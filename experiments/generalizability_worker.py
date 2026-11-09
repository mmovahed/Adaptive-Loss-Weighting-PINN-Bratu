"""Standalone, reference-blind matched-candidate comparison for R1-C8/R2-C4.
Python >=3.7; intentionally does not modify the original Bratu trainers.
"""
import argparse
import itertools
import json
import math
import os
import socket
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from models.pinn import PINN
from rl.ddqn_agent import DDQNAgent
from training.adaptive_weighting import (LearningRateAnnealingBalancer, ReLoBRaLoBalancer)
from problems.generalizability_benchmarks import make_problem
from utils.bratu_eval import set_seed

BENCHMARKS=('helmholtz2d','burgers1d_time')
METHODS=('fixed_equal','lra','relobralo','ddqn')
SEEDS=(42,123,456,789,1000)
PROTOCOL='generalizability_matched_v1'
STRIDE=1000003
MULTIPLIERS=(.8,.9,1.,1.1,1.2)


def index_case(index):
    if index<0 or index>=len(BENCHMARKS)*len(METHODS)*len(SEEDS):
        raise ValueError('Index must be between 0 and 39')
    return (BENCHMARKS[index//20],METHODS[(index//5)%4],SEEDS[index%5])


def _score(model,problem,grids):
    model.eval()
    terms=problem.loss_components(model,grids)
    return np.asarray([float(x.detach().item()) for x in terms],dtype=np.float64)


def _state(current,previous,weights,step,total_steps):
    current=np.maximum(current,1e-12)
    previous=np.maximum(previous,1e-12)
    weights=np.asarray(weights,dtype=np.float64)
    return np.asarray(list(np.log(current+1e-12))+
        list(np.log(current+1e-12)-np.log(previous+1e-12))+
        list(weights/100.)+[float(step)/total_steps],dtype=np.float32)


def _apply_action(weights,action):
    m=len(weights)
    factors=list(itertools.product(MULTIPLIERS,repeat=m))
    raw=np.asarray(weights,dtype=np.float64)*np.asarray(factors[int(action)])
    return np.clip(raw,.01,100.)


def _lra_update_multi(physics,aux_losses,weights,model):
    """Independent reference-style LRA gradients for each auxiliary loss.

    The m=2 worker delegates to the ORIGINAL two-loss LRA class instead.
    The extension computes gradients of CURRENTLY WEIGHTED auxiliary losses.
    """
    pars=[p for p in model.parameters() if p.requires_grad and p.ndim>=2]
    gp=torch.autograd.grad(physics,pars,retain_graph=True,allow_unused=True)
    numerator=max(float(torch.max(torch.abs(v)).item()) for v in gp if v is not None)
    new=np.asarray(weights,dtype=np.float64).copy()
    for i,li in enumerate(aux_losses,start=1):
        grads=torch.autograd.grad(float(weights[i])*li,pars,retain_graph=True,allow_unused=True)
        vals=[float(torch.mean(torch.abs(v)).item()) for v in grads if v is not None]
        if vals:
            hat=numerator/max(float(np.mean(vals)),1e-12)
            if np.isfinite(hat):
                new[i]=0.9*weights[i]+0.1*hat
    return new


def run_candidate(problem,method,seed,grids,validation,steps,device,hidden,warmup,interval):
    set_seed(seed)
    model=PINN(input_dim=problem.input_dim,output_dim=1,hidden_layers=hidden).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    m=len(problem.loss_names)
    weights=np.ones(m,dtype=np.float64)
    lra2=LearningRateAnnealingBalancer(smoothing=.9) if method=='lra' and m==2 else None
    balancer=ReLoBRaLoBalancer(n_losses=m,alpha=.999,rho_probability=.9999,
           temperature=.1,random_seed=seed+7919) if method=='relobralo' else None
    weights_log=[]; losses_log=[]; steps_log=[]
    agent_metrics={'actions':0,'q_updates':0,'q_seconds':0.,'action_seconds':0.}

    def update_pinn():
        model.train(); opt.zero_grad()
        terms=problem.loss_components(model,grids)
        if method=='relobralo':
            weights[:]=balancer.update([float(x.detach().item()) for x in terms])
        loss=sum(float(w)*x for w,x in zip(weights,terms))
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError('Nonfinite weighted training loss (seed={}, method={})'.format(seed,method))
        loss.backward(); opt.step()
        return terms

    start=time.perf_counter()
    if method!='ddqn':
        for step in range(1,steps+1):
            terms=update_pinn()
            if method=='lra' and step%10==0:
                fresh=problem.loss_components(model,grids)
                if m==2:
                    weights[1]=lra2.update(fresh[0],fresh[1],model.parameters())
                else:
                    weights[:]=_lra_update_multi(fresh[0],fresh[1:],weights,model)
            if step==1 or step%20==0 or step==steps:
                steps_log.append(step); weights_log.append(weights.copy())
                losses_log.append([float(v.detach().item()) for v in terms])
        return model,{'seconds':time.perf_counter()-start,'weights':weights.copy(),
                      'steps':np.asarray(steps_log),'weight_trace':np.asarray(weights_log),
                      'loss_trace':np.asarray(losses_log),'controller':agent_metrics},None

    raise RuntimeError('DDQN requires the persistent-controller case-level loop')


def train_ddqn_candidate(problem,agent,seed,grids,validation,steps,device,hidden,warmup,interval):
    set_seed(seed)
    model=PINN(input_dim=problem.input_dim,output_dim=1,hidden_layers=hidden).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    m=len(problem.loss_names)
    weights=np.ones(m,dtype=np.float64)
    count_actions=0; q_updates=0; q_time=0.; action_time=0.
    trace=[]; step_trace=[]

    def train_for(n):
        for _ in range(n):
            model.train(); opt.zero_grad()
            terms=problem.loss_components(model,grids)
            value=sum(float(w)*term for w,term in zip(weights,terms))
            if not bool(torch.isfinite(value).item()):
                raise RuntimeError('Nonfinite DDQN PINN loss')
            value.backward(); opt.step()

    start=time.perf_counter()
    initial=min(warmup,steps)
    train_for(initial)
    current=_score(model,problem,validation)
    prev=current.copy()
    state=_state(current,prev,weights,initial,steps)
    trace.append(weights.copy()); step_trace.append(initial)
    for begin in range(initial+1,steps+1,interval):
        ta=time.perf_counter(); action=agent.select_action(state)
        action_time+=time.perf_counter()-ta
        count_actions+=1
        weights=_apply_action(weights,action)
        n=min(interval,steps-begin+1)
        train_for(n)
        updated=_score(model,problem,validation)
        reward=float(np.tanh(np.log((np.sum(current)+1e-12)/(np.sum(updated)+1e-12))))
        step=begin+n-1
        next_state=_state(updated,current,weights,step,steps)
        agent.remember(state,action,reward,next_state,step>=steps)
        tq=time.perf_counter(); old_steps=agent.train_steps
        agent.train_step()
        q_time+=time.perf_counter()-tq
        q_updates+=agent.train_steps-old_steps
        current=updated; state=next_state
        trace.append(weights.copy()); step_trace.append(step)
    return model,{'seconds':time.perf_counter()-start,'weights':weights,
                  'steps':np.asarray(step_trace),'weight_trace':np.asarray(trace),
                  'controller':{'actions':count_actions,'q_updates':q_updates,
                                'q_seconds':q_time,'action_seconds':action_time}}


def run_case(benchmark,method,outer_seed,steps=15000,candidates=5,n_collocation=256,
             n_validation=512,n_test=2000,hidden=(64,64,64),warmup=200,interval=20,device='cpu'):
    if method not in METHODS: raise ValueError(method)
    problem=make_problem(benchmark)
    assert len(problem.loss_names) in (2,3)
    set_seed(outer_seed)
    train_grids=problem.training_grids(n_collocation,device)
    val_grids=problem.validation_grids(n_validation,device)
    model_list=[]; infos=[]; scores=[]; comp=[]
    agent=None
    if method=='ddqn':
        set_seed(outer_seed)
        m=len(problem.loss_names)
        agent=DDQNAgent(state_dim=3*m+1,action_dim=5**m,learning_rate=1e-3,
          gamma=.99,epsilon_start=1.,epsilon_end=.05,epsilon_decay=.995,
          batch_size=32,memory_size=5000,target_update_frequency=100,device=device)
    t0=time.perf_counter()
    for j in range(1,candidates+1):
        seed=outer_seed+STRIDE*j
        if agent is None:
            model,info,_=run_candidate(problem,method,seed,train_grids,val_grids,
                               steps,device,hidden,warmup,interval)
        else:
            model,info=train_ddqn_candidate(problem,agent,seed,train_grids,val_grids,
                               steps,device,hidden,warmup,interval)
        terms=_score(model,problem,val_grids)
        score=float(np.sum(terms))
        if not np.isfinite(score): raise RuntimeError('Nonfinite validation score')
        infos.append(info); model_list.append(model); scores.append(score); comp.append(terms)
        print('CANDIDATE benchmark={} method={} seed={} S_val={:.8e} seconds={:.2f}'.format(
              benchmark,method,seed,score,info['seconds']),flush=True)
    winner=int(np.argmin(scores))
    # IMPORTANT: No reference/test grid exists until AFTER winner is fixed.
    test_x=problem.test_points(device,n_test)
    with torch.no_grad():
        ref=problem.exact(test_x).detach().cpu().numpy().reshape(-1)
        pred=np.stack([model(test_x).detach().cpu().numpy().reshape(-1) for model in model_list])
    errors=pred-ref[None,:]
    rel=np.linalg.norm(errors,axis=1)/(np.linalg.norm(ref)+1e-12)
    mse=np.mean(errors**2,axis=1)
    mae=np.mean(np.abs(errors),axis=1)
    ctrl={k:sum(float(x['controller'][k]) for x in infos) for k in
          ('actions','q_updates','q_seconds','action_seconds')}
    return {'benchmark':benchmark,'method':method,'outer_seed':outer_seed,
        'selection_score':np.asarray(scores), 'validation_components':np.stack(comp),
        'candidate_seed':np.asarray([outer_seed+STRIDE*j for j in range(1,candidates+1)]),
        'candidate_relative_l2':rel,'candidate_mse':mse,'candidate_mae':mae,
        'candidate_training_seconds':np.asarray([i['seconds'] for i in infos]),
        'selected_candidate':winner+1,'selected_relative_l2':float(rel[winner]),
        'selected_mse':float(mse[winner]),'selected_mae':float(mae[winner]),
        'selected_score':float(scores[winner]),'selected_predictions':pred[winner],
        'reference_predictions':ref,'test_coordinates':test_x.detach().cpu().numpy(),
        'weight_trace':infos[winner]['weight_trace'], 'step_trace':infos[winner]['steps'],
        'loss_trace':infos[winner].get('loss_trace',np.zeros((0,len(problem.loss_names)))),
        'selected_weights':infos[winner]['weights'], 'n_losses':len(problem.loss_names),
        'total_seconds':time.perf_counter()-t0,'hostname':socket.gethostname(),
        'controller_actions':ctrl['actions'],
        'controller_updates':ctrl['q_updates'],'controller_q_seconds':ctrl['q_seconds'],
        'controller_action_seconds':ctrl['action_seconds']}


def save_result(result,dest,steps,candidates):
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    data=dict(result)
    data.update({'protocol_version':PROTOCOL,'steps':int(steps),'n_candidates':int(candidates)})
    tmp=dest.with_name(dest.name+'.tmp')
    with tmp.open('wb') as fh: np.savez_compressed(fh,**data)
    os.replace(str(tmp),str(dest))
    return dest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=int,default=None)
    parser.add_argument('--benchmark',choices=BENCHMARKS)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--mode',choices=('production','smoke'),default='production')
    parser.add_argument('--steps',type=int,default=15000)
    parser.add_argument('--candidates',type=int,default=5)
    parser.add_argument('--collocation',type=int,default=256)
    parser.add_argument('--validation',type=int,default=512)
    parser.add_argument('--test',type=int,default=2000)
    parser.add_argument('--hidden',type=int,default=64)
    parser.add_argument('--warmup',type=int,default=200)
    parser.add_argument('--interval',type=int,default=20)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if args.index is not None:
        args.benchmark,args.method,args.seed=index_case(args.index)
    if not args.benchmark or not args.method: parser.error('Specify --index or benchmark and method')
    if args.steps<1 or args.candidates<1 or args.validation<1 or args.collocation<1 or args.test<4: parser.error('Invalid positive settings')
    if args.interval<1 or args.warmup<0: parser.error('Invalid DDQN settings')
    if args.mode=='production':
        expected=(15000,5,256,512,2000,64,200,20)
        actual=(args.steps,args.candidates,args.collocation,args.validation,
                args.test,args.hidden,args.warmup,args.interval)
        if actual!=expected: parser.error('Production configuration is fixed; use --mode smoke for reduced budgets')
    os.environ['OMP_NUM_THREADS']='1'
    try: torch.set_num_threads(1)
    except Exception: pass
    dest=ROOT/'results'/'generalizability_v1'/args.mode/args.benchmark/args.method/('seed_{}.npz'.format(args.seed))
    config_json=json.dumps({'steps':args.steps,'candidates':args.candidates,
         'collocation':args.collocation,'validation':args.validation,'test':args.test,
         'hidden_layers':[args.hidden]*3,'learning_rate':0.001,'warmup':args.warmup,
         'interval':args.interval,'seed_stride':STRIDE},sort_keys=True)
    if dest.exists() and not args.overwrite:
        with np.load(str(dest),allow_pickle=False) as d:
            if str(d['protocol_version'].item())!=PROTOCOL or str(d['config_json'].item())!=config_json:
                raise RuntimeError('Existing result uses a different protocol or configuration: '+str(dest))
        print('SKIP validated existing '+str(dest),flush=True); return
    device='cpu'
    out=run_case(args.benchmark,args.method,args.seed,steps=args.steps,
         candidates=args.candidates,n_collocation=args.collocation,n_validation=args.validation,
         n_test=args.test,hidden=(args.hidden,)*3,warmup=args.warmup,
         interval=args.interval,device=device)
    out['config_json']=config_json
    save_result(out,dest,args.steps,args.candidates)
    print('RESULT_JSON: '+json.dumps({'benchmark':args.benchmark,'method':args.method,
           'seed':args.seed,'relative_l2':out['selected_relative_l2'],
           'validation_score':out['selected_score'],'training_seconds':float(sum(out['candidate_training_seconds'])),
           'output':str(dest)},sort_keys=True),flush=True)


if __name__=='__main__': main()
