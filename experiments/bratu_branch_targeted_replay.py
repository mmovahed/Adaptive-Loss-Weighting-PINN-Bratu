"""Isolated branch-diagnostic rerun for five original DDQN cases.

No original training scripts or result files are modified. This script reproduces
five complete candidate episodes per case (persistent DDQN controller) and saves
ALL five final predictions, including the candidates not selected by validation.
Compatible with Python 3.7 and the original Bratu benchmark interfaces.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from configs.bratu_config import (N_TEST, N_VALIDATION, TOTAL_PINN_STEPS,
    SELECTION_CANDIDATES, CANDIDATE_SEED_STRIDE, DDQN_TRAIN_EPISODES)
from experiments.bratu_ddqn_single import run_controller_episode
from problems.bratu import BratuProblem
from rl.ddqn_agent import DDQNAgent
from utils.bratu_eval import (evaluate_candidate_models, make_validation_grid, set_seed)
from utils.fair_selection import make_candidate_seeds, select_best_candidate

TARGET_CASES = [
    (3.0, 1000),
    (3.4, 42),
    (3.4, 456),
    (3.4, 789),
    (3.4, 1000),
]


def reference_branches(x, lam):
    x = np.asarray(x, dtype=np.float64).ravel()
    ac = brentq(lambda a: 2.0/a-np.tanh(a/2.0), 1, 5)
    f = lambda a: 2.0*a*a/(np.cosh(a/2.0)**2)-lam
    a_lower = brentq(f, 1.0e-8, ac)
    a_upper = brentq(f, ac, 50.0)
    def u(a):
        return 2.0*(np.log(np.cosh(a/2.0))-np.log(np.cosh(a*(x-0.5))))
    return u(a_lower), u(a_upper)


def run_case(index, smoke=False):
    lam, outer_seed = TARGET_CASES[index]
    if not smoke and TOTAL_PINN_STEPS != 15000:
        raise RuntimeError('Production requires exactly 15000 steps per candidate.')
    if DDQN_TRAIN_EPISODES != SELECTION_CANDIDATES or SELECTION_CANDIDATES != 5:
        raise RuntimeError('This rerun requires the original persistent 5-candidate DDQN protocol.')
    if torch.cuda.is_available():
        raise RuntimeError('This audit is specified for CPU, to match original runs.')
    device = 'cpu'
    set_seed(outer_seed)
    seeds = make_candidate_seeds(outer_seed, SELECTION_CANDIDATES, CANDIDATE_SEED_STRIDE)
    problem = BratuProblem(lam=lam)
    x_val = make_validation_grid(N_VALIDATION, device)
    set_seed(outer_seed)
    agent = DDQNAgent(state_dim=7, action_dim=25, learning_rate=1e-3,
        gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=0.995, batch_size=32, memory_size=5000,
        target_update_frequency=100, device=device)
    models, infos = [], []
    for j, seed in enumerate(seeds, start=1):
        model, info = run_controller_episode(agent, problem, x_val, device, seed)
        models.append(model)
        infos.append(info)
        print('Candidate {}/5: seed={}, S_val={:.8g}'.format(j, seed, info['validation_score']), flush=True)
    chosen = select_best_candidate([info['validation_score'] for info in infos])
    evaluated = evaluate_candidate_models(models, problem, N_TEST, device)
    xx = np.asarray(evaluated[0]['x'], dtype=np.float64).ravel()
    uu_l, uu_u = reference_branches(xx, lam)
    predictions = np.stack([np.asarray(r['y_pred']).ravel() for r in evaluated], axis=0)
    # Unnormalized distances, for comparison to upper/lower on a common grid.
    dlo = np.sqrt(np.mean((predictions-uu_l[None,:])**2, axis=1))
    dup = np.sqrt(np.mean((predictions-uu_u[None,:])**2, axis=1))
    outdir = ROOT / 'results' / 'branch_audit_v1' / ('smoke' if smoke else 'production')
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / 'ddqn_lambda_{}_seed_{}.npz'.format(str(lam).replace('.','p'), outer_seed)
    if fname.exists() and not smoke:
        raise RuntimeError('Refusing to overwrite existing audit: {}'.format(fname))
    np.savez_compressed(str(fname), lambda_value=np.asarray([lam]), outer_seed=np.asarray([outer_seed]),
        candidate_seeds=np.asarray(seeds, dtype=np.int64), selected_candidate=np.asarray([chosen+1]),
        x=xx, y_pred_all=predictions, y_ref_numerical=np.asarray(evaluated[0]['y_ref']),
        y_lower_analytic=uu_l, y_upper_analytic=uu_u,
        relative_l2_rerun=np.asarray([r['relative_l2'] for r in evaluated]),
        validation_score_rerun=np.asarray([q['validation_score'] for q in infos]),
        distance_lower_rms=dlo, distance_upper_rms=dup,
        closer_upper=(dup<dlo).astype(np.int8),
        n_steps=np.asarray([TOTAL_PINN_STEPS]), method=np.asarray(['ddqn']))
    print('OUTPUT {}'.format(fname),flush=True)
    print('CLOSER_UPPER {}'.format((dup<dlo).tolist()),flush=True)
    return fname

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=int,required=True,choices=range(5))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    run_case(args.index,args.smoke)
