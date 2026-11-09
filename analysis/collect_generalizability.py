"""Audited collection; accepts ONLY full production results (40 cases)."""
import csv
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.generalizability_worker import (BENCHMARKS,METHODS,SEEDS,PROTOCOL,STRIDE)


def _write(path,records,cols):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=cols); writer.writeheader(); writer.writerows(records)


def collect(base):
    base=Path(base); expected=[]; selected=[]; candidates=[]
    configs=set()
    for bench in BENCHMARKS:
        for method in METHODS:
            for seed in SEEDS:
                path=base/bench/method/('seed_{}.npz'.format(seed))
                if not path.exists(): expected.append(str(path)); continue
                with np.load(str(path),allow_pickle=False) as d:
                    if str(d['protocol_version'].item())!=PROTOCOL or int(d['steps'])!=15000 or int(d['n_candidates'])!=5:
                        raise RuntimeError('Wrong protocol or budget: '+str(path))
                    actual=str(d['benchmark'].item()),str(d['method'].item()),int(d['outer_seed'])
                    if actual!=(bench,method,seed): raise RuntimeError('Identity mismatch '+str(path))
                    seeds=list(map(int,d['candidate_seed']))
                    if seeds != [seed+STRIDE*j for j in range(1,6)]: raise RuntimeError('Unmatched candidate seeds: '+str(path))
                    scores=np.asarray(d['selection_score'],dtype=float)
                    idx=int(d['selected_candidate'])-1
                    if idx!=int(np.argmin(scores)): raise RuntimeError('Selection mismatch '+str(path))
                    errs=np.asarray(d['candidate_relative_l2'],dtype=float)
                    durations=np.asarray(d['candidate_training_seconds'],dtype=float)
                    if scores.size!=5 or errs.size!=5 or durations.size!=5 or not np.all(np.isfinite(errs)) or not np.all(np.isfinite(scores)):
                        raise RuntimeError('Missing, nonfinite or malformed data: '+str(path))
                    config=str(d['config_json'].item())
                    configs.add(config)
                    selected.append({'benchmark':bench,'method':method,'seed':seed,
                        'selected_candidate':idx+1,'selected_relative_l2':float(errs[idx]),
                        'selected_validation':float(scores[idx]),'five_candidate_seconds':float(np.sum(durations)),
                        'host':str(d['hostname'].item())})
                    for j in range(5):
                        candidates.append({'benchmark':bench,'method':method,'seed':seed,'candidate':j+1,
                           'candidate_seed':seeds[j],'relative_l2':float(errs[j]),
                           'validation_score':float(scores[j]),'train_seconds':float(durations[j]),
                           'selected':int(j==idx)})
    if expected:
        raise RuntimeError('Missing {} of 40 results; first missing: {}'.format(len(expected),expected[0]))
    if len(configs)!=1: raise RuntimeError('Inconsistent training configurations among results')
    base.mkdir(parents=True,exist_ok=True)
    summary=[]
    for bench in BENCHMARKS:
        for method in METHODS:
            v=np.asarray([x['selected_relative_l2'] for x in selected if x['benchmark']==bench and x['method']==method])
            c=np.asarray([x['relative_l2'] for x in candidates if x['benchmark']==bench and x['method']==method])
            t=np.asarray([x['five_candidate_seconds'] for x in selected if x['benchmark']==bench and x['method']==method])
            summary.append({'benchmark':bench,'method':method,'n_cases':len(v),
              'mean_relative_l2':float(v.mean()),'sd_relative_l2':float(v.std(ddof=1)),
              'median_relative_l2':float(np.median(v)),'candidate_q90':float(np.percentile(c,90)),
              'mean_five_candidate_seconds':float(t.mean())})
    wins=[]
    for bench in BENCHMARKS:
        for seed in SEEDS:
            items=[x for x in selected if x['benchmark']==bench and x['seed']==seed and x['method']!='fixed_equal']
            best=min(items,key=lambda x:x['selected_relative_l2'])
            wins.append({'benchmark':bench,'seed':seed,'winning_method':best['method'],
                         'winning_relative_l2':best['selected_relative_l2']})
    _write(base/'selected_cases.csv',selected,list(selected[0].keys()))
    _write(base/'all_candidates.csv',candidates,list(candidates[0].keys()))
    _write(base/'method_summary.csv',summary,list(summary[0].keys()))
    _write(base/'adaptive_winners.csv',wins,list(wins[0].keys()))
    with (base/'COLLECTION_VALIDATED.txt').open('w') as f:
        f.write('40 cases, 200 candidates, protocol/budget/seed/selection/finite checks passed.\n')
        f.write('Report accuracy within each benchmark; do not compare absolute errors across PDEs.\n')
        f.write('Five outer seeds per PDE yield descriptive comparisons; no significance claims.\n')
        f.write('Wall-clock durations are descriptive unless compute hardware is controlled.\n')
    print('Validated 40 runs and 200 candidates; CSV files written to '+str(base))


if __name__=='__main__':
    collect(ROOT/'results'/'generalizability_v1'/'production')
