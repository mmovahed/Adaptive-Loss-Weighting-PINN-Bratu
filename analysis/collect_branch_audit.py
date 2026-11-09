"""Collect five targeted DDQN replay cases and compare to archived original metrics."""
import csv
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
ORIG=ROOT/'reference/original_candidate_results.csv'
OUT=ROOT/'results/branch_audit_v1/production'
CASES=[(3.0,1000),(3.4,42),(3.4,456),(3.4,789),(3.4,1000)]

def read_reference():
    with ORIG.open('r',newline='') as f:
        d=list(csv.DictReader(f))
    return {(float(r['lambda']),int(r['outer_seed']),int(r['candidate'])):float(r['test_relative_l2']) for r in d if r['method']=='ddqn'}

def main():
    reference=read_reference()
    rows=[]
    for lam,seed in CASES:
        fp=OUT/'ddqn_lambda_{}_seed_{}.npz'.format(str(lam).replace('.','p'),seed)
        if not fp.exists():
            raise RuntimeError('Missing result: {}'.format(fp))
        with np.load(str(fp)) as z:
            if int(z['n_steps'][0])!=15000 or z['y_pred_all'].shape!=(5,2000):
                raise RuntimeError('Incomplete/wrong-budget result: {}'.format(fp))
            for j in range(5):
                orig=reference[(lam,seed,j+1)]
                rerun=float(z['relative_l2_rerun'][j])
                delta=abs(rerun-orig)
                # This is a concordance CHECK, not a proof of trajectory identity.
                agreement=delta<=max(1e-4,0.02*abs(orig))
                rows.append(dict(lambda_value=lam,seed=seed,candidate=j+1,
                    original_relative_l2=orig,rerun_relative_l2=rerun,
                    absolute_error_difference=delta,metric_agreement_2pct=agreement,
                    distance_lower_rms=float(z['distance_lower_rms'][j]),
                    distance_upper_rms=float(z['distance_upper_rms'][j]),
                    rerun_closer_branch='upper' if int(z['closer_upper'][j]) else 'lower'))
    out=OUT/'replay_branch_comparison.csv'
    with out.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print('Comparison saved:',out)
    for r in rows:
        if r['original_relative_l2']>0.322 or r['lambda_value']==3.0 and r['original_relative_l2']>.99:
            print(r)
    print('Rerun metric agreement: {}/25'.format(sum(r['metric_agreement_2pct'] for r in rows)))
    print('NOTE: Matching a scalar error does not establish identity of the original model; assess before making a branch claim about archived runs.')

if __name__=='__main__': main()
