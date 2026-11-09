#!/usr/bin/env python3
"""One-command reconstruction of manuscript Fig1.pdf through Fig9.pdf."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEFAULT_ROOT=HERE.parents[1]


def call(*args):
    subprocess.run([sys.executable,*map(str,args)],check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT,help='Repository root')
    ap.add_argument('--out',type=Path,default=None,help='Destination for Fig1...Fig9')
    args=ap.parse_args()
    root=args.root.resolve()
    results=root/'results'
    out=(args.out or root/'build'/'figures').resolve()
    out.mkdir(parents=True,exist_ok=True)
    scratch=out/'supporting_outputs'
    scratch.mkdir(parents=True,exist_ok=True)
    call(HERE/'generate_article_outputs.py','--results',results,'--out',scratch)
    originals={
       1:'fig1_selected_accuracy_adaptive_median_iqr',
       2:'fig2_candidate_distribution_adaptive',
       3:'fig3_adaptive_mean_rank',
       5:'fig5_pointwise_error_lambda3p4_seed123',
       6:'fig6_weight_trajectories_lambda3p4_seed123',
    }
    for i,stem in originals.items():
        for ext in ('pdf','png'):
            shutil.copyfile(scratch/f'{stem}.{ext}',out/f'Fig{i}.{ext}')
    call(HERE/'make_time_error_figure.py',results/'final'/'final_results.csv',out)
    call(HERE/'make_branch_figures.py','--results',results,'--out',out)
    files=[out/f'Fig{i}.pdf' for i in range(1,10)]
    for p in files:
        if not p.exists() or p.stat().st_size==0:raise FileNotFoundError(p)
    print(f'REPRO_FIGURES_OK: 9 PDF figures written to {out}')


if __name__=='__main__':main()
