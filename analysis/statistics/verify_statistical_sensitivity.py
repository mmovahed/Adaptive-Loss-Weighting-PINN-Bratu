"""Reproduce the R2-C5 seed-blocked sensitivity analysis.

Run next to adaptive_selected_60.csv with Python 3 + NumPy/Pandas/SciPy.
This uses final errors of selected candidates only; it does not use all-candidate
errors as additional independent inferential observations.
"""
from itertools import permutations, product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, friedmanchisquare, wilcoxon

base = Path(__file__).resolve().parent
df = pd.read_csv(base / "adaptive_selected_60.csv")
methods = ["lra", "relobralo", "ddqn"]
seeds = sorted(df.seed.unique().tolist())
lambdas = sorted(df["lambda"].unique().tolist())
assert len(seeds) == 5 and len(lambdas) == 4
assert len(df) == 60
v = {(int(r.seed), float(r["lambda"]), r.method): float(r.relative_l2)
     for _, r in df.iterrows()}
M = np.array([[[v[(s, la, method)] for method in methods]
               for la in lambdas] for s in seeds])
R = np.apply_along_axis(rankdata, -1, M)
mean_ranks = R.mean(axis=(0, 1))
assert np.allclose(mean_ranks, [1.45, 2.3, 2.25])
num_cases = len(seeds)*len(lambdas)

def friedman_statistic(ranks):
    totals = np.sum(ranks, axis=(0,1))
    m=len(methods)
    return 12.0 / (num_cases*m*(m+1)) * np.sum(totals**2) - 3*num_cases*(m+1)

obs = friedman_statistic(R)
label_permutations = np.array(list(permutations(range(3))), dtype=int)
per_seed_rank_totals = R.sum(axis=1)
per_seed_relabelled = [[row[perm] for perm in label_permutations]
                       for row in per_seed_rank_totals]
extreme=0
for indices in product(range(6), repeat=5):
    totals = sum(per_seed_relabelled[i][indices[i]] for i in range(5))
    statistic = 12/(num_cases*3*4)*np.sum(totals**2)-3*num_cases*4
    extreme += statistic >= obs-1e-9
exact_omnibus = extreme/(6**5)
assert extreme == 360 and np.isclose(exact_omnibus, 0.046296296296296294)

pairs = [(0,1),(0,2),(1,2)]
rows=[]
for a,b in pairs:
    # One independent replicate is the mean log error difference across 4 lambdas
    # for a single shared outer seed; the statistic therefore has n=5.
    seed_diffs = np.log10(M[:,:,a] / M[:,:,b]).mean(axis=1)
    test = wilcoxon(seed_diffs, method="exact", alternative="two-sided")
    rows.append({"comparison":f"{methods[a]} vs {methods[b]}",
                 "n_outer_seeds":len(seeds),
                 "exact_wilcoxon_p": float(test.pvalue),
                 "geometric_mean_ratio_second_over_first": float(10**(-seed_diffs.mean())),
                 "seed_blocks_first_lower":int(np.sum(seed_diffs < 0))})
# Holm adjustment with 3 pairwise tests, retaining monotonicity
ps=np.array([row["exact_wilcoxon_p"] for row in rows])
order=np.argsort(ps)
adjusted=np.zeros(3)
prev=0
for k,index in enumerate(order):
    adjusted[index]=min(1.0, max(prev, ps[index]*(3-k)))
    prev=adjusted[index]
for r,adj in zip(rows,adjusted): r["holm_adjusted_p"] = adj
pd.DataFrame(rows).to_csv(base/"seed_blocked_pairwise_tests.csv",index=False)

summary = pd.DataFrame([{"test":"Conventional Friedman (20 lambda-seed cases)",
                         "statistic":obs,
                         "p_value":friedmanchisquare(*[M[:,:,k].ravel() for k in range(3)]).pvalue,
                         "n_blocks":20},
                        {"test":"Exact method-label permutations within five seed blocks",
                         "statistic":obs,"p_value":exact_omnibus,"n_blocks":5}])
summary.to_csv(base/"omnibus_statistical_tests.csv",index=False)
print(summary.to_string(index=False))
print(pd.DataFrame(rows).to_string(index=False))
