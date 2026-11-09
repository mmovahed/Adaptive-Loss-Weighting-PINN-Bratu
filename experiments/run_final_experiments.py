import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

DEFAULT_METHODS = {
    "fixed_equal": "bratu_experiment.py",
    "fixed_selected": "bratu_fixed_selected.py",
    "lra": "bratu_lra.py",
    "relobralo": "bratu_relobralo.py",
    "ddqn": "bratu_ddqn_single.py",
}
DEFAULT_LAMBDAS = [1.0, 2.0, 3.0, 3.4]
DEFAULT_SEEDS = [42, 123, 456, 789, 1000]


def _selected_methods():
    raw = os.environ.get("BRATU_METHODS", "").strip()
    if not raw:
        return DEFAULT_METHODS
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in DEFAULT_METHODS]
    if unknown:
        raise ValueError("Unknown methods: {}".format(unknown))
    return {name: DEFAULT_METHODS[name] for name in names}


def _selected_lambdas():
    raw = os.environ.get("BRATU_LAMBDAS", "").strip()
    return DEFAULT_LAMBDAS if not raw else [float(x.strip()) for x in raw.split(",") if x.strip()]


def _selected_seeds():
    raw = os.environ.get("BRATU_SEEDS", "").strip()
    return DEFAULT_SEEDS if not raw else [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_result_json(output):
    marker = "RESULT_JSON:"
    for line in reversed(output.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):].strip())
    return None


def run_experiment(method_name, script_name, lam, seed):
    print("\n" + "=" * 80)
    print("METHOD={} | LAMBDA={} | SEED={}".format(method_name, lam, seed))
    print("=" * 80)
    env = os.environ.copy()
    env["BRATU_LAMBDA"] = str(lam)
    env["BRATU_SEED"] = str(seed)
    script_path = PROJECT_ROOT / "experiments" / script_name

    start = time.perf_counter()
    completed = subprocess.run(
        [PYTHON, str(script_path)], cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    wall_time_seconds = time.perf_counter() - start
    output = completed.stdout + "\n" + completed.stderr
    print(output)
    if completed.returncode != 0:
        return None
    result = parse_result_json(output)
    if result is None:
        return None
    result["wall_time_seconds"] = wall_time_seconds
    return result


def main():
    methods = _selected_methods(); lambdas = _selected_lambdas(); seeds = _selected_seeds()
    total = len(methods) * len(lambdas) * len(seeds)
    final_dir = PROJECT_ROOT / "results" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    summary_file = final_dir / "final_results.csv"

    print("=" * 80)
    print("BRATU ADAPTIVE LOSS-WEIGHTING BENCHMARK")
    print("=" * 80)
    print("Methods : {}".format(list(methods.keys())))
    print("Lambdas : {}".format(lambdas))
    print("Seeds   : {}".format(seeds))
    print("Runs    : {}".format(total))
    print("=" * 80)

    rows, failures = [], []
    current = 0
    for lam in lambdas:
        for seed in seeds:
            for method_name, script_name in methods.items():
                current += 1
                print("\n[{}/{}]".format(current, total))
                metrics = run_experiment(method_name, script_name, lam, seed)
                if metrics is None:
                    failures.append((method_name, lam, seed)); continue
                row = {
                    "method": method_name, "lambda": lam, "seed": seed,
                    "protocol_version": metrics.get("protocol_version"),
                    "selection_protocol": metrics.get("selection_protocol"),
                    "n_candidates": metrics.get("n_candidates"),
                    "selected_candidate": metrics.get("selected_candidate"),
                    "selected_candidate_seed": metrics.get("selected_candidate_seed"),
                    "validation_score": metrics.get("validation_score"),
                    "mse": metrics.get("mse"), "mae": metrics.get("mae"),
                    "max_error": metrics.get("max_error"),
                    "relative_l2": metrics.get("relative_l2"),
                    "alpha": metrics.get("alpha"), "beta": metrics.get("beta"),
                    "candidate_train_seconds_total": metrics.get("candidate_train_seconds_total"),
                    "candidate_train_seconds_mean": metrics.get("candidate_train_seconds_mean"),
                    "wall_time_seconds": metrics.get("wall_time_seconds"),
                }
                rows.append(row)

    fieldnames = [
        "method", "lambda", "seed", "protocol_version", "selection_protocol",
        "n_candidates", "selected_candidate", "selected_candidate_seed",
        "validation_score", "mse", "mae", "max_error", "relative_l2",
        "alpha", "beta", "candidate_train_seconds_total",
        "candidate_train_seconds_mean", "wall_time_seconds",
    ]
    with open(str(summary_file), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)

    print("\nSuccessful: {}/{}".format(len(rows), total))
    print("Failed    : {}".format(len(failures)))
    print("Saved     : {}".format(summary_file))
    if failures:
        for failure in failures:
            print("FAILED: {}".format(failure))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
