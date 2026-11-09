"""Bratu DDQN ablation dispatcher. Run from the project root:
python experiments/bratu_ddqn_ablation.py --index 0

Indices 0-19: instrumented reproduction of the original unnormalized controller;
indices 20-39: normalized alpha+beta=2, everything else matched.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ddqn_ablation_support import decode_array_index, ablation_output_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="Short, isolated test outputs only")
    parser.add_argument("--smoke-steps", type=int, default=40, help="Only with --smoke; exercise controller training")
    args = parser.parse_args()
    mode, lam, seed = decode_array_index(args.index)
    os.environ["BRATU_DDQN_ABLATION_MODE"] = mode
    os.environ["BRATU_LAMBDA"] = str(lam)
    os.environ["BRATU_SEED"] = str(seed)
    os.environ["BRATU_DDQN_ABLATION_TAG"] = "smoke" if args.smoke else "production"
    if args.smoke:
        if args.smoke_steps < 20:
            raise ValueError("--smoke-steps must be at least 20")
        os.environ.update({
            "BRATU_TOTAL_PINN_STEPS": str(args.smoke_steps),
            "BRATU_N_COLLOCATION": "8",
            "BRATU_N_VALIDATION": "12",
            "BRATU_N_TEST": "50",
            "BRATU_WARMUP_STEPS": "10",
            "BRATU_DECISION_INTERVAL": "10",
            "BRATU_SELECTION_CANDIDATES": "2",
            "BRATU_DDQN_TRAIN_EPISODES": "2",
        })
    else:
        if args.smoke_steps != 40:
            raise ValueError("--smoke-steps is only valid together with --smoke")
        # Guard against an accidental low-budget scientific run. PBS pins all values.
        expected = {
            "BRATU_TOTAL_PINN_STEPS": 15000,
            "BRATU_N_COLLOCATION": 256,
            "BRATU_N_VALIDATION": 512,
            "BRATU_N_TEST": 2000,
            "BRATU_WARMUP_STEPS": 200,
            "BRATU_DECISION_INTERVAL": 20,
            "BRATU_SELECTION_CANDIDATES": 5,
            "BRATU_DDQN_TRAIN_EPISODES": 5,
            "BRATU_CANDIDATE_SEED_STRIDE": 1000003,
        }
        for k, v in expected.items():
            if int(os.environ.get(k, v)) != v:
                raise RuntimeError("Production setting {} must equal {}".format(k, v))
    output_npz, output_json = ablation_output_paths(ROOT, mode, lam, seed)
    if output_npz.exists() and output_json.exists():
        with output_json.open("r") as handle:
            prior = json.load(handle)
        if (prior.get("mode") != mode or float(prior.get("lambda")) != float(lam)
                or int(prior.get("outer_seed")) != int(seed)):
            raise RuntimeError("Previous result has mismatched metadata: {}".format(output_json))
        print("SKIP complete result: {}".format(output_json), flush=True)
        return
    if output_npz.exists() != output_json.exists():
        raise RuntimeError("Incomplete output pair; inspect and delete ONLY the partial ablation case before retry: {} {}".format(output_npz, output_json))
    print("DDQN ABLATION | index={} | mode={} | lambda={} | seed={} | smoke={}".format(
        args.index, mode, lam, seed, args.smoke), flush=True)
    from experiments.bratu_ddqn_ablation_worker import main as run_case
    run_case()


if __name__ == "__main__":
    main()
