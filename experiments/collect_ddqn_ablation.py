"""Validate and collect full 2x4x5 Bratu DDQN ablation results."""
import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.ddqn_ablation_support import ablation_output_paths, decode_array_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--partial", action="store_true", help="only for smoke tests")
    args = parser.parse_args()
    if args.partial and not args.smoke:
        raise RuntimeError("Partial output is only permitted for smoke mode")
    os.environ["BRATU_DDQN_ABLATION_TAG"] = "smoke" if args.smoke else "production"
    rows = []
    missing = []
    import numpy as np
    for i in range(40):
        mode, lam, seed = decode_array_index(i)
        npz_path, json_path = ablation_output_paths(ROOT, mode, lam, seed)
        if not npz_path.exists() or not json_path.exists():
            missing.append(i)
            continue
        with json_path.open() as fh:
            r = json.load(fh)
        expected = (mode, float(lam), int(seed))
        actual = (r["mode"], float(r["lambda"]), int(r["outer_seed"]))
        if expected != actual:
            raise RuntimeError("Metadata mismatch in case {}".format(i))
        with np.load(str(npz_path), allow_pickle=False) as data:
            n = int(r["n_candidates"])
            if len(data["candidate_seed"]) != n:
                raise RuntimeError("Candidate count mismatch in {}".format(i))
            if abs(float(r["relative_l2"]) - float(data["candidate_test_relative_l2"][int(r["selected_candidate"])-1])) > 1e-9:
                raise RuntimeError("Selected error mismatch: {}".format(i))
            if mode == "normalized":
                total = data["alpha"] + data["beta"]
                terminal_total = data["candidate_terminal_alpha"] + data["candidate_terminal_beta"]
                if (float(np.max(np.abs(total - 2))) > 1e-6 or
                        float(np.max(np.abs(terminal_total - 2))) > 1e-6):
                    raise RuntimeError("Normalized weights do not sum to 2: {}".format(i))
            else:
                # No identity claim about the original being normalized.
                pass
            if not args.smoke:
                if n != 5 or r["pinn_steps_per_candidate"] != 15000:
                    raise RuntimeError("Unexpected production training budget: {}".format(i))
                # 5*(15000-200)/20=3700 action calls; replay warm-up 31.
                if int(r["candidate_agent_action_calls_total"]) != 3700 or int(r["candidate_agent_optimizer_updates_total"]) != 3669:
                    raise RuntimeError("DDQN controller update count mismatch: {}".format(i))
                if r["final_replay_size"] != 3700:
                    raise RuntimeError("Replay persistence mismatch: {}".format(i))
        rows.append({
            "index": i, "mode": mode, "lambda": lam, "seed": seed,
            "selected_relative_l2": r["relative_l2"],
            "selected_validation_score": r["validation_score"],
            "selected_candidate": r["selected_candidate"],
            "five_candidate_seconds": r["candidate_train_seconds_total"],
            "agent_action_seconds": r["candidate_agent_action_seconds_total"],
            "agent_update_seconds": r["candidate_agent_update_seconds_total"],
            "agent_optimizer_updates": r["candidate_agent_optimizer_updates_total"],
            "agent_action_calls": r["candidate_agent_action_calls_total"],
            "validation_eval_seconds": r["candidate_validation_seconds_total"],
            "replay_size": r["final_replay_size"], "host": r["host"],
        })
    if missing and not args.partial:
        raise RuntimeError("Missing {} of 40 jobs; indices: {}".format(len(missing), missing))
    if not rows:
        raise RuntimeError("No ablation results to collect")
    fingerprints = set()
    for r in rows:
        _, p = ablation_output_paths(ROOT, r["mode"], r["lambda"], r["seed"])
        with p.open() as handle:
            fingerprints.add(json.load(handle).get("source_sha256"))
    if len(fingerprints) != 1:
        raise RuntimeError("Mixed code versions in ablation results: {}".format(fingerprints))
    root = ROOT / "results" / "ddqn_ablation_v1" / ("smoke" if args.smoke else "production") / "summary"
    root.mkdir(parents=True, exist_ok=True)
    out = root / "all_cases.csv"
    with out.open("w", newline="") as fh:
        writer=csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if len(rows) == 40:
        pairs=[]
        lookup={(x["mode"],x["lambda"],x["seed"]):x for x in rows}
        for lam in (1.0,2.0,3.0,3.4):
            for seed in (42,123,456,789,1000):
                a=lookup[("original_control",lam,seed)]
                b=lookup[("normalized",lam,seed)]
                pairs.append({"lambda":lam,"seed":seed,
                              "original_error":a["selected_relative_l2"],
                              "normalized_error":b["selected_relative_l2"],
                              "original_seconds":a["five_candidate_seconds"],
                              "normalized_seconds":b["five_candidate_seconds"],
                              "original_host":a["host"],"normalized_host":b["host"],
                              "normalized_better":int(b["selected_relative_l2"]<a["selected_relative_l2"]),
                              "same_host":int(a["host"]==b["host"])})
        with (root/"matched_pairs.csv").open("w",newline="") as fh:
            writer=csv.DictWriter(fh,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
        summary=[]
        for mode in ("original_control","normalized"):
            for lam in (1.0,2.0,3.0,3.4):
                subset=[r for r in rows if r["mode"]==mode and r["lambda"]==lam]
                errs=[r["selected_relative_l2"] for r in subset]
                summary.append({"mode":mode,"lambda":lam,"n":len(subset),
                                "mean_error":statistics.mean(errs),
                                "sd_error":statistics.stdev(errs),
                                "median_error":statistics.median(errs),
                                "mean_train_seconds":statistics.mean(r["five_candidate_seconds"] for r in subset),
                                "mean_controller_update_seconds":statistics.mean(r["agent_update_seconds"] for r in subset),
                                "mean_controller_action_seconds":statistics.mean(r["agent_action_seconds"] for r in subset),
                                "mean_controller_updates":statistics.mean(r["agent_optimizer_updates"] for r in subset)})
        with (root/"summary_by_lambda.csv").open("w",newline="") as fh:
            writer=csv.DictWriter(fh,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print("Collected {} / 40; missing {}".format(len(rows), missing))
    print("Summary directory: {}".format(root))


if __name__=="__main__":
    main()
