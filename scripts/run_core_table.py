"""Phase 1: reproduce Table \\ref{tab:performance_revised} — all 6 methods on
one dataset, 5 seeds each. This is the "Kaggle account N runs 1 dataset"
job: pass --dataset to scope it, so accounts 1/2/3 each run this script with
a different --dataset and the results merge back into one table afterward.

Usage (per account):
    python -m scripts.run_core_table --dataset PHEME-Event --data_root data --out runs/core_pheme.json
    python -m scripts.run_core_table --dataset FNN-Poli-Time --data_root data --out runs/core_poli.json
    python -m scripts.run_core_table --dataset FNN-Gossip-Time --data_root data --out runs/core_gossip.json
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from eakd_cfnd.config import DEFAULT_SEEDS, METHODS, RunConfig
from eakd_cfnd.data import load_dataset
from eakd_cfnd.train import run_cil_experiment
from scripts.common import aggregate_seeds, run_with_checkpoint, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["PHEME-Event", "FNN-Poli-Time", "FNN-Gossip-Time"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--fact_check_api_key", default=os.environ.get("GOOGLE_FACT_CHECK_API_KEY"))
    parser.add_argument("--ckpt_dir", default=None,
                         help="per-(method,seed) checkpoint dir, resumed automatically if it "
                              "already has results (e.g. from a prior Kaggle session that hit "
                              "its time limit). Defaults to <out's dir>/checkpoints/<dataset>.")
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir or (Path(args.out).parent / "checkpoints" / args.dataset)

    tasks = load_dataset(args.dataset, root=args.data_root)
    results = {"dataset": args.dataset, "methods": {}}

    for method in METHODS:
        per_seed = []
        for seed in args.seeds:
            def _run(method=method, seed=seed):
                config = RunConfig(dataset=args.dataset, method=method, seed=seed)
                return run_cil_experiment(tasks, config, api_key=args.fact_check_api_key)
            summary = run_with_checkpoint(ckpt_dir, f"{method}_seed{seed}", _run)
            per_seed.append(summary)
            print(f"[{args.dataset}] {method} seed={seed} -> "
                  f"acc={summary['avg_accuracy']:.1f} bwt={summary['bwt']:.1f}")
        results["methods"][method] = {
            "per_seed": per_seed,
            "aggregate": aggregate_seeds(per_seed),
        }
        save_json(results, args.out)  # re-save after every method so a killed session leaves a usable partial table


if __name__ == "__main__":
    main()
