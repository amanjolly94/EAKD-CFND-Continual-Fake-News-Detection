"""Phase 4: hyperparameter sensitivity sweep — Reviewer 1 Major Concern 5 /
Reviewer 2 point 5 / Reviewer 3 point 4: sweep theta_uncertainty and beta,
report accuracy/F1/BWT/API-query-rate/overhead for each setting. Uses fewer
seeds per point (--seeds default 3, not 5) since this is a sweep over many
configurations, not a single headline number — matches how sensitivity
analyses are conventionally scoped in the literature this paper cites.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from eakd_cfnd.config import RunConfig
from eakd_cfnd.data import load_dataset
from eakd_cfnd.train import run_cil_experiment
from scripts.common import aggregate_seeds, run_with_checkpoint, save_json

DEFAULT_THETA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_BETA_GRID = [0.25, 0.5, 1.0, 1.5, 2.0]


def sweep(tasks, dataset, param_name, grid, fixed_theta, fixed_beta, seeds, api_key, ckpt_dir):
    rows = []
    for value in grid:
        theta = value if param_name == "theta_uncertainty" else fixed_theta
        beta = value if param_name == "beta" else fixed_beta
        per_seed = []
        wall_start = time.monotonic()
        for seed in seeds:
            def _run(seed=seed, theta=theta, beta=beta):
                config = RunConfig(dataset=dataset, method="EAKD-CFND", seed=seed,
                                    uncertainty_threshold=theta, beta=beta)
                return run_cil_experiment(tasks, config, api_key=api_key)
            summary = run_with_checkpoint(ckpt_dir, f"{param_name}_{value}_seed{seed}", _run)
            per_seed.append(summary)
        wall_elapsed = time.monotonic() - wall_start
        agg = aggregate_seeds(per_seed)
        api_rates = [r.get("verification_cost", {}).get("api_call_rate") for r in per_seed]
        api_rates = [r for r in api_rates if r is not None]
        rows.append({
            param_name: value,
            "accuracy": agg["avg_accuracy"],
            "f1": agg["avg_f1"],
            "bwt": agg["bwt"],
            "api_call_rate": sum(api_rates) / len(api_rates) if api_rates else None,
            "wall_time_s_per_seed": wall_elapsed / len(seeds),
        })
        print(f"[sensitivity] {param_name}={value} -> "
              f"acc={agg['avg_accuracy']['mean']:.1f} bwt={agg['bwt']['mean']:.1f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FNN-Poli-Time",
                         choices=["PHEME-Event", "FNN-Poli-Time", "FNN-Gossip-Time"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out", default="runs/sensitivity.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 123])
    parser.add_argument("--fixed_theta", type=float, default=0.7)  # matches manuscript's chosen value
    parser.add_argument("--fixed_beta", type=float, default=1.0)
    parser.add_argument("--theta_grid", nargs="+", type=float, default=DEFAULT_THETA_GRID)
    parser.add_argument("--beta_grid", nargs="+", type=float, default=DEFAULT_BETA_GRID)
    parser.add_argument("--fact_check_api_key", default=os.environ.get("GOOGLE_FACT_CHECK_API_KEY"))
    parser.add_argument("--ckpt_dir", default=None,
                         help="per-(param,value,seed) checkpoint dir, resumed automatically "
                              "across killed/restarted Kaggle sessions. Defaults to "
                              "<out's dir>/checkpoints/sensitivity/<dataset>.")
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir or (Path(args.out).parent / "checkpoints" / "sensitivity" / args.dataset)

    tasks = load_dataset(args.dataset, root=args.data_root)

    theta_rows = sweep(tasks, args.dataset, "theta_uncertainty", args.theta_grid,
                        args.fixed_theta, args.fixed_beta, args.seeds, args.fact_check_api_key, ckpt_dir)
    save_json({"dataset": args.dataset, "fixed_theta": args.fixed_theta, "fixed_beta": args.fixed_beta,
                "theta_sweep": theta_rows, "beta_sweep": []}, args.out)

    beta_rows = sweep(tasks, args.dataset, "beta", args.beta_grid,
                       args.fixed_theta, args.fixed_beta, args.seeds, args.fact_check_api_key, ckpt_dir)

    save_json({
        "dataset": args.dataset,
        "fixed_theta": args.fixed_theta,
        "fixed_beta": args.fixed_beta,
        "theta_sweep": theta_rows,
        "beta_sweep": beta_rows,
    }, args.out)


if __name__ == "__main__":
    main()
