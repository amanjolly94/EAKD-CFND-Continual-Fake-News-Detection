"""Exploratory phase (not one of the paper's original 7): head-to-head test
of EAKD-CFND, DER, and EAKD-CFND+Replay hybrid variants on PHEME-Event -- the
dataset where the real 1a run (2026-08-13) showed DER at 84.8% accuracy
against EAKD-CFND's 16.8%. Tests whether adding a small DER-style buffer to
EAKD-CFND's existing uncertainty-adaptive KD + verification mechanism closes
that gap, isolating whether the buffer alone explains DER's advantage.

Variant keys accepted (via --variants): "EAKD-CFND", "DER", and
"EAKD-CFND+Replay" optionally suffixed with "+PP" (DER++'s label-replay
term) and/or "+UW" (uncertainty-weighted replay) -- e.g.
"EAKD-CFND+Replay+PP+UW" runs both extensions together.

Usage:
    python -m scripts.run_hybrid_experiment --dataset PHEME-Event --data_root data --out runs/hybrid.json \\
        --variants EAKD-CFND DER "EAKD-CFND+Replay+PP"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from eakd_cfnd.config import DEFAULT_SEEDS, RunConfig
from eakd_cfnd.data import load_dataset
from eakd_cfnd.methods import DER, EAKDCFND, EAKDCFNDReplay
from eakd_cfnd.train import run_cil_experiment
from eakd_cfnd.verification import ExternalVerifier
from scripts.common import aggregate_seeds, run_with_checkpoint, save_json


def build(method_key: str, config: RunConfig, api_key: str | None,
          service_account_file: str | None = None):
    if method_key == "DER":
        return DER(config)
    verifier = None
    if service_account_file:
        verifier = ExternalVerifier(theta_uncertainty=config.uncertainty_threshold,
                                     service_account_file=service_account_file)
    elif api_key:
        verifier = ExternalVerifier(theta_uncertainty=config.uncertainty_threshold, api_key=api_key)
    if method_key == "EAKD-CFND":
        return EAKDCFND(config, verifier=verifier)
    if method_key.startswith("EAKD-CFND+Replay"):
        return EAKDCFNDReplay(config, verifier=verifier,
                               der_plus_plus="+PP" in method_key,
                               uncertainty_weighted_replay="+UW" in method_key)
    raise ValueError(method_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PHEME-Event",
                         choices=["PHEME-Event", "FNN-Poli-Time", "FNN-Gossip-Time"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out", default="runs/hybrid.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", default=["EAKD-CFND", "EAKD-CFND+Replay", "DER"])
    parser.add_argument("--fact_check_api_key", default=os.environ.get("GOOGLE_FACT_CHECK_API_KEY"))
    parser.add_argument("--fact_check_service_account", default=os.environ.get("GOOGLE_FACT_CHECK_SERVICE_ACCOUNT"),
                         help="path to a Google Cloud service-account JSON key; takes precedence "
                              "over --fact_check_api_key if both are set (the static key path is dead).")
    parser.add_argument("--ckpt_dir", default=None,
                         help="per-(method,seed) checkpoint dir, resumed automatically across "
                              "killed/restarted Kaggle sessions. Defaults to <out's dir>/checkpoints/hybrid/<dataset>.")
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir or (Path(args.out).parent / "checkpoints" / "hybrid" / args.dataset)

    tasks = load_dataset(args.dataset, root=args.data_root)
    results = {"dataset": args.dataset, "methods": {}}

    for method_key in args.variants:
        per_seed = []
        for seed in args.seeds:
            def _run(method_key=method_key, seed=seed):
                config = RunConfig(dataset=args.dataset, method="EAKD-CFND", seed=seed)
                method = build(method_key, config, args.fact_check_api_key,
                               args.fact_check_service_account)
                return run_cil_experiment(tasks, config, method=method)
            summary = run_with_checkpoint(ckpt_dir, f"{method_key}_seed{seed}", _run)
            per_seed.append(summary)
            print(f"[hybrid:{args.dataset}] {method_key} seed={seed} -> "
                  f"acc={summary['avg_accuracy']:.1f} bwt={summary['bwt']:.1f}")
        results["methods"][method_key] = {
            "per_seed": per_seed,
            "aggregate": aggregate_seeds(per_seed),
        }
        save_json(results, args.out)


if __name__ == "__main__":
    main()
