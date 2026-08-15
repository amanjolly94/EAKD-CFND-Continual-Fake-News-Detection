"""Shared helpers for the phase scripts: aggregate seeds into the same
mean+-SD, n=5 format as Table \\ref{tab:performance_revised}, and persist
results as JSON so a later phase (e.g. stats significance) can consume an
earlier phase's output without rerunning anything.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def aggregate_seeds(per_seed_results: list[dict], keys: tuple[str, ...] = ("avg_accuracy", "avg_f1", "bwt", "fwt")) -> dict:
    """per_seed_results: list of the dicts returned by
    train.run_cil_experiment(), one per seed. Returns {key: {"mean", "sd", "n"}}
    — the exact shape stats.significance_table() expects."""
    agg = {}
    n = len(per_seed_results)
    for key in keys:
        values = [r[key] for r in per_seed_results if key in r and r[key] == r[key]]  # drop NaN
        if not values:
            agg[key] = {"mean": None, "sd": None, "n": 0}
            continue
        agg[key] = {
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": n,
        }
    return agg


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_with_checkpoint(ckpt_dir: str | Path, key: str, run_fn):
    """Run run_fn() once, checkpointing its result under ckpt_dir/<key>.json.
    If that checkpoint already exists -- e.g. because a prior Kaggle session
    hit its time limit mid-sweep and this is a resumed re-run of the same
    kernel -- skip re-running and return the cached result instead. Lets any
    loop over (config, seed) units become resumable across session restarts
    just by wrapping each unit's work in this call; no change to the loop
    structure itself."""
    ckpt_file = Path(ckpt_dir) / f"{key}.json"
    if ckpt_file.exists():
        print(f"[resume] {key}: using checkpoint")
        return load_json(ckpt_file)
    result = run_fn()
    save_json(result, ckpt_file)
    return result
