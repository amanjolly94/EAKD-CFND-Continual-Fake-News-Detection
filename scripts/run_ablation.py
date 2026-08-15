"""Phase 2: reproduce Table \\ref{tab:ablation} on FNN-Poli-Time (matches the
manuscript's ablation scope exactly — see "Ablation Studies", subsec:ablation).
Configs: Full EAKD-CFND, Standard KD (fixed weight), Random alpha(x),
Loss-based alpha (no uncertainty), No External Verification, Fine-tuning.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from eakd_cfnd.config import DEFAULT_SEEDS, RunConfig
from eakd_cfnd.data import load_dataset
from eakd_cfnd.methods import EAKDCFND, FineTuning, LwF
from eakd_cfnd.train import run_cil_experiment
from eakd_cfnd.verification import ExternalVerifier
from scripts.common import aggregate_seeds, run_with_checkpoint, save_json


def _slug(row_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", row_name).strip("_")


class RandomAlphaEAKD(EAKDCFND):
    """'EAKD w/ Random alpha(x)' ablation row: replace the uncertainty-derived
    weight with a uniform-random one, isolating whether the SIGNAL matters or
    any per-instance variation would do."""
    def step(self, model, prev_model, batch, prior_task_classes):
        from eakd_cfnd.methods import kd_loss
        logits = model(**batch["inputs"]).logits
        if prev_model is None or not prior_task_classes:
            return F.cross_entropy(logits, batch["labels"]), {}
        with torch.no_grad():
            teacher_logits = prev_model(**batch["inputs"]).logits[:, prior_task_classes]
        alpha = torch.rand(logits.shape[0], device=logits.device)
        task_loss = F.cross_entropy(logits, batch["labels"], reduction="none")
        loss_kd = kd_loss(logits[:, prior_task_classes], teacher_logits, self.config.kd_temperature)
        return (task_loss + alpha * loss_kd).mean(), {}


class LossBasedAlphaEAKD(EAKDCFND):
    """'EAKD w/o Uncertainty (Loss-based alpha)': weight by the teacher's own
    task loss instead of its output-distribution uncertainty — tests whether
    ANY per-instance signal works as well as the specific uncertainty signal."""
    def step(self, model, prev_model, batch, prior_task_classes):
        from eakd_cfnd.methods import kd_loss
        from eakd_cfnd import uncertainty as unc
        logits = model(**batch["inputs"]).logits
        if prev_model is None or not prior_task_classes:
            return F.cross_entropy(logits, batch["labels"]), {}
        with torch.no_grad():
            # Full (unsliced) class space here: batch["labels"] are the CURRENT
            # task's global class ids, which aren't among prior_task_classes, so
            # cross_entropy against a prior_task_classes-sliced teacher would
            # index out of range (this crashed with a CUDA device-side assert
            # during phase-2 row 4 of a real Kaggle run). The prior_task_classes
            # slice is still used below, but only for the KD term, matching
            # EAKDCFND.step's shape-matched student/teacher comparison.
            teacher_logits_full = prev_model(**batch["inputs"]).logits
            teacher_logits_prior = teacher_logits_full[:, prior_task_classes]
            teacher_loss_per_instance = F.cross_entropy(teacher_logits_full, batch["labels"], reduction="none")
        self._update_running_bounds(teacher_loss_per_instance)
        norm = unc.normalize(teacher_loss_per_instance, self._running_min, self._running_max)
        alpha = unc.adaptive_weight(norm, self.config.beta)
        task_loss = F.cross_entropy(logits, batch["labels"], reduction="none")
        loss_kd = kd_loss(logits[:, prior_task_classes], teacher_logits_prior, self.config.kd_temperature)
        return (task_loss + alpha * loss_kd).mean(), {}


ABLATION_CONFIGS = {
    "EAKD-CFND (Full)": ("EAKD-CFND", True),           # (method_key, use_verification)
    "Standard KD (lambda=1)": ("LwF", False),
    "EAKD w/ Random alpha(x)": ("RandomAlpha", False),
    "EAKD w/o Uncertainty (Loss-based alpha)": ("LossBasedAlpha", False),
    "EAKD w/o External Verification": ("EAKD-CFND", False),
    "Fine-tuning (No KD)": ("FT", False),
}


def build_method_for_row(row_name: str, config: RunConfig, api_key: str | None):
    key, use_verification = ABLATION_CONFIGS[row_name]
    verifier = None
    if use_verification and api_key:
        verifier = ExternalVerifier(api_key=api_key, theta_uncertainty=config.uncertainty_threshold)
    if key == "EAKD-CFND":
        return EAKDCFND(config, verifier=verifier)
    if key == "RandomAlpha":
        return RandomAlphaEAKD(config, verifier=None)
    if key == "LossBasedAlpha":
        return LossBasedAlphaEAKD(config, verifier=None)
    if key == "LwF":
        return LwF(config)
    if key == "FT":
        return FineTuning(config)
    raise ValueError(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out", default="runs/ablation.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--fact_check_api_key", default=os.environ.get("GOOGLE_FACT_CHECK_API_KEY"))
    parser.add_argument("--ckpt_dir", default=None,
                         help="per-(row,seed) checkpoint dir, resumed automatically across "
                              "killed/restarted Kaggle sessions. Defaults to <out's dir>/checkpoints/ablation.")
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir or (Path(args.out).parent / "checkpoints" / "ablation")

    tasks = load_dataset("FNN-Poli-Time", root=args.data_root)
    results = {"dataset": "FNN-Poli-Time", "rows": {}}

    for row_name in ABLATION_CONFIGS:
        per_seed = []
        for seed in args.seeds:
            def _run(row_name=row_name, seed=seed):
                config = RunConfig(dataset="FNN-Poli-Time", method="EAKD-CFND", seed=seed)
                method = build_method_for_row(row_name, config, args.fact_check_api_key)
                return run_cil_experiment(tasks, config, method=method)
            summary = run_with_checkpoint(ckpt_dir, f"{_slug(row_name)}_seed{seed}", _run)
            per_seed.append(summary)
            print(f"[ablation] {row_name} seed={seed} -> acc={summary['avg_accuracy']:.1f} bwt={summary['bwt']:.1f}")
        results["rows"][row_name] = {"per_seed": per_seed, "aggregate": aggregate_seeds(per_seed)}
        save_json(results, args.out)


if __name__ == "__main__":
    main()
