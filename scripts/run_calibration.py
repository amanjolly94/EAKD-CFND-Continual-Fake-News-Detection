"""Phase 3: calibration validation — Reviewer 1 Major Concern 4 / Reviewer 2
point 6. Runs EAKD-CFND with each of the four uncertainty signals
(entropy, msp, mc_dropout, ensemble), collects (confidence, correct) pairs
on held-out data, and computes ECE + a ranked comparison. This is the exact
gap the manuscript's Limitations section names as open.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eakd_cfnd import uncertainty as unc
from eakd_cfnd.calibration import compare_calibration, expected_calibration_error
from eakd_cfnd.config import DEFAULT_SEEDS, RunConfig
from eakd_cfnd.data import cumulative_seen_classes, load_dataset
from eakd_cfnd.train import TextClassificationDataset, collate, set_seed
from scripts.common import run_with_checkpoint, save_json

SIGNALS = ("entropy", "msp", "mc_dropout", "ensemble")


def train_teacher(tasks, config, device):
    """Train a plain fine-tuned model through all-but-the-last task, to play
    the role of 'teacher' whose calibration we're measuring — mirrors how
    omega(x) is always computed from a teacher in the main method."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    total_classes = sorted({c for t in tasks for c in t.classes})
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(total_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    for task in tasks[:-1]:
        loader = DataLoader(TextClassificationDataset(task.train, tokenizer),
                             batch_size=config.batch_size, shuffle=True,
                             collate_fn=lambda b: collate(b, device))
        for _ in range(config.epochs_per_task):
            for batch in loader:
                optimizer.zero_grad()
                logits = model(**batch["inputs"]).logits
                loss = F.cross_entropy(logits, batch["labels"])
                loss.backward()
                optimizer.step()
    model.eval()
    return model, tokenizer, total_classes


def ensemble_of(base_config, tasks, device, size):
    models = []
    for i in range(size):
        cfg = copy.deepcopy(base_config)
        cfg.seed = base_config.seed * 1000 + i  # deterministic but distinct per-member seed
        m, _, _ = train_teacher(tasks, cfg, device)
        models.append(m)
    return models


def evaluate_signal(signal, model, tokenizer, total_classes, held_out_task, prior_classes,
                     config, device, ensemble_models=None):
    loader = DataLoader(TextClassificationDataset(held_out_task.test, tokenizer),
                         batch_size=config.batch_size, collate_fn=lambda b: collate(b, device))
    confidences, correct = [], []
    for batch in loader:
        if signal == "mc_dropout":
            _var, mean_probs = unc.mc_dropout_uncertainty(model, batch["inputs"], n_passes=config.mc_dropout_passes)
            probs = mean_probs
        elif signal == "ensemble":
            _var, mean_probs = unc.ensemble_uncertainty(ensemble_models, batch["inputs"])
            probs = mean_probs
        else:
            with torch.no_grad():
                logits = model(**batch["inputs"]).logits[:, prior_classes]
            probs = F.softmax(logits, dim=-1)
        conf, pred_idx = probs.max(dim=-1)
        preds = torch.tensor(prior_classes, device=device)[pred_idx]
        confidences.extend(conf.tolist())
        correct.extend((preds == batch["labels"]).long().tolist())
    return expected_calibration_error(confidences, correct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FNN-Poli-Time",
                         choices=["PHEME-Event", "FNN-Poli-Time", "FNN-Gossip-Time"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out", default="runs/calibration.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--ckpt_dir", default=None,
                         help="per-seed checkpoint dir (each seed trains a teacher + ensemble "
                              "and evaluates all 4 signals as one unit), resumed automatically "
                              "across killed/restarted Kaggle sessions. Defaults to "
                              "<out's dir>/checkpoints/calibration/<dataset>.")
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir or (Path(args.out).parent / "checkpoints" / "calibration" / args.dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = load_dataset(args.dataset, root=args.data_root)
    prior_classes = cumulative_seen_classes(tasks, len(tasks) - 2)  # all classes before the held-out (last) task
    held_out_task = tasks[-1]

    per_seed_results = {signal: [] for signal in SIGNALS}
    for seed in args.seeds:
        def _run(seed=seed):
            set_seed(seed)
            config = RunConfig(dataset=args.dataset, method="EAKD-CFND", seed=seed)
            model, tokenizer, total_classes = train_teacher(tasks, config, device)
            ensemble_models = ensemble_of(config, tasks, device, config.ensemble_size)
            return {
                signal: evaluate_signal(signal, model, tokenizer, total_classes, held_out_task,
                                         prior_classes, config, device, ensemble_models)["ece"]
                for signal in SIGNALS
            }
        seed_eces = run_with_checkpoint(ckpt_dir, f"seed{seed}", _run)
        for signal in SIGNALS:
            per_seed_results[signal].append(seed_eces[signal])
            print(f"[calibration] seed={seed} signal={signal} ECE={seed_eces[signal]:.4f}")

    aggregate = {
        signal: {"ece": sum(v) / len(v)} for signal, v in per_seed_results.items() if v
    }
    save_json({
        "dataset": args.dataset,
        "per_seed_ece": per_seed_results,
        "aggregate_ece": aggregate,
        "ranking": compare_calibration(aggregate),
    }, args.out)


if __name__ == "__main__":
    main()
