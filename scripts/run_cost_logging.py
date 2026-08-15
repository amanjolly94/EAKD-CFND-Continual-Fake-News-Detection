"""Phase 5a: quantitative cost/scalability accounting — Reviewer 1 Major
Concern 6, Reviewer 2 points 2 and 7, Reviewer 3 point 5. Measures, for every
method in Table \\ref{tab:performance_revised}: wall-clock training time,
inference latency per instance, peak GPU memory; and for EAKD-CFND
specifically, the External Verification cost breakdown (API call rate,
latency, failure rate) via verification.VerificationStats — the exact
numbers the manuscript's Efficiency Analysis section currently only
describes qualitatively.

NOTE on scope: this script does NOT add new baseline methods beyond the six
already in Table \\ref{tab:performance_revised} (see the reviewer response
letter — implementing an additional adaptive-KD baseline correctly enough to
trust is a separate, larger piece of work than instrumenting the existing
six, and a half-verified new baseline is worse than none).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eakd_cfnd.config import METHODS, RunConfig
from eakd_cfnd.data import cumulative_seen_classes, load_dataset
from eakd_cfnd.methods import build_method
from eakd_cfnd.train import TextClassificationDataset, collate, set_seed
from eakd_cfnd.verification import ExternalVerifier
from scripts.common import run_with_checkpoint, save_json


def measure_method(method_name, tasks, config, device, api_key):
    set_seed(config.seed)
    total_classes = sorted({c for t in tasks for c in t.classes})
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(total_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    verifier = None
    if method_name == "EAKD-CFND" and api_key:
        verifier = ExternalVerifier(api_key=api_key, theta_uncertainty=config.uncertainty_threshold)
    method = build_method(config, verifier=verifier)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_start = time.monotonic()
    prev_model = None
    for task_id, task in enumerate(tasks):
        prior_classes = cumulative_seen_classes(tasks, task_id - 1) if task_id > 0 else []
        loader = DataLoader(TextClassificationDataset(task.train, tokenizer),
                             batch_size=config.batch_size, shuffle=True,
                             collate_fn=lambda b: collate(b, device))
        model.train()
        for _ in range(config.epochs_per_task):
            for batch in loader:
                optimizer.zero_grad()
                loss, _ = method.step(model, prev_model, batch, prior_classes)
                loss.backward()
                optimizer.step()
        import copy
        prev_model = copy.deepcopy(model)
        prev_model.eval()
        for p in prev_model.parameters():
            p.requires_grad = False
    train_time_s = time.monotonic() - train_start

    # Inference latency: single-instance forward passes, matching how a
    # streaming deployment would actually call the model (one item at a time).
    model.eval()
    inference_loader = DataLoader(TextClassificationDataset(tasks[-1].test, tokenizer),
                                   batch_size=1, collate_fn=lambda b: collate(b, device))
    latencies = []
    with torch.no_grad():
        for batch in inference_loader:
            start = time.monotonic()
            model(**batch["inputs"])
            latencies.append(time.monotonic() - start)

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else None

    result = {
        "method": method_name,
        "train_time_s": train_time_s,
        "mean_inference_latency_ms": (sum(latencies) / len(latencies)) * 1000 if latencies else None,
        "peak_gpu_memory_mb": peak_memory_mb,
    }
    if verifier is not None:
        result["verification_cost"] = verifier.stats.summary()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FNN-Poli-Time",
                         choices=["PHEME-Event", "FNN-Poli-Time", "FNN-Gossip-Time"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out", default="runs/cost_logging.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fact_check_api_key", default=os.environ.get("GOOGLE_FACT_CHECK_API_KEY"))
    parser.add_argument("--ckpt_dir", default=None,
                         help="per-method checkpoint dir, resumed automatically across "
                              "killed/restarted Kaggle sessions. Defaults to "
                              "<out's dir>/checkpoints/cost_logging/<dataset>.")
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir or (Path(args.out).parent / "checkpoints" / "cost_logging" / args.dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = load_dataset(args.dataset, root=args.data_root)

    results = []
    for method_name in METHODS:
        def _run(method_name=method_name):
            config = RunConfig(dataset=args.dataset, method=method_name, seed=args.seed)
            return measure_method(method_name, tasks, config, device, args.fact_check_api_key)
        r = run_with_checkpoint(ckpt_dir, method_name, _run)
        results.append(r)
        print(f"[cost] {method_name}: train={r['train_time_s']:.1f}s "
              f"infer={r['mean_inference_latency_ms']:.1f}ms/instance")
        save_json({"dataset": args.dataset, "device": device, "results": results}, args.out)


if __name__ == "__main__":
    main()
