"""CIL training loop shared by all six methods (see methods.py) — this is
what scripts/run_*.py call. One method-specific `step()` call per batch;
everything else (task sequencing, teacher snapshotting, evaluation) is
identical across methods, so results are only ever different because of the
method, not because of a divergent training procedure — the R1 Major
Concern 7 ask ("all baselines use the same backbone/optimizer/epochs/task
splits") holds by construction here, not just by manual bookkeeping.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import RunConfig
from .data import Task, cumulative_seen_classes
from .evaluate import ResultMatrix
from .methods import build_method
from .verification import ExternalVerifier, VerificationStats


class TextClassificationDataset(Dataset):
    def __init__(self, instances, tokenizer, max_length: int = 128):
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        enc = self.tokenizer(inst.text, truncation=True, padding="max_length",
                              max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": inst.label,
            "veracity": inst.veracity,
            "text": inst.text,
        }


def collate(batch, device):
    input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
    attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)
    labels = torch.tensor([b["label"] for b in batch], device=device)
    veracity = torch.tensor([b["veracity"] for b in batch], device=device)
    texts = [b["text"] for b in batch]
    return {
        "inputs": {"input_ids": input_ids, "attention_mask": attention_mask},
        "labels": labels,
        "veracity": veracity,
        "texts": texts,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_cil_experiment(tasks: list[Task], config: RunConfig, api_key: str | None = None,
                        method=None,
                        device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    """Trains `method` (or the standard method for `config.method` if not
    given) sequentially over `tasks`, matching Problem Formulation
    (subsec:problem): only D_i is available while training on T_i. Passing a
    pre-built `method` lets callers (e.g. the ablation script's RandomAlpha/
    LossBasedAlpha rows, which aren't in the standard METHODS registry) reuse
    this exact loop instead of re-implementing it. Returns the metrics
    summary (evaluate.ResultMatrix.summary()) plus per-method diagnostics
    (e.g. EAKD-CFND's verification cost stats)."""
    set_seed(config.seed)

    total_classes = sorted({c for t in tasks for c in t.classes})
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(total_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    verification_stats = None
    if method is None:
        verifier = None
        if config.method == "EAKD-CFND":
            verification_stats = VerificationStats()
            if api_key:
                verifier = ExternalVerifier(
                    api_key=api_key, theta_uncertainty=config.uncertainty_threshold,
                    stats=verification_stats)
            # if no api_key, EAKD-CFND runs with verifier=None: every high-uncertainty
            # instance falls back to its original label (Algorithm 1's designed
            # behavior when the API path is unavailable) — a legitimate, disclosed
            # partial-mode run, not a silent skip.
        method = build_method(config, verifier=verifier)
    elif getattr(method, "verifier", None) is not None:
        verification_stats = method.verifier.stats

    result_matrix = ResultMatrix(n_tasks=len(tasks))
    prev_model = None

    for task_id, task in enumerate(tasks):
        prior_classes = cumulative_seen_classes(tasks, task_id - 1) if task_id > 0 else []
        method.on_task_start(task_id, model, prev_model)
        if config.method == "LUD" and prev_model is not None:
            calib_loader = DataLoader(
                TextClassificationDataset(task.train, tokenizer), batch_size=config.batch_size,
                collate_fn=lambda b: collate(b, device))
            method.estimate_class_confidence(prev_model, calib_loader, prior_classes)

        train_loader = DataLoader(
            TextClassificationDataset(task.train, tokenizer), batch_size=config.batch_size,
            shuffle=True, collate_fn=lambda b: collate(b, device))

        model.train()
        for epoch in range(config.epochs_per_task):
            for batch in train_loader:
                optimizer.zero_grad()
                loss, _diagnostics = method.step(model, prev_model, batch, prior_classes)
                loss.backward()
                optimizer.step()

        if config.method == "EWC":
            fisher_loader = DataLoader(
                TextClassificationDataset(task.train, tokenizer), batch_size=config.batch_size,
                collate_fn=lambda b: collate(b, device))
            method.consolidate(model, fisher_loader)

        # Evaluate on every task seen so far (fills column task_id of every
        # completed row, and row task_id across all seen tasks).
        model.eval()
        with torch.no_grad():
            for eval_task_id in range(task_id + 1):
                eval_loader = DataLoader(
                    TextClassificationDataset(tasks[eval_task_id].test, tokenizer),
                    batch_size=config.batch_size, collate_fn=lambda b: collate(b, device))
                y_true, y_pred = [], []
                for batch in eval_loader:
                    logits = model(**batch["inputs"]).logits[:, total_classes]
                    preds = torch.tensor(total_classes, device=device)[logits.argmax(dim=-1)]
                    y_true.extend(batch["labels"].tolist())
                    y_pred.extend(preds.tolist())
                result_matrix.record(task_id, eval_task_id, y_true, y_pred, total_classes)

        prev_model = copy.deepcopy(model)
        prev_model.eval()
        for p in prev_model.parameters():
            p.requires_grad = False

    summary = result_matrix.summary()
    summary["run_id"] = config.run_id()
    if verification_stats is not None:
        summary["verification_cost"] = verification_stats.summary()
    return summary
