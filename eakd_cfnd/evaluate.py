"""Metrics matching Evaluation Metrics (subsec:metrics): Average Accuracy,
Average F1 (macro), Backward Transfer, Forward Transfer. Standard
Lopez-Paz & Ranzato (2017) formulas, since the manuscript names the metrics
but doesn't spell out the formula — this is the standard CL definition and
is what any reviewer would assume "BWT"/"FWT" means without further
qualification.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


class ResultMatrix:
    """R[i][j] = metric evaluated on task j's test set, after training
    through task i (i.e. the model has seen tasks 0..i). Both accuracy and
    macro-F1 are tracked in parallel matrices."""

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        self.acc = np.full((n_tasks, n_tasks), np.nan)
        self.f1 = np.full((n_tasks, n_tasks), np.nan)
        self.baseline_acc = np.full(n_tasks, np.nan)  # random-init accuracy on task j, for FWT

    def record(self, after_task: int, eval_task: int, y_true, y_pred, all_classes: list[int]):
        self.acc[after_task, eval_task] = float(np.mean(np.array(y_true) == np.array(y_pred)))
        self.f1[after_task, eval_task] = f1_score(
            y_true, y_pred, labels=all_classes, average="macro", zero_division=0)

    def record_baseline(self, eval_task: int, y_true, y_pred):
        """Accuracy of an untrained/random model on task j — used as the FWT
        reference point (Lopez-Paz & Ranzato's b_i)."""
        self.baseline_acc[eval_task] = float(np.mean(np.array(y_true) == np.array(y_pred)))

    def average_accuracy(self) -> float:
        """ACC = (1/T) sum_j R[T-1, j] — final-model accuracy averaged over all tasks."""
        final_row = self.acc[self.n_tasks - 1]
        return float(np.nanmean(final_row))

    def average_f1(self) -> float:
        final_row = self.f1[self.n_tasks - 1]
        return float(np.nanmean(final_row))

    def backward_transfer(self) -> float:
        """BWT = 1/(T-1) * sum_{i=0}^{T-2} (R[T-1, i] - R[i, i]).
        Negative = forgetting (final performance on task i is worse than
        right after learning it); this matches the sign convention already
        used in the manuscript (e.g. "-25.6% under naive fine-tuning")."""
        T = self.n_tasks
        if T < 2:
            return 0.0
        terms = [self.acc[T - 1, i] - self.acc[i, i] for i in range(T - 1)]
        return float(np.nanmean(terms)) * 100.0  # manuscript reports BWT as a percentage

    def forward_transfer(self) -> float:
        """FWT = 1/(T-1) * sum_{i=1}^{T-1} (R[i-1, i] - b_i), where b_i is the
        baseline (random-init) accuracy on task i. Requires record_baseline()
        to have been called for tasks 1..T-1."""
        T = self.n_tasks
        if T < 2:
            return 0.0
        terms = []
        for i in range(1, T):
            if not np.isnan(self.baseline_acc[i]):
                terms.append(self.acc[i - 1, i] - self.baseline_acc[i])
        return float(np.nanmean(terms)) * 100.0 if terms else float("nan")

    def summary(self) -> dict:
        return {
            "avg_accuracy": self.average_accuracy() * 100.0,
            "avg_f1": self.average_f1() * 100.0,
            "bwt": self.backward_transfer(),
            "fwt": self.forward_transfer(),
        }
