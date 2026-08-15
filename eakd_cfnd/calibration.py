"""Expected Calibration Error + reliability-diagram data. This is the
calibration validation Reviewer 1 (Major Concern 4) and Reviewer 2 explicitly
asked for and the manuscript's Limitations section (as revised) names as
missing — this module is what closes that gap once run on real predictions.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict:
    """Standard ECE (Guo et al. 2017 binning definition): partition [0,1]
    confidence into n_bins equal-width bins, ECE = sum_b (|B_b|/N) * |acc(b) - conf(b)|.

    Args:
        confidences: per-instance predicted-class probability, shape [N]
        correct: per-instance 0/1 whether the prediction was correct, shape [N]
    Returns dict with "ece" (scalar) and "bins" (per-bin data for the
    reliability diagram: bin edges, accuracy, confidence, count).
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # last bin is closed on both ends so confidence==1.0 is included
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": lo, "hi": hi, "count": 0, "accuracy": None, "confidence": None})
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (count / n) * abs(bin_acc - bin_conf)
        bins.append({"lo": lo, "hi": hi, "count": count, "accuracy": float(bin_acc), "confidence": float(bin_conf)})
    return {"ece": float(ece), "bins": bins, "n_bins": n_bins, "n_instances": n}


def compare_calibration(results_by_signal: dict[str, dict]) -> dict:
    """results_by_signal: {"entropy": {...ece dict...}, "mc_dropout": {...}, "ensemble": {...}}
    Returns a ranked comparison (lower ECE = better calibrated), directly
    answering Reviewer 1 Major Concern 4's "compare entropy/MSP with stronger
    methods like MC Dropout or ensemble-based uncertainty"."""
    ranked = sorted(results_by_signal.items(), key=lambda kv: kv[1]["ece"])
    return {
        "ranking_best_to_worst": [name for name, _ in ranked],
        "ece_by_signal": {name: r["ece"] for name, r in results_by_signal.items()},
    }
