"""Significance testing. welch_t_test reproduces exactly the computation
already added to the manuscript's new "Statistical Significance" section
(Table \\ref{tab:significance}) — same formula, so results from real reruns
are directly comparable to what's already published. paired_t_test is added
for when real per-seed runs (not just aggregate mean/SD) are available,
which is a strictly stronger test than the unpaired approximation the
manuscript had to use.
"""
from __future__ import annotations

import math

from scipy import stats as scipy_stats


def welch_t_test(mean1: float, sd1: float, n1: int, mean2: float, sd2: float, n2: int) -> dict:
    """Welch's two-sample t-test from summary statistics only (mean, SD, n
    per group) — the same computation used in the manuscript when per-seed
    values weren't individually retained."""
    se = math.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
    if se == 0:
        raise ValueError("Standard error is zero; cannot compute t-statistic (identical, zero-variance groups?)")
    t = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (sd1 ** 2 / n1 + sd2 ** 2 / n2) ** 2
    denom = (sd1 ** 2 / n1) ** 2 / (n1 - 1) + (sd2 ** 2 / n2) ** 2 / (n2 - 1)
    df = num / denom

    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t), df))
    return {"t": t, "df": df, "p_value": p_value, "delta": mean1 - mean2}


def paired_t_test(values1: list[float], values2: list[float]) -> dict:
    """Paired t-test on matched per-seed runs — use this instead of
    welch_t_test whenever raw per-seed results are actually available (i.e.
    the same 5 seed indices were used for both methods), since it's strictly
    more powerful than the unpaired approximation."""
    if len(values1) != len(values2):
        raise ValueError("paired_t_test requires equal-length, seed-matched lists")
    t, p_value = scipy_stats.ttest_rel(values1, values2)
    return {"t": float(t), "df": len(values1) - 1, "p_value": float(p_value),
            "delta": sum(values1) / len(values1) - sum(values2) / len(values2)}


def significance_table(performance: dict, comparisons: list[tuple[str, str]], dataset: str) -> list[dict]:
    """performance: {method_name: {"mean": float, "sd": float, "n": int}, ...} for one dataset.
    comparisons: [("EAKD-CFND", "DER"), ("EAKD-CFND", "LUD"), ...]
    Reproduces the exact table structure used in the manuscript
    (tab:significance), so this can regenerate/extend it once real per-seed
    or new-baseline results are available.
    """
    rows = []
    for a, b in comparisons:
        pa, pb = performance[a], performance[b]
        result = welch_t_test(pa["mean"], pa["sd"], pa["n"], pb["mean"], pb["sd"], pb["n"])
        rows.append({
            "comparison": f"{a} vs. {b}",
            "dataset": dataset,
            "delta_pp": round(result["delta"], 1),
            "t": round(result["t"], 2),
            "df": round(result["df"], 1),
            "p_value": result["p_value"],
        })
    return rows
