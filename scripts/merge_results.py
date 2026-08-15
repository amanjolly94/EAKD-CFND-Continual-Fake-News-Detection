"""Merge real experiment output (runs/*.json from the phase scripts) into the
manuscript's LaTeX table format.

Three tables already exist in main_minor_revison1.tex with placeholder rows
to replace outright: tab:performance_revised, tab:significance, tab:ablation.
Three more phases (calibration, sensitivity, cost logging) have NO table
skeleton yet -- they're currently prose-only Limitations/Future-Work bullets
(see main_minor_revison1.tex lines ~516-530) -- so this script emits new,
self-contained table snippets for those instead of trying to slot numbers
into a table that doesn't exist. Deciding where in the manuscript those new
tables belong, and rewriting the surrounding prose, is a content decision
left to a human editing pass; this script only produces correct numbers in
ready-to-paste LaTeX.

Usage:
    python -m scripts.merge_results --runs_dir runs --out runs/merged_tables.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eakd_cfnd.stats import paired_t_test, welch_t_test
from scripts.common import load_json

DATASET_FILES = {
    "PHEME-Event": "core_pheme.json",
    "FNN-Poli-Time": "core_poli.json",
    "FNN-Gossip-Time": "core_gossip.json",
}
METHOD_ORDER = ["FT", "EWC", "LwF", "DER", "LUD", "EAKD-CFND"]
METHOD_LABEL = {
    "FT": "Fine-tuning (FT)", "EWC": "EWC", "LwF": "LwF",
    "DER": "DER (Buffer=200)", "LUD": "LUD", "EAKD-CFND": "\\textbf{EAKD-CFND (Ours)}",
}
ABLATION_ROW_ORDER = [
    "EAKD-CFND (Full)", "Standard KD (lambda=1)", "EAKD w/ Random alpha(x)",
    "EAKD w/o Uncertainty (Loss-based alpha)", "EAKD w/o External Verification",
    "Fine-tuning (No KD)",
]
ABLATION_ROW_LABEL = {
    "EAKD-CFND (Full)": "\\textbf{EAKD-CFND (Full)}",
    "Standard KD (lambda=1)": "Standard KD ($\\lambda=1$)",
    "EAKD w/ Random alpha(x)": "EAKD w/ Random $\\alpha(x)$",
    "EAKD w/o Uncertainty (Loss-based alpha)": "EAKD w/o Uncertainty (Loss-based $\\alpha$)",
    "EAKD w/o External Verification": "EAKD w/o External Verification",
    "Fine-tuning (No KD)": "Fine-tuning (No KD)",
}


def _fmt(mean: float, sd: float, bold: bool = False) -> str:
    s = f"{mean:.1f} $\\pm$ {sd:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def load_core_tables(runs_dir: Path) -> dict[str, dict]:
    """dataset -> method -> {"per_seed": [...], "aggregate": {...}}, for
    whichever of the three core-table phases (1a/1b/1c) have actually run."""
    out = {}
    for dataset, fname in DATASET_FILES.items():
        p = runs_dir / fname
        if p.exists():
            out[dataset] = load_json(p)["methods"]
    return out


def performance_table_rows(core: dict[str, dict]) -> list[str]:
    rows = []
    for method in METHOD_ORDER:
        label = METHOD_LABEL[method]
        cells = []
        for dataset in DATASET_FILES:
            if dataset not in core or method not in core[dataset]:
                continue
            agg = core[dataset][method]["aggregate"]
            bold = method == "EAKD-CFND"
            acc = _fmt(agg["avg_accuracy"]["mean"], agg["avg_accuracy"]["sd"], bold)
            f1 = _fmt(agg["avg_f1"]["mean"], agg["avg_f1"]["sd"], bold)
            bwt = _fmt(agg["bwt"]["mean"], agg["bwt"]["sd"], bold)
            fwt = _fmt(agg["fwt"]["mean"], agg["fwt"]["sd"], bold) if agg["fwt"]["mean"] is not None else "--"
            cells.append(f"& {dataset} & {acc} & {f1} & {bwt} & {fwt} \\\\")
        if not cells:
            continue
        rows.append(f"\\multirow{{{len(cells)}}}{{*}}{{{label}}} " + cells[0])
        rows.extend(cells[1:])
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    return rows


def significance_rows(core: dict[str, dict]) -> list[str]:
    """Real per-seed values are available now, so this uses paired_t_test
    (seed-matched) instead of the manuscript's original welch_t_test
    (unpaired, summary-stats-only) -- strictly more powerful given the same
    5 seed indices were used for every method. Falls back to Welch's test
    per-dataset if per-seed lists are ever missing or length-mismatched."""
    rows = []
    for dataset in DATASET_FILES:
        if dataset not in core or "EAKD-CFND" not in core[dataset]:
            continue
        for baseline in ("DER", "LUD"):
            if baseline not in core[dataset]:
                continue
            ours = core[dataset]["EAKD-CFND"]
            base = core[dataset][baseline]
            ours_acc = [r["avg_accuracy"] for r in ours["per_seed"]]
            base_acc = [r["avg_accuracy"] for r in base["per_seed"]]
            if len(ours_acc) == len(base_acc) and len(ours_acc) > 1:
                result = paired_t_test(ours_acc, base_acc)
            else:
                oa, ba = ours["aggregate"]["avg_accuracy"], base["aggregate"]["avg_accuracy"]
                result = welch_t_test(oa["mean"], oa["sd"], oa["n"], ba["mean"], ba["sd"], ba["n"])
            p = result["p_value"]
            p_str = f"$p<{0.001:.3f}$" if p < 0.001 else f"$p<{p:.3f}$"
            rows.append(
                f"EAKD-CFND vs.\\ {baseline} & {dataset} & "
                f"{result['delta']:+.1f} & {result['t']:.2f} & {p_str} \\\\"
            )
    return rows


def ablation_table_rows(ablation: dict | None) -> list[str]:
    if not ablation:
        return []
    rows = []
    for row_name in ABLATION_ROW_ORDER:
        if row_name not in ablation["rows"]:
            continue
        agg = ablation["rows"][row_name]["aggregate"]
        bold = row_name == "EAKD-CFND (Full)"
        acc = _fmt(agg["avg_accuracy"]["mean"], agg["avg_accuracy"]["sd"], bold)
        bwt = _fmt(agg["bwt"]["mean"], agg["bwt"]["sd"], bold)
        rows.append(f"{ABLATION_ROW_LABEL[row_name]} & {acc} & {bwt} \\\\")
    return rows


def calibration_table_snippet(calibration: dict | None) -> str | None:
    if not calibration:
        return None
    lines = [
        "% NEW TABLE -- no existing skeleton; closes R1 Major Concern 4 / R1 Minor",
        "% Concern (calibration validation) -- currently a Limitations bullet",
        "% (main_minor_revison1.tex ~line 516). Needs manual placement + a",
        "% sentence replacing/updating that bullet once inserted.",
        "\\begin{table}[ht!]",
        "\\centering",
        f"\\caption{{Expected Calibration Error (ECE) by Uncertainty Signal on {calibration['dataset']}}}\\label{{tab:calibration}}",
        "\\begin{tabular}{@{}lc@{}}",
        "\\toprule",
        "\\textbf{Uncertainty Signal} & \\textbf{ECE} \\\\",
        "\\midrule",
    ]
    ranking = calibration.get("ranking", {}).get("ranking_best_to_worst", list(calibration["aggregate_ece"]))
    for signal in ranking:
        ece = calibration["aggregate_ece"][signal]["ece"]
        bold = signal == ranking[0]
        val = f"\\textbf{{{ece:.3f}}}" if bold else f"{ece:.3f}"
        signal_label = {"entropy": "Entropy", "msp": "Max Softmax Prob.",
                         "mc_dropout": "MC Dropout", "ensemble": "Deep Ensemble"}.get(signal, signal)
        lines.append(f"{signal_label} & {val} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def sensitivity_table_snippet(sensitivity: dict | None) -> str | None:
    if not sensitivity:
        return None
    lines = [
        "% NEW TABLE -- no existing skeleton; closes R1 Major Concern 5 / R3",
        "% sensitivity ask -- currently a Limitations bullet",
        "% (main_minor_revison1.tex ~line 517).",
        "\\begin{table}[ht!]",
        "\\centering",
        f"\\caption{{Hyperparameter Sensitivity on {sensitivity['dataset']} "
        f"(fixed $\\theta_{{uncertainty}}={sensitivity['fixed_theta']}$, $\\beta={sensitivity['fixed_beta']}$ "
        "unless swept)}\\label{tab:sensitivity}",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "\\textbf{Param} & \\textbf{Value} & \\textbf{Acc.\\ (\\%)} & \\textbf{F1 (\\%)} & \\textbf{BWT (\\%)} & \\textbf{API Rate} \\\\",
        "\\midrule",
    ]
    for sweep_name, rows in (("$\\theta_{uncertainty}$", sensitivity["theta_sweep"]),
                              ("$\\beta$", sensitivity["beta_sweep"])):
        for r in rows:
            value = r.get("theta_uncertainty", r.get("beta"))
            api_rate = r["api_call_rate"]
            api_str = f"{api_rate:.2f}" if api_rate is not None else "--"
            lines.append(
                f"{sweep_name} & {value} & {r['accuracy']['mean']:.1f} & "
                f"{r['f1']['mean']:.1f} & {r['bwt']['mean']:.1f} & {api_str} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def cost_logging_table_snippet(cost: dict | None) -> str | None:
    if not cost:
        return None
    lines = [
        "% NEW TABLE -- no existing skeleton; closes R1 Major Concern 6 / R2",
        "% points 2 and 7 / R3 scalability ask -- currently a Limitations",
        "% bullet (main_minor_revison1.tex ~line 518).",
        "\\begin{table}[ht!]",
        "\\centering",
        f"\\caption{{Training/Inference Cost and External Verification Overhead on {cost['dataset']}}}\\label{{tab:cost}}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Train (s)} & \\textbf{Infer (ms/inst.)} & \\textbf{Peak Mem (MB)} & \\textbf{API Call Rate} \\\\",
        "\\midrule",
    ]
    for r in cost["results"]:
        vc = r.get("verification_cost")
        api_str = f"{vc['api_call_rate']:.2f}" if vc else "--"
        mem = f"{r['peak_gpu_memory_mb']:.0f}" if r["peak_gpu_memory_mb"] is not None else "--"
        lines.append(
            f"{r['method']} & {r['train_time_s']:.1f} & {r['mean_inference_latency_ms']:.2f} & "
            f"{mem} & {api_str} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    if cost["results"] and cost["results"][-1].get("verification_cost"):
        vc = cost["results"][-1]["verification_cost"]
        lines.append(
            f"% EAKD-CFND verification detail: {vc['n_api_calls']} calls / "
            f"{vc['n_instances']} instances, failure_rate={vc['failure_rate']:.3f}, "
            f"mean_latency_s={vc['mean_latency_s']}"
        )
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--out", default="runs/merged_tables.tex")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)

    core = load_core_tables(runs_dir)
    ablation = load_json(runs_dir / "ablation.json") if (runs_dir / "ablation.json").exists() else None
    calibration = load_json(runs_dir / "calibration.json") if (runs_dir / "calibration.json").exists() else None
    sensitivity = load_json(runs_dir / "sensitivity.json") if (runs_dir / "sensitivity.json").exists() else None
    cost = load_json(runs_dir / "cost_logging.json") if (runs_dir / "cost_logging.json").exists() else None

    out_parts = []

    out_parts.append("% ===== tab:performance_revised rows (replace existing \\midrule-separated rows) =====")
    out_parts.extend(performance_table_rows(core))

    out_parts.append("\n% ===== tab:significance rows (replace existing rows; now paired/seed-matched, "
                      "not the manuscript's original unpaired Welch approximation) =====")
    out_parts.extend(significance_rows(core))

    if ablation:
        out_parts.append("\n% ===== tab:ablation rows (replace existing rows) =====")
        out_parts.extend(ablation_table_rows(ablation))

    for snippet in (calibration_table_snippet(calibration),
                    sensitivity_table_snippet(sensitivity),
                    cost_logging_table_snippet(cost)):
        if snippet:
            out_parts.append("\n" + snippet)

    out_text = "\n".join(out_parts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(out_text, encoding="utf-8")
    print(f"wrote {args.out}")

    missing = [name for name, present in [
        ("1a PHEME-Event", "PHEME-Event" in core), ("1b FNN-Poli-Time", "FNN-Poli-Time" in core),
        ("1c FNN-Gossip-Time", "FNN-Gossip-Time" in core), ("ablation", ablation is not None),
        ("calibration", calibration is not None), ("sensitivity", sensitivity is not None),
        ("cost_logging", cost is not None),
    ] if not present]
    if missing:
        print("Not yet available (skipped):", ", ".join(missing))


if __name__ == "__main__":
    main()
