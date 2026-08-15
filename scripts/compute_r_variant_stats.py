"""Paired significance stats for EAKD-CFND-R variants vs. EAKD-CFND and vs.
DER, on FNN-Poli-Time (complete, 5 variants) and FNN-Gossip-Time (partial,
3 of 5 variants -- plain replay and +PP+UW are still running as of this
write). Reuses compute_review_stats.paired_stats/holm_bonferroni.

Reads real per-seed data from runs/hybrid_*.json, never invents numbers.
Run from code/: python -m scripts.compute_r_variant_stats
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.compute_review_stats import holm_bonferroni, paired_stats

RUNS = Path(__file__).resolve().parent.parent / "runs"
SEEDS = ["13", "42", "123", "2024", "31415"]

REPLAY_KEYS = {
    "EAKD-CFND+Replay", "EAKD-CFND+Replay+PP",
    "EAKD-CFND+Replay+UW", "EAKD-CFND+Replay+PP+UW",
}

FILES = {
    ("FNN-Poli-Time", "plain replay"): "hybrid_base_poli.json",
    ("FNN-Poli-Time", "+PP"): "hybrid_pp_poli.json",
    ("FNN-Poli-Time", "+UW"): "hybrid_uw_poli.json",
    ("FNN-Poli-Time", "+PP+UW"): "hybrid_both_poli.json",
    ("FNN-Poli-Time", "no-verification"): "hybrid_noverify_poli.json",
    ("FNN-Gossip-Time", "plain replay"): "hybrid_base_gossip.json",
    ("FNN-Gossip-Time", "+PP"): "hybrid_pp_gossip.json",
    ("FNN-Gossip-Time", "+UW"): "hybrid_uw_gossip.json",
    ("FNN-Gossip-Time", "+PP+UW"): "hybrid_both_gossip.json",
    ("FNN-Gossip-Time", "no-verification"): "hybrid_noverify_gossip.json",
}


def seed_accs(per_seed):
    by_seed = {r["run_id"].split("_seed")[-1]: r["avg_accuracy"] for r in per_seed}
    return [by_seed[s] for s in SEEDS]


CORE_FILES = {"FNN-Poli-Time": "core_poli.json", "FNN-Gossip-Time": "core_gossip.json"}
_core_cache = {}


def core_baseline(dataset, method):
    if dataset not in _core_cache:
        _core_cache[dataset] = json.load(open(RUNS / CORE_FILES[dataset]))
    return seed_accs(_core_cache[dataset]["methods"][method]["per_seed"])


def main():
    all_comparisons = []
    print("=" * 78)
    print("EAKD-CFND-R variants vs. EAKD-CFND and vs. DER (paired, seed-matched)")
    print("=" * 78)
    for (dataset, variant), fname in FILES.items():
        d = json.load(open(RUNS / fname))
        methods = d["methods"]
        replay_key = next(k for k in methods if k in REPLAY_KEYS)
        r_accs = seed_accs(methods[replay_key]["per_seed"])
        # the no-verification file only logs the replay method itself; pull
        # the EAKD-CFND/DER reference arms from the core comparison table
        # instead of assuming every hybrid file re-logged them.
        eakd_accs = (seed_accs(methods["EAKD-CFND"]["per_seed"]) if "EAKD-CFND" in methods
                     else core_baseline(dataset, "EAKD-CFND"))
        der_accs = (seed_accs(methods["DER"]["per_seed"]) if "DER" in methods
                    else core_baseline(dataset, "DER"))

        vs_eakd = paired_stats(r_accs, eakd_accs)
        vs_der = paired_stats(r_accs, der_accs)
        all_comparisons.append((f"{variant} vs. EAKD-CFND", dataset, vs_eakd))
        all_comparisons.append((f"{variant} vs. DER", dataset, vs_der))
        print(f"{dataset:18s} {variant:16s} "
              f"vs EAKD-CFND: Δ={vs_eakd['mean_diff']:+6.2f}pp t={vs_eakd['t']:+7.2f} "
              f"p={vs_eakd['p']:.4g} d_z={vs_eakd['cohens_dz']:+.3f}  |  "
              f"vs DER: Δ={vs_der['mean_diff']:+6.2f}pp t={vs_der['t']:+7.2f} "
              f"p={vs_der['p']:.4g} d_z={vs_der['cohens_dz']:+.3f}")

    print()
    print("=" * 78)
    print("Holm-Bonferroni correction across all comparisons above")
    print("=" * 78)
    pvals = [c[2]["p"] for c in all_comparisons]
    adj = holm_bonferroni(pvals)
    for (label, dataset, stats), p_adj in zip(all_comparisons, adj):
        sig = "*" if p_adj < 0.05 else " "
        print(f"{sig} {label:28s} {dataset:18s} p_raw={stats['p']:.4g}  p_holm={p_adj:.4g}")



if __name__ == "__main__":
    main()
