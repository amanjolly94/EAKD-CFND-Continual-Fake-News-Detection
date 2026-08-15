"""Runnable self-check for the logic most likely to silently corrupt results
if wrong: the significance-test formula, the verification module's
fallback/conflict branching (Algorithm 1), and ECE binning. No GPU, no
dataset, no network required — run with:

    python -m tests.test_core

This is not a full test suite (no pytest, no fixtures) — one assert-based
script per ponytail's "non-trivial logic needs one runnable check" rule.
"""
from __future__ import annotations

from eakd_cfnd.calibration import expected_calibration_error
from eakd_cfnd.stats import welch_t_test
from eakd_cfnd.verification import ExternalVerifier, VerificationStats


def test_welch_t_matches_manuscript():
    """Regression test against the four t-values already published in
    main_minor_revison1.tex's Table \\ref{tab:significance} — computed from
    Table \\ref{tab:performance_revised}'s own mean/SD, n=5 per group. If
    this ever fails, either the formula changed or the manuscript's table
    needs correcting, not the other way around."""
    cases = [
        # (mean1, sd1, mean2, sd2, expected_t, expected_delta)
        (80.3, 0.9, 78.5, 1.0, 2.99, 1.8),   # EAKD-CFND vs DER, PHEME-Event
        (80.3, 0.9, 77.3, 1.1, 4.72, 3.0),   # EAKD-CFND vs LUD, PHEME-Event
        (86.2, 0.8, 84.0, 0.9, 4.09, 2.2),   # EAKD-CFND vs DER, FNN-Poli-Time
        (86.2, 0.8, 83.1, 1.0, 5.41, 3.1),   # EAKD-CFND vs LUD, FNN-Poli-Time
    ]
    for mean1, sd1, mean2, sd2, expected_t, expected_delta in cases:
        result = welch_t_test(mean1, sd1, 5, mean2, sd2, 5)
        assert abs(result["t"] - expected_t) < 0.01, (
            f"t mismatch: got {result['t']:.2f}, manuscript says {expected_t}")
        assert abs(result["delta"] - expected_delta) < 0.01, (
            f"delta mismatch: got {result['delta']:.2f}, manuscript says {expected_delta}")
    print("test_welch_t_matches_manuscript: OK (4/4 match Table \\ref{tab:significance})")


def test_welch_t_symmetry():
    """Swapping the two groups should flip the sign of t and delta, not change their magnitude."""
    a = welch_t_test(80.3, 0.9, 5, 78.5, 1.0, 5)
    b = welch_t_test(78.5, 1.0, 5, 80.3, 0.9, 5)
    assert abs(a["t"] + b["t"]) < 1e-9
    assert abs(a["delta"] + b["delta"]) < 1e-9
    assert abs(a["df"] - b["df"]) < 1e-9
    print("test_welch_t_symmetry: OK")


def test_verification_no_call_below_threshold():
    """Algorithm 1, line 1-2: omega(x) <= theta -> no API call, original label returned."""
    stats = VerificationStats()
    verifier = ExternalVerifier(api_key="unused", theta_uncertainty=0.7, stats=stats)
    label = verifier.effective_label("some headline", omega_x=0.3, original_label=1)
    assert label == 1
    assert len(stats.entries) == 1
    assert stats.entries[0].called is False
    assert stats.entries[0].outcome == "no_call_needed"
    print("test_verification_no_call_below_threshold: OK")


def test_verification_api_error_falls_back():
    """Algorithm 1, line 6-8: API call raises -> fallback to original label, logged as a failure."""
    class BrokenVerifier(ExternalVerifier):
        def _search(self, query):
            import requests
            raise requests.RequestException("simulated network failure")

    stats = VerificationStats()
    verifier = BrokenVerifier(api_key="unused", theta_uncertainty=0.7, stats=stats)
    label = verifier.effective_label("some headline", omega_x=0.9, original_label=0)
    assert label == 0
    assert stats.entries[0].outcome == "fallback_api_error"
    summary = stats.summary()
    assert summary["n_api_calls"] == 1
    assert summary["n_failures"] == 1
    assert summary["failure_rate"] == 1.0
    print("test_verification_api_error_falls_back: OK")


def test_verification_conflict_resolution():
    """Algorithm 1, line 13-18: single result -> use it; majority -> majority
    vote; tie -> fallback to original label."""
    assert ExternalVerifier._resolve_conflict([1], original_label=0) == (1, "used_result")
    assert ExternalVerifier._resolve_conflict([1, 1, 0], original_label=0) == (1, "used_result")
    assert ExternalVerifier._resolve_conflict([1, 0], original_label=0) == (0, "fallback_tied_conflict")
    print("test_verification_conflict_resolution: OK")


def test_ece_perfectly_calibrated_is_zero():
    """A model whose confidence exactly equals its bin accuracy should score ECE=0."""
    confidences = [0.9] * 10
    correct = [1] * 9 + [0]  # 90% accuracy in the 0.9 bin, matching confidence exactly
    result = expected_calibration_error(confidences, correct, n_bins=10)
    assert result["ece"] < 1e-9, f"expected ECE=0, got {result['ece']}"
    print("test_ece_perfectly_calibrated_is_zero: OK")


def test_ece_overconfident_is_positive():
    """A model that's always 99% confident but only 50% correct should have high ECE."""
    confidences = [0.99] * 20
    correct = [1] * 10 + [0] * 10
    result = expected_calibration_error(confidences, correct, n_bins=10)
    assert result["ece"] > 0.4, f"expected large ECE for overconfident model, got {result['ece']}"
    print("test_ece_overconfident_is_positive: OK")


if __name__ == "__main__":
    test_welch_t_matches_manuscript()
    test_welch_t_symmetry()
    test_verification_no_call_below_threshold()
    test_verification_api_error_falls_back()
    test_verification_conflict_resolution()
    test_ece_perfectly_calibrated_is_zero()
    test_ece_overconfident_is_positive()
    print("\nAll self-checks passed.")
