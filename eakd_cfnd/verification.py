"""Uncertainty-Driven External Verification module — direct implementation of
Algorithm 1 in main_minor_revison1.tex (subsubsec:external_verification).
Every branch here corresponds 1:1 to an \\If/\\Return line in the paper's
algorithm box, and every call is logged (latency, call count, failure/fallback
reason) so scripts/run_cost_logging.py can produce the quantitative cost
table Reviewer 1 Major Concern 6 / Reviewer 2 point 2 asked for, which the
manuscript's Limitations section currently names as missing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests
from difflib import SequenceMatcher

GOOGLE_FACT_CHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GOOGLE_FACT_CHECK_OAUTH_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Verdict -> task label space mapping (fake=1, real=0). Extend as needed for
# textual ratings actually observed in the API's `textualRating` field.
VERDICT_TO_LABEL = {
    "false": 1, "pants on fire": 1, "pants-fire": 1, "mostly false": 1,
    "fake": 1, "incorrect": 1, "misleading": 1,
    "true": 0, "mostly true": 0, "correct": 0, "accurate": 0,
}


@dataclass
class VerificationLogEntry:
    query: str
    called: bool
    latency_s: float | None
    n_results: int
    n_after_filter: int
    outcome: str          # "no_call_needed" | "used_result" | "fallback_api_error" |
                            # "fallback_no_relevant_result" | "fallback_tied_conflict"
    label: int | None      # resulting y_effective's veracity component, if determined here


@dataclass
class VerificationStats:
    """Aggregate cost accounting across a run — this IS the cost table."""
    entries: list[VerificationLogEntry] = field(default_factory=list)

    def summary(self) -> dict:
        called = [e for e in self.entries if e.called]
        n_total = len(self.entries)
        n_called = len(called)
        failures = [e for e in called if e.outcome.startswith("fallback")]
        latencies = [e.latency_s for e in called if e.latency_s is not None]
        return {
            "n_instances": n_total,
            "n_api_calls": n_called,
            "api_call_rate": n_called / n_total if n_total else 0.0,
            "n_failures": len(failures),
            "failure_rate": len(failures) / n_called if n_called else 0.0,
            "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
            "p95_latency_s": sorted(latencies)[int(0.95 * len(latencies))] if latencies else None,
            "total_api_time_s": sum(latencies) if latencies else 0.0,
        }


class ExternalVerifier:
    """Config mirrors Algorithm 1's \\Require line exactly: theta_uncertainty,
    similarity threshold tau_sim, an authenticated caller, and the original
    label as the universal fallback.

    Auth: pass exactly one of `api_key` (legacy `key=` query-param auth) or
    `service_account_file` (a Google Cloud service-account JSON key; calls
    authenticate via an OAuth2 bearer token instead). The static API key was
    disabled on the project this ships against; service_account_file is the
    live path."""

    def __init__(self, theta_uncertainty: float, api_key: str | None = None,
                 service_account_file: str | None = None, tau_sim: float = 0.5,
                 timeout_s: float = 5.0, stats: VerificationStats | None = None):
        if bool(api_key) == bool(service_account_file):
            raise ValueError("Pass exactly one of api_key or service_account_file")
        self.api_key = api_key
        self._credentials = None
        if service_account_file:
            from google.oauth2 import service_account
            self._credentials = service_account.Credentials.from_service_account_file(
                service_account_file, scopes=GOOGLE_FACT_CHECK_OAUTH_SCOPES)
        self.theta_uncertainty = theta_uncertainty
        self.tau_sim = tau_sim
        self.timeout_s = timeout_s
        self.stats = stats if stats is not None else VerificationStats()

    def _auth_headers(self) -> dict:
        if self._credentials is None:
            return {}
        if not self._credentials.valid:
            from google.auth.transport.requests import Request
            self._credentials.refresh(Request())
        return {"Authorization": f"Bearer {self._credentials.token}"}

    def effective_label(self, x_text: str, omega_x: float, original_label: int) -> int:
        """Algorithm 1, lines 1-24. Returns y_effective (veracity, 0/1)."""
        if omega_x <= self.theta_uncertainty:
            self.stats.entries.append(VerificationLogEntry(
                query=x_text, called=False, latency_s=None, n_results=0,
                n_after_filter=0, outcome="no_call_needed", label=original_label))
            return original_label

        query = self._build_query(x_text)
        start = time.monotonic()
        try:
            results = self._search(query)
            latency = time.monotonic() - start
        except (requests.RequestException, TimeoutError) as exc:
            latency = time.monotonic() - start
            self.stats.entries.append(VerificationLogEntry(
                query=query, called=True, latency_s=latency, n_results=0,
                n_after_filter=0, outcome="fallback_api_error", label=original_label))
            return original_label  # failure fallback (Algorithm 1, line 8)

        filtered = [r for r in results if self._similarity(r.get("claim_text", ""), x_text) >= self.tau_sim]
        if not filtered:
            self.stats.entries.append(VerificationLogEntry(
                query=query, called=True, latency_s=latency, n_results=len(results),
                n_after_filter=0, outcome="fallback_no_relevant_result", label=original_label))
            return original_label  # Algorithm 1, line 11

        mapped = [self._map_verdict_to_label(r.get("verdict", "")) for r in filtered]
        mapped = [m for m in mapped if m is not None]
        if not mapped:
            self.stats.entries.append(VerificationLogEntry(
                query=query, called=True, latency_s=latency, n_results=len(results),
                n_after_filter=len(filtered), outcome="fallback_no_relevant_result", label=original_label))
            return original_label

        label, outcome = self._resolve_conflict(mapped, original_label)
        self.stats.entries.append(VerificationLogEntry(
            query=query, called=True, latency_s=latency, n_results=len(results),
            n_after_filter=len(filtered), outcome=outcome, label=label))
        return label

    # -- Algorithm 1 sub-routines -------------------------------------------------

    @staticmethod
    def _build_query(x_text: str, max_words: int = 20) -> str:
        """BuildQuery(x): use the first max_words words as the search query —
        a simple, replaceable heuristic; swap for claim-span extraction if a
        stronger extractor is available."""
        return " ".join(x_text.split()[:max_words])

    def _search(self, query: str) -> list[dict]:
        """FactCheckAPI.search(q) via the Google Fact Check Tools API
        \\citep{ref_googleapi}. Raises on transport failure so the caller's
        try/except drives the failure-fallback branch."""
        params = {"query": query}
        if self.api_key:
            params["key"] = self.api_key
        resp = requests.get(
            GOOGLE_FACT_CHECK_ENDPOINT,
            params=params,
            headers=self._auth_headers(),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for claim in data.get("claims", []):
            claim_text = claim.get("text", "")
            for review in claim.get("claimReview", []):
                results.append({
                    "claim_text": claim_text,
                    "verdict": review.get("textualRating", ""),
                })
        return results

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """sim(r.claim, x) — SequenceMatcher ratio is a cheap, dependency-free
        stand-in; swap for embedding cosine similarity if higher precision on
        the filter step is needed."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    @staticmethod
    def _map_verdict_to_label(verdict: str) -> int | None:
        """MapVerdictToLabel(r.verdict). Returns None for verdicts not in the
        table (e.g. "Unproven", "Mixture") — these are excluded from the
        conflict-resolution vote, matching Algorithm 1's L set construction
        (only verdicts that map cleanly participate)."""
        return VERDICT_TO_LABEL.get(verdict.strip().lower())

    @staticmethod
    def _resolve_conflict(mapped_labels: list[int], original_label: int) -> tuple[int, str]:
        """Algorithm 1, lines 13-18: single label -> use it; strict majority
        -> majority vote; tie -> fallback to original label."""
        if len(set(mapped_labels)) == 1:
            return mapped_labels[0], "used_result"
        counts = {v: mapped_labels.count(v) for v in set(mapped_labels)}
        best = max(counts.values())
        winners = [v for v, c in counts.items() if c == best]
        if len(winners) == 1:
            return winners[0], "used_result"
        return original_label, "fallback_tied_conflict"
