"""Teacher uncertainty signal omega(x) (Section subsubsec:uncertainty_signal),
plus the two calibration-focused alternatives Reviewer 1 (Major Concern 4)
and Reviewer 2 asked for: MC Dropout and deep-ensemble variance. Having all
four behind one interface is what makes the calibration-comparison phase
(scripts/run_calibration.py) a config change, not a rewrite.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """H(P(x)) = -sum p_i log p_i, Eq. in subsubsec:uncertainty_signal. Higher = more uncertain."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def max_softmax_probability(logits: torch.Tensor) -> torch.Tensor:
    """MSP uncertainty = 1 - max_i p_i (low max prob = high uncertainty)."""
    probs = F.softmax(logits, dim=-1)
    return 1.0 - probs.max(dim=-1).values


@torch.no_grad()
def mc_dropout_uncertainty(model, inputs, n_passes: int = 20) -> torch.Tensor:
    """MC Dropout (Gal & Ghahramani, ref_gal2016_dropout): keep dropout active
    at inference, run n_passes forward passes, use predictive variance as the
    uncertainty signal. This is the calibration-focused baseline Reviewer 1
    Major Concern 4 asks EAKD-CFND's entropy/MSP signal to be compared against.
    """
    was_training = model.training
    model.train()  # keep dropout layers active
    probs_stack = []
    for _ in range(n_passes):
        logits = model(**inputs).logits
        probs_stack.append(F.softmax(logits, dim=-1))
    model.train(was_training)
    probs_stack = torch.stack(probs_stack, dim=0)  # [n_passes, B, C]
    mean_probs = probs_stack.mean(dim=0)
    # predictive variance of the winning class, averaged over classes as the
    # scalar per-instance uncertainty signal
    variance = probs_stack.var(dim=0).mean(dim=-1)
    return variance, mean_probs


@torch.no_grad()
def ensemble_uncertainty(models: list, inputs) -> torch.Tensor:
    """Deep ensemble uncertainty: disagreement (variance) across `ensemble_size`
    independently-seeded models, the other calibration baseline Reviewer 1
    Major Concern 4 asks for."""
    probs_stack = [F.softmax(m(**inputs).logits, dim=-1) for m in models]
    probs_stack = torch.stack(probs_stack, dim=0)
    mean_probs = probs_stack.mean(dim=0)
    variance = probs_stack.var(dim=0).mean(dim=-1)
    return variance, mean_probs


def normalize(uncertainty: torch.Tensor, running_min: float, running_max: float) -> torch.Tensor:
    """normalize(omega(x)) -> [0, 1], per Eq. eq:alpha_uncertainty. Uses a
    running min/max (updated by the caller from validation-set statistics)
    rather than per-batch min/max, so alpha() is stable across batches."""
    denom = max(running_max - running_min, 1e-8)
    return ((uncertainty - running_min) / denom).clamp(0.0, 1.0)


def adaptive_weight(omega_normalized: torch.Tensor, beta: float) -> torch.Tensor:
    """alpha(omega(x)) = max(0, 1 - beta * normalize(omega(x))), Eq. eq:alpha_uncertainty."""
    return (1.0 - beta * omega_normalized).clamp(min=0.0)


def get_uncertainty_signal(kind: str):
    """Dispatch used by config.RunConfig.uncertainty_signal."""
    if kind == "entropy":
        return entropy
    if kind == "msp":
        return max_softmax_probability
    if kind == "mc_dropout":
        return mc_dropout_uncertainty
    if kind == "ensemble":
        return ensemble_uncertainty
    raise ValueError(f"Unknown uncertainty_signal {kind!r}")
