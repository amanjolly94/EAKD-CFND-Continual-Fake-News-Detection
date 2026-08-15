"""The six methods compared in Table \\ref{tab:performance_revised}: FT, EWC,
LwF, DER, LUD, EAKD-CFND (Ours). Implemented as one `Method` interface so
train.py's task loop doesn't need a per-method branch — only the loss
computation differs, matching how Table \\ref{tab:feature_comparison_corrected}
frames these as points on the same design space (rehearsal / regularization /
distillation / adaptive-weighting / external-knowledge), not unrelated
algorithms.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import uncertainty as unc
from .verification import ExternalVerifier


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """L_KD(x) = tau^2 * KL(sigma(z^T/tau) || sigma(z^S/tau)), Eq. eq:kd_loss."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return (temperature ** 2) * F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


class Method:
    """Base class. `step` computes the total loss for one batch and returns
    (loss, per_instance_diagnostics) — diagnostics feed logging/calibration,
    they don't affect the gradient."""

    name = "base"

    def __init__(self, config):
        self.config = config

    def on_task_start(self, task_id: int, model, prev_model):
        pass

    def step(self, model, prev_model, batch, prior_task_classes: list[int]) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError


class FineTuning(Method):
    """Lower bound: no forgetting protection at all."""
    name = "FT"

    def step(self, model, prev_model, batch, prior_task_classes):
        logits = model(**batch["inputs"]).logits
        loss = F.cross_entropy(logits, batch["labels"])
        return loss, {}


class EWC(Method):
    """Elastic Weight Consolidation \\citep{ref_kirkpatrick2017}: quadratic
    penalty on parameters important to previous tasks, weighted by the
    diagonal Fisher information estimated at the end of each task."""
    name = "EWC"

    def __init__(self, config, ewc_lambda: float = 1000.0):
        super().__init__(config)
        self.ewc_lambda = ewc_lambda
        self.fisher: dict[str, torch.Tensor] = {}
        self.optimal_params: dict[str, torch.Tensor] = {}

    def on_task_start(self, task_id: int, model, prev_model):
        if task_id == 0:
            return
        # Fisher/optimal_params are expected to have been populated by
        # `consolidate()` at the end of the previous task (see train.py).

    def consolidate(self, model, dataloader):
        """Call at the end of a task: estimate diagonal Fisher information
        and snapshot current parameters as the new EWC anchor point."""
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        n_batches = 0
        for batch in dataloader:
            model.zero_grad()
            logits = model(**batch["inputs"]).logits
            log_probs = F.log_softmax(logits, dim=-1)
            sampled = torch.multinomial(log_probs.exp(), 1).squeeze(-1)
            loss = F.nll_loss(log_probs, sampled)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            n_batches += 1
        for n in fisher:
            fisher[n] /= max(n_batches, 1)
        self.fisher = fisher
        self.optimal_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        model.train()

    def step(self, model, prev_model, batch, prior_task_classes):
        logits = model(**batch["inputs"]).logits
        loss = F.cross_entropy(logits, batch["labels"])
        if self.fisher:
            penalty = sum(
                (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
                for n, p in model.named_parameters() if n in self.fisher
            )
            loss = loss + self.ewc_lambda * penalty
        return loss, {}


class LwF(Method):
    """Learning without Forgetting: fixed-weight KD from the frozen teacher
    over prior-task classes, no replay, no per-instance adaptation."""
    name = "LwF"

    def __init__(self, config, kd_weight: float = 1.0):
        super().__init__(config)
        self.kd_weight = kd_weight

    def step(self, model, prev_model, batch, prior_task_classes):
        logits = model(**batch["inputs"]).logits
        task_loss = F.cross_entropy(logits, batch["labels"])
        if prev_model is None or not prior_task_classes:
            return task_loss, {}
        with torch.no_grad():
            teacher_logits = prev_model(**batch["inputs"]).logits[:, prior_task_classes]
        student_logits_prior = logits[:, prior_task_classes]
        loss_kd = kd_loss(student_logits_prior, teacher_logits, self.config.kd_temperature)
        return task_loss + self.kd_weight * loss_kd, {}


@dataclass
class ReplayBuffer:
    capacity: int
    items: list = field(default_factory=list)
    _seen: int = 0

    def add(self, item):
        """Reservoir sampling, so every seen instance has equal probability
        of being retained regardless of stream length."""
        self._seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = random.randint(0, self._seen - 1)
            if j < self.capacity:
                self.items[j] = item

    def sample(self, batch_size: int) -> list:
        if not self.items:
            return []
        return random.sample(self.items, min(batch_size, len(self.items)))


class DER(Method):
    """Dark Experience Replay \\citep{ref_buzzega2020}: rehearsal buffer of
    (input, teacher_logits) pairs, replayed with an MSE loss against the
    logits stored at collection time (the "dark experience" — the full
    output distribution, not just the label)."""
    name = "DER"

    def __init__(self, config, alpha: float = 0.5):
        super().__init__(config)
        self.alpha = alpha
        self.buffer = ReplayBuffer(capacity=config.der_buffer_size)

    def step(self, model, prev_model, batch, prior_task_classes):
        logits = model(**batch["inputs"]).logits
        task_loss = F.cross_entropy(logits, batch["labels"])

        with torch.no_grad():
            for i in range(batch["labels"].shape[0]):
                self.buffer.add({
                    "inputs": {k: v[i:i + 1].detach().cpu() for k, v in batch["inputs"].items()},
                    "logits": logits[i:i + 1].detach().cpu(),
                })

        replay = self.buffer.sample(batch["labels"].shape[0])
        if not replay:
            return task_loss, {}
        device = logits.device
        replay_inputs = {k: torch.cat([r["inputs"][k] for r in replay]).to(device) for k in replay[0]["inputs"]}
        replay_target_logits = torch.cat([r["logits"] for r in replay]).to(device)
        replay_logits = model(**replay_inputs).logits
        loss_der = F.mse_loss(replay_logits, replay_target_logits)
        return task_loss + self.alpha * loss_der, {}


class LUD(Method):
    """Logits Uncertainty Distillation \\citep{ref_lud2024}: distillation
    weight is CATEGORY-level — one confidence estimate per class, applied
    uniformly to every instance of that class. This is the key structural
    difference from EAKD-CFND's instance-level weighting (see the Novelty
    section, subsec:novelty), reproduced faithfully here so the comparison in
    Table \\ref{tab:performance_revised} is actually testing that difference."""
    name = "LUD"

    def __init__(self, config, kd_weight: float = 1.0):
        super().__init__(config)
        self.kd_weight = kd_weight
        self.class_confidence: dict[int, float] = {}

    def on_task_start(self, task_id: int, model, prev_model):
        pass  # class_confidence is populated by `estimate_class_confidence` before training starts

    @torch.no_grad()
    def estimate_class_confidence(self, prev_model, dataloader, prior_task_classes: list[int]):
        """Category-level confidence = mean per-class MSP over the teacher's
        predictions on a calibration pass — this replaces per-instance
        omega(x) with one scalar per class, the defining LUD characteristic."""
        if prev_model is None:
            return
        sums: dict[int, float] = {c: 0.0 for c in prior_task_classes}
        counts: dict[int, int] = {c: 0 for c in prior_task_classes}
        for batch in dataloader:
            logits = prev_model(**batch["inputs"]).logits[:, prior_task_classes]
            msp = F.softmax(logits, dim=-1).max(dim=-1).values
            preds = logits.argmax(dim=-1)
            for c_idx, cls in enumerate(prior_task_classes):
                mask = preds == c_idx
                if mask.any():
                    sums[cls] += msp[mask].sum().item()
                    counts[cls] += int(mask.sum().item())
        self.class_confidence = {c: (sums[c] / counts[c] if counts[c] else 0.0) for c in prior_task_classes}

    def step(self, model, prev_model, batch, prior_task_classes):
        logits = model(**batch["inputs"]).logits
        task_loss = F.cross_entropy(logits, batch["labels"])
        if prev_model is None or not prior_task_classes:
            return task_loss, {}
        with torch.no_grad():
            teacher_logits = prev_model(**batch["inputs"]).logits[:, prior_task_classes]
            teacher_pred_class_idx = teacher_logits.argmax(dim=-1)
            weights = torch.tensor(
                [self.class_confidence.get(prior_task_classes[i], 0.0) for i in teacher_pred_class_idx.tolist()],
                device=logits.device,
            )
        student_logits_prior = logits[:, prior_task_classes]
        per_instance_kd = kd_loss(student_logits_prior, teacher_logits, self.config.kd_temperature)
        loss_kd = (weights.mean()) * per_instance_kd  # category-level scalar applied uniformly to the batch
        return task_loss + self.kd_weight * loss_kd, {"lud_class_weight_mean": weights.mean().item()}


class EAKDCFND(Method):
    """Ours. Instance-level adaptive KD weighting + uncertainty-triggered
    external verification, exactly Algorithm 1 + Eq. eq:total_loss."""
    name = "EAKD-CFND"

    def __init__(self, config, verifier: ExternalVerifier | None = None):
        super().__init__(config)
        self.verifier = verifier
        self.uncertainty_fn = unc.get_uncertainty_signal(config.uncertainty_signal)
        self._running_min = float("inf")
        self._running_max = float("-inf")

    def _update_running_bounds(self, omega: torch.Tensor):
        self._running_min = min(self._running_min, omega.min().item())
        self._running_max = max(self._running_max, omega.max().item())

    def step(self, model, prev_model, batch, prior_task_classes):
        logits = model(**batch["inputs"]).logits

        if prev_model is None or not prior_task_classes:
            loss = F.cross_entropy(logits, batch["labels"])
            return loss, {}

        with torch.no_grad():
            teacher_logits = prev_model(**batch["inputs"]).logits[:, prior_task_classes]
            omega = self.uncertainty_fn(teacher_logits)
            if isinstance(omega, tuple):  # mc_dropout/ensemble return (variance, mean_probs)
                omega = omega[0]
        self._update_running_bounds(omega)
        omega_norm = unc.normalize(omega, self._running_min, self._running_max)
        alpha = unc.adaptive_weight(omega_norm, self.config.beta)

        y_effective = batch["labels"]
        if self.verifier is not None:
            texts = batch.get("texts")
            if texts is not None:
                new_labels = y_effective.clone()
                for i, (text, w) in enumerate(zip(texts, omega.tolist())):
                    orig_veracity = int(batch["veracity"][i].item())
                    verified_veracity = self.verifier.effective_label(text, w, orig_veracity)
                    if verified_veracity != orig_veracity:
                        # relabel within the same task-conditioned class space
                        new_labels[i] = batch["labels"][i] - orig_veracity + verified_veracity
                y_effective = new_labels

        task_loss = F.cross_entropy(logits, y_effective, reduction="none")
        student_logits_prior = logits[:, prior_task_classes]
        loss_kd = kd_loss(student_logits_prior, teacher_logits, self.config.kd_temperature)
        # Eq. eq:total_loss: mean over batch of (task loss + alpha(omega(x)) * L_KD(x))
        total = (task_loss + alpha * loss_kd).mean()
        return total, {"omega_mean": omega.mean().item(), "alpha_mean": alpha.mean().item()}


class EAKDCFNDReplay(EAKDCFND):
    """Hybrid: EAKD-CFND's instance-level uncertainty-adaptive KD + external
    verification, plus a small DER-style reservoir replay buffer.

    Motivated by a real result from this session (2026-08-13): on
    PHEME-Event, DER (buffer=200) reached 84.8% accuracy while every
    distillation-only method including EAKD-CFND collapsed to ~16-17% --
    KD from a teacher that is itself just deepcopy(student) at each task
    boundary, with no grounding in real past examples, cannot prevent
    catastrophic forgetting the way replay does. Not one of the paper's
    original six baselines -- exploratory.

    Two independent extensions, off by default (plain DER-style logit
    replay only):
      der_plus_plus: adds DER++'s second replay term -- a SEPARATE sampled
        batch replayed with cross-entropy against the STORED GROUND-TRUTH
        LABEL, on top of the logit-MSE term. Buffer items carry `label` too.
      uncertainty_weighted_replay: instead of a flat replay_weight, weights
        each replayed instance's MSE loss by the CURRENT teacher's
        uncertainty about it -- ties the paper's own instance-level signal
        into the replay mechanism itself, rather than having KD-uncertainty
        and replay sit as two unrelated additive terms. Uses the FULL
        (unsliced) class space for that uncertainty computation, not
        prior_task_classes -- buffer items can include current-task
        instances (added earlier in the same task's loop, before the task
        finishes), whose true classes aren't in prior_task_classes, so
        slicing there would measure confidence over classes that item
        doesn't even belong to (the same category of mismatch that broke
        run_ablation.py's LossBasedAlphaEAKD earlier -- entropy/msp don't
        need a label, so this doesn't crash the way that did, but slicing
        would still be measuring the wrong thing for those items)."""
    name = "EAKD-CFND+Replay"

    def __init__(self, config, verifier: ExternalVerifier | None = None,
                 buffer_capacity: int | None = None, replay_weight: float = 0.5,
                 der_plus_plus: bool = False, uncertainty_weighted_replay: bool = False):
        super().__init__(config, verifier=verifier)
        self.buffer = ReplayBuffer(capacity=buffer_capacity if buffer_capacity is not None else config.der_buffer_size)
        self.replay_weight = replay_weight
        self.der_plus_plus = der_plus_plus
        self.uncertainty_weighted_replay = uncertainty_weighted_replay
        tags = [t for t, flag in [("PP", der_plus_plus), ("UW", uncertainty_weighted_replay)] if flag]
        if tags:
            self.name += "+" + "+".join(tags)

    def step(self, model, prev_model, batch, prior_task_classes):
        logits = model(**batch["inputs"]).logits

        with torch.no_grad():
            for i in range(batch["labels"].shape[0]):
                self.buffer.add({
                    "inputs": {k: v[i:i + 1].detach().cpu() for k, v in batch["inputs"].items()},
                    "logits": logits[i:i + 1].detach().cpu(),
                    "label": batch["labels"][i:i + 1].detach().cpu(),
                })

        if prev_model is None or not prior_task_classes:
            total = F.cross_entropy(logits, batch["labels"])
            diag = {}
        else:
            with torch.no_grad():
                teacher_logits = prev_model(**batch["inputs"]).logits[:, prior_task_classes]
                omega = self.uncertainty_fn(teacher_logits)
                if isinstance(omega, tuple):
                    omega = omega[0]
            self._update_running_bounds(omega)
            omega_norm = unc.normalize(omega, self._running_min, self._running_max)
            alpha = unc.adaptive_weight(omega_norm, self.config.beta)

            y_effective = batch["labels"]
            if self.verifier is not None:
                texts = batch.get("texts")
                if texts is not None:
                    new_labels = y_effective.clone()
                    for i, (text, w) in enumerate(zip(texts, omega.tolist())):
                        orig_veracity = int(batch["veracity"][i].item())
                        verified_veracity = self.verifier.effective_label(text, w, orig_veracity)
                        if verified_veracity != orig_veracity:
                            new_labels[i] = batch["labels"][i] - orig_veracity + verified_veracity
                    y_effective = new_labels

            task_loss = F.cross_entropy(logits, y_effective, reduction="none")
            student_logits_prior = logits[:, prior_task_classes]
            loss_kd = kd_loss(student_logits_prior, teacher_logits, self.config.kd_temperature)
            total = (task_loss + alpha * loss_kd).mean()
            diag = {"omega_mean": omega.mean().item(), "alpha_mean": alpha.mean().item()}

        replay = self.buffer.sample(batch["labels"].shape[0])
        if replay:
            device = logits.device
            replay_inputs = {k: torch.cat([r["inputs"][k] for r in replay]).to(device) for k in replay[0]["inputs"]}
            replay_target_logits = torch.cat([r["logits"] for r in replay]).to(device)
            replay_logits = model(**replay_inputs).logits

            if self.uncertainty_weighted_replay and prev_model is not None:
                with torch.no_grad():
                    replay_teacher_full = prev_model(**replay_inputs).logits
                    replay_omega = self.uncertainty_fn(replay_teacher_full)
                    if isinstance(replay_omega, tuple):
                        replay_omega = replay_omega[0]
                    replay_omega_norm = unc.normalize(replay_omega, self._running_min, self._running_max)
                per_instance_replay = F.mse_loss(replay_logits, replay_target_logits, reduction="none").mean(dim=-1)
                loss_replay = (replay_omega_norm * per_instance_replay).mean()
                diag_extra = {"replay_omega_mean": replay_omega.mean().item()}
            else:
                loss_replay = F.mse_loss(replay_logits, replay_target_logits)
                diag_extra = {}

            total = total + self.replay_weight * loss_replay
            diag["loss_replay"] = loss_replay.item()
            diag.update(diag_extra)

            if self.der_plus_plus:
                replay2 = self.buffer.sample(batch["labels"].shape[0])
                if replay2:
                    replay2_inputs = {k: torch.cat([r["inputs"][k] for r in replay2]).to(device) for k in replay2[0]["inputs"]}
                    replay2_labels = torch.cat([r["label"] for r in replay2]).squeeze(-1).to(device)
                    replay2_logits = model(**replay2_inputs).logits
                    loss_replay_label = F.cross_entropy(replay2_logits, replay2_labels)
                    total = total + self.replay_weight * loss_replay_label
                    diag["loss_replay_label"] = loss_replay_label.item()

        return total, diag


def build_method(config, verifier: ExternalVerifier | None = None) -> Method:
    registry = {
        "FT": FineTuning,
        "EWC": EWC,
        "LwF": LwF,
        "DER": DER,
        "LUD": LUD,
    }
    if config.method == "EAKD-CFND":
        return EAKDCFND(config, verifier=verifier)
    if config.method not in registry:
        raise ValueError(f"Unknown method {config.method!r}")
    return registry[config.method](config)
