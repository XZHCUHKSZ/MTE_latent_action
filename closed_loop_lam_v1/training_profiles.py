"""Explicit training profiles for baseline-fidelity and matched-budget runs.

The legacy profile exists only to reproduce earlier internal runs. New results
must select either ``matched`` or ``fidelity`` and persist the resolved settings.
No profile changes the data split or action-label ledger.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


PROFILE_NAMES = ("legacy", "matched", "fidelity")


@dataclass(frozen=True)
class MethodTrainingSettings:
    profile: str
    method: str
    representation_updates: int
    policy_updates: int
    representation_lr: float
    policy_lr: float
    use_linear_warmup_decay: bool
    laom_max_offset: int
    laom_target_tau: float
    laom_supervision_coef: float
    latent_dim_override: int | None
    hidden_dim_override: int | None
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_method_training_settings(
    method: str,
    profile: str,
    requested_representation_updates: int,
    requested_policy_updates: int,
    batch_size: int,
    train_transition_count: int,
    default_lr: float = 1e-3,
) -> MethodTrainingSettings:
    """Resolve all method-specific choices before a training loop starts.

    ``matched`` keeps the caller's update counts but restores official semantic
    defaults for LAPO/LAOM. ``fidelity`` additionally restores the official LAPO
    Stage-1/2 step counts and the LAOM 150-epoch representation schedule. Visual
    encoders remain replaced by MLPs because this project stores structured state.
    """
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown training profile {profile!r}; expected {PROFILE_NAMES}")
    if requested_representation_updates <= 0 or requested_policy_updates <= 0:
        raise ValueError("Training update counts must be positive")
    if batch_size <= 0 or train_transition_count <= 0:
        raise ValueError("batch_size and train_transition_count must be positive")

    settings = dict(
        profile=profile,
        method=method,
        representation_updates=int(requested_representation_updates),
        policy_updates=int(requested_policy_updates),
        representation_lr=float(default_lr),
        policy_lr=float(default_lr),
        use_linear_warmup_decay=False,
        # These values reproduce the pre-audit internal adapter behavior only.
        laom_max_offset=10,
        laom_target_tau=.001,
        laom_supervision_coef=.01,
        latent_dim_override=None,
        hidden_dim_override=None,
        source="legacy_internal_protocol",
    )

    if profile in ("matched", "fidelity"):
        settings.update(
            use_linear_warmup_decay=True,
            laom_max_offset=1,
            laom_target_tau=.01,
            laom_supervision_coef=.05,
            source="project_method_matched_budget_schedule",
        )
        if method == "lapo_state_adapter":
            settings.update(
                representation_lr=3e-4,
                policy_lr=2e-4,
                source="lapo_official_semantics_structured_state_adapter",
            )
        elif method in ("laom_state_adapter", "laom_supervised_state_adapter"):
            settings.update(
                representation_lr=3e-4,
                policy_lr=3e-4,
                source="laom_official_semantics_structured_state_adapter",
            )

    if profile == "fidelity":
        if method == "lapo_state_adapter":
            settings.update(representation_updates=50_000, policy_updates=60_000)
        elif method in ("laom_state_adapter", "laom_supervised_state_adapter"):
            batches_per_epoch = max(1, math.ceil(train_transition_count / batch_size))
            settings.update(
                representation_updates=150 * batches_per_epoch,
                latent_dim_override=256,
                hidden_dim_override=512,
            )

    return MethodTrainingSettings(**settings)


def linear_warmup_decay_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    """Simple dependency-free warmup followed by linear decay."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0:
        raise ValueError("step cannot be negative")
    warmup_steps = max(0, min(int(warmup_steps), int(total_steps)))
    if warmup_steps and step < warmup_steps:
        return max(1e-3, float(step + 1) / float(warmup_steps))
    remaining = max(1, total_steps - warmup_steps)
    return max(0., float(total_steps - step) / float(remaining))
