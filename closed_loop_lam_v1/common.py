from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from closed_loop_lam_v1.training_profiles import (
    linear_warmup_decay_scale,
    resolve_method_training_settings,
)
from closed_loop_lam_v1.unified_models import (
    BidirectionalMobiusTreeEdgeCARA,
    MIFCARAIncidenceFlow,
    MatchedCFDeepSets,
    MobiusGraphEdgeCARA,
    MobiusSimple,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def save_json(path: str | Path, value: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def trajectory_split(n_episodes: int, seed: int, train_frac: float = .7,
                     val_frac: float = .15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_episodes < 3:
        raise ValueError("At least three complete trajectories are required")
    eps = np.arange(n_episodes, dtype=np.int64)
    np.random.default_rng(seed).shuffle(eps)
    n_train = max(1, min(n_episodes - 2, int(round(train_frac * n_episodes))))
    n_val = max(1, min(n_episodes - n_train - 1, int(round(val_frac * n_episodes))))
    return eps[:n_train], eps[n_train:n_train + n_val], eps[n_train + n_val:]


def source_group_trajectory_split(source_ids: np.ndarray, seed: int,
                                  train_frac: float = .7,
                                  val_frac: float = .15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split whole source trials, keeping all roles/chunks in one partition."""
    source_ids = np.asarray(source_ids)
    unique = np.unique(source_ids)
    train_sources, val_sources, test_sources = trajectory_split(
        len(unique), seed, train_frac, val_frac)
    train_values = unique[train_sources]; val_values = unique[val_sources]; test_values = unique[test_sources]
    return (np.where(np.isin(source_ids, train_values))[0],
            np.where(np.isin(source_ids, val_values))[0],
            np.where(np.isin(source_ids, test_values))[0])


def stratified_trajectory_split(group_columns: Iterable[np.ndarray], seed: int,
                                train_frac: float = .7,
                                val_frac: float = .15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trajectory split preserving partner/demonstrator/role cells."""
    cols = [np.asarray(x) for x in group_columns]
    if not cols or any(len(x) != len(cols[0]) for x in cols):
        raise ValueError("Stratified split requires equally sized group columns")
    keys = np.stack(cols, axis=1); rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for key in np.unique(keys, axis=0):
        ix = np.where(np.all(keys == key, axis=1))[0]; rng.shuffle(ix)
        if len(ix) < 3:
            raise ValueError(f"Each partner/demonstrator/role cell needs >=3 trajectories; {key.tolist()} has {len(ix)}")
        n_train = max(1, min(len(ix) - 2, int(round(train_frac * len(ix)))))
        n_val = max(1, min(len(ix) - n_train - 1, int(round(val_frac * len(ix)))))
        train.extend(ix[:n_train]); val.extend(ix[n_train:n_train+n_val]); test.extend(ix[n_train+n_val:])
    train = np.asarray(train, np.int64); val = np.asarray(val, np.int64); test = np.asarray(test, np.int64)
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def nested_label_episodes(train_eps: np.ndarray, budget: int, seed: int) -> np.ndarray:
    eps = np.asarray(train_eps, np.int64).copy()
    np.random.default_rng(seed).shuffle(eps)
    return np.sort(eps[:min(int(budget), len(eps))])


def nested_stratified_label_episodes(train_eps: np.ndarray, group_columns: Iterable[np.ndarray],
                                     budget: int, seed: int) -> np.ndarray:
    """Nested round-robin label subset across partner-skill/role groups."""
    train_eps = np.asarray(train_eps, np.int64)
    cols = [np.asarray(x) for x in group_columns]
    keys = np.stack([x[train_eps] for x in cols], axis=1)
    rng = np.random.default_rng(seed)
    groups = []
    for key in np.unique(keys, axis=0):
        members = train_eps[np.all(keys == key, axis=1)].copy(); rng.shuffle(members)
        groups.append(members.tolist())
    rng.shuffle(groups); order = []
    while any(groups):
        for members in groups:
            if members:
                order.append(members.pop())
    return np.sort(np.asarray(order[:min(int(budget), len(order))], np.int64))


def flat_indices(episodes: Iterable[int], horizon_steps: int) -> np.ndarray:
    episodes = np.asarray(list(episodes), np.int64)
    if len(episodes) == 0:
        return np.zeros(0, np.int64)
    return (episodes[:, None] * horizon_steps + np.arange(horizon_steps)[None, :]).reshape(-1)


def valid_step_mask(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Return the authoritative transition mask, or all-valid for legacy data."""
    shape = tuple(np.asarray(data["actions"]).shape[:2])
    if "valid_step_mask" not in data:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(data["valid_step_mask"], dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"valid_step_mask {mask.shape} does not match actions {shape}")
    if np.any(np.diff(mask.astype(np.int8), axis=1) > 0):
        raise ValueError("valid_step_mask must be a valid prefix followed only by padding")
    return mask


def valid_flat_indices(data: Dict[str, np.ndarray], episodes: Iterable[int]) -> np.ndarray:
    """Flatten complete trajectories while excluding explicit terminal padding."""
    mask = valid_step_mask(data)
    episodes = np.asarray(list(episodes), np.int64)
    if len(episodes) == 0:
        return np.zeros(0, np.int64)
    base = flat_indices(episodes, mask.shape[1])
    return base[mask[episodes].reshape(-1)]


def fit_mean_std(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(x, np.float32).mean(axis=0)
    sd = np.asarray(x, np.float32).std(axis=0)
    return mu.astype(np.float32), np.where(sd > eps, sd, 1.).astype(np.float32)


class ContinuousLAM(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, obs_dim),
        )

    def encode(self, obs: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([obs, future], dim=-1))

    def forward(self, obs: torch.Tensor, future: torch.Tensor):
        z = self.encode(obs, future)
        return self.decoder(torch.cat([obs, z], dim=-1)), z


class VQLAM(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, hidden: int, codebook_size: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z_dim),
        )
        self.codebook = nn.Embedding(codebook_size, z_dim)
        nn.init.uniform_(self.codebook.weight, -1. / codebook_size, 1. / codebook_size)
        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, obs_dim),
        )

    def quantize(self, z_e: torch.Tensor):
        dist = (z_e.pow(2).sum(-1, keepdim=True) - 2 * z_e @ self.codebook.weight.T
                + self.codebook.weight.pow(2).sum(-1)[None])
        ids = dist.argmin(-1); z_q = self.codebook(ids)
        return z_e + (z_q - z_e).detach(), z_q, ids

    def forward(self, obs: torch.Tensor, future: torch.Tensor):
        z_e = self.encoder(torch.cat([obs, future], dim=-1)); z_st, z_q, ids = self.quantize(z_e)
        pred = self.decoder(torch.cat([obs, z_st], dim=-1))
        return pred, z_e, z_q, z_st, ids


class LAPOEMAQuantizer(nn.Module):
    """EMA multi-codebook VQ used by the official LAPO implementation.

    This is the vector-observation adapter of VQEmbeddingEMA in
    third_party/LAPO_official/lapo/models.py. The codebook layout, nearest
    neighbour assignment, EMA updates, Laplace smoothing, commitment loss and
    straight-through estimator follow that implementation.
    """

    def __init__(self, num_codebooks: int = 2, num_discrete_latents: int = 4,
                 emb_dim: int = 16, num_embs: int = 64,
                 commitment_cost: float = .05, decay: float = .999,
                 epsilon: float = 1e-5):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_discrete_latents = num_discrete_latents
        self.emb_dim = emb_dim
        self.num_embs = num_embs
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        embedding = torch.zeros(num_codebooks, num_embs, emb_dim)
        embedding.uniform_(-1 / num_embs * 5, 1 / num_embs * 5)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(num_codebooks, num_embs))
        self.register_buffer("ema_weight", embedding.clone())

    @property
    def latent_dim(self) -> int:
        return self.num_codebooks * self.num_discrete_latents * self.emb_dim

    def forward(self, x: torch.Tensor):
        b = len(x); n = self.num_codebooks; h = self.num_discrete_latents
        d = self.emb_dim; m = self.num_embs
        if x.shape[-1] != self.latent_dim:
            raise ValueError(f"LAPO latent dim must be {self.latent_dim}, got {x.shape[-1]}")
        xv = x.view(b, n, d, h, 1).permute(1, 0, 3, 4, 2)
        x_flat = xv.detach().reshape(n, -1, d)
        distances = torch.baddbmm(
            torch.sum(self.embedding ** 2, dim=2).unsqueeze(1)
            + torch.sum(x_flat ** 2, dim=2, keepdim=True),
            x_flat, self.embedding.transpose(1, 2), alpha=-2., beta=1.)
        indices = torch.argmin(distances, dim=-1)
        encodings = F.one_hot(indices, m).float()
        quantized = torch.gather(
            self.embedding, 1, indices.unsqueeze(-1).expand(-1, -1, d)
        ).view_as(xv)
        if self.training:
            with torch.no_grad():
                self.ema_count.mul_(self.decay).add_((1 - self.decay) * encodings.sum(1))
                count_sum = self.ema_count.sum(-1, keepdim=True)
                smoothed = ((self.ema_count + self.epsilon)
                            / (count_sum + m * self.epsilon) * count_sum)
                dw = torch.bmm(encodings.transpose(1, 2), x_flat)
                self.ema_weight.mul_(self.decay).add_((1 - self.decay) * dw)
                self.embedding.copy_(self.ema_weight / smoothed.unsqueeze(-1))
        vq_loss = self.commitment_cost * F.mse_loss(xv, quantized.detach())
        quantized_st = quantized.detach() + (xv - xv.detach())
        avg_probs = encodings.mean(1)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum(-1)).sum()
        la_q = quantized_st.permute(1, 0, 4, 2, 3).reshape(b, self.latent_dim)
        inds = indices.view(n, b, h, 1).permute(1, 0, 2, 3)
        return la_q, vq_loss, perplexity, inds


class LAPOStateAdapter(nn.Module):
    """Structured-state adapter for LAPO Stage 1.

    The official CNN/UNet are replaced by MLPs because MPE/Overcooked provide
    vectors. Temporal semantics and objective are kept:
    [o_(t-1), o_t, o_(t+1)] -> la and FDM([o_(t-1), o_t], la_q) -> o_(t+1).
    """

    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.vq = LAPOEMAQuantizer()
        la_dim = self.vq.latent_dim
        self.idm = nn.Sequential(
            nn.Linear(obs_dim * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, la_dim),
        )
        self.world_model = nn.Sequential(
            nn.Linear(obs_dim * 2 + la_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs_history: torch.Tensor):
        la = self.idm(obs_history.flatten(1))
        la_q, vq_loss, perplexity, indices = self.vq(la)
        wm_in = torch.cat([obs_history[:, :2].flatten(1), la_q], dim=-1)
        pred = self.world_model(wm_in)
        return pred, la, la_q, vq_loss, perplexity, indices


class LAOMMLPBlock(nn.Module):
    """Transformer-style residual MLP block used by the official LAOM heads."""

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, expand * dim), nn.ReLU6(),
            nn.Linear(expand * dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class LAOMConditionedHead(nn.Module):
    """Condition on the source inputs at every block, matching LAOM's MLP heads."""

    def __init__(self, condition_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(condition_dim, hidden),
            nn.Linear(condition_dim + hidden, hidden),
            nn.Linear(condition_dim + hidden, hidden),
        ])
        self.blocks = nn.ModuleList([LAOMMLPBlock(hidden) for _ in range(3)])
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        x = self.blocks[0](self.projections[0](condition))
        x = self.blocks[1](self.projections[1](torch.cat([x, condition], dim=-1)))
        x = self.blocks[2](self.projections[2](torch.cat([x, condition], dim=-1)))
        return self.output(x)


class LAOMStateAdapter(nn.Module):
    """Structured-state adapter of the official LAOM/LAOM+supervision objective.

    Kept from the official implementation: a shared online encoder, multi-step
    inverse head, one-step latent temporal-consistency target, EMA target encoder,
    continuous (non-quantized) latent, and an optional linear true-action head.
    CNNs and image augmentations are intentionally replaced for vector states.
    """

    def __init__(self, obs_dim: int, latent_dim: int, hidden: int,
                 true_action_dim: int | None = None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.inverse_head = LAOMConditionedHead(2 * hidden, hidden, latent_dim)
        self.forward_head = LAOMConditionedHead(hidden + latent_dim, hidden, hidden)
        self.true_action_head = (
            nn.Linear(latent_dim, true_action_dim) if true_action_dim is not None else None
        )
        self.target_encoder = deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    def infer(self, obs: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_emb = self.encoder(obs)
        future_emb = self.encoder(future)
        latent = self.inverse_head(torch.cat([obs_emb, future_emb], dim=-1))
        return latent, obs_emb

    def forward(self, obs: torch.Tensor, future: torch.Tensor):
        latent, obs_emb = self.infer(obs, future)
        predicted_next = self.forward_head(torch.cat([obs_emb.detach(), latent], dim=-1))
        return predicted_next, latent, obs_emb.detach()

    @torch.no_grad()
    def target(self, next_obs: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(next_obs)

    @torch.no_grad()
    def update_target(self, tau: float = .001) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(1. - tau).add_(online, alpha=tau)

    @torch.no_grad()
    def label(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        return self.infer(obs, next_obs)[0]


class EdgeCARA(nn.Module):
    def __init__(self, state_dim: int, z_dim: int, context_dim: int, hidden: int):
        super().__init__()
        self.z_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z_dim),
        )
        self.c_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, context_dim),
        )
        self.common_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, state_dim),
        )
        self.context_decoder = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, state_dim),
        )


class EdgeCARAMobiusAttention(nn.Module):
    """Explicit first-order Mobius residuals with context-aware aggregation.

    The first edge is required to be the empty-context edge g_i(empty).
    Every remaining edge is converted to the first-order residual
    mu_i(S) = g_i(S) - g_i(empty). FORMAL-V2 currently supplies singleton S,
    so these residuals are the pairwise Mobius interaction terms mu_ij.
    """

    def __init__(self, state_dim: int, mask_dim: int, z_dim: int, hidden: int):
        super().__init__()
        self.direct_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z_dim),
        )
        self.interaction_encoder = nn.Sequential(
            nn.Linear(state_dim + mask_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z_dim),
        )
        self.attention = nn.Sequential(
            nn.Linear(2 * z_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.interaction_projection = nn.Linear(z_dim, z_dim, bias=False)
        self.output_norm = nn.LayerNorm(z_dim)
        self.direct_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, state_dim),
        )
        self.interaction_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, state_dim),
        )

    def forward(self, edges: torch.Tensor, context_masks: torch.Tensor):
        direct = edges[:, 0]
        direct_z = self.direct_encoder(direct)
        if edges.shape[1] == 1:
            empty = edges.new_zeros((edges.shape[0], 0, edges.shape[-1]))
            weights = edges.new_zeros((edges.shape[0], 0))
            return self.output_norm(direct_z), direct_z, empty, weights

        residuals = edges[:, 1:] - direct[:, None]
        masks = context_masks[None, 1:].expand(edges.shape[0], -1, -1)
        interaction_z = self.interaction_encoder(
            torch.cat([residuals, masks], dim=-1).reshape(
                -1, residuals.shape[-1] + masks.shape[-1]
            )
        ).reshape(edges.shape[0], edges.shape[1] - 1, -1)
        query = direct_z[:, None].expand_as(interaction_z)
        logits = self.attention(torch.cat([query, interaction_z], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        correction = (weights[..., None] * self.interaction_projection(interaction_z)).sum(1)
        z = self.output_norm(direct_z + correction)
        return z, direct_z, interaction_z, weights


class EdgeCARAMobiusLattice(nn.Module):
    """Generic Mobius inversion over the observed counterfactual subset lattice.

    Unlike ``EdgeCARAMobiusAttention``, this remains mathematically correct when
    pair or higher-order contexts are collected.  A context family may be partial,
    but its subset-incidence matrix must be nonsingular and start with the empty set.
    """

    def __init__(self, state_dim: int, context_masks: torch.Tensor,
                 z_dim: int, hidden: int, policy_aligned: bool = False,
                 obs_dim: int | None = None):
        super().__init__()
        masks = context_masks.detach().float()
        if masks.ndim != 2 or torch.any(masks[0] != 0):
            raise ValueError("Mobius contexts must be [contexts, agents] with empty set first")
        subset = ((masks[:, None, :] <= masks[None, :, :]).all(-1)).float()
        # subset[s, t] above means S_s subset S_t; zeta[row S, col T] needs T subset S.
        zeta = subset.T
        if int(torch.linalg.matrix_rank(zeta).item()) != zeta.shape[0]:
            raise ValueError("Counterfactual context family has singular subset incidence matrix")
        self.register_buffer("context_masks", masks)
        self.register_buffer("zeta", zeta)
        self.register_buffer("mobius", torch.linalg.inv(zeta))
        self.coefficient_encoder = nn.Sequential(
            nn.Linear(state_dim + masks.shape[-1], hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z_dim),
        )
        heads = 4 if z_dim % 4 == 0 else 1
        self.token_attention = nn.MultiheadAttention(z_dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(z_dim)
        self.coefficient_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, state_dim),
        )
        self.policy_aligned = policy_aligned
        if policy_aligned:
            if obs_dim is None:
                raise ValueError("policy-aligned Mobius model requires obs_dim")
            self.latent_prior = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z_dim),
            )
        else:
            self.latent_prior = None

    def forward(self, edges: torch.Tensor):
        coefficients = torch.einsum("cd,bdk->bck", self.mobius, edges)
        masks = self.context_masks[None].expand(edges.shape[0], -1, -1)
        tokens = self.coefficient_encoder(torch.cat([coefficients, masks], dim=-1))
        attended, weights = self.token_attention(
            tokens[:, :1], tokens, tokens, need_weights=True, average_attn_weights=True
        )
        z = self.output_norm(tokens[:, 0] + attended[:, 0])
        return z, tokens, coefficients, weights[:, 0]

    def reconstruct(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficient_hat = self.coefficient_decoder(tokens)
        edge_hat = torch.einsum("cd,bdk->bck", self.zeta, coefficient_hat)
        return edge_hat, coefficient_hat


class MultiViewAE(nn.Module):
    def __init__(self, state_dim: int, z_dim: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, state_dim),
        )


def _offdiag_cov(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.)
    z = z - z.mean(0, keepdim=True)
    cov = z.T @ z / (z.shape[0] - 1)
    return ((cov - torch.diag(torch.diag(cov))) ** 2).mean()


def _variance_floor(z: torch.Tensor) -> torch.Tensor:
    return F.relu(1. - torch.sqrt(z.var(0, unbiased=False) + 1e-4)).pow(2).mean()


def _cross_cov(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.)
    z = (z - z.mean(0)) / (z.std(0, unbiased=False) + 1e-4)
    c = (c - c.mean(0)) / (c.std(0, unbiased=False) + 1e-4)
    return ((z.T @ c / (z.shape[0] - 1)) ** 2).mean()


@dataclass
class RepresentationResult:
    z: np.ndarray
    train_loss: float
    parameter_count: int
    diagnostics: dict | None = None


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    if len(indices) == 0:
        raise ValueError("Empty training transition set")
    yield rng.choice(indices, size=min(batch_size, len(indices)), replace=len(indices) < batch_size)


def _encode_chunks(fn, n: int, chunk: int = 8192) -> np.ndarray:
    ans = []
    with torch.no_grad():
        for start in range(0, n, chunk):
            ans.append(fn(slice(start, min(n, start + chunk))).cpu().numpy())
    return np.concatenate(ans).astype(np.float32)


def _profile_scheduler(optimizer, enabled: bool, total_steps: int):
    if not enabled:
        return None
    warmup_steps = min(max(1, total_steps // 30), 1000)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: linear_warmup_decay_scale(step, total_steps, warmup_steps),
    )


def train_representation(data: Dict[str, np.ndarray], method: str, train_eps: np.ndarray,
                         seed: int, z_dim: int = 16, hidden: int = 128,
                         updates: int = 800, batch_size: int = 512,
                         lr: float = 1e-3, device: str = "cpu",
                         labeled_eps: np.ndarray | None = None,
                         true_action_dim: int | None = None,
                         training_profile: str = "legacy",
                         requested_policy_updates: int | None = None) -> RepresentationResult:
    """Train a representation and return one latent per transition.

    Every method is action-free except the explicitly named
    ``laom_supervised_state_adapter``, which requires ``labeled_eps`` and may
    read actions only from those complete trajectories.
    """
    seed_all(seed)
    obs = data["obs"][:, :-1].reshape(-1, data["obs"].shape[-1]).astype(np.float32)
    root = data["cf_root"].reshape(-1, obs.shape[-1]).astype(np.float32)
    cf_e = data["cf_ego_null"].reshape(-1, obs.shape[-1]).astype(np.float32)
    cf_o = data["cf_other_null"].reshape(-1, obs.shape[-1]).astype(np.float32)
    cf_b = data["cf_both_null"].reshape(-1, obs.shape[-1]).astype(np.float32)
    steps = data["actions"].shape[1]
    transition_mask = valid_step_mask(data)
    tr = valid_flat_indices(data, train_eps)
    rng = np.random.default_rng(seed + 17)
    settings = resolve_method_training_settings(
        method=method,
        profile=training_profile,
        requested_representation_updates=updates,
        requested_policy_updates=(updates if requested_policy_updates is None else requested_policy_updates),
        batch_size=batch_size,
        train_transition_count=len(tr),
        default_lr=lr,
    )
    effective_updates = settings.representation_updates
    effective_lr = settings.representation_lr
    diagnostics = settings.to_dict()
    requested_z_dim, requested_hidden = int(z_dim), int(hidden)
    z_dim = int(settings.latent_dim_override or z_dim)
    hidden = int(settings.hidden_dim_override or hidden)
    diagnostics.update(
        requested_z_dim=requested_z_dim,
        effective_z_dim=z_dim,
        requested_hidden=requested_hidden,
        effective_hidden=hidden,
    )

    if method == "edge_cara_mif":
        required = (
            "cf_horizon_entity_root", "cf_horizon_entity_ego_null",
            "cf_context_masks", "mif_entity_adjacency", "effect_horizons",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                "edge_cara_mif requires explicit [time, coalition, entity, feature] data; "
                f"missing rich-schema fields {missing}"
            )
        rich_root = np.asarray(data["cf_horizon_entity_root"], np.float32)
        rich_ego = np.asarray(data["cf_horizon_entity_ego_null"], np.float32)
        if rich_root.shape != rich_ego.shape or rich_root.ndim != 6:
            raise ValueError(
                "MIF entity arrays must match [episode,step,time,coalition,entity,feature]"
            )
        if rich_root.shape[:2] != data["actions"].shape[:2]:
            raise ValueError("MIF rich arrays do not align with stored trajectories")
        effect_horizons = np.asarray(data["effect_horizons"], np.int64)
        if len(effect_horizons) != rich_root.shape[2] or len(effect_horizons) < 2:
            raise ValueError("MIF requires at least two explicit, aligned effect horizons")
        rich_values = np.concatenate([rich_root, rich_root - rich_ego], axis=-1)
        flat_values = rich_values.reshape(-1, *rich_values.shape[2:])
        vm, vs = fit_mean_std(flat_values[tr].reshape(-1, rich_values.shape[-1]))
        values_n = (flat_values - vm) / vs
        values_t = torch.as_tensor(values_n, device=device)
        masks_t = torch.as_tensor(np.asarray(data["cf_context_masks"], np.float32), device=device)
        adjacency_t = torch.as_tensor(np.asarray(data["mif_entity_adjacency"], np.float32), device=device)
        net = MIFCARAIncidenceFlow(
            value_dim=rich_values.shape[-1],
            context_masks=masks_t,
            entity_adjacency=adjacency_t,
            time_steps=rich_values.shape[2],
            z_dim=z_dim,
            hidden=hidden,
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        for _ in range(effective_updates):
            ix = next(_batches(tr, batch_size, rng)); batch = values_t[ix]
            visible = torch.rand(*batch.shape[:-1], 1, device=device) > .30
            # Preserve a boundary condition while masking the remainder of the complex.
            visible[:, 0, 0] = True
            zbar, reconstructed, _ = net(batch, visible)
            hidden_mask = (~visible).expand_as(batch)
            visible_expanded = visible.expand_as(batch)
            hidden_loss = (
                F.mse_loss(reconstructed[hidden_mask], batch[hidden_mask])
                if bool(hidden_mask.any()) else batch.new_tensor(0.)
            )
            visible_loss = F.mse_loss(
                reconstructed[visible_expanded], batch[visible_expanded]
            )
            loss = hidden_loss + .25 * visible_loss + _variance_floor(zbar) + .02 * _offdiag_cov(zbar)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.detach())
        net.eval()
        z = _encode_chunks(
            lambda sl: net(
                values_t[sl],
                torch.ones(*values_t[sl].shape[:-1], 1, dtype=torch.bool, device=device),
            )[0],
            len(values_t),
        )
        diagnostics.update(
            information_object="MIF_exact_mobius_time_x_coalition_x_entity",
            effect_horizons=effect_horizons.tolist(),
            entity_count=int(rich_values.shape[-2]),
            entity_feature_dim=int(rich_values.shape[-1]),
            masked_completion_probability=.30,
        )
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
        )

    if method in ("laom_state_adapter", "laom_supervised_state_adapter"):
        supervised = method == "laom_supervised_state_adapter"
        if supervised and (labeled_eps is None or len(labeled_eps) == 0):
            raise ValueError("laom_supervised_state_adapter requires labeled_eps")
        obs_seq = data["obs"].astype(np.float32)
        current_obs = obs_seq[:, :-1].reshape(-1, obs_seq.shape[-1])
        om, os = fit_mean_std(current_obs[tr])
        obs_n = (obs_seq - om) / os
        obs_t = torch.as_tensor(obs_n, device=device)
        max_offset = min(settings.laom_max_offset, steps)
        valid_time_count = steps - max_offset + 1
        if valid_time_count <= 0:
            raise ValueError("LAOM multi-step offset does not fit the stored trajectories")
        valid_lengths = transition_mask.sum(axis=1)
        train_pairs = np.asarray(
            [(int(ep), t) for ep in train_eps
             for t in range(max(0, min(valid_time_count, int(valid_lengths[ep]) - max_offset + 1)))],
            dtype=np.int64,
        )
        if len(train_pairs) == 0:
            raise ValueError("No valid LAOM multi-step windows remain after terminal masking")
        actions_array = np.asarray(data["actions"])
        discrete_actions = actions_array.ndim == 2
        supervised_action_dim = None
        if supervised:
            if discrete_actions and true_action_dim is None:
                raise ValueError(
                    "Discrete supervised LAOM requires the declared true_action_dim; "
                    "it is not inferred by inspecting hidden action labels"
                )
            supervised_action_dim = (
                int(true_action_dim) if discrete_actions else int(actions_array.shape[-1])
            )
            labeled_pairs = np.asarray(
                [(int(ep), t) for ep in np.asarray(labeled_eps)
                 for t in range(max(0, min(valid_time_count, int(valid_lengths[ep]) - max_offset + 1)))],
                dtype=np.int64,
            )
            if len(labeled_pairs) == 0:
                raise ValueError("No labeled LAOM windows remain after terminal masking")
        net = LAOMStateAdapter(
            obs_seq.shape[-1], z_dim, hidden, true_action_dim=supervised_action_dim
        ).to(device)
        opt = torch.optim.Adam(
            [parameter for parameter in net.parameters() if parameter.requires_grad], lr=effective_lr
        )
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        loss_value = math.nan
        for _ in range(effective_updates):
            chosen = rng.choice(
                len(train_pairs), size=min(batch_size, len(train_pairs)),
                replace=len(train_pairs) < batch_size,
            )
            pair = train_pairs[chosen]
            offsets = rng.integers(1, max_offset + 1, size=len(pair))
            ep = torch.as_tensor(pair[:, 0], device=device)
            tm = torch.as_tensor(pair[:, 1], device=device)
            future_tm = tm + torch.as_tensor(offsets, device=device)
            current = obs_t[ep, tm]
            future = obs_t[ep, future_tm]
            next_obs = obs_t[ep, tm + 1]
            predicted_next, _, _ = net(current, future)
            loss = F.mse_loss(predicted_next, net.target(next_obs).detach())
            if supervised:
                label_chosen = rng.choice(
                    len(labeled_pairs), size=min(batch_size, len(labeled_pairs)),
                    replace=len(labeled_pairs) < batch_size,
                )
                label_pair = labeled_pairs[label_chosen]
                label_offsets = rng.integers(1, max_offset + 1, size=len(label_pair))
                lep = torch.as_tensor(label_pair[:, 0], device=device)
                ltm = torch.as_tensor(label_pair[:, 1], device=device)
                lfuture = ltm + torch.as_tensor(label_offsets, device=device)
                _, label_latent, _ = net(obs_t[lep, ltm], obs_t[lep, lfuture])
                prediction = net.true_action_head(label_latent)
                targets = actions_array[label_pair[:, 0], label_pair[:, 1]]
                if discrete_actions:
                    supervised_loss = F.cross_entropy(
                        prediction, torch.as_tensor(targets, dtype=torch.long, device=device)
                    )
                else:
                    supervised_loss = F.mse_loss(
                        prediction, torch.as_tensor(targets, dtype=torch.float32, device=device)
                    )
                loss = loss + settings.laom_supervision_coef * supervised_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            net.update_target(tau=settings.laom_target_tau)
            loss_value = float(loss.detach())
        net.eval()
        current_all = torch.as_tensor(
            obs_n[:, :-1].reshape(-1, obs_seq.shape[-1]), device=device
        )
        next_all = torch.as_tensor(
            obs_n[:, 1:].reshape(-1, obs_seq.shape[-1]), device=device
        )
        z = _encode_chunks(lambda sl: net.label(current_all[sl], next_all[sl]), len(current_all))
        diagnostics.update(
            effective_max_offset=int(max_offset),
            effective_target_tau=float(settings.laom_target_tau),
            effective_supervision_coef=(float(settings.laom_supervision_coef) if supervised else None),
        )
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters() if p.requires_grad), diagnostics
        )

    if method == "lapo_state_adapter":
        # LAPO labels the last adjacent transition in a three-frame window.
        # Pad only inside the episode; never borrow history from another one.
        obs_seq = data["obs"].astype(np.float32)
        previous = np.concatenate([obs_seq[:, :1], obs_seq[:, :-2]], axis=1)
        current = obs_seq[:, :-1]
        nxt = obs_seq[:, 1:]
        history = np.stack([previous, current, nxt], axis=2)
        om, os = fit_mean_std(current.reshape(-1, obs_seq.shape[-1])[tr])
        history_n = (history - om) / os
        ht = torch.as_tensor(history_n.reshape(-1, 3, obs_seq.shape[-1]), device=device)
        net = LAPOStateAdapter(obs_seq.shape[-1], hidden).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        last_perplexity = math.nan
        for _ in range(effective_updates):
            ix = next(_batches(tr, batch_size, rng))
            pred, _, _, vq_loss, perplexity, _ = net(ht[ix])
            loss = F.mse_loss(pred, ht[ix, -1]) + vq_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.detach()); last_perplexity = float(perplexity.detach())
        net.eval()
        z = _encode_chunks(lambda sl: net(ht[sl])[1], len(ht))
        diagnostics.update(
            vq_perplexity=last_perplexity,
            vq_active_codes=int((net.vq.ema_count > 0).sum().item()),
            vq_total_codes=int(net.vq.ema_count.numel()),
        )
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
        )

    if method in ("continuous_lam", "lapo_vq"):
        om, os = fit_mean_std(obs[tr]); fm, fs = fit_mean_std(root[tr])
        on = (obs - om) / os; fn = (root - fm) / fs
        ot = torch.as_tensor(on, device=device); ft = torch.as_tensor(fn, device=device)
        net = (ContinuousLAM(obs.shape[-1], z_dim, hidden) if method == "continuous_lam"
               else VQLAM(obs.shape[-1], z_dim, hidden)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        for _ in range(effective_updates):
            ix = next(_batches(tr, batch_size, rng))
            if method == "continuous_lam":
                pred, _ = net(ot[ix], ft[ix]); loss = F.mse_loss(pred, ft[ix])
            else:
                pred, z_e, z_q, _, _ = net(ot[ix], ft[ix])
                loss = F.mse_loss(pred, ft[ix]) + F.mse_loss(z_q, z_e.detach()) + .25 * F.mse_loss(z_e, z_q.detach())
            opt.zero_grad(); loss.backward(); opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.detach())
        if method == "continuous_lam":
            z = _encode_chunks(lambda sl: net.encode(ot[sl], ft[sl]), len(obs))
        else:
            z = _encode_chunks(lambda sl: net(ot[sl], ft[sl])[2], len(obs))
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
        )

    if "cf_context_root" in data and "cf_context_ego_null" in data:
        context_root = data["cf_context_root"].reshape(
            -1, data["cf_context_root"].shape[2], obs.shape[-1]).astype(np.float32)
        context_ego = data["cf_context_ego_null"].reshape(
            -1, data["cf_context_ego_null"].shape[2], obs.shape[-1]).astype(np.float32)
        if context_root.shape != context_ego.shape or context_root.shape[0] != len(obs):
            raise ValueError("Malformed pairwise counterfactual context arrays")
        edges = context_root - context_ego
        if "cf_context_masks" not in data:
            raise ValueError("Explicit counterfactual contexts require cf_context_masks")
        context_masks = np.asarray(data["cf_context_masks"], dtype=np.float32)
        if context_masks.ndim != 2 or context_masks.shape[0] != edges.shape[1]:
            raise ValueError(
                "cf_context_masks must have shape [num_contexts, num_agents]"
            )
        if np.any(context_masks[0]):
            raise ValueError("The first counterfactual context must be the empty set")
    else:
        # Backward-compatible two-context data: S=empty and S=all non-ego agents.
        # FORMAL-V2 MPE data must use the explicit singleton-context arrays above.
        edges = np.stack([root - cf_e, cf_o - cf_b], axis=1).astype(np.float32)
        # Legacy two-agent data: S=empty followed by S={other agent}.
        context_masks = np.asarray([[0., 0.], [0., 1.]], dtype=np.float32)
    if method == "edge_cara_root_only":
        edges = edges[:, :1]
    elif method == "edge_cara_shuffled_cf":
        # Negative control: preserve the marginal edge distribution while breaking
        # the observation-to-counterfactual pairing at the transition level.
        edges = edges[rng.permutation(len(edges))]
    edge_train = edges[tr].reshape(-1, edges.shape[-1])
    em, es = fit_mean_std(edge_train)
    en = (edges - em[None, None]) / es[None, None]
    et = torch.as_tensor(en, device=device)
    mt = torch.as_tensor(context_masks, device=device)

    if method == "random_edge_encoder":
        projection = rng.normal(0., 1. / np.sqrt(edges.shape[-1]),
                                size=(edges.shape[-1], z_dim)).astype(np.float32)
        z = en.mean(1) @ projection
        return RepresentationResult(z.astype(np.float32), math.nan, 0, diagnostics)

    if method == "matched_cf_deepsets":
        net = MatchedCFDeepSets(edges.shape[-1], mt, z_dim, hidden).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        for _ in range(effective_updates):
            ix = next(_batches(tr, batch_size, rng)); rb = et[ix]
            zbar, _, edge_hat = net(rb)
            loss = F.mse_loss(edge_hat, rb) + _variance_floor(zbar) + .02 * _offdiag_cov(zbar)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.detach())
        net.eval()
        z = _encode_chunks(lambda sl: net(et[sl])[0], len(obs))
        diagnostics["information_object"] = "GS_raw_counterfactual_edges"
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
        )

    if method == "edge_cara_mobius_simple":
        net = MobiusSimple(edges.shape[-1], mt, z_dim, hidden).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        for _ in range(effective_updates):
            ix = next(_batches(tr, batch_size, rng)); rb = et[ix]
            zbar, _, coefficients, coefficient_hat, edge_hat = net(rb)
            loss = (
                F.mse_loss(coefficient_hat, coefficients)
                + F.mse_loss(edge_hat, rb)
                + _variance_floor(zbar) + .02 * _offdiag_cov(zbar)
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.detach())
        net.eval()
        z = _encode_chunks(lambda sl: net(et[sl])[0], len(obs))
        diagnostics["information_object"] = "MU_exact_mobius_simple_backbone"
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
        )

    if method == "edge_cara_mobius_graph":
        net = MobiusGraphEdgeCARA(edges.shape[-1], mt, z_dim, hidden).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        for _ in range(effective_updates):
            ix = next(_batches(tr, batch_size, rng)); rb = et[ix]
            zbar, _, _, coefficients, coefficient_hat, edge_hat = net(rb)
            loss = (
                F.mse_loss(coefficient_hat, coefficients)
                + F.mse_loss(edge_hat, rb)
                + _variance_floor(zbar) + .02 * _offdiag_cov(zbar)
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.detach())
        net.eval()
        z = _encode_chunks(lambda sl: net(et[sl])[0], len(obs))
        diagnostics["information_object"] = "MU_exact_mobius_graph_backbone"
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
        )

    if method in (
        "edge_cara_mobius_tree", "edge_cara_mobius_tree_upward",
        "edge_cara_mobius_tree_shuffled",
    ):
        use_downward = method != "edge_cara_mobius_tree_upward"
        net = BidirectionalMobiusTreeEdgeCARA(
            EdgeCARA(edges.shape[-1], z_dim, z_dim, hidden),
            edges.shape[-1], mt, z_dim, hidden,
            use_upward=True, use_downward=use_downward,
        ).to(device)
        tree_permutation = torch.arange(edges.shape[1], device=device)
        if method == "edge_cara_mobius_tree_shuffled" and edges.shape[1] > 2:
            tree_permutation[1:] = torch.roll(tree_permutation[1:], shifts=1)
        opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        for _ in range(effective_updates):
            ix = next(_batches(tr, batch_size, rng)); rb = et[ix]
            if method == "edge_cara_mobius_tree_shuffled":
                # Negative structural control: retain node marginals and model
                # capacity but break which coefficient occupies which tree node.
                tree_input = rb[:, tree_permutation]
            else:
                tree_input = rb
            (
                zbar, common_z, base_tokens, context_tokens, _, _, _,
                coefficients, coefficient_hat, node_coefficient_hat, edge_hat, _,
            ) = net(tree_input)
            common = tree_input.mean(1)
            deviations = tree_input - common[:, None]
            common_hat = net.base.common_decoder(common_z)
            deviation_hat = net.base.context_decoder(
                context_tokens.reshape(-1, z_dim)
            ).reshape_as(tree_input)
            cbar = context_tokens.mean(1)
            base_loss = (
                F.mse_loss(common_hat, common)
                + F.mse_loss(common_hat[:, None] + deviation_hat, tree_input)
                + .5 * F.mse_loss(deviation_hat, deviations)
                + 2. * F.mse_loss(
                    base_tokens, common_z[:, None].expand_as(base_tokens)
                )
                + .02 * _cross_cov(common_z, cbar)
            )
            tree_loss = (
                F.mse_loss(coefficient_hat, coefficients)
                + F.mse_loss(edge_hat, tree_input)
                + .25 * F.mse_loss(node_coefficient_hat, coefficients)
            )
            loss = (
                base_loss + tree_loss + _variance_floor(zbar)
                + .02 * _offdiag_cov(zbar)
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.detach())
        net.eval()
        z = _encode_chunks(lambda sl: net(et[sl][:, tree_permutation])[0], len(obs))
        diagnostics.update(
            information_object="original_EdgeCARA_plus_bidirectional_Mobius_tree_residual",
            context_order=int(context_masks.sum(axis=1).max()),
            context_count=int(context_masks.shape[0]),
            upward_sweep=True,
            downward_sweep=use_downward,
            full_four_agent_tree=bool(
                context_masks.shape[-1] == 4
                and int(context_masks.sum(axis=1).max()) == 3
                and context_masks.shape[0] == 8
            ),
            final_tree_gate=float(torch.sigmoid(net.tree_gate_logit).detach().cpu()),
        )
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
        )

    if method == "multi_view_ae":
        net = MultiViewAE(edges.shape[-1], z_dim, hidden).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        for _ in range(effective_updates):
            ix = next(_batches(tr, batch_size, rng)); x = et[ix].reshape(-1, edges.shape[-1])
            zv = net.encoder(x); pred = net.decoder(zv)
            loss = F.mse_loss(pred, x); opt.zero_grad(); loss.backward(); opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.detach())
        z = _encode_chunks(
            lambda sl: net.encoder(et[sl].reshape(-1, edges.shape[-1])).reshape(
                et[sl].shape[0], edges.shape[1], z_dim
            ).mean(1),
            len(obs),
        )
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
        )

    if method == "edge_cara_mobius_attention":
        net = EdgeCARAMobiusAttention(
            edges.shape[-1], context_masks.shape[-1], z_dim, hidden
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        for _ in range(effective_updates):
            ix = next(_batches(tr, batch_size, rng)); rb = et[ix]
            zbar, direct_z, interaction_z, _ = net(rb, mt)
            direct_hat = net.direct_decoder(direct_z)
            direct = rb[:, 0]
            if rb.shape[1] > 1:
                residuals = rb[:, 1:] - direct[:, None]
                residual_hat = net.interaction_decoder(
                    interaction_z.reshape(-1, z_dim)
                ).reshape_as(residuals)
                reconstructed = direct_hat[:, None] + residual_hat
                structural_loss = (
                    F.mse_loss(residual_hat, residuals)
                    + F.mse_loss(reconstructed, rb[:, 1:])
                )
            else:
                structural_loss = rb.new_tensor(0.)
            loss = (
                F.mse_loss(direct_hat, direct)
                + structural_loss
                + _variance_floor(zbar)
                + .02 * _offdiag_cov(zbar)
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.detach())
        net.eval()
        z = _encode_chunks(lambda sl: net(et[sl], mt)[0], len(obs))
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
        )

    if method in ("edge_cara_mobius_lattice", "edge_cara_mobius_policy_aligned"):
        policy_aligned = method == "edge_cara_mobius_policy_aligned"
        om, os = fit_mean_std(obs[tr])
        obs_current = torch.as_tensor((obs - om) / os, device=device)
        net = EdgeCARAMobiusLattice(
            edges.shape[-1], mt, z_dim, hidden,
            policy_aligned=policy_aligned, obs_dim=obs.shape[-1],
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
        scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
        for _ in range(effective_updates):
            ix = next(_batches(tr, batch_size, rng)); rb = et[ix]
            zbar, tokens, coefficients, _ = net(rb)
            edge_hat, coefficient_hat = net.reconstruct(tokens)
            loss = (
                F.mse_loss(coefficient_hat, coefficients)
                + F.mse_loss(edge_hat, rb)
                + _variance_floor(zbar)
                + .02 * _offdiag_cov(zbar)
            )
            if policy_aligned:
                prior = net.latent_prior(obs_current[ix])
                # LPWM-style inverse/posterior to current-state prior alignment.
                # The weaker reverse term also shapes the edge latent while the
                # reconstruction and variance terms prevent a constant solution.
                loss = loss + F.mse_loss(prior, zbar.detach()) + .25 * F.mse_loss(
                    zbar, prior.detach()
                )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.detach())
        net.eval()
        z = _encode_chunks(lambda sl: net(et[sl])[0], len(obs))
        return RepresentationResult(
            z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
        )

    if method not in ("edge_cara", "edge_cara_root_only",
                      "edge_cara_no_invariance", "edge_cara_shuffled_cf"):
        raise ValueError(f"Unknown representation method: {method}")

    net = EdgeCARA(edges.shape[-1], z_dim, z_dim, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=effective_lr); loss_value = math.nan
    scheduler = _profile_scheduler(opt, settings.use_linear_warmup_decay, effective_updates)
    for _ in range(effective_updates):
        ix = next(_batches(tr, batch_size, rng)); rb = et[ix]
        common = rb.mean(1); q = rb - common[:, None]
        zv = net.z_encoder(rb.reshape(-1, rb.shape[-1])).reshape(
            len(ix), edges.shape[1], z_dim
        )
        cv = net.c_encoder(q.reshape(-1, q.shape[-1])).reshape(
            len(ix), edges.shape[1], z_dim
        )
        zbar = zv.mean(1); cbar = cv.mean(1)
        common_hat = net.common_decoder(zbar)
        q_hat = net.context_decoder(cv.reshape(-1, z_dim)).reshape_as(rb)
        invariance_weight = 0. if method == "edge_cara_no_invariance" else 2.
        loss = (
            F.mse_loss(common_hat, common)
            + F.mse_loss(common_hat[:, None] + q_hat, rb)
            + .5 * F.mse_loss(q_hat, q)
            + invariance_weight * F.mse_loss(zv, zbar[:, None].expand_as(zv))
            + _variance_floor(zbar) + .02 * _offdiag_cov(zbar) + .02 * _cross_cov(zbar, cbar)
        )
        opt.zero_grad(); loss.backward(); opt.step()
        if scheduler is not None:
            scheduler.step()
        loss_value = float(loss.detach())
    z = _encode_chunks(
        lambda sl: net.z_encoder(et[sl].reshape(-1, edges.shape[-1])).reshape(
            et[sl].shape[0], edges.shape[1], z_dim
        ).mean(1),
        len(obs),
    )
    return RepresentationResult(
        z, loss_value, sum(p.numel() for p in net.parameters()), diagnostics
    )


class RecurrentPolicy(nn.Module):
    def __init__(self, obs_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, obs: torch.Tensor, hidden_state=None):
        y, h = self.gru(obs, hidden_state)
        return self.head(y), h


class ActionDecoder(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, z):
        return self.net(z)


@dataclass
class LatentPipeline:
    policy: RecurrentPolicy
    decoder: ActionDecoder
    obs_mean: np.ndarray
    obs_std: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray
    discrete: bool

    def act_step(self, obs: np.ndarray, hidden_state, device: str = "cpu"):
        x = (np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std
        xt = torch.as_tensor(x[None, None], device=device)
        with torch.no_grad():
            z, h = self.policy(xt, hidden_state)
            out = self.decoder(z[:, -1])
        if self.discrete:
            return int(out.argmax(-1).item()), h
        return np.clip(out[0].cpu().numpy(), -1., 1.).astype(np.float32), h


def train_latent_pipeline(data: Dict[str, np.ndarray], z_flat: np.ndarray,
                          train_eps: np.ndarray, val_eps: np.ndarray, labeled_eps: np.ndarray,
                          seed: int, discrete: bool, action_dim: int,
                          policy_updates: int = 800, decoder_updates: int = 500,
                          batch_episodes: int = 16, lr: float = 1e-3,
                          device: str = "cpu",
                          shared_policy: LatentPipeline | None = None,
                          method: str = "unspecified",
                          training_profile: str = "legacy") -> Tuple[LatentPipeline, dict]:
    seed_all(seed)
    obs = data["obs"][:, :-1].astype(np.float32)
    actions = data["actions"]
    e, t, d = obs.shape
    z = z_flat.reshape(e, t, -1).astype(np.float32)
    mask = valid_step_mask(data)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    train_flat = valid_flat_indices(data, train_eps)
    om, os = ((shared_policy.obs_mean, shared_policy.obs_std) if shared_policy is not None
              else fit_mean_std(obs.reshape(-1, d)[train_flat]))
    zm, zs = ((shared_policy.z_mean, shared_policy.z_std) if shared_policy is not None
              else fit_mean_std(z.reshape(-1, z.shape[-1])[train_flat]))
    on = (obs - om) / os; zn = (z - zm) / zs
    ot = torch.as_tensor(on, device=device); zt = torch.as_tensor(zn, device=device)
    settings = resolve_method_training_settings(
        method=method,
        profile=training_profile,
        requested_representation_updates=policy_updates,
        requested_policy_updates=policy_updates,
        batch_size=max(1, batch_episodes * t),
        train_transition_count=max(1, len(train_flat)),
        default_lr=lr,
    )
    effective_policy_updates = settings.policy_updates
    effective_policy_lr = settings.policy_lr
    policy_loss = math.nan; policy_selected_update = 0
    if shared_policy is None:
        policy = RecurrentPolicy(d, z.shape[-1]).to(device)
        opt = torch.optim.Adam(policy.parameters(), lr=effective_policy_lr)
        scheduler = _profile_scheduler(
            opt, settings.use_linear_warmup_decay, effective_policy_updates
        )
        rng = np.random.default_rng(seed + 101)
        best_val = math.inf; best_state = None
        # A fixed cadence makes endpoint sweeps comparable. A horizon-dependent
        # cadence can select different checkpoints solely because a longer run
        # evaluated validation less often.
        check_every = 100
        for update in range(1, effective_policy_updates + 1):
            ix = rng.choice(train_eps, min(batch_episodes, len(train_eps)), replace=len(train_eps) < batch_episodes)
            pred, _ = policy(ot[ix]); selected_mask = mask_t[ix]
            loss = F.mse_loss(pred[selected_mask], zt[ix][selected_mask])
            opt.zero_grad(); loss.backward(); opt.step()
            if scheduler is not None:
                scheduler.step()
            policy_loss = float(loss.detach())
            if update == 1 or update % check_every == 0 or update == effective_policy_updates:
                with torch.no_grad():
                    vp, _ = policy(ot[val_eps]); val_mask = mask_t[val_eps]
                    candidate = float(F.mse_loss(vp[val_mask], zt[val_eps][val_mask]))
                if candidate < best_val:
                    best_val = candidate; policy_selected_update = update
                    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        if best_state is not None:
            policy.load_state_dict(best_state)
    else:
        policy = shared_policy.policy

    decoder = ActionDecoder(z.shape[-1], action_dim).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    # Deployment uses the latent policy output, not the encoder's oracle latent.
    # Training the decoder on oracle z creates a train/deploy distribution shift,
    # especially in long-horizon discrete control. Keep the policy frozen and use
    # its predicted latents on exactly the same scarce labeled trajectories.
    with torch.no_grad():
        deploy_z, _ = policy(ot[labeled_eps])
        validation_z, _ = policy(ot[val_eps])
    labeled_mask = mask_t[labeled_eps].reshape(-1)
    validation_mask = mask_t[val_eps].reshape(-1)
    lz = deploy_z.reshape(-1, z.shape[-1])[labeled_mask].detach()
    validation_z = validation_z.reshape(-1, z.shape[-1])[validation_mask].detach()
    if discrete:
        la = torch.as_tensor(actions[labeled_eps].reshape(-1), dtype=torch.long, device=device)[labeled_mask]
        validation_actions = torch.as_tensor(
            actions[val_eps].reshape(-1), dtype=torch.long, device=device
        )[validation_mask]
    else:
        la = torch.as_tensor(actions[labeled_eps].reshape(-1, action_dim), dtype=torch.float32, device=device)[labeled_mask]
        validation_actions = torch.as_tensor(
            actions[val_eps].reshape(-1, action_dim), dtype=torch.float32, device=device
        )[validation_mask]
    decoder_loss = math.nan; decoder_selected_update = 0
    best_decoder_val = math.inf; best_decoder_state = None
    decoder_check_every = 25
    for update in range(1, decoder_updates + 1):
        n = len(lz); ix = torch.randint(0, n, (min(512, n),), device=device)
        out = decoder(lz[ix]); loss = F.cross_entropy(out, la[ix]) if discrete else F.mse_loss(out, la[ix])
        opt.zero_grad(); loss.backward(); opt.step(); decoder_loss = float(loss.detach())
        if update == 1 or update % decoder_check_every == 0 or update == decoder_updates:
            with torch.no_grad():
                validation_output = decoder(validation_z)
                candidate = float(
                    F.cross_entropy(validation_output, validation_actions)
                    if discrete else F.mse_loss(validation_output, validation_actions)
                )
            if candidate < best_decoder_val:
                best_decoder_val = candidate; decoder_selected_update = update
                best_decoder_state = {
                    key: value.detach().cpu().clone()
                    for key, value in decoder.state_dict().items()
                }
    if best_decoder_state is not None:
        decoder.load_state_dict(best_decoder_state)

    with torch.no_grad():
        vpred, _ = policy(ot[val_eps]); val_mask = mask_t[val_eps]
        latent_val_mse = float(F.mse_loss(vpred[val_mask], zt[val_eps][val_mask]))
        vz = vpred.reshape(-1, z.shape[-1])[validation_mask]; va = decoder(vz)
        target = validation_actions
        action_val = float((va.argmax(-1) == target).float().mean()) if discrete else float(F.mse_loss(va, target))
    pipe = LatentPipeline(policy, decoder, om, os, zm, zs, discrete)
    return pipe, dict(policy_train_loss=policy_loss, decoder_train_loss=decoder_loss,
                      policy_reused=shared_policy is not None,
                      policy_selected_update=policy_selected_update,
                      policy_validation_cadence=100,
                      decoder_selected_update=decoder_selected_update,
                      decoder_validation_loss=best_decoder_val,
                      decoder_validation_cadence=25,
                      decoder_latent_source="predicted_frozen_policy",
                      training_profile=training_profile,
                      policy_effective_updates=int(effective_policy_updates),
                      policy_effective_lr=float(effective_policy_lr),
                      latent_val_mse=latent_val_mse,
                      action_val_accuracy=action_val if discrete else None,
                      action_val_mse=None if discrete else action_val)


@dataclass
class DirectPolicy:
    policy: RecurrentPolicy
    obs_mean: np.ndarray
    obs_std: np.ndarray
    discrete: bool

    def act_step(self, obs: np.ndarray, hidden_state, device: str = "cpu"):
        x = (np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std
        xt = torch.as_tensor(x[None, None], device=device)
        with torch.no_grad():
            out, h = self.policy(xt, hidden_state)
        if self.discrete:
            return int(out[0, -1].argmax().item()), h
        return np.clip(out[0, -1].cpu().numpy(), -1., 1.).astype(np.float32), h


def train_direct_bc(data: Dict[str, np.ndarray], fit_eps: np.ndarray, val_eps: np.ndarray,
                    seed: int, discrete: bool, action_dim: int, updates: int = 1000,
                    batch_episodes: int = 16, lr: float = 1e-3,
                    device: str = "cpu", class_balanced: bool = False,
                    selection_metric: str = "accuracy") -> Tuple[DirectPolicy, dict]:
    if selection_metric not in ("accuracy", "balanced_accuracy", "mse"):
        raise ValueError(f"Unknown BC selection metric: {selection_metric}")
    if not discrete and selection_metric != "mse":
        selection_metric = "mse"
    seed_all(seed)
    obs = data["obs"][:, :-1].astype(np.float32); actions = data["actions"]
    mask = valid_step_mask(data)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    fit_flat = valid_flat_indices(data, fit_eps)
    om, os = fit_mean_std(obs.reshape(-1, obs.shape[-1])[fit_flat]); on = (obs - om) / os
    ot = torch.as_tensor(on, device=device)
    target = torch.as_tensor(actions, dtype=torch.long if discrete else torch.float32, device=device)
    class_weights = None
    if discrete and class_balanced:
        fit_mask = mask_t[fit_eps]
        counts = torch.bincount(target[fit_eps][fit_mask], minlength=action_dim).float()
        present = counts > 0
        class_weights = torch.zeros_like(counts)
        class_weights[present] = counts[present].sum() / counts[present]
        class_weights[present] /= class_weights[present].mean()

    def classification_scores(logits, truth):
        pred = logits.argmax(-1)
        accuracy = float((pred == truth).float().mean())
        recalls = []
        for cls in range(action_dim):
            mask = truth == cls
            if bool(mask.any()):
                recalls.append((pred[mask] == truth[mask]).float().mean())
        balanced = float(torch.stack(recalls).mean()) if recalls else float("nan")
        return accuracy, balanced

    policy = RecurrentPolicy(obs.shape[-1], action_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr); rng = np.random.default_rng(seed + 313)
    train_loss = math.nan; best_score = -math.inf if discrete else math.inf
    best_state = None; best_update = 0; check_every = max(10, updates // 50)
    for update in range(1, updates + 1):
        ix = rng.choice(fit_eps, min(batch_episodes, len(fit_eps)), replace=len(fit_eps) < batch_episodes)
        pred, _ = policy(ot[ix])
        selected_mask = mask_t[ix]
        loss = (F.cross_entropy(pred[selected_mask], target[ix][selected_mask],
                                weight=class_weights) if discrete
                else F.mse_loss(pred[selected_mask], target[ix][selected_mask]))
        opt.zero_grad(); loss.backward(); opt.step(); train_loss = float(loss.detach())
        if update == 1 or update % check_every == 0 or update == updates:
            with torch.no_grad():
                vp, _ = policy(ot[val_eps])
                val_mask = mask_t[val_eps]
                if discrete:
                    val_accuracy, val_balanced = classification_scores(
                        vp[val_mask], target[val_eps][val_mask])
                    candidate = val_balanced if selection_metric == "balanced_accuracy" else val_accuracy
                else:
                    candidate = float(F.mse_loss(vp[val_mask], target[val_eps][val_mask]))
            improved = candidate > best_score if discrete else candidate < best_score
            if improved:
                best_score = candidate; best_update = update
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    if best_state is not None:
        policy.load_state_dict(best_state)
    with torch.no_grad():
        pred, _ = policy(ot[val_eps])
        val_mask = mask_t[val_eps]
        if discrete:
            score, balanced_score = classification_scores(
                pred[val_mask], target[val_eps][val_mask])
        else:
            score = float(F.mse_loss(pred[val_mask], target[val_eps][val_mask])); balanced_score = None
    return DirectPolicy(policy, om, os, discrete), dict(train_loss=train_loss,
        val_accuracy=score if discrete else None,
        val_balanced_accuracy=balanced_score if discrete else None,
        val_mse=None if discrete else score, selected_update=best_update,
        class_balanced=bool(class_balanced), selection_metric=selection_metric)


class InverseDynamics(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))

    def forward(self, obs, nxt):
        return self.net(torch.cat([obs, nxt], dim=-1))


def train_idm_relabel(data: Dict[str, np.ndarray], train_eps: np.ndarray, val_eps: np.ndarray,
                      labeled_eps: np.ndarray, seed: int, discrete: bool, action_dim: int,
                      idm_updates: int = 500, bc_updates: int = 1000,
                      lr: float = 1e-3, device: str = "cpu") -> Tuple[DirectPolicy, dict]:
    """Published-style scarce IDM: label all demonstrations, then behavior-clone them."""
    seed_all(seed)
    obs = data["obs"][:, :-1].astype(np.float32); nxt = data["obs"][:, 1:].astype(np.float32)
    actions = data["actions"]; mask = valid_step_mask(data)
    train_flat = valid_flat_indices(data, train_eps)
    om, os = fit_mean_std(obs.reshape(-1, obs.shape[-1])[train_flat])
    on, nnxt = (obs - om) / os, (nxt - om) / os
    ot, nt = torch.as_tensor(on, device=device), torch.as_tensor(nnxt, device=device)
    target = torch.as_tensor(actions, dtype=torch.long if discrete else torch.float32, device=device)
    idm = InverseDynamics(obs.shape[-1], action_dim).to(device); opt = torch.optim.Adam(idm.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 919); idm_loss = math.nan
    labeled_mask = torch.as_tensor(mask[labeled_eps].reshape(-1), dtype=torch.bool, device=device)
    labeled_flat_obs = ot[labeled_eps].reshape(-1, obs.shape[-1])[labeled_mask]
    labeled_flat_nxt = nt[labeled_eps].reshape(-1, obs.shape[-1])[labeled_mask]
    labeled_target = ((target[labeled_eps].reshape(-1)[labeled_mask]) if discrete else
                      target[labeled_eps].reshape(-1, action_dim)[labeled_mask])
    for _ in range(idm_updates):
        n = len(labeled_flat_obs); ix = torch.as_tensor(rng.integers(0, n, min(512, n)), device=device)
        pred = idm(labeled_flat_obs[ix], labeled_flat_nxt[ix])
        loss = F.cross_entropy(pred, labeled_target[ix]) if discrete else F.mse_loss(pred, labeled_target[ix])
        opt.zero_grad(); loss.backward(); opt.step(); idm_loss = float(loss.detach())
    with torch.no_grad():
        pseudo_logits = idm(ot.reshape(-1, obs.shape[-1]), nt.reshape(-1, obs.shape[-1]))
        pseudo = pseudo_logits.argmax(-1).reshape(actions.shape) if discrete else pseudo_logits.reshape(actions.shape)
    relabeled = dict(data); relabeled["actions"] = pseudo.cpu().numpy().astype(actions.dtype)
    policy, diag = train_direct_bc(relabeled, train_eps, val_eps, seed + 1, discrete, action_dim,
                                   updates=bc_updates, lr=lr, device=device)
    diag["idm_train_loss"] = idm_loss
    return policy, diag


def normalized_return(method_return: float, random_return: float, full_bc_return: float) -> float:
    den = full_bc_return - random_return
    # A reference that is not better than random is not a valid normalization anchor.
    return float((method_return - random_return) / den) if den > 1e-8 else math.nan
