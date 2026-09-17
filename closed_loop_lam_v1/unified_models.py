"""Counterfactual-set controls and unified incidence models.

These implementations are project-native. They do not copy external model
classes. Exact zeta/Mobius operators are buffers derived from dataset masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def lattice_operators(context_masks: torch.Tensor) -> dict[str, torch.Tensor]:
    masks = context_masks.detach().float()
    if masks.ndim != 2 or masks.shape[0] == 0 or torch.any(masks[0] != 0):
        raise ValueError("contexts must be [coalitions, agents] with the empty coalition first")
    subset = (masks[:, None, :] <= masks[None, :, :]).all(-1)
    zeta = subset.T.float()
    if int(torch.linalg.matrix_rank(zeta).item()) != zeta.shape[0]:
        raise ValueError("context family has a singular subset-incidence matrix")
    order = masks.sum(-1, keepdim=True)
    order_difference = order - order.T
    # hasse[parent, child] iff parent is a subset and child adds exactly one entity.
    hasse = (subset & (order_difference.T == 1)).float()
    undirected_hasse = hasse + hasse.T
    coalition_degree = undirected_hasse.sum(-1, keepdim=True).clamp_min(1.)
    entity_incidence = masks.clone()
    entity_degree = entity_incidence.sum(0, keepdim=True).T.clamp_min(1.)
    coalition_entity_degree = entity_incidence.sum(-1, keepdim=True).clamp_min(1.)
    return {
        "masks": masks,
        "zeta": zeta,
        "mobius": torch.linalg.inv(zeta),
        "order": order,
        "hasse": hasse,
        "undirected_hasse": undirected_hasse,
        "coalition_degree": coalition_degree,
        "entity_incidence": entity_incidence,
        "entity_degree": entity_degree,
        "coalition_entity_degree": coalition_entity_degree,
    }


class MatchedCFDeepSets(nn.Module):
    """Information-matched control over raw counterfactual context edges (GS)."""

    def __init__(self, state_dim: int, context_masks: torch.Tensor,
                 z_dim: int, hidden: int):
        super().__init__()
        ops = lattice_operators(context_masks)
        self.register_buffer("context_masks", ops["masks"])
        mask_dim = self.context_masks.shape[-1]
        self.edge_lift = nn.Sequential(
            nn.Linear(state_dim + mask_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * z_dim, hidden), nn.ReLU(), nn.Linear(hidden, z_dim)
        )
        self.output_norm = nn.LayerNorm(z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + mask_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, state_dim),
        )

    def forward(self, edges: torch.Tensor):
        masks = self.context_masks[None].expand(edges.shape[0], -1, -1)
        tokens = self.edge_lift(torch.cat([edges, masks], dim=-1))
        pooled = tokens.mean(1)
        z = self.output_norm(self.readout(torch.cat([tokens[:, 0], pooled], dim=-1)))
        global_tokens = z[:, None].expand(-1, edges.shape[1], -1)
        edge_hat = self.decoder(torch.cat([global_tokens, masks], dim=-1))
        return z, tokens, edge_hat


class MobiusSimple(nn.Module):
    """Exact Mobius coordinates with a deliberately simple shared encoder (MU-Simple)."""

    def __init__(self, state_dim: int, context_masks: torch.Tensor,
                 z_dim: int, hidden: int):
        super().__init__()
        ops = lattice_operators(context_masks)
        for name in ("context_masks", "zeta", "mobius", "order"):
            source = "masks" if name == "context_masks" else name
            self.register_buffer(name, ops[source])
        mask_dim = self.context_masks.shape[-1]
        self.coefficient_lift = nn.Sequential(
            nn.Linear(state_dim + mask_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * z_dim, hidden), nn.ReLU(), nn.Linear(hidden, z_dim)
        )
        self.output_norm = nn.LayerNorm(z_dim)
        self.coefficient_decoder = nn.Sequential(
            nn.Linear(z_dim + mask_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, edges: torch.Tensor):
        coefficients = torch.einsum("cd,bdk->bck", self.mobius, edges)
        masks = self.context_masks[None].expand(edges.shape[0], -1, -1)
        max_order = self.order.max().clamp_min(1.)
        orders = (self.order / max_order)[None].expand(edges.shape[0], -1, -1)
        tokens = self.coefficient_lift(torch.cat([coefficients, masks, orders], dim=-1))
        z = self.output_norm(self.readout(torch.cat([tokens[:, 0], tokens.mean(1)], dim=-1)))
        global_tokens = z[:, None].expand(-1, edges.shape[1], -1)
        coefficient_hat = self.coefficient_decoder(
            torch.cat([global_tokens, masks, orders], dim=-1)
        )
        edge_hat = torch.einsum("cd,bdk->bck", self.zeta, coefficient_hat)
        return z, tokens, coefficients, coefficient_hat, edge_hat


class _IncidenceFlowCell(nn.Module):
    def __init__(self, z_dim: int, hidden: int, message_types: int):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear((message_types + 1) * z_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )
        self.gates = nn.Parameter(torch.zeros(message_types))
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, state: torch.Tensor, messages: list[torch.Tensor]) -> torch.Tensor:
        if len(messages) != len(self.gates):
            raise ValueError("message count does not match typed incidence gates")
        gated = [torch.sigmoid(gate) * message for gate, message in zip(self.gates, messages)]
        return self.norm(state + self.update(torch.cat([state, *gated], dim=-1)))


class MobiusGraphEdgeCARA(nn.Module):
    """Stable main candidate: exact Mobius tokens plus Hasse/entity incidence flow."""

    def __init__(self, state_dim: int, context_masks: torch.Tensor,
                 z_dim: int, hidden: int, layers: int = 2):
        super().__init__()
        ops = lattice_operators(context_masks)
        for name in (
            "masks", "zeta", "mobius", "order", "hasse", "entity_incidence",
            "entity_degree", "coalition_entity_degree",
        ):
            self.register_buffer(name, ops[name])
        mask_dim = self.masks.shape[-1]
        self.coefficient_lift = nn.Sequential(
            nn.Linear(state_dim + mask_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )
        self.entity_seed = nn.Parameter(torch.randn(mask_dim, z_dim) * .02)
        self.coalition_cells = nn.ModuleList(
            [_IncidenceFlowCell(z_dim, hidden, message_types=3) for _ in range(layers)]
        )
        self.entity_cells = nn.ModuleList(
            [_IncidenceFlowCell(z_dim, hidden, message_types=1) for _ in range(layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * z_dim, hidden), nn.SiLU(), nn.Linear(hidden, z_dim)
        )
        self.output_norm = nn.LayerNorm(z_dim)
        self.coefficient_decoder = nn.Sequential(
            nn.Linear(z_dim + mask_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, edges: torch.Tensor):
        coefficients = torch.einsum("cd,bdk->bck", self.mobius, edges)
        masks = self.masks[None].expand(edges.shape[0], -1, -1)
        orders = (self.order / self.order.max().clamp_min(1.))[None].expand(
            edges.shape[0], -1, -1
        )
        coalition = self.coefficient_lift(torch.cat([coefficients, masks, orders], dim=-1))
        entity = self.entity_seed[None].expand(edges.shape[0], -1, -1)

        for coalition_cell, entity_cell in zip(self.coalition_cells, self.entity_cells):
            entity_message = torch.einsum(
                "ca,bcz->baz", self.entity_incidence, coalition
            ) / self.entity_degree[None]
            entity = entity_cell(entity, [entity_message])
            from_entities = torch.einsum(
                "ca,baz->bcz", self.entity_incidence, entity
            ) / self.coalition_entity_degree[None]
            from_parents = torch.einsum("pc,bpz->bcz", self.hasse, coalition)
            parent_degree = self.hasse.sum(0).clamp_min(1.)[None, :, None]
            from_parents = from_parents / parent_degree
            from_children = torch.einsum("pc,bcz->bpz", self.hasse, coalition)
            child_degree = self.hasse.sum(1).clamp_min(1.)[None, :, None]
            from_children = from_children / child_degree
            coalition = coalition_cell(
                coalition, [from_parents, from_children, from_entities]
            )

        z = self.output_norm(self.readout(
            torch.cat([coalition[:, 0], coalition.mean(1)], dim=-1)
        ))
        global_tokens = z[:, None].expand(-1, edges.shape[1], -1)
        coefficient_hat = self.coefficient_decoder(
            torch.cat([global_tokens, masks, orders], dim=-1)
        )
        edge_hat = torch.einsum("cd,bdk->bck", self.zeta, coefficient_hat)
        return z, coalition, entity, coefficients, coefficient_hat, edge_hat


class BidirectionalMobiusTreeEdgeCARA(nn.Module):
    """Original Edge-CARA with an explicit bottom-up/top-down Mobius-tree residual.

    The observed, downward-closed coalition family is a Hasse DAG (a tree only
    when every non-root node has one parent).  We retain the historical
    ``Mobius Tree`` name while exposing the mathematically precise object.  A
    complete four-agent/one-ego experiment uses all eight subsets of the three
    non-ego agents, so the sweeps have depths 3 -> 0 and 0 -> 3.

    ``base`` must be an EdgeCARA-compatible module.  Keeping it as an injected
    module makes the ablation literal: this model contains the same z/c
    encoders and common/context decoders as the runnable original Edge-CARA.
    """

    def __init__(self, base: nn.Module, state_dim: int,
                 context_masks: torch.Tensor, z_dim: int, hidden: int,
                 *, use_upward: bool = True, use_downward: bool = True):
        super().__init__()
        if not use_upward and not use_downward:
            raise ValueError("Mobius Tree must enable at least one directed sweep")
        ops = lattice_operators(context_masks)
        for name in ("masks", "zeta", "mobius", "order", "hasse"):
            self.register_buffer(name, ops[name])
        self.base = base
        self.use_upward = bool(use_upward)
        self.use_downward = bool(use_downward)
        mask_dim = self.masks.shape[-1]
        self.coefficient_lift = nn.Sequential(
            nn.Linear(state_dim + mask_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )
        self.upward_cell = _IncidenceFlowCell(z_dim, hidden, message_types=1)
        self.downward_cell = _IncidenceFlowCell(z_dim, hidden, message_types=1)
        self.tree_readout = nn.Sequential(
            nn.Linear(3 * z_dim, hidden), nn.SiLU(), nn.Linear(hidden, z_dim)
        )
        # Start near the original Edge-CARA while allowing useful tree signal
        # immediately.  sigmoid(-2.197...) = 0.1.
        self.tree_gate_logit = nn.Parameter(torch.tensor(-2.1972246))
        self.output_norm = nn.LayerNorm(z_dim)
        self.global_coefficient_decoder = nn.Sequential(
            nn.Linear(z_dim + mask_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        # This auxiliary head makes both directed sweeps directly trainable;
        # the main global decoder above still forces the exported latent to
        # carry enough information to reconstruct the counterfactual complex.
        self.node_coefficient_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.SiLU(), nn.Linear(hidden, state_dim)
        )

    def _level_mask(self, level: int, state: torch.Tensor) -> torch.Tensor:
        return (self.order.squeeze(-1) == float(level))[None, :, None].to(state.device)

    def forward(self, edges: torch.Tensor):
        batch, contexts, _ = edges.shape
        coefficients = torch.einsum("cd,bdk->bck", self.mobius, edges)
        masks = self.masks[None].expand(batch, -1, -1)
        max_order = int(self.order.max().item())
        orders = (self.order / max(float(max_order), 1.))[None].expand(batch, -1, -1)
        initial = self.coefficient_lift(torch.cat([coefficients, masks, orders], dim=-1))

        upward = initial
        if self.use_upward:
            child_degree = self.hasse.sum(1).clamp_min(1.)[None, :, None]
            for level in range(max_order - 1, -1, -1):
                from_children = torch.einsum("pc,bcz->bpz", self.hasse, upward)
                candidate = self.upward_cell(upward, [from_children / child_degree])
                upward = torch.where(self._level_mask(level, upward), candidate, upward)

        downward = upward
        if self.use_downward:
            parent_degree = self.hasse.sum(0).clamp_min(1.)[None, :, None]
            for level in range(1, max_order + 1):
                from_parents = torch.einsum("pc,bpz->bcz", self.hasse, downward)
                candidate = self.downward_cell(downward, [from_parents / parent_degree])
                downward = torch.where(self._level_mask(level, downward), candidate, downward)

        # Literal original Edge-CARA branch.
        base_tokens = self.base.z_encoder(edges.reshape(-1, edges.shape[-1])).reshape(
            batch, contexts, -1
        )
        common_z = base_tokens.mean(1)
        deviations = edges - edges.mean(1, keepdim=True)
        context_tokens = self.base.c_encoder(
            deviations.reshape(-1, deviations.shape[-1])
        ).reshape(batch, contexts, -1)

        order_weight = (1. + self.order.squeeze(-1))[None, :, None]
        order_pool = (downward * order_weight).sum(1) / order_weight.sum(1)
        tree_z = self.tree_readout(torch.cat(
            [downward[:, 0], downward.mean(1), order_pool], dim=-1
        ))
        gate = torch.sigmoid(self.tree_gate_logit)
        z = self.output_norm(common_z + gate * tree_z)

        global_z = z[:, None].expand(-1, contexts, -1)
        coefficient_hat = self.global_coefficient_decoder(
            torch.cat([global_z, masks, orders], dim=-1)
        )
        edge_hat = torch.einsum("cd,bdk->bck", self.zeta, coefficient_hat)
        node_coefficient_hat = self.node_coefficient_decoder(downward)
        return (
            z, common_z, base_tokens, context_tokens, initial, upward, downward,
            coefficients, coefficient_hat, node_coefficient_hat, edge_hat, gate,
        )


def _normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    adjacency = adjacency.detach().float()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(1.)
    return adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1.)


class MIFCARAIncidenceFlow(nn.Module):
    """Rich-data MIF-CARA prototype over time x coalition x entity cells.

    The coalition axis is converted to exact Mobius coefficients before typed
    incidence flow and converted back with zeta inversion for reconstruction.
    It requires explicit time and entity axes and therefore refuses legacy
    single-horizon flat-state datasets.
    """

    def __init__(self, value_dim: int, context_masks: torch.Tensor,
                 entity_adjacency: torch.Tensor, time_steps: int,
                 z_dim: int, hidden: int, layers: int = 3):
        super().__init__()
        if time_steps < 2:
            raise ValueError("MIF-CARA requires at least two explicit time/horizon cells")
        ops = lattice_operators(context_masks)
        coalition_adjacency = ops["undirected_hasse"]
        coalition_adjacency.fill_diagonal_(1.)
        self.register_buffer("mobius", ops["mobius"])
        self.register_buffer("zeta", ops["zeta"])
        self.register_buffer("coalition_adjacency", _normalized_adjacency(coalition_adjacency))
        self.register_buffer("entity_adjacency", _normalized_adjacency(entity_adjacency))
        time_adjacency = torch.zeros(time_steps, time_steps)
        index = torch.arange(time_steps - 1)
        time_adjacency[index, index + 1] = 1.
        time_adjacency[index + 1, index] = 1.
        self.register_buffer("time_adjacency", _normalized_adjacency(time_adjacency))
        self.register_buffer("context_masks", ops["masks"])
        self.value_dim = int(value_dim)
        self.time_embedding = nn.Parameter(torch.randn(time_steps, z_dim) * .02)
        self.coalition_embedding = nn.Parameter(
            torch.randn(self.coalition_adjacency.shape[0], z_dim) * .02
        )
        self.entity_embedding = nn.Parameter(
            torch.randn(self.entity_adjacency.shape[0], z_dim) * .02
        )
        self.input_lift = nn.Sequential(
            nn.Linear(value_dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, z_dim)
        )
        self.cells = nn.ModuleList(
            [_IncidenceFlowCell(z_dim, hidden, message_types=3) for _ in range(layers)]
        )
        self.value_decoder = nn.Sequential(
            nn.Linear(2 * z_dim, hidden), nn.SiLU(), nn.Linear(hidden, value_dim)
        )
        self.readout = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.SiLU(), nn.Linear(hidden, z_dim), nn.LayerNorm(z_dim)
        )

    def forward(self, values: torch.Tensor, visible_mask: torch.Tensor):
        if values.ndim != 5:
            raise ValueError("values must be [batch,time,coalition,entity,value_dim]")
        expected = (
            self.time_adjacency.shape[0], self.coalition_adjacency.shape[0],
            self.entity_adjacency.shape[0], self.value_dim,
        )
        if tuple(values.shape[1:]) != expected:
            raise ValueError(f"MIF-CARA expected trailing shape {expected}, got {tuple(values.shape[1:])}")
        if visible_mask.shape != values.shape[:-1] + (1,):
            raise ValueError("visible_mask must match values except for a singleton feature axis")
        visible = visible_mask.to(values.dtype)
        coefficients = torch.einsum("cd,btdek->btcek", self.mobius, values)
        state = self.input_lift(torch.cat([coefficients * visible, visible], dim=-1))
        for cell in self.cells:
            coalition_message = torch.einsum(
                "cd,btdez->btcez", self.coalition_adjacency, state
            )
            entity_message = torch.einsum(
                "ef,btcfz->btcez", self.entity_adjacency, state
            )
            time_message = torch.einsum(
                "tu,buced->btced", self.time_adjacency, state
            )
            state = cell(state, [coalition_message, entity_message, time_message])
        latent = self.readout(state[:, -1, 0].mean(1))
        coordinate = (
            self.time_embedding[:, None, None, :]
            + self.coalition_embedding[None, :, None, :]
            + self.entity_embedding[None, None, :, :]
        )
        coordinate = coordinate[None].expand(values.shape[0], -1, -1, -1, -1)
        global_latent = latent[:, None, None, None, :].expand_as(coordinate)
        reconstructed_coefficients = self.value_decoder(
            torch.cat([global_latent, coordinate], dim=-1)
        )
        reconstructed = torch.einsum(
            "cd,btdek->btcek", self.zeta, reconstructed_coefficients
        )
        return latent, reconstructed, state
