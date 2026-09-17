from __future__ import annotations
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None
MPE_AVAILABLE = True
MPE_BACKEND = 'unknown'
_MPE_IMPORT_ERROR = None
try:
    from mpe2 import simple_spread_v3
    MPE_BACKEND = 'mpe2'
except Exception as e1:
    try:
        from pettingzoo.mpe import simple_spread_v3
        MPE_BACKEND = 'pettingzoo.mpe'
    except Exception as e2:
        MPE_AVAILABLE = False
        _MPE_IMPORT_ERROR = (e1, e2)
        simple_spread_v3 = None
MECHANISMS = ['static', 'periodic', 'independent', 'mimic', 'follow', 'compete', 'cooperate', 'adversarial']
MECH_TO_ID = {m: i for i, m in enumerate(MECHANISMS)}
ID_TO_MECH = {i: m for m, i in MECH_TO_ID.items()}
TOPOLOGIES = ['star', 'chain', 'team', 'pairwise', 'dense']
TOPO_TO_ID = {t: i for i, t in enumerate(TOPOLOGIES)}
ID_TO_TOPO = {i: t for t, i in TOPO_TO_ID.items()}
RELATION_TYPES = ['none', 'static', 'periodic', 'independent', 'mimic', 'follow', 'compete', 'cooperate', 'block', 'adversarial', 'avoid', 'near', 'same_landmark', 'collision_risk', 'velocity_alignment']
REL_TO_ID = {r: i for i, r in enumerate(RELATION_TYPES)}
ID_TO_REL = {i: r for r, i in REL_TO_ID.items()}
DIRECTION_LABEL_DIM = 9
NOOP_DIRECTION_ID = 0
METHOD_ALIASES = {'vanilla': 'lam', 'lam': 'lam', 'lam_delta': 'lam_delta', 'delta_lam': 'lam_delta', 'lam_null': 'lam_null', 'lam_plus_null': 'lam_null', 'cara_direct': 'cara_lam_direct', 'cara_lam_direct': 'cara_lam_direct', 'cara_total': 'cara_lam_total', 'cara_lam_total': 'cara_lam_total', 'factorized_cara': 'factorized_cara_lam', 'factorized_cara_lam': 'factorized_cara_lam', 'oracle_action': 'oracle_action'}
DEFAULT_METHODS = 'lam,lam_delta,lam_null,cara_lam_direct,cara_lam_total,oracle_action,factorized_cara_lam'
FORCE8 = np.asarray([[1.0, 0.0], [math.sqrt(0.5), math.sqrt(0.5)], [0.0, 1.0], [-math.sqrt(0.5), math.sqrt(0.5)], [-1.0, 0.0], [-math.sqrt(0.5), -math.sqrt(0.5)], [0.0, -1.0], [math.sqrt(0.5), -math.sqrt(0.5)]], dtype=np.float32)
args_obs_mode_global = 'state_native'
args_pixel_size_global = 32

# Final implementation source: whosmove_lam_pettingzoo_mpe_v2_1_multistep.py:135

def require_mpe() -> None:
    if not MPE_AVAILABLE:
        raise RuntimeError(
            "MPE/PettingZoo environment could not be imported.\n"
            "Install with:\n\n"
            "    pip install mpe2 pettingzoo gymnasium pygame\n\n"
            f"Original import errors: {_MPE_IMPORT_ERROR}"
        )

# Final implementation source: whosmove_lam_pettingzoo_mpe_v2_1_multistep.py:160

def one_hot(index: int, n: int) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    if 0 <= int(index) < n:
        x[int(index)] = 1.0
    return x

# Final implementation source: whosmove_lam_pettingzoo_mpe_v2_1_multistep.py:174

def relation_for_mechanism(mechanism: str) -> str:
    return {
        "static": "static",
        "periodic": "periodic",
        "independent": "independent",
        "mimic": "mimic",
        "follow": "follow",
        "compete": "compete",
        "cooperate": "cooperate",
        "adversarial": "adversarial",
    }[mechanism]

# Final implementation source: whosmove_lam_pettingzoo_mpe_v2_1_multistep.py:187

def resize_frame_nn(frame: np.ndarray, size: int) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    h, w = frame.shape[:2]
    ys = np.linspace(0, h - 1, int(size)).round().astype(np.int64)
    xs = np.linspace(0, w - 1, int(size)).round().astype(np.int64)
    return frame[ys][:, xs]

# Final implementation source: whosmove_lam_pettingzoo_mpe_v2_1_multistep.py:199

def safe_norm(x: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    return np.linalg.norm(np.asarray(x, dtype=np.float32), axis=axis, keepdims=keepdims)

# Final implementation source: whosmove_lam_pettingzoo_mpe_v2_1_multistep.py:203

def unit_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)

# Final implementation source: whosmove_lam_pettingzoo_mpe_v2_1_multistep.py:211

def force_to_direction_id(force: np.ndarray, eps: float = 1e-5) -> int:
    f = np.asarray(force, dtype=np.float32).ravel()
    if f.shape[0] < 2 or float(np.linalg.norm(f[:2])) < eps:
        return 0
    ang = math.atan2(float(f[1]), float(f[0]))
    if ang < 0:
        ang += 2 * math.pi
    bucket = int(np.floor((ang + math.pi / 8) / (math.pi / 4))) % 8
    return 1 + bucket

# Final implementation source: whosmove_lam_pettingzoo_mpe_v2_1_multistep.py:222

def direction_id_to_force(idx: int) -> np.ndarray:
    idx = int(idx)
    if idx <= 0:
        return np.zeros(2, dtype=np.float32)
    return FORCE8[(idx - 1) % 8].copy()

# Final implementation source: whosmove_lam_pettingzoo_mpe_v2_1_multistep.py:229

def discrete_id_to_force(a: int) -> np.ndarray:
    # V1 / PettingZoo discrete convention: 0 noop, 1 left, 2 right, 3 down, 4 up.
    a = int(a)
    if a == 1:
        return np.asarray([-1.0, 0.0], dtype=np.float32)
    if a == 2:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    if a == 3:
        return np.asarray([0.0, -1.0], dtype=np.float32)
    if a == 4:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    return np.zeros(2, dtype=np.float32)

# Final implementation source: whosmove_lam_pettingzoo_mpe_v2_1_multistep.py:354

class WhoseMoveMPENativeEnv:
    def __init__(
        self,
        num_agents: int = 4,
        mechanism: str = "mixed",
        topology: str = "star",
        topology_mode: str = "template",
        action_mode: str = "continuous",
        relation_strength: float = 1.0,
        edge_dropout: float = 0.0,
        edge_type_noise: float = 0.0,
        edge_weight_noise: float = 0.0,
        dynamic_edge_radius: float = 0.75,
        collision_radius: float = 0.18,
        blocking_width: float = 0.20,
        force_scale: float = 1.0,
        seed: int = 0,
        max_cycles: int = 64,
        render_mode: str = "rgb_array",
    ):
        require_mpe()
        if mechanism != "mixed" and mechanism not in MECH_TO_ID:
            raise ValueError(f"Unknown mechanism {mechanism}")
        if topology not in TOPO_TO_ID:
            raise ValueError(f"Unknown topology {topology}")
        if topology_mode not in ["template", "dynamic", "hybrid"]:
            raise ValueError("topology_mode must be template, dynamic, or hybrid")
        if action_mode not in ["discrete", "force8", "continuous"]:
            raise ValueError("action_mode must be discrete, force8, or continuous")
        self.num_agents = int(num_agents)
        self.mechanism = mechanism
        self.topology = topology
        self.topology_mode = topology_mode
        self.action_mode = action_mode
        self.relation_strength = float(relation_strength)
        self.edge_dropout = float(edge_dropout)
        self.edge_type_noise = float(edge_type_noise)
        self.edge_weight_noise = float(edge_weight_noise)
        self.dynamic_edge_radius = float(dynamic_edge_radius)
        self.collision_radius = float(collision_radius)
        self.blocking_width = float(blocking_width)
        self.force_scale = float(force_scale)
        self.seed = int(seed)
        self.max_cycles = int(max_cycles)
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.previous_forces = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.previous_action_ids = np.zeros(self.num_agents, dtype=np.int64)
        self.last_factual_forces = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.last_factual_action_ids = np.zeros(self.num_agents, dtype=np.int64)
        self.relation_graph = np.zeros((self.num_agents, self.num_agents), dtype=np.int64)
        self.relation_weight = np.zeros((self.num_agents, self.num_agents), dtype=np.float32)
        self.dynamic_relation_graph = np.zeros((self.num_agents, self.num_agents), dtype=np.int64)
        self.dynamic_relation_weight = np.zeros((self.num_agents, self.num_agents), dtype=np.float32)
        self._make_env(seed)

    def _make_env(self, seed: int) -> None:
        self.env = simple_spread_v3.parallel_env(
            N=self.num_agents,
            local_ratio=0.5,
            max_cycles=self.max_cycles,
            continuous_actions=True,
            render_mode=self.render_mode,
        )
        self.obs, self.infos = self.env.reset(seed=int(seed))
        self.agent_names = list(getattr(self.env, "possible_agents", list(self.obs.keys())))[: self.num_agents]
        self.relation_graph, self.relation_weight = self.sample_template_relation_graph()
        self.update_dynamic_relation_graph()

    def get_world(self) -> Any:
        obj = self.env
        for _ in range(8):
            if hasattr(obj, "unwrapped"):
                unwrapped = getattr(obj, "unwrapped")
                if unwrapped is not obj:
                    obj = unwrapped
                    continue
            if hasattr(obj, "aec_env"):
                obj = getattr(obj, "aec_env")
                continue
            break
        if hasattr(obj, "world"):
            return getattr(obj, "world")
        raise RuntimeError("Could not access MPE world from PettingZoo/MPE wrapper.")

    def _capture_world_state(self) -> Dict[str, Any]:
        world = self.get_world()
        agents = []
        for a in list(getattr(world, "agents", [])):
            item = {
                "p_pos": np.asarray(a.state.p_pos, dtype=np.float64).copy(),
                "p_vel": np.asarray(a.state.p_vel, dtype=np.float64).copy(),
            }
            if hasattr(a.state, "c"):
                item["c"] = np.asarray(a.state.c, dtype=np.float64).copy()
            action = getattr(a, "action", None)
            if action is not None:
                try:
                    item["action_u"] = np.asarray(action.u, dtype=np.float64).copy()
                except Exception:
                    pass
                try:
                    item["action_c"] = np.asarray(action.c, dtype=np.float64).copy()
                except Exception:
                    pass
            agents.append(item)
        landmarks = []
        for lm in list(getattr(world, "landmarks", [])):
            landmarks.append({
                "p_pos": np.asarray(lm.state.p_pos, dtype=np.float64).copy(),
                "p_vel": np.asarray(lm.state.p_vel, dtype=np.float64).copy(),
            })
        return {"agents": agents, "landmarks": landmarks}

    def _restore_world_state(self, state: Dict[str, Any]) -> None:
        world = self.get_world()
        for a, item in zip(list(getattr(world, "agents", [])), state.get("agents", [])):
            a.state.p_pos = np.asarray(item["p_pos"], dtype=np.float64).copy()
            a.state.p_vel = np.asarray(item["p_vel"], dtype=np.float64).copy()
            if "c" in item and hasattr(a.state, "c"):
                a.state.c = np.asarray(item["c"], dtype=np.float64).copy()
            action = getattr(a, "action", None)
            if action is not None and "action_u" in item:
                try:
                    action.u = np.asarray(item["action_u"], dtype=np.float64).copy()
                except Exception:
                    pass
            if action is not None and "action_c" in item:
                try:
                    action.c = np.asarray(item["action_c"], dtype=np.float64).copy()
                except Exception:
                    pass
        for lm, item in zip(list(getattr(world, "landmarks", [])), state.get("landmarks", [])):
            lm.state.p_pos = np.asarray(item["p_pos"], dtype=np.float64).copy()
            lm.state.p_vel = np.asarray(item["p_vel"], dtype=np.float64).copy()
        self._refresh_obs_from_world()

    def _refresh_obs_from_world(self) -> None:
        vec = self.state_vector()
        self.obs = {name: vec.copy() for name in self.agent_names}
        self.infos = {name: {} for name in self.agent_names}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "world_state": self._capture_world_state(),
            "previous_forces": self.previous_forces.copy(),
            "previous_action_ids": self.previous_action_ids.copy(),
            "last_factual_forces": self.last_factual_forces.copy(),
            "last_factual_action_ids": self.last_factual_action_ids.copy(),
            "relation_graph": self.relation_graph.copy(),
            "relation_weight": self.relation_weight.copy(),
            "dynamic_relation_graph": self.dynamic_relation_graph.copy(),
            "dynamic_relation_weight": self.dynamic_relation_weight.copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        self.previous_forces = np.asarray(snap["previous_forces"], dtype=np.float32).copy()
        self.previous_action_ids = np.asarray(snap["previous_action_ids"], dtype=np.int64).copy()
        self.last_factual_forces = np.asarray(snap["last_factual_forces"], dtype=np.float32).copy()
        self.last_factual_action_ids = np.asarray(snap["last_factual_action_ids"], dtype=np.int64).copy()
        self.relation_graph = np.asarray(snap["relation_graph"], dtype=np.int64).copy()
        self.relation_weight = np.asarray(snap["relation_weight"], dtype=np.float32).copy()
        self.dynamic_relation_graph = np.asarray(snap["dynamic_relation_graph"], dtype=np.int64).copy()
        self.dynamic_relation_weight = np.asarray(snap["dynamic_relation_weight"], dtype=np.float32).copy()
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(snap["rng_state"])
        self._restore_world_state(snap["world_state"])

    # ----------------------- physical state helpers --------------------------
    def agent_pos(self, i: int) -> np.ndarray:
        world = self.get_world()
        return np.asarray(world.agents[int(i)].state.p_pos, dtype=np.float32)

    def agent_vel(self, i: int) -> np.ndarray:
        world = self.get_world()
        return np.asarray(world.agents[int(i)].state.p_vel, dtype=np.float32)

    def landmark_pos(self, i: int) -> np.ndarray:
        world = self.get_world()
        if not getattr(world, "landmarks", None):
            return np.zeros(2, dtype=np.float32)
        idx = int(np.clip(int(i), 0, len(world.landmarks) - 1))
        return np.asarray(world.landmarks[idx].state.p_pos, dtype=np.float32)

    def all_agent_pos(self) -> np.ndarray:
        return np.stack([self.agent_pos(i) for i in range(self.num_agents)], axis=0).astype(np.float32)

    def all_agent_vel(self) -> np.ndarray:
        return np.stack([self.agent_vel(i) for i in range(self.num_agents)], axis=0).astype(np.float32)

    def all_landmark_pos(self) -> np.ndarray:
        return np.stack([self.landmark_pos(i) for i in range(self.num_agents)], axis=0).astype(np.float32)

    def pairwise_agent_distances(self) -> np.ndarray:
        pos = self.all_agent_pos()
        return safe_norm(pos[:, None, :] - pos[None, :, :], axis=-1).astype(np.float32)

    def agent_landmark_distances(self) -> np.ndarray:
        pos = self.all_agent_pos()
        lm = self.all_landmark_pos()
        return safe_norm(pos[:, None, :] - lm[None, :, :], axis=-1).astype(np.float32)

    def nearest_landmark_ids(self) -> np.ndarray:
        return np.argmin(self.agent_landmark_distances(), axis=1).astype(np.int64)

    def collision_risk_matrix(self) -> np.ndarray:
        pos = self.all_agent_pos()
        vel = self.all_agent_vel()
        rel_pos = pos[:, None, :] - pos[None, :, :]
        rel_vel = vel[:, None, :] - vel[None, :, :]
        dist = safe_norm(rel_pos, axis=-1)
        closing = np.sum(rel_pos * rel_vel, axis=-1) < 0.0
        risk = ((dist < self.collision_radius) & closing).astype(np.float32)
        np.fill_diagonal(risk, 0.0)
        return risk

    def blocking_matrix(self) -> np.ndarray:
        n = self.num_agents
        mat = np.zeros((n, n), dtype=np.float32)
        ego = self.agent_pos(0)
        tgt = self.landmark_pos(int(self.nearest_landmark_ids()[0]))
        segment = tgt - ego
        seg_len2 = float(np.dot(segment, segment)) + 1e-8
        for i in range(1, n):
            p = self.agent_pos(i)
            alpha = float(np.dot(p - ego, segment) / seg_len2)
            if 0.0 < alpha < 1.0:
                closest = ego + alpha * segment
                d = float(np.linalg.norm(p - closest))
                if d < self.blocking_width:
                    mat[i, 0] = 1.0
        return mat

    def physical_features(self) -> Dict[str, np.ndarray]:
        return {
            "pos": self.all_agent_pos(),
            "vel": self.all_agent_vel(),
            "landmarks": self.all_landmark_pos(),
            "pairwise_dist": self.pairwise_agent_distances(),
            "landmark_dist": self.agent_landmark_distances(),
            "nearest_landmark": self.nearest_landmark_ids(),
            "collision_risk": self.collision_risk_matrix(),
            "blocking": self.blocking_matrix(),
        }

    def state_vector(self) -> np.ndarray:
        parts: List[np.ndarray] = []
        for i in range(self.num_agents):
            parts.append(self.agent_pos(i))
            parts.append(self.agent_vel(i))
        for j in range(self.num_agents):
            parts.append(self.landmark_pos(j))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def state_native_vector(self) -> np.ndarray:
        pos = self.all_agent_pos()
        vel = self.all_agent_vel()
        lm = self.all_landmark_pos()
        pair_d = self.pairwise_agent_distances()
        lm_d = self.agent_landmark_distances()
        collision = self.collision_risk_matrix()
        nearest = self.nearest_landmark_ids()
        parts: List[np.ndarray] = [self.state_vector()]
        # Relative coordinates to ego give the representation direct physical meaning.
        parts.append((pos[1:] - pos[0]).reshape(-1).astype(np.float32))
        parts.append((vel[1:] - vel[0]).reshape(-1).astype(np.float32))
        parts.append(pair_d.reshape(-1).astype(np.float32))
        parts.append(lm_d.reshape(-1).astype(np.float32))
        parts.append(collision.reshape(-1).astype(np.float32))
        parts.append(one_hot(int(nearest[0]), self.num_agents))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def pixel_vector(self, pixel_size: int = 32) -> np.ndarray:
        try:
            frame = self.env.render()
            if isinstance(frame, list):
                frame = frame[0]
            frame = np.asarray(frame)
            if frame.size == 0:
                raise RuntimeError("empty render")
            pix = resize_frame_nn(frame, int(pixel_size)).astype(np.float32) / 255.0
            return pix.reshape(-1).astype(np.float32)
        except Exception:
            return self.state_vector()

    def observation_vector(self, obs_mode: str = "state_native", pixel_size: int = 32) -> np.ndarray:
        if obs_mode == "state":
            return self.state_vector()
        if obs_mode == "state_native":
            return self.state_native_vector()
        pix = self.pixel_vector(pixel_size=pixel_size)
        if obs_mode == "pixel":
            return pix
        if obs_mode == "state_pixel":
            return np.concatenate([self.state_native_vector(), pix], axis=0).astype(np.float32)
        raise ValueError("obs_mode must be state, state_native, pixel, or state_pixel")

    # ----------------------- topology and mechanism --------------------------
    def _mechanism_for_agent(self, i: int) -> str:
        if self.mechanism != "mixed":
            return self.mechanism
        options = ["independent", "mimic", "follow", "compete", "cooperate", "adversarial", "periodic"]
        return options[(i - 1) % len(options)] if i > 0 else "independent"

    def sample_template_relation_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.num_agents
        g = np.zeros((n, n), dtype=np.int64)
        w = np.zeros((n, n), dtype=np.float32)
        rng = np.random.default_rng(self.seed + 13007)

        def add(src: int, tgt: int, rel: str, strength: Optional[float] = None):
            if src == tgt or src < 0 or tgt < 0 or src >= n or tgt >= n:
                return
            g[src, tgt] = REL_TO_ID[rel]
            w[src, tgt] = self.relation_strength if strength is None else float(strength)

        if self.mechanism == "static":
            for i in range(1, n):
                add(i, 0, "static")
        elif self.topology == "star":
            for i in range(1, n):
                add(i, 0, relation_for_mechanism(self._mechanism_for_agent(i)))
        elif self.topology == "chain":
            for i in range(1, n):
                add(i, i - 1, relation_for_mechanism(self._mechanism_for_agent(i)))
        elif self.topology == "team":
            if n > 1:
                add(1, 0, "cooperate")
            for i in range(2, n):
                add(i, 0 if i % 2 == 0 else 1, "compete" if i % 2 == 0 else "adversarial")
        elif self.topology == "pairwise":
            for i in range(1, n, 2):
                add(i, i - 1, relation_for_mechanism(self._mechanism_for_agent(i)))
                if i + 1 < n:
                    add(i + 1, i, relation_for_mechanism(self._mechanism_for_agent(i + 1)))
        elif self.topology == "dense":
            for i in range(1, n):
                add(i, 0, relation_for_mechanism(self._mechanism_for_agent(i)))
                for j in range(1, n):
                    if i != j and rng.random() < 0.75:
                        add(i, j, relation_for_mechanism(self._mechanism_for_agent(i)), strength=max(0.2, self.relation_strength * 0.75))

        # Template graph perturbation, retained for comparability with V1_FIXED3/MultiGrid.
        candidates = ["follow", "avoid", "mimic", "compete", "cooperate", "block", "adversarial"]
        for i in range(1, n):
            for j in range(n):
                if i == j or g[i, j] <= 0:
                    continue
                if rng.random() < self.edge_dropout:
                    g[i, j] = 0
                    w[i, j] = 0.0
                    continue
                if rng.random() < self.edge_type_noise:
                    g[i, j] = REL_TO_ID[str(rng.choice(candidates))]
                if self.edge_weight_noise > 0:
                    w[i, j] = float(np.clip(w[i, j] + rng.normal(0.0, self.edge_weight_noise), 0.05, 1.0))
            if np.sum(g[i] > 0) == 0:
                tgt = max(0, i - 1) if self.topology == "chain" else 0
                g[i, tgt] = REL_TO_ID[relation_for_mechanism(self._mechanism_for_agent(i))]
                w[i, tgt] = self.relation_strength
        return g, w

    def update_dynamic_relation_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.num_agents
        g = np.zeros((n, n), dtype=np.int64)
        w = np.zeros((n, n), dtype=np.float32)
        pos = self.all_agent_pos()
        vel = self.all_agent_vel()
        dmat = self.pairwise_agent_distances()
        nearest = self.nearest_landmark_ids()
        collision = self.collision_risk_matrix()
        blocking = self.blocking_matrix()
        for i in range(1, n):
            for j in range(n):
                if i == j:
                    continue
                rel = "none"
                weight = 0.0
                if collision[i, j] > 0:
                    rel = "collision_risk"
                    weight = 1.0
                elif blocking[i, j] > 0:
                    rel = "block"
                    weight = 1.0
                elif nearest[i] == nearest[j]:
                    rel = "same_landmark"
                    weight = 0.8
                else:
                    vi = vel[i]
                    vj = vel[j]
                    cos = float(np.dot(vi, vj) / ((np.linalg.norm(vi) * np.linalg.norm(vj)) + 1e-8))
                    if dmat[i, j] < self.dynamic_edge_radius:
                        rel = "near"
                        weight = float(1.0 / (1.0 + dmat[i, j]))
                    if cos > 0.75 and np.linalg.norm(vi) > 1e-4 and np.linalg.norm(vj) > 1e-4:
                        rel = "velocity_alignment"
                        weight = max(weight, 0.75)
                if rel != "none":
                    g[i, j] = REL_TO_ID[rel]
                    w[i, j] = float(np.clip(weight, 0.05, 1.0))
            if np.sum(g[i] > 0) == 0:
                # Always keep at least one physical context edge for non-ego agents.
                j = 0 if i != 0 else 1
                rel = "near" if dmat[i, j] < self.dynamic_edge_radius else "follow"
                g[i, j] = REL_TO_ID[rel]
                w[i, j] = float(np.clip(1.0 / (1.0 + dmat[i, j]), 0.05, 1.0))
        self.dynamic_relation_graph = g
        self.dynamic_relation_weight = w
        if self.topology_mode == "dynamic":
            self.relation_graph = g.copy()
            self.relation_weight = w.copy()
        elif self.topology_mode == "hybrid":
            # Union of template and dynamic relations: template provides episode-level role,
            # dynamic graph provides state-dependent physical contact.
            mask = g > 0
            self.relation_graph = self.relation_graph.copy()
            self.relation_weight = self.relation_weight.copy()
            self.relation_graph[mask] = g[mask]
            self.relation_weight[mask] = np.maximum(self.relation_weight[mask], w[mask])
        return g, w

    # ----------------------- force policies ----------------------------------
    def move_force_toward(self, i: int, target: np.ndarray) -> np.ndarray:
        return unit_vector(np.asarray(target, dtype=np.float32) - self.agent_pos(i))

    def move_force_away(self, i: int, target: np.ndarray) -> np.ndarray:
        return unit_vector(self.agent_pos(i) - np.asarray(target, dtype=np.float32))

    def random_force(self) -> Tuple[np.ndarray, int]:
        if self.action_mode == "discrete":
            a = int(self.rng.integers(0, 5))
            f = discrete_id_to_force(a)
            return f, force_to_direction_id(f)
        if self.action_mode == "force8":
            a = int(self.rng.integers(0, DIRECTION_LABEL_DIM))
            return direction_id_to_force(a), a
        # continuous
        raw = self.rng.normal(0.0, 1.0, size=2).astype(np.float32)
        f = unit_vector(raw) * float(self.rng.uniform(0.25, 1.0))
        return f.astype(np.float32), force_to_direction_id(f)

    def periodic_force(self, t: int, i: int) -> np.ndarray:
        return FORCE8[(t + i) % 8].copy()

    def ego_policy(self, t: int) -> Tuple[np.ndarray, int]:
        return self.random_force()

    def alt_ego_policy(self, t: int, ego_force: np.ndarray) -> Tuple[np.ndarray, int]:
        # Pick an opposite-ish alternative to make alt effects visible.
        if float(np.linalg.norm(ego_force)) < 1e-5:
            f = FORCE8[(t + 3) % 8].copy()
        else:
            f = -unit_vector(ego_force)
        return f.astype(np.float32), force_to_direction_id(f)

    def _relation_for_agent(self, i: int) -> Tuple[int, str]:
        rel_ids = self.relation_graph[i]
        targets = np.where(rel_ids > 0)[0]
        target_agent = int(targets[0]) if len(targets) else 0
        rel = ID_TO_REL[int(rel_ids[target_agent])] if len(targets) else relation_for_mechanism(self._mechanism_for_agent(i))
        if self.mechanism != "mixed":
            rel = relation_for_mechanism(self.mechanism)
        return target_agent, rel

    def other_policy_force(self, i: int, t: int, ego_force: np.ndarray, branch: str) -> Tuple[np.ndarray, int]:
        if branch == "null_direct":
            f = self.last_factual_forces[i].copy()
            return f, int(self.last_factual_action_ids[i])
        target_agent, rel = self._relation_for_agent(i)
        if rel == "static":
            f = np.zeros(2, dtype=np.float32)
        elif rel == "periodic":
            f = self.periodic_force(t, i)
        elif rel == "independent":
            f = self.move_force_toward(i, self.landmark_pos(i))
        elif rel == "mimic":
            f = ego_force.copy() if branch != "null_reactive" else self.previous_forces[0].copy()
        elif rel == "follow":
            f = self.move_force_toward(i, self.agent_pos(target_agent))
        elif rel in ["near", "velocity_alignment"]:
            f = 0.5 * self.move_force_toward(i, self.agent_pos(target_agent)) + 0.5 * self.previous_forces[target_agent]
            f = unit_vector(f)
        elif rel in ["avoid", "collision_risk"]:
            f = self.move_force_away(i, self.agent_pos(target_agent))
        elif rel == "same_landmark":
            f = self.move_force_toward(i, self.landmark_pos(int(self.nearest_landmark_ids()[target_agent])))
        elif rel == "cooperate":
            # Complementary landmark assignment: each non-ego moves toward a different landmark.
            f = self.move_force_toward(i, self.landmark_pos((i + 1) % self.num_agents))
        elif rel == "compete":
            f = self.move_force_toward(i, self.landmark_pos(int(self.nearest_landmark_ids()[0])))
        elif rel in ["block", "adversarial"]:
            ego = self.agent_pos(0)
            target = self.landmark_pos(int(self.nearest_landmark_ids()[0]))
            midpoint = 0.5 * (ego + target)
            f = self.move_force_toward(i, midpoint)
        else:
            f, _ = self.random_force()
        f = unit_vector(f) if np.linalg.norm(f) > 1.0 else f.astype(np.float32)
        return f.astype(np.float32), force_to_direction_id(f)

    def joint_forces(self, ego_force: np.ndarray, ego_id: int, t: int, branch: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.topology_mode in ["dynamic", "hybrid"]:
            self.update_dynamic_relation_graph()
        forces = np.zeros((self.num_agents, 2), dtype=np.float32)
        ids = np.zeros(self.num_agents, dtype=np.int64)
        forces[0] = np.asarray(ego_force, dtype=np.float32)
        ids[0] = int(ego_id)
        for i in range(1, self.num_agents):
            forces[i], ids[i] = self.other_policy_force(i, t=t, ego_force=forces[0], branch=branch)
        if branch == "factual":
            self.last_factual_forces = forces.copy()
            self.last_factual_action_ids = ids.copy()
        return forces, ids

    def _manual_world_step(self, forces: np.ndarray) -> None:
        world = self.get_world()
        agents = list(getattr(world, "agents", []))[: self.num_agents]
        dim_c = int(getattr(world, "dim_c", 0) or 0)
        for i, agent in enumerate(agents):
            if not hasattr(agent, "action"):
                continue
            accel = getattr(agent, "accel", None)
            sensitivity = 5.0 if accel is None else float(accel)
            f = np.asarray(forces[i], dtype=np.float64).ravel()[:2]
            agent.action.u = f * sensitivity * self.force_scale
            try:
                agent.action.c = np.zeros(dim_c, dtype=np.float64)
            except Exception:
                pass
        world.step()
        self._refresh_obs_from_world()

    def step_with_forces(self, forces: np.ndarray, ids: np.ndarray) -> None:
        self._manual_world_step(forces)
        self.previous_forces = np.asarray(forces, dtype=np.float32).copy()
        self.previous_action_ids = np.asarray(ids, dtype=np.int64).copy()

    def agent_effect_matrix_proxy(self) -> np.ndarray:
        n = self.num_agents
        mat = np.zeros((n, n), dtype=np.float32)
        graph = self.relation_graph if self.topology_mode != "dynamic" else self.dynamic_relation_graph
        weight = self.relation_weight if self.topology_mode != "dynamic" else self.dynamic_relation_weight
        d = self.pairwise_agent_distances()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if graph[i, j] > 0:
                    mat[i, j] = float(weight[i, j] * (1.0 / (1.0 + d[i, j])))
        return mat
