from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from .native_particle import WhoseMoveMPENativeEnv,force_to_direction_id

# Final implementation source: closed_loop_lam_v1/mpe_env.py:21

def make_env(seed: int, num_agents: int = 3, max_cycles: int = 25):
    return WhoseMoveMPENativeEnv(
        num_agents=num_agents, mechanism="independent", topology="star",
        topology_mode="template", action_mode="continuous", seed=seed,
        max_cycles=max_cycles, render_mode=None,
    )

# Final implementation source: closed_loop_lam_v1/mpe_env.py:29

def teacher_assignment(env: WhoseMoveMPENativeEnv) -> np.ndarray:
    """Return the globally minimum-distance agent-to-landmark assignment."""
    pos = env.all_agent_pos(); landmarks = env.all_landmark_pos()
    cost = np.linalg.norm(pos[:, None] - landmarks[None], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    assignment = np.zeros(env.num_agents, np.int64); assignment[rows] = cols
    return assignment

# Final implementation source: closed_loop_lam_v1/mpe_env.py:38

def teacher_forces(env: WhoseMoveMPENativeEnv, quality: str,
                   rng: np.random.Generator,
                   assignment: np.ndarray | None = None) -> np.ndarray:
    pos = env.all_agent_pos(); landmarks = env.all_landmark_pos()
    # Legacy datasets recomputed Hungarian assignment at every step.  New
    # datasets pass an episode-fixed initial assignment so the demonstrated
    # controller is history-identifiable instead of discontinuously relabeling
    # targets after small learner-induced state shifts.
    if assignment is None:
        assignment = teacher_assignment(env)
    assignment = np.asarray(assignment, np.int64)
    if assignment.shape != (env.num_agents,):
        raise ValueError(f"teacher assignment must have shape ({env.num_agents},)")
    forces = np.zeros((env.num_agents, 2), np.float32)
    noise_prob = {"high": .05, "mid": .20, "low": .45}[quality]
    for i in range(env.num_agents):
        delta = landmarks[assignment[i]] - pos[i]
        norm = float(np.linalg.norm(delta))
        f = delta / max(norm, 1e-6)
        # Mild collision avoidance, shared across data collection and evaluation.
        for j in range(env.num_agents):
            if i == j:
                continue
            sep = pos[i] - pos[j]; dist = float(np.linalg.norm(sep))
            if dist < .25:
                f += .35 * sep / max(dist, 1e-6)
        f = f / max(1., float(np.linalg.norm(f)))
        if rng.random() < noise_prob:
            f = rng.normal(size=2); f = f / max(1e-6, float(np.linalg.norm(f)))
        forces[i] = f.astype(np.float32)
    return forces

# Final implementation source: closed_loop_lam_v1/mpe_env.py:71

def team_reward(env: WhoseMoveMPENativeEnv) -> float:
    raw = env.env.unwrapped
    scenario, world = raw.scenario, env.get_world()
    global_reward = float(scenario.global_reward(world))
    local_ratio = float(getattr(raw, "local_ratio", .5))
    per_agent = [
        local_ratio * float(scenario.reward(agent, world)) + (1. - local_ratio) * global_reward
        for agent in world.agents[:env.num_agents]
    ]
    return float(np.mean(per_agent))

# Final implementation source: closed_loop_lam_v1/mpe_env.py:83

def collision_count(env: WhoseMoveMPENativeEnv) -> float:
    d = env.pairwise_agent_distances()
    return float(np.sum(np.triu((d < .12).astype(np.float32), 1)))

# Final implementation source: closed_loop_lam_v1/mpe_env.py:88

def coverage_distance(env: WhoseMoveMPENativeEnv) -> float:
    pos, lm = env.all_agent_pos(), env.all_landmark_pos()
    return float(np.mean(np.min(np.linalg.norm(pos[:, None] - lm[None], axis=-1), axis=0)))
