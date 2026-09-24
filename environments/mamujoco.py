from __future__ import annotations
from gymnasium_robotics.envs.multiagent_mujoco.mamujoco_v1 import get_parts_and_edges
import json
from pathlib import Path
from typing import Sequence
import numpy as np
from gymnasium_robotics.envs.multiagent_mujoco.mamujoco_v1 import parallel_env
SUPPORTED_TASKS={"Ant":"4x2","CoupledHalfCheetah":"1p1"}

# Final implementation source: closed_loop_lam_v1/mamujoco_env.py:55

def make_env(seed: int, scenario: str = "Ant", agent_conf: str | None = None,
             agent_obsk: int = 1):
    """Create and reset an official Gymnasium-Robotics MaMuJoCo environment."""
    if scenario not in SUPPORTED_TASKS:
        raise ValueError(f"Supported qualification tasks are {tuple(SUPPORTED_TASKS)}")
    agent_conf = agent_conf or SUPPORTED_TASKS[scenario]
    env = parallel_env(
        scenario=scenario,
        agent_conf=agent_conf,
        agent_obsk=agent_obsk,
        render_mode=None,
    )
    # The upstream constructor keeps this graph in a local variable.  Retain a
    # read-only reference for our incidence prior without changing the env.
    _, official_edges, _ = get_parts_and_edges(scenario, agent_conf)
    env._adapter_mujoco_edges = tuple(official_edges)
    env.reset(seed=int(seed))
    return env

# Final implementation source: closed_loop_lam_v1/mamujoco_env.py:112

def state_vector(env) -> np.ndarray:
    """Official centralised MaMuJoCo state used by the latent-action model."""
    return np.asarray(env.state(), dtype=np.float32).copy()

# Final implementation source: closed_loop_lam_v1/mamujoco_env.py:151

def _action_dict(env, joint_action: np.ndarray) -> dict[str, np.ndarray]:
    joint_action = np.asarray(joint_action, np.float32)
    if joint_action.shape[0] != env.num_agents:
        raise ValueError(f"Expected {env.num_agents} agent actions, got {joint_action.shape}")
    result = {}
    for i, agent in enumerate(env.possible_agents):
        expected = env.action_space(agent).shape
        action = joint_action[i, :expected[0]]
        if action.shape != expected:
            raise ValueError(f"{agent} action has {action.shape}, expected {expected}")
        result[agent] = action.copy()
    return result

# Final implementation source: closed_loop_lam_v1/mamujoco_env.py:165

def _step(env, joint_action: np.ndarray):
    return env.step(_action_dict(env, joint_action))

# Final implementation source: closed_loop_lam_v1/mamujoco_env.py:205

def evaluate_policy(policy, seeds: Sequence[int], max_steps: int,
                    scenario: str = "Ant", agent_conf: str | None = None,
                    teacher_checkpoint: str | Path | None = None,
                    teacher_gate: str | Path | None = None,
                    teacher_ego: bool = False, random_ego: bool = False,
                    device: str = "cpu") -> dict:
    """Closed-loop ego evaluation with the gated central teacher as partners.

    The teacher proposes all four local actions from the current state; learned
    and random conditions replace only agent 0.  Agents 1..N therefore remain
    responsive closed-loop partners under every condition.
    """
    if teacher_ego and random_ego:
        raise ValueError("teacher_ego and random_ego are mutually exclusive")
    if teacher_checkpoint is None or teacher_gate is None:
        raise ValueError("MaMuJoCo evaluation requires a gated teacher checkpoint")
    agent_conf = agent_conf or SUPPORTED_TASKS[scenario]
    from environments.teacher import CentralSACTeacher, sha256_file
    checkpoint = Path(teacher_checkpoint)
    gate_path = Path(teacher_gate)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(checkpoint)
    if not gate.get("passed", False):
        raise ValueError(f"Teacher gate did not pass: {gate_path}")
    if gate.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Teacher checkpoint hash does not match the evaluation gate")
    if gate.get("scenario") != scenario or gate.get("agent_conf") != agent_conf:
        raise ValueError("Teacher gate task does not match the evaluation task")
    teacher = CentralSACTeacher(checkpoint, scenario, device=device)
    returns, lengths = [], []
    for seed in np.asarray(seeds, dtype=np.int64):
        env = make_env(int(seed), scenario, agent_conf)
        rng = np.random.default_rng(int(seed) + 1_700_003)
        hidden = None
        total = 0.0
        length = 0
        try:
            for _ in range(int(max_steps)):
                action = teacher.joint_action(env)
                if random_ego:
                    action[0] = rng.uniform(-1.0, 1.0, action.shape[-1])
                elif not teacher_ego:
                    if policy is None:
                        raise ValueError("A learned policy is required unless an anchor flag is set")
                    ego_action, hidden = policy.act_step(
                        state_vector(env), hidden, device=device)
                    action[0] = np.asarray(ego_action, np.float32)
                _, reward, terminated, truncated, _ = _step(env, action)
                total += float(np.mean(list(reward.values())))
                length += 1
                if any(terminated.values()) or any(truncated.values()):
                    break
        finally:
            env.close()
        returns.append(total); lengths.append(length)
    values = np.asarray(returns, np.float64)
    return {
        "mean_return": float(values.mean()),
        "std_return": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "se_return": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        "mean_episode_length": float(np.mean(lengths)),
        "episode_returns": values.tolist(),
        "episode_lengths": [int(x) for x in lengths],
        "teacher_checkpoint_sha256": checkpoint_hash,
        "partner_policy": "gated centralized SAC with ego action overridden",
    }
