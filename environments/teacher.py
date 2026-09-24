from __future__ import annotations
from .mamujoco import state_vector
import hashlib
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC

# Final implementation source: closed_loop_lam_v1/mamujoco_teacher.py:37

def global_to_local_action(env, global_action: np.ndarray) -> np.ndarray:
    """Invert the upstream actuator partition without assuming joint order."""
    global_action = np.asarray(global_action, np.float32).reshape(-1)
    local_dims = [env.action_space(agent).shape[0] for agent in env.possible_agents]
    if len(set(local_dims)) != 1:
        raise ValueError("The current rectangular schema requires equal local action dimensions")
    local = np.zeros((env.num_agents, local_dims[0]), np.float32)
    used = []
    for i, partition in enumerate(env.agent_action_partitions):
        for j, node in enumerate(partition):
            action_id = int(node.act_ids)
            if not 0 <= action_id < len(global_action):
                raise ValueError(f"Invalid official actuator id {action_id}")
            local[i, j] = global_action[action_id]
            used.append(action_id)
    if sorted(used) != list(range(len(global_action))):
        raise ValueError(f"Official partition is not a bijection over actions: {used}")
    return local

# Final implementation source: closed_loop_lam_v1/mamujoco_teacher.py:66

class CentralSACTeacher:
    def __init__(self, checkpoint: str | Path, scenario: str, device: str = "cpu"):
        self.checkpoint = Path(checkpoint)
        self.scenario = scenario
        self.model = SAC.load(str(self.checkpoint), device=device)

    def global_action(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(
            np.asarray(observation, np.float32), deterministic=True)
        return np.asarray(action, np.float32).reshape(-1)

    def joint_action(self, env) -> np.ndarray:
        return global_to_local_action(env, self.global_action(state_vector(env)))

# Final implementation source: closed_loop_lam_v1/mamujoco_teacher.py:81

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
