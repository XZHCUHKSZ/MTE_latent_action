"""Evaluation-only simulator access. Never imported by a training entry point.

MaMuJoCo restore follows the archived integration-state adapter; original model,
reward and agent partition are unchanged. This module defines no learned method.
"""
from copy import deepcopy
import numpy as np


class EffectEnvironment:
    def __init__(self, name, seed, teacher=None):
        self.name = name
        self.teacher = teacher
        self.rng = np.random.default_rng(seed + 901)
        if name == 'mpe':
            from environments.mpe import make_env
            self.env = make_env(seed, 4, 25)
        elif name == 'mamujoco':
            from environments.mamujoco import make_env
            self.env = make_env(seed, 'Ant', '4x2')
            if teacher is None:
                raise ValueError('A checked frozen teacher is required')
        else:
            raise ValueError(name)

    def observation(self):
        if self.name == 'mpe':
            return np.concatenate([self.env.all_agent_pos().ravel(),
                                   self.env.all_landmark_pos().ravel()]).astype(np.float32)
        return np.asarray(self.env.state(), np.float32).copy()

    def action(self):
        if self.name == 'mpe':
            from environments.mpe import teacher_forces
            return teacher_forces(self.env, 'high', self.rng, None)
        return self.teacher.joint_action(self.env)

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (4, 2) or not np.isfinite(action).all():
            raise ValueError('Expected four physical agents with two finite commands')
        if self.name == 'mpe':
            from environments.native_particle import force_to_direction_id
            self.env.step_with_forces(action, np.array([force_to_direction_id(a) for a in action]))
            return False
        from environments.mamujoco import _step
        _, _, terminated, truncated, _ = _step(self.env, action)
        return any(terminated.values()) or any(truncated.values())

    def _chain(self):
        obj = self.env.single_agent_env
        while True:
            yield obj
            if not hasattr(obj, 'env'):
                break
            obj = obj.env

    def snapshot(self):
        if self.name == 'mpe':
            return self.env.snapshot()
        import mujoco
        base = self.env.single_agent_env.unwrapped
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(base.model, spec), np.float64)
        mujoco.mj_getState(base.model, base.data, state, spec)
        return dict(physics=state, agents=list(self.env.agents),
                    elapsed=[getattr(e, '_elapsed_steps', None) for e in self._chain()],
                    rng=deepcopy(base.np_random.bit_generator.state))

    def restore(self, snap):
        if self.name == 'mpe':
            self.env.restore(snap)
            return
        import mujoco
        base = self.env.single_agent_env.unwrapped
        mujoco.mj_setState(base.model, base.data, snap['physics'], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(base.model, base.data)
        for wrapper, elapsed in zip(self._chain(), snap['elapsed']):
            if elapsed is not None:
                wrapper._elapsed_steps = elapsed
        base.np_random.bit_generator.state = deepcopy(snap['rng'])
        self.env.agents = list(snap['agents'])

    def roll(self, snap, actions):
        self.restore(snap)
        obs, done, live = [], False, []
        for action in actions:
            live.append(not done)
            if not done:
                done = self.step(action)
            obs.append(self.observation())
        return np.stack(obs), np.array(live, bool)

    def close(self):
        (self.env.env if self.name == 'mpe' else self.env).close()
