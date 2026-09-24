import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OBS_DIM=105
from training.grounding_mamujoco import GroundedPolicy

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:218

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:248

def evaluate(policy, seeds, max_steps=200, teacher_checkpoint=None, teacher_gate=None,
             anchor=None):
    """Reuse the original official evaluator and frozen partner policy unchanged.

    Call only after training has finished. No data collection, simulator restore,
    branch construction, or online update is performed. Original teacher and
    random anchors use the same full 200-step protocol, with no warmup override.
    """
    if anchor not in (None, 'teacher', 'random'):
        raise ValueError('Unknown anchor')
    if anchor is None and not isinstance(policy, GroundedPolicy):
        raise TypeError('Use GroundedPolicy for learned evaluation')
    from environments.mamujoco import evaluate_policy
    if teacher_gate is None:
        raise ValueError('Pass the archived teacher gate path explicitly')
    gate_path = Path(teacher_gate)
    if not gate_path.is_absolute():
        gate_path = ROOT / gate_path
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    checkpoint = Path(teacher_checkpoint or gate['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    seeds = [int(x) for x in seeds]
    if not seeds or len(seeds) != len(set(seeds)) or max_steps < 2:
        raise ValueError('Evaluation requires unique seeds and at least two steps')
    result = evaluate_policy(policy, seeds, int(max_steps), scenario='Ant', agent_conf='4x2',
        teacher_checkpoint=checkpoint, teacher_gate=gate_path, device='cpu',
        teacher_ego=anchor == 'teacher', random_ego=anchor == 'random')
    result.update(episode_seeds=seeds, evaluation_steps=int(max_steps),
        common_warmup_action=None, observation_dim=OBS_DIM, anchor=anchor,
        evaluator_source='closed_loop_lam_v1.mamujoco_env.evaluate_policy',
        evidence='post-freeze simulator development control; no pretraining simulator access',
        warmup_note='none; preserve original actions and rewards from t0 through final valid step')
    return result
