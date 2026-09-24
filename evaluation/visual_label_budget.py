"""Path-only binding of the accepted RGB rollout to new grounded decoders.

The original rollout, renderer, recurrent controller and physical environment
are called unchanged. Development actions are evaluation-only diagnostics.
"""
from mte.method_names import resolve_visual_arm
from pathlib import Path
import atexit
from training import visual_label_budget as B
from visual import train as V
from utils.atomic import atomic_json
from utils.io import read


def evaluate(out, arm, seed, budget_root, assets, evaluation_seeds):
    arm = resolve_visual_arm(arm)
    from visual import rollout as R, render as render_module, dcs_source
    from utils.access import guard
    source = B.ARCHIVE / str(seed)
    B.configure(seed, budget_root, 8)
    # Binding paths does not alter model bodies or evaluation mathematics.
    render_module.ROOT = B.WORKSPACE
    dcs_source.IMAGES = B.WORKSPACE / 'visual_multiagent_2026_09_10/laom_complexity_v1/assets/DAVIS/JPEGImages/480p'
    from visual import scene
    scene.IMAGES = dcs_source.IMAGES
    V.OLD = Path(assets)
    V.RT = source
    R.RT = Path(budget_root)
    R.ARMS = [arm]
    R.SEEDS = list(evaluation_seeds)
    original_controller = V.controller

    def controller(name):
        V.RT = Path(budget_root)
        try:
            return original_controller(name)
        finally:
            V.RT = source

    R.controller = controller
    R.progress = lambda phase, **kw: atomic_json(out/'progress.json', dict(phase=phase, **kw))
    config = read(V.OLD/'visual_config.json')
    gate = read(B.WORKSPACE/config['teacher_gate'])
    cp = Path(gate['checkpoint'])
    if not cp.is_absolute():
        cp = B.WORKSPACE/cp
    allowed = V.policy_paths() + [source/'frontend/model.pt', source/'features/features.npz',
        source/'features/projection.pt', Path(budget_root)/('ground_'+arm)/'decoder.pt', cp]
    allowed += [V.OLD/f'labels/dev/{i:04d}.npy' for i in range(8)]
    audit = guard(out, allowed, pretraining=False)
    audit['phase'] = 'frozen_budgeted_control_evaluation'
    atexit.register(lambda: atomic_json(out/'access_audit.json', audit))
    result = R.evaluate(out, arm)
    assert not audit['violations']
    atomic_json(out/'access_audit.json', audit)
    return result
