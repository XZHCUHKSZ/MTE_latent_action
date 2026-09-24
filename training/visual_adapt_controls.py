"""Method/budget binding for the accepted supervised history-copy adapter."""
from training import visual_label_budget as B
from training import visual_supervision_control as S
from utils.atomic import atomic_json

ARMS = {
    'lapo_solo': 'anchor_solo_lapo_state_adapter',
    'lapo_aux': 'anchor_plus_lapo_state_adapter',
    'laom_solo': 'anchor_solo_laom_state_adapter',
    'laom_aux': 'anchor_plus_laom_state_adapter',
    'base_solo': 'anchor_only',
    'duplicate_aux': 'anchor_duplicate',
    'random_aux': 'anchor_plus_random_edge_encoder',
}


def base_arm(arm):
    family, composition, initialization, access = arm.split('_')
    assert initialization == 'pretrained' and access in ('frozen', 'trainable')
    return ARMS[family + '_' + composition]


def ground(dest, root, seed, budget, arm, updates):
    B.configure(seed, root, budget, updates=updates)
    S.base_arm = base_arm
    B.V.progress = lambda phase, **kw: atomic_json(dest/'progress.json', dict(phase=phase, **kw))
    # S uses V.pair: duplicate=True evaluates ONE history then concatenates twice.
    return S.ground(dest, arm)


controller = S.controller
