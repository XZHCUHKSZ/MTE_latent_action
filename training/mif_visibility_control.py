"""Explicit coordinate/visibility controls around the accepted MIF trainer.

No model class or scientific loss is redefined. The native arm calls the existing
masked trainer unchanged. A scoped construction adapter changes only fixed basis
buffers and/or encoder visibility. Loss masks remain unchanged in every arm.
"""
import hashlib
import numpy as np
import torch
from mte import structured_controls as original
from training.representation_mamujoco import train_history_policy
from utils.artifacts import rich

ARMS = ('masked_mobius', 'masked_raw', 'visible_mobius', 'visible_raw')


def train(s, arm, config, out, progress):
    assert arm in ARMS
    x, valid, n, a, b, masks, adjacency, support = rich(s)
    z, stats, checkpoint = representation(a, b, support, n, masks, adjacency,
        s['seed'], config['representation_updates'], arm, progress)
    torch.save(checkpoint, out / 'representation.pt')
    history = train_history_policy(x, z, valid, n, s['seed'], config['policy_updates'],
        out / 'policy', progress, method='edge_cara_mif')
    return dict(representation=stats, history=history, native_action_labels_read=0,
                simulator_queries=0, configuration=arm)


def representation(a, b, valid, n, masks, adjacency, seed, updates, arm, progress):
    assert arm in ARMS
    original_factory = original.make_model
    recorded = {}
    mask_hash = hashlib.sha256()
    counters = {'training_calls': 0, 'hidden_cells': 0, 'total_cells': 0}

    def build(dim, cm, adj, hidden, native_arm, device='cuda'):
        net = original_factory(dim, cm, adj, hidden, native_arm, device)
        initial = hashlib.sha256()
        for name, parameter in net.named_parameters():
            initial.update(name.encode())
            initial.update(parameter.detach().cpu().numpy().tobytes())
        recorded['initial_parameters_sha256'] = initial.hexdigest()
        if arm.endswith('_raw'):
            with torch.no_grad():
                for name in ('mobius', 'zeta'):
                    getattr(net, name).copy_(torch.eye(len(cm), device=device))

        def visibility_hook(module, args):
            values, loss_visibility = args
            if module.training:
                bits = loss_visibility.detach().cpu().numpy()
                mask_hash.update(bits.tobytes())
                counters['training_calls'] += 1
                counters['hidden_cells'] += int((~bits).sum())
                counters['total_cells'] += bits.size
                # Return a NEW tensor. Never mutate the caller's loss mask.
                if arm.startswith('visible_'):
                    return values, torch.ones_like(loss_visibility)
            return None

        net.register_forward_pre_hook(visibility_hook)
        return net

    # Process-local, explicit ablation adapter; original factory restored even
    # after errors. No source file, model forward, optimizer, or loss is edited.
    original.make_model = build
    try:
        z, stats, checkpoint = original.train(a, b, valid, n, masks, adjacency,
            'mamujoco', seed, updates, 'mif_edge', progress)
    finally:
        original.make_model = original_factory
    assert counters['training_calls'] == updates
    normalization = hashlib.sha256()
    for name in ('canonical_mean', 'canonical_std', 'input_mean', 'input_std'):
        normalization.update(checkpoint[name].tobytes())
    stats.update(recorded, **counters, configuration=arm,
        loss_masks_sha256=mask_hash.hexdigest(), normalization_sha256=normalization.hexdigest(),
        original_trainer='mte.structured_controls.train',
        encoder_visibility='all' if arm.startswith('visible_') else 'native coefficient-node mask',
        loss='unchanged native hidden MSE + .25 visible MSE + variance floor + .02 covariance',
        basis='identity' if arm.endswith('_raw') else 'Mobius/zeta',
        masking_semantics='Same node-index loss mask, different coordinate bases; not identical hidden raw-table information.')
    checkpoint['ablation_configuration'] = arm
    return z, stats, checkpoint
