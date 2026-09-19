"""Temporal indexing and absence of action/future information at deployment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mte.temporal_mpe import indices
from closed_loop_lam_v1 import common as C
from training.route_representation import factory


def main():
    e, t = indices(3)
    assert len(t) == 66 and t.min() == 1 and t.max() == 22
    assert (t+2 <= 24).all() and (t+3 <= 25).all()
    assert np.array_equal(np.bincount(e), [22, 22, 22])
    # Timestamp-valued trajectories detect start/stop/horizon mistakes.
    x = np.broadcast_to(np.arange(26)[None, :, None], (3, 26, 16)).astype(np.float32)
    for horizon in (1, 2, 3):
        assert np.all(x[:, 1+horizon:23+horizon]-x[:, 1:23] == horizon)
    masks = np.zeros((8, 4), bool)
    for c in range(8): masks[c, 1:] = [(c >> j) & 1 for j in range(3)]
    for model in ('graph', 'mif'):
        adj = np.eye(8, dtype=np.float32)
        C.seed_all(42); a = factory(model, 'mobius', 8, masks, adj, 'mpe')
        C.seed_all(42); b = factory(model, 'raw_coordinates', 8, masks, adj, 'mpe')
        assert all(torch.equal(q, r) for q, r in zip(a.parameters(), b.parameters()))
        assert sum(q.numel() for q in a.parameters()) == sum(q.numel() for q in b.parameters())
        assert torch.equal(b.mobius, torch.eye(8, device='cuda'))
    print('PASS: time bounds, target horizons, raw/matched identical trainable initialization and parameter count')


if __name__ == '__main__': main()
