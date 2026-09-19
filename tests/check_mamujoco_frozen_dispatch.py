"""Real two-update dispatch parity against an archived mixed dispatcher.

Requires the scientific CUDA environment, but no external data or simulator.
Pass --reference /path/to/original/experiments/limited_labels.py. The reference
can be obtained from Git commit 2a784fa without restoring retired files in-place.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from experiments import mamujoco_frozen_control as current

METHODS = ['entity_target', 'entity_joint', 'lapo_joint', 'laom_joint_k3',
           'continuous_lam', 'lapo_state_adapter', 'laom_state_adapter',
           'edge_cara_h1', 'edge_cara_h3', 'edge_cara_mobius_simple',
           'edge_cara_mobius_graph', 'edge_cara_mif', 'edge_cara_mobius_tree',
           'random_edge_encoder', 'matched_cf_deepsets']


def check(reference):
    spec = importlib.util.spec_from_file_location('archived_mamujoco_dispatch', reference)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(20720)
    x = rng.normal(size=(4, 6, 105)).astype(np.float32)
    mask = np.ones((4, 5), bool)
    a = rng.normal(size=(4, 5, 3, 8, 105)).astype(np.float32)
    b = rng.normal(size=a.shape).astype(np.float32)
    cm = np.zeros((8, 4), np.float32)
    for k in range(8):
        cm[k, 1:] = [(k >> j) & 1 for j in range(3)]
    checks = []
    with tempfile.TemporaryDirectory(prefix='mamu-dispatch-parity-') as temporary:
        root = Path(temporary).resolve()
        np.savez(root/'train.npz', observations=x[:2], valid_mask=mask[:2])
        np.savez(root/'dev.npz', observations=x[2:], valid_mask=mask[2:])
        np.savez(root/'endpoints.npz', actual=a, reference=b, masks=cm)
        suite = dict(env='mamujoco', seed=73, n=2, train=str(root/'train.npz'),
                     dev=str(root/'dev.npz'), endpoints=str(root/'endpoints.npz'),
                     endpoint_version='constant_support_repaired', codes=str(root/'codes.npz'))
        config = dict(backend_updates={'mamujoco': 2}, policy_updates=2)
        try:
            current.train_history(dict(env='mpe'), 'entity_target', root, print, config)
        except ValueError:
            pass
        else:
            raise AssertionError('Retired MPE route was not rejected before IO')
        for method in METHODS:
            shape = (4, 5, 4, 16) if method in ('entity_target', 'entity_joint') else (4, 5, 64)
            np.savez(root/'codes.npz', z=rng.normal(size=shape).astype(np.float32))
            paths = []
            for name, module in [('old', old), ('current', current)]:
                out = root/(method+'_'+name)
                out.mkdir()
                C.seed_all(73)
                result = module.train_history(suite, method, out, lambda *a, **k: None, config)
                assert result['native_action_labels_read'] == result['simulator_queries'] == 0
                paths.append(out)
            errors = []

            def compare(u, v):
                if torch.is_tensor(u):
                    assert torch.is_tensor(v) and u.shape == v.shape
                    error = float((u-v).abs().max()) if u.numel() else 0.
                    assert error <= 1e-6, error
                    errors.append(error)
                elif isinstance(u, dict):
                    assert u.keys() == v.keys()
                    for key in u:
                        compare(u[key], v[key])
                elif isinstance(u, np.ndarray):
                    assert np.array_equal(u, v)
                elif isinstance(u, (list, tuple)):
                    assert len(u) == len(v)
                    for uu, vv in zip(u, v):
                        compare(uu, vv)
                else:
                    assert u == v, (u, v)

            files = list(paths[0].rglob('*.pt'))
            assert files
            for f in files:
                compare(torch.load(f, map_location='cpu', weights_only=False),
                        torch.load(paths[1]/f.relative_to(paths[0]), map_location='cpu', weights_only=False))
            checks.append(dict(method=method, checkpoints=len(files), tensors=len(errors),
                               max_abs_error=max(errors), passed=True))
            print('PASS', method, max(errors), flush=True)
    return dict(passed=True, mpe_rejected_before_io=True, checks=checks,
                original_dispatch_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                new_dispatch_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
                scope='Synthetic observation-only two-update dispatcher parity; no simulator or action labels; not performance reproduction')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = check(args.reference.resolve())
    if args.report:
        with args.report.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    print(json.dumps(dict(passed=True, methods=len(report['checks']))))
