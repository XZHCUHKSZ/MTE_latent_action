"""Naming compatibility contracts: no training or experimental data."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mte.method_names import FAMILY, LABELS, resolve_method, paper_name, resolve_mpe_config

class NamingContracts(unittest.TestCase):
    def test_all_variants_and_compositions(self):
        self.assertEqual(len(set(FAMILY.values())), 5)
        for key, native in FAMILY.items():
            public = LABELS[key]
            self.assertEqual(resolve_method(public), native)
            self.assertEqual(resolve_method(key), native)
            self.assertEqual(resolve_method(native), native)
            self.assertEqual(paper_name(native), public)
            for interface, prefixes in [('visual', ('anchor_solo_', 'anchor_plus_')), ('coupled', ('solo_', 'aux_'))]:
                for route, prefix in zip(['Solo', 'Aux'], prefixes):
                    old = prefix + native
                    self.assertEqual(resolve_method(public + '-' + route, interface), old)
                    self.assertEqual(resolve_method(old, interface), old)
                    self.assertEqual(paper_name(old), public + '-' + route)
    def test_mpe_protocol_and_config_preservation(self):
        pairs = {'PC-Mean-h2': 'edge_h2', 'PC-Mean-h3': 'edge_h3',
                 'PC-Set': 'simple', 'PC-Graph': 'graph', 'PC-Sweep': 'tree', 'PC-Field': 'mif'}
        for public, old in pairs.items():
            for route, prefix in [('Solo', ''), ('Aux', 'base+')]:
                name = public + '-' + route
                self.assertEqual(resolve_method(name, 'mpe'), prefix + old)
                self.assertEqual(paper_name(prefix + old, 'mpe'), name)
        for name in ['PC-Mean', 'PC-Mean-h1', 'PC-Field-h2']:
            with self.assertRaises(ValueError): resolve_method(name, 'mpe')
        old = {'methods': ['base', 'base+mif'], 'latent_methods': ['mif'],
               'budgets': [8,16], 'overrides': {'seed': 5}}
        public = dict(old, methods=['base','PC-Field-Aux'], latent_methods=['PC-Field'])
        resolved = resolve_mpe_config(public)
        self.assertEqual(resolved, old)
        resolved['overrides']['seed'] = 99
        self.assertEqual(public['overrides']['seed'], 5)
        self.assertEqual(resolve_mpe_config(old), old)

    def test_horizon_is_not_discarded(self):
        for h in [1, 2, 3]:
            public = 'PC-Mean-h' + str(h)
            old = 'edge_cara_h' + str(h)
            self.assertEqual(resolve_method(public), old)
            self.assertEqual(paper_name(old), public)
    def test_invalid_names_fail_before_dispatch(self):
        for public in ['PC-Field-h2', 'PC-Sweep-h1', 'PC-New', 'PC-Mean-h4', 'PC-Field-Solo']:
            with self.assertRaises(ValueError): resolve_method(public)
        with self.assertRaises(ValueError): resolve_method('PC-Mean-h2', 'visual')
        with self.assertRaises(ValueError): resolve_method('PC-Field', 'unknown')
    def test_baselines_and_legacy_ids_remain_unchanged(self):
        for name in ['base', 'anchor_only', 'anchor_duplicate', 'continuous_lam', 'lapo_state_adapter', 'laom_state_adapter', 'matched_cf_deepsets', 'random_edge_encoder', 'bc', 'idm']:
            for interface in ['method', 'visual', 'coupled']:
                self.assertEqual(resolve_method(name, interface), name)

if __name__ == '__main__': unittest.main()
