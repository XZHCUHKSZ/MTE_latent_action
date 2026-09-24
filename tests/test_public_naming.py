"""Public naming integration, CPU-only; no scientific training or evidence reads."""
import ast
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mte.method_names import (normalize_config, normalize_job, resolve_control,
    resolve_visual_arm, resolve_coupled_ground, resolve_method, display_name, report_text)


class PublicNamingIntegration(unittest.TestCase):
    def test_all_original_configs_are_identity_transforms(self):
        for path in (ROOT/'configs').glob('*.json'):
            original = json.loads(path.read_text(encoding='utf-8-sig'))
            for interface in ('method', 'mpe', 'control', 'visual'):
                with self.subTest(config=path.name, interface=interface):
                    self.assertEqual(normalize_config(original, interface), original)

    def test_qualified_controls_keep_all_axes(self):
        for public, old in [('PC-Field-edge','mif_edge'),('PC-Field-pair','mif_pair'),
                            ('PC-Set-matched','simple_matched'),('PC-Graph-raw','graph_raw'),
                            ('PC-Graph-complement','graph_complement'),('PC-Mean-h3-root','edge_h3_root')]:
            self.assertEqual(resolve_control(public), old)
        for family, old in [('Set','simple'),('Graph','graph'),('Sweep','tree'),('Field','mif')]:
            for composition in ('Solo','Aux'):
                for initialization in ('pretrained','random'):
                    for access in ('frozen','trainable','baseonly','auxonly'):
                        public = f'PC-{family}-{composition}-{initialization}-{access}'
                        native = f'{old}_{composition.lower()}_{initialization}_{access}'
                        self.assertEqual(resolve_visual_arm(public), native)
                        self.assertEqual(resolve_control(public), native)
                        self.assertEqual(display_name(native), public)
        self.assertEqual(resolve_method('PC-Graph-raw-Aux','mpe'), 'base+graph_raw')
        self.assertEqual(resolve_method('PC-Mean-h3-root-Aux','mpe'), 'base+edge_h3_root')
        for name in ('PC-Mean-h1-root', 'PC-Field-Aux-random-frozen', 'PC-Field-h2'):
            with self.assertRaises(ValueError): resolve_method(name,'mpe')

    def test_branch_and_scale_job_matrices(self):
        from experiments.visual_branch_attribution import jobs as branch
        from experiments.mpe_agent_scale_core import jobs as scale, location
        old = json.loads((ROOT/'configs/visual_branch_attribution.json').read_text())
        public = copy.deepcopy(old)
        public['families'] = [display_name(f) for f in old['families']]
        for phase in ('smoke','parity','formal'):
            for stage in ('ground','evaluate'):
                self.assertEqual(branch(public,phase,stage),branch(old,phase,stage))
        self.assertEqual(public['families'][0], 'PC-Set')
        old = json.loads((ROOT/'configs/mpe_agent_scale_core.json').read_text())
        public = copy.deepcopy(old)
        public['methods'] = [display_name(m,'mpe') for m in old['methods']]
        for phase in ('smoke','formal'):
            for stage in ('collect','pretrain','ground','evaluate'):
                self.assertEqual(scale(public,phase,stage),scale(old,phase,stage))
        job = ('formal',4,123,'ground','PC-Field-Aux','frozen')
        oldjob = (*job[:4],'base+mif',job[-1])
        self.assertEqual(location(Path('scratch'),job),location(Path('scratch'),oldjob))
        self.assertEqual(job[4],'PC-Field-Aux')

    def test_visual_budget_matrices_and_paths(self):
        from experiments import visual_low_label as low, visual_supervision_inventory as inv
        for module, file in [(low,'visual_low_label'), (inv,'visual_supervision_inventory')]:
            old = json.loads((ROOT/'configs'/f'{file}.json').read_text())
            public = copy.deepcopy(old)
            for field in ('arms','fair_arms','main_model','secondary_model'):
                if field not in public: continue
                value=public[field]
                public[field]=[display_name(v) for v in value] if isinstance(value,list) else display_name(value)
            self.assertEqual(list(module.phases(public)),list(module.phases(old)))
            job=('formal',908711,2,'PC-Field-Solo','ground')
            oldjob=(*job[:3],'anchor_solo_edge_cara_mif',job[-1])
            self.assertEqual(module.job_path(Path('scratch'),job), module.job_path(Path('scratch'),oldjob))

    def test_temporal_plans_preserve_protocols(self):
        from experiments.mpe_inventory_completion import p_for
        from experiments.mpe_evidence_completion import plan
        for fn,file,fields in [(p_for,'mpe_inventory_completion',('methods','latent_methods')),
                              (plan,'mpe_evidence_completion',('full_original_arms','scaling_original_arms','new_control_configs','scaling_control_configs','probe_methods','probe_auxiliaries'))]:
            old=json.loads((ROOT/'configs'/f'{file}.json').read_text())
            public=copy.deepcopy(old)
            for field in fields: public[field]=[display_name(v,'mpe') for v in old[field]]
            for n in [old['full_n'],32,175]: self.assertEqual(fn(public,n),fn(old,n))

    def test_actual_visual_and_coupled_api_bindings(self):
        from visual import train as visual
        from training import coupled_cheetah_frozen as coupled
        from training.visual_branch_attribution import base_arm
        with patch.object(visual,'RT',Path('scratch')):
            self.assertEqual(visual.source('PC-Field'), visual.source('edge_cara_mif'))
        for family in ('Set','Graph','Sweep','Field'):
            public=f'PC-{family}-Aux-pretrained-frozen'
            self.assertEqual(base_arm(public),base_arm(resolve_control(public)))
        aliases=coupled.public_arms()
        self.assertEqual(len(aliases),19)
        self.assertEqual([resolve_coupled_ground(x) for x in aliases],coupled.arms())
        self.assertEqual(resolve_coupled_ground('PC-Field-Solo'),'solo_edge_cara_mif__Frozen')
        with self.assertRaises(ValueError):resolve_coupled_ground('PC-Field-Solo__Adapt')
        # Reaches the actual pair/restore path before any model or file is loaded.
        class Stop(Exception):pass
        with patch.object(coupled,'restore',side_effect=Stop) as restore:
            for arm in ('PC-Field-Solo','solo_edge_cara_mif'):
                with self.assertRaises(Stop):coupled.pair(Path('scratch'),arm,'cpu')
                self.assertEqual(restore.call_args.args[0],Path('scratch/edge_cara_mif/policy/policy.pt'))

    def test_report_labels_do_not_touch_values_paths_or_baselines(self):
        lines=['# MIF / Simple/Graph/Tree',
               '|base+mif|anchor_solo_edge_cara_mif|mif_aux_pretrained_frozen|12.3456|0.1234|',
               'MTE denotes a matched target edge; outputs/mif/result.json and `edge_cara_mif` retain provenance.']
        text=report_text(lines,'control')
        self.assertIn('PC-Set/PC-Graph/PC-Sweep',text)
        self.assertIn('|PC-Field-Aux|PC-Field-Solo|PC-Field-Aux-pretrained-frozen|12.3456|0.1234|',text)
        self.assertIn('MTE denotes a matched target edge',text)
        self.assertIn('outputs/mif/result.json',text)
        self.assertIn('`edge_cara_mif`',text)
        self.assertEqual(report_text(text.splitlines(),'control'),text)
        for baseline in ['anchor_solo_lapo_state_adapter','anchor_plus_continuous_lam','bc','idm']:
            self.assertEqual(display_name(baseline),baseline)

    def test_every_markdown_report_writer_uses_formatter(self):
        count=0
        for folder in ('experiments','evaluation'):
            for path in (ROOT/folder).glob('*.py'):
                tree=ast.parse(path.read_text(encoding='utf-8-sig'))
                for node in ast.walk(tree):
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and 'RESULTS_CN.md' in ast.unparse(node.func.value):
                        self.assertIn('report_text(',ast.unparse(node.args[0]),str(path))
                        count+=1
        self.assertEqual(count,15)  # 14 writers; visibility copies the same rendered report.

if __name__=='__main__': unittest.main()
