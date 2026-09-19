"""Current verified evidence, with manuscript snapshots separate. Never trains or selects winners."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def current_report(name):
    if name == 'mpe':
        s = read('results/completed_runs/mpe_inventory_completion_2026_09_18/summary.json')
        assert s['completed_cells'] == s['expected_cells'] == 1305 and not s['missing']
        return dict(protocol='MPE t+2 inverse / direct h1,h2,h3; complete inventory', rows=s['means'])
    if name == 'mamujoco':
        return dict(protocol='Original constant-support-repaired frozen main controls',
                    rows=read('results/current/mamujoco_controls.json'))
    if name == 'visual':
        summary = read('results/completed_runs/visual_supervision_inventory_2026_09_19/summary.json')
        assert len(summary['records']) == 620 and summary['new_results'] == 360 and summary['reused_results'] == 260
        return dict(protocol='Complete visual B1/B2/B4/B8 inventory; 500 original-arm cells + 120 MIF supervision-factorial cells',
                    manuscript_status='The earlier 180-cell subset is in the manuscript; this full inventory is completed follow-up evidence',
                    primary_evaluation='27 fixed conditions: 908604-908630; not the historical all-30 mean',
                    evidence=summary)
    if name == 'visual-paper':
        return read('results/completed_runs/visual_low_label_2026_09_19_r1/summary.json')
    if name == 'adapt':
        return dict(scope='Separate fixed-budget supervised history-copy + decoder studies; never pooled with Frozen',
                    studies={n:read('results/completed_runs/'+n+'/summary.json') for n in
                        ('policy_finetune_pilot_2026_09_20_r1', 'mte_finetune_extension_2026_09_20')},
                    mif_budget_curve='results/completed_runs/visual_supervision_inventory_2026_09_19/summary.json')
    if name == 'branches':
        return read('results/completed_runs/visual_branch_attribution_2026_09_20/summary.json')
    if name == 'current':
        return read('provenance/current_scientific_state.json')
    if name == 'route':
        return read('results/completed_runs/mamujoco_route_completion_2026_09_19/summary.json')
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true')
    p.add_argument('--report', choices=['current', 'mpe', 'mamujoco', 'visual', 'visual-paper', 'route', 'adapt', 'branches'])
    p.add_argument('--check-results', action='store_true')
    a = p.parse_args()
    if a.list:
        for r in read('provenance/experiment_catalog.json')['experiments']:
            print(r['id'], r['entry'], r['manuscript_status'], sep=' | ')
    if a.report:
        print(json.dumps(current_report(a.report), ensure_ascii=False, indent=2))
    if a.check_results:
        from inspect_release import verify
        verify()
        for name in ('current', 'mpe', 'mamujoco', 'visual', 'visual-paper', 'route', 'adapt', 'branches'):
            current_report(name)
        old = current_report('visual-paper')['records']
        full = current_report('visual')['evidence']['records']
        index = {(r['seed'],r['budget'],r['arm']):r for r in full}
        assert len(index) == 620
        for r in old:
            q = index[(r['seed'],r['budget'],r['arm'])]
            assert r['source'] == q['source'] and abs(r['mean_return']-q['mean_return']) < 1e-10
        print('PASS: current reports; all 180 visual manuscript cells preserved inside the full 620-cell inventory; no winner selection')
    if not any((a.list, a.report, a.check_results)):
        p.print_help()


if __name__ == '__main__':
    main()
