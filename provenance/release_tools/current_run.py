"""Current paper evidence only. Never starts training or selects winning routes."""
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
        return read('results/completed_runs/visual_low_label_2026_09_19_r1/summary.json')
    if name == 'route':
        return read('results/completed_runs/mamujoco_route_completion_2026_09_19/summary.json')
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true')
    p.add_argument('--report', choices=['mpe', 'mamujoco', 'visual', 'route'])
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
        for name in ('mpe', 'mamujoco', 'visual', 'route'):
            current_report(name)
        print('PASS: current report sources; historical MPE is excluded from default reports')
    if not any((a.list, a.report, a.check_results)):
        p.print_help()


if __name__ == '__main__':
    main()
