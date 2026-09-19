"""Stage the exact accepted composite script in a NEW directory and check its data.

Requires the separately supplied recorded RGB asset, numpy, scipy and matplotlib.
Does not edit the manuscript, train models, or claim byte-identical PDF metadata.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def run(rgb, out):
    import numpy as np
    lock = read(ROOT / 'provenance/manuscript_numeric_verification_2026_09_20.json')
    assert hashlib.sha256(rgb.read_bytes()).hexdigest() == lock['recorded_rgb']['sha256'], 'Wrong recorded RGB asset'
    script = ROOT / 'provenance/manuscript_sources/draw_reference_composites.py'
    assert hashlib.sha256(script.read_bytes()).hexdigest() == lock['figure_script']['sha256']
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('scripts', 'evidence', 'pictures', 'qa'):
        (out / folder).mkdir()
    shutil.copyfile(script, out / 'scripts' / script.name)
    evidence = ROOT / 'results/manuscript_snapshot'
    for name in ('figure_numeric_current.json', 'figure_scaling_current.json',
                 'control_ablation_current.json', 'visual_low_label_summary.json'):
        shutil.copyfile(evidence / name, out / 'evidence' / name)
    shutil.copyfile(rgb, out / 'evidence/illustration_rgb.npy')
    env = os.environ.copy()
    env.pop('MTE_FIG2_ONLY', None)
    subprocess.run([sys.executable, str(out / 'scripts' / script.name)], env=env, check=True)
    actual = read(out / 'evidence/reference_composite_data.json')
    expected = read(evidence / 'figure2_ablation_update_audit.json')
    expected += [r for r in read(evidence / 'reference_composite_data.json') if r['panel'].startswith('Fig3 ')]
    assert len(actual) == len(expected)
    for observed, original in zip(actual, expected):
        assert observed.keys() == original.keys()
        for key in original:
            if key == 'seed_values':
                np.testing.assert_allclose(observed[key], original[key], rtol=0, atol=1e-12)
            else:
                assert observed[key] == original[key], (key, observed[key], original[key])
    report = dict(status='pass', audited_series=len(actual),
                  script_sha256=lock['figure_script']['sha256'],
                  recorded_rgb_sha256=lock['recorded_rgb']['sha256'],
                  scope='Exact accepted script executed; Fig2 current audit and Fig3 accepted audit matched. Figure 1 is a separate accepted design asset. No pixel-level or PDF-byte identity claim.')
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.rgb.resolve(), args.out.resolve())
