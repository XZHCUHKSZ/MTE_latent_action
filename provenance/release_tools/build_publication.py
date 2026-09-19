"""Build a reviewable publication checkout; never commits, pushes or deletes files.

Scientific Python and configuration files are copied byte-for-byte. Only textual
evidence paths are normalized. Run from the archival workspace with explicit paths.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

FOLDERS = ('closed_loop_lam_v1', 'configs', 'docs', 'environments', 'evaluation',
           'experiments', 'mte', 'provenance', 'results', 'tests', 'third_party',
           'training', 'utils', 'visual')
EXCLUDE = {'experiments/visual_supervision_inventory_resume.py',
           'tests/check_visual_supervision_resume.py'}
EXCLUDE.update({'docs/review_response_experiment_plan_2026_09_17.md', 'docs/evidence_synthesis_and_remaining_2026_09_19.md', 'docs/review_gap_check_2026_09_19.md', 'docs/review_wording_response_2026_09_19.md', 'docs/visual_supervision_resume_2026_09_19.md', 'docs/review_remaining_and_next_2026_09_18.md', 'docs/weak_controls_audit_2026_09_19.md', 'docs/mpe_post_repair_evidence_audit_2026_09_18.md', 'docs/review_actions_2026_09_19_evening.md'})
RETIREMENTS = json.loads((Path(__file__).with_name('retirements.json')).read_text(encoding='utf-8'))['files']
EXCLUDE.update(r['path'] for r in RETIREMENTS)
# Local audit development records remain in the archive, not the curated delivery.
EXCLUDE.update({'provenance/release_tools/current_state_audit_2026_09_20.json', 'provenance/release_tools/current_state_audit_2026_09_20_r1.json'})
# Integration refers to the manuscript snapshot shipped with this release index.
RUNS = [
 ('mpe_temporal_full_scale_5seeds', 'mpe_temporal_scale', 'paper', 'MPE temporal structural controls'),
 ('mpe_evidence_completion_2026_09_18', 'mpe_evidence_completion', 'paper', 'Additional structural and donor controls'),
 ('mpe_inventory_completion_2026_09_18', 'mpe_inventory_completion', 'paper', 'Complete MPE configuration, label and data inventory'),
 ('mpe_partner_shift_2026_09_18_r1', 'mpe_partner_shift', 'followup', 'Fixed-policy partner perturbation diagnostic'),
 ('mpe_physical_interaction_onset_2026_09_18', 'mpe_physical_interaction_onset', 'followup', 'Simulator second-order physical-effect onset'),
 ('mpe_donor_transport_audit_2026_09_18', 'mpe_donor_transport_audit', 'followup', 'Read-only donor transport diagnostic'),
 ('mpe_donor_zero_audit_2026_09_19', 'mpe_donor_zero_audit', 'paper', 'Zero reference and query-state oracle diagnostic'),
 ('mpe_matching_input_control_2026_09_19', 'mpe_matching_input_control', 'paper', 'Same-resource Graph16 input control'),
 ('mamujoco_route_completion_2026_09_19', 'mamujoco_route_completion', 'paper', 'Seven-route matched/raw and Global16 comparison'),
 ('mamujoco_mif_visibility_2026_09_19', 'mamujoco_mif_visibility', 'paper', 'Native masking by coordinate-system factorial'),
 ('mamujoco_frozen_matching_audit_2026_09_19', 'mamujoco_frozen_matching_audit', 'followup', 'Frozen matching implementation audit'),
 ('mte_agent_resource_audit_2026_09_19', 'mte_agent_resource_audit', 'followup', 'Agent interface and computational resource audit'),
 ('visual_low_label_2026_09_19_r1', 'visual_low_label', 'paper', 'Visual low-label curves for nine configurations'),
 ('visual_supervision_inventory_2026_09_19', 'visual_supervision_inventory', 'followup', 'Complete visual label inventory and supervision factorial'),
 ('policy_finetune_pilot_2026_09_20_r1', 'policy_finetune_pilot', 'followup', 'Cross-method visual and MPE history adaptation'),
 ('mte_finetune_extension_2026_09_20', 'mte_finetune_extension', 'followup', 'MTE-family visual and MaMuJoCo adaptation'),
 ('visual_branch_attribution_2026_09_20', 'visual_branch_attribution', 'followup', 'Selective branch adaptation and capacity-matched LAOM control'),
]

README = '''# MTE: paper code and auditable experimental evidence

This repository organizes observation-only MTE pretraining, action grounding,
closed-loop evaluation, and structural controls across MPE, MaMuJoCo and visual
control. The 2026-09-20 release adds the completed experiments and audits from
September 18–20 to the original September 17 code release.

**Start with [the paper-to-code map](docs/PAPER_CODE_MAP.md),
[reproduction instructions](docs/REPRODUCIBILITY.md), and
[the experiment catalogue](provenance/experiment_catalog.json).**

## Two evaluation routes

| Route | Observation-only stage | Action-labelled stage | Evidence |
|---|---|---|---|
| Frozen | Learn representation and history; freeze them | Train the action decoder at the declared label budget | Current manuscript main experiments and structural controls |
| Adapt | Reuse the same observation-pretrained artifacts | Update a history copy and decoder using the same declared labels | Completed follow-up studies, separately indexed below |

Both routes retain the original observation-only pretraining boundary. Adapt uses
supervised downstream training. Its results retain their own labels, protocols
and comparisons; the repository never selects the better route per evaluation
episode or overwrites the Frozen scores.

The current manuscript snapshot is `iclr2027_mpe_completed_revision_2026_09_18`
with the September 19 visual low-label integration. The newer supervision,
fine-tuning and branch-attribution studies are complete **follow-up evidence**;
they are not yet incorporated into that manuscript snapshot. The catalogue
records this distinction for each experiment.

## Read results without training

Python 3.12 and its standard library are sufficient for these commands:

```bash
python inspect_release.py --list
python inspect_release.py --verify
python inspect_release.py --report visual_branch_attribution_2026_09_20
python run.py --check-results
```

`run.py` now selects current MPE, MaMuJoCo, visual and route evidence explicitly.
For example, `python run.py --report mpe` reads the corrected 1,305-cell MPE
inventory. `python run.py --report mamujoco` reads only MaMuJoCo main controls.
`python run.py --report visual` reads the full 620-cell visual inventory;
`--report visual-paper` retains the 180-cell manuscript subset. `--report adapt`
and `--report branches` expose the separate completed adaptation studies.
Use `--report current` and [the current-state audit](docs/CURRENT_SCIENTIFIC_STATE_CN.md)
for selection decisions, verified overlaps and remaining coverage.
See [the implementation lineage audit](docs/METHOD_LINEAGE_AUDIT_CN.md) for
superseded entry points and the exact meanings of Base, MTE-Aux and Global16.

## Organization

| Directory | Responsibility |
|---|---|
| `experiments/` | Question-specific entry points and formal experiment orchestration |
| `configs/`, `docs/` | Scientific protocols, access rules and experiment explanations |
| `mte/`, `training/` | Matched observation construction, representation/history learning and grounding |
| `environments/`, `visual/`, `evaluation/` | Environment adapters, RGB processing and evaluation |
| `closed_loop_lam_v1/` | Preserved shared method implementations |
| `results/` | Numeric evidence, all-direction comparisons and Chinese reports |
| `provenance/` | Source lineage, historical verification records and release hashes |
| `tests/` | Interface checks and asset-dependent scientific preflights |
| `third_party/` | Required upstream sources and attribution |

## Reproduction and interpretation

The public bundle contains source, protocols and compact evidence, not the large
frozen datasets, checkpoints, teacher weights or licensed visual assets. Full
training/replay requires the original asset tree described in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). No experiments run automatically.

Comparisons average evaluation episodes within each upstream seed. Each study's
declared budget aggregation and Holm family remain separate. Five-seed exact
two-sided sign permutation tests have minimum p = 0.0625. Reports retain positive
and negative differences, intervals and exact p values; positive means describe
observed gains. LAOM/LAPO state/feature adapters keep their declared scope.

The manuscript's main evidence, additional controls, and exploratory adaptation
answer different questions. Same-task adaptation does not add new environments,
larger agent populations or entity-aligned visual slots. The branch study tests
which history updates improve control while preserving both information inputs.

See [release changes](docs/RELEASE_2026_09_20.md) for validation and provenance.
Existing third-party licenses and attribution are retained.
'''

REPRODUCTION = '''# Reproduction and asset contract

## Evidence inspection

Run `python inspect_release.py --verify` from the repository root. This checks
published file hashes, catalogue targets, and syntax/JSON integrity. The
release manifest records the archive source hash and published hash separately.
Result-path normalization preserves JSON scalar values, arrays and ordering.

## Scientific environment

Use the versions recorded in `provenance/environment_versions.json` and
`requirements.txt`. `requirements-reproduction.txt` adds the orchestration
dependency (`psutil`) to that original list. PyTorch/CUDA and torchvision must be installed explicitly
for the chosen platform; this release records the original environment rather
than silently substituting new library versions. Orchestration also needs
`psutil` (the release validation record records its tested version).

From the repository root, with the scientific environment installed:

```bash
python tests/check_imports.py
python tests/check_grounding_contract.py
```

The first checks imports and unresolved global references. The second is a CPU
contract check with actual recurrent networks and a recording decoder stub;
it tests label partitioning, feature routing and update budgets, not performance.
`tests/check_mamujoco_frozen_dispatch.py --reference <archived-limited_labels.py>`
checks fifteen original MaMuJoCo paths with two real updates using synthetic
observations; obtain the reference from commit `2a784fa` or the original archive.
Other `tests/check_*.py` scripts include real GPU training/replays and require
the assets below. They use new output paths and reject existing partial runs.

## External assets for training and replay

The scientific runners retain their original workspace-relative routing. For
archival replay, place this checkout at
`<workspace>/paper/MTE_Paper_Code_Final_2026_09_17/` and restore the source
trees named in the chosen `configs/*.json`, its experiment documentation and
source manifests. A default clone with no assets supports result inspection,
not full retraining. The exact scientific runners/configurations are unchanged.

Required asset classes include observation-only train/dev exports, frozen
matched endpoints and valid masks, original representation/history weights,
budgeted target-action exports, reference decoder weights and trajectory replay
records. Visual studies also use frozen RGB/PCA artifacts and the declared
DAVIS/distracting-control assets; MaMuJoCo uses its fixed teacher and simulator
configuration. Respect the original datasets' and third parties' licenses.

Public metadata uses `WORKSPACE_ROOT/`, `PACKAGE_ROOT/`, or
`LOCAL_ASSET_ROOT/` to denote original machine-specific paths. These are provenance
identifiers, not downloadable URLs. Full replay requires resolving the original
asset paths, not replacing data with newly generated trajectories under old labels.
Exported status fields omit PIDs and timestamps; archive hashes identify their
original files. These sanitized exports are not substitutes for original audit
manifests during replay.

The archival managers also call `<workspace>/tools/verify_frozen_foundation.py`
with the original workspace freeze manifest. That guard covers additional
historical research outside this paper. Those unrelated records are not included
in this publication: original manager replay requires that original workspace
guard and its referenced files. The public integrity inspector checks this
release and does not bypass or replace the original scientific gate.

Protocol entry points expose their own CLI (`python -m experiments.<name> --help`).
Read the matching documentation first: some entry points require a manager/worker
subcommand. Run the specified qualification/replay checks before full jobs.
Never bypass source hashes, native parity or label-access guards to fit a new
machine. New results belong in a fresh run directory.

## Observation and supervision boundary

Observation-only representation/history learning and budgeted action grounding
are separate stages. Frozen trains a decoder. Adapt trains copies of selected
history branches plus the decoder within the fixed target-action budget. The
RGB/PCA front-end and original archived models stay unchanged. Training simulation
queries and partner-action labels remain prohibited where specified. Evaluation
simulator/oracle diagnostics are explicitly separated from deployable policies.

## Historical records

September 17 mixed-MPE reports and obsolete delivery guides have been removed
from the current checkout. They remain available in Git history and original
archives. The current file inventory is `provenance/release_manifest.json`;
current checks are in `provenance/release_validation.json`. Per-experiment
qualification records retain their original scientific meaning. The retirement
list gives each removed path, its reason and replacement.
'''

INSPECTOR = '''"""Read-only release integrity and experiment catalogue inspector."""
import argparse, ast, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def verify():
    manifest=read(ROOT/"provenance/release_manifest.json")
    for retired in manifest["excluded"]:
        assert not (ROOT/retired).exists(), ("retired file present", retired)
    for row in manifest["files"]:
        p=ROOT/row["path"]
        assert p.is_file(),row["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row["published_sha256"],row["path"]
        if p.suffix==".json": read(p)
        if p.suffix==".py": ast.parse(p.read_text(encoding="utf-8-sig"),filename=str(p))
    for row in read(ROOT/"provenance/experiment_catalog.json")["experiments"]:
        for key in ("entry", "summary", "status"):
            assert (ROOT/row[key]).is_file(),(row["id"],key)
        assert read(ROOT/row["status"])["status"]=="complete",row["id"]
    print(json.dumps({"status":"pass","files_checked":len(manifest["files"]),"scope":"Published integrity and syntax; no training or replay"}))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list",action="store_true");ap.add_argument("--verify",action="store_true");ap.add_argument("--report")
    a=ap.parse_args(); rows=read(ROOT/"provenance/experiment_catalog.json")["experiments"]
    if a.list:
        for r in rows: print(r["id"],r["manuscript_status"],r["question"],sep=" | ")
    if a.verify: verify()
    if a.report:
        row=next((r for r in rows if r["id"]==a.report),None)
        if row is None: ap.error("Unknown experiment; use --list")
        print(json.dumps(read(ROOT/row["summary"]),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
'''

def sha(data):
    return hashlib.sha256(data).hexdigest()

def build(source, manuscript, destination):
    source, manuscript, destination = map(lambda p: p.resolve(), (source, manuscript, destination))
    assert source != destination and (destination / '.git').is_dir()
    workspace = source.parents[1]
    inventory = {}

    def normalize(s):
        # String-only transforms: never alter numeric results or scientific code.
        for root, name in ((source, 'PACKAGE_ROOT'), (workspace, 'WORKSPACE_ROOT')):
            for old in (str(root), root.as_posix(), str(root).replace('\\', '\\\\')):
                s = s.replace(old, name)
        s = re.sub(r'[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s"<>]+', 'LOCAL_ASSET_ROOT', s)
        return s

    def clean(obj):
        if isinstance(obj, str): return normalize(obj)
        if isinstance(obj, list): return [clean(x) for x in obj]
        if isinstance(obj, dict):
            out = {normalize(k): clean(v) for k, v in obj.items()}
            assert len(out) == len(obj), 'Path normalization key collision'
            return out
        return obj

    def emit(path, data, origin=None, original=None, transform='generated'):
        out=destination/path
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes(data)
        inventory[path] = dict(path=path,published_sha256=sha(data),bytes=len(data),
                               origin=origin,source_sha256=sha(original) if original is not None else None,
                               transform=transform)

    def export(src, path, origin):
        if path in EXCLUDE: return
        data=src.read_bytes();published=data
        if src.suffix=='.json':
            raw=json.loads(data.decode('utf-8-sig'));sanitized=clean(raw)
            if raw != sanitized: published=(json.dumps(sanitized,ensure_ascii=False,indent=2)+'\n').encode()
        elif src.suffix in ('.md','.tex','.txt'):
            published=normalize(data.decode('utf-8-sig')).encode()
        emit(path,published,origin,data,'exact' if data==published else 'path_normalization')

    def document(path, text): emit(path,text.encode())
    def metadata(path, value): document(path,json.dumps(value,ensure_ascii=False,indent=2)+'\n')

    for folder in FOLDERS:
        for f in sorted((source/folder).rglob('*')):
            rel=f.relative_to(source).as_posix()
            if not f.is_file() or '__pycache__' in f.parts or rel in EXCLUDE: continue
            if f.suffix not in ('.py','.json','.md','.txt','.tex') and f.name!='LICENSE': continue
            export(f,rel,'package/'+rel)
    for name in ('run.py','requirements.txt','DEPENDENCIES.md','版本归并说明.md','冒烟测试说明.md','模型与数据边界.md','阅读顺序.md'):
        if (source/name).exists(): export(source/name,name,'package/'+name)
    export(source/'README.md','docs/README_PRE_RELEASE_ARCHIVE.md','package/README.md')
    document('README.md',README)
    document('docs/REPRODUCIBILITY.md',REPRODUCTION)
    document('inspect_release.py',INSPECTOR)
    export(source/'provenance/release_tools/current_run.py', 'run.py', 'package/provenance/release_tools/current_run.py')
    document('.gitattributes','* -text\n')
    document('requirements-reproduction.txt','-r requirements.txt\npsutil==7.2.2\n')
    document('.gitignore','__pycache__/\n*.pyc\noutputs/\nlocal_assets/\n*.pt\n*.npz\n*.npy\n.venv*/\n*.log\n.env\n.env.*\n')

    experiments=[]
    for name, module, integration, question in RUNS:
        run=source/'outputs'/name
        status=json.loads((run/'status.json').read_text(encoding='utf-8-sig'))
        assert status['status']=='complete',(name,status.get('status'))
        prefix=f'results/completed_runs/{name}'
        st={'status':'complete','run_id':name,'archival_status_sha256':sha((run/'status.json').read_bytes())}
        metadata(prefix+'/status.json',st)
        assert (run/'summary.json').exists(),name
        export(run/'summary.json',prefix+'/summary.json',f'package/outputs/{name}/summary.json')
        for extra in ('qualification.json','pairing_checks.json','factorial_checks.json'):
            if (run/extra).exists():export(run/extra,prefix+'/'+extra,f'package/outputs/{name}/{extra}')
        row=dict(id=name,question=question,manuscript_status=integration,entry=f'experiments/{module}.py',
                 summary=prefix+'/summary.json',status=prefix+'/status.json')
        for key, relative in [('protocol',f'configs/{module}.json'),('documentation',f'docs/{module}.md'),('report',f'results/{name}/RESULTS_CN.md')]:
            if (source/relative).exists():row[key]=relative
        experiments.append(row)
    metadata('provenance/experiment_catalog.json',dict(release='2026-09-20',
        manuscript_snapshot=manuscript.name,experiments=experiments,
        supporting_entry_points=[dict(entry='experiments/mamujoco_frozen_control.py', scope='Original repaired MaMuJoCo main-method representation/history API; rejects MPE'), dict(entry='experiments/visual_control.py', scope='Original visual frontend and frozen-control stage API')]))

    metadata('provenance/retirement_audit.json', dict(scope='Retired from current delivery; original archives preserved', files=RETIREMENTS))
    numeric=json.loads((manuscript/'evidence/unchanged_numeric_summary.json').read_text(encoding='utf-8'))
    metadata('results/current/mamujoco_controls.json', [r for r in numeric if r['environment']=='mamujoco'])
    evidence=[]
    for f in sorted((manuscript/'evidence').iterdir()):
        if f.is_file() and f.suffix in ('.json','.tex','.md'):
            rel='results/manuscript_snapshot/'+f.name
            export(f,rel,'manuscript/evidence/'+f.name);evidence.append(rel)
    metadata('provenance/manuscript_snapshot.json',dict(directory=manuscript.name,
        files=[dict(path=n,sha256=sha((manuscript/n).read_bytes())) for n in ('main.tex','main.pdf','EVIDENCE_MAP.md')],
        evidence=evidence,adaptation_integrated=False))
    mapping=['# Paper-to-code and evidence map','',
             'The manuscript snapshot and hashes are recorded in `provenance/manuscript_snapshot.json`.',
             'Current plotted/table exports are in `results/manuscript_snapshot/`. The original mixed reports',
             'are retired. Use the completed MPE inventory and `results/current/mamujoco_controls.json`.','',
             '| Experiment/question | Entry point | Evidence | Manuscript status |','|---|---|---|---|']
    for r in experiments:
        mapping.append(f"| {r['question']} | [{r['entry']}](../{r['entry']}) | [{r['id']}](../{r['summary']}) | {r['manuscript_status']} |")
    mapping += ['', '`paper` means this experiment supplies evidence used in the current manuscript; it does not',
                'mean every result row is printed. `followup` means separately completed evidence awaiting integration.',
                '', '## Protocol distinctions', '',
                '- Original MaMuJoCo route/full-visibility MIF and native masked MIF have separate objectives and labels.',
                '- Graph16 matching-input controls preserve endpoint information and keep their own results; the main Graph8 scores are unchanged.',
                '- Raw and random-edge controls retain MTE inputs. They test encoder increments within that construction.',
                '- Donor query-state oracle re-encoding uses evaluation simulator outcomes; its improvement is diagnostic, not a deployed policy repair.',
                '- Visual B1/B2/B4/B8 supervision inventory, B2 cross-family adaptation, MPE B32 pilot and MaMuJoCo B8 adaptation retain distinct scopes.',
                '- MaMuJoCo adaptation uses one decoder seed with five upstream seeds; the route factorial uses five nested decoder repeats.',
                '- Branch attribution retains both information branches. The selected branch receives gradients; capacity-matched LAOM uses its state adapter.',
                '- Fixed-budget adaptation is exploratory follow-up on existing tasks. New tasks, larger agent counts and visual entity alignment remain separate work.','']
    document('docs/PAPER_CODE_MAP.md','\n'.join(mapping))
    document('docs/RELEASE_2026_09_20.md','''# Release 2026-09-20

This release updates the existing GitHub codebase without replacing scientific
definitions or rewriting Git history. It adds seventeen completed experiment
records, current manuscript numerical exports, and a standard-library evidence
inspector. Frozen and Adapt results are indexed separately.

Scientific Python/config files are copied byte-for-byte from the accepted code
baseline. A new MaMuJoCo-only dispatcher removes the obsolete MPE branch from
the mixed dispatcher; its fifteen original method paths passed actual two-update
checkpoint parity with zero tensor differences. The current report launcher and
retirement map replace superseded default entry points. Only machine-specific paths in textual evidence are normalized.
`release_manifest.json` records original and published hashes for every exported
file. Internal review/editing plans, dataset/checkpoint/log/cache files and the one-off process-recovery script
are excluded. Its associated recovery test is also excluded; the scientific
supervision runner and its qualification records remain included.

The source archive and completed formal results remain unchanged. In particular,
the shared core files and historic protocol hashes are preserved. Retired September 17
reports remain in original archives and Git history; current publication checks
are recorded in `provenance/release_validation.json`.

This publication performs integrity, import/interface and small synthetic checks.
It does not rerun every formal training job or re-estimate published performance.
The original per-run reports contain the full scientific qualification records.

The manuscript PDF and figures have not been edited by this code release.
''')
    # Every existing tracked delivery file is also covered; do not silently delete
    # remote content. The caller must inspect the git diff before committing.
    import subprocess
    tracked=subprocess.check_output(['git','-C',str(destination),'ls-files','-z']).decode().split('\0')
    for rel in tracked:
        if rel and rel not in inventory and rel not in EXCLUDE and rel not in ('provenance/release_manifest.json','provenance/release_validation.json') and (destination/rel).is_file():
            data=(destination/rel).read_bytes()
            inventory[rel]=dict(path=rel,published_sha256=sha(data),bytes=len(data),origin='previous_publication',source_sha256=sha(data),transform='retained')
    manifest=dict(release='2026-09-20',files=list(sorted(inventory.values(),key=lambda x:x['path'])),
                  excluded=sorted(EXCLUDE),manifest_self_excluded=True,
                  validation_record_excluded='provenance/release_validation.json')
    (destination/'provenance/release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(inventory),'experiments':len(experiments),'bytes':sum(r['bytes'] for r in inventory.values())}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--manuscript',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    args=ap.parse_args();build(args.source,args.manuscript,args.destination)
