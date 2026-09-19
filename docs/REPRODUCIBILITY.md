# Reproduction and asset contract

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

The original `provenance/package_files.json`, `validation.json`, `smoke_report.json`
and Chinese September 17 guides describe that earlier snapshot. The current file
inventory is `provenance/release_manifest.json`; current checks are in
`provenance/release_validation.json`. Recorded historical PASS results are evidence
from their original runs, not a claim that this publication reran all experiments.
