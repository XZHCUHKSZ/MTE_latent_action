# Running the code

## Scientific code and assets

Install the recorded numerical/visual environments in separate environments as
needed; see `requirements.txt` and `DEPENDENCIES.md`. Torch and
Torchvision wheels must match the recorded platform. External author baselines
also need their pinned repositories and original dependency environments in
`DEPENDENCIES.md`. FLAM uses Linux and author dependencies;
its unlicensed source and perceptual weights are not redistributed here.

A clone includes code and protocols only, **not a self-contained
retraining dataset**. Required external assets include observation-only train/dev
exports, frozen matched endpoints/masks, pretrained representation/history
artifacts, fixed target-action exports, teachers and visual render assets (including
DAVIS where used). Preserve their original SHA-256 identities. Collecting new data
creates a new reproduction, not a replacement for the reported evidence.

The established MPE/MaMuJoCo/Ant stages retain workspace-relative asset contracts.
For archival replay, install this tree at `<workspace>/paper/MTE_Paper_Code_Final_2026_09_17/`
and restore the frozen asset paths referenced by the selected protocol. Stage
functions are APIs, not promises that every file is a standalone training CLI.
Use the mapped `worker`, `pretrain`, `ground`, `evaluate`, or readout API with its
recorded parameters, one stage per fresh process. `visual_control.py --help`
provides the established explicit visual-stage interface.

External Ant upstream scripts preserve their original preflight/source guards.
They additionally require the original OTF source manifest and FLAM source,
perceptual-asset and build/preflight records referenced in those scripts. These
archive records are not included in this code-only release, so **a default
clone cannot launch those archival upstream scripts**. We do not disable their
checks or silently label an unqualified replay as the original experiment.
External downstream `external_ant_stages.worker` needs OUT/UP bindings to the
selected three-seed run, its original source/freeze records, and the 27-condition
configuration. There is no published continuation manager.

## Coupled Frozen stages

Use the exact two frozen SAC teachers and their `teacher_freeze.json`/`result.json`
sidecars. `configs/coupled_frozen_five_seed.json` fixes the acquisition, training,
evaluation seeds, all budgets and both teacher strata. Restore observation arrays
under `<data-root>/seed<teacher>/acquire/data64/`, or acquire a new reproduction
with `collect` using the same teacher and fixed acquisition seeds.

```sh
python -m experiments.coupled_frozen init --run-dir outputs/coupled_reproduction --data-root local_assets/coupled/observations --teacher-root local_assets/coupled/teachers
python -m experiments.coupled_frozen seal-data --run-dir outputs/coupled_reproduction
python -m experiments.coupled_frozen pre --run-dir outputs/coupled_reproduction --teacher 202622200 --seed 202623500
```

Complete `pre` for all ten teacher/seed pairs, then `seal-pre`, then `labels`.
Complete `ground --teacher ... --seed ... --arm ...` for all 190 controllers, then
`seal-ground`, then `evaluate` for all ten pairs. `arms()` in
`training/coupled_cheetah_frozen.py` lists the 19 allowed arms. The two seal stages
refuse incomplete artifacts. Grounding verifies frozen observation models;
evaluation verifies all frozen controllers. Run each command in a fresh process
because label/simulator access guards are process-wide. No Adapt arm is accepted.

## Protocol scope

Frozen grounding trains only a decoder. Adapt changes permitted history copies.
Solo/Aux is a separate input-composition choice. The new Coupled entry accepts
only Frozen arms; existing Ant and population Adapt code remains because those
experiments appear in the manuscript.

MaMuJoCo protocols retain dataset split IDs and schedules. Resolve
`WORKSPACE_ROOT` against your immutable assets. For RGB Ant, place a copy of
`configs/visual_config.json` in `--asset-dir` alongside separately supplied labels.

`tests/check_coupled_frozen.py --reference <archived-adapter.py>` is an optional
synthetic CPU extraction-parity check; it requires the separately held original
adapter. No recorded experimental outcomes are bundled with these tests.
