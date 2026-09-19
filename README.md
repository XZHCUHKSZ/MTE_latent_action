# MTE: paper code and auditable experimental evidence

This repository organizes observation-only MTE pretraining, action grounding,
closed-loop evaluation, and structural controls across MPE, MaMuJoCo and visual
control. The 2026-09-20 release adds the completed experiments and audits from
September 18–20 to the original September 17 code release.

**Start with [the paper-to-code map](docs/PAPER_CODE_MAP.md),
[reproduction instructions](docs/REPRODUCIBILITY.md), and
[the experiment catalogue](provenance/experiment_catalog.json).**

The [manuscript alignment audit](docs/MANUSCRIPT_ALIGNMENT_CN.md) binds all 21
tables and three included figures to the accepted PDF. It records 371 recomputed
mean/SD cells, 1,570 checked figure input values, and the limits of that verification.

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
