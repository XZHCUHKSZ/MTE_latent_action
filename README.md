# MTE experimental code

Code and protocols corresponding to the 24 September 2026 professor-verbatim
manuscript. **No experimental results, numeric result exports, reports, logs,
weights, datasets, or manuscript files are included.**

- MPE and MaMuJoCo method, budget, scaling, structural and readout experiments.
- RGB Ant label-budget, initialization and branch controls.
- Four/eight-agent MPE population experiments.
- The manuscript's three-seed OTF/FLAM comparisons.
- Clean-RGB CoupledHalfCheetah, Frozen only.

Only Adapt experiments already present in the manuscript are retained.
Deferred external five-seed extensions, Coupled Adapt, unused tasks, background
managers and retry dispatchers are excluded.

See the [paper-to-code map](docs/PAPER_CODE_MAP.md),
[reproduction guide](docs/REPRODUCIBILITY.md), and [dependencies](DEPENDENCIES.md).

## Checks and entry points

```sh
python inspect_release.py --verify
python tests/check_release_interfaces.py
python -m experiments.coupled_frozen --help
python -m experiments.visual_control --help
```

The first command checks source/configuration syntax and the code-only file
policy without training. Interface checks require NumPy/SciPy. Experiment stages
need the dependencies and separately supplied assets described in the guide.

## Layout

`experiments/`: stage entry points; `configs/`: protocols; `mte/`, `training/`:
representation, history and grounding; `environments/`, `visual/`, `evaluation/`:
data adapters and evaluation; `tests/`: bounded code checks. Licensed third-party
source retains its original attribution. Shared core implementations are unchanged.
