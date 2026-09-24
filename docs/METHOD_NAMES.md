# Predictive-comparison names

The manuscript describes the framework as **controlled predictive comparisons**, abbreviated **PC** for predictive comparison. A **matched target edge (MTE)** is the mathematical difference of the matched predictions.

| Paper variant | Short figure/table name | Previous name | Frozen implementation ID |
|---|---|---|---|
| PC-Mean | Mean | Edge-CARA / Edge | `edge_cara` |
| PC-Set | Set | Simple | `edge_cara_mobius_simple` |
| PC-Graph | Graph | Graph | `edge_cara_mobius_graph` |
| PC-Sweep | Sweep | Tree | `edge_cara_mobius_tree` |
| PC-Field | Field | MIF | `edge_cara_mif` |

Mean pools encoded matched differences; Set pools subset-coefficient tokens; Graph passes messages on subset/partner connections; Sweep performs ordered downward/upward passes on the subset DAG; Field includes endpoints and differences across horizons, subsets and output groups. Sweep avoids calling a Hasse DAG a strict tree. These are labels for existing implementations, not new algorithms.

Solo/Aux remains the composition axis; Frozen/Adapt remains the grounding-update axis. They are independent. A named horizon is an existing protocol choice, not a new family member.

```sh
python -m mte.method_names
python -m mte.method_names --resolve PC-Field-Solo --interface visual
python -m mte.method_names --resolve PC-Mean-h2
python -m mte.method_names --resolve PC-Field-Aux --interface mpe
```

Use `resolve_method(name, interface)` before a legacy training API; `paper_name(id)` formats old IDs for new displays. The curated visual CLI accepts `--method PC-Field` for pretraining and `--method PC-Field-Solo` for grounding. The curated Coupled CLI accepts `--arm PC-Field-Solo`. The curated MaMuJoCo training API accepts bare PC names. Existing configs, core code, checkpoint namespaces and completed results keep their historical identifiers, so they still select exactly the original implementation and training profile. The same registry is installed in the editable paper-code baseline.

This is a naming/interface update only. No new scientific run, change of budget, numerical result, or model parameter is included.

For MPE, the curated release inventory entry normalizes public names in caller-supplied `methods` and `latent_methods` lists without changing the supplied object or other parameters. `PC-Mean-h2/h3` requires an explicit horizon; `PC-Field-Solo/Aux` maps to `mif`/`base+mif`. `paper_name(id, "mpe")` displays the corresponding paper label. Frozen protocol files retain their original identifiers.


## Complete public interfaces

The registry now binds the visual encoder, pair, grounding and controller APIs;
MaMuJoCo representation and history APIs; Coupled pair and grounding APIs;
MPE inventory, temporal and population APIs; coordinate/route controls; and
visual label-budget, initialization and branch controls. Public config/job inputs
are copied before normalization. Data paths, seeds, budgets, update counts,
losses, architecture and stored evidence IDs are unchanged.

| Interface | Example | Meaning |
|---|---|---|
| Encoder | `PC-Field` | Original Field encoder |
| Visual composition | `PC-Field-Solo` | Original standalone history |
| Coupled grounding | `PC-Field-Solo` or `PC-Field-Solo__Frozen` | Same Frozen arm; Adapt rejected |
| MPE composition | `PC-Mean-h3-Aux` | Explicit horizon, original Base-plus-Mean composition |
| MPE coordinate control | `PC-Graph-raw-Aux` | Same raw-coordinate auxiliary control |
| Structural control | `PC-Field-pair` | Same endpoint-pair input control |
| Branch control | `PC-Field-Aux-pretrained-auxonly` | Same initialization and gradient-access setting |

Use `resolve_control` for short structural/branch IDs, `resolve_visual_arm` for
visual compositions and qualified history controls, and `resolve_coupled_ground`
for complete Frozen Coupled arms. `visual.train.public_methods/public_arms` and
`training.coupled_cheetah_frozen.public_arms` expose the public catalogs.

All 14 Markdown report generators now format method labels through `report_text`.
New human-readable reports use PC names; their machine-readable JSON records,
checkpoint paths, source hashes and completed reports retain original IDs.
Third-party baseline names are unchanged. A matched target edge remains an MTE.

## Intentional compatibility identifiers

Old spellings still occur in the frozen core class names, dispatch IDs,
checkpoint keys, archived protocol files (including their original prose),
module/import filenames, provenance paths, and the explicit compatibility table
and tests. These are necessary reproducibility identifiers, not additional
paper methods. The `mte` package and repository URL remain stable. We do not
rewrite a frozen configuration or rename a serialized class merely to remove
search hits. Generic graph/tree variables and third-party source keep their
original meanings.

```sh
python tests/test_method_names.py
python tests/test_public_naming.py
python tests/check_release_interfaces.py
python inspect_release.py --verify
```

The integration suite uses synthetic inputs and CPU-only interface calls. It
checks public/legacy job matrices and paths, config non-mutation, horizon and
composition preservation, report labels and Frozen-only binding. It does not
train a policy, rerun an experiment, or validate performance. PyTorch,
NumPy and SciPy are needed for integration checks.
