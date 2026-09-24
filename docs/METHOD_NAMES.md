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
