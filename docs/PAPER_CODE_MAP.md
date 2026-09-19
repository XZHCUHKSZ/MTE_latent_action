# Paper-to-code and evidence map

The manuscript snapshot and hashes are recorded in `provenance/manuscript_snapshot.json`.
See [the per-table alignment audit](MANUSCRIPT_ALIGNMENT_CN.md) for all 21 tables, included figures, numerical checks and remaining replay gaps.
Current plotted/table exports are in `results/manuscript_snapshot/`. The original mixed reports
are retired. Use the completed MPE inventory and `results/current/mamujoco_controls.json`.

| Experiment/question | Entry point | Evidence | Manuscript status |
|---|---|---|---|
| MPE temporal structural controls | [experiments/mpe_temporal_scale.py](../experiments/mpe_temporal_scale.py) | [mpe_temporal_full_scale_5seeds](../results/completed_runs/mpe_temporal_full_scale_5seeds/summary.json) | paper |
| Additional structural and donor controls | [experiments/mpe_evidence_completion.py](../experiments/mpe_evidence_completion.py) | [mpe_evidence_completion_2026_09_18](../results/completed_runs/mpe_evidence_completion_2026_09_18/summary.json) | paper |
| Complete MPE configuration, label and data inventory | [experiments/mpe_inventory_completion.py](../experiments/mpe_inventory_completion.py) | [mpe_inventory_completion_2026_09_18](../results/completed_runs/mpe_inventory_completion_2026_09_18/summary.json) | paper |
| Fixed-policy partner perturbation diagnostic | [experiments/mpe_partner_shift.py](../experiments/mpe_partner_shift.py) | [mpe_partner_shift_2026_09_18_r1](../results/completed_runs/mpe_partner_shift_2026_09_18_r1/summary.json) | followup |
| Simulator second-order physical-effect onset | [experiments/mpe_physical_interaction_onset.py](../experiments/mpe_physical_interaction_onset.py) | [mpe_physical_interaction_onset_2026_09_18](../results/completed_runs/mpe_physical_interaction_onset_2026_09_18/summary.json) | followup |
| Read-only donor transport diagnostic | [experiments/mpe_donor_transport_audit.py](../experiments/mpe_donor_transport_audit.py) | [mpe_donor_transport_audit_2026_09_18](../results/completed_runs/mpe_donor_transport_audit_2026_09_18/summary.json) | followup |
| Zero reference and query-state oracle diagnostic | [experiments/mpe_donor_zero_audit.py](../experiments/mpe_donor_zero_audit.py) | [mpe_donor_zero_audit_2026_09_19](../results/completed_runs/mpe_donor_zero_audit_2026_09_19/summary.json) | paper |
| Same-resource Graph16 input control | [experiments/mpe_matching_input_control.py](../experiments/mpe_matching_input_control.py) | [mpe_matching_input_control_2026_09_19](../results/completed_runs/mpe_matching_input_control_2026_09_19/summary.json) | paper |
| Seven-route matched/raw and Global16 comparison | [experiments/mamujoco_route_completion.py](../experiments/mamujoco_route_completion.py) | [mamujoco_route_completion_2026_09_19](../results/completed_runs/mamujoco_route_completion_2026_09_19/summary.json) | paper |
| Native masking by coordinate-system factorial | [experiments/mamujoco_mif_visibility.py](../experiments/mamujoco_mif_visibility.py) | [mamujoco_mif_visibility_2026_09_19](../results/completed_runs/mamujoco_mif_visibility_2026_09_19/summary.json) | paper |
| Frozen matching implementation audit | [experiments/mamujoco_frozen_matching_audit.py](../experiments/mamujoco_frozen_matching_audit.py) | [mamujoco_frozen_matching_audit_2026_09_19](../results/completed_runs/mamujoco_frozen_matching_audit_2026_09_19/summary.json) | followup |
| Agent interface and computational resource audit | [experiments/mte_agent_resource_audit.py](../experiments/mte_agent_resource_audit.py) | [mte_agent_resource_audit_2026_09_19](../results/completed_runs/mte_agent_resource_audit_2026_09_19/summary.json) | followup |
| Visual low-label curves for nine configurations | [experiments/visual_low_label.py](../experiments/visual_low_label.py) | [visual_low_label_2026_09_19_r1](../results/completed_runs/visual_low_label_2026_09_19_r1/summary.json) | paper |
| Complete visual label inventory and supervision factorial | [experiments/visual_supervision_inventory.py](../experiments/visual_supervision_inventory.py) | [visual_supervision_inventory_2026_09_19](../results/completed_runs/visual_supervision_inventory_2026_09_19/summary.json) | followup |
| Cross-method visual and MPE history adaptation | [experiments/policy_finetune_pilot.py](../experiments/policy_finetune_pilot.py) | [policy_finetune_pilot_2026_09_20_r1](../results/completed_runs/policy_finetune_pilot_2026_09_20_r1/summary.json) | followup |
| MTE-family visual and MaMuJoCo adaptation | [experiments/mte_finetune_extension.py](../experiments/mte_finetune_extension.py) | [mte_finetune_extension_2026_09_20](../results/completed_runs/mte_finetune_extension_2026_09_20/summary.json) | followup |
| Selective branch adaptation and capacity-matched LAOM control | [experiments/visual_branch_attribution.py](../experiments/visual_branch_attribution.py) | [visual_branch_attribution_2026_09_20](../results/completed_runs/visual_branch_attribution_2026_09_20/summary.json) | followup |

`paper` means this experiment supplies evidence used in the current manuscript; it does not
mean every result row is printed. `followup` means separately completed evidence awaiting integration.

## Protocol distinctions

- Original MaMuJoCo route/full-visibility MIF and native masked MIF have separate objectives and labels.
- Graph16 matching-input controls preserve endpoint information and keep their own results; the main Graph8 scores are unchanged.
- Raw and random-edge controls retain MTE inputs. They test encoder increments within that construction.
- Donor query-state oracle re-encoding uses evaluation simulator outcomes; its improvement is diagnostic, not a deployed policy repair.
- Visual B1/B2/B4/B8 supervision inventory, B2 cross-family adaptation, MPE B32 pilot and MaMuJoCo B8 adaptation retain distinct scopes.
- MaMuJoCo adaptation uses one decoder seed with five upstream seeds; the route factorial uses five nested decoder repeats.
- Branch attribution retains both information branches. The selected branch receives gradients; capacity-matched LAOM uses its state adapter.
- Fixed-budget adaptation is exploratory follow-up on existing tasks. New tasks, larger agent counts and visual entity alignment remain separate work.
