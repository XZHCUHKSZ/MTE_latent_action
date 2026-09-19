# 最新论文与代码逐项绑定

本发布绑定 `iclr2027_mpe_completed_revision_2026_09_18` 的最新接受稿（含 9 月 19 日视觉低标签更新），不是只按目录日期选旧稿。PDF SHA256：`52e01af2b1a6921a203498e962fcc20f46f7a6d8610bff9554443d47af987f5b`。论文与图没有被本次代码整理改写。

## 实测通过的范围

- 五张主回报表的 371 个均值/样本标准差单元格，按论文打印精度独立复算通过。
- 图中数值、数据规模和路线输入的 1,570 个种子值已与对应完整结果核对。
- 21 张表逐项记录正文行号、块哈希、证据与实现路径；这不表示剩余所有诊断表均重新训练或独立复算。
- 3 个实际引用 PDF 图均锁定哈希。图 2/3 的原画图脚本原字节收录；图 1 是单独设计图，仅提供接受稿资产指纹，不伪造其生成器。

## 逐表清单

|表|问题|证据|实现/范围|验证级别|
|---|---|---|---|---|
|A1|MPE main returns I|[mpe_completed.json](../results/manuscript_snapshot/mpe_completed.json)|[experiments/mpe_inventory_completion.py](../experiments/mpe_inventory_completion.py)|numeric_recomputed|
|A2|MPE main returns II|[mpe_completed.json](../results/manuscript_snapshot/mpe_completed.json)|[experiments/mpe_inventory_completion.py](../experiments/mpe_inventory_completion.py)|numeric_recomputed|
|A3|MaMuJoCo main returns I|[mamujoco_controls.json](../results/current/mamujoco_controls.json)|[experiments/mamujoco_frozen_control.py](../experiments/mamujoco_frozen_control.py)<br>[training/grounding_mamujoco.py](../training/grounding_mamujoco.py)|numeric_recomputed|
|A4|MaMuJoCo main returns II|[mamujoco_controls.json](../results/current/mamujoco_controls.json)|[experiments/mamujoco_frozen_control.py](../experiments/mamujoco_frozen_control.py)<br>[training/grounding_mamujoco.py](../training/grounding_mamujoco.py)|numeric_recomputed|
|A5|Visual B8 and low-label returns|[visual_summary.json](../results/manuscript_snapshot/visual_summary.json)<br>[visual_low_label_summary.json](../results/manuscript_snapshot/visual_low_label_summary.json)|[experiments/visual_control.py](../experiments/visual_control.py)<br>[experiments/visual_low_label.py](../experiments/visual_low_label.py)|numeric_recomputed|
|A6|MPE native outcomes|[mpe_completed.json](../results/manuscript_snapshot/mpe_completed.json)|[experiments/mpe_inventory_completion.py](../experiments/mpe_inventory_completion.py)|source_bound|
|A7|MPE data scaling|[mpe_completed.json](../results/manuscript_snapshot/mpe_completed.json)|[experiments/mpe_inventory_completion.py](../experiments/mpe_inventory_completion.py)|source_bound|
|A8|MaMuJoCo scaling endpoints|[unchanged_numeric_scaling.json](../results/manuscript_snapshot/unchanged_numeric_scaling.json)|[experiments/mamujoco_frozen_control.py](../experiments/mamujoco_frozen_control.py)|source_bound|
|A9|MPE history action/skill probes|[mpe_completed.json](../results/manuscript_snapshot/mpe_completed.json)|[experiments/mpe_inventory_completion.py](../experiments/mpe_inventory_completion.py)<br>[evaluation/probes.py](../evaluation/probes.py)|source_bound|
|A10|MaMuJoCo history action/skill probes|[mamujoco_action.json](../results/probes/mamujoco_action.json)|[experiments/action_information.py](../experiments/action_information.py)<br>[evaluation/probes.py](../evaluation/probes.py)|source_bound|
|A11|MaMuJoCo correspondence reassignment|[historical_correspondence.json](../results/manuscript_snapshot/historical_correspondence.json)<br>[historical_secondary_tables.tex](../results/manuscript_snapshot/historical_secondary_tables.tex)|存档数值；暂无独立当前入口|archived_numeric_record_no_dedicated_current_runner|
|A12|MIF versus formal Base|[cross_setting_control_gains_2026_09_19.json](../results/manuscript_snapshot/cross_setting_control_gains_2026_09_19.json)|[training/composition.py](../training/composition.py)|source_bound|
|A13|Literature and access comparison|接受稿中带引用的文献表|文献归纳，不代表运行官方基线|literature_table_not_executed_baselines|
|A14|Visual supervision and parameter counts|[visual_summary.json](../results/manuscript_snapshot/visual_summary.json)|[experiments/visual_control.py](../experiments/visual_control.py)|source_bound|
|A15|MPE structural and matching contrasts|[mpe_completed.json](../results/manuscript_snapshot/mpe_completed.json)<br>[mpe_matching_input_control_2026_09_19.json](../results/manuscript_snapshot/mpe_matching_input_control_2026_09_19.json)|[experiments/mpe_temporal_scale.py](../experiments/mpe_temporal_scale.py)<br>[experiments/mpe_evidence_completion.py](../experiments/mpe_evidence_completion.py)<br>[experiments/mpe_matching_input_control.py](../experiments/mpe_matching_input_control.py)|source_bound|
|A16|MPE structural budget returns|[mpe_completed.json](../results/manuscript_snapshot/mpe_completed.json)|[experiments/mpe_temporal_scale.py](../experiments/mpe_temporal_scale.py)<br>[experiments/mpe_evidence_completion.py](../experiments/mpe_evidence_completion.py)|source_bound|
|A17|MaMuJoCo original structural controls|[historical_secondary_tables.tex](../results/manuscript_snapshot/historical_secondary_tables.tex)<br>[primary_holm.json](../results/structural/primary_holm.json)|[experiments/resource_controls.py](../experiments/resource_controls.py)<br>[mte/structured_controls.py](../mte/structured_controls.py)|source_bound|
|A18|MPE physical donor and query-state oracle|[mpe_completed.json](../results/manuscript_snapshot/mpe_completed.json)<br>[mpe_donor_zero_audit_2026_09_19.json](../results/manuscript_snapshot/mpe_donor_zero_audit_2026_09_19.json)|[experiments/mpe_donor_zero_audit.py](../experiments/mpe_donor_zero_audit.py)|source_bound|
|A19|MPE routes versus Global16|[route_comparison_current.json](../results/manuscript_snapshot/route_comparison_current.json)|[experiments/mpe_inventory_completion.py](../experiments/mpe_inventory_completion.py)|figure_input_recomputed|
|A20|MPE representation/history coefficient readout|[mpe_completed.json](../results/manuscript_snapshot/mpe_completed.json)|[experiments/mpe_inventory_completion.py](../experiments/mpe_inventory_completion.py)|source_bound|
|A21|MaMuJoCo coefficient readout|[rows.json](../results/interactions/agent_readout/rows.json)<br>[mamujoco_representation_action_readout.tex](../results/manuscript_snapshot/mamujoco_representation_action_readout.tex)|[experiments/partner_interactions.py](../experiments/partner_interactions.py)<br>[evaluation/probes.py](../evaluation/probes.py)|source_bound|

## 可复核入口

```bash
python inspect_release.py --verify
python provenance/release_tools/check_manuscript_alignment.py --manuscript /path/to/accepted-manuscript --release . --report /path/to/new-report.json
python evaluation/reproduce_paper_figures.py --rgb /path/to/illustration_rgb.npy --out /path/to/new-figure-directory
```

第二条需要本地接受稿，拒绝覆盖已有报告；第三条需原始记录帧，SHA256 锁定在数值核验记录中，生成到不存在的新目录。RGB 数据不随代码分发。图的生成数据核对不等于 PDF 字节或像素布局完全一致。完整训练仍需要外部数据和冻结权重，详见 REPRODUCIBILITY.md。

## 方法语义与版本边界

- MPE 使用修正后的 t+2 逆模型和直接多时域读出；原混合 worker 的旧 MPE 路径不是当前入口。MaMuJoCo 原始方法通过独立 frozen-control 分发器保留。
- 论文早期源码名 frontend.py/worker.py 是历史来源定位；现发布对应 mte/frontends.py 和各环境 experiments/training 入口。provenance/function_sources.json、reorganized_interfaces.json 保留历史沿革，不是当前运行选择器。
- Base 是全系统分支；MTE-Aux 是 Base 与 MTE 分支组合；Global16 是另一个全系统表示路线。三者不得因改称系统分支而混成同一对照。
- MaMuJoCo 875 路线 MIF 与原生 masked MIF 的训练目标不同；500 遮蔽×坐标研究及微调研究保留各自协议。raw/random 对照仍保留 MTE 输入，不是删除整个 MTE。
- 620 视觉库存、跨方法微调、MTE 微调扩展、分支归因属于已完成后续证据，尚未合入本 PDF；Frozen 和 Adapt 分开报告，不择优替换原表。
- 原 EVIDENCE_MAP 的早期待办段有历史状态；以该文件末尾低标签合入记录和本发布 catalogue 的 paper/followup 标识为准。

## 可复现性尚有的缺口

A11 有冻结数值和来源记录，但当前整理包没有独立的整套重跑入口；A13 是文献比较，不能充当官方结构基线复现。外部训练资产未随包分发。本发布明确这些差别，不将哈希一致、脚本导入或数值复算说成所有实验从头重现。
