# 当前有效版本与证据总核对（2026-09-20）

当前发布以**最新完成、来源完整、协议一致的结果**为准。修复版替代相同问题的旧实现，完整补齐版替代不完整的查询入口；Frozen、监督适配、不同训练目标分别报告。正式版本不按各预算或种子的最高回报拼接。

机器可读索引见 `provenance/current_scientific_state.json`。方法定义与退役文件见 `METHOD_LINEAGE_AUDIT_CN.md` 和 `provenance/retirement_audit.json`。

## 这次查清并处理的事项

1. **MPE 正式时间对齐已更换。** 当前使用 t+2 逆推/前端目标和直接 h1/h2/h3 匹配预测，控制标签仍对齐决策时刻。默认入口采用 1,305 单元的完整 inventory；旧一步 MPE 默认报告与分派器已退役。环境积分器和冻结共享方法没有被本次发布改写。
2. **系统分支的身份已区分。** MTE-Aux 是 Base + MTE history；Global-Joint16-Aux 是相同 Base + 全系统 16 维 history，用来替换并比较第二分支。原独立局部分支不是 Global-Joint16，旧数值不能改名复用。视觉 Base 为 32 维，其余相应数值分支配置见各自协议。
3. **视觉默认入口已升级。** `python run.py --report visual` 读取 620 单元完整清单；`--report visual-paper` 保留论文采用的 180 单元子集。旧 180 单元全部包含在 620 中，来源与均值逐项相同，不能把两批计作 800 个独立结果。
4. **视觉评估口径已核清。** 原 B8 文件的总均值使用 30 个条件；后续主统计按预定的 908604–908630 共 27 个条件，从相同逐回合记录重新汇总。新旧总均值不同不等于覆盖了原结果。全 30 条记录仍保留。
5. **监督适配与原 Frozen 并列索引。** Adapt 更新 history 副本和 decoder，不重训 RGB/PCA 或原表示，也不覆盖旧 history。预训练仍只读取观测；下游 Adapt 明确使用预算内动作标签。其收益单独配对计算，不替换 Frozen 主实验和消融。
6. **选择性更新确实发生在指定分支。** 分支归因的四格始终保留两路信息，区别是哪些 history 收到梯度。重新核对了 Base/辅助分支初末权重、25 组四格配对与五种子跨来源同容量记录；MIF 不是唯一测试模型。
7. **旧名字不等于无用实现。** 已退役 25 个旧入口、配置、旧默认结果或说明；实际仍被正式实验导入的旧名函数保留，并注明依赖。完成的实验档案、原始哈希和旧交付包不删除。

## 各问题采用哪一套结果

| 问题 | 当前有效证据 | 保留的区别 |
|---|---|---|
| MPE 主控制、标签/数据规模、结构对照 | `mpe_inventory_completion_2026_09_18`，1,305 单元；关联 temporal scale/evidence completion | 修正后的时间协议；旧一步成绩不混入 |
| MPE 显式 matching 输入 | `mpe_matching_input_control_2026_09_19`，15 单元 | Graph16 三种输入保留同一端点信息，测试输入组织；不是原 Graph8 或完整无 MTE 对照 |
| MPE zero/donor 机制 | `mpe_donor_zero_audit_2026_09_19` 及 transport/onset 诊断 | 原 zero 对照有效；query-state 重编码约 90.8% 的误差下降是模拟器 oracle 诊断，不是部署修复 |
| MaMuJoCo 主 Frozen 控制 | `results/current/mamujoco_controls.json` | 原常量输入保护后的主结果，原生 masked MIF |
| MaMuJoCo Global16/raw 路线 | `mamujoco_route_completion_2026_09_19`，875 单元 | 路线 MIF 是 full-visibility 整表重建；5 上游种子 × 5 decoder 重复 |
| MaMuJoCo 遮蔽 × 坐标 | `mamujoco_mif_visibility_2026_09_19`，500 单元 | 原生 masked 与 visible 对照保留随机 loss 分组；遮蔽发生在各自坐标系 |
| 视觉全家族低标签及 MIF 监督控制 | `visual_supervision_inventory_2026_09_19`，620 单元 | 原 25 配置 × 4 预算 × 5 种子 = 500；另 120 个 MIF 监督格 |
| 视觉/MPE 跨方法适配 | `policy_finetune_pilot_2026_09_20_r1`，35 新 + 35 Frozen 配对 | 视觉 B2；MPE B32；LAPO/LAOM 为既有 state-adapter 路线 |
| MTE 家族视觉/MaMuJoCo 适配 | `mte_finetune_extension_2026_09_20`，45 新 + 45 Frozen 配对 | 视觉 B2；MaMuJoCo B8、单个 decoder 种子；非路线 full-visibility MIF |
| 视觉分支贡献 | `visual_branch_attribution_2026_09_20`，100 单元，其中 50 新 | Simple/Graph/Tree/MIF/LAOM；四格均保留 Base 和辅助信息 |

其余已完成伙伴扰动、冻结 matching 和 agent 资源诊断同样保留在 17 项目录内。资源耗时或合成张量测试不替代更多 agent 的正式控制实验。

## 适配结果如何使用

视觉低标签下的监督适配是实质性补强：MIF-Solo 在 B1/B2/B4/B8 的均值均高于本轮 BC-batch256，对应差值约 +12.38/+7.41/+12.60/+10.40。这里 BC 与 MIF 的架构、训练路径不同；同架构的预训练/随机初始化四格回答另一项问题，不能用 BC 的比较替代它。

跨方法检验也保留了全部方向。视觉 B2 的 LAPO/LAOM 同样受益；MTE 的 Simple/Graph/Tree Aux 均改善，但 Solo 并非全正。MPE B32 的 Graph-Aux 和 LAOM 小幅下降、LAPO 小幅提高。MaMuJoCo B8 的 Graph 提高约 +10.31，原生 MIF 约 −0.99，Simple 约 −12.44。因此当前支持的是**视觉中的有效监督适配路线与明确的适用条件**，不能写成三类环境、所有 MTE 模型统一正增益。

视觉分支归因进一步显示，MIF 仅更新 Base、仅更新 MTE history 都取得正的平均增量（约 +36.15、+26.44）。两路一起更新相对仅更新 Base 的额外均值约 +1.43。MIF 在四种模式下相对同容量 LAOM 来源 history 的均值均为正，但跨来源比较也完整保留了 Graph 全更新略低于 LAOM 的结果。此实验能区分**来源信息、更新位置和容量**，选择性冻结本身不等于移除 MTE 信息。

上述统计以五个上游种子为配对单位：先在种子内平均预定预算/评价条件，再计算种子差值。精确双侧符号置换的最小 p 为 0.0625；t95、精确 p、Holm 和所有负向结果都保留在原报告及 JSON 中。均值优势按均值优势陈述。

## 复核范围与结果

本轮直接读取原档案，核验 17 项完成状态、来源及权重哈希、可用访问审计、唯一身份、逐控制器回报来源和视觉 27 条件聚合。从逐控制器结果独立重算八类后续研究的五种子差值及其 t95/精确 p/Holm；对监督四格、微调配对和选择性分支匹配重新读取原训练身份记录验证。

具体数量和报告哈希由 `provenance/current_scientific_state.json` 记录；明细见 `provenance/release_tools/current_state_audit_2026_09_20_r2.json`。这是已归档证据的完整性与统计复核，没有重训全部实验，也没有新增科学成绩。早期审计工具调试记录保留在本地，格式适配完成后的报告作为发布依据。

## 论文与未完成范围

当前论文快照仍为 `iclr2027_mpe_completed_revision_2026_09_18`，包含 9 月 19 日合入的 180 单元视觉低标签结果。620 完整清单、跨方法适配、MTE 适配扩展和分支归因是**已完成但尚未合入该 PDF**的证据。代码目录已全部收录；不能把代码发布完成说成论文更新完成。

Frozen 与 Adapt 的完整主实验/全消融预算矩阵尚未对齐：跨家族适配目前主要为视觉 B2、MPE B32、MaMuJoCo B8。真正新任务、更多 agent 的控制验证、实体对齐视觉 slot、官方结构基线，以及可部署 donor/history 改善仍是独立缺口。本轮没有启动这些实验，也没有自动更改论文、图表或 PDF。
