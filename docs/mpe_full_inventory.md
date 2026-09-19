# MPE全清单补测：不再遗漏Tree及其他正文/附录模型

2026-09-18用户授权。入口 `experiments/mpe_inventory_completion.py`，新输出 `outputs/mpe_inventory_completion_2026_09_18`。此前两轮已经完成的结果只读复用；不改冻结核心、已完成结果或旧快照。

## 覆盖标准

以 `configs/numeric.json` 的原论文29个配置标签逐项核对，不用“主要模型”代替全清单。entity_target和anchor_only对应同一个Base，合并为28个唯一配置；另外保留当前Global-Joint16-Aux路线对照，共29个唯一配置。别名在 `configs/mpe_inventory_completion.json:paper_aliases` 显式登记。

- 完整N700：每配置5个seed × B8/16/32/64/128/256。
- 固定B32规模曲线：每配置5个seed × N32/175/350/700，N700复用完整矩阵的B32，不重复计入样本。
- 全部配置合计1,305个唯一单元；已有600个可复用，新增705个。既有raw/complement/root-only等结构诊断仍独立保留，不混为全清单主比较或删除。
- Tree-Solo与Tree-Aux同样覆盖全部四个N、五个seed；不再只报告它的N700。
- 仍按seed为统计重复单位，100个共享评估episode先在seed内汇总。扩展清单属于开发覆盖，不重新包装成未见数据确认。

## 原清单的适配对应

|原标签|时间修正版|
|---|---|
|continuous_lam / anchor_continuous|continuous / base+continuous|
|lapo_state_adapter / laom_state_adapter|lapo_state / laom_state|
|lapo_joint / laom_joint_k3|lapo / laom（先前已完成的时间适配）|
|entity_joint|四个既有entity code拼接后训练history|
|entity_target / anchor_only|同一个base，结果复用，不双计|
|anchor_duplicate|同一个冻结Base输出复制，非独立训练双分支|
|anchor_laom|base+laom_state|
|edge_cara / anchor_edge_h1|edge_h2 / base+edge_h2（显式时间修正，不把旧h1结果改名）|
|edge_cara_h3 / anchor_edge_h3|edge_h3 / base+edge_h3|
|mobius_simple/graph/tree、mif及各anchor版|simple/graph/tree/mif及对应base+版本|
|matched_cf_deepsets / anchor_matched|deepsets / base+deepsets|
|random_edge_encoder / anchor_random_edge|random / base+random|
|bc_scarce / idm_relabel|bc / idm|

## 模型与时序边界

Simple/Graph/MIF沿用已声明的route目标；Edge/Tree/DeepSets/Random直接调用冻结的 `common.train_representation`。所有修正版使用同一套观察训练的frontend和直接h1/h2/h3预测端点。DeepSets/Random采用h3，与Simple/Graph一致；这不是原h1基线的逐位复现。

Continuous调用原冻结函数，adapter只把预测目标设为t+2。State-LAPO/LAOM保留原类、128 hidden、原latent尺寸、原目标、优化器和matched schedule；显式窗口改为LAPO[t−1,t,t+2]、LAOM[t,t+2]且重构t+2。它们与已经运行的joint frontend在容量/训练配置上不同，不将两套结果说成相同模型的重复证据。均为本地适配，不是作者官方benchmark复现。

IDM使用原InverseDynamics与原relabel/BC训练流程，仅把未来输入从t+1改为t+2；原预算1000个IDM更新、1200个pseudo-policy更新。原生动作只能在冻结后的grounding进程读取；验证动作包含在B内。IDM确实属于动作监督grounding基线，不能称其所有训练都observation-only。

BC以相同B动作训练，观察标准化使用对应N的无标签观察。因而N改变时其统计量也可能改变；这次按每个N实际重跑BC，不伪称是同一个恒定点。

## 诊断与统计

N700补齐全部latent模型的representation/history动作和单伙伴系数探针，沿用前轮分区、ridge选择和开发诊断范围。BC/IDM没有无动作latent预训练分支，因此不伪造其R层探针；它们的动作验证误差、原生回报仍在全部预算/N下给出。

物理donor评估依赖共同冻结frontend/bridge，N700未改变，复用上一轮全部五种子结果。不为每个仅改变压缩器的模型重复模拟器分支并当成新独立证据。

新增大矩阵完整报告，不在看完结果后挑一个预算/成员开展显著性宣称。原先三项与后续两项预定结构检验保持原结果、原检验家族。未经预定的28配置排行和规模差异首先作为描述性覆盖；不以增加配置来消除旧阴性结果。

## 执行与验收

1. 核对原29标签无遗漏、别名及时间适配有记录。
2. 复用checkpoint先验哈希、复制到独立新目录；不读取旧grounding标签进入预训练。
3. 四路预训练，所有N全部预训练冻结后再开放各B动作。
4. 新成员动作/交互探针、缺失的grounding及评估自动接续；每N/seed/B独立写入。
5. 完成需1,305/1,305身份唯一、705新增、600复用、全权重与代码指纹通过，且五个N700 probes均完成。
6. 再更新论文图表与复现映射；未完成前不称论文已全部同步。

这轮解决MPE模型/预算/规模清单完整性，不承诺所有方法胜出，不包括新伙伴任务、官方FLAM复现或视觉/其他环境的新增实验。

## 2026-09-18论文同步交付

完整矩阵已同步至 `paper/iclr2027_mpe_completed_revision_2026_09_18/main.tex`
（仓库根目录相对路径），以用户最后上传的单文件稿为底稿。正文9页、全文30页；
更新MPE设置、回报/规模/动作/交互表、五项结构配对、同状态与训练donor物理效应，
并修正主图的旧一步标签。MaMuJoCo/视觉结果未重跑。
该目录的 `EVIDENCE_MAP.md` 和 `修改说明与缺项核对_CN.md` 给出来源及未解决问题。
交付ZIP为 `paper/MTE_ICLR2027_MPE_Complete_30Pages_2026_09_18.zip`，31个文件。
按用户要求恢复原版六面板结果图及四面板资源曲线，MPE均使用当前结果；
正文先展示完整MTE分支对Global-Joint16的控制增益，再报告单操作配对及其不确定性。
这是论文同步，不改变本清单的科学协议、完成结果或原有统计检验家族。
