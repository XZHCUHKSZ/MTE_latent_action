# 视觉监督路径与完整家族低标签补齐

## 问题与预先固定设计
BC与MIF共享冻结RGB前端和PCA。BC有78978个接受动作梯度的history参数；MIF-Solo有18946个动作decoder参数与80784个冻结history参数。因此原比较同时涉及预训练表示、动作梯度能否更新history、网络装配与容量。参数计数本身不能解释回报差距，也不能据此判定BC对照无效。

本轮不缩小BC、不修改原MIF。Solo/Aux各采用相同现有RecurrentPolicy与ActionDecoder，交叉预训练/随机history初始化、冻结/动作微调。保留原预训练冻结结果，新增三格。预训练与随机的trainable格有完全相同的总参数和可训练参数；冻不冻结是待测实验因素。RGB/PCA和预训练导出的无动作normalizer共用，随机history控制不称完全无预训练。所有组共享decoder初值、动作标签、样本索引、优化器与600次更新；微调组梯度穿过完整200步history。正常BC和IDM仍是外部直接监督参照。

五固定上游seed908711–715；四预算B1/2/4/8；标签200/400/800/1600；每次监督更新256个目标动作样本。新6格×4预算×5seed=120个控制单元。冻结格为已完成证据。预算内的latent尺度取原预训练history输出，固定且四格共用。随机history不是新增独立前端训练seed。

## 其余家族覆盖
原25配置中9配置已有低标签结果；剩16配置在B1/2/4补240组，并复用B8。包含Edge、Simple、Graph、Tree、DeepSets、Random-edge、Continuous的Solo/Aux，以及重复Base和full-batch BC。后者沿用历史bc_full1600标识，低预算实际全批B×200，元数据显式修正，不能误写所有预算都是1600，也不并入等呈现次数主对照。

最终原家族500单元+监督控制120单元=620；360新增260复用。所有闭环仍30个共同条件，27个未用于旧开发的条件为主分析；不是新环境、更多agent或agent-aligned视觉slot实验。

## 门槛与统计
8格两更新smoke、40个冻结适配器逐权重/预测复现检查、80个缺失家族B8 decoder复现、80个原908601逐步动作/回报完全相同重放，全部通过后启动正式训练；全部新grounding冻结后才评价。共享初始化、归一化、标签与采样的哈希逐seed/预算/组合检查。审计限制每个训练worker只访问该预算动作标签，训练无模拟器调用。

先每seed内平均B1/2/4及27条件，Solo预定三比较：history微调收益、同架构trainable预训练减随机、初始化×微调差中之差，三项同一Holm族。Aux另一个三项族。五seed双侧精确符号置换最低p=.0625；完整报告方向、t95、精确p及Holm。B8和原家族逐格描述不择优改主模型。

## 执行
入口experiments/visual_supervision_inventory.py，协议configs/visual_supervision_inventory.json。最多四worker。PAUSE只停止新派发，存活任务自然结束；partial不自动覆盖或重试。新目录outputs/visual_supervision_inventory_2026_09_19。核心与完成结果只读，不自动改论文或上传。
