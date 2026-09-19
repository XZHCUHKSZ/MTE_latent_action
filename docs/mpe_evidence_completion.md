# MPE补齐：原始Edge-CARA、数据规模与机制诊断

用户于2026-09-18授权补齐前次审计中MPE相关的缺项。此协议在运行正式补测前固定；不会按胜负换种子、指标或预算。旧480组结果只读保留，论文暂不提前填写结果。

## X1：基础家族成员能否支持控制？

N700、原五个训练seed45–49、六档B8/16/32/64/128/256、原每配置100个评估episode。补8配置，共240行：Edge-h2与Edge-h3的Solo/Aux、Tree的Solo/Aux、Base+Simple-raw、Base+Edge-h3-root-only。

Edge-CARA不是新写的替代模型：直接调用冻结的 `common.train_representation`，保留原EdgeCARA网络、common/context分解、invariance及原损失。Tree同样直接调用原函数。环境adapter只把h2/h3预测端点送进已有接口，填零actions数组仅满足shape，不读取真实动作。原h1版本不改名冒充h2。

本轮Simple/Graph/MIF沿用已声明的route full-visibility目标；原Edge/Tree沿用自身原目标。因此家族排行榜不是等损失/等FLOPs消融。Edge多上下文与root-only调用同一原函数，原生固定训练设置/参数一致，但输入信息和归一化随上下文变化，解释为多上下文路线对照。Simple/raw使用同表、同初始化、同网络预算的现有route训练器。

本轮新增主结构比较仅两项：Simple-Aux减Simple-raw-Aux、Edge-h3-Aux减root-only-Aux。先在每seed内求六预算log2轴归一化AUC差，再跨五seed配对区间；这两项单独Holm校正。前轮三项结果不重算检验家族以改善显著性，也不并入本轮冒称预注册。

## X2：相同标签下，更多无标签观察是否有益？

固定B32和相同标签/验证索引，五种子，N32、175、700嵌套。新增前两个N，700复用本轮及补测同协议模型。

九配置：Base、LAPO、LAOM、Base+Global-Joint16、Edge-h2/h3-Aux、Simple/Graph/MIF-Aux。新增2×5×9=90行；合计新增330，复用480。BC原B32可作为固定直接监督参考，不声称其在每个新N重新训练或获得了N缩小后的归一化。

所有N使用相同frontend/readout/representation/history/decoder更新数3200/1200/3200/1200/1000，原batch256及route batch64。开发观察分区保持150集。因此它测的是固定开发观察可用量下增加训练观察，不能把N说成包括开发集在内的全部无标签访问量。

主规模对比为各配置N700−N32，九项单独Holm；N175用于观察曲线形状，不按结果选择主终点。不以某个家族成员在测试上最佳替换固定Graph/MIF等报告配置。

## X3：动作和伙伴信息能否进入history？

冻结后读取全部新家族与现有raw/partner/Global/LAPO/LAOM特征，区分transition-informed representation R与部署history H。Aux同时报告纯分支及拼接特征。

动作探针采用六预算相同fit/validation标签划分，测试为后75个开发episode；交互探针拟合最多4096个训练转移，前75个开发episode选ridge，后75测试。reader使用同一六档ridge候选，标准化只用拟合集。伙伴目标固定为共同预测表的单伙伴Möbius系数，按三个伙伴与h2/h3报告，不把各自自建目标用于横向排名。

这些是开发诊断：开发观察曾用于history checkpoint选择，不称全新untouched测试。动作/物理标签的reader拟合与原B预算控制学习分开记账，不回流更新policy。不同方法特征宽度如实记录；joint基线与Aux不是等宽对照。Edge/Tree原函数只返回编码，保留其完整R/H数组和history权重，不伪称保存了原函数未返回的representation权重。

## X4：真实训练donor的预测差分是否有物理含义？

在冻结后的独立进程复现bridge训练时的1024样本bank和最近历史规则；先核对前64个训练donor索引与归档完全一致。用64个新episode、每episode四个时刻，从同一快照仅替换目标第一步为donor原生动作，随后使用共同factual动作续接；三步都报告。原生joint_actions只在此评估进程读取。

比较：只替换目标code的matched预测、同时替换伙伴code的mismatched预测、零效应预测。random-donor使用自己的对应物理真值，不能错拿另一个donor效应制造劣势。保留h1物理零效应负对照，报告目标code跨历史迁移误差、伙伴code变化与history距离。不能将code swap认定为真实action intervention。

另列相同episode校准划分（前32拟合、中16验证、后16测试）的history-only、history+target-slot-pair、history+MTE reader；reader结果与直接物理单位误差分开，reader特征维数不同。这是语义诊断，不是部署时提供未来帧或模拟器接口。

## 调度、恢复与交付

新增入口 `experiments/mpe_evidence_completion.py`。四进程上限，顺序为大N补成员→冻结诊断→大N新增grounding→小N预训练→小Ngrounding。每阶段状态持续写入独立目录；每个seed、N、budget独占输出，完成结果不覆盖。旧checkpoint复制到新目录并校验；导出动作前须完成对应N的全部预训练。

运行开始固定科学源代码哈希；完成核对权重未变。失败保留日志，不自动换种子或覆盖不完整预训练。修复中途科学代码需新运行版本并记录，不能直接绕过哈希。

```text
python tests/check_temporal_completion.py
python -m experiments.mpe_evidence_completion --workspace WORKSPACE --out outputs/mpe_evidence_completion_2026_09_18
```

冒烟配置独立，不计性能证据。正式结束后以新协议整体更新MPE正文、表图、时间定义、结果来源；保留原协议归档。未完成前不能说论文已经全部修正。

本次范围不包括新增FLAM官方复现、新环境/新伙伴泛化、视觉低标签曲线、全高阶物理交互或全部历史29配置复刻。这些是此前列出的后续增强项，而不是本次MPE时间修正的必要全量重跑。单独隔离endpoint参数化或MIF每个传播轴仍需专门实验，当前补测不宣称一次证明所有组件。
