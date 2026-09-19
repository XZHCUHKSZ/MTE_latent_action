# 审查建议核对与补实验方案

日期：2026-09-17。状态：供用户审阅的方案；未启动训练、未修改论文或已有结果，未上传本文件到 GitHub。

## 1. 判断依据

核对对象为用户提供的 12 张审查截图、当前 `paper/iclr2027_control_ablation_figures_2026_09_17/main.tex`、本整理包的配置、入口及结果记录。审查截图未附其实际审阅 PDF，不能假定其所有引文都来自当前版本。

当前稿已有三个与截图不同之处：

- 正文已报告 Global-Joint16 对照：MPE 中 MIF/Graph/Simple 的平均回报差为 +0.228/+0.178/+0.151；MaMuJoCo 五次 decoder 重复汇总中 MIF 为 696.99，对照 688.51。后者是 +8.48，不能与三次共同重复的另一汇总混用。
- 这一比较每个环境只有两个上游种子。decoder 重复是嵌套重复，不是五个独立表示训练种子。入口 `experiments/route_ablation.py` 也明确标为 development experiment。
- 当前稿没有“family 是看完结果后才提出”的原断言。它区分家族矩阵的描述性结果与未做过的 held-out 模型选择检验。不能把“不同成员有不同优势”推断为“家族是事后创造的”；具体诊断实验的事后开发性质则须保留。

Global-Joint16 是完整学习路线对照。MTE 路线使用 full-visibility reconstruction 与 variance/covariance penalties，区别于正式主矩阵的 masked objective。正向结果支持该路线配置；单靠这组结果尚不能隔离相减、Möbius 或某一种图传播。

## 2. 逐项提取与优先级

| 截图 | 建议/质疑 | 判断与处理 | 优先级 |
|---|---|---|---|
| 1 | 与 FLAM/OTF-LAM/object-centric 方法区分；COMA 类比；避免物理因果措辞 | 合理。当前附录已有 FLAM/OTF 引用，补的是机制区分与可行基线，不是从零补引用。COMA 仅作概念比较，非同监督条件基线。 | P1；术语 P0 |
| 2 | 理论主要支持 Simple/Graph，强控制实例主要是 MIF | 合理。统一为“共同 MTE 算子—编码方式—历史迁移—控制”的分层验证；不要求每个家族成员都由同一机制获益。 | P0 |
| 3 | 已知目标 slot 的收益与 MTE 收益混在一起 | 核心问题成立；所引旧对照不是当前主比较。要增加同 slot、同预测表内的机制消融。 | P0 |
| 4 | 个别结构增益尚未确定 | 当前新路线对照提供正向证据，但没有取代原资源控制的研究问题。分别报告。 | P0 |
| 5 | family 被事后选出，必须固定一个算法 | 前半句不能照收。保留家族贡献，预先锁定新验证中的主配置和次配置，不按新测试结果更换代表。 | P0/P1 |
| 6 | 视觉 camera slot 不等于 agent slot | 合理，而且当前第 2 节已说明。低标签曲线不能解决身份对应问题。若强调视觉 agent 归因，需要额外验证或新视觉分组方案。 | P1 |
| 7 | 实验按四个主张组织 | 合理，可直接改组织方式：目标效应、伙伴上下文、历史可用性、控制收益。保留现有数据，不扩写大量附录。 | P0 |
| 8 | 仅评估时测真实 target-action effect | 最高价值建议。保持训练隔离，先验证模拟器恢复与效应对齐，再测模型。不是把旧有 simulator-MTE 监督重新放回预训练。 | P0 |
| 9 | 新任务/新伙伴验证；加入结构化基线和 LAOM+ | 方向合理，不必一次完成全部四种迁移。FLAM 优先；OTF 可行性需查完整接口；LAOM+ 是有动作监督的独立轨道。 | P1 |
| 10 | 八层消融与规模扩展 | 不宜一次堆成巨型矩阵。拆成少数单因素配对，标出不同信息与计算的变化。大规模 agent 扩展排后。 | 消融 P0；扩展 P2 |
| 11 | 视觉低标签预算、监督微调 | 合理。25% 不能充分展示低标签区间；微调只在预训练冻结存档后进入独立 grounding 轨道。 | P1 |
| 12 | 围绕单一算子、固定方法、真实效应和 held-out 验证重组 | 主方向可采纳，不接受它把所有当前证据视为无效，也不能把主会评价当作客观评分。 | P0/P1 |

## 3. 第一优先：真实动作效应的独立评估

### 要回答的问题

在已知 slot 条件相同的情况下，MTE 是否比原 slot code、普通差分和 joint latent 更准确地表达“只改变目标 agent 动作造成的系统变化”？这检验预测对比的语义，不再仅检查能否读出模型自己生成的系数。

### 冻结与信息边界

1. 对既有 observation-only checkpoints、预处理、donor 规则和输入接口固定版本。所有预训练与 history 训练结束后，另启评估进程。
2. 评估器可读动作、完整模拟器状态并执行分支；训练器不可读这些文件。评估产生的分支 RGB/状态轨迹同样禁止回流训练。
3. 先用 development 场景调试恢复精度、时间索引和测量单位。调试数据不作为确认结果。冻结测量方案后，用新 episode seeds 评估。
4. 既有 checkpoint 上得到的是冻结模型的外部语义诊断，不自动变成全新训练种子的确认实验。

### 真实参照如何构造

从同一完整快照出发，固定伙伴当前动作、随机数/外部扰动及第一步之后的记录动作序列。只替换目标动作，得到两个真实端点，并相减。定义 h=1 为主时间尺度，h=2、3 为次级，投影到预先固定的同一 outcome 坐标。

MaMuJoCo 原 `closed_loop_lam_v1/mamujoco_env.py` 已有 snapshot/restore；整理包 MPE `environments/native_particle.py` 也有。可以在新评估适配器中复用，但要先做“相同快照+相同动作重复 rollout 一致”的检查。不能修改被冻结的原始实现。

两个重要约束：

- 目标动作可以影响别的 agent 或整个身体。正确目标不是“只有目标坐标变化”，而是“所有变化的起因仅为目标动作替换”。误伤这一点会把真实交互当成泄漏。
- 动作在模拟器中是明确的，但 latent code 不一定唯一对应动作。不能直接认定 donor code 等于 donor 的原生动作。正式 donor 规则的 code swap 与真实 donor-action swap 应配对测量，并把失配作为结果。

先设两个分开的诊断：A 为同状态下的可控 paired transitions，用于检查 slot 混合和 code replacement；B 使用原方法按相近历史选择的 donor，检查实际构造在跨状态情况下的语义。A 属于额外评估输入，不能称为标准部署，也不能用它替换 B 的正式 donor 结果。

### 对照与指标

- 原 target-slot 表示、joint LAPO/LAOM 表示：通过相同容量的评估 reader 预测真实效应；输入是同一组观察和有向转移对，禁止只给其中一组原生动作或完整状态。
- MTE 原预测差分：若预测器输出与真实 outcome 同坐标，先直接计算误差，不为 MTE 单独拟合校准器。
- MTE、普通未匹配差分、random-donor 差分：分别记录更改了伙伴匹配还是 donor 选择；不能再次用打乱整张 history–target 对应来冒充 matching-only 消融。
- 若要跨 latent 空间比较，另列统一 reader 诊断。reader 只在独立评估 calibration split 拟合，在未见 episode 测试；不更新 policy，不用于挑选主模型。该结果标为诊断监督，不称为无监督训练指标。
- 主要指标为固定尺度的 outcome effect MSE；尺度只从允许的校准/开发分区确定。报告每个环境、每个训练 seed 的配对差和区间。次指标为方向一致性、按实体分组误差、同状态不同伙伴上下文下的误差。近零效应需单列，不靠余弦相似度虚增得分。
- 同时报告零效应/context-only reader 基线、单端点预测误差及效应大小，避免大量微小效应导致“全预测零”看起来很好。

### 针对 Möbius 的额外价值

在同一快照下组成目标动作与一个伙伴动作的四种组合，以四个真实端点的二阶差分作为 target–partner interaction 参照。与当前 learned coefficient probe 区别在于，这次参照来自独立物理分支。先做单伙伴，不立刻扩张全高阶组合。固定同一方向与时间尺度，分别报告 Edge、Simple、Graph、MIF 的保留能力。

history policy 无法仅凭同一个历史预测评估器任意指定的未来动作对。因此上述 paired-effect 检查首先针对 transition-informed representation；部署层另测自然任务上的真实动作读出与控制，不把需要未来分支的信息要求强加给 history policy。

## 4. 第二优先：少量但能归因的结构消融

固定 slot frontend、预测表、train/dev/test episodes、输出 16-D、Base、history network、grounding labels 和 decoder。建议包含以下配对，而不是把所有方法排成一条复杂度阶梯：

| 配对 | 固定什么 | 回答什么 |
|---|---|---|
| 已知 target-slot latent vs MTE route | slot 定义、观察权限、标签与部署接口 | 已知 slot 之外，完整路线有多少额外收益；属于路线级比较 |
| 匹配端点 pair vs endpoint+edge | 同一两个端点、网络与训练目标/预算 | 可逆参数化是否使有限容量模型更容易利用对比 |
| 正确 partner matching vs partner-mismatched difference | 相同 history、目标 codes、donor 池与预测器；只改变一个端点的伙伴 codes | 固定伙伴的价值；避免破坏整条 history 对齐 |
| 单一伙伴上下文 vs 多上下文表 | frontend 与最终网络接口；单独记录真实计算差 | 多伙伴设置是否补充有用信息 |
| raw edge table vs Möbius coordinates | 完全相同表信息、token 数、encoder、loss、updates | subset 坐标组织的价值 |
| structured vs pooled/unstructured encoder | 同一输入表、latent 宽度、可训练参数预算与训练预算 | 编码结构的价值 |

pair 与 endpoint+edge、raw 与完整 Möbius 坐标都是可逆变换。若有优势，解释为有限模型/优化的归纳偏置，不能写成增加信息量。单上下文与多上下文则不是等信息比较，应明确标出，而非笼统宣称所有消融等信息。

训练 loss 必须同配对一致，尤其不能把新 full-visibility loss 的正向结果直接归因于老正式 masked objective 的 MTE。多任务目标的改善应另设配对。

不把旧 Base+独立局部分支恢复为本轮主实验。它检验增加局部表示分支的竞争性，不是匹配/差分的直接消融；原有有效记录仍保留来源，不宣布数值无效。现在优先回答同 slot、同表内机制，以及 Base+Global-Joint16 路线比较。

## 5. 家族、主配置和确认实验

保留 Edge、Simple、Graph、MIF 的家族定义和既有完整结果。新实验中建议 Graph 为机制主配置（其设计最直接对应 subset/伙伴结构），MIF 为固定的完整控制配置，Simple 为结构简化对照，Edge 为单差分对照。此选择在新结果出现之前写入协议；不可做完后换最好者当唯一主方法。

路线级确认至少采用五个独立 upstream seeds；每个 seed 共用相同数据划分和配对条件。多预算和 decoder 重复在 seed 内汇总，不能当独立样本。是否沿用原五种子或采用新五种子，按问题区分：旧种子复验是扩展既有证据，新种子才是新增训练随机性的确认。

默认建议先在 MPE 与 MaMuJoCo 各冻结一个预先指定预算进行关键配对；再按固定完整预算网格出曲线。预算在看新结果前选定，依据低标签定位，不选历史上最好看的点。控制主指标保持原生 return，action probe 和真实效应误差分别回答信息与语义问题。

确认测试的主要对照和检验数量先固定。对主要配对报告 effect size、paired interval 与预先选定的检验，多比较采用 Holm。五种子并不保证显著，更不能以“没过零”为理由不断加 seed 到显著为止；事先固定样本数/停止规则。环境原生回报不同，不把 MPE 与 MaMuJoCo 原始分数直接平均。

新伙伴策略泛化优先于立即换整个环境：固定伙伴数量、target-action space 和观察接口，预先规定伙伴策略变化强度，冻结 target policy，不做测试反馈调参；同时报告参考策略在这些条件下的难度。若更换实体数导致维度/结构变化，则属于额外适配实验，不冒充零样本泛化。

## 6. 视觉 BC、低标签与语义

BC 强不是代码错误的证据，也不能称为理论上限。LAOM 原文说明直接监督和 IDM 在 distractor 条件下可以很强，并另用表示阶段的动作监督提高表现；这支持把监督使用方式分开研究，不证明本方法已优于 BC。

推荐复用冻结视觉表征，先扩充 1%、2.5%、5%、10%、25% 的唯一标注 transitions 预算。现有预算以完整 trajectory 为单位时，小于一条 episode 的比例需要明确新增 transition/segment 采样协议；绝不能把每个 frame stack、增强次数或 epoch 数算作新的标签。所有方法同一标签索引，预处理与验证动作的额外权限也要计账，episode-level test split 不变。

同协议比较 MIF-Aux、MIF-Solo、Graph、LAPO、LAOM、Feature BC、IDM；表征可复用但 decoder 按各预算重训。标签集嵌套，保留 25% 旧点。低预算有望更清楚展示预训练用途，不能预言一定反超 BC。

新增 supervised grounding fine-tune 作为第二轨道：无动作预训练完成并保存后，允许同样少量标签调整 history features，所有方法获得相同权限与调参预算。它仍可称 observation-only pretraining，但不再属于原先 frozen-history 协议。LAOM+ 使用动作监督学习表示，应单独标明，不能混称 observation-only baseline。

camera slot 的语义缺口须独立处理：当前视觉结果继续表述为 RGB 控制实例。若做 object/limb slot，优先从 RGB 无监督获得；不能把模拟器 limb mask、真实分割或跟踪 ID 偷渡入训练。模拟器 limb identity 可以只用于冻结后的评估。低标签曲线不能替代这项身份验证。

## 7. 文献与新基线

- FLAM：正式论文和官方代码已找到，优先做接口/预算可行性审计。它学习因子化状态与动作，可帮助回答“分解本身是否足够”。本地 known-slot 对照不能改名为原版 FLAM。若做适配，明确保留/改变哪些组件，并记录参数与训练成本。
- OTF-LAM：原文提出 observed-transition primitives；适合讨论 agent ambiguity 下的观测变化混合。需要继续核实代码、视觉接口与数据要求，不能凭摘要宣布已能直接接入数值环境。
- object-centric：是方法类别，需选定具体可复现模型，不能只在表中造一个“object-centric baseline”。
- COMA：固定其他 agent 动作的类比有助于解释对比操作；它使用 centralized critic、动作与奖励，不能算本 observation-only 条件下的直接竞争方法。

一轮不必同时复刻所有新方法。先补 FLAM 或明确声明的 factorized-LAM 适配，再决定 OTF。LAPO/LAOM 原基线始终保留。

来源（均为原论文或作者官方代码）：

- FLAM：https://arxiv.org/html/2602.16229v2
- FLAM code：https://github.com/wangzizhao/flam
- OTF-LAM：https://arxiv.org/html/2606.30544v1
- LAOM：https://arxiv.org/html/2502.00379v5
- COMA：https://arxiv.org/abs/1705.08926

## 8. 执行顺序与交付门槛

1. P0 审计：核对最新版声明、模型输入、loss、upstream seeds、checkpoint 与结果来源；冻结新方案。先不改已有论文结果。
2. P0 评估冒烟：利用已有 snapshots 接口，先 MPE 后 MaMuJoCo；验证相同动作重放、action ordering、horizon/projection 与随机状态。先出真实效应和 slot 混合诊断，不启动全模型重训。
3. P0 两环境结构配对：先用开发种子验证实现/数值稳定；正式矩阵与样本数随后冻结，成功与失败均交付。主要模型 Graph、MIF，配套 Simple、Edge、LAPO、LAOM 与 Base。
4. P1 预先固定方法的确认：五个独立 upstream seeds；加入一套固定的新伙伴条件。不要用 decoder 重复数充当独立 seed 数。
5. P1 视觉标签曲线及 FLAM 适配可行性；优先复用表征，只重训必要模块。
6. P2 scalability：先实测已有 2^p 表的生成时间、内存，再在支持的实体数上考虑低阶截断/采样。不能以合成表运行快声称大规模环境控制已验证。

时长在 snapshot 冒烟和一个训练单元完成后按吞吐估算，分开报告端点生成、representation、history、grounding、rollout 成本；当前不承诺若干小时内全部结束。

代码改动在本整理包下新增 evaluation/experiment/config 文件，结果另建 run 路径；修改前后跑 frozen-foundation verifier，不改原始核心或已完成结果。开发协议、正式协议与图表来源保持一一对应。正文仍按 9 页、全稿约 30 页规划：用目标效应图和机制—控制配对图替换重复呈现，避免再靠不断加附录回应质疑。

## 9. 什么结果足以回答什么主张

- 真实效应更准：支持 target-oriented 语义，不自动支持更高控制 return。
- 同信息 raw/Möbius 或 endpoint/edge 配对获益：支持特定结构的归纳偏置价值。
- 同 slot 条件下 MTE 的 history 表示/控制更好：减少“仅来自已知 slot”的解释空间。
- 新伙伴与新 seeds 下固定配置有收益：支持复现性和这一范围的迁移，仍不是所有环境普遍最优。
- 上述层次方向不一致：分别定位 representation、history transfer 与 decoder 的瓶颈，不通过更换指标掩盖。

优先补的是“独立物理参照 + 同 slot 的结构配对 + 固定方法的确认”。这比重跑一轮全家族回报排行榜，更直接回答此次审查。
