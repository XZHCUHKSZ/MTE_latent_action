# 视觉B2的分支适配归因与同容量非MTE对照

用户2026-09-20授权轻量检验：视觉Aux组合的监督适配收益来自Base、MTE辅助history，还是增加一条可训练分支本身？保留已有MTE主线和全部冻结证据，不预设结论。

## 设计

沿用已检查的五个上游种子908711–715、B2（400个二维目标动作向量）、600更新、Adam 0.001、每步256原采样索引、冻结RGB/PCA前端。评价为原30个共同条件，908604–630的27个条件用于主统计。评价回报不用于模型选择。

五个辅助history来源：Simple、Graph、Tree、MIF、LAOM-state-adapter。每个来源均为16维、80784个history参数；共用32维Base history（visual_laom_target，82848参数）及相同48维输入的decoder。跨来源decoder初始化、标签、采样、观察输入和架构参数量逐项核验。LAPO为128维，不纳入“同容量”控制。

每个来源四格：

|格|Base history|辅助history|decoder|来源|
|---|---|---|---|---|
|frozen|冻结|冻结|训练|旧结果复用|
|baseonly|训练|冻结|训练|新增|
|auxonly|冻结|训练|训练|新增|
|trainable|训练|训练|训练|旧结果复用|

共100格=50新增+50复用，不扩预算、不增训练数据、不重训前端。MIF仍为原锁定主模型；Simple/Graph/Tree为家族跟进，五个来源和全部方向一并保留。

## 能区分什么

1. `baseonly−frozen`：保留原辅助信息时，适配Base的增量。
2. `auxonly−frozen`：Base保持固定时，适配辅助history的增量。
3. `trainable−baseonly`：已适配Base后，额外适配辅助history的增量。
4. `trainable−auxonly`：已适配辅助history后，额外适配Base的增量。
5. `trainable−baseonly−auxonly+frozen`：两支同时适配相对单支效应相加的交互项。

每种MTE在四格中均减去相应LAOM格，比较相同容量、相同更新范围下MTE来源history的相对控制价值。这区别于“只要多一条分支就行”的解释。LAOM是有独立预训练信息的有效控制，不使用故意弱化的空白/零向量作为唯一对照。

选择性冻结不删除分支信息：若baseonly改善，不能据此断言MTE信息无用。跨方法保持同一种无动作归一化算法，但latent统计数值依其原表示各自计算；同一方法的四格数值完全一致。Base和辅助分支的输出维度/参数量本来不同，单支更新之间不是精确同参数比较；MTE与LAOM对应格的架构与可训练参数量相同。

这检验MTE来源history与一个具体非MTE来源的相对价值，不单独证明匹配构造、Möbius编码、所有MTE组成或通用因果机制。原MTE、LAOM各自预训练目标和轨迹保持原样。LAOM-state-adapter不是官方端到端像素LAOM复现。

## 数值门槛与信息边界

独立适配器`training/visual_branch_attribution.py`来源于已验证的`visual_supervision_control`；只扩展分支梯度开关，不改变共同科学类、loss、优化器、批次或训练步数。冻结分支逐份比较更新前后完整权重指纹，必须完全相同；解冻分支须发生更新。所有格均使用因果history推理，逐步和整序列预测误差≤1e-5。

原无动作预训练和所有历史文件只读。新增grounding只读B2目标动作、不读伙伴动作、不查询模拟器。先10条真实选择性更新smoke与原环境单回合；50个旧冻结/全微调600更新数值复现（decoder/fit maxabs≤1e-6、全微调history逐张量相等）和50原908601逐步动作/奖励完全重放均通过后，才派发50个正式grounding。25个四格配对与五种子的跨方法同容量检查通过，冻结50份新权重后再评价。

最多四路。失败停止派发、保留partial；不自动重试或覆盖。源manifest、所有来源权重、访问审计、foundation verifier均须通过。输出`outputs/visual_branch_attribution_2026_09_20`和`results/visual_branch_attribution_2026_09_20/RESULTS_CN.md`。论文图表不自动改写、不上传GitHub。

## 统计与研究阶段

本轮是在看到既有微调结果后发起的机制探索，不包装成全新确认实验。先每上游seed平均27条件，再五seed配对。五个分支比较按每来源各一个Holm族；四种MTE×四格对LAOM的16比较构成另一个Holm族。保留均值、五seed差值、正向种子数、t95、精确双侧符号置换p及Holm p。五seed精确p最低0.0625。不根据中途结果减少模型、替换对照、扩种子或改变论文主线。
