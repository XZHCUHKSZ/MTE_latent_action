# 强对照审计与后续实验（2026-09-19）

## 审计结论

审稿指出的强对照需要保留，但不能把不同证据层级合并为“整个MTE无用”。目前可以纠正两个过度推论，并定位一个真实实现适用性问题；不能据此宣称所有结构均有无可辩驳优势。

|对照|实际检验什么|核查发现|合理结论|
|---|---|---|---|
|zero-effect|冻结预测差分的真实物理效应MSE|所有旧原始/标准化指标复算及逐样本重放通过；h2不是纯零效应集合|对实际跨状态donor的负结果有效；不等于控制策略回报差，也不直接否定表示用途|
|Random-edge|MTE表的平均和随机投影能否替代学习式编码器|`common.py`先计算`context_root-context_ego`，再`en.mean(1) @ projection`；随后仍训练history/decoder|其强度限制复杂编码器的增量贡献；不是“不用MTE”的证据|
|自身raw|同一MTE表是否需要Möbius坐标|保留匹配端点差分，只将Möbius/zeta改为identity|检验坐标选择，不检验匹配构造是否有用|
|Base-duplicate|复制Base输入后下游参数化/优化能否解释收益|没有引入新表示信息，但可改变decoder优化|是有效简单控制，不能因其强就删除；也不相当于两套独立训练的局部表征|
|Independent/Global16|局部或全系统辅助表示能否提供替代控制收益|输入范围、目标与MTE路线不同；Global16是全系统分支，不是旧双局部分支|可作为竞争路线/容量参考；不能把不同输入和目标误称完全等价消融，也不能因此宣布无效|

## zero 对照核验与新诊断

固定原seed45–49、每seed64个原评价episode、每episode4个query、nearest/random两种donor，共2560对记录。未训练或改权重，未换donor，未筛选样本，未拟合缩放。

- 180个旧原始/标准化MSE重新核算一致；donor身份、truth、matched、mismatched与history逐条复现。
- donor动作索引和`t→t+2` code窗口独立重编码一致；h1负对照物理效应严格为零。
- h2目标效应在各seed全部query中均非零；只看target、只看非零效应或最高效应四分位，原跨状态donor仍差于zero。不能归咎于大量静止样本或伙伴零坐标把指标稀释。
- 新诊断只将同一donor动作在query状态执行并重新编码，固定原预测器与伙伴code。

|nearest donor|旧code差分MSE↓|query状态重编码MSE↓|zero MSE↓|
|---|---:|---:|---:|
|h1，全部agent坐标|0.00839212|0.0000938325|0|
|h2，全部agent坐标|0.03210072|0.0000480771|0.000523119|
|h3，全部agent坐标|0.06022960|0.000289537|0.00160205|
|h2，仅目标坐标|0.12605437|0.000152783|0.00209248|

h2重编码后全坐标误差比zero低90.8%，五个seed均更低；原跨状态donor约为zero的61.4倍，目标坐标约60.2倍。最高真实效应四分位仍约31.2倍。因此不能推翻原zero比较，但可以反驳“该结果已经证明匹配预测构造本身没有目标效应信息”。当前证据更具体地指向跨状态code的可输运性问题。

**重编码使用模拟器干预后的结果，是oracle诊断。它没有修复真实训练donor，不构成新的闭环收益、部署方法、跨任务确认或因果识别。** 旧负结果保留。共同评价episode不冒充独立训练重复。五seed双侧精确符号置换最小p=.0625；此项是机制定位，不作确认性显著宣称。

完整输出：`results/mpe_donor_zero_audit_2026_09_19/RESULTS_CN.md`、`VERIFICATION.json`；逐条数组在对应`outputs/`。

## Random-edge 的强度应如何解释

N700、六预算、五seed等权描述均值：Base −17.40037，Base-duplicate −17.31913，Random-edge-Aux −17.27199，Graph-Aux −17.26975，Tree-Aux −17.25700。

Random-edge-Aux保留MTE表，与Graph非常接近，这与“MTE表可以被简单编码器利用”相容；同时，它确实削弱“图编码器本身带来大的额外收益”的说法。没有matching与非matching的同资源控制前，不能把Random优于Base单独归因于matching，而忽略额外表示、history网络与decoder参数化。

## 已启动的同资源输入机制控制

`configs/mpe_matching_input_control.json` / `experiments/mpe_matching_input_control.py`。

- 固定5seed、N700/B32、100原共同评价episode；3臂×5seed=15闭环结果，不扩大预算寻找正结果。
- 固定同一冻结预测器与h3端点表、同一现有Graph类、16输入坐标、16维latent、同初始化/端点归一化/采样/训练步数，复用原history和grounding实现。
- 三臂：`[A,A-B]`、`[A,A-B_complement]`、`[A,B]`；全表均可逆，信息量相同。
- 所有臂共享端点对`[A,B]`重建目标，避免matched独享差分监督。
- 两项主比较：matched减context_mismatch、matched减endpoints；全部保留，五seed配对，t95/精确p/Holm。
- 所有上游冻结后才导出B32标签。预训练guard、未来观测因果性检查、初始化/参数/目标/归一化/采样哈希一致性均在正式流程核验。
- 三臂2更新完整冒烟已通过，包括grounding、两episode闭环和source/weight检查。冒烟仅验证实现与接口。

这是**匹配差分作为输入归纳偏置**的控制，不是原主实验edge-only Graph8的复现，也不是无MTE全流程基线：端点表本身仍来自冻结预测器和原重组规则。新名称和独立结果目录保留这一区别，不替换论文原模型数值。

## 尚需解决

1. 实际可部署donor的支持/输运问题：本轮定位未修复。任何后续支持约束或生成式替代需独立协议，并复测控制，不能直接拿oracle作为原方法优势。
2. 伙伴政策扰动已完成，但真正未开发任务确认、agent数扩展、官方结构化基线仍未完成。不能把这15单元称为解决这些问题。
3. 视觉低标签适配和协议草稿已存在，但尚未通过复现门槛或正式启动。本轮优先完成matching控制，避免同时启动大量新协议。

论文主线、教授前文、图表和旧结果均未自动修改；也未上传GitHub。


## 正式完成补记（2026-09-19 16:36）

上述15组控制已全部完成并验收。matched对直接端点均值+0.01137（3/5正向），对context_mismatch为−0.01061（1/5正向）；两项Holm p均.625。本轮不能作为匹配结构一致优越的确认。全部结果、范围与验收见`results/mpe_matching_input_control_2026_09_19/RESULTS_CN.md`和`VERIFICATION.json`。论文未自动更新。
