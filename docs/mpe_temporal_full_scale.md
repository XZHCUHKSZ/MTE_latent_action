# MPE 修正后恢复规模：五种子、六档标签预算

本轮由用户于 2026-09-17 授权，目的为检验小规模时间修正与结构消融信号能否在原数据/更新/评估规模重复。不是为了搜索到正结果；不依据中途结果改变种子、预算、方法或停止时间。旧结果、旧交付包和冻结核心均保留。

## 固定规模

|项目|本轮|
|---|---|
|训练随机种子|45、46、47、48、49；与开发轮40、41分开报告|
|数据来源|依次使用原seed40–44的五份数据及原训练/验证划分|
|每份数据|700训练episode，150验证episode|
|标签预算|8、16、32、64、128、256条训练轨迹；各预算含1/4验证轨迹|
|frontend / readout updates|3200 / 1200|
|family representation / history updates|3200 / 1200|
|action decoder / BC updates|1000 / 3200|
|批量|frontend/readout 256；family route64，沿用原route实验，而非formal masked MIF批量|
|control configs|16个，清单同小规模修正版|
|控制评估|每配置100个相同的新episode种子，23步，保留原reward与warmup|
|物理诊断|每模型种子64个新episode ×4时刻；h2主检验，h1/h3同时报告|
|组合数|5×6×16=480组控制结果，合计48,000条控制评估episode|
|并行|预训练最多4进程；grounding按seed×budget拆成30任务、最多4进程|

这是五个新的训练初始化，不是五份全新未见数据或新任务。之前已经使用过这些环境和数据作开发，不能称完全untouched的确认实验。

## 固定方法，不混协议

使用已验证的 `mte/temporal_mpe.py` 和 `training/temporal_family_mpe.py`：位置观测-only、inverse/feature target t+2、直接预测物理h1/h2/h3；原类不改；MTE两端共享history和partner assignments。LAPO/LAOM同样时间适配，明确是本地适配。

Simple/Graph/MIF维持已归档route的full-visibility reconstruction+variance/covariance目标，不因恢复3200updates就改称formal masked MIF。原offset1对照也使用同一个新的direct readout，仅隔离窗口作用，不冒充旧递归协议原样复现。

完整配置：Base、LAPO、LAOM、Simple/Graph/MIF Solo与Aux、Graph-raw-Aux、MIF-raw-Aux、Graph-partner-complement-Aux、Global-Joint16-Aux、旧窗口Base与Graph-Aux、BC。恢复的是主要规模，不是把所有历史29种配置、Tree/IDM等同时重新启动。

paired matched/raw共享可训练初始化、架构、数据和训练预算；原生基线间维度、结构和共同frontend/readout成本不同，不声称整个矩阵等参数/FLOPs。Base/BC和不利结果保留。

## 分析在运行前固定

主比较为 Graph-Aux−Graph-raw-Aux、MIF-Aux−MIF-raw-Aux、Graph-Aux−Graph-partner-complement-Aux。每个训练seed先将回报差在log2预算轴上梯形积分，除以log2(256)−log2(8)，得到一个平均曲线差。再跨五个训练seed作配对汇总，报告差值、t型95%区间和三项Holm校正。t推断依赖seed差值的分布假设；n=5较小，不将预算或100个评估episode充当额外训练重复。

所有预算的回报、碰撞、覆盖距离与预算内动作验证误差单独公开。中途表格仅描述性，不对未完成矩阵滚动宣称显著。物理效应的matched/mismatched/zero预测误差单列，不和控制回报混为同一成功标准，也不替代nearest-history-donor的完整物理验证。

## 隔离、调度与恢复

先导出positions-only，全部五个seed预训练完成、记录global_freeze后才导出动作。每个grounding任务独占`seedXX/grounding/budget_B/`，不存在共享结果文件写竞争。统计由单个manager每10秒汇总，并维护`status.json`。全部模型/decoder只按原validation规则选checkpoint，不用最终rollout选模型。

实施文件指纹与冻结权重均校验。运行失败保留日志，不悄悄换seed。完整预训练阶段可跳过、grounding可按已完成配置恢复；未完成预训练需先审查失败，调度器拒绝无提示覆盖。科学源代码变动后拒绝在原目录续跑。

```text
python tests/check_temporal_scale.py
python -m experiments.mpe_temporal_scale --workspace WORKSPACE --out outputs/mpe_temporal_full_scale_5seeds
```

输出看`status.json`、`summary.json`、`RESULTS_CN.md`及每任务日志。本轮不自动修改论文数值或公开GitHub内容；需等完整结果审查后再形成论文更新。
