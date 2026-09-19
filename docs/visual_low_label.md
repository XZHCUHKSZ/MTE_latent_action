# X3b：减少动作标签时，冻结视觉MTE路线是否仍有控制收益？

本轮扩展原视觉实验的标签曲线，保持32条无标签训练轨迹、原五个训练种子908711–908715、四智能体Ant、原RGB前端与冻结history模型。不是新环境、更多agent、视觉实体分割或重新预训练。

## 固定比较

MIF-Solo/Aux、LAPO-Solo/Aux、LAOM-Solo/Aux、Base、feature BC batch256、IDM，共9配置。前三者中的LAPO/LAOM沿用论文feature-space适配，不称官方完整pixel复现。

标签为相同训练轨迹前缀B=1/2/4/8，每条200个目标动作：200/400/800/1600个动作标签，占32条训练轨迹3.125%/6.25%/12.5%/25%。新增前三档135控制单元，原B8的45单元核验后复用。伙伴动作标签不读。相同预算方法共用标签集；没有以回报选择轨迹。

主模型锁定MIF-Solo；主比较为对Base、LAPO-Solo、LAOM-Solo、BC、IDM的五项。各上游seed内先平均27主评价条件，再等权平均B1/B2/B4，最后五seed配对。报告均值差、正向seed、t95、精确双侧符号置换及五项Holm。MIF-Aux对Base/LAPO-Aux/LAOM-Aux为另三项次要族；每预算曲线为描述性。五种子精确p最低.0625。不将预算、回合或动作重复计为独立训练seed。

## 信息与计算边界

视觉前端、投影和潜表示history全部复用冻结原件。潜表示只训练ActionDecoder；BC直接训练同原版GRU，IDM先用该预算真实标签学习逆模型，再对32条无标签轨迹生成伪标签训练GRU。更新数/优化器/原batch规则均不变。相同独立标签预算不意味着训练信息流相同。离线dev动作只在最终评价读取，不用于拟合、选checkpoint或调参。

评价原30条件、每条最多200步，其中908604–908630的27条件构成主结果，旧3开发条件保留次要统计。原MSAA0、GL_DITHER关闭、目标相机渲染、伙伴响应策略、视觉背景与相机变化不变。

## 准入与运行

入口 `python -m experiments.visual_low_label manage`；协议 `configs/visual_low_label.json`。从包根运行，正式输出独立目录 `outputs/visual_low_label_2026_09_19_r1`。

1. 六个低预算两更新smoke，涵盖latent、BC、IDM的B1/B4。
2. 五seed×九配置重训B8 decoder，逐权重/归一化比较旧件，max abs≤1e-6；IDM逆模型也检查。
3. 45次旧条件908601独立闭环重放，动作和每步reward要求逐项完全一致。
4. 前三项全部通过才新增135个低预算grounding；全部训练完成后冻结，才派发正式评价。四路上限，保留至少4GiB可用RAM。
5. 完成后校验来源哈希、135新+45复用身份、审计、权重、foundation，再输出完整正负结果。

`PAUSE`只停止新派发、在任务边界暂停；已完成单元可续跑。运行器检测存活manager不重复启动；若任务存在partial或失败则保留现场，不自动覆盖/改协议/重试。工作进程退出后检查审计再计为完成。原始B8、历史源码与已完成实验均只读。结果不自动写进论文、不自动上传GitHub。

代码：`training/visual_label_budget.py`仅改变原监督训练轨迹数量；`evaluation/visual_label_budget.py`仅绑定路径并调用原`visual.rollout.evaluate`和原渲染器。来源记录在正式运行的`source_manifest.json`。

首次准备目录保留：管理器查询已结束worker的PID触发NoSuchProcess，在B8核验阶段退出；新增低标签控制结果为0。r1仅修正状态读取竞态，不改模型、预算、种子或评价协议。
