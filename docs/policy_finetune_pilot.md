# 其他方法的history微调：视觉与MPE固定预算轻量试验

用户2026-09-20授权轻量验证其他方法（特别LAPO/LAOM）能否同样受益，并询问MPE。结果正负均保留，不按性能调参或筛选种子。

入口 experiments/policy_finetune_pilot.py；适配 training/policy_finetune_pilot.py；配置 configs/policy_finetune_pilot.json。输出 outputs/policy_finetune_pilot_2026_09_20。原核心方法、协议、权重及已完成结果不改，新增文件作为下游监督适配。

视觉：LAPO/LAOM各Solo/Aux，五种子908711–715，B2=400动作向量=32条观察轨迹的6.25%，600更新、每步256采样。20新评价，对应20旧冻结结果；30原共同episode，其中27主统计。既有MIF微调/冻结及BC/IDM在B2作背景参照。直接复用visual_supervision_control.ground，仅绑定不同源策略；RGB/PCA、无动作normalizer冻结，Aux两支history均接受动作梯度。是论文既定state-adapter，不是官方端到端像素复现。

MPE：N700/B32，Graph-Aux(base+graph)、LAPO、LAOM，五种子45–49，15新评价及15旧冻结参照。32标签轨迹共704动作向量，其中528训练、176预算内验证。保留x0 warmup、t1–22对齐、1000更新、原decoder初始化和RNG、每25步验证及最优验证checkpoint选择，history与decoder共同保存同一步。100原共同episode。原BC3200更新，仅作背景参照，不能称同更新量比较。

同一方法冻结/微调保持架构、初始化、归一化、标签、输入、采样和验证划分一致。不同方法latent宽度/组合不同，不声称所有模型参数完全相等。没有新动作标签、前端训练、模拟器训练查询、伙伴标签或新任务。视觉评价dev动作只用于旧诊断，不用于训练或选择。MPE验证标签算在B32内。

先7条真实2更新CUDA训练路径与7个原环境单回合冒烟；随后35冻结decoder权重maxabs<=1e-6复现、视觉预测复现及35个原episode重放。全部通过才35完整微调；全部新权重冻结后再35完整评价。source_manifest与ground_freeze核对，35配对身份核查，审计无violations。进度文件读取容错不作用于结果与权重；失败停派发保留partial。最多四路。PAUSE停新派发，存活任务自然结束，不自动清理partial。

每上游seed内平均episode再五seed配对；视觉四项微调增益一个Holm族，MPE三项另一个族。全部方向、t95、精确符号置换p和Holm保留，最小p=.0625。固定预算探索，不冒充全预算优势、新种子或新任务。微调后跨模型排序为描述性；共同监督收益不归为MTE独有。

无动作latent/history预训练后做少标签策略适配，属于latent-action研究中的合理下游路线。本轮为history+decoder监督适配，不是冻结表示decoder-only探针，也不要求微调history保留原latent语义。LAPO原文的离线少标签路线训练decoder，在线策略微调另用RL；本轮不是逐字复现。LAOM的监督可以进入latent模型训练，而本轮冻结上游模型。

参考：https://proceedings.iclr.cc/paper_files/paper/2024/file/27985d21f0b751b933d675930aa25022-Paper-Conference.pdf
https://arxiv.org/abs/2502.00379
https://arxiv.org/abs/2410.11758

保持原冻结主实验，不自动修改论文或上传GitHub。真实预检见 outputs/policy_finetune_pilot_preflight_2026_09_20/CHECKS.json。

启动记录：初始准备目录在构建来源清单时因Windows路径分隔符不同中止，尚未派发任何任务。仅将来源键标准化为正斜杠，不改变文件或哈希。失败目录保留；正式目标改为 outputs/policy_finetune_pilot_2026_09_20_r1，结果目录同名。
