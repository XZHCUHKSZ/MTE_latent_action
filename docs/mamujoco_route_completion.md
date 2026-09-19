# MaMuJoCo 五种子路线与坐标配对补齐

2026-09-19 用户授权开始。正式目录：`outputs/mamujoco_route_completion_2026_09_19/`。

## 验证目的

1. 完整MTE分支相较同样拼接Base的Global-Joint16是否有控制收益？
2. 同一编码器、预测表、目标、参数量和训练预算下，Möbius坐标相较raw坐标是否有控制收益？

两个问题分别统计。Global16的上游目标和计算不同，不是单操作消融。raw保留原MTE表，不能称为“完全无MTE”。MIF是MaMuJoCo主要模型，Graph/Simple为预先固定的家族比较，不能结束后换赢家作主模型。

## 固定规模

- N90；原数据/前端种子0–4。0/1曾经观察过，不宣称五个全新确认种子或新任务。
- B4/8/16/32/64，包含动作拟合和验证标签；五次decoder重复202609160–164。
- 七配置：MIF/Graph/Simple各matched与raw，加Global16；全部Aux拼接相同冻结Base。
- 35个上游表示/history配置：14复用、21新训练。
- 875个控制单元：278完整旧单元复用、597新增。原部分MIF记录从更早的decoder重复归档核验复用；不补写旧目录。
- 每个控制单元沿用原50个共同评估episode、200步上限、原teacher和目标agent接口。
- 原Base/LAPO/LAOM/BC的100个种子×预算背景记录保留在baseline_references.json；这些记录不是同decoder重复的新配对单元。

## 方法与信息边界

调用整理包已有 `training/route_representation.py`、`experiments/route_ablation.py`、history/grounding、composition和官方环境评估适配。raw在新入口注册已有factory的raw_coordinates模式，不修改科学类、训练函数或受保护核心文件。

三类matched/raw均为原full-visibility路线目标，3200 representation更新、1200原latent-history更新；Global16采用原3200-update前端与1200-update history。grounding沿用800更新。不是原masked主实验的静默替换，也没有加入曾试过的history semantic loss。

预训练独立进程只允许观察导出、修复后的预测端点；禁止动作文件和模拟器导入。全部35组模型检查冻结与Base归一化一致后，才开放新grounding的预算标签。来源SHA256、协议和代码指纹冻结。原始结果不修改。

## 分析计划

每个模型先在每个上游seed内平均五个decoder重复和全部五个预算，再取matched−global和matched−raw。独立推断单位为五个上游seed。

路线三项、坐标三项是两个预先固定Holm检验族。报告所有六项的均值、逐seed差、正向seed数、描述性配对t95区间、精确双侧符号置换p及Holm p。五个seed最小精确p为0.0625；不按结果追加seed或替换指标。此设计检验重复性与效应大小，不保证显著或优势成立。

## 运行与续跑

入口 `experiments/mamujoco_route_completion.py`，协议 `configs/mamujoco_route_completion.json`。

两路预训练，全部冻结后四路grounding/评估；新派发要求至少6 GiB可用内存。manager使用独占文件锁防重复启动。失败时停止新派发，等待其他正在运行的任务结束，保留错误与partial供检查。

`PAUSE`文件仅停止新派发，不杀正在执行的任务。完成的独立任务可以复用续跑；不完整目录不会被自动覆盖。断电前应先等待正在执行任务保存完整结果，避免声称支持任意update级断点。

阶段进度见status.json；每个任务有progress.json和独立日志。完整结果只有全部875单元、来源/模型哈希及基础核验通过后才标complete。论文和GitHub不自动更新。

## 已通过的启动验证

- 七条预训练路径，2更新；MIF matched/raw与Global16三条grounding和2episode×5步评估路径。
- 三个matched/raw配对参数量、归一化摘要一致；raw的Mobius/zeta均为单位阵；初始化配对检查通过。
- 全局冻结后才读取grounding标签，访问审计无违规。
- 原MIF控制器在当前整理接口重放两个200步episode，回报误差0。
- 冻结基础检查前后PASS，11文件。

冒烟报告在 `outputs/mamujoco_route_completion_smoke_2026_09_19/`。冒烟只验证接口，不代表正式表现。
