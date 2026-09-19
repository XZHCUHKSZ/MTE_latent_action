# MPE 时间接口修正：小规模开发实验

## 修正依据

原始 seed40–44 的每个 1,000 episode 档案均满足 p[t+1]=p[t]+0.1*v[t]，最大残差约 2.24e-7。动作 a[t] 先改变 v[t+1]，到 p[t+2] 才出现位置效应。旧位置逆模型读取 p[t],p[t+1]，不能在同状态干预下区分当前动作。动作标签和 history 输出均对齐 t；不是回报算式或标签整列偏移。

旧结果保留。旧控制得分不因该发现自动失效，但不能据此主张一步转移识别了当前动作。新实验不修改环境积分器，不添加模拟器速度，不把动作/物理分支用于预训练。

## 新协议

- 2 个开发训练种子 40、41；沿用其训练/开发 episode 划分的前 256/64 条；不使用原测试集训练。
- 预算 B=8、32，包括其中 1/4 验证轨迹。原始每条轨迹仅 t=1..22 共22个动作纳入 grounding。
- 所有修正后的局部/联合 frontend 均使用未来 t+2；LAOM 使用固定 offset2，LAPO 三个输入时刻为 t-1,t,t+2。原适配类不改。明确标为本地时间适配，不冒充官方复现。
- frontend LAOM 的预测目标同时改为 t+2。仅延长 inverse 窗口、却继续只重构不含当前动作的一步位置，不能构成充分的时间修正。
- 预测器采用原 ObservationReadout 类，输出24维，直接预测物理 h=1,2,3 的8维位置增量。原历史+code条件和 MTE 的两端共享 history/partner 规则不变。这里的多步目标含后续行为的不确定性，仍称 learned predictive contrast，不称已识别物理干预。
- 这是显式新适配协议，不是原 recursive one-step bridge 的原样重跑。h1 保留为时间负对照，h2 为主要物理检验，h3 供既有 single-horizon family 使用。
- Simple/Graph/MIF 使用原类和已有 route 的 full-visibility reconstruction+variance/covariance 损失；不混称原 formal masked MIF 目标。
- matched/raw 使用相同初始可训练参数、数据、输入信息、训练步数、history结构和decoder宽度；raw仅替换现有坐标变换buffer。Graph partner-complement明确改变负端partner条件。
- 资源匹配指这些成对家族消融。LAPO、LAOM、Base 与 MTE 的原生表示维度、架构以及共享 frontend/readout 成本不同，不宣称整个基线矩阵等参数或总算力相等。协议 scope 中的共享 resources 不能扩大解释为全矩阵等计算量。
- 保留 Base、joint LAPO、joint LAOM、Simple/Graph/MIF Solo/Aux、Global-Joint16 Aux、BC。Global16 是全局转移分支，不是第二个局部 target-slot 分支。小实验暂不扩到 Tree、IDM、所有预算，不称完整原规模替换。
- 一个 offset1 的 Base/Graph 配对时间对照也使用新 direct readout；用于隔离窗口作用，不能当作原递归协议完整复现。

## 信息隔离与评估

curator 单独从原 archive 的 obs member 导出 positions-only 文件。训练在子进程安装文件白名单和 simulator import 禁止。全部种子预训练结束并落盘 global_freeze 后，另行导出 B 条轨迹的动作。history 输入为 x0..xt，未来后缀扰动必须不影响当前输出。冻结checkpoint在grounding前后比对hash。

所有模型在相同32个新评估episode上运行23步控制；零动作warmup、partner策略、reward、碰撞和覆盖距离计算完全沿用旧代码。不用episode充当训练种子。

主比较提前固定 Graph-Aux vs Graph-raw-Aux、MIF-Aux vs MIF-raw-Aux、Graph-Aux vs partner-complement-Aux。两预算、两种子全部报告。跨零/负向结果保留。两种子只作开发证据，不报告五种子显著性或普遍优势。训练步数较正式规模少，不能直接和旧大表数字比较。

独立的 post-freeze 物理评估：16新episode ×4时刻，从同状态改变首步target动作，固定续行；h2为主。它检验时间响应以及matched/mismatched误差，也报告零预测误差。该零动作参照诊断不代替原nearest-history donor的完整语义验证。

## 运行

在此包根目录：

```text
python tests/check_mpe_temporal_repair.py
python -m experiments.mpe_temporal_repair --workspace WORKSPACE --out outputs/mpe_temporal_repair_development
```

两个seed最多两路并行，每进程2个CPU线程；固定任务清单跑完停止，不自动搜索至正向。目录中 protocol、implementation_sources、status、逐stage日志、原始逐episode结果和 RESULTS_CN.md 可追踪。失败保留证据，不覆写未完成预训练目录。

2026-09-17 已通过 2-update 端到端冒烟：全部16个control配置、文件访问白名单、26份训练权重的冻结检查、物理分支精确重放，以及raw/matched初始参数一致性。冒烟结果仅验证可运行，不作性能证据。正式开发任务目录为 `outputs/mpe_temporal_repair_development`，具体完成状态以该目录的 `status.json` 为准。
