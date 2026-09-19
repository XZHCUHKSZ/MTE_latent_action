# 当前实现与退役路线核对（2026-09-20）

本次按当前论文、正式结果和代码调用关系核对。当前发布入口只指向有效路线；原始档案、完成结果和旧交付包保持原样，历史材料可通过旧版本追溯。逐文件退役依据见 `provenance/release_tools/retirements.json`。

## MPE 的实际修正

位置观测满足先用已有速度更新位置、再由当前动作影响后续速度的时间顺序。原 `p[t],p[t+1]` 输入不能在同状态下辨别当前动作的位置效应。正式修正版使用 `t+2` 逆推窗口，同时把前端重建目标改为 `t+2`；匹配预测器直接预测 h1/h2/h3 增量，history 和动作标签仍对齐决策时刻 t。

正式 MPE 入口为 `experiments/mpe_inventory_completion.py`，结构对照由 `mpe_temporal_scale.py` 和 `mpe_evidence_completion.py` 补齐。对应 1,305 个唯一控制单元。旧的一步 MPE 分派入口、递归 endpoint 构造器及混合旧 MPE 数值的默认报告从当前交付移除。

这改变了环境适配中的时间索引、训练目标与预测路径。共享 `common.py`、`unified_models.py`、`training_profiles.py` 与当前冻结原件字节一致；本次核验没有改动这些类，也没有修改环境积分器。不能把旧一步 MPE 结果当成新协议的复现。

## 主方法和系统分支对照

| 路线 | Base | 第二分支 | 作用 |
|---|---|---|---|
| MTE-Solo | 无 | MTE history | 测试 MTE 单独控制 |
| MTE-Aux | 原目标槽 history | MTE history | 主方法的组合控制 |
| Base-only | 原目标槽 history | 无 | 原目标码参照 |
| Base-duplicate | 原目标槽 history | 同一 Base 输出重复 | 参数化/容量参照 |
| Global-Joint16-Aux | 相同 Base | 全系统 16 维 history | 替换 MTE 分支的系统路线对照 |

系统分支替换的是对照中的第二分支，不是把 MTE 主方法改为全系统码。旧的独立局部分支与 Global-Joint16 目标不同，不能重命名后复用。当前路线结论使用完整 Global-Joint16 配对；旧独立分支若出现在保留的论文诊断中，只按其原身份解释。

## MaMuJoCo 和视觉

- 原 MaMuJoCo 主实验使用常量输入保护后的端点，原生 MIF 保留 masked 目标。`experiments/mamujoco_frozen_control.py` 从旧混合入口提取原 MaMuJoCo 操作，并拒绝 MPE 输入；标签划分、原方法训练函数和 history 路由不变。
- 875 组路线实验的 MIF 使用 full-visibility 整表重建；500 组 masking×coordinate 实验保持另一个明确目标。两者各有协议和结果名，不能用其中一个覆盖另一个。
- 视觉主实验仍使用其原 RGB/PCA 与 history。低标签补齐改变 grounding 标签预算；Adapt 另外更新 history 副本和 decoder。新微调及分支归因是已完成的后续证据，不冒充已经写入当前 PDF 的主实验。
- raw、Random-edge 仍携带 MTE 输入。强对照或负向比较是有效证据，全部保留；退役依据是实现/协议被替代或入口不再使用，不是结果是否有利。

## 旧名字但仍有必要的实现

| 保留文件 | 当前实际依赖 |
|---|---|
| `experiments/mpe_temporal_repair.py` | 正式 inventory、scale、matching 和 donor 入口复用数据导出、标签边界等函数；开发运行本身不列为主实验 |
| `experiments/route_ablation.py` | 正式 MaMuJoCo 路线实验调用其 `train_global`；文件早期开发说明不决定当前运行种子数，正式配置决定 |
| `training/representation_mpe.py` | 修正版仍复用端点张量适配函数；其旧训练函数不作为当前 MPE 入口 |
| `configs/numeric.json` | MPE 数据源和原 train/dev ID 的谱系依赖；不再作为当前 MPE 训练协议 |
| `experiments/target_effects.py` | 冻结 MaMuJoCo 诊断复用收集/指纹函数 |
| `mte/coordinate_controls.py` | 已有 partner readout 支持函数仍引用；历史 pilot 不再作为当前训练入口 |

不能只按文件名中的日期、pilot、repair 或 auxiliary 删除代码。上述保留项有实际调用或原始证据需求；其用途与当前主入口分开说明。

## 发布核验

重新检查模块导入、静态依赖、时间索引、标签/验证预算、matched/raw 参数一致性及发布文件哈希。对新 MaMuJoCo 分派执行与原入口的真实两更新配对检查；该小检查用于确认分派等价，不作为性能复现。完整训练的结果与资格门槛沿用各自正式档案。本次不改论文数值、PDF 或图表。
