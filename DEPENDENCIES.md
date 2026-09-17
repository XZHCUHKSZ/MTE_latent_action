# 运行环境

整理与检查使用仓库的 Python 3.12 虚拟环境。确切已安装版本见 `provenance/environment_versions.json`。
PyTorch 使用与 CUDA 驱动兼容的安装包；本机版本记录在上面的 JSON 中。不要把旧 `vendor/` 目录当成可迁移的 torchvision 安装。

训练需要 GPU；结果阅读命令只需要 Python 标准库。环境评估另外需要 MuJoCo / Gymnasium-Robotics、冻结的 SAC teacher 和 DAVIS 图像。
`third_party/laom/` 保留所用官方神经网络及 distracting-control-suite 源码和 Apache-2.0 LICENSE。
LAPO/LAOM 数值及 feature-space 适配的来源说明仍在核心类中；没有因换文件名而消除来源。

大数据、模型权重、DAVIS、teacher checkpoint 不放入这个代码包。数据和权重必须沿用论文的冻结版本，不能新采集后仍标称原结果。

视觉 augmentation 还需要 torchvision；原实验 vendor 的元数据为 `0.28.0+cu130`，对应本机 `torch 2.13.0+cu130`。新包不复制二进制 vendor；复现时安装对应 wheel。此依赖未在新目录重新安装。
