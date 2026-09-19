# MPE 同信息、同资源 matching 输入控制

目的与所有边界见 `weak_controls_audit_2026_09_19.md`。
入口：`python -m experiments.mpe_matching_input_control --out outputs/NEW_DIRECTORY`。
协议：`configs/mpe_matching_input_control.json`。三臂固定N700/B32、5seed；原共享模型不改。

输入为两组8维端点坐标，调用现有Graph类；与原8维edge-only Graph区分。统一重建归一化端点A/B，逐臂只改变输入可逆坐标映射。context_mismatch用固定伙伴complement，保持全部信息、端点来源和样本数不变。配对检查包括参数量、初始化、归一化、重建目标、批次序列哈希；history的未来后缀不能改变过去输出。

训练进程仅能访问导出观测及冻结预测端点，不能读动作或模拟器。全部history冻结后才复制已有B32动作标签。grounding复用原实现；所有输出及checkpoint在新目录。

正式规模15个表示/history路径、15个闭环结果。两路预训练，全部完成后四路grounding。2更新冒烟使用原完整观测和归一化，但缩短表示/history/decoder步数且仅2个评价episode；不能把冒烟数值视为正式结果。

入口拒绝既有输出目录，防止覆盖；当前入口不自动续跑，也不实现PAUSE。管理器失败停止新派发并保存failure.json，已在运行的子进程不强杀。遇到关机或中断先检查并做显式续跑设计，不对旧目录重新运行入口。

主统计：每上游seed一个B32均值，两项配对差及同族Holm。旧开发数据和旧评价episode复用，非新环境确认；无结果相关追加种子/预算/模型。
