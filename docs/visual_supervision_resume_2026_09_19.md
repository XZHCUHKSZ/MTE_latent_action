# 视觉监督清单：进度读取故障与续跑

用户授权：2026-09-19「修复，然后继续」。

原管理器在读取 `formal/seed908712/b4/evaluate_mif_aux_pretrained_trainable/progress.json` 时发生 PermissionError 并退出。错误位于进度显示读取，不是训练、模型或回报计算。文件随后可读，符合瞬时共享访问冲突的表现，但未确认操作系统层面的具体原因。

故障后残留的三个评价任务正常结束。续跑前确认无存活的实验进程：360 个 grounding 和 246 个 evaluation 均完整保存，无 partial。剩余 114 个评价任务。

独立入口 `experiments.visual_supervision_inventory_resume` 只给原管理器的 `progress.json` 读取增加短重试；连续失败显示 telemetry_unavailable。结果、访问审计、协议、来源和权重读取仍严格失败；底层 I/O 错误不被吞掉。worker 仍调用原入口，训练、评价、随机种子及统计协议均不变。

不修改原 runner、source_manifest 或 ground_freeze；独立 resume_record 记录新增入口的哈希和原始清单哈希。旧 manager_failure 和原 stdout/stderr 保留为故障证据，续跑使用独立日志。原 qualification 不重写，直接验证既有资格检查及冻结权重后，从评价阶段补齐。

故障注入检查：`python -m unittest discover -s tests -p check_visual_supervision_resume.py`。启动前只读核验：`python -m experiments.visual_supervision_inventory_resume --out outputs/visual_supervision_inventory_2026_09_19 --check-only`。正式续跑去掉 `--check-only`，后台窗口隐藏。

监视以最新 status、resume_record 和独立续跑日志为准。历史 manager_failure 不代表新一轮又失败；新故障会写入独立 manager.resume_failure 文件。保留原始 620 个结果身份要求和全部科学校验。
