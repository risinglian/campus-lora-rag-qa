# LoRA 训练 Loss 曲线与过程分析

- 数据来源：`outputs/qwen25-1.5b-campus-lora/trainer_state.json`
- 曲线图片：`report/image/lora_loss_curve.png`
- 曲线数据：`report/evidence/13_loss_curve_data.csv`
- 图片生成状态：PIL generated actual line chart because matplotlib failed: ModuleNotFoundError("No module named 'matplotlib'")

## 关键训练结果

| 指标 | 真实取值 |
| --- | ---: |
| global step | 81 |
| epoch | 3.0 |
| train_loss | 1.2578617422669023 |
| eval_loss | 0.730340301990509 |
| train_runtime 秒 | 150.2109 |

## 分析

训练日志显示 LoRA SFT 正常完成，最终产生可加载的 adapter。loss 曲线只能说明训练过程正常和损失收敛趋势，不能单独证明问答效果；最终效果仍应以 `eval_results/compare_12_latest.md` 与 `eval_results/evaluation_summary.md` 的 12 题三模式评估为主。由于数据集规模为 240 条，报告中不夸大泛化能力，而是结合 Base、LoRA、LoRA+RAG 的同题对比分析微调带来的变化。
