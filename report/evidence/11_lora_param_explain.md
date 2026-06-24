# LoRA 参数核验与作用说明

| 参数 | 真实取值 | 作用 | 为什么这样设置 |
| --- | --- | --- | --- |
| 基础模型路径 | /root/autodl-tmp/llm_workspace/models/Qwen2.5-1.5B-Instruct | 指定被微调的 Qwen2.5-1.5B-Instruct 权重位置 | 使用本地模型，避免远程服务器联网下载 |
| LoRA rank / r | 8 | 控制低秩矩阵的秩，决定 adapter 表达能力 | r=8 在小数据集上兼顾表达能力和过拟合风险 |
| LoRA alpha | 16 | 控制 LoRA 分支缩放强度 | alpha=16 与 r=8 对应 alpha/r=2，增强领域适配但不过度覆盖基础模型 |
| LoRA dropout | 0.05 | 对 LoRA 分支进行正则化 | 0.05 用于降低 240 条小样本训练的过拟合风险 |
| target_modules | ['k_proj', 'q_proj', 'o_proj', 'gate_proj', 'down_proj', 'v_proj', 'up_proj'] | 决定在哪些线性层注入 LoRA adapter | target=all 让注意力和前馈相关线性层都参与领域适配 |
| learning_rate | 1.0e-4 | 控制 adapter 参数更新步长 | 1e-4 适合 LoRA 小规模 SFT，训练速度和稳定性折中 |
| num_train_epochs | 3.0 | 训练轮数 | 3 轮足以完成课程数据适配，同时避免过度记忆 |
| per_device_train_batch_size | 1 | 单卡每步 batch size | 设为 1 降低单步显存占用 |
| gradient_accumulation_steps | 8 | 梯度累积步数 | 设为 8，使等效 batch size 为 8，提高训练稳定性 |
| bf16 | true | 是否使用 bf16 混合精度 | 4090D 支持 bf16，可节省显存并提升速度 |
| cutoff_len / max_seq_length | 1024 | 训练样本最大长度截断 | 控制上下文长度和显存占用 |
| lr_scheduler_type | cosine | 学习率调度策略 | 按配置使用，避免训练后期震荡 |
| warmup_ratio | 0.05 | 学习率 warmup 比例 | 训练初期逐步升高学习率，提高稳定性 |
| logging_steps | 5 | 日志记录间隔 | 便于观察训练 loss 变化 |
| save_steps | 50 | checkpoint 保存间隔 | 保留训练中间状态，降低训练中断风险 |

证据来源：`configs/lora_sft.yaml` 与 `outputs/qwen25-1.5b-campus-lora/adapter_config.json`。