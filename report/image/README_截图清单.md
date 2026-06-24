# 截图清单（按报告证据编号）

> 截图时请优先截远程终端中对应文件或命令输出，文件路径均位于 `/root/autodl-tmp/llm_workspace`。

| 编号 | 建议截图文件名 | 截图内容 | 证据来源/命令 |
| --- | --- | --- | --- |
| 1 | `01_project_tree.png` | 项目目录、关键文件列表 | `cat report/evidence/00_project_tree.log` |
| 2 | `02_env_gpu.png` | Python/依赖/GPU 环境 | `cat report/evidence/01_env.log` |
| 3 | `03_dataset_check.png` | `campus_qa.json` 记录数、字段、重复检查 | `cat report/evidence/05_dataset_check.log` |
| 4 | `04_lora_training_state.png` | LoRA 训练 step、epoch、loss | `cat report/evidence/03_train_state_summary.log` |
| 5 | `05_adapter_output.png` | `adapter_model.safetensors`、`adapter_config.json` 文件 | `cat report/evidence/02_lora_outputs.log` |
| 6 | `06_rag_hash_code.png` | RAG 使用 `HashEmbeddingFunction` 而不是 BGE | `cat report/evidence/06_rag_code_scan_clean.log` |
| 7 | `07_rag_retrieval.png` | 三个问题的 ChromaDB 检索命中结果 | `cat report/evidence/07_rag_retrieval_test.log` |
| 8 | `08_three_mode_eval.png` | 12 题 Base/LoRA/LoRA+RAG 原始对比 | `cat eval_results/compare_12_latest.md` |
| 9 | `09_scoring_summary.png` | 平均分、高质量率等统计 | `cat eval_results/evaluation_summary.md` |
| 10 | `10_final_demo.png` | LoRA+RAG 单题演示输出 | `python main.py --mode lora_rag --question "校园卡丢了怎么办？"` |

说明：第 6 张截图尤其重要，报告中不能把 HashEmbeddingFunction 写成 BGE；除非后续真实完成 BGE 替换并重新跑通检索与评估。
