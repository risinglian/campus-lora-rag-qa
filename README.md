# 基于 LoRA 与 RAG 的校园智能问答助手

本项目是《大模型微调与优化》课程个人大作业，目标是构建一个面向校园服务场景的智能问答助手。系统基于本地部署的 `Qwen2.5-1.5B-Instruct`，使用 LoRA 进行参数高效微调，并结合 ChromaDB 检索增强生成（RAG）提供校园知识依据。

> 真实性说明：当前 RAG embedding 实现为 `HashEmbeddingFunction`，不是 BGE。基础模型本体不上传到 GitHub；LoRA adapter 约 35 MB，作为课程复现实验成果保留在 `outputs/qwen25-1.5b-campus-lora/`。

## 技术路线

- **Base 模型**：直接调用 `Qwen2.5-1.5B-Instruct`，作为原始基线。
- **LoRA 微调**：使用 `data/campus_qa.json` 进行 SFT，输出轻量 adapter。
- **ChromaDB RAG**：将 `data/campus_docs.json` 写入 ChromaDB，推理时检索 Top-K 校园知识。
- **三模式对比**：统一比较 `base`、`lora`、`lora_rag` 三种模式的回答质量。

## 环境配置

真实运行环境记录见 `report/evidence/01_env.log`。

| 项目 | 版本 / 配置 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.4.0+cu121 |
| Transformers | 4.45.2 |
| PEFT | 0.19.1 |
| ChromaDB | 1.5.9 |
| sentence-transformers | 5.6.0 |
| GPU | NVIDIA GeForce RTX 4090 D，24564 MiB |

## 安装依赖

建议使用已有 Conda 环境；依赖安装方式如下：

```bash
pip install -r requirements.txt
```

## 数据说明

- `data/campus_qa.json`：LoRA 微调数据，共 240 条校园问答样本，包含 `instruction`、`input`、`output` 字段。
- `data/campus_docs.json`：RAG 校园知识文档，用于构建 ChromaDB 知识库。
- `data/test_questions_12.json`：12 题三模式评估集，覆盖域内已见、域内泛化、域外问题、通用能力、指令遵循和综合问题。

## LoRA 训练方法

```bash
llamafactory-cli train configs/lora_sft.yaml
```

训练配置见 `configs/lora_sft.yaml`，adapter 输出目录为：

```text
outputs/qwen25-1.5b-campus-lora/
```

关键结果：

- global step：81
- epoch：3.0
- train_loss：1.2578617422669023
- eval_loss：0.730340301990509
- adapter 参数量：9,232,384，约占基础模型 0.598063%

## RAG 知识库构建

```bash
python scripts/build_chroma_db.py
```

该脚本读取 `data/campus_docs.json`，使用 `src/rag_utils.py` 中的离线 `HashEmbeddingFunction` 生成向量并写入 ChromaDB。选择 Hash embedding 是为了保证离线服务器可复现；它不是 BGE，语义检索能力也弱于 BGE。

## 三种运行方式

```bash
python main.py --mode base --question "校园卡丢了怎么办？"
python main.py --mode lora --question "校园卡丢了怎么办？"
python main.py --mode lora_rag --question "校园卡丢了怎么办？"
```

模式说明：

- `base`：只加载基础模型。
- `lora`：加载基础模型 + LoRA adapter。
- `lora_rag`：先检索 ChromaDB，再使用 LoRA 模型结合参考资料回答。

## 评估方法

```bash
python scripts/eval_compare.py
```

评估脚本默认读取 `data/test_questions_12.json`，分别运行 Base、LoRA、LoRA+RAG 三种模式，并输出 JSON 与 Markdown 结果。已有完整评估结果保存在 `eval_results/`。

## 主要结果

评分统计见 `eval_results/evaluation_summary.md`。

| 模式 | 平均分 | 高质量回答率 | 域内问题平均分 |
| --- | ---: | ---: | ---: |
| Base | 4.25 | 75.0% | 4.00 |
| LoRA | 4.00 | 66.7% | 4.14 |
| LoRA+RAG | 4.58 | 91.7% | 5.00 |

结论：LoRA+RAG 在域内校园服务问题上表现最好，说明微调后的回答风格与外部知识检索能够形成互补。但 LoRA 并非所有指标都优于 Base，域外拒答仍需额外规则增强。

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── main.py
├── configs/
│   └── lora_sft.yaml
├── data/
│   ├── campus_qa.json
│   ├── campus_docs.json
│   └── test_questions_12.json
├── src/
│   └── rag_utils.py
├── scripts/
│   ├── build_chroma_db.py
│   ├── check_campus_qa.py
│   ├── eval_compare.py
│   └── inspect_lora_trainable_params.py
├── outputs/
│   └── qwen25-1.5b-campus-lora/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── eval_results/
│   ├── compare_12_latest.json
│   ├── compare_12_latest.md
│   ├── evaluation_results.csv
│   ├── evaluation_scored.csv
│   └── evaluation_summary.md
└── report/
    ├── evidence/
    └── image/
```

## 主要代码说明

- `main.py`：统一推理入口，支持 Base、LoRA、LoRA+RAG 三种模式。
- `src/rag_utils.py`：RAG 检索工具，包含离线 `HashEmbeddingFunction` 与 ChromaDB Top-K 检索。
- `scripts/build_chroma_db.py`：读取校园知识文档并构建 ChromaDB。
- `scripts/eval_compare.py`：读取 12 题测试集并运行三模式评估。
- `configs/lora_sft.yaml`：LoRA 微调配置文件。

## 局限性

- 当前 RAG 使用 `HashEmbeddingFunction`，不是 BGE；离线可复现，但语义检索能力有限。
- 知识库规模较小，只覆盖课程实验中的校园服务场景。
- 域外拒答仍需增强，例如金融预测、医疗诊断等问题应加入领域分类或安全规则。
- 基础模型本体需要用户自行准备到 `models/Qwen2.5-1.5B-Instruct/`。

## 提交说明

- 基础模型文件未上传，`.gitignore` 已忽略 `models/`、`cache/`、`conda_envs/`、`pip_cache/` 等大目录。
- LoRA adapter 文件 `outputs/qwen25-1.5b-campus-lora/adapter_model.safetensors` 约 35 MB，低于 GitHub 单文件限制，本次保留上传。
- 若下载仓库后缺少基础模型，请先准备 `Qwen2.5-1.5B-Instruct` 本地权重，或修改 `main.py --model-path` 指向自己的模型目录。
