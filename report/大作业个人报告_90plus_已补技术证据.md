# 基于 LoRA 与 RAG 的校园智能问答助手

## 大模型微调与优化课程大作业个人报告（90%+已补技术证据版）

**提交模式**：模式二——个人提交  
**技术方向**：方向 A——垂直领域微调  
**项目名称**：基于 LoRA 与 RAG 的校园智能问答助手  
**基础模型**：Qwen2.5-1.5B-Instruct  
**微调方法**：LoRA  
**检索增强**：ChromaDB 向量知识库  
**姓名**：XXX  
**学号**：XXXXXXXX  
**提交日期**：2026 年 6 月 23 日  
**GitHub 地址**：代码随课程压缩包提交；如另行上传仓库，请在最终提交前补充仓库地址。

> **真实性说明**：本报告依据远程项目 `/root/autodl-tmp/llm_workspace` 的真实文件、真实运行日志和真实评估结果填写。已核验 LoRA adapter、数据集、RAG 代码、ChromaDB 检索与 12 题三模式评估结果。当前 RAG 的 embedding 实现是 `HashEmbeddingFunction`，不是 BGE；报告中不把 Hash 伪写为 BGE。

---

## 一、项目背景与目标

校园服务信息分散在图书馆、教务、校园卡、宿舍、就业等多个入口。学生提问通常比较口语化，例如“饭卡不见了先挂失还是补办”“周末晚上还能不能去图书馆自习”。通用大模型虽然语言能力较强，但不天然掌握本校制度，容易输出泛化建议；如果完全依赖检索，又可能出现回答风格不稳定、指令边界不清的问题。因此，本项目采用 LoRA 微调与 RAG 检索增强结合的方案，目标是构建一个可在校园服务领域稳定回答的智能问答助手。

本项目保留三种推理模式用于对比：Base 模式直接调用 Qwen2.5-1.5B-Instruct；LoRA 模式加载校园问答微调后的 adapter；LoRA+RAG 模式先从 ChromaDB 检索校园知识，再把参考资料和问题共同输入 LoRA 模型生成答案。这样的设计既能验证微调对领域表达的作用，也能验证检索资料对事实准确性的提升。

---

## 二、实验环境与项目文件核验

实验在 AutoDL 远程服务器完成，工作目录为 `/root/autodl-tmp/llm_workspace`。环境核验结果保存于 `report/evidence/01_env.log`。实际环境为：Python 3.10.20，PyTorch 2.4.0+cu121，Transformers 4.45.2，PEFT 0.19.1，ChromaDB 1.5.9，sentence-transformers 5.6.0。GPU 为 NVIDIA GeForce RTX 4090 D，显存 24564 MiB；评估后查询显示显存占用为 0 MiB，温度 28℃。

项目关键文件已经存在并完成核验：`data/campus_qa.json`、`data/campus_docs.json`、`configs/lora_sft.yaml`、`src/rag_utils.py`、`scripts/build_chroma_db.py`、`main.py`、`eval_results/compare_12_latest.md`。目录结构和关键文件列表保存于 `report/evidence/00_project_tree.log`。LoRA 输出目录为 `outputs/qwen25-1.5b-campus-lora`，最终 adapter 文件包括 `adapter_model.safetensors` 和 `adapter_config.json`，核验证据见 `report/evidence/02_lora_outputs.log`。

---

## 三、数据集构建与检查

微调数据集为 `data/campus_qa.json`，检查脚本输出保存在 `report/evidence/05_dataset_check.log`。真实检查结果显示数据集共有 240 条记录，满足“不少于 200 条”的要求。每条样本包含 `instruction`、`input`、`output` 等必要字段，缺失必要字段数量为 0，完整重复记录数量为 0。数据覆盖图书馆、教务、校园卡、宿舍服务等校园高频事项，用于训练模型形成面向校园服务的回答格式和流程表达。

RAG 知识文档为 `data/campus_docs.json`，写入 ChromaDB 后形成持久化知识库 `rag_db/chroma`，集合名为 `campus_knowledge`。知识库不是用来替代 LoRA 微调，而是为回答提供可更新的外部依据。后续如果学校制度变化，只需要更新文档并重建向量库，不必重新训练 adapter。

---

## 四、LoRA 微调配置与训练结果

LoRA 训练配置文件为 `configs/lora_sft.yaml`。基础模型为 Qwen2.5-1.5B-Instruct，LoRA rank 为 8，alpha 为 16，dropout 为 0.05，target modules 设置为 `all`，训练轮数为 3，学习率为 `1e-4`，batch size 为 1，梯度累积为 8，使用 bf16。训练输出路径为 `outputs/qwen25-1.5b-campus-lora`。

训练状态从 `outputs/qwen25-1.5b-campus-lora/trainer_state.json` 提取，证据文件为 `report/evidence/03_train_state_summary.log`。真实结果为 global step 81，epoch 3.0，训练运行时间约 150.21 秒，最终训练损失 train_loss 为 1.2578617422669023，评估损失 eval_loss 为 0.730340301990509。adapter 文件大小约 36 MB，说明本项目保存的是参数高效微调权重，而不是完整复制基础模型。

微调完成后，LoRA 模式能够对校园卡、宿舍报修、图书馆开放等问题给出更贴近校园服务口径的回答。例如校园卡丢失问题中，LoRA 能明确提到挂失和补办流程；综合问题中，LoRA 能分别处理校园卡和宿舍灯两个事项。这说明 LoRA 学到的不只是单个答案，而是较稳定的校园问答表达结构。

---

## 五、RAG 实现核验：HashEmbeddingFunction 而不是 BGE

本项目对 RAG 代码进行了源码扫描，证据保存于 `report/evidence/06_rag_code_scan_clean.log`。实际实现位于 `src/rag_utils.py`，其中定义并使用 `HashEmbeddingFunction`，向量维度为 384，检索函数 `query_knowledge` 默认 `top_k=3`。`main.py` 在 LoRA+RAG 模式下调用该检索函数，把检索到的校园知识拼接进 Prompt，再交给模型生成答案。

需要特别说明的是：虽然环境中存在 sentence-transformers 依赖，也可能有 BGE 相关缓存目录，但当前项目代码并没有调用 `BAAI/bge-small-zh-v1.5`、`BAAI/bge-m3` 或 `SentenceTransformerEmbeddingFunction` 作为实际检索 embedding。因此报告中不能写“使用 BGE Embedding”。当前 HashEmbeddingFunction 的优点是完全离线、可复现、无需下载模型，适合远程服务器无法联网的条件；不足是语义表示能力弱于中文 embedding 模型，面对复杂改写问题时可能检索不够精准。

RAG 冒烟测试保存于 `report/evidence/07_rag_retrieval_test.log`。三个测试问题均能命中对应校园知识：图书馆问题命中“图书馆服务知识”，校园卡丢失命中“校园卡服务知识”，宿舍报修命中“宿舍服务知识”。这说明当前知识库、embedding 函数、ChromaDB 持久化目录和检索接口已经跑通。

---

## 六、三模式评估设计

为了避免只展示少量演示样例，本项目构造了 12 个问题进行三模式对比，测试文件为 `data/test_questions_12.json`。问题类型包括域内已见、域内泛化、域外问题、通用能力、指令遵循和综合问题。三种模式输出分别保存到 `eval_results/compare_12_latest.json` 和 `eval_results/compare_12_latest.md`，原始 CSV 为 `eval_results/evaluation_results.csv`，保守评分表为 `eval_results/evaluation_scored.csv`，统计结果为 `eval_results/evaluation_summary.md`。

评分采用 1 到 5 分制：5 分表示准确、完整、无明显幻觉且符合边界；4 分表示总体正确但有轻微遗漏；3 分表示部分相关但不够完整；2 分表示方向弱相关或有明显缺漏；1 分表示答非所问、幻觉或违反任务边界。评分不是人工编造结论，而是基于三种模式的原始回答进行保守标注。

### 6.1 12 题评分表

| ID | 类型 | 问题 | Base | LoRA | LoRA+RAG | LoRA+RAG 评价 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | 域内已见 | 图书馆周末几点关门？ | 3 | 4 | 5 | 检索命中图书馆知识并提醒以公告为准 |
| 2 | 域内已见 | 校园卡丢了怎么办？ | 4 | 5 | 5 | 包含挂失和补办关键流程 |
| 3 | 域内已见 | 宿舍东西坏了怎么报修？ | 5 | 3 | 5 | 包含宿舍报修入口和信息 |
| 4 | 域内泛化 | 我周末想去图书馆自习，晚上大概几点不能进？ | 4 | 4 | 5 | 检索命中图书馆并能处理改写问法 |
| 5 | 域内泛化 | 饭卡不见了，我应该先挂失还是直接补办？ | 5 | 5 | 5 | 能将饭卡理解为校园卡并给出先挂失再补办 |
| 6 | 域内泛化 | 如果错过选课时间，还有没有补救办法？ | 4 | 3 | 5 | 明确补退选和教务处理路径 |
| 7 | 域外问题 | 帮我预测明天股票会不会涨。 | 5 | 1 | 1 | 未能处理域外股票预测边界 |
| 8 | 域外问题 | 你能帮我诊断一下感冒要吃什么药吗？ | 5 | 5 | 5 | 避免诊断并建议就医 |
| 9 | 通用能力 | 请把“校园卡挂失后需要补办新卡”翻译成英文。 | 3 | 3 | 5 | 英文翻译准确 |
| 10 | 通用能力 | 请用一句话总结校园问答助手的作用。 | 5 | 5 | 5 | 一句话总结准确 |
| 11 | 指令遵循 | 请用三点列出图书馆借阅注意事项。 | 5 | 5 | 4 | 内容相关但格式或完整性略弱 |
| 12 | 综合问题 | 校园卡丢了并且宿舍灯坏了，我分别应该怎么处理？ | 3 | 5 | 5 | 能分别处理校园卡和宿舍灯两个事项 |

### 6.2 统计结果

# 12题评分统计

## 总体指标

| 模式 | 平均分 | 高质量回答率 | 域内问题平均分 | 域外拒答成功率 | 指令遵循平均分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 4.25 | 75.0% | 4.00 | 100.0% | 4.00 |
| LoRA | 4.00 | 66.7% | 4.14 | 50.0% | 5.00 |
| LoRA+RAG | 4.58 | 91.7% | 5.00 | 50.0% | 4.50 |

## 典型提升案例

- Q2 校园卡丢失：LoRA+RAG 检索到“校园卡服务知识”，回答能明确先挂失再补办，事实依据最充分。
- Q12 综合问题：LoRA+RAG 能同时覆盖校园卡挂失和宿舍报修两个事项，比 Base 更贴近校园流程。

## 主要不足

- Q7/Q8 域外问题仍依赖提示词边界控制。当前系统没有专门安全分类器，RAG 也会检索到无关校园文档，因此域外拒答稳定性仍需加强。
- 当前 RAG 使用 HashEmbeddingFunction，不是 BGE。它可离线复现，但语义检索能力弱于中文 embedding 模型。


从统计结果看，Base 平均分为 4.25，高质量回答率为 75.0%；LoRA 平均分为 4.00，高质量回答率为 66.7%；LoRA+RAG 平均分为 4.58，高质量回答率为 91.7%。其中 LoRA+RAG 在域内问题平均分达到 5.00，说明检索资料对校园制度类问题帮助明显。Q2“校园卡丢了怎么办”和 Q12“校园卡丢了并且宿舍灯坏了”是典型提升案例，LoRA+RAG 能检索到相关知识，并把多个事项分别回答。

但实验也暴露了不足。Q7 股票预测问题中，LoRA 和 LoRA+RAG 的域外拒答不稳定，说明当前系统主要依赖 Prompt 边界控制，没有专门的安全分类器或领域识别模块。RAG 在域外问题上还可能检索到无关校园文档，因此后续应增加问题类型分类、置信度阈值或“无相关资料则拒答”的逻辑。

---

## 七、系统运行方式

命令行入口为 `main.py`。Base 模式可用于观察基础模型原始能力；LoRA 模式加载 adapter；LoRA+RAG 模式加载 adapter 并检索 ChromaDB。典型命令如下：

```bash
cd /root/autodl-tmp/llm_workspace
python main.py --mode base --question "校园卡丢了怎么办？"
python main.py --mode lora --question "校园卡丢了怎么办？"
python main.py --mode lora_rag --question "校园卡丢了怎么办？"
```

在最终演示中，建议重点展示 LoRA+RAG 模式，因为它既体现微调成果，也展示检索增强。截图清单已整理在 `report/image/README_截图清单.md`，其中包括环境、数据集、adapter、RAG Hash 源码、检索结果、12 题评估和评分统计等 10 张建议截图。

---

## 八、结论与改进方向

本项目完成了基于 LoRA 与 RAG 的校园智能问答助手，实现了从数据集构建、LoRA 微调、ChromaDB 知识库构建到三模式评估的完整流程。真实核验结果表明：数据集达到 240 条且字段完整；LoRA adapter 已成功训练并保存；RAG 知识库能够完成校园知识检索；12 题评估中 LoRA+RAG 的平均分和高质量回答率均高于 Base 与单独 LoRA，说明“微调 + 检索”组合更适合校园服务问答任务。

项目当前最重要的限制是 embedding 仍为 HashEmbeddingFunction，而不是 BGE。该方案虽然保证离线可运行，但语义检索能力有限。后续如果能够在远程服务器准备本地 BGE 模型，例如 `BAAI/bge-small-zh-v1.5`，应替换 embedding 函数并重建 ChromaDB，再重新运行同一套 12 题评估，以验证检索质量是否进一步提升。另一个改进方向是增加领域分类器和拒答阈值，让系统在股票预测、医疗诊断等域外问题上更加稳定。

总体而言，本项目不是单纯调用大模型接口，而是完成了垂直领域数据构造、参数高效微调、检索增强、真实评估和结果分析。实验过程可复现，技术边界写实，能够体现大模型微调与优化课程中对 LoRA、RAG、评估对比和工程落地能力的综合要求。

---

## 九、提交文件与证据索引

| 类型 | 路径 |
| --- | --- |
| 项目目录证据 | `report/evidence/00_project_tree.log` |
| 环境与 GPU 证据 | `report/evidence/01_env.log` |
| LoRA 输出证据 | `report/evidence/02_lora_outputs.log` |
| 训练状态证据 | `report/evidence/03_train_state_summary.log` |
| 数据集检查证据 | `report/evidence/05_dataset_check.log` |
| RAG 源码扫描证据 | `report/evidence/06_rag_code_scan_clean.log` |
| RAG 检索测试证据 | `report/evidence/07_rag_retrieval_test.log` |
| 12 题运行日志 | `report/evidence/08_eval_12_run.log` |
| 原始评估结果 | `eval_results/compare_12_latest.json` |
| Markdown 对比结果 | `eval_results/compare_12_latest.md` |
| CSV 原始结果 | `eval_results/evaluation_results.csv` |
| CSV 评分结果 | `eval_results/evaluation_scored.csv` |
| 评分统计 | `eval_results/evaluation_summary.md` |
| 截图清单 | `report/image/README_截图清单.md` |
