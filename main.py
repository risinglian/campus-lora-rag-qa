#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.rag_utils import query_knowledge

DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_LORA = ROOT / "outputs" / "qwen25-1.5b-campus-lora"
DEFAULT_CHROMA = ROOT / "rag_db" / "chroma"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，统一控制 Base、LoRA、LoRA+RAG 三种运行模式。"""
    parser = argparse.ArgumentParser(description="Campus QA assistant with Base, LoRA, and RAG modes.")
    parser.add_argument("--mode", choices=["base", "lora", "rag", "lora_rag"], default="base")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--adapter-path", default=str(DEFAULT_LORA))
    parser.add_argument("--chroma-path", default=str(DEFAULT_CHROMA))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def build_prompt(question: str, contexts: list[dict[str, object]]) -> str:
    """根据检索结果构造 Prompt。

    Base 与 LoRA 模式不传入上下文，直接回答用户问题；LoRA+RAG 模式会把
    ChromaDB 检索到的校园知识放入 Prompt，要求模型优先依据资料回答。
    """
    if not contexts:
        return question
    context_text = "\n\n".join(
        f"[{index}] {item['title']}\n{item['content']}" for index, item in enumerate(contexts, start=1)
    )
    return (
        "请基于以下校园知识回答问题。若知识库没有直接答案，请说明需要以学校最新通知为准。\n\n"
        f"校园知识：\n{context_text}\n\n"
        f"问题：{question}"
    )


def load_model(model_path: Path, adapter_path: Path | None):
    """加载基础模型，并在 LoRA/LoRA+RAG 模式下额外加载 PEFT adapter。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path is not None:
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found: {adapter_path}. Train first with configs/lora_sft.yaml."
            )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, question: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    """按 Qwen chat template 组装对话并生成最终回答。"""
    messages = [
        {"role": "system", "content": "你是校园智能问答助手，回答应准确、简洁，并提醒用户以学校最新通知为准。"},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

    generated = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main() -> None:
    """主流程：按模式决定是否检索 RAG、是否加载 LoRA，然后输出回答。"""
    args = parse_args()
    mode = args.mode
    use_rag = mode in {"rag", "lora_rag"}
    use_lora = mode in {"lora", "lora_rag"}

    contexts: list[dict[str, object]] = []
    if use_rag:
        chroma_path = Path(args.chroma_path)
        if not chroma_path.exists():
            raise FileNotFoundError(
                f"ChromaDB path not found: {chroma_path}. Run scripts/build_chroma_db.py first."
            )
        contexts = query_knowledge(chroma_path, args.question, args.top_k)

    prompt = build_prompt(args.question, contexts)
    tokenizer, model = load_model(Path(args.model_path), Path(args.adapter_path) if use_lora else None)
    answer = generate_answer(tokenizer, model, prompt, args.max_new_tokens, args.temperature, args.top_p)

    print(f"mode: {mode}")
    if contexts:
        print("retrieved:")
        for item in contexts:
            print(f"- {item['title']} distance={item['distance']}")
    print("answer:")
    print(answer)


if __name__ == "__main__":
    main()
