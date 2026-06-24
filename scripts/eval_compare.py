#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "data" / "test_questions_12.json"
DEFAULT_PYTHON = ROOT / "conda_envs" / "llm_course" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    """解析评估参数，默认对 12 个问题运行 Base、LoRA、LoRA+RAG。"""
    parser = argparse.ArgumentParser(description="Run Base, LoRA, and LoRA+RAG comparison tests.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--modes", nargs="+", default=["base", "lora", "lora_rag"])
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, object]]:
    """读取 12 题评估文件；若只有字符串，也转为统一字典结构。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    questions: list[dict[str, object]] = []
    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            questions.append({"id": index, "type": "未分类", "question": item})
        else:
            questions.append({
                "id": item.get("id", index),
                "type": item.get("type", "未分类"),
                "question": item["question"],
            })
    return questions


def run_one(python_bin: str, mode: str, question: str, max_new_tokens: int) -> subprocess.CompletedProcess[str]:
    """调用 main.py 运行单个模式，保证评估与真实命令行推理路径一致。"""
    command = [
        python_bin,
        str(ROOT / "main.py"),
        "--mode", mode,
        "--question", question,
        "--max-new-tokens", str(max_new_tokens),
    ]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def extract_answer(stdout: str) -> str:
    """从 main.py 输出中提取 answer 部分，便于生成 Markdown 摘要。"""
    marker = "answer:\n"
    if marker in stdout:
        return stdout.split(marker, 1)[1].strip()
    return stdout.strip()


def main() -> None:
    """对固定 12 题依次运行三种模式，并保存 JSON/Markdown 结果。"""
    args = parse_args()
    questions = load_questions(Path(args.questions))
    results = []

    for question_item in questions:
        for mode in args.modes:
            completed = run_one(args.python, mode, question_item["question"], args.max_new_tokens)
            results.append({
                "id": question_item["id"],
                "type": question_item["type"],
                "question": question_item["question"],
                "mode": mode,
                "returncode": completed.returncode,
                "answer": extract_answer(completed.stdout),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            })

    output_dir = ROOT / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"compare_{timestamp}.json"
    md_path = output_dir / f"compare_{timestamp}.md"
    latest_json = output_dir / "compare_latest.json"
    latest_md = output_dir / "compare_latest.md"

    json_text = json.dumps(results, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")

    lines = ["# Base / LoRA / LoRA+RAG 对比评估", "", f"- 生成时间：{timestamp}", ""]
    for question_item in questions:
        lines.append(f"## Q{question_item['id']} {question_item['type']}：{question_item['question']}")
        for item in results:
            if item["id"] == question_item["id"]:
                lines.append(f"- **{item['mode']}**：{item['answer']}")
        lines.append("")
    md_text = "\n".join(lines)
    md_path.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
