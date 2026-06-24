#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag_utils import get_chroma_collection, load_docs


def parse_args() -> argparse.Namespace:
    """解析知识库构建参数，默认读取 data/campus_docs.json。"""
    parser = argparse.ArgumentParser(description="Build the campus ChromaDB knowledge base.")
    parser.add_argument("--docs", default=str(ROOT / "data" / "campus_docs.json"))
    parser.add_argument("--db", default=str(ROOT / "rag_db" / "chroma"))
    parser.add_argument("--collection", default="campus_knowledge")
    return parser.parse_args()


def main() -> None:
    """将校园知识文档写入 ChromaDB。

    每条文档使用 id 作为主键、content 作为向量化文本、title 作为 metadata。
    具体 embedding 函数由 `src.rag_utils.get_chroma_collection` 绑定。
    """
    args = parse_args()
    docs_path = Path(args.docs)
    db_path = Path(args.db)
    db_path.mkdir(parents=True, exist_ok=True)

    docs = load_docs(docs_path)
    collection = get_chroma_collection(db_path, args.collection)

    collection.upsert(
        ids=[item["id"] for item in docs],
        documents=[item["content"] for item in docs],
        metadatas=[{"title": item["title"]} for item in docs],
    )

    print(f"docs: {len(docs)}")
    print(f"db: {db_path}")
    print(f"collection: {args.collection}")
    print(f"collection_count: {collection.count()}")


if __name__ == "__main__":
    main()
