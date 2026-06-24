from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class HashEmbeddingFunction:
    """离线 Hash embedding，用于在无网络环境下构建 ChromaDB。

    课程项目原计划可替换为 BGE 等中文向量模型，但远程服务器无法稳定下载模型。
    因此这里使用确定性的字符/二元组 Hash 向量，保证知识库构建和评估可离线复现。
    该方法不是语义 embedding，效果弱于 BGE；报告中也如实说明当前实现为 Hash。
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def name(self) -> str:
        return "campus_hash_embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in input]

    def embed_query(self, input):
        """兼容 ChromaDB 查询接口，对单条问题或多条文本生成向量。"""
        if isinstance(input, str):
            return self.embed_text(input)
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """兼容 ChromaDB 文档写入接口，为知识文档生成向量。"""
        return self(input)

    def embed_text(self, text: str) -> list[float]:
        """将文本映射到固定维度向量，并进行 L2 归一化。"""
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> Iterable[str]:
        """同时使用单字和相邻二元组，提高关键词匹配的稳定性。"""
        compact = "".join(text.lower().split())
        for char in compact:
            yield char
        for index in range(max(0, len(compact) - 1)):
            yield compact[index : index + 2]


def load_docs(path: Path) -> list[dict[str, str]]:
    """读取并校验校园知识文档，要求每条包含 id/title/content。"""
    with path.open("r", encoding="utf-8") as file:
        docs = json.load(file)
    if not isinstance(docs, list):
        raise ValueError(f"Expected a list of documents in {path}")
    for index, item in enumerate(docs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Document {index} is not an object")
        for field in ["id", "title", "content"]:
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Document {index} is missing {field}")
    return docs


def get_chroma_collection(db_path: Path, collection_name: str = "campus_knowledge"):
    """打开或创建 ChromaDB collection，并绑定离线 Hash embedding 函数。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=HashEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def query_knowledge(db_path: Path, question: str, top_k: int = 3) -> list[dict[str, object]]:
    """从 ChromaDB 中检索与问题最相近的 Top-K 校园知识。

    返回值包含文档 id、标题、正文和距离分数，供 `main.py` 组装 RAG Prompt。
    """
    collection = get_chroma_collection(db_path)
    result = collection.query(query_texts=[question], n_results=top_k)
    rows: list[dict[str, object]] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        rows.append({
            "id": doc_id,
            "title": (metadata or {}).get("title", doc_id),
            "content": document,
            "distance": distance,
        })
    return rows
