"""RAG 检索器：知识库按「条目」切片成 chunk，按相似度检索 top-k。

这是第 1 周「RAG = 先检索再生成」里的 Retrieval 环节。
关键设计（贴近真实 RAG）：
    - 不是把整篇文档向量化，而是按「每条政策」切 chunk —— 粒度更细，检索更准；
    - 每个文档取「最相关的那个 chunk」作为它的得分，再排序返回 top-k 篇文档。
这样短 query 和精准命中的条目相似度高，和无关条目的差距能被阈值干净区分。
"""
import os
from dataclasses import dataclass

import numpy as np

from .embedder import CharEmbedder


@dataclass
class RetrievedDoc:
    doc_id: str      # 文档编号，如 D1
    content: str     # 命中的 chunk 正文
    score: float     # 与 query 的余弦相似度


class KnowledgeBase:
    def __init__(self, kb_dir: str, embedder: CharEmbedder):
        self.embedder = embedder
        self.chunks: list[tuple[str, str]] = []   # (doc_id, chunk_text)
        self.doc_ids: list[str] = []
        self._load(kb_dir)

    def _load(self, kb_dir: str):
        for fname in sorted(os.listdir(kb_dir)):
            if not fname.endswith(".md"):
                continue
            doc_id = fname.split("_", 1)[0]
            with open(os.path.join(kb_dir, fname), encoding="utf-8") as f:
                for line in f:
                    text = line.strip().lstrip("- ").strip()
                    if text and not text.startswith("#"):  # 跳过标题行
                        self.chunks.append((doc_id, text))
                        if doc_id not in self.doc_ids:
                            self.doc_ids.append(doc_id)

    def search(self, query: str, top_k: int = 2) -> list[RetrievedDoc]:
        qvec = self.embedder.embed(query)
        # 每个文档取「最相关 chunk」作为代表，得到文档级得分
        best: dict[str, tuple[float, str]] = {}
        for doc_id, chunk in self.chunks:
            score = float(np.dot(qvec, self.embedder.embed(chunk)))
            if doc_id not in best or score > best[doc_id][0]:
                best[doc_id] = (score, chunk)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        return [RetrievedDoc(doc_id, chunk, score)
                for doc_id, (score, chunk) in ranked[:top_k]]