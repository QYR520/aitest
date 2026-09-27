"""Embedding 引擎：把文字变成向量，语义相近 -> 向量相近。

第 1 周核心：Embedding = 文字变向量坐标，语义相近则坐标相近。
这里提供两档实现（自动选优）：
    1. STEmbedder：真实中文语义模型 BAAI/bge-small-zh（本地已缓存，离线可跑）
    2. CharEmbedder：字符 n-gram 特征哈希（纯 numpy 降级，无任何模型依赖）

选优逻辑：优先加载 bge-small-zh，加载失败（缺模型/缺库）才降级到字符版。
这正好对应真实项目里的「embedding 模型降级处理」。
"""
import os

# 强制离线，避免 sentence-transformers 联网检查更新（模型已在本地）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import re

import numpy as np


class STEmbedder:
    """真实语义 embedding：BAAI/bge-small-zh（中文向量模型）。

    相比字符 n-gram，它能捕捉「同义改写」：如"退货期限是几天"和"7天内可退"向量很近。
    模型用类级单例缓存，所有实例共享，避免重复加载。
    """

    _model = None

    def _ensure_model(self):
        if STEmbedder._model is None:
            from sentence_transformers import SentenceTransformer
            STEmbedder._model = SentenceTransformer("BAAI/bge-small-zh")
        return STEmbedder._model

    def embed(self, text: str) -> np.ndarray:
        return self._ensure_model().encode(text, normalize_embeddings=True)

    def similarity(self, a: str, b: str) -> float:
        return float(np.dot(self.embed(a), self.embed(b)))


class CharEmbedder:
    """字符 n-gram 特征哈希的轻量 embedding（降级方案）。

    纯标准库 + numpy，不依赖任何模型。原理：中文里"退货"和"退货政策"
    共享"退货"等 n-gram，所以向量距离近。
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, token: str) -> int:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        return h % self.dim

    def _ngrams(self, text: str):
        text = re.sub(r"\s+", "", text)
        for ch in text:
            yield ch
        for i in range(len(text) - 1):
            yield text[i : i + 2]

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in self._ngrams(text):
            vec[self._bucket(token)] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def similarity(self, a: str, b: str) -> float:
        return float(np.dot(self.embed(a), self.embed(b)))


def get_embedder():
    """返回当前可用的最优 embedder：优先真实语义模型，失败降级字符版。"""
    try:
        e = STEmbedder()
        e._ensure_model()  # 立即触发加载，加载失败即刻降级
        return e
    except Exception:
        return CharEmbedder()