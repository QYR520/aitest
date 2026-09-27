"""检索质量指标 + 幻觉/一致率计算。

对应第 2 周「RAG 测试/检索质量」与「幻觉/事实性」的公式。
高频面试陷阱：Recall@k 的分母是「相关文档总数」，不是 k！
"""
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    ndcg: float
    retrieved: list[str]
    relevant: list[str]


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """该捞的捞全没有：top-k 里命中的相关数 / 相关总数。分母是相关总数！"""
    if not relevant:
        return 1.0  # 没有相关文档，视为不适用
    topk = retrieved[:k]
    hits = [d for d in topk if d in relevant]
    return len(hits) / len(relevant)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    """第一个「该中的」文档排第几：位置倒数。排第1->1，排第4->1/4。"""
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg(retrieved: list[str], relevant: list[str]) -> float:
    """按位置打折的加权：既中、又排得对才高分。这里是简化版 DCG/IDCG。"""
    def dcg(seq):
        return sum((1.0 if d in relevant else 0.0) / (1.0 if i == 0 else __import__("math").log2(i + 2))
                   for i, d in enumerate(seq))
    ideal = sorted(relevant, key=lambda d: d)  # 理想排序：相关全都在前
    idcg = dcg(ideal)
    return dcg(retrieved) / idcg if idcg > 0 else 0.0


def compute_retrieval(retrieved: list[str], relevant: list[str], k: int) -> RetrievalMetrics:
    if not relevant:
        return RetrievalMetrics(1.0, 1.0, 1.0, retrieved, relevant)
    return RetrievalMetrics(
        recall_at_k=round(recall_at_k(retrieved, relevant, k), 4),
        mrr=round(mrr(retrieved, relevant), 4),
        ndcg=round(ndcg(retrieved, relevant), 4),
        retrieved=retrieved,
        relevant=relevant,
    )