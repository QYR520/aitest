"""判定方法（Judge）：回答「这条输出怎么打分」。

对应第 2 周「准确性测试」的三种打分方式，选型决策树：
    答案唯一/格式固定 -> 硬真值对比（exact）
    答案多说法但意思固定 -> 语义相似度（semantic）
    答案需要综合判断 -> LLM-as-Judge（llm_judge）
    能硬判断绝不用软判断。

每种判定返回一个 Verdict（通过？分数？理由），而不是简单 True/False，
这正是「AI 测试按维度打分」与传统测试「断言对错」的区别。
"""
import json
import re
from dataclasses import dataclass

from sut.embedder import CharEmbedder, get_embedder


@dataclass
class Verdict:
    passed: bool
    score: float      # 0~1，越接近 1 越好
    detail: str = ""


def judge_exact(actual: str, expected: str) -> Verdict:
    """硬真值：期望答案是否字面出现在实际回答里。

    适用：答案唯一且格式固定（如工具返回、固定拒答语）。
    """
    if expected and expected in actual:
        return Verdict(True, 1.0, f"命中硬真值「{expected}」")
    return Verdict(False, 0.0, f"未命中「{expected}」，实际：「{actual}」")


def judge_semantic(actual: str, expected: str, embedder: CharEmbedder,
                   threshold: float = 0.5) -> Verdict:
    """语义相似度：把两者转向量算余弦，超过阈值判通过。

    适用：答案有多种说法但意思固定（如"7天内可退"vs"一周内可退"）。
    """
    score = embedder.similarity(actual, expected)
    return Verdict(
        score >= threshold, score,
        f"语义相似度 {score:.3f}（阈值 {threshold}）",
    )


def judge_faithfulness(answer: str, knowledge_text: str,
                       embedder: CharEmbedder, threshold: float = 0.5) -> Verdict:
    """忠实度（反幻觉）：回答是否编造了知识库里没有的内容。

    对应业界 RAGAS 的 faithfulness——把回答拆成「事实声明」逐条对照检索到的
    资料，看有没有无依据的编造。mock 项目用「回答 vs 资料」语义相似度近似：

    1. 拒答语（抱歉/无法/没有相关信息）→ 没有任何事实声明，天然忠实；
    2. 知识区为空却给了实质回答 → 典型幻觉（编造了库里不存在的事实）；
    3. 知识区非空 → 取回答与资料每条的最大相似度作为忠实度，低于阈值判编造。

    注意与 rejection 的区别：rejection 测「行为上该不该拒答」，faithfulness
    测「内容上有没有编造」——是两把不同的尺子。
    """
    REJECT = ("抱歉", "无法", "没有相关信息")
    if any(w in answer for w in REJECT):
        return Verdict(True, 1.0, "拒答，未编造任何事实")
    lines = [ln.strip() for ln in (knowledge_text or "").splitlines() if ln.strip()]
    if not lines:
        return Verdict(False, 0.0, f"知识区为空却回答「{answer[:30]}」，疑似幻觉")
    sim = max(embedder.similarity(answer, ln) for ln in lines)
    return Verdict(
        sim >= threshold, round(sim, 4),
        f"忠实度 {sim:.3f}（对资料最大相似度，阈值 {threshold}）",
    )


def judge_llm(actual: str, expected: str, embedder: CharEmbedder,
              keywords: list[str] | None = None) -> Verdict:
    """LLM-as-Judge：让「更强的模型」按标准打分。

    真实做法：把「标准答案 + 打分规则 + 待评回答」一起发给一个更强的 LLM，
    让它输出 0-1 分和理由。这里 mock 成「关键词命中 + 语义相似度」的合成分，
    用于离线演示「判定交给模型」这条链路；换成真实接口时只改这一处即可。
    """
    sim = embedder.similarity(actual, expected)
    kw_hit = 0.0
    if keywords:
        hit = sum(1 for k in keywords if k in actual)
        kw_hit = hit / len(keywords)
    score = round(0.6 * sim + 0.4 * kw_hit, 3)
    return Verdict(
        score >= 0.5, score,
        f"LLM裁判合成分 {score:.3f}（相似度 {sim:.3f} + 关键词 {kw_hit:.2f}）",
    )


class LLMJudge:
    """LLM-as-Judge：真正让一个 LLM 当裁判，按 Rubric 对回答打分。

    与上面 judge_llm（纯本地「关键词+相似度」合成）的区别：
    这里把「问题 + 标准答案 + 待评回答」组织成 prompt 发给一个 LLM，
    并解析它返回的 JSON 打分结构 —— 这才是业界说的 LLM-as-Judge。

    两种模式：
    - openai（真实）：调真实接口，LLM 输出 {"score","pass","reason"}，我们解析。
    - mock（离线）：降级为 judge_llm 的合成打分，保证无 key 也能跑。
    """

    # Rubric 是关键：给法官「明确的打分标准」，否则分数不可复现（面试必问）
    RUBRIC = (
        "你是客观的 AI 回答质量评审员。请严格依据【标准答案】判断【模型回答】是否正确。\n"
        "\n"
        "【问题】{question}\n"
        "【标准答案】{gold}\n"
        "【模型回答】{actual}\n"
        "\n"
        "评分规则：\n"
        "1. 含义与标准答案一致（措辞不同可接受）-> score 0.8~1.0，pass=true\n"
        "2. 含义部分正确但遗漏关键信息 -> score 0.5~0.79，pass=true\n"
        "3. 答非所问 / 意思相反 / 含错误事实 -> score 0~0.49，pass=false\n"
        "\n"
        "只输出一行 JSON，禁止输出任何其它文字：\n"
        '{{"score": <0到1小数>, "pass": <true/false>, "reason": "<一句话理由>"}}'
    )

    def __init__(self, llm, embedder=None):
        self.llm = llm
        self.embedder = embedder or get_embedder()

    def judge(self, question: str, gold: str, actual: str,
              keywords: list[str] | None = None) -> Verdict:
        if self.llm.provider == "mock":
            # 离线：沿用本地合成判定（行为确定、可复现）
            return judge_llm(actual, gold, self.embedder, keywords)
        return self._real_judge(question, gold, actual)

    def _real_judge(self, question: str, gold: str, actual: str) -> Verdict:
        prompt = self.RUBRIC.format(question=question, gold=gold, actual=actual)
        messages = [
            {"role": "system", "content": "你是客观的评测裁判，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = self.llm.complete(messages)
        except Exception as e:
            return Verdict(False, 0.0, f"法官调用失败：{e}")
        data = self._parse_json(raw)
        if data is None:
            return Verdict(False, 0.0, f"法官输出无法解析为 JSON：{raw[:120]}")
        score = float(data.get("score", 0.0))
        passed = bool(data.get("pass", score >= 0.5))
        reason = data.get("reason", "")
        return Verdict(passed, score, f"[LLM裁判] {reason}（score={score:.2f}）")

    @staticmethod
    def _parse_json(raw: str):
        """从 LLM 输出里稳健地抠出 JSON（LLM 常带 ```json 或前后废话）。"""
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, TypeError):
            return None


JUDGES = {
    "exact": judge_exact,
    "semantic": judge_semantic,
    "llm_judge": judge_llm,
}