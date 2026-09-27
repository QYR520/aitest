"""评测 Runner：Test Runner 的核心循环。

对应最开始讲的那个骨架：
    遍历用例 -> 执行被测系统(SUT) -> 按维度打分 -> 汇总
只是把「== 断言」换成了「多维度打分」，把「函数」换成了「LLM 应用」。
"""
import time
from collections import Counter
from pathlib import Path

import yaml

from sut.embedder import get_embedder
from sut.retrieval import KnowledgeBase
from sut.llm import LLMClient
from sut.agent import CustomerServiceAgent
from eval.judges import Verdict, judge_exact, judge_semantic, LLMJudge
from eval.metrics import compute_retrieval
from eval.tracing import get_tracer
from eval.safety import evaluate_safety

REJECT_WORDS = ["抱歉", "无法", "没有相关信息"]


class TestRunner:
    __test__ = False  # 告诉 pytest 这不是测试类，避免误收集

    def __init__(self, config: dict, suite: dict, prompt_fn=None):
        self.config = config
        self.suite = suite
        self.embedder = get_embedder()
        self.kb = KnowledgeBase(config["sut"]["knowledge_base_dir"], self.embedder)
        self.llm = LLMClient(config["llm"])
        # 法官模型：可配置独立的（通常更强的）法官，缺省复用 SUT 的 llm
        self.judge = LLMJudge(LLMClient(config.get("llm_judge") or config["llm"]),
                              self.embedder)
        self.agent = CustomerServiceAgent(config["sut"], self.kb, self.llm,
                                          prompt_fn=prompt_fn)
        # 可观测性：把每个用例的执行轨迹(trace)上报 Langfuse（无 key 时自动 no-op）
        self.tracer = get_tracer(config.get("langfuse"), model=self.llm.model)

    # ---------- 主流程 ----------
    def run(self) -> dict:
        results = []
        for case in self.suite["cases"]:
            results.append(self.run_case(case))
        summary = self._summarize(results)
        self.tracer.flush()  # 把已上报的 trace 推送到 Langfuse
        return summary

    def run_case(self, case: dict) -> dict:
        history: list[str] = []
        final_answer = ""
        last_trace = None
        for turn in case["turns"]:  # 支持多轮
            t0 = time.perf_counter()
            answer, trace = self.agent.answer(history, turn)
            trace["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            history.append(turn)
            final_answer, last_trace = answer, trace

        golden = case.get("golden", {})
        verdicts: dict[str, Verdict] = {}

        # ① 内容判定（准确性）：三种 judge 由用例指定（决策树落地）
        judge_cfg = case.get("judge", {})
        jtype = judge_cfg.get("type", "semantic")
        if golden.get("answer") and jtype in ("exact", "semantic", "llm_judge"):
            if jtype == "exact":
                verdicts["accuracy"] = judge_exact(final_answer, golden["answer"])
            elif jtype == "semantic":
                verdicts["accuracy"] = judge_semantic(
                    final_answer, golden["answer"], self.embedder,
                    judge_cfg.get("threshold", 0.5),
                )
            else:
                question = " | ".join(case["turns"])
                verdicts["accuracy"] = self.judge.judge(
                    question, golden["answer"], final_answer,
                    judge_cfg.get("keywords"),
                )

        # ② 检索质量：RecR@k / MRR / nDCG（独立于生成）
        retrieval = None
        if golden.get("relevant_doc_ids") is not None and last_trace and last_trace["intent"] == "rag":
            retrieval = compute_retrieval(
                last_trace["retrieved_docs"],
                golden["relevant_doc_ids"],
                self.config["sut"].get("top_k", 2),
            )

        # ③ 工具正确性：选对工具 + 传对参数
        if golden.get("expected_tool"):
            exp = golden["expected_tool"]
            ok = (
                last_trace is not None
                and last_trace["intent"] == "tool"
                and last_trace["tool_name"] == exp["name"]
                and last_trace["tool_args"] == exp["args"]
            )
            got = f"工具={last_trace.get('tool_name') if last_trace else None} 参数={last_trace.get('tool_args') if last_trace else None}"
            verdicts["tool_correctness"] = Verdict(ok, 1.0 if ok else 0.0,
                                                   f"期望调用 {exp['name']}{exp['args']}，实际 {got}")

        # ④ 拒答（不可回答）：该拒答时拒了没
        if golden.get("expect_unanswerable"):
            rejected = any(w in final_answer for w in REJECT_WORDS)
            verdicts["rejection"] = Verdict(
                rejected, 1.0 if rejected else 0.0,
                "正确拒答" if rejected else f"未拒答（疑似幻觉）：「{final_answer}」",
            )

        # ⑤ 安全/注入
        if case.get("category") == "injection":
            passed, score, detail = evaluate_safety(final_answer, True)
            verdicts["safety"] = Verdict(passed, score, detail)

        # ⑥ 功能完成度（task_completion）：只要产出了非空回答且有正确意图就算「活干了」
        produced = bool(final_answer.strip())
        verdicts["task_completion"] = Verdict(produced, 1.0 if produced else 0.0, "正常作出回答")

        # 综合通过：所有 Verdict 维度都 passed（retrieval 是诊断指标，不强制）
        passed = all(v.passed for v in verdicts.values())

        result = {
            "id": case["id"],
            "category": case.get("category", ""),
            "question": " | ".join(case["turns"]),
            "answer": final_answer,
            "verdicts": verdicts,
            "retrieval": retrieval,
            "passed": passed,
            "trace": last_trace,
        }
        self.tracer.record_case(result)  # 上报 trace + 各维度打分到 Langfuse
        return result

    # ---------- 汇总 ----------
    def _summarize(self, results: list[dict]) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r["passed"])

        # 各维度通过率
        dim_totals: dict[str, int] = {}
        dim_passed: dict[str, int] = {}
        for r in results:
            for dim, v in r["verdicts"].items():
                dim_totals[dim] = dim_totals.get(dim, 0) + 1
                if v.passed:
                    dim_passed[dim] = dim_passed.get(dim, 0) + 1

        dim_rate = {d: round(dim_passed.get(d, 0) / dim_totals[d], 4) for d in dim_totals}

        # 成本统计（token）
        prompt_tokens = sum(
            (r["trace"] or {}).get("usage", {}).get("prompt_tokens", 0) for r in results)
        completion_tokens = sum(
            (r["trace"] or {}).get("usage", {}).get("completion_tokens", 0) for r in results)

        # 延迟
        latencies = [r["trace"]["latency_ms"] for r in results if r.get("trace") and "latency_ms" in r["trace"]]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        meta = self.suite.get("suite", {})  # 套件元信息在 suite: 子键下
        return {
            "suite": meta.get("name", ""),
            "description": meta.get("description", ""),
            "llm": {"provider": self.llm.provider, "model": self.llm.model,
                    "temperature": self.llm.temperature},
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "dim_pass_rate": dim_rate,
            "cost": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                     "total_tokens": prompt_tokens + completion_tokens},
            "avg_latency_ms": avg_latency,
            "results": results,
        }

    # ---------- 一致性测试：同输入跑 N 次，算一致率 ----------
    def run_consistency(self, case_id: str, n: int = 10) -> dict:
        case = next(c for c in self.suite["cases"] if c["id"] == case_id)
        answers = []
        for _ in range(n):
            history = []
            ans = ""
            for turn in case["turns"]:
                ans, _ = self.agent.answer(history, turn)
                history.append(turn)
            answers.append(ans)
        most_common, count = Counter(answers).most_common(1)[0]
        rate = count / n
        return {
            "case_id": case_id,
            "n": n,
            "consistency_rate": rate,
            "unique_answers": len(set(answers)),
            "most_common_answer": most_common,
            "sample_answers": answers[:5],
        }

    # ---------- Prompt A/B / 消融：换 prompt，比同一批用例的通过率 ----------
    def compare_prompts(self, case_ids: list[str], prompt_b) -> dict:
        """用默认 prompt(A) 和自定义 prompt(B) 各跑一批用例，对比通过率。

        控制变量：模型/温度/输入/检索全不变，只换 prompt -> 可归因（第 2 周铁律）。
        """
        cases = [c for c in self.suite["cases"] if c["id"] in case_ids]
        baseline = self._run_cases_with(cases, self.agent)

        agent_b = CustomerServiceAgent(self.config["sut"], self.kb, self.llm,
                                       prompt_fn=prompt_b)
        variant = self._run_cases_with(cases, agent_b)

        return {
            "cases": case_ids,
            "prompt_a_pass_rate": baseline / len(cases) if cases else 0,
            "prompt_b_pass_rate": variant / len(cases) if cases else 0,
        }

    def _run_cases_with(self, cases: list[dict], agent) -> int:
        passed = 0
        for case in cases:
            history = []
            ans = ""
            trace = None
            for turn in case["turns"]:
                ans, trace = agent.answer(history, turn)
                history.append(turn)
            golden = case.get("golden", {})
            if golden.get("expect_unanswerable"):
                if any(w in ans for w in REJECT_WORDS):
                    passed += 1
            elif case.get("category") == "injection":
                ok, _, _ = evaluate_safety(ans, True)
                if ok:
                    passed += 1
            elif golden.get("answer"):
                v = judge_semantic(ans, golden["answer"], self.embedder,
                                   case.get("judge", {}).get("threshold", 0.5))
                if v.passed:
                    passed += 1
        return passed