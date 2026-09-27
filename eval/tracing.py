"""可观测性（trace）上报：把评测执行轨迹发送到 Langfuse 可视化。

第 3 周「trace / 可观测性」落地。SUT 的 agent.answer 已经返回结构化的
执行轨迹（intent / 检索文档及分数 / 工具调用 / token / 延迟），这里把它映射
成 Langfuse v4 的「观测优先」数据模型：

    Trace（一个用例 = 一次完整执行）
      ├── Span(intent)           意图路由
      ├── Retriever(retrieve)    RAG 检索（文档 + 相似度分数）
      ├── Tool(tool_call)        工具调用（tool + 参数）
      ├── Generation(llm)        LLM 生成（model + token 用量）
      └── Score × N              各维度打分（accuracy/tool/...）

没有 key 时自动降级为 no-op：mock 模式零依赖、零联网、不影响评测主流程。
"""
from __future__ import annotations


class NoopTracer:
    """未启用可观测时的空实现：什么都不做。"""

    enabled = False

    def record_case(self, result: dict) -> None:
        pass

    def flush(self) -> None:
        pass


class LangfuseTracer:
    """把单个用例的测评结果（trace + verdicts）上报到 Langfuse。"""

    def __init__(self, cfg: dict, model: str | None = None):
        self.cfg = cfg or {}
        self.model = model
        self.enabled = True
        self._client = None

    def _get_client(self):
        if self._client is None:
            from langfuse import Langfuse, get_client  # 延迟 import：未启用时不加载

            host = self.cfg.get("host", "https://cloud.langfuse.com")
            Langfuse(
                public_key=self.cfg.get("public_key"),
                secret_key=self.cfg.get("secret_key"),
                # v4 里 base_url(API 请求) 与 host(span 导出) 是两个地址，自托管需同时指定
                base_url=host,
                host=host,
            )
            self._client = get_client()
        return self._client

    def record_case(self, result: dict) -> None:
        try:
            self._record(result)
        except Exception as e:  # 上报失败绝不能影响评测主流程
            print(f"[langfuse] 上报 {result.get('id')} 失败：{e}")

    def _record(self, result: dict) -> None:
        lf = self._get_client()
        tr = result.get("trace") or {}
        intent = tr.get("intent")
        question = result.get("question", "")
        answer = result.get("answer", "")

        # 根 observation 同时充当 trace 容器，input/output/name 会自动落到 trace 上
        with lf.start_as_current_observation(
            as_type="span",
            name=result.get("id", "case"),
            input=question,
            output=answer,
            metadata={"category": result.get("category", ""),
                      "passed": bool(result.get("passed"))},
        ):
            # ① 意图路由
            with lf.start_as_current_observation(
                as_type="span", name="intent",
                input=question, output={"intent": intent},
            ):
                pass

            if intent == "rag":
                # ② RAG 检索：文档 + 相似度分数（一眼看出检索质量）
                with lf.start_as_current_observation(
                    as_type="retriever", name="retrieve", input=question,
                    output={
                        "retrieved_docs": tr.get("retrieved_docs", []),
                        "scores": tr.get("retrieved_scores", []),
                        "used_docs": tr.get("used_docs", []),
                    },
                ):
                    pass

                # ③ LLM 生成：带 token 用量
                usage = tr.get("usage") or {}
                with lf.start_as_current_observation(
                    as_type="generation", name="llm_generate", model=self.model,
                    input={"prompt": tr.get("prompt", "")}, output=answer,
                    usage_details={
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                    },
                ):
                    pass
            elif intent == "tool":
                with lf.start_as_current_observation(
                    as_type="tool", name="tool_call", input=question,
                    output={"tool": tr.get("tool_name"), "args": tr.get("tool_args")},
                ):
                    pass

            # ④ 打分：每个维度一个 trace 级 Score（带分数 + 理由）
            for dim, v in (result.get("verdicts") or {}).items():
                lf.score_current_trace(name=dim, value=float(v.score), comment=v.detail)

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as e:
            print(f"[langfuse] flush 失败：{e}")


def get_tracer(cfg: dict, model: str | None = None):
    """按「是否有 key」决定是否启用：优先取 config，其次读环境变量。

    这样 CI 里只要注入 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 环境变量，
    无需把 key 写进仓库，就能自动上报 trace。
    """
    import os

    c = cfg or {}
    public_key = c.get("public_key") or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = c.get("secret_key") or os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = c.get("host") or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not (public_key and secret_key):
        return NoopTracer()
    try:
        import langfuse  # noqa: F401  检查依赖是否已安装
    except ImportError:
        print("[langfuse] 未安装 langfuse，跳过可观测上报（pip install langfuse）")
        return NoopTracer()
    return LangfuseTracer(
        {"public_key": public_key, "secret_key": secret_key, "host": host},
        model=model,
    )