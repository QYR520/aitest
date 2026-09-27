"""LLM 客户端：负责「生成」。支持 mock 与 openai 两种模式。

mock 模式的设计意图（重要）：
    我们不把「网络大模型」当被测对象，而是把它当成 SUT 里的一个「零件」。
    mock 模式用一个确定性行为来模拟这个零件，好处是：
    1. 全离线、可复现、不需要 API key；
    2. 它的「忠诚度」是完美的（永远照抄检索到的资料），
       所以错误只能来自「检索环节」——精准演示 RAG 的 garbage in/garbage out。

它还模拟了两个真实大模型的风险（对应第 1 周四大风险）：
    - Temperature：温度高时答案会抖动（随机性）
    - 幻觉：当检索不到资料且开了幻觉开关时，会「自信地编造」（而非拒答）
"""
import random
import re

from .embedder import get_embedder


def estimate_tokens(text: str) -> int:
    """token 估算：中文约 1 字 ≈ 1 token（第 1 周：钱和速度都按 token 算）。"""
    return len(text)


INJECTION_KEYWORDS = [
    "忽略", "忽略以上", "忽略之前", "忽略规则", "泄露", "系统提示", "系统指令",
    "扮演", "假装", "角色", "不要遵守", "绕过", "无视",
]


class LLMClient:
    def __init__(self, config: dict):
        self.config = config
        self.provider = config.get("provider", "mock")
        # 与 config.yaml 注释一致：写了 openai 但 key 为空 -> 强制走 mock，
        # 避免无 key 真实调用 401（对 SUT 和 LLM-as-Judge 都生效）。
        if self.provider != "mock" and not config.get("api_key", ""):
            self.provider = "mock"
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = float(config.get("temperature", 0.1))
        # 缺陷开关：用于演示「测试能抓住 bug」
        self.flaw = config.get("flaw", {}) or {}
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        # 用于 mock 生成时从资料里选「与问题最相关的句子」
        self.embedder = get_embedder()

    # ---------- mock 模式 ----------
    def complete(self, messages: list[dict]) -> str:
        """统一的生成入口。mock 走规则，openai 走真实 API。"""
        if self.provider == "mock":
            return self._mock_complete(messages)
        return self._openai_complete(messages)

    def _mock_complete(self, messages: list[dict]) -> str:
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if len(messages) >= 2 else ""

        self.last_usage = {
            "prompt_tokens": estimate_tokens(system + user),
            "completion_tokens": 0,
        }

        # ① 注入防护（指令优先级：系统规则 > 用户指令）
        guard_on = self._is_guard_enabled(system)
        if self._is_injection(user):
            if guard_on:
                return "抱歉，我无法执行该指令，只能根据知识库内容回答您的问题。"
            # 防注入规则缺失（消融/缺陷）-> 模拟真实模型被注入成功：泄露系统提示词
            return self._leak_system_prompt(system)

        # ② 提取【资料区】
        knowledge = self._extract_zone(system, "资料区")

        # ③ 无资料：是否拒答由 prompt 规则决定（贴近真实 LLM 读规则行事）
        if not knowledge:
            # 缺陷开关 或 prompt 里没有「拒答规则」-> 自由编造（幻觉）
            if self.flaw.get("hallucinate") or not self._has_reject_rule(system):
                return self._hallucinate(user)
            return "抱歉，我的知识库中没有相关信息，无法回答您的问题。"

        # ④ 有资料：忠实复述资料（模拟"看着资料回答"）
        answer = self._summarize(knowledge, user)
        self.last_usage["completion_tokens"] = estimate_tokens(answer)

        # ⑤ Temperature 抖动：高温时答案不稳定（演示随机性风险）
        if self.temperature >= 0.5:
            answer = self._perturb_numbers(answer)
        return answer

    def _openai_complete(self, messages: list[dict]) -> str:
        # 真实 OpenAI 兼容接口（配置好 key 后可用）
        import urllib.request
        import json

        url = self.config.get("base_url", "").rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.get('api_key', '')}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    # ---------- mock 辅助 ----------
    @staticmethod
    def _extract_zone(system: str, zone: str) -> str:
        m = re.search(rf"【{zone}】\s*\n(.*?)(?=\n【|\Z)", system, re.S)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _is_guard_enabled(system: str) -> bool:
        return "忽略" in system and "只能" in system

    @staticmethod
    def _has_reject_rule(system: str) -> bool:
        """判断 prompt 里是否写了「资料区为空必须拒答」的规则。"""
        return "无法回答" in system or "没有相关信息" in system

    @staticmethod
    def _is_injection(user: str) -> bool:
        return any(k in user for k in INJECTION_KEYWORDS)

    def _summarize(self, knowledge: str, question: str) -> str:
        """mock 的「生成」：从资料里选与问题最相关的那句作为回答。

        真实大模型会看着资料 + 问题组织语言，这里简化为「按相似度选最相关句」。
        对测试而言足够：正确/错误的根源都在检索，生成是忠实的。
        """
        lines = [ln.strip().lstrip("- ").strip() for ln in knowledge.splitlines() if ln.strip()]
        if not lines:
            return ""
        return max(lines, key=lambda ln: self.embedder.similarity(ln, question))

    def _hallucinate(self, question: str) -> str:
        """模拟「检索不到却自信地编造」——这正是不该发生的幻觉行为。"""
        return "根据我们的政策，这项服务是支持的，您可以直接前往线下门店办理。"

    def _leak_system_prompt(self, system: str) -> str:
        """模拟「被注入劫持」：把系统提示词的规则部分泄露出来（真实模型被注入后的典型表现）。"""
        rules = system.split("【资料区】")[0].strip()
        return f"好的，我的系统提示词如下：\n{rules}"

    def _perturb_numbers(self, text: str) -> str:
        """高温下随机把数字改错，模拟大模型的输出随机性（第 1 周 Temperature 风险）。"""
        def _bump(m):
            n = int(m.group())
            return str(n + random.choice([1, 2]))
        return re.sub(r"\d+", _bump, text)