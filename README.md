# LLM Test Runner（第 3 周项目）

一个从头搭起来的 **AI 测试运行器**：喂测试用例 → 调被测的客服 Agent → 按维度打分 → 出报告。

核心价值：把第 1、2 周学的「AI 测试 8 类方法 + 判定维度」全部落地成能跑、能测、能抓 bug 的代码。

## 交互式 Web 测试台（推荐：点按钮测，可视化看结果）

```bash
python web_server.py
```

浏览器打开 **http://localhost:8000**，4 个功能页：

| 页面 | 玩法 | 验证的知识 |
|------|------|-----------|
| 🔍 RAG 检索实验室 | 输入问题 + 拖 Top-K 滑块 → 看排位表 + 相似度柱状图 + 命中/拒答提示 | 检索质量、Top-K、相似度阈值 |
| ⚖️ Prompt 对比器 | 编辑自定义提示词 → 对比 A/B 通过率（删掉规则 = 消融实验） | Prompt 测试、消融、控制变量 |
| 🎲 一致性测试台 | 调温度 + 次数 → 看一致率和唯一答案数 | 一致性测试、温度混杂因素 |
| 📊 全量评测 | 一键跑 15 条用例 → 通过率 + 维度柱状图 + 逐条判定 | LLM Test Runner 闭环 |

纯标准库（http.server）零依赖，**不需要 streamlit/flask**。配置想接真实模型：改 config.yaml 的 provider/base_url/api_key。

# CLI 方式（备选）

在 `phase5` 目录下：

```bash
# 一键跑全部评测 + 一致性测试，生成报告
python run.py

# 额外跑 Prompt 消融对比（有防护 vs 无防护）
python run.py --ab

# 跑端到端自测（4 个测试）
pytest
```

跑完生成两份报告在 `reports/` 下：
- `report.html`：**可视化报告，双击在浏览器打开**（通过/失败红绿标注 + 维度通过率柱状图 + 检索指标）
- `report.md`：Markdown 版本

mock 模式**不需要 API key、不需要联网**，环境里有 numpy + PyYAML 就能跑通。

## 目录结构

```
phase5/
├── run.py                     # 一键入口
├── config.yaml                # 配置（含缺陷开关、温度）
├── sut/                       # ★ 被测系统（不是测试，是要被测试的对象）
│   ├── embedder.py            #   Embedding：文字转向量（字符 n-gram 特征哈希）
│   ├── retrieval.py           #   RAG 检索：知识库相似度 top-k
│   ├── tools.py               #   工具：get_order_status（Function Calling 的执行部分）
│   ├── llm.py                 #   LLM 客户端：mock / openai，含幻觉、温度模拟
│   └── agent.py               #   Agent 主循环：意图路由 + 检索 + 拼 prompt + 生成
├── eval/                      # ★ 测试层（评测）
│   ├── runner.py              #   Test Runner 核心循环 + 一致性 + Prompt A/B
│   ├── judges.py              #   判定方法：硬真值 / 语义相似度 / LLM-as-Judge
│   ├── metrics.py             #   Recall@k / MRR / nDCG
│   ├── safety.py              #   注入检测
│   └── reporter.py            #   报告生成
├── data/
│   ├── knowledge_base/        #   知识库 4 篇（D1退货/D2运费/D3保修/D4支付）
│   └── test_suites/           #   评测集（15 条用例）
└── tests/test_e2e.py          #   端到端自测
```

## 前两周知识点 → 代码落地对照

| 学过的知识 | 落在哪 |
|-----------|--------|
| Embedding / 向量相似度 | `sut/embedder.py` |
| RAG（检索→增强→生成） | `sut/retrieval.py` + `sut/agent.py` |
| Function Calling + Agent | `sut/tools.py` + `sut/agent.py`（意图路由） |
| Temperature 随机性 | `sut/llm.py` 里 `_perturb_numbers`，温度 ≥0.5 答案抖动 |
| 幻觉（Hallucination） | `sut/llm.py` 缺陷开关 `flaw.hallucinate` |
| 判定三选一（硬真值/语义/LLM裁判） | `eval/judges.py` |
| Recall@k / MRR / nDCG | `eval/metrics.py` |
| 不可回答测试（拒答） | 评测集 TC08/09 + `runner.py` 的 rejection 判定 |
| Prompt Injection / 红队 | 评测集 TC14/15 + `eval/safety.py` |
| 一致性测试（跑 N 次算一致率） | `runner.py` 的 `run_consistency` |
| Prompt 测试（A/B、消融） | `runner.py` 的 `compare_prompts` |
| 成本（token 统计）/ 延迟 | 每个用例 trace 里的 `usage` + `latency_ms` |

## 三个「会出错」的演示

想看到「测试真的能抓 bug」，改 `config.yaml` 里的开关，再跑一次对比：

1. **幻觉缺陷**：`flaw.hallucinate: true` → 不可回答用例（TC08/09）从 PASS 变 FAIL
2. **温度导致不一致**：`temperature: 0.8` → 一致性测试的一致率从 100% 掉下来
3. **消融掉防护规则**：`python run.py --ab` → 无防护 prompt 的通过率明显变低

## 核心思想（面试一句话）

传统测试 `== 断言对错`；AI 测试对同一条输出**按维度打分**（准确性/幻觉/工具/检索…），
一条输出可能「相关性过、准确性挂」，从而定位是「检索错了」还是「prompt 没约束好」。

## 可观测性（trace → Langfuse）

跑评测时，每个用例的执行轨迹（意图路由 / 检索命中的文档及分数 / 工具调用 / token 用量 / 延迟）
会被上报到 **Langfuse** 可视化平台，你可以点开每一条 trace 看 SUT 内部到底怎么跑的。

开启（默认关闭，随便跑 mock 不受影响）：

1. 到 <https://cloud.langfuse.com> 注册并创建项目，复制 `public_key`(pk-lf-…) 和 `secret_key`(sk-lf-…)
2. `config.yaml` 里把 `langfuse.enabled` 改 `true`，填入两个 key
3. `pip install langfuse`
4. 重新 `python run.py`，打开 Langfuse 页面就能看到评测的 trace 和每个维度的 Score