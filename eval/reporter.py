"""报告生成：把评测结果渲染成 Markdown / HTML 报告 + 终端输出。"""
import html
from pathlib import Path

from eval.judges import Verdict
from eval.metrics import RetrievalMetrics


def _verdict_line(name: str, v: Verdict) -> str:
    mark = "通过" if v.passed else "失败"
    return f"{name}={mark}({v.score:.3f})"


def render_markdown(summary: dict) -> str:
    """渲染成 Markdown，可存 reports/*.md，也可直接展示。"""
    lines = []
    llm = summary["llm"]
    lines.append(f"# 评测报告：{summary['suite']}")
    lines.append("")
    lines.append(f"> {summary['description']}")
    lines.append("")
    lines.append(f"- LLM：{llm['provider']} / {llm['model']} / temperature={llm['temperature']}")
    lines.append(f"- 总用例：{summary['total']}　通过：{summary['passed']}　通过率：{summary['pass_rate']*100:.1f}%")
    lines.append(f"- 平均延迟：{summary['avg_latency_ms']} ms　总 token：{summary['cost']['total_tokens']}（输入 {summary['cost']['prompt_tokens']} / 输出 {summary['cost']['completion_tokens']}）")
    lines.append("")

    lines.append("## 各维度通过率")
    lines.append("")
    lines.append("| 维度 | 通过率 |")
    lines.append("|------|--------|")
    for dim, rate in summary["dim_pass_rate"].items():
        lines.append(f"| {dim} | {rate*100:.1f}% |")
    lines.append("")

    lines.append("## 用例明细")
    lines.append("")
    lines.append("| ID | 类别 | 问题 | 回答 | 判定 | 通过 |")
    lines.append("|----|------|------|------|------|------|")
    for r in summary["results"]:
        vtext = "；".join(_verdict_line(d, v) for d, v in r["verdicts"].items())
        if r.get("retrieval") and r["retrieval"].relevant:
            rm = r["retrieval"]
            vtext += f"；检索 recall@{summary.get('_k', 2)}={rm.recall_at_k} MRR={rm.mrr} nDCG={rm.ndcg}"
        mark = "✅" if r["passed"] else "❌"
        lines.append(f"| {r['id']} | {r['category']} | {r['question']} | {r['answer']} | {vtext} | {mark} |")
    lines.append("")

    return "\n".join(lines)


def render_terminal(summary: dict) -> str:
    """终端简版输出。"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"评测报告 | {summary['suite']} | {summary['llm']['provider']}/{summary['llm']['model']}")
    lines.append(f"总 {summary['total']} 条，通过 {summary['passed']} 条，通过率 {summary['pass_rate']*100:.1f}%")
    lines.append(f"平均延迟 {summary['avg_latency_ms']} ms | 总 token {summary['cost']['total_tokens']}")
    lines.append("-" * 60)
    for r in summary["results"]:
        vtext = "  ".join(f"{d}:{'✓' if v.passed else '✗'}" for d, v in r["verdicts"].items())
        if r.get("retrieval") and r["retrieval"].relevant:
            rm = r["retrieval"]
            vtext += f"  recall@{summary.get('_k',2)}={rm.recall_at_k}"
        flag = "PASS" if r["passed"] else "FAIL"
        lines.append(f"[{flag}] {r['id']} {r['category']}")
        lines.append(f"      问：{r['question']}")
        lines.append(f"      答：{r['answer']}")
        lines.append(f"      {vtext}")
    lines.append("=" * 60)
    return "\n".join(lines)


def render_html(summary: dict) -> str:
    """自包含 HTML 报告：内嵌 CSS，双击即可在浏览器查看，无需服务器。"""
    llm = summary["llm"]
    rate = summary["pass_rate"] * 100
    overall_color = "#16a34a" if summary["pass_rate"] >= 0.8 else "#dc2626"

    # 维度通过率 -> CSS 柱状图（纯 div 宽度比例）
    dim_bars = ""
    for dim, r in summary["dim_pass_rate"].items():
        pct = r * 100
        color = "#16a34a" if r >= 0.8 else ("#d97706" if r >= 0.5 else "#dc2626")
        dim_bars += f"""
        <div class="dim-row">
          <span class="dim-name">{html.escape(dim)}</span>
          <div class="bar-track"><div class="bar" style="width:{pct:.1f}%;background:{color}"></div></div>
          <span class="dim-pct">{pct:.0f}%</span>
        </div>"""

    # 用例明细表格
    rows = ""
    for r in summary["results"]:
        flag = "通过" if r["passed"] else "失败"
        cls = "pass" if r["passed"] else "fail"
        vtexts = "".join(
            f"<div class='verdict'><b>{html.escape(d)}</b> {('✓' if v.passed else '✗')} "
            f"<span class='vscore'>{v.score:.2f}</span></div>"
            for d, v in r["verdicts"].items()
        )
        if r.get("retrieval") and r["retrieval"].relevant:
            rm = r["retrieval"]
            vtexts += (f"<div class='verdict'><b>检索</b> recall@{summary.get('_k', 2)}={rm.recall_at_k}"
                       f" MRR={rm.mrr} nDCG={rm.ndcg}</div>")
        rows += f"""
        <tr class="{cls}">
          <td>{r['id']}</td>
          <td>{html.escape(r['category'])}</td>
          <td class="q">{html.escape(r['question'])}</td>
          <td class="a">{html.escape(r['answer'])}</td>
          <td class="v">{vtexts}</td>
          <td class="flag {cls}">{flag}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>AI 评测报告 - {html.escape(summary['suite'])}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; background: #f5f6f8; color: #1f2937; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; margin: 4px 0 2px; }}
  .sub {{ color: #6b7280; font-size: 13px; margin-bottom: 16px; }}
  .cards {{ display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }}
  .card {{ background: #fff; border-radius: 10px; padding: 16px 20px; flex: 1; min-width: 180px;
           box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .card .num {{ font-size: 30px; font-weight: 700; }}
  .card .lbl {{ font-size: 12px; color: #6b7280; margin-top: 2px; }}
  .section {{ background: #fff; border-radius: 10px; padding: 18px 20px; margin: 16px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  h2 {{ font-size: 16px; margin: 0 0 12px; }}
  .dim-row {{ display: flex; align-items: center; gap: 10px; margin: 8px 0; }}
  .dim-name {{ width: 130px; font-size: 13px; }}
  .bar-track {{ flex: 1; background: #e5e7eb; border-radius: 6px; height: 14px; overflow: hidden; }}
  .bar {{ height: 100%; border-radius: 6px; min-width: 2px; }}
  .dim-pct {{ width: 42px; text-align: right; font-size: 13px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eceef1;
           vertical-align: top; }}
  th {{ background: #f9fafb; font-size: 12px; color: #6b7280; }}
  tr.pass td.flag {{ color: #16a34a; font-weight: 700; }}
  tr.fail td.flag {{ color: #dc2626; font-weight: 700; }}
  tr.fail {{ background: #fff7f7; }}
  .q, .a {{ max-width: 260px; }}
  .verdict {{ font-size: 12px; margin: 2px 0; }}
  .vscore {{ color: #9ca3af; }}
  .meta {{ font-size: 12px; color: #6b7280; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>AI 评测报告 · {html.escape(summary['suite'])}</h1>
  <div class="sub">{html.escape(summary['description'])}</div>
  <div class="meta">LLM：{llm['provider']} / {llm['model']} / temperature={llm['temperature']}　
    评测时间：{summary.get('timestamp', '-')}</div>

  <div class="cards">
    <div class="card"><div class="num" style="color:{overall_color}">{rate:.1f}%</div>
      <div class="lbl">总通过率（{summary['passed']}/{summary['total']}）</div></div>
    <div class="card"><div class="num">{summary['avg_latency_ms']} ms</div>
      <div class="lbl">平均延迟</div></div>
    <div class="card"><div class="num">{summary['cost']['total_tokens']}</div>
      <div class="lbl">总 token（输入 {summary['cost']['prompt_tokens']} / 输出 {summary['cost']['completion_tokens']}）</div></div>
    <div class="card"><div class="num">{consistency_pct(summary)}%</div>
      <div class="lbl">一致性（跑 {summary.get('consistency_n', '-')} 次）</div></div>
  </div>

  <div class="section">
    <h2>各维度通过率</h2>
    {dim_bars}
  </div>

  <div class="section">
    <h2>用例明细（{summary['total']} 条）</h2>
    <table>
      <tr><th>ID</th><th>类别</th><th>问题</th><th>回答</th><th>判定</th><th>结果</th></tr>
      {rows}
    </table>
  </div>
</div>
</body>
</html>"""


def consistency_pct(summary: dict) -> float:
    """从 summary 里取一致率（若存在），否则返回 '-'。"""
    v = summary.get("consistency_rate")
    return round(v * 100, 0) if v is not None else 0