# -*- coding: utf-8 -*-
"""实验历史对比报告生成器。

流程（Langfuse 拉取 + 本地缓存）：
    1. 从 Langfuse Public API 拉取全部 trace；
    2. 只保留带 metadata.run 标签的 trace（即每次实验跑），按 run 分组；
    3. 从 metadata.verdicts_json 还原「每个用例 × 每个维度」的打分与对错；
    4. 缓存到 reports/history.json；
    5. 生成自包含的 reports/compare.html（时间轴 + 维度对比 + 逐用例对错网格），
       双击即可在浏览器查看，无需服务器。

用法：
    python build_compare_report.py          # 拉最新数据并生成对比页
    python build_compare_report.py --offline   # 不联网，仅用本地缓存重新渲染
"""
import argparse
import base64
import html
import json
import os
import urllib.request
from datetime import datetime

import yaml

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE, "reports")
HISTORY_JSON = os.path.join(REPORT_DIR, "history.json")
COMPARE_HTML = os.path.join(REPORT_DIR, "compare.html")

# run 标签 -> 页面展示名（便于人类阅读）
RUN_LABELS = {
    "baseline": "基线（全部规则）",
    "exp1_hallucinate": "实验1｜幻觉开关",
    "exp2_threshold": "实验2｜检索阈值0.99",
    "exp3_del_reject": "实验3｜删拒答规则",
    "exp4_del_inject": "实验4｜删防注入规则",
    "exp5_temperature": "实验5｜温度0.8",
}


def load_config():
    with open(os.path.join(BASE, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_all_traces(pk, sk, host):
    """分页拉取全部 trace。auth 用 Basic（public_key:secret_key）。"""
    cred = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    base = host.rstrip("/") + "/api/public/traces"
    traces = []
    page = 1
    while True:
        url = f"{base}?page={page}&limit=100&orderBy=timestamp.asc"
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {cred}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        traces.extend(data.get("data", []))
        meta = data.get("meta", {})
        total_pages = int(meta.get("totalPages", 1) or 1)
        if page >= total_pages:
            break
        page += 1
    return traces


def summarize_run(run, traces):
    """把一个 run 的多条 trace 汇总成对比所需的统计量。"""
    total = passed = 0
    cases = {}
    dim_stats = {}  # dim -> {total, passed, score_sum}
    ts_min = None
    for t in traces:
        md = t.get("metadata") or {}
        p = bool(md.get("passed"))
        total += 1
        passed += int(p)
        cases[t.get("name")] = p
        ts = t.get("timestamp")
        if ts and (ts_min is None or ts < ts_min):
            ts_min = ts
        # verdicts_json 可能是 JSON 字符串，也可能已被 Langfuse 还原成 dict
        raw = md.get("verdicts_json") or {}
        if isinstance(raw, str):
            try:
                verdicts = json.loads(raw)
            except (TypeError, ValueError):
                verdicts = {}
        else:
            verdicts = raw or {}
        for dim, v in verdicts.items():
            d = dim_stats.setdefault(dim, {"total": 0, "passed": 0, "score_sum": 0.0})
            d["total"] += 1
            d["passed"] += int(bool(v.get("passed")))
            try:
                d["score_sum"] += float(v.get("score") or 0.0)
            except (TypeError, ValueError):
                pass

    return {
        "run": run,
        "label": RUN_LABELS.get(run, run),
        "time": ts_min,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "dim_pass_rate": {d: round(s["passed"] / s["total"], 4)
                          for d, s in dim_stats.items()},
        "dim_avg_score": {d: round(s["score_sum"] / s["total"], 4)
                          for d, s in dim_stats.items()},
        "cases": cases,
    }


def group_and_summarize(traces):
    """只保留带 run 标签的 trace，分组并汇总，按时间升序。"""
    grouped = {}
    for t in traces:
        run = (t.get("metadata") or {}).get("run")
        if run:
            grouped.setdefault(run, []).append(t)
    runs = [summarize_run(r, ts) for r, ts in grouped.items()]
    runs.sort(key=lambda r: r["time"] or "")
    return runs


# ---------- HTML 渲染 ----------
def rate_color(rate):
    if rate >= 0.9:
        return "#16a34a"
    if rate >= 0.7:
        return "#d97706"
    return "#dc2626"


def render_trend(runs):
    """总通过率折线（内联 SVG，无外部依赖）。"""
    n = len(runs)
    w, h, pad = 620, 200, 30
    inner_w, inner_h = w - pad * 2, h - pad * 2
    max_rate = 1.0
    pts = []
    for i, r in enumerate(runs):
        x = pad + (inner_w * i / (n - 1) if n > 1 else inner_w / 2)
        y = pad + inner_h * (1 - r["pass_rate"] / max_rate)
        pts.append((x, y))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{rate_color(runs[i]["pass_rate"])}"/>'
        f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-size="11" '
        f'fill="#374151">{runs[i]["pass_rate"]*100:.0f}%</text>'
        for i, (x, y) in enumerate(pts)
    )
    labels = "".join(
        f'<text x="{x:.1f}" y="{h - 8}" text-anchor="middle" font-size="10" fill="#6b7280">'
        f'{html.escape(runs[i]["label"])}</text>'
        for i, (x, y) in enumerate(pts)
    )
    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" style="background:#fff;border-radius:10px">
      <line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#e5e7eb"/>
      <text x="{pad-6}" y="{pad}" font-size="10" fill="#6b7280">100%</text>
      <text x="{pad-6}" y="{pad+inner_h}" font-size="10" fill="#6b7280">0%</text>
      <polyline points="{line}" fill="none" stroke="#2563eb" stroke-width="2"/>
      {dots}
      {labels}
    </svg>"""


def render_summary_table(runs):
    rows = ""
    for r in runs:
        dims = " · ".join(f"{d} {r['dim_pass_rate'].get(d, 0)*100:.0f}%"
                          for d in sorted(r["dim_pass_rate"]))
        rows += f"""
        <tr>
          <td class="nowrap">{html.escape(r['label'])}</td>
          <td class="nowrap muted">{r['time']}</td>
          <td class="nowrap"><b style="color:{rate_color(r['pass_rate'])}">{r['pass_rate']*100:.0f}%</b></td>
          <td>{r['passed']}/{r['total']}</td>
          <td class="muted dimlist">{html.escape(dims)}</td>
        </tr>"""
    return f"""
    <table>
      <tr><th>实验</th><th>时间</th><th>通过率</th><th>通过/总数</th><th>各维度通过率</th></tr>
      {rows}
    </table>"""


def render_dim_heatmap(runs):
    """维度 × run 热力图：单元格显示 通过率% 与 平均分。"""
    dims = sorted({d for r in runs for d in r["dim_pass_rate"]})
    head = "".join(f"<th>{html.escape(r['label'])}</th>" for r in runs)
    rows = ""
    for d in dims:
        cells = ""
        for r in runs:
            pr = r["dim_pass_rate"].get(d)
            sc = r["dim_avg_score"].get(d)
            if pr is None:
                cells += "<td class='na'>—</td>"
            else:
                cells += (f"<td style='background:{rate_color(pr)}22'>"
                          f"<b style='color:{rate_color(pr)}'>{pr*100:.0f}%</b>"
                          f"<div class='mutedsmall'>{sc:.2f}</div></td>")
        rows += f"<tr><td class='dimname'><b>{html.escape(d)}</b></td>{cells}</tr>"
    return f"""
    <table>
      <tr><th>维度</th>{head}</tr>
      {rows}
    </table>"""


def render_case_grid(runs):
    """逐用例对错网格：行=用例，列=run，✓绿/✗红。"""
    case_ids = sorted({cid for r in runs for cid in r["cases"]})
    head = "".join(f"<th>{html.escape(r['label'])}</th>" for r in runs)
    rows = ""
    for cid in case_ids:
        cells = ""
        for r in runs:
            p = r["cases"].get(cid)
            if p is None:
                cells += "<td class='na'>—</td>"
            elif p:
                cells += "<td class='pass'>✓</td>"
            else:
                cells += "<td class='fail'>✗</td>"
        rows += f"<tr><td class='dimname'>{html.escape(cid)}</td>{cells}</tr>"
    return f"""
    <table class="grid">
      <tr><th>用例</th>{head}</tr>
      {rows}
    </table>"""


def render_html(runs, generated_at):
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>AI 测评实验历史对比</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; background: #f5f6f8; color: #1f2937; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; margin: 4px 0 2px; }}
  .sub {{ color: #6b7280; font-size: 13px; margin-bottom: 18px; }}
  .section {{ background: #fff; border-radius: 10px; padding: 18px 20px; margin: 16px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  h2 {{ font-size: 16px; margin: 0 0 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 7px 9px; border-bottom: 1px solid #eceef1;
           vertical-align: middle; }}
  th {{ background: #f9fafb; font-size: 11px; color: #6b7280; }}
  td.pass {{ color: #16a34a; font-weight: 700; text-align: center; }}
  td.fail {{ color: #dc2626; font-weight: 700; text-align: center; background: #fff1f1; }}
  td.na {{ color: #c0c4cc; text-align: center; }}
  td.dimname {{ font-size: 12px; white-space: nowrap; }}
  td.muted, .muted {{ color: #6b7280; font-size: 12px; }}
  .mutedsmall {{ color: #9ca3af; font-size: 11px; }}
  .nowrap {{ white-space: nowrap; }}
  td.dimlist {{ font-size: 11px; }}
  table.grid th {{ position: sticky; top: 0; background: #f9fafb; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>AI 测评 · 实验历史对比</h1>
  <div class="sub">数据来源：Langfuse（按 run 标签分组） · 生成时间 {html.escape(generated_at)} ·
    每个用例/each 维度展示「通过率% = 对错」与「平均分 = 打分」</div>

  <div class="section">
    <h2>总通过率随时间（次数序）变化</h2>
    {render_trend(runs)}
  </div>

  <div class="section">
    <h2>各实验汇总</h2>
    {render_summary_table(runs)}
  </div>

  <div class="section">
    <h2>各维度 × 实验（通过率 / 平均分）</h2>
    {render_dim_heatmap(runs)}
  </div>

  <div class="section">
    <h2>逐用例对错（行=用例，列=实验）</h2>
    {render_case_grid(runs)}
  </div>
</div>
</body>
</html>"""


def refresh(offline: bool = False):
    """拉取 Langfuse（或读缓存）→ 写 history.json → 写 compare.html。

    返回 payload（含 runs），供 CLI 与 Web 后端复用；无数据时返回 None。
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    runs = None
    if not offline:
        cfg = load_config()
        lf = cfg.get("langfuse", {})
        pk, sk, host = lf.get("public_key", ""), lf.get("secret_key", ""), lf.get("host", "")
        if not (pk and sk):
            print("未配置 langfuse key，尝试用本地缓存渲染…")
        else:
            try:
                traces = fetch_all_traces(pk, sk, host)
                runs = group_and_summarize(traces)
                print(f"从 Langfuse 拉取 {len(traces)} 条 trace，"
                      f"识别到 {len(runs)} 个实验 run")
            except Exception as e:
                print(f"拉取 Langfuse 失败：{e}，回落本地缓存…")

    if not runs:
        if os.path.exists(HISTORY_JSON):
            with open(HISTORY_JSON, encoding="utf-8") as f:
                runs = json.load(f).get("runs", [])
            print("已从本地缓存 history.json 加载")
        if not runs:
            print("没有可用的对比数据（既无 Langfuse 数据也无本地缓存）。")
            return None

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {"generated_at": generated_at, "runs": runs}

    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(COMPARE_HTML, "w", encoding="utf-8") as f:
        f.write(render_html(runs, generated_at))

    print("\n对比报告已生成：")
    print("  缓存  ：", HISTORY_JSON)
    print("  可视化：", COMPARE_HTML)
    print("\n双击打开 " + COMPARE_HTML + " 即可在浏览器查看")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="不联网，仅用本地缓存重新渲染")
    args = ap.parse_args()
    refresh(offline=args.offline)


if __name__ == "__main__":
    main()