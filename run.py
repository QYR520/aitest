# -*- coding: utf-8 -*-
"""LLM Test Runner 一键运行入口。

用法（在 phase5 目录下）：
    python run.py               # 跑全部评测 + 一致性测试，生成报告
    python run.py --ab          # 额外跑 Prompt 消融对比（有防护 vs 无防护）
"""
import os
import sys
import argparse

# 项目根目录（用当前文件所在位置定位，不依赖运行时 cwd）
BASE = os.path.dirname(os.path.abspath(__file__))
# 保证从任意目录运行都能 import sut / eval 包
sys.path.insert(0, BASE)

import yaml

from eval.runner import TestRunner
from eval.reporter import render_markdown, render_terminal, render_html


def load_config(path=None):
    """读配置，并把里面的相对路径统一转成基于项目根目录的绝对路径。

    这样无论从哪个目录运行 `python run.py`，config.yaml / 知识库 / 报告目录
    都能被正确定位（不再依赖 cwd）。
    """
    path = path or os.path.join(BASE, "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section, key in (("sut", "knowledge_base_dir"),
                         ("run", "suite_file"),
                         ("run", "report_dir")):
        if section in cfg and key in cfg[section] and not os.path.isabs(cfg[section][key]):
            cfg[section][key] = os.path.join(BASE, cfg[section][key])
    return cfg


def load_suite(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def prompt_no_guard(knowledge_text: str) -> str:
    """消融版 prompt：删掉「拒答规则」和「防注入规则」。

    用于 Prompt 测试的消融对比：同输入、同温度、只删指令，
    观察质量是否下降（第 2 周「控制变量只动一个」）。
    """
    return f"""你是客服助手，请根据【资料区】回答用户问题。

【资料区】
{knowledge_text}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab", action="store_true", help="跑 Prompt 消融对比")
    args = parser.parse_args()

    config = load_config()
    suite = load_suite(config["run"]["suite_file"])
    runner = TestRunner(config, suite)

    # ① 主评测
    summary = runner.run()
    summary["_k"] = config["sut"]["top_k"]

    # ② 一致性测试
    consistency = runner.run_consistency(
        config["run"]["consistency_case"], config["run"]["consistency_n"])

    # ③ 报告（Markdown + 自包含 HTML，双击即可浏览器查看）
    from datetime import datetime
    summary["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary["consistency_rate"] = consistency["consistency_rate"]
    summary["consistency_n"] = consistency["n"]

    md = render_markdown(summary)
    md += "\n\n## 一致性测试\n\n"
    md += (f"- 用例 {consistency['case_id']} 跑 {consistency['n']} 次，"
           f"一致率 {consistency['consistency_rate']*100:.1f}%，"
           f"唯一答案数 {consistency['unique_answers']}\n")

    print(render_terminal(summary))
    print("\n[一致性] 用例", consistency["case_id"],
          "跑", consistency["n"], "次，一致率",
          f"{consistency['consistency_rate']*100:.1f}%")

    os.makedirs(config["run"]["report_dir"], exist_ok=True)
    report_md = os.path.join(config["run"]["report_dir"], "report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(md)
    report_html = os.path.join(config["run"]["report_dir"], "report.html")
    with open(report_html, "w", encoding="utf-8") as f:
        f.write(render_html(summary))
    print("\n报告已生成：")
    print("  Markdown：", report_md)
    print("  可视化  ：", report_html)
    print("\n双击打开 " + report_html + " 即可在浏览器查看可视化结果")

    # ④ 可选：Prompt 消融对比
    if args.ab:
        unhang = ["TC08", "TC09", "TC14", "TC15"]  # 拒答/注入类用例
        ab = runner.compare_prompts(unhang, prompt_no_guard)
        print("\n[Prompt 消融] 有防护通过率", f"{ab['prompt_a_pass_rate']*100:.0f}%",
              "vs 无防护", f"{ab['prompt_b_pass_rate']*100:.0f}%",
              "（说明删掉的规则确实在起作用）")


if __name__ == "__main__":
    main()