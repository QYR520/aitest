# -*- coding: utf-8 -*-
"""端到端测试：验证整个 Test Runner 跑通，且基线（无缺陷）下用例应通过。

这是「测试之上的测试」——确保评测框架本身是可靠的。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from eval.runner import TestRunner


def _load():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = yaml.safe_load(open(os.path.join(base, "config.yaml"), encoding="utf-8"))
    suite = yaml.safe_load(
        open(os.path.join(base, config["run"]["suite_file"]), encoding="utf-8"))
    return config, suite


def test_runner_importable():
    config, suite = _load()
    runner = TestRunner(config, suite)
    assert runner is not None


def test_all_cases_run():
    config, suite = _load()
    runner = TestRunner(config, suite)
    summary = runner.run()
    # 基线模式（无缺陷）下，所有用例都应通过
    assert summary["total"] == len(suite["cases"])
    assert summary["pass_rate"] == 1.0, "基线模式下有用例失败，请看报告"


def test_hallucination_flaw_is_caught():
    """缺陷注入：打开幻觉开关后，不可回答用例应被测试抓出失败。"""
    config, suite = _load()
    config["llm"]["flaw"] = {"hallucinate": True}
    runner = TestRunner(config, suite)
    summary = runner.run()
    # 打开幻觉后，不可回答用例（TC08/TC09）的拒答维度应失败
    failed_ids = {r["id"] for r in summary["results"] if not r["passed"]}
    assert {"TC08", "TC09"}.issubset(failed_ids), "幻觉缺陷未被测试捕获"


def test_consistency_runs():
    config, suite = _load()
    runner = TestRunner(config, suite)
    result = runner.run_consistency("TC01", n=5)
    assert result["n"] == 5
    assert 0 <= result["consistency_rate"] <= 1.0