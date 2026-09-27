"""工具层：客服 Agent 可调用的工具（Function Calling 的「执行」部分）。

第 1 周学过：Function Calling 里 LLM 负责「决策」（我要调哪个工具），
程序负责「执行」。这里的 get_order_status 就是被调用的真实工具。
测试「工具正确性」就是看 Agent 有没有选对工具、传对参数。
"""

# mock 订单数据库：模拟后端真实的订单状态查询
ORDER_DB = {
    "A12345": "正在配送中",
    "B88888": "已签收",
    "C00001": "已发货",
}

# 工具说明书（schema）：真实场景下会注册给 LLM 看
TOOL_SCHEMAS = {
    "get_order_status": {
        "name": "get_order_status",
        "description": "根据订单号查询订单物流状态",
        "parameters": {"order_id": "string，订单号"},
    }
}

# 订单号正则：用于意图识别（mock 模式下模拟 LLM 的决策）
ORDER_ID_PATTERN = r"[A-Z]\d{5}"


def call_tool(name: str, args: dict) -> str:
    """执行工具并返回结果（模拟真实后端接口调用）。"""
    if name == "get_order_status":
        order_id = args.get("order_id", "")
        status = ORDER_DB.get(order_id, "未找到该订单")
        return f"订单 {order_id} 状态：{status}"
    return "没有这个工具"