"""
对话级别奖励函数，供 custom_reward_function.path/name 加载。

NaviAgentLoop 在 AgentLoopOutput.extra_fields 中已经写入了 turn_scores（任务完成/
交互轮次奖励）和 tool_rewards（每次工具调用的步骤级奖励），这里直接复用这些已计算
好的信号做一次聚合，不重复计算工具/状态相关的逐步奖励逻辑。
"""

from typing import Any, Optional


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> float:
    extra_info = extra_info or {}

    turn_scores = extra_info.get("turn_scores") or []
    tool_rewards = extra_info.get("tool_rewards") or []

    if not turn_scores and not tool_rewards:
        return 0.0

    turn_score = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0
    tool_score = sum(tool_rewards) / len(tool_rewards) if tool_rewards else 0.0

    return turn_score + 0.1 * tool_score
