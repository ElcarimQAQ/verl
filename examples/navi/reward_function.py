# Copyright 2025 Navi Agent Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Navigation Agent Reward Function for online RL training.
参考 swe_agent/reward.py 和 custom_reward_v7.py 的实现

Reward structure (for navi data sources):
  1.0       — 任务完成（导航成功启动/到达）
  0.5-0.9   — 部分完成（正确的工具调用序列，但未最终完成）
  0.2-0.5   — 有效工具调用，有进展
  0.1-0.2   — 有工具调用但效率低
  0.0       — 无工具调用或无效
  -0.1      — 过早终止（1-2轮就结束）
  -0.05     — 长时间无进展（>=10轮但无有效工具）

用法（在启动命令中指定）::

    custom_reward_function.path=examples/navi/reward_function.py \
    custom_reward_function.name=compute_score
"""

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# 导航工具名称列表
NAVI_TOOL_NAMES = {
    "poi_search", "navigation_start", "notify_user_msg", "navigation_route",
    "navigation_info_query", "navigation_memory", "navigation_pathPoint",
    "filter", "wiki_search", "reject"
}

# 标记为导航数据的 data_source 取值。
# 训练数据若使用通用的 "rule" 作为 data_source，也按导航数据处理，
# 避免回退到 default_compute_score 时抛出 NotImplementedError。
NAVI_DATA_SOURCES = {"navi", "navi_agent", "navigation", "rule"}


def _safe_float(x, default=0.0) -> float:
    """安全转换为float"""
    try:
        if x is None:
            return default
        x = float(x)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _parse_tool_calls_from_text(text: str) -> List[Dict]:
    """从文本中解析工具调用"""
    if not text or not isinstance(text, str):
        return []

    tool_calls = []
    # 匹配 {"name": "...", "arguments": {...}} 格式
    pattern = r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}'
    matches = re.findall(pattern, text)

    for name, args_str in matches:
        try:
            args = json.loads(args_str)
            tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            tool_calls.append({"name": name, "arguments": {}})

    return tool_calls


def _is_navi_data(ground_truth: Any, solution_str: str) -> bool:
    """判断是否为导航数据"""
    # 检查 ground_truth 中是否有导航工具调用
    if isinstance(ground_truth, dict):
        tool_calls = ground_truth.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    if name in NAVI_TOOL_NAMES:
                        return True

    # 检查 solution_str 中是否有导航工具调用
    tools = _parse_tool_calls_from_text(solution_str)
    for tool in tools:
        if tool.get("name") in NAVI_TOOL_NAMES:
            return True

    return False


def _check_tool_validity(tool_name: str, tool_args: dict) -> tuple:
    """检查工具调用的有效性"""
    required_params = {
        "poi_search": ["mode", "keyword"],
        "navigation_start": ["mode", "desLocationReference"],
        "notify_user_msg": ["msg"],
    }

    if tool_name in required_params:
        missing = [p for p in required_params[tool_name] if p not in tool_args]
        if missing:
            return False, f"Missing required params: {missing}"

    if tool_name == "poi_search":
        if tool_args.get("mode") not in ["关键词", "周边", "沿途"]:
            return False, f"Invalid search mode: {tool_args.get('mode')}"

    return True, "Valid"


def _calculate_tool_reward(tool_name: str, tool_args: dict, is_valid: bool) -> float:
    """计算工具调用的奖励"""
    if not is_valid:
        return -0.5

    base_rewards = {
        "poi_search": 0.3,
        "navigation_start": 0.5,
        "notify_user_msg": 0.2,
        "navigation_route": 0.2,
        "navigation_info_query": 0.1,
        "navigation_memory": 0.1,
        "navigation_pathPoint": 0.1,
        "filter": 0.1,
        "wiki_search": 0.2,
        "reject": 0.1,
    }

    reward = base_rewards.get(tool_name, 0.1)

    # 额外奖励：参数完整性
    if tool_name == "poi_search" and "keyword" in tool_args:
        keyword = tool_args.get("keyword", "")
        if keyword and len(str(keyword)) >= 2:
            reward += 0.1

    if tool_name == "notify_user_msg":
        msg = tool_args.get("msg", "")
        completion_signals = ["导航已开始", "已到达", "导航结束", "任务完成", "导航已完成",
                             "开始导航", "已为您规划", "导航已启动"]
        if any(signal in msg for signal in completion_signals):
            reward += 0.5

    return reward


def _detect_tool_usage(solution_str: str) -> dict:
    """检测模型使用了哪些工具"""
    text = solution_str or ""
    tool_calls = _parse_tool_calls_from_text(text)

    tool_names = {tc["name"] for tc in tool_calls}

    return {
        "used_poi_search": "poi_search" in tool_names,
        "used_navigation_start": "navigation_start" in tool_names,
        "used_notify_user": "notify_user_msg" in tool_names,
        "used_navigation_route": "navigation_route" in tool_names,
        "used_filter": "filter" in tool_names,
        "used_any_tool": len(tool_names) > 0,
        "num_tools": len(tool_calls),
        "tool_names": list(tool_names),
    }


def _check_task_completion(extra_info: Optional[dict]) -> tuple:
    """检查任务是否完成"""
    if not extra_info:
        return False, 0.0

    # 从 extra_info 获取任务完成信息
    is_completed = extra_info.get("task_completed", False)
    completion_score = _safe_float(extra_info.get("completion_score", 0.0))

    # 检查 turn_scores 中是否有完成信号
    turn_scores = extra_info.get("turn_scores", [])
    if turn_scores:
        max_turn_score = max(turn_scores) if turn_scores else 0.0
        if max_turn_score >= 0.8:  # 高分表示任务完成
            is_completed = True
            completion_score = max(completion_score, max_turn_score)

    return is_completed, completion_score


def _aggregate_tool_rewards(extra_info: Optional[dict]) -> float:
    """从 extra_info 聚合工具奖励"""
    if not extra_info:
        return 0.0

    # 优先使用 tool_rewards
    tool_rewards = extra_info.get("tool_rewards", [])
    if tool_rewards:
        return sum(tool_rewards) / max(len(tool_rewards), 1)

    # 其次使用 turn_scores
    turn_scores = extra_info.get("turn_scores", [])
    if turn_scores:
        return sum(turn_scores) / max(len(turn_scores), 1)

    return 0.0


def _extract_expected_tools_from_ground_truth(ground_truth: Any) -> List[Dict]:
    """从 ground_truth 中提取期望的工具调用"""
    expected_tools = []

    if isinstance(ground_truth, dict):
        # 直接 tool_calls 字段
        tool_calls = ground_truth.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if name:
                        expected_tools.append({"name": name, "arguments": args})
    elif isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
            return _extract_expected_tools_from_ground_truth(gt)
        except Exception:
            pass

    return expected_tools


# ---------------------------------------------------------------------------
# VERL-compatible compute_score entry point
# ---------------------------------------------------------------------------


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> float:
    """Custom reward function for Navigation Agent with tool-use shaping.

    Args:
        data_source: 数据来源标识
        solution_str: 模型输出的解决方案字符串
        ground_truth: 真实标签
        extra_info: 额外信息，包含多轮对话的上下文
            - num_turns: 对话轮数
            - tool_rewards: 每个工具调用的奖励列表
            - turn_scores: 每轮的分数列表
            - task_completed: 任务是否完成
            - completion_score: 完成分数

    Returns:
        float: 奖励分数
    """
    # 检查是否为导航数据
    # 支持多种判断方式：data_source, ground_truth 中的工具调用
    is_navi = data_source in NAVI_DATA_SOURCES

    # 如果 data_source 不是 navi，检查 ground_truth 是否包含导航工具
    if not is_navi:
        expected_tools = _extract_expected_tools_from_ground_truth(ground_truth)
        for tool in expected_tools:
            if tool.get("name") in NAVI_TOOL_NAMES:
                is_navi = True
                break

    # 如果仍然不是导航数据，使用默认计算方式
    if not is_navi:
        try:
            from verl.utils.reward_score import default_compute_score

            return default_compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **kwargs,
            )
        except NotImplementedError:
            # 未知 data_source 时不应中断整个训练，按 0 分处理并告警。
            logger.warning(
                "No reward function implemented for data_source=%r and it is not "
                "recognized as navi data; returning 0.0",
                data_source,
            )
            return 0.0

    # === 以下是导航数据的处理逻辑 ===

    # 获取多轮对话信息
    num_turns = 0
    if extra_info is not None:
        num_turns = int(extra_info.get("num_turns", 0) or 0)

    # 解析 ground_truth
    expected_tools = _extract_expected_tools_from_ground_truth(ground_truth)

    # 检测工具使用情况
    tools = _detect_tool_usage(solution_str)

    # 从 extra_info 获取实际执行的奖励
    tool_reward_from_execution = _aggregate_tool_rewards(extra_info)

    # 检查任务完成
    is_completed, completion_score = _check_task_completion(extra_info)

    # 任务完成: 高奖励
    if is_completed:
        score = 0.8 + _clip(completion_score * 0.2, 0.0, 0.2)
        logger.info(
            f"Navi reward: score={score:.2f} (task completed), turns={num_turns}, "
            f"tools={tools['tool_names']}, completion_score={completion_score:.2f}"
        )
        return score

    # 有工具执行奖励
    if tool_reward_from_execution > 0:
        score = _clip(tool_reward_from_execution, 0.0, 0.9)
        logger.info(
            f"Navi reward: score={score:.2f} (from execution), turns={num_turns}, "
            f"tools={tools['tool_names']}"
        )
        return score

    # 解析模型输出的工具调用并计算奖励
    predicted_tools = _parse_tool_calls_from_text(solution_str)

    if not predicted_tools:
        # 没有任何工具调用
        if num_turns <= 2:
            # 过早终止
            score = -0.1
        elif num_turns >= 10:
            # 长时间无进展
            score = -0.05
        else:
            score = 0.0
        logger.info(
            f"Navi reward: score={score:.2f} (no tools), turns={num_turns}"
        )
        return score

    # 计算工具调用奖励
    total_reward = 0.0
    error_penalty = -0.5
    theta_validity = 0.4
    theta_name = 0.3
    theta_args = 0.3

    for pred_tool in predicted_tools:
        pred_name = pred_tool.get("name", "")
        pred_args = pred_tool.get("arguments", {})

        # 只处理导航工具
        if pred_name not in NAVI_TOOL_NAMES:
            continue

        # 检查工具调用有效性
        is_valid, reason = _check_tool_validity(pred_name, pred_args)

        if not is_valid:
            total_reward += error_penalty
            continue

        # 计算基础奖励
        tool_reward = _calculate_tool_reward(pred_name, pred_args, is_valid)

        # 如果有期望的工具，检查匹配
        if expected_tools:
            best_match_score = 0.0
            for exp_tool in expected_tools:
                exp_name = exp_tool.get("name", "")
                exp_args = exp_tool.get("arguments", {})

                # 名称匹配
                name_match = 1.0 if pred_name == exp_name else 0.0

                # 参数匹配
                if name_match > 0:
                    arg_matches = 0
                    total_args = len(exp_args)
                    if total_args > 0:
                        for key, value in exp_args.items():
                            if key in pred_args:
                                if pred_args[key] == value:
                                    arg_matches += 1
                                elif isinstance(value, str) and value.lower() in str(pred_args.get(key, "")).lower():
                                    arg_matches += 0.5
                        arg_match = arg_matches / total_args
                    else:
                        arg_match = 1.0
                else:
                    arg_match = 0.0

                match_score = theta_name * name_match + theta_args * arg_match
                best_match_score = max(best_match_score, match_score)

            tool_reward = tool_reward * (theta_validity + best_match_score)

        total_reward += tool_reward

    # 聚合奖励
    navi_tools = [t for t in predicted_tools if t.get("name") in NAVI_TOOL_NAMES]
    if navi_tools:
        score = total_reward / max(len(navi_tools), 1)
    else:
        score = 0.0

    # 长时间无有效进展
    if num_turns >= 10 and not tools["used_navigation_start"] and score >= 0.0:
        score = _clip(score - 0.1, -0.1, 0.5)

    # 最终 clip
    score = _clip(score, -1.0, 1.0)

    logger.info(
        f"Navi reward: score={score:.2f}, turns={num_turns}, "
        f"tools={tools['tool_names']}, num_tools={tools['num_tools']}"
    )

    return score


# 别名
navi_conversation_reward_func = compute_score
