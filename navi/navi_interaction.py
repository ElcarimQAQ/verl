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
Navigation Interaction Module for user simulation.
参考 collabllm_interation.py 和 agent_execution_engine_v2.py 中的 UserMockEngine 实现
"""

import asyncio
import copy
import json
import logging
import math
import os
import re
from typing import Any, Optional
from uuid import uuid4

import litellm

from navi.interaction_base import BaseInteraction
from navi.tool_rules import NOTIFY_TOOLS, USER_VISIBLE_TOOL_RESULTS
from navi.utils import extract_json
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger("NaviInteraction")
# 默认设置为 INFO 级别，确保 start_interaction 等关键日志能够输出
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# 终止信号
TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"

_REQUIRED_RESPONSE_KEYS = {"current_status", "thought", "response"}
_SATISFACTION_KEY = "satisfaction"
_META_HINTS = ("```", "<|", "|>", "current_status", '"thought"', '"response"')
_TRANSIENT_MSG_HINTS = (
    "connection error",
    "connection reset",
    "timeout",
    "timed out",
    "service unavailable",
    "overloaded",
    "502",
    "503",
    "504",
    "temporarily unavailable",
)
_FALLBACK_FOLLOWUPS = (
    "还有别的方案吗？",
    "再帮我看看附近还有什么。",
    "继续吧。",
    "还有多远？",
    "麻烦再确认一下。",
    "好的，下一步呢？",
)


def _safe_satisfaction(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return 1.0 if number > 0 else 0.0


def _looks_like_meta(text: str) -> bool:
    return len(text) > 80 or any(hint.lower() in text.lower() for hint in _META_HINTS)


def _is_transient_error(error: Exception) -> bool:
    if type(error).__name__ in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "ServiceUnavailableError",
    }:
        return True
    message = str(error).lower()
    return any(hint in message for hint in _TRANSIENT_MSG_HINTS)


def _extract_response_json(text: str) -> Optional[dict[str, Any]]:
    """Return the last complete response object, ignoring quoted GT JSON."""
    try:
        result = extract_json(text)
        if isinstance(result, dict) and _REQUIRED_RESPONSE_KEYS <= result.keys():
            return result
    except Exception:
        pass

    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", text):
        try:
            result, _ = decoder.raw_decode(text[match.start() :])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(result, dict) and _REQUIRED_RESPONSE_KEYS <= result.keys():
            candidates.append(result)
    return candidates[-1] if candidates else None


# 用户模拟Prompt模板
USER_PROMPT_TEMPLATE = """你是一名车载导航系统的用户，正在与AI助手交互完成导航任务。你的目标是生成真实、自然的用户回复。

## 输入信息：
你将获得：
- 任务描述：你尝试完成的导航任务类型
- 参考目标：这可能包含用户的完整请求或期望的结果
- 聊天历史：你（作为用户）与AI之间的对话

输入：
<|任务描述开始（AI不可见）|>
{task_desc}
<|任务描述结束|>

<|参考目标开始（AI不可见）|>
{single_turn_prompt}
<|参考目标结束|>

<|期望结果开始（AI不可见，用于判断AI是否真正满足了你的需求）|>
{ground_truth}
<|期望结果结束|>

<|聊天历史开始|>
{chat_history}
<|聊天历史结束|>


## 指导原则：
- 保持角色：扮演人类用户，而非AI助手
- 自然交互：使用简洁、口语化的表达方式
- 信息补充：如果AI需要更多信息，提供必要的细节
- 任务导向：保持对话聚焦于导航目标
- 模拟真实用户：可能存在模糊表述、简短回复等真实用户行为

## 特殊情况：
- 当AI播报"开始导航"、"已为您规划路线"等表示导航开始的内容时，表示任务已完成
- 当AI播报"已到达目的地"、"导航结束"时，表示任务已完成
- 当AI无法满足需求时，可以终止对话

## 输出格式：
输出一个JSON对象，包含以下字段：
- "current_status": 简要描述AI当前提供的服务状态
- "thought": 你作为用户决定下一步说什么的思考过程
- "response": 你作为用户的回复内容

当任务完成或AI无法继续帮助时，使用 "{termination_signal}" 作为回复。

注意：确保JSON格式正确，包含所有必要字段。"""


class NaviInteraction(BaseInteraction):
    """Navigation user simulator used by the asynchronous agent loop."""

    def __init__(self, config: dict):
        super().__init__(config)
        model_config = copy.deepcopy(config)
        model_config.pop("enable_log", None)
        self.name = model_config.pop("name", "navi")
        self.user_model = model_config.pop("user_model")
        self.termination_signal = model_config.pop("termination_signal", TERMINATION_SIGNAL)
        self.num_retries = int(model_config.pop("num_retries", 3))
        if "max_tokens" in model_config:
            model_config["max_tokens"] = int(model_config["max_tokens"])
        if "temperature" in model_config:
            model_config["temperature"] = float(model_config["temperature"])
        extra_body = model_config.get("extra_body")
        if isinstance(extra_body, dict):
            template_kwargs = extra_body.get("chat_template_kwargs")
            if isinstance(template_kwargs, dict) and "enable_thinking" in template_kwargs:
                template_kwargs["enable_thinking"] = str(template_kwargs["enable_thinking"]).lower() == "true"
        self.user_model_kwargs = model_config
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[dict[str, Any] | str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        assert "single_turn_prompt" in kwargs, "single_turn_prompt is required in interaction_kwargs"
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "interaction_kwargs": kwargs,
            "fallback_count": 0,
        }
        return instance_id

    @rollout_trace_op
    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any] | str], **kwargs
    ) -> tuple[bool, str, float, dict]:
        last_message = messages[-1] if messages else None
        if isinstance(last_message, dict):
            assert last_message.get("role") in {"system", "assistant"}, "最后一条消息必须来自system或assistant"
        else:
            assert last_message is not None, "messages 不能为空"

        state = self._instance_dict[instance_id]
        interaction_kwargs = state["interaction_kwargs"]
        ground_truth = state.get("ground_truth")
        if isinstance(ground_truth, (dict, list)):
            ground_truth_text = json.dumps(ground_truth, ensure_ascii=False, indent=2)
        else:
            ground_truth_text = ground_truth or "（未提供，以参考目标为准）"
        env_diff_summary = kwargs.get("env_diff_summary") or "（无环境状态变化信息）"
        env_state_text = kwargs.get("env_state_text") or "（无当前环境状态信息）"
        prompt = USER_PROMPT_TEMPLATE.format(
            task_desc=interaction_kwargs.get("task_desc", "车载导航任务"),
            single_turn_prompt=interaction_kwargs["single_turn_prompt"],
            ground_truth=ground_truth_text,
            chat_history=self._parse_messages(messages, strip_sys_prompt=True),
            termination_signal=self.termination_signal,
        )
        prompt += (
            "\n\n<|环境状态差分开始（AI不可见）|>\n"
            f"{env_diff_summary}\n<|环境状态差分结束|>\n"
            "<|当前环境状态开始（AI不可见）|>\n"
            f"{env_state_text}\n<|当前环境状态结束|>\n"
            "请在输出 JSON 中额外给出 satisfaction（0 或 1）；只有工具真实执行且满足 ground_truth 才能为 1。"
        )

        response = ""
        satisfaction = None
        for attempt in range(self.num_retries):
            try:
                message = (
                    (
                        await litellm.acompletion(
                            model=self.user_model,
                            messages=[{"role": "user", "content": prompt}],
                            **self.user_model_kwargs,
                        )
                    )
                    .choices[0]
                    .message
                )
                raw_response = message.content
                if raw_response is None:
                    raw_response = getattr(message, "reasoning_content", None)
                    provider_fields = getattr(message, "provider_specific_fields", {}) or {}
                    raw_response = raw_response or provider_fields.get("reasoning_content")
            except Exception as error:
                if _is_transient_error(error):
                    await asyncio.sleep(min(2**attempt, 8))
                logger.warning("[NaviInteraction] user model error: %s", error)
                continue

            extracted = _extract_response_json(raw_response) if isinstance(raw_response, str) else None
            if isinstance(extracted, dict) and _REQUIRED_RESPONSE_KEYS <= extracted.keys():
                candidate = extracted.get("response")
                if isinstance(candidate, str):
                    response = candidate
                    satisfaction = _safe_satisfaction(extracted.get(_SATISFACTION_KEY))
                    state["fallback_count"] = 0
                    break
            plain = (raw_response or "").strip().strip('"').strip("'") if isinstance(raw_response, str) else ""
            if plain and not _looks_like_meta(plain):
                response = plain
                state["fallback_count"] = 0
                break

        is_fallback = not response
        if is_fallback:
            fallback_count = int(state.get("fallback_count", 0))
            if fallback_count == 0:
                response = interaction_kwargs.get("single_turn_prompt") or "请继续帮我处理导航任务。"
            else:
                response = _FALLBACK_FOLLOWUPS[(fallback_count - 1) % len(_FALLBACK_FOLLOWUPS)]
            state["fallback_count"] = fallback_count + 1

        state["response"] = response
        should_terminate = self.termination_signal in response
        reward = satisfaction if satisfaction is not None else 0.0
        metrics = {"has_satisfaction": satisfaction is not None, "is_fallback": is_fallback}
        return should_terminate, response, reward, metrics

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)

    def _parse_messages(self, messages, strip_sys_prompt=True) -> str:
        if messages is None:
            return ""
        normalized = []
        for message in messages:
            if isinstance(message, str):
                normalized.append({"role": "assistant", "content": message})
            elif isinstance(message, dict):
                normalized.append(message)
        if strip_sys_prompt:
            normalized = [message for message in normalized if message.get("role") != "system"]
        normalized = [self._fill_notify_content(self._remove_think_block(message)) for message in normalized]

        chat_lines = []
        tool_names_by_id: dict[str, str] = {}
        pending_tool_names: list[str] = []
        for message in normalized:
            role = message.get("role")
            if role == "assistant":
                chat_lines.append(f"**Assistant**: {message.get('content') or ''}")
                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function")
                    if not isinstance(function, dict) or not function.get("name"):
                        continue
                    tool_name = function["name"]
                    pending_tool_names.append(tool_name)
                    if tool_call.get("id") is not None:
                        tool_names_by_id[str(tool_call["id"])] = tool_name
                    if tool_name not in NOTIFY_TOOLS:
                        summary = self._tool_call_summary(tool_name, function.get("arguments"))
                        chat_lines.append(f"**[AI动作]** {summary}")
            elif role == "tool":
                call_id = message.get("tool_call_id")
                tool_name = tool_names_by_id.get(str(call_id)) if call_id is not None else None
                if tool_name is None and pending_tool_names:
                    tool_name = pending_tool_names.pop(0)
                elif tool_name in pending_tool_names:
                    pending_tool_names.remove(tool_name)
                if tool_name is None or tool_name in USER_VISIBLE_TOOL_RESULTS:
                    chat_lines.append(f"**Tool**: {message.get('content') or ''}")
            else:
                label = role.capitalize() if role else "Unknown"
                chat_lines.append(f"**{label}**: {message.get('content') or ''}")
        return "\n".join(chat_lines)

    @staticmethod
    def _fill_notify_content(message: dict) -> dict:
        if message.get("role") != "assistant":
            return message
        parts = []
        content = message.get("content")
        if isinstance(content, str):
            content = re.sub(r"<\|im_\w+\|>", "", content).strip()
            if content:
                parts.append(content)
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            if not isinstance(function, dict) or function.get("name") not in NOTIFY_TOOLS:
                continue
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if isinstance(arguments, dict) and isinstance(arguments.get("msg"), str) and arguments["msg"]:
                parts.append(arguments["msg"])
        if not parts:
            return message
        result = dict(message)
        result["content"] = "\n".join(parts)
        return result

    @staticmethod
    def _tool_call_summary(tool_name: str, arguments: Any) -> str:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        fields_by_tool = {
            "poi_search": ("keyword", "mode"),
            "charging_station_search": ("keyword", "mode"),
            "navigation_start": ("desLocationReference", "routeType", "mode"),
            "navigation_route": ("mode", "routeType"),
            "navigation_control": ("action",),
            "navigation_mapZoom": ("action",),
            "navigation_memory": ("operation", "name"),
            "filter": ("distance_min", "distance_max", "rating_min", "rating_max"),
            "wiki_search": ("query",),
            "memory_search": ("query",),
        }
        parts = [f"调用了 {tool_name}"]
        for field in fields_by_tool.get(tool_name, ()):
            value = arguments.get(field)
            if value not in (None, "", []):
                rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
                parts.append(f"{field}={rendered[:80]}")
        return ", ".join(parts)

    @staticmethod
    def _remove_think_block(message: dict) -> dict:
        content = message.get("content")
        if not isinstance(content, str):
            return message
        result = dict(message)
        result["content"] = re.sub(r"<tool_call>.*?uaiya>", "", content, flags=re.DOTALL).strip()
        return result
