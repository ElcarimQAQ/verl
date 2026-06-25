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
import os
import re
from typing import Any, Optional
from uuid import uuid4

import litellm

from navi.interaction_base import BaseInteraction
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger('NaviInteraction')
# 默认设置为 INFO 级别，确保 start_interaction 等关键日志能够输出
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# 终止信号
TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"

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
    """
    导航系统的用户交互模拟器

    主要功能：
    - `start_interaction`: 启动一个交互实例
    - `generate_response`: 生成用户回复
    - `finalize_interaction`: 结束交互实例
    """

    def __init__(self, config: dict):
        super().__init__(config)
        _config = copy.deepcopy(config)

        _config.pop("enable_log", None)

        self.name = _config.pop("name")
        self.user_model = _config.pop("user_model")

        self.termination_signal = _config.pop("termination_signal", TERMINATION_SIGNAL)
        self.num_retries = _config.pop("num_retries", 3)

        self.user_model_kwargs = _config

        self._instance_dict = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        """启动交互实例"""
        logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
        logger.info(f"[NaviInteraction] start_interaction called with instance_id={instance_id}, kwargs={kwargs}")
        
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        self.interaction_kwargs = kwargs
        logger.info(f"[NaviInteraction] start_interaction completed, instance_id={instance_id}, interaction_kwargs={self.interaction_kwargs}")
        assert "single_turn_prompt" in kwargs, "single_turn_prompt is required in interaction_kwargs"
        return instance_id

    @rollout_trace_op
    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        """生成用户回复"""
        assert messages[-1]["role"] in ["system", "assistant"], (
            "最后一条消息必须来自system或assistant"
        )

        chat_history = self._parse_messages(messages, strip_sys_prompt=True)
        prompt = USER_PROMPT_TEMPLATE.format(
            task_desc=self.interaction_kwargs.get("task_desc", "车载导航任务"),
            single_turn_prompt=self.interaction_kwargs["single_turn_prompt"],
            chat_history=chat_history,
            termination_signal=self.termination_signal,
        )

        response = ""
        for i in range(self.num_retries):
            try:
                full_response = (
                    (
                        await litellm.acompletion(
                            model=self.user_model,
                            messages=[{"role": "user", "content": prompt}],
                            **self.user_model_kwargs,
                        )
                    )
                    .choices[0]
                    .message.content
                )
            except litellm.RateLimitError as e:
                logger.warning(f"[NaviInteraction] RateLimitError: {e}. Retrying...")
                await asyncio.sleep(max(2**i, 60))
                continue
            except Exception as e:
                logger.exception(f"[NaviInteraction] Error: {e}")
                continue

            try:
                if isinstance(full_response, str):
                    full_response = self._extract_json(full_response)
            except Exception as e:
                logger.warning(f"[NaviInteraction] JSON提取失败: {e}. Retrying...")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {"current_status", "thought", "response"}.issubset(keys):
                    response = full_response.pop("response")
                    if isinstance(response, str):
                        break
                    else:
                        logger.warning(
                            f"[NaviInteraction] 无效响应: {response}"
                        )
                        continue
                else:
                    logger.warning(f"[NaviInteraction] 缺少必要字段: {keys}")
                    continue

        self._instance_dict[instance_id]["response"] = response
        logger.debug(f"[NaviInteraction] User: {response}")

        # 检查是否应该终止
        should_terminate_sequence = self.termination_signal in response

        # 导航特定：检查是否包含导航完成信号
        nav_completion_signals = ["好的，开始吧", "可以了", "谢谢", "好的", "没问题"]
        if any(signal in response for signal in nav_completion_signals):
            should_terminate_sequence = True

        reward = 0.0

        return should_terminate_sequence, response, reward, {}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """结束交互实例"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    def _parse_messages(self, messages, strip_sys_prompt=True):
        """解析消息列表为对话历史字符串"""
        if messages is None:
            return ""

        if strip_sys_prompt:
            messages = [msg for msg in messages if msg["role"] != "system"]

        messages = [self._remove_think_block(msg) for msg in messages]

        chat = "\n".join(
            f"**{m['role'].capitalize()}**: {m.get('content', '')}"
            for m in messages
        )

        return chat

    def _remove_think_block(self, msg: dict):
        """移除think块"""
        if "content" in msg and isinstance(msg["content"], str):
            msg["content"] = re.sub(r"<tool_call>.*?uaiya>", "", msg["content"], flags=re.DOTALL).strip()
        return msg

    def _extract_json(self, s: str) -> dict:
        """从字符串中提取JSON对象"""
        def convert_value(value):
            true_values = {"true": True, "false": False, "null": None}
            value_lower = value.lower()
            if value_lower in true_values:
                return true_values[value_lower]
            try:
                if "." in value or "e" in value.lower():
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value

        def skip_whitespace(s, pos):
            while pos < len(s) and s[pos] in " \t\n\r":
                pos += 1
            return pos

        def parse_string(s, pos):
            quote_char = s[pos]
            assert quote_char in ('"', "'")
            pos += 1
            result = ""
            while pos < len(s):
                c = s[pos]
                if c == "\\":
                    pos += 1
                    if pos >= len(s):
                        raise ValueError("Invalid escape sequence")
                    c = s[pos]
                    escape_sequences = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", quote_char: quote_char}
                    result += escape_sequences.get(c, c)
                elif c == quote_char:
                    pos += 1
                    converted_value = convert_value(result)
                    return converted_value, pos
                else:
                    result += c
                pos += 1
            raise ValueError("Unterminated string")

        def parse_key(s, pos):
            pos = skip_whitespace(s, pos)
            if s[pos] in ('"', "'"):
                key, pos = parse_string(s, pos)
                return key, pos
            else:
                raise ValueError(f"Expected string for key at position {pos}")

        def parse_number(s, pos):
            start = pos
            while pos < len(s) and s[pos] in "-+0123456789.eE":
                pos += 1
            num_str = s[start:pos]
            try:
                if "." in num_str or "e" in num_str.lower():
                    return float(num_str), pos
                else:
                    return int(num_str), pos
            except ValueError:
                raise ValueError(f"Invalid number at position {start}: {num_str}")

        def parse_array(s, pos):
            lst = []
            assert s[pos] == "["
            pos += 1
            pos = skip_whitespace(s, pos)
            while pos < len(s) and s[pos] != "]":
                value, pos = parse_value(s, pos)
                lst.append(value)
                pos = skip_whitespace(s, pos)
                if pos < len(s) and s[pos] == ",":
                    pos += 1
                    pos = skip_whitespace(s, pos)
            if pos >= len(s) or s[pos] != "]":
                raise ValueError('Expected "]"')
            pos += 1
            return lst, pos

        def parse_object(s, pos):
            obj = {}
            assert s[pos] == "{"
            pos += 1
            pos = skip_whitespace(s, pos)
            while pos < len(s) and s[pos] != "}":
                pos = skip_whitespace(s, pos)
                key, pos = parse_key(s, pos)
                pos = skip_whitespace(s, pos)
                if pos >= len(s) or s[pos] != ":":
                    raise ValueError(f'Expected ":" at position {pos}')
                pos += 1
                pos = skip_whitespace(s, pos)
                value, pos = parse_value(s, pos)
                obj[key] = value
                pos = skip_whitespace(s, pos)
                if pos < len(s) and s[pos] == ",":
                    pos += 1
            if pos >= len(s) or s[pos] != "}":
                raise ValueError('Expected "}"')
            pos += 1
            return obj, pos

        def parse_value(s, pos):
            pos = skip_whitespace(s, pos)
            if pos >= len(s):
                raise ValueError("Unexpected end of input")
            if s[pos] == "{":
                return parse_object(s, pos)
            elif s[pos] == "[":
                return parse_array(s, pos)
            elif s[pos] in ('"', "'"):
                return parse_string(s, pos)
            elif s[pos : pos + 4].lower() == "true":
                return True, pos + 4
            elif s[pos : pos + 5].lower() == "false":
                return False, pos + 5
            elif s[pos : pos + 4].lower() == "null":
                return None, pos + 4
            elif s[pos] in "-+0123456789.":
                return parse_number(s, pos)
            else:
                raise ValueError(f"Unexpected character at position {pos}: {s[pos]}")

        json_start = s.index("{")
        json_end = s.rfind("}")
        s = s[json_start : json_end + 1]

        s = s.strip()
        result, pos = parse_value(s, 0)
        return result
