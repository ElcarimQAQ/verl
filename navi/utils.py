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
Navigation Agent Utilities
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def is_valid_messages(msg: dict) -> bool:
    """
    检查消息是否有效，包括：
    1. </think> 标签配对
    2. 标签内外内容非空
    3. 没有嵌套的 think 块
    4. 移除 <|im_end|> 后内容非空
    """
    content = msg.get("content")
    if not isinstance(content, str):
        return True

    if not content.strip():
        return False

    num_think_open = content.count("<tool_call>")
    num_think_close = content.count("uaiya>")

    if num_think_open != num_think_close:
        return False

    if num_think_open > 1:
        return False

    if num_think_open == 0:
        visible_content = content
    else:
        match = re.search(r"<tool_call>(.*?)uaiya>", content, re.DOTALL)
        if not match or not match.group(1).strip():
            return False
        visible_content = re.sub(r"<tool_call>.*?uaiya>", "", content, flags=re.DOTALL)

    visible_content = visible_content.strip()

    if visible_content.endswith("<|im_end|>"):
        visible_content = visible_content[: -len("<|im_end|>")]

    if not visible_content.strip():
        return False

    return True


def parse_tool_calls_from_content(content: str) -> List[Dict]:
    """
    从内容中解析工具调用

    支持格式：
    1. JSON格式的工具调用
    2. 函数调用格式: function_name(arg1=value1, arg2=value2)
    """
    tool_calls = []

    # 尝试解析JSON格式的工具调用
    try:
        # 查找所有可能的JSON对象
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+\"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}'
        matches = re.findall(json_pattern, content)
        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logger.debug(f"Failed to parse JSON tool calls: {e}")

    # 尝试解析函数调用格式
    if not tool_calls:
        func_pattern = r'(\w+)\s*\(([^)]*)\)'
        matches = re.findall(func_pattern, content)
        for func_name, args_str in matches:
            try:
                args = parse_function_args(args_str)
                tool_calls.append({
                    "name": func_name,
                    "arguments": args
                })
            except Exception as e:
                logger.debug(f"Failed to parse function call: {e}")

    return tool_calls


def parse_function_args(args_str: str) -> Dict:
    """解析函数参数字符串为字典"""
    args = {}
    if not args_str.strip():
        return args

    # 匹配 key=value 或 key="value" 格式
    pattern = r'(\w+)\s*=\s*(?:["\']([^"\']*)["\']|([^,\s]+))'
    matches = re.findall(pattern, args_str)

    for match in matches:
        key = match[0]
        value = match[1] if match[1] else match[2]
        args[key] = value

    return args


def remove_think_block(msg: dict) -> dict:
    """移除think块"""
    if "content" in msg and isinstance(msg["content"], str):
        msg["content"] = re.sub(r"<tool_call>.*?uaiya>", "", msg["content"], flags=re.DOTALL).strip()
    return msg


def parse_messages(messages: List[dict], strip_sys_prompt: bool = True) -> str:
    """
    解析消息列表为对话历史字符串

    Args:
        messages: 消息列表，每个元素包含 'role' 和 'content'
        strip_sys_prompt: 是否移除系统提示

    Returns:
        格式化的对话历史字符串
    """
    if messages is None:
        return ""

    if strip_sys_prompt:
        messages = [msg for msg in messages if msg.get("role") != "system"]

    messages = [remove_think_block(msg) for msg in messages]

    chat = "\n".join(
        f"**{m.get('role', 'unknown').capitalize()}**: {m.get('content', '')}"
        for m in messages
    )

    return chat


def check_navigation_completion(tool_name: str, tool_return: str, args: Dict) -> Tuple[bool, str]:
    """
    检查导航任务是否完成

    Args:
        tool_name: 工具名称
        tool_return: 工具返回结果
        args: 工具参数

    Returns:
        (is_completed, reason): 是否完成及原因
    """
    if tool_name == "notify_user_msg":
        msg = args.get("msg", "")

        # 检查导航开始信号
        start_signals = ["开始导航", "已为您规划", "导航已启动", "正在为您导航", "已为您导航"]
        if any(signal in msg for signal in start_signals):
            return True, "导航已开始"

        # 检查导航结束信号
        end_signals = ["导航结束", "已到达", "任务完成", "导航已完成", "祝您"]
        if any(signal in msg for signal in end_signals):
            return True, "任务完成"

        # 消息为空
        if not msg:
            return True, "播报内容为空"

    # 检查工具返回中的导航结束信号
    if "导航结束" in tool_return or "已到达目的地" in tool_return:
        return True, "导航结束信号"

    return False, "任务未完成"


def validate_tool_call(tool_name: str, tool_args: Dict, history: List[Dict]) -> Tuple[bool, str]:
    """
    验证工具调用的有效性

    Args:
        tool_name: 工具名称
        tool_args: 工具参数
        history: 历史消息

    Returns:
        (is_valid, reason): 是否有效及原因
    """
    # 规则1: navigation_start 前必须已经获取了目的地
    if tool_name == "navigation_start":
        has_destination = False
        for msg in history:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                try:
                    data = json.loads(content) if isinstance(content, str) else content
                    if isinstance(data, dict):
                        # 检查POI搜索结果
                        if "pois" in data or "destination" in data:
                            has_destination = True
                            break
                except json.JSONDecodeError:
                    if "destination" in content.lower() or "目的地" in content:
                        has_destination = True
                        break

        if not has_destination:
            return False, "navigation_start called without confirmed destination"

    # 规则2: 检查参数完整性（与 navi_sandbox_tool.NAVI_TOOL_SCHEMAS 中的 v21 工具定义保持一致）
    required_params = {
        "poi_search": ["mode", "keyword"],
        "navigation_start": ["mode", "desLocationReference"],
        "navigation_route": ["mode"],
        "navigation_info_query": ["mode"],
        "navigation_memory": ["mode"],
        "navigation_pathPoint": ["mode"],
        "filter": ["reference"],
        "notify_user_msg": ["msg"],
        "wiki_search": ["query"],
    }

    if tool_name in required_params:
        for param in required_params[tool_name]:
            if param not in tool_args or not tool_args[param]:
                return False, f"Missing required parameter: {param}"

    return True, "valid"


def extract_json(s: str) -> Any:
    """
    从字符串中提取JSON对象

    Args:
        s: 包含JSON的字符串

    Returns:
        解析后的Python对象
    """
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
                return convert_value(result), pos
            else:
                result += c
            pos += 1
        raise ValueError("Unterminated string")

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
            raise ValueError(f"Invalid number: {num_str}")

    def parse_array(s, pos):
        lst = []
        pos += 1  # skip [
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
        return lst, pos + 1

    def parse_object(s, pos):
        obj = {}
        pos += 1  # skip {
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "}":
            pos = skip_whitespace(s, pos)
            # parse key
            if s[pos] in ('"', "'"):
                key, pos = parse_string(s, pos)
            else:
                raise ValueError(f"Expected string key at position {pos}")

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
        return obj, pos + 1

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
        elif s[pos:pos+4].lower() == "true":
            return True, pos + 4
        elif s[pos:pos+5].lower() == "false":
            return False, pos + 5
        elif s[pos:pos+4].lower() == "null":
            return None, pos + 4
        elif s[pos] in "-+0123456789.":
            return parse_number(s, pos)
        else:
            raise ValueError(f"Unexpected character at position {pos}: {s[pos]}")

    # Find JSON boundaries
    json_start = s.find("{")
    if json_start == -1:
        json_start = s.find("[")
    if json_start == -1:
        raise ValueError("No JSON object found")

    if s[json_start] == "{":
        json_end = s.rfind("}")
    else:
        json_end = s.rfind("]")

    s = s[json_start:json_end + 1].strip()
    result, _ = parse_value(s, 0)
    return result
