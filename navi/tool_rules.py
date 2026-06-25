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

"""Per-tool hard-rule validators for the navi agent (schema / business rules only).

纯静态校验：只判断"这次工具调用本身符不符合该工具的硬规则"，
不涉及 GT 比对、不涉及任务完成、不涉及 sandbox 执行结果。

供在线 agent loop 在**每步**工具调用时做 per-step 合法性打分：
合规 -> 0 分，违规 -> 负分。任务完成 / 工具名是否选对 / 与 GT 匹配度
都不在这一层，分别交给 turn_scores 和最终 compute_score。
"""

import re
from typing import Any, Dict, Tuple

# API 白名单（来自离线 reward 脚本 ALLOWED_APIS）。不在列表内的工具一律视为非法。
ALLOWED_APIS = {
    "poi_search", "navigation_start", "notify_user_msg", "reject", "wiki_search",
    "navigation_route", "NEXT_PAGE", "BACK_PAGE", "SELECT_PAGE", "SCROLL_PAGE",
    "navigation_control", "navigation_mapZoom", "navigation_function_switch",
    "navigation_broadCastMode_set", "navigation_info_query",
    "navigation_roadcondition_query", "share_poi", "navigation_memory", "filter",
}

_ROUTE_TYPES = {"智能推荐", "速度最快", "不走高速", "高速优先", "避免拥堵", "避免收费", "大路优先"}


def _check_allowed_keys(args: dict, allowed: set, tool: str) -> Tuple[bool, str]:
    for k in args:
        if k not in allowed:
            return False, f"{tool} 不允许的 key: `{k}`"
    return True, ""


def validate_navigation_start(args: Dict[str, Any]) -> Tuple[bool, str]:
    ok, msg = _check_allowed_keys(
        args,
        {"mode", "desLocationReference", "pathPointReference",
         "pathpointNegativeReference", "routeType", "includedRoadName", "excludedRoadName"},
        "navigation_start",
    )
    if not ok:
        return ok, msg

    path_points = args.get("pathPointReference", [])
    if not isinstance(path_points, list):
        path_points = [path_points] if path_points else []
    location_count = (1 if args.get("desLocationReference") else 0) + len(path_points)

    mode = args.get("mode")
    if location_count == 1 and mode == "展示规划":
        return False, "单地点导航不能使用 `展示规划` mode"
    if location_count >= 2 and mode == "导航":
        return False, f"多地点导航（{location_count}个地点）不能使用 `导航` mode"

    rt = args.get("routeType")
    if rt is not None and rt not in _ROUTE_TYPES:
        return False, f"navigation_start 的 routeType 非法：{rt}"

    for key in ("includedRoadName", "excludedRoadName"):
        v = args.get(key)
        if v is not None and (not isinstance(v, list) or len(v) == 0):
            return False, f"navigation_start 的 {key} 如果存在必须是非空列表"
    return True, ""


def validate_navigation_route(args: Dict[str, Any]) -> Tuple[bool, str]:
    ok, msg = _check_allowed_keys(
        args, {"mode", "routeType", "includedRoadName", "excludedRoadName", "routeIndex"},
        "navigation_route",
    )
    if not ok:
        return ok, msg
    mode = args.get("mode")
    if mode is None:
        return False, "navigation_route 必须包含 mode 参数"
    if mode not in {"路线偏好", "刷新"}:
        return False, f"navigation_route 的 mode 非法：`{mode}`"
    rt = args.get("routeType")
    if rt is not None and rt not in _ROUTE_TYPES:
        return False, f"navigation_route 的 routeType 非法：`{rt}`"
    ri = args.get("routeIndex")
    if ri is not None:
        try:
            if float(ri) < 1:
                return False, f"routeIndex 必须是正数：`{ri}`"
        except (TypeError, ValueError):
            return False, f"routeIndex 必须是数字：`{ri}`"
    for key in ("includedRoadName", "excludedRoadName"):
        v = args.get(key)
        if v is not None and (not isinstance(v, list) or len(v) == 0):
            return False, f"navigation_route 的 {key} 如果存在必须是非空列表"
    return True, ""


def validate_filter(args: Dict[str, Any]) -> Tuple[bool, str]:
    ok, msg = _check_allowed_keys(
        args,
        {"reference", "distance_min", "distance_max", "rating_min", "rating_max",
         "price_min", "price_max", "businessHour"},
        "filter",
    )
    if not ok:
        return ok, msg
    if not args.get("reference"):
        return False, "filter 必须包含 reference 参数"

    def _range(lo_key, hi_key, lo_bound=None, hi_bound=None) -> Tuple[bool, str]:
        lo, hi = args.get(lo_key), args.get(hi_key)
        for name, val in ((lo_key, lo), (hi_key, hi)):
            if val is not None:
                try:
                    f = float(val)
                except (TypeError, ValueError):
                    return False, f"{name} 必须是数字"
                if lo_bound is not None and (f < lo_bound or f > hi_bound):
                    return False, f"{name}({val}) 必须在 {lo_bound}-{hi_bound} 范围内"
        if lo is not None and hi is not None and float(lo) > float(hi):
            return False, f"{lo_key}({lo}) 不能大于 {hi_key}({hi})"
        return True, ""

    for r in (_range("distance_min", "distance_max"),
              _range("rating_min", "rating_max", 0.0, 5.0),
              _range("price_min", "price_max")):
        if not r[0]:
            return r

    bh = args.get("businessHour")
    if bh is not None:
        if not isinstance(bh, list):
            return False, "businessHour 必须是列表"
        pats = [re.compile(r"^\d{1,2}:\d{2}-\d{1,2}:\d{2}$"),
                re.compile(r"^(before|after)-\d{1,2}:\d{2}$"),
                re.compile(r"^\d{1,2}:\d{2}-(before|after)$")]
        for slot in bh:
            if not isinstance(slot, str):
                return False, "businessHour 的每个元素必须是字符串"
            if not any(p.match(slot) for p in pats) and slot != "00:00-24:00":
                return False, f"businessHour 格式错误：{slot}"
    return True, ""


def validate_notify_user_msg(args: Dict[str, Any]) -> Tuple[bool, str]:
    ok, msg = _check_allowed_keys(args, {"msg", "status"}, "notify_user_msg")
    if not ok:
        return ok, msg
    text = args.get("msg", "")
    if not isinstance(text, str):
        return False, "msg 必须是字符串"
    if "抱歉" in text:
        return False, "notify_user_msg 的 msg 禁止包含'抱歉'"
    if len(text) > 70:
        return False, f"notify_user_msg 的 msg 字数过长（{len(text)}字）"
    if "已为您找到以下地点，要去哪个呢？" in text:
        return False, "notify_user_msg 禁止使用模糊回复"
    return True, ""


def validate_poi_search(args: Dict[str, Any]) -> Tuple[bool, str]:
    return _check_allowed_keys(
        args, {"mode", "keyword", "centerLocationReference", "radius"}, "poi_search")


def validate_wiki_search(args: Dict[str, Any]) -> Tuple[bool, str]:
    return _check_allowed_keys(args, {"query"}, "wiki_search")


# 工具名 -> 细粒度校验函数。未列出的工具表示暂无 schema 级硬规则（仅做白名单校验）。
TOOL_VALIDATORS = {
    "navigation_start": validate_navigation_start,
    "navigation_route": validate_navigation_route,
    "filter": validate_filter,
    "notify_user_msg": validate_notify_user_msg,
    "poi_search": validate_poi_search,
    "wiki_search": validate_wiki_search,
}


def validate_tool_args(tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, str]:
    """统一入口：API 白名单 + 该工具的细粒度硬规则。

    Returns:
        (is_valid, reason)。``is_valid`` 为 True 时 ``reason`` 为空串。
    """
    if tool_name not in ALLOWED_APIS:
        return False, f"API `{tool_name}` 不在允许列表中"
    validator = TOOL_VALIDATORS.get(tool_name)
    if validator is None:
        return True, ""
    if not isinstance(tool_args, dict):
        return False, f"{tool_name} 的 arguments 必须是对象"
    return validator(tool_args)
