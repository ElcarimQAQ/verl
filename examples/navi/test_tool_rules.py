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

"""Tests for the per-tool hard-rule validators in ``examples/navi/tool_rules.py``.

Run from the repo root::

    pytest examples/navi/test_tool_rules.py
"""

import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("navi_tool_rules", os.path.join(_HERE, "tool_rules.py"))
tool_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool_rules)
validate_tool_args = tool_rules.validate_tool_args


@pytest.mark.parametrize(
    "expect_ok, name, args",
    [
        # navigation_start: 多地点不能用「导航」mode
        (False, "navigation_start", {"mode": "导航", "desLocationReference": "a", "pathPointReference": ["b"]}),
        (True, "navigation_start", {"mode": "导航", "desLocationReference": "a"}),
        # 单地点不能用「展示规划」
        (False, "navigation_start", {"mode": "展示规划", "desLocationReference": "a"}),
        (True, "navigation_start", {"mode": "展示规划", "desLocationReference": "a", "pathPointReference": ["b"]}),
        # 非法 routeType / 未定义 key
        (False, "navigation_start", {"mode": "导航", "desLocationReference": "a", "routeType": "瞎写"}),
        (False, "navigation_start", {"mode": "导航", "desLocationReference": "a", "badkey": 1}),
        # 黑白名单非空
        (False, "navigation_start", {"mode": "导航", "desLocationReference": "a", "includedRoadName": []}),
        # navigation_route
        (False, "navigation_route", {"routeType": "智能推荐"}),  # 缺 mode
        (False, "navigation_route", {"mode": "刷新", "routeType": "瞎写"}),
        (True, "navigation_route", {"mode": "刷新", "routeType": "智能推荐"}),
        (False, "navigation_route", {"mode": "刷新", "routeIndex": 0}),
        (True, "navigation_route", {"mode": "刷新", "routeIndex": 2}),
        # filter
        (False, "filter", {"rating_min": 1}),  # 缺 reference
        (False, "filter", {"reference": "poi_search@1", "rating_min": 4, "rating_max": 2}),
        (True, "filter", {"reference": "poi_search@1", "rating_min": 2, "rating_max": 4}),
        (False, "filter", {"reference": "poi_search@1", "rating_max": 9}),  # 超出 0-5
        (False, "filter", {"reference": "poi_search@1", "distance_min": 5, "distance_max": 1}),
        (True, "filter", {"reference": "poi_search@1", "businessHour": ["09:00-18:00"]}),
        (True, "filter", {"reference": "poi_search@1", "businessHour": ["00:00-24:00"]}),
        (False, "filter", {"reference": "poi_search@1", "businessHour": ["9点到6点"]}),
        # notify_user_msg
        (False, "notify_user_msg", {"msg": "抱歉没找到"}),
        (False, "notify_user_msg", {"msg": "已为您找到以下地点，要去哪个呢？"}),
        (False, "notify_user_msg", {"msg": "啊" * 61}),  # 过长
        (True, "notify_user_msg", {"msg": "已为您规划路线"}),
        # poi_search / wiki_search key 校验
        (False, "poi_search", {"mode": "关键词", "keyword": "加油站", "badkey": 1}),
        (True, "poi_search", {"mode": "关键词", "keyword": "加油站"}),
        (False, "wiki_search", {"q": "东方明珠"}),  # 错误 key
        (True, "wiki_search", {"query": "东方明珠"}),
        # 工具白名单
        (False, "unknown_api", {}),
        # 无细粒度规则的工具：只过白名单即可
        (True, "navigation_memory", {"whatever": 1}),
    ],
)
def test_validate_tool_args(expect_ok, name, args):
    ok, reason = validate_tool_args(name, args)
    assert ok == expect_ok, f"{name} args={args}: got ok={ok} reason={reason!r}"
    if ok:
        assert reason == ""
    else:
        assert reason  # 违规必须给出非空 reason


def test_non_dict_args_rejected():
    ok, reason = validate_tool_args("filter", "not-a-dict")
    assert ok is False and reason
