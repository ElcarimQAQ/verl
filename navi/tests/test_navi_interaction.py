# Copyright 2025 Navi Agent Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import copy
import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

if "litellm" not in sys.modules:
    try:
        import litellm  # noqa: F401
    except ImportError:
        litellm_stub = types.ModuleType("litellm")
        litellm_stub.RateLimitError = type("RateLimitError", (Exception,), {})
        litellm_stub.acompletion = None
        sys.modules["litellm"] = litellm_stub

try:
    import numpy  # noqa: F401
except ImportError:
    verl_stub = types.ModuleType("verl")
    verl_stub.__path__ = []
    utils_stub = types.ModuleType("verl.utils")
    utils_stub.__path__ = []
    trace_stub = types.ModuleType("verl.utils.rollout_trace")
    trace_stub.rollout_trace_op = lambda function: function
    sys.modules.update({"verl": verl_stub, "verl.utils": utils_stub, "verl.utils.rollout_trace": trace_stub})

from navi.navi_interaction import NaviInteraction, _extract_response_json, _safe_satisfaction


def _completion(content):
    message = SimpleNamespace(content=content, provider_specific_fields={})
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class TestNaviInteraction(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.interaction = NaviInteraction({"user_model": "test-model", "num_retries": 1})

    async def test_structured_ground_truth_and_satisfaction_reach_user_model(self):
        ground_truth = {
            "role": "assistant",
            "content": "已开始导航",
            "tool_calls": [{"function": {"name": "navigation_start", "arguments": {}}}],
        }
        instance_id = await self.interaction.start_interaction(
            ground_truth=ground_truth, single_turn_prompt="导航去机场"
        )
        answer = json.dumps(
            {"current_status": "完成", "thought": "已执行", "response": "好的", "satisfaction": 1},
            ensure_ascii=False,
        )
        completion = AsyncMock(return_value=_completion(answer))
        with patch("navi.navi_interaction.litellm.acompletion", new=completion) as call:
            terminated, response, reward, metrics = await self.interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "已开始导航"}],
                env_diff_summary="进入导航态",
                env_state_text="目的地：机场",
            )

        prompt = call.await_args.kwargs["messages"][0]["content"]
        self.assertIn(json.dumps(ground_truth, ensure_ascii=False, indent=2), prompt)
        self.assertIn("进入导航态", prompt)
        self.assertFalse(terminated)
        self.assertEqual(response, "好的")
        self.assertEqual(reward, 1.0)
        self.assertEqual(metrics, {"has_satisfaction": True, "is_fallback": False})

    async def test_failed_model_uses_rotating_nonterminating_fallback(self):
        instance_id = await self.interaction.start_interaction(single_turn_prompt="导航去机场")
        error = RuntimeError("invalid model")
        with patch("navi.navi_interaction.litellm.acompletion", new=AsyncMock(side_effect=error)):
            first = await self.interaction.generate_response(instance_id, [{"role": "assistant", "content": ""}])
            second = await self.interaction.generate_response(instance_id, [{"role": "assistant", "content": ""}])

        self.assertEqual(first, (False, "导航去机场", 0.0, {"has_satisfaction": False, "is_fallback": True}))
        self.assertEqual(second[1], "还有别的方案吗？")

    def test_parse_messages_exposes_actions_and_visible_results_without_mutation(self):
        messages = [
            {"role": "user", "content": "去机场"},
            {
                "role": "assistant",
                "content": "<|im_end|>",
                "tool_calls": [
                    {
                        "id": "call-nav",
                        "type": "function",
                        "function": {
                            "name": "navigation_start",
                            "arguments": {"desLocationReference": "机场", "mode": "导航"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-nav", "content": '{"errorCode": 0}'},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-wiki",
                        "type": "function",
                        "function": {"name": "wiki_search", "arguments": {"query": "机场"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-wiki", "content": "very long internal result"},
        ]
        original = copy.deepcopy(messages)

        rendered = self.interaction._parse_messages(messages)

        self.assertIn("调用了 navigation_start", rendered)
        self.assertIn("errorCode", rendered)
        self.assertIn("调用了 wiki_search", rendered)
        self.assertNotIn("very long internal result", rendered)
        self.assertEqual(messages, original)

    def test_notify_message_is_rendered_as_assistant_content(self):
        message = {
            "role": "assistant",
            "content": "<|im_end|>",
            "tool_calls": [
                {
                    "function": {
                        "name": "notify_user_msg",
                        "arguments": json.dumps({"msg": "已为您开始导航"}, ensure_ascii=False),
                    }
                }
            ],
        }
        self.assertIn("已为您开始导航", self.interaction._parse_messages([message]))
        self.assertEqual(message["content"], "<|im_end|>")


class TestNaviInteractionHelpers(unittest.TestCase):
    def test_extracts_last_valid_json_after_ground_truth(self):
        text = (
            'ground truth: {"tool_calls": []}\n'
            '{"current_status":"done","thought":"ok","response":"结束","satisfaction":1}'
        )
        self.assertEqual(_extract_response_json(text)["response"], "结束")

    def test_satisfaction_is_binary_and_rejects_invalid_values(self):
        self.assertEqual(_safe_satisfaction("0"), 0.0)
        self.assertEqual(_safe_satisfaction(0.2), 1.0)
        self.assertIsNone(_safe_satisfaction(float("nan")))
        self.assertIsNone(_safe_satisfaction("invalid"))


if __name__ == "__main__":
    unittest.main()
