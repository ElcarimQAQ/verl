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
NaviAgentLoop 快速单测，跳过 Ray/vLLM/真实模型加载。

启动一次 `train_rl_navi*.sh` 需要拉起 Ray、加载真实模型权重、起 vLLM/SGLang
server，少则几分钟、多则几十分钟，非常不适合迭代调试 `navi_agent_loop.py` /
`navi_interaction.py` 里的状态机和奖励逻辑。这里把唯一真正"重"的两个依赖
mock 掉：

- LLM 生成：``FakeServerManager.generate`` 直接从一个预先写好的回复队列里弹出
  下一条回复（经过真实的 HermesToolParser 解析，所以工具调用解析逻辑仍然是
  真实路径），不需要真实模型/GPU。
- 用户模拟器：用 ``StubInteraction`` 替代真实的 ``NaviInteraction``，不发起
  litellm/网络请求。

其余路径（沙盒工具分发 MockSandboxExecutor、AgentState 状态转移、
interaction 生命周期管理、奖励计算）都是真实代码路径。

运行方式（需要在已安装 torch/omegaconf/transformers 等 verl 依赖的环境中）：

    pytest navi/tests/test_navi_agent_loop.py -v
    # 或不依赖 pytest：
    python -m unittest navi.tests.test_navi_agent_loop -v
"""

import asyncio
import json
import unittest

from omegaconf import OmegaConf

from navi.navi_agent_loop import NaviAgentLoop
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.workers.rollout.replica import TokenOutput


class FakeTokenizer:
    """最小可用 tokenizer：单词级编码，足以让真实 HermesToolParser 正常解析。"""

    pad_token = "<pad>"

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._inv_vocab: dict[int, str] = {}

    def _tok(self, word: str) -> int:
        if word not in self._vocab:
            idx = len(self._vocab)
            self._vocab[word] = idx
            self._inv_vocab[idx] = word
        return self._vocab[word]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._tok(w) for w in text.split(" ")] if text else []

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self._inv_vocab.get(i, "") for i in ids)

    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True, tokenize: bool = True, **kwargs
    ):
        parts = []
        for m in messages:
            parts.append(f"<{m.get('role', '')}>")
            content = m.get("content") or ""
            if content:
                parts.append(content)
        if add_generation_prompt:
            parts.append("<assistant>")
        text = " ".join(parts)
        return self.encode(text) if tokenize else text


class FakeServerManager:
    """按调用顺序弹出预先写好的 assistant 回复文本，不做真实推理。"""

    def __init__(self, tokenizer: FakeTokenizer, replies: list[str]):
        self.tokenizer = tokenizer
        self._replies = list(replies)
        self.call_count = 0

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs) -> TokenOutput:
        assert self._replies, "FakeServerManager replies exhausted: 场景脚本写的回复数不够"
        text = self._replies.pop(0)
        self.call_count += 1
        return TokenOutput(token_ids=self.tokenizer.encode(text))


class StubInteraction:
    """替代 NaviInteraction：不发起任何网络请求，按脚本顺序返回用户回复。"""

    def __init__(self, scripted_responses: list[tuple[bool, str, float]]):
        # 每项: (should_terminate, response_text, reward)
        self._script = list(scripted_responses)
        self.config = {"enable_log": False}
        self.started: list[str] = []
        self.finalized: list[str] = []

    async def start_interaction(self, instance_id: str, **kwargs) -> str:
        self.started.append(instance_id)
        return instance_id

    async def generate_response(self, instance_id: str, messages, **kwargs):
        assert self._script, "StubInteraction script exhausted"
        should_terminate, text, reward = self._script.pop(0)
        return should_terminate, text, reward, {}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self.finalized.append(instance_id)


def _make_trainer_config(num_repeat_rollouts: int = 1) -> DictConfigWrap:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 4096,
                    "response_length": 4096,
                    "multi_turn": {
                        "max_user_turns": 8,
                        "max_assistant_turns": 8,
                        "max_parallel_calls": 4,
                        "max_tool_response_length": 4096,
                        "tool_response_truncate_side": "right",
                        "format": "hermes",
                        "num_repeat_rollouts": num_repeat_rollouts,
                    },
                }
            }
        }
    )
    return DictConfigWrap(config)


def _make_data_config() -> DictConfigWrap:
    config = OmegaConf.create(
        {
            "apply_chat_template_kwargs": {},
            "mm_processor_kwargs": {},
            "continuous_token": {"enable": False},
        }
    )
    return DictConfigWrap(config)


def _tool_call_text(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>{payload}</tool_call>"


class NaviAgentLoopTestCase(unittest.TestCase):
    """所有测试共享的构造逻辑，子类只需要提供 replies/interaction 脚本。"""

    def _build_loop(self, tokenizer: FakeTokenizer, server_manager, num_repeat_rollouts: int = 1) -> NaviAgentLoop:
        from verl.utils.dataset.rl_dataset import RLHFDataset
        from navi.navi_agent_loop import MockSandboxExecutor

        agent_loop = NaviAgentLoop(
            trainer_config=_make_trainer_config(num_repeat_rollouts),
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=RLHFDataset,
            data_config=_make_data_config(),
            agent={},
        )
        # 强制使用 MockSandboxExecutor：如果 navi_lab.sandbox.core2 在当前环境可导入
        # （比如在真实训练机器上跑这些单测），__init__ 会优先选用真实 SandboxExecutor，
        # 这些单测就会变成悄悄发起真实网络请求的集成测试，结果随环境漂移。
        agent_loop._sandbox_executor_cls = MockSandboxExecutor
        return agent_loop

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)


class TestToolCallThenTerminate(NaviAgentLoopTestCase):
    """模型先调用 poi_search，拿到结果后不再调用工具 -> 直接终止（无 interaction 配置）。"""

    def test_tool_call_then_plain_terminate(self):
        tokenizer = FakeTokenizer()
        replies = [
            _tool_call_text("poi_search", {"mode": "关键词", "keyword": "咖啡"}),
            "好的，已经为您找到附近的咖啡馆。",
        ]
        server_manager = FakeServerManager(tokenizer, replies)
        agent_loop = self._build_loop(tokenizer, server_manager)

        output = self._run(
            agent_loop.run(
                sampling_params={},
                raw_prompt=[{"role": "user", "content": "帮我找个咖啡馆"}],
            )
        )

        self.assertEqual(server_manager.call_count, 2)
        # poi_search 成功但不构成"完成"：per-step 合规奖励为 0（不奖不罚）
        self.assertEqual(output.extra_fields["tool_rewards"], [0.0])
        # 没有任务完成 -> turn_scores 不应出现完成的 1.0
        self.assertNotIn(1.0, output.extra_fields["turn_scores"])


class TestNonNotifyToolCompletion(NaviAgentLoopTestCase):
    """非 notify 工具直接完成任务（结果命中"已到达目的地"信号）-> 完成奖励走
    turn_scores，不进 tool_rewards，且立即终止。"""

    def test_completion_routes_to_turn_scores(self):
        from navi.navi_agent_loop import MockSandboxExecutor

        class CompletingSandbox(MockSandboxExecutor):
            # 让任意工具的执行结果命中 check_navigation_completion 的"已到达目的地"信号
            def execute(self, call):
                return {"status": "0", "info": "已到达目的地"}

        tokenizer = FakeTokenizer()
        replies = [
            _tool_call_text("navigation_start", {"mode": "导航", "desLocationReference": "目的地"}),
        ]
        server_manager = FakeServerManager(tokenizer, replies)
        agent_loop = self._build_loop(tokenizer, server_manager)
        agent_loop._sandbox_executor_cls = CompletingSandbox

        output = self._run(
            agent_loop.run(
                sampling_params={},
                raw_prompt=[{"role": "user", "content": "带我去目的地"}],
            )
        )

        # 工具调用后任务即完成并终止，只生成一次
        self.assertEqual(server_manager.call_count, 1)
        # 完成奖励写入 turn_scores
        self.assertIn(1.0, output.extra_fields["turn_scores"])
        # per-step tool_rewards 只反映合规性（成功 -> 0.0），绝不含完成的 1.0
        self.assertNotIn(1.0, output.extra_fields["tool_rewards"])
        self.assertTrue(all(r <= 0.0 for r in output.extra_fields["tool_rewards"]))


class TestNotifyUserMsgTriggersInteraction(NaviAgentLoopTestCase):
    """notify_user_msg -> INTERACTING -> 用户模拟器回复终止信号 -> TERMINATED。"""

    def test_notify_user_msg_then_interaction_terminates(self):
        tokenizer = FakeTokenizer()
        replies = [
            _tool_call_text("notify_user_msg", {"msg": "已为您找到咖啡馆"}),
        ]
        server_manager = FakeServerManager(tokenizer, replies)
        agent_loop = self._build_loop(tokenizer, server_manager)

        stub_interaction = StubInteraction([(True, "好的，谢谢", 0.5)])
        agent_loop.interaction_config_file = "stub://navi"
        agent_loop.interaction_map = {"navi": stub_interaction}

        output = self._run(
            agent_loop.run(
                sampling_params={},
                raw_prompt=[{"role": "user", "content": "帮我找个咖啡馆"}],
            )
        )

        self.assertEqual(server_manager.call_count, 1)
        self.assertEqual(stub_interaction.started, [stub_interaction.finalized[0]])
        self.assertEqual(len(stub_interaction.finalized), 1)
        messages = output.extra_fields["messages"]["messages"][0]
        self.assertTrue(any(m.role == "user" and m.content == "好的，谢谢" for m in messages))


class TestNotifyUserMsgInteractionContinues(NaviAgentLoopTestCase):
    """用户模拟器第一次要求继续 -> 回到 GENERATING -> 模型不再调用工具，但 interaction
    仍未结束，所以再次进入 INTERACTING -> 用户模拟器第二次回复终止信号 -> TERMINATED。

    注意：只要 agent_data.interaction 非空，没有工具调用的纯文本回复也会被路由回
    INTERACTING（由用户模拟器决定是否终止），而不会直接终止，所以这里必须给
    StubInteraction 准备两轮脚本。
    """

    def test_interaction_continue_then_terminate(self):
        tokenizer = FakeTokenizer()
        replies = [
            _tool_call_text("notify_user_msg", {"msg": "已为您找到咖啡馆"}),
            "好的，正在为您导航。",
        ]
        server_manager = FakeServerManager(tokenizer, replies)
        agent_loop = self._build_loop(tokenizer, server_manager)

        stub_interaction = StubInteraction(
            [
                (False, "麻烦再帮我看一下评分", 0.0),
                (True, "好的，谢谢", 0.3),
            ]
        )
        agent_loop.interaction_config_file = "stub://navi"
        agent_loop.interaction_map = {"navi": stub_interaction}

        output = self._run(
            agent_loop.run(
                sampling_params={},
                raw_prompt=[{"role": "user", "content": "帮我找个咖啡馆"}],
            )
        )

        self.assertEqual(server_manager.call_count, 2)
        self.assertEqual(len(stub_interaction.finalized), 1)


if __name__ == "__main__":
    unittest.main()
