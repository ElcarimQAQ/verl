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
Navigation Agent Loop for online RL training.

参考实现：
- agent_execution_engine_v2.py: 导航 Agent 执行引擎
- collabllm_agent_loop.py: CollabLLM 多轮对话 Agent 循环

核心功能：
1. 多轮工具调用（POI 搜索、路线规划、导航开始等）
2. 与沙盒 (SandboxExecutor) 的交互 - 模式 A：直接在 AgentLoop 中管理
3. 用户模拟器交互（notify_user_msg 后生成用户回复）
4. 步骤级别奖励计算
"""

import asyncio
import copy as copy_module
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from navi.utils import (
    check_navigation_completion,
    is_valid_messages,
    validate_tool_call,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.interactions.base import BaseInteraction
from verl.tools.schemas import ToolResponse
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.schemas import Message

logger = logging.getLogger('NaviAgent')
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# 配置 handler，确保在 Ray worker 中也能输出日志
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
    logger.addHandler(handler)


# 导航工具终止信号
TERMINATION_SIGNALS = ["导航已开始", "已到达", "导航结束", "任务完成", "导航已完成"]


class NaviAgentLoop(ToolAgentLoop):
    """
    导航 Agent 循环，继承自 ToolAgentLoop，支持：
    1. 多轮工具调用（搜索 POI、规划路线、开始导航等）
    2. 与沙盒 (SandboxExecutor) 的交互 - 模式 A：直接在 AgentLoop 中管理实例
    3. 用户模拟器交互（notify_user_msg 后生成用户回复）
    4. 步骤级别的奖励计算

    沙盒交互方式（模式 A）：
    - 在 run() 开始时创建一个 SandboxExecutor 实例
    - 整个 trajectory 过程中保持同一个实例，维持状态
    - 通过线程池异步执行同步的 sandbox 调用
    """

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        **kwargs,
    ):
        super().__init__(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
            **kwargs,
        )

        # 调试日志：检查父类初始化的工具和工具 schema
        logger.info(f"[NaviAgent] __init__ completed")
        logger.info(f"[NaviAgent] self.tools = {list(self.tools.keys()) if hasattr(self, 'tools') else 'NOT FOUND'}")
        # logger.info(f"[NaviAgent] self.interaction_config_file = {getattr(self, 'interaction_config_file', 'NOT FOUND')}")

        # 从配置中读取 agent.templates.system_template
        agent_config = kwargs.get('agent', {})
        templates = agent_config.get('templates', {})
        system_template = templates.get('system_template', None)
        
        # 如果配置了 system_template，覆盖默认的 system_prompt
        if system_template:
            self.system_prompt = self.tokenizer.encode(system_template, add_special_tokens=False)
            logger.info(f"[NaviAgent] Loaded system_template from config, length={len(self.system_prompt)} tokens")

        # 线程池用于同步沙盒调用
        self._executor = ThreadPoolExecutor(max_workers=10)

        # 尝试加载 SandboxExecutor
        self._sandbox_executor_cls = None
        try:
            from navi_lab.sandbox.core2 import SandboxExecutor
            self._sandbox_executor_cls = SandboxExecutor
            logger.debug("[NaviAgent] Loaded SandboxExecutor from navi_lab.sandbox.core2")
        except Exception as e:
            logger.debug(f"[NaviAgent] navi_lab.sandbox.core2 import failed: {e}, using MockSandboxExecutor")
            self._sandbox_executor_cls = MockSandboxExecutor

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        主运行循环

        参考 agent_execution_engine_v2.py 中的 run_single_trajectory
        """
        messages = list(kwargs["raw_prompt"])
        image_data = kwargs.get("multi_modal_data", {}).get("image", None)
        if image_data is not None and len(image_data) > 0:
            image_data = copy_module.deepcopy(image_data)
        else:
            image_data = None  # Pass None to vLLM for non-multimodal models
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # 获取环境信息
        extra_info = kwargs.get("extra_info", {})
        env_info = extra_info.get("env_info", self._get_default_env())
        user_query = extra_info.get("single_turn_prompt", "")
        if not user_query and messages:
            # 从 messages 中提取用户 query
            for msg in messages:
                if msg.get("role") == "user":
                    user_query = msg.get("content", "")
                    break

        # ========================================
        # 模式 A 核心：创建并保持 sandbox 实例
        # ========================================
        sandbox = self._sandbox_executor_cls(
            initial_env=env_info,
            initial_history=[],
            initial_history_detail=[],
            use_real_car_format=True
        )
        sandbox_type = "RealSandbox (navi_lab)" if self._sandbox_executor_cls.__module__.startswith("navi_lab") else "MockSandbox"
        logger.debug(f"[NaviAgent] Created {sandbox_type} instance for trajectory: {request_id}")

        # 初始化交互模块
        interaction = None
        interaction_kwargs = {}
        logger.info(f"[NaviAgent] === RUN METHOD START ===")
        logger.info(f"[NaviAgent] interaction_config_file: {self.interaction_config_file}")
        logger.info(f"[NaviAgent] extra_info keys: {kwargs.get('extra_info', {}).keys() if kwargs.get('extra_info') else 'None'}")
        logger.info(f"[NaviAgent] messages count: {len(messages)}")
        logger.info(f"[NaviAgent] sandbox type: {type(sandbox).__name__}")
        if self.interaction_config_file:
            # 尝试从 extra_info 中获取 interaction_kwargs
            extra_info = kwargs.get("extra_info", {})
            interaction_kwargs = extra_info.get("interaction_kwargs", {})
            
            logger.info(f"[NaviAgent] interaction_kwargs from data: {interaction_kwargs}")
            logger.info(f"[NaviAgent] extra_info content: {extra_info}")
            logger.info(f"[NaviAgent] interaction_config_file is set, checking interaction_kwargs")
            
            # 如果数据中没有 interaction_kwargs，自动构建默认配置
            if not interaction_kwargs or "name" not in interaction_kwargs:
                logger.info(f"[NaviAgent] interaction_kwargs is empty or missing 'name', auto-generating default")
                # 从 messages 中提取用户查询
                user_query = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_query = msg.get("content", "")
                        break
                
                # 构建默认 interaction_kwargs
                interaction_kwargs = {
                    "name": "navi",
                    "task_desc": "车载导航任务",
                    "single_turn_prompt": user_query,
                }
                logger.info(f"[NaviAgent] Auto-generated interaction_kwargs: {interaction_kwargs}")
            else:
                logger.info(f"[NaviAgent] Using interaction_kwargs from data: {interaction_kwargs}")
            
            interaction_name = interaction_kwargs["name"]
            logger.info(f"[NaviAgent] Looking for interaction: {interaction_name}, available: {list(self.interaction_map.keys())}")
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. "
                    f"Available interactions: {list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            logger.info(f"[NaviAgent] Found interaction module, starting interaction with kwargs: {interaction_kwargs}")
            await interaction.start_interaction(request_id, **interaction_kwargs)
            logger.info(f"[NaviAgent] Interaction started successfully")
        logger.info(f"[NaviAgent] === RUN METHOD INIT COMPLETE ===")

        # 打印初始 messages（调试用）
        logger.debug(f"\n{'='*80}")
        logger.debug(f"[NaviAgent] === INITIAL MESSAGES (count={len(messages)}) ===")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = str(msg.get('content', ''))[:200]
            logger.debug(f"  [{i}] role={role}, content={content}...")
        logger.debug(f"{'='*80}\n")

        # 创建 AgentData 实例，并附加 sandbox 实例
        agent_data = AgentData(
            messages=messages,
            image_data=None,
            video_data=None,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        # 将 sandbox 实例存储在 extra_fields 中，供后续工具调用使用
        agent_data.extra_fields["sandbox"] = sandbox
        agent_data.extra_fields["env_info"] = env_info
        agent_data.extra_fields["user_query"] = user_query

        # 首先生成模型响应
        # logger.info(f"[NaviAgent] === BEFORE FIRST GENERATION ===")
        # logger.info(f"[NaviAgent] agent_data.messages count: {len(agent_data.messages)}")
        # logger.info(f"[NaviAgent] agent_data.prompt_ids length: {len(agent_data.prompt_ids) if hasattr(agent_data, 'prompt_ids') else 'N/A'}")
        await self._handle_pending_state(agent_data, sampling_params)
        # logger.info(f"[NaviAgent] Before _handle_generating_state, messages: {len(agent_data.messages)}")
        # logger.info(f"[NaviAgent] === ENTERING _handle_generating_state ===")
        status = await self._handle_generating_state(agent_data, sampling_params)
        logger.info(f"[NaviAgent] After _handle_generating_state, status={status}, tool_calls={len(agent_data.tool_calls) if hasattr(agent_data, 'tool_calls') else 'N/A'}")
        logger.info(f"[NaviAgent] === FIRST GENERATION COMPLETE ===")

        if status == AgentState.TERMINATED:
            num_repeats = 0
        else:
            # 使用 OmegaConf 的 select 方法或直接访问，避免 .get() 返回 None
            try:
                multi_turn_config = self.config.actor_rollout_ref.rollout.multi_turn
                num_repeats = multi_turn_config.get("num_repeat_rollouts", 1) if multi_turn_config else 1
            except (KeyError, AttributeError, TypeError):
                num_repeats = 1
            if num_repeats is None:
                num_repeats = 1  # Default to 1 if not configured
        logger.info(f"[NaviAgent] num_repeats={num_repeats}, status={status}")
        
        # 打印分隔线，清晰显示每个 agent loop（调试用）
        logger.debug(f"\n{'='*80}")
        logger.debug(f"[NaviAgent] === STARTING {num_repeats} AGENT LOOP REPEATS ===")
        logger.debug(f"[NaviAgent] Request ID: {request_id}")
        logger.debug(f"{'='*80}\n")
        
        interaction_requests = [copy_module.deepcopy(agent_data) for _ in range(num_repeats)]
        messages_lst = []

        for idx, _agent_data in enumerate(interaction_requests):
            # 打印每个 repeat 的开始标记（调试用）
            logger.debug(f"\n{'#'*80}")
            logger.debug(f"{'#'*80}")
            logger.debug(f"[NaviAgent] >>>>>>>>>>>>> AGENT LOOP #{idx+1}/{num_repeats} <<<<<<<<<<<")
            logger.debug(f"{'#'*80}")
            logger.debug(f"{'#'*80}\n")
            
            # 每个 repeat 需要独立的 sandbox 实例
            _agent_data.extra_fields["sandbox"] = self._sandbox_executor_cls(
                initial_env=env_info,
                initial_history=[],
                initial_history_detail=[],
                use_real_car_format=True
            )
            _agent_data.extra_fields["repeat_id"] = idx  # 标记 repeat ID

            logger.info(f"[NaviAgent] === AGENT LOOP #{idx+1}/{num_repeats} START, messages count: {len(_agent_data.messages)} ===")
            if _agent_data.messages:
                last_msg = _agent_data.messages[-1]
                logger.info(f"[NaviAgent] Last message role: {last_msg.get('role', 'N/A')}, content preview: {str(last_msg.get('content', ''))[:100]}")

            # 移除 is_valid_messages 检查，因为第一次生成后的 assistant 消息是正常的
            # is_valid_messages 主要用于检查 <think> 标签配对，但模型生成的消息格式可能不完全符合预期
            # if not is_valid_messages(_agent_data.messages[-1]):
            #     logger.warning(f"[NaviAgent] is_valid_messages returned False, breaking for loop")
            #     break

            prev_msg_len = len(_agent_data.messages)
            
            # 根据第一次生成的结果确定初始状态
            if hasattr(_agent_data, 'tool_calls') and _agent_data.tool_calls:
                initial_state = AgentState.PROCESSING_TOOLS
                logger.info(f"[NaviAgent] Starting repeat with PROCESSING_TOOLS state (has {len(_agent_data.tool_calls)} tool calls)")
            elif self.interaction_config_file:
                initial_state = AgentState.INTERACTING
                logger.info(f"[NaviAgent] Starting repeat with INTERACTING state (has interaction)")
            else:
                initial_state = AgentState.TERMINATED
                logger.info(f"[NaviAgent] Starting repeat with TERMINATED state (no tool calls, no interaction)")
            
            # 记录开始时的状态
            start_turns = _agent_data.user_turns + _agent_data.assistant_turns
            start_tool_rewards = len(_agent_data.tool_rewards)
            start_turn_scores = len(_agent_data.turn_scores)
            
            await self.run_agent_data_loop(_agent_data, sampling_params, initial_state)
            
            # 记录结束时的状态
            end_turns = _agent_data.user_turns + _agent_data.assistant_turns
            end_tool_rewards = len(_agent_data.tool_rewards)
            end_turn_scores = len(_agent_data.turn_scores)
            
            # 打印该 agent loop 的统计信息（调试用）
            logger.info(f"\n{'-'*60}")
            logger.info(f"[NaviAgent] >>> AGENT LOOP #{idx+1}/{num_repeats} SUMMARY <<<")
            logger.info(f"  Total turns: {start_turns} → {end_turns}")
            logger.debug(f"  Tool rewards count: {start_tool_rewards} → {end_tool_rewards}")
            logger.debug(f"  Turn scores count: {start_turn_scores} → {end_turn_scores}")
            if _agent_data.turn_scores:
                logger.debug(f"  Turn scores: {_agent_data.turn_scores}")
            if _agent_data.tool_rewards:
                logger.debug(f"  Tool rewards: {_agent_data.tool_rewards}")
            logger.debug(f"  Final messages count: {len(_agent_data.messages)}")
            logger.debug(f"{'-'*60}\n")
            
            # 转换消息为 Message 对象，需要处理 tool_calls 的 arguments 字段
            # 工具解析器返回的 arguments 是 JSON 字符串，但 Message 需要字典类型
            messages_for_output = []
            for msg in _agent_data.messages:
                msg_copy = copy_module.deepcopy(msg)
                if msg_copy.get("tool_calls"):
                    for tc in msg_copy["tool_calls"]:
                        if isinstance(tc.get("function", {}).get("arguments"), str):
                            try:
                                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode tool call arguments: {tc['function']['arguments']}")
                messages_for_output.append(Message(**msg_copy))
            messages_lst.append(messages_for_output)

            if interaction and interaction.config.get("enable_log"):
                if len(messages_lst[-1]) > prev_msg_len:
                    logger.debug(f"Assistant: ...{messages_lst[-1][prev_msg_len - 1].content[-100:]}")
                    logger.debug(f"User:      {messages_lst[-1][prev_msg_len].content[:100]}...")
            
            # 打印每个 repeat 的结束标记（调试用）
            logger.debug(f"\n{'#'*80}")
            logger.debug(f"[NaviAgent] <<<<<<<<<<< AGENT LOOP #{idx+1}/{num_repeats} FINISHED <<<<<<<<<<<")
            logger.debug(f"{'#'*80}\n")

        # 打印所有 agent loop 完成的标记（调试用）
        logger.debug(f"\n{'='*80}")
        logger.debug(f"[NaviAgent] === ALL {num_repeats} AGENT LOOPS COMPLETED ===")
        logger.debug(f"{'='*80}\n")
        
        # 构建输出
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "messages": {"messages": messages_lst},
                "final_env_state": sandbox.export_state() if hasattr(sandbox, 'export_state') else {},
            },
        )
        return output

    def _filter_think_blocks(self, content: str) -> str:
        """过滤掉思考块内容，只保留实际回复"""
        # 移除 <think>...</think> 块
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # 移除  块
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # 移除 <thought>...</thought> 块
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
        return content

    def _print_messages_trajectory(self, agent_data: AgentData, stage: str):
        """打印当前 messages 轨迹（过滤思考块）"""
        logger.debug(f"\n{'='*80}")
        logger.debug(f"[NaviAgent] === MESSAGES TRAJECTORY @ {stage} ===")
        logger.debug(f"[NaviAgent] Total messages: {len(agent_data.messages)}")
        logger.debug(f"[NaviAgent] user_turns={agent_data.user_turns}, assistant_turns={agent_data.assistant_turns}")
        for i, msg in enumerate(agent_data.messages):
            role = msg.get('role', 'unknown')
            content = str(msg.get('content', ''))
            # 过滤思考块
            content = self._filter_think_blocks(content)
            # 截断长内容
            if len(content) > 300:
                content = content[:300] + "..."
            # 如果有 tool_calls，也打印
            tool_calls = msg.get('tool_calls', None)
            if tool_calls:
                logger.debug(f"  [{i}] role={role}, tool_calls={len(tool_calls)}")
                for tc in tool_calls:
                    tc_name = tc.get('function', {}).get('name', 'unknown')
                    logger.debug(f"       tool: {tc_name}")
            else:
                logger.debug(f"  [{i}] role={role}, content={content}")
        logger.debug(f"{'='*80}\n")

    async def run_agent_data_loop(
        self, agent_data: AgentData, sampling_params: dict[str, Any], state: AgentState
    ):
        """运行 Agent 数据循环"""
        while state != AgentState.TERMINATED:
            logger.info(f"[NaviAgent] === run_agent_data_loop iteration, current state: {state} ===")
            # 打印当前 messages 轨迹
            self._print_messages_trajectory(agent_data, f"run_agent_data_loop state={state.value}")
            if state == AgentState.PENDING:
                logger.info(f"[NaviAgent] Entering PENDING state handler")
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                logger.info(f"[NaviAgent] Entering GENERATING state handler")
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                logger.info(f"[NaviAgent] Entering PROCESSING_TOOLS state handler, tool_calls count: {len(agent_data.tool_calls) if hasattr(agent_data, 'tool_calls') else 'N/A'}")
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                logger.info(f"[NaviAgent] Entering INTERACTING state handler")
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"[NaviAgent] Invalid state: {state}")
                state = AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """
        处理工具调用状态：执行工具并准备响应

        使用模式 A：直接调用 SandboxExecutor

        关键逻辑：
        - notify_user_msg 执行后应该进入 INTERACTING 状态，触发用户模拟器
        - 其他工具执行后进入 GENERATING 状态，继续生成
        """
        # 关键日志：确认进入 PROCESSING_TOOLS 状态处理
        logger.info(f"[NaviAgent] >>> ENTERING _handle_processing_tools_state")
        logger.info(f"[NaviAgent] >>> agent_data.tool_calls count: {len(agent_data.tool_calls) if hasattr(agent_data, 'tool_calls') else 'N/A'}")
        logger.info(f"[NaviAgent] >>> self.max_parallel_calls: {self.max_parallel_calls}")
        if hasattr(agent_data, 'tool_calls') and agent_data.tool_calls:
            for i, tc in enumerate(agent_data.tool_calls):
                logger.info(f"[NaviAgent] >>> Tool call {i}: name={tc.name}, arguments={tc.arguments[:200] if tc.arguments else 'None'}")
        
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        # 获取 sandbox 实例
        sandbox = agent_data.extra_fields.get("sandbox")
        logger.info(f"[NaviAgent] >>> sandbox from extra_fields: {type(sandbox).__name__ if sandbox else 'None'}")

        tasks = []
        tool_calls_info = []  # 保存完整的 tool_call 信息
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._execute_tool_with_sandbox(tool_call, sandbox, agent_data))
            # 解析 arguments 以便后续检查
            try:
                tool_args = json.loads(tool_call.arguments) if tool_call.arguments else {}
            except json.JSONDecodeError:
                tool_args = {}
            tool_calls_info.append({
                "name": tool_call.name,
                "arguments": tool_args,
            })

        responses = await asyncio.gather(*tasks)

        # 标记是否有 notify_user_msg 工具调用
        has_notify_user_msg = False

        # 处理工具响应
        for (tool_response, tool_reward, extra_info), tool_info in zip(responses, tool_calls_info):
            tool_name = tool_info["name"]
            tool_args = tool_info["arguments"]

            # notify_user_msg 特殊处理：不添加 tool response，直接触发用户模拟器
            if tool_name == "notify_user_msg":
                has_notify_user_msg = True
                if tool_reward is not None:
                    agent_data.tool_rewards.append(tool_reward)

                # 打印工具调用信息
                logger.info(f"[NaviAgent] ========== TOOL CALL: {tool_name} (SKIPPED - triggers interaction) ==========")
                logger.info(f"[NaviAgent] Tool arguments: {json.dumps(tool_args, ensure_ascii=False)}")
                logger.debug(f"[NaviAgent] Tool reward: {tool_reward}")
                logger.info(f"[NaviAgent] ========== END TOOL CALL: {tool_name} ==========")

                # 检查任务完成 - 检查 arguments.msg 字段
                msg_content = tool_args.get("msg", "")
                if any(signal in msg_content for signal in TERMINATION_SIGNALS):
                    logger.info(f"[NaviAgent] Task completed with signal in msg: {msg_content[:50]}")
                    agent_data.turn_scores.append(1.0)
                    return AgentState.TERMINATED

                # notify_user_msg 不添加 tool response，直接进入 INTERACTING 状态
                logger.info(f"[NaviAgent] notify_user_msg executed, skipping tool response, entering INTERACTING state")
                continue

            # 其他工具：正常添加 tool response
            message = {"role": "tool", "content": tool_response.text or ""}
            add_messages.append(message)

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            # 打印工具返回 - 增强版日志
            logger.info(f"[NaviAgent] ========== TOOL RESPONSE: {tool_name} ==========")
            print(f"[NaviAgent] Tool reward: {tool_reward}")
            # 完整打印工具返回内容（限制长度避免日志过多）
            response_text = tool_response.text or "None"
            preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            logger.info(f"[NaviAgent] Tool {tool_name}: reward={tool_reward}, response_len={len(response_text)}")
            logger.debug(f"[NaviAgent] Tool response preview: {preview}")

        # 如果只有 notify_user_msg，直接进入 INTERACTING 状态
        if has_notify_user_msg and len(add_messages) == 0:
            logger.info(f"[NaviAgent] Only notify_user_msg executed, entering INTERACTING state directly")
            return AgentState.INTERACTING

        # 有其他工具响应，正常处理
        agent_data.messages.extend(add_messages)

        # 构建响应 IDs
        response_ids = await self.apply_chat_template(
            add_messages,
            images=new_images_this_turn,
            videos=None,
            remove_system_prompt=True,
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1

        # 如果同时有 notify_user_msg 和其他工具，进入 INTERACTING 状态
        if has_notify_user_msg and self.interaction_config_file:
            logger.info(f"[NaviAgent] notify_user_msg + other tools executed, entering INTERACTING state")
            return AgentState.INTERACTING

        return AgentState.GENERATING

    async def _execute_tool_with_sandbox(
        self,
        tool_call,
        sandbox,
        agent_data: AgentData,
    ) -> tuple[ToolResponse, float, dict]:
        """
        使用 SandboxExecutor 执行工具调用

        参考 agent_execution_engine_v2.py 中的 execute_tool_async

        注意：wiki_search 使用 ToolMockEngine（本地 vLLM 模型），避免外部 API 限制影响训练
        """
        tool_name = tool_call.name
        
        # 关键日志：确认进入沙盒执行
        logger.info(f"[NaviAgent] >>> ENTERING _execute_tool_with_sandbox")
        logger.info(f"[NaviAgent] >>> tool_call.name: {tool_name}")
        logger.info(f"[NaviAgent] >>> tool_call.arguments: {tool_call.arguments[:200] if tool_call.arguments else 'None'}")
        logger.info(f"[NaviAgent] >>> sandbox type: {type(sandbox).__name__}")

        try:
            tool_args = json.loads(tool_call.arguments) if tool_call.arguments else {}
        except json.JSONDecodeError as e:
            logger.error(f"[NaviAgent] Failed to decode tool call: {e}")
            logger.error(f"[NaviAgent] Raw tool call arguments: {tool_call.arguments!r}")
            logger.error(f"[NaviAgent] Tool name: {tool_name}, request_id: {agent_data.request_id}")
            tool_args = {}

        # 验证工具调用
        is_valid, reason = validate_tool_call(tool_name, tool_args, agent_data.messages)
        if not is_valid:
            logger.warning(f"[NaviAgent] Invalid tool call: {reason}")
            return ToolResponse(text=f"Error: {reason}"), -0.5, {"error": reason}

        # wiki_search 等工具使用 MockSandboxExecutor（集成了 ToolMockEngine）
        # 其他工具使用真实 Sandbox
        if tool_name == "wiki_search":
            mock_sandbox = MockSandboxExecutor(
                initial_env={},
                initial_history=[],
                initial_history_detail=[],
                use_real_car_format=True,
                use_llm_mock=True  # 启用 LLM Mock
            )
            sandbox_to_use = mock_sandbox
            sandbox_type = "MockSandbox (LLM Mock for wiki_search)"
        else:
            sandbox_to_use = sandbox
            sandbox_type = "MockSandbox" if isinstance(sandbox, MockSandboxExecutor) else "RealSandbox"

        # 构建调用格式
        # 日志：开始执行工具调用
        logger.debug(f"[NaviAgent] === Sandbox Execution Start ===")
        logger.debug(f"[NaviAgent] Sandbox type: {sandbox_type}")
        logger.debug(f"[NaviAgent] Tool name: {tool_name}")
        logger.debug(f"[NaviAgent] Tool args: {json.dumps(tool_args, ensure_ascii=False)}")
        call = {
            "name": tool_name,
            "arguments": tool_args
        }

        try:
            # 使用线程池执行同步沙盒调用
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor,
                sandbox_to_use.execute,
                call
            )

            # 处理结果
            if isinstance(result, dict):
                result_str = json.dumps(result, ensure_ascii=False)
            else:
                result_str = str(result)

            # 计算奖励
            is_completed, completion_reason = check_navigation_completion(tool_name, result_str, tool_args)
            tool_reward = self._calculate_tool_reward(tool_name, result, tool_args, is_completed)

            if is_completed:
                logger.info(f"[NaviAgent] Navigation completed: {completion_reason}")
                tool_reward = 1.0

            logger.debug(f"[NaviAgent] === Sandbox Execution Result ===")
            logger.debug(f"[NaviAgent] Result (first 500 chars): {result_str[:500]}")
            logger.debug(f"[NaviAgent] Tool reward: {tool_reward}")
            logger.debug(f"[NaviAgent] === Sandbox Execution End ===")

            return ToolResponse(text=result_str), tool_reward, {"completed": is_completed}

        except Exception as e:
            logger.error(f"[NaviAgent] Sandbox execution error: {e}")
            import traceback
            traceback.print_exc()
            return ToolResponse(text=f"[ERROR] Sandbox execution failed: {e}"), -1.0, {"error": str(e)}

    def _calculate_tool_reward(
        self,
        tool_name: str,
        result: Any,
        args: Dict,
        is_completed: bool
    ) -> float:
        """
        计算工具调用的奖励

        参考 agent_execution_engine_v2.py 中的 pre_judge_tool_return
        """
        reward = 0.0

        # 检查错误
        if isinstance(result, dict):
            error_code = result.get("errorCode")
            if error_code is not None and error_code not in [0, 200]:
                return -0.5

            if result.get("status") == "0":
                return -0.5

        # 工具特定奖励
        if tool_name == "poi_search":
            if isinstance(result, dict) and "pois" in result:
                pois = result.get("pois", [])
                reward = 0.1 * min(len(pois), 5)
        elif tool_name == "navigation_start":
            reward = 0.5
        elif tool_name == "notify_user_msg":
            reward = 0.1
        else:
            reward = 0.1

        if is_completed:
            reward = max(reward, 1.0)

        return reward

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """
        处理交互状态：获取用户输入
        
        重写父类方法，添加日志来确认交互模块被调用
        """
        logger.info(f"[NaviAgent] _handle_interacting_state called, interaction={agent_data.interaction is not None}")

        # 打印进入交互状态时的 messages
        self._print_messages_trajectory(agent_data, "ENTERING INTERACTING STATE")

        if agent_data.interaction is None:
            logger.warning("[NaviAgent] No interaction module, terminating")
            return AgentState.TERMINATED
        
        logger.info(f"[NaviAgent] Calling interaction.generate_response, messages count={len(agent_data.messages)}")
        
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )

        # 简洁的模拟用户回复日志
        logger.info(f"[NaviAgent] Simulated user response (len={len(interaction_responses) if interaction_responses else 0}, terminate={should_terminate_sequence})")
        if interaction_responses:
            # 只打印前100字符，避免日志过多
            preview = interaction_responses[:100] + "..." if len(interaction_responses) > 100 else interaction_responses
            logger.debug(f"[NaviAgent] User: {preview}")

        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # Check termination condition
        if should_terminate_sequence:
            logger.info("[NaviAgent] Interaction requested termination")
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """
        重写父类方法，添加详细日志来调试工具调用解析
        """
        add_messages: list[dict[str, Any]] = []

        from verl.utils.profiler import simple_timer
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Extract tool calls - 关键调试点（必须在终止条件检查之前）
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        logger.info(f"[NaviAgent] Tool parser result: tool_calls count={len(agent_data.tool_calls)}")
        if agent_data.tool_calls:
            for i, tc in enumerate(agent_data.tool_calls):
                logger.info(f"[NaviAgent] Tool call {i}: name={tc.name}, arguments={tc.arguments[:200] if tc.arguments else 'None'}")
        else:
            # 记录模型生成的原始文本，帮助调试（过滤思考块）
            decoded_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            filtered_text = self._filter_think_blocks(decoded_text)
            # 只打印过滤后的实际回复内容
            if filtered_text:
                logger.info(f"[NaviAgent] Model response (filtered): {filtered_text[:500]}")
            else:
                logger.info(f"[NaviAgent] Model response: [thinking only, no visible output]")

        # Check termination conditions - 工具调用优先于终止条件
        # 如果有工具调用，即使达到长度限制也要先执行工具
        if agent_data.tool_calls:
            logger.info(f"[NaviAgent] Has tool calls, skipping termination checks, will enter PROCESSING_TOOLS")
        elif not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            logger.info(f"[NaviAgent] Terminated: response_mask length ({len(agent_data.response_mask)}) >= response_length ({self.response_length})")
            return AgentState.TERMINATED
        elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            logger.info(f"[NaviAgent] Terminated: assistant_turns ({agent_data.assistant_turns}) >= max ({self.max_assistant_turns})")
            return AgentState.TERMINATED
        elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            logger.info(f"[NaviAgent] Terminated: user_turns ({agent_data.user_turns}) >= max ({self.max_user_turns})")
            return AgentState.TERMINATED

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            logger.info(f"[NaviAgent] Next state: PROCESSING_TOOLS (has {len(agent_data.tool_calls)} tool calls)")
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            logger.info(f"[NaviAgent] Next state: INTERACTING (no tool calls, has interaction)")
            return AgentState.INTERACTING
        else:
            logger.info(f"[NaviAgent] Next state: TERMINATED (no tool calls, no interaction)")
            return AgentState.TERMINATED

    def _get_default_env(self) -> Dict:
        """获取默认环境配置"""
        return {
            "current_location": {
                "name": "当前位置",
                "cityname": "上海市",
                "location": "121.626803,31.256862",
                "formatted_address": "上海市浦东新区",
                "district": "浦东新区"
            },
            "destination": None,
            "waypoints": [],
            "routeType": None,
            "memory": [],
            "routeInfo": None,
        }



class MockSandboxExecutor:
    """
    Mock 沙盒执行器，用于测试环境
    参考 agent_execution_engine_v2.py 中 SandboxExecutor 的接口
    
    集成 ToolMockEngine，支持使用本地 LLM 模型生成动态响应
    """

    # 需要使用 LLM Mock 的工具列表
    LLM_MOCK_TOOLS = ["wiki_search"]

    def __init__(self, initial_env, initial_history, initial_history_detail, 
                 use_real_car_format=True, use_llm_mock=False):
        self.env_state = copy_module.deepcopy(initial_env) if initial_env else {}
        self.history = copy_module.deepcopy(initial_history) if initial_history else []
        self.history_detail = copy_module.deepcopy(initial_history_detail) if initial_history_detail else {}
        self.use_llm_mock = use_llm_mock
        
        # 延迟初始化 ToolMockEngine（避免循环导入）
        self._tool_mock_engine = None

    def _get_tool_mock_engine(self):
        """延迟获取 ToolMockEngine 实例"""
        if self._tool_mock_engine is None:
            from navi.tool_mock import ToolMockEngine
            self._tool_mock_engine = ToolMockEngine()
        return self._tool_mock_engine

    def execute(self, call):
        """执行工具调用"""
        tool_name = call.get("name", "")
        args = call.get("arguments", {})

        # 如果启用了 LLM Mock 且工具在 LLM_MOCK_TOOLS 列表中，使用 LLM 生成响应
        if self.use_llm_mock and tool_name in self.LLM_MOCK_TOOLS:
            try:
                mock_engine = self._get_tool_mock_engine()
                response_str = mock_engine.generate_mock_response(tool_name, args)
                logger.info(f"[MockSandbox] LLM Mock response for {tool_name}: {response_str[:200]}...")
                return json.loads(response_str)
            except Exception as e:
                logger.warning(f"[MockSandbox] LLM Mock failed for {tool_name}: {e}, using fallback")
                # 继续使用硬编码的 fallback

        # 根据工具名返回 mock 结果（硬编码 fallback）
        if tool_name == "poi_search":
            keyword = args.get("keyword", "地点")
            return {
                "pois": [
                    {
                        "id": "MOCK001",
                        "name": f"示例{keyword}1",
                        "address": "上海市浦东新区示例路 1 号",
                        "location": "121.500000,31.200000",
                        "distance": "500",
                        "type": "生活服务"
                    },
                    {
                        "id": "MOCK002",
                        "name": f"示例{keyword}2",
                        "address": "上海市浦东新区示例路 2 号",
                        "location": "121.510000,31.210000",
                        "distance": "800",
                        "type": "生活服务"
                    }
                ]
            }
        elif tool_name == "navigation_start":
            return {
                "status": "1",
                "info": "导航开始",
                "route": {"distance": "5000", "duration": "600"}
            }
        elif tool_name == "notify_user_msg":
            msg = args.get("msg", "")
            return {"status": "1", "info": "消息已播报", "msg": msg}
        elif tool_name == "navigation_route":
            return {"status": "1", "distance": "5000", "duration": "600", "info": "路线查询成功"}
        elif tool_name == "navigation_info_query":
            return {"status": "1", "distance": "5公里", "duration": "10分钟", "info": "查询成功"}
        elif tool_name == "navigation_memory":
            return {"status": "1", "info": "记忆操作成功"}
        elif tool_name == "filter":
            return {"status": "1", "info": "筛选成功"}
        elif tool_name == "wiki_search":
            # Fallback for wiki_search
            return {
                "status": "1",
                "info": f"关于'{args.get('query', '')}'的百科信息",
                "summary": "这是一个示例百科搜索结果（fallback）"
            }
        else:
            return {"status": "1", "info": f"Mock执行 {tool_name}"}

    def export_state(self):
        return copy_module.deepcopy(self.env_state)

    def export_history(self):
        return copy_module.deepcopy(self.history)

    def export_history_detail(self):
        return copy_module.deepcopy(self.history_detail)
