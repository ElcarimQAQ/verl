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
Navigation Sandbox Tool for verl framework.
封装 deepthink-agent 中的 SandboxExecutor 为 verl Tool 接口
"""

import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionPropertySchema, OpenAIFunctionParametersSchema, OpenAIFunctionSchema, OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# 导航工具定义
NAVI_TOOL_SCHEMAS = {
    "poi_search": OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="poi_search",
            description="搜索POI（兴趣点），支持周边搜索、关键词搜索、沿途搜索等模式",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "mode": OpenAIFunctionPropertySchema(
                        type="string",
                        description="搜索模式：周边、关键词、沿途",
                        enum=["周边", "关键词", "沿途"]
                    ),
                    "keyword": OpenAIFunctionPropertySchema(
                        type="string",
                        description="搜索关键词，如：加油站、餐厅、停车场等"
                    ),
                    "centerLocationReference": OpenAIFunctionPropertySchema(
                        type="string",
                        description="中心位置引用，格式如：poi_search@1#1，表示使用第一次搜索的第一个结果作为中心"
                    ),
                },
                required=["mode", "keyword"]
            )
        )
    ),
    "navigation_start": OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="navigation_start",
            description="开始导航，设置目的地和途经点",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "mode": OpenAIFunctionPropertySchema(
                        type="string",
                        description="导航模式：导航、重新导航",
                        enum=["导航", "重新导航"]
                    ),
                    "desLocationReference": OpenAIFunctionPropertySchema(
                        type="string",
                        description="目的地引用，格式如：poi_search@1#1 或 H1#2（历史记忆引用）"
                    ),
                    "pathPointReference": OpenAIFunctionPropertySchema(
                        type="array",
                        description="途经点引用列表"
                    ),
                    "pathpointNegativeReference": OpenAIFunctionPropertySchema(
                        type="string",
                        description="避让点引用"
                    ),
                    "routeType": OpenAIFunctionPropertySchema(
                        type="string",
                        description="路线偏好：时间优先、距离优先、大路优先、红绿灯少"
                    ),
                },
                required=["mode", "desLocationReference"]
            )
        )
    ),
    "navigation_route": OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="navigation_route",
            description="查询或修改导航路线信息",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "mode": OpenAIFunctionPropertySchema(
                        type="string",
                        description="操作模式：多远多久、路线偏好、路线详情"
                    ),
                    "routeType": OpenAIFunctionPropertySchema(
                        type="string",
                        description="路线偏好类型"
                    ),
                    "includedRoadName": OpenAIFunctionPropertySchema(
                        type="string",
                        description="指定经过的道路名称"
                    ),
                    "excludedRoadName": OpenAIFunctionPropertySchema(
                        type="string",
                        description="指定避让的道路名称"
                    ),
                },
                required=["mode"]
            )
        )
    ),
    "navigation_info_query": OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="navigation_info_query",
            description="查询导航信息，如距离、时间等",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "mode": OpenAIFunctionPropertySchema(
                        type="string",
                        description="查询模式：多远多久"
                    ),
                },
                required=["mode"]
            )
        )
    ),
    "navigation_memory": OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="navigation_memory",
            description="管理导航记忆，如设置家、公司等常用地点",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "mode": OpenAIFunctionPropertySchema(
                        type="string",
                        description="操作模式：记住、删除、查询",
                        enum=["记住", "删除", "查询"]
                    ),
                    "locationType": OpenAIFunctionPropertySchema(
                        type="string",
                        description="位置类型：家、公司"
                    ),
                    "locationReference": OpenAIFunctionPropertySchema(
                        type="string",
                        description="位置引用"
                    ),
                },
                required=["mode"]
            )
        )
    ),
    "navigation_pathPoint": OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="navigation_pathPoint",
            description="管理导航途经点",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "mode": OpenAIFunctionPropertySchema(
                        type="string",
                        description="操作模式：添加、删除",
                        enum=["添加", "删除"]
                    ),
                    "pathPointReference": OpenAIFunctionPropertySchema(
                        type="string",
                        description="途经点引用"
                    ),
                    "insertIndex": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="插入位置索引"
                    ),
                },
                required=["mode"]
            )
        )
    ),
    "filter": OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="filter",
            description="筛选POI搜索结果",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "reference": OpenAIFunctionPropertySchema(
                        type="string",
                        description="要筛选的结果引用，如：poi_search@1"
                    ),
                    "rating_min": OpenAIFunctionPropertySchema(
                        type="number",
                        description="最低评分"
                    ),
                    "businessHour": OpenAIFunctionPropertySchema(
                        type="array",
                        description="营业时间筛选"
                    ),
                },
                required=["reference"]
            )
        )
    ),
    "notify_user_msg": OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="notify_user_msg",
            description="向用户播报消息，用于告知用户当前状态或询问信息",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "msg": OpenAIFunctionPropertySchema(
                        type="string",
                        description="要播报给用户的消息内容"
                    ),
                },
                required=["msg"]
            )
        )
    ),
    "wiki_search": OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="wiki_search",
            description="搜索百科知识，用于回答关于地点、景点的知识性问题",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "query": OpenAIFunctionPropertySchema(
                        type="string",
                        description="搜索查询内容"
                    ),
                },
                required=["query"]
            )
        )
    ),
}


class NaviSandboxState:
    """导航沙盒状态管理"""

    def __init__(self, initial_env: Optional[Dict] = None):
        self.env_state = copy.deepcopy(initial_env) if initial_env else self._get_default_env()
        self.history: List[Dict] = []
        self.history_detail: List[Dict] = []

    def _get_default_env(self) -> Dict:
        """获取默认环境状态"""
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
            "includedRoadName": None,
            "excludedRoadName": None,
            "currentTime": "12:00:00",
            "haveHome": None,
            "haveCompany": None,
            "memoryInfo": []
        }


class NaviSandboxTool(BaseTool):
    """
    导航沙盒工具

    封装 SandboxExecutor 为 verl Tool 接口，支持：
    1. POI搜索
    2. 导航规划
    3. 路线查询
    4. 记忆管理
    5. 用户播报
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: Dict[str, NaviSandboxState] = {}
        self._sandbox_executor_cls = None

    def _get_sandbox_executor(self):
        """延迟加载 SandboxExecutor"""
        if self._sandbox_executor_cls is None:
            try:
                # 尝试从 deepthink-agent 导入
                from navi_lab.sandbox.core2 import SandboxExecutor
                self._sandbox_executor_cls = SandboxExecutor
            except ImportError:
                logger.warning("navi_lab.sandbox.core2 not found, using mock executor")
                self._sandbox_executor_cls = MockSandboxExecutor
        return self._sandbox_executor_cls

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """创建沙盒实例"""
        if instance_id is None:
            instance_id = str(uuid4())

        # 从kwargs获取初始环境配置
        initial_env = kwargs.get("initial_env", None)
        if initial_env is None:
            initial_env = self.config.get("initial_env", None)

        # 创建沙盒状态
        self._instances[instance_id] = NaviSandboxState(initial_env)

        logger.debug(f"[NaviSandbox] Created instance: {instance_id}")
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        执行工具调用

        Args:
            instance_id: 实例ID
            parameters: 工具参数

        Returns:
            (tool_response, tool_reward, tool_metrics)
        """
        if instance_id not in self._instances:
            return ToolResponse(text="Error: Instance not found"), 0.0, {}

        state = self._instances[instance_id]
        tool_name = self.name

        # 构建调用格式
        call = {
            "name": tool_name,
            "arguments": parameters
        }

        try:
            # 使用真实 SandboxExecutor 或 Mock
            SandboxExecutor = self._get_sandbox_executor()
            executor = SandboxExecutor(
                initial_env=state.env_state,
                initial_history=state.history,
                initial_history_detail=state.history_detail,
                use_real_car_format=self.config.get("use_real_car_format", True)
            )

            # 执行工具调用
            result = executor.execute(call)

            # 更新状态
            state.env_state = executor.export_state()
            state.history = executor.export_history()
            state.history_detail = executor.export_history_detail()

            # 处理结果
            if isinstance(result, dict):
                result_str = json.dumps(result, ensure_ascii=False)
            else:
                result_str = str(result)

            # 计算步骤奖励
            tool_reward = self._calculate_step_reward(tool_name, result, parameters)
            tool_metrics = {"tool_name": tool_name, "success": "errorCode" not in str(result)}

            logger.debug(f"[NaviSandbox] Execute {tool_name}: {result_str[:100]}...")

            return ToolResponse(text=result_str), tool_reward, tool_metrics

        except Exception as e:
            logger.error(f"[NaviSandbox] Execute error: {e}")
            return ToolResponse(text=f"Error: {str(e)}"), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """计算实例的累积奖励"""
        if instance_id not in self._instances:
            return 0.0

        state = self._instances[instance_id]

        # 检查是否有目的地
        has_destination = state.env_state.get("destination") is not None

        # 检查是否有导航路线
        has_route = state.env_state.get("routeInfo") is not None

        reward = 0.0
        if has_destination:
            reward += 0.3
        if has_route:
            reward += 0.3

        return reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """释放实例"""
        if instance_id in self._instances:
            del self._instances[instance_id]
            logger.debug(f"[NaviSandbox] Released instance: {instance_id}")

    def _calculate_step_reward(self, tool_name: str, result: Any, parameters: dict) -> float:
        """
        计算步骤级别的奖励

        参考 agent_execution_engine_v2.py 中的 pre_judge_tool_return
        """
        # 检查错误
        if isinstance(result, dict):
            error_code = result.get("errorCode")
            if error_code is not None and error_code not in [0, 200]:
                return -0.5

            if result.get("status") == "0":
                return -0.5

        # 工具特定奖励
        if tool_name == "navigation_start":
            # 成功开始导航给予正向奖励
            return 0.5
        elif tool_name == "poi_search":
            # 成功搜索POI
            if isinstance(result, dict) and "pois" in result:
                return 0.2
        elif tool_name == "notify_user_msg":
            msg = parameters.get("msg", "")
            # 检查任务完成信号
            completion_signals = ["开始导航", "已为您规划", "导航已启动", "已到达"]
            if any(signal in msg for signal in completion_signals):
                return 1.0

        return 0.1  # 默认小正奖励


class MockSandboxExecutor:
    """
    Mock沙盒执行器，用于测试环境
    当真实 SandboxExecutor 不可用时使用
    """

    def __init__(self, initial_env, initial_history, initial_history_detail, use_real_car_format=True):
        self.env_state = copy.deepcopy(initial_env) if initial_env else {}
        self.history = copy.deepcopy(initial_history) if initial_history else []
        self.history_detail = copy.deepcopy(initial_history_detail) if initial_history_detail else []

    def execute(self, call):
        """Mock执行"""
        tool_name = call.get("name", "")
        args = call.get("arguments", {})

        # 返回mock结果
        if tool_name == "poi_search":
            return {
                "pois": [
                    {
                        "id": "MOCK001",
                        "name": f"示例{args.get('keyword', '地点')}1",
                        "address": "上海市浦东新区示例路1号",
                        "location": "121.500000,31.200000",
                        "distance": "500",
                        "type": "生活服务"
                    },
                    {
                        "id": "MOCK002",
                        "name": f"示例{args.get('keyword', '地点')}2",
                        "address": "上海市浦东新区示例路2号",
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
                "route": {
                    "distance": "5000",
                    "duration": "600"
                }
            }
        elif tool_name == "notify_user_msg":
            return {"status": "1", "info": "消息已播报"}
        else:
            return {"status": "1", "info": f"Mock执行 {tool_name}"}

    def export_state(self):
        return copy.deepcopy(self.env_state)

    def export_history(self):
        return copy.deepcopy(self.history)

    def export_history_detail(self):
        return copy.deepcopy(self.history_detail)


def create_navi_tools(config: Optional[Dict] = None) -> List[NaviSandboxTool]:
    """
    创建所有导航工具实例

    Args:
        config: 工具配置

    Returns:
        工具列表
    """
    config = config or {}
    tools = []

    for tool_name, tool_schema in NAVI_TOOL_SCHEMAS.items():
        tool = NaviSandboxTool(
            config=config.get(tool_name, {}),
            tool_schema=tool_schema
        )
        tools.append(tool)

    return tools
