# Navigation Agent Online RL Training

基于verl框架实现的导航Agent在线强化学习训练方案。

## 最新更新 (2026-04-02)

### 新增功能
- **NPU 支持**: 新增 `train_rl_navi_npu.sh` 训练脚本，支持华为昇腾 NPU 设备
- **思考块过滤**: 添加 `_filter_think_blocks` 方法，自动过滤 `erable...urchased`、`...`、`<thought>...</thought>` 等思考块内容
- **约束推理**: 在系统提示中添加 "Constrained Reasoning" 指导，限制思考块内容不超过200字
- **增强日志**: 添加更清晰的 agent loop 调试日志，包括每个 repeat 的开始/结束标记
- **测试文件**: 新增 `test_agent_loop.py`、`test_navi_agent_loop.py` 等测试脚本

### 配置变更
- `max_response_length`: 512 → 1024
- `max_turns`: 30 → 10
- `rollout.n`: 2 → 4
- 新增 `num_repeat_rollouts`: 2

## 训练结果 (2026-04-01)

### 训练摘要

| 指标 | 初始值 | 最终值 | 变化 |
|------|--------|--------|------|
| **平均奖励** | 0.11 | **0.30** | +172% |
| **最小奖励** | -0.10 | **0.08** | 模型学会工具调用 |
| **Perplexity** | 2.53 | **2.06** | -18.6% |
| **Entropy** | 0.90 | **0.76** | 更确定输出 |

### 工具调用奖励分布

| 工具 | 奖励范围 | 说明 |
|------|----------|------|
| `navigation_start` | 0.42 | 最高奖励，成功开始导航 |
| `poi_search` | 0.34-0.40 | POI搜索成功 |
| `notify_user_msg` | 0.14 | 用户播报 |
| `wiki_search` | 0.08 | 百科搜索（使用LLM Mock） |
| `navigation_pathPoint` | 0.04 | 途经点管理 |
| `filter` | 0.04 | 结果筛选 |
| 无工具调用 | -0.10 | 惩罚 |

### Checkpoint 位置

```
/home/ma-user/work/yanglibing/git_obs/verl-npu/outputs/navi_agent_debug/global_step_166/
```

## 快速开始

### Debug模式 (4 GPU, Qwen3-0.6B)

使用空闲GPU 2-5进行快速调试：

```bash
cd /home/ma-user/work/yanglibing/git_obs/verl-npu
sh recipe/navi/train_rl_navi.sh
```

### NPU模式 (8 NPU, Qwen3-0.6B)

在华为昇腾 NPU 上运行：

```bash
cd /home/ma-user/work/yanglibing/git_obs/verl-npu
sh recipe/navi/train_rl_navi_npu.sh
```

> ⚠️ **注意**: NPU 模式需要先激活 CANN 环境变量，如未激活请取消脚本中的注释：
> ```bash
> source /usr/local/Ascend/ascend-toolkit/set_env.sh
> source /usr/local/Ascend/nnal/atb/set_env.sh
> ```

### 完整训练 (8 GPU, Qwen3-30B-A3B)

需要清空所有GPU后运行：

```bash
cd /home/ma-user/work/yanglibing/git_obs/verl-npu
sh recipe/navi/train_rl_navi_full.sh
```

> ⚠️ **注意**: Qwen3-30B-A3B 模型在当前配置下可能会遇到 OOM (Out of Memory) 问题，建议：
> - 降低 `batch_size`
> - 减少 `max_prompt_length` 或 `max_response_length`
> - 或者使用更小的模型

### 本地配置说明

| 配置项 | Debug模式 | NPU模式 | 完整训练 |
|--------|-----------|---------|----------|
| 模型 | Qwen3-0.6B | Qwen3-0.6B | Qwen3-30B-A3B |
| 设备 | 4 GPU (2,3,4,5) | 8 NPU | 8 GPU |
| batch_size | 8 | 8 | 32 |
| max_prompt_length | 6144 | 6144 | 8192 |
| max_response_length | 1024 | 1024 | 2048 |
| tensor_parallel | 4 | 4 | 8 |
| max_turns | 10 | 10 | 8 |
| rollout_n | 4 | 4 | 2 |
| num_repeat_rollouts | 2 | 2 | - |
| use_remove_padding | True | False | True |
| enable_chunked_prefill | 默认 | False | 默认 |

## 概述

本实现参考了以下主要来源：
- `agent_execution_engine_v2.py`: 导航Agent执行引擎，包含轨迹生成、工具调用、沙盒执行、质检逻辑
- `recipe/collabllm`: verl框架中的CollabLLM多轮对话训练实现
- `recipe/swe_agent`: SWE-Agent外部控制模式和工具交互实现

## 架构设计

```
recipe/navi/
├── __init__.py                 # 包初始化
├── navi_agent_loop.py          # 导航Agent循环（核心）- 包含MockSandboxExecutor
├── navi_interaction.py         # 用户交互模拟器
├── tool_mock.py                # 工具Mock引擎（使用本地LLM生成响应）
├── navi_sandbox_tool.py        # 沙盒工具封装（SandboxExecutor → verl Tool）
├── reward_function.py          # 奖励函数
├── process_dataset.py          # 数据处理脚本
├── utils.py                    # 工具函数
├── train_rl_navi.sh            # Debug训练脚本 (4 GPU)
├── train_rl_navi_npu.sh        # NPU训练脚本 (8 NPU)
├── train_rl_navi_full.sh       # 完整训练脚本 (8 GPU)
├── README.md                   # 使用文档
├── config/
│   ├── agent.yaml              # Agent配置
│   ├── navi_interaction_config.yaml  # 交互配置（本地vLLM服务）
│   └── navi_tool_config.yaml   # 工具配置（沙盒工具定义）
├── metrics/
│   ├── __init__.py
│   ├── accuracy.py             # 准确率指标
│   ├── step_validity.py        # 步骤有效性指标
│   └── efficiency.py           # 效率指标
└── tests/                      # 测试文件
    ├── test_agent_loop.py      # Agent循环测试
    ├── test_navi_agent_loop.py # 导航Agent循环测试
    ├── test_navi_interaction.py # 交互测试
    └── test_interaction_load.py # 交互加载测试
```

## 核心组件

### 1. NaviAgentLoop (`navi_agent_loop.py`)

继承自 `ToolAgentLoop`，实现导航特定的Agent循环逻辑：

```python
class NaviAgentLoop(ToolAgentLoop):
    """
    核心功能：
    1. 多轮工具调用支持（POI搜索、路线规划、开始导航）
    2. 与SandboxExecutor的交互（真实API或Mock）
    3. 用户模拟器交互集成
    4. 任务完成检测
    5. 步骤级别奖励计算
    6. 思考块过滤（erable...urchased等）
    """

    async def run(self, sampling_params, **kwargs):
        # 1. 初始化沙盒实例
        sandbox = self._sandbox_executor_cls(...)

        # 2. 生成模型响应
        # 3. 执行工具调用（通过沙盒或verl Tool）
        # 4. 与用户模拟器交互
        # 5. 返回训练输出

    def _filter_think_blocks(self, content: str) -> str:
        """过滤掉思考块内容，只保留实际回复"""
        # 移除 erable...urchased 块
        # 移除  块
        # 移除 <thought>...</thought> 块
        return content
```

### 状态流转

```
PENDING → GENERATING → PROCESSING_TOOLS ↔ GENERATING
                ↓              ↓
           INTERACTING ←──────┘
                ↓
           TERMINATED
```

**关键状态说明：**

| 状态 | 触发条件 | 行为 |
|------|----------|------|
| `PENDING` | 初始化 | 准备prompt，初始化沙盒 |
| `GENERATING` | 模型生成 | 调用vLLM生成响应，解析工具调用 |
| `PROCESSING_TOOLS` | 有工具调用 | 执行工具，计算工具奖励 |
| `INTERACTING` | `notify_user_msg`后或无工具调用 | 调用用户模拟器生成回复，拼接user消息 |
| `TERMINATED` | 任务完成/超长/最大轮次 | 结束轨迹，返回奖励 |

**INTERACTING状态关键逻辑：**

当 `notify_user_msg` 工具执行后，系统进入 `INTERACTING` 状态：
1. 调用 `interaction.generate_response()` 生成模拟用户回复
2. 将回复包装为 `{"role": "user", "content": interaction_responses}` 拼接到 messages
3. 编码后追加到 `prompt_ids`
4. 返回 `GENERATING` 状态继续生成

### 2. MockSandboxExecutor (`navi_agent_loop.py`)

Mock沙盒执行器，集成了 `ToolMockEngine`，支持使用本地LLM模型生成动态响应：

```python
class MockSandboxExecutor:
    """
    Mock沙盒执行器，集成 ToolMockEngine
    """

    # 需要使用 LLM Mock 的工具列表
    LLM_MOCK_TOOLS = ["wiki_search"]

    def __init__(self, ..., use_llm_mock=False):
        self.use_llm_mock = use_llm_mock
        self._tool_mock_engine = None  # 延迟初始化

    def execute(self, call):
        # 如果启用了 LLM Mock，使用本地 LLM 生成响应
        if self.use_llm_mock and tool_name in self.LLM_MOCK_TOOLS:
            return json.loads(self._get_tool_mock_engine().generate_mock_response(...))

        # 否则使用硬编码 fallback
        ...
```

**关键特性：**
- `wiki_search` 等工具使用本地 vLLM 模型生成响应，避免外部 API 限制（如502错误）
- 其他工具使用硬编码的 fallback 响应
- 延迟初始化 `ToolMockEngine`，避免循环导入

### 3. ToolMockEngine (`tool_mock.py`)

工具 Mock 引擎，使用本地 vLLM 模型生成工具返回：

```python
class ToolMockEngine:
    """
    工具 Mock 引擎，使用本地 vLLM 模型生成工具返回

    配置参考 navi_interaction_config.yaml:
    - api_base: http://localhost:12200/v1
    - model: Qwen3-235B-A22B-Thinking-2507-FP8
    """

    # Wiki 搜索专用模板
    WIKI_SEARCH_PROMPT = """你是一个百科知识查询工具API模拟器..."""

    def generate_mock_response(self, tool_name: str, tool_args: Dict) -> str:
        """使用本地 LLM 模型生成工具返回"""
        # 选择合适的模板
        # 调用本地 vLLM 服务
        # 提取 JSON 响应
```

**配置（与 `navi_interaction_config.yaml` 一致）：**
```yaml
api_base: "http://localhost:12200/v1"
model: "Qwen3-235B-A22B-Thinking-2507-FP8"
api_key: "sk-no-key-required"
```

### 4. NaviInteraction (`navi_interaction.py`)

用户交互模拟器，用于：
- 模拟真实用户回复
- 检测任务完成信号
- 生成对话级别的交互数据

### 5. NaviRewardManager (`reward_function.py`)

奖励管理器，计算多个指标：
- **accuracy**: 任务完成准确率
- **step_validity**: 步骤有效性（工具调用是否正确）
- **efficiency**: 效率（避免冗余调用）

## 支持的工具

| 工具名称 | 描述 | Mock方式 |
|----------|------|----------|
| `poi_search` | POI搜索 | 硬编码 |
| `navigation_start` | 开始导航 | 硬编码 |
| `navigation_route` | 路线查询 | 硬编码 |
| `navigation_info_query` | 导航信息查询 | 硬编码 |
| `navigation_memory` | 记忆管理 | 硬编码 |
| `navigation_pathPoint` | 途经点管理 | 硬编码 |
| `filter` | 结果筛选 | 硬编码 |
| `notify_user_msg` | 用户播报 | 硬编码 |
| `wiki_search` | 百科搜索 | **LLM Mock** |

## 使用方法

### 1. 准备数据

将导航训练数据转换为parquet格式：

```bash
# 从JSONL转换
python recipe/navi/process_dataset.py \
    --input /path/to/navi_data.jsonl \
    --output ./data/navi

# 或创建示例数据集
python recipe/navi/process_dataset.py \
    --create_sample \
    --output ./data/navi \
    --num_samples 100
```

### 2. 配置数据路径

编辑 `train_rl_navi.sh`，设置数据路径：

```bash
DATA_TRAIN=/path/to/train.parquet
DATA_VAL=/path/to/val.parquet
```

### 3. 启动训练

```bash
# Debug模式 (推荐先测试)
sh recipe/navi/train_rl_navi.sh

# NPU模式
sh recipe/navi/train_rl_navi_npu.sh

# 完整训练
sh recipe/navi/train_rl_navi_full.sh

# 从checkpoint恢复
sh recipe/navi/train_rl_navi.sh /path/to/checkpoint
```

### 4. 使用真实沙盒

如果安装了 `navi_lab` 包（包含 `SandboxExecutor`），系统会自动加载真实的沙盒执行器。否则使用Mock沙盒。

```bash
# 安装navi_lab（如果需要真实沙盒）
pip install -e /path/to/deepthink-agent
```

## 配置说明

### Agent配置 (`config/agent.yaml`)

```yaml
- name: navi_agent
  _target_: recipe.navi.navi_agent_loop.NaviAgentLoop
```

### 工具配置 (`config/navi_tool_config.yaml`)

定义所有导航工具的schema和实现类：

```yaml
tools:
  - class_name: recipe.navi.navi_sandbox_tool.NaviSandboxTool
    config:
      type: native
      use_real_car_format: true
    tool_schema:
      type: function
      function:
        name: poi_search
        description: 搜索POI
        parameters:
          type: object
          properties:
            mode:
              type: string
              description: 搜索模式
            keyword:
              type: string
              description: 搜索关键词
          required:
            - mode
            - keyword
```

### 交互配置 (`config/navi_interaction_config.yaml`)

使用本地 vLLM 服务：

```yaml
interaction:
  - name: "navi"
    class_name: "recipe.navi.navi_interaction.NaviInteraction"
    config: {
      "user_model": "openai/Qwen3-235B-A22B-Thinking-2507-FP8",
      "api_base": "http://localhost:12200/v1",
      "api_key": "sk-no-key-required",
      "num_retries": 3,
      "max_tokens": 512,
      "temperature": 1.0,
      "enable_log": true
    }
```

## 奖励计算

### 步骤级别奖励

```python
def _calculate_tool_reward(tool_name, result, args, is_completed):
    # 检查工具返回有效性
    if "errorCode" in result and result["errorCode"] not in [0, 200]:
        return -0.5

    # 工具特定奖励
    if tool_name == "poi_search":
        return 0.1 * len(result.get("pois", []))
    elif tool_name == "navigation_start":
        return 0.5
    elif tool_name == "notify_user_msg":
        return 0.1

    # 任务完成奖励
    if is_completed:
        return 1.0

    return 0.1
```

### 对话级别奖励

```python
# 多指标加权组合
total_reward = (
    accuracy_score * accuracy_weight +
    step_validity_score * step_validity_weight +
    efficiency_score * efficiency_weight
)
```

## 导航任务完成信号

系统检测以下信号判断任务完成：
- "开始导航"、"已为您规划"、"导航已启动"
- "导航结束"、"已到达"、"任务完成"

## 与原始实现的对照

| 原始实现 | verl实现 |
|----------|----------|
| `NaviAgentExecutionEngine` | `NaviAgentLoop` |
| `SandboxExecutor` | `NaviSandboxTool` + 直接调用 |
| `NaviTrajectoryData` | `NaviTrajectoryState` |
| `execute_tool_async` | `_execute_with_sandbox` |
| `pre_judge_tool_return` | `_calculate_tool_reward` |
| `check_task_completion` | `check_navigation_completion` |
| `UserMockEngine` | `NaviInteraction` |
| `ToolMockEngine` | `tool_mock.ToolMockEngine` (集成到 MockSandboxExecutor) |

## 训练日志示例

### 训练进度输出

```
Training Progress:  49%|████▉     | 81/166 [52:41<36:37, 25.85s/it]
```

### 奖励计算日志

```
Navi reward: score=0.42, turns=2, tools=['navigation_start'], num_tools=1
Navi reward: score=0.40, turns=2, tools=['poi_search'], num_tools=1
Navi reward: score=0.14, turns=2, tools=['notify_user_msg'], num_tools=1
Navi reward: score=-0.10 (no tools), turns=2
```

### 工具调用日志

```
[NaviAgent] Tool parser result: tool_calls count=1
[NaviAgent] Tool call 0: name=poi_search, arguments={"mode": "关键词", "keyword": "星巴克"}
[NaviAgent] Next state: PROCESSING_TOOLS (has 1 tool calls)
```

### LLM Mock 日志（wiki_search）

```
[NaviAgent] === Sandbox Execution Start ===
[NaviAgent] Sandbox type: MockSandbox (LLM Mock for wiki_search)
[ToolMock] LLM response: {"status": "1", "info": "查询成功", ...}
[MockSandbox] LLM Mock response for wiki_search: {"status": "1", ...}
```

### 交互状态日志

```
[NaviAgent] notify_user_msg executed, skipping tool response, entering INTERACTING state
[NaviAgent] _handle_interacting_state called, interaction=True
[NaviAgent] Calling interaction.generate_response, messages count=7
[NaviAgent] Interaction response: None...
```

### Step指标输出

```
step:166 - actor/entropy:0.76 - critic/score/mean:0.30 - critic/score/max:0.40
         - rollout_corr/training_ppl:2.06 - actor/grad_norm:2.22
         - response_length/mean:303.0 - num_turns/mean:2.0
```

## 扩展开发

### 添加新的工具

1. 在 `config/navi_tool_config.yaml` 中添加工具定义
2. 在 `navi_sandbox_tool.py` 中添加对应的schema
3. 在 `MockSandboxExecutor.execute()` 中添加工具逻辑
4. 如果需要 LLM Mock，将工具名添加到 `LLM_MOCK_TOOLS` 列表

### 添加新的LLM Mock工具

1. 在 `tool_mock.py` 的 `ToolMockEngine` 中添加专用 Prompt 模板
2. 将工具名添加到 `MockSandboxExecutor.LLM_MOCK_TOOLS` 列表
3. 在调用 `MockSandboxExecutor` 时设置 `use_llm_mock=True`

### 添加新的指标

1. 在 `metrics/` 目录下创建新文件，如 `new_metric.py`
2. 实现 `compute_score` 函数
3. 在训练脚本中添加权重配置

## 依赖

- verl >= 0.7.0
- torch
- transformers
- litellm
- pyarrow
- pandas
- requests
- navi_lab (可选，用于真实沙盒)

## 参考

1. `agent_execution_engine_v2.py`: 导航Agent执行引擎实现
2. `recipe/collabllm`: CollabLLM多轮对话训练
3. `recipe/swe_agent`: SWE-Agent外部控制模式
4. verl框架文档: https://github.com/volcengine/verl
