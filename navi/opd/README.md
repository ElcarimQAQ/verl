# Navi On-Policy Distillation（OPD）运行指南

本目录提供 Navi 任务的 OPD 启动脚本，用于让 Qwen3.5 student 在自己生成的轨迹上学习 teacher 的 token 分布。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `run_opd_35b_35b_16npu.sh` | 35B student + NaviAgent 35B teacher，单节点 16 张 Ascend NPU |
| `run_opd_35b_122b_16npu.sh` | 35B student + 122B teacher，单节点 16 张 Ascend NPU |
| `opd_smoke_reward.py` | v0 trainer 兼容用的常量 reward；不参与当前直接蒸馏目标 |

## OPD 流程

当前脚本运行的是单轮 continuation OPD：

1. 从 Parquet 数据读取 prompt；
2. student 使用 vLLM 在线生成 response；
3. teacher 对 student 生成的 prompt + response 计算 top-k log-prob；
4. actor 计算 student 在相同 token 位置上的分布；
5. 使用 `forward_kl_topk` 直接反向传播并更新 student。

这不是传统 SFT。SFT 对数据集中固定的 assistant answer 做 teacher forcing；OPD 的 response 由 student 在线采样，再由 teacher 指导。

当前脚本也不是在线 ToolAgentLoop：数据可以包含结构化 JSON `tool_calls` 历史，但 student 新生成的工具调用不会被解析或执行。要做 Agentic OPD，需要额外启用 async rollout、multi-turn、工具配置及与 checkpoint 匹配的 tool parser。

## 默认硬件布局

单节点暴露 16 张 NPU：

```text
NPU 0-7    actor resource pool
            - FSDP2 actor training
            - student vLLM rollout
            - student/reference log-prob

NPU 8-15   teacher resource pool
            - teacher vLLM inference
```

默认并行配置：

| 组件 | NPU 数 | 并行方式 |
| --- | ---: | --- |
| Actor pool | 8 | FSDP2；rollout TP=2 |
| Teacher pool | 8 | TP=8、EP=8、单 replica |

单 teacher 的 replica 数由 verl 自动计算：

```text
num_replicas = teacher_pool_size / per_replica_world_size
             = 8 / 8
             = 1
```

如果将 teacher TP 改成 2，则会自动得到 4 个 replica。多 replica 可能提高吞吐，但每个设备上的权重分片更大；对 15K context 的 35B/122B 模型，默认优先使用 TP=8 降低单设备内存压力。

## 前置条件

目标机器应具备：

- 单节点 16 张 Ascend 910B（或与脚本配置等价的 NPU）；
- 可用的 Ray、PyTorch NPU、vLLM Ascend 和 verl 环境；
- `/usr/local/Ascend/ascend-toolkit/set_env.sh`；
- `/usr/local/Ascend/nnal/atb/set_env.sh`；
- student/teacher checkpoint；
- Navi train/validation Parquet 数据。

启动前建议验证模型和数据路径：

```bash
test -e /path/to/student
test -e /path/to/teacher
test -e /path/to/rl_train.parquet
test -e /path/to/rl_validation.parquet
```

脚本本身也会执行路径检查，并验证 teacher pool 大小可以被 teacher TP 整除。

## Tokenizer 兼容性

OPD 的 teacher/student 必须共享相同的 token 到 ID 映射。仅仅 `vocab_size` 相同并不够。

建议启动前执行：

```python
from transformers import AutoTokenizer

student = AutoTokenizer.from_pretrained(
    "/path/to/student",
    trust_remote_code=True,
)
teacher = AutoTokenizer.from_pretrained(
    "/path/to/teacher",
    trust_remote_code=True,
)

assert len(student) == len(teacher)
assert student.get_vocab() == teacher.get_vocab()
assert student.special_tokens_map == teacher.special_tokens_map
```

如果 token ID 映射不同，teacher 的 top-k token ID 在 student 词表中会指向其他 token，forward KL 将失去正确语义。

## 运行 35B → 35B

默认运行：

```bash
bash navi/opd/run_opd_35b_35b_16npu.sh
```

显式指定路径：

```bash
PROJECT_DIR=/home/ma-user/work/yanglibing/git_obs/verl-npu \
STUDENT_MODEL=/home/ma-user/work/models/Qwen3.5-35B-A3B \
TEACHER_MODEL=/home/ma-user/work/LFR/model/NaviAgent/checkpoint-200 \
TRAIN_FILE=/path/to/rl_train.parquet \
VAL_FILE=/path/to/rl_validation.parquet \
OUT_DIR=/path/to/output \
bash navi/opd/run_opd_35b_35b_16npu.sh
```

## 运行 35B → 122B

```bash
PROJECT_DIR=/home/ma-user/work/yanglibing/git_obs/verl-npu \
STUDENT_MODEL=/home/ma-user/work/models/Qwen3.5-35B-A3B \
TEACHER_MODEL=/path/to/Qwen3.5-122B-A10B \
TRAIN_FILE=/path/to/rl_train.parquet \
VAL_FILE=/path/to/rl_validation.parquet \
bash navi/opd/run_opd_35b_122b_16npu.sh
```

122B teacher 在 8 张 64GB NPU 上是否可运行取决于 checkpoint dtype、vLLM Ascend 版本、模型实现及运行时临时内存。如果 teacher 初始化阶段 OOM，应优先增加 teacher 设备数或确认 TP=8 实际生效；仅降低 batch 无法减少模型权重内存。

## 默认长度与 batch

两份脚本默认：

```text
train batch                 8
PPO mini batch              8
micro batch per GPU         1
max prompt length       14960
max response length       512
max model length         15473
max tokens per GPU       16384
```

`data.filter_overlong_prompts=True` 会过滤超过 prompt 上限的样本，`data.truncation=error` 用于防止仍有异常长样本时静默破坏多轮工具轨迹。

正式训练前应统计 chat-template 渲染后的 token 长度分布，而不是原始字符串长度：

```python
from datasets import load_dataset
from transformers import AutoTokenizer

model_path = "/path/to/student"
data_path = "/path/to/rl_train.parquet"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
dataset = load_dataset("parquet", data_files=data_path, split="train")

lengths = []
for row in dataset:
    ids = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=True,
        add_generation_prompt=True,
    )
    lengths.append(len(ids))

lengths.sort()
for p in (50, 90, 95, 99):
    index = min(len(lengths) - 1, int(len(lengths) * p / 100))
    print(f"P{p}: {lengths[index]}")
print("max:", max(lengths))
```

## 蒸馏配置

默认 loss：

```text
loss_mode             forward_kl_topk
topk                   8
use_policy_gradient    False
use_task_rewards       False
chunked top-k          True
chunk size             1024 tokens
```

含义：

- teacher 为每个位置返回概率最高的 8 个 token；
- student 在对应 token 上计算近似 forward KL；
- distillation loss 直接 backward；
- 不将蒸馏 loss 当作 policy-gradient reward；
- 不混入任务 reward；
- 长序列按 token chunk 计算以降低峰值内存。

## 为什么 reward 恒定返回 1.0

`opd_smoke_reward.py` 是 trainer 兼容层。v0 postprocessing 仍可能调用 reward function，而 Navi 的自定义 `data_source` 不在默认 reward dispatcher 中。

当前配置为：

```text
use_task_rewards=False
use_policy_gradient=False
```

因此常量 `1.0` 不进入直接蒸馏梯度，只防止默认 dispatcher 对未知数据源抛错。

如果以后启用 `use_task_rewards=True`，必须换成真正能区分轨迹质量的 Navi reward，不能继续使用常量 reward。

## 内存优化配置

脚本默认采用：

- actor gradient checkpointing；
- FSDP2 parameter/optimizer offload；
- forward 后 reshard；
- actor 与 rollout log-prob dynamic batching；
- rollout TP=2；
- teacher TP=8/EP=8；
- actor vLLM utilization 0.30；
- teacher vLLM utilization 0.60；
- chunked forward-KL top-k。

第一次运行建议使用更保守参数：

```bash
TRAIN_BATCH_SIZE=4 \
PPO_MINI_BATCH_SIZE=4 \
ACTOR_GPU_MEMORY_UTILIZATION=0.25 \
TEACHER_GPU_MEMORY_UTILIZATION=0.50 \
bash navi/opd/run_opd_35b_35b_16npu.sh \
    distillation.distillation_loss.chunked_topk_chunk_size=512
```

确认稳定后再逐步增加 batch 或 chunk size。

## OOM 排查

需要先根据 traceback 判断 OOM 阶段。

### Actor vLLM 初始化/KV cache OOM

常见日志包含 cache block 或 engine initialization 错误。尝试：

```bash
ACTOR_GPU_MEMORY_UTILIZATION=0.20 \
MAX_PROMPT_LENGTH=12288 \
MAX_TOKEN_LEN_PER_GPU=13312 \
bash navi/opd/run_opd_35b_35b_16npu.sh
```

### Teacher vLLM 初始化 OOM

尝试降低 KV cache 预算：

```bash
TEACHER_GPU_MEMORY_UTILIZATION=0.45 \
bash navi/opd/run_opd_35b_35b_16npu.sh
```

如果权重本身无法加载，降低 utilization 或 batch 不足以解决，应增加 teacher NPU 数、提高有效 TP，或使用支持的 weight/CPU offload 方案。

### Actor backward / forward-KL OOM

优先减小 chunk：

```bash
bash navi/opd/run_opd_35b_35b_16npu.sh \
    distillation.distillation_loss.chunked_topk_chunk_size=512
```

仍然 OOM：

```bash
DISTILLATION_TOPK=4 \
bash navi/opd/run_opd_35b_35b_16npu.sh \
    distillation.distillation_loss.chunked_topk_chunk_size=256 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=256
```

### Teacher 请求并发/KV cache OOM

降低全局 batch：

```bash
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
bash navi/opd/run_opd_35b_35b_16npu.sh
```

降低 batch 可以减少并发请求，但不能解决单条 15K 序列的 logits 或模型权重内存问题。

## 输出与日志

默认输出：

```text
outputs/opd_35b_35b_16npu/
outputs/opd_35b_122b_16npu/
```

日志：

```text
logs/opd_35b_35b_16npu_<timestamp>.log
logs/opd_35b_122b_16npu_<timestamp>.log
```

默认 `save_freq=-1`、`test_freq=-1`、`total_epochs=1`，适合作为 smoke run。正式训练可以通过尾部 Hydra overrides 修改：

```bash
bash navi/opd/run_opd_35b_35b_16npu.sh \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=3
```

## 启动前检查清单

1. student/teacher checkpoint 路径存在；
2. train/validation Parquet 路径存在；
3. teacher/student token 到 ID 映射完全一致；
4. 16 张 NPU 均被 Ray 发现；
5. actor pool 和 teacher pool 总数不超过实际设备数；
6. teacher pool 能被 teacher TP 整除；
7. prompt P99/max 与 `MAX_PROMPT_LENGTH` 匹配；
8. `MAX_TOKEN_LEN_PER_GPU >= MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1`；
9. `use_task_rewards=False` 时才使用常量 smoke reward；
10. 先完成小 batch smoke，再提高吞吐配置。

## 语法检查

修改脚本后至少运行：

```bash
bash -n navi/opd/run_opd_35b_35b_16npu.sh
bash -n navi/opd/run_opd_35b_122b_16npu.sh
```
