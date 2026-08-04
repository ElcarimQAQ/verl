#!/usr/bin/env bash
# OPD (on-policy distillation) — Qwen3.5-35B-A3B student +
# NaviAgent Qwen3.5-35B-A3B teacher on 16 Ascend NPUs.
#
# Resource layout on one 16-device node:
#   actor pool:   8 NPUs, rollout TP=2
#   teacher pool: 8 NPUs, inference TP=8, EP=8, one replica
#
# The Parquet dataset keeps tool calls as structured JSON messages. This script
# performs single-turn OPD continuation; it does not enable ToolAgentLoop or a
# tool-call parser. To execute tools online, add the async multi-turn settings
# matching the checkpoint's tool protocol separately.
#
# Usage:
#   STUDENT_MODEL=/path/to/Qwen3.5-35B-A3B \
#   TEACHER_MODEL=/path/to/NaviAgent-Qwen3.5-35B-A3B \
#   bash navi/opd/run_opd_35b_35b_16npu.sh

# Do not enable global `set -u`: Ascend environment scripts may reference
# variables that are not defined by the caller.
set -o pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/ma-user/work/yanglibing/git_obs/verl-npu}
STUDENT_MODEL=${STUDENT_MODEL:-/home/ma-user/work/models/Qwen3.5-35B-A3B}
TEACHER_MODEL=${TEACHER_MODEL:-/home/ma-user/work/LFR/model/NaviAgent-SFT-data0530-Qwen3.6-35B-A3B-0603/checkpoint-200}
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_DIR}/data/navi_pipeline_rounds/rl_train.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_DIR}/data/navi_pipeline_rounds/rl_validation.parquet}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/outputs/opd_35b_35b_16npu}

# Expose all 16 devices to Ray. The trainer and distillation configurations
# create disjoint actor and teacher resource pools below.
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export VERL_NPU_VISIBLE_DEVICES=${VERL_NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}
export WORLD_SIZE=16

set +eu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set +x

export VLLM_PLUGINS=ascend
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
unset RANK_TABLE_FILE
unset RANK_ID

export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_OVER_OFI=${HCCL_OVER_OFI:-1}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export HCCL_EVENT_TIMEOUT=${HCCL_EVENT_TIMEOUT:-7200}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-7200}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export P2P_HCCL_BUFFSIZE=${P2P_HCCL_BUFFSIZE:-30}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-700}
export HCCL_BUFFSIZE_EP=${HCCL_BUFFSIZE_EP:-700}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_TLS_ENABLE=${HCCL_TLS_ENABLE:-0}

ulimit -n 50000

NNODES=${NNODES:-1}
ACTOR_NPUS_PER_NODE=${ACTOR_NPUS_PER_NODE:-8}
TEACHER_NPUS_PER_NODE=${TEACHER_NPUS_PER_NODE:-8}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TEACHER_TP=${TEACHER_TP:-8}
TEACHER_EP=${TEACHER_EP:-8}

# Long rounds can approach 15K prompt tokens. Keep the global batch modest so
# rollout/teacher request bursts do not exhaust KV cache on 64-GB 910Bs.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-14960}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_NUM_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-16384}

ACTOR_GPU_MEMORY_UTILIZATION=${ACTOR_GPU_MEMORY_UTILIZATION:-0.30}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.60}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

if ((TEACHER_NPUS_PER_NODE * NNODES < TEACHER_TP)); then
    echo "Teacher pool is smaller than TEACHER_TP=${TEACHER_TP}." >&2
    exit 2
fi
if (((TEACHER_NPUS_PER_NODE * NNODES) % TEACHER_TP != 0)); then
    echo "Teacher pool size must be divisible by TEACHER_TP=${TEACHER_TP}." >&2
    exit 2
fi
for required_path in "${STUDENT_MODEL}" "${TEACHER_MODEL}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done

start_time=$(date +%Y%m%d_%H%M%S)
mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
LOG_FILE=${LOG_FILE:-${PROJECT_DIR}/logs/opd_35b_35b_16npu_${start_time}.log}
cd "${PROJECT_DIR}" || exit 2

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    reward.custom_reward_function.path="${PROJECT_DIR}/navi/opd/opd_smoke_reward.py"
    reward.custom_reward_function.name=compute_score
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.trust_remote_code=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=False
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.entropy_from_logits_with_chunking=False
    actor_rollout_ref.ref.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ACTOR_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.calculate_log_probs=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=opd_navi
    trainer.experiment_name=qwen3_5_35b_from_35b_a3b_16npu
    trainer.n_gpus_per_node=${ACTOR_NPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="${OUT_DIR}"
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_NPUS_PER_NODE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.expert_parallel_size=${TEACHER_EP}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_NUM_TOKENS}
    distillation.teacher_models.teacher_model.inference.enforce_eager=true
    distillation.distillation_loss.loss_mode=forward_kl_topk
    distillation.distillation_loss.topk=${DISTILLATION_TOPK}
    distillation.distillation_loss.use_chunked_topk=True
    distillation.distillation_loss.chunked_topk_chunk_size=1024
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

ray stop --force 2>/dev/null || true

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${DISTILLATION[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
