#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <prefiller|decoder>" >&2
    exit 1
fi

ROLE="$1"
if [[ "$ROLE" != "prefiller" && "$ROLE" != "decoder" ]]; then
    echo "Invalid role: $ROLE" >&2
    exit 1
fi

export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

MODEL="${MODEL_PATH:-/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}" 
HTTP_PORT="${HTTP_PORT:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MODEL_MAX_LEN="${MODEL_MAX_LEN:-104752}"
KV_RPC_PORT="${KV_RPC_PORT:-}"
SKIP_LAST_N_TOKENS="${SKIP_LAST_N_TOKENS:-1}"
CONFIG_FILE="${LMCACHE_CONFIG_FILE:-$SCRIPT_DIR/configs/lmcache-${ROLE}-config.yaml}"
CUDA_DEVICE_SET="${CUDA_VISIBLE_DEVICES:-}"

if [[ -z "$HTTP_PORT" ]]; then
    echo "HTTP_PORT environment variable must be set before invoking disagg_vllm_launcher." >&2
    exit 1
fi

if [[ -z "$CUDA_DEVICE_SET" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set for $ROLE." >&2
    exit 1
fi

if [[ -z "$KV_RPC_PORT" ]]; then
    echo "KV_RPC_PORT must be provided for $ROLE." >&2
    exit 1
fi

KV_TRANSFER_CONFIG=$(python3 - <<'PY'
import json
import os
role = os.environ["ROLE"]
rpc_port = os.environ["KV_RPC_PORT"]
skip_tokens = int(os.environ.get("SKIP_LAST_N_TOKENS", "1"))
config = {
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_producer" if role == "prefiller" else "kv_consumer",
    "kv_connector_extra_config": {
        "discard_partial_chunks": False,
        "lmcache_rpc_port": rpc_port,
    },
}
if role == "decoder":
    config["kv_connector_extra_config"]["skip_last_n_tokens"] = skip_tokens
print(json.dumps(config))
PY
)

CMD=(vllm serve "$MODEL" \
    --port "$HTTP_PORT" \
    --max-model-len "$MODEL_MAX_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --disable-log-requests \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-transfer-config "$KV_TRANSFER_CONFIG")

if [[ "$SERVED_MODEL_NAME" != "$MODEL" ]]; then
    CMD+=(--served-model-name "$MODEL")
fi

# Add user-provided extra arguments, if any.
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=( $EXTRA_VLLM_ARGS )
    CMD+=("${extra_args[@]}")
fi

upper_role=${ROLE^^}
role_extra_var="${upper_role}_EXTRA_VLLM_ARGS"
if [[ -n "${!role_extra_var:-}" ]]; then
    # shellcheck disable=SC2206
    role_extra=( ${!role_extra_var} )
    CMD+=("${role_extra[@]}")
fi

echo "Launching $ROLE on port $HTTP_PORT with CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_SET"

UCX_TLS=cuda_ipc,cuda_copy,tcp \
    LMCACHE_CONFIG_FILE="$CONFIG_FILE" \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_SET" \
    "${CMD[@]}"
