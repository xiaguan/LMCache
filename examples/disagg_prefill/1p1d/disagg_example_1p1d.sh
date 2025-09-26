#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="configs/disagg_launch.yaml"
PIDS=()
TEMP_CONFIGS=()

print_usage() {
    cat <<'USAGE'
Usage: bash disagg_example_1p1d.sh [--config CONFIG_PATH]

Options:
  --config PATH   Path to a YAML file with launch parameters (default: configs/disagg_launch.yaml)
  -h, --help      Show this help message and exit
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            if [[ $# -lt 2 ]]; then
                echo "--config requires a path argument" >&2
                exit 1
            fi
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

check_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "HF_TOKEN is not set. Please export a valid Hugging Face token." >&2
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN does not look correct. It should start with hf_." >&2
        exit 1
    fi
    echo "HF_TOKEN is set and looks valid."
}

check_num_gpus() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found; cannot verify GPU count." >&2
        exit 1
    fi
    local num_gpus
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if (( num_gpus < 2 )); then
        echo "Need at least 2 GPUs to run disaggregated prefill." >&2
        exit 1
    fi
    echo "Detected $num_gpus GPUs."
}

ensure_python_library_installed() {
    local module="$1"
    echo "Checking if Python module '$module' is installed..."
    python3 -c "import $module" >/dev/null 2>&1 || {
        if [[ "$module" == "nixl" ]]; then
            echo "Module '$module' missing. Please install from https://github.com/ai-dynamo/nixl" >&2
        else
            echo "Module '$module' missing. Install via: pip install $module" >&2
        fi
        exit 1
    }
    echo "Module '$module' available."
}

load_config() {
    local config_path="$1"
    local export_script
    export_script=$(python3 - "$config_path" <<'PY'
import shlex
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required but not installed") from exc

config_path = Path(sys.argv[1]).resolve()
with config_path.open("r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}

model_cfg = data.get("model", {})
model_name = model_cfg.get(
    "name",
    "/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)
served_name = model_cfg.get("served_name") or Path(model_name).name
service_name = data.get("service_name") or served_name
max_model_len = int(model_cfg.get("max_model_len", 104752))
tensor_parallel = int(model_cfg.get("tensor_parallel_size", 1))

runtime_cfg = data.get("runtime", {})
log_dir = runtime_cfg.get("log_dir", "logs")

proxy_cfg = data.get("proxy", {})
proxy_host = proxy_cfg.get("host", "localhost")
proxy_port = int(proxy_cfg.get("port", 9100))
proxy_zmq_host = proxy_cfg.get("zmq_host", proxy_host)
proxy_zmq_port = int(proxy_cfg.get("zmq_port", 7500))

pref_cfg = data.get("prefillers", {})
pref_count = int(pref_cfg.get("count", 1))
if pref_count < 1:
    raise SystemExit("prefillers.count must be >= 1")

pref_ports = pref_cfg.get("ports")
if pref_ports is not None:
    if len(pref_ports) != pref_count:
        raise SystemExit("Length of prefillers.ports must equal prefillers.count")
    pref_http_ports = [int(p) for p in pref_ports]
else:
    start = int(pref_cfg.get("http_port_start", 7100))
    pref_http_ports = [start + i for i in range(pref_count)]

pref_devices = pref_cfg.get("devices")
if pref_devices is None:
    pref_devices = list(range(pref_count))
if len(pref_devices) != pref_count:
    raise SystemExit("prefillers.devices length must match prefillers.count")

pref_host_value = pref_cfg.get("hosts") or pref_cfg.get("host", "localhost")
if isinstance(pref_host_value, str):
    pref_hosts = [pref_host_value] * pref_count
else:
    pref_hosts = list(pref_host_value)
    if len(pref_hosts) == 1 and pref_count > 1:
        pref_hosts *= pref_count
    if len(pref_hosts) != pref_count:
        raise SystemExit("prefillers.hosts length must match prefillers.count or be 1")

pref_kv_prefix = pref_cfg.get("kv_rpc_prefix", "prefiller")

# Decoder configuration
dec_cfg = data.get("decoders", {})
dec_count = int(dec_cfg.get("count", 1))
if dec_count < 1:
    raise SystemExit("decoders.count must be >= 1")

if dec_count != pref_count:
    # Warn but allow mismatch; the proxy can broadcast uneven counts if desired
    pass

dec_ports = dec_cfg.get("ports")
if dec_ports is not None:
    if len(dec_ports) != dec_count:
        raise SystemExit("Length of decoders.ports must equal decoders.count")
    dec_http_ports = [int(p) for p in dec_ports]
else:
    start = int(dec_cfg.get("http_port_start", 7200))
    dec_http_ports = [start + i for i in range(dec_count)]

init_ports = dec_cfg.get("init_ports")
if init_ports is not None:
    if len(init_ports) != dec_count:
        raise SystemExit("decoders.init_ports length mismatch")
    dec_init_ports = [int(p) for p in init_ports]
else:
    start = int(dec_cfg.get("init_port_start", 7300))
    dec_init_ports = [start + i for i in range(dec_count)]

alloc_ports = dec_cfg.get("alloc_ports")
if alloc_ports is not None:
    if len(alloc_ports) != dec_count:
        raise SystemExit("decoders.alloc_ports length mismatch")
    dec_alloc_ports = [int(p) for p in alloc_ports]
else:
    start = int(dec_cfg.get("alloc_port_start", 7400))
    dec_alloc_ports = [start + i for i in range(dec_count)]

skip_tokens = dec_cfg.get("skip_last_n_tokens", 1)
if isinstance(skip_tokens, list):
    if len(skip_tokens) != dec_count:
        raise SystemExit("decoders.skip_last_n_tokens length mismatch")
    dec_skip_tokens = [int(x) for x in skip_tokens]
else:
    dec_skip_tokens = [int(skip_tokens) for _ in range(dec_count)]

dec_devices = dec_cfg.get("devices")
if dec_devices is None:
    dec_devices = list(range(dec_count))
if len(dec_devices) != dec_count:
    raise SystemExit("decoders.devices length must match decoders.count")

dec_host_value = dec_cfg.get("hosts") or dec_cfg.get("host", "localhost")
if isinstance(dec_host_value, str):
    dec_hosts = [dec_host_value] * dec_count
else:
    dec_hosts = list(dec_host_value)
    if len(dec_hosts) == 1 and dec_count > 1:
        dec_hosts *= dec_count
    if len(dec_hosts) != dec_count:
        raise SystemExit("decoders.hosts length must match decoders.count or be 1")

dec_kv_prefix = dec_cfg.get("kv_rpc_prefix", "decoder")

service_slug = service_name.lower()
service_slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in service_slug)
while "--" in service_slug:
    service_slug = service_slug.replace("--", "-")
service_slug = service_slug.strip("-") or "lmcache"

pref_rpc_ids = [f"{service_slug}-{pref_kv_prefix}-{idx}" for idx in range(pref_count)]
dec_rpc_ids = [f"{service_slug}-{dec_kv_prefix}-{idx}" for idx in range(dec_count)]

pref_hosts_out = " ".join(shlex.quote(h) for h in pref_hosts)
pref_ports_out = " ".join(str(p) for p in pref_http_ports)
pref_devices_out = " ".join(str(d) for d in pref_devices)
pref_rpc_out = " ".join(shlex.quote(r) for r in pref_rpc_ids)

dec_hosts_out = " ".join(shlex.quote(h) for h in dec_hosts)
dec_ports_out = " ".join(str(p) for p in dec_http_ports)
dec_devices_out = " ".join(str(d) for d in dec_devices)
dec_init_out = " ".join(str(p) for p in dec_init_ports)
dec_alloc_out = " ".join(str(p) for p in dec_alloc_ports)
dec_rpc_out = " ".join(shlex.quote(r) for r in dec_rpc_ids)
dec_skip_out = " ".join(str(s) for s in dec_skip_tokens)

lines = [
    f"MODEL_PATH={shlex.quote(str(model_name))}",
    f"SERVED_MODEL_NAME={shlex.quote(str(served_name))}",
    f"SERVICE_NAME={shlex.quote(str(service_name))}",
    f"SERVICE_SLUG={shlex.quote(service_slug)}",
    f"TENSOR_PARALLEL_SIZE={tensor_parallel}",
    f"MODEL_MAX_LEN={max_model_len}",
    f"LOG_DIR={shlex.quote(str(log_dir))}",
    f"PROXY_HOST={shlex.quote(str(proxy_host))}",
    f"PROXY_PORT={proxy_port}",
    f"PROXY_ZMQ_HOST={shlex.quote(str(proxy_zmq_host))}",
    f"PROXY_ZMQ_PORT={proxy_zmq_port}",
    f"PREFILLER_COUNT={pref_count}",
    f"PREFILLER_HOSTS=({pref_hosts_out})",
    f"PREFILLER_HTTP_PORTS=({pref_ports_out})",
    f"PREFILLER_DEVICES=({pref_devices_out})",
    f"PREFILLER_RPC_IDS=({pref_rpc_out})",
    f"DECODER_COUNT={dec_count}",
    f"DECODER_HOSTS=({dec_hosts_out})",
    f"DECODER_HTTP_PORTS=({dec_ports_out})",
    f"DECODER_DEVICES=({dec_devices_out})",
    f"DECODER_INIT_PORTS=({dec_init_out})",
    f"DECODER_ALLOC_PORTS=({dec_alloc_out})",
    f"DECODER_RPC_IDS=({dec_rpc_out})",
    f"DECODER_SKIP_LAST_TOKENS=({dec_skip_out})",
]

print("\n".join(lines))
PY
    )

    eval "$export_script"
}

cleanup() {
    local status=${1:-0}
    echo "Stopping launched processes..."
    trap - INT TERM USR1 EXIT

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            echo "Sending SIGTERM to PID $pid"
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done

    sleep 2

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            echo "Force killing PID $pid"
            kill -9 "$pid" >/dev/null 2>&1 || true
        fi
    done

    for cfg in "${TEMP_CONFIGS[@]}"; do
        if [[ -f "$cfg" ]]; then
            rm -f "$cfg"
        fi
    done

    echo "Cleanup complete."
    exit "$status"
}

join_by() {
    local IFS="$1"
    shift
    local first=1
    for element in "$@"; do
        if (( first )); then
            printf '%s' "$element"
            first=0
        else
            printf '%s%s' "$IFS" "$element"
        fi
    done
}

wait_for_server() {
    local host="$1"
    local port="$2"
    local timeout_seconds=1200
    local start_time
    start_time=$(date +%s)

    echo "Waiting for server on ${host}:${port}..."

    while true; do
        if curl -s "http://${host}:${port}/v1/completions" >/dev/null; then
            return 0
        fi
        local now
        now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then
            echo "Timeout waiting for ${host}:${port}" >&2
            return 1
        fi
        sleep 1
    done
}

render_prefiller_config() {
    local output_path
    output_path=$(mktemp)
    python3 - "$SCRIPT_DIR/configs/lmcache-prefiller-config.yaml" "$PROXY_ZMQ_HOST" "$PROXY_ZMQ_PORT" "$output_path" <<'PY'
import sys

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required but not installed") from exc

base_path, proxy_host, proxy_port, out_path = sys.argv[1:5]
with open(base_path, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}

data["pd_proxy_host"] = proxy_host
data["pd_proxy_port"] = int(proxy_port)

with open(out_path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(data, handle, sort_keys=False)
PY
    TEMP_CONFIGS+=("$output_path")
    printf '%s' "$output_path"
}

ensure_log_dir() {
    mkdir -p "$LOG_DIR"
}

launch_proxy() {
    local pref_host_csv pref_port_csv dec_host_csv dec_port_csv dec_init_csv dec_alloc_csv
    pref_host_csv=$(join_by , "${PREFILLER_HOSTS[@]}")
    pref_port_csv=$(join_by , "${PREFILLER_HTTP_PORTS[@]}")
    dec_host_csv=$(join_by , "${DECODER_HOSTS[@]}")
    dec_port_csv=$(join_by , "${DECODER_HTTP_PORTS[@]}")
    dec_init_csv=$(join_by , "${DECODER_INIT_PORTS[@]}")
    dec_alloc_csv=$(join_by , "${DECODER_ALLOC_PORTS[@]}")

    python3 ../disagg_proxy_server.py \
        --host "$PROXY_HOST" \
        --port "$PROXY_PORT" \
        --prefiller-host "$pref_host_csv" \
        --prefiller-port "$pref_port_csv" \
        --num-prefillers "$PREFILLER_COUNT" \
        --decoder-host "$dec_host_csv" \
        --decoder-port "$dec_port_csv" \
        --decoder-init-port "$dec_init_csv" \
        --decoder-alloc-port "$dec_alloc_csv" \
        --num-decoders "$DECODER_COUNT" \
        --proxy-host "$PROXY_ZMQ_HOST" \
        --proxy-port "$PROXY_ZMQ_PORT" \
        > >(tee "$LOG_DIR/proxy.log") 2>&1 &
    local proxy_pid=$!
    PIDS+=($proxy_pid)
}

launch_decoders() {
    local base_config="$SCRIPT_DIR/configs/lmcache-decoder-config.yaml"
    for (( idx=0; idx<DECODER_COUNT; idx++ )); do
        local port="${DECODER_HTTP_PORTS[$idx]}"
        local device="${DECODER_DEVICES[$idx]}"
        local rpc_id="${DECODER_RPC_IDS[$idx]}"
        local skip_tokens="${DECODER_SKIP_LAST_TOKENS[$idx]}"
        local log_file="$LOG_DIR/decoder_${idx}.log"

        LMCACHE_CONFIG_FILE="$base_config" \
        CUDA_VISIBLE_DEVICES="$device" \
        HTTP_PORT="$port" \
        KV_RPC_PORT="$rpc_id" \
        SKIP_LAST_N_TOKENS="$skip_tokens" \
        MODEL_PATH="$MODEL_PATH" \
        SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
        SERVICE_NAME="$SERVICE_NAME" \
        SERVICE_SLUG="$SERVICE_SLUG" \
        TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
        MODEL_MAX_LEN="$MODEL_MAX_LEN" \
        ROLE="decoder" \
        bash "$SCRIPT_DIR/disagg_vllm_launcher.sh" decoder \
            > >(tee "$log_file") 2>&1 &
        local pid=$!
        PIDS+=($pid)
    done
}

launch_prefillers() {
    for (( idx=0; idx<PREFILLER_COUNT; idx++ )); do
        local port="${PREFILLER_HTTP_PORTS[$idx]}"
        local device="${PREFILLER_DEVICES[$idx]}"
        local rpc_id="${PREFILLER_RPC_IDS[$idx]}"
        local log_file="$LOG_DIR/prefiller_${idx}.log"
        local config_path
        config_path=$(render_prefiller_config)

        LMCACHE_CONFIG_FILE="$config_path" \
        CUDA_VISIBLE_DEVICES="$device" \
        HTTP_PORT="$port" \
        KV_RPC_PORT="$rpc_id" \
        MODEL_PATH="$MODEL_PATH" \
        SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
        SERVICE_NAME="$SERVICE_NAME" \
        SERVICE_SLUG="$SERVICE_SLUG" \
        TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
        MODEL_MAX_LEN="$MODEL_MAX_LEN" \
        ROLE="prefiller" \
        bash "$SCRIPT_DIR/disagg_vllm_launcher.sh" prefiller \
            > >(tee "$log_file") 2>&1 &
        local pid=$!
        PIDS+=($pid)
    done
}

main() {
    echo "Warning: LMCache disaggregated prefill support for vLLM v1 is experimental and subject to change."

    ensure_python_library_installed yaml
    ensure_python_library_installed lmcache
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm

    check_num_gpus

    load_config "$CONFIG_FILE"
    ensure_log_dir

    trap 'cleanup $?' INT TERM USR1 EXIT

    echo "Launching proxy..."
    launch_proxy

    echo "Launching decoder instances (${DECODER_COUNT})..."
    launch_decoders

    echo "Launching prefiller instances (${PREFILLER_COUNT})..."
    launch_prefillers

    echo "Waiting for decoder HTTP endpoints..."
    for (( idx=0; idx<DECODER_COUNT; idx++ )); do
        wait_for_server "${DECODER_HOSTS[$idx]}" "${DECODER_HTTP_PORTS[$idx]}"
    done

    echo "Waiting for prefiller HTTP endpoints..."
    for (( idx=0; idx<PREFILLER_COUNT; idx++ )); do
        wait_for_server "${PREFILLER_HOSTS[$idx]}" "${PREFILLER_HTTP_PORTS[$idx]}"
    done

    echo "Waiting for proxy HTTP endpoint..."
    wait_for_server "$PROXY_HOST" "$PROXY_PORT"

    echo "==================================================="
    echo "All services are running. Send requests to ${PROXY_HOST}:${PROXY_PORT}."
    echo "Logs available under $LOG_DIR/"
    echo "Press Ctrl+C to stop all components."
    echo "==================================================="

    while true; do
        sleep 1
    done
}

main
