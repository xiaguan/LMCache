#!/usr/bin/env python3
"""Python launcher for the 1p1d disaggregated prefill example.

This script replaces the original shell-based launcher to provide richer
configuration handling, safer cleanup, and automatic shortening of LMCache RPC
socket names to avoid hitting ZeroMQ's 107-character IPC path limit.
"""

from __future__ import annotations

# Standard library
import argparse
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import atexit

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install it with `pip install pyyaml`."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "disagg_launch.yaml"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs"

PROCESSES: list[subprocess.Popen] = []
LOG_HANDLES: list = []
TEMP_FILES: list[Path] = []
CLEANED_UP = False


@dataclass
class LaunchConfig:
    model_path: str
    served_name: str
    service_name: str
    service_slug: str
    tensor_parallel: int
    max_model_len: int
    log_dir: Path
    proxy_host: str
    proxy_port: int
    proxy_zmq_host: str
    proxy_zmq_port: int
    pref_hosts: list[str]
    pref_ports: list[int]
    pref_devices: list[str]
    pref_rpc_ids: list[str]
    dec_hosts: list[str]
    dec_ports: list[int]
    dec_devices: list[str]
    dec_init_ports: list[int]
    dec_alloc_ports: list[int]
    dec_rpc_ids: list[str]
    dec_skip_last_tokens: list[int]


def ensure_env_prereqs() -> None:
    """Check for minimum runtime prerequisites."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable is not set; public models may be inaccessible.", file=sys.stderr)

    if not shutil_which("nvidia-smi"):
        raise SystemExit("nvidia-smi not found; cannot verify GPU availability.")

    num_gpus = count_gpus()
    if num_gpus < 2:
        raise SystemExit("At least 2 GPUs are required for disaggregated prefill.")

    required_modules = ["lmcache", "pandas", "datasets", "vllm", "yaml"]
    for module in required_modules:
        try:
            __import__(module)
        except ImportError as exc:  # pragma: no cover
            msg = f"Missing required Python module '{module}'. Install via `pip install {module}`."
            raise SystemExit(msg) from exc


def count_gpus() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):  # pragma: no cover
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def shutil_which(cmd: str) -> str | None:
    try:
        from shutil import which
    except ImportError:  # pragma: no cover
        return None
    return which(cmd)


def slugify(value: str, max_len: int = 24) -> str:
    import re

    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    if not value:
        value = "lmcache"
    if len(value) <= max_len:
        return value
    digest = hashlib.sha1(value.encode()).hexdigest()[:6]
    truncated = value[: max_len - 7].rstrip("-")
    if not truncated:
        truncated = value[: max_len - 7]
    return f"{truncated}-{digest}"


def sanitize_name(value: str, max_len: int) -> str:
    return slugify(value, max_len=max_len)


def to_list(value, count: int, name: str) -> list:
    if isinstance(value, list):
        result = value
    elif isinstance(value, (tuple, set)):
        result = list(value)
    else:
        result = [value]

    if len(result) == 1 and count > 1:
        result = result * count

    if len(result) != count:
        raise SystemExit(f"Length of '{name}' must be 1 or match the instance count ({count}).")

    return list(result)


def expand_ports(base: Sequence[int] | None, start: int, count: int, name: str) -> list[int]:
    if base is not None:
        if len(base) != count:
            raise SystemExit(f"Length of '{name}' must match the instance count ({count}).")
        return [int(p) for p in base]
    return [int(start) + i for i in range(count)]


def make_rpc_ids(prefix: str, count: int, index_start: int = 1) -> list[str]:
    base = sanitize_name(prefix, max_len=32)
    return [f"{base}{idx}" for idx in range(index_start, index_start + count)]


def load_config(path: Path) -> LaunchConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    model_cfg = data.get("model", {})
    model_path = str(model_cfg.get(
        "name",
        "/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ))
    served_name = str(model_cfg.get("served_name") or Path(model_path).name)
    service_name = str(data.get("service_name") or served_name)
    service_slug = slugify(service_name)
    max_model_len = int(model_cfg.get("max_model_len", 104752))
    tensor_parallel = int(model_cfg.get("tensor_parallel_size", 1))

    runtime_cfg = data.get("runtime", {})
    log_dir = Path(runtime_cfg.get("log_dir", DEFAULT_LOG_DIR)).resolve()

    proxy_cfg = data.get("proxy", {})
    proxy_host = str(proxy_cfg.get("host", "localhost"))
    proxy_port = int(proxy_cfg.get("port", 9100))
    proxy_zmq_host = str(proxy_cfg.get("zmq_host", proxy_host))
    proxy_zmq_port = int(proxy_cfg.get("zmq_port", 7500))

    pref_cfg = data.get("prefillers", {})
    pref_count = int(pref_cfg.get("count", 1))
    if pref_count < 1:
        raise SystemExit("prefillers.count must be >= 1")

    pref_ports = expand_ports(pref_cfg.get("ports"), pref_cfg.get("http_port_start", 7100), pref_count, "prefillers.ports")
    pref_devices = [str(x) for x in to_list(pref_cfg.get("devices", list(range(pref_count))), pref_count, "prefillers.devices")]
    pref_hosts = [str(x) for x in to_list(pref_cfg.get("hosts") or pref_cfg.get("host", "localhost"), pref_count, "prefillers.hosts")]
    pref_rpc_ids = pref_cfg.get("rpc_ids")
    if pref_rpc_ids is not None:
        if len(pref_rpc_ids) != pref_count:
            raise SystemExit("prefillers.rpc_ids length must match prefillers.count")
        pref_rpc_ids = [str(x) for x in pref_rpc_ids]
    else:
        pref_rpc_prefix = str(pref_cfg.get("kv_rpc_prefix", "producer"))
        pref_rpc_ids = make_rpc_ids(pref_rpc_prefix, pref_count)

    dec_cfg = data.get("decoders", {})
    dec_count = int(dec_cfg.get("count", 1))
    if dec_count < 1:
        raise SystemExit("decoders.count must be >= 1")

    dec_ports = expand_ports(dec_cfg.get("ports"), dec_cfg.get("http_port_start", 7200), dec_count, "decoders.ports")
    dec_devices = [str(x) for x in to_list(dec_cfg.get("devices", list(range(dec_count))), dec_count, "decoders.devices")]
    dec_hosts = [str(x) for x in to_list(dec_cfg.get("hosts") or dec_cfg.get("host", "localhost"), dec_count, "decoders.hosts")]
    dec_init_ports = expand_ports(dec_cfg.get("init_ports"), dec_cfg.get("init_port_start", 7300), dec_count, "decoders.init_ports")
    dec_alloc_ports = expand_ports(dec_cfg.get("alloc_ports"), dec_cfg.get("alloc_port_start", 7400), dec_count, "decoders.alloc_ports")
    dec_rpc_ids = dec_cfg.get("rpc_ids")
    if dec_rpc_ids is not None:
        if len(dec_rpc_ids) != dec_count:
            raise SystemExit("decoders.rpc_ids length must match decoders.count")
        dec_rpc_ids = [str(x) for x in dec_rpc_ids]
    else:
        dec_rpc_prefix = str(dec_cfg.get("kv_rpc_prefix", "consumer"))
        dec_rpc_ids = make_rpc_ids(dec_rpc_prefix, dec_count)

    skip_tokens = dec_cfg.get("skip_last_n_tokens", 1)
    if isinstance(skip_tokens, list):
        if len(skip_tokens) != dec_count:
            raise SystemExit("decoders.skip_last_n_tokens length must match decoders.count")
        dec_skip_last_tokens = [int(x) for x in skip_tokens]
    else:
        dec_skip_last_tokens = [int(skip_tokens) for _ in range(dec_count)]

    return LaunchConfig(
        model_path=model_path,
        served_name=served_name,
        service_name=service_name,
        service_slug=service_slug,
        tensor_parallel=tensor_parallel,
        max_model_len=max_model_len,
        log_dir=log_dir,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
        proxy_zmq_host=proxy_zmq_host,
        proxy_zmq_port=proxy_zmq_port,
        pref_hosts=pref_hosts,
        pref_ports=pref_ports,
        pref_devices=pref_devices,
        pref_rpc_ids=pref_rpc_ids,
        dec_hosts=dec_hosts,
        dec_ports=dec_ports,
        dec_devices=dec_devices,
        dec_init_ports=dec_init_ports,
        dec_alloc_ports=dec_alloc_ports,
        dec_rpc_ids=dec_rpc_ids,
        dec_skip_last_tokens=dec_skip_last_tokens,
    )


def join_csv(items: Iterable) -> str:
    return ",".join(str(x) for x in items)


def start_process(cmd: list[str], log_path: Path | None = None, env: dict[str, str] | None = None) -> None:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        LOG_HANDLES.append(log_handle)
        stdout = log_handle
        stderr = subprocess.STDOUT
    else:
        stdout = None
        stderr = None

    process = subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        env=env,
        start_new_session=True,
    )
    PROCESSES.append(process)


def start_proxy(cfg: LaunchConfig) -> None:
    proxy_log = cfg.log_dir / "proxy.log"
    cmd = [
        sys.executable,
        str((SCRIPT_DIR.parent / "disagg_proxy_server.py").resolve()),
        "--host",
        cfg.proxy_host,
        "--port",
        str(cfg.proxy_port),
        "--prefiller-host",
        join_csv(cfg.pref_hosts),
        "--prefiller-port",
        join_csv(cfg.pref_ports),
        "--num-prefillers",
        str(len(cfg.pref_hosts)),
        "--decoder-host",
        join_csv(cfg.dec_hosts),
        "--decoder-port",
        join_csv(cfg.dec_ports),
        "--decoder-init-port",
        join_csv(cfg.dec_init_ports),
        "--decoder-alloc-port",
        join_csv(cfg.dec_alloc_ports),
        "--num-decoders",
        str(len(cfg.dec_hosts)),
        "--proxy-host",
        cfg.proxy_zmq_host,
        "--proxy-port",
        str(cfg.proxy_zmq_port),
    ]
    print(f"Launching proxy on {cfg.proxy_host}:{cfg.proxy_port} (ZMQ {cfg.proxy_zmq_host}:{cfg.proxy_zmq_port})")
    start_process(cmd, proxy_log)


def render_prefiller_config(proxy_host: str, proxy_port: int) -> Path:
    base_config = SCRIPT_DIR / "configs" / "lmcache-prefiller-config.yaml"
    with base_config.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["pd_proxy_host"] = proxy_host
    data["pd_proxy_port"] = int(proxy_port)

    tmp = tempfile.NamedTemporaryFile("w", suffix="-prefiller.yaml", delete=False)
    yaml.safe_dump(data, tmp, sort_keys=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    TEMP_FILES.append(tmp_path)
    return tmp_path


def start_prefillers(cfg: LaunchConfig, pref_config_path: Path) -> None:
    for idx, (host, port, device, rpc_id) in enumerate(
        zip(cfg.pref_hosts, cfg.pref_ports, cfg.pref_devices, cfg.pref_rpc_ids)
    ):
        log_path = cfg.log_dir / f"prefiller_{idx}.log"
        env = os.environ.copy()
        env.update(
            {
                "LMCACHE_CONFIG_FILE": str(pref_config_path),
                "CUDA_VISIBLE_DEVICES": device,
                "HTTP_PORT": str(port),
                "KV_RPC_PORT": rpc_id,
                "MODEL_PATH": cfg.model_path,
                "SERVED_MODEL_NAME": cfg.served_name,
                "SERVICE_NAME": cfg.service_name,
                "SERVICE_SLUG": cfg.service_slug,
                "TENSOR_PARALLEL_SIZE": str(cfg.tensor_parallel),
                "MODEL_MAX_LEN": str(cfg.max_model_len),
                "ROLE": "prefiller",
            }
        )
        cmd = ["bash", str((SCRIPT_DIR / "disagg_vllm_launcher.sh").resolve()), "prefiller"]
        print(f"Launching prefiller[{idx}] on {host}:{port} (GPU {device}) with RPC '{rpc_id}'")
        start_process(cmd, log_path, env)


def start_decoders(cfg: LaunchConfig) -> None:
    base_config = SCRIPT_DIR / "configs" / "lmcache-decoder-config.yaml"
    for idx, (host, port, device, rpc_id, init_port, alloc_port, skip_tokens) in enumerate(
        zip(
            cfg.dec_hosts,
            cfg.dec_ports,
            cfg.dec_devices,
            cfg.dec_rpc_ids,
            cfg.dec_init_ports,
            cfg.dec_alloc_ports,
            cfg.dec_skip_last_tokens,
        )
    ):
        log_path = cfg.log_dir / f"decoder_{idx}.log"
        env = os.environ.copy()
        env.update(
            {
                "LMCACHE_CONFIG_FILE": str(base_config),
                "CUDA_VISIBLE_DEVICES": device,
                "HTTP_PORT": str(port),
                "KV_RPC_PORT": rpc_id,
                "MODEL_PATH": cfg.model_path,
                "SERVED_MODEL_NAME": cfg.served_name,
                "SERVICE_NAME": cfg.service_name,
                "SERVICE_SLUG": cfg.service_slug,
                "TENSOR_PARALLEL_SIZE": str(cfg.tensor_parallel),
                "MODEL_MAX_LEN": str(cfg.max_model_len),
                "ROLE": "decoder",
                "DECODER_INIT_PORT": str(init_port),
                "DECODER_ALLOC_PORT": str(alloc_port),
                "SKIP_LAST_N_TOKENS": str(skip_tokens),
            }
        )
        cmd = ["bash", str((SCRIPT_DIR / "disagg_vllm_launcher.sh").resolve()), "decoder"]
        print(f"Launching decoder[{idx}] on {host}:{port} (GPU {device}) with RPC '{rpc_id}'")
        start_process(cmd, log_path, env)


def wait_for_server(host: str, port: int, timeout: int = 1200) -> None:
    url = f"http://{host}:{port}/v1/completions"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except urllib.error.HTTPError as http_err:
            if http_err.code < 500:
                return
        except urllib.error.URLError:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {url}")


def cleanup() -> None:
    global CLEANED_UP
    if CLEANED_UP:
        return
    CLEANED_UP = True

    if PROCESSES:
        print("Stopping launched processes...")

    for proc in PROCESSES:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(2)
    for proc in PROCESSES:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for proc in PROCESSES:
        try:
            proc.wait(timeout=5)
        except Exception:  # pragma: no cover
            pass

    for handle in LOG_HANDLES:
        try:
            handle.close()
        except Exception:  # pragma: no cover
            pass
    for path in TEMP_FILES:
        try:
            path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover
            pass

    if PROCESSES:
        print("Cleanup complete.")


def finish(exit_code: int = 0) -> None:
    cleanup()
    sys.exit(exit_code)


def install_signal_handlers() -> None:
    def handler(signum, frame):  # type: ignore[override]
        print(f"Received signal {signum}; shutting down...")
        finish(0)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the disaggregated prefill 1p1d example using Python",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip prerequisite checks (HF token, nvidia-smi, Python modules).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the config and print the derived settings without launching.",
    )
    return parser.parse_args()


def print_summary(cfg: LaunchConfig) -> None:
    print("=== Launch Summary ===")
    print(f"Model path          : {cfg.model_path}")
    print(f"Served model name   : {cfg.served_name}")
    print(f"Service slug        : {cfg.service_slug}")
    print(f"Tensor parallel size: {cfg.tensor_parallel}")
    print(f"Max model length    : {cfg.max_model_len}")
    print(f"Log directory       : {cfg.log_dir}")
    print(f"Proxy HTTP          : {cfg.proxy_host}:{cfg.proxy_port}")
    print(f"Proxy ZMQ           : {cfg.proxy_zmq_host}:{cfg.proxy_zmq_port}")
    print(f"Prefillers ({len(cfg.pref_hosts)}) :")
    for idx, (host, port, dev, rpc_id) in enumerate(
        zip(cfg.pref_hosts, cfg.pref_ports, cfg.pref_devices, cfg.pref_rpc_ids)
    ):
        print(f"  [{idx}] host={host} port={port} device={dev} rpc={rpc_id}")
    print(f"Decoders ({len(cfg.dec_hosts)})   :")
    for idx, (host, port, dev, rpc_id, init_port, alloc_port) in enumerate(
        zip(
            cfg.dec_hosts,
            cfg.dec_ports,
            cfg.dec_devices,
            cfg.dec_rpc_ids,
            cfg.dec_init_ports,
            cfg.dec_alloc_ports,
        )
    ):
        print(
            f"  [{idx}] host={host} port={port} init={init_port} alloc={alloc_port} "
            f"device={dev} rpc={rpc_id}"
        )
    print("======================")


atexit.register(cleanup)


def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    print_summary(cfg)

    if args.dry_run:
        return

    if not args.skip_checks:
        ensure_env_prereqs()

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    install_signal_handlers()

    print("Warning: LMCache disaggregated prefill support for vLLM v1 is experimental and subject to change.")

    pref_config_path = render_prefiller_config(cfg.proxy_zmq_host, cfg.proxy_zmq_port)

    try:
        start_proxy(cfg)
        start_decoders(cfg)
        start_prefillers(cfg, pref_config_path)

        print("Waiting for decoder HTTP endpoints...")
        for host, port in zip(cfg.dec_hosts, cfg.dec_ports):
            wait_for_server(host, port)
        print("Waiting for prefiller HTTP endpoints...")
        for host, port in zip(cfg.pref_hosts, cfg.pref_ports):
            wait_for_server(host, port)
        print("Waiting for proxy HTTP endpoint...")
        wait_for_server(cfg.proxy_host, cfg.proxy_port)

        print("==============================")
        print(f"All services are running. Send requests to {cfg.proxy_host}:{cfg.proxy_port}.")
        print(f"Logs are under {cfg.log_dir}")
        print("Press Ctrl+C to terminate.")
        print("Example benchmark command:")
        bench_lines = [
            f"  vllm bench serve --port {cfg.proxy_port} --seed $(date +%s) \\",
            f"      --model {cfg.model_path} \\",
            "      --dataset-name random --random-input-len 7500 --random-output-len 200 \\",
            "      --num-prompts 30 --burstiness 100 --request-rate 1 --ignore-eos",
        ]
        print("\n".join(bench_lines))
        print("==============================")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        finish(0)
    except Exception as exc:
        print(f"Encountered error: {exc}", file=sys.stderr)
        finish(1)


if __name__ == "__main__":
    main()
