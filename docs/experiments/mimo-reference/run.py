#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Orchestrate one bounded two-Spark MiMo reference boot from the workstation.

Starts a memory guard on each node, launches the worker rank then the head
rank of the pinned SGLang image, waits for loopback health on the head, runs
workload.py, and tears down while waiting for GPU memory to be released.
Node names, addresses and interfaces are this measured deployment's inputs,
not llmpalooza configuration.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REMOTE = ".local/share/llmp/mimo-reference"


def ssh(host, command, check=True, timeout=None):
    return subprocess.run(["ssh", host, command], check=check, text=True, capture_output=True, timeout=timeout)


def node_state(host, timeout=10):
    script = ("python3 -c \"import json,subprocess;"
              "m={l.split(':')[0]:int(l.split()[1])*1024 for l in open('/proc/meminfo')};"
              "a=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],"
              "capture_output=True,text=True,check=True,timeout=5).stdout.split();"
              "print(json.dumps({'mem_available':m['MemAvailable'],'compute_apps':a}))\"")
    return json.loads(ssh(host, script, timeout=timeout).stdout)


def server_args(cfg, rank):
    args = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", "/model", "--served-model-name", "mimo",
        # Config parsing only: audited configuration_mimo_v2.py; SGLang's own model code executes.
        "--trust-remote-code",
        "--tp-size", "2", "--ep-size", "2", "--nnodes", "2", "--node-rank", str(rank),
        "--dist-init-addr", f"{cfg['head_address']}:{cfg['dist_port']}",
        "--host", "127.0.0.1", "--port", str(cfg["port"]),
        "--context-length", str(cfg["context_length"]),
        "--max-total-tokens", str(cfg["max_total_tokens"]),
        "--max-running-requests", "4", "--chunked-prefill-size", "2048", "--page-size", "64",
        "--kv-cache-dtype", "fp8_e4m3", "--swa-full-tokens-ratio", "0.03",
        "--mem-fraction-static", str(cfg["mem_fraction"]),
        "--attention-backend", "triton",
        # The engine builds this architecture's multimodal processor even
        # without this flag (boot 1); kept enabled, requests stay text-only.
        "--enable-multimodal", "--mm-attention-backend", "triton_attn",
        "--moe-runner-backend", "flashinfer_mxfp4", "--moe-a2a-backend", "none", "--moe-dense-tp-size", "1",
        "--weight-loader-drop-cache-after-load",
        "--enable-cache-report", "--random-seed", "42", "--log-level", "info",
    ]
    if not cfg["cuda_graph"]:
        args.append("--disable-cuda-graph")
    return args + cfg["extra_args"]


def docker_run(cfg, rank, name, home, user):
    env = {
        "NCCL_SOCKET_IFNAME": cfg["interface"], "GLOO_SOCKET_IFNAME": cfg["interface"],
        "NCCL_IB_HCA": cfg["nccl_ib_hca"], "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        # Non-root: JIT/compile caches live in the per-node cache mount.
        "HOME": "/cache", "XDG_CACHE_HOME": "/cache/.cache", "TRITON_CACHE_DIR": "/cache/triton",
    }
    command = ["sudo", "-n", "docker", "run", "-d", "--name", name, "--gpus", "all", "--user", user,
               "--network", "host", "--shm-size", "16g", "--cap-add", "IPC_LOCK",
               "--ulimit", "memlock=-1:-1", "--device", "/dev/infiniband",
               "--mount", f"type=bind,src={home}/{REMOTE}/model,dst=/model,readonly",
               "--mount", f"type=bind,src={home}/{REMOTE}/cache,dst=/cache"]
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    command += [cfg["image"], *server_args(cfg, rank)]
    return shlex.join(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="new run directory name under the node reference root")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--health-timeout", type=int, default=3600)
    parser.add_argument("--no-workload", action="store_true")
    parser.add_argument("--results", type=Path, required=True, help="private local directory for copied results")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    head, worker = cfg["head"], cfg["worker"]
    nodes = {head: 0, worker: 1}
    run_dir = f"{REMOTE}/runs/{args.name}"
    record = {"config": cfg, "server_args_head": server_args(cfg, 0), "events": {}}
    started = time.monotonic()
    local = args.results.expanduser() / args.name
    local.mkdir(parents=True, exist_ok=True, mode=0o700)

    def event(label):
        record["events"][label] = round(time.monotonic() - started, 3)
        print(f"[{record['events'][label]:9.1f}s] {label}", flush=True)

    for host in nodes:
        state = node_state(host)
        if state["compute_apps"] or state["mem_available"] < cfg["min_start_available_gib"] << 30:
            sys.exit(f"{host} not idle: {state}")
        record.setdefault("preflight", {})[host] = state
        # The run directory itself must be new; its parents may already exist.
        ssh(host, f"mkdir -p -m 700 {REMOTE}/runs {REMOTE}/cache && mkdir -m 700 {run_dir}")
        for f in ("guard.py", "workload.py"):
            subprocess.run(["scp", "-q", str(HERE / f), f"{host}:{run_dir}/"], check=True)
    names = {host: f"llmp-mimo-rank{rank}" for host, rank in nodes.items()}
    guard_attempts = []
    container_attempts = []
    try:
        # Track attempts before SSH: a timeout can leave remote work running.
        for host, name in names.items():
            guard_attempts.append(host)
            # Background only the guard (not a subshell holding ssh's stdout), so ssh returns.
            ssh(host, f"cd {run_dir} || exit 1; nohup python3 guard.py --container {name} --guard-gib {cfg['guard_gib']} "
                      f"--output guard.json > guard.log 2>&1 < /dev/null & echo $! > guard.pid", timeout=60)
        event("guards started")
        for host in (worker, head):
            home = ssh(host, "printf %s \"$HOME\"").stdout
            user = ssh(host, "printf %s:%s \"$(id -u)\" \"$(id -g)\"").stdout
            container_attempts.append(host)
            ssh(host, docker_run(cfg, nodes[host], names[host], home, user))
        event("containers started")
        deadline = time.monotonic() + args.health_timeout
        while True:
            for host, name in names.items():
                status = ssh(host, f"sudo -n docker inspect -f '{{{{.State.Status}}}}' {name}", check=False).stdout.strip()
                if status != "running":
                    raise RuntimeError(f"{name} on {host} is {status or 'missing'}")
            if ssh(head, f"curl -sf -m 5 http://127.0.0.1:{cfg['port']}/health", check=False).returncode == 0:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("health timeout")
            time.sleep(10)
        event("healthy")
        record["post_boot_state"] = {host: node_state(host) for host in nodes}
        if not args.no_workload:
            out = ssh(head, f"cd {run_dir} && python3 workload.py --port {cfg['port']} --output workload.json",
                      check=False)
            record["workload_stdout"] = out.stdout[-4000:]
            record["workload_returncode"] = out.returncode
            event("workload finished")
            out.check_returncode()
    except Exception as error:  # record and fall through to teardown
        record["error"] = repr(error)
        event("error")
    finally:
        def cleanup_attempt(host, phase, action):
            try:
                return action()
            except Exception as error:
                record.setdefault("cleanup_errors", []).append({
                    "host": host, "phase": phase, "error": repr(error),
                })
                record.setdefault("error", "teardown incomplete; see cleanup_errors")
                return None

        released = {}
        pending = set(nodes)
        try:
            for host in (head, worker):
                if host not in container_attempts:
                    continue
                name = names[host]
                cleanup_attempt(host, "container", lambda: ssh(
                    host, f"sudo -n docker stop -t 60 {name}; "
                    f"sudo -n docker logs {name} > {run_dir}/{name}.log 2>&1; "
                    f"sudo -n docker rm -f {name}", timeout=90))
            event("container teardown attempted")
            deadline = time.monotonic() + 300
            polling = set(nodes)
            while polling and time.monotonic() < deadline:
                for host in sorted(polling):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    state = cleanup_attempt(host, "release observation", lambda: node_state(
                        host, timeout=min(10, remaining)))
                    if state is None:
                        # Unknown is not released. Do not delay healthy-node cleanup
                        # with repeated attempts to reach a disconnected node.
                        polling.discard(host)
                    elif not state["compute_apps"]:
                        released[host] = {"seconds": round(time.monotonic() - started, 3), **state}
                        pending.discard(host)
                        polling.discard(host)
                if polling:
                    time.sleep(min(2, max(0, deadline - time.monotonic())))
        finally:
            record["released"] = released
            record["not_released"] = sorted(pending)
            if pending:
                record.setdefault("error", "GPU memory release was not confirmed on every node")
            for host in guard_attempts:
                cleanup_attempt(host, "guard", lambda: ssh(
                    host, f"cd {run_dir} && kill -TERM $(cat guard.pid) && sleep 2", timeout=10))
            event("guard shutdown attempted")
            # Save the local result even if a remote copy fails or is interrupted.
            result_path = local / "run.json"
            result_path.write_text(json.dumps(record, indent=2) + "\n")
            try:
                for host in nodes:
                    cleanup_attempt(host, "results", lambda: subprocess.run(
                        ["rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
                         f"{host}:{run_dir}/", str(local / host)], check=True, timeout=60))
            finally:
                result_path.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({k: record.get(k) for k in (
                "events", "error", "released", "not_released", "cleanup_errors",
            )}, indent=2))
    return 1 if "error" in record else 0


if __name__ == "__main__":
    sys.exit(main())
