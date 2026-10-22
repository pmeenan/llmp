#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned generic Llama reference inside an outer spark-job --gpu job."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

IMAGE = "sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
DOCKER = ["sudo", "-n", "docker"]
CONFIG_SHA256 = "e7cbee89103cc488ef55c142c2a2a376ea00ce67b49c3ae9282d944107d97c33"


def server_args(port=18973):
    """The exact CLI checked by validate_reference_cli.py in the pinned image."""
    return [
        "-m", "vllm.entrypoints.openai.api_server", "--model", "/model",
        "--served-model-name", "llama33", "--host", "127.0.0.1",
        "--port", str(port), "--load-format", "instanttensor",
        "--dtype", "bfloat16", "--kv-cache-dtype", "float16",
        "--gpu-memory-utilization", "0.65", "--max-model-len", "12288",
        "--max-num-seqs", "1", "--max-num-batched-tokens", "512",
        "--max-logprobs", "1", "--logprobs-mode", "raw_logprobs",
        "--generation-config", "vllm", "--no-enable-prefix-caching",
        "--no-enable-log-requests",
    ]


def stop(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--behavior-inputs", type=Path)
    parser.add_argument("--behavior-sha256")
    parser.add_argument("--phase", choices=["short", "depth", "behavior"], default="short")
    parser.add_argument("--repo", type=Path, default=Path("/home/pmeenan/src/llmp"))
    args = parser.parse_args()
    if args.phase == "behavior":
        if args.behavior_inputs is None or args.behavior_sha256 is None:
            parser.error("behavior phase requires --behavior-inputs and --behavior-sha256")
    elif args.ids is None:
        parser.error("quality phases require --ids")
    args.out.mkdir(parents=True, exist_ok=False)
    if hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("checkpoint config differs from the registered Llama source")
    archive_raw = (args.checkpoint / "llmp-archive.json").read_bytes()
    archive = json.loads(archive_raw)
    if archive.get("repo") != "nvidia/Llama-3.3-70B-Instruct-NVFP4" or archive.get("commit") != "f792b707002f2c613e0729c6b37bbcedc4efa94e":
        raise ValueError("archive receipt names a different checkpoint revision")
    (args.out / "source-archive.json").write_bytes(archive_raw)
    pylib = Path.home() / ".local/share/mia-load-study/pylib"
    cache = Path.home() / ".cache/vllm/triton-u3-llama"
    cache.mkdir(parents=True, exist_ok=True)
    pin_path = args.repo / "docs/experiments/fast-swap/mia-launch.json"
    pins = json.loads(pin_path.read_text())
    if pins["image"] != IMAGE:
        raise ValueError("loader pin names a different reference image")
    for relative, expected in pins["loader_files"].items():
        if hashlib.sha256((pylib / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"installed instanttensor payload changed: {relative}")
    name = "llmp-u3-llama-" + uuid.uuid4().hex[:12]
    port = 18973
    url = f"http://127.0.0.1:{port}"
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    except OSError:
        pass
    else:
        raise RuntimeError(f"reference port is already occupied: {port}")
    command = DOCKER + [
        "run", "--detach", "--name", name, "--device", "nvidia.com/gpu=all",
        "--network", "host", "--ipc", "host", "--entrypoint", "python3",
        "-v", f"{args.checkpoint.resolve()}:/model:ro",
        "-v", f"{pylib}:/opt/instanttensor:ro", "-v", f"{cache}:/triton",
        "-e", "PYTHONPATH=/opt/instanttensor", "-e", "TRITON_CACHE_DIR=/triton",
        "-e", "VLLM_USE_V2_MODEL_RUNNER=1", IMAGE,
    ] + server_args(port)
    (args.out / "launch.json").write_text(json.dumps(
        {"image": IMAGE, "argv": command, "phase": args.phase,
         "loader_pin_sha256": hashlib.sha256(pin_path.read_bytes()).hexdigest()}, indent=2) + "\n")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        subprocess.run(command, check=True)
        deadline = time.monotonic() + 420
        while True:
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("reference readiness deadline exceeded")
            state = subprocess.check_output(
                DOCKER + ["inspect", "--format", "{{.State.Running}}", name], text=True).strip()
            if state != "true":
                raise RuntimeError("reference container exited before readiness")
            time.sleep(2)
        (args.out / "container.json").write_bytes(
            subprocess.check_output(DOCKER + ["inspect", name]))
        collector = Path(__file__).with_name("collect_quality.py")
        # Independent request histories, with prefix caching disabled; neither
        # request inherits a conversation or state from the previous request.
        if args.phase == "behavior":
            subprocess.run([
                "python3", str(Path(__file__).with_name("collect_behavior.py")),
                "--url", url, "--model", "llama33", "--inputs", str(args.behavior_inputs),
                "--inputs-sha256", args.behavior_sha256, "--out", str(args.out / "behavior"),
            ], check=True)
        for repeat in ([] if args.phase == "behavior" else [1, 2]):
            subprocess.run([
                "python3", str(collector), "--url", url, "--model", "llama33",
                "--ids", str(args.ids), "--out", str(args.out / f"repeat-{repeat}"),
                "--phase", args.phase,
            ], check=True)
    finally:
        # The unpredictable owned name makes cleanup safe even if docker run
        # created the container but returned an error or received SIGTERM.
        # A detached container is outside the supervisor's process group.
        # Failed/full log storage must not bypass retirement, and diagnostics
        # must leave time for removal before the supervisor's SIGKILL grace.
        try:
            with (args.out / "container.log").open("wb") as output:
                subprocess.run(DOCKER + ["logs", name], stdout=output,
                               stderr=subprocess.STDOUT, timeout=2, check=False)
        finally:
            removed = subprocess.run(DOCKER + ["rm", "--force", name],
                                     timeout=5, check=False)
            if removed.returncode:
                # Nonexistence after a failed create is fine; a surviving
                # container must be surfaced before handing off the host.
                found = subprocess.run(DOCKER + ["inspect", name],
                                       capture_output=True, timeout=1, check=False)
                if found.returncode == 0:
                    raise RuntimeError(f"owned container survived cleanup: {name}")


if __name__ == "__main__":
    main()
