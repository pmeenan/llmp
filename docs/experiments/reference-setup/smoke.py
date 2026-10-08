#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""GPU inference and cross-process slot restore on the pinned reference.

Run on the chosen Spark. Raw results and synthetic conversation state go to
an explicitly supplied, new external directory. No latency benchmark claim.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    require(1024 <= args.port <= 65535, "Choose an unprivileged TCP port")
    root = Path(__file__).resolve().parent
    pins = json.loads((root / "artifacts.json").read_text())
    model = args.model.resolve(strict=True)
    require(model.is_file(), "Model must be a regular file")
    require(model.stat().st_size == pins["model"]["size_bytes"], "Model size mismatch")
    with model.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(digest == pins["model"]["sha256"], "Model SHA-256 mismatch")
    output = args.output.resolve()
    # mkdir(exist_ok=False) prevents overwriting previous evidence or slot files.
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    slots = output / "slots"
    slots.mkdir(mode=0o700)
    docker = shlex.split(os.environ.get("DOCKER", "docker"))
    image = pins["engine"]["image"]
    name = "llmp-reference-" + uuid.uuid4().hex[:12]
    api_key = uuid.uuid4().hex
    base = f"http://127.0.0.1:{args.port}"
    # Local API calls must not traverse a proxy configured in the environment.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = {"image": image, "model_sha256": digest, "checks": {}}
    active = False

    def command(*parts, **kwargs):
        return subprocess.run(docker + list(parts), check=True, **kwargs)

    def request(path, payload=None, timeout=180):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + api_key})
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    def start():
        nonlocal active
        cmd = [
            "create", "--name", name, "--pull", "never",
            "--device", "nvidia.com/gpu=all", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--memory", "48g", "--memory-swap", "48g", "--pids-limit", "512",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
            "--publish", f"127.0.0.1:{args.port}:8080",
            "--mount", f"type=bind,src={model},dst=/model.gguf,readonly",
            "--mount", f"type=bind,src={slots},dst=/slots",
            "--env", "CUDA_DISABLE_PTX_JIT=1",
            "--entrypoint", "/app/llama-server", image,
            "--model", "/model.gguf", "--alias", "reference-gemma4",
            "--host", "0.0.0.0", "--port", "8080", "--n-gpu-layers", "999",
            "--ctx-size", "8192", "--parallel", "1", "--threads", "8",
            "--threads-batch", "8", "--batch-size", "512", "--ubatch-size", "512",
            "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--flash-attn", "on", "--cache-ram", "0", "--no-webui",
            "--swa-full",
            "--slot-save-path", "/slots", "--slots", "--metrics",
            "--verbosity", "4", "--api-key", api_key,
        ]
        results["docker_create_arguments"] = cmd[:-1] + ["<generated-per-run>"]
        # Preserve the unique name even if the process is forcibly interrupted.
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        command(*cmd, stdout=subprocess.DEVNULL)
        active = True
        command("start", name, stdout=subprocess.DEVNULL)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            state = command("inspect", "--format", "{{.State.Running}}", name,
                            capture_output=True, text=True).stdout.strip()
            require(state == "true", "Reference server exited during startup")
            try:
                if request("/health", timeout=2).get("status") == "ok":
                    return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(1)
        raise RuntimeError("Reference server did not become healthy in 300 seconds")

    def stop(label):
        nonlocal active
        if active:
            try:
                command("stop", "--time", "30", name, stdout=subprocess.DEVNULL)
            finally:
                try:
                    with (output / f"{label}.log").open("w") as stream:
                        command("logs", name, stdout=stream, stderr=subprocess.STDOUT)
                finally:
                    command("rm", "--force", name, stdout=subprocess.DEVNULL)
                    active = False

    def complete(prompt, cache=True):
        result = request("/completion", {
            "prompt": prompt, "n_predict": 32, "temperature": 0,
            "seed": 42, "cache_prompt": cache, "id_slot": 0,
            "return_tokens": True,
        })
        require(result.get("tokens_predicted", 0) > 0, "No tokens generated")
        require(bool(result.get("tokens")), "Reference did not return token IDs")
        return result

    try:
        start()
        results["models"] = request("/v1/models")
        chat = request("/v1/chat/completions", {
            "model": "reference-gemma4",
            "messages": [{"role": "user", "content": "Reply with the single word READY."}],
            "temperature": 0, "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        require(bool(chat.get("choices")), "Chat API returned no choices")
        require(chat.get("usage", {}).get("completion_tokens", 0) > 0,
                "Chat API did not generate tokens")
        results["chat"] = chat
        # Below the 1024-token SWA window: tests complete prefix reuse, not
        # the long-context/partially-retained SWA policy of the next experiment.
        prompt = "".join(f"Record {i}: the storage shelf contains blue boxes.\n"
                         for i in range(48))
        prompt += "\nQuestion: What color are the boxes?\nAnswer:"
        prompt_tokens = request("/tokenize", {"content": prompt, "add_special": True})["tokens"]
        first = complete(prompt_tokens, cache=False)
        results["initial"] = first
        results["memory_after_initial"] = command(
            "stats", "--no-stream", "--format", "{{json .}}", name,
            capture_output=True, text=True).stdout.strip()
        results["host_meminfo_after_initial"] = Path("/proc/meminfo").read_text()
        saved = request("/slots/0?action=save", {"filename": "gemma-slot.bin"})
        require(saved.get("n_saved", 0) > 0 and saved.get("n_written", 0) > 0,
                "Slot save did not write state")
        results["save"] = saved
        require(len(prompt_tokens) <= saved["n_saved"] < 1024,
                "Saved state must cover the initial prompt within the SWA window")
        suffix = request("/tokenize", {
            "content": "\nQuestion: Name the color again.\nAnswer:", "add_special": False,
        })["tokens"]
        continuation = prompt_tokens + first["tokens"] + suffix
        require(len(continuation) + 32 < 1024, "Smoke must remain within SWA window")
        results["resident_continuation"] = complete(continuation)
        stop("first-process")
        start()
        restored = request("/slots/0?action=restore", {"filename": "gemma-slot.bin"})
        require(restored.get("n_restored") == saved["n_saved"], "Restore token count differs")
        require(restored.get("n_read") == saved["n_written"], "Restore byte count differs")
        results["restore"] = restored
        results["restored_continuation"] = complete(continuation)
        warm = results["resident_continuation"]
        restored_result = results["restored_continuation"]
        require(restored_result["tokens"] == warm["tokens"],
                "Restored continuation differs from resident continuation")
        pending_tokens = len(continuation) - saved["n_saved"]
        require(pending_tokens > 0, "Continuation must extend the saved token prefix")
        restored_prefill = restored_result["timings"]["prompt_n"]
        require(restored_prefill == warm["timings"]["prompt_n"],
                "Restore did more prefill work than the resident continuation")
        require(pending_tokens <= restored_prefill <= pending_tokens + 1,
                "Restore re-prefilled beyond the suffix and at most one cached tail token")
        results["continuation_token_count"] = len(continuation)
        results["expected_pending_tokens"] = pending_tokens
        stop("last-process")
        for label in ("first-process", "last-process"):
            require("offloaded 31/31 layers to GPU" in (output / f"{label}.log").read_text(),
                    "Full GPU offload was not confirmed in the reference log")
        results["checks"] = {
            "all_layers_gpu_ptx_jit_disabled": True,
            "chat_api": True, "completion_api": True,
            "cross_process_restore_counts": True,
            "restored_equals_resident_token_ids": True,
            "restored_prefill_only_suffix_and_at_most_one_tail_token": True,
        }
        print(json.dumps({"checks": results["checks"], "output": str(output)}, indent=2))
    finally:
        try:
            stop("last-process")
        finally:
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    os.umask(0o077)
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    main()
