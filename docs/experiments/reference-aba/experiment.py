#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned llama.cpp A→B→A reference experiment; run on the selected Linux node.

All generated traces, model files, state, telemetry, logs, and raw results must
be outside Git. Python orchestrates the external native reference only.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import mmap
import os
import platform
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
GIB = 1 << 30
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
LIBC.mincore.restype = ctypes.c_int


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def memory():
    return {k.rstrip(":"): int(v) * 1024 for k, v, *rest in
            (line.split() for line in Path("/proc/meminfo").read_text().splitlines())}


def system_counters(device):
    fields = [int(n) for n in (Path("/sys/class/block") / device / "stat").read_text().split()]
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return {"read_bytes": fields[2] * 512, "write_bytes": fields[6] * 512,
            "pswpin": int(vm["pswpin"]), "pswpout": int(vm["pswpout"]),
            "oom_kill": int(vm["oom_kill"])}


def cache(path):
    size = path.stat().st_size
    pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
        view = (ctypes.c_char * size).from_buffer(mapping)
        flags = (ctypes.c_ubyte * pages)()
        rc = LIBC.mincore(ctypes.addressof(view), size, flags)
        del view
        check(rc == 0, f"mincore failed: errno {ctypes.get_errno()}")
        resident = sum(flag & 1 for flag in flags)
    return {"size_bytes": size, "resident_pages": resident, "pages": pages,
            "resident_fraction": resident / pages}


def condition(path, kind):
    if kind == "warm":
        with path.open("rb") as stream:
            while stream.read(8 << 20):
                pass
    else:
        with path.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    state = cache(path)
    check(state["resident_fraction"] >= .999 if kind == "warm" else state["resident_pages"] == 0,
          f"Cannot establish {kind} cache for {path.name}: {state}")
    return state


class Telemetry:
    def __init__(self, output, release_pressure=None, pressure_alive=None):
        self.output = output
        self.done = threading.Event()
        self.samples = []
        self.release_pressure = release_pressure
        self.low_memory = False
        self.pressure_alive = pressure_alive
        self.pressure_lost = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.done.is_set():
            m = memory()
            self.samples.append({"monotonic_ns": time.monotonic_ns(), **m})
            if m["MemAvailable"] < 2 * GIB and self.release_pressure:
                self.low_memory = True
                self.release_pressure()
            if self.pressure_alive and not self.pressure_alive():
                self.pressure_lost = True
            self.done.wait(.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.done.set()
        self.thread.join()
        dump(self.output / "memory-samples.json", self.samples)

    def summary(self, start=0, end=1 << 64):
        samples = [s for s in self.samples if start <= s["monotonic_ns"] <= end]
        check(bool(samples), "No memory samples cover the measured cycle")
        return {"sample_interval_ms": 50, "sample_count": len(samples),
                "min_available_bytes": min(s["MemAvailable"] for s in samples),
                "max_total_minus_available_bytes": max(s["MemTotal"]-s["MemAvailable"] for s in samples),
                "max_dirty_bytes": max(s["Dirty"] for s in samples),
                "max_writeback_bytes": max(s["Writeback"] for s in samples)}


class Router:
    def __init__(self, args, pins, directory, normal=False, maximum=1):
        self.args, self.pins, self.directory = args, pins, directory
        self.normal, self.maximum = normal, maximum
        self.name = "llmp-aba-" + uuid.uuid4().hex[:12]
        self.key = uuid.uuid4().hex
        self.docker = shlex.split(os.environ.get("DOCKER", "docker"))
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.states = directory / "states"
        self.states.mkdir(mode=0o700)
        common = ["version = 1", "[*]", "n-gpu-layers = 999", "threads = 8",
                  "threads-batch = 8", "parallel = 1", "batch-size = 512", "ubatch-size = 512",
                  "cache-type-k = f16", "cache-type-v = f16", "flash-attn = on",
                  "slot-save-path = /states", "slots = true", "metrics = true",
                  "load-on-startup = false", "stop-timeout = 60"]
        if not normal:
            common += ["cache-ram = 0"]
        for key, ctx in (("A", 32768), ("B", 8192)):
            common += [f"[{key}]", f"model = /models/{pins['models'][key]['filename']}", f"ctx-size = {ctx}"]
            if key == "A" and not normal:
                common += ["swa-full = true"]
        (directory / "models.ini").write_text("\n".join(common) + "\n")
        self.created = False

    def cmd(self, *parts, **kwargs):
        return subprocess.run(self.docker + list(parts), check=True, **kwargs)

    def request(self, endpoint, payload=None, timeout=600):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.args.port}" + endpoint, data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        with self.opener.open(req, timeout=timeout) as response:
            return json.load(response)

    def start(self):
        argv = ["create", "--name", self.name, "--pull", "never", "--device", "nvidia.com/gpu=all",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--user", f"{os.getuid()}:{os.getgid()}", "--pids-limit", "512",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m", "--publish", f"127.0.0.1:{self.args.port}:8080",
                "--mount", f"type=bind,src={self.args.models.resolve()},dst=/models,readonly",
                "--mount", f"type=bind,src={self.directory / 'models.ini'},dst=/models.ini,readonly",
                "--mount", f"type=bind,src={self.states},dst=/states",
                "--env", "CUDA_DISABLE_PTX_JIT=1", "--entrypoint", "/app/llama-server", self.pins["image"],
                "--models-preset", "/models.ini", "--models-max", str(self.maximum),
                "--host", "0.0.0.0", "--port", "8080", "--no-webui", "--verbosity", "4",
                "--api-key", self.key]
        dump(self.directory / "container.json", {"name": self.name, "argv": argv[:-1] + ["<ephemeral>"]})
        self.cmd(*argv, stdout=subprocess.DEVNULL)
        self.created = True
        self.cmd("start", self.name, stdout=subprocess.DEVNULL)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                data = self.request("/models", timeout=2)
                if {m["id"] for m in data["data"]} == {"A", "B"}:
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(.1)
        raise RuntimeError("Router startup timed out")

    def stop(self):
        if self.created:
            try:
                self.cmd("stop", "--time", "60", self.name, stdout=subprocess.DEVNULL)
            finally:
                try:
                    with (self.directory / "server.log").open("w") as stream:
                        self.cmd("logs", self.name, stdout=stream, stderr=subprocess.STDOUT)
                finally:
                    self.cmd("rm", "--force", self.name, stdout=subprocess.DEVNULL)
                    self.created = False

    def status(self):
        return {item["id"]: item["status"] for item in self.request("/models")["data"]}

    def wait(self, model, value):
        start = time.monotonic()
        while time.monotonic() - start < 300:
            state = self.status()[model]
            check(not state.get("failed"), f"Model {model} failed: {state}")
            if state["value"] == value:
                return state
            time.sleep(.02)
        raise RuntimeError(f"Model {model} did not become {value}")

    def load(self, model):
        state = self.status()[model]
        if state["value"] == "loaded":
            return state
        self.request("/models/load", {"model": model})
        return self.wait(model, "loaded")

    def unload(self, model):
        self.request("/models/unload", {"model": model})
        self.wait(model, "unloaded")

    def tokens(self, model, content, add_special=False):
        return self.request("/tokenize", {"model": model, "content": content,
                                         "add_special": add_special, "parse_special": True})["tokens"]

    def template(self, model, content):
        return self.template_messages(model, [{"role": "user", "content": content}])

    def template_messages(self, model, messages):
        response = self.request("/apply-template", {"model": model, "messages": messages,
                                                   "chat_template_kwargs": {"enable_thinking": False}})
        return self.tokens(model, response["prompt"])

    def complete(self, model, tokens, predict=128, cache_prompt=True):
        payload = {"model": model, "prompt": tokens, "n_predict": predict, "temperature": 0,
                   "seed": 42, "cache_prompt": cache_prompt, "id_slot": 0, "return_tokens": True,
                   "stream": True}
        req = urllib.request.Request(f"http://127.0.0.1:{self.args.port}/completion",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        start = time.monotonic_ns()
        first = visible = None
        first_io = None
        ids, text_parts, final = [], [], None
        with self.opener.open(req, timeout=600) as response:
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                now = time.monotonic_ns()
                check("error" not in event, f"Stream error: {event}")
                if event.get("tokens"):
                    if first is None:
                        first_io = system_counters(self.args.device)
                    first = now if first is None else first
                    ids.extend(event["tokens"])
                if event.get("content"):
                    visible = now if visible is None else visible
                    text_parts.append(event["content"])
                if event.get("stop"):
                    final = event
        check(first is not None and final is not None, "Incomplete generated-token stream")
        check(not final.get("truncated"), "Context was truncated")
        return {"request_start_ns": start, "first_token_ns": first, "first_text_ns": visible,
                "end_ns": time.monotonic_ns(), "tokens": ids, "content": "".join(text_parts),
                "io_at_first_token": first_io,
                "final": final, "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}

    def save(self, model):
        started = time.monotonic_ns()
        file = self.states / (model + ".bin")
        result = self.request("/slots/0?action=save", {"model": model, "filename": file.name})
        written = time.monotonic_ns()
        check(result["n_saved"] > 0 and result["n_written"] == file.stat().st_size, "State save size mismatch")
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
        fd = os.open(self.states, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return {**result, "api_elapsed_s": (written-started)/1e9,
                "durable_elapsed_s": (time.monotonic_ns()-started)/1e9,
                "allocated_bytes": file.stat().st_blocks * 512}

    def restore(self, model, saved):
        file = self.states / (model + ".bin")
        result = self.request("/slots/0?action=restore", {"model": model, "filename": file.name})
        check(result["n_restored"] == saved["n_saved"], "State restore token mismatch")
        check(result["n_read"] == saved["n_written"] == file.stat().st_size, "State restore byte mismatch")
        return result


def records(start, count):
    return "".join(f"Record {i:04d}: service svc-{i%17} uses shard {i%11}; planned timeout {20+i%80} milliseconds. Status is healthy.\n"
                   for i in range(start, start+count))


def prepare(args, pins):
    target = args.output.resolve()
    target.mkdir(parents=True, mode=0o700, exist_ok=False)
    router = Router(args, pins, target, maximum=2)
    trace = {"version": 1, "A_turns": [], "sampling": {"temperature": 0, "seed": 42}}
    try:
        router.start()
        router.load("A")
        messages = [{"role": "user", "content": "Keep this technical notebook for later questions.\n" + records(0, 512) +
                     "\nReply with only CHECK1."}]
        prompt = router.template_messages("A", messages)
        for turn in range(3):
            check(len(prompt) + 128 < 32768, "Trace exceeds context")
            result = router.complete("A", prompt)
            trace["A_turns"].append({"prompt": prompt, "expected_tokens": result["tokens"], "result": result})
            question = (records(512 + turn*64, 64) + f"\nReply with only CHECK{turn+2}.") if turn < 2 else (
                "Using the notebook, list the integers from one through thirty-two in order, separated by commas. Do not add anything else.")
            messages += [{"role": "assistant", "content": result["content"]},
                         {"role": "user", "content": question}]
            previous = prompt + result["tokens"]
            prompt = router.template_messages("A", messages)
            # Completed Gemma assistant turns drop the generation-only empty
            # thought prefix. Preserve the canonical template and measure that
            # short tail recomputation rather than inventing chat delimiters.
            check(common_prefix(previous, prompt) >= len(previous)-64,
                  "Chat template rewrote more than the short generation tail")
        trace["A_continuation"] = prompt
        trace["A_save"] = router.save("A")
        resident = router.complete("A", prompt)
        trace["A_expected_continuation"] = resident["tokens"]
        trace["A_resident_probe"] = resident
        router.unload("A")
        router.load("A")
        trace["A_restore"] = router.restore("A", trace["A_save"])
        trace["A_restored_probe"] = router.complete("A", prompt)
        check(trace["A_restored_probe"]["tokens"] == resident["tokens"], "Long-context A restore differs")
        verify_reuse(trace["A_restored_probe"], trace["A_save"], prompt,
                     a_reusable_prefix(trace, trace["A_save"]["n_saved"]))
        router.unload("A")
        router.load("B")
        trace["B_prompt"] = router.template("B", "A service retried a request after a timeout. Give three short checks for avoiding duplicate work.")
        b = router.complete("B", trace["B_prompt"], predict=64)
        trace["B_expected_tokens"] = b["tokens"]
        trace["B_initial_probe"] = b
        trace["B_save"] = router.save("B")
        trace["B_probe_continuation"] = trace["B_prompt"] + b["tokens"] + router.tokens("B", "\nName the first check again:")
        trace["B_resident_probe"] = router.complete("B", trace["B_probe_continuation"], predict=32)
        router.unload("B")
        router.load("B")
        trace["B_restore"] = router.restore("B", trace["B_save"])
        trace["B_restored_probe"] = router.complete("B", trace["B_probe_continuation"], predict=32)
        trace["B_restored_matches_resident"] = trace["B_restored_probe"]["tokens"] == trace["B_resident_probe"]["tokens"]
        check(trace["B_restored_matches_resident"], "B restore differs from resident continuation")
        verify_reuse(trace["B_restored_probe"], trace["B_save"], trace["B_probe_continuation"])
    finally:
        router.stop()
        dump(target / "prepare-results.json", trace)
    frozen = freeze_trace(trace)
    dump(target / "trace.json", frozen)
    print(json.dumps({"trace": str(target / "trace.json"), "sha256": sha(target / "trace.json"),
                      "A_prompt_lengths": [len(t["prompt"]) for t in trace["A_turns"]],
                      "A_continuation_length": len(trace["A_continuation"]),
                      "B_restored_matches_resident": trace["B_restored_matches_resident"]}, indent=2))


def freeze_trace(trace):
    reused = a_reusable_prefix(trace, trace["A_save"]["n_saved"])
    return {"version": 1, "sampling": trace["sampling"],
            "A_turns": [{"prompt": t["prompt"], "expected_tokens": t["expected_tokens"]} for t in trace["A_turns"]],
            "A_continuation": trace["A_continuation"], "A_expected_continuation": trace["A_expected_continuation"],
            "B_prompt": trace["B_prompt"], "B_expected_tokens": trace["B_expected_tokens"],
            "A_reuse": {"n_saved": trace["A_save"]["n_saved"], "common_prefix_tokens": reused,
                        "rewritten_saved_tail_tokens": trace["A_save"]["n_saved"]-reused}}


CASES = {
    "resident": ("resident", "warm", 0, False),
    "warm_restore": ("restore", "warm", 0, False),
    "warm_recompute": ("recompute", "warm", 0, False),
    "cold_restore": ("restore", "cold", 0, False),
    "cold_recompute": ("recompute", "cold", 0, False),
    "pressure_restore": ("restore", "cold", 80, False),
    "pressure_recompute": ("recompute", "cold", 80, False),
    "normal_pressure_restore": ("restore", "cold", 80, True),
    "normal_pressure_recompute": ("recompute", "cold", 80, True),
}


class Pressure:
    def __init__(self, program, gib, directory, expected_sha):
        self.program, self.gib, self.directory = program, gib, directory
        self.process = None
        self.lock = threading.Lock()
        self.receipt = {"locked_bytes": 0}
        self.expected_sha = expected_sha

    def start(self):
        if not self.gib:
            return
        check(sha(self.program) == self.expected_sha, "Pressure helper hash mismatch")
        self.process = subprocess.Popen(["sudo", "-n", str(self.program.resolve()), str(self.gib)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        line = self.process.stdout.readline()
        check(bool(line), "Pressure allocation failed")
        self.receipt = json.loads(line)
        status = Path(f"/proc/{self.receipt['pid']}/status").read_text()
        fields = {s.split(":")[0]: s.split(":")[1].strip() for s in status.splitlines() if ":" in s}
        self.receipt.update({"rss_bytes": int(fields["VmRSS"].split()[0])*1024,
                             "vmlck_bytes": int(fields["VmLck"].split()[0])*1024})
        check(self.receipt["vmlck_bytes"] == self.gib*GIB and self.receipt["rss_bytes"] >= self.gib*GIB,
              "Pressure is not fully resident and locked")
        dump(self.directory / "pressure.json", self.receipt)

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def verify(self):
        if self.gib:
            check(self.alive(), "Pressure helper exited before measured work completed")
            status = Path(f"/proc/{self.receipt['pid']}/status").read_text()
            locked = next(int(s.split()[1])*1024 for s in status.splitlines() if s.startswith("VmLck:"))
            check(locked == self.gib*GIB, "Pressure backing was unlocked during the trial")

    def release(self):
        with self.lock:
            if self.process and self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()

    def stop(self):
        self.release()
        if self.process:
            check(self.process.wait(timeout=60) == 0, "Pressure helper did not release cleanly")
            self.process.stdout.close()


def common_prefix(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


def a_reusable_prefix(trace, saved_count):
    turn = trace["A_turns"][-1]
    saved_tokens = (turn["prompt"] + turn["expected_tokens"])[:saved_count]
    return common_prefix(saved_tokens, trace["A_continuation"])


def verify_reuse(completion, saved, prompt, cached_prefix=None):
    t = completion["final"]["timings"]
    cached_prefix = saved["n_saved"] if cached_prefix is None else cached_prefix
    check(0 < cached_prefix <= saved["n_saved"], "Invalid reusable prefix")
    expected = len(prompt) - cached_prefix
    check(expected > 0, "Continuation does not extend the saved prefix")
    check(expected <= t["prompt_n"] <= expected+1, "Unexpected re-prefill after restore")
    check(cached_prefix-1 <= t["cache_n"] <= cached_prefix, "Saved prefix was not reused")
    check(t["prompt_n"] + t["cache_n"] == len(prompt), "Prompt/cache token accounting disagrees")


def verify_swap(counters, allowance):
    check(all(v >= 0 for v in counters.values()), "VM counters went backwards")
    traffic = (counters["pswpin"] + counters["pswpout"]) * mmap.PAGESIZE
    check(counters["oom_kill"] == 0, "OOM occurred in trial")
    check(traffic <= allowance, "Observed swap exceeded the pinned allowance")
    return traffic


def run_trial(args, pins, trace, case, directory):
    mode, temperature, held_gib, normal = CASES[case]
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    router = Router(args, pins, directory, normal=normal, maximum=2 if mode == "resident" else 1)
    pressure = Pressure(args.ballast, held_gib, directory, pins["ballast_sha256"])
    result = {"case": case, "mode": mode, "cache_condition": temperature,
              "ballast_gib": held_gib, "normal_reference": normal,
              "trace_sha256": sha(args.trace), "image": pins["image"], "passed": False}
    before_trial = system_counters(args.device)
    monitor = Telemetry(directory, pressure.release if held_gib else None, pressure.alive if held_gib else None)
    try:
        pressure.start()
        result["pressure"] = pressure.receipt
        result["physical_memory_bytes"] = memory()["MemTotal"]
        monitor.__enter__()
        router.start()
        if temperature == "warm":
            for spec in pins["models"].values():
                condition(args.models / spec["filename"], "warm")
        result["A_effective_status"] = router.load("A")
        if mode == "resident":
            result["B_effective_status"] = router.load("B")
        result["initial_A"] = [router.complete("A", turn["prompt"]) for turn in trace["A_turns"]]
        result["initial_outputs_match_trace"] = [got["tokens"] == turn["expected_tokens"]
                                                 for got, turn in zip(result["initial_A"], trace["A_turns"])]
        check(all(result["initial_outputs_match_trace"]), "Initial conversation diverged from frozen history")
        check(not monitor.low_memory, "Pressure released on low-memory guard")
        pressure.verify()

        def switch(outgoing, incoming, prompt, predict):
            pressure.verify()
            measured = {}
            if mode != "resident":
                file = args.models / pins["models"][incoming]["filename"]
                if temperature == "cold":
                    condition(file, "cold")
                measured["weight_cache_before_switch"] = cache(file)
                check(measured["weight_cache_before_switch"]["resident_fraction"] >= .999
                      if temperature == "warm" else measured["weight_cache_before_switch"]["resident_pages"] == 0,
                      "Incoming weight-cache condition changed")
                if incoming == "A" and mode == "restore":
                    state = router.states / "A.bin"
                    if temperature == "cold":
                        condition(state, "cold")
                    measured["state_cache_before_switch"] = cache(state)
            counters = system_counters(args.device)
            start = time.monotonic_ns()
            measured["start_ns"] = start
            if mode == "restore":
                measured["save_outgoing"] = router.save(outgoing)
            before_load = time.monotonic_ns()
            if mode != "resident":
                measured["incoming_effective_status"] = router.load(incoming)
                check(router.status()[outgoing]["value"] == "unloaded", "Outgoing model still runs")
            loaded = time.monotonic_ns()
            measured["eviction_and_load_s"] = (loaded-before_load)/1e9
            if incoming == "A" and mode == "restore":
                measured["restore_incoming"] = router.restore("A", result["A_to_B"]["save_outgoing"])
            restored = time.monotonic_ns()
            measured["restore_api_s"] = (restored-loaded)/1e9
            completion = router.complete(incoming, prompt, predict=predict)
            measured["completion"] = completion
            measured["first_token_s"] = (completion["first_token_ns"]-start)/1e9
            measured["first_text_s"] = None if completion["first_text_ns"] is None else (completion["first_text_ns"]-start)/1e9
            measured["request_first_token_s"] = (completion["first_token_ns"]-completion["request_start_ns"])/1e9
            measured["io_through_first_token"] = {k: completion["io_at_first_token"][k]-v for k, v in counters.items()}
            measured["io_through_completion"] = {k: v-counters[k] for k, v in system_counters(args.device).items()}
            measured["memory_after"] = memory()
            check(not monitor.low_memory, "Pressure released on low-memory guard")
            check(not monitor.pressure_lost, "Pressure helper disappeared during measurement")
            pressure.verify()
            return measured

        result["A_to_B"] = switch("A", "B", trace["B_prompt"], 64)
        result["B_to_A"] = switch("B", "A", trace["A_continuation"], 128)
        a = result["B_to_A"]["completion"]
        b = result["A_to_B"]["completion"]
        result["A_continuation_matches_trace"] = a["tokens"] == trace["A_expected_continuation"]
        result["B_output_matches_trace"] = b["tokens"] == trace["B_expected_tokens"]
        check(result["B_output_matches_trace"], "B output differs from frozen reference")
        if mode == "recompute":
            timings = a["final"]["timings"]
            check(timings["cache_n"] == 0 and timings["prompt_n"] == len(trace["A_continuation"]),
                  "Recompute arm did not start with an empty prompt cache")
        if mode == "restore" and not normal:
            saved = result["A_to_B"]["save_outgoing"]
            result["expected_A_reused_tokens"] = a_reusable_prefix(trace, saved["n_saved"])
            check(saved["n_saved"] == trace["A_reuse"]["n_saved"] and
                  result["expected_A_reused_tokens"] == trace["A_reuse"]["common_prefix_tokens"],
                  "Saved/reusable prefix differs from the frozen trace")
            verify_reuse(a, saved, trace["A_continuation"], result["expected_A_reused_tokens"])
            check(result["A_continuation_matches_trace"], "Restored output differs from resident reference")
        if mode == "resident":
            check(result["A_continuation_matches_trace"], "Resident output differs from trace reference")
            prefix = a_reusable_prefix(trace, len(trace["A_continuation"]))
            verify_reuse(a, {"n_saved": prefix}, trace["A_continuation"])
        result["memory"] = monitor.summary(result["A_to_B"]["start_ns"], a["end_ns"])
        result["peak_spill_logical_bytes"] = sum(p.stat().st_size for p in router.states.iterdir())
        result["peak_spill_allocated_bytes"] = sum(p.stat().st_blocks*512 for p in router.states.iterdir())
        result["trial_vm_deltas"] = {k: system_counters(args.device)[k]-before_trial[k] for k in ("pswpin", "pswpout", "oom_kill")}
        result["page_size_bytes"] = mmap.PAGESIZE
        result["swap_allowance_bytes"] = pins["swap_allowance_bytes"]
        result["trial_swap_traffic_bytes"] = verify_swap(result["trial_vm_deltas"], pins["swap_allowance_bytes"])
        pressure.verify()
        router.stop()
        log = (directory / "server.log").read_text()
        check("offloaded 31/31 layers to GPU" in log and "offloaded 42/42 layers to GPU" in log,
              "Full GPU offload missing from model logs")
        if mode != "resident":
            check("models_max limit reached, removing LRU" in log or "evicting idle LRU" in log,
                  "Native router eviction was not observed")
        result["passed"] = True
    finally:
        # Give the system its headroom back before any slow Docker cleanup.
        try:
            try:
                pressure.stop()
            finally:
                try:
                    router.stop()
                finally:
                    if monitor.thread.ident is not None:
                        monitor.__exit__()
        except BaseException as error:
            result["passed"] = False
            result["cleanup_error"] = str(error)
            raise
        finally:
            dump(directory / "result.json", result)
    return result


def run(args, pins):
    check(args.trace is not None and args.ballast is not None, "run requires --trace and --ballast")
    trace = json.loads(args.trace.read_text())
    check(sha(args.trace) == pins["trace_sha256"], "Frozen trace hash mismatch")
    cases = args.cases.split(",")
    check(all(c in CASES for c in cases), "Unknown case")
    check(1 <= args.repeats <= 10, "Repeat count outside experiment bound")
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    dump(output / "run.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                              "host": platform.node(), "kernel": platform.release(),
                              "machine": platform.machine(), "device": args.device,
                              "python": platform.python_version(), "cases": cases,
                              "gpu_driver": subprocess.check_output(
                                  ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
                              "model_filesystem": subprocess.check_output(
                                  ["findmnt", "-no", "SOURCE,FSTYPE", "--target", str(args.models)], text=True).strip(),
                              "repeats": args.repeats, "pins": pins,
                              "harness_sha256": sha(Path(__file__)),
                              "physical_memory_bytes": memory()["MemTotal"]})
    # Interleave cases per repetition; retain actual order as provenance.
    order = []
    for repeat in range(args.repeats):
        for case in cases:
            name = f"{repeat+1:02d}-{case}"
            print("START", name, flush=True)
            result = run_trial(args, pins, trace, case, output / name)
            order.append(name)
            dump(output / "order.json", order)
            print(json.dumps({"trial": name, "A_to_B_s": result["A_to_B"]["first_token_s"],
                              "B_to_A_s": result["B_to_A"]["first_token_s"], "passed": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--device", required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--ballast", type=Path)
    parser.add_argument("--cases", default="resident,warm_restore,warm_recompute,cold_restore,cold_recompute,pressure_restore,pressure_recompute,normal_pressure_restore,normal_pressure_recompute")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    pins = json.loads((HERE / "artifacts.json").read_text())
    for spec in pins["models"].values():
        path = args.models / spec["filename"]
        check(path.stat().st_size == spec["size_bytes"] and sha(path) == spec["sha256"], "GGUF integrity failure")
    check(args.models.is_dir() and 1024 <= args.port <= 65535, "Invalid model directory or port")
    if args.action == "prepare":
        prepare(args, pins)
    else:
        run(args, pins)


if __name__ == "__main__":
    os.umask(0o077)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    main()
