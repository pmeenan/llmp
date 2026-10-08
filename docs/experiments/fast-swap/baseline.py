# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""M3 baselines: load, prefill, decode, peak memory and swap of reference engines.

Runs on a Spark, never on the workstation. Python only orchestrates the
reference engines over their HTTP APIs; raw output goes to a new external
directory and the checked-in report carries the aggregates. Coarse by design
(D-085): a few repeats, medians, no calibration protocol.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
PROMPTS = json.loads((HERE / "prompts.json").read_text())
DOCKER = shlex.split(os.environ.get("DOCKER", "sudo -n docker"))
# The pinned b10964 image; LLAMA_IMAGE overrides it (pins.json's local
# b11254 build, llmp-llamacpp:b11254-cuda13, for the Qwen3.8 GGUF study).
LLAMA_IMAGE = os.environ.get("LLAMA_IMAGE", "ghcr.io/ggml-org/llama.cpp@sha256:"
                             "837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
GIB = 1 << 30


def filler(count, tag):
    """The fixed synthetic log used for prefill and the swap's saved context."""
    statuses = ("ok", "retry", "timeout", "ok", "ok")
    lines = [f"Run {tag}. The following service log lists one record per line."]
    for i in range(count):
        lines.append(f"Record {i:05d}: service svc-{i * 7 % 23:02d} on shard {i * 5 % 11} "
                     f"answered in {(i * 37) % 97 + 3} ms with status {statuses[i % 5]}.")
    return "\n".join(lines) + "\n"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class Memory(threading.Thread):
    """Samples MemAvailable every 200 ms; peak use is a drop from a baseline."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.running = True

    @staticmethod
    def available():
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("MemAvailable missing")

    def run(self):
        while self.running:
            self.samples.append((time.monotonic(), self.available()))
            time.sleep(0.2)

    def minimum(self, start, end=None):
        end = end or time.monotonic()
        values = [v for t, v in self.samples if start <= t <= end]
        return min(values) if values else None


def evict(paths):
    """Drop the files' pages from the page cache (cold-load condition)."""
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def model_files(paths):
    out = []
    for path in map(Path, paths):
        out += sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    return out


class Client:
    def __init__(self, base, model=None, key=None):
        self.base, self.model, self.key = base.rstrip("/"), model, key

    def _request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        return urllib.request.Request(self.base + path, data=body, headers=headers)

    def get(self, path, timeout=10):
        with OPENER.open(self._request(path), timeout=timeout) as response:
            raw = response.read()
        try:
            return json.loads(raw)
        except ValueError:
            return raw.decode(errors="replace")

    def post(self, path, payload, timeout=3600):
        with OPENER.open(self._request(path, payload), timeout=timeout) as response:
            return json.load(response)

    def ready(self, path):
        try:
            with OPENER.open(self._request(path), timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return False

    def stream(self, path, payload, timeout=3600):
        """Streams a completion; times the first and last nonempty piece."""
        payload = dict(payload, stream=True, stream_options={"include_usage": True})
        if self.model:
            payload.setdefault("model", self.model)
        result = {"t_send": time.monotonic(), "t_first": None, "t_last": None,
                  "pieces": 0, "text": "", "usage": None, "timings": None,
                  "engine_stats": None}
        with OPENER.open(self._request(path, payload), timeout=timeout) as response:
            for raw in response:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                now = time.monotonic()
                result["usage"] = chunk.get("usage") or result["usage"]
                result["timings"] = chunk.get("timings") or result["timings"]
                result["engine_stats"] = chunk.get("tensorfold") or result["engine_stats"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or delta.get("reasoning_content")
                             or delta.get("reasoning") or choice.get("text") or "")
                    if piece:
                        result["pieces"] += 1
                        result["text"] += piece
                        result["t_first"] = result["t_first"] or now
                        result["t_last"] = now
        result["t_end"] = time.monotonic()
        return result

    def chat(self, messages, max_tokens, extra=None):
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
        payload.update(extra or {})
        return self.stream("/v1/chat/completions", payload)


def spec_metrics(client):
    """vLLM's speculative-decoding counters, if the engine exports them."""
    try:
        text = client.get("/metrics")
    except Exception:  # noqa: BLE001 - optional endpoint
        return {}
    values = {}
    if isinstance(text, str):
        for line in text.splitlines():
            if "spec_decode" in line and not line.startswith("#"):
                name, _, value = line.rpartition(" ")
                try:
                    values[name] = float(value)
                except ValueError:
                    pass
    return values


def count_tokens(client, tokenizer, text):
    if tokenizer == "llama":
        return len(client.post("/tokenize", {"content": text})["tokens"])
    if tokenizer == "vllm":
        return client.post("/tokenize", {"model": client.model, "prompt": text})["count"]
    raise ValueError(tokenizer)


def fit_filler(client, tokenizer, target):
    """Smallest record count whose filler reaches the target token count."""
    if tokenizer == "usage":
        sizes = {}
        for count in (50, 200):
            result = client.chat([{"role": "user", "content": filler(count, "fit")}], 1)
            sizes[count] = result["usage"]["prompt_tokens"]
        per = (sizes[200] - sizes[50]) / 150
        return max(1, round(200 + (target - sizes[200]) / per))
    low, high = 1, 4096
    while low < high:
        mid = (low + high) // 2
        if count_tokens(client, tokenizer, filler(mid, "fit-00000000")) >= target:
            high = mid
        else:
            low = mid + 1
    return low


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def measure(client, args, output):
    """Prefill and decode on a warm server, per decode variant."""
    extra = json.loads(args.extra)
    client.chat(PROMPTS["first_token"]["messages"], 8, extra)  # warm-up
    report = {"prefill": {}, "decode": {}}
    for target in PROMPTS["prefill_tokens"]:
        count = fit_filler(client, args.tokenizer, target)
        runs = []
        for repeat in range(args.repeats):
            tag = f"{uuid.uuid4().hex[:8]}"
            content = filler(count, tag) + "\n" + PROMPTS["prefill_question"]
            result = client.chat([{"role": "user", "content": content}], 1, extra)
            seconds = result["t_end"] - result["t_send"]
            tokens = (result["usage"] or {}).get("prompt_tokens")
            timings = result["timings"] or {}
            runs.append({"prompt_tokens": tokens, "seconds": seconds,
                         "client_tok_s": tokens / seconds if tokens else None,
                         "engine_prompt_n": timings.get("prompt_n"),
                         "engine_tok_s": timings.get("prompt_per_second")})
        report["prefill"][str(target)] = {
            "records": count, "runs": runs,
            "median_client_tok_s": median(r["client_tok_s"] for r in runs),
            "median_engine_tok_s": median(r["engine_tok_s"] for r in runs)}
    for variant in args.variant or ["default={}"]:
        name, _, variant_json = variant.partition("=")
        variant_extra = dict(extra, **json.loads(args.decode_extra), **json.loads(variant_json))
        report["decode"][name] = {}
        for prompt in PROMPTS["decode"]:
            before = spec_metrics(client)
            runs = []
            for repeat in range(args.repeats):
                result = client.chat(prompt["messages"], PROMPTS["decode_tokens"], variant_extra)
                tokens = (result["usage"] or {}).get("completion_tokens")
                span = (result["t_last"] or 0) - (result["t_first"] or 0)
                timings = result["timings"] or {}
                runs.append({"completion_tokens": tokens, "pieces": result["pieces"],
                             "first_piece_s": result["t_first"] - result["t_send"],
                             "client_tok_s": (tokens - 1) / span if tokens and span > 0 else None,
                             "engine_tok_s": timings.get("predicted_per_second"),
                             "draft_n": timings.get("draft_n"),
                             "draft_n_accepted": timings.get("draft_n_accepted"),
                             "engine_stats": result["engine_stats"],
                             "text_head": result["text"][:200]})
            after = spec_metrics(client)
            report["decode"][name][prompt["id"]] = {
                "runs": runs,
                "median_client_tok_s": median(r["client_tok_s"] for r in runs),
                "median_engine_tok_s": median(r["engine_tok_s"] for r in runs),
                "spec_metrics_delta": {k: after[k] - before.get(k, 0) for k in after}}
    dump(output / "measure.json", report)
    return report


def reference(client, kind, output, repeats=2):
    """Greedy tokens and top-k logprobs for the fixed chat prompts."""
    count, top = PROMPTS["reference_tokens"], PROMPTS["reference_top_logprobs"]
    rows = []
    for prompt in PROMPTS["chat"]:
        if kind == "vllm":
            ids = client.post("/tokenize", {"model": client.model, "messages": prompt["messages"],
                                            "add_generation_prompt": True})["tokens"]
        else:
            text = client.post("/apply-template", {"messages": prompt["messages"]})["prompt"]
            ids = client.post("/tokenize", {"content": text, "add_special": False,
                                            "parse_special": True})["tokens"]
        attempts = []
        for _ in range(repeats):
            if kind == "vllm":
                body = client.post("/v1/completions", {
                    "model": client.model, "prompt": ids, "max_tokens": count, "temperature": 0,
                    "logprobs": top, "return_tokens_as_token_ids": True})
                logprobs = body["choices"][0]["logprobs"]
                tokens = [int(t.split(":", 1)[1]) for t in logprobs["tokens"]]
                steps = [{"id": tid, "logprob": lp,
                          "top": sorted(([int(k.split(":", 1)[1]), v] for k, v in (alt or {}).items()),
                                        key=lambda kv: -kv[1])}
                         for tid, lp, alt in zip(tokens, logprobs["token_logprobs"],
                                                 logprobs["top_logprobs"])]
            else:
                body = client.post("/completion", {
                    "prompt": ids, "n_predict": count, "temperature": 0, "n_probs": top,
                    "cache_prompt": False, "post_sampling_probs": False, "ignore_eos": False})
                steps = [{"id": p["id"], "logprob": p["logprob"],
                          "top": [[alt["id"], alt["logprob"]] for alt in p["top_logprobs"]]}
                         for p in body["completion_probabilities"]]
            attempts.append(steps)
        same = all([s["id"] for s in a] == [s["id"] for s in attempts[0]] for a in attempts)
        rows.append({"id": prompt["id"], "messages": prompt["messages"], "prompt_token_ids": ids,
                     "greedy_token_ids": [s["id"] for s in attempts[0]], "steps": attempts[0],
                     "repeats": repeats, "repeats_identical": same,
                     "other_repeats_token_ids": [[s["id"] for s in a] for a in attempts[1:]]})
    dump(output / "reference.json", rows)
    return rows


def wait_ready(client, health, deadline_s, process=None):
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        if client.ready(health):
            return time.monotonic()
        if process is not None and process.poll() not in (None, 0):
            raise RuntimeError(f"start command exited with {process.returncode}")
        time.sleep(0.5)
    raise TimeoutError("server not ready")


def first_token(client, extra):
    first = PROMPTS["first_token"]
    result = client.chat(first["messages"], first["max_tokens"], extra)
    return result["t_first"] or result["t_end"]


def llama_command(name, port, model, args, states=None):
    """The pinned llama.cpp server; ROOT is mounted read-only at /models and
    paths under it (the model, and a drafter in ARGS) are given as /models/..."""
    root = Path(os.environ.get("MODEL_ROOT", Path.home() / ".local/share/llmp"))
    model = Path(model).resolve().relative_to(root)
    argv = DOCKER + ["run", "-d", "--name", name, "--device", "nvidia.com/gpu=all",
                     "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                     "--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,size=256m",
                     "--publish", f"127.0.0.1:{port}:8080",
                     "--mount", f"type=bind,src={root},dst=/models,readonly",
                     "--env", "CUDA_DISABLE_PTX_JIT=1"]
    if states:
        argv += ["--mount", f"type=bind,src={states},dst=/states"]
    argv += ["--entrypoint", "/app/llama-server", LLAMA_IMAGE, "-m", f"/models/{model}",
             "--host", "0.0.0.0", "--port", "8080", "--metrics"] + shlex.split(args)
    if states:
        argv += ["--slot-save-path", "/states"]
    return argv


def stop_container(name, output):
    with open(output / f"{name}.log", "w") as log:
        subprocess.run(DOCKER + ["logs", name], stdout=log, stderr=subprocess.STDOUT, check=False)
    subprocess.run(DOCKER + ["stop", "-t", "30", name], capture_output=True, check=False)
    subprocess.run(DOCKER + ["rm", "-f", name], capture_output=True, check=False)


def session(args):
    """Start an engine cold, time it to ready and first token, measure, stop."""
    output = Path(args.output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    extra = json.loads(args.extra)
    memory = Memory()
    memory.start()
    time.sleep(1)
    if args.evict:
        evict(model_files(args.evict))
    time.sleep(1)
    baseline = memory.available()
    client = Client(f"http://127.0.0.1:{args.port}", args.model)
    name = None
    process = None
    t0 = time.monotonic()
    if args.llama_model:
        name = "llmp-m3base-" + uuid.uuid4().hex[:8]
        subprocess.run(llama_command(name, args.port, args.llama_model, args.llama_args),
                       check=True, capture_output=True)
    else:
        log = open(output / "start.log", "w")
        process = subprocess.Popen(args.start, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    report = {"arguments": {k: v for k, v in vars(args).items() if k != "func"},
              "baseline_available_bytes": baseline}
    try:
        t_ready = wait_ready(client, args.health, args.ready_timeout, process)
        if client.model is None:
            client.model = client.get("/v1/models")["data"][0]["id"]
        t_first = first_token(client, extra)
        report["load"] = {"ready_s": t_ready - t0, "first_token_s": t_first - t0,
                          "min_available_to_ready_bytes": memory.minimum(t0, t_ready)}
        print(json.dumps(report["load"]), flush=True)
        dump(output / "session.json", report)
        if not args.no_measure:
            report["measure"] = measure(client, args, output)
        if args.reference:
            report["reference_file"] = str(output / "reference.json")
            reference(client, args.reference, output)
    finally:
        report["min_available_bytes"] = memory.minimum(t0)
        report["peak_drop_gib"] = (baseline - report["min_available_bytes"]) / GIB
        if name:
            stop_container(name, output)
        elif args.stop:
            subprocess.run(args.stop, shell=True, check=False,
                           stdout=open(output / "stop.log", "w"), stderr=subprocess.STDOUT)
        if process is not None and process.poll() is None:
            process.terminate()
        memory.running = False
        dump(output / "session.json", report)
    print(json.dumps({"load": report.get("load"), "peak_drop_gib": report["peak_drop_gib"]}))


def swap(args):
    """One or more llama.cpp A->B->A cycles, A holding a saved 8K conversation."""
    output = Path(args.output).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    states = output / "states"
    states.mkdir(mode=0o700)
    memory = Memory()
    memory.start()
    client = Client(f"http://127.0.0.1:{args.port}")
    spec = PROMPTS["swap"]
    files = {"A": model_files([Path(args.a).parent]), "B": model_files([Path(args.b).parent])}
    llama_args = {"A": args.a_args, "B": args.b_args}
    models = {"A": args.a, "B": args.b}

    def start(key):
        name = f"llmp-m3swap-{key}-" + uuid.uuid4().hex[:8]
        t = time.monotonic()
        subprocess.run(llama_command(name, args.port, models[key], llama_args[key], states),
                       check=True, capture_output=True)
        ready = wait_ready(client, "/health", 900)
        return name, ready - t

    def save(key, name):
        t = time.monotonic()
        body = client.post("/slots/0?action=save", {"filename": f"{key}.bin"})
        return time.monotonic() - t, body

    def stop(name):
        t = time.monotonic()
        stop_container(name, output)
        return time.monotonic() - t

    cycles = []
    for cycle in range(args.cycles):
        evict(files["A"])
        name_a, _ = start("A")
        count = fit_filler(client, "llama", spec["context_tokens"])
        text = client.post("/apply-template", {"messages": [
            {"role": "user", "content": filler(count, f"swap{cycle}") + "\n" + spec["question"]}]})["prompt"]
        prompt = client.post("/tokenize", {"content": text, "parse_special": True})["tokens"]
        first = client.post("/completion", {"prompt": prompt, "n_predict": spec["reply_tokens"],
                                            "temperature": 0, "cache_prompt": True,
                                            "return_tokens": True, "id_slot": 0})
        tail = client.post("/tokenize", {"content": "\n\n" + spec["continuation"]})["tokens"]
        continuation = prompt + first["tokens"] + tail
        evict(files["B"])
        row = {"cycle": cycle, "a_context_tokens": len(prompt), "baseline_available_bytes": memory.available()}
        t0 = time.monotonic()
        row["a_save_s"], row["a_save"] = save("A", name_a)
        row["a_stop_s"] = stop(name_a)
        name_b, row["b_start_to_ready_s"] = start("B")
        t_first = first_token(client, {"cache_prompt": True, "id_slot": 0})
        row["a_to_b_first_token_s"] = t_first - t0
        row["a_to_b_min_available_bytes"] = memory.minimum(t0)
        evict(files["A"])
        t1 = time.monotonic()
        row["b_save_s"], row["b_save"] = save("B", name_b)
        row["b_stop_s"] = stop(name_b)
        name_a, row["a_start_to_ready_s"] = start("A")
        t = time.monotonic()
        row["a_restore"] = client.post("/slots/0?action=restore", {"filename": "A.bin"})
        row["a_restore_s"] = time.monotonic() - t
        result = client.stream("/completion", {"prompt": continuation, "n_predict": spec["reply_tokens"],
                                               "temperature": 0, "cache_prompt": True, "id_slot": 0})
        row["b_to_a_first_token_s"] = (result["t_first"] or result["t_end"]) - t1
        row["b_to_a_timings"] = result["timings"]
        row["b_to_a_min_available_bytes"] = memory.minimum(t1)
        stop(name_a)
        cycles.append(row)
        print(json.dumps({k: v for k, v in row.items() if k.endswith("_s")}), flush=True)
        dump(output / "swap.json", {"arguments": {k: v for k, v in vars(args).items() if k != "func"},
                                    "cycles": cycles})
    memory.running = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("session")
    s.add_argument("output")
    s.add_argument("--port", type=int, default=18080)
    s.add_argument("--model", help="served model name; read from /v1/models if omitted")
    s.add_argument("--llama-model", help="GGUF path; runs the pinned llama.cpp image")
    s.add_argument("--llama-args", default="")
    s.add_argument("--start", help="shell command that starts the engine")
    s.add_argument("--stop", help="shell command that stops the engine")
    s.add_argument("--health", default="/health")
    s.add_argument("--ready-timeout", type=float, default=1800)
    s.add_argument("--evict", nargs="*", help="files or directories to drop from the page cache first")
    s.add_argument("--tokenizer", choices=["llama", "vllm", "usage"], default="llama")
    s.add_argument("--extra", default="{}", help="JSON fields added to every request")
    s.add_argument("--decode-extra", default='{"ignore_eos": true}')
    s.add_argument("--variant", action="append", help="name=JSON decode variant (e.g. spec_off)")
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--no-measure", action="store_true")
    s.add_argument("--reference", choices=["llama", "vllm"])
    s.set_defaults(func=session)
    w = sub.add_parser("swap")
    w.add_argument("output")
    w.add_argument("--a", required=True)
    w.add_argument("--b", required=True)
    w.add_argument("--a-args", default="")
    w.add_argument("--b-args", default="")
    w.add_argument("--port", type=int, default=18081)
    w.add_argument("--cycles", type=int, default=1)
    w.set_defaults(func=swap)
    e = sub.add_parser("evict", help="drop files or directories from the page cache")
    e.add_argument("paths", nargs="+")
    e.set_defaults(func=lambda a: evict(model_files(a.paths)))
    args = parser.parse_args()
    os.umask(0o077)
    args.func(args)


if __name__ == "__main__":
    main()
