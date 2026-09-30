# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Long-context runs: the comparators and jitLLM's runtime at depth (README.md).

Runs on a Spark, never on the workstation; Python only drives engines over
HTTP (and llama.cpp's tools in their container). Raw output goes to a new
external directory; the checked-in report carries the aggregates.

    longctx.py llama OUT --image IMG --model GGUF --args "..." PROMPT.json...
        llama.cpp's server: each prompt's chat messages rendered and
        tokenized by the server (/apply-template, /tokenize), then one
        streamed /completion from those IDs (greedy, cache_prompt off,
        --top N log-probabilities). With --session, a session prompt's turns
        through the chat route with the prompt cache on, each answer sent
        back without its reasoning (the turn-reuse comparator).
    longctx.py llama-ppl OUT --image IMG --model GGUF --text PPL.txt --ctx L...
        llama-perplexity over one window of each L (its rule: the second
        half scored), and the window's IDs (llama-tokenize) for jitLLM.
    longctx.py vllm OUT --start CMD --stop CMD [--ppl PPL.txt --ctx L...] PROMPT.json...
        A vLLM server (Mia's recipe): the prompt IDs from /tokenize with the
        template, one streamed /v1/completions each (greedy, --top N
        log-probabilities, a unique cache_salt so nothing is reused), then
        prompt log-probabilities over each perplexity window.
    longctx.py jitllm OUT --runtime BIN --config TOML --model NAME PROMPT.json...
        jitllm-runtime serving the chat route: one streamed greedy
        /v1/chat/completions per prompt; --session runs a session prompt's
        follow-up turns, the client sending back its answers (with or
        without the reasoning, --reasoning keep|drop).

Every run samples MemAvailable every 200 ms (peak = the drop from before the
engine started) and writes OUT/run.json and one OUT/<prompt>.json each.
Prefill = first streamed piece − request sent (one decode step included);
decode = (completion tokens − 1) ÷ (last piece − first piece).
New records retain the requested output budget and terminal stream status;
early natural stops are labeled separately from full-budget timings. Retrieval
keeps the historical reasoning-plus-visible metric and also checks the visible
answer, which passes only after a completed stop response.

Protocol fields (also retained in run.json's compact prompt summaries):
    requested_tokens / completion_tokens: the sent budget and actual count;
    full_budget: actual count equals budget, or null if a count is unavailable;
    early_stop: a completed natural stop before that budget;
    finish_reason / seen_done / stream_complete: terminal status; /v1/ streams
        need both a finish reason and [DONE], native llama /completion needs
        its explicit stop envelope instead. Incomplete streams record errors
        and do not produce throughput values;
    needles_found: historical reasoning-plus-visible presence, unchanged;
    visible_needles_found / visible_retrieval_pass: visible presence only,
        passing only if every fact is visible and finish_reason is stop in
        a complete stream. A length finish is still a timing, not an answer.
        This searches the content/text channel; raw completion APIs need not
        separate reasoning, and model-specific hidden markup is not inferred.
The runtime's cold page-in warm-up is saved and must complete successfully
before timed prompts start. Synthetic checks: test_longctx.py on a Spark.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fast-swap"))
from baseline import Client, Memory, dump, wait_ready  # noqa: E402

DOCKER = shlex.split(os.environ.get("DOCKER", "sudo -n docker"))
GIB = 1 << 30
ROOT = Path(os.environ.get("MODEL_ROOT", Path.home() / ".local/share/jitllm"))


def gate(min_gib):
    """Refuse to start an engine beside another model process."""
    free = Memory.available() / GIB
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    running = subprocess.run(DOCKER + ["ps", "-q"], capture_output=True, text=True).stdout.strip()
    if free < min_gib or apps or running:
        raise SystemExit(f"busy: {free:.1f} GiB available, GPU processes [{apps}], "
                         f"containers [{running}]")
    return free


def prompt_files(paths):
    return [json.loads(Path(p).read_text()) for p in paths]


def stream_chunks(response, record):
    """Retain partial results if the connection or SSE JSON ends badly."""
    try:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                record["seen_done"] = True
                return
            chunk = json.loads(data)
            if not isinstance(chunk, dict):
                raise ValueError("SSE data is not a JSON object")
            if "error" in chunk:
                record["error"] = chunk["error"]
                return
            yield chunk
    except (ValueError, OSError, http.client.HTTPException) as error:
        record["error"] = {"code": "stream_read_failed", "message": str(error)}


def stream(client, path, payload, memory):
    """A streamed request: timings, text, per-token IDs and log-probabilities."""
    payload = dict(payload, stream=True)
    if path.startswith("/v1/"):
        payload["stream_options"] = {"include_usage": True}
    out = {"t_send": time.monotonic(), "t_first": None, "t_last": None, "pieces": 0,
           "text": "", "reasoning": "", "ids": [], "steps": [], "usage": None, "timings": None,
           "requested_tokens": payload.get("max_tokens", payload.get("n_predict")),
           "finish_reason": None, "seen_done": False, "stream_complete": False,
           "llama_stop": False}
    try:
        response = __import__("baseline").OPENER.open(client._request(path, payload),
                                                      timeout=4 * 3600)
    except urllib.error.HTTPError as e:
        with e:
            out["error"] = {"status": e.code, "body": e.read().decode(errors="replace")[:2000]}
        out["t_end"] = time.monotonic()
        out["min_available_bytes"] = memory.minimum(out["t_send"], out["t_end"])
        return out
    except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
        out["error"] = {"code": "stream_open_failed", "message": str(error)}
        out["t_end"] = time.monotonic()
        out["min_available_bytes"] = memory.minimum(out["t_send"], out["t_end"])
        return out
    with response:
        for chunk in stream_chunks(response, out):
            now = time.monotonic()
            out["usage"] = chunk.get("usage") or out["usage"]
            out["timings"] = chunk.get("timings") or out["timings"]
            piece = ""
            if "choices" not in chunk and chunk.get("stop"):  # native llama terminal envelope
                out["llama_stop"] = True
                out["llama_stop_details"] = {
                    k: chunk[k] for k in ("stop_type", "stopped_eos", "stopped_limit",
                                         "stopped_word", "stopping_word") if k in chunk}
                stop_type = chunk.get("stop_type")
                if stop_type == "limit" or chunk.get("stopped_limit"):
                    out["finish_reason"] = "length"
                elif stop_type in ("eos", "word") or chunk.get("stopped_eos") or (
                        chunk.get("stopped_word")):
                    out["finish_reason"] = "stop"
            if "content" in chunk and "choices" not in chunk:  # llama.cpp /completion
                piece = chunk.get("content") or ""
                out["text"] += piece
                for p in chunk.get("completion_probabilities") or []:
                    out["ids"].append(p["id"])
                    out["steps"].append({"id": p["id"], "logprob": p["logprob"],
                                         "top": [[a["id"], a["logprob"]]
                                                 for a in p.get("top_logprobs") or []]})
                if chunk.get("tokens"):
                    out.setdefault("token_chunks", []).append(chunk["tokens"])
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason") is not None:
                    out["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta") or {}
                content = delta.get("content") or choice.get("text") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                out["text"] += content
                out["reasoning"] += reasoning
                piece = content + reasoning
                lp = choice.get("logprobs")
                if lp and lp.get("tokens"):  # vLLM /v1/completions
                    for tok, value, alt in zip(lp["tokens"], lp["token_logprobs"],
                                               lp["top_logprobs"] or [None] * len(lp["tokens"])):
                        tid = int(tok.split(":", 1)[1])
                        out["ids"].append(tid)
                        out["steps"].append({"id": tid, "logprob": value, "top": sorted(
                            ([int(k.split(":", 1)[1]), v] for k, v in (alt or {}).items()),
                            key=lambda kv: -kv[1])})
            if piece or chunk.get("completion_probabilities"):
                out["pieces"] += 1
                out["t_first"] = out["t_first"] or now
                out["t_last"] = now
    terminal = out["seen_done"] if path.startswith("/v1/") else out["llama_stop"]
    out["stream_complete"] = bool(terminal and isinstance(out["finish_reason"], str)
                                  and out["finish_reason"] and not out.get("error"))
    if not out["stream_complete"] and not out.get("error"):
        out["error"] = {"code": "incomplete_stream", "message":
                        "Missing terminal stream marker or finish reason",
                        "seen_done": out["seen_done"], "llama_stop": out["llama_stop"],
                        "finish_reason": out["finish_reason"]}
    out["t_end"] = time.monotonic()
    out["min_available_bytes"] = memory.minimum(out["t_send"], out["t_end"])
    return out


def summarize(record, prompt_tokens, completion_tokens):
    first, last, send = record["t_first"], record["t_last"], record["t_send"]
    record["prompt_tokens"] = prompt_tokens
    record["completion_tokens"] = completion_tokens
    complete = bool(record.get("stream_complete") and not record.get("error"))
    requested = record.get("requested_tokens")
    known_counts = requested is not None and completion_tokens is not None
    record["full_budget"] = completion_tokens == requested if known_counts else None
    record["early_stop"] = (complete and record.get("finish_reason") == "stop"
                            and completion_tokens < requested) if known_counts else None
    record["completed_stop"] = complete and record.get("finish_reason") == "stop"
    if not complete:
        record["completion_status"] = "error" if record.get("error") else "incomplete"
    elif record["early_stop"]:
        record["completion_status"] = "early_stop"
    elif record["full_budget"]:
        record["completion_status"] = "full_budget"
    elif record.get("finish_reason") == "length" and known_counts:
        record["completion_status"] = "short_length"
    else:
        record["completion_status"] = "complete"
    record["prefill_s"] = first - send if first else None
    record["prefill_tok_s"] = (prompt_tokens / (first - send)
                               if complete and first and prompt_tokens else None)
    span = (last or 0) - (first or 0)
    record["decode_tok_s"] = ((completion_tokens - 1) / span
                              if complete and completion_tokens and completion_tokens > 1
                              and span > 0 else None)
    return record


def needles(prompt, text):
    return {n["id"]: n["passphrase"] in text for n in prompt.get("needles", [])}


def failed(out, run, name, error):
    """One case's failure, recorded (with an HTTP error's body) so the rest run."""
    body = None
    if isinstance(error, urllib.error.HTTPError):
        body = error.read().decode(errors="replace")[:2000]
    run.setdefault("failures", {})[name] = {"error": repr(error), "body": body}
    dump(out / "run.json", run)
    print(name, "FAILED", repr(error), body or "", flush=True)


def report(out, name, prompt, record):
    # Preserve the historical metric; reasoning alone is not a visible answer.
    record["needles_found"] = needles(prompt, record.get("reasoning", "") + record["text"])
    record["visible_needles_found"] = needles(prompt, record["text"])
    record["visible_retrieval_pass"] = (
        bool(record.get("completed_stop")) and all(record["visible_needles_found"].values())
        if record["visible_needles_found"] else None)
    dump(out / f"{name}.json", record)
    short = {k: record.get(k) for k in ("prompt_tokens", "completion_tokens", "prefill_s",
                                       "requested_tokens", "full_budget", "early_stop",
                                       "completion_status", "finish_reason", "seen_done",
                                       "stream_complete", "completed_stop", "prefill_tok_s",
                                       "decode_tok_s", "needles_found", "visible_needles_found",
                                       "visible_retrieval_pass", "error")}
    print(name, json.dumps(short), flush=True)
    return short


# ------------------------------------------------------------------ llama.cpp

def llama_argv(name, port, image, binary, model, args, extra_mounts=()):
    model = Path(model).resolve().relative_to(ROOT)
    argv = DOCKER + ["run", "-d", "--name", name, "--device", "nvidia.com/gpu=all",
                     "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                     "--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,size=256m",
                     "--network", "bridge", "--publish", f"127.0.0.1:{port}:8080",
                     "--mount", f"type=bind,src={ROOT},dst=/models,readonly",
                     "--env", "CUDA_DISABLE_PTX_JIT=1"]
    for src, dst in extra_mounts:
        argv += ["--mount", f"type=bind,src={src},dst={dst}"]
    argv += ["--entrypoint", binary, image, "-m", f"/models/{model}"]
    return argv + shlex.split(args)


def stop_container(name, out):
    with open(out / f"{name}.log", "w") as log:
        subprocess.run(DOCKER + ["logs", name], stdout=log, stderr=subprocess.STDOUT, check=False)
    subprocess.run(DOCKER + ["stop", "-t", "30", name], capture_output=True, check=False)
    subprocess.run(DOCKER + ["rm", "-f", name], capture_output=True, check=False)


def cmd_llama(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    gate(args.min_gib)
    memory = Memory()
    memory.start()
    time.sleep(1)
    baseline = Memory.available()
    name = "jitllm-lc-" + uuid.uuid4().hex[:8]
    argv = llama_argv(name, args.port, args.image, "/app/llama-server", args.model,
                      f"--host 0.0.0.0 --port 8080 --metrics {args.args}")
    run = {"argv": argv, "baseline_available_bytes": baseline, "prompts": {}}
    t0 = time.monotonic()
    subprocess.run(argv, check=True, capture_output=True)
    client = Client(f"http://127.0.0.1:{args.port}")
    try:
        run["ready_s"] = wait_ready(client, "/health", args.ready_timeout) - t0
        if args.session:
            # The turn-reuse comparator: the chat route with the prompt cache
            # on, the client sending back each answer without its reasoning;
            # llama.cpp reports how many prompt tokens it reused (cache_n).
            for prompt in prompt_files(args.prompts):
                messages = list(prompt["messages"])
                for turn, followup in enumerate([None] + prompt["followups"]):
                    case = f"{prompt['id']}-turn{turn + 1}-drop"
                    try:
                        if followup is not None:
                            messages.append({"role": "user", "content": followup})
                        record = stream(client, "/v1/chat/completions",
                                        {"messages": messages, "temperature": 0,
                                         "max_tokens": prompt["generate_tokens"],
                                         "cache_prompt": True}, memory)
                        t = record.get("timings") or {}
                        usage = record.get("usage") or {}
                        summarize(record, usage.get("prompt_tokens"),
                                  usage.get("completion_tokens"))
                        record["cached_tokens"] = t.get("cache_n")
                        run["prompts"][case] = report(out, case, prompt, record)
                        run["prompts"][case]["cached_tokens"] = record["cached_tokens"]
                        run["prompts"][case]["prompt_n"] = t.get("prompt_n")
                        messages.append({"role": "assistant", "content": record["text"]})
                        dump(out / "run.json", run)
                    except Exception as error:  # noqa: BLE001 - recorded; the rest run
                        failed(out, run, case, error)
                        break
            return
        for prompt in prompt_files(args.prompts):
            try:
                text = client.post("/apply-template", {"messages": prompt["messages"]})["prompt"]
                ids = client.post("/tokenize", {"content": text, "add_special": False,
                                                "parse_special": True})["tokens"]
                payload = {"prompt": ids, "n_predict": args.tokens or prompt["generate_tokens"],
                           "temperature": 0, "cache_prompt": False, "n_probs": args.top,
                           "post_sampling_probs": False}
                record = stream(client, "/completion", payload, memory)
                record["prompt_ids"] = ids
                t = record.get("timings") or {}
                summarize(record, len(ids), t.get("predicted_n"))
                run["prompts"][prompt["id"]] = report(out, prompt["id"], prompt, record)
                dump(out / "run.json", run)
            except Exception as error:  # noqa: BLE001 - recorded; the rest run
                failed(out, run, prompt["id"], error)
    finally:
        stop_container(name, out)
        run["min_available_bytes"] = memory.minimum(t0)
        run["peak_drop_gib"] = (baseline - run["min_available_bytes"]) / GIB
        memory.running = False
        dump(out / "run.json", run)
    print(json.dumps({"peak_drop_gib": run["peak_drop_gib"]}), flush=True)


def cmd_llama_ppl(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    text = Path(args.text).resolve()
    mounts = [(text.parent, "/text")]
    run = {"windows": {}}
    # The window's IDs as llama-perplexity takes them (the model's BOS rule).
    ids_argv = llama_argv("jitllm-lc-tok-" + uuid.uuid4().hex[:8], 0, args.image,
                          "/app/llama-tokenize", args.model,
                          f"-f /text/{text.name} --ids --log-disable", mounts)
    ids_argv = [a for a in ids_argv if a not in ("-d",)]
    ids_argv[ids_argv.index("--publish"):ids_argv.index("--publish") + 2] = []
    ids_argv.insert(ids_argv.index("run") + 1, "--rm")
    tokens = json.loads(subprocess.run(ids_argv, check=True, capture_output=True,
                                       text=True).stdout.strip().splitlines()[-1])
    for ctx in args.ctx:
        gate(args.min_gib)
        memory = Memory()
        memory.start()
        time.sleep(1)
        baseline = Memory.available()
        (out / f"ppl-{ctx}.ids").write_text("ppl\t" + " ".join(map(str, tokens[:ctx])) + "\n")
        argv = llama_argv("jitllm-lc-ppl-" + uuid.uuid4().hex[:8], 0, args.image,
                          "/app/llama-perplexity", args.model,
                          f"-f /text/{text.name} -c {ctx} --chunks 1 {args.args}", mounts)
        argv = [a for a in argv if a != "-d"]
        argv[argv.index("--publish"):argv.index("--publish") + 2] = []
        argv.insert(argv.index("run") + 1, "--rm")
        t0 = time.monotonic()
        result = subprocess.run(argv, capture_output=True, text=True)
        seconds = time.monotonic() - t0
        (out / f"ppl-{ctx}.log").write_text(result.stdout + result.stderr)
        final = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", result.stderr +
                          result.stdout)
        run["windows"][str(ctx)] = {
            "ppl": float(final.group(1)) if final else None,
            "stderr": float(final.group(2)) if final else None,
            "seconds": seconds, "exit": result.returncode,
            "peak_drop_gib": (baseline - memory.minimum(t0)) / GIB}
        memory.running = False
        print(ctx, json.dumps(run["windows"][str(ctx)]), flush=True)
        dump(out / "run.json", run)


# ------------------------------------------------------------------ vLLM

def cmd_vllm(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    gate(args.min_gib)
    memory = Memory()
    memory.start()
    time.sleep(1)
    baseline = Memory.available()
    run = {"start": args.start, "baseline_available_bytes": baseline, "prompts": {},
           "windows": {}}
    t0 = time.monotonic()
    log = open(out / "start.log", "w")
    process = subprocess.Popen(args.start, shell=True, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    client = Client(f"http://127.0.0.1:{args.port}", args.model)
    try:
        run["ready_s"] = wait_ready(client, "/health", args.ready_timeout, process) - t0
        for prompt in prompt_files(args.prompts):
            try:
                ids = client.post("/tokenize", {"model": args.model,
                                                "messages": prompt["messages"],
                                                "add_generation_prompt": True})["tokens"]
                before = spec_counters(client)
                payload = {"model": args.model, "prompt": ids,
                           "max_tokens": args.tokens or prompt["generate_tokens"],
                           "temperature": 0, "logprobs": args.top,
                           "return_tokens_as_token_ids": True, "cache_salt": uuid.uuid4().hex}
                record = stream(client, "/v1/completions", payload, memory)
                record["prompt_ids"] = ids
                record["spec_delta"] = {k: v - before.get(k, 0)
                                        for k, v in spec_counters(client).items()}
                usage = record.get("usage") or {}
                summarize(record, usage.get("prompt_tokens", len(ids)),
                          usage.get("completion_tokens", len(record["ids"])))
                run["prompts"][prompt["id"]] = report(out, prompt["id"], prompt, record)
                dump(out / "run.json", run)
            except Exception as error:  # noqa: BLE001 - recorded; the rest run
                failed(out, run, prompt["id"], error)
        if args.ppl:
            text = Path(args.ppl).read_text()
            tokens = client.post("/tokenize", {"model": args.model, "prompt": text,
                                               "add_special_tokens": False})["tokens"]
            for ctx in args.ctx:
                ids = tokens[:ctx]
                (out / f"ppl-{ctx}.ids").write_text("ppl\t" + " ".join(map(str, ids)) + "\n")
                t1 = time.monotonic()
                body = client.post("/v1/completions", {
                    "model": args.model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                    "prompt_logprobs": 1, "cache_salt": uuid.uuid4().hex}, timeout=4 * 3600)
                seconds = time.monotonic() - t1
                lps = body["choices"][0]["prompt_logprobs"]
                nll = []
                for position in range(1, len(ids)):
                    entry = lps[position]
                    nll.append(-entry[str(ids[position])]["logprob"])
                (out / f"ppl-{ctx}.nll.json").write_text(json.dumps(nll) + "\n")
                half = nll[ctx // 2:]
                mean = sum(half) / len(half)
                run["windows"][str(ctx)] = {"tokens": len(ids), "scored": len(half),
                                            "mean_nll": mean, "ppl": __import__("math").exp(mean),
                                            "seconds": seconds}
                print(ctx, json.dumps(run["windows"][str(ctx)]), flush=True)
                dump(out / "run.json", run)
    finally:
        subprocess.run(args.stop, shell=True, check=False, stdout=open(out / "stop.log", "w"),
                       stderr=subprocess.STDOUT)
        if process.poll() is None:
            process.terminate()
        run["min_available_bytes"] = memory.minimum(t0)
        run["peak_drop_gib"] = (baseline - run["min_available_bytes"]) / GIB
        memory.running = False
        dump(out / "run.json", run)
    print(json.dumps({"peak_drop_gib": run["peak_drop_gib"]}), flush=True)


def spec_counters(client):
    try:
        text = client.get("/metrics")
    except Exception:  # noqa: BLE001 - optional endpoint
        return {}
    values = {}
    for line in (text if isinstance(text, str) else "").splitlines():
        if "spec_decode" in line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            try:
                values[name] = float(value)
            except ValueError:
                pass
    return values


# ------------------------------------------------------------------ jitLLM

def cmd_jitllm(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    # The runtime's process lock lives here and is refused in a directory
    # others may change; a supervisor (nsys, spark-job) may leave umask 0.
    out.chmod(0o755)
    gate(args.min_gib)
    memory = Memory()
    memory.start()
    time.sleep(1)
    baseline = Memory.available()
    run = {"config": Path(args.config).read_text(), "baseline_available_bytes": baseline,
           "prompts": {}}
    t0 = time.monotonic()
    with open(out / "service.log", "w") as log:
        process = subprocess.Popen([args.runtime, "--config", args.config, "--anchor",
                                    str(out / "anchor")], stdout=log, stderr=subprocess.STDOUT)
    client = Client(f"http://127.0.0.1:{args.port}", args.model)
    try:
        while "jitllm-runtime: ready" not in (out / "service.log").read_text():
            if process.poll() is not None:
                raise SystemExit("the runtime exited: " + (out / "service.log").read_text()[-2000:])
            time.sleep(0.5)
        run["ready_s"] = time.monotonic() - t0
        # A short warm-up pages the model in (the first request's swap), so
        # no measured prefill includes it.
        warm = stream(client, "/v1/chat/completions",
                      {"model": args.model, "messages": [{"role": "user", "content": "Hi"}],
                       "temperature": 0, "max_tokens": 8}, memory)
        run["warmup_s"] = warm["t_end"] - warm["t_send"]
        usage = warm.get("usage") or {}
        summarize(warm, usage.get("prompt_tokens"), usage.get("completion_tokens"))
        run["warmup"] = report(out, "warmup", {}, warm)
        dump(out / "run.json", run)
        if warm.get("error") or not warm["stream_complete"]:
            error = RuntimeError("cold page-in warm-up failed; no timed requests were sent")
            failed(out, run, "warmup", error)
            raise error
        for prompt in prompt_files(args.prompts):
            messages = list(prompt["messages"])
            turns = [None] + (prompt["followups"] if args.session else [])
            for turn, followup in enumerate(turns):
                case = prompt["id"] + (f"-turn{turn + 1}-{args.reasoning}" if args.session else "")
                if process.poll() is not None:
                    failed(out, run, case, RuntimeError("the runtime has exited"))
                    break
                try:
                    if followup is not None:
                        messages.append({"role": "user", "content": followup})
                    payload = {"model": args.model, "messages": messages, "temperature": 0,
                               "max_tokens": args.tokens or prompt["generate_tokens"]}
                    record = stream(client, "/v1/chat/completions", payload, memory)
                    # A request stopped by the route's deadline (600 s, D-097)
                    # keeps the chunks it ran, and the same request again
                    # resumes from them: the client retries, and the prefill is
                    # timed from the first send (the retries' overhead
                    # included).
                    attempts = [record.get("error")]
                    while record.get("error") and len(attempts) <= args.retries and (
                            "deadline" in json.dumps(record["error"]).lower()
                            or "504" in json.dumps(record["error"])):
                        first_send = record["t_send"]
                        record = stream(client, "/v1/chat/completions", payload, memory)
                        record["t_send"] = first_send
                        attempts.append(record.get("error"))
                    record["attempts"] = attempts
                    usage = record.get("usage") or {}
                    summarize(record, usage.get("prompt_tokens"), usage.get("completion_tokens"))
                    record["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens")
                    run["prompts"][case] = report(out, case, prompt, record)
                    run["prompts"][case]["cached_tokens"] = record["cached_tokens"]
                    run["prompts"][case]["attempts"] = attempts
                    reply = {"role": "assistant", "content": record["text"]}
                    if args.reasoning == "keep" and record["reasoning"]:
                        reply["reasoning_content"] = record["reasoning"]
                    messages.append(reply)
                    dump(out / "run.json", run)
                except Exception as error:  # noqa: BLE001 - recorded; the rest run
                    failed(out, run, case, error)
                    break
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        run["min_available_bytes"] = memory.minimum(t0)
        run["peak_drop_gib"] = (baseline - run["min_available_bytes"]) / GIB
        memory.running = False
        run["requests"] = [line for line in (out / "service.log").read_text().splitlines()
                           if " request " in line or "swap" in line or "budget" in line]
        dump(out / "run.json", run)
    print(json.dumps({"peak_drop_gib": run["peak_drop_gib"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)

    def common(p):
        p.add_argument("out")
        p.add_argument("--port", type=int, default=18150)
        p.add_argument("--tokens", type=int, default=0)
        p.add_argument("--top", type=int, default=5)
        p.add_argument("--min-gib", type=float, default=110)
        p.add_argument("--ready-timeout", type=float, default=2400)

    p = sub.add_parser("llama")
    common(p)
    p.add_argument("--image", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--args", default="")
    p.add_argument("--session", action="store_true")
    p.add_argument("prompts", nargs="+")
    p.set_defaults(func=cmd_llama)

    p = sub.add_parser("llama-ppl")
    common(p)
    p.add_argument("--image", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--args", default="")
    p.add_argument("--text", required=True)
    p.add_argument("--ctx", type=int, action="append", required=True)
    p.set_defaults(func=cmd_llama_ppl)

    p = sub.add_parser("vllm")
    common(p)
    p.add_argument("--start", required=True)
    p.add_argument("--stop", required=True)
    p.add_argument("--model", default="qwen3.8-flash-next")
    p.add_argument("--ppl")
    p.add_argument("--ctx", type=int, action="append", default=[])
    p.add_argument("prompts", nargs="*")
    p.set_defaults(func=cmd_vllm)

    p = sub.add_parser("jitllm")
    common(p)
    p.add_argument("--runtime", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--session", action="store_true")
    p.add_argument("--reasoning", choices=["keep", "drop"], default="drop")
    p.add_argument("--retries", type=int, default=0)
    p.add_argument("prompts", nargs="+")
    p.set_defaults(func=cmd_jitllm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
