#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end agent-style tool requests: tokens/s at C1 and C4, and greedy outputs.

    agent_e2e.py --runtime BIN --family qwen|gemma --out DIR [--port P] [--modes ...]
                 [--cells CASE:TOOLS,...]

Starts llmp-runtime on loopback with one model, sends streamed chat requests
(tools present with a tool_choice of auto, none (`off`), required or named,
temperature 0; and the same messages with no tools, `none`, as the
reference), and writes DIR/<family>.json with each request's timings, usage
and its full greedy output (reasoning, content, tool calls).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time

SHARE = pathlib.Path(os.path.expanduser("~/.local/share/llmp"))
QWEN = "02354af162f30b6ec6a117d5311da7d642c5be3128867750be55eeb7b6cd38b9"
QWEN_MTP = "bb49b7da7edb78202a64f1f9ab049412a10f2a9f73eb4eef66608e81b011f263"
GEMMA26 = "a1f1c523b126353f0d186dff0602f91651b05d65118006ed48958ae4f275e9e1"
QWEN_FILES = SHARE / "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6"

MODELS = {
    "qwen": (
        "qwen3.8",
        f'[models."qwen3.8"]\nartifact = "{QWEN}"\ndrafter = "{QWEN_MTP}"\nspeculation = true\n'
        "draft_vocab = 65536\ndepth_cost_ratio = 1.16\ncontext = 16384\nprefill_chunk = 4096\n"
        f'max_slots = 4\ntokenizer = "{QWEN_FILES}/tokenizer.json"\n'
        f'chat_template = "{QWEN_FILES}/chat_template.jinja"\n',
    ),
    "gemma": (
        "gemma26",
        f'[models.gemma26]\nartifact = "{GEMMA26}"\ncontext = 8192\nprefill_chunk = 1024\n'
        "max_slots = 4\n",
    ),
}

SYSTEM = (
    "You are a coding and research agent working in a user's project. You can call tools to "
    "read files, search the web and look up the weather. Call a tool whenever it would give a "
    "better answer than guessing, and call several tools at once when the questions are "
    "independent. After the tools answer, reply briefly.\n\n"
    "Guidelines:\n- Be concise.\n- Prefer tools over memory for facts that change.\n"
    "- Never invent file contents.\n"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city's name."},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web and return the top results.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file of the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
]

QUESTIONS = [
    "What is the weather in Paris right now?",
    "Is it raining in Tokyo at the moment?",
    "What's the temperature in Berlin today?",
    "How windy is it in Madrid right now?",
]
CITIES = ["Paris", "Tokyo", "Berlin", "Madrid"]
RESULTS = [
    '{"temperature_c": 18, "conditions": "light rain", "wind_kmh": 12}',
    '{"temperature_c": 24, "conditions": "heavy rain", "wind_kmh": 30}',
    '{"temperature_c": 9, "conditions": "overcast", "wind_kmh": 5}',
    '{"temperature_c": 27, "conditions": "sunny", "wind_kmh": 41}',
]
TEXT = [
    "Explain how a hash table handles collisions, in about 200 words.",
    "Explain how TCP congestion control works, in about 200 words.",
    "Explain what a B-tree is and why databases use it, in about 200 words.",
    "Explain how garbage collection works in Java, in about 200 words.",
]


def messages(case: str, q: int) -> list:
    system = {"role": "system", "content": SYSTEM}
    if case == "text":
        return [system, {"role": "user", "content": TEXT[q]}]
    user = {"role": "user", "content": QUESTIONS[q]}
    if case == "call":
        return [system, user]
    call = {
        "id": f"call_{q}",
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": CITIES[q]})},
    }
    return [
        system,
        user,
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": f"call_{q}", "content": RESULTS[q]},
    ]


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def chat(port: int, body: dict) -> dict:
    record: dict = {"error": None}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    reasoning, content = [], []
    calls: dict[int, dict] = {}
    try:
        record["t_send"] = time.monotonic()
        conn.request(
            "POST", "/v1/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        if response.status != 200:
            record["error"] = f"HTTP {response.status}: {response.read()[:600]!r}"
            return record
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            now = time.monotonic()
            if data == b"[DONE]":
                record["t_end"] = now
                break
            event = json.loads(data)
            if "error" in event:
                record["error"] = str(event["error"])
                break
            if event.get("usage"):
                record["usage"] = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                got = False
                if delta.get("reasoning"):
                    reasoning.append(delta["reasoning"])
                    got = True
                if delta.get("content"):
                    content.append(delta["content"])
                    got = True
                for call in delta.get("tool_calls") or []:
                    slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                    function = call.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
                    got = True
                if got:
                    record.setdefault("t_first", now)
                    record["t_last"] = now
                if choice.get("finish_reason"):
                    record["finish"] = choice["finish_reason"]
    except Exception as e:  # noqa: BLE001
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    record["reasoning"] = "".join(reasoning)
    record["content"] = "".join(content)
    record["tool_calls"] = [calls[i] for i in sorted(calls)]
    if not record["error"] and "usage" in record and "t_first" in record and "t_end" in record:
        tokens = record["usage"].get("completion_tokens", 0)
        record["decode_s"] = record["t_end"] - record["t_first"]
        record["total_s"] = record["t_end"] - record["t_send"]
        record["decode_tok_s"] = (tokens - 1) / record["decode_s"] if record["decode_s"] > 0 else 0
    return record


def body(model: str, case: str, q: int, tools: str, max_tokens: int) -> dict:
    out = {
        "model": model,
        "messages": messages(case, q),
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools != "none":
        out["tools"] = TOOLS
        if tools == "auto":
            out["tool_choice"] = "auto"
        elif tools == "off":
            out["tool_choice"] = "none"
        elif tools == "required":
            out["tool_choice"] = "required"
        elif tools == "named":
            out["tool_choice"] = {"type": "function", "function": {"name": "get_weather"}}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--family", required=True, choices=sorted(MODELS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=18119)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--modes", default="c1,c4,choices")
    # case:tools cells for c1 and c4 (tools: auto, off = tool_choice none,
    # required, named, none = no tools).
    parser.add_argument("--cells", default="text:auto,text:none,call:auto,answer:auto")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = out / f"run-{args.family}"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(mode=0o700)
    model, toml = MODELS[args.family]
    store = "u2-v1-artifacts" if (SHARE / "u2-v1-artifacts").is_dir() else "artifacts"
    config = run / "llmp.toml"
    config.write_text(
        f'schema_version = 2\n[storage]\ndata_dir = "{run / "data"}"\n'
        f'installed = "{SHARE / store}"\n[client]\nbind = ["loopback"]\nport = {args.port}\n' + toml
    )
    if port_open(args.port):
        print(f"port {args.port} is in use", file=sys.stderr)
        return 1
    log = open(run / "runtime.log", "wb")
    env = dict(os.environ, CUDA_DISABLE_PTX_JIT="1")
    started = time.monotonic()
    proc = subprocess.Popen(
        [args.runtime, "--config", str(config), "--anchor", str(run / "enrollment")],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )
    result: dict = {"family": args.family, "runtime": args.runtime, "max_tokens": args.max_tokens}
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"runtime exited {proc.returncode} before ready")
            if time.monotonic() - started > 400:
                raise RuntimeError("runtime not ready in 400 s")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=5)
                conn.request("GET", "/v1/models")
                ok = conn.getresponse().status == 200
                conn.close()
                if ok:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        result["ready_s"] = time.monotonic() - started
        modes = args.modes.split(",")
        records: list[dict] = []

        def one(case: str, tools: str, q: int, repeat: int, c: int) -> dict:
            r = chat(args.port, body(model, case, q, tools, args.max_tokens))
            r.update(case=case, tools=tools, question=q, repeat=repeat, c=c)
            print(json.dumps({k: r.get(k) for k in (
                "case", "tools", "question", "repeat", "c", "error", "finish", "decode_tok_s",
                "usage")}), flush=True)
            return r

        # (case, tools): a text answer with tools present and without (the
        # reference), the agent's call turn and its answer after the result.
        cells = [tuple(cell.split(":")) for cell in args.cells.split(",")]
        if "c1" in modes:
            for repeat in range(args.repeats):
                for case, tools in cells:
                    records.append(one(case, tools, 0, repeat, 1))
        if "c4" in modes:
            for repeat in range(args.repeats):
                for case, tools in cells:
                    start = time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(4) as pool:
                        futures = [pool.submit(one, case, tools, q, repeat, 4) for q in range(4)]
                        wave = [f.result() for f in futures]
                    wall = time.monotonic() - start
                    tokens = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in wave)
                    for r in wave:
                        r["wave_s"] = wall
                        r["wave_tok_s"] = tokens / wall if wall > 0 else 0
                    print(json.dumps({"wave": case, "tools": tools, "repeat": repeat,
                                      "tokens": tokens, "wall_s": wall}), flush=True)
                    records.extend(wave)
        if "choices" in modes:
            for tools in ("required", "named"):
                for case in ("call", "answer"):
                    records.append(one(case, tools, 0, 0, 1))
        result["records"] = records
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        print(result["error"], file=sys.stderr)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log.close()
    (out / f"{args.family}.json").write_text(json.dumps(result, indent=1))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
