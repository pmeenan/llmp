#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Retain the two registered greedy replies and bind them to an existing capture.

One family/arm per exclusive spark-job --gpu --timeout 600. Output is private,
fresh and external to Git. No score passage, timing verdict or new quality bound.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import sys

from capture import EXPECTED, HERE, load_quality_pair, load_speed_gate, sha256_file, verify_inputs, verify_score


def data_lines(chunks, raw_sink):
    """Preserve received bytes; assemble lines before decoding split UTF-8/CRLF."""
    pending = b""
    for chunk in chunks:
        raw_sink.write(chunk)
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line.strip()
    if pending:
        yield pending.strip()


def parse_sse(chunks, raw_sink):
    """The pinned speed_gate contract: each delta's reasoning, then content."""
    events, reasoning, content, ordered = [], [], [], []
    finish = usage = None
    done = False
    for line in data_lines(chunks, raw_sink):
        if not line or line.startswith(b":"):
            continue
        if not line.startswith(b"data: "):
            raise ValueError("unexpected SSE line outside the registered data-line contract")
        data = line[6:]
        if data == b"[DONE]":
            done = True
            break
        event = json.loads(data)
        if not isinstance(event, dict) or "error" in event:
            raise ValueError("malformed or failed SSE event")
        choices = event.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("greedy stream must contain one choice or its usage event")
        if event.get("usage") is not None:
            if usage is not None:
                raise ValueError("duplicate usage event")
            usage = event["usage"]
        if not choices and event.get("usage") is None:
            raise ValueError("empty event without usage")
        for choice in choices:
            if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0:
                raise ValueError("unexpected choice identity")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise ValueError("missing delta")
            if set(delta) - {"role", "reasoning", "content"}:
                raise ValueError("unexpected non-text delta")
            fields = []
            for name in ("reasoning", "content"):
                value = delta.get(name)
                if value is None:
                    value = ""
                if not isinstance(value, str):
                    raise ValueError("text delta is not a string")
                fields.append(value)
            if finish is not None and any(fields):
                raise ValueError("text after completion")
            reasoning.append(fields[0])
            content.append(fields[1])
            ordered.extend(fields)
            if choice.get("finish_reason") is not None:
                if finish is not None or choice["finish_reason"] not in ("length", "stop"):
                    raise ValueError("duplicate or unsupported finish reason")
                finish = choice["finish_reason"]
        events.append(event)
    if not done or finish is None or not any(ordered):
        raise ValueError("stream lacks DONE, successful finish or reply text")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int for k in
                                        ("prompt_tokens", "completion_tokens", "total_tokens")):
        raise ValueError("missing integer usage")
    if (usage["prompt_tokens"] <= 0 or not 0 < usage["completion_tokens"] <= 128 or
            usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
        raise ValueError("invalid usage for the registered request")
    stream_text = "".join(ordered)
    return {"events": events, "reasoning": "".join(reasoning), "content": "".join(content),
            "stream_text": stream_text, "reply_sha256": hashlib.sha256(stream_text.encode()).hexdigest(),
            "finish": finish, "usage": usage}


def collect_reply(port, body, raw_path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=420)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        with raw_path.open("xb") as raw:
            if response.status != 200:
                raw.write(response.read())
                raise ValueError(f"greedy HTTP status {response.status}")
            result = parse_sse(iter(response.readline, b""), raw)
        result["raw_sse_sha256"] = sha256_file(raw_path)
        return result
    finally:
        connection.close()


def match_reply(reply, original):
    if original.get("error") or any(reply[k] != original.get(k) for k in
                                    ("reply_sha256", "finish", "usage")):
        raise ValueError("supplementary reply differs from the captured digest/finish/usage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=HERE.parents[2])
    parser.add_argument("--family", required=True, choices=("qwen38", "qwen38-gguf", "deepseek"))
    parser.add_argument("--arm", required=True, choices=("A", "C", "D"))
    parser.add_argument("--runtime", required=True, type=pathlib.Path)
    parser.add_argument("--quality-receipt", required=True, type=pathlib.Path,
                        help="exact completed quality receipt whose replies must reproduce")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--share", type=pathlib.Path, default=pathlib.Path("~/.local/share/llmp").expanduser())
    parser.add_argument("--corpus", type=pathlib.Path,
                        default=pathlib.Path("~/.local/share/llmp/corpus/m3-reference-corpus.txt").expanduser())
    args = parser.parse_args()
    os.umask(0o077)
    out = args.out.expanduser().absolute()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    pins = json.loads((HERE / "llm-capture-pins.json").read_text())
    repo = args.repo.expanduser().resolve(strict=True)
    if sha256_file(repo / "tools/speed_gate.py") != pins["speed_gate_sha256"]:
        raise SystemExit("speed_gate helper differs from registration")
    runtime = args.runtime.expanduser().resolve(strict=True)
    before = sha256_file(runtime)
    if before != EXPECTED[args.arm] or before != pins["arms"][args.arm]["sha256"]:
        raise SystemExit("runtime differs from frozen arm")
    sg = load_speed_gate(repo, debug_plans=False)
    qp = load_quality_pair(repo, pins["quality_pair_sha256"])
    family = sg.FAMILIES[args.family]
    if list(sg.QUALITY_PROMPTS) != pins["quality"]["greedy_prompts"] or family.max_tokens != 128:
        raise SystemExit("greedy requests differ from registration")
    template, inputs = verify_inputs(sg, args, pins)
    prior_path = args.quality_receipt.expanduser().resolve(strict=True)
    label = {"A": "old", "C": "candidate-c", "D": "candidate-d"}[args.arm]
    if prior_path.name != f"{args.family}-quality-{label}.json":
        raise SystemExit("wrong family/arm receipt filename")
    prior_sha = sha256_file(prior_path)
    if args.arm == "A" and prior_sha != pins["greedy_baseline_receipts"][args.family]:
        raise SystemExit("baseline receipt differs from the retained governed A capture")
    if args.arm in ("C", "D"):
        metadata = json.loads((prior_path.parent.parent / "capture-metadata.json").read_text())
        expected_metadata = {"family": args.family, "arm": args.arm, "runtime_sha256": before,
                             "runtime_sha256_after": before, "inputs": inputs,
                             "capture_sha256": sha256_file(HERE / "capture.py"),
                             "parser_sha256": sha256_file(HERE / "plan_reports.py"),
                             "pins_sha256": sha256_file(HERE / "llm-capture-pins.json"),
                             "speed_gate_sha256": pins["speed_gate_sha256"],
                             "quality_pair_sha256": pins["quality_pair_sha256"]}
        if any(metadata.get(k) != value for k, value in expected_metadata.items()):
            raise SystemExit(f"{args.arm} capture metadata differs from the registered source/inputs")
    prior = json.loads(prior_path.read_text())
    if (prior.get("error") or prior.get("family") != args.family or prior.get("label") != label or
            prior.get("runtime_sha256") != before or prior.get("config_template") != template or
            prior.get("runtime_env") != {} or prior.get("stop", {}).get("returncode") != 0 or
            len(prior.get("replies", [])) != len(sg.QUALITY_PROMPTS)):
        raise SystemExit("prior quality capture is incomplete or belongs to another arm/recipe")
    expected_prompt = sg.passage(sg.load_corpus(args.corpus.expanduser()), 100003, 4000).encode()
    verify_score(prior_path, prior, expected_prompt, args.family, qp)
    work = out / "work"
    work.mkdir(mode=0o700)
    config = work / "node.toml"
    config.write_text(sg.render_config(template, args.share.expanduser().resolve(), work / "data", 18114))
    result = {"format": "llmp-u4-greedy-text/1", "family": args.family, "arm": args.arm,
              "runtime": str(runtime), "runtime_sha256_before": before,
              "collector_sha256": sha256_file(pathlib.Path(__file__)),
              "capture_helper_sha256": sha256_file(HERE / "capture.py"),
              "pins_sha256": sha256_file(HERE / "llm-capture-pins.json"),
              "speed_gate_sha256": pins["speed_gate_sha256"],
              "quality_pair_sha256": pins["quality_pair_sha256"], "inputs": inputs,
              "config_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
              "prior_quality_receipt": str(prior_path), "prior_quality_receipt_sha256": prior_sha,
              "debug_plans": False, "timing_sample": False,
              "ready_timeout_s": 420, "stop_timeout_s": 20, "supervisor_timeout_s": 600,
              "replies": [], "error": None, "qualified": False}
    service = None
    try:
        service = sg.Service(runtime, config, work, 18114, sg.parse_runtime_env(None))
        service.wait_ready()
        for index, prompt in enumerate(sg.QUALITY_PROMPTS):
            body = {"model": family.model, "stream": True, "temperature": 0,
                    "stream_options": {"include_usage": True}, "max_tokens": family.max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
            (out / f"request-{index}.json").write_text(json.dumps(body) + "\n")
            reply = collect_reply(18114, body, out / f"reply-{index}.sse")
            result["replies"].append(reply)
            match_reply(reply, prior["replies"][index])
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if service is not None:
            result["stop"] = service.stop()
        result["runtime_sha256_after"] = sha256_file(runtime)
    result["qualified"] = (result["error"] is None and len(result["replies"]) == 2 and
                           result.get("stop", {}).get("returncode") == 0 and
                           result["runtime_sha256_after"] == before)
    (out / "greedy-text.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"family": args.family, "arm": args.arm, "qualified": result["qualified"],
                      "error": result["error"]}))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
