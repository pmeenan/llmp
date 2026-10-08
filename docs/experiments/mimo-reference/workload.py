#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded text workload against the loopback reference endpoint on the head.

Synthetic prompts only; no user data. Records client-side first-token time,
token arrival gaps, usage and RDMA port counters around each request.
"""

import argparse
import http.client
import json
from pathlib import Path
import statistics
import time

WORDS = ("amber basalt cobalt delta ember fjord granite harbor indigo juniper kelp lumen meadow "
         "nickel onyx prairie quartz ridge sierra tundra umber violet willow xenon yarrow zephyr").split()
HCAS = ("rocep1s0f1", "roceP2p1s0f1")


def rdma_counters():
    out = {}
    for hca in HCAS:
        base = Path(f"/sys/class/infiniband/{hca}/ports/1/counters")
        try:
            # port_{xmit,rcv}_data count 4-byte words.
            out[hca] = {k: int((base / f"port_{k}_data").read_text()) * 4 for k in ("xmit", "rcv")}
        except OSError:
            out[hca] = None
    return out


def counter_delta(before, after):
    return {h: None if before[h] is None or after[h] is None else
            {k: after[h][k] - before[h][k] for k in before[h]} for h in before}


def document(tag, records):
    state = sum(map(ord, tag)) * 2654435761 % (1 << 32)
    lines = [f"Reference document {tag}."]
    for i in range(records):
        state = (1103515245 * state + 12345) % (1 << 31)
        a, b, v = WORDS[state % 26], WORDS[(state >> 5) % 26], state % 9973
        lines.append(f"Record {i:05d}: the {a} {b} sample measured {v} units.")
    return "\n".join(lines)


def request(host, port, model, content, max_tokens, ignore_eos=False, timeout=1800):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    counters_before = rdma_counters()
    sent = time.monotonic()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:500]!r}")
    arrivals, text, usage, finish = [], [], None, None
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                arrivals.append(time.monotonic() - sent)
                text.append(piece)
            finish = choice.get("finish_reason") or finish
    total = time.monotonic() - sent
    conn.close()
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    result = {
        "first_content_seconds": round(arrivals[0], 4) if arrivals else None,
        "total_seconds": round(total, 4),
        "content_chunks": len(arrivals),
        "usage": usage,
        "finish_reason": finish,
        "text": "".join(text),
        "rdma_bytes": counter_delta(counters_before, rdma_counters()),
    }
    if gaps:
        ordered = sorted(gaps)
        result["gap_seconds"] = {
            "median": round(statistics.median(ordered), 5),
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 5),
            "p99": round(ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))], 5),
            "max": round(ordered[-1], 5),
        }
        completion = (usage or {}).get("completion_tokens")
        if completion and completion > 1:
            result["decode_tokens_per_second"] = round((completion - 1) / (arrivals[-1] - arrivals[0]), 3)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", default="mimo")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    host = "127.0.0.1"
    cases = []

    def run(name, content, max_tokens, **kw):
        row = {"case": name, **request(host, args.port, args.model, content, max_tokens, **kw)}
        cases.append(row)
        print(json.dumps({k: row.get(k) for k in ("case", "first_content_seconds", "total_seconds", "usage",
                                                  "decode_tokens_per_second")}), flush=True)
        return row

    question = "What is 19 + 23? Reply with only the number."
    for i in range(3):
        run(f"smoke-{i}", question, 16)
    # Prefill ladder: distinct documents so no prompt prefix is shared.
    for records, label in ((60, "1k"), (240, "4k"), (960, "16k")):
        for rep in range(2):
            run(f"prefill-{label}-{rep}", document(f"{label}-{rep}", records) +
                "\nWhich record has the largest value? Answer with its number.", 1)
    # Exact repeat of the last long prompt: prefix-cache reuse on the resident server.
    run("prefill-16k-1-repeat", document("16k-1", 960) +
        "\nWhich record has the largest value? Answer with its number.", 1)
    for rep in range(2):
        run(f"decode-256-{rep}", document(f"decode-{rep}", 60) +
            "\nWrite a long, detailed essay about these records.", 256, ignore_eos=True)
    smoke = [c["text"].strip() for c in cases if c["case"].startswith("smoke")]
    summary = {
        "smoke_answers": smoke,
        "smoke_contains_42": all("42" in s for s in smoke),
        "smoke_identical": len(set(smoke)) == 1,
    }
    args.output.write_text(json.dumps({"summary": summary, "cases": cases}, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
