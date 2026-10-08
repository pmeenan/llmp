# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Perplexity reference from a vLLM server: prompt log-probabilities of a fixed text.

Usage: ppl.py BASE_URL MODEL TEXT_FILE TOKENS OUTPUT.json [--start CMD --stop CMD]

Tokenizes TEXT_FILE with the server's /tokenize (no chat template, no special
tokens), keeps the first TOKENS tokens, and asks /v1/completions for
prompt_logprobs 1 at temperature 0. Records each position's log-probability
of the actual next token and the top-1 token, and the perplexity over
positions 1..TOKENS-1. With --start, the server is started first (and the
load timed) and stopped afterwards with --stop.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time

from baseline import Client, wait_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("model")
    parser.add_argument("text", type=Path)
    parser.add_argument("tokens", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--stop")
    parser.add_argument("--ready-timeout", type=float, default=2400)
    args = parser.parse_args()
    client = Client(args.base, args.model)
    timings = {}
    process = None
    t0 = time.monotonic()
    if args.start:
        log = open(args.output.with_suffix(".start.log"), "w")
        process = subprocess.Popen(args.start, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    try:
        timings["start_to_ready_s"] = wait_ready(client, "/health", args.ready_timeout, process) - t0
        text = args.text.read_text()
        ids = client.post("/tokenize", {"model": args.model, "prompt": text,
                                        "add_special_tokens": False})["tokens"]
        total = len(ids)
        ids = ids[:args.tokens]
        t = time.monotonic()
        body = client.post("/v1/completions", {"model": args.model, "prompt": ids, "max_tokens": 1,
                                               "temperature": 0, "prompt_logprobs": 1})
        timings["request_s"] = time.monotonic() - t
    finally:
        if args.stop:
            subprocess.run(args.stop, shell=True, check=False)
    prompt_logprobs = body["choices"][0]["prompt_logprobs"]
    if len(prompt_logprobs) != len(ids) or prompt_logprobs[0] is not None:
        raise RuntimeError(f"unexpected prompt_logprobs length {len(prompt_logprobs)}")
    target, top1, top1_logprob = [], [], []
    for position in range(1, len(ids)):
        entries = {int(k): v for k, v in prompt_logprobs[position].items()}
        target.append(entries[ids[position]]["logprob"])
        best = min(entries, key=lambda k: entries[k]["rank"])
        top1.append(best)
        top1_logprob.append(entries[best]["logprob"])
    nll = -sum(target) / len(target)
    result = {"text_tokens_total": total, "tokens": len(ids), "scored_positions": len(target),
              "mean_nll": nll, "perplexity": math.exp(nll),
              "top1_agreement": sum(a == b for a, b in zip(top1, ids[1:])) / len(target),
              "timings": timings, "token_ids": ids,
              "target_logprobs": [round(v, 5) for v in target],
              "top1_token_ids": top1, "top1_logprobs": [round(v, 5) for v in top1_logprob]}
    args.output.write_text(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
