#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect registered full-vocabulary target likelihoods from either engine."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.request

IDS_SHA256 = "cb67d66726c438813f2f4378b9943e83e7db10b0fe80326cbf14e991ed558f9d"


def extract(response, ids, phase):
    """Top-k here limits reported alternatives, never the softmax denominator."""
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("expected exactly one successful completion choice")
    scores = choices[0].get("prompt_logprobs")
    if not isinstance(scores, list) or len(scores) != len(ids) or scores[0] is not None:
        raise ValueError("prompt-score shape/BOS does not match the registered IDs")
    usage = response.get("usage", {})
    if usage.get("prompt_tokens") != len(ids):
        raise ValueError("engine consumed a different number of prompt tokens")
    # vLLM exposes this field optionally; native authenticates the submitted IDs
    # through actual-target map keys and the exact prompt count instead.
    returned_ids = choices[0].get("prompt_token_ids")
    if returned_ids is not None and returned_ids != ids:
        raise ValueError("engine returned a different prompt token sequence")
    first = 1 if phase == "short" else 8192
    records = []
    for position in range(first, len(ids)):
        row = scores[position]
        key = str(ids[position])
        if not isinstance(row, dict) or key not in row:
            raise ValueError(f"missing actual-target score at position {position}")
        entry = row[key]
        value = entry.get("logprob") if isinstance(entry, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid target score at position {position}")
        if not math.isfinite(value) or value > 0:
            raise ValueError(f"non-finite or positive log probability at {position}")
        records.append({"document": 0 if phase == "short" else 1,
                        "position": position, "target": ids[position], "nll": -value})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["short", "depth"], default="short")
    args = parser.parse_args()
    raw_ids = args.ids.read_bytes()
    if hashlib.sha256(raw_ids).hexdigest() != IDS_SHA256:
        raise ValueError("IDs are not the frozen pre-registration input")
    full_ids = json.loads(raw_ids)
    if len(full_ids) != 9216 or full_ids[0] != 128000:
        raise ValueError("registered Llama input geometry is invalid")
    ids = full_ids[:1024] if args.phase == "short" else full_ids
    args.out.mkdir(parents=True, exist_ok=True)
    body = {"model": args.model, "prompt": ids, "max_tokens": 0,
            "temperature": 0, "top_p": 1, "echo": True,
            "prompt_logprobs": 1, "logprobs": 1,
            "add_special_tokens": False, "stream": False}
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=540) as response:
        raw = response.read()
    (args.out / "response.json").write_bytes(raw)
    records = extract(json.loads(raw), ids, args.phase)
    with (args.out / "losses.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    receipt = {"model": args.model, "phase": args.phase, "ids_sha256": IDS_SHA256,
               "response_sha256": hashlib.sha256(raw).hexdigest(),
               "targets": len(records), "request": body}
    (args.out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    print(f"Collected {len(records)} full-vocabulary actual-target losses")


if __name__ == "__main__":
    main()
