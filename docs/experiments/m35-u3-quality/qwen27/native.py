#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect identical frozen Qwen targets from an already supervised native service."""
import argparse
import json
import os
from pathlib import Path
import urllib.request

from contract import PHASES, encoded, frozen, native_records, select_cases, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="qwen27")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--prereg-commit", required=True)
    parser.add_argument("--phase", choices=["short", "depth", "behavior"], required=True)
    parser.add_argument("--case", help="one frozen behavior case per bounded job")
    parser.add_argument("--repeat", type=int, choices=[1, 2], default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec, all_ids, behavior, protocol_sha = frozen(args.protocol, args.inputs, args.prereg_commit)
    cases = select_cases(args.phase, args.case, behavior)
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    requests = []
    if args.phase == "behavior":
        for case in cases:
            requests.append((case["id"], {"model": args.model, "prompt": case["ids"],
                             "max_tokens": 512, "temperature": 0, "top_p": 1,
                             "add_special_tokens": False, "stream": False}))
    else:
        count, first, expected, document = PHASES[args.phase]
        ids = all_ids[:count]
        requests.append((args.phase, {"model": args.model, "prompt": ids, "max_tokens": 0,
                         "temperature": 0, "top_p": 1, "echo": True, "prompt_logprobs": 1,
                         "logprobs": 1, "add_special_tokens": False, "stream": False}))
    summaries = []
    for name, body in requests:
        (args.out / f"{name}-request.json").write_bytes(encoded(body))
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/completions",
                                         data=encoded(body), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=540) as response:
            raw = response.read()
        (args.out / f"{name}-response.json").write_bytes(raw)
        obj = json.loads(raw)
        if args.phase == "behavior":
            choices = obj.get("choices")
            if (obj.get("error") or not isinstance(choices, list) or len(choices) != 1
                    or obj.get("usage", {}).get("prompt_tokens") != len(body["prompt"])):
                raise ValueError("native behavior refused or prompt count differs")
            choice = choices[0]
            if choice.get("finish_reason") not in ("stop", "length") or not isinstance(choice.get("text"), str):
                raise ValueError("native behavior has an invalid finish reason or text")
            summaries.append({"id": name, "text": choice["text"], "finish_reason": choice["finish_reason"],
                              "usage": obj["usage"], "response_sha256": sha(raw)})
        else:
            records = native_records(obj, ids, first, document)
            if len(records) != expected:
                raise ValueError("native target count differs")
            losses = b"".join(encoded(row) + b"\n" for row in records)
            (args.out / "losses.jsonl").write_bytes(losses)
            summaries.append({"targets": len(records), "response_sha256": sha(raw),
                              "losses_sha256": sha(losses)})
    (args.out / "receipt.json").write_bytes(encoded({"protocol_sha256": protocol_sha,
        "prereg_commit": args.prereg_commit, "phase": args.phase, "repeat": args.repeat,
        "ids_sha256": spec["ids_sha256"], "expected_native_artifact": spec["native_artifact"],
        "expected_native_runtime_sha256": spec["native_runtime_sha256"], "completed": True, "records": summaries}))


if __name__ == "__main__":
    main()
