#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Separate, fixed notebook-recall check; never part of the 27 timing trials."""
import argparse
import json
import os
from pathlib import Path
import re
import signal

from experiment import HERE, Router, a_reusable_prefix, check, common_prefix, dump, records, sha, verify_reuse

QUESTION = ("From the notebook, give the service name, shard number and timeout in milliseconds "
            "for records 0063 and 0612. Use just two lines in this format: "
            "Record NNNN: SERVICE; shard SHARD; timeout TIMEOUT ms.")
EXPECTED = {"0063": ("svc-12", 8, 83), "0612": ("svc-0", 7, 72)}


def score(content):
    observed = {}
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    well_formed = len(lines) == len(EXPECTED)
    pattern = r"Record\s+(\d+)\s*:\s*(svc-\d+)\s*;\s*shard\s+(\d+)\s*;\s*timeout\s+(\d+)\s+ms\.?"
    for line in lines:
        match = re.fullmatch(pattern, line, re.IGNORECASE)
        if match is None:
            well_formed = False
            continue
        number, service, shard, timeout = match.groups()
        record = f"{int(number):04d}"
        if record in observed:
            well_formed = False
        observed[record] = (service.lower(), int(shard), int(timeout))
    return {"correct": well_formed and observed == EXPECTED,
            "well_formed": well_formed, "observed": observed, "expected": EXPECTED}


def run(args):
    pins = json.loads((HERE / "artifacts.json").read_text())
    check(sha(args.trace) == pins["trace_sha256"], "Frozen trace hash mismatch")
    spec = pins["models"]["A"]
    model = args.models / spec["filename"]
    check(model.stat().st_size == spec["size_bytes"] and sha(model) == spec["sha256"], "A GGUF integrity failure")
    trace = json.loads(args.trace.read_text())
    target = args.output.resolve()
    target.mkdir(parents=True, mode=0o700, exist_ok=False)
    router = Router(args, pins, target, maximum=1)
    result = {"passed": False, "question": QUESTION, "trace_sha256": pins["trace_sha256"],
              "image": pins["image"], "harness_sha256": sha(Path(__file__)), "pressure_gib": 0}
    try:
        router.start()
        result["effective_status"] = router.load("A")
        messages = []
        for i, turn in enumerate(trace["A_turns"]):
            text = ("Keep this technical notebook for later questions.\n" + records(0, 512)
                    if i == 0 else records(512+(i-1)*64, 64)) + f"\nReply with only CHECK{i+1}."
            messages.append({"role": "user", "content": text})
            check(router.template_messages("A", messages) == turn["prompt"], "Reconstructed history differs from frozen trace")
            reply = router.complete("A", turn["prompt"])
            check(reply["tokens"] == turn["expected_tokens"], "Initial output differs from frozen trace")
            messages.append({"role": "assistant", "content": reply["content"]})
        saved = result["save"] = router.save("A")
        check(saved["n_saved"] == trace["A_reuse"]["n_saved"], "Saved prefix differs from the fixed trace")
        messages.append({"role": "user", "content": QUESTION})
        prompt = router.template_messages("A", messages)
        last = trace["A_turns"][-1]
        prefix = common_prefix((last["prompt"] + last["expected_tokens"])[:saved["n_saved"]], prompt)
        check(prefix == a_reusable_prefix(trace, saved["n_saved"]), "Recall rewrote a different history prefix")
        result["prompt"] = prompt
        result["expected_reused_tokens"] = prefix
        result["resident"] = router.complete("A", prompt)
        verify_reuse(result["resident"], saved, prompt, prefix)
        router.unload("A")
        router.load("A")
        result["restore"] = router.restore("A", saved)
        result["restored"] = router.complete("A", prompt)
        verify_reuse(result["restored"], saved, prompt, prefix)
        check(result["resident"]["tokens"] == result["restored"]["tokens"], "Recall output differs after restore")
        result["facts"] = score(result["restored"]["content"])
        # Equal but incorrect facts are inconclusive recall, not evidence of KV corruption.
        result["passed"] = True
    finally:
        try:
            router.stop()
        except BaseException:
            result["passed"] = False
            raise
        finally:
            dump(target / "result.json", result)
    print(json.dumps({"continuation_tokens_equal": result["passed"], "facts": result["facts"],
                      "saved_tokens": saved["n_saved"], "reused_tokens": prefix,
                      "restored_timings": result["restored"]["final"]["timings"]}, indent=2))


if __name__ == "__main__":
    os.umask(0o077)
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--port", default=18081, type=int)
    run(parser.parse_args())
