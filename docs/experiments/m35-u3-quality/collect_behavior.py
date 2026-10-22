#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect fixed-template behavioral replies; semantic verdicts require review."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw_inputs = args.inputs.read_bytes()
    if hashlib.sha256(raw_inputs).hexdigest() != args.inputs_sha256:
        raise ValueError("behavior input differs from the registered identity")
    inputs = json.loads(raw_inputs)
    if inputs["format"] != "llmp-u3-llama-behavior-inputs/1":
        raise ValueError("unexpected behavior input format")
    if [case["id"] for case in inputs["cases"]] != [
            "capital", "haiku", "sky", "fibonacci", "primes", "french"]:
        raise ValueError("behavior coverage differs from the protocol")
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"model": args.model, "inputs_sha256": args.inputs_sha256,
              "semantic_verdict": "pending_review", "cases": []}
    failed = False
    for case in inputs["cases"]:
        try:
            body = {"model": args.model, "prompt": case["ids"], "max_tokens": 256,
                    "temperature": 0, "top_p": 1, "add_special_tokens": False,
                    "stream": False}
            request = urllib.request.Request(
                args.url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
            obj = json.loads(raw)
            if obj["usage"]["prompt_tokens"] != len(case["ids"]):
                raise ValueError("engine consumed a different prompt length")
            if len(obj["choices"]) != 1 or not obj["choices"][0].get("text"):
                raise ValueError("missing completed behavioral reply")
            (args.out / f'{case["id"]}.json').write_bytes(raw)
            result["cases"].append({"id": case["id"], "response": obj,
                                    "response_sha256": hashlib.sha256(raw).hexdigest()})
        except Exception as error:
            failed = True
            result["cases"].append({"id": case["id"], "error": str(error)})
    (args.out / "behavior.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if failed:
        raise SystemExit("one or more behavioral requests failed; see behavior.json")


if __name__ == "__main__":
    main()
