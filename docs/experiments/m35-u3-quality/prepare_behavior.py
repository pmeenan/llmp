#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze the six source-template prompts without loading model weights."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

PROMPTS_SHA256 = "c697236c56a09b0a3f2550f7514b3e4d826e1d14a96b4d1c79e3bd33a3a6f859"
TOKENIZER_SHA256 = "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b"
TEMPLATE_SHA256 = "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.prompts.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROMPTS_SHA256:
        raise ValueError("behavior prompt source changed")
    if hashlib.sha256((args.checkpoint / "tokenizer.json").read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("tokenizer source changed")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("source chat template changed")
    cases = []
    for case in json.loads(raw)["chat"]:
        ids = tokenizer.apply_chat_template(case["messages"], tokenize=True,
                                             add_generation_prompt=True, return_dict=False)
        cases.append({"id": case["id"], "messages": case["messages"], "ids": ids})
    result = {"format": "llmp-u3-llama-behavior-inputs/1", "prompts_sha256": PROMPTS_SHA256,
              "tokenizer_sha256": TOKENIZER_SHA256, "template_sha256": TEMPLATE_SHA256,
              "max_tokens": 256, "cases": cases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    args.out.write_bytes(encoded)
    print("Behavior input SHA256:", hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    main()
