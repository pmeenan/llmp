#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Inputs-only preparation with the exact source tokenizer/template; never loads weights."""
import argparse
import json
import os
from pathlib import Path

from contract import CORPUS, METADATA, encoded, metadata, sha, valid_ids

CASES = [
    {"id": "capital", "prompt": "What is the capital of France? Reply with only the city name.",
     "kind": "text", "expected": "Paris"},
    {"id": "arithmetic", "prompt": "Calculate 17 times 23. Reply with only the integer.",
     "kind": "text", "expected": "391"},
    {"id": "sort", "prompt": "Sort these integers in ascending order: 9, 2, 1, 2. Return only a JSON array.",
     "kind": "json", "expected": [1, 2, 2, 9]},
    {"id": "structured", "prompt": "Return only a JSON object with exactly two keys: planet set to Mars and moons set to the integer 2.",
     "kind": "json", "expected": {"planet": "Mars", "moons": 2}},
    {"id": "context", "prompt": "Use this table: Mira has code AZ-417; Niko has code BX-208. What is Mira's code? Reply with only the code.",
     "kind": "text", "expected": "AZ-417"},
    {"id": "fibonacci", "prompt": "Write only a Python function named fib(n) that returns a list of the first n Fibonacci numbers, starting 0, 1. For n=0 return []. Keep it concise. Do not include markdown, tests, or explanation.",
     "kind": "python_fib", "checks": {"0": [], "1": [0], "7": [0, 1, 1, 2, 3, 5, 8]}},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if list(Path("/dev").glob("nvidia*")):
        raise ValueError("inputs-only container unexpectedly exposes GPU devices")
    from tokenizers import Tokenizer
    from tensorfold.cuda.chat_template import ChatTemplate
    from source_adapter import authenticate_source
    source = authenticate_source()
    metadata(args.checkpoint)
    raw = args.corpus.read_bytes()
    if len(raw) != 817275 or sha(raw) != CORPUS:
        raise ValueError("retained corpus changed")
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    config = json.loads((args.checkpoint / "config.json").read_bytes())
    vocab = config.get("text_config", config)["vocab_size"]
    ids = tokenizer.encode(raw.decode(), add_special_tokens=False).ids[:9216]
    valid_ids(ids, vocab, count=9216)
    template = ChatTemplate(args.checkpoint)
    cases = []
    for case in CASES:
        messages = [{"role": "user", "content": case["prompt"]}]
        rendered = template.render(messages, tools=None, enable_thinking=False)
        # Refuse a dynamic date or nondeterministic template before freezing anything.
        if rendered != template.render(messages, tools=None, enable_thinking=False):
            raise ValueError("source template is not deterministic")
        tokens = tokenizer.encode(rendered, add_special_tokens=False).ids
        valid_ids(tokens, vocab)
        if len(tokens) + 512 > 12288:
            raise ValueError("behavior does not fit the registered context")
        cases.append({**case, "messages": messages, "rendered": rendered, "ids": tokens})
    behavior = {"format": "llmp-u3-qwen27-behavior/1", "enable_thinking": False,
                "max_tokens": 512, "cases": cases}
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "ids.json").write_bytes(encoded(ids))
    (args.out / "behavior.json").write_bytes(encoded(behavior))
    receipt = {"corpus_sha256": CORPUS, "metadata": METADATA, "vocab_size": vocab,
               "count": len(ids), "first_id": ids[0], "add_special_tokens": False,
               "ids_sha256": sha(encoded(ids)), "behavior_sha256": sha(encoded(behavior)),
               "source": source, "weight_loads": 0}
    (args.out / "inputs.json").write_bytes(encoded(receipt))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
