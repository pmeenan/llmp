#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""llmpalooza's fixed inputs from the vLLM oracle's references (stdlib only).

  python3 reference_inputs.py FAST_SWAP_DIR OUT_DIR

Reads reference-qwen3.8-nvfp4-vllm.json and reference-qwen3.8-nvfp4-vllm-ppl.json
and writes, for llmp_qwen38_exec:
  prompts.tsv  name<TAB>the chat prompt's rendered token IDs (the oracle's)
  forced.tsv   name<TAB>the oracle's 32 greedy tokens (teacher forcing)
  ppl.tsv      the perplexity text's token IDs (the oracle's tokenization)
"""
import json
import sys
from pathlib import Path


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    src, out = Path(argv[1]), Path(argv[2])
    greedy = json.loads((src / "reference-qwen3.8-nvfp4-vllm.json").read_text())
    ppl = json.loads((src / "reference-qwen3.8-nvfp4-vllm-ppl.json").read_text())
    prompts, forced = [], []
    for p in greedy["prompts"]:
        prompts.append(p["id"] + "\t" + " ".join(str(t) for t in p["prompt_token_ids"]))
        forced.append(p["id"] + "\t" + " ".join(str(t) for t in p["greedy_token_ids"]))
    (out / "prompts.tsv").write_text("\n".join(prompts) + "\n")
    (out / "forced.tsv").write_text("\n".join(forced) + "\n")
    (out / "ppl.tsv").write_text("async-model\t" + " ".join(str(t) for t in ppl["token_ids"]) + "\n")


if __name__ == "__main__":
    main(sys.argv)
