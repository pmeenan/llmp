#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Export an EXL3 fixture's exactly decoded weights for the FP64 oracle.

External reference tooling for the backend proof's P0; it does not implement
llmpalooza inference. Runs in the reference container with the GPU. Each EXL3
linear is exported as ExLlamaV3 stores it before its Hadamard transforms:
the trellis decoded by upstream's own reconstruction kernel into FP16 in the
rotated basis (exact, since decoding produces FP16 values), and its sign
vectors suh and svh as FP16. oracle.py applies the transforms in FP64, so no
FP16 rounding of the original-basis weights enters the oracle. Biases, norms
and the BF16 embedding are copied from the checkpoint. The 128-point
Hadamard matrix is exported unscaled (entries +-1).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = next(f for f in json.loads(args.pins.read_text())["fixtures"]
                   if f["repository"].endswith(args.model.name))
    if digest(args.model / fixture["filename"]) != fixture["published_sha256"]:
        raise ValueError("fixture hash mismatch")
    from exllamav3 import Config, Model
    from exllamav3.modules.quant.exl3_lib.quantize import get_hadamard_dt
    from safetensors import safe_open
    config = Config.from_directory(str(args.model))
    model = Model.from_config(config)
    model.load(device="cuda:0")
    arrays = {"hadamard128": get_hadamard_dt(128, "cpu", torch.float32).to(torch.int8).numpy()}
    with torch.inference_mode():
        for module in model:
            inner = getattr(module, "inner", None)
            if inner is None or type(inner).__name__ != "LinearEXL3":
                continue
            key = module.key
            suh = inner.unpack_bf(inner.su) if inner.su is not None else inner.suh
            svh = inner.unpack_bf(inner.sv) if inner.sv is not None else inner.svh
            arrays[f"{key}.inner"] = inner.get_inner_weight_tensor().cpu().numpy()
            arrays[f"{key}.suh"] = suh.half().cpu().numpy()
            arrays[f"{key}.svh"] = svh.half().cpu().numpy()
    with safe_open(str(args.model / fixture["filename"]), "pt") as source:
        for key in source.keys():
            if key.endswith((".bias", "norm.weight", "layernorm.weight", "embed_tokens.weight")):
                tensor = source.get_tensor(key)
                arrays[key] = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
    linears = sum(k.endswith(".inner") for k in arrays)
    if linears != 169:
        raise ValueError(f"expected 169 EXL3 linears, found {linears}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    print(json.dumps({"fixture": fixture["repository"], "linears": linears, "arrays": len(arrays),
                      "output_sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
