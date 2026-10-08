#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize the pinned Gemma GGUF using the reference image's GGUF reader.

Run inside the pinned image, after verifying the artifact hash. This is an
experiment inspector, not llmpalooza's future untrusted-checkpoint validator.
"""

import json
import re
import sys

from gguf import GGUFReader


def summarize(path):
    reader = GGUFReader(path, mode="r")
    metadata = {
        name: field.contents()
        for name, field in reader.fields.items()
        if name.startswith(("general.", "gemma4."))
    }
    if metadata.get("general.architecture") != "gemma4":
        raise ValueError("This inspector accounts only for the selected Gemma 4")
    count = metadata["gemma4.expert_count"]
    layers = metadata["gemma4.block_count"]
    expert_layers = {}
    shared_layers = {}
    total_bytes = 0
    for tensor in reader.tensors:
        total_bytes += tensor.n_bytes
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate_up|down)_exps\.(weight|scale)", tensor.name)
        if match:
            if int(tensor.shape[-1]) != count or tensor.n_bytes % count:
                raise ValueError(f"Unexpected expert axis: {tensor.name}")
            layer = expert_layers.setdefault(int(match[1]), {})
            layer[match[2] + "." + match[3]] = {
                "type": tensor.tensor_type.name,
                "shape": [int(n) for n in tensor.shape],
                "bytes_per_expert": tensor.n_bytes // count,
            }
        elif "_exps" in tensor.name:
            raise ValueError(f"Unaccounted expert tensor: {tensor.name}")
        shared = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight", tensor.name)
        if shared:
            shared_layers.setdefault(int(shared[1]), {})[shared[2]] = tensor.n_bytes
    if set(expert_layers) != set(range(layers)):
        raise ValueError("Missing routed-expert layer")
    for layer in expert_layers.values():
        if set(layer) != {"gate_up.weight", "down.weight", "down.scale"}:
            raise ValueError("Incomplete expert projection closure")
    if set(shared_layers) != set(range(layers)) or any(
        set(layer) != {"gate", "up", "down"} for layer in shared_layers.values()
    ):
        raise ValueError("Incomplete shared expert projection closure")
    per_layer = {
        str(i): sum(p["bytes_per_expert"] for p in projections.values())
        for i, projections in sorted(expert_layers.items())
    }
    expert_bytes = sum(per_layer.values()) * count
    return {
        "metadata": metadata,
        "tensor_count": len(reader.tensors),
        "tensor_elements": sum(tensor.n_elements for tensor in reader.tensors),
        "tensor_payload_bytes": total_bytes,
        "routed_expert_payload_bytes": expert_bytes,
        "non_routed_tensor_bytes": total_bytes - expert_bytes,
        "shared_expert_projection_bytes": sum(sum(layer.values()) for layer in shared_layers.values()),
        "expert_bytes_per_layer": per_layer,
        "expert_projection_layouts": expert_layers,
    }


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=2))
