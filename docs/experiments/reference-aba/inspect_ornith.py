#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Inspect the hash-verified Ornith GGUF with the pinned reference GGUF reader.

Experiment accounting only; not llmpalooza's future untrusted-input importer.
"""
import json
import re
import sys

from gguf import GGUFReader


def summarize(path):
    reader = GGUFReader(path, mode="r")
    metadata = {name: field.contents() for name, field in reader.fields.items()
                if name.startswith(("general.", "qwen35moe."))}
    if metadata.get("general.architecture") != "qwen35moe":
        raise ValueError("Expected the pinned Ornith qwen35moe artifact")
    layers = metadata["qwen35moe.block_count"]
    experts = metadata["qwen35moe.expert_count"]
    primary_layers = layers - metadata["qwen35moe.nextn_predict_layers"]
    if (layers, primary_layers, experts) != (41, 40, 256):
        raise ValueError("Unexpected primary/MTP/expert layout")
    routed, shared = {}, {}
    total = mtp = 0
    for tensor in reader.tensors:
        total += tensor.n_bytes
        block = re.match(r"blk\.(\d+)\.", tensor.name)
        if block and int(block[1]) >= primary_layers:
            mtp += tensor.n_bytes
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
        if match:
            if int(tensor.shape[-1]) != experts or tensor.n_bytes % experts:
                raise ValueError("Unexpected expert tensor axis")
            routed.setdefault(int(match[1]), {})[match[2]] = {
                "type": tensor.tensor_type.name, "bytes_per_expert": tensor.n_bytes // experts}
        elif "_exps" in tensor.name:
            raise ValueError("Unaccounted expert tensor")
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_shexp\.weight", tensor.name)
        if match:
            shared.setdefault(int(match[1]), {})[match[2]] = tensor.n_bytes
    for partition in (routed, shared):
        if set(partition) != set(range(layers)) or any(set(p) != {"gate", "up", "down"} for p in partition.values()):
            raise ValueError("Incomplete expert closure")
    per_layer = {i: sum(p["bytes_per_expert"] for p in projections.values()) for i, projections in routed.items()}
    routed_bytes = sum(per_layer.values()) * experts
    return {"metadata": metadata, "tensor_count": len(reader.tensors),
            "tensor_elements": sum(t.n_elements for t in reader.tensors),
            "tensor_payload_bytes": total, "routed_expert_payload_bytes": routed_bytes,
            "non_routed_tensor_bytes": total-routed_bytes,
            "mtp_tensor_payload_bytes": mtp, "primary_tensor_payload_bytes": total-mtp,
            "primary_routed_expert_bytes": sum(per_layer[i] for i in range(primary_layers))*experts,
            "shared_projection_bytes": sum(sum(p.values()) for p in shared.values()),
            "expert_bytes_per_layer": per_layer, "expert_projection_layouts": routed}


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=2))
