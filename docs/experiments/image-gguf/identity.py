#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Match repackaged safetensors against the upstream diffusers components.

Name-agnostic: each tensor's content is hashed after conversion to the
comparison dtype, and the candidate's multiset of (shape, hash) is matched
against the reference's. Shapes ignore singleton dimensions (a 2D kernel
stored as 3D with depth 1 has the same bytes). An unmatched candidate tensor
also matches when it splits along dim 0 into equal row blocks that are all
reference tensors (a fused projection); every match consumes its references. Run in the pinned image-reference container after
both inputs are hash-verified. Reads tensors one at a time.
Usage: identity.py --dtype bfloat16 --candidate A.safetensors --reference B1 [B2 ...]
"""

import argparse
import collections
import hashlib
import json

from safetensors import safe_open
import torch


def key(tensor):
    shape = tuple(n for n in tensor.shape if n != 1)
    return shape, hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy()).hexdigest()


def fingerprints(paths, dtype, keep_unmatched=None):
    out = collections.Counter()
    sizes = {}
    stored = collections.Counter()
    tensors = {}
    for path in paths:
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                stored[str(tensor.dtype)] += 1
                converted = tensor.to(dtype).contiguous()
                k = key(converted)
                out[k] += 1
                sizes[k] = converted.numel() * converted.element_size()
                if keep_unmatched is not None and k not in keep_unmatched:
                    tensors.setdefault(k, converted)
    return out, sizes, stored, tensors


def split_matches(tensor, available):
    """Return the reference keys consumed if tensor is a row-concatenation of them."""
    for parts in (2, 3, 4, 6, 8):
        if tensor.ndim < 2 or tensor.shape[0] % parts:
            continue
        keys = [key(chunk) for chunk in tensor.chunk(parts, dim=0)]
        need = collections.Counter(keys)
        if all(available[k] >= n for k, n in need.items()):
            return keys
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--reference", nargs="+", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    ref, _, ref_stored, _ = fingerprints(args.reference, dtype)
    cand, cand_sizes, cand_stored, loose = fingerprints(args.candidate, dtype, keep_unmatched=ref)
    matched = cand & ref
    available = ref - matched
    fused = collections.Counter()
    for k, n in (cand - ref).items():
        for _ in range(n):
            keys = split_matches(loose[k], available)
            if keys is None:
                break
            available -= collections.Counter(keys)
            fused[k] += 1
    unmatched = cand - matched - fused
    print(json.dumps({
        "label": args.label,
        "comparison_dtype": args.dtype,
        "candidate_stored_dtypes": dict(cand_stored),
        "reference_stored_dtypes": dict(ref_stored),
        "candidate_tensors": sum(cand.values()),
        "reference_tensors": sum(ref.values()),
        "matched_tensors": sum(matched.values()),
        "row_concatenated_tensors": sum(fused.values()),
        "unused_reference_tensors": sum(available.values()),
        "candidate_bytes": sum(cand_sizes[k] * n for k, n in cand.items()),
        "matched_bytes": sum(cand_sizes[k] * n for k, n in (matched + fused).items()),
        "unmatched_candidate_shapes": sorted({str(k[0]) for k in unmatched})[:20],
    }))


if __name__ == "__main__":
    main()
