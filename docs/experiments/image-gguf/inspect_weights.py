#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Inventory the pinned denoiser representations by tensor type and block.

Run inside the pinned image-reference container (GGUF reader from
/app/gguf-py) after hash verification. Reads headers and tensor metadata only;
this is an experiment inspector, not llmpalooza's untrusted-checkpoint validator.
"""

import collections
import json
from pathlib import Path
import re
import struct
import sys

from gguf import GGUFReader

BLOCK = re.compile(r"(?:^|\.)(?:transformer_blocks|blocks?|blk)\.(\d+)\.")


def safetensors_inventory(path):
    with open(path, "rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        if length > 64 << 20:
            raise ValueError("implausible safetensors header")
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    rows = []
    for name, info in header.items():
        start, end = info["data_offsets"]
        rows.append((name, info["dtype"], end - start, info["shape"]))
    return rows, {"header_bytes": 8 + length}


def gguf_inventory(path):
    reader = GGUFReader(path, mode="r")
    rows = [(t.name, t.tensor_type.name, int(t.n_bytes), [int(n) for n in t.shape]) for t in reader.tensors]
    metadata = {k: f.contents() for k, f in reader.fields.items()
                if k.startswith("general.") and not k.startswith("general.tags")}
    return rows, {"metadata": metadata, "data_offset": int(reader.data_offset)}


def summarize(path):
    rows, extra = gguf_inventory(path) if path.suffix == ".gguf" else safetensors_inventory(path)
    by_type = collections.Counter()
    count_by_type = collections.Counter()
    blocks = collections.defaultdict(int)
    outside = 0
    for name, dtype, size, _ in rows:
        by_type[dtype] += size
        count_by_type[dtype] += 1
        match = BLOCK.search(name)
        if match:
            blocks[int(match[1])] += size
        else:
            outside += size
    block_sizes = sorted(blocks.values())
    return {
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "tensors": len(rows),
        "tensor_bytes": sum(r[2] for r in rows),
        "bytes_by_type": dict(by_type.most_common()),
        "tensors_by_type": dict(count_by_type.most_common()),
        "blocks": len(blocks),
        "block_bytes_min_max": [block_sizes[0], block_sizes[-1]] if block_sizes else None,
        "non_block_bytes": outside,
        **extra,
    }


if __name__ == "__main__":
    print(json.dumps([summarize(Path(p)) for p in sys.argv[1:]], indent=2, default=str))
