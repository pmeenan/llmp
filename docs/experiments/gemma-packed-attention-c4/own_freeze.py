#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze same-policy full heads and initialized owner states before oracle use."""
import hashlib
import json
import os
import pathlib
import sys

INPUT_SHA = "b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610"
HEAD_BYTES = 32 * 4 * 262144 * 4


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(root):
    pre = root / "pre-native.json"
    identity = json.loads(pre.read_text())
    for item in identity["source_files"]:
        if digest(pathlib.Path(item["path"])) != item["sha256"]:
            raise ValueError("source changed after prefreeze")
    for key in ("native_binary", "build_receipt"):
        item = identity[key]
        if digest(pathlib.Path(item["path"])) != item["sha256"]:
            raise ValueError("compiled identity changed after prefreeze")
    result = {"pre_native_sha256": digest(pre), "same_policy_own_noise": 0,
              "steps": 32, "owners": 4, "full_head_bytes": HEAD_BYTES, "policies": {}}
    for policy in ("control", "candidate"):
        records = []
        for repeat in ("first", "repeat"):
            directory = root / f"{policy}-{repeat}"
            if digest(directory / "inputs.i32") != INPUT_SHA:
                raise ValueError("input identity changed")
            files = {"heads.f32": HEAD_BYTES}
            for owner in range(4):
                files[f"owner-{owner}.state"] = None
            hashes = {}
            for name, expected in files.items():
                path = directory / name
                size = path.stat().st_size
                if expected is not None and size != expected or size <= 0:
                    raise ValueError("incomplete head/state witness")
                hashes[name] = {"sha256": digest(path), "bytes": size}
            records.append(hashes)
        if records[0] != records[1]:
            raise ValueError("same-policy complete outputs/state differ")
        result["policies"][policy] = records[0]
    out = root / "native-own-frozen.json"
    with out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    out.chmod(0o400)
    print(json.dumps({"freeze_sha256": digest(out), **result}))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("own_freeze.py ROOT")
    freeze(pathlib.Path(sys.argv[1]))
