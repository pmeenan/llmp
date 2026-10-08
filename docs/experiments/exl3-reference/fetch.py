#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Fetch only the pinned EXL3 fixture data; never import checkpoint code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_file(url, target, size, expected):
    if target.is_symlink():
        raise ValueError(f"symlink destination: {target}")
    if target.exists():
        if target.stat().st_size == size and digest(target) == expected:
            return
        raise ValueError(f"existing file fails verification: {target}")
    fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".partial", dir=target.parent)
    partial = Path(name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=120) as response:
            if response.status != 200:
                raise ValueError(f"unexpected HTTP status {response.status}")
            count = 0
            h = hashlib.sha256()
            while block := response.read(min(8 << 20, size - count + 1)):
                count += len(block)
                if count > size:
                    raise ValueError("response exceeds pinned size")
                h.update(block)
                out.write(block)
            if count != size or h.hexdigest() != expected:
                raise ValueError(f"size/hash mismatch: {target.name}")
            out.flush()
            os.fsync(out.fileno())
        # Publish without replacing a file that appeared during the download.
        os.link(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pins", type=Path, default=Path(__file__).with_name("pins.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    pins = json.loads(args.pins.read_text())
    receipts = []
    for fixture in pins["fixtures"]:
        label = fixture["repository"].rsplit("-", 1)[1]
        folder = args.output / label
        folder.mkdir(mode=0o700, exist_ok=True)
        if folder.is_symlink():
            raise ValueError("symlink fixture directory")
        files = fixture["metadata_files_verified"] + [{
            "filename": fixture["filename"],
            "size_bytes": fixture["published_size_bytes"],
            "sha256": fixture["published_sha256"],
        }]
        for row in files:
            name = row["filename"]
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("fixture names must be single path components")
            url = f"https://huggingface.co/{fixture['repository']}/resolve/{fixture['revision']}/{name}?download=true"
            fetch_file(url, folder / name, row["size_bytes"], row["sha256"])
            receipts.append({"fixture": label, **row})
            print(f"verified {label}/{name}", flush=True)
    (args.output / "verified-files.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
