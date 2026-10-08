#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Fetch a pinned Hugging Face snapshot; verify every byte, never run its code."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import urllib.request

BLOCK = 8 << 20


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def destination(root, relative):
    pure = PurePosixPath(relative)
    parts = pure.parts
    if (not parts or pure.is_absolute() or str(pure) != relative
            or any(p in ("", ".", "..") for p in parts)):
        raise ValueError(f"unsafe pinned path: {relative!r}")
    directory = root
    for part in parts[:-1]:
        directory = directory / part
        if directory.is_symlink():
            raise ValueError(f"symlink in destination: {directory}")
    return root.joinpath(*parts)


def verified(target, row):
    if target.is_symlink():
        raise ValueError(f"symlink destination: {target}")
    if not target.exists():
        return False
    if not target.is_file() or target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
        raise ValueError(f"existing file fails verification: {target}")
    return True


def fetch(url, target, row):
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if verified(target, row):
        return "present"
    fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".partial", dir=target.parent)
    partial = Path(name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "llmp-reference-fetch"})
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise ValueError(f"unexpected HTTP status {response.status}")
            count = 0
            h = hashlib.sha256()
            while block := response.read(min(BLOCK, row["bytes"] - count + 1)):
                count += len(block)
                if count > row["bytes"]:
                    raise ValueError("response exceeds pinned size")
                h.update(block)
                out.write(block)
            if count != row["bytes"] or h.hexdigest() != row["sha256"]:
                raise ValueError(f"size/hash mismatch: {row['path']}")
            out.flush()
            os.fsync(out.fileno())
        # Publish without replacing a file that appeared during the download.
        os.link(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return "fetched"


def verify_tree(root, rows):
    expected = {row["path"] for row in rows}
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink in snapshot: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra or missing:
        raise ValueError(f"snapshot differs from pins: extra={extra[:5]} missing={missing[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pins", type=Path, required=True, help="an experiment's pins.json")
    parser.add_argument("--section", default="model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--only", action="append", default=[],
                        help="fetch only pinned paths with this prefix; skips the exact-tree check")
    args = parser.parse_args()
    section = json.loads(args.pins.read_text())[args.section]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    rows = [r for r in section["files"] if not args.only or any(r["path"].startswith(p) for p in args.only)]
    base = f"https://huggingface.co/{section['repository']}/resolve/{section['revision']}/"

    def one(row):
        status = fetch(base + row["path"], destination(root, row["path"]), row)
        print(f"{status} {row['path']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        # Largest first so the long transfers overlap the small ones.
        for future in [pool.submit(one, r) for r in sorted(rows, key=lambda r: -r["bytes"])]:
            future.result()
    if not args.only:
        verify_tree(root, section["files"])
    print(f"verified {len(rows)} files, {sum(r['bytes'] for r in rows)} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
