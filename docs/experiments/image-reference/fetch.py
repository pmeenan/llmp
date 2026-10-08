# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Fetch the fixed reference dataset; never execute checkpoint code."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def verify(path, item):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]


def verify_directory(root, pins):
    expected_files = {item["path"] for item in pins["files"]}
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in model directory: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise ValueError("Model directory must contain exactly the pinned files")
    for item in pins["files"]:
        if not verify(root / item["path"], item):
            raise ValueError(f"Unverified input: {item['path']}")


def main():
    pins = json.loads(Path(__file__).with_name("pins.json").read_text())
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for item in pins["files"]:
        path = root / item["path"]
        if not path.resolve().is_relative_to(root):
            raise ValueError("Path escapes destination")
        path.parent.mkdir(parents=True, exist_ok=True)
        if verify(path, item):
            continue
        url = (f"https://huggingface.co/{pins['model']}/resolve/"
               f"{pins['revision']}/{item['path']}")
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".fetch-", delete=False) as stream:
            partial = Path(stream.name)
            try:
                subprocess.run(["curl", "--fail", "--location", url], stdout=stream, check=True)
                stream.flush()
                if not verify(partial, item):
                    raise ValueError(f"Integrity mismatch: {item['path']}")
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        print("Verified", item["path"], flush=True)


if __name__ == "__main__":
    main()
