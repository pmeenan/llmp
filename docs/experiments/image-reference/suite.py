# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded resident-pipeline matrix; model load occurs once, not per case."""
import gc
import json
import os
from pathlib import Path
import sys

import measure


def main():
    model, output = sys.argv[1:]
    root = Path(output)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    retained = {}
    cases = [
        ("plain-512-4", ["--size", "512", "--steps", "4", "--plain"]),
        ("instrumented-512-4", ["--size", "512", "--steps", "4"]),
        ("plain-repeat-512-4", ["--size", "512", "--steps", "4", "--plain"]),
        ("no-cache-512-4", ["--size", "512", "--steps", "4", "--no-cache"]),
        ("t2i-1024-4", ["--size", "1024", "--steps", "4"]),
        ("t2i-1024-40", ["--size", "1024", "--steps", "40"]),
        ("t2i-2048-40", ["--size", "2048", "--steps", "40"]),
        ("edit-512-4", ["--size", "512", "--steps", "4", "--edit"]),
        ("edit-repeat-512-4", ["--size", "512", "--steps", "4", "--edit"]),
        ("cancel-512-4", ["--size", "512", "--steps", "4", "--cancel-after", "1"]),
        ("after-cancel-512-4", ["--size", "512", "--steps", "4", "--plain"]),
        ("phase-release-512-4", ["--size", "512", "--steps", "4", "--release-phases"]),
    ]
    results = {}
    try:
        for name, arguments in cases:
            results[name] = measure.main([model, str(root / name), *arguments], retained)
        baseline = results["plain-512-4"]["output"]["pixels_sha256"]
        for name in ("instrumented-512-4", "plain-repeat-512-4", "after-cancel-512-4", "phase-release-512-4"):
            if results[name]["output"]["pixels_sha256"] != baseline:
                raise RuntimeError(f"Fixed-configuration pixel mismatch: {name}")
        if results["edit-512-4"]["output"] != results["edit-repeat-512-4"]["output"]:
            raise RuntimeError("Repeated editing output differs")
        (root / "checks.json").write_text(json.dumps({"fixed_configuration_pixels_match": True,
            "repeated_edit_matches": True, "after_cancel_matches": True,
            "phase_release_matches": True}, indent=2) + "\n")
    finally:
        import torch
        retained.clear()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        (root / "release.json").write_text(json.dumps({"cuda_allocated": torch.cuda.memory_allocated(),
            "cuda_reserved": torch.cuda.memory_reserved()}) + "\n")


if __name__ == "__main__":
    os.umask(0o077)
    main()
