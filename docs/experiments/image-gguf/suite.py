#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the bounded GGUF image suite sequentially; one container per case.

Each denoiser's files are hash-verified by its first successful case; later
cases in the same session skip re-verification. Cases never overlap, so host memory observations are
attributable to one case.
"""

import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

# name, denoiser, placement, size, steps, extra run.py flags
CASES = [
    ("q4-1024-4-cold", "q4_k_m", "resident", 1024, 4, ["--cold"]),
    ("q4-1024-4-a", "q4_k_m", "resident", 1024, 4, []),
    ("q4-1024-4-b", "q4_k_m", "resident", 1024, 4, []),
    ("q4-512-4", "q4_k_m", "resident", 512, 4, []),
    ("q4-1024-40", "q4_k_m", "resident", 1024, 40, []),
    ("q4-2048-40", "q4_k_m", "resident", 2048, 40, []),
    ("q8-512-4", "q8_0", "resident", 512, 4, []),
    ("q8-1024-4", "q8_0", "resident", 1024, 4, []),
    ("q8-1024-40", "q8_0", "resident", 1024, 40, []),
    ("q8-2048-40", "q8_0", "resident", 2048, 40, []),
    ("bf16-512-4", "bf16", "resident", 512, 4, []),
    ("bf16-1024-4-a", "bf16", "resident", 1024, 4, []),
    ("bf16-1024-4-b", "bf16", "resident", 1024, 4, []),
    ("bf16-1024-40", "bf16", "resident", 1024, 40, []),
    ("bf16-2048-40", "bf16", "resident", 2048, 40, []),
    # Release the text encoder and VAE outside their phases; denoiser resident.
    ("q4-1024-40-phase", "q4_k_m", "phase-disk", 1024, 40, []),
    # Disk-backed parameters: unconstrained, then a managed budget below the
    # 4.6 GB denoiser so each step re-stages segments; warm, then cold.
    ("q4-1024-4-disk", "q4_k_m", "all-disk", 1024, 4, []),
    ("q4-1024-4-disk-3g", "q4_k_m", "all-disk", 1024, 4, ["--max-vram", "cuda0=3"]),
    ("q4-1024-4-disk-3g-cold", "q4_k_m", "all-disk", 1024, 4, ["--max-vram", "cuda0=3", "--cold"]),
    ("q4-1024-4-disk-2g", "q4_k_m", "all-disk", 1024, 4, ["--max-vram", "cuda0=2"]),
    # Control: resident, but with the spatial VAE tiling sd.cpp fell back to
    # under the 3 GiB budget (32x32 latent tiles, default 0.5 overlap).
    ("q4-1024-4-vae-tiled", "q4_k_m", "resident", 1024, 4,
     ["--extra=--vae-tiling", "--extra=--vae-tile-size", "--extra=32x32"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private suite directory")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    failures = []
    verified = set()  # denoisers whose files (with shared te/vae) passed hash checks
    for name, denoiser, placement, size, steps, extra in CASES:
        if args.only and name not in args.only:
            continue
        command = [sys.executable, str(HERE / "run.py"), "--models", str(args.models),
                   "--output", str(args.output / name), "--denoiser", denoiser, "--placement", placement,
                   "--size", str(size), "--steps", str(steps), *extra]
        if denoiser in verified:
            command.append("--skip-verify")
        print(f"== {name}", flush=True)
        code = subprocess.run(command).returncode
        if code == 0:
            verified.add(denoiser)
        if code:
            failures.append(name)
    print(f"failures: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
