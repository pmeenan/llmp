#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Record upstream's kernel launches for each BP-F2 kernel case.

External reference tooling for the backend proof's P0; it does not implement
llmpalooza inference. Runs in the reference container with a frozen tuning cache
(EXLLAMAV3_TUNE_CACHE) and the arm's EXL3_* settings. It builds the cases as
../exl3-reference/measure.py does (the protocol's real projections, or its
seeded synthetic trellises), warms each case, then profiles one invocation
with the PyTorch profiler and records every kernel launched: its name, grid,
block, shared memory (static plus dynamic, as the profiler reports it) and
registers per thread, in launch order. It also records the mapped cuBLAS
libraries (versions and SHA-256) and the extension's SHA-256. A
native kernel case is checked against this record before it is timed.
"""

import argparse
import ctypes
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def launches(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as trace:
        prof.export_chrome_trace(trace.name)
        events = json.loads(Path(trace.name).read_text())["traceEvents"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    return [{"name": e["name"], "grid": e["args"].get("grid"), "block": e["args"].get("block"),
             "shared_memory": e["args"].get("shared memory"), "registers": e["args"].get("registers per thread")}
            for e in kernels]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True, help="../exl3-reference/protocol.json")
    parser.add_argument("--mode", choices=("kernels", "synthetic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("EXLLAMAV3_TUNE_CACHE"):
        raise ValueError("name the frozen tuning cache with EXLLAMAV3_TUNE_CACHE")
    cache_before = digest(Path(os.environ["EXLLAMAV3_TUNE_CACHE"]))
    protocol = json.loads(args.protocol.read_text())
    torch.manual_seed(protocol["seed"])
    from exllamav3 import Config, Model
    from exllamav3.modules.quant.exl3 import LinearEXL3
    config = Config.from_directory(str(args.model))
    model = Model.from_config(config)
    model.load(device="cuda:0")
    torch.manual_seed(protocol["seed"])
    if args.mode == "synthetic":
        specs = [(f"synthetic-{k}-{n}-K{rate}", k, n, rate)
                 for k, n in protocol["kernel_synthetic_shapes"] for rate in protocol["kernel_synthetic_rates"]]
    else:
        specs = [(name, None, None, None) for name in protocol["kernel_real_keys"]]
    records = []
    with torch.inference_mode():
        for name, k, n, rate in specs:
            if args.mode == "synthetic":
                trellis = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * rate), dtype=torch.int16, device="cuda")
                suh = (torch.randint(0, 2, (k,), device="cuda") * 2 - 1).half() / np.sqrt(k)
                svh = (torch.randint(0, 2, (n,), device="cuda") * 2 - 1).half()
                linear = LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis,
                                    mcg=torch.tensor(0, dtype=torch.int32, device="cuda"), key=name)
            else:
                linear = model.find_module(name).inner
            for rows in protocol["kernel_rows"]:
                x = torch.randn((1, rows, linear.in_features), device="cuda", dtype=torch.float16) * .1
                for _ in range(protocol["warmup_calls"]):
                    linear.forward(x, {})
                torch.cuda.synchronize()
                records.append({"set": args.mode, "name": name, "rows": rows,
                                "launches": launches(lambda: linear.forward(x, {}))})
            del linear
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if ("libcublas" in line or "exllamav3_ext" in line) and line.split()[-1].startswith("/")})
    versions = {p: ctypes.CDLL(p).cublasLtGetVersion() for p in mapped if "libcublasLt" in p}
    for p in mapped:
        if "libcublas.so" in p:
            parts = []
            for prop in (0, 1, 2):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
                value = ctypes.c_int()
                ctypes.CDLL(p).cublasGetProperty(prop, ctypes.byref(value))
                parts.append(value.value)
            versions[p] = ".".join(map(str, parts))
    extension = [p for p in mapped if "exllamav3_ext" in p]
    library_sha256 = {p: digest(Path(p)) for p in mapped if "libcublas" in p}
    cache_after = digest(Path(os.environ["EXLLAMAV3_TUNE_CACHE"]))
    args.output.write_text(json.dumps({"environment": {k: v for k, v in os.environ.items()
                                                       if k.startswith(("EXL3_", "EXLLAMAV3_"))},
                                       "cublas_mapped": mapped, "cublas_versions": versions,
                                       "cublas_sha256": library_sha256,
                                       "extension": {p: digest(Path(p)) for p in extension},
                                       "tune_cache_sha256": {"before": cache_before, "after": cache_after},
                                       "cases": records}, indent=1) + "\n")
    print("DONE", len(records), flush=True)


if __name__ == "__main__":
    main()
