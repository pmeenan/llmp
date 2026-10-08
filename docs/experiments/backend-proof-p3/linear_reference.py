#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Upstream ExLlamaV3's output for every case of the per-linear sweep (BP-N5; cases.py).

Reference-only experiment tooling for the backend proof's P3; it does not implement llmpalooza
inference. Runs in the reference container (run_reference.sh) with upstream's extension built by
the SDK's NVCC 13.4.92 (the EXL3 bridge build), the SDK's cuBLAS 13.8.0.4 bind-mounted, a tuning
cache named by EXLLAMAV3_TUNE_CACHE, EXL3_HGEMM_F16ACC=0, and EXL3_GEMV=0 for EXL3-G or unset for
EXL3-O. It loads the fixture through upstream's Model API and, for each case, calls upstream's
extension exactly as upstream's modules do, one step at a time, hashing (SHA-256) every buffer a
step writes:

  packed  exl3_gemm(x, trellis, y, suh, a_had, svh, -1, mcg, mul1, 0), the call of
          BC_LinearEXL3 (which exl3_gemm sends to the GEMV kernel where EXL3-O's heuristic
          takes it), then the bias through upstream's add (add_kernel_hhh): stages a_had, y, out;
  multi   exl3_mgemm for gate and up as the gated MLP calls it: stages a_had (both slabs), y;
  recon   reconstruct_hgemm's steps: had_r_128 of x into xh; per slice of at most 32,768 columns
          the rotated weights (reconstruct / reconstruct_slice) and hgemm_recon; had_r_128 of the
          output in place; the bias with PyTorch's add: stages xh, w0.., gemm, y, out;
  fused   per slice reconstruct_had_slice and hgemm_recon of x; the bias: stages w0.., y, out.

Each case's final output is also computed by upstream's own module call (LinearEXL3.forward, or,
for the pair, the stepwise mgemm call itself, which is the MLP's), and the record says whether the
steps reproduce it bit for bit. Per linear it hashes the weights as loaded and the full
reconstructed weights, rotated (reconstruct) and fused (reconstruct_had_slice). With --profile it
also records the kernels each case's product launches (name, grid, block), as
../backend-proof-p0/exl3_launch_record.py does. Output: one JSON document; raw tensors are never
written.
"""

import argparse
import ctypes
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as sweep  # noqa: E402


def digest_file(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sha(tensor):
    torch.cuda.synchronize()
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def launches(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as trace:
        prof.export_chrome_trace(trace.name)
        events = json.loads(Path(trace.name).read_text())["traceEvents"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    return [{"name": e["name"], "grid": e["args"].get("grid"), "block": e["args"].get("block"),
             "shared_memory": e["args"].get("shared memory")} for e in kernels]


def out_dtype(linear):
    return linear.default_out_dtype


def packed(ext, linear, x):
    rows = x.shape[0]
    y = torch.empty((rows, linear.out_features), dtype=out_dtype(linear), device=x.device)
    a_had = torch.empty_like(x)
    tag = ext.exl3_gemm(x, linear.trellis, y, linear.suh, a_had, linear.svh, -1, linear.mcg, linear.mul1, 0)
    stages = {"a_had": sha(a_had), "y": sha(y)}
    out = y
    if linear.bias is not None:
        out = y.clone()
        ext.add(out, linear.bias, out)
    stages["out"] = sha(out)
    return stages, out, int(tag)


def multi(ext, mlp, x):
    mgu = mlp.multi_gu[0]
    rows = x.shape[0]
    xv = x.view(1, rows, x.shape[1])
    guh = torch.empty((2, rows, x.shape[1]), dtype=mlp.interm_dtype, device=x.device)
    gu = torch.empty((2, rows, mgu.out_features), dtype=mlp.interm_dtype, device=x.device)
    tag = ext.exl3_mgemm(xv, mgu.ptrs_trellis, gu, mgu.ptrs_suh, guh, mgu.ptrs_svh, None, None, mgu.K, -1,
                         mgu.mcg, mgu.mul1, -1, -1, 0, 1, None, None)
    # The MLP allocates the transformed-input scratch in its intermediate dtype (F32 here), but the
    # kernel writes F16 slabs (A_had + j * m * k halves): only those bytes are its output.
    written = guh.view(torch.uint8).reshape(-1)[:2 * rows * x.shape[1] * 2]
    return {"a_had": sha(written), "y": sha(gu), "out": sha(gu)}, gu, int(tag)


def reconstructed(ext, linear, x, fused):
    rows = x.shape[0]
    k, n = linear.in_features, linear.out_features
    y = torch.empty((rows, n), dtype=out_dtype(linear), device=x.device)
    stages = {}
    if fused:
        xh = x
    else:
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, linear.suh, None, 1.0)
        stages["xh"] = sha(xh)
    widths = sweep.slices(n)
    w_ = torch.empty((k * widths[0],), dtype=torch.half, device=x.device)
    start = 0
    for i, width in enumerate(widths):
        w = w_[:k * width].view(k, width)
        if fused:
            ext.reconstruct_had_slice(w, linear.trellis, linear.suh, linear.svh[start:], linear.K, linear.mcg,
                                      linear.mul1, start)
        elif start == 0 and width == n:
            ext.reconstruct(w, linear.trellis, linear.K, linear.mcg, linear.mul1)
        else:
            ext.reconstruct_slice(w, linear.trellis, linear.K, linear.mcg, linear.mul1, start)
        stages[f"w{i}"] = sha(w)
        ext.hgemm_recon(xh, w, y[:, start:start + width])
        start += width
    if not fused:
        stages["gemm"] = sha(y)
        ext.had_r_128(y, y, None, linear.svh, 1.0)
    stages["y"] = sha(y)
    out = y
    if linear.bias is not None:
        out = y.clone()
        out += linear.bias
    stages["out"] = sha(out)
    return stages, out


def weights(ext, linear):
    k, n = linear.in_features, linear.out_features
    record = {"k": k, "n": n, "K": linear.K, "mcg": bool(linear.mcg), "mul1": bool(linear.mul1),
              "out": "F32" if out_dtype(linear) == torch.float else "F16",
              "trellis": sha(linear.trellis), "suh": sha(linear.suh), "svh": sha(linear.svh),
              "bias": sha(linear.bias) if linear.bias is not None else None}
    w = torch.empty((k, n), dtype=torch.half, device=linear.trellis.device)
    ext.reconstruct(w, linear.trellis, linear.K, linear.mcg, linear.mul1)
    record["W"] = sha(w)
    ext.reconstruct_had_slice(w, linear.trellis, linear.suh, linear.svh, linear.K, linear.mcg, linear.mul1, 0)
    record["W_fused"] = sha(w)
    del w
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", choices=sweep.FIXTURES, required=True)
    parser.add_argument("--arm", choices=sweep.ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="record each product's kernel launches")
    parser.add_argument("--no-weights", action="store_true", help="skip the per-linear weight hashes")
    parser.add_argument("--only", help="run only cases whose id contains this")
    args = parser.parse_args()
    environment = {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "EXLLAMAV3_", "CUBLAS"))}
    if os.environ.get("EXL3_HGEMM_F16ACC") != "0" or "EXLLAMAV3_TUNE_CACHE" not in os.environ:
        raise SystemExit("set EXL3_HGEMM_F16ACC=0 and name the tuning cache with EXLLAMAV3_TUNE_CACHE")
    if (os.environ.get("EXL3_GEMV") == "0") != (args.arm == "G"):
        raise SystemExit("EXL3-G runs with EXL3_GEMV=0, EXL3-O without EXL3_GEMV")
    cache = Path(os.environ["EXLLAMAV3_TUNE_CACHE"])
    cache_before = digest_file(cache) if cache.exists() else None

    from exllamav3 import Config, Model
    from exllamav3.ext import exllamav3_ext as ext
    config = Config.from_directory(str(args.model))
    model = Model.from_config(config)
    model.load(device="cuda:0")
    device = torch.device("cuda:0")

    started = time.time()
    records = {"weights": {}, "cases": []}
    with torch.inference_mode():
        modules = {name: model.find_module(name).inner for name in sweep.linears()}
        if not args.no_weights:
            for name, linear in modules.items():
                records["weights"][name] = weights(ext, linear)
        for case in sweep.cases():
            if args.only and args.only not in case["id"]:
                continue
            first = modules[case["linears"][0]]
            x = torch.from_numpy(sweep.inputs(args.fixture, case["key"], case["rows"], first.in_features)).to(device)
            record = {"id": case["id"], "path": case["path"], "rows": case["rows"], "linears": case["linears"],
                      "x": sha(x)}
            if case["path"] == "packed":
                record["stages"], out, record["tag"] = packed(ext, first, x)
                reference = first.forward(x, {})
                product = lambda: packed(ext, first, x)  # noqa: E731
            elif case["path"] == "multi":
                mlp = model.find_module(case["key"].rsplit(".gate_up", 1)[0])
                record["stages"], out, record["tag"] = multi(ext, mlp, x)
                reference = None  # the stepwise call is the MLP's own
                product = lambda: multi(ext, mlp, x)  # noqa: E731
            else:
                fused = case["path"] == "fused"
                record["stages"], out = reconstructed(ext, first, x, fused)
                reference = first.forward(x, {})
                product = lambda: reconstructed(ext, first, x, fused)  # noqa: E731
            record["module_equal"] = None if reference is None else bool(
                torch.equal(out.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)))
            if args.profile:
                record["launches"] = launches(product)
            # A second call of the same steps repeats every stage.
            record["repeat_equal"] = product()[0] == record["stages"]
            records["cases"].append(record)
            print(record["id"], record["path"], record["module_equal"], record["repeat_equal"], flush=True)

    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if ("libcublas" in line or "exllamav3_ext" in line) and line.split()[-1].startswith("/")})
    versions = {p: ctypes.CDLL(p).cublasLtGetVersion() for p in mapped if "libcublasLt" in p}
    extension = [p for p in mapped if "exllamav3_ext" in p]
    records.update({
        "fixture": args.fixture, "arm": args.arm, "environment": environment,
        "mapped": mapped, "cublaslt_versions": versions,
        "library_sha256": {p: digest_file(Path(p)) for p in mapped},
        "extension": {p: digest_file(Path(p)) for p in extension},
        "tune_cache": {"before": cache_before, "after": digest_file(cache),
                       "bytes_base64": __import__("base64").b64encode(cache.read_bytes()).decode()},
        "torch": torch.__version__, "seconds": round(time.time() - started, 1),
    })
    args.output.write_text(json.dumps(records, indent=1) + "\n")
    print("DONE", len(records["cases"]), flush=True)


if __name__ == "__main__":
    main()
