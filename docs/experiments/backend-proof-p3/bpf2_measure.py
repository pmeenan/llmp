#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""BP-F2's reference arm (pre-registered at P3 entry; docs/backend-proof.md, "Performance protocol").

External reference harness in the M0 reference container; it implements nothing of llmpalooza. It is
M0's measure.py (../exl3-reference/measure.py, imported unchanged: its model loading, manifest,
timing and checks) with its kernel modes over BP-F2's case set instead of M0's:

  BPF2_CASES=bpf2-cases.txt BPF2_SET=real-40|real-45|synthetic \\
    bpf2_measure.py --model MODEL --mode kernels|synthetic --protocol P --pins PINS --output DIR

It takes measure.py's arguments, so timing_session.sh runs it in measure.py's place
(MEASURE=/harness/.../bpf2_measure.py). For the cases of BPF2_SET (bpf2_cases.py's lines
SET|NAME|ROWS|KIND), in file order, it times exactly as measure.py does (the protocol's warm-up
calls, a CUDA graph of ten invocations bracketed by events, 31 replays; the repeat and the capture
checked bit for bit against the eager call) and writes kernels.json in measure.py's format, so
timing_stats.py and timing_protocol.py read it unchanged. What it times per KIND:
- `linear`: upstream's LinearEXL3.forward under the protocol's profile (EXL3-O: upstream's own
  choice of GEMV, GEMM or reconstruction), except that on the reconstruction path the bias add is
  ExLlamaV3's add kernel (ext.add, add_kernel_hhh) where upstream's module adds with PyTorch: the
  native plan's owner, as the operation plan's probe (../backend-proof-p0/exl3_op_plan.py) moved it;
- `multi`: layer 0's gate and up through ext.exl3_mgemm, called as the gated MLP calls it for up to
  32 rows (P3 part 1's linear_reference.py `multi`).
The synthetic set's linears are measure.py's (random trellis and signs from the protocol's seed).
"""

import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/experiment/experiment")
import measure  # noqa: E402  (M0's harness, unchanged)
from checks import compare, summary  # noqa: E402


def load_cases():
    wanted = os.environ["BPF2_SET"]
    cases = []
    for line in Path(os.environ["BPF2_CASES"]).read_text().splitlines():
        set_name, name, rows, kind = line.split("|")
        if set_name == wanted:
            cases.append((name, int(rows), kind))
    if not cases:
        raise ValueError(f"no BP-F2 cases for {wanted}")
    return cases


def install_bias_add(ext):
    """The reconstruction path's bias through ExLlamaV3's add kernel, as the native plan owns it."""
    from exllamav3.modules.quant.exl3 import LinearEXL3
    recon = LinearEXL3.reconstruct_hgemm

    def recon_bias(self, x, out_dtype):
        bias = self.bias
        if bias is None:
            return recon(self, x, out_dtype)
        self.bias = None
        try:
            y = recon(self, x, out_dtype)
        finally:
            self.bias = bias
        ext.add(y, bias, y)
        return y
    LinearEXL3.reconstruct_hgemm = recon_bias


def kernels(model, protocol, output, synthetic):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3 import LinearEXL3
    install_bias_add(ext)
    torch.manual_seed(protocol["seed"])
    rows_out = []
    linears = {}
    for name, rows, kind in load_cases():
        if kind == "multi":
            mlp = model.find_module(name.removesuffix(".gate_up"))
            mgu = mlp.multi_gu[0]
            k, n, rate = mgu.in_features, mgu.out_features, mgu.K
        elif synthetic:
            if name not in linears:
                _, k_, n_, rate_ = name.split("-")
                k, n, rate = int(k_), int(n_), int(rate_[1:])
                trellis = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * rate), dtype=torch.int16,
                                        device="cuda")
                suh = (torch.randint(0, 2, (k,), device="cuda") * 2 - 1).half() / np.sqrt(k)
                svh = (torch.randint(0, 2, (n,), device="cuda") * 2 - 1).half()
                linears[name] = LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis,
                                           mcg=torch.tensor(0, dtype=torch.int32, device="cuda"), key=name)
            linear = linears[name]
            k, n, rate = linear.in_features, linear.out_features, linear.K
        else:
            linear = model.find_module(name).inner
            k, n, rate = linear.in_features, linear.out_features, linear.K
        if kind == "multi":
            x = torch.randn((1, rows, k), device="cuda", dtype=torch.float16) * .1
            guh = torch.empty((2, rows, k), dtype=mlp.interm_dtype, device="cuda")
            gu = torch.empty((2, rows, n), dtype=mlp.interm_dtype, device="cuda")

            def operation():
                ext.exl3_mgemm(x, mgu.ptrs_trellis, gu, mgu.ptrs_suh, guh, mgu.ptrs_svh, None, None, mgu.K, -1,
                               mgu.mcg, mgu.mul1, -1, -1, 0, 1, None, None)
                return gu
            path = "multi"
        else:
            x = torch.randn((1, rows, k), device="cuda", dtype=torch.float16) * .1

            def operation():
                return linear.forward(x, {})
            path = "packed" if rows <= 144 else ("fused_reconstruct" if rows >= 1024 else "reconstruct")
        for _ in range(protocol["warmup_calls"]):
            operation()
        torch.cuda.synchronize()
        eager = operation().float().cpu().numpy().copy()
        same = compare(eager, operation().float().cpu().numpy())
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        graph = torch.cuda.CUDAGraph()
        invocations = protocol["kernel_replays_per_sample"]
        a = torch.cuda.Event(enable_timing=True, external=True)
        b = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            a.record()
            for invocation in range(invocations):
                value = operation()
                if invocation == invocations - 1:
                    captured = value
                del value
            b.record()
        for _ in range(protocol["warmup_calls"]):
            graph.replay()
        torch.cuda.synchronize()
        capture_check = compare(eager, captured.float().cpu().numpy())
        elapsed = []
        for _ in range(protocol["kernel_repeats"]):
            graph.replay()
            b.synchronize()
            elapsed.append(a.elapsed_time(b) * 1000 / invocations)
        trace = measure.profile_cuda(graph.replay)
        trace["profiled_invocations"] = invocations
        trace["active_device_us"] /= invocations
        rows_out.append({"name": name, "k": k, "n": n, "K": rate, "rows": rows, "kind": kind, "path": path,
                         "repeat": same, "capture_check": capture_check, "graph_replay_us": elapsed,
                         "summary": summary(elapsed), "profile": trace, "memory": measure.memory()})
        measure.write_json(output / "kernels.json", rows_out)
        print("kernel", name, rows, rows_out[-1]["summary"]["median"], flush=True)
        del graph, captured, x
        gc.collect()
    return rows_out


measure.kernels = kernels

if __name__ == "__main__":
    with torch.inference_mode():
        measure.main()
