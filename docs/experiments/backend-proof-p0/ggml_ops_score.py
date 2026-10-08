#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Score upstream's and GGML's non-linear operations against FP64.

External reference tooling for the backend proof's P0; it does not implement
llmpalooza inference. Input: ggml_ops_capture.py directories. For every captured
operation it computes the exact result in FP64 from the captured inputs and
compares with it (a) upstream's output as captured and (b) each candidate
GGML path, run through the reference shim (ggml_ops.py) on the same inputs:

- rms_norm: rms_norm and mul fused (one graph) or unfused, F32 output and
  after GGML's F32-to-F16 copy (what an FP16 linear input receives);
- add: the fused residual add of post_attention_layernorm (bit equality with
  upstream, and error);
- rope: NEOX RoPE computed on F16 data (F16 out), or in F32 (F32 out, and
  F32 rounded to F16 by GGML's copy);
- attention: every ggml_ops.PATHS entry, F32 output and rounded to F16, per
  query position; plus rope_attention, attention whose q comes from each
  RoPE candidate applied to the captured pre-RoPE q, against FP64 RoPE then
  FP64 attention (K/V as captured);
- swiglu: GGML's swiglu on the FP32 gate and up, F16 and F32 output;
- get_rows: the embedding lookup (bit equality with upstream).

Statistics per case: largest absolute error, RMS error, and RMS error
relative to the FP64 result's RMS; for attention also each query row's
relative RMS error. They are aggregated per fixture, operation, candidate
and phase (each prefix's prefill; its single-token steps 0 and 15) over the
captured layers: the maximum of max_abs and of the relative errors, and the
RMS of the RMS errors. Operation scratch per call is recorded too: GGML's
pool peak and materialized intermediates (ggml_shim.cu), upstream's PyTorch
allocation peak above its start less its output (attention only).

--hashes writes the SHA-256 of every GGML output, for comparing two builds of
the shim; --compare-hashes checks two such files for equality.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ggml_ops import PATHS, GGMLOps

THETA, HEAD_DIM = 1e6, 64


def fp64(a, device="cuda"):
    return torch.from_numpy(np.asarray(a)).to(device).double()


def stats(cand, ref, rows=False):
    err = cand.double() - ref
    out = {"max_abs": err.abs().max().item(), "rms": err.pow(2).mean().sqrt().item(),
           "ref_rms": ref.pow(2).mean().sqrt().item()}
    out["rel_rms"] = out["rms"] / out["ref_rms"]
    if rows:
        e = err.reshape(err.shape[0], -1)
        r = ref.reshape(ref.shape[0], -1)
        row = e.pow(2).mean(1).sqrt() / r.pow(2).mean(1).sqrt()
        out["row_rel_rms"] = row.cpu().numpy()
    return out


def rope64(x, pos0):
    inv = THETA ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float64, device=x.device) / HEAD_DIM)
    ang = (torch.arange(x.shape[0], dtype=torch.float64, device=x.device) + pos0)[:, None] * inv[None]
    cos = torch.cat((ang.cos(), ang.cos()), -1)[:, None]
    sin = torch.cat((ang.sin(), ang.sin()), -1)[:, None]
    half = HEAD_DIM // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


def attn64(q, k, v, past, scale):
    n, heads = q.shape[0], q.shape[1]
    group = heads // k.shape[1]
    kk = k.repeat_interleave(group, 1).transpose(0, 1)
    vv = v.repeat_interleave(group, 1).transpose(0, 1)
    s = q.transpose(0, 1) @ kk.transpose(1, 2) * scale
    mask = torch.arange(k.shape[0], device=q.device)[None] > (torch.arange(n, device=q.device)[:, None] + past)
    s = s.masked_fill(mask, float("-inf"))
    return (torch.softmax(s, -1) @ vv).transpose(0, 1)


class Scorer:
    def __init__(self, ops):
        self.g = ops
        self.cases = defaultdict(list)  # (fixture, op, candidate, phase) -> [stats]
        self.hashes = {}
        self.checks = defaultdict(lambda: {"cases": 0, "equal": 0})
        self.memory = defaultdict(list)  # (op, candidate, phase) -> [(pool, intermediates)]
        self.routes = {}

    def add(self, fixture, op, cand, phase, s, key=None, tensor=None):
        self.cases[(fixture, op, cand, phase)].append(s)
        if tensor is not None:
            self.hashes[f"{fixture}/{phase}/{key}/{op}/{cand}"] = hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()

    def check(self, name, equal):
        self.checks[name]["cases"] += 1
        self.checks[name]["equal"] += int(bool(equal))

    def scratch(self, op, cand, phase):
        self.memory[(op, cand, phase)].append(self.g.last_scratch())

    def norm(self, fixture, phase, key, d, kind):
        x = torch.from_numpy(d[f"{key}.x"]).cuda().float()
        if f"{key}.residual_before" in d:
            before = torch.from_numpy(d[f"{key}.residual_before"]).cuda().float()
            after = torch.from_numpy(d[f"{key}.residual_after"]).cuda().float()
            summed = self.g.add(before, x)
            self.check("add: GGML equals upstream's fused residual add", torch.equal(summed, after.reshape(summed.shape)))
            ref_add = fp64(d[f"{key}.residual_before"]) + fp64(d[f"{key}.x"])
            self.add(fixture, "add", "upstream", phase, stats(after, ref_add))
            self.add(fixture, "add", "ggml", phase, stats(summed.reshape(after.shape), ref_add), key, summed)
            x = after
        x = x.reshape(-1, x.shape[-1])
        w = torch.from_numpy(d[f"{key}.weight"]).cuda()
        eps = float(d[f"{key}.eps"])
        x64 = x.double()
        ref = x64 / torch.sqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * w.double()
        y = torch.from_numpy(d[f"{key}.y"]).cuda()
        self.add(fixture, "rms_norm", "upstream", phase, stats(y.reshape(ref.shape), ref))
        outs = {}
        for fused in (True, False):
            for dtype in (torch.float32, torch.float16):
                name = f"ggml_{'fused' if fused else 'unfused'}_{'f32' if dtype == torch.float32 else 'f16'}"
                outs[name] = self.g.rms_norm(x, w.float(), eps, dtype, fused)
                self.scratch("rms_norm", name, kind)
                self.add(fixture, "rms_norm", name, phase, stats(outs[name], ref), key, outs[name])
        self.check("rms_norm: GGML (F16 output) equals upstream", torch.equal(outs["ggml_fused_f16"], y.reshape(ref.shape)))
        self.check("rms_norm: fused equals unfused (F32)", torch.equal(outs["ggml_fused_f32"], outs["ggml_unfused_f32"]))
        self.check("rms_norm: F16 output equals F32 output rounded", torch.equal(outs["ggml_fused_f16"], outs["ggml_fused_f32"].half()))

    def rope(self, fixture, phase, key, d, kind):
        pos = int(d[f"{key}.position"])
        out = {}
        for part in ("q", "k"):
            before = torch.from_numpy(d[f"{key}.{part}_before"]).cuda()[0]
            after = torch.from_numpy(d[f"{key}.{part}_after"]).cuda()[0]
            ref = rope64(before.double(), pos)
            self.add(fixture, "rope", "upstream", phase, stats(after, ref))
            c = {"ggml_f16": self.g.rope_neox(before, pos, THETA, torch.float16, torch.float16),
                 "ggml_f32": self.g.rope_neox(before, pos, THETA, torch.float32, torch.float32),
                 "ggml_f32_to_f16": self.g.rope_neox(before, pos, THETA, torch.float32, torch.float16)}
            for name, t in c.items():
                self.scratch("rope", name, kind)
                self.add(fixture, "rope", name, phase, stats(t, ref), f"{key}.{part}", t)
            self.check("rope: GGML (F16) equals upstream", torch.equal(c["ggml_f16"], after))
            self.check("rope: F16 compute equals F32 compute rounded to F16", torch.equal(c["ggml_f16"], c["ggml_f32_to_f16"]))
            out[part] = c
        return out

    def attention(self, fixture, phase, key, d, kind, rope_key=None):
        q = torch.from_numpy(d[f"{key}.q"]).cuda()
        k = torch.from_numpy(d[f"{key}.k_cache"]).cuda().contiguous()
        v = torch.from_numpy(d[f"{key}.v_cache"]).cuda().contiguous()
        past, scale = int(d[f"{key}.past"]), float(d[f"{key}.scale"])
        n_kv = k.shape[0]
        assert n_kv == past + q.shape[0]
        ref = attn64(q.double(), k.double(), v.double(), past, scale)
        o = torch.from_numpy(d[f"{key}.o"]).cuda()
        self.add(fixture, "attention", "upstream", phase, stats(o, ref, rows=True))
        self.memory[("attention", "upstream", kind)].append(
            (int(d[f"{key}.torch_peak_bytes"]) - int(d[f"{key}.output_bytes"]), 0))
        self.routes[f"n_q={q.shape[0]} n_kv={n_kv}"] = self.g.nonfa_route(q.shape[0], n_kv, k.shape[1], k.shape[2])
        for path in PATHS:
            t = self.g.attention(q, k, v, n_kv, past, scale, path, torch.float32)
            self.scratch("attention", path, kind)
            self.add(fixture, "attention", f"{path}_f32", phase, stats(t, ref, rows=True), key, t)
            self.add(fixture, "attention", f"{path}_f16", phase, stats(t.half(), ref, rows=True))
            self.check(f"attention: GGML {path} (F16 output) equals upstream", torch.equal(t.half(), o))
            if phase.startswith("prefill 32"):
                t16 = self.g.attention(q, k, v, n_kv, past, scale, path, torch.float16)
                self.check("attention: F16 output equals F32 output rounded", torch.equal(t16, t.half()))
        if rope_key is not None:
            q_before = torch.from_numpy(d[f"{rope_key}.q_before"]).cuda()[0]
            ref = attn64(rope64(q_before.double(), past), k.double(), v.double(), past, scale)
            self.add(fixture, "rope_attention", "upstream", phase, stats(o, ref, rows=True))
            for rope_name, dtype in (("rope_f16", torch.float16), ("rope_f32", torch.float32)):
                qr = self.g.rope_neox(q_before, past, THETA, dtype, dtype)
                for path in ("fa_auto", "fa_vec", "fa_tile", "nonfa_llama", "nonfa_f32_notf32"):
                    t = self.g.attention(qr, k, v, n_kv, past, scale, path, torch.float32)
                    self.add(fixture, "rope_attention", f"{rope_name}+{path}_f16", phase, stats(t.half(), ref, rows=True))

    def swiglu(self, fixture, phase, key, d, kind):
        g = torch.from_numpy(d[f"{key}.gate"]).cuda()
        u = torch.from_numpy(d[f"{key}.up"]).cuda()
        g64, u64 = g.double(), u.double()
        ref = g64 / (1 + torch.exp(-g64)) * u64
        a = torch.from_numpy(d[f"{key}.out"]).cuda()
        self.add(fixture, "swiglu", "upstream", phase, stats(a, ref))
        for name, dtype in (("ggml_f32_to_f16", torch.float16), ("ggml_f32", torch.float32)):
            t = self.g.swiglu(g, u, dtype)
            self.scratch("swiglu", name, kind)
            self.add(fixture, "swiglu", name, phase, stats(t, ref), key, t)
            if dtype == torch.float16:
                self.check("swiglu: GGML (F16 output) equals upstream", torch.equal(t, a))

    def aggregate(self):
        out = {}
        for (fixture, op, cand, phase), cases in sorted(self.cases.items()):
            agg = {"cases": len(cases), "max_abs": max(c["max_abs"] for c in cases),
                   "rms": math.sqrt(sum(c["rms"] ** 2 for c in cases) / len(cases)),
                   "rel_rms_max": max(c["rel_rms"] for c in cases),
                   "rel_rms_mean": sum(c["rel_rms"] for c in cases) / len(cases)}
            if "row_rel_rms" in cases[0]:
                rows = np.concatenate([c["row_rel_rms"] for c in cases])
                agg["row_rel_rms_max"] = float(rows.max())
                agg["row_rel_rms_quantiles"] = {q: float(np.quantile(rows, float(q))) for q in ("0.5", "0.9", "0.99")}
            out.setdefault(op, {}).setdefault(cand, {}).setdefault(fixture, {})[phase] = agg
        # Each candidate's relative RMS error against upstream's, per fixture and phase.
        ratios = {}
        for op, cands in out.items():
            if "upstream" not in cands:
                continue
            for cand, fixtures in cands.items():
                if cand == "upstream":
                    continue
                r = [fixtures[f][p]["rel_rms_max"] / cands["upstream"][f][p]["rel_rms_max"]
                     for f in fixtures for p in fixtures[f] if cands["upstream"][f][p]["rel_rms_max"] > 0]
                ratios.setdefault(op, {})[cand] = {"min": min(r), "median": float(np.median(r)), "max": max(r)}
        memory = {}
        for (op, cand, kind), values in sorted(self.memory.items()):
            memory.setdefault(op, {}).setdefault(cand, {})[kind] = {
                "pool_peak_bytes_max": max(v[0] for v in values), "intermediate_bytes_max": max(v[1] for v in values)}
        return out, ratios, memory


def phase_of(tag, prefix):
    return f"prefill {prefix}" if tag == "prefill" else f"steps after {prefix}"


def kind_of(tag, prefix):
    return f"prefill {prefix}" if tag == "prefill" else f"step at {prefix + int(tag[4:])}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--captures", type=Path, nargs="+")
    parser.add_argument("--embedding", nargs="*", default=[],
                        help="FIXTURE=weights npz (exl3_export.py) for the get_rows check")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hashes", type=Path)
    parser.add_argument("--compare-hashes", type=Path, nargs=2)
    args = parser.parse_args()
    if args.compare_hashes:
        a, b = (json.loads(p.read_text()) for p in args.compare_hashes)
        differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        print(json.dumps({"outputs": len(a), "identical": not differ, "differing": differ[:50],
                          "differing_count": len(differ)}, indent=1))
        return
    ops = GGMLOps(args.library)
    scorer = Scorer(ops)
    embeddings = dict(e.split("=", 1) for e in args.embedding)
    inputs = {}
    for cap in args.captures:
        manifest = json.loads((cap / "manifest.json").read_text())
        assert manifest["passed"]
        fixture = cap.name
        inputs[fixture] = {"capture": str(cap), "fixture": manifest["fixture"], "layers": manifest["layers"],
                           "capture_harness_sha256": manifest["harness_sha256"], "mlp_bc_none": manifest["mlp_bc_none"]}
        for path in sorted(cap.glob("capture-*.npz"), key=lambda p: int(p.stem.split("-")[1])):
            prefix = int(path.stem.split("-")[1])
            d = np.load(path)
            tags = sorted({k.split(".")[0] for k in d.files})
            for tag in tags:
                phase, kind = phase_of(tag, prefix), kind_of(tag, prefix)
                for layer in manifest["layers"]:
                    base = f"{tag}.L{layer}"
                    for norm in ("attn_norm", "mlp_norm"):
                        if f"{base}.{norm}.x" in d.files:
                            scorer.norm(fixture, phase, f"{base}.{norm}", d, kind)
                    if f"{base}.rope.q_before" in d.files:
                        scorer.rope(fixture, phase, f"{base}.rope", d, kind)
                    if f"{base}.attn.q" in d.files:
                        rope_key = f"{base}.rope" if f"{base}.rope.q_before" in d.files else None
                        scorer.attention(fixture, phase, f"{base}.attn", d, kind, rope_key)
                    if f"{base}.swiglu.gate" in d.files:
                        scorer.swiglu(fixture, phase, f"{base}.swiglu", d, kind)
                if f"{tag}.final.norm.x" in d.files:
                    scorer.norm(fixture, phase, f"{tag}.final.norm", d, kind)
                if tag == "prefill" and "prefill.embed.ids" in d.files and fixture in embeddings:
                    table = np.load(embeddings[fixture])["model.embed_tokens.weight"]
                    up = torch.from_numpy(d["prefill.embed.out"]).cuda().float().reshape(-1, table.shape[1])
                    for name, dtype in (("bf16", torch.bfloat16), ("f16", torch.float16)):
                        t = torch.from_numpy(table.astype(np.float32)).cuda().to(dtype)
                        if not torch.equal(t.float(), torch.from_numpy(table.astype(np.float32)).cuda()):
                            continue  # this table type cannot hold the exported values exactly
                        got = ops.get_rows(t, torch.from_numpy(d["prefill.embed.ids"]))
                        scorer.check(f"get_rows ({name} table): GGML equals upstream's lookup", torch.equal(got, up))
            print("scored", fixture, prefix, flush=True)
    results, ratios, memory = scorer.aggregate()
    record = {"library": str(args.library), "build": ops.build_info, "inputs": inputs,
              "exactness_checks": dict(scorer.checks), "nonfa_routes": scorer.routes,
              "relative_rms_ratio_to_upstream": ratios, "scratch": memory, "results": results}
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    if args.hashes:
        args.hashes.write_text(json.dumps(scorer.hashes, indent=0, sort_keys=True) + "\n")
    print("DONE", args.output)


if __name__ == "__main__":
    main()
