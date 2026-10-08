#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Tier E, operation level: native EXL3-G's GGML-derived operations recomputed by the bridge.

Reference-only experiment tooling for the backend proof's P3 (docs/backend-proof.md, Tier E, "GGML-
derived operations inside the EXL3 plan"); it does not implement llmpalooza inference and shares no code
with native's planner. Native records, with --record-ops DIR, every operation of its EXL3-G run of
the held-out trajectories (prefixes 32, 144, 145, 1,023 and 1,024, each prefilled at P = 0, then 16
single-token steps):

  DIR/manifest.json  {"format": "llmp-exl3-ops/1", "fixture", "arm": "G", "artifact": <artifact id>,
                      "prefixes": [...], "suffix": 16}
  DIR/<prefix>/<phase>.json, .bin
                     phase 0 is the prefill, phases 1..16 the steps at P = prefix + phase - 1. The JSON
                     lists the operations in execution order, {"op", "layer" (-1 outside layers),
                     "inputs": [T], "weights": [W], "outputs": [T]}, T = {"name" (the record's tensor name), "dtype",
                     "shape" (row-major), "sha256"} plus, for outputs, "offset" and "bytes" of the raw
                     little-endian bytes in the .bin (lm_head's logits: hash only). The first
                     operation, "inputs", outputs the host-built ids, positions and mask; kv_write
                     outputs carry "cells": [P, P+n]; "attention" covers the record's
                     attention.kv_max, attention and attention.combine.

Checks, phase by phase (streamed):
- plan: the operations, in order, are the record's for the phase kind (n, P, Npad) in every layer,
  with the record's input and output tensor names; a phase kind the record lacks fails;
- dtype chain: every tensor's dtype and shape are the record's `tensors` (a noted view shape is
  accepted; the K/V cache is [n, 2, 64] as kv_write writes it and [Npad, 2, 64] as attention reads it);
- wiring: every input's SHA-256 equals the recorded output of the operation that produced it (the
  previous layer's resid.out for resid.in, embed.out for layer 0's); weights (embedding table, norm
  scales widened exactly to F32, q/k/v biases) equal the artifact's; attention's K and V over cells
  [0, Npad) equal this layer's kv_write outputs of every phase so far below P + n, zeros above;
  kv_write writes cells [P, P+n) and its output is its input byte for byte;
- host inputs: ids equal the held-out IDs, positions P..P+n-1, the mask 0 where column j <= P + i
  and -inf elsewhere (F16), all built here;
- values: each GGML operation is recomputed from its recorded inputs by the bridge's library (P0's
  GGML shim: the pinned build of libggml-cuda, NVCC 13.4.92, sm_121a) and must be bit-identical:
  embed (get_rows, BF16 table to F32), attn/mlp/final norm (rms_norm fused with mul, F32), every
  cast (cpy F32<->F16), rope_q/rope_k (NEOX, F32), attention (the vector kernel's launcher, forced;
  F32 out), both residual adds (F32) and swiglu (F32). Semantic parameters come from this script and
  the artifact's config (eps, RoPE base, n_ctx_orig, scale head_dim^-1/2, positions, mask, KV cells);
  linears and bias adds (ExLlamaV3's, gated in P3 part 1) are checked for wiring and dtypes only;
- linear weights: every operation carries "weights": [{"name", "sha256"}], the load-time hash of what
  lies at each address native bound for the linear it runs (empty for every other operation): "trellis",
  "suh", "svh" for a linear; for gate_up both linears' ("gate_proj.trellis" ... "up_proj.svh"), then
  "table.trellis[0]", "table.trellis[1]", "table.suh[0]" ... "table.svh[1]", what lies at the two
  addresses each of the multi-GEMM's device tables holds. Each must be the artifact's tensor of this
  operation's linear and layer (the tables': gate's, then up's), so a linear bound to another layer's or
  tensor's weights, or a stale table, fails;
- RE-010: each phase's recorded logits hash equals native's uninstrumented logits (--uninstrumented
  NATIVE_OUT, required: logits-<p>.prefill.npy / .suffix.npy, F32 widened from F16) narrowed to F16.
  NATIVE_OUT's manifest.json must be llmp_exl3_exec's for this fixture, arm G and artifact, with
  neither capture nor record_ops, and no mismatched evaluation.

Every library mapped into the process whose name starts with libggml must hash to the record's pins,
except libggml_shim.so (it adds the cast entry point), which must hash to SHIM_SHA256, the cuda134b
build this script was reviewed with. Prints a per-operation summary
per phase; exits 0 only when everything is exact and complete, 1 on any difference (the first is
named: prefix, phase, layer, op, tensor), 2 on a malformed or incomplete recording.

In the reference container (run_container.sh), with the artifact store and recording mounted:
  GGML_OPS_LIB=/p0/ggmlops/cuda134b/libggml_shim.so op_tier_e.py --artifact /artifacts/<id> \\
      --ids /p0/heldout-ids.i64le [--record exl3-op-plan-g.json] --uninstrumented DIR RECORDING
"""

import argparse
import hashlib
import json
import math
import mmap
import os
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend-proof-p0"))

HELD_OUT_SHA256 = "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c"
SHIM_SHA256 = "70904fc3ad46e3e8ad54e4c5461814bd80c75ee21eb33d09af04f906a53c4440"   # ggmlops/cuda134b
PREFIXES, SUFFIX, KV_PAD = (32, 144, 145, 1023, 1024), 16, 256
NP = {"float32": np.float32, "float16": np.float16, "int32": np.int32}
ITEM = {"float32": 4, "float16": 2, "bfloat16": 2, "int32": 4}
MERGED = ("attention.kv_max", "attention.combine")   # inside native's one attention operation
WEIGHTS = {"attn_norm.w": "model.layers.{l}.input_layernorm.weight",
           "mlp_norm.w": "model.layers.{l}.post_attention_layernorm.weight",
           "final_norm.w": "model.norm.weight", "embed_table": "model.embed_tokens.weight",
           "q_proj.bias": "model.layers.{l}.self_attn.q_proj.bias",
           "k_proj.bias": "model.layers.{l}.self_attn.k_proj.bias",
           "v_proj.bias": "model.layers.{l}.self_attn.v_proj.bias"}
# The linear operations and the artifact's linear each reads, by layer ("lm_head" outside the layers).
LINEARS = {"q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj", "v_proj": "self_attn.v_proj",
           "o_proj": "self_attn.o_proj", "gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj",
           "down_proj": "mlp.down_proj", "lm_head": "lm_head"}
PARTS = ("trellis", "suh", "svh")
CASTS = ("attn_norm.cast", "rope_q.cast", "rope_k.cast", "rope_k.cast_out", "attention.cast", "mlp_norm.cast",
         "swiglu.cast", "final_norm.cast")
NORMS = {"attn_norm": "attn_norm.w", "mlp_norm": "mlp_norm.w", "final_norm": "final_norm.w"}


class Malformed(Exception):
    """The recording is malformed or incomplete (exit 2)."""


class Unpinned(Exception):
    """The bridge's library is not the pinned build (exit 2: nothing can be judged)."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(condition, message):
    if not condition:
        raise Malformed(message)


# --------------------------------------------------------------------------- reference inputs


class Artifact:
    """A v0 artifact (docs/artifact-format.md): identity, config and plain resources, chunk-verified."""

    def __init__(self, root):
        manifest = (root / "manifest.json").read_bytes()
        need(sha(manifest) == root.name, f"{root}: manifest.json does not hash to the artifact id")
        files = {f["path"]: f for f in json.loads(manifest)["files"]}
        index = (root / "index.json").read_bytes()
        need(sha(index) == files["index.json"]["sha256"], "index.json differs from the manifest")
        config = (root / "meta" / "config.json").read_bytes()
        need(sha(config) == files["meta/config.json"]["sha256"], "meta/config.json differs from the manifest")
        self.root, self.index, self.config = root, json.loads(index), json.loads(config)
        # The checkpoint it was prepared from (the manifest's source), which names the fixture.
        self.source = {s["name"]: s["sha256"] for s in json.loads(manifest)["source"]}
        self.by_role = {role: r for r in self.index["resources"] for role in r["roles"]}

    def read(self, name):
        """(repr, bytes) of a resource; the chunks it lies in must hash as the index records."""
        r = self.by_role[name]
        group = self.index["groups"][r["group"]]
        shard = self.index["shards"][group["shard"]]
        chunk = self.index["chunk_bytes"]
        first, last = r["offset"] // chunk, (r["offset"] + r["bytes"] - 1) // chunk
        with open(self.root / shard["path"], "rb") as f:
            f.seek(shard["data_offset"] + group["offset"] + first * chunk)
            span = f.read(min((last + 1) * chunk, group["stored_bytes"]) - first * chunk)
        for i in range(first, last + 1):
            piece = span[(i - first) * chunk:(i - first + 1) * chunk]
            need(sha(piece) == self.index["chunk_sha256"][group["first_chunk"] + i], f"{name}: chunk hash differs")
        start = r["offset"] - first * chunk
        return r["repr"], span[start:start + r["bytes"]]


class Reference:
    """What the bridge takes from its own code, the artifact and the held-out IDs."""

    def __init__(self, artifact, ids_path, record, ops, torch):
        self.torch, self.ops, self.record = torch, ops, record
        c = artifact.config
        m = record["model"]
        self.layers, self.hidden, self.heads = c["num_hidden_layers"], c["hidden_size"], c["num_attention_heads"]
        self.kv_heads, self.head_dim = c["num_key_value_heads"], c["hidden_size"] // c["num_attention_heads"]
        self.eps, self.theta = float(c["rms_norm_eps"]), float(c["rope_theta"])
        self.n_ctx_orig, self.vocab = c["max_position_embeddings"], c["vocab_size"]
        # The fixture whose checkpoint the artifact was prepared from, as the record's probes name
        # it (config.json's quantization bits say 4 for the mixed 4.5 bpw fixture too).
        fixtures = [f for f, p in record["provenance"]["probes"].items()
                    if p["weight_sha256"] == artifact.source.get("model.safetensors")]
        need(len(fixtures) == 1, "the artifact was prepared from neither fixture's checkpoint")
        self.fixture = fixtures[0]
        mine = {"layers": self.layers, "hidden": self.hidden, "heads": self.heads, "kv_heads": self.kv_heads,
                "head_dim": self.head_dim, "vocab": self.vocab, "rms_norm_eps": self.eps, "rope_theta": self.theta}
        need(all(m[k] == v for k, v in mine.items()), f"the artifact's config differs from the record's model {m}")
        self.scale = self.head_dim ** -0.5
        ids = Path(ids_path).read_bytes()
        need(sha(ids) == HELD_OUT_SHA256, "held-out ID identity mismatch")
        self.ids = np.frombuffer(ids, dtype="<i8")
        self.weights = {}   # (name, layer) -> (sha256, dtype, shape, device tensor)
        for layer in [-1] + list(range(self.layers)):
            for name, role in WEIGHTS.items():
                if ("{l}" in role) != (layer >= 0):
                    continue
                rep, data = artifact.read(role.format(l=layer))
                if name.endswith(".w"):     # BF16 widened exactly to F32
                    need(rep["dtype"] == "BF16", f"{role}: {rep}")
                    bits = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
                    array, dtype = bits.view(np.float32), "float32"
                    tensor = torch.from_numpy(array.copy()).cuda()
                elif name == "embed_table":
                    need(rep["dtype"] == "BF16", f"{role}: {rep}")
                    array, dtype = np.frombuffer(data, dtype="<i2"), "bfloat16"
                    tensor = torch.from_numpy(array.copy()).view(torch.bfloat16).reshape(rep["shape"]).cuda()
                else:
                    need(rep["dtype"] == "F16", f"{role}: {rep}")
                    array, dtype = np.frombuffer(data, dtype="<f2"), "float16"
                    tensor = torch.from_numpy(array.copy()).cuda()
                self.weights[(name, layer)] = (sha(array.tobytes()), dtype, list(rep["shape"]), tensor)
        # Every linear's trellis and side vectors, as the artifact holds them (their bytes are what
        # native loads and hashes at the address it binds).
        self.linears = {}   # (linear, layer, part) -> sha256
        for layer in [-1] + list(range(self.layers)):
            for op, linear in LINEARS.items():
                if (op == "lm_head") != (layer < 0):
                    continue
                prefix = linear if layer < 0 else f"model.layers.{layer}.{linear}"
                for part in PARTS:
                    self.linears[(op, layer, part)] = sha(artifact.read(f"{prefix}.{part}")[1])

    def linear_weights(self, op, layer):
        """The weights native must record for linear operation op in layer, [(name, sha256)] in order; for
        the multi-GEMM (gate_up) both linears' and then what its three device tables point at (the gate
        linear's tensor, then up's), or None for an operation that reads no linear."""
        if op == "gate_up":
            names = [(f"{lin}.{part}", self.linears[(lin, layer, part)]) for lin in ("gate_proj", "up_proj")
                     for part in PARTS]
            return names + [(f"table.{part}[{i}]", self.linears[(lin, layer, part)])
                            for part in PARTS for i, lin in enumerate(("gate_proj", "up_proj"))]
        if op in LINEARS:
            return [(part, self.linears[(op, layer, part)]) for part in PARTS]
        return None

    def host_inputs(self, prefix, phase, n, past, npad):
        ids = self.ids[:prefix] if phase == 0 else self.ids[prefix + phase - 1:prefix + phase]
        mask = np.where(np.arange(npad)[None, :] <= past + np.arange(n)[:, None], 0.0, -np.inf).astype("<f2")
        return {"ids": ids.astype("<i4"), "positions": np.arange(past, past + n, dtype="<i4"), "mask": mask}

    def recompute(self, op, x, n, past, host):
        """The bridge's output of GGML operation op from its recorded inputs x (device tensors, by name);
        IDs, positions, the mask and K/V cells are the bridge's own (host, and x's caches)."""
        torch, ops = self.torch, self.ops
        f32 = torch.float32
        if op == "embed":
            return ops.get_rows(x["embed_table"], torch.from_numpy(host["ids"].copy()))
        if op in NORMS:
            return ops.rms_norm(x[next(iter(x))], x[NORMS[op]], self.eps, out_dtype=f32, fused=True)
        if op in CASTS:
            (src,) = x.values()
            return ops.cast(src, torch.float16 if src.dtype == f32 else f32)
        if op in ("rope_q", "rope_k"):
            src = x[op + ".in"].reshape(n, -1, self.head_dim)
            return ops.rope_neox(src, past, self.theta, f32, f32, n_ctx_orig=self.n_ctx_orig)
        if op == "attention":
            q = x["q_rope"].reshape(n, self.heads, self.head_dim)
            k, v = x["k_cache"][:past + n].contiguous(), x["v_cache"][:past + n].contiguous()
            return ops.attention(q, k, v, past + n, past, self.scale, "fa_vec", out_dtype=f32)
        if op in ("attn_residual_add", "mlp_residual_add"):
            a, b = x.values()
            return ops.add(a, b)
        if op == "swiglu":
            return ops.swiglu(x["gate"], x["up"], f32)
        raise ValueError(op)


# --------------------------------------------------------------------------- the check


def shape_of(spec, n, npad):
    """A `tensors` shape string ('[n, Npad]', '[4096, 2, 64] per layer') as a list."""
    dims = spec.split(" per ")[0].strip().strip("[]").split(",")
    return [{"n": n, "Npad": npad}.get(d.strip()) or int(d) for d in dims]


class Check:
    def __init__(self, ref, record, recording, uninstrumented, prefixes=PREFIXES):
        self.ref, self.record, self.dir, self.native = ref, record, recording, uninstrumented
        self.prefixes = prefixes
        self.specs = {o["op"]: o for seg in record["operations"].values() for o in seg}
        self.specs["inputs"] = {"op": "inputs", "inputs": [], "outputs": ["ids", "positions", "mask"]}
        self.first = None
        self.failures = 0

    def fail(self, where, what):
        self.failures += 1
        if self.first is None:
            self.first = f"{where}: {what}"
            print("FIRST DIFFERENCE", self.first, flush=True)

    def kind(self, n, past, npad):
        for pk in self.record["phase_kinds"]:
            if pk["kind"] == "prefill" and n > 1 and pk["rows"] == n and past == 0:
                return pk
            if pk["kind"] != "prefill" and n == 1 and pk["padded_kv_length"] == npad:
                return pk
        return None

    def expected(self, pk):
        layer = [op for op in pk["layer_order"] if op not in MERGED]
        return ([("inputs", -1)] + [(o["op"], -1) for o in pk["embedding"]]
                + [(op, lay) for lay in range(self.ref.layers) for op in layer]
                + [(op, -1) for op in pk["output_order"]])

    def dtype_ok(self, t, n, npad, role):
        """The dtype chain: t's dtype and shape against the record's tensors table."""
        name = t["name"]
        if name in ("k_cache", "v_cache"):
            want = [npad if role == "attention" else n, self.ref.kv_heads, self.ref.head_dim]
            return t["dtype"] == "float16" and t["shape"] == want
        entry = self.record["tensors"].get(name)
        if entry is None:
            return False
        shapes = [shape_of(entry["shape"], n, npad)]
        note = entry.get("note", "")
        if "viewed [" in note:
            shapes.append(shape_of(note.split("viewed ", 1)[1].split("]")[0] + "]", n, npad))
        return t["dtype"] == entry["dtype"] and t["shape"] in shapes

    def run(self):
        manifest = json.loads((self.dir / "manifest.json").read_text())
        need(manifest.get("format") == "llmp-exl3-ops/1" and manifest.get("arm") == "G",
             f"not an EXL3-G operation recording: {manifest}")
        need(manifest.get("fixture") == self.ref.fixture, f"fixture {manifest.get('fixture')} is not the artifact's")
        need(manifest.get("suffix") == SUFFIX, "suffix must be 16")
        need(sorted(manifest.get("prefixes", [])) == sorted(self.prefixes), f"prefixes {manifest.get('prefixes')}")
        for prefix in manifest["prefixes"]:
            cap = -(-(prefix + SUFFIX) // KV_PAD) * KV_PAD
            kv = {(c, lay): np.zeros((cap, self.ref.kv_heads, self.ref.head_dim), "<f2")
                  for c in ("k_cache", "v_cache") for lay in range(self.ref.layers)}
            for phase in range(SUFFIX + 1):
                self.phase(prefix, phase, kv)
        return self.failures == 0

    def phase(self, prefix, phase, kv):
        n, past = (prefix, 0) if phase == 0 else (1, prefix + phase - 1)
        npad = -(-(past + n) // KV_PAD) * KV_PAD
        where = f"prefix {prefix} phase {phase}"
        path = self.dir / str(prefix) / f"{phase}.json"
        need(path.exists() and path.with_suffix(".bin").exists(), f"{where}: missing {path.name} or its .bin")
        doc = json.loads(path.read_text())
        need((doc.get("prefix"), doc.get("phase"), doc.get("rows"), doc.get("past"), doc.get("npad"))
             == (prefix, phase, n, past, npad), f"{where}: header {[doc.get(k) for k in ('rows', 'past', 'npad')]}")
        pk = self.kind(n, past, npad)
        if pk is None:
            self.fail(where, f"phase kind (n {n}, P {past}, Npad {npad}) is not in the record")
            return
        got = [(o.get("op"), o.get("layer")) for o in doc["ops"]]
        want = self.expected(pk)
        if got != want:
            remaining = iter(want)
            if len(got) < len(want) and all(g in remaining for g in got):
                raise Malformed(f"{where}: incomplete: {len(want) - len(got)} operations missing, the first "
                                f"{next(w for w in want if w not in set(got))}")
            i = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            self.fail(where, f"operation {i} is {got[i] if i < len(got) else None}, the record's "
                             f"{want[i] if i < len(want) else None}")
            return
        blob = path.with_suffix(".bin")
        with open(blob, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if size else b""
            try:
                counts = self.ops(doc["ops"], mm, prefix, phase, n, past, npad, kv, where)
            finally:
                if size:
                    mm.close()
        print(f"{where} ({pk['phase_kind']}): " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in counts.items()), flush=True)

    def ops(self, ops, mm, prefix, phase, n, past, npad, kv, where):
        torch, ref = self.ref.torch, self.ref
        produced = {}       # (layer, name) -> (sha256, dtype, shape, offset, bytes)
        counts = {}

        def count(key, ok):
            c = counts.setdefault(key, [0, 0])
            c[0] += int(bool(ok))
            c[1] += 1
            return ok

        def load(entry):
            _, dtype, shape, offset, nbytes = entry
            array = np.frombuffer(mm[offset:offset + nbytes], dtype=NP[dtype])
            return torch.from_numpy(array.reshape(shape).copy()).cuda()

        host = ref.host_inputs(prefix, phase, n, past, npad)
        last = ref.layers - 1
        for op in ops:
            name, layer = op["op"], op["layer"]
            at = f"{where} layer {layer} op {name}"
            spec = self.specs[name]
            want_in = [s.split(" (")[0] for s in spec["inputs"]]   # "resid.out (layer 23)"
            for part, names in (("inputs", want_in), ("outputs", spec["outputs"])):
                if [t.get("name") for t in op[part]] != names:
                    self.fail(at, f"{part} {[t.get('name') for t in op[part]]}, the record's {names}")
                    return counts
            for t in op["inputs"] + op["outputs"]:
                need(isinstance(t.get("sha256"), str) and isinstance(t.get("shape"), list) and "dtype" in t,
                     f"{at}: tensor {t.get('name')} lacks sha256, dtype or shape")
                if not count("dtype", self.dtype_ok(t, n, npad, name)):
                    self.fail(f"{at} tensor {t['name']}", f"dtype {t['dtype']} shape {t['shape']}")
            # Outputs: stored bytes present and hashing as recorded.
            outs = {}
            for t in op["outputs"]:
                nbytes = math.prod(t["shape"]) * ITEM.get(t["dtype"], 0)
                if t["name"] == "logits":
                    outs[t["name"]] = (t["sha256"], t["dtype"], t["shape"], None, nbytes)
                    continue
                need(t.get("bytes") == nbytes and isinstance(t.get("offset"), int) and 0 <= t["offset"]
                     and t["offset"] + nbytes <= len(mm), f"{at}: output {t['name']} not stored as its shape")
                need(sha(mm[t["offset"]:t["offset"] + nbytes]) == t["sha256"],
                     f"{at}: stored {t['name']} does not hash to its sha256")
                outs[t["name"]] = (t["sha256"], t["dtype"], t["shape"], t["offset"], nbytes)
            # Inputs: wiring to their producers.
            values = {}
            for t in op["inputs"]:
                key = t["name"]
                if key in WEIGHTS:
                    wsha, _, _, tensor = ref.weights[(key, layer if "{l}" in WEIGHTS[key] else -1)]
                    ok, values[key] = t["sha256"] == wsha, tensor
                elif key in ("k_cache", "v_cache"):
                    cells = kv[(key, layer)][:npad]
                    ok, values[key] = t["sha256"] == sha(cells.tobytes()), torch.from_numpy(cells.copy()).cuda()
                else:
                    src = ((layer - 1, "resid.out") if layer > 0 else (-1, "embed.out")) if key == "resid.in" else \
                        (last, "resid.out") if key == "resid.out" and layer == -1 else \
                        (-1, key) if key in host else (layer, key)
                    entry = produced.get(src)
                    ok = entry is not None and t["sha256"] == entry[0]
                    if ok and name in self.GGML and entry[1] in NP:
                        values[key] = load(entry)
                if not count("wiring", ok):
                    self.fail(f"{at} tensor {key}", "input is not its producer's recorded output")
            # The linears' weights (and the multi-GEMM's tables) at the addresses native bound: the
            # artifact's tensors of this operation's linear and layer, never another's.
            need(isinstance(op.get("weights"), list), f"{at}: no weights list (a recording before linear weights)")
            want_w = ref.linear_weights(name, layer)
            got_w = [(w.get("name"), w.get("sha256")) for w in op["weights"]]
            if want_w is None:
                if not count("linear weights", not got_w):
                    self.fail(at, f"records linear weights {[n for n, _ in got_w]} for an operation that reads none")
            elif [n for n, _ in got_w] != [n for n, _ in want_w]:
                count("linear weights", False)
                self.fail(at, f"linear weights {[n for n, _ in got_w]}, the plan's {[n for n, _ in want_w]}")
            else:
                for (wname, got), (_, want) in zip(got_w, want_w):
                    if not count("linear weights", got == want):
                        self.fail(f"{at} tensor {wname}", "is not the artifact's tensor for this linear and layer")
            # The operation itself.
            if name == "inputs":
                for key, array in host.items():
                    if not count("host inputs", outs[key][0] == sha(array.tobytes())):
                        self.fail(f"{at} tensor {key}", "differs from the bridge's own construction")
            elif name.startswith("kv_write"):
                t = op["outputs"][0]
                ok = t.get("cells") == [past, past + n] and t["sha256"] == op["inputs"][0]["sha256"]
                if count("kv_write", ok):
                    array = np.frombuffer(mm[t["offset"]:t["offset"] + t["bytes"]], "<f2")
                    kv[(t["name"], layer)][past:past + n] = array.reshape(n, ref.kv_heads, ref.head_dim)
                else:
                    self.fail(f"{at} tensor {t['name']}", f"cells {t.get('cells')} or bytes differ from its input")
            elif name in self.GGML and len(values) == len(op["inputs"]):
                result = ref.recompute(name, values, n, past, host)
                data = result.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                (out,) = op["outputs"]
                if not count(name, sha(data) == out["sha256"]):
                    self.fail(f"{at} tensor {out['name']}", "the bridge's recomputation differs")
            elif name in self.GGML:   # an input not wired (or not loadable): never a silent skip
                count(name, False)
                self.fail(f"{at} tensor {op['outputs'][0]['name']}", "not recomputed: an input is not wired")
            if name == "lm_head" and self.native is not None:
                self.logits(prefix, phase, op["outputs"][0], at, count)
            for key, entry in outs.items():
                produced[(layer, key)] = entry
        return counts

    GGML = {"embed", "attn_norm", "mlp_norm", "final_norm", "rope_q", "rope_k", "attention", "attn_residual_add",
            "mlp_residual_add", "swiglu", *CASTS}

    def logits(self, prefix, phase, out, at, count):
        path = self.native / f"logits-{prefix}.{'prefill' if phase == 0 else 'suffix'}.npy"
        need(path.exists(), f"missing {path}")
        rows = np.load(path, mmap_mode="r")
        rows = rows if phase == 0 else rows[phase - 1:phase]
        narrow = np.ascontiguousarray(rows, dtype="<f4").astype("<f2")
        ok = np.array_equal(narrow.astype("<f4"), rows, equal_nan=True) and sha(narrow.tobytes()) == out["sha256"]
        if not count("uninstrumented logits", ok):
            self.fail(f"{at} tensor logits", "differs from the uninstrumented run's logits")


def uninstrumented_control(native, ref, artifact_id):
    """native (the uninstrumented run's --out) is llmp_exl3_exec's plain EXL3-G run of this artifact."""
    path = Path(native) / "manifest.json"
    need(path.exists(), f"the uninstrumented control has no {path}")
    m = json.loads(path.read_text())
    want = {"harness": "llmp_exl3_exec", "fixture": ref.fixture, "arm": "G", "artifact": artifact_id,
            "capture": False, "record_ops": False, "mismatches": 0}
    got = {k: m.get(k) for k in want}
    need(got == want, f"{path} is not an uninstrumented EXL3-G run of this artifact: {got}, want {want}")


def mapped_libraries(record):
    """Every mapped libggml* file hashes to the record's pins (libggml_shim.so to SHIM_SHA256)."""
    pins = next(iter(record["provenance"]["probes"].values()))["library_sha256"]
    by_name = {Path(k).name: v for k, v in pins.items()}
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if "/libggml" in line and line.split()[-1].startswith("/")})
    report = {}
    for path in mapped:
        name = Path(path).name
        report[path] = digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        pin = SHIM_SHA256 if name == "libggml_shim.so" else by_name.get(name)
        if pin != digest:
            raise Unpinned(f"{path} ({digest}) is not the pinned build ({pin})")
    if not any("libggml-cuda" in p for p in mapped):
        raise Unpinned("libggml-cuda is not mapped")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--artifact", type=Path, required=True, help="the installed artifact directory")
    parser.add_argument("--ids", type=Path, required=True, help="the held-out IDs (little-endian int64)")
    parser.add_argument("--record", type=Path, default=HERE / "exl3-op-plan-g.json")
    parser.add_argument("--uninstrumented", type=Path, required=True,
                        help="native's uninstrumented output directory (the RE-010 control)")
    parser.add_argument("--lib", default=os.environ.get("GGML_OPS_LIB"), help="the shim (default GGML_OPS_LIB)")
    args = parser.parse_args()
    return check(args.recording, args.artifact, args.ids, args.record, args.uninstrumented, args.lib)


def check(recording, artifact, ids, record_path, uninstrumented, lib, prefixes=PREFIXES):
    import torch
    import ggml_ops
    record = json.loads(Path(record_path).read_text())
    try:
        ops = ggml_ops.GGMLOps(lib)
        for path, digest in mapped_libraries(record).items():
            print("library", path, digest)
        ref = Reference(Artifact(Path(artifact).resolve()), ids, record, ops, torch)
        manifest = json.loads((Path(recording) / "manifest.json").read_text())
        need(manifest.get("artifact") == Path(artifact).resolve().name, "the recording names another artifact")
        need(uninstrumented is not None, "no uninstrumented control (RE-010)")
        uninstrumented_control(uninstrumented, ref, Path(artifact).resolve().name)
        with torch.inference_mode():
            ok = Check(ref, record, Path(recording), uninstrumented, prefixes)
            exact = ok.run()
    except Malformed as error:
        print("MALFORMED OR INCOMPLETE:", error, flush=True)
        return 2
    except Unpinned as error:
        print("NOT THE PINNED BRIDGE:", error, flush=True)
        return 2
    except Exception:   # a recording this script cannot read is never a pass
        traceback.print_exc()
        print("MALFORMED OR INCOMPLETE: unreadable recording (above)", flush=True)
        return 2
    print("PASS: every GGML operation exact, wiring, weights and dtype chain complete" if exact else
          f"FAIL: {ok.failures} differences; first: {ok.first}", flush=True)
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
