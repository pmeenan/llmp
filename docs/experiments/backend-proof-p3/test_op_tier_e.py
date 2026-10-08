#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Self-test of op_tier_e.py on synthetic recordings (reference only; GPU, reference container).

Writes a recording of the 32-row prefix's trajectory (the prefill and 16 single-token steps) in
native's --record-ops format, with every GGML operation computed by the shim (the same kernels
op_tier_e.py recomputes with), the artifact's weights, the held-out IDs, and synthetic linears
(seeded random projections: op_tier_e.py checks only their wiring, dtypes and the artifact weights
they record). It then runs
op_tier_e.check on that recording, restricted to that prefix, and on altered copies, each of which
must give the stated exit code and first difference:

  exact            0                       the recording as written, with its uninstrumented logits
  flipped bit      1  rope_q, q_rope       one bit of an output, hashes updated (a wrong result)
  corrupt bytes    2                       one stored bit flipped, hashes not updated
  mis-wired input  1  mlp_norm, resid.mid  a stale buffer: the input hash is resid.in's
  wrong dtype      1  swiglu.cast          swiglu.out declared bfloat16
  missing layer    2                       layer 12 of one step dropped
  stale K cell     1  attention, k_cache   the K input over [0, Npad) hashes with one cell zeroed
  wrong mask       1  inputs, mask         one future column unmasked, consumers rewired to it
  wrong cells      1  kv_write.k           cells [P, P+n+1]
  norm weight      1  attn_norm            layer 3's norm scale is layer 4's
  linear weight    1  q_proj, trellis      layer 3's q_proj trellis is layer 4's
  stale table      1  gate_up, table       the multi-GEMM's up trellis table entry points at layer 7's
  no weights       2                       an operation's weights list missing (an older recording)
  logits           1  lm_head              one uninstrumented logit changed (RE-010)
  instrumented     2                       the control's manifest says record_ops (not uninstrumented)
  no control       2                       no uninstrumented control at all

  run_container.sh test_op_tier_e.py --artifact /artifacts/<id> --ids /p0/heldout-ids.i64le \\
      --lib /p0/ggmlops/cuda134b/libggml_shim.so --work /out/selftest
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backend-proof-p0"))
import op_tier_e  # noqa: E402

PREFIX, SUFFIX = 32, 16


def sha(data):
    return hashlib.sha256(data).hexdigest()


def generate(out, artifact_dir, ids_path, lib):
    """A recording of the 32-row trajectory, and its uninstrumented logits, in out/."""
    import torch
    import ggml_ops
    ops = ggml_ops.GGMLOps(lib)
    art = op_tier_e.Artifact(artifact_dir)
    cfg = art.config
    layers, hidden, heads, kvh = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"], \
        cfg["num_key_value_heads"]
    hd, inter, vocab, eps, theta = hidden // heads, cfg["intermediate_size"], cfg["vocab_size"], \
        cfg["rms_norm_eps"], cfg["rope_theta"]
    f16, f32 = torch.float16, torch.float32
    names = {torch.float32: "float32", torch.float16: "float16", torch.int32: "int32"}

    def plain(role, widen):
        rep, data = art.read(role)
        if widen:   # BF16 -> F32 exactly
            array = (np.frombuffer(data, "<u2").astype(np.uint32) << 16).view(np.float32)
            return {"sha256": sha(array.tobytes()), "dtype": "float32", "shape": rep["shape"],
                    "t": torch.from_numpy(array.copy()).cuda()}
        return {"sha256": sha(data), "dtype": "float16", "shape": rep["shape"],
                "t": torch.from_numpy(np.frombuffer(data, "<f2").copy()).cuda()}
    rep, data = art.read("model.embed_tokens.weight")
    table = torch.from_numpy(np.frombuffer(data, "<i2").copy()).view(torch.bfloat16).reshape(rep["shape"]).cuda()
    weights = {("embed_table", -1): {"sha256": sha(data), "dtype": "bfloat16", "shape": rep["shape"]},
               ("final_norm.w", -1): plain("model.norm.weight", True)}
    for layer in range(layers):
        p = f"model.layers.{layer}."
        weights[("attn_norm.w", layer)] = plain(p + "input_layernorm.weight", True)
        weights[("mlp_norm.w", layer)] = plain(p + "post_attention_layernorm.weight", True)
        for x in "qkv":
            weights[(f"{x}_proj.bias", layer)] = plain(p + f"self_attn.{x}_proj.bias", False)
    # The linears' weights as native records them (the artifact's bytes at each bound address).
    lin = {}
    for layer in [-1] + list(range(layers)):
        for op_name, sub in (("lm_head", "lm_head"),) if layer < 0 else (
                ("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"), ("v_proj", "self_attn.v_proj"),
                ("o_proj", "self_attn.o_proj"), ("gate_proj", "mlp.gate_proj"), ("up_proj", "mlp.up_proj"),
                ("down_proj", "mlp.down_proj")):
            for part in ("trellis", "suh", "svh"):
                role = sub if layer < 0 else f"model.layers.{layer}.{sub}"
                lin[(op_name, layer, part)] = sha(art.read(f"{role}.{part}")[1])

    def linear_weights(name, layer):
        parts = ("trellis", "suh", "svh")
        if name == "gate_up":
            out = [{"name": f"{x}.{part}", "sha256": lin[(x, layer, part)]} for x in ("gate_proj", "up_proj")
                   for part in parts]
            return out + [{"name": f"table.{part}[{i}]", "sha256": lin[(x, layer, part)]}
                          for part in parts for i, x in enumerate(("gate_proj", "up_proj"))]
        if (name, layer, "trellis") in lin:
            return [{"name": part, "sha256": lin[(name, layer, part)]} for part in parts]
        return []
    gen = torch.Generator(device="cuda").manual_seed(20260927)
    mats = {name: torch.randn(k, m, generator=gen, device="cuda") / k ** 0.5 for name, k, m in (
        ("q", hidden, hidden), ("k", hidden, kvh * hd), ("v", hidden, kvh * hd), ("o", hidden, hidden),
        ("gate", hidden, inter), ("up", hidden, inter), ("down", inter, hidden), ("head", hidden, 64))}

    def linear(x, name, dtype):
        y = x.float().reshape(x.shape[0], -1) @ mats[name]
        return (y.repeat(1, vocab // 64) if name == "head" else y).to(dtype).contiguous()

    ids_all = np.fromfile(ids_path, dtype="<i8")
    cap = -(-(PREFIX + SUFFIX) // 256) * 256
    kcache = [np.zeros((cap, kvh, hd), "<f2") for _ in range(layers)]
    vcache = [np.zeros((cap, kvh, hd), "<f2") for _ in range(layers)]
    (out / str(PREFIX)).mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({"format": "llmp-exl3-ops/1", "fixture": "4.0bpw", "arm": "G",
                                                   "artifact": artifact_dir.name, "prefixes": [PREFIX],
                                                   "suffix": SUFFIX}))
    logits_rows = []
    for phase in range(SUFFIX + 1):
        n, past = (PREFIX, 0) if phase == 0 else (1, PREFIX + phase - 1)
        npad = -(-(past + n) // 256) * 256
        blob = open(out / str(PREFIX) / f"{phase}.bin", "wb")
        ops_list, offset = [], [0]

        def o(name, t, **extra):
            """An output entry, stored (t: torch tensor or numpy array)."""
            array = t.detach().contiguous().cpu().numpy() if hasattr(t, "detach") else np.ascontiguousarray(t)
            data = array.tobytes()
            entry = {"name": name, "dtype": str(array.dtype), "shape": list(array.shape),
                     "sha256": sha(data), **extra}
            if name != "logits":
                entry.update(offset=offset[0], bytes=len(data))
                blob.write(data)
                offset[0] += len(data)
            return entry

        def i(entry):
            return {k: entry[k] for k in ("name", "dtype", "shape", "sha256")}

        def w(name, layer):
            e = weights[(name, layer)]
            return {"name": name, "dtype": e["dtype"], "shape": e["shape"], "sha256": e["sha256"]}

        def rename(entry, name, shape=None):
            return {**i(entry), "name": name, **({"shape": shape} if shape else {})}

        def op(name, layer, inputs, outputs):
            ops_list.append({"op": name, "layer": layer, "inputs": inputs, "weights": linear_weights(name, layer),
                             "outputs": outputs})
            return outputs
        ids = (ids_all[:PREFIX] if phase == 0 else ids_all[PREFIX + phase - 1:PREFIX + phase]).astype("<i4")
        mask = np.where(np.arange(npad)[None, :] <= past + np.arange(n)[:, None], 0.0, -np.inf).astype("<f2")
        e_ids, e_pos, e_mask = op("inputs", -1, [], [o("ids", ids), o("positions", np.arange(past, past + n,
                                                                                               dtype="<i4")),
                                                     o("mask", mask)])
        x = ops.get_rows(table, torch.from_numpy(ids))
        (e_x,) = op("embed", -1, [w("embed_table", -1), i(e_ids)], [o("embed.out", x)])
        for layer in range(layers):
            e_in = rename(e_x, "resid.in")
            a = ops.rms_norm(x, weights[("attn_norm.w", layer)]["t"], eps, out_dtype=f32)
            (e_a,) = op("attn_norm", layer, [e_in, w("attn_norm.w", layer)], [o("attn_norm.f32", a)])
            a16 = ops.cast(a, f16)
            (e_a16,) = op("attn_norm.cast", layer, [i(e_a)], [o("attn_norm.out", a16)])
            proj = {}
            for p, width in (("q", hidden), ("k", kvh * hd), ("v", kvh * hd)):
                g = linear(a16, p, f16)
                (e_g,) = op(f"{p}_proj", layer, [i(e_a16)], [o(f"{p}_proj.gemm", g)])
                y = g + weights[(f"{p}_proj.bias", layer)]["t"]
                (proj[p],) = op(f"{p}_proj.bias_add", layer, [i(e_g), w(f"{p}_proj.bias", layer)], [o(p, y)])
                proj[p + "_t"] = y
            rq = ops.cast(proj["q_t"].view(n, heads, hd), f32)
            (e_rq,) = op("rope_q.cast", layer, [i(proj["q"])], [o("rope_q.in", rq)])
            qr = ops.rope_neox(rq, past, theta, f32, f32)
            (e_qr,) = op("rope_q", layer, [i(e_rq), i(e_pos)], [o("q_rope", qr)])
            rk = ops.cast(proj["k_t"].view(n, kvh, hd), f32)
            (e_rk,) = op("rope_k.cast", layer, [i(proj["k"])], [o("rope_k.in", rk)])
            kr32 = ops.rope_neox(rk, past, theta, f32, f32)
            (e_kr32,) = op("rope_k", layer, [i(e_rk), i(e_pos)], [o("k_rope.f32", kr32)])
            kr = ops.cast(kr32, f16)
            (e_kr,) = op("rope_k.cast_out", layer, [i(e_kr32)], [o("k_rope", kr)])
            v = proj["v_t"].view(n, kvh, hd)
            kcache[layer][past:past + n] = kr.cpu().numpy()
            vcache[layer][past:past + n] = v.cpu().numpy()
            op("kv_write.k", layer, [i(e_kr)], [o("k_cache", kr, cells=[past, past + n])])
            op("kv_write.v", layer, [i(proj["v"])], [o("v_cache", v, cells=[past, past + n])])
            kin, vin = kcache[layer][:npad], vcache[layer][:npad]
            att = ops.attention(qr, torch.from_numpy(kin[:past + n].copy()).cuda(),
                                torch.from_numpy(vin[:past + n].copy()).cuda(), past + n, past, hd ** -0.5, "fa_vec",
                                out_dtype=f32)
            (e_att,) = op("attention", layer, [i(e_qr), {"name": "k_cache", "dtype": "float16", "shape": list(kin.shape),
                                                         "sha256": sha(kin.tobytes())},
                                               {"name": "v_cache", "dtype": "float16", "shape": list(vin.shape),
                                                "sha256": sha(vin.tobytes())}, i(e_mask)],
                          [o("attn.f32", att)])
            ao = ops.cast(att, f16).reshape(n, hidden)
            (e_ao,) = op("attention.cast", layer, [i(e_att)], [o("attn.out", ao)])
            ob = linear(ao, "o", f32)
            (e_o,) = op("o_proj", layer, [i(e_ao)], [o("o_proj.out", ob)])
            mid = ops.add(x, ob)
            (e_mid,) = op("attn_residual_add", layer, [e_in, i(e_o)], [o("resid.mid", mid)])
            m = ops.rms_norm(mid, weights[("mlp_norm.w", layer)]["t"], eps, out_dtype=f32)
            (e_m,) = op("mlp_norm", layer, [i(e_mid), w("mlp_norm.w", layer)], [o("mlp_norm.f32", m)])
            m16 = ops.cast(m, f16)
            (e_m16,) = op("mlp_norm.cast", layer, [i(e_m)], [o("mlp_norm.out", m16)])
            gate, up = linear(m16, "gate", f32), linear(m16, "up", f32)
            if n <= 32:
                e_g, e_u = op("gate_up", layer, [i(e_m16)], [o("gate", gate), o("up", up)])
            else:
                (e_g,) = op("gate_proj", layer, [i(e_m16)], [o("gate", gate)])
                (e_u,) = op("up_proj", layer, [i(e_m16)], [o("up", up)])
            s = ops.swiglu(gate, up, f32)
            (e_s,) = op("swiglu", layer, [i(e_g), i(e_u)], [o("swiglu.f32", s)])
            s16 = ops.cast(s, f16)
            (e_s16,) = op("swiglu.cast", layer, [i(e_s)], [o("swiglu.out", s16)])
            d = linear(s16, "down", f32)
            (e_d,) = op("down_proj", layer, [i(e_s16)], [o("down_proj.out", d)])
            x = ops.add(mid, d)
            (e_x,) = op("mlp_residual_add", layer, [i(e_mid), i(e_d)], [o("resid.out", x)])
        fn = ops.rms_norm(x, weights[("final_norm.w", -1)]["t"], eps, out_dtype=f32)
        (e_fn,) = op("final_norm", -1, [i(e_x), w("final_norm.w", -1)], [o("final_norm.f32", fn)])
        fn16 = ops.cast(fn, f16)
        (e_fn16,) = op("final_norm.cast", -1, [i(e_fn)], [o("final_norm.out", fn16)])
        logits = linear(fn16, "head", f16)
        op("lm_head", -1, [i(e_fn16)], [o("logits", logits)])
        logits_rows.append(logits.float().cpu().numpy())
        blob.close()
        (out / str(PREFIX) / f"{phase}.json").write_text(json.dumps(
            {"prefix": PREFIX, "phase": phase, "rows": n, "past": past, "npad": npad, "ops": ops_list}))
    control = out / "uninstrumented"   # as llmp_exl3_exec writes an uninstrumented run's --out
    control.mkdir()
    np.save(control / f"logits-{PREFIX}.prefill.npy", logits_rows[0])
    np.save(control / f"logits-{PREFIX}.suffix.npy", np.concatenate(logits_rows[1:]))
    (control / "manifest.json").write_text(json.dumps(
        {"harness": "llmp_exl3_exec", "fixture": "4.0bpw", "arm": "G", "artifact": artifact_dir.name,
         "capture": False, "record_ops": False, "evaluations": 2, "mismatches": 0}))


# --------------------------------------------------------------------------- alterations


def phase_doc(root, phase):
    path = root / str(PREFIX) / f"{phase}.json"
    return path, json.loads(path.read_text())


def find(doc, name, layer):
    return next(o for o in doc["ops"] if o["op"] == name and o["layer"] == layer)


def rewire(doc, tensor, layer, old, new):
    """Every input named tensor in layer (or anywhere, layer None) with hash old gets hash new."""
    for o in doc["ops"]:
        for t in o["inputs"]:
            if t["name"] == tensor and t["sha256"] == old and (layer is None or o["layer"] == layer):
                t["sha256"] = new


def flip(root, phase, entry, rehash):
    blob = root / str(PREFIX) / f"{phase}.bin"
    data = bytearray(blob.read_bytes())
    data[entry["offset"] + 5] ^= 0x01
    blob.write_bytes(bytes(data))
    if rehash:
        old, entry["sha256"] = entry["sha256"], sha(bytes(data[entry["offset"]:entry["offset"] + entry["bytes"]]))
        return old
    return None


def alter_flip(root):
    path, doc = phase_doc(root, 0)
    out = find(doc, "rope_q", 5)["outputs"][0]
    old = flip(root, 0, out, True)
    rewire(doc, "q_rope", 5, old, out["sha256"])
    path.write_text(json.dumps(doc))


def alter_corrupt(root):
    path, doc = phase_doc(root, 3)
    flip(root, 3, find(doc, "attn_norm", 2)["outputs"][0], False)


def alter_miswire(root):
    path, doc = phase_doc(root, 2)
    norm = find(doc, "mlp_norm", 7)
    norm["inputs"][0]["sha256"] = find(doc, "attn_norm", 7)["inputs"][0]["sha256"]   # resid.in's
    path.write_text(json.dumps(doc))


def alter_dtype(root):
    path, doc = phase_doc(root, 0)
    find(doc, "swiglu.cast", 3)["outputs"][0]["dtype"] = "bfloat16"
    find(doc, "down_proj", 3)["inputs"][0]["dtype"] = "bfloat16"
    path.write_text(json.dumps(doc))


def alter_missing_layer(root):
    path, doc = phase_doc(root, 4)
    doc["ops"] = [o for o in doc["ops"] if o["layer"] != 12]
    path.write_text(json.dumps(doc))


def alter_stale_k(root):
    path, doc = phase_doc(root, 5)
    k = find(doc, "attention", 9)["inputs"][1]
    k0 = find(json.loads((root / str(PREFIX) / "0.json").read_text()), "kv_write.k", 9)["outputs"][0]
    # The layer's cells as op_tier_e must build them, with cell 3 (from the prefill) zeroed.
    cells = np.zeros(k["shape"], "<f2")
    for phase in range(6):
        d = json.loads((root / str(PREFIX) / f"{phase}.json").read_text()) if phase else None
        e = find(d, "kv_write.k", 9)["outputs"][0] if phase else k0
        data = (root / str(PREFIX) / f"{phase}.bin").read_bytes()[e["offset"]:e["offset"] + e["bytes"]]
        lo, hi = e["cells"]
        cells[lo:hi] = np.frombuffer(data, "<f2").reshape(hi - lo, *k["shape"][1:])
    assert sha(cells.tobytes()) == k["sha256"]
    cells[3] = 0
    k["sha256"] = sha(cells.tobytes())
    path.write_text(json.dumps(doc))


def alter_mask(root):
    path, doc = phase_doc(root, 0)
    out = find(doc, "inputs", -1)["outputs"][2]
    blob = root / str(PREFIX) / "0.bin"
    data = bytearray(blob.read_bytes())
    data[out["offset"] + 2:out["offset"] + 4] = np.float16(0).tobytes()   # row 0 sees column 1
    blob.write_bytes(bytes(data))
    old, out["sha256"] = out["sha256"], sha(bytes(data[out["offset"]:out["offset"] + out["bytes"]]))
    rewire(doc, "mask", None, old, out["sha256"])
    path.write_text(json.dumps(doc))


def alter_cells(root):
    path, doc = phase_doc(root, 6)
    find(doc, "kv_write.k", 0)["outputs"][0]["cells"][1] += 1
    path.write_text(json.dumps(doc))


def alter_norm_weight(root):
    path, doc = phase_doc(root, 1)
    find(doc, "attn_norm", 3)["inputs"][1]["sha256"] = find(doc, "attn_norm", 4)["inputs"][1]["sha256"]
    path.write_text(json.dumps(doc))


def alter_linear_weight(root):
    path, doc = phase_doc(root, 0)
    find(doc, "q_proj", 3)["weights"][0]["sha256"] = find(doc, "q_proj", 4)["weights"][0]["sha256"]
    path.write_text(json.dumps(doc))


def alter_stale_table(root):
    """gate_up's table still points up's trellis at the next layer's (a rewrite that never happened)."""
    path, doc = phase_doc(root, 2)
    table = next(w for w in find(doc, "gate_up", 6)["weights"] if w["name"] == "table.trellis[1]")
    table["sha256"] = next(w for w in find(doc, "gate_up", 7)["weights"] if w["name"] == "up_proj.trellis")["sha256"]
    path.write_text(json.dumps(doc))


def alter_no_weights(root):
    path, doc = phase_doc(root, 0)
    del find(doc, "down_proj", 0)["weights"]
    path.write_text(json.dumps(doc))


def alter_logits(root):
    path = root / "uninstrumented" / f"logits-{PREFIX}.suffix.npy"
    logits = np.load(path)
    logits[7, 1000] += 0.5
    np.save(path, logits)


def alter_instrumented(root):
    path = root / "uninstrumented" / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "record_ops": True}))


def alter_no_control(root):
    shutil.rmtree(root / "uninstrumented")


CASES = (("exact", None, 0, None), ("flipped bit", alter_flip, 1, "prefix 32 phase 0 layer 5 op rope_q tensor q_rope"),
         ("corrupt bytes", alter_corrupt, 2, None),
         ("mis-wired input", alter_miswire, 1, "prefix 32 phase 2 layer 7 op mlp_norm tensor resid.mid"),
         ("wrong dtype", alter_dtype, 1, "prefix 32 phase 0 layer 3 op swiglu.cast tensor swiglu.out"),
         ("missing layer", alter_missing_layer, 2, None),
         ("stale K cell", alter_stale_k, 1, "prefix 32 phase 5 layer 9 op attention tensor k_cache"),
         ("wrong mask", alter_mask, 1, "prefix 32 phase 0 layer -1 op inputs tensor mask"),
         ("wrong cells", alter_cells, 1, "prefix 32 phase 6 layer 0 op kv_write.k tensor k_cache"),
         ("norm weight", alter_norm_weight, 1, "prefix 32 phase 1 layer 3 op attn_norm tensor attn_norm.w"),
         ("linear weight", alter_linear_weight, 1, "prefix 32 phase 0 layer 3 op q_proj tensor trellis"),
         ("stale table", alter_stale_table, 1, "prefix 32 phase 2 layer 6 op gate_up tensor table.trellis[1]"),
         ("no weights", alter_no_weights, 2, None),
         ("logits", alter_logits, 1, "prefix 32 phase 8 layer -1 op lm_head tensor logits"),
         ("instrumented", alter_instrumented, 2, None), ("no control", alter_no_control, 2, None))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--lib", required=True)
    parser.add_argument("--record", type=Path, default=HERE / "exl3-op-plan-g.json")
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    base = args.work / "recording"
    if not (base / "manifest.json").exists():
        generate(base, args.artifact.resolve(), args.ids, args.lib)
    results = []
    for name, alter, want_code, want_first in CASES:
        root = args.work / "case"
        shutil.rmtree(root, ignore_errors=True)
        shutil.copytree(base, root)
        if alter:
            alter(root)
        print(f"=== {name}", flush=True)
        captured = []
        original = op_tier_e.Check.fail

        def fail(self, where, what, _captured=captured):
            _captured.append(where)
            return original(self, where, what)
        op_tier_e.Check.fail = fail
        try:
            code = op_tier_e.check(root, args.artifact, args.ids, args.record, root / "uninstrumented", args.lib,
                                   prefixes=(PREFIX,))
        finally:
            op_tier_e.Check.fail = original
        first = captured[0] if captured else None
        ok = code == want_code and (want_first is None or first == want_first)
        results.append((name, ok, code, first))
        print(f"=== {name}: exit {code} (want {want_code}), first {first}: {'as expected' if ok else 'UNEXPECTED'}",
              flush=True)
    for name, ok, code, first in results:
        print(f"{'ok  ' if ok else 'FAIL'} {name}: exit {code}, first {first}")
    return 0 if all(r[1] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
