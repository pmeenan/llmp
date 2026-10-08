#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Builds the artifact reader's test corpus with the M0 prototype, the v0 format's oracle (D-056).

    artifact_corpus.py OUT          the corpus, rebuilt from scratch under OUT
    artifact_corpus.py view ARTIFACT  one artifact's view (below), as the C++ test prints it

Everything comes from docs/experiments/artifact-layout/layout.py: the prototype
builds small artifacts from synthetic GGUF and safetensors checkpoints, and its
verifier decides every mutated case. OUT holds:

    types.txt                 the prototype's GGML type, dtype and native-type tables
    golden/<name>/<id>/       valid artifacts
    golden/<name>.view        the prototype's view of each: shards, groups, chunks,
                              resources, slices, row lookups and read plans
    cases/<case>/<id>/        one mutation each of a valid artifact
    cases.txt                 "<case> <id> <verdict>": the prototype's verdict when a loader
                              opens it (verify with deep=False), "ok" or the rule it names
    fuzz/<n>.manifest, fuzz/<n>.index
                              seeded random mutations of a valid artifact's documents (the
                              index only where it changed), then fixed probes of the reader's
                              two known divergences; the rest of the artifact is its golden
                              copy, and its name is the manifest's digest
    fuzz.txt                  "<n> <base> <verdict> <unread>": the prototype's verdict on each,
                              and its verdict when kept .kv.gguf contents are not read (as
                              the reader does not read them; the test requires them the same
                              unless <verdict> is meta)

Each case names the rule it is meant to break, and the build fails if the
prototype disagrees, so the corpus cannot silently stop covering a rule.
Fuzz cases name no rule: whatever the prototype says is the verdict the
reader must match.
Cases change file contents only; links, hard links and special files are
made by the C++ test itself, since a deployed copy of the build tree would
not keep them.
"""

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pathlib  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402

PROTOTYPE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "experiments" / "artifact-layout"
sys.path.insert(0, str(PROTOTYPE))
import layout as L  # noqa: E402
from test_layout import KV, gguf_bytes, rand, st_bytes, toy_tensors  # noqa: E402


# ---------------------------------------------------------------- sources


def tiny_tensors():
    """A small MoE: 2 layers, 2 experts, a tied F16 embedding and output.
    Q4_K gate slices (ne0 = 256) carry GGML's over-read; Q6_K down slices
    (ne0 = 512) do not."""
    emb = rand(2 * 256 * 8, 1)
    tensors = [("token_embd.weight", [256, 8], 1, emb)]
    for layer in range(2):
        tensors += [
            (f"blk.{layer}.attn_norm.weight", [256], 0, rand(1024, 10 + layer)),
            (f"blk.{layer}.ffn_gate_inp.weight", [256, 2], 0, rand(2048, 30 + layer)),
            (f"blk.{layer}.ffn_gate_exps.weight", [256, 4, 2], 12, rand(144 * 4 * 2, 40 + layer)),
            (f"blk.{layer}.ffn_down_exps.weight", [512, 4, 2], 14, rand(210 * 2 * 4 * 2, 50 + layer)),
        ]
    return tensors + [("output_norm.weight", [256], 0, rand(1024, 60)), ("output.weight", [256, 8], 1, emb)]


def exl3_entries():
    """An EXL3-style safetensors checkpoint: a BF16 row table, plain norms and
    two trellis linears (K = 4 and K = 8) with their side vectors and markers."""
    out = [("model.embed_tokens.weight", "BF16", [64, 32], rand(64 * 32 * 2, 1)),
           ("model.layers.0.input_layernorm.weight", "F16", [32], rand(64, 2)),
           ("model.norm.weight", "F16", [32], rand(64, 3))]
    for prefix, k, seed in (("model.layers.0.mlp.up_proj", 4, 10), ("lm_head", 8, 20)):
        out += [(f"{prefix}.trellis", "I16", [2, 4, 16 * k], rand(2 * 4 * 16 * k * 2, seed)),
                (f"{prefix}.suh", "F16", [32], rand(64, seed + 1)),
                (f"{prefix}.svh", "F16", [64], rand(128, seed + 2)),
                (f"{prefix}.mcg", "I32", [], rand(4, seed + 3))]
    return out


def build_gguf(work, name, kv, tensors, shard_target):
    src_dir = work / "src" / name
    src_dir.mkdir(parents=True)
    path = src_dir / f"{name}.gguf"
    path.write_bytes(gguf_bytes(kv, tensors))
    src = L.load_sources([str(path)])
    plan = L.plan(src, tie_check=True, shard_target=shard_target)
    return L.build(plan, src, work / "installed" / name, [str(path)])


def build_exl3(work, name):
    src_dir = work / "src" / name
    src_dir.mkdir(parents=True)
    (src_dir / "config.json").write_text('{"model_type": "qwen2"}')
    path = src_dir / "model.safetensors"
    path.write_bytes(st_bytes(exl3_entries()))
    src = L.load_sources([str(path)])
    plan = L.plan(src, tie_check=True)
    return L.build(plan, src, work / "installed" / name, [str(path)],
                   meta_files=[("meta/config.json", str(src_dir / "config.json"))])


# ---------------------------------------------------------------- views


def repr_text(rep):
    if rep["family"] == "ggml":
        return f"ggml:{rep['type']}:{'x'.join(map(str, rep['ne']))}"
    text = f"{rep['family']}:{rep.get('role', '-')}:{rep['dtype']}:{'x'.join(map(str, rep['shape']))}"
    if rep.get("role") == "trellis":
        text += f":k{rep['k_bits']}:in{rep['in_features']}:out{rep['out_features']}:{rep['codebook']}"
    return text


def dash(v):
    return "-" if v is None else str(v)


def view(root):
    """The reader's view of a verified artifact, one fact per line."""
    manifest, index = L.verify(root, deep=False)
    lines = [f"artifact {pathlib.Path(root).name}"]
    m = manifest["model"]
    lines.append(f"model {m['architecture']} {m['expert_count']} {','.join(m['representation'])}")
    for s in manifest["source"]:
        lines.append(f"source {s['name']} {s['bytes']} {s['sha256']}")
    for f in manifest["files"]:
        lines.append(f"file {f['path']} {f['role']} {f['bytes']} {f['sha256']}")
    for i, s in enumerate(index["shards"]):
        lines.append(f"shard {i} {s['path']} {s['data_offset']} {s['data_bytes']} {s['header_sha256']}")
    everything = []
    for gid, g in enumerate(index["groups"]):
        chunks = -(-g["stored_bytes"] // L.CHUNK)
        lines.append(f"group {gid} {g['kind']} {dash(g['layer'])} {dash(g['expert'])} {g['shard']} "
                     f"{g['offset']} {g['used_bytes']} {g['stored_bytes']} {g['first_chunk']} {chunks}")
        for k in range(chunks):
            path, off, n = L.chunk_range(index, gid, k)
            lines.append(f"chunk {gid} {k} {path} {off} {n} {index['chunk_sha256'][g['first_chunk'] + k]}")
            everything.append((gid, k))
    for r in index["resources"]:
        c = L.closure(index, r["group"], r["offset"], r["readable_bytes"])
        lines.append(f"resource {r['name']} {','.join(r['roles'])} {r['group']} {r['offset']} {r['bytes']} "
                     f"{r['readable_bytes']} {r.get('access', '-')} {repr_text(r['repr'])} {c[0]} {c[-1]}")
        if r.get("access") == "rows":
            n_rows, row_bytes = L._row_geometry(r)
            rows = sorted({0, n_rows // 2, n_rows - 1})
            keys = L.rows_to_chunks(index, r["name"], rows)
            lines.append(f"rows {r['name']} {n_rows} {row_bytes} {','.join(map(str, rows))} "
                         f"{','.join(f'{g}:{k}' for g, k in keys)}")
    for a in index["expert_arrays"]:
        lines.append(f"array {a['name']} {a['layer']} {a['count']} {a['first_group']} {a['group_offset']} "
                     f"{a['slice_bytes']} {a['readable_bytes']} {repr_text(a['repr'])}")
        for e in range(a["count"]):
            c = L.closure(index, a["first_group"] + e, a["group_offset"], a["readable_bytes"])
            lines.append(f"slice {a['name']} {e} {a['first_group'] + e} {c[0]} {c[-1]}")
    # Read plans: every chunk missing, under three limits, then every other
    # chunk missing with the rest resident.
    plans = [("all", everything, (), L.MAX_RUN, 1024), ("all", everything, (), 4 << 20, 3),
             ("all", everything, (), 1, 1), ("alternate", everything[::2], everything[1::2], L.MAX_RUN, 1024)]
    for label, missing, resident, max_run, max_segments in plans:
        for path, off, length, segs in L.coalesce(index, missing, set(resident), max_run, max_segments):
            lines.append(f"run {label} {max_run} {max_segments} {path} {off} {length} "
                         + " ".join(f"{g}:{k}:{n}" for (g, k), n in segs))
    return "\n".join(lines) + "\n"


def types_text():
    lines = [f"ggml {name} {tid} {blck} {size}" for tid, (name, blck, size) in sorted(L.GGML_TYPES.items())]
    lines += [f"dtype {name} {size}" for name, size in sorted(L.ST_SIZES.items())]
    lines += [f"native {name}" for name in sorted(L.ST_NATIVE)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- mutations


class Case:
    """One copy of a valid artifact, mutated so that (ideally) one rule fires.
    Helpers re-hash what they change and re-list every file, then rename the
    directory to the new manifest digest, so only the rule under test can fire."""

    def __init__(self, root):
        self.a = root

    def load(self, rel):
        return json.loads((self.a / rel).read_bytes())

    def publish(self, mbytes):
        (self.a / "manifest.json").write_bytes(mbytes)
        new = self.a.with_name(hashlib.sha256(mbytes).hexdigest())
        if new != self.a:
            os.rename(self.a, new)
            self.a = new

    def relist(self, mutate=None, raw=None):
        m = self.load("manifest.json")
        for f in m["files"]:
            if (self.a / f["path"]).exists():
                data = (self.a / f["path"]).read_bytes()
                f["bytes"], f["sha256"] = len(data), hashlib.sha256(data).hexdigest()
        if mutate:
            mutate(m)
        self.publish(raw(m) if raw else L.dumps(m))

    def index(self, fn=None, raw=None):
        doc = self.load("index.json")
        if fn:
            fn(doc)
        (self.a / "index.json").write_bytes(raw(doc) if raw else L.dumps(doc))
        self.relist()

    def shard_header(self, fn, shard=0):
        """Rewrites a shard's JSON header in place (same length) and fixes its digest."""
        path = self.a / f"data/{shard:05d}.safetensors"
        data = bytearray(path.read_bytes())
        (n,) = struct.unpack("<Q", data[:8])
        text = fn(bytes(data[8:8 + n]).rstrip(b" "))
        assert len(text) <= n, "a header rewrite must keep its length"
        data[8:8 + n] = text + b" " * (n - len(text))
        path.write_bytes(data)
        off = self.load("index.json")["shards"][shard]["data_offset"]
        digest = hashlib.sha256(bytes(data[:off])).hexdigest()
        self.index(lambda i: i["shards"][shard].update(header_sha256=digest))


def res(doc, name):
    return next(r for r in doc["resources"] if r["name"] == name)


def swap(items, a, b):
    items[a], items[b] = items[b], items[a]


# A trellis whose in_features (16 * 2^59 = 2^63) is beyond 64 signed bits:
# a consistent descriptor, so the rule that refuses it is the next one.
HUGE_TRELLIS = {"family": "exl3", "role": "trellis", "dtype": "I16", "shape": [2**59, 4, 64], "k_bits": 4,
                "in_features": 2**63, "out_features": 64, "codebook": "mcg"}


def replace_once(old, new):
    def f(m):
        data = L.dumps(m)
        assert old in data, old
        return data.replace(old, new, 1)
    return f


# (case, base, intended verdict, mutation). Bases: "tiny" (GGUF MoE, several
# shards, tied embedding), "exl3" (safetensors, EXL3 parts, kept config.json).
CASES = [
    ("valid-tiny", "tiny", "ok", lambda c: None),
    ("valid-exl3", "exl3", "ok", lambda c: None),
    # versions, profile, identity, encoding
    ("version-1", "tiny", "unsupported-version", lambda c: c.relist(lambda m: m.update(format_version=1))),
    ("version-bool", "tiny", "unsupported-version", lambda c: c.relist(lambda m: m.update(format_version=False))),
    ("version-string", "tiny", "unsupported-version", lambda c: c.relist(lambda m: m.update(format_version="0"))),
    ("version-big", "tiny", "unsupported-version",
     lambda c: c.relist(lambda m: m.update(format_version=2**64))),
    ("profile-other", "tiny", "unsupported-profile",
     lambda c: c.relist(lambda m: m["layout"].update(chunk_bytes=1 << 20))),
    ("profile-extra-key", "tiny", "unsupported-profile", lambda c: c.relist(lambda m: m["layout"].update(x=1))),
    ("profile-bool", "tiny", "unsupported-profile",
     lambda c: c.relist(lambda m: m["layout"].update(profile_version=False))),
    ("profile-float", "tiny", "json",
     lambda c: c.relist(raw=replace_once(b'"file_alignment":4096', b'"file_alignment":4096.0'))),
    ("format-other", "tiny", "format", lambda c: c.relist(lambda m: m.update(format="llmp-other"))),
    ("format-not-object", "tiny", "format", lambda c: c.relist(raw=lambda m: L.dumps([m]))),
    ("manifest-indented", "tiny", "canonical",
     lambda c: c.relist(raw=lambda m: json.dumps(m, sort_keys=True, indent=1).encode())),
    ("manifest-unsorted", "tiny", "canonical",
     lambda c: c.relist(raw=lambda m: (json.dumps(dict(reversed(list(m.items()))), separators=(",", ":"))
                                       + "\n").encode())),
    ("manifest-no-newline", "tiny", "canonical", lambda c: c.relist(raw=lambda m: L.dumps(m)[:-1])),
    ("manifest-two-newlines", "tiny", "canonical", lambda c: c.relist(raw=lambda m: L.dumps(m) + b"\n")),
    ("manifest-negative-zero", "tiny", "canonical",
     lambda c: c.relist(raw=replace_once(b'"profile_version":0', b'"profile_version":-0'))),
    ("manifest-escape", "tiny", "canonical",
     lambda c: c.relist(raw=replace_once(b'"name":"artifact-layout/layout.py"',
                                         b'"name":"artifact\\u002dlayout/layout.py"'))),
    ("manifest-escaped-newline", "tiny", "schema",
     lambda c: c.relist(raw=replace_once(b'"version":"m0-prototype"', b'"version":"m0\\nprototype"'))),
    ("manifest-del", "tiny", "canonical",
     lambda c: c.relist(raw=replace_once(b'"version":"m0-prototype"', b'"version":"m0\x7fprototype"'))),
    ("manifest-bom", "tiny", "json", lambda c: c.relist(raw=lambda m: b"\xef\xbb\xbf" + L.dumps(m))),
    ("manifest-utf16", "tiny", "json", lambda c: c.relist(raw=lambda m: L.dumps(m).decode().encode("utf-16"))),
    ("manifest-non-ascii", "tiny", "json",
     lambda c: c.relist(raw=replace_once(b'"version":"m0-prototype"', "\"version\":\"m0é\"".encode()))),
    ("manifest-control", "tiny", "json",
     lambda c: c.relist(raw=replace_once(b'"version":"m0-prototype"', b'"version":"m0\tp"'))),
    ("manifest-duplicate-key", "tiny", "json",
     lambda c: c.relist(raw=replace_once(b'{"converter"', b'{"format":"x","converter"'))),
    ("manifest-float", "tiny", "json",
     lambda c: c.relist(raw=replace_once(b'"expert_count":2', b'"expert_count":2.0'))),
    ("manifest-exponent", "tiny", "json",
     lambda c: c.relist(raw=replace_once(b'"expert_count":2', b'"expert_count":2e0'))),
    ("manifest-nan", "tiny", "json", lambda c: c.relist(raw=replace_once(b'"expert_count":2', b'"expert_count":NaN'))),
    ("manifest-leading-zero", "tiny", "json",
     lambda c: c.relist(raw=replace_once(b'"expert_count":2', b'"expert_count":02'))),
    ("manifest-long-integer", "tiny", "json",
     lambda c: c.relist(raw=replace_once(b'"expert_count":2', b'"expert_count":123456789012345678901'))),
    ("manifest-trailing-data", "tiny", "json", lambda c: c.relist(raw=lambda m: L.dumps(m) + b"{}")),
    ("manifest-trailing-comma", "tiny", "json",
     lambda c: c.relist(raw=replace_once(b'"expert_count":2', b'"expert_count":2,'))),
    ("manifest-deep", "tiny", "json",
     lambda c: c.relist(raw=replace_once(b'"expert_count":2', b'"expert_count":' + b"[" * 9 + b"]" * 9))),
    ("manifest-too-large", "tiny", "file-size", lambda c: c.relist(raw=lambda m: L.dumps(m) + b" " * (1 << 20))),
    ("identity", "tiny", "identity", lambda c: c.publish_as("0" * 64)),
    ("identity-uppercase", "tiny", "identity", lambda c: c.publish_as(c.a.name.upper())),
    # manifest schema
    ("manifest-extra-key", "tiny", "schema", lambda c: c.relist(lambda m: m.update(extra=1))),
    ("manifest-missing-key", "tiny", "schema", lambda c: c.relist(lambda m: m.pop("converter"))),
    ("experimental-false", "tiny", "schema", lambda c: c.relist(lambda m: m.update(experimental=False))),
    ("experimental-one", "tiny", "schema", lambda c: c.relist(lambda m: m.update(experimental=1))),
    ("model-extra-key", "tiny", "schema", lambda c: c.relist(lambda m: m["model"].update(extra=1))),
    ("model-architecture", "tiny", "schema", lambda c: c.relist(lambda m: m["model"].update(architecture="a b"))),
    ("model-architecture-long", "tiny", "schema",
     lambda c: c.relist(lambda m: m["model"].update(architecture="a" * 201))),
    ("model-expert-count-big", "tiny", "schema",
     lambda c: c.relist(lambda m: m["model"].update(expert_count=(1 << 20) + 1))),
    ("model-expert-count-negative", "tiny", "schema",
     lambda c: c.relist(lambda m: m["model"].update(expert_count=-1))),
    ("model-representation-empty", "tiny", "schema",
     lambda c: c.relist(lambda m: m["model"].update(representation=[]))),
    ("model-representation-unsorted", "exl3", "schema",
     lambda c: c.relist(lambda m: m["model"].update(representation=["plain", "exl3"]))),
    ("model-representation-unknown", "tiny", "schema",
     lambda c: c.relist(lambda m: m["model"].update(representation=["ggml", "onnx"]))),
    ("model-representation-mismatch", "tiny", "schema",
     lambda c: c.relist(lambda m: m["model"].update(representation=["ggml", "plain"]))),
    ("source-empty", "tiny", "schema", lambda c: c.relist(lambda m: m.update(source=[]))),
    ("source-bytes-string", "tiny", "schema", lambda c: c.relist(lambda m: m["source"][0].update(bytes="1"))),
    ("source-sha-upper", "tiny", "schema",
     lambda c: c.relist(lambda m: m["source"][0].update(sha256=m["source"][0]["sha256"].upper()))),
    ("source-duplicate", "tiny", "canonical", lambda c: c.relist(lambda m: m["source"].append(dict(m["source"][0])))),
    ("source-unsorted", "exl3", "canonical", lambda c: c.relist(lambda m: m["source"].reverse())),
    ("transformation-kind", "tiny", "schema",
     lambda c: c.relist(lambda m: m["transformations"].append({"kind": "requantize"}))),
    ("transformation-not-object", "tiny", "schema", lambda c: c.relist(lambda m: m["transformations"].append(5))),
    ("transformation-keys", "tiny", "schema",
     lambda c: c.relist(lambda m: next(t for t in m["transformations"] if t["kind"] == "dedupe-identical").pop(
         "sha256"))),
    ("transformation-count-zero", "tiny", "schema",
     lambda c: c.relist(lambda m: next(t for t in m["transformations"] if t["kind"] == "expert-slice").update(count=0))),
    ("transformation-unsorted", "tiny", "canonical", lambda c: c.relist(lambda m: m["transformations"].reverse())),
    ("transformation-missing-slice", "tiny", "schema",
     lambda c: c.relist(lambda m: m.update(transformations=[t for t in m["transformations"]
                                                             if t.get("tensor") != "blk.0.ffn_gate_exps.weight"]))),
    ("transformation-missing-dedupe", "tiny", "schema",
     lambda c: c.relist(lambda m: m.update(transformations=[t for t in m["transformations"]
                                                             if t["kind"] != "dedupe-identical"]))),
    ("transformation-extra-dedupe", "exl3", "schema",
     lambda c: c.relist(lambda m: m["transformations"].append(
         {"kind": "dedupe-identical", "resource": "model.norm.weight", "role": "x", "sha256": "0" * 64}))),
    ("converter-keys", "tiny", "schema", lambda c: c.relist(lambda m: m["converter"].update(build="x"))),
    ("converter-type", "tiny", "schema", lambda c: c.relist(lambda m: m.update(converter=5))),
    ("converter-empty", "tiny", "schema", lambda c: c.relist(lambda m: m["converter"].update(name=""))),
    ("files-one", "tiny", "schema", lambda c: c.relist(lambda m: m.update(files=m["files"][:1]))),
    ("files-bytes-negative", "tiny", "schema", lambda c: c.relist(lambda m: m["files"][0].update(bytes=-1))),
    ("files-role-unknown", "tiny", "path", lambda c: c.relist(lambda m: m["files"][0].update(role="blob"))),
    ("files-second-index", "tiny", "path", lambda c: c.relist(lambda m: m["files"][-1].update(role="index"))),
    ("files-shard-path", "tiny", "path",
     lambda c: c.relist(lambda m: m["files"][0].update(path="data/0.safetensors"))),
    ("files-meta-path", "exl3", "path",
     lambda c: c.relist(lambda m: next(f for f in m["files"] if f["role"] == "source-metadata").update(
         path="meta/../config.json"))),
    ("files-unsorted", "tiny", "canonical", lambda c: c.relist(lambda m: m["files"].reverse())),
    ("files-no-index", "tiny", "file-set",
     lambda c: c.relist(lambda m: m.update(files=[f for f in m["files"] if f["role"] != "index"]))),
    # the file set, sizes and kept metadata
    ("unlisted-file", "tiny", "file-set", lambda c: (c.a / "meta" / "extra.txt").write_text("x")),
    ("unexpected-directory", "tiny", "file-set", lambda c: (c.a / "junk").mkdir()),
    ("nested-directory", "tiny", "file-set", lambda c: (c.a / "data" / "sub").mkdir()),
    ("listed-missing", "exl3", "file-set", lambda c: c.remove_listed("meta/config.json")),
    ("size-mismatch", "tiny", "file-size", lambda c: c.append("data/00000.safetensors", b"\0")),
    ("meta-not-a-source", "exl3", "file-set",
     lambda c: c.relist(lambda m: next(s for s in m["source"] if s["name"] == "config.json").update(
         sha256="0" * 64))),
    ("meta-kv-without-gguf", "tiny", "file-set",
     lambda c: c.relist(lambda m: m["source"][0].update(name="other.gguf"))),
    ("meta-empty-kv", "tiny", "meta", lambda c: c.write_listed("meta/tiny.kv.gguf", b"")),
    # A manifest expert count that disagrees with the expert arrays: the
    # prototype first finds it disagreeing with the kept GGUF metadata; the
    # reader, which does not read that, with the index (expert-array). A
    # dense model's count is compared with nothing, and slices are bounded by
    # each array's own count.
    ("manifest-expert-count", "tiny", "meta", lambda c: c.relist(lambda m: m["model"].update(expert_count=3))),
    ("manifest-expert-count-dense", "exl3", "ok", lambda c: c.relist(lambda m: m["model"].update(expert_count=3))),
    # index.json: document
    ("index-hash", "tiny", "hash", lambda c: c.write("index.json", c.read("index.json").replace(b"head", b"heae"))),
    ("index-indented", "tiny", "canonical", lambda c: c.index(raw=lambda d: json.dumps(d, sort_keys=True, indent=1).encode())),
    ("index-truncated", "tiny", "json", lambda c: c.index(raw=lambda d: L.dumps(d)[:-100])),
    ("index-duplicate-key", "tiny", "json",
     lambda c: c.index(raw=lambda d: L.dumps(d).replace(b'{"chunk_bytes"', b'{"format":"x","chunk_bytes"', 1))),
    ("index-float", "tiny", "json",
     lambda c: c.index(raw=lambda d: L.dumps(d).replace(b'"chunk_bytes":2097152', b'"chunk_bytes":1e400', 1))),
    ("index-deep", "tiny", "json", lambda c: c.index(raw=lambda d: b"[" * 200000)),
    ("index-non-ascii", "tiny", "json", lambda c: c.index(raw=lambda d: L.dumps(d).replace(b"head", b"h\xc3\xa9", 1))),
    ("index-list", "tiny", "schema", lambda c: c.index(raw=lambda d: L.dumps([d]))),
    ("index-extra-key", "tiny", "schema", lambda c: c.index(lambda d: d.update(extra=1))),
    ("index-resources-null", "tiny", "schema", lambda c: c.index(lambda d: d.update(resources=None))),
    ("index-version", "tiny", "unsupported-version", lambda c: c.index(lambda d: d.update(format_version=1))),
    ("index-alignment", "tiny", "unsupported-version", lambda c: c.index(lambda d: d.update(file_alignment=8192))),
    ("index-format", "tiny", "unsupported-version", lambda c: c.index(lambda d: d.update(format="jitllm-artifact"))),
    # shards
    ("shards-empty", "tiny", "schema", lambda c: c.index(lambda d: d.update(shards=[]))),
    ("shard-keys", "tiny", "schema", lambda c: c.index(lambda d: d["shards"][0].pop("header_sha256"))),
    ("shard-path-order", "tiny", "file-set", lambda c: c.index(lambda d: swap(d["shards"], 0, 1))),
    ("shard-data-offset-small", "tiny", "schema", lambda c: c.index(lambda d: d["shards"][0].update(data_offset=0))),
    ("shard-data-offset-unaligned", "tiny", "bounds",
     lambda c: c.index(lambda d: d["shards"][0].update(data_offset=d["shards"][0]["data_offset"] + 1))),
    ("shard-data-bytes-unaligned", "tiny", "bounds",
     lambda c: c.index(lambda d: d["shards"][0].update(data_bytes=d["shards"][0]["data_bytes"] + 1))),
    ("shard-wrong-size", "tiny", "file-set",
     lambda c: c.index(lambda d: d["shards"][0].update(data_bytes=d["shards"][0]["data_bytes"] + 4096))),
    ("shard-header-sha", "tiny", "schema", lambda c: c.index(lambda d: d["shards"][0].update(header_sha256="x"))),
    ("shard-unlisted", "tiny", "file-set",
     lambda c: c.index(lambda d: d.update(shards=d["shards"][:-1]))),
    # groups
    ("groups-empty", "tiny", "schema", lambda c: c.index(lambda d: d.update(groups=[]))),
    ("group-keys", "tiny", "schema", lambda c: c.index(lambda d: d["groups"][0].update(extra=1))),
    ("group-kind", "tiny", "schema", lambda c: c.index(lambda d: d["groups"][0].update(kind="dense"))),
    ("group-layer-on-head", "tiny", "schema",
     lambda c: c.index(lambda d: next(g for g in d["groups"] if g["kind"] == "head").update(layer=0))),
    ("group-layer-bool", "tiny", "schema",
     lambda c: c.index(lambda d: next(g for g in d["groups"] if g["kind"] == "layer").update(layer=False))),
    ("group-expert-null", "tiny", "schema",
     lambda c: c.index(lambda d: next(g for g in d["groups"] if g["kind"] == "expert").update(expert=None))),
    ("group-expert-on-layer", "tiny", "schema",
     lambda c: c.index(lambda d: next(g for g in d["groups"] if g["kind"] == "layer").update(expert=0))),
    ("group-expert-duplicate", "tiny", "expert-array",
     lambda c: c.index(lambda d: [g for g in d["groups"] if g["kind"] == "expert"][1].update(expert=0))),
    ("group-shard-range", "tiny", "schema", lambda c: c.index(lambda d: d["groups"][-1].update(shard=99))),
    ("group-shard-order", "tiny", "bounds", lambda c: c.index(lambda d: d["groups"][-1].update(shard=0))),
    ("group-stored", "tiny", "bounds",
     lambda c: c.index(lambda d: d["groups"][0].update(stored_bytes=d["groups"][0]["stored_bytes"] + 4096))),
    ("group-stored-small", "tiny", "schema", lambda c: c.index(lambda d: d["groups"][0].update(stored_bytes=0))),
    ("group-offset", "tiny", "bounds", lambda c: c.index(lambda d: d["groups"][1].update(offset=4096 * 100))),
    ("group-first-chunk", "tiny", "bounds", lambda c: c.index(lambda d: d["groups"][1].update(first_chunk=7))),
    ("group-used-mismatch", "tiny", "bounds",
     lambda c: c.index(lambda d: d["groups"][0].update(used_bytes=d["groups"][0]["used_bytes"] - 1))),
    ("chunks-count", "tiny", "schema", lambda c: c.index(lambda d: d["chunk_sha256"].pop())),
    ("chunks-hex", "tiny", "schema", lambda c: c.index(lambda d: d["chunk_sha256"].__setitem__(0, "g" * 64))),
    # resources
    ("resource-keys", "tiny", "schema", lambda c: c.index(lambda d: d["resources"][0].update(extra=1))),
    ("resource-name", "tiny", "schema", lambda c: c.index(lambda d: d["resources"][0].update(name="a b"))),
    ("resource-reserved", "tiny", "schema",
     lambda c: c.index(lambda d: d["resources"][0].update(name="__metadata__", roles=["__metadata__"]))),
    ("resource-duplicate-name", "tiny", "schema",
     lambda c: c.index(lambda d: d["resources"][1].update(name=d["resources"][0]["name"],
                                                          roles=[d["resources"][0]["name"]]))),
    ("resource-group-range", "tiny", "schema", lambda c: c.index(lambda d: d["resources"][0].update(group=10**6))),
    ("resource-in-expert-group", "tiny", "expert-array",
     lambda c: c.index(lambda d: d["resources"][0].update(
         group=next(i for i, g in enumerate(d["groups"]) if g["kind"] == "expert")))),
    ("roles-first", "tiny", "schema", lambda c: c.index(lambda d: d["resources"][0].update(roles=["x"]))),
    ("roles-empty", "tiny", "schema", lambda c: c.index(lambda d: d["resources"][0].update(roles=[]))),
    ("roles-too-many", "tiny", "bounds",
     lambda c: c.index(lambda d: d["resources"][0].update(
         roles=[d["resources"][0]["name"]] + [f"alias{i}" for i in range(9)]))),
    ("roles-unsorted", "tiny", "canonical",
     lambda c: c.index(lambda d: d["resources"][0].update(roles=[d["resources"][0]["name"], "b", "a"]))),
    ("roles-reserved", "tiny", "schema",
     lambda c: c.index(lambda d: d["resources"][0].update(roles=[d["resources"][0]["name"], "__metadata__"]))),
    ("roles-twice", "tiny", "schema",
     lambda c: c.index(lambda d: d["resources"][0].update(roles=[d["resources"][0]["name"],
                                                                 d["resources"][1]["name"]]))),
    ("roles-name-array", "tiny", "schema",
     lambda c: c.index(lambda d: d["resources"][0].update(roles=[d["resources"][0]["name"],
                                                                 d["expert_arrays"][0]["name"]]))),
    ("access-other", "tiny", "schema", lambda c: c.index(lambda d: d["resources"][0].update(access="cols"))),
    ("resource-unsorted", "tiny", "canonical", lambda c: c.index(lambda d: swap(d["resources"], 0, 1))),
    ("resource-offset-misaligned", "tiny", "alignment",
     lambda c: c.index(lambda d: d["resources"][1].update(offset=d["resources"][1]["offset"] + 8))),
    ("resource-overlap", "tiny", "overlap",
     lambda c: c.index(lambda d: res(d, "blk.0.ffn_gate_inp.weight").update(
         offset=res(d, "blk.0.ffn_gate_inp.weight")["offset"] - 256))),
    ("resource-outside-group", "tiny", "bounds",
     lambda c: c.index(lambda d: res(d, "blk.0.ffn_gate_inp.weight").update(offset=1 << 20))),
    ("group-empty", "tiny", "bounds", lambda c: c.move_resources_out_of_group()),
    # representations
    ("repr-not-object", "tiny", "repr", lambda c: c.index(lambda d: d["resources"][0].update(repr=[]))),
    ("repr-family", "tiny", "repr", lambda c: c.index(lambda d: d["resources"][0]["repr"].update(family="onnx"))),
    ("repr-family-type", "tiny", "repr", lambda c: c.index(lambda d: d["resources"][0]["repr"].update(family=1))),
    ("repr-ggml-type", "tiny", "repr", lambda c: c.index(lambda d: d["resources"][0]["repr"].update(type="Q9_Z"))),
    ("repr-ggml-keys", "tiny", "schema", lambda c: c.index(lambda d: d["resources"][0]["repr"].update(extra=1))),
    ("repr-ggml-ne-missing", "tiny", "schema", lambda c: c.index(lambda d: d["resources"][0]["repr"].pop("ne"))),
    ("repr-ggml-ne-empty", "tiny", "schema", lambda c: c.index(lambda d: d["resources"][0]["repr"].update(ne=[]))),
    ("repr-ggml-ne-dims", "tiny", "repr",
     lambda c: c.index(lambda d: d["resources"][0]["repr"].update(ne=[256, 1, 1, 1, 1]))),
    ("repr-ggml-ne-zero", "tiny", "repr", lambda c: c.index(lambda d: d["resources"][0]["repr"].update(ne=[0]))),
    ("repr-ggml-ne-huge", "tiny", "repr",
     lambda c: c.index(lambda d: d["resources"][0]["repr"].update(ne=[2**32, 2**32 + 1]))),
    ("repr-ggml-block", "tiny", "repr",
     lambda c: c.index(lambda d: d["expert_arrays"][0]["repr"].update(ne=[255, 4]))),
    ("repr-bytes", "tiny", "repr", lambda c: c.index(lambda d: d["resources"][0].update(bytes=d["resources"][0]["bytes"] + 1))),
    ("repr-bytes-zero", "tiny", "schema", lambda c: c.index(lambda d: d["resources"][0].update(bytes=0))),
    ("repr-readable", "tiny", "repr",
     lambda c: c.index(lambda d: d["resources"][0].update(readable_bytes=d["resources"][0]["readable_bytes"] + 1))),
    ("rows-not-2d", "tiny", "repr",
     lambda c: c.index(lambda d: next(r for r in d["resources"] if r.get("access") == "rows")["repr"].update(
         ne=[256, 4, 2]))),
    ("exl3-k-bits", "exl3", "repr", lambda c: c.index(lambda d: c.trellis(d)["repr"].update(k_bits=7))),
    ("exl3-codebook", "exl3", "repr", lambda c: c.index(lambda d: c.trellis(d)["repr"].update(codebook="mul1"))),
    ("exl3-in-features", "exl3", "repr", lambda c: c.index(lambda d: c.trellis(d)["repr"].update(in_features=33))),
    ("exl3-role-other", "exl3", "repr",
     lambda c: c.index(lambda d: res(d, "model.layers.0.mlp.up_proj.suh")["repr"].update(role="su"))),
    ("exl3-trellis-keys", "exl3", "schema", lambda c: c.index(lambda d: c.trellis(d)["repr"].pop("codebook"))),
    ("exl3-plain-part", "exl3", "repr",
     lambda c: c.index(lambda d: res(d, "model.layers.0.mlp.up_proj.mcg").update(
         repr={"family": "plain", "dtype": "I32", "shape": []}))),
    ("exl3-plain-role-key", "exl3", "schema",
     lambda c: c.index(lambda d: res(d, "model.norm.weight")["repr"].update(role="trellis"))),
    ("exl3-dtype", "exl3", "repr", lambda c: c.index(lambda d: res(d, "model.norm.weight")["repr"].update(dtype="F17"))),
    ("exl3-shape-negative", "exl3", "repr",
     lambda c: c.index(lambda d: res(d, "model.norm.weight")["repr"].update(shape=[-32]))),
    ("exl3-empty", "exl3", "repr",
     lambda c: c.index(lambda d: res(d, "model.norm.weight")["repr"].update(shape=[0, 32]))),
    ("exl3-side-length", "exl3", "repr",
     lambda c: c.index(lambda d: res(d, "model.layers.0.mlp.up_proj.suh").update(
         repr={"family": "exl3", "role": "suh", "dtype": "F16", "shape": [16, 2]}))),
    ("exl3-closure", "exl3", "repr", lambda c: c.drop_resource("model.layers.0.mlp.up_proj.svh")),
    ("exl3-in-features-64-bit", "exl3", "schema",
     lambda c: c.index(lambda d: c.trellis(d).update(repr=dict(HUGE_TRELLIS), bytes="x"))),
    ("exl3-in-features-64-bit-bytes", "exl3", "repr",
     lambda c: c.index(lambda d: c.trellis(d).update(repr=dict(HUGE_TRELLIS)))),
    ("exl3-in-features-65-bit", "exl3", "schema",
     lambda c: c.index(lambda d: c.trellis(d).update(
         repr=dict(HUGE_TRELLIS, shape=[2**60, 4, 64], in_features=2**64), bytes="x"))),
    # expert arrays
    ("array-keys", "tiny", "schema", lambda c: c.index(lambda d: d["expert_arrays"][0].update(extra=1))),
    ("array-count", "tiny", "expert-array", lambda c: c.index(lambda d: d["expert_arrays"][0].update(count=1))),
    ("array-count-cap", "tiny", "bounds",
     lambda c: c.index(lambda d: d["expert_arrays"][0].update(count=(1 << 18) + 1))),
    ("array-first-group", "tiny", "schema",
     lambda c: c.index(lambda d: d["expert_arrays"][0].update(first_group=len(d["groups"])))),
    ("array-not-ggml", "tiny", "expert-array",
     lambda c: c.index(lambda d: d["expert_arrays"][0].update(repr={"family": "plain", "dtype": "U8", "shape": [576]}))),
    ("array-exl3-64-bit", "tiny", "expert-array",
     lambda c: c.index(lambda d: d["expert_arrays"][0].update(repr=dict(HUGE_TRELLIS)))),
    ("array-3d", "tiny", "expert-array",
     lambda c: c.index(lambda d: d["expert_arrays"][0]["repr"].update(ne=[256, 2, 2]))),
    ("array-slice-bytes", "tiny", "repr",
     lambda c: c.index(lambda d: d["expert_arrays"][0].update(slice_bytes=d["expert_arrays"][0]["slice_bytes"] + 1))),
    ("array-readable", "tiny", "repr",
     lambda c: c.index(lambda d: d["expert_arrays"][0].update(readable_bytes=d["expert_arrays"][0]["slice_bytes"]))),
    ("array-offset-misaligned", "tiny", "alignment",
     lambda c: c.index(lambda d: d["expert_arrays"][1].update(group_offset=d["expert_arrays"][1]["group_offset"] + 8))),
    ("array-overlap", "tiny", "overlap",
     lambda c: c.index(lambda d: d["expert_arrays"][1].update(group_offset=d["expert_arrays"][1]["group_offset"] - 256))),
    ("array-name-duplicate", "tiny", "schema",
     lambda c: c.index(lambda d: d["expert_arrays"][0].update(name=d["resources"][0]["name"]))),
    ("array-name-reserved", "tiny", "schema",
     lambda c: c.index(lambda d: d["expert_arrays"][0].update(name="__metadata__"))),
    ("array-unsorted", "tiny", "canonical", lambda c: c.index(lambda d: swap(d["expert_arrays"], 0, 1))),
    ("array-layer", "tiny", "expert-array", lambda c: c.index(lambda d: d["expert_arrays"][0].update(layer=1))),
    ("expert-group-uncovered", "tiny", "expert-array", lambda c: c.drop_layer_arrays(1)),
    # shard headers
    ("header-text", "tiny", "container",
     lambda c: c.shard_header(lambda t: t.replace(b'"dtype":"F32"', b'"dtype":"I32"', 1))),
    ("header-sha", "tiny", "hash", lambda c: c.index(lambda d: d["shards"][0].update(header_sha256="0" * 64))),
    ("header-data-offset", "tiny", "container", lambda c: c.grow_header()),
]


class Mutations(Case):
    def publish_as(self, name):
        new = self.a.with_name(name)
        os.rename(self.a, new)
        self.a = new

    def read(self, rel):
        return (self.a / rel).read_bytes()

    def write(self, rel, data):
        (self.a / rel).write_bytes(data)

    def append(self, rel, data):
        with open(self.a / rel, "ab") as f:
            f.write(data)

    def write_listed(self, rel, data):
        self.write(rel, data)
        self.relist()

    def remove_listed(self, rel):
        os.remove(self.a / rel)

    @staticmethod
    def trellis(d):
        return res(d, "model.layers.0.mlp.up_proj.trellis")

    def drop_resource(self, name):
        self.index(lambda d: d.update(resources=[r for r in d["resources"] if r["name"] != name]))

    def move_resources_out_of_group(self):
        """Moves layer 0's resources into layer 1's group at free offsets, so layer 0's group is empty."""
        def f(d):
            layer = [i for i, g in enumerate(d["groups"]) if g["kind"] == "layer"]
            g0, g1 = layer[0], layer[1]
            end = d["groups"][g1]["used_bytes"]
            for r in d["resources"]:
                if r["group"] == g0:
                    end = L.align(end, L.MEMBER_ALIGN)
                    r.update(group=g1, offset=end)
                    end += r["readable_bytes"]
            d["resources"].sort(key=lambda r: (r["group"], r["offset"]))
        self.index(f)

    def drop_layer_arrays(self, layer):
        def f(d):
            d["expert_arrays"] = [a for a in d["expert_arrays"] if a["layer"] != layer]
        self.index(f)
        self.relist(lambda m: m.update(transformations=[t for t in m["transformations"]
                                                         if not t.get("tensor", "").startswith(f"blk.{layer}.")]))

    def grow_header(self):
        """Moves shard 0's data section 4 KiB later: the header's length prefix
        and padding change, the index follows, the header no longer is the canonical one."""
        path = self.a / "data/00000.safetensors"
        data = bytearray(path.read_bytes())
        (n,) = struct.unpack("<Q", data[:8])
        data[:8] = struct.pack("<Q", n + 4096)
        data[8 + n:8 + n] = b" " * 4096
        path.write_bytes(data)
        off = n + 8 + 4096

        def f(d):
            d["shards"][0].update(data_offset=off, header_sha256=hashlib.sha256(bytes(data[:off])).hexdigest())
        self.index(f)


def verdict(root):
    try:
        L.verify(root, deep=False)
    except L.ArtifactError as e:
        return e.code
    return "ok"


def verdict_unread(root):
    """The prototype's verdict with kept .kv.gguf contents unread, as the
    reader leaves them: what it would say past its `meta` checks."""
    check = L._check_kv_gguf
    L._check_kv_gguf = lambda data, where: {}
    try:
        return verdict(root)
    finally:
        L._check_kv_gguf = check


# ---------------------------------------------------------------- fuzz

FUZZ_CASES = 800
FUZZ_NUMBERS = [0, 1, -1, 7, 32, 255, 256, 257, 4095, 4096, 4097, 2097151, 2097152, 2097153, 2**31, 2**32,
                2**32 + 1, 2**59, 2**63 - 1, 2**63, 2**64 - 1, 2**64, 99999999999999999999]
FUZZ_STRINGS = ["", "x", "__metadata__", "~pad.0", "mcg", "trellis", "suh", "rows", "ggml", "exl3", "plain",
                "U8", "Q4_K", "F16", "I16", "I32", "expert", "layer", "head", "table", "global", "toy",
                "index", "shard", "source-metadata", "index.json", "data/00000.safetensors", "0" * 64]
FUZZ_SUFFIXES = [".suh", ".svh", ".mcg", ".trellis", ".bias", "#0", "x", "/.."]
# The divergences the fuzz test allows, each probed so that its filter is
# exercised and its exact rule checked: a manifest that disagrees with kept
# GGUF metadata (the prototype says meta; the reader, which does not read it,
# must give the prototype's verdict past that check), and a string escape
# (the reader refuses it as canonical wherever the prototype refuses).
FUZZ_PROBES = [
    ("tiny", lambda c: c.relist(lambda m: m["model"].update(architecture="other"))),
    ("tiny", lambda c: c.relist(lambda m: m["model"].update(expert_count=1))),
    ("tiny", lambda c: c.relist(lambda m: m["model"].update(expert_count=3))),
    ("tiny", lambda c: c.relist(lambda m: m["model"].update(architecture="other", expert_count=4))),
    ("tiny", lambda c: c.relist(raw=replace_once(b'"format":"', b'"format":"\\u0041'))),
    ("tiny", lambda c: c.relist(raw=replace_once(b'"version":"m0-prototype"', b'"version":"m0\\"prototype"'))),
    ("exl3", lambda c: c.relist(raw=replace_once(b'"version":"m0-prototype"', b'"version":"m0\\\\prototype"'))),
    ("exl3", lambda c: c.index(raw=lambda d: L.dumps(d).replace(b'"name":"', b'"name":"\\u0078', 1))),
]


def fuzz_tree(doc, rng, names):
    """One to three random edits of a parsed document, at random depths,
    mostly inside its lists (the header keys fail early and stop there)."""
    for _ in range(rng.randint(1, 3)):
        parent, key, node = None, None, doc
        for _ in range(7):
            if isinstance(node, (dict, list)) and node and rng.random() < 0.9:
                if isinstance(node, dict):
                    lists = sorted(k for k, v in node.items() if isinstance(v, list))
                    use_lists = parent is None and lists and rng.random() < 0.85
                    key = rng.choice(lists if use_lists else sorted(node))
                else:
                    key = rng.randrange(len(node))
                parent, node = node, node[key]
            else:
                break
        if parent is None:
            continue
        # Integers are mostly nudged, so that edits reach the layout checks.
        op = rng.choice([6, 7]) if type(node) is int and rng.random() < 0.6 else rng.randrange(11)
        if op == 0:
            parent[key] = rng.choice(FUZZ_NUMBERS)
        elif op == 1:
            parent[key] = rng.choice(FUZZ_STRINGS + names)
        elif op == 2:
            parent[key] = rng.choice([None, True, False])
        elif op == 3 and isinstance(node, list) and node:
            node.pop(rng.randrange(len(node)))
        elif op == 4 and isinstance(node, list) and node:
            node.append(json.loads(json.dumps(rng.choice(node))))
        elif op == 5 and isinstance(node, dict) and node:
            node.pop(rng.choice(sorted(node)))
        elif op == 6 and type(node) is int:
            parent[key] = node + rng.choice([1, -1, 256, -256, 4096, -4096, 2097152, -2097152, 144])
        elif op == 7 and type(node) is int:
            parent[key] = node * rng.choice([2, 16, 0, -1])
        elif op == 8 and isinstance(parent, list) and len(parent) > 1:
            other = rng.randrange(len(parent))
            parent[key], parent[other] = parent[other], parent[key]
        elif op == 9 and isinstance(node, str) and node:
            parent[key] = node + rng.choice(FUZZ_SUFFIXES)
        elif op == 10:
            parent[key] = [0] if rng.random() < 0.5 else {}
    return doc


def fuzz_bytes(data, rng):
    """One to three random byte edits: JSON punctuation, deletions, copies and raw bytes."""
    data = bytearray(data)
    for _ in range(rng.randint(1, 3)):
        at, op = rng.randrange(len(data)), rng.randrange(4)
        if op == 0:
            data[at] = rng.choice(b'0123456789{}[],:"-ax \\\n')
        elif op == 1:
            del data[at:at + 1 + rng.randrange(8)]
        elif op == 2:
            start = rng.randrange(len(data))
            data[at:at] = data[start:start + rng.randrange(32)]
        else:
            data[at] = rng.randrange(256)
    return bytes(data)


def fuzz_corpus(out, bases, work):
    """FUZZ_CASES mutations of the tiny and exl3 artifacts' index or manifest,
    as trees (re-hashed and re-listed, so they reach the checks past the
    digests) or as raw bytes, then FUZZ_PROBES, each judged by the prototype."""
    rng = random.Random(2026)
    fuzz = out / "fuzz"
    fuzz.mkdir()
    lines = []

    def mutation(n):
        if n >= FUZZ_CASES:
            return FUZZ_PROBES[n - FUZZ_CASES]
        base = rng.choice(["tiny", "tiny", "exl3"])
        index = json.loads((bases[base] / "index.json").read_bytes())
        names = [r["name"] for r in index["resources"]] + [a["name"] for a in index["expert_arrays"]]
        kind = rng.choices(range(4), weights=[10, 3, 4, 3])[0]
        if kind == 0:
            return base, lambda c: c.index(lambda d: fuzz_tree(d, rng, names))
        if kind == 1:
            return base, lambda c: c.index(raw=lambda d: fuzz_bytes(L.dumps(d), rng))
        if kind == 2:
            return base, lambda c: c.relist(lambda m: fuzz_tree(m, rng, names))
        return base, lambda c: c.relist(raw=lambda m: fuzz_bytes(L.dumps(m), rng))

    for n in range(FUZZ_CASES + len(FUZZ_PROBES)):
        base, mutate = mutation(n)
        src = bases[base]
        target = work / "fuzz" / str(n) / src.name
        shutil.copytree(src, target)
        c = Mutations(target)
        mutate(c)
        assert c.a.name == hashlib.sha256(c.read("manifest.json")).hexdigest()
        (fuzz / f"{n}.manifest").write_bytes(c.read("manifest.json"))
        if c.read("index.json") != (src / "index.json").read_bytes():
            (fuzz / f"{n}.index").write_bytes(c.read("index.json"))
        # Both judged in full, so that the test checks, rather than the
        # corpus assumes, that they differ only where the first is meta.
        lines.append(f"{n} {base} {verdict(c.a)} {verdict_unread(c.a)}")
        shutil.rmtree(target.parent)
    (out / "fuzz.txt").write_text("\n".join(lines) + "\n")


def build_corpus(out):
    if out.exists():
        shutil.rmtree(out)
    work = out / "work"
    bases = {
        "tiny": build_gguf(work, "tiny", {"general.architecture": (8, "toy"), "toy.expert_count": (4, 2),
                                          "general.alignment": (4, 32)}, tiny_tensors(), 12 << 10),
        "exl3": build_exl3(work, "exl3"),
        "moe": build_gguf(work, "moe", KV, toy_tensors(), 6 << 20),
    }
    (out / "types.txt").write_text(types_text())
    golden = out / "golden"
    golden.mkdir()
    for name, root in bases.items():
        shutil.copytree(root, golden / name / root.name)
        (golden / f"{name}.view").write_text(view(golden / name / root.name))
    lines, wrong = [], []
    for case, base, intended, mutate in CASES:
        target = out / "cases" / case / bases[base].name
        shutil.copytree(bases[base], target)
        c = Mutations(target)
        mutate(c)
        got = verdict(c.a)
        if got != intended:
            wrong.append(f"{case}: the prototype says {got}, the case intends {intended}")
        lines.append(f"{case} {c.a.name} {got}")
    fuzz_corpus(out, bases, work)
    shutil.rmtree(work)
    if wrong:
        raise SystemExit("artifact corpus: cases that do not break their rule:\n  " + "\n  ".join(wrong))
    (out / "cases.txt").write_text("\n".join(lines) + "\n")
    (out / "stamp").write_text("")


def main(argv):
    if len(argv) == 3 and argv[1] == "view":
        sys.stdout.write(view(pathlib.Path(argv[2])))
        return 0
    if len(argv) != 2:
        raise SystemExit(__doc__)
    build_corpus(pathlib.Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
