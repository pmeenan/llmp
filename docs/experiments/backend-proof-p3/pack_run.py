#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Packs native's full-model output into the reference runs' layout (reference only).

Reference-side experiment tooling for the backend proof's P3 (Tier C); it shares no code with llmpalooza.
Native writes, per prefix p of the held-out trajectories, in NATIVE_DIR:
  logits-<p>.prefill.npy          float32 [p, 151936]   every prefill row
  logits-<p>.suffix.npy           float32 [16, 151936]  the 16 single-token steps
and, with capture,
  capture-<p>.blocks.npy          float32 [24, p, 896]  the residual stream after each decoder block
  capture-<p>.final_norm.npy      float16 [p, 896]      the final norm's output
  capture-<p>.kv.npy              float16 [48, p, 2, 64]       K then V per layer, cells [0, p) after the prefill
  capture-<p>.kv_after_suffix.npy float16 [48, p + 16, 2, 64] the same after the 16 steps

  pack_run.py NATIVE_DIR OUT_DIR --ids HELD_OUT_IDS --uninstrumented RUNG3_DIR --artifact ARTIFACT_DIR
  pack_run.py --reference NATIVE_LAYOUT_DIR OUT_DIR --ids HELD_OUT_IDS   (validation on reference runs)

NATIVE_DIR/manifest.json must be llmp_exl3_exec's capture run, with no mismatched evaluation, and
every logits .npy must equal RUNG3_DIR's (an uninstrumented run of the same fixture, arm and artifact)
byte for byte (RE-010). The run's fixture is its artifact's (fixture_identity.py: the checkpoint it was
prepared from), never its label: the manifest must name ARTIFACT_DIR's id and a fixture label that
agrees. The identity and that comparison go to OUT_DIR/native.json, which
tierc_check.py requires (it checks the fixture against its bounds and fails a run whose captured
logits differ). --reference packs a directory without a native manifest and writes no native.json.

It writes OUT_DIR/logits-<p>.npz (ids int64 (1, p + 16), prefill float32 (1, p, V), suffix float32
(1, 16, V)) and OUT_DIR/capture-<p>.npz (blocks float32 (24, 1, p, 896), final_norm float32 (1, p, 896),
kv float32 (48, p, 2, 64), kv_after_suffix float32 (48, p + 16, 2, 64)), exactly as
../backend-proof-p0/exl3_heldout.py saves a run, which ../backend-proof-p0/oracle_compare.py scores:
F16 is widened exactly, nothing else changes. Every file's dtype and shape is checked, and a prefix
with any capture file must have all four. Exits 2 on anything missing or malformed.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_identity import artifact_fixture  # noqa: E402

HELD_OUT_SHA256 = "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c"
LAYERS, HIDDEN, KV_HEADS, HEAD_DIM, VOCAB, SUFFIX = 24, 896, 2, 64, 151936, 16
CAPTURE = {"blocks": ("float32", lambda p: (LAYERS, p, HIDDEN), lambda a, p: a.reshape(LAYERS, 1, p, HIDDEN)),
           "final_norm": ("float16", lambda p: (p, HIDDEN), lambda a, p: a.reshape(1, p, HIDDEN)),
           "kv": ("float16", lambda p: (2 * LAYERS, p, KV_HEADS, HEAD_DIM), lambda a, p: a),
           "kv_after_suffix": ("float16", lambda p: (2 * LAYERS, p + SUFFIX, KV_HEADS, HEAD_DIM), lambda a, p: a)}


class Malformed(Exception):
    pass


def load(path, dtype, shape):
    if not path.exists():
        raise Malformed(f"missing {path}")
    array = np.load(path, mmap_mode="r")
    if array.dtype != np.dtype(dtype) or array.shape != shape:
        raise Malformed(f"{path}: {array.dtype} {array.shape}, expected {dtype} {shape}")
    return array


def identity(native, uninstrumented, prefixes, artifact):
    """native.json: the capture run's identity and its RE-010 comparison with the uninstrumented run. The
    fixture is the artifact's (fixture_identity.py), never the run's label, which must agree with it."""
    try:
        fixture, artifact_id = artifact_fixture(artifact)
    except (OSError, KeyError, ValueError) as error:
        raise Malformed(f"the artifact names no fixture: {error}") from error
    keys = ("harness", "fixture", "arm", "artifact", "plan_file_sha256", "capture", "record_ops", "mismatches")
    manifests = {}
    for role, d in (("capture", native), ("uninstrumented", uninstrumented)):
        path = d / "manifest.json"
        if not path.exists():
            raise Malformed(f"missing {path}")
        manifests[role] = {k: json.loads(path.read_text()).get(k) for k in keys}
    cap, plain = manifests["capture"], manifests["uninstrumented"]
    if cap["harness"] != "llmp_exl3_exec" or cap["capture"] is not True or cap["mismatches"] != 0:
        raise Malformed(f"{native} is not a llmp_exl3_exec capture run: {cap}")
    if plain["harness"] != "llmp_exl3_exec" or plain["capture"] is not False or \
            plain["record_ops"] is not False or plain["mismatches"] != 0:
        raise Malformed(f"{uninstrumented} is not an uninstrumented llmp_exl3_exec run: {plain}")
    same_run = all(cap[k] == plain[k] for k in ("fixture", "arm", "artifact", "plan_file_sha256"))
    if not same_run:
        raise Malformed(f"the capture and uninstrumented runs differ in identity: {cap} {plain}")
    if cap["artifact"] != artifact_id:
        raise Malformed(f"the run is of artifact {cap['artifact']}, not {artifact_id}")
    if cap["fixture"] != fixture:
        raise Malformed(f"the run is labelled {cap['fixture']}, but its artifact is {fixture}'s")
    names = [f"logits-{p}.{part}.npy" for p in prefixes for part in ("prefill", "suffix")]
    differ = [n for n in names if not (uninstrumented / n).exists()
              or (native / n).read_bytes() != (uninstrumented / n).read_bytes()]
    return {"fixture": fixture, "fixture_from": "the artifact's source checkpoint", "arm": cap["arm"],
            "artifact": cap["artifact"],
            "plan_file_sha256": cap["plan_file_sha256"], "uninstrumented": str(uninstrumented),
            "logits_equal_uninstrumented": not differ, "logits_differing": differ}


def pack(native, out, ids_path, uninstrumented=None, artifact=None):
    ids_bytes = ids_path.read_bytes()
    if hashlib.sha256(ids_bytes).hexdigest() != HELD_OUT_SHA256:
        raise Malformed("held-out ID identity mismatch")
    ids = np.frombuffer(ids_bytes, dtype="<i8")
    prefixes = sorted(int(m[1]) for p in native.glob("logits-*.prefill.npy")
                      if (m := re.fullmatch(r"logits-(\d+)\.prefill\.npy", p.name)))
    if not prefixes:
        raise Malformed(f"no logits-<p>.prefill.npy in {native}")
    out.mkdir(parents=True, exist_ok=True)
    for p in prefixes:
        prefill = load(native / f"logits-{p}.prefill.npy", "float32", (p, VOCAB))
        suffix = load(native / f"logits-{p}.suffix.npy", "float32", (SUFFIX, VOCAB))
        np.savez(out / f"logits-{p}.npz", ids=ids[:p + SUFFIX].astype(np.int64).reshape(1, -1),
                 suffix=np.asarray(suffix).reshape(1, SUFFIX, VOCAB), prefill=np.asarray(prefill).reshape(1, p, VOCAB))
        present = [k for k in CAPTURE if (native / f"capture-{p}.{k}.npy").exists()]
        if present and len(present) != len(CAPTURE):
            raise Malformed(f"prefix {p}: capture files {present}, expected all of {list(CAPTURE)}")
        if present:
            arrays = {}
            for key, (dtype, shape, layout) in CAPTURE.items():
                array = load(native / f"capture-{p}.{key}.npy", dtype, shape(p))
                arrays[key] = layout(np.asarray(array).astype(np.float32), p)
            np.savez(out / f"capture-{p}.npz", **arrays)
        print(f"prefix {p}: logits{' and capture' if present else ''}")
    if uninstrumented is not None:
        record = identity(native, uninstrumented, prefixes, artifact)
        (out / "native.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"{record['fixture']} {record['arm']}: captured logits equal the uninstrumented run's: "
              f"{record['logits_equal_uninstrumented']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("native", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--ids", type=Path, required=True, help="the held-out IDs (little-endian int64)")
    parser.add_argument("--uninstrumented", type=Path, help="native's uninstrumented run (the RE-010 control)")
    parser.add_argument("--reference", action="store_true", help="a reference run in native's layout: no identity")
    parser.add_argument("--artifact", type=Path, help="the installed artifact the native run loaded (its fixture)")
    args = parser.parse_args()
    if (args.uninstrumented is None) != args.reference:
        parser.error("exactly one of --uninstrumented (a native run) and --reference")
    if (args.artifact is None) != args.reference:
        parser.error("a native run needs --artifact, whose fixture it is; a reference run takes none")
    try:
        pack(args.native, args.out, args.ids, args.uninstrumented, args.artifact)
    except Malformed as error:
        print("MALFORMED OR INCOMPLETE:", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
