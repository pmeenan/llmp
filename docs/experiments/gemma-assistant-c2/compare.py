#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Whole finite C2 rows, exact bytes and distribution comparison; no tolerance fitting."""
import array
import hashlib
import json
import math
import pathlib
import sys

VOCAB = 262144
WIDTH = 2816


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def values(data, count):
    if len(data) != count * 4 or sys.byteorder != "little":
        raise ValueError("closed little-endian F32 byte count differs")
    out = array.array("f", data)
    if not all(math.isfinite(v) for v in out):
        raise ValueError("nonfinite full output")
    return out


def row(native, reference, width):
    a, b = values(native, width), values(reference, width)
    aw = max(range(width), key=a.__getitem__)
    bw = max(range(width), key=b.__getitem__)
    bsecond = max(b[i] for i in range(width) if i != bw)
    record = {"byte_exact": native == reference,
              "max_raw_delta": max(abs(x-y) for x, y in zip(a, b)),
              "strict_argmax_difference": aw != bw,
              "reference_top2_margin": b[bw]-bsecond,
              "reference_selected_margin": b[bw]-b[aw],
              "positive_margin_difference": aw != bw and b[bw] > b[aw]}
    if native == reference:
        record.update(tv=0.0, chosen_nll_delta=0.0)
    else:
        am, bm = max(a), max(b)
        ae, be = [math.exp(x-am) for x in a], [math.exp(x-bm) for x in b]
        az, bz = math.fsum(ae), math.fsum(be)
        record.update(tv=math.fsum(abs(x/az-y/bz) for x, y in zip(ae, be))/2,
                      chosen_nll_delta=(math.log(az)+am-a[bw])-(math.log(bz)+bm-b[bw]))
    return record


def compare(native_dir, reference_dir, mode):
    ref_head = (reference_dir / "heads.f32").read_bytes()
    ref_feature = (reference_dir / "postprojection.f32").read_bytes()
    values(ref_head, 6 * VOCAB)
    values(ref_feature, 6 * WIDTH)
    records = []
    feature_records = []
    for step in range(3):
        for role, width in (("head", VOCAB), ("projection", WIDTH)):
            first = native_dir / f"{mode}-chain3-repeat0-step{step}-{role}.f32"
            repeat = native_dir / f"{mode}-chain3-repeat1-step{step}-{role}.f32"
            data = first.read_bytes()
            values(data, 2 * width)
            if data != repeat.read_bytes():
                raise ValueError("posthoc native own repeat differs")
            if step == 0:
                for r in (0, 1):
                    if data != (native_dir / f"{mode}-chain1-repeat{r}-step0-{role}.f32").read_bytes():
                        raise ValueError("posthoc one/three-step initial bytes differ")
            source = ref_head if role == "head" else ref_feature
            for owner in range(2):
                a = data[owner*width*4:(owner+1)*width*4]
                offset = (step*2+owner)*width*4
                b = source[offset:offset+width*4]
                result = row(a, b, width) if role == "head" else {
                    "byte_exact": a == b,
                    "max_raw_delta": max(abs(x-y) for x, y in zip(values(a,width),values(b,width)))}
                result.update(step=step, owner=owner,
                              native_sha256=hashlib.sha256(a).hexdigest(),
                              reference_sha256=hashlib.sha256(b).hexdigest())
                (records if role == "head" else feature_records).append(result)
    return {"mode": mode, "head_rows": 6, "feature_rows": 6,
            "byte_exact_heads": sum(r["byte_exact"] for r in records),
            "byte_exact_features": sum(r["byte_exact"] for r in feature_records),
            "strict_argmax_differences": sum(r["strict_argmax_difference"] for r in records),
            "positive_margin_differences": sum(r["positive_margin_difference"] for r in records),
            "max_raw_head_delta": max(r["max_raw_delta"] for r in records),
            "max_raw_feature_delta": max(r["max_raw_delta"] for r in feature_records),
            "max_tv": max(r["tv"] for r in records),
            "max_abs_chosen_nll_delta": max(abs(r["chosen_nll_delta"]) for r in records),
            "raw_rows_external": records, "raw_feature_rows_external": feature_records}


def selftest():
    def pack(xs):
        return array.array("f", xs).tobytes()
    zero = row(pack([0.0,1.0,2.0,3.0]),pack([-0.0,1.0,2.0,3.0]),4)
    assert not zero["byte_exact"] and zero["max_raw_delta"] == 0
    tie = row(pack([0.0,1.0,0.0,0.0]),pack([1.0,1.0,0.0,0.0]),4)
    assert tie["strict_argmax_difference"] and not tie["positive_margin_difference"]
    tied_top_candidate_below = row(pack([0.0,0.0,3.0,0.0]),pack([2.0,2.0,1.0,0.0]),4)
    assert tied_top_candidate_below["reference_top2_margin"] == 0
    assert tied_top_candidate_below["reference_selected_margin"] == 1
    assert tied_top_candidate_below["positive_margin_difference"]
    positive = row(pack([0.0,2.0,0.0,0.0]),pack([2.0,1.0,0.0,0.0]),4)
    assert positive["positive_margin_difference"] and positive["tv"] > 0
    for data in (pack([math.inf,0.0,0.0,0.0]),pack([0.0,0.0,0.0])):
        try:
            row(data,pack([0.0]*4),4)
        except ValueError:
            pass
        else:
            raise AssertionError("hostile row accepted")
    print("C2_COMPARE_CONTROLS 6 PASS")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) == 5 and sys.argv[3] in ("serial", "joined"):
        result = compare(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3])
        with pathlib.Path(sys.argv[4]).open("x") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
        print(json.dumps({k:v for k,v in result.items() if not k.startswith("raw_")},allow_nan=False))
    else:
        raise SystemExit("NATIVE_DIR REFERENCE_DIR serial|joined NEW_JSON or --selftest")
