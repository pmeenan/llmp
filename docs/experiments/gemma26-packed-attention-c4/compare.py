#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Full heads: exact bytes/raw/winners; sixteen explicitly selected NLL/TV rows."""
import array
import hashlib
import json
import math
import pathlib
import sys
import tempfile

VOCAB = 262144
ROWS = 128
SELECTED = {step * 4 + owner for step in (0, 1, 2, 31) for owner in range(4)}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checked_input(path, expected_sha, expected_bytes):
    raw = path.read_bytes()
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("likelihood input identity changed")
    return raw


def decode(raw):
    a = array.array("f")
    if a.itemsize != 4 or len(raw) != VOCAB * 4:
        raise ValueError("incomplete F32 head")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    if not all(math.isfinite(x) for x in a):
        raise ValueError("nonfinite head")
    return a


def row(ah, bh, target, sampled):
    a, b = decode(ah), decode(bh)
    aw = max(range(VOCAB), key=a.__getitem__)
    bw = max(range(VOCAB), key=b.__getitem__)
    strict = aw != bw
    margin = b[bw] - b[aw]
    result = {"byte_exact": ah == bh, "strict": strict,
              "positive_reference_winner_margin": strict and margin > 0,
              "tied_reference_winner": strict and margin == 0,
              "reference_winner_over_native_choice": margin,
              "max_raw": max(abs(x-y) for x, y in zip(a, b, strict=True))}
    if sampled:
        def distribution(r):
            peak = max(r)
            exp = [math.exp(x-peak) for x in r]
            total = math.fsum(exp)
            return exp, total, math.log(total)+peak-r[target]
        ae, az, anll = distribution(a)
        be, bz, bnll = distribution(b)
        result.update(tv=0.5*math.fsum(abs(x/az-y/bz) for x, y in zip(ae, be, strict=True)),
                      native_nll=anll, reference_nll=bnll, signed_nll_delta=anll-bnll)
    return result


def compare(root, policy, ref_first, ref_repeat, expected):
    own_path = root / "native-own-frozen.json"
    if digest(own_path) != expected:
        raise ValueError("native freeze changed")
    own = json.loads(own_path.read_text())
    if digest(root / "pre-native.json") != own["pre_native_sha256"]:
        raise ValueError("source prefreeze changed")
    native = root / f"{policy}-first/heads.f32"
    if digest(native) != own["policies"][policy]["heads.f32"]["sha256"]:
        raise ValueError("native complete outputs changed")
    reference_sha = digest(ref_first / "heads.f32")
    if digest(ref_repeat / "heads.f32") != reference_sha:
        raise ValueError("reference complete repeats differ")
    for path in (native, ref_first / "heads.f32", ref_repeat / "heads.f32"):
        if path.stat().st_size != ROWS * VOCAB * 4:
            raise ValueError("incomplete bounded full heads")
    input_sha = "b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610"
    for directory in (root / f"{policy}-first", ref_first, ref_repeat):
        if digest(directory / "inputs.i32") != input_sha:
            raise ValueError("fixed input identity changed")
    ids = array.array("i")
    ids.frombytes(checked_input(root / "ids.i32", input_sha, 1024 * 4))
    if sys.byteorder != "little":
        ids.byteswap()
    counts = {"byte_exact_rows": 0, "strict_argmax_differences": 0,
              "positive_reference_winner_margin_differences": 0, "strict_at_reference_tie": 0}
    maximum = 0.0
    phases = {name: {"rows": 0, "byte_exact_rows": 0, "strict_differences": 0, "max_raw": 0.0}
              for name in ("first_wave", "first_three_waves", "later_29_waves")}
    selected = []
    with native.open("rb") as a, (ref_first / "heads.f32").open("rb") as b:
        for index in range(ROWS):
            step, owner = divmod(index, 4)
            metrics = row(a.read(VOCAB*4), b.read(VOCAB*4), ids[68+owner+step], index in SELECTED)
            counts["byte_exact_rows"] += metrics["byte_exact"]
            counts["strict_argmax_differences"] += metrics["strict"]
            counts["positive_reference_winner_margin_differences"] += metrics["positive_reference_winner_margin"]
            counts["strict_at_reference_tie"] += metrics["tied_reference_winner"]
            maximum = max(maximum, metrics["max_raw"])
            for name in ((["first_wave"] if step == 0 else []) +
                         (["first_three_waves"] if step < 3 else ["later_29_waves"])):
                phase = phases[name]
                phase["rows"] += 1
                phase["byte_exact_rows"] += metrics["byte_exact"]
                phase["strict_differences"] += metrics["strict"]
                phase["max_raw"] = max(phase["max_raw"], metrics["max_raw"])
            if index in SELECTED:
                selected.append({"row": index, "step": step, "owner": owner,
                                 "target_position": 68+owner+step,
                                 "next_forced_id": ids[68+owner+step], **metrics})
    return {"native_own_freeze_sha256": expected, "policy": policy,
            "native_sha256": digest(native), "reference_sha256": reference_sha,
            "input_sha256": input_sha, "complete_rows": ROWS, "vocab": VOCAB,
            **counts, "max_raw_delta": maximum, "own_repeat_margin_movement": 0,
            "phase_counts": phases, "selected_likelihood_rows": selected,
            "scope": "selected rows are not corpus PPL or a whole-model quality gate"}


def selftest():
    global VOCAB
    VOCAB = 4
    def raw(a):
        return array.array("f", a).tobytes()
    a, b = raw([0, 0, 3, 0]), raw([2, 2, 1, 0])
    r = row(a, b, 2, True)
    assert r["strict"] and r["positive_reference_winner_margin"]
    assert r["reference_winner_over_native_choice"] == 1
    r = row(raw([0.0]*4), raw([-0.0]*4), 0, True)
    assert not r["byte_exact"] and r["max_raw"] == 0 and not r["strict"]
    for values in ([math.nan, 0, 0, 0], [math.inf, 0, 0, 0]):
        try:
            row(raw(values), b, 2, False)
            raise AssertionError("nonfinite accepted")
        except ValueError:
            pass
    try:
        row(b"", b, 2, False)
        raise AssertionError("truncation accepted")
    except ValueError:
        pass
    canonical = bytes(range(8))
    expected_sha = hashlib.sha256(canonical).hexdigest()
    with tempfile.TemporaryDirectory(prefix="gemma-packed-input-control-") as directory:
        path = pathlib.Path(directory) / "ids.i32"
        path.write_bytes(canonical)
        assert checked_input(path, expected_sha, len(canonical)) == canonical
        for changed in (b"\xff" + canonical[1:], canonical[:-1]):
            path.write_bytes(changed)
            try:
                checked_input(path, expected_sha, len(canonical))
                raise AssertionError("changed likelihood input accepted")
            except ValueError:
                pass
    print("COMPARE_SELFTEST tie-selected-margin/signed-zero/nonfinite/truncation/input-identity PASS")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) == 7 and sys.argv[2] in ("control", "candidate"):
        result = compare(pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]),
                         pathlib.Path(sys.argv[4]), sys.argv[5])
        with pathlib.Path(sys.argv[6]).open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(json.dumps(result))
    else:
        raise SystemExit("compare.py ROOT control|candidate REF_FIRST REF_REPEAT FREEZE_SHA OUTPUT | --selftest")
