#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded all-head identity/argmax/raw screen; eight labelled likelihood rows."""
import array
import hashlib
import json
import math
import pathlib
import sys

VOCAB = 262144
STEPS = 32
SEEDS = (45518, 107, 101)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1048576):
            h.update(data)
    return h.hexdigest()


def decode(data):
    row = array.array("f")
    row.frombytes(data)
    if sys.byteorder != "little":
        row.byteswap()
    return row


def argmax(row):
    best = -math.inf
    chosen = 0
    for i, value in enumerate(row):
        if not math.isfinite(value):
            raise ValueError("non-finite completed head")
        if value > best:
            best, chosen = value, i
    return chosen


def record(path, owners):
    if path.stat().st_size != owners * STEPS * VOCAB * 4:
        raise ValueError("incomplete bounded owner heads")
    choices = []
    with path.open("rb") as stream:
        for _ in range(owners * STEPS):
            choices.append(argmax(decode(stream.read(VOCAB * 4))))
    return {"sha256": digest(path), "rows": owners * STEPS, "argmax": choices}


def freeze(root, owners):
    names = ("scalar-a", "joined-first", "joined-repeat", "scalar-b", "ordinary")
    files = {name: root / name / "heads.f32" for name in names}
    hashes = {name: digest(path) for name, path in files.items()}
    if len({hashes[name] for name in names[:4]}) != 1:
        raise ValueError("solo/joined or own-repeat full-head identity failed")
    inputs = {name: root / name / "inputs.i32" for name in names}
    supplied_sha = None
    if inputs["joined-first"].exists():
        hashes_input = {name: digest(path) for name, path in inputs.items()}
        if len(set(hashes_input.values())) != 1:
            raise ValueError("native supplied inputs differ")
        supplied_sha = hashes_input["joined-first"]
        read_inputs(inputs["joined-first"])
    candidate = record(files["joined-first"], owners)
    ordinary = (candidate if hashes["ordinary"] == candidate["sha256"]
                else record(files["ordinary"], owners))
    result = {"owners": owners, "vocab": VOCAB, "steps": STEPS,
              "source_identity_sha256": digest(root / "source-identities.json"),
              "files": hashes, "candidate": candidate, "ordinary": ordinary,
              "own_repeat_raw_noise": 0, "solo_joined_byte_exact": True,
              "supplied_input_sha256": supplied_sha}
    path = root / "native-frozen.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    path.chmod(0o444)
    print(json.dumps({"freeze_sha256": digest(path), **result}))


def read_inputs(path):
    data = path.read_bytes()
    ids = array.array("i")
    if len(data) != 4096 or ids.itemsize != 4:
        raise ValueError("invalid supplied ID byte count")
    ids.frombytes(data)
    if sys.byteorder != "little":
        ids.byteswap()
    if ids[0] != 2 or any(i < 0 or i >= VOCAB for i in ids):
        raise ValueError("invalid supplied ID payload")
    return ids


def distribution(row):
    peak = max(row)
    unscaled = [math.exp(value - peak) for value in row]
    total = math.fsum(unscaled)
    return unscaled, total, peak


def compared_row(ah, bh, native_choice, reference_choice):
    ar, br = decode(ah), decode(bh)
    strict = native_choice != reference_choice
    margin = br[reference_choice] - br[native_choice]
    return ar, br, {"byte_exact": ah == bh, "strict": strict,
                    "reference_winner_margin": margin, "outside_zero_margin": strict and margin > 0,
                    "reference_tie": strict and margin == 0,
                    "max_raw": 0.0 if ah == bh else max(abs(x-y) for x,y in zip(ar,br,strict=True))}


def compare(root, first, repeat, expected):
    frozen = root / "native-frozen.json"
    if digest(frozen) != expected:
        raise ValueError("candidate freeze identity changed")
    own = json.loads(frozen.read_text())
    owners = own["owners"]
    if digest(root / "source-identities.json") != own["source_identity_sha256"]:
        raise ValueError("native source identities changed")
    for name, sha in own["files"].items():
        if digest(root / name / "heads.f32") != sha:
            raise ValueError("native complete output changed")
    supplied = None
    if own.get("supplied_input_sha256"):
        for path in [root / name / "inputs.i32" for name in own["files"]] + [first / "inputs.i32", repeat / "inputs.i32"]:
            if digest(path) != own["supplied_input_sha256"]:
                raise ValueError("matched supplied inputs changed")
        supplied = read_inputs(root / "joined-first/inputs.i32")
    reference = record(first / "heads.f32", owners)
    if digest(repeat / "heads.f32") != reference["sha256"]:
        raise ValueError("reference repeat changed full heads")
    candidate = own["candidate"]
    differences = sum(a != b for a, b in zip(candidate["argmax"], reference["argmax"], strict=True))
    selected = {step * owners + owner for step in (0, 7, 15, 30) for owner in (0, owners - 1)}
    byte_exact = 0
    maximum = 0.0
    outside = 0
    ties = 0
    sampled = []
    with (root / "joined-first/heads.f32").open("rb") as a, (first / "heads.f32").open("rb") as b:
        for index in range(STEPS * owners):
            ah, bh = a.read(VOCAB * 4), b.read(VOCAB * 4)
            ar, br, metrics = compared_row(ah, bh, candidate["argmax"][index], reference["argmax"][index])
            byte_exact += metrics["byte_exact"]
            maximum = max(maximum, metrics["max_raw"])
            outside += metrics["outside_zero_margin"]
            ties += metrics["reference_tie"]
            if index in selected:
                step, owner = divmod(index, owners)
                target_position = (10 if supplied is None else 68) + owner + step
                target = (SEEDS[(step + 4 + owner) % len(SEEDS)] if supplied is None else supplied[target_position])
                ap, az, am = distribution(ar)
                bp, bz, bm = distribution(br)
                tv = 0.5 * math.fsum(abs(x / az - y / bz) for x, y in zip(ap, bp, strict=True))
                sampled.append({"row": index, "step": step, "owner": owner,
                                "target_position": target_position, "next_forced_id": target,
                                "tv": tv, "native_nll": math.log(az) + am - ar[target],
                                "reference_nll": math.log(bz) + bm - br[target]})
    print(json.dumps({"native_freeze_sha256": expected, "reference": reference,
                      "owners": owners, "supplied_input_sha256": own.get("supplied_input_sha256"), "complete_rows": owners * STEPS,
                      "byte_exact_rows": byte_exact, "strict_argmax_differences": differences,
                      "outside_zero_own_repeat_margin_noise": outside, "strict_differences_at_reference_tie": ties,
                      "margin_noise_basis": "candidate complete-head byte-exact repeat implies zero winner-margin movement",
                      "all_head_max_raw_delta": maximum,
                      "selected_likelihood_rows": sampled, "corpus_ppl_claim": False}, indent=2))


def selftest():
    assert argmax([1.0, 1.0, -1.0]) == 0
    assert array.array("f", [0.0]).tobytes() != array.array("f", [-0.0]).tobytes()
    candidate = array.array("f", [1.0, 2.0, -1.0])
    reference = array.array("f", [3.0, 3.0, -1.0])
    _, _, metrics = compared_row(candidate.tobytes(), reference.tobytes(), argmax(candidate), argmax(reference))
    assert metrics["strict"] and metrics["reference_tie"] and not metrics["outside_zero_margin"]
    _, _, metrics = compared_row(array.array("f", [0.0]).tobytes(), array.array("f", [-0.0]).tobytes(), 0, 0)
    assert not metrics["byte_exact"] and metrics["max_raw"] == 0.0
    for value in (math.inf, -math.inf, math.nan):
        try:
            argmax([0.0, value])
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite head accepted")
    print("signed-zero/tie/non-finite controls passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif len(sys.argv) == 4 and sys.argv[1] == "freeze":
        count = int(sys.argv[3])
        if not 1 <= count <= 12:
            raise ValueError("unbounded owner count")
        freeze(pathlib.Path(sys.argv[2]), count)
    elif len(sys.argv) == 6 and sys.argv[1] == "compare":
        compare(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]), sys.argv[5])
    else:
        raise SystemExit("freeze ROOT OWNERS | compare ROOT REF_FIRST REF_REPEAT FREEZE_SHA | selftest")
