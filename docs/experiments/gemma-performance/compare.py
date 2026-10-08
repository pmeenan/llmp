#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""ROOT calibrate|oracle: freeze native-own noise before reading the oracle.

ROOT contains native-quality1/native-norm1/reference-quality{1,2} directories.
Raw inputs/outputs stay external. If greedies diverge, later numerical rows
are refused until both paths are rerun on the same reference-forced prefix.
"""
import array
import hashlib
import heapq
import json
import math
import pathlib
import sys

ROWS, VOCAB = 32, 262144


def row(root, mode, index):
    raw = (root / mode / f"logits-{index}.f32").read_bytes()
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    assert len(values) == VOCAB and all(map(math.isfinite, values))
    return values, hashlib.sha256(raw).hexdigest()


def best(values):
    return max(range(VOCAB), key=values.__getitem__)


def distribution(values):
    maximum = max(values)
    weights = [math.exp(x - maximum) for x in values]
    total = math.fsum(weights)
    return weights, total, maximum + math.log(total)


def main():
    assert len(sys.argv) >= 3
    root, action = pathlib.Path(sys.argv[1]), sys.argv[2]
    if action == "pair":
        assert len(sys.argv) in (6, 8), "ROOT pair LEFT RIGHT ROWS [LEFT_OFFSET RIGHT_OFFSET]"
        left, right, count = sys.argv[3], sys.argv[4], int(sys.argv[5])
        offsets = [int(x) for x in sys.argv[6:]] if len(sys.argv) == 8 else [0, 0]
        assert 1 <= count <= 32 and min(offsets) >= 0 and max(offsets) + count <= 32
        rows, common = [], True
        for i in range(count):
            a, ah = row(root, left, i + offsets[0])
            b, bh = row(root, right, i + offsets[1])
            ai, bi = best(a), best(b)
            item = {"left_row": i + offsets[0], "right_row": i + offsets[1],
                    "ids": [ai, bi], "common_prefix": common, "byte_exact": ah == bh}
            if common:
                aw, at, al = distribution(a)
                bw, bt, bl = distribution(b)
                item.update(raw=max(abs(x-y) for x, y in zip(a,b)),
                            chosen_nll=abs((al-a[bi])-(bl-b[bi])),
                            tv=.5*math.fsum(abs(x/at-y/bt) for x,y in zip(aw,bw)))
            rows.append(item)
            common &= ai == bi
        print(json.dumps({"rows": rows, "strict_ids_equal": common,
                          "scope": "caller must establish identical initial prefixes and row offsets"}, indent=2))
        return 0 if common else 1
    assert len(sys.argv) == 3
    frozen = root / "native-noise-frozen.json"
    if action == "calibrate":
        assert not frozen.exists(), "refusing to replace an existing calibration"
        moves, ids, hashes = [], [], {}
        exact, difference = True, 0.0
        for i in range(ROWS):
            a, ah = row(root, "native-quality1", i)
            b, bh = row(root, "native-norm1", i)
            first, second = heapq.nlargest(2, range(VOCAB), key=a.__getitem__)
            assert first == best(b), f"native calibration prefixes diverge at {i}"
            ids.append(first)
            hashes[f"native-quality1/{i}"] = ah
            hashes[f"native-norm1/{i}"] = bh
            moves.append(abs((a[first] - a[second]) - (b[first] - b[second])))
            difference = max(difference, max(abs(x - y) for x, y in zip(a, b)))
            exact &= ah == bh
        result = {
            "sample_count": ROWS,
            "calibration": "native ordinary versus independently norm-fused, device masks",
            "p99_top2_margin_movement": sorted(moves)[math.ceil(.99 * ROWS) - 1],
            "native_greedy_ids": ids, "sha256": hashes,
            "full_logits_byte_exact": exact, "max_raw_delta": difference,
            "scope": "literal short screen only; no representative quality allowance",
        }
        frozen.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "sha256"}, indent=2))
        print("freeze_sha256=" + hashlib.sha256(frozen.read_bytes()).hexdigest())
        return 0
    assert action == "oracle"
    calibration = json.loads(frozen.read_text())
    deltas, nll_deltas, tvs = [], [], []
    mismatches, ids, hashes = [], [], {}
    common_prefix = True
    for i in range(ROWS):
        native, nh = row(root, "native-quality1", i)
        reference, rh = row(root, "reference-quality1", i)
        _, repeat_hash = row(root, "reference-quality2", i)
        assert rh == repeat_hash, f"reference own-repeat differs at {i}"
        assert nh == calibration["sha256"][f"native-quality1/{i}"]
        ni, ri = best(native), best(reference)
        assert ni == calibration["native_greedy_ids"][i]
        ids.append(ri)
        hashes[f"reference-quality1/{i}"] = rh
        if common_prefix:
            deltas.append(max(abs(x - y) for x, y in zip(native, reference)))
            nw, ns, nlse = distribution(native)
            rw, rs, rlse = distribution(reference)
            nll_deltas.append(abs((nlse - native[ri]) - (rlse - reference[ri])))
            tvs.append(.5 * math.fsum(abs(x / ns - y / rs) for x, y in zip(nw, rw)))
        if ni != ri:
            mismatches.append({"row": i, "native": ni, "reference": ri,
                               "common_prefix": common_prefix,
                               "reference_margin": reference[ri] - reference[ni]})
            common_prefix = False
    result = {
        "native_noise_freeze_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "frozen_bound": calibration["p99_top2_margin_movement"],
        "reference_repeat_byte_exact": True, "rows": ROWS,
        "strict_greedy_mismatches": mismatches, "common_prefix_rows": len(deltas),
        "max_raw_delta": max(deltas), "max_chosen_nll_delta": max(nll_deltas),
        "max_full_softmax_tv": max(tvs), "reference_ids": ids, "sha256": hashes,
    }
    (root / "quality-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sha256"}, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
