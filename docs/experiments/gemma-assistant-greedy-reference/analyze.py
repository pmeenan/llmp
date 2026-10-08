#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded own freeze and strict one-unit comparison; no numerical allowance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

VOCAB = 262144
INPUT_SHA = "b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(1 << 20):
            h.update(data)
    return h.hexdigest()


def write_new(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def floats(path, count):
    require(path.is_file() and path.stat().st_size == count * 4, f"bad float extent: {path}")
    values = [v[0] for v in struct.iter_unpack("<f", path.read_bytes())]
    require(all(math.isfinite(v) for v in values), f"nonfinite output: {path}")
    return values


def tokens(path):
    require(path.stat().st_size == 16, "token array is not four rows")
    values = list(struct.unpack("<4i", path.read_bytes()))
    require(all(0 <= v < VOCAB for v in values), "noncanonical token")
    return values


def winner(row):
    return max(range(len(row)), key=row.__getitem__)


def judge(proposal, heads, vocab=VOCAB):
    matched = 0
    while matched < 3 and proposal[matched + 1] == winner(heads[matched * vocab:(matched + 1) * vocab]):
        matched += 1
    return matched + 1, winner(heads[matched * vocab:(matched + 1) * vocab])


def freeze(root, profile, engine, mode, source, source_sha, official, official_sha, out):
    require(digest(source) == source_sha and digest(official) == official_sha,
            "source/official admission identity differs")
    final = json.loads(official.read_text())
    require(final.get("rc") == 0 and final.get("state") == "done",
            "acquisition has not officially retired successfully")
    require(digest(root / "inputs.i32") == INPUT_SHA, "actual input identity differs")
    retired = json.loads((root / "retired.json").read_text())
    require(retired == {"successful_teardown": True, "rounds": 2}, "missing successful retirement")
    width = 5376 if profile == "31" else 2816
    first, repeat = root / "first", root / "repeat"
    files = []
    for path in sorted(first.iterdir()):
        require(path.is_file() and (repeat / path.name).is_file(), "missing repeated output")
        sha = digest(path)
        require(sha == digest(repeat / path.name), f"own repeat differs: {path.name}")
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha})
    require({p.name for p in first.iterdir()} == {p.name for p in repeat.iterdir()},
            "repeat output set differs")
    meta = json.loads((first / "completion.json").read_text())
    require(meta["profile"] == profile and meta["feature_width"] == width and
            meta["vocab"] == VOCAB and meta["mode"] == mode and meta["past"] == 64 and meta["prefill_query_rows"] == 64 and
            meta["verify_rows"] == 4 and meta["draft_protected_target"] is True and
            meta["pending_anchor_committed"] is False, "actual chronology differs")
    proposal = tokens(first / "proposal.i32")
    initial = floats(first / "initial-head.f32", VOCAB)
    heads = floats(first / "verify-heads.f32", 4 * VOCAB)
    features = floats(first / "verify-features.f32", 4 * width)
    floats(first / "initial-feature.f32", width)
    drafts = floats(first / "draft-heads.f32", 3 * VOCAB if mode == "unit" else 0)
    floats(first / "draft-features.f32", 3 * width if mode == "unit" else 0)
    keep, next_anchor = judge(proposal, heads) if mode == "unit" else (4, winner(heads[-VOCAB:]))
    require(meta["keep"] == keep and meta["completed_endpoint"] == 64 + keep and
            meta["next_anchor"] == next_anchor, "retired acceptance differs")
    committed = tokens(first / "committed.i32")
    require(committed == proposal[:keep] + [0] * (4 - keep), "pending/tail tokens committed")
    selected = (first / "selected-head.f32").read_bytes()
    selected_feature = (first / "selected-feature.f32").read_bytes()
    require(selected == (first / "verify-heads.f32").read_bytes()[(keep-1)*VOCAB*4:keep*VOCAB*4] and
            selected_feature == (first / "verify-features.f32").read_bytes()[(keep-1)*width*4:keep*width*4],
            "selected carry is not actual keep-1 row")
    if mode == "unit":
        require(proposal[0] == winner(initial), "initial anchor differs")
        require(proposal[1:] == [winner(drafts[i*VOCAB:(i+1)*VOCAB]) for i in range(3)],
                "actual draft proposal differs")
    else:
        ids = struct.unpack("<1024i", (root / "inputs.i32").read_bytes())
        require(proposal == list(ids[64:68]), "teacher control changed its predeclared block")
    if engine == "native":
        require(meta["same_four_query_control_exact"] is True and
                meta["rejected_before_sha256"] == meta["rejected_after_sha256"],
                "native independent control or rejected restoration failed")
    else:
        require(meta["semantic_cell_endpoint_checked"] is True and
                meta["physical_rejected_zero_claim"] is False, "public logical rejection differs")
        for name in ("protected-state.bin", "accepted-state.bin"):
            require(0 < (first / name).stat().st_size <= (64 if profile == "31" else 32) * (1 << 20),
                    "public opaque-state bound differs")
    write_new(out, {"version": 1, "profile": profile, "engine": engine, "mode": mode,
                    "source_sha256": source_sha, "official_final_sha256": official_sha,
                    "inputs_sha256": INPUT_SHA, "retired_sha256": digest(root / "retired.json"),
                    "own_full_bytes_exact": True, "all_outputs_finite": True,
                    "proposal": proposal, "keep": keep, "next_anchor": next_anchor,
                    "files": files})


def row_compare(a, b):
    native, reference = winner(a), winner(b)
    margin = b[reference] - b[native]
    return {"native_winner": native, "reference_winner": reference,
            "reference_margin": margin, "strict_positive_difference": native != reference and margin > 0,
            "exact_tie_difference": native != reference and margin == 0,
            "max_raw_delta": max(abs(x-y) for x, y in zip(a, b))}


def nll(row, target):
    maximum = max(row)
    return maximum + math.log(math.fsum(math.exp(v-maximum) for v in row)) - row[target]


def compare(native, reference, native_own, native_sha, reference_own, reference_sha, out):
    require(digest(native_own) == native_sha and digest(reference_own) == reference_sha,
            "externally frozen own-proof identities differ")
    own_n, own_r = json.loads(native_own.read_text()), json.loads(reference_own.read_text())
    require(own_n["profile"] == own_r["profile"] and own_n["mode"] == own_r["mode"] and
            own_n["engine"] == "native" and own_r["engine"] == "reference" and
            own_n["own_full_bytes_exact"] and own_r["own_full_bytes_exact"], "own gates differ")
    for root, own in ((native, own_n), (reference, own_r)):
        require(own["inputs_sha256"] == INPUT_SHA and digest(root / "inputs.i32") == INPUT_SHA and
                digest(root / "retired.json") == own["retired_sha256"], "frozen global input/retirement changed")
        for entry in own["files"]:
            require(digest(root / "first" / entry["path"]) == entry["sha256"], "frozen output changed")
    pn, pr = own_n["proposal"], own_r["proposal"]
    divergence = next((i for i, (a, b) in enumerate(zip(pn, pr)) if a != b), None)
    result = {"version": 1, "profile": own_n["profile"], "mode": own_n["mode"],
              "native_own_sha256": native_sha, "reference_own_sha256": reference_sha,
              "proposal_gate_pass": divergence is None, "first_proposal_divergence": divergence,
              "native_proposal": pn, "reference_proposal": pr}
    if divergence is not None:
        result["qualification"] = "Different transaction operands; no full-transaction comparison or numerical waiver."
        write_new(out, result)
        return 1
    native, reference = native / "first", reference / "first"
    width = 5376 if own_n["profile"] == "31" else 2816
    rows = []
    a0, b0 = floats(native / "initial-head.f32", VOCAB), floats(reference / "initial-head.f32", VOCAB)
    av, bv = floats(native / "verify-heads.f32", 4 * VOCAB), floats(reference / "verify-heads.f32", 4 * VOCAB)
    rows.append({"phase": "initial", **row_compare(a0, b0)})
    for row in range(4):
        rows.append({"phase": "verify", "row": row,
                     **row_compare(av[row*VOCAB:(row+1)*VOCAB], bv[row*VOCAB:(row+1)*VOCAB])})
    assistant_rows = []
    if own_n["mode"] == "unit":
        ad = floats(native / "draft-heads.f32", 3 * VOCAB)
        bd = floats(reference / "draft-heads.f32", 3 * VOCAB)
        for row in range(3):
            assistant_rows.append({"row": row, **row_compare(ad[row*VOCAB:(row+1)*VOCAB],
                                                             bd[row*VOCAB:(row+1)*VOCAB])})
    feature_summary = {}
    for name, count in (("initial-feature.f32", width), ("draft-features.f32", 3*width if own_n["mode"] == "unit" else 0),
                        ("verify-features.f32", 4*width), ("selected-feature.f32", width)):
        a, b = floats(native / name, count), floats(reference / name, count)
        feature_summary[name] = {"byte_exact": digest(native / name) == digest(reference / name),
                                 "max_raw_delta": max((abs(x-y) for x, y in zip(a, b)), default=0.0)}
    # Conditional continuation under the generated four-token proposal. The
    # final verified row is intentionally not assigned a fifth likelihood target.
    ids = struct.unpack("<1024i", (native.parent / "inputs.i32").read_bytes())
    na = [nll(a0, ids[64])] + [nll(av[row*VOCAB:(row+1)*VOCAB], ids[65+row]) for row in range(3)]
    nb = [nll(b0, ids[64])] + [nll(bv[row*VOCAB:(row+1)*VOCAB], ids[65+row]) for row in range(3)]
    result.update({"target_rows": rows, "assistant_rows": assistant_rows, "features": feature_summary,
                   "native_keep": own_n["keep"], "reference_keep": own_r["keep"],
                   "native_next_anchor": own_n["next_anchor"], "reference_next_anchor": own_r["next_anchor"],
                   "strict_positive_differences": sum(r["strict_positive_difference"] for r in rows),
                   "exact_tie_differences": sum(r["exact_tie_difference"] for r in rows),
                   "conditional_targets": 4, "native_conditional_nll": math.fsum(na)/4,
                   "reference_conditional_nll": math.fsum(nb)/4,
                   "conditional_loss_ratio_delta": math.expm1((math.fsum(na)-math.fsum(nb))/4),
                   "final_verify_row_scored": False, "cross_engine_kv_byte_claim": False})
    an, ar = (native / "initial-head.f32").read_bytes(), (reference / "initial-head.f32").read_bytes()
    vn, vr = (native / "verify-heads.f32").read_bytes(), (reference / "verify-heads.f32").read_bytes()
    result["target_byte_exact_rows"] = int(an == ar) + sum(vn[i*VOCAB*4:(i+1)*VOCAB*4] == vr[i*VOCAB*4:(i+1)*VOCAB*4] for i in range(4))
    result["strict_choice_gate_pass"] = result["strict_positive_differences"] == 0
    result["conditional_loss_gate_pass"] = result["conditional_loss_ratio_delta"] <= 0.03
    result["acceptance_carry_gate_pass"] = own_n["keep"] == own_r["keep"] and own_n["next_anchor"] == own_r["next_anchor"]
    result["qualification"] = "One matched C1/P64 unit; conditional loss is not corpus PPL, serving or scalar-width qualification."
    write_new(out, result)
    return 0 if result["strict_choice_gate_pass"] and result["conditional_loss_gate_pass"] and result["acceptance_carry_gate_pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    own = sub.add_parser("own")
    for name in ("root", "source", "official", "out"):
        own.add_argument(name, type=Path)
    own.add_argument("source_sha")
    own.add_argument("official_sha")
    own.add_argument("profile", choices=("26", "31"))
    own.add_argument("engine", choices=("native", "reference"))
    own.add_argument("mode", choices=("unit", "teacher"))
    cross = sub.add_parser("compare")
    for name in ("native", "reference", "native_own", "reference_own", "out"):
        cross.add_argument(name, type=Path)
    cross.add_argument("native_sha")
    cross.add_argument("reference_sha")
    args = parser.parse_args()
    if args.command == "own":
        freeze(args.root, args.profile, args.engine, args.mode, args.source, args.source_sha,
               args.official, args.official_sha, args.out)
        return 0
    return compare(args.native, args.reference, args.native_own, args.native_sha,
                   args.reference_own, args.reference_sha, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
