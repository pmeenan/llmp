#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Own repeat and equal-output gates for fixed32; no numerical allowance."""
import argparse
import json
import math
from pathlib import Path
import struct

from analyze import INPUT_SHA, VOCAB, digest, require, row_compare, winner, write_new

WIDTH = 5376
PHASES = ("quality-first", "quality-repeat", "timed-first", "timed-repeat")


def read_json(path, expected=None):
    if expected is not None:
        require(digest(path) == expected, f"JSON identity changed: {path}")
    return json.loads(path.read_text())


def read_tokens(path):
    require(path.stat().st_size == 32 * 4, "fixed32 token extent changed")
    result = list(struct.unpack("<32i", path.read_bytes()))
    require(all(0 <= x < VOCAB for x in result), "fixed32 noncanonical token")
    return result


def finite_file(path, count):
    require(path.stat().st_size == count * 4, f"bad F32 extent: {path}")
    # One vocabulary row at a time; no full-trace Python float list.
    with path.open("rb") as f:
        while block := f.read(4 * VOCAB):
            require(len(block) % 4 == 0, "partial float")
            require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", block)),
                    f"nonfinite output: {path}")


def float_row(path, index=0):
    with path.open("rb") as f:
        f.seek(index * VOCAB * 4)
        data = f.read(VOCAB * 4)
    require(len(data) == VOCAB * 4, "missing predictive row")
    return list(struct.unpack(f"<{VOCAB}f", data))


def counters(metadata, tokens):
    require(metadata["emitted"] == 32 and metadata["endpoint"] == 96 and
            metadata["prefill_queries"] == 64 and metadata["max_rows"] == 128 and
            metadata["context"] == 4096 and metadata["eog_ignored"] is True and
            metadata["eos_id"] == 106 and
            metadata["eos_occurrences"] == tokens.count(106), "fixed32 recipe mismatch")
    spec = metadata["mode"] == "unit32"
    require(metadata["mode"] in ("plain32", "unit32"), "unknown fixed32 mode")
    count = rows = drafts = 0
    require(1 <= len(metadata["units"]) <= 32, "bad unit count")
    for unit in metadata["units"]:
        depth = min(3, 31-count) if spec and count < 31 else 0
        require(unit["past"] == 64+count and unit["depth"] == depth and
                unit["verified_rows"] == depth+1 and 1 <= unit["keep"] <= depth+1 and
                unit["keep"] <= 32-count, "bad unit envelope")
        count += unit["keep"]
        rows += depth+1
        drafts += depth
    require(count == 32 and rows == metadata["verified_rows"] and
            drafts == metadata["drafted_rows"], "fixed32 work accounting mismatch")
    require(math.isfinite(metadata["seconds"]) and metadata["seconds"] >= 0,
            "invalid elapsed")
    return rows, sum(u["depth"] > 0 for u in metadata["units"])


def own(root, engine, mode, source, source_sha, official, official_sha, output):
    require(engine in ("native", "reference") and mode in ("plain32", "unit32"),
            "closed31 performance arm required")
    source_data = read_json(source, source_sha)
    done = read_json(official, official_sha)
    require(done["state"] == "done" and done["rc"] == 0, "producer did not retire successfully")
    require(source_data["profile"] == "31" and source_data["engine"] == engine and
            source_data["mode"] == mode, "source binding mismatch")
    require(digest(root/"inputs.i32") == INPUT_SHA, "input identity changed")
    retired = read_json(root/"retired.json")
    require(retired == {"successful_teardown": True, "warm": 1,
                       "quality_rounds": 2, "timed_rounds": 2}, "teardown recipe mismatch")
    require({x.name for x in root.iterdir()} == set(PHASES) | {"warm", "inputs.i32", "retired.json"}
            and (root/"warm").is_dir() and not list((root/"warm").iterdir()),
            "unexpected fixed32 root/warm outputs")
    phases = {}
    files = []
    for phase in PHASES:
        directory = root/phase
        tokens = read_tokens(directory/"tokens.i32")
        metadata = read_json(directory/"completion.json")
        rows, last_rows = counters(metadata, tokens)
        require(metadata["mode"] == mode and metadata["phase"] ==
                ("quality" if phase.startswith("quality") else "timed"), "phase mismatch")
        limit = (256 if mode == "unit32" else 64) << 20 if engine == "native" else None
        if limit is not None:
            require(metadata["actual_vector_capacity_bytes"] <= limit-(1 << 20),
                    "native actual caller capacity exceeds grant")
        else:
            bound = (384 if mode == "unit32" else 192) << 20
            require(metadata["actual_vector_capacity_bytes"] + (129 << 20) <= bound,
                    "original known vector capacity exceeds bound")
        expected = {"tokens.i32", "completion.json", "final-head.f32"}
        finite_file(directory/"final-head.f32", VOCAB)
        if engine == "reference":
            require(metadata["semantic_cell_endpoint_checked"] is True and
                    metadata["physical_rejected_zero_claim"] is False, "original cell witness absent")
            expected.add("end-state.bin")
            require(0 < (directory/"end-state.bin").stat().st_size <= 128 << 20,
                    "original state snapshot exceeds bound")
        else:
            require(len(metadata["semantic_state_sha256"]) == 64 and
                    all(c in "0123456789abcdef" for c in metadata["semantic_state_sha256"]),
                    "native completed state witness absent")
        if mode == "unit32":
            expected.add("final-feature.f32")
            finite_file(directory/"final-feature.f32", WIDTH)
        if phase.startswith("quality"):
            expected.add("prediction-heads.f32")
            finite_file(directory/"prediction-heads.f32", 32*VOCAB)
            for i, token in enumerate(tokens):
                require(winner(float_row(directory/"prediction-heads.f32", i)) == token,
                        "emitted token is not target-authoritative")
            if mode == "unit32":
                for name, count in (("initial-feature.f32", WIDTH),
                                    ("verify-heads.f32", rows*VOCAB),
                                    ("verify-features.f32", rows*WIDTH),
                                    ("last-draft-heads.f32", last_rows*VOCAB)):
                    expected.add(name)
                    finite_file(directory/name, count)
        require({x.name for x in directory.iterdir()} == expected,
                "unexpected/missing fixed32 phase outputs")
        for name in sorted(expected):
            path = directory/name
            files.append({"path": f"{phase}/{name}", "bytes": path.stat().st_size,
                          "sha256": digest(path)})
        phases[phase] = metadata
    # Quality traces have identical chronology and complete bytes, including metadata.
    first = root/"quality-first"
    for path in first.iterdir():
        require(digest(path) == digest(root/"quality-repeat"/path.name),
                "full quality own repeat differs")
    for phase in ("timed-first", "timed-repeat"):
        for name in ("tokens.i32", "final-head.f32") + (("final-feature.f32",) if mode == "unit32" else ()) + (("end-state.bin",) if engine == "reference" else ()):
            require(digest(first/name) == digest(root/phase/name),
                    "timed output differs from quality witness")
        a, b = dict(phases["quality-first"]), dict(phases[phase])
        for d in (a, b):
            d.pop("seconds")
            d.pop("phase")
        require(a == b and phases[phase]["seconds"] > 0, "timed chronology/state differs")
    write_new(output, {"schema": 1, "engine": engine, "mode": mode, "profile": "31",
                       "source_sha256": source_sha, "official_final_sha256": official_sha,
                       "inputs_sha256": INPUT_SHA, "retired_sha256": digest(root/"retired.json"),
                       "full_quality_repeat_exact": True, "timed_matches_quality": True,
                       "all_retained_outputs_finite": True, "files": files,
                       "phases": phases})


def admit(root, proof, expected):
    frozen = read_json(proof, expected)
    require(frozen["full_quality_repeat_exact"] and frozen["timed_matches_quality"] and
            frozen["all_retained_outputs_finite"], "own proof failed")
    require(digest(root/"inputs.i32") == frozen["inputs_sha256"] == INPUT_SHA and
            digest(root/"retired.json") == frozen["retired_sha256"], "global frozen input changed")
    for f in frozen["files"]:
        p = root/f["path"]
        require(p.stat().st_size == f["bytes"] and digest(p) == f["sha256"],
                "frozen phase payload changed")
    return frozen


def first_divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def pair(left, right):
    a, b = read_tokens(left/"quality-first/tokens.i32"), read_tokens(right/"quality-first/tokens.i32")
    first = first_divergence(a, b)
    if first is not None:
        row = row_compare(float_row(left/"quality-first/prediction-heads.f32", first),
                          float_row(right/"quality-first/prediction-heads.f32", first))
        return {"sequence_equal": False, "first_common_prefix_divergence": first,
                "divergence_head": row, "later_different_histories_compared": False}
    final = row_compare(float_row(left/"quality-first/final-head.f32"),
                        float_row(right/"quality-first/final-head.f32"))
    return {"sequence_equal": True, "first_common_prefix_divergence": None,
            "pending_anchor_equal": winner(float_row(left/"quality-first/final-head.f32")) ==
                                    winner(float_row(right/"quality-first/final-head.f32")),
            "pending_head": final}


def compare(roots, proofs, hashes, output):
    names = ("native_plain", "native_unit", "reference_plain", "reference_unit")
    admitted = {n: admit(r, p, h) for n, r, p, h in zip(names, roots, proofs, hashes)}
    for n, d in admitted.items():
        require(d["engine"] == ("native" if n.startswith("native") else "reference") and
                d["mode"] == ("plain32" if n.endswith("plain") else "unit32"), "arm binding swapped")
    gates = {"native_unit_vs_plain": pair(roots[1], roots[0]),
             "reference_unit_vs_plain": pair(roots[3], roots[2]),
             "native_plain_vs_reference_plain": pair(roots[0], roots[2]),
             "native_unit_vs_reference_unit": pair(roots[1], roots[3])}
    passed = all(g["sequence_equal"] and g.get("pending_anchor_equal", False) for g in gates.values())
    times = {n: [d["phases"][p]["seconds"] for p in ("timed-first", "timed-repeat")]
             for n, d in admitted.items()}
    work = {}
    for name, proof in admitted.items():
        phase = proof["phases"]["quality-first"]
        histogram = {str(keep): sum(u["keep"] == keep for u in phase["units"])
                     for keep in range(1, 5)}
        work[name] = {"committed": 32, "units": len(phase["units"]),
                      "accepted_prefix_distribution": histogram,
                      "draft_rows": phase["drafted_rows"],
                      "target_query_and_published_head_rows": phase["verified_rows"],
                      "scalar_tail_units": sum(u["depth"] == 0 for u in phase["units"]),
                      "eos_occurrences_ignored": phase["eos_occurrences"]}
    result = {"schema": 1, "requested_committed_tokens": 32, "endpoint": 96,
              "eog_ignored_literal_diagnostic": True, "continuation_gates": gates,
              "continuation_pass": passed, "own_proof_sha256": dict(zip(names, hashes)),
              "timed_seconds": times, "timed_spread_seconds":
                  {n: max(v)-min(v) for n, v in times.items()},
              "work": work, "timing_interpretation_qualified": passed,
              "heads_scalar_vs_four_byte_equality_assumed": False,
              "no_numerical_allowance": True,
              "qualification": "Short literal32 screen; no chat-stop, serving, sampling, 26B or sustained performance claim."}
    if passed:
        means = {n: sum(v)/len(v) for n, v in times.items()}
        result["mean_seconds"] = means
        result["unit_relative_to_plain"] = {e: means[e+"_unit"]/means[e+"_plain"]-1
                                            for e in ("native", "reference")}
        result["native_unit_relative_to_reference_unit"] = means["native_unit"]/means["reference_unit"]-1
    write_new(output, result)
    return passed


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("own")
    for name in ("root", "source", "official", "output"):
        p.add_argument(name, type=Path)
    for name in ("source_sha", "official_sha", "engine", "mode"):
        p.add_argument(name)
    p = sub.add_parser("compare")
    p.add_argument("roots", type=Path, nargs=4)
    p.add_argument("proofs", type=Path, nargs=4)
    p.add_argument("hashes", nargs=4)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "own":
        own(args.root, args.engine, args.mode, args.source, args.source_sha,
            args.official, args.official_sha, args.output)
        return 0
    return 0 if compare(args.roots, args.proofs, args.hashes, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
