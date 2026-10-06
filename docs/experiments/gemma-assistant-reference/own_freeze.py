#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate one bounded original-image acquisition; raw arrays stay external."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import sys

INPUT_SHA = "b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610"
PIN = "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4"
IMAGE = "sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7"


def identity(path):
    assert path.is_file() and not path.is_symlink(), path
    sha = hashlib.sha256()
    with path.open("rb") as file:
        while data := file.read(1 << 20):
            sha.update(data)
    return {"bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def exact(path, size, kind=None):
    assert path.stat().st_size == size, (path, size)
    data = path.read_bytes()
    if kind:
        values = array.array(kind)
        values.frombytes(data)
        assert sys.byteorder == "little"
        if kind == "f":
            assert values.itemsize == 4 and all(math.isfinite(x) for x in values), path
        else:
            assert values.itemsize == 4 and all(0 <= x < 262144 for x in values), path
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--profile", choices=("26", "31"), default="26")
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    profile31 = args.profile == "31"
    width, local_width, global_width = (5376, 4096, 2048) if profile31 else (2816, 2048, 1024)
    local_layer, global_layer = (58, 59) if profile31 else (28, 29)
    profile = "31B" if profile31 else "26B-A4B"
    completion = json.loads((root / "completion.json").read_text())
    count = completion["owners"]
    assert count in (1, 2) and completion["version"] == 1
    assert not profile31 or (count == 1 and completion["mode"] == "serial")
    assert completion["mode"] in ("serial", "batch")
    assert completion["physical_draft_batch"] == (count if completion["mode"] == "batch" else 1)
    assert count == 2 or completion["mode"] == "serial"
    assert all(completion[key] is True for key in (
        "target_state_unchanged", "same_shape_own_repeat_byte_exact", "constant_query_position"))
    assert completion["step_order"] == "step-owner-value"
    assert identity(root / "inputs.i32") == {"bytes": 4096, "sha256": INPUT_SHA}
    expected = {"inputs.i32", "completion.json"}
    for owner in range(count):
        prefix = f"owner-{owner}/"
        directory = root / prefix
        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["version"] == 1 and metadata["profile"] == profile
        assert metadata["owner"] == metadata["stream"] == metadata["sequence"] == owner
        assert metadata["completed_endpoint"] == metadata["query_position"] == 64 + owner
        assert metadata["feature_position"] == 63 + owner
        assert metadata["feature_width"] == width and metadata["vocabulary"] == 262144
        assert metadata["local_window"] == 1024 and metadata["read_cells"] == 256
        assert metadata["local_capacity"] == 1280 and metadata["global_capacity"] == 4096
        assert metadata["local_layer"] == local_layer and metadata["global_layer"] == global_layer
        exact(directory / "feature.f32", width * 4, "f")
        exact(directory / "anchor.i32", 4, "i")
        assert 0 < (directory / "target-state.bin").stat().st_size <= (64 if profile31 else 32) << 20
        expected |= {prefix + name for name in (
            "metadata.json", "feature.f32", "anchor.i32", "target-state.bin")}
        for name, cache_width, capacity in (("local", local_width, 1280), ("global", global_width, 4096)):
            for role in ("k", "v"):
                exact(directory / f"{name}-{role}.f16", cache_width * 256 * 2)
                physical = exact(directory / f"{name}-physical-{role}.f16", cache_width * capacity * 2)
                assert physical[:cache_width * 256 * 2] == (directory / f"{name}-{role}.f16").read_bytes()
                assert not any(physical[cache_width * (64 + owner) * 2:])
                expected |= {prefix + f"{name}-{role}.f16", prefix + f"{name}-physical-{role}.f16"}
            positions = array.array("i")
            positions.frombytes(exact(directory / f"{name}-positions.i32", capacity * 4))
            membership = exact(directory / f"{name}-membership.u8", capacity)
            assert list(positions[:64 + owner]) == list(range(64 + owner))
            assert all(x == -1 for x in positions[64 + owner:])
            assert membership == bytes([1] * (64 + owner) + [0] * (capacity - 64 - owner))
            expected |= {prefix + f"{name}-positions.i32", prefix + f"{name}-membership.u8"}
    for steps in (1, 3):
        prefix = f"steps-{steps}/"
        directory = root / prefix
        incoming = exact(directory / "incoming-feature.f32", steps * count * width * 4, "f")
        anchors = exact(directory / "incoming-anchor.i32", steps * count * 4, "i")
        heads = exact(directory / "heads.f32", steps * count * 262144 * 4, "f")
        projected = exact(directory / "postprojection.f32", steps * count * width * 4, "f")
        assert heads == exact(directory / "repeat-heads.f32", len(heads), "f")
        assert projected == exact(directory / "repeat-postprojection.f32", len(projected), "f")
        for owner in range(count):
            assert incoming[owner * width * 4:(owner + 1) * width * 4] == (
                root / f"owner-{owner}/feature.f32").read_bytes()
            assert anchors[owner * 4:(owner + 1) * 4] == (root / f"owner-{owner}/anchor.i32").read_bytes()
        for step in range(1, steps):
            for owner in range(count):
                row = step * count + owner
                previous = row - count
                assert incoming[row * width * 4:(row + 1) * width * 4] == (
                    projected[previous * width * 4:(previous + 1) * width * 4])
                values = array.array("f")
                values.frombytes(heads[previous * 262144 * 4:(previous + 1) * 262144 * 4])
                winner = max(range(262144), key=values.__getitem__)
                chosen = array.array("i")
                chosen.frombytes(anchors[row * 4:(row + 1) * 4])
                assert chosen[0] == winner
        expected |= {prefix + name for name in (
            "incoming-feature.f32", "incoming-anchor.i32", "heads.f32", "postprojection.f32",
            "repeat-heads.f32", "repeat-postprojection.f32")}
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    assert actual == expected, (actual - expected, expected - actual)
    for path in root.rglob("*"):
        assert not path.is_symlink() and path.stat().st_mode & 0o077 == 0, path
    headers = json.loads((args.work / "header-closure.json").read_text())
    for name, record in headers.items():
        assert identity(args.work / "headers" / name) == record
    source = {name: identity(args.source / name) for name in (
        "llama_assistant.cc", "reference.sh", "PROTOCOL.md", "own_freeze.py")}
    before = json.loads(args.pre.read_text())
    assert source == before["source_files"]
    assert headers == before["headers"]
    assert identity(args.work / "llama_assistant")["sha256"] == before["client_sha256"]
    assert before["input"] == identity(root / "inputs.i32")
    if profile31:
        assert before["profile"] == "31B"
        assert before["opaque_state_bound_bytes"] == 64 << 20
        assert before["known_vector_bound_bytes"] == 384 << 20
        retired = json.loads((args.work / (before["owned_name"] + "-container-retired.json")).read_text())
        assert retired["container_absent_after_checked_docker_query"] is True
    final = json.loads((args.job / "final.json").read_text())
    assert final["rc"] == 0 and final["state"] == "done", final
    steps_record = json.loads((args.job / "steps.json").read_text())
    assert len(steps_record["steps"]) == 1
    assert steps_record["steps"][0]["state"] == "done" and steps_record["steps"][0]["rc"] == 0
    receipt = {
        "schema": 1, "source_pin": PIN, "image": IMAGE,
        "completion": completion, "profile": profile, "canonical_input_sha256": INPUT_SHA,
        "client": identity(args.work / "llama_assistant"), "sources": source,
        "pre_acquisition_source_freeze": identity(args.pre),
        "supervised_retirement": {"final": identity(args.job / "final.json"),
                                  "steps": identity(args.job / "steps.json"),
                                  "log": identity(args.job / "log")},
        "headers": headers,
        "original_library_symbols_log": identity(args.work / "symbols1.log"),
        "files": {name: identity(root / name) for name in sorted(expected)},
        "calibration_boundary": "stage0 inputs only before native endogenous freeze; later stock inputs withheld",
    }
    fd = os.open(args.receipt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as file:
        json.dump(receipt, file, indent=2)
        file.write("\n")
    print(f"REFERENCE_OWN_FREEZE owners={count} mode={completion['mode']} files={len(expected)} sha256={identity(args.receipt)['sha256']}")


if __name__ == "__main__":
    main()
