#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Exclusive bounded source/capture/native freeze, before original outputs."""
import argparse
import array
import datetime
import hashlib
import json
import math
import pathlib
import struct

NAMES = ("input", "gate", "up", "down", "activation", "down-output", "params")
SIZES = (5376 * 4, 65028240, 65028240, 94832640, 21504 * 4, 5376 * 4, 256)
SOURCE_FILES = (
    "benchmarks/CMakeLists.txt",
    "benchmarks/gemma_dense_ffn_capture.cc",
    "benchmarks/gemma_dense_ffn_capture.h",
    "benchmarks/gemma_dense_ffn_capture_wrap.cc",
    "benchmarks/gemma_dense_ffn_replay.cc",
    "docs/experiments/gemma-dense-ffn/replay_inputs.h",
    "docs/experiments/gemma-dense-ffn/llama_ffn.cc",
    "docs/experiments/gemma-dense-ffn/reference.sh",
    "docs/experiments/gemma-dense-ffn/PROTOCOL.md",
    "docs/experiments/gemma-dense-ffn/own_freeze.py",
)
IDS_SHA = "b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610"
ARTIFACT = "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08"
INDEX_SHA = "9ae365a17ed73528262d7befdb505bf07e6695e03f0fa36f69e69099dbd5f34c"


def identity(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(1 << 20):
            h.update(data)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def finite_file(path, size):
    assert path.stat().st_size == size, path
    v = array.array("f")
    with path.open("rb") as f:
        v.fromfile(f, size // 4)
    assert all(math.isfinite(x) for x in v), path
    return identity(path)


def retired(job):
    folder = pathlib.Path.home() / ".local/share/llmp/jobs" / job
    final = json.loads((folder / "final.json").read_text())
    assert final["state"] == "done" and final["rc"] == 0, final
    return {name: identity(folder / name) for name in ("final.json", "log", "job.json")}


def capture(path):
    m = json.loads((path / "operands.json").read_text())
    assert m["origin"] == "native dense31 scalar layer0"
    assert m["position"] == 67 and m["graphs"] is False and m["addresses_observed"] is True
    assert len(set(m["resource_indices"])) == 3
    result = {}
    for name, size in zip(NAMES, SIZES):
        p = path / (name + ".bin")
        assert p.stat().st_size == size, (name, size)
        if name in ("input", "activation", "down-output"):
            result[name] = finite_file(p, size)
        else:
            result[name] = identity(p)
        if name in ("gate", "up", "down"):
            d = m["tensors"][name]
            expected_ne = [21504, 5376, 1, 1] if name == "down" else [5376, 21504, 1, 1]
            assert d["ne"] == expected_ne and d["root_ne"] == expected_ne
            assert d["type"] == (14 if name == "down" else 12)
            assert d["nb"][1] == (17640 if name == "down" else 3024)
            assert d["relative_offset"] == 0 and d["root_bytes"] == (94832640 if name == "down" else 65028096)
            block = 210 if name == "down" else 144
            logical = 94832640 if name == "down" else 65028096
            offsets = (208,) if name == "down" else (0, 2)
            data = p.read_bytes()
            for offset in range(0, logical, block):
                assert all(math.isfinite(struct.unpack_from("<e", data, offset + k)[0]) for k in offsets), name
            assert not any(data[logical:]), name
    parameters = struct.unpack("<64i", (path / "params.bin").read_bytes())
    assert parameters == (0,) * 48 + (1,) + (0,) * 15
    assert m["op_params"] == [list(parameters[i:i+16]) for i in range(0, 64, 16)]
    result["metadata"] = identity(path / "operands.json")
    result["heads"] = finite_file(path / "heads.f32", 262144 * 4)
    assert identity(path / "inputs.i32") == {"bytes": 4096, "sha256": IDS_SHA}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("source", "capture", "native"))
    p.add_argument("root", type=pathlib.Path)
    p.add_argument("scratch", type=pathlib.Path)
    p.add_argument("job")
    p.add_argument("--source-record", default="source-frozen.json", choices=("source-frozen.json", "source-replay-frozen.json", "source-replay2-frozen.json"))
    p.add_argument("--native-dirs", nargs=2, default=("native-first", "native-repeat"))
    args = p.parse_args()
    root, base = args.root.resolve(), args.scratch.resolve()
    binary_paths = {
        "capture": root / "build/spark-native/benchmarks/llmp_gemma_dense_ffn_capture",
        "native": root / "build/spark-native/benchmarks/llmp_gemma_dense_ffn_replay",
        "original": base / "llama_ffn",
    }
    source = {name: identity(root / name) for name in SOURCE_FILES}
    binaries = {name: identity(path) for name, path in binary_paths.items()}
    assert identity(base / "inputs.i32") == {"bytes": 4096, "sha256": IDS_SHA}
    artifact = pathlib.Path.home() / ".local/share/llmp/m3-artifacts" / ARTIFACT
    assert identity(artifact / "manifest.json")["sha256"] == ARTIFACT
    assert identity(artifact / "index.json")["sha256"] == INDEX_SHA
    ancestry = json.loads((base / "source-ancestry.json").read_text())
    assert ancestry["base"] == "c67dd4d" and ancestry["tracked_production_diff"] == []
    assert set(ancestry["dirty_paths"]).issubset(SOURCE_FILES)
    assert ancestry["source_files"] == source
    libraries = dict(line.split()[::-1] for line in (base / "libraries.sha256").read_text().splitlines())
    assert libraries == {
        "/app/libggml-cuda.so": "5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e",
        "/app/libggml.so": "f09875d0593d950f0b1f4f57803235cb06209f774c524c005fb192000d5d7418",
        "/app/libggml-base.so": "ed0a6a3f65472059789159569080d08e7f1dcc8bdd7a35d127efbd1b3f4e273e",
    }
    environment = {
        "inputs": identity(base / "inputs.i32"),
        "artifact_manifest": identity(artifact / "manifest.json"),
        "artifact_index": identity(artifact / "index.json"),
        "build_receipt": identity(root / "build/spark-native/llmp-receipt.json"),
        "headers": {x.name: identity(x) for x in sorted((base / "headers").iterdir()) if x.is_file()},
        "original_library_witness": identity(base / "libraries.sha256"),
        "source_ancestry": identity(base / "source-ancestry.json"),
    }
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "mode": args.mode, "official_acquisition_retirement": retired(args.job)}
    if args.mode == "source":
        result.update({"base": "c67dd4d", "sources": source, "binaries": binaries,
                       "environment": environment, "ancestry": ancestry})
    else:
        pre = json.loads((base / args.source_record).read_text())
        assert pre["sources"] == source and pre["binaries"] == binaries
        assert pre["environment"] == environment
        result["source_freeze"] = identity(base / args.source_record)
        if args.mode == "capture":
            first, repeat = capture(base / "capture-first"), capture(base / "capture-repeat")
            for key in NAMES + ("heads",):
                assert first[key] == repeat[key], key
            controls = [finite_file(base / name / "heads.f32", 262144 * 4) for name in ("control-first", "control-repeat")]
            assert first["heads"] == controls[0] == controls[1]
            result.update({"first": first, "repeat": repeat, "controls": controls,
                           "own_complete_byte_noise": 0})
        else:
            cap = json.loads((base / "capture-frozen.json").read_text())
            assert capture(base / "capture-first") == cap["first"]
            capture_pre = json.loads((base / "source-frozen.json").read_text())
            assert cap["source_freeze"] == identity(base / "source-frozen.json")
            # A corrected standalone binder marker may be compiled after the
            # unchanged capture. Keep the original pre-capture proof immutable;
            # this is a separate replay freeze before any original outputs.
            for key in ("capture", "original"):
                assert capture_pre["binaries"][key] == binaries[key]
            for key in SOURCE_FILES:
                if key not in ("benchmarks/gemma_dense_ffn_replay.cc", "docs/experiments/gemma-dense-ffn/own_freeze.py"):
                    assert capture_pre["sources"][key] == source[key], key
            for key in ("inputs", "artifact_manifest", "artifact_index", "headers", "original_library_witness"):
                assert capture_pre["environment"][key] == environment[key], key
            assert identity(base / "capture-build-receipt.json") == capture_pre["environment"]["build_receipt"]
            assert identity(base / "capture-source-ancestry.json") == capture_pre["environment"]["source_ancestry"]
            result["capture_source_freeze"] = cap["source_freeze"]
            heads = []
            fresh = []
            for name in args.native_dirs:
                heads.extend(finite_file(base / name / f, 26880 * 4) for f in ("first.f32", "repeat.f32"))
                fresh.append(finite_file(base / name / "fresh.f32", 26880 * 4))
            assert all(x == heads[0] for x in heads)
            assert fresh[0] == fresh[1] and fresh[0] != heads[0]
            captured = (base / "capture-first/activation.bin").read_bytes() + (base / "capture-first/down-output.bin").read_bytes()
            assert hashlib.sha256(captured).hexdigest() == heads[0]["sha256"]
            result.update({"capture_freeze": identity(base / "capture-frozen.json"),
                           "full_outputs": heads, "fresh_outputs": fresh,
                           "actual_capture_reconstruction_exact": True, "own_complete_byte_noise": 0})
    target = base / (args.source_record if args.mode == "source" else args.mode + "-frozen.json")
    with target.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print("FROZEN", target, identity(target))


if __name__ == "__main__":
    main()
