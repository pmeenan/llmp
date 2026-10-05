#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""C4 ordinary-products candidate freeze; no solo/joined equality claim."""
import importlib.util
import json
import pathlib
import sys

source = pathlib.Path(__file__).resolve().parents[1] / "gemma-joined-serving/analyze.py"
spec = importlib.util.spec_from_file_location("joined_analysis", source)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def freeze(root):
    names = ("joined-first", "joined-repeat", "rows-norm-control")
    hashes = {name: analysis.digest(root / name / "heads.f32") for name in names}
    if len({hashes[name] for name in names[:2]}) != 1:
        raise ValueError("same-policy joined own-repeat full bytes differ")
    expected_input = "b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610"
    for name in names:
        path = root / name / "inputs.i32"
        if analysis.digest(path) != expected_input:
            raise ValueError("natural supplied inputs changed")
        analysis.read_inputs(path)
    candidate = analysis.record(root / "joined-first/heads.f32", 4)
    control = analysis.record(root / "rows-norm-control/heads.f32", 4)
    result = {"owners": 4, "vocab": analysis.VOCAB, "steps": analysis.STEPS,
              "source_identity_sha256": analysis.digest(root / "source-identities.json"),
              "files": hashes, "candidate": candidate, "rows_norm_control": control,
              "own_repeat_raw_noise": 0, "joined_own_repeat_byte_exact": True,
              "supplied_input_sha256": expected_input}
    path = root / "native-frozen.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    path.chmod(0o444)
    print(json.dumps({"freeze_sha256": analysis.digest(path), **result}))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("own_freeze.py ROOT")
    freeze(pathlib.Path(sys.argv[1]))
