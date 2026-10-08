#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the fused-route arms on a Spark with the unmodified pinned llama.cpp image.

All inputs and outputs other than this harness are external. Containers run
without network access. The rejected graph-output patch is not built here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import uuid

from compare_routes import compare_outputs, compare_records, compare_routes, read_lines, read_routes

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "pins.json").read_text())
IMAGE = PINS["reference"]["image"]
BASE_ENV = ["--env", "CUDA_DISABLE_PTX_JIT=1"]
NO_FUSION = ["--env", "GGML_CUDA_DISABLE_FUSION=1", "--env", "GGML_CUDA_DISABLE_GRAPHS=1"]
# name: (harness mode, extra environment); upstream fusion and CUDA graphs unless NO_FUSION
ARMS = {
    "untraced": (0, []),
    "legacy-split": (1, []),
    "fusion-boundary": (4, []),
    "untraced-nofusion": (0, NO_FUSION),
    "fusion-boundary-nofusion": (4, NO_FUSION),
}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def container(docker, base, tail, **kwargs):
    name = "llmp-fused-routes-" + uuid.uuid4().hex[:12]
    try:
        subprocess.run(base + ["--name", name] + tail, check=True, **kwargs)
    finally:
        result = subprocess.run(docker + ["rm", "-f", name], capture_output=True, text=True)
        if result.returncode and "No such container" not in result.stderr:
            raise RuntimeError("Container cleanup failed: " + result.stderr)


def verify_reference(reference, model_id):
    receipt = json.loads((reference / "receipt.json").read_text())
    runs = {r["name"]: r for r in receipt["runs"]}
    names = {}
    for trace in (0, 1):
        name = f"{model_id}-b512-r1-s0-t{trace}"
        run = runs[name]
        if sha(reference / (name + ".predictions")) != run["predictions_sha256"] or \
           sha(reference / (name + ".jsonl")) != run["events_sha256"]:
            raise ValueError("Reference capture mismatch: " + name)
        names[trace] = name
    if not (receipt["cuda_disable_fusion"] and receipt["cuda_disable_graphs"]):
        raise ValueError("Reference capture is not the fusion/graph-disabled configuration")
    return names


def require_equal_controls(comparisons):
    """Reject failed controls while leaving expected-difference arms diagnostic."""
    required = []
    for name in ("fusion_boundary_vs_untraced", "fusion_boundary_nofusion_vs_untraced_nofusion"):
        for kind in ("predictions", "logit_hashes"):
            row = comparisons[name][kind]
            required.append((f"{name}/{kind}", row, ("records",), ("different_records",)))
    name = "untraced_nofusion_vs_reference_untraced_predictions"
    required.append((name, comparisons[name], ("records",), ("different_records",)))
    name = "boundary_routes_nofusion_vs_reference_legacy_routes"
    required.append((name, comparisons[name], ("events", "rows"),
                     ("different_events", "different_ordered_rows", "different_expert_sets")))
    for name, row, counts, differences in required:
        if any(type(row[key]) is not int or row[key] <= 0 for key in counts) or \
           any(type(row[key]) is not int or row[key] != 0 for key in differences):
            raise ValueError(f"Required equality control failed: {name}: {row}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "models", "source", "inputs", "reference", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    os.umask(0o077)
    for k in ("spec", "models", "source", "inputs", "reference", "output"):
        setattr(a, k, getattr(a, k).resolve())
    spec = json.loads(a.spec.read_text())
    if sha(a.spec) != PINS["workload"]["spec_sha256"]:
        raise ValueError("Spec mismatch")
    models = [m for m in spec["models"] if m["id"] in PINS["workload"]["models"]]
    for m in models:
        for f in m["files"]:
            if sha(a.models / f["path"]) != f["sha256"]:
                raise ValueError("Model mismatch")
        if sha(a.inputs / m["tokens"]) != m["tokens_sha256"]:
            raise ValueError("Input mismatch")
    references = {m["id"]: verify_reference(a.reference, m["id"]) for m in models}
    for name, digest in PINS["reference"]["headers"].items():
        if sha(a.source / name) != digest:
            raise ValueError("Header mismatch: " + name)
    a.output.mkdir(mode=0o700)
    docker = shlex.split(os.environ.get("DOCKER", "docker"))
    base = docker + ["run", "--rm", "--network", "none", "--read-only", "--user", f"{os.getuid()}:{os.getgid()}",
                     "--tmpfs", "/tmp:rw,size=1g"]
    for path, dest, readonly in [(HERE, "/harness", True), (a.source, "/source", True), (a.inputs, "/inputs", True),
                                 (a.models, "/models", True), (a.output, "/output", False)]:
        base += ["--mount", f"type=bind,src={path},dst={dest}" + (",readonly" if readonly else "")]
    harness = PINS["harness"]["build"]
    container(docker, base, ["--entrypoint", harness[0], IMAGE, *harness[1:]])
    receipt = dict(image=IMAGE, revision=PINS["reference"]["source_revision"], spec_sha256=sha(a.spec),
                   harness_sha256=sha(HERE / "route_capture.cc"), runner_sha256=sha(Path(__file__)),
                   compare_sha256=sha(HERE / "compare_routes.py"), executable_sha256=sha(a.output / "capture"),
                   host=os.uname().nodename,
                   gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
                   arms=[], comparisons={})
    for m in models:
        for arm, (mode, env) in ARMS.items():
            name = f"{m['id']}-{arm}"
            (a.output / name).mkdir()
            tail = ["--device", "nvidia.com/gpu=all", *BASE_ENV, *env, "--entrypoint", "/output/capture", IMAGE,
                    "/models/" + m["files"][0]["path"], "/inputs/" + m["tokens"],
                    f"/output/{name}/events.jsonl", f"/output/{name}/predictions", f"/output/{name}/logits",
                    str(m["layers"]), str(m["experts"]), str(m["topk"]), str(m["context"]), "512", str(mode), "1", "0"]
            with (a.output / name / "log").open("x") as log:
                container(docker, base, tail, stdout=log, stderr=subprocess.STDOUT)
            text = (a.output / name / "log").read_text(errors="replace")
            offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
            if not offload or offload[1] != offload[2] or int(offload[1]) < m["layers"]:
                raise ValueError("Full GPU offload not verified")
            library_event = json.loads((a.output / name / "events.jsonl").read_text().splitlines()[0])
            if library_event != {"event": "library", "mode": mode, "libllama": "/app/libllama.so.0"}:
                raise ValueError(f"{name}: unexpected library {library_event}")
            receipt["arms"].append(dict(name=name, model=m["id"], mode=mode, env=env,
                                        **{f + "_sha256": sha(a.output / name / f)
                                           for f in ("events.jsonl", "predictions", "logits", "log")}))
            (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(name, "done", flush=True)
        d = lambda arm: a.output / f"{m['id']}-{arm}"
        ref = references[m["id"]]
        receipt["comparisons"][m["id"]] = {
            "fusion_boundary_vs_untraced": compare_outputs(d("untraced"), d("fusion-boundary")),
            "legacy_split_vs_untraced": compare_outputs(d("untraced"), d("legacy-split")),
            "fusion_boundary_nofusion_vs_untraced_nofusion":
                compare_outputs(d("untraced-nofusion"), d("fusion-boundary-nofusion")),
            "untraced_nofusion_vs_untraced": compare_outputs(d("untraced"), d("untraced-nofusion")),
            "untraced_nofusion_vs_reference_untraced_predictions":
                compare_records(read_lines(a.reference / (ref[0] + ".predictions"), 2),
                                read_lines(d("untraced-nofusion") / "predictions", 2)),
            "boundary_routes_nofusion_vs_reference_legacy_routes":
                compare_routes(read_routes(a.reference / (ref[1] + ".jsonl")),
                               read_routes(d("fusion-boundary-nofusion") / "events.jsonl")),
            "boundary_routes_vs_legacy_split_routes":
                compare_routes(read_routes(d("legacy-split") / "events.jsonl"),
                               read_routes(d("fusion-boundary") / "events.jsonl")),
            "boundary_routes_fusion_vs_nofusion":
                compare_routes(read_routes(d("fusion-boundary-nofusion") / "events.jsonl"),
                               read_routes(d("fusion-boundary") / "events.jsonl")),
            "reference": {"untraced": ref[0], "legacy_routes": ref[1]},
        }
        (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # Preserve the diagnostic comparisons even when acceptance fails.
        require_equal_controls(receipt["comparisons"][m["id"]])
    print("complete", flush=True)


if __name__ == "__main__":
    main()
