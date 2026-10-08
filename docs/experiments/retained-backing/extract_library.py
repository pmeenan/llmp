#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Extract D-056's measured group sizes into library.json (headers only).

Runs artifact-layout's pinned planner (layout.py) over the model sources the
layout study planned and keeps only what the swap-trace generator needs: each
dense group's kind, layer and used/stored bytes, and each routed layer's expert
closure. Every total is checked against that study's results.json, so the
output is D-056's measured plan, not a new one.

    python3 extract_library.py OUT.json PROFILE=SOURCE[,SOURCE...] ...
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYOUT = HERE.parent / "artifact-layout"
LAYOUT_SHA256 = "a0d1980a9eddd1adf60863cf7b825396700691fd17b3cd62ca063fda2e0b6809"
PROFILES = ("qwen25", "exl3-4.0bpw", "exl3-4.5bpw", "gemma", "ornith", "qwen38", "dsv4")


def load_layout():
    path = LAYOUT / "layout.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != LAYOUT_SHA256:
        raise SystemExit("layout.py is not the planner results.json was measured with")
    spec = importlib.util.spec_from_file_location("layout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract(layout, profile, paths, measured):
    p = layout.plan(layout.load_sources(paths), tie_check=True)
    groups = p["groups"]
    s = layout.stats(p)
    for key in ("groups", "chunks", "group_used_bytes", "disk_bytes", "handle_backing_bytes"):
        if s[key] != measured[key]:
            raise SystemExit(f"{profile}: {key} {s[key]} differs from results.json {measured[key]}")
    dense, experts = [], {}
    for g in groups:
        if g["kind"] == "expert":
            closure = experts.setdefault(g["layer"], dict(layer=g["layer"], count=0, used=g["used"], stored=g["stored"]))
            if (closure["used"], closure["stored"]) != (g["used"], g["stored"]):
                raise SystemExit(f"{profile}: layer {g['layer']} expert closures are not uniform")
            closure["count"] += 1
        else:
            dense.append(dict(kind=g["kind"], layer=g["layer"], used=g["used"], stored=g["stored"]))
    return dict(profile=profile, model=measured["model"], architecture=p["arch"], experts=p["experts"],
                dense=dense, expert_layers=[experts[k] for k in sorted(experts)],
                totals={k: s[k] for k in ("groups", "chunks", "group_used_bytes", "disk_bytes", "handle_backing_bytes")})


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    layout = load_layout()
    results = json.loads((LAYOUT / "results.json").read_text())
    out = dict(format="llmp-retained-backing-library", version=1,
               planner=dict(path="docs/experiments/artifact-layout/layout.py", sha256=LAYOUT_SHA256),
               chunk_bytes=layout.CHUNK, file_align=layout.FILE_ALIGN, profiles=[])
    for arg in argv[2:]:
        profile, _, sources = arg.partition("=")
        if profile not in PROFILES or not sources:
            raise SystemExit(f"bad profile argument {arg!r}")
        out["profiles"].append(extract(layout, profile, sources.split(","), results["plans"][profile]))
    with open(argv[1], "x") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv)
