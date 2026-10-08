#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""One host-VMM diagnosis session on a Spark (README.md in this directory).

  diag_session.py OUT_DIR --bench BIN [--repeats N] [--file F] [--dir D]
                  [--note TEXT] [--only info,micro,copy,restore,ggml]

Runs llmp_vmm_diag_bench (benchmarks/vmm_diag_bench.cc) as separate
processes: `info`; `micro` over every memory arm; `copy` between pairs of
arms; `restore` (with --dir, on the SSD to read from) in place into host
VMM and through a host-VMM landing zone copied into device VMM, at two and
four reads in flight; and `ggml` for each buffer placement, the placements in a rotated
order in each of N repeats. Before every process it waits until no
compute process is on the GPU and the load average is below --max-load, and
records the GPU's clocks, temperature, power and throttle reasons before
and after. Each process's CSV goes to OUT_DIR/<name>.csv, and
OUT_DIR/manifest.json records the host, the identities of the benchmark and
the cuBLAS it loads, and every process's conditions. Diagnostic only: no
gate reads it. Needs Python 3.12 and nvidia-smi.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

GPU_STATE = ("clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,"
             "clocks_throttle_reasons.active")
GPU_SETTINGS = ("name,driver_version,persistence_mode,clocks.max.sm,clocks.applications.graphics,"
                "power.limit,compute_mode")

MICRO_ARMS = ["malloc", "dvmm", "hvmm", "hvmm-jit", "hvmm-gpu", "hvmm-2m", "hvmm-host", "pinned",
              "registered", "pageable", "pageable-thp", "registered-thp", "managed",
              "managed-prefetch"]
INFO_ARMS = ["dvmm-cpu", *MICRO_ARMS]
COPY_PAIRS = [("hvmm", "dvmm"), ("hvmm", "malloc"), ("registered-thp", "dvmm"), ("pinned", "dvmm"),
              ("dvmm", "dvmm"), ("malloc", "malloc"), ("hvmm", "hvmm")]
# (weights, activations, scratch, cuBLAS workspace[, outputs apart from the
# activations])
PLACEMENTS = [
    ("malloc", "malloc", "malloc", "malloc"),
    ("dvmm", "dvmm", "dvmm", "dvmm"),
    ("hvmm", "hvmm", "hvmm", "hvmm"),
    ("hvmm-jit", "hvmm-jit", "hvmm-jit", "hvmm-jit"),
    ("hvmm", "malloc", "malloc", "malloc"),
    ("hvmm", "dvmm", "dvmm", "dvmm"),
    ("malloc", "hvmm", "malloc", "malloc"),
    ("malloc", "hvmm", "malloc", "malloc", "malloc"),
    ("malloc", "malloc", "malloc", "malloc", "hvmm"),
    ("malloc", "malloc", "hvmm", "hvmm"),
    ("hvmm-2m", "dvmm", "dvmm", "dvmm"),
    ("registered-thp", "dvmm", "dvmm", "dvmm"),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                         check=True, capture_output=True, text=True).stdout.strip()
    return dict(zip(query.split(","), (v.strip() for v in out.split(","))))


def compute_apps():
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
                          check=True, capture_output=True, text=True).stdout.strip()


def wait_idle(max_load, deadline_s=900):
    start = time.monotonic()
    while True:
        apps = compute_apps()
        load = os.getloadavg()[0]
        if not apps and load < max_load:
            return load
        if time.monotonic() - start > deadline_s:
            sys.exit(f"the host did not become idle: load {load}, compute apps {apps!r}")
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--file", help="a file for info's direct-read check")
    parser.add_argument("--dir", help="a directory on the SSD for restore's unnamed file")
    parser.add_argument("--max-load", type=float, default=0.5)
    parser.add_argument("--only", default="info,micro,copy,restore,ggml")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bench = args.bench.resolve()
    cublas = bench.parent.parent / "cublas"
    manifest = {
        "schema": 1,
        "started": now(),
        "note": args.note,
        "host": platform.node(),
        "kernel": platform.release(),
        "gpu": smi(GPU_SETTINGS),
        "governor": Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip(),
        "bench_sha256": sha256(bench),
        "cublas_sha256": {p.name: sha256(p) for p in sorted(cublas.glob("libcublas*.so.13"))},
        "script_sha256": sha256(__file__),
        "processes": [],
    }
    only = set(args.only.split(","))

    def run(name, argv):
        load = wait_idle(args.max_load)
        before = {"at": now(), **smi(GPU_STATE), "load": load}
        with open(args.out / f"{name}.csv", "x") as out, open(args.out / f"{name}.err", "x") as err:
            result = subprocess.run([str(bench), *argv], stdout=out, stderr=err, check=False)
        after = {"at": now(), **smi(GPU_STATE), "load": os.getloadavg()[0],
                 "compute_apps": compute_apps()}
        manifest["processes"].append({"name": name, "argv": argv, "status": result.returncode,
                                      "before": before, "after": after})
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"{name}: status {result.returncode}, SM {before['clocks.sm']} -> "
              f"{after['clocks.sm']}, {after['temperature.gpu']} C", flush=True)
        if result.returncode != 0:
            sys.exit(f"{name} failed; see {name}.err")

    if "info" in only:
        argv = ["info", *[a for arm in INFO_ARMS for a in ("--arm", arm)]]
        if args.file:
            argv += ["--file", args.file]
        run("info", argv)
    for repeat in range(args.repeats):
        if "micro" in only:
            arms = MICRO_ARMS[repeat % len(MICRO_ARMS):] + MICRO_ARMS[:repeat % len(MICRO_ARMS)]
            run(f"micro-r{repeat}",
                ["micro", "--rounds", "3", *[a for arm in arms for a in ("--arm", arm)]])
        if "copy" in only:
            for source, destination in COPY_PAIRS:
                run(f"copy-{source}-{destination}-r{repeat}",
                    ["copy", "--from", source, "--to", destination, "--rounds", "1"])
        if "restore" in only and args.dir:
            for depth in (2, 4):
                for method, landing, destination in (("none", "hvmm", "hvmm"),
                                                     ("memcpy", "hvmm", "dvmm"),
                                                     ("kernel", "hvmm", "dvmm")):
                    run(f"restore-{method}-d{depth}-r{repeat}",
                        ["restore", "--dir", args.dir, "--gib", "8", "--landing", landing,
                         "--to", destination, "--method", method, "--depth", str(depth),
                         "--rounds", "3"])
        if "ggml" in only:
            order = list(range(len(PLACEMENTS)))
            order = order[::-1] if repeat % 2 else order
            shift = repeat // 2
            order = order[shift:] + order[:shift]
            for index in order:
                weights, acts, scratch, workspace, *outputs = PLACEMENTS[index]
                run(f"ggml-p{index}-r{repeat}",
                    ["ggml", "--weights", weights, "--acts", acts, "--scratch", scratch,
                     "--workspace", workspace,
                     *[a for arm in outputs for a in ("--outputs", arm)]])
    manifest["finished"] = now()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
