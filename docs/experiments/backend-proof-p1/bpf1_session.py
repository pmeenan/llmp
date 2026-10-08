#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""One BP-F1 timing session on a Spark (docs/backend-proof.md, "Performance protocol").

  bpf1_session.py OUT_DIR --harness BIN --cases FILE --arm-a KIND --arm-b KIND
                  --order primary|mirrored --source-commit SHA --source-diff-sha256 HEX
                  [--registry backend-proof.md --calibration FILE] [--note TEXT]

KIND is cuda-malloc, host-vmm or device-vmm; A is the reference
(cuda-malloc), B the candidate. A calibration or holdout session is A/A:
both arms cuda-malloc. The session:

- refuses to start if any compute process is on the GPU or the load
  average is above --max-load, and records the host, driver, GPU state,
  clock and power settings, the CPU governor, and the identities of the
  harness, the case file, the cuBLAS it loads, this script and the
  source tree (commit and the SHA-256 of its uncommitted diff, computed by
  the caller: the Spark has no checkout);
- runs one discarded warm-up process per arm, then one more immediately
  before the first timed block;
- runs eight timed blocks, each a fresh process of every case, in the order
  A1 B1 B2 A2 B3 A3 A4 B4 (primary) or B1 A1 A2 B2 A3 B3 B4 A4 (mirrored);
- records SM and memory clocks, temperature, power, P-state and active
  clock-throttle reasons before and after every block, and stops if
  another compute process appears on the GPU.

A VMM arm runs only under the BP-F1 rule pre-registered for its memory
kind: host VMM under rule v1 (D-034), device VMM under rule v2 (D-081).
The SHA-256 of --calibration, --harness and --cases must each equal the one
written in --registry (docs/backend-proof.md) as "BP-F1 calibration
SHA-256: `<hex>`", "BP-F1 harness SHA-256: `<hex>`" and "BP-F1 cases
SHA-256: `<hex>`" for v1, and as "BP-F1 v2 calibration SHA-256: `<hex>`"
and so on for v2. The harness then records the calibration's hash in each
block. bpf1_stats.py summarizes a session. Needs Python 3.12 and
nvidia-smi.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

KINDS = ("cuda-malloc", "host-vmm", "device-vmm")
# The pre-registered rule each VMM memory kind is compared under.
RULES = {"host-vmm": "v1", "device-vmm": "v2"}
ORDERS = {"primary": "A1 B1 B2 A2 B3 A3 A4 B4", "mirrored": "B1 A1 A2 B2 A3 B3 B4 A4"}
REGISTERED = re.compile(r"BP-F1 (?:(v2) )?(calibration|harness|cases) SHA-256: `([0-9a-f]{64})`")
GPU_STATE = ("clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,"
             "clocks_throttle_reasons.active")
GPU_SETTINGS = ("name,driver_version,persistence_mode,clocks.max.sm,clocks.max.mem,"
                "clocks.applications.graphics,power.limit,compute_mode")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered(registry_text, what="calibration", rule="v1"):
    """The SHA-256 of BP-F1's `what` (calibration, harness or cases) pre-registered in
    backend-proof.md for `rule` (v1 or v2), or None if there is not exactly one."""
    found = {digest for version, name, digest in REGISTERED.findall(registry_text)
             if name == what and (version or "v1") == rule}
    return found.pop() if len(found) == 1 else None


def registered_calibration(registry_text, rule="v1"):
    """The calibration SHA-256 pre-registered in backend-proof.md, or None if there is not exactly one."""
    return registered(registry_text, "calibration", rule)


def check_registration(arms, registry, calibration, harness=None, cases=None):
    """The calibration hash a VMM session runs under; None for a session without VMM.

    A VMM session also needs the harness binary and case file pre-registered with that
    calibration, under the rule for its memory kind (RULES)."""
    kinds = {kind for kind in arms if kind in RULES}
    if not kinds:
        return None
    if len(kinds) != 1:
        raise SystemExit("a session compares one VMM kind against cudaMalloc")
    kind = kinds.pop()
    rule = RULES[kind]
    if registry is None or calibration is None:
        raise SystemExit(f"a {kind} arm needs --registry and --calibration")
    text = Path(registry).read_text()
    checked = {}
    for what, path in (("calibration", calibration), ("harness", harness), ("cases", cases)):
        expected = registered(text, what, rule)
        if expected is None:
            raise SystemExit(f"{registry} pre-registers no BP-F1 rule {rule} {what}: {kind} is not compared")
        actual = sha256(path) if path is not None else None
        if actual != expected:
            raise SystemExit(f"the {what} {path} has SHA-256 {actual}, not the pre-registered {expected}")
        checked[what] = actual
    return checked["calibration"]


def smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return dict(zip(query.split(","), (v.strip() for v in out.split(","))))


def compute_apps():
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.strip()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def governor():
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return path.read_text().strip() if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--arm-a", choices=KINDS, required=True)
    parser.add_argument("--arm-b", choices=KINDS, required=True)
    parser.add_argument("--order", choices=ORDERS, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-diff-sha256", required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--note", default="")
    parser.add_argument("--max-load", type=float, default=1.0)
    args = parser.parse_args()

    arms = {"A": args.arm_a, "B": args.arm_b}
    if args.arm_a != "cuda-malloc":
        raise SystemExit("A, the reference, is cuda-malloc")
    calibration = check_registration(arms.values(), args.registry, args.calibration, args.harness, args.cases)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"session directory not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    busy = compute_apps()
    load = os.getloadavg()
    if busy:
        raise SystemExit(f"the GPU is in use:\n{busy}")
    if load[0] > args.max_load:
        raise SystemExit(f"load average {load[0]:.2f} is above {args.max_load}")

    cublas = args.harness.resolve().parent.parent / "cublas"
    manifest = {
        "session": args.out.name, "note": args.note, "started": now(),
        "order": ORDERS[args.order], "arms": arms,
        "host": {"node": platform.node(), "kernel": platform.release(), "machine": platform.machine(),
                 "load_average_at_start": load, "cpu_governor": governor(), "python": sys.version.split()[0]},
        "gpu_settings_at_start": smi(GPU_SETTINGS),
        "identities": {
            "harness_sha256": sha256(args.harness), "cases_sha256": sha256(args.cases),
            "session_script_sha256": sha256(__file__),
            "cublas": {p.name: sha256(p) for p in sorted(cublas.glob("libcublas*.so*"))},
            "source_commit": args.source_commit, "source_diff_sha256": args.source_diff_sha256,
        },
        "calibration_sha256": calibration,
        "blocks": [],
    }
    manifest_path = args.out / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    def run(arm, label):
        command = [str(args.harness), "--cases", str(args.cases), "--memory", arms[arm],
                   "--output", str(args.out / f"{label}.json")]
        if arms[arm] in RULES:
            command += ["--calibration-sha256", calibration]
        with open(args.out / f"{label}.log", "w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            manifest["failed"] = {"block": label, "exit": result.returncode, "at": now()}
            save()
            raise SystemExit(f"{label} exited {result.returncode}; see {args.out / (label + '.log')}")

    save()
    order = ORDERS[args.order].split()
    for arm in ("A", "B"):
        run(arm, f"warm-{arm}")
    run(order[0][0], "warm-first")
    for step in order:
        if busy := compute_apps():
            manifest["failed"] = {"block": step, "reason": "another compute process", "at": now()}
            save()
            raise SystemExit(f"another process is on the GPU before {step}:\n{busy}")
        before = {"at": now(), **smi(GPU_STATE), "load": os.getloadavg()[0]}
        run(step[0], step)
        after = {"at": now(), **smi(GPU_STATE), "load": os.getloadavg()[0]}
        manifest["blocks"].append({"block": step, "memory": arms[step[0]], "before": before, "after": after})
        save()
    manifest["ended"] = now()
    manifest["gpu_settings_at_end"] = smi(GPU_SETTINGS)
    manifest["cpu_governor_at_end"] = governor()
    manifest["compute_apps_at_end"] = compute_apps()
    save()
    print(f"session {args.out.name} done")


if __name__ == "__main__":
    main()
