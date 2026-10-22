#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect the native arm using the existing supervised Service lifecycle."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tomllib


def stop(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--phase", choices=["short", "depth", "behavior"], default="short")
    parser.add_argument("--behavior-inputs", type=Path)
    parser.add_argument("--behavior-sha256")
    args = parser.parse_args()
    if args.phase == "behavior":
        if args.behavior_inputs is None or args.behavior_sha256 is None:
            parser.error("behavior phase requires the frozen input file and SHA256")
    elif args.ids is None:
        parser.error("quality phases require --ids")
    config = tomllib.loads(args.config.read_text())
    model = config.get("models", {}).get("llama33", {})
    expected = {"artifact": "9cca62f1e6582324de2ebce34f38b89a8aa8890cdced957a50782e407ff4d7ee",
                "context": 12288, "max_slots": 1, "prefill_chunk": 512,
                "kv_cache_encoding": "f16"}
    if any(model.get(key) != value for key, value in expected.items()):
        raise ValueError("native config differs from the registered model/geometry")
    if config.get("client", {}).get("port") != 18974:
        raise ValueError("native config does not select the registered loopback port")
    # Import the established runner rather than introduce another process/port
    # owner. Its process stays in the spark-job group and cleanup is waited.
    sys.path.insert(0, str(args.repo / "tools"))
    from speed_gate import Service
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"runtime": str(args.runtime.resolve()),
               "runtime_sha256": hashlib.sha256(args.runtime.read_bytes()).hexdigest(),
               "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
               "phase": args.phase}
    (args.out / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    service = None
    try:
        service = Service(args.runtime.resolve(), args.config.resolve(), args.out, 18974, {})
        service.wait_ready(timeout=420)
        if args.phase == "behavior":
            subprocess.run([
                "python3", str(Path(__file__).with_name("collect_behavior.py")),
                "--url", "http://127.0.0.1:18974", "--model", "llama33",
                "--inputs", str(args.behavior_inputs), "--inputs-sha256", args.behavior_sha256,
                "--out", str(args.out / "behavior"),
            ], check=True)
        else:
            for repeat in [1, 2]:
                subprocess.run([
                    "python3", str(Path(__file__).with_name("collect_quality.py")),
                    "--url", "http://127.0.0.1:18974", "--model", "llama33",
                    "--ids", str(args.ids), "--phase", args.phase,
                    "--out", str(args.out / f"repeat-{repeat}"),
                ], check=True)
    finally:
        if service is not None:
            (args.out / "retirement.json").write_text(
                json.dumps(service.stop(timeout=20), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
