#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect the native arm using the existing supervised Service lifecycle."""
import argparse
import hashlib
import json
import os
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
    parser.add_argument("--phase", choices=["short", "depth", "behavior", "fibonacci512"], default="short")
    parser.add_argument("--behavior-inputs", type=Path)
    parser.add_argument("--behavior-sha256")
    args = parser.parse_args()
    if args.phase in {"behavior", "fibonacci512"}:
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
    supplement = args.phase == "fibonacci512"
    if supplement:
        import fibonacci512 as fib
        os.umask(0o077)
        fib.inputs(args.behavior_inputs, args.behavior_sha256)
        fib.native_preflight(args.runtime, config, args.repo)
    # Import the established runner rather than introduce another process/port
    # owner. Its process stays in the spark-job group and cleanup is waited.
    sys.path.insert(0, str(args.repo / "tools"))
    from speed_gate import Service
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"runtime": str(args.runtime.resolve()),
               "runtime_sha256": hashlib.sha256(args.runtime.read_bytes()).hexdigest(),
               "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
               "phase": args.phase}
    if supplement:
        receipt["helper_sha256"] = {
            name: fib.sha(Path(__file__).with_name(name))
            for name in ("native.py", "fibonacci512.py")}
        receipt["service_sha256"] = fib.SERVICE_SHA
        dependencies = subprocess.check_output(["ldd", str(args.runtime.resolve())],
                                               stderr=subprocess.STDOUT, timeout=10)
        (args.out / "dependencies.txt").write_bytes(dependencies)
        if b"not found" in dependencies:
            raise ValueError("frozen native runtime has missing dependencies")
        receipt["dependencies_sha256"] = hashlib.sha256(dependencies).hexdigest()
        receipt["input_sha256"] = fib.INPUT_SHA
        receipt["metadata_sha256"] = fib.source_metadata(
            Path(model["tokenizer"]).parent)
    (args.out / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    service = None
    try:
        service = Service(args.runtime.resolve(), args.config.resolve(), args.out, 18974, {})
        service.wait_ready(timeout=420)
        if supplement:
            fib.collect("http://127.0.0.1:18974", args.behavior_inputs,
                        args.out / "fibonacci512")
        elif args.phase == "behavior":
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
        try:
            if service is not None:
                retirement = service.stop(timeout=20)
                (args.out / "retirement.json").write_text(
                    json.dumps(retirement, sort_keys=True) + "\n")
                if supplement and retirement != {"signal": "TERM", "returncode": 0}:
                    raise RuntimeError("supplement runtime did not retire cleanly")
        finally:
            if supplement:
                receipt["runtime_sha256_after"] = fib.sha(args.runtime)
                receipt["config_sha256_after"] = fib.sha(args.config)
                (args.out / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
                if (receipt["runtime_sha256_after"] != fib.RUNTIME_SHA or
                        receipt["config_sha256_after"] != receipt["config_sha256"]):
                    raise RuntimeError("supplement runtime/config changed during collection")


if __name__ == "__main__":
    main()
