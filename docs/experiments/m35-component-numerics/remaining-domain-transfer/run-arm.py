#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""One registered U4 arm. Collection requires a separate reviewed receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def fail(message: str, code: int = 2) -> None:
    print(f"run-arm: {message}", file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipt(path: Path, expected: dict, label: str) -> dict:
    if not path.is_absolute():
        fail(f"{label}: pinned path is not absolute: {path}")
    try:
        info = path.lstat()
    except OSError as error:
        fail(f"{label}: cannot stat {path}: {error}")
    if not stat.S_ISREG(info.st_mode):
        fail(f"{label}: expected regular non-symlink file: {path}")
    mode = stat.S_IMODE(info.st_mode)
    expected_mode = int(expected["mode"], 8) if isinstance(expected["mode"], str) else expected["mode"]
    if mode != expected_mode:
        fail(f"{label}: mode mismatch for {path}: got {mode:04o}, expected {expected_mode:04o}")
    actual_sha = sha256_file(path)
    if actual_sha != expected["sha256"]:
        fail(f"{label}: SHA-256 mismatch for {path}: {actual_sha}")
    if "bytes" in expected and info.st_size != expected["bytes"]:
        fail(f"{label}: size mismatch for {path}: {info.st_size}")
    return {"path": str(path), "bytes": info.st_size, "mode": f"{mode:04o}", "sha256": actual_sha}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"expected a JSON object in {path}")
    return value


def write_exclusive(path: Path, data: bytes, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def validate(pins_path: Path, arm_name: str, *, require_fresh_output: bool = True) -> tuple[dict, dict, dict, list[str], dict, dict]:
    pins = read_json(pins_path)
    if pins.get("schema_version") != 1 or pins.get("enabled") is not False:
        fail("pins must use schema_version 1 and remain statically disabled")
    arm = pins.get("arms", {}).get(arm_name)
    if not isinstance(arm, dict):
        fail(f"unknown registered arm {arm_name!r}")
    host = pins.get("hosts", {}).get(arm.get("host"))
    if not isinstance(host, dict):
        fail("arm names an unknown actual host")
    actual_host = socket.gethostname().split(".", 1)[0]
    if actual_host != host.get("actual_hostname"):
        fail(f"wrong host: expected {host.get('actual_hostname')}, got {actual_host}")

    build, binary_name = arm.get("build"), arm.get("binary")
    binary = host.get("binaries", {}).get(build, {}).get(binary_name)
    if not isinstance(binary, dict):
        fail("arm does not resolve to a host-local pinned binary")
    binary_receipt = file_receipt(Path(binary["path"]), binary, "executable")

    libraries = host.get("libraries")
    if not isinstance(libraries, dict) or not libraries:
        fail("host has no pinned dynamic dependency closure")
    library_receipts = [file_receipt(Path(item["path"]), item, f"library {name}")
                        for name, item in sorted(libraries.items())]

    store = Path(host["store_path"])
    if not store.is_absolute() or not store.is_dir():
        fail(f"pinned artifact store is absent: {store}")
    artifact_keys = [(arm["target_artifact"], arm["target_id"])]
    if arm.get("drafter_artifact"):
        artifact_keys.append((arm["drafter_artifact"], arm["drafter_id"]))
    artifact_receipts = []
    artifacts = pins.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("pins do not inline the frozen artifact metadata")
    for key, artifact_id in artifact_keys:
        artifact = artifacts.get(key)
        if not isinstance(artifact, dict) or artifact.get("id") != artifact_id:
            fail(f"artifact identity is not bound by pins: {artifact_id}")
        root = store / artifact_id
        checked = {}
        for filename, field in (("manifest.json", "manifest_sha256"),
                                ("spec.json", "spec_sha256"), ("index.json", "index_sha256")):
            path = root / filename
            try:
                info = path.lstat()
            except OSError as error:
                fail(f"artifact metadata missing {path}: {error}")
            if not stat.S_ISREG(info.st_mode):
                fail(f"artifact metadata is not a regular file: {path}")
            actual = sha256_file(path)
            if actual != artifact[field]:
                fail(f"artifact metadata hash mismatch: {path}")
            checked[field] = actual
        artifact_receipts.append({"key": key, "id": artifact_id, "path": str(root), **checked})

    input_pins = pins.get("inputs")
    input_key = arm.get("input_key", arm.get("model"))
    if not isinstance(input_pins, dict) or input_key not in input_pins:
        fail("arm does not name an inline frozen input recipe")
    policy = arm.get("expected_target_policy")
    if not isinstance(policy, str) or not policy:
        fail("arm lacks the source-derived expected target policy")
    if arm.get("drafter_id") and not isinstance(arm.get("expected_drafter_policy"), str):
        fail("speculative arm lacks the expected drafter policy")
    if arm.get("transport") not in ("configured-default", "full-head"):
        fail("arm has an unsupported transport")

    parent = Path(arm["output_parent"])
    export = Path(arm["export_dir"])
    if not parent.is_absolute() or export != parent / "export":
        fail("output_parent and export_dir must be absolute and export_dir must be output_parent/export")
    if require_fresh_output and os.path.lexists(parent):
        fail(f"fresh output namespace already exists: {parent}")
    argv = [str(Path(binary["path"])), "--gtest_also_run_disabled_tests",
            f"--gtest_filter={arm['filter']}", f"--gtest_output=xml:{parent / 'gtest.xml'}"]

    child_env = {k: v for k, v in os.environ.items()
                 if not k.startswith("LLMP_U4_") and k not in {
                    "LD_PRELOAD", "LD_LIBRARY_PATH", "LLMP_TEST_MODELS", "LLMP_TEST_V1_STORE",
                    "LLMP_TEST_DATA", "LLMP_TEST_SCRATCH", "LLMP_DSV4_TARGET_STORE"}}
    child_env.update({
        "LD_LIBRARY_PATH": host["library_dir"],
        "LLMP_TEST_MODELS": host["models_root"],
        "LLMP_TEST_V1_STORE": host["store_path"],
        "LLMP_DSV4_TARGET_STORE": host["store_path"],
        "LLMP_TEST_DATA": host["test_data_directory"],
        "LLMP_TEST_SCRATCH": str(parent / "scratch"),
        "LLMP_U4_MODEL": arm["model"],
        "LLMP_U4_STORE": host["store_path"],
        "LLMP_U4_TARGET_ID": arm["target_id"],
        "LLMP_U4_TRANSPORT": arm["transport"],
        "LLMP_U4_EXPORT_DIR": str(export),
        "LLMP_U4_EXPECTED_TARGET_POLICY": arm["expected_target_policy"],
    })
    if arm.get("drafter_id"):
        child_env["LLMP_U4_DRAFTER_ID"] = arm["drafter_id"]
        child_env["LLMP_U4_EXPECTED_DRAFTER_POLICY"] = arm["expected_drafter_policy"]
    safe_env = {k: child_env[k] for k in sorted(child_env)
                if k.startswith(("LLMP_U4_", "LLMP_TEST_", "LLMP_DSV4_", "LD_LIBRARY_PATH"))}
    return pins, arm, host, argv, {"binary": binary_receipt, "libraries": library_receipts,
                                  "artifacts": artifact_receipts}, {"env": child_env, "safe_env": safe_env}


def ldd_resolution(binary: Path, host: dict) -> dict:
    env = {"LD_LIBRARY_PATH": host["library_dir"], "PATH": "/usr/bin:/bin", "LANG": "C"}
    result = subprocess.run(["/usr/bin/ldd", str(binary)], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        fail(f"ldd failed for {binary}: {result.stderr.strip()}")
    libraries = host["libraries"]
    resolved = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(\S+)", line)
        if match:
            soname, path_text = match.groups()
            if path_text == "not":
                fail(f"ldd reports unresolved dependency for {binary}: {line.strip()}")
            if soname not in libraries:
                fail(f"ldd found unpinned dependency for {binary}: {soname}")
            resolved_path = Path(path_text)
            pinned_path = Path(libraries[soname]["path"])
            if resolved_path.resolve() != pinned_path.resolve():
                fail(f"LD_LIBRARY_PATH resolved {soname} to {resolved_path}, expected {pinned_path}")
            resolved[soname] = {"path": str(resolved_path), "pinned_path": str(pinned_path),
                                "sha256": libraries[soname]["sha256"]}
            continue
        # The ELF interpreter is reported without `=>` and is loaded before
        # LD_LIBRARY_PATH is consulted. Authenticate its bytes against the
        # same pinned closure member while recording the actual system path.
        direct = re.match(r"\s*(/\S+)\s+\(0x[0-9a-fA-F]+\)", line)
        if direct and Path(direct.group(1)).name in libraries:
            soname = Path(direct.group(1)).name
            actual_path = Path(direct.group(1))
            actual_sha = sha256_file(actual_path)
            if actual_sha != libraries[soname]["sha256"]:
                fail(f"system ELF interpreter hash mismatch for {actual_path}")
            resolved[soname] = {"path": str(actual_path),
                                "pinned_copy": libraries[soname]["path"],
                                "sha256": actual_sha, "same_bytes_as_pinned_copy": True}
    missing = sorted(set(libraries) - set(resolved))
    if missing:
        fail(f"ldd did not account for pinned dependency entries for {binary}: {missing}")
    return {"binary": str(binary), "returncode": result.returncode, "resolved": resolved,
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest()}


def validate_registration(path: Path, pins_path: Path, executor_path: Path) -> dict:
    receipt = read_json(path)
    if receipt.get("enabled") is not True:
        fail("external registration is not enabled")
    if not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("preregistration_commit", ""))):
        fail("registration must bind the full preregistration commit hash")
    if receipt.get("pins_sha256") != sha256_file(pins_path):
        fail("registration pins hash does not match this file")
    if receipt.get("executor_sha256") != sha256_file(executor_path):
        fail("registration executor hash does not match this helper")
    return receipt


def export_manifest(export_dir: Path) -> tuple[list[dict], bool]:
    entries = []
    for path in sorted(export_dir.rglob("*")):
        if path.is_symlink():
            fail(f"unexpected symlink in diagnostic export: {path}")
        if path.is_file():
            info = path.stat()
            entries.append({"path": str(path.relative_to(export_dir)), "bytes": info.st_size,
                            "mode": f"{stat.S_IMODE(info.st_mode):04o}", "sha256": sha256_file(path)})
    status = export_dir / "status.jsonl"
    complete = False
    if status.is_file():
        lines = [line for line in status.read_text(encoding="utf-8").splitlines() if line]
        complete = len(lines) >= 2 and json.loads(lines[0]).get("status") == "incomplete" \
            and json.loads(lines[-1]).get("status") == "complete"
    return entries, complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true", help="validate pins without creating output or launching")
    modes.add_argument("--run", action="store_true", help="run exactly one arm after external registration enables it")
    parser.add_argument("--registration", type=Path,
                        help="external enablement receipt; mandatory with --run")
    args = parser.parse_args()
    pins_path = args.pins.resolve()
    pins, arm, host, argv, checks, env_data = validate(pins_path, args.arm)
    if args.check_only:
        if args.registration:
            fail("--registration is not used in check-only mode")
        resolved = ldd_resolution(Path(checks["binary"]["path"]), host)
        print(json.dumps({"arm": args.arm, "host": host["actual_hostname"],
                          "binary": checks["binary"], "libraries": checks["libraries"],
                          "artifacts": checks["artifacts"], "argv_preview": argv,
                          "ldd_resolution": resolved,
                          "output_parent": arm["output_parent"], "export_dir": arm["export_dir"],
                          "collection": "not executed; metadata check only"}, sort_keys=True, indent=2))
        return 0

    if not args.registration:
        fail("--run requires an external reviewed registration receipt")
    registration_path = args.registration.resolve()
    registration = validate_registration(registration_path, pins_path, Path(__file__).resolve())
    parent = Path(arm["output_parent"])
    parent_parent = parent.parent
    if not parent_parent.is_dir():
        fail(f"registered output parent must already exist: {parent_parent}")
    parent_parent_info = parent_parent.stat()
    if parent_parent_info.st_uid != os.geteuid() or (parent_parent_info.st_mode & 0o022):
        fail("registered output parent must be owned by this user and not group/world writable")
    old_umask = os.umask(0o077)
    try:
        os.mkdir(parent, 0o700)
    finally:
        os.umask(old_umask)
    if stat.S_IMODE(parent.stat().st_mode) != 0o700:
        fail("new output namespace is not mode 0700")

    status_path = parent / "run.json"
    started = {"state": "running", "arm": args.arm, "host": host["actual_hostname"],
               "registration_commit": registration["preregistration_commit"],
               "argv": argv, "binary": checks["binary"], "libraries": checks["libraries"],
               "artifacts": checks["artifacts"], "environment": env_data["safe_env"],
               "output_parent": str(parent), "export_dir": arm["export_dir"]}
    write_exclusive(status_path, (json.dumps(started, sort_keys=True, indent=2) + "\n").encode())
    stdout_path, stderr_path = parent / "stdout.log", parent / "stderr.log"
    start = time.monotonic()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        os.chmod(stdout_path, 0o600)
        os.chmod(stderr_path, 0o600)
        # Recheck the exact launch closure immediately before starting the test.
        launch_ldd = ldd_resolution(Path(checks["binary"]["path"]), host)
        result = subprocess.run(argv, cwd=host.get("working_directory", "/home/pmeenan/src/llmp"), env=env_data["env"],
                                stdout=stdout, stderr=stderr, check=False)
        stdout.flush(); os.fsync(stdout.fileno())
        stderr.flush(); os.fsync(stderr.fileno())
    elapsed = time.monotonic() - start
    # Collection may not replace its own executable, libraries, or frozen
    # artifact metadata. Revalidate those immutable inputs after the child exits.
    _, _, _, _, post_checks, _ = validate(pins_path, args.arm, require_fresh_output=False)
    immutable_inputs_unchanged = post_checks == checks
    xml_path = parent / "gtest.xml"
    xml_summary = None
    xml_ok = False
    try:
        root = ET.parse(xml_path).getroot()
        cases = list(root.iter("testcase"))
        failures = list(root.iter("failure")) + list(root.iter("error"))
        skipped = [case for case in cases if case.find("skipped") is not None]
        selected_class, selected_name = arm["filter"].split(".", 1)
        case = cases[0] if len(cases) == 1 else None
        root_failures = int(root.attrib.get("failures", "0")) + int(root.attrib.get("errors", "0"))
        xml_summary = {"testcases": len(cases), "failures_or_errors": len(failures),
                       "root_failures_or_errors": root_failures, "skipped": len(skipped),
                       "testsuites_tests": int(root.attrib.get("tests", len(cases))),
                       "selected_class": case.attrib.get("classname") if case is not None else None,
                       "selected_name": case.attrib.get("name") if case is not None else None,
                       "selected_status": case.attrib.get("status") if case is not None else None,
                       "selected_result": case.attrib.get("result") if case is not None else None}
        xml_ok = (len(cases) == 1 and not failures and not skipped and root_failures == 0
                  and xml_summary["testsuites_tests"] == 1
                  and case.attrib.get("classname") == selected_class
                  and case.attrib.get("name") == selected_name
                  and case.attrib.get("status") == "run"
                  and case.attrib.get("result") == "completed")
    except (OSError, ET.ParseError, ValueError):
        pass
    export_dir = Path(arm["export_dir"])
    files, export_complete = export_manifest(export_dir) if export_dir.is_dir() else ([], False)
    passed = result.returncode == 0 and xml_ok and export_complete and immutable_inputs_unchanged
    finished = {**started, "state": "passed" if passed else "failed", "returncode": result.returncode,
                "elapsed_seconds": elapsed, "gtest_xml": str(xml_path), "gtest_xml_summary": xml_summary,
                "launch_ldd_resolution": launch_ldd,
                "immutable_inputs_unchanged": immutable_inputs_unchanged,
                "post_run_immutable_inputs": post_checks,
                "export_complete": export_complete, "export_files": files,
                "stdout_sha256": sha256_file(stdout_path), "stderr_sha256": sha256_file(stderr_path)}
    tmp = parent / "run.json.tmp"
    write_exclusive(tmp, (json.dumps(finished, sort_keys=True, indent=2) + "\n").encode())
    os.replace(tmp, status_path)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
