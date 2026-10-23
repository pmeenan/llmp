#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run exactly two registered current-candidate replies after external enablement."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time


def fail(message: str, code: int = 2) -> None:
    print(f"run-replies: {message}", file=sys.stderr)
    raise SystemExit(code)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(data, dict):
        fail(f"expected JSON object in {path}")
    return data


def check_file(path: Path, pin: dict, label: str) -> dict:
    if not path.is_absolute():
        fail(f"{label}: expected absolute path: {path}")
    try:
        info = path.lstat()
    except OSError as error:
        fail(f"{label}: cannot stat {path}: {error}")
    if not stat.S_ISREG(info.st_mode):
        fail(f"{label}: expected regular non-symlink file: {path}")
    mode = stat.S_IMODE(info.st_mode)
    expected_mode = int(pin.get("mode", "0444"), 8) if isinstance(pin.get("mode", "0444"), str) else pin["mode"]
    actual_hash = sha256(path)
    if actual_hash != pin["sha256"] or mode != expected_mode:
        fail(f"{label}: hash/mode mismatch for {path}: {actual_hash}/{mode:04o}")
    if "bytes" in pin and info.st_size != pin["bytes"]:
        fail(f"{label}: size mismatch for {path}")
    return {"path": str(path), "bytes": info.st_size, "mode": f"{mode:04o}", "sha256": actual_hash}


def ldd_resolution(runtime: Path, library_dir: Path, expected: dict) -> dict:
    environment = {"LD_LIBRARY_PATH": str(library_dir), "PATH": "/usr/bin:/bin", "LANG": "C"}
    result = subprocess.run(["/usr/bin/ldd", str(runtime)], env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        fail(f"ldd failed: {result.stderr.strip()}")
    found = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(\S+)", line)
        if match:
            soname, resolved = match.groups()
            if resolved == "not":
                fail(f"unresolved dependency: {line.strip()}")
            if soname not in expected:
                fail(f"un-pinned dependency: {soname}")
            target = Path(resolved)
            pinned = library_dir / soname
            if target.resolve() != pinned.resolve():
                fail(f"{soname} resolves to {target}, expected {pinned}")
            found[soname] = {"path": str(target), "sha256": sha256(target)}
        else:
            direct = re.match(r"\s*(/\S+)\s+\(0x[0-9a-fA-F]+\)", line)
            if direct:
                target = Path(direct.group(1))
                soname = target.name
                if soname in expected:
                    digest = sha256(target)
                    if digest != expected[soname]["sha256"]:
                        fail(f"system loader differs from pinned bytes: {target}")
                    found[soname] = {"path": str(target), "sha256": digest,
                                     "same_bytes_as_pinned_copy": True}
    missing = sorted(set(expected) - set(found))
    if missing:
        fail(f"ldd did not resolve pinned dependencies: {missing}")
    return {"returncode": result.returncode, "resolved": found,
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest()}


def load_helpers(repo: Path, pins: dict):
    sys.dont_write_bytecode = True
    helpers = pins["runtime_helpers"]
    paths = {
        "speed_gate.py": repo / helpers["speed_gate.py"]["path"],
        "greedy_capture.py": repo / helpers["greedy_capture.py"]["path"],
        "capture.py": repo / helpers["capture.py"]["path"],
    }
    for name, path in paths.items():
        if sha256(path) != helpers[name]["sha256"]:
            fail(f"pinned helper changed: {path}")
    capture_directory = str(paths["greedy_capture.py"].parent)
    sys.path.insert(0, capture_directory)
    sg_spec = importlib.util.spec_from_file_location("u4_reply_speed_gate", paths["speed_gate.py"])
    if not sg_spec or not sg_spec.loader:
        fail("cannot load pinned speed_gate helper")
    sg = importlib.util.module_from_spec(sg_spec)
    sys.modules[sg_spec.name] = sg
    sg_spec.loader.exec_module(sg)
    gc_spec = importlib.util.spec_from_file_location("u4_reply_greedy_capture", paths["greedy_capture.py"])
    if not gc_spec or not gc_spec.loader:
        fail("cannot load pinned greedy capture helper")
    gc = importlib.util.module_from_spec(gc_spec)
    sys.modules[gc_spec.name] = gc
    gc_spec.loader.exec_module(gc)
    for name, path in paths.items():
        if sha256(path) != helpers[name]["sha256"]:
            fail(f"pinned helper changed while importing: {path}")
    return sg, gc


def artifact_checks(pins: dict, family: dict) -> list[dict]:
    store = Path(family["store"])
    if not store.is_dir():
        fail(f"artifact store missing: {store}")
    receipts = []
    for key, artifact in family["artifacts"].items():
        aid = artifact["id"]
        root = store / aid
        checked = {}
        for name, field in (("manifest.json", "manifest_sha256"),
                            ("index.json", "index_sha256"), ("spec.json", "spec_sha256")):
            path = root / name
            if not path.is_file() or path.is_symlink():
                fail(f"artifact metadata absent or unsafe: {path}")
            digest = sha256(path)
            if digest != artifact[field]:
                fail(f"artifact metadata mismatch: {path}")
            checked[name] = digest
        receipts.append({"key": key, "id": aid, "metadata": checked})
    for source in family["tokenizer_template"]["pins"].values():
        if isinstance(source, dict) and "path" in source and "sha256" in source:
            path = Path(source["path"])
            if sha256(path) != source["sha256"]:
                fail(f"tokenizer/template source hash mismatch: {path}")
    return receipts


def validate_request_recipe(family: dict) -> None:
    limits = family.get("collection_limits", {})
    expected_limits = {"sequential_requests": 2, "retry": False, "score_passage": False,
                       "timing_sample": False, "debug_plans": False, "temperature": 0,
                       "max_tokens": 128, "stream": True, "include_usage": True}
    if any(limits.get(key) != value for key, value in expected_limits.items()):
        fail("family collection limits differ from the fixed two-reply recipe")
    requests = family.get("requests")
    if not isinstance(requests, list) or [item.get("index") for item in requests] != [0, 1]:
        fail("family must contain exactly requests 0 and 1 in order")
    for entry in requests:
        body = entry.get("request")
        if not isinstance(body, dict):
            fail("registered request body is missing")
        if (body.get("model") != family.get("model") or body.get("temperature") != 0 or
                body.get("max_tokens") != 128 or body.get("stream") is not True or
                body.get("stream_options") != {"include_usage": True}):
            fail(f"request {entry['index']} differs from the registered greedy bounds")
        messages = body.get("messages")
        if (not isinstance(messages, list) or len(messages) != 1 or
                not isinstance(messages[0], dict) or
                messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str)):
            fail(f"request {entry['index']} must be the single registered user prompt")
        reference = entry.get("reference", {})
        required = entry.get("reference_match_is_required", {})
        usage = reference.get("usage", {})
        if (required.get("finish") != "length" or required.get("completion_tokens") != 128 or
                type(required.get("prompt_tokens")) is not int or
                any(usage.get(key) != value for key, value in required.items() if key != "finish") or
                reference.get("finish") != "length" or
                not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("request_raw_sha256", "")))):
            fail(f"request {entry['index']} lacks its authenticated reference bounds")


def validate(pins_path: Path, repo: Path, family_name: str, *, require_fresh_output: bool = True):
    pins = read_json(pins_path)
    if pins.get("format") != "llmp-u4-current-replies-prospective/1" or pins.get("enabled") is not False:
        fail("pins must remain the expected statically disabled version")
    family = pins.get("families", {}).get(family_name)
    if not isinstance(family, dict):
        fail(f"unknown family {family_name}")
    validate_request_recipe(family)
    actual_host = socket.gethostname().split(".", 1)[0]
    if actual_host != family["host"]:
        fail(f"wrong host: expected {family['host']}, got {actual_host}")
    runtime_helpers = pins["runtime_helpers"]
    runtime = Path(family["current_runtime"]["path"])
    runtime_receipt = check_file(runtime, {"sha256": family["current_runtime"]["sha256"], "mode": family["current_runtime"]["mode"]}, "runtime")
    closure = runtime_helpers["library_closure"]
    receipt_path = Path(closure["receipt_path"])
    if sha256(receipt_path) != closure["receipt_sha256"]:
        fail("flat library closure receipt hash mismatch")
    library_dir = Path(closure["directory"])
    library_receipts = {}
    for soname, record in sorted(closure["unique_sonames"].items()):
        path = library_dir / soname
        library_receipts[soname] = check_file(path, {**record, "mode": "0444"}, f"library {soname}")
    artifacts = artifact_checks(pins, family)
    template = family["config"]["current_equivalent_template"]
    config_hash = hashlib.sha256(template.encode()).hexdigest()
    if config_hash != family["config"]["current_equivalent_template_sha256"]:
        fail("current equivalent config template hash mismatch")
    out = Path(family["output_parent"])
    if not out.is_absolute():
        fail("output namespace must be absolute")
    if require_fresh_output and os.path.lexists(out):
        fail(f"output namespace already exists: {out}")
    helper_receipts = {name: {"path": str(repo / item["path"]), "sha256": item["sha256"]}
                       for name, item in runtime_helpers.items() if name.endswith(".py")}
    for name, item in runtime_helpers.items():
        if name.endswith(".py"):
            path = repo / item["path"]
            if sha256(path) != item["sha256"]:
                fail(f"pinned helper changed: {path}")
    return {"pins": pins, "family": family, "runtime": runtime, "runtime_receipt": runtime_receipt,
            "library_dir": library_dir, "library_receipts": library_receipts,
            "artifacts": artifacts, "config_template_sha256": config_hash,
            "helpers": helper_receipts, "output_parent": out,
            "ldd": ldd_resolution(runtime, library_dir, closure["unique_sonames"])}


def validate_registration(path: Path, pins_path: Path, executor_path: Path, pins: dict) -> dict:
    receipt = read_json(path)
    if receipt.get("enabled") is not True:
        fail("external receipt does not enable collection")
    if not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("preregistration_commit", ""))):
        fail("external receipt lacks full 40-character reviewed commit")
    if receipt.get("pins_sha256") != sha256(pins_path):
        fail("external receipt pins hash mismatch")
    if receipt.get("executor_sha256") != sha256(executor_path):
        fail("external receipt executor hash mismatch")
    dependencies = {
        name: pins["runtime_helpers"][name]["sha256"]
        for name in ("speed_gate.py", "greedy_capture.py", "capture.py")
    }
    dependencies["library_closure_receipt"] = pins["runtime_helpers"]["library_closure"]["receipt_sha256"]
    if receipt.get("dependency_hashes") != dependencies:
        fail("external receipt dependency hashes do not match the pins")
    return receipt


def write_private(path: Path, value: dict, *, exclusive: bool = False) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def config_text(template: str, share: Path, data: Path, port: int) -> str:
    return (template.replace("{share}", str(share)).replace("{data}", str(data))
            .replace("{port}", str(port)))


def collect_requests(gc, port: int, requests: list[dict], output: Path) -> list[dict]:
    """The sole model-facing loop: the two registered serial chat requests."""
    replies = []
    for entry in requests:
        index = entry["index"]
        body = entry["request"]
        request_path = output / f"request-{index}.json"
        with request_path.open("x", encoding="utf-8") as stream:
            json.dump(body, stream, sort_keys=True, indent=2)
            stream.write("\n")
        reply = gc.collect_reply(port, body, output / f"reply-{index}.sse")
        reference = entry["reference"]
        required = entry["reference_match_is_required"]
        usage_matches = all(reply["usage"].get(key) == value for key, value in required.items()
                            if key != "finish")
        finish_matches = reply["finish"] == required["finish"] == "length"
        token_budget_matches = reply["usage"].get("completion_tokens") == 128
        prompt_matches = reply["usage"].get("prompt_tokens") == reference["usage"].get("prompt_tokens")
        replies.append({"index": index, "request": body, "request_sha256": sha256(request_path),
                        "reply": reply, "comparison_to_retained_A": {
                            "reply_sha256_matches": reply["reply_sha256"] == reference["reply_sha256"],
                            "usage_matches": usage_matches, "finish_matches": finish_matches,
                            "completion_budget_matches": token_budget_matches,
                            "prompt_tokens_match": prompt_matches}})
        if not (usage_matches and finish_matches and token_budget_matches and prompt_matches):
            raise RuntimeError(f"reply {index} differs from registered usage/finish bounds")
    return replies


def run(args) -> int:
    pins_path = args.pins.resolve(strict=True)
    repo = args.repo.resolve(strict=True)
    if args.run:
        if not args.registration:
            fail("--run requires an external enablement receipt")
        registration = validate_registration(args.registration.resolve(strict=True), pins_path,
                                             Path(__file__).resolve(), read_json(pins_path))
    else:
        registration = None
    state = validate(pins_path, repo, args.family)
    family = state["family"]
    service_pins = state["pins"]["service"]
    if args.check_only:
        print(json.dumps({"family": args.family, "host": family["host"],
                          "runtime": state["runtime_receipt"], "libraries": state["library_receipts"],
                          "artifacts": state["artifacts"], "config_template_sha256": state["config_template_sha256"],
                          "helpers": state["helpers"], "ldd": state["ldd"],
                          "output_parent_absent": not os.path.lexists(state["output_parent"]),
                          "collection": "not executed"}, sort_keys=True, indent=2))
        return 0

    os.umask(0o077)
    # Import only after the external enablement receipt and all file pins pass,
    # but before reserving the irreversible fresh output namespace.
    sg, gc = load_helpers(repo, state["pins"])
    if service_pins["bind"] != "127.0.0.1":
        fail("only the pinned loopback bind is supported")
    if sg.port_open(service_pins["port"]):
        fail(f"registered local HTTP port {service_pins['port']} is already occupied")

    out = state["output_parent"]
    base = out.parent
    if not base.is_dir() or base.stat().st_uid != os.geteuid() or (base.stat().st_mode & 0o022):
        fail("output parent must already exist, be user-owned, and not group/world writable")
    old_umask = os.umask(0o077)
    try:
        os.mkdir(out, 0o700)
    finally:
        os.umask(old_umask)
    if stat.S_IMODE(out.stat().st_mode) != 0o700:
        fail("fresh output namespace is not private mode 0700")
    work = out / "work"
    work.mkdir(mode=0o700)
    data = work / "data"
    data.mkdir(mode=0o700)
    template = family["config"]["current_equivalent_template"]
    config = work / "node.toml"
    port = service_pins["port"]
    config.write_text(config_text(template, Path(family["share_root"]), data, port), encoding="utf-8")
    os.chmod(config, 0o600)
    run_receipt = {"state": "running", "family": args.family, "host": family["host"],
                   "registration_commit": registration["preregistration_commit"],
                   "registration_sha256": sha256(args.registration.resolve(strict=True)),
                   "pins_sha256": sha256(pins_path), "executor_sha256": sha256(Path(__file__).resolve()),
                   "runtime": state["runtime_receipt"], "libraries": state["library_receipts"],
                   "artifacts": state["artifacts"], "ldd": state["ldd"],
                   "config_path": str(config), "config_sha256": sha256(config),
                   "config_template_sha256": state["config_template_sha256"],
                   "bind": service_pins["bind"], "port": port, "debug_plans": False, "timing_sample": False,
                   "runtime_environment": {"LD_LIBRARY_PATH": str(state["library_dir"]),
                                           "CUDA_DISABLE_PTX_JIT": "1",
                                           "filtered_parent_prefixes": ["LLMP_", "LD_", "CUDA_", "NVIDIA_"]},
                   "output_parent": str(out), "started_unix_ns": time.time_ns()}
    run_path = out / "run.json"
    write_private(run_path, run_receipt, exclusive=True)
    service = None
    error = None
    replies = []
    stop_receipt = None
    started = time.monotonic()
    try:
        parent_env = os.environ.copy()
        clean_env = {key: value for key, value in parent_env.items()
                     if not key.startswith(("LLMP_", "LD_", "CUDA_", "NVIDIA_"))}
        os.environ.clear()
        os.environ.update(clean_env)
        try:
            service = sg.Service(state["runtime"], config, work, port,
                                 {"LD_LIBRARY_PATH": str(state["library_dir"])})
        finally:
            os.environ.clear()
            os.environ.update(parent_env)
        ready_seconds = service.wait_ready(420)
        replies = collect_requests(gc, port, family["requests"], out)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        if service is not None:
            try:
                stop_receipt = service.stop(20)
            except Exception as exception:
                error = error or f"stop {type(exception).__name__}: {exception}"
        elapsed = time.monotonic() - started

    post_runtime_hash = sha256(state["runtime"])
    try:
        post_state = validate(pins_path, repo, args.family, require_fresh_output=False)
        immutable_match = (post_state["runtime_receipt"] == state["runtime_receipt"] and
                           post_state["library_receipts"] == state["library_receipts"] and
                           post_state["artifacts"] == state["artifacts"] and
                           post_state["helpers"] == state["helpers"])
    except SystemExit as exception:
        immutable_match = False
        error = error or f"post-run immutable input check failed: {exception.code}"
    if post_runtime_hash != family["current_runtime"]["sha256"]:
        immutable_match = False
    passed = (error is None and len(replies) == 2 and stop_receipt is not None and
              stop_receipt.get("returncode") == 0 and immutable_match)
    final = {**run_receipt, "state": "passed" if passed else "failed",
             "returncode": stop_receipt.get("returncode") if stop_receipt else None,
             "stop": stop_receipt, "elapsed_seconds": elapsed, "ready_seconds": locals().get("ready_seconds"),
             "error": error, "candidate_runtime_sha256_after": post_runtime_hash,
             "runtime_log_path": str(work / "runtime.log"),
             "runtime_log_sha256": sha256(work / "runtime.log") if (work / "runtime.log").is_file() else None,
             "immutable_inputs_unchanged": immutable_match, "replies": replies,
             "quality_verdict": "requires independent semantic review; no answer-completion claim"}
    tmp = out / "run.json.tmp"
    write_private(tmp, final, exclusive=True)
    os.replace(tmp, run_path)
    return 0 if passed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--family", required=True, choices=("qwen38-gguf", "deepseek"))
    parser.add_argument("--repo", type=Path, default=Path("/home/pmeenan/src/llmp"))
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args(argv)
    if args.check_only and args.registration:
        fail("--registration is invalid with --check-only")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
