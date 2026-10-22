#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Capture paid production LLM plan reports on the frozen current v1 artifacts.

Run one arm per supervised exclusive Spark job, after U4 checks/timing. This
reuses speed_gate.py's exact quality prompts and score passage; it only adds
--debug-plans to its service process and normalizes runtime.log plan JSON.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

EXPECTED = {
    "A": "405c5930accdc009ad8e824d2fcdeff00f063877f0d936c2b1a5dda6b28e519e",
    "B": "e47f5b30150e9e8a357fa204c423de10ab706c8c8a7f02f0bc48887b5b2a04af",
    "C": "6e7bc211551d17b7ba752d319b91d0937c29ab471853b7382da1fbb22093d68f",
}
HERE = pathlib.Path(__file__).resolve().parent


def load_speed_gate(repo, *, debug_plans=True):
    spec = importlib.util.spec_from_file_location("llmp_speed_gate", repo / "tools/speed_gate.py")
    assert spec and spec.loader
    sg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sg
    spec.loader.exec_module(sg)

    class DebugService(sg.Service):
        """Same service launch, adding only opt-in production plan reporting."""
        def __init__(self, runtime, config, run_dir, port, runtime_env):
            self.port = port
            if sg.port_open(port):
                raise RuntimeError(f"something already listens on port {port}")
            self.log = open(run_dir / "runtime.log", "wb")
            self.started = time.monotonic()
            self.proc = subprocess.Popen(
                [str(runtime), "--config", str(config), "--anchor", str(run_dir / "enrollment")]
                + (["--debug-plans"] if debug_plans else []),
                stdout=self.log, stderr=subprocess.STDOUT,
                env=sg.arm_env(runtime_env))

        def wait_ready(self, timeout=420):
            return super().wait_ready(timeout)

        def stop(self, timeout=20):
            return super().stop(timeout)

    sg.Service = DebugService
    return sg


def sha256_file(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_quality_pair(repo, expected_sha256):
    path = repo / "tools/quality_pair.py"
    if sha256_file(path) != expected_sha256:
        raise SystemExit("quality_pair.py differs from the registered helper")
    spec = importlib.util.spec_from_file_location("llmp_quality_pair", path)
    assert spec and spec.loader
    qp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qp)
    return qp


def verify_score(record_path, receipt, expected_prompt, family, qp):
    """Require every target from the frozen passage before accepting an inventory."""
    expected_targets = {"qwen38": 996, "qwen38-gguf": 996, "deepseek": 966}[family]
    prompt_name = record_path.stem + "-score.txt"
    response_name = record_path.stem + "-score.json"
    prompt = receipt.get("score_prompt", {})
    response = receipt.get("score_response", {})
    if prompt.get("file") != prompt_name or response.get("file") != response_name:
        raise ValueError("score files do not have this exact quality run's names")
    actual_prompt = (record_path.parent / prompt_name).read_bytes()
    prompt_digest = hashlib.sha256(actual_prompt).hexdigest()
    if actual_prompt != expected_prompt or prompt.get("sha256") != prompt_digest:
        raise ValueError("score prompt bytes/hash differ from the frozen passage")
    # Reuse the pinned pairing parser for response hash, canonical actual target IDs,
    # echo agreement and finite nonpositive per-target log-probabilities.
    losses = qp.from_speed_gate(record_path, receipt)
    payload = json.loads((record_path.parent / response_name).read_bytes())
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("scoring must return exactly one choice")
    rows = choices[0].get("prompt_logprobs")
    echo = (choices[0].get("logprobs") or {}).get("tokens")
    if (len(rows) != expected_targets + 1 or len(losses.rows) != expected_targets or
            not isinstance(echo, list) or len(echo) != len(rows)):
        raise ValueError("scoring lacks the exact registered target/echo count")
    # The first echo is unscored but must also have a canonical token identity.
    qp.token_id(echo[0], "unscored first echo")
    usage = payload.get("usage") or {}
    if (type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] != len(rows) or
            type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] != 0 or
            receipt.get("score", {}).get("tokens") != expected_targets):
        raise ValueError("scoring usage/receipt differs from the complete prompt")
    targets = [losses.rows[(0, position)][0] for position in range(1, len(rows))]
    return {"targets": expected_targets, "prompt_tokens": len(rows),
            "ordered_target_ids_sha256": qp.ids_digest(targets),
            "ordered_target_ids_digest_format": "SHA-256 of json.dumps(ordered target ID list)",
            "scope": "single-arm target inventory for later A/B alignment; not tokenizer identity"}


def verify_inputs(sg, args, pins):
    share = args.share.expanduser().resolve(strict=True)
    store = share / sg.resolve_store(sg.V1_STORE, share)
    template = pins["configs"][args.family]
    if template != sg.config_template(sg.FAMILIES[args.family]):
        raise SystemExit("generated current configuration differs from registration")
    # Pin the actual configured target/drafter closure, including absent optional specs.
    import tomllib
    models = tomllib.loads(template.replace("{port}", "18114"))["models"]
    observed = {}
    for model in models.values():
        for key in ("artifact", "drafter"):
            if key not in model:
                continue
            identity = model[key]
            observed[identity] = {}
            for name, expected in pins["artifacts"][identity].items():
                path = store / identity / name
                actual = sha256_file(path) if path.exists() else None
                if actual != expected:
                    raise SystemExit(f"artifact metadata differs: {path}")
                observed[identity][name] = actual
    sources = {}
    for name, expected in pins["source_files"].items():
        if name.startswith("models/") and args.family == "deepseek":
            continue
        path = args.corpus.expanduser() if name.startswith("corpus/") else share / name
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise SystemExit(f"source input differs: {path}")
        sources[name] = {"path": str(path.resolve()), "sha256": actual}
    return template, {"store": str(store.resolve()), "artifacts": observed, "source_files": sources}


from plan_reports import parse_log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=pathlib.Path, default=HERE.parents[2],
                   help="repository containing the pinned tools/speed_gate.py")
    p.add_argument("--family", required=True, choices=("qwen38", "qwen38-gguf", "deepseek"))
    p.add_argument("--arm", required=True, choices=("A", "B", "C"))
    p.add_argument("--runtime", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path,
                   help="private per-arm directory; use same parent for A and B")
    p.add_argument("--share", type=pathlib.Path, default=pathlib.Path("~/.local/share/llmp").expanduser())
    p.add_argument("--corpus", type=pathlib.Path,
                   default=pathlib.Path("~/.local/share/llmp/corpus/m3-reference-corpus.txt").expanduser())
    a = p.parse_args()
    os.umask(0o077)
    out = a.out.expanduser().absolute()
    # Reserve the exact namespace once, including failed attempts. Never erase a prior run.
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    pins = json.loads((HERE / "llm-capture-pins.json").read_text())
    repo = a.repo.expanduser().resolve(strict=True)
    speed_gate = repo / "tools/speed_gate.py"
    if sha256_file(speed_gate) != pins["speed_gate_sha256"]:
        raise SystemExit("speed_gate.py differs from the registered helper")
    runtime = a.runtime.expanduser().resolve(strict=True)
    actual = sha256_file(runtime)
    if actual != EXPECTED[a.arm] or actual != pins["arms"][a.arm]["sha256"]:
        raise SystemExit(f"runtime SHA-256 {actual} != frozen {a.arm} {EXPECTED[a.arm]}")
    sg = load_speed_gate(repo)
    qp = load_quality_pair(repo, pins["quality_pair_sha256"])
    family = sg.FAMILIES[a.family]
    if (list(sg.QUALITY_PROMPTS) != pins["quality"]["greedy_prompts"] or
            family.max_tokens != 128 or family.score_chars != 4000):
        raise SystemExit("quality inputs differ from registration")
    template, inputs = verify_inputs(sg, a, pins)
    config_template = out / "current-v1-node.toml.template"
    config_template.write_text(template)
    capture_metadata = {"family": a.family, "arm": a.arm, "runtime": str(runtime),
                        "runtime_sha256": actual, "repo": str(repo), "inputs": inputs,
                        "capture_sha256": sha256_file(pathlib.Path(__file__)),
                        "parser_sha256": sha256_file(HERE / "plan_reports.py"),
                        "config_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                        "pins_sha256": sha256_file(HERE / "llm-capture-pins.json"),
                        "speed_gate_sha256": sha256_file(speed_gate),
                        "quality_pair_sha256": pins["quality_pair_sha256"],
                        "ready_timeout_s": 420, "stop_timeout_s": 20,
                        "supervisor_timeout_s": 600, "debug_plans": True}
    (out / "capture-metadata.json").write_text(json.dumps(capture_metadata, indent=2) + "\n")
    # cmd_quality creates one fresh, private data dir and records exact template,
    # runtime hash, prompt/score hashes and target logprobs. Same host/port/config
    # semantics are held fixed across arms; only fresh data/anchor paths vary.
    args = argparse.Namespace(
        family=a.family, runtime=str(runtime), label={"A": "old", "B": "new", "C": "candidate-c"}[a.arm],
        out=str(out), share=str(a.share.expanduser().resolve()), corpus=str(a.corpus.expanduser().resolve()), port=18114,
        config_template=str(config_template), runtime_env=None)
    rc = sg.cmd_quality(args)
    after = sha256_file(runtime)
    capture_metadata["runtime_sha256_after"] = after
    (out / "capture-metadata.json").write_text(json.dumps(capture_metadata, indent=2) + "\n")
    if after != actual:
        raise SystemExit("runtime changed during the capture")
    quality_dir = out / "quality"
    name = f"{family.name}-quality-{args.label}"
    log = out / "work" / name / "runtime.log"
    record_path = quality_dir / f"{name}.json"
    if not log.is_file() or not record_path.is_file():
        raise SystemExit("quality run lacks its exact log/receipt")
    receipt = json.loads(record_path.read_text())
    if rc != 0 or receipt.get("error") or receipt.get("runtime_sha256") != actual:
        raise SystemExit("failed or mismatched quality run cannot qualify operand capture")
    if len(receipt.get("replies", [])) != len(sg.QUALITY_PROMPTS) or any(
            r.get("error") or not r.get("finish") for r in receipt["replies"]):
        raise SystemExit("greedy controls did not all complete successfully")
    expected_prompt = sg.passage(sg.load_corpus(a.corpus.expanduser()), 100003, 4000).encode()
    score_validation = verify_score(record_path, receipt, expected_prompt, a.family, qp)
    if receipt.get("stop", {}).get("returncode") != 0:
        raise SystemExit("runtime did not retire successfully")
    parsed = parse_log(log)
    parsed.update({"capture_metadata": capture_metadata, "family": a.family, "arm": a.arm, "runtime": str(runtime),
                   "runtime_sha256": actual, "quality_receipt": str(record_path),
                   "capture_parser_sha256": sg.sha256_file(pathlib.Path(__file__).with_name("plan_reports.py")),
                   "quality_receipt_sha256": sg.sha256_file(record_path),
                   "config_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                   "score_prompt_sha256": receipt.get("score_prompt", {}).get("sha256"),
                   "score_response_sha256": receipt.get("score_response", {}).get("sha256"),
                   "score": receipt.get("score"), "score_validation": score_validation,
                   "return_code": rc,
                   "debug_log": str(log), "debug_log_sha256": sg.sha256_file(log)})
    (quality_dir / f"{a.family}-quality-{args.label}-plans.json").write_text(
        json.dumps(parsed, indent=2, sort_keys=True) + "\n")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
