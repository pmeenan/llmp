#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect the retained four-document Mia target control on frozen U4 A or D.

A complete capture is not a quality pass. The unchanged strict Mia bounds are
reported separately; no drafter qualification or legacy exception is inferred.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
import pathlib

import capture
from plan_reports import parse_log

HERE = pathlib.Path(__file__).resolve().parent
EXPECTED = {arm: capture.EXPECTED[arm] for arm in ("A", "D")}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reference_inputs(root, pins, qp):
    """Authenticate documents, exact first/target IDs, and retained loss positions."""
    for name, expected in pins["reference_files"].items():
        raw = (root / name).read_bytes()
        if len(raw) != expected["bytes"] or digest(raw) != expected["sha256"]:
            raise ValueError(f"retained reference changed: {name}")
    inputs = json.loads((root / "inputs.json").read_bytes())
    docs = inputs["documents"]
    if [d["document"] for d in docs] != list(range(4)):
        raise ValueError("expected exactly four ordered reference documents")
    losses, _ = qp.load(str(root / "losses.jsonl"))
    wanted = {}
    for d in docs:
        number, ids = d["document"], d["tokens"]
        raw = (root / f"doc{number}.txt").read_bytes()
        if (len(ids) != 255 or any(type(i) is not int or i < 0 for i in ids) or
                digest(raw) != d["text_sha256"]):
            raise ValueError("reference text/token inventory differs")
        for position, target in enumerate(ids[1:], 1):
            wanted[number, position] = target
    if {k: v[0] for k, v in losses.rows.items()} != wanted:
        raise ValueError("reference target IDs/positions do not match all four inputs")
    return docs, losses


def validate_response(path, receipt, ids, qp):
    """Use actual scored targets, never a top-one substitute, including exact first ID."""
    losses = qp.from_speed_gate(path, receipt)
    payload = json.loads((path.parent / receipt["score_response"]["file"]).read_bytes())
    if payload.get("error") or len(payload.get("choices", [])) != 1:
        raise ValueError("expected one successful score choice")
    choice = payload["choices"][0]
    rows = choice["prompt_logprobs"]
    echo = (choice.get("logprobs") or {}).get("tokens")
    if (len(ids) != 255 or len(rows) != 255 or len(losses.rows) != 254 or
            not isinstance(echo, list) or len(echo) != 255 or
            [qp.token_id(x, "reference echo") for x in echo] != ids):
        raise ValueError("all 255 echoed reference IDs and 254 target rows are required")
    if {p: value[0] for (_, p), value in losses.rows.items()} != dict(enumerate(ids[1:], 1)):
        raise ValueError("scored keys differ from the exact reference targets")
    usage = payload.get("usage") or {}
    if (type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] != 255 or
            type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] != 0):
        raise ValueError("scoring usage differs from the complete reference document")
    return losses


def associate_documents(parsed, documents, model):
    """Associate all paid scoring rows, retaining the global installed registry.

    The isolated service receives one synchronous document request at a time:
    that request is the sole owner. Completion records report total input rows,
    not owner IDs or token positions. The exact echoed/target IDs are checked
    separately. Capacity may split its 255 input rows into several score steps.
    """
    result = []
    seen = set()
    previous_end = 0
    for d in documents:
        start, end = d["log_start_line"], d["log_end_line"]
        if (type(start) is not int or type(end) is not int or
                start < previous_end or end <= start):
            raise ValueError("document log intervals are invalid or overlap")
        previous_end = end
        uses = []
        rows = 0
        for plan in parsed["plans"]:
            for done in plan["completions"]:
                if start < done["log_line"] <= end:
                    if (plan["model"] != model or type(done["stage"]) is not int or
                            done["stage"] != 0 or type(done["phase"]) is not int or
                            done["phase"] != 4):  # execution::Phase::kScore
                        raise ValueError("target-only score ran an unexpected model/stage/phase")
                    if type(done["rows"]) is not int or not 0 < done["rows"] <= 255:
                        raise ValueError("invalid successful target score input-row count")
                    key = (plan["model"], done["sequence"])
                    if key in seen:
                        raise ValueError("completion belongs to more than one document")
                    seen.add(key)
                    rows += done["rows"]
                    uses.append({"model": model, "identity": plan["identity"],
                                 "completion": done})
        if rows != 255:
            raise ValueError("document must complete exactly 255 target score input rows")
        result.append({"document": d["document"], "score_input_rows": rows, "uses": sorted(
            uses, key=lambda u: u["completion"]["log_line"])})
    all_uses = {(p["model"], c["sequence"]) for p in parsed["plans"] for c in p["completions"]}
    if seen != all_uses:
        raise ValueError("unassociated successful plan use outside document requests")
    return result


def log_lines(path):
    # Production's report sink flushes at each record's end, before the completed
    # transaction returns; HTTP response completion therefore closes this interval.
    return len(path.read_bytes().splitlines())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=pathlib.Path, default=None)
    p.add_argument("--arm", required=True, choices=EXPECTED)
    p.add_argument("--runtime", required=True, type=pathlib.Path)
    p.add_argument("--reference", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--share", type=pathlib.Path, default=pathlib.Path("~/.local/share/llmp"))
    a = p.parse_args()
    os.umask(0o077)
    out = a.out.expanduser().absolute()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    repo = (a.repo if a.repo else HERE.parents[2]).expanduser().resolve(strict=True)
    pins = json.loads((HERE / "mia-capture-pins.json").read_bytes())
    original = json.loads((HERE / "llm-capture-pins.json").read_bytes())
    for name, expected in pins["helpers"].items():
        path = repo / name if "/" in name else HERE / name
        if capture.sha256_file(path) != expected:
            raise ValueError(f"registered helper changed: {name}")
    runtime = a.runtime.expanduser().resolve(strict=True)
    before = capture.sha256_file(runtime)
    if before != EXPECTED[a.arm] or before != original["arms"][a.arm]["sha256"]:
        raise ValueError("runtime differs from the frozen arm")
    sg = capture.load_speed_gate(repo)
    qp = capture.load_quality_pair(repo, original["quality_pair_sha256"])
    reference = a.reference.expanduser().resolve(strict=True)
    docs, retained = reference_inputs(reference, pins, qp)
    # Reuse the complete v1 artifact/tokenizer/template validation. The ordinary
    # corpus is authenticated by this shared helper, but is not sent for scoring.
    a.family = "qwen38"
    a.corpus = a.share.expanduser() / "corpus/m3-reference-corpus.txt"
    template, inputs = capture.verify_inputs(sg, a, original)
    prereg, prereg_sha = qp.load_prereg(str(repo / pins["preregistration"]))
    if prereg["reference_sha256"] != pins["reference_files"]["losses.jsonl"]["sha256"]:
        raise ValueError("preregistration refers to another reference")
    run = sg.fresh_run_dir(out, "qwen38-mia-" + a.arm)
    config = run / "node.toml"
    config.write_text(sg.render_config(template, a.share.expanduser().resolve(), run / "data", 18114))
    log = run / "runtime.log"
    record = {"format": "llmp-u4-mia-target-capture/1", "arm": a.arm,
              "runtime": str(runtime), "runtime_sha256": before, "source": original["arms"][a.arm],
              "inputs": inputs, "reference": str(reference), "reference_files": pins["reference_files"],
              "helpers": pins["helpers"], "collector_sha256": capture.sha256_file(pathlib.Path(__file__)),
              "pins_sha256": capture.sha256_file(HERE / "mia-capture-pins.json"),
              "template_sha256": digest(template.encode()), "config_sha256": capture.sha256_file(config),
              "facts": sg.host_facts(), "documents": [], "error": None,
              "scope": "target-only 1016-loss control; no drafter quality or legacy exception inferred"}
    service = None
    collected = qp.Losses(str(out / "losses.jsonl"), "losses")
    try:
        service = sg.Service(runtime, config, run, 18114, sg.parse_runtime_env(None))
        record["ready_seconds"] = service.wait_ready()
        for d in docs:
            number = d["document"]
            raw = (reference / f"doc{number}.txt").read_bytes()
            if digest(raw) != d["text_sha256"]:
                raise ValueError("document changed after reference verification")
            body = {"model": sg.FAMILIES["qwen38"].model, "prompt": raw.decode(), "max_tokens": 0,
                    "prompt_logprobs": 0, "echo": True, "logprobs": 0,
                    "return_tokens_as_token_ids": True}
            request = json.dumps(body).encode()
            (out / f"doc{number}-request.json").write_bytes(request)
            start = log_lines(log)
            conn = http.client.HTTPConnection("127.0.0.1", 18114, timeout=120)
            try:
                conn.request("POST", "/v1/completions", request, {"Content-Type": "application/json"})
                response = conn.getresponse()
                response_raw = response.read()
                status = response.status
            finally:
                conn.close()
            end = log_lines(log)
            response_name = f"doc{number}-response.json"
            (out / response_name).write_bytes(response_raw)
            receipt = {"document": number, "request_sha256": digest(request), "prompt_sha256": digest(raw),
                       "score_response": {"file": response_name, "sha256": digest(response_raw)},
                       "http_status": status, "log_start_line": start, "log_end_line": end}
            record["documents"].append(receipt)
            if status != 200:
                raise ValueError(f"document {number}: HTTP {status}")
            losses = validate_response(out / f"doc{number}.json", receipt, d["tokens"], qp)
            receipt["ordered_target_ids_sha256"] = qp.ids_digest(d["tokens"][1:])
            receipt["echo_ids_sha256"] = qp.ids_digest(d["tokens"])
            receipt["targets"] = len(losses.rows)
            for (_, position), (target, nll) in losses.rows.items():
                collected.add(number, position, target, nll)
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        if service is not None:
            record["stop"] = service.stop()
        record["runtime_sha256_after"] = capture.sha256_file(runtime)
        write_json(out / "capture.json", record)
    if (record["error"] or record.get("stop", {}).get("returncode") != 0 or
            record["runtime_sha256_after"] != before):
        raise ValueError("incomplete capture, retirement, or changed runtime; see capture.json")
    try:
        parsed = parse_log(log)
        record["document_plan_uses"] = associate_documents(parsed, record["documents"], sg.FAMILIES["qwen38"].model)
        comparison = qp.compare(qp.pair(collected, retained), None)
        checks = qp.check_bounds(comparison, prereg["bounds"])
        record["mia_strict"] = {"prereg_sha256": prereg_sha, "comparison": comparison, "checks": checks,
                                "status": "PASS" if all(c["ok"] for c in checks) else "FAIL"}
        (out / "losses.jsonl").write_text("".join(json.dumps({"document": d, "position": p,
            "target": target, "nll": nll}, allow_nan=False) + "\n"
            for (d, p), (target, nll) in sorted(collected.rows.items())))
        write_json(out / "plans.json", parsed)
        record.update(capture_complete=True, losses_sha256=capture.sha256_file(out / "losses.jsonl"),
                      plans_sha256=capture.sha256_file(out / "plans.json"),
                      debug_log=str(log), debug_log_sha256=capture.sha256_file(log))
        write_json(out / "capture.json", record)
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record.pop("capture_complete", None)
        write_json(out / "capture.json", record)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
