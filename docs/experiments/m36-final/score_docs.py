#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Score fixed text documents through the literal completions route, one process per call.

    score_docs.py --family F --runtime RUNTIME --template FILE --out DIR DOC0.txt [DOC1.txt ...]

Serves the family's node configuration (`speed_gate.py`'s own Service and renderer, so the
configuration, the data directory and the port are the gate's), posts each document to
/v1/completions with the quality control's body (`prompt_logprobs` 0, `echo`,
`return_tokens_as_token_ids`) and writes DIR/losses.jsonl, one
`{"document": D, "position": P, "target": ID, "nll": X}` a line in the form
`quality_pair.py` and `collect-qwen-reference.py` share, plus DIR/score-docs.json (the
runtime's and the documents' SHA-256, each document's token count and mean NLL).
Documents are numbered by their order on the command line. Prompts are never kept.
Run it beside tools/speed_gate.py (it imports it) under the Spark's GPU supervisor.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import speed_gate as sg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(sg.FAMILIES))
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--share", default="~/.local/share/llmp")
    ap.add_argument("--port", type=int, default=sg.PORT)
    ap.add_argument("--runtime-env", action="append", metavar="NAME=VALUE")
    ap.add_argument("docs", nargs="+")
    args = ap.parse_args()
    family = sg.FAMILIES[args.family]
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = pathlib.Path(args.runtime).resolve()
    share = pathlib.Path(args.share).expanduser().resolve()
    env = sg.parse_runtime_env(args.runtime_env)
    run_dir = sg.fresh_run_dir(out, f"{family.name}-score-docs")
    template = pathlib.Path(args.template).expanduser().read_text()
    config = run_dir / "node.toml"
    config.write_text(sg.render_config(template, share, run_dir / "data", args.port))
    record = {"format": "llmp-score-docs/1", "family": family.name,
              "runtime": str(runtime), "runtime_sha256": sg.sha256_file(runtime),
              "config_sha256": hashlib.sha256(template.encode()).hexdigest(),
              "runtime_env": env, "facts": sg.host_facts(), "documents": [], "error": None}
    rows = []
    service = None
    try:
        service = sg.Service(runtime, config, run_dir, args.port, env)
        service.wait_ready()
        for number, name in enumerate(args.docs):
            text = pathlib.Path(name).read_text()
            conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=3600)
            conn.request("POST", "/v1/completions", json.dumps(
                {"model": family.model, "prompt": text, "max_tokens": 0, "prompt_logprobs": 0,
                 "echo": True, "logprobs": 0, "return_tokens_as_token_ids": True}),
                {"Content-Type": "application/json"})
            response = conn.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"scoring {name}: HTTP {response.status}: {body[:400]!r}")
            choice = json.loads(body)["choices"][0]
            lp = choice["prompt_logprobs"]
            if not lp or lp[0] is not None:
                raise RuntimeError(f"{name}: prompt_logprobs must start with the unscored token")
            nll = []
            for position in range(1, len(lp)):
                label, entry = next(iter(lp[position].items()))
                target = int(label[len("token_id:"):] if label.startswith("token_id:") else label)
                rows.append({"document": number, "position": position, "target": target,
                             "nll": -float(entry["logprob"])})
                nll.append(-float(entry["logprob"]))
            record["documents"].append({
                "document": number, "file_sha256": sg.sha256_file(pathlib.Path(name)),
                "tokens": len(lp), "targets": len(nll), "mean_nll": sum(nll) / len(nll)})
    except Exception as e:  # the record says how the run failed
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        if service is not None:
            record["stop"] = service.stop()
    (out / "losses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "score-docs.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"error": record["error"], "documents": record["documents"]}), flush=True)
    return 1 if record["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
