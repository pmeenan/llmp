# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded, supervised scalar HTTP controls for the two approved profiles.

Run from a built Spark tree inside an installed spark-job --gpu job. Raw
responses and runtime logs stay in the supplied external output directory.
This is an own-engine continuity control, not a model-reference comparison.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


ARTIFACTS = {
    "gemma26": "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3",
    "gemma31": "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08",
}
PROMPT = [2, 818, 5279, 529, 7001, 563]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--installed", type=pathlib.Path, required=True)
    parser.add_argument("--runtime", type=pathlib.Path, required=True)
    parser.add_argument("--switches-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in ("spill", "state", "checkpoints"):
        (args.output / name).mkdir(mode=0o700)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    config = args.output / "node.toml"
    config.write_text(
        'schema_version=2\n[storage]\n'
        f'installed="{args.installed.resolve()}"\n'
        + "".join(f'{name}="{args.output.resolve()}/{name}"\n'
                  for name in ("spill", "state", "checkpoints"))
        + "".join(f'[models.{model}]\nartifact="{artifact}"\n'
                  'context=4096\nprefill_chunk=128\nmax_slots=12\n'
                  for model, artifact in ARTIFACTS.items())
        + f'[client]\nbind="127.0.0.1:{port}"\nmodel_turn_seconds=1\n'
        '[memory]\nkeep_across_restart=false\n'
    )
    os.chmod(config, 0o600)
    results = {
        "format": "jitllm-gemma31-scalar-http-control-v1",
        "artifacts": ARTIFACTS,
        "runtime_sha256": hashlib.sha256(args.runtime.read_bytes()).hexdigest(),
        "cases": [],
        "switches_only": args.switches_only,
        "model_turn_seconds": 1,
    }
    log_path = args.output / "runtime.log"
    log = log_path.open("w")
    process = subprocess.Popen(
        [str(args.runtime.resolve()), "--config", str(config.resolve()),
         "--anchor", str((args.output / "anchor").resolve())],
        stdout=log, stderr=log,
    )

    def open_post(path, body):
        return urllib.request.urlopen(urllib.request.Request(
            base + path, json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}), timeout=180)

    def post(path, body):
        with open_post(path, body) as response:
            return json.load(response)

    def literal(model, seed, count=4, **extra):
        return {"model": model, "prompt": PROMPT, "max_tokens": count,
                "temperature": 0.7, "top_k": 16, "seed": seed,
                "return_tokens_as_token_ids": True, **extra}

    def choice(body):
        return post("/v1/completions", body)["choices"][0]

    def stream(path, body, first=None, cancel=False):
        text = ""
        done = False
        with open_post(path, {**body, "stream": True}) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    done = True
                    break
                event = json.loads(line[6:])
                assert "error" not in event, event
                for item in event.get("choices", []):
                    piece = item.get("text", item.get("delta", {}).get("content", ""))
                    text += piece
                    if piece and first is not None:
                        first.set()
                    if piece and cancel:
                        return text
        assert done, "SSE did not complete"
        return text

    def case(name, run):
        started = time.monotonic()
        try:
            detail = run()
            record = {"name": name, "passed": True, "detail": detail}
        except Exception as error:  # Keep later bounded controls useful.
            record = {"name": name, "passed": False,
                      "error": f"{type(error).__name__}: {error}"}
        record["seconds"] = time.monotonic() - started
        results["cases"].append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    try:
        for _ in range(600):
            if process.poll() is not None:
                raise RuntimeError("runtime exited during startup")
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=1) as response:
                    models = json.load(response)
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            raise RuntimeError("runtime startup timed out")
        assert {item["id"] for item in models["data"]} == set(ARTIFACTS)

        if not args.switches_only:
            for model in ARTIFACTS:
                def cohorts(model=model):
                    expected = [choice(literal(model, seed)) for seed in range(200, 212)]
                    for count in (1, 2, 4, 8, 12):
                        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                            actual = list(pool.map(lambda seed: choice(literal(model, seed)),
                                                   range(200, 200 + count)))
                        assert actual == expected[:count]
                    return {"owner_counts": [1, 2, 4, 8, 12], "tokens_per_owner": 4}
                case(model + ":scalar-cohorts", cohorts)

                def chat(model=model):
                    body = {"model": model,
                            "messages": [{"role": "user", "content": "Reply with one short word."}],
                            "max_tokens": 8, "temperature": 0.7, "top_k": 16, "seed": 2718}
                    plain = post("/v1/chat/completions", body)["choices"][0]
                    content = plain["message"]["content"]
                    assert stream("/v1/chat/completions", body) == content
                    assert content, "stop control needs nonempty generated content"
                    delimiter = content[:min(3, len(content))]
                    stopped = post("/v1/chat/completions", {**body, "stop": delimiter})["choices"][0]
                    assert stopped["message"]["content"] == ""
                    assert stopped["finish_reason"] == "stop"
                    for change in (
                        {"tools": [{"type": "function", "function": {"name": "test"}}]},
                        {"messages": [{"role": "tool", "content": "test", "tool_call_id": "x"}]},
                    ):
                        try:
                            post("/v1/chat/completions", {**body, **change})
                        except urllib.error.HTTPError as error:
                            assert error.code == 400
                        else:
                            raise AssertionError("unsupported tool request admitted")
                    stream("/v1/chat/completions", body, cancel=True)
                    assert post("/v1/chat/completions", body)["choices"][0] == plain
                    return {"SSE_exact": True, "stop_delimiter_bytes": len(delimiter.encode()),
                            "tool_refusals": 2, "disconnect_followup_exact": True}
                case(model + ":chat-SSE-stop-refusal-cancel", chat)

                def likelihood(model=model):
                    ids = (PROMPT * 22)[:128]
                    body = literal(model, 772, count=0, prompt=ids, prompt_logprobs=1,
                                   echo=True, logprobs=1)
                    baseline = choice(body)
                    assert len(baseline["prompt_logprobs"]) == 128
                    assert baseline["prompt_logprobs"][0] is None
                    assert len(baseline["logprobs"]["token_logprobs"]) == 128
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        scored = pool.submit(choice, body)
                        peer = pool.submit(choice, literal(model, 200))
                        assert scored.result() == baseline
                        assert peer.result() == choice(literal(model, 200))
                    return {"scored_rows": 127, "first_null": True, "interleaved_repeat_exact": True}
                case(model + ":likelihood-alignment", likelihood)

        for source, target in (("gemma26", "gemma31"), ("gemma31", "gemma26")):
            def switching(source=source, target=target):
                body = {"model": source,
                        "messages": [{"role": "user", "content":
                                      "Continue this numbered list with many entries, one per line: 1. alpha"}],
                        "max_tokens": 128, "temperature": 0.7, "top_k": 16, "seed": 511}
                baseline = post("/v1/chat/completions", body)
                expected = baseline["choices"][0]["message"]["content"]
                actual_count = baseline["usage"]["completion_tokens"]
                assert actual_count >= 64, "pending switch needs a continuing response"
                target_body = literal(target, 512, count=8, logprobs=1)
                peer_expected = choice(target_body)
                marker = f"{source}'s request paused for a model switch"
                before = log_path.read_text().count(marker)
                first = threading.Event()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    ongoing = pool.submit(stream, "/v1/chat/completions", body, first)
                    deadline = time.monotonic() + 120
                    while not first.wait(0.1):
                        if ongoing.done():
                            ongoing.result()  # Surface a refused stream immediately.
                            raise AssertionError("no first stream publication")
                        assert time.monotonic() < deadline, "no first stream publication"
                    peer = pool.submit(choice, target_body)
                    assert ongoing.result() == expected
                    assert peer.result() == peer_expected
                pauses = log_path.read_text().count(marker) - before
                assert pauses > 0, f"{source}->{target} did not pause its active source"
                return {"source_tokens": actual_count, "target_max_tokens": 8,
                        "source_pause_messages": pauses,
                        "owned_outputs_exact": True}
            case(source + "->" + target + ":pending-switch", switching)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        try:
            results["runtime_exit"] = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            results["runtime_exit"] = "retirement-timeout-killed"
        log.close()
        text = log_path.read_text()
        results["model_switch_pause_messages"] = text.count("request paused for a model switch")
        (args.output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    assert results["runtime_exit"] == 0, results["runtime_exit"]
    assert results["model_switch_pause_messages"] >= 2, "no actual pending-switch evidence"
    assert all(item["passed"] for item in results["cases"]), "bounded controls failed"


if __name__ == "__main__":
    main()
