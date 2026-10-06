# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Gemma31 candidate HTTP lifetime controls; no reference or timing claim."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import signal
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request


ARTIFACT = "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08"
INPUT_SHA = "90bf82fd26b7040c513e7370eca2cd283ac8fd15ce89a2e8d05d6f7e670b5ba2"
RUNTIME_SHA = "4b6809227f847da6f628babae1665181e3f022f9aabad58dc2fefbb04130d8ae"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--installed", type=pathlib.Path, required=True)
    parser.add_argument("--runtime", type=pathlib.Path, required=True)
    parser.add_argument("--inputs", type=pathlib.Path, required=True)
    args = parser.parse_args()
    inputs = args.inputs.read_bytes()
    assert len(inputs) == 131072 and hashlib.sha256(inputs).hexdigest() == INPUT_SHA
    assert hashlib.sha256(args.runtime.read_bytes()).hexdigest() == RUNTIME_SHA
    prompt = list(struct.unpack_from("<256i", inputs))
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in ("spill", "state", "checkpoints"):
        (args.output / name).mkdir(mode=0o700)
    anchor = args.output / "enrollment"
    assert not os.path.lexists(anchor)
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
        + f'[models.gemma31]\nartifact="{ARTIFACT}"\n'
        'context=8192\nprefill_chunk=256\nmax_slots=4\n'
        f'[client]\nbind="127.0.0.1:{port}"\n'
        '[memory]\nkeep_across_restart=false\n'
    )
    os.chmod(config, 0o600)
    results = {"schema": 1, "artifact": ARTIFACT, "runtime_sha256": RUNTIME_SHA,
               "inputs_sha256": INPUT_SHA, "cases": [], "default_adoption": False}
    log_path = args.output / "runtime.log"
    log = log_path.open("x")
    process = subprocess.Popen(
        [str(args.runtime.resolve()), "--config", str(config.resolve()), "--anchor", str(anchor)],
        stdout=log, stderr=log,
    )

    def open_post(path, body):
        return urllib.request.urlopen(urllib.request.Request(
            base + path, json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}), timeout=120)

    def post(path, body):
        with open_post(path, body) as response:
            return json.load(response)

    def chat(text="Reply with one short word.", count=16):
        return {"model": "gemma31", "messages": [{"role": "user", "content": text}],
                "max_tokens": count, "temperature": 0}

    def stream(body, cancel=False, first=None, start=None, all_first=None, departed=None):
        if start is not None:
            start.wait(timeout=30)
        text, done, reason = "", False, None
        post_departure_publications = 0
        disconnect = False
        with open_post("/v1/chat/completions", {**body, "stream": True}) as response:
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
                    piece = item.get("delta", {}).get("content", "")
                    text += piece
                    reason = item.get("finish_reason") or reason
                    was_started = first is not None and first.is_set()
                    if piece and was_started and departed is not None and departed.is_set():
                        post_departure_publications += 1
                    if piece and first is not None:
                        first.set()
                    if piece and cancel:
                        if all_first is not None:
                            deadline = time.monotonic() + 60
                            for event in all_first:
                                assert event.wait(max(0, deadline - time.monotonic())), \
                                    "all owners must publish before departure"
                        disconnect = True
                        break
                    if piece and departed is not None and not departed.is_set():
                        assert departed.wait(timeout=60), "coordinated departure missing"
                if disconnect:
                    break
        # The response has closed before releasing peers to read later SSE.
        if disconnect:
            if departed is not None:
                departed.set()
            return {"text": text, "disconnected": True}
        assert done and reason in ("stop", "length"), "SSE retirement missing"
        return {"text": text, "finish_reason": reason, "done": done,
                "post_departure_publications": post_departure_publications}

    def case(name, run):
        started = time.monotonic()
        try:
            detail = run()
            record = {"name": name, "passed": True, "detail": detail}
        except Exception as error:
            record = {"name": name, "passed": False,
                      "error": f"{type(error).__name__}: {error}"}
        record["seconds"] = time.monotonic() - started
        results["cases"].append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        assert record["passed"], record

    try:
        for _ in range(600):
            assert process.poll() is None, "runtime exited during startup"
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=1) as response:
                    models = json.load(response)
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            raise AssertionError("runtime startup timed out")
        assert [item["id"] for item in models["data"]] == ["gemma31"]
        marker = "Gemma diagnostic gemma31: "
        lines = [line.split(marker, 1)[1] for line in log_path.read_text().splitlines()
                 if marker in line]
        assert len(lines) == 1
        status = json.loads(lines[0])
        assert status["gemma31_candidate"] and status["dispatch"] == "joined-candidate"
        assert not status["row_invariant"] and status["max_rows"] == 256
        assert all(status[key] for key in ("norm_rope_requested", "norm_add_requested",
                                          "owner_attention_requested"))
        results["actual_startup_recipe"] = status

        def continuation():
            def literal(ids, count):
                response = post("/v1/completions", {
                    "model": "gemma31", "prompt": ids, "max_tokens": count,
                    "temperature": 0, "logprobs": 1, "return_tokens_as_token_ids": True,
                })
                choice = response["choices"][0]
                tokens = [int(token.removeprefix("token_id:"))
                          for token in choice["logprobs"]["tokens"]]
                assert all(0 <= token < 262144 for token in tokens)
                assert choice["finish_reason"] in ("stop", "length")
                return tokens
            expected = literal(prompt, 16)
            assert len(expected) >= 2, "continuation needs a nontrivial generated prefix"
            keep = min(4, len(expected) // 2)
            first = literal(prompt, keep)
            assert first == expected[:keep]
            rest = literal(prompt + first, 16 - keep)
            assert first + rest == expected
            return {"committed_prefix_tokens": keep, "continuation_tokens": len(rest),
                    "same_scalar_query_arithmetic": True, "exact_token_continuation": True}
        case("literal-owned-continuation", continuation)

        def stop_cancel():
            body = chat()
            plain = post("/v1/chat/completions", body)["choices"][0]
            content = plain["message"]["content"]
            assert content
            actual = stream(body)
            assert actual["text"] == content and actual["finish_reason"] == plain["finish_reason"]
            delimiter = content[:min(3, len(content))]
            stopped = post("/v1/chat/completions", {**body, "stop": delimiter})["choices"][0]
            assert stopped["message"]["content"] == "" and stopped["finish_reason"] == "stop"
            assert stream(body, cancel=True)["disconnected"]
            assert post("/v1/chat/completions", body)["choices"][0] == plain
            return {"SSE_exact": True, "stop_suppressed": True,
                    "disconnect_followup_exact": True}
        case("chat-SSE-stop-disconnect", stop_cancel)

        def departed_owner():
            start = threading.Barrier(4)
            first = [threading.Event() for _ in range(4)]
            departed = threading.Event()
            bodies = [chat(f"Continue this numbered list with many entries, one per line: 1. item {i}",
                           128) for i in range(4)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                runs = [pool.submit(stream, bodies[i], i == 0, first[i], start, first, departed)
                        for i in range(4)]
                answers = [run.result() for run in runs]
            assert all(event.is_set() for event in first), "every request must publish"
            assert answers[0]["disconnected"]
            assert all(answer["done"] and answer["text"] for answer in answers[1:])
            assert all(answer["post_departure_publications"] > 0 for answer in answers[1:])
            followup = post("/v1/chat/completions", chat())["choices"][0]
            assert followup["message"]["content"]
            return {"concurrent_requests": 4, "departed_owners": 1,
                    "completed_peers": 3, "followup_healthy": True,
                    "peer_publications_after_disconnect":
                        [answer["post_departure_publications"] for answer in answers[1:]],
                    "progress_scope": "client-observed SSE publication and completion",
                    "scalar_batch_equality_claim": False}
        case("C4-departed-owner-peer-progress", departed_owner)
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
        marker = "Gemma diagnostic final gemma31: "
        lines = [line.split(marker, 1)[1] for line in log_path.read_text().splitlines()
                 if marker in line]
        if len(lines) == 1:
            results["actual_final_recipe"] = json.loads(lines[0])
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    assert results["runtime_exit"] == 0
    final = results["actual_final_recipe"]
    assert final["joined_groups"] > 0 and final["joined_units"] > final["joined_groups"]
    assert len(results["cases"]) == 3 and all(case["passed"] for case in results["cases"])


if __name__ == "__main__":
    main()
