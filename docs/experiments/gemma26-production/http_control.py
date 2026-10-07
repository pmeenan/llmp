# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Ordinary Gemma26/31 HTTP lifecycle, stock top2 IDs and same-geometry repeats."""

import argparse
import array
import concurrent.futures
import hashlib
import heapq
import json
import math
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


PROFILES = {
    "gemma26": ("4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3", 1024),
    "gemma31": ("32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08", 256),
}
INPUT_SHA = "90bf82fd26b7040c513e7370eca2cd283ac8fd15ce89a2e8d05d6f7e670b5ba2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--installed", type=pathlib.Path, required=True)
    parser.add_argument("--runtime", type=pathlib.Path, required=True)
    parser.add_argument("--inputs", type=pathlib.Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="gemma26")
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--build-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--reference-input", type=pathlib.Path, required=True)
    parser.add_argument("--reference-input-sha256", required=True)
    parser.add_argument("--reference-heads", type=pathlib.Path, required=True)
    parser.add_argument("--reference-head-sha256", nargs=3, required=True,
                        metavar=("PREFIX", "SCALAR", "FRESH"))
    parser.add_argument("--case", choices=("all", "literal", "chat", "peers"), default="all")
    args = parser.parse_args()
    model = args.profile
    artifact, prefill = PROFILES[model]
    inputs = args.inputs.read_bytes()
    assert len(inputs) == 131072 and hashlib.sha256(inputs).hexdigest() == INPUT_SHA
    assert hashlib.sha256(args.runtime.read_bytes()).hexdigest() == args.runtime_sha256
    prefix = 256 if model == "gemma26" else 128
    prompt = list(struct.unpack_from(f"<{prefix}i", inputs))
    reference_input = args.reference_input.read_bytes()
    assert hashlib.sha256(reference_input).hexdigest() == args.reference_input_sha256
    assert len(reference_input) == (prefix + 4) * 4
    reference_ids = list(struct.unpack(f"<{prefix + 4}i", reference_input))
    assert reference_ids[:prefix] == prompt
    assert all(0 <= token < 262144 for token in reference_ids)
    reference_top2 = {}
    reference_hashes = {}
    for name, expected_sha in zip((f"prefix{prefix}", f"scalar{prefix + 4}",
                                    f"fresh{prefix + 4}"), args.reference_head_sha256):
        raw = (args.reference_heads / (name + ".f32")).read_bytes()
        assert len(raw) == 262144 * 4 and hashlib.sha256(raw).hexdigest() == expected_sha
        row = array.array("f")
        row.frombytes(raw)
        assert len(row) == 262144 and all(math.isfinite(x) for x in row)
        reference_top2[name] = heapq.nlargest(2, range(len(row)), key=row.__getitem__)
        reference_hashes[name] = expected_sha
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
        + f'[models.{model}]\nartifact="{artifact}"\n'
        'context=8192\nmax_slots=4\n'
        f'[client]\nbind="127.0.0.1:{port}"\n'
        '[memory]\nkeep_across_restart=false\n'
    )
    os.chmod(config, 0o600)
    results = {"schema": 1, "profile": model, "artifact": artifact,
               "runtime_sha256": args.runtime_sha256, "source_commit": args.source_commit,
               "build_receipt_sha256": hashlib.sha256(args.build_receipt.read_bytes()).hexdigest(),
               "harness_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
               "inputs_sha256": INPUT_SHA, "cases": [], "ordinary_production": True,
               "selected_case": args.case,
               "reference_input_sha256": args.reference_input_sha256,
               "reference_head_sha256": reference_hashes,
               "reference_scope": "same-geometry first-row top2 IDs; no HTTP likelihood allowance",
               "scope": "client-observed HTTP lifecycle; no backend cancellation or C4 shape claim"}
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
        return {"model": model, "messages": [{"role": "user", "content": text}],
                "max_tokens": count, "temperature": 0}

    def stream(body, cancel=False, first=None, start=None, all_first=None, departed=None):
        if start is not None:
            start.wait(timeout=30)
        text, done, reason = "", False, None
        read_events_after_disconnect = 0
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
                        read_events_after_disconnect += 1
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
                "read_events_after_disconnect": read_events_after_disconnect}

    def case(group, name, run):
        if args.case not in ("all", group):
            return
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

    try:
        inspected = subprocess.run(
            [str(args.runtime.resolve()), "--config", str(config.resolve()),
             "--anchor", str(anchor), "settings", "--json"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        resolved = json.loads(inspected.stdout)
        assert len(resolved["models"]) == 1 and resolved["models"][0]["name"] == model
        settings = resolved["models"][0]["settings"]
        for key, value, source in (("context", 8192, "override"),
                                   ("max_slots", 4, "override"),
                                   ("prefill_chunk", prefill, "fallback"),
                                   ("speculation", False, "derived")):
            assert settings[key]["value"] == value and settings[key]["source"] == source
        assert settings["prefill_chunk"]["basis"] == f"qualified bounded {model.title()} serving envelope"
        results["resolved_settings"] = resolved
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
        assert [item["id"] for item in models["data"]] == [model]
        log_text = log_path.read_text()
        settings_lines = [line for line in log_text.splitlines()
                          if f"model {model}: settings " in line]
        assert len(settings_lines) == 1, "actual startup settings missing"
        startup = settings_lines[0]
        for field in ("context=8192 (override)", "max_slots=4 (override)",
                      f"prefill_chunk={prefill} (fallback)", "speculation=false (derived)"):
            assert field in startup, (field, startup)
        assert f"prefill chunks of {prefill} rows" in log_text
        assert "Gemma diagnostic" not in log_text, "arithmetic diagnostics must stay off"
        results["startup_settings"] = startup

        def continuation():
            def literal(ids, count, label):
                response = post("/v1/completions", {
                    "model": model, "prompt": ids, "max_tokens": count,
                    "temperature": 0, "logprobs": 2, "return_tokens_as_token_ids": True,
                })
                (args.output / f"literal-{label}.json").write_text(
                    json.dumps({"prompt_ids": ids, "response": response}, indent=2) + "\n")
                choice = response["choices"][0]
                scores = choice["logprobs"]
                assert all(token.startswith("token_id:") for token in scores["tokens"])
                assert len(scores["token_logprobs"]) == len(scores["top_logprobs"]) == count
                assert all(math.isfinite(value) for value in scores["token_logprobs"])
                assert all(len(row) == 2 and all(token.startswith("token_id:") and
                                                math.isfinite(value) for token, value in row.items())
                           for row in scores["top_logprobs"])
                tokens = [int(token.removeprefix("token_id:")) for token in scores["tokens"]]
                assert len(tokens) == count and all(0 <= token < 262144 for token in tokens)
                assert choice["finish_reason"] == "length"
                return tokens, choice

            def top2(choice):
                row = choice["logprobs"]["top_logprobs"][0]
                assert len(row) == 2 and all(math.isfinite(x) for x in row.values())
                return [int(token.removeprefix("token_id:"))
                        for token in sorted(row, key=row.__getitem__, reverse=True)]

            expected, whole = literal(prompt, 16, "whole")
            first, _ = literal(prompt, 4, "first-four")
            assert first == expected[:4], "same-geometry first-four prefix differs"
            repeated, repeated_whole = literal(prompt, 16, "whole-repeat")
            assert repeated == expected and repeated_whole == whole, "same-geometry repeat differs"
            rest, split = literal(prompt + first, 12, "fresh-split")
            repeated_rest, repeated_split = literal(prompt + first, 12, "fresh-split-repeat")
            assert repeated_rest == rest and repeated_split == split, "fresh-geometry repeat differs"
            assert top2(whole) == reference_top2[f"prefix{prefix}"], "stock prefix top2 IDs differ"
            _, fixed = literal(reference_ids, 1, "stock-fixed-prefix")
            assert top2(fixed) == reference_top2[f"fresh{prefix + 4}"], "stock fresh top2 IDs differ"
            joined = first + rest
            comparison = {"uninterrupted_ids": expected, "prefix_ids": first,
                          "continuation_prompt_ids": prompt + first, "continuation_ids": rest,
                          "first_divergence": next((i for i, pair in enumerate(zip(expected, joined))
                                                    if pair[0] != pair[1]), None),
                          "cross_geometry_scope": "descriptive; stock also changes with prefill geometry"}
            (args.output / "literal-comparison.json").write_text(
                json.dumps(comparison, indent=2) + "\n")
            return {"first_four_prefix_exact": True, "continuation_tokens": len(rest),
                    "same_geometry_token_and_score_repeat_exact": True,
                    "same_geometry_stock_first_row_top2_ids_exact": True,
                    "cross_geometry_first_divergence": comparison["first_divergence"],
                    "cross_geometry_matching_positions": sum(a == b for a, b in zip(expected, joined)),
                    "exact_saved_state_continuation_claim": False}
        case("literal", "literal-same-geometry-repeat-and-stock-IDs", continuation)

        def checkpoint_replay():
            body = chat("Continue this numbered list with many entries: 1. checkpoint replay entry", 16)
            first = post("/v1/chat/completions", body)
            short = post("/v1/chat/completions", {**body, "max_tokens": 4})
            replay = post("/v1/chat/completions", body)
            for label, response in (("first", first), ("short", short), ("replay", replay)):
                (args.output / f"chat-checkpoint-{label}.json").write_text(
                    json.dumps(response, indent=2) + "\n")
            assert first["usage"]["completion_tokens"] == replay["usage"]["completion_tokens"] == 16
            assert short["usage"]["completion_tokens"] == 4
            assert first["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
            cached = [response["usage"]["prompt_tokens_details"]["cached_tokens"]
                      for response in (short, replay)]
            assert all(value > 0 for value in cached), "chat checkpoint reuse missing"
            assert replay["choices"] == first["choices"], "chat checkpoint response replay differs"
            return {"request_completion_tokens": [16, 4, 16], "cached_tokens": cached,
                    "response_and_finish_reason_exact": True,
                    "scope": "reported chat prompt checkpoint reuse and exact response replay"}
        case("chat", "chat-cached-checkpoint-replay", checkpoint_replay)

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
            stopped_stream = stream({**body, "stop": delimiter})
            assert stopped_stream["text"] == "" and stopped_stream["finish_reason"] == "stop"
            assert stream(body, cancel=True)["disconnected"]
            assert post("/v1/chat/completions", body)["choices"][0] == plain
            return {"SSE_exact": True, "stop_suppressed": True,
                    "disconnect_followup_exact": True}
        case("chat", "chat-SSE-stop-disconnect", stop_cancel)

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
            assert all(answer["read_events_after_disconnect"] > 0 for answer in answers[1:])
            followup = post("/v1/chat/completions", chat())["choices"][0]
            assert followup["message"]["content"]
            return {"concurrent_requests": 4, "departed_owners": 1,
                    "completed_peers": 3, "followup_healthy": True,
                    "peer_read_events_after_disconnect":
                        [answer["read_events_after_disconnect"] for answer in answers[1:]],
                    "progress_scope": "client-observed SSE reads and completion; reads may be buffered",
                    "scalar_batch_equality_claim": False}
        case("peers", "four-requests-departed-client-peer-completion", departed_owner)
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
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    assert results["runtime_exit"] == 0, results["runtime_exit"]
    assert len(results["cases"]) == ({"all": 4, "literal": 1, "chat": 2, "peers": 1}[args.case])
    assert all(case["passed"] for case in results["cases"])

if __name__ == "__main__":
    main()
