#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Slice K1 long-context Q8_0 vs F16 KV quality evaluation tool.

Evaluates opt-in lossy KV cache encoding (kv_cache_encoding = "q8_0") against
baseline (kv_cache_encoding = "f16") on migrated LLM families across long contexts, each
served from its v1 import (D-118).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import http.client
import json
import math
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import speed_gate  # noqa: E402  (the published v1 IDs)

PORT = 18114
DEFAULT_SHARE = "~/.local/share/llmp"
DEFAULT_CORPUS = "~/.local/share/llmp/corpus/m3-reference-corpus.txt"
DIAGNOSTIC_PREFIX = "LLMP_DIAGNOSTIC_"

QUALITY_PROMPTS = (
    "Explain, step by step, how a hash table handles collisions.",
    "Write a short story about a lighthouse keeper who finds a map.",
)

# The v1 imports of every family (D-118), served from one store.
V1_STORE = "u2-v1-artifacts"


def resolve_store(store: str, share: pathlib.Path) -> str:
    """Resolve `store` under `share`, falling back to `artifacts` for v1 stores."""
    if store == V1_STORE and os.environ.get("LLMP_TEST_V1_STORE"):
        return os.environ["LLMP_TEST_V1_STORE"]
    if (share / store).is_dir():
        return store
    if store == V1_STORE and (share / "artifacts").is_dir():
        return "artifacts"
    return store
DSV4 = speed_gate.DSV4_V1
DSPARK = speed_gate.DSPARK_V1
QWEN = speed_gate.QWEN_V1
QWEN_MTP = speed_gate.QWEN_MTP_V1
QWEN_GGUF = speed_gate.QWEN_GGUF_V1
GEMMA2 = speed_gate.GEMMA2_V1
GEMMA3 = speed_gate.GEMMA3_V1
GEMMA26 = speed_gate.GEMMA26_V1
GEMMA26_ASSISTANT = speed_gate.GEMMA26_ASSISTANT_V1
GEMMA31 = speed_gate.GEMMA31_V1
GEMMA31_ASSISTANT = speed_gate.GEMMA31_ASSISTANT_V1
QWEN_TOKENIZER = "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6"


def _qwen_files() -> str:
    return (
        f'tokenizer = "{{share}}/{QWEN_TOKENIZER}/tokenizer.json"\n'
        f'chat_template = "{{share}}/{QWEN_TOKENIZER}/chat_template.jinja"\n'
    )


# Gemma 4's instruction-tuned checkpoints are not raw-text language models:
# scored as bare text after BOS (as llama.cpp's llama-perplexity does), both
# engines put most of the mass on control tokens and give perplexities in the
# tens of thousands. The passage is therefore scored as the model's own turn
# of its chat format (enable_thinking off: an empty thought channel), and only
# the passage's rows count. The wrapper ends with a special token, so the
# passage's tokenization does not depend on it.
GEMMA4_TURN = (
    "<|turn>user\nWrite the next part of the project documentation.<turn|>\n"
    "<|turn>model\n<|channel>thought\n<channel|>"
)


@dataclasses.dataclass(frozen=True)
class Family:
    name: str
    model: str
    store: str
    models_toml: str
    target_tokens: int  # the passage's tokens to score, at most
    prefix: str = ""  # text before the passage, not scored (chat-format wrapper)


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        Family(
            "gemma2",
            "gemma2",
            V1_STORE,
            f'[models.gemma2]\nartifact = "{GEMMA2}"\ncontext = 8192\nprefill_chunk = 128\n'
            "max_slots = 2\n",
            target_tokens=8000,
        ),
        Family(
            "gemma3",
            "gemma3",
            V1_STORE,
            f'[models.gemma3]\nartifact = "{GEMMA3}"\ncontext = 8448\nprefill_chunk = 128\n'
            "max_slots = 1\n",
            target_tokens=8000,
        ),
        Family(
            "gemma26",
            "gemma26",
            V1_STORE,
            f'[models.gemma26]\nartifact = "{GEMMA26}"\ncontext = 8192\nprefill_chunk = 1024\n'
            "max_slots = 4\n",
            target_tokens=8000,
            prefix=GEMMA4_TURN,
        ),
        Family(
            "gemma26-assistant",
            "gemma26",
            V1_STORE,
            f'[models.gemma26]\nartifact = "{GEMMA26}"\ndrafter = "{GEMMA26_ASSISTANT}"\n'
            "speculation = true\ncontext = 8192\nprefill_chunk = 1024\nmax_slots = 4\n",
            target_tokens=1024,
            prefix=GEMMA4_TURN,
        ),
        Family(
            "gemma31",
            "gemma31",
            V1_STORE,
            f'[models.gemma31]\nartifact = "{GEMMA31}"\ncontext = 8192\nprefill_chunk = 256\n'
            "max_slots = 4\n",
            target_tokens=8000,
            prefix=GEMMA4_TURN,
        ),
        Family(
            "gemma31-assistant",
            "gemma31",
            V1_STORE,
            f'[models.gemma31]\nartifact = "{GEMMA31}"\ndrafter = "{GEMMA31_ASSISTANT}"\n'
            "speculation = true\ncontext = 8192\nprefill_chunk = 256\nmax_slots = 4\n",
            target_tokens=1024,
            prefix=GEMMA4_TURN,
        ),
        Family(
            "qwen38",
            "qwen3.8",
            V1_STORE,
            f'[models."qwen3.8"]\nartifact = "{QWEN}"\ndrafter = "{QWEN_MTP}"\nspeculation = true\n'
            "draft_vocab = 65536\ndepth_cost_ratio = 1.16\ncontext = 33792\nprefill_chunk = 4096\n"
            "max_slots = 4\n" + _qwen_files(),
            target_tokens=16000,
        ),
        Family(
            "qwen38-gguf",
            "qwen3.8-gguf",
            V1_STORE,
            f'[models."qwen3.8-gguf"]\nartifact = "{QWEN_GGUF}"\ncontext = 33792\n'
            "prefill_chunk = 4096\nmax_slots = 4\n" + _qwen_files(),
            target_tokens=16000,
        ),
        Family(
            "deepseek",
            "deepseek",
            V1_STORE,
            f'[models.deepseek]\nartifact = "{DSV4}"\ndrafter = "{DSPARK}"\nspeculation = true\n'
            'wave_form = "plain"\ncontext = 262144\nprefill_chunk = 4096\nmax_slots = 4\n',
            target_tokens=16000,
        ),
    )
}

STANDARD_LLM_FAMILIES = (
    "gemma2",
    "gemma3",
    "gemma26",
    "gemma31",
    "qwen38",
    "qwen38-gguf",
    "deepseek",
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(path: pathlib.Path) -> str:
    data = path.read_bytes()
    return data.decode("utf-8")


MAX_PASSAGE_CHARS = 200_000


def passage(corpus: str, index: int, chars: int) -> str:
    """The index-th passage's first `chars` characters, from a line start.

    The start does not depend on `chars`, so a shorter passage is a prefix of
    a longer one with the same index.
    """
    if chars > MAX_PASSAGE_CHARS:
        raise ValueError(f"a passage of at most {MAX_PASSAGE_CHARS} characters")
    span = len(corpus) - MAX_PASSAGE_CHARS - 1
    if span <= 0:
        return corpus[:chars]
    start = (index * 7919 * 211) % span
    newline = corpus.find("\n", start)
    start = newline + 1 if 0 <= newline < span else start
    return corpus[start : start + chars]


def parse_runtime_env(pairs: list[str] | None) -> dict[str, str]:
    """Diagnostic switches, NAME=VALUE each, sorted by name."""
    env: dict[str, str] = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if not sep or not name.startswith(DIAGNOSTIC_PREFIX) or name in env:
            raise SystemExit(f"--runtime-env {pair!r}: not a single {DIAGNOSTIC_PREFIX}*=VALUE")
        env[name] = value
    return dict(sorted(env.items()))


def arm_env(runtime_env: dict[str, str]) -> dict[str, str]:
    """The runtime process's environment."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(DIAGNOSTIC_PREFIX)}
    env.update(runtime_env, CUDA_DISABLE_PTX_JIT="1")
    return env


def make_config(
    family: Family, encoding: str, share: pathlib.Path, data: pathlib.Path, port: int
) -> str:
    store = resolve_store(family.store, share)
    template = (
        'schema_version = 2\n[storage]\ndata_dir = "{data}"\n'
        f'installed = "{{share}}/{store}"\n'
        "[client]\nbind = [\"loopback\"]\nport = {port}\n"
        + family.models_toml.rstrip()
        + f'\nkv_cache_encoding = "{encoding}"\n'
    )
    return (
        template.replace("{share}", str(share))
        .replace("{data}", str(data))
        .replace("{port}", str(port))
    )


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def host_facts() -> dict:
    facts = {"host": os.uname().nodename, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        facts["gpu_compute_apps"] = apps.splitlines() if apps else []
        facts["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        facts["gpu_error"] = str(e)
    return facts


def fresh_run_dir(out: pathlib.Path, name: str) -> pathlib.Path:
    run_dir = out / "work" / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    os.chmod(run_dir, 0o700)
    return run_dir


class Service:
    def __init__(
        self,
        runtime: pathlib.Path,
        config: pathlib.Path,
        run_dir: pathlib.Path,
        port: int,
        runtime_env: dict[str, str],
    ):
        self.port = port
        if port_open(port):
            raise RuntimeError(f"something already listens on port {port}")
        self.log = open(run_dir / "runtime.log", "wb")
        env = arm_env(runtime_env)
        self.started = time.monotonic()
        self.proc = subprocess.Popen(
            [str(runtime), "--config", str(config), "--anchor", str(run_dir / "enrollment")],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def wait_ready(self, timeout: float = 900) -> float:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the runtime exited with {self.proc.returncode} before ready")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                conn.request("GET", "/v1/models")
                status = conn.getresponse().status
                conn.close()
                if status == 200:
                    if self.proc.poll() is not None:
                        raise RuntimeError(f"the runtime exited with {self.proc.returncode}")
                    return time.monotonic() - self.started
            except OSError:
                pass
            time.sleep(0.2)
        raise RuntimeError("the runtime was not ready in time")

    def stop(self, timeout: float = 180) -> dict:
        ended = {"signal": "TERM"}
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                ended["signal"] = "KILL"
        ended["returncode"] = self.proc.returncode
        self.log.close()
        return ended


def stream_chat(port: int, body: dict) -> dict:
    """One streamed chat request: timings, usage, and reply digest."""
    record: dict = {"error": None}
    payload = json.dumps(body).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    try:
        conn.connect()
        record["t_send"] = time.monotonic()
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            record["error"] = f"HTTP {response.status}: {response.read()[:400]!r}"
            return record
        digest = hashlib.sha256()
        times: list[float] = []
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            now = time.monotonic()
            if data == b"[DONE]":
                record["t_end"] = now
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                record["usage"] = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                text = (delta.get("reasoning") or "") + (delta.get("content") or "")
                if text:
                    times.append(now)
                    digest.update(text.encode())
                    record.setdefault("t_first", now)
                    record["t_last"] = now
                if choice.get("finish_reason"):
                    record["finish"] = choice["finish_reason"]
        record["chunks"] = len(times)
        record["times"] = times
        record["reply_sha256"] = digest.hexdigest()
        for key in ("t_first", "t_end", "usage"):
            if key not in record:
                record["error"] = f"the stream ended without {key}"
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    return record


BUCKET_TOKENS = 2048


def nll_ppl(scores: list[float]) -> tuple[float, float]:
    nll = -sum(scores) / len(scores)
    return nll, math.exp(nll)


def compute_metrics(scores: list[float]) -> dict:
    n = len(scores)
    if n == 0:
        raise ValueError("no scored tokens")

    overall_nll, overall_ppl = nll_ppl(scores)
    early_n = min(1024, n)
    early_nll, early_ppl = nll_ppl(scores[:early_n])
    late_n = min(2048, n)
    late_nll, late_ppl = nll_ppl(scores[n - late_n :])
    # Depth buckets: passage positions [i, i + BUCKET_TOKENS).
    buckets = [
        {"first": i, "tokens": len(scores[i : i + BUCKET_TOKENS]),
         "nll": nll_ppl(scores[i : i + BUCKET_TOKENS])[0]}
        for i in range(0, n, BUCKET_TOKENS)
    ]

    return {
        "tokens": n,
        "overall_nll": overall_nll,
        "overall_ppl": overall_ppl,
        "early_tokens": early_n,
        "early_nll": early_nll,
        "early_ppl": early_ppl,
        "late_tokens": late_n,
        "late_nll": late_nll,
        "late_ppl": late_ppl,
        "buckets": buckets,
    }


def post_json(port: int, path: str, body: dict) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
    finally:
        conn.close()
    if response.status != 200:
        raise RuntimeError(f"{path}: HTTP {response.status}: {data[:400]!r}")
    return json.loads(data)


def prompt_tokens(port: int, model: str, text: str) -> int:
    """The prompt's token count (BOS included), by a one-token completion."""
    body = {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0}
    return int(post_json(port, "/v1/completions", body)["usage"]["prompt_tokens"])


def fit_passage(port: int, family: Family, corpus: str, prefix_tokens: int) -> str:
    """The longest passage (index 100003) whose tokens fit target_tokens.

    Sized from the server's own token counts (a cheap prefill each), so every
    tokenizer gets its own length and the context is never exceeded.
    """
    chars = family.target_tokens * 3
    best = ""
    for _ in range(8):
        text = passage(corpus, 100003, chars)
        tokens = prompt_tokens(port, family.model, family.prefix + text) - prefix_tokens
        if tokens <= family.target_tokens:
            best = text if len(text) > len(best) else best
            if tokens >= family.target_tokens * 0.98:
                break
        chars = int(len(text) * family.target_tokens * 0.99 / max(tokens, 1))
    if not best:
        raise RuntimeError("no passage fits the target token count")
    return best


def score_rows(rows: list, first: int) -> tuple[list[int], list[float]]:
    """The scored targets' IDs and log-probabilities from row `first` on.

    With prompt_logprobs = 0 each row holds its own target only (keyed by ID).
    """
    ids: list[int] = []
    scores: list[float] = []
    for row in rows[max(first, 1) :]:
        if not row or len(row) != 1:
            raise RuntimeError("a scored row without exactly its target")
        token, value = next(iter(row.items()))
        ids.append(int(token))
        scores.append(float(value["logprob"]))
    return ids, scores


def evaluate_arm(
    family: Family,
    encoding: str,
    runtime: pathlib.Path,
    share: pathlib.Path,
    corpus: str,
    out: pathlib.Path,
    port: int,
    runtime_env: dict[str, str],
    timeout: float,
    keep_data: bool,
    text: str | None = None,
) -> dict:
    """One arm; `text` is the passage the other arm scored (None: fit it here)."""
    name = f"{family.name}-k1-{encoding}"
    run_dir = fresh_run_dir(out, name)
    config = run_dir / "node.toml"
    cfg_text = make_config(family, encoding, share, run_dir / "data", port)
    config.write_text(cfg_text)

    record: dict = {
        "encoding": encoding,
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "error": None,
    }

    service = None
    try:
        service = Service(runtime, config, run_dir, port, runtime_env)
        record["ready_s"] = service.wait_ready(timeout)

        # 1. Teacher-forced scoring of the passage (behind its chat-format
        # wrapper, if any; only the passage's rows count).
        prefix_tokens = prompt_tokens(port, family.model, family.prefix) if family.prefix else 1
        if text is None:
            text = fit_passage(port, family, corpus, prefix_tokens)
        record["prefix_tokens"] = prefix_tokens
        record["prefix_sha256"] = hashlib.sha256(family.prefix.encode()).hexdigest()
        record["prompt_chars"] = len(text)
        record["prompt_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        record["passage"] = text

        started = time.monotonic()
        resp_json = post_json(
            port,
            "/v1/completions",
            {
                "model": family.model,
                "prompt": family.prefix + text,
                "max_tokens": 0,
                "prompt_logprobs": 0,
                "return_tokens_as_token_ids": True,
            },
        )
        record["score_s"] = time.monotonic() - started
        rows = resp_json["choices"][0]["prompt_logprobs"]
        record["prompt_tokens"] = len(rows)
        ids, scores = score_rows(rows, prefix_tokens)
        if not scores:
            raise RuntimeError("scoring: 0 tokens scored in prompt_logprobs")
        record["ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        record["scores"] = scores
        record.update(compute_metrics(scores))

        # 2. Greedy stream_chat validation
        chat_body = {
            "model": family.model,
            "stream": True,
            "temperature": 0,
            "stream_options": {"include_usage": True},
            "max_tokens": 64,
            "messages": [{"role": "user", "content": QUALITY_PROMPTS[0]}],
        }
        chat_res = stream_chat(port, chat_body)
        if chat_res.get("error"):
            raise RuntimeError(f"chat stream failed: {chat_res['error']}")

        record["chat_reply_sha256"] = chat_res.get("reply_sha256")
        record["chat_tokens"] = chat_res.get("usage", {}).get("completion_tokens", 0)
        record["chat_chunks"] = chat_res.get("chunks", 0)

    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        if service is not None:
            record["stop"] = service.stop()
        if not keep_data:
            shutil.rmtree(run_dir / "data", ignore_errors=True)

    return record


def arm_path(out: pathlib.Path, family: Family, encoding: str) -> pathlib.Path:
    return out / f"{family.name}-k1-{encoding}.json"


def combine(family: Family, runtime: pathlib.Path, f16: dict, q8_0: dict) -> dict:
    if f16["prompt_sha256"] != q8_0["prompt_sha256"] or f16["ids_sha256"] != q8_0["ids_sha256"]:
        raise RuntimeError("the arms scored different passages")
    delta = {}
    for part in ("overall", "early", "late"):
        delta[f"{part}_nll_delta"] = q8_0[f"{part}_nll"] - f16[f"{part}_nll"]
        delta[f"{part}_ppl_change_pct"] = (
            (q8_0[f"{part}_ppl"] - f16[f"{part}_ppl"]) / f16[f"{part}_ppl"] * 100.0
        )
    delta["bucket_nll_delta"] = [
        {"first": a["first"], "tokens": a["tokens"], "nll_delta": b["nll"] - a["nll"]}
        for a, b in zip(f16["buckets"], q8_0["buckets"])
    ]
    pairs = list(zip(f16["scores"], q8_0["scores"]))
    delta["mean_abs_logprob_diff"] = sum(abs(a - b) for a, b in pairs) / len(pairs)
    delta["max_abs_logprob_diff"] = max(abs(a - b) for a, b in pairs)
    slim = {k: v for k, v in f16.items() if k not in ("scores", "passage")}
    slim_q8 = {k: v for k, v in q8_0.items() if k not in ("scores", "passage")}
    return {
        "format": "llmp-k1-quality/2",
        "family": family.name,
        "model": family.model,
        "runtime": str(runtime),
        "runtime_sha256": sha256_file(runtime),
        "prefix_tokens": f16["prefix_tokens"],
        "prompt_chars": f16["prompt_chars"],
        "prompt_sha256": f16["prompt_sha256"],
        "prompt_tokens": f16["prompt_tokens"],
        "tokens": f16["tokens"],
        "f16": slim,
        "q8_0": slim_q8,
        "delta": delta,
        "facts": host_facts(),
        "error": None,
    }


def evaluate_family(
    family: Family,
    runtime: pathlib.Path,
    share: pathlib.Path,
    corpus: str,
    out: pathlib.Path,
    port: int,
    runtime_env: dict[str, str],
    timeout: float,
    keep_data: bool,
    arms: tuple[str, ...] = ("f16", "q8_0"),
) -> dict:
    """Runs the named arms (each saved alone) and combines both when present.

    An arm not named is read from its saved file, so the two arms of a long
    family can run as separately supervised jobs; an arm scores its saved
    twin's passage when there is one.
    """
    results: dict[str, dict] = {}
    for encoding in ("f16", "q8_0"):
        path = arm_path(out, family, encoding)
        if encoding not in arms:
            if path.exists():
                results[encoding] = json.loads(path.read_text())
            continue
        twin = "q8_0" if encoding == "f16" else "f16"
        other = results.get(twin) or (
            json.loads(arm_path(out, family, twin).read_text())
            if arm_path(out, family, twin).exists()
            else None
        )
        text = other.get("passage") if other and not other.get("error") else None
        record = evaluate_arm(
            family, encoding, runtime, share, corpus, out, port, runtime_env, timeout, keep_data,
            text,
        )
        path.write_text(json.dumps(record) + "\n")
        if record.get("error"):
            print(f"ERROR: {family.name} arm {encoding} failed: {record['error']}", file=sys.stderr)
            return {"family": family.name, "error": f"{encoding}: {record['error']}"}
        results[encoding] = record
        print(
            json.dumps({"family": family.name, "encoding": encoding, "tokens": record["tokens"],
                        "overall_ppl": round(record["overall_ppl"], 4),
                        "score_s": round(record["score_s"], 1)}),
            flush=True,
        )
    if "f16" not in results or "q8_0" not in results:
        return {"family": family.name, "error": None, "partial": sorted(results)}

    record = combine(family, runtime, results["f16"], results["q8_0"])
    (out / f"{family.name}-k1.json").write_text(json.dumps(record, indent=2) + "\n")
    f16, q8_0, d = record["f16"], record["q8_0"], record["delta"]
    chat_ok = (
        f16.get("chat_reply_sha256") is not None and q8_0.get("chat_reply_sha256") is not None
    )
    chat = "same" if f16.get("chat_reply_sha256") == q8_0.get("chat_reply_sha256") else "differs"
    print(
        f"| {family.name} | {f16['tokens']} | {f16['overall_ppl']:.4f} | {q8_0['overall_ppl']:.4f} | "
        f"{d['overall_ppl_change_pct']:+.3f}% | {d['early_ppl_change_pct']:+.3f}% | "
        f"{d['late_ppl_change_pct']:+.3f}% | {'PASS' if chat_ok else 'FAIL'} ({chat}) |",
        flush=True,
    )
    return record


def print_table_header() -> None:
    print(
        "| Family | Scored Tokens | F16 Overall PPL | Q8_0 Overall PPL | Overall PPL Δ | "
        "Early PPL Δ (0-1K) | Late PPL Δ (last 2K) | Chat |",
        flush=True,
    )
    print(
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Slice K1 long-context Q8_0 vs F16 KV quality evaluation tool."
    )
    parser.add_argument(
        "--family",
        required=True,
        help=f"model family to evaluate, or 'all'. Choices: {sorted(FAMILIES)} or 'all'",
    )
    parser.add_argument(
        "--with-assistant",
        action="store_true",
        help="for gemma26 and gemma31, evaluate with assistant drafter enabled",
    )
    parser.add_argument("--runtime", required=True, help="path to llmp-runtime binary")
    parser.add_argument("--out", required=True, help="output directory for evaluation results")
    parser.add_argument(
        "--share", default=DEFAULT_SHARE, help=f"share directory (default {DEFAULT_SHARE})"
    )
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS, help=f"reference corpus path (default {DEFAULT_CORPUS})"
    )
    parser.add_argument("--port", type=int, default=PORT, help=f"serving port (default {PORT})")
    parser.add_argument(
        "--runtime-env",
        action="append",
        metavar="NAME=VALUE",
        help="diagnostic switch (LLMP_DIAGNOSTIC_*) for the runtime (repeatable)",
    )
    parser.add_argument(
        "--keep-data", action="store_true", help="retain run data directories after execution"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="timeout in seconds for waiting until runtime is ready (default 900.0)",
    )
    parser.add_argument(
        "--arm",
        choices=("both", "f16", "q8_0"),
        default="both",
        help="run one arm only (the other is read from its saved file when present)",
    )
    parser.add_argument(
        "--target-tokens", type=int, help="override the family's passage token budget"
    )

    args = parser.parse_args(argv)

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    share = pathlib.Path(args.share).expanduser().resolve()
    corpus_path = pathlib.Path(args.corpus).expanduser().resolve()
    if not corpus_path.exists():
        raise SystemExit(f"corpus file not found: {corpus_path}")
    corpus = load_corpus(corpus_path)

    runtime = pathlib.Path(args.runtime).resolve()
    if not runtime.exists():
        raise SystemExit(f"runtime binary not found: {runtime}")

    runtime_env = parse_runtime_env(args.runtime_env)
    arms = ("f16", "q8_0") if args.arm == "both" else (args.arm,)

    target_families: list[str]
    if args.family == "all":
        target_families = list(STANDARD_LLM_FAMILIES)
        if args.with_assistant:
            target_families = [
                "gemma26-assistant"
                if f == "gemma26"
                else "gemma31-assistant"
                if f == "gemma31"
                else f
                for f in target_families
            ]
    else:
        fam_name = args.family
        if args.with_assistant:
            if fam_name == "gemma26":
                fam_name = "gemma26-assistant"
            elif fam_name == "gemma31":
                fam_name = "gemma31-assistant"
        if fam_name not in FAMILIES:
            raise SystemExit(f"unknown family: {fam_name!r}. Choices: {sorted(FAMILIES)} or 'all'")
        target_families = [fam_name]

    print_table_header()

    has_error = False
    for fam_name in target_families:
        family = FAMILIES[fam_name]
        if args.target_tokens:
            family = dataclasses.replace(family, target_tokens=args.target_tokens)
        res = evaluate_family(
            family,
            runtime,
            share,
            corpus,
            out,
            args.port,
            runtime_env,
            args.timeout,
            args.keep_data,
            arms,
        )
        if res.get("error"):
            has_error = True

    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
