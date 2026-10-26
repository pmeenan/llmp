#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""M3.6's matched speed gate: an old and a new `llmp-runtime` on one model family.

The protocol is docs/experiments/m36-baselines/README.md; this tool is its
only implementation, so the protocol's constants live here (FAMILIES, the
schedule, the statistic and the stopping rule). Change them only with that
report, never between the runs of one comparison.

    speed_gate.py families
    speed_gate.py plan --family F --old RUNTIME --new RUNTIME --out DIR [--blocks N] [--first B]
                       [--old-config TEMPLATE] [--new-config TEMPLATE]
                       [--old-env NAME=VALUE ...] [--new-env NAME=VALUE ...]
                       [--job-runs K --steps PREFIX]
    speed_gate.py run --family F --runtime RUNTIME --label old|new --out DIR
                      (--block B --position P | --prime) [--session S]
                      [--config-template TEMPLATE] [--runtime-env NAME=VALUE ...]
    speed_gate.py quality --family F --runtime RUNTIME --label old|new --out DIR
                          [--config-template TEMPLATE] [--runtime-env NAME=VALUE ...]
    speed_gate.py verdict DIR [--family F] [--json]
    speed_gate.py spread DIR [--family F] [--json]
    speed_gate.py operating

A *run* is one fresh runtime process with a fresh data directory. For an
LLM family it serves the family's fixed configuration on loopback, and the
client drives it over the chat route: one warm-up pass of every cell, then
REPEATS measured passes, each cell C1, C2, ... in turn. A cell releases its
requests together; each is an 8K-class (Gemma3: 4K-class) prompt cut from
the corpus, distinct within the run, greedy, streamed, with `max_tokens`
generated. Per cell the client takes, from its own monotonic clock:

  prefill_s     release to the last request's first streamed token
  decode_tok_s  the sum of the requests' rates while every request decodes
                (after the last first token, before the first last token),
                each measured between its own streamed chunks in that window
                (a chunk is a step, worth the request's mean tokens a chunk);
                with one request, (tokens - 1) / its first-to-last span
  wall_tok_s    all completed tokens / (release to the last request's end)

A run's value of a metric is the mean of its measured passes (each pass
pays different prompts, the same in every run). Whole-node
MemAvailable is sampled every 20 ms from before the process starts until it
has exited; peak_gib is the largest drop. An image family's run is one
`chat --image-prompt` process: generation_s is the generation's time from
its model being ready to its pixels, whose digest must equal that arm's
predeclared output pin. Reference image quality is a separate control.

A *block* is four runs, old/new/new/old (ABBA), so linear drift cancels.
Its slowdown for a metric is ln(mean(new) / mean(old)) for a time and the
inverse for a rate, positive when new is slower. Over n blocks the gate is
non-inferiority at the margin GATE (1%): PASS when the upper bound
(mean + t(0.975, n-1) * sd / sqrt(n)) is at most ln(1 + GATE); FAIL when
the lower bound (mean - the same half-width) exceeds it; otherwise run
another block, up to the family's max_blocks, after which the metric has
NOT PASSED. Each look's bound is one-sided 97.5% because the rule looks
after every block from min_blocks on: at 95% a look, a true 1% slowdown
would pass about one comparison in ten (`operating` measures both). No
verdict is given before the family's min_blocks, chosen from the measured
same-binary (A-vs-A) spread so that a one-sided 95% bound's half-width is
expected to be at most RESOLUTION (0.5%). peak_gib is reported as an
absolute difference, not a baseline-relative acceptance gate (owner,
2026-10-13). Hardware fit and resource admission still apply. Only the first
max_blocks blocks count. A family passes when every speed metric passes and its
runs are valid: none failed or recorded a problem (token counts, cached
prompt tokens, pixels, another GPU process), each ran in a session whose
priming run succeeded before it, and every old run used one
runtime and configuration, every new run one, and the protocol's current
family constants, and the arms paid the same work: every cell's prompt indices,
width and per-request prompt_tokens equal in every run (a tokenizer difference
between the arms' artifacts); otherwise its verdict is INVALID (rerun the block: its
records are replaced). `verdict` exits 0 only when every family judged
passes, 2 while one needs more blocks, 1 otherwise.

A *session* is one supervised job's steps (one `plan` output, or with
`--job-runs` one of its step files). It starts with a priming run of the
old arm (PRIME_ARM: why that arm is there): a whole run, which must
succeed, whose record goes to primes/ and never to runs/, so no statistic
reads it. Every measured run records its session; a run without one, or
whose session's priming run is missing, failed, recorded a problem or did
not start before it, makes its family INVALID.

The node configuration an arm gets is the family's default (config_text): both
arms serve the v1 imports from `m36-v1-artifacts` (falling back to `artifacts`,
D-118). `run --config-template` gives an arm its own
instead: the whole node configuration with {share}, {data} and {port} for the
share directory, the run's data directory and its port. Use it when a newer runtime needs other keys or a
re-imported artifact for the same work; `plan --old-config/--new-config`
passes each arm its template, and the records keep its text and digest.

A runtime's diagnostic switches are environment variables named
LLMP_DIAGNOSTIC_* (never configuration): `run --runtime-env NAME=VALUE`
gives an arm's runtime one (repeatable; no other names), `plan
--old-env/--new-env` passes them on, and the records keep them. A switch
the runtime binary does not name is refused (`plan` when the binary is
there, every `run`): it would change nothing and mislabel the arm. An arm's
runs, and the priming runs of PRIME_ARM, must all have the same ones. The
gate uses one where the baseline's work differs from the new runtime's by a
known defect (LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1 for Gemma 2 and
Gemma 3; docs/experiments/m36-baselines/README.md), so both do equal work.

`spread` reads same-binary runs (old and new both the baseline; give the new
arm the old template, `--new-config`, since its default is the v1 imports) and reports
each metric's run-to-run and block-to-block spread and the blocks its bound
needs; min_blocks below come from it. It reads runs with or without a
primed session (the records min_blocks come from predate priming) and
counts those without one.

`quality` runs one process: two fixed greedy C1 replies (their digests;
`--retain-streams` additionally keeps raw SSE and client timings) and the mean negative log-likelihood of a
fixed corpus passage through the literal completions route's prompt
scores. It is a quick runtime-level control beside each family's own
oracle controls, not a replacement for them.

Ordinary speed records keep token counts, timings and SHA-256 digests only.
Quality records retain the scored passage and response; `--retain-streams`
also retains greedy replies for explicitly enrolled quality comparisons.
Run it under the Spark's GPU supervisor
(docs/workflow.md); a run that fails records its error and exits nonzero,
so a steps queue moves on.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import http.client
import json
import math
import mmap
import os
import pathlib
import random
import re
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True

GATE = 0.01  # a new implementation may be at most 1% slower (engine-components.md)
RESOLUTION = 0.005  # the bound's expected half-width when min_blocks is chosen
PORT = 18114
CORPUS_SHA256 = "55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb"
IMAGE_PROMPT = "A red ceramic teapot on a plain wooden table, soft daylight, no text."
IMAGE_NOISE = "references/qwen-image-2.1/native/ref1/latents_init.bf16"
IMAGE_NOISE_SHA256 = "eb7333893ca0409a100955e5305482bb9de8225ee3c21d33fd01fd6c16c042bd"
IMAGE_PIXELS_SHA256 = "3b7770ca720e3ae4bb0633dad36ace14e3c65beed0440dec06bde21c10e29fb7"
QWEN_TOKENIZER = "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6"
INSTRUCTION = "Rewrite the following text in full, keeping every detail and its order:"
QUALITY_PROMPTS = (
    "Explain, step by step, how a hash table handles collisions.",
    "Write a short story about a lighthouse keeper who finds a map.",
)

# One-sided Student t quantiles by degrees of freedom, 95% (min_blocks' design
# resolution) and 97.5% (each look's bound, so that looking after every block
# from min_blocks to max_blocks keeps the overall false-PASS rate near 5%;
# `speed_gate.py operating` measures it). Beyond the table, df 20's value
# (conservative; max_blocks stays far below it).
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860,
       9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753,
       16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def t95(df: int) -> float:
    return T95.get(df, T95[20]) if df >= 1 else math.inf


def t_look(df: int) -> float:
    """The quantile of every look's one-sided bound: 97.5%."""
    return T975.get(df, T975[20]) if df >= 1 else math.inf


@dataclasses.dataclass(frozen=True)
class Family:
    name: str
    kind: str  # "llm" or "image"
    model: str  # the configured model's name
    store: str  # the installed store, under the share directory
    models_toml: str  # its [models.*] tables; {share} is the share directory
    prompt_chars: int = 0  # corpus characters a prompt
    max_tokens: int = 128
    cells: tuple = (1,)
    repeats: int = 2  # measured passes a run
    min_blocks: int = 3
    max_blocks: int = 6
    score_chars: int = 4000


# The v1 imports of every family (`u2-v1-artifacts` / `artifacts`, D-118), by
# name from the pinned in-tree map the models tests read
# (tests/unit/data/v1_frozen/v1-ids.json), so a re-import changes one file.
V1_IDS = pathlib.Path(__file__).resolve().parent.parent / "tests/unit/data/v1_frozen/v1-ids.json"


def published_ids(path: pathlib.Path = V1_IDS) -> dict[str, str]:
    """Each published name's v1 ID (a composition's own ID for the image)."""
    data = json.loads(path.read_text())
    return {name: data["artifacts"][key] for key, name in data["names"].items()}


_PUBLISHED = published_ids()
DSV4_V1 = _PUBLISHED["deepseek"]
DSPARK_V1 = _PUBLISHED["dspark"]
QWEN_V1 = _PUBLISHED["qwen38-modelopt"]
QWEN_MTP_V1 = _PUBLISHED["qwen38-modelopt-mtp"]
QWEN_GGUF_V1 = _PUBLISHED["qwen38-gguf"]
GEMMA2_V1 = _PUBLISHED["gemma2"]
GEMMA3_V1 = _PUBLISHED["gemma3"]
GEMMA26_V1 = _PUBLISHED["gemma26"]
GEMMA26_ASSISTANT_V1 = _PUBLISHED["gemma26-assistant"]
GEMMA31_V1 = _PUBLISHED["gemma31"]
GEMMA31_ASSISTANT_V1 = _PUBLISHED["gemma31-assistant"]
IMAGE_V1 = _PUBLISHED["qwen-image"]
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


def _qwen_files() -> str:
    return (f'tokenizer = "{{share}}/{QWEN_TOKENIZER}/tokenizer.json"\n'
            f'chat_template = "{{share}}/{QWEN_TOKENIZER}/chat_template.jinja"\n')


def _gemma(name: str, artifact: str, context: int, chunk: int, slots: int) -> str:
    return (f'[models.{name}]\nartifact = "{artifact}"\ncontext = {context}\n'
            f"prefill_chunk = {chunk}\nmax_slots = {slots}\n")


def _qwen(artifact: str, drafter: str) -> str:
    return (f'[models."qwen3.8"]\nartifact = "{artifact}"\ndrafter = "{drafter}"\n'
            "speculation = true\ndraft_vocab = 65536\ndepth_cost_ratio = 1.16\n"
            "context = 33792\nprefill_chunk = 4096\nmax_slots = 4\n" + _qwen_files())


def _qwen_gguf(artifact: str) -> str:
    return (f'[models."qwen3.8-gguf"]\nartifact = "{artifact}"\ncontext = 33792\n'
            "prefill_chunk = 4096\nmax_slots = 4\n" + _qwen_files())


def _deepseek(artifact: str, drafter: str) -> str:
    return (f'[models.deepseek]\nartifact = "{artifact}"\ndrafter = "{drafter}"\n'
            "speculation = true\n"
            'wave_form = "plain"\ncontext = 262144\nprefill_chunk = 4096\nmax_slots = 4\n')


FAMILIES = {f.name: f for f in (
    Family("gemma2", "llm", "gemma2", V1_STORE,
           _gemma("gemma2", GEMMA2_V1, 8192, 128, 2),
           prompt_chars=26000, cells=(1, 2), repeats=6,
           min_blocks=5, max_blocks=10),
    Family("gemma3", "llm", "gemma3", V1_STORE,
           _gemma("gemma3", GEMMA3_V1, 4096, 128, 2),
           prompt_chars=11500, cells=(1, 2), repeats=6,
           min_blocks=7, max_blocks=12),
    Family("gemma26", "llm", "gemma26", V1_STORE,
           _gemma("gemma26", GEMMA26_V1, 8192, 1024, 4),
           prompt_chars=26000, cells=(1, 4), min_blocks=9, max_blocks=12),
    Family("gemma31", "llm", "gemma31", V1_STORE,
           _gemma("gemma31", GEMMA31_V1, 8192, 256, 4),
           prompt_chars=26000, cells=(1, 4), repeats=1,
           min_blocks=7, max_blocks=12),
    Family("qwen38", "llm", "qwen3.8", V1_STORE, _qwen(QWEN_V1, QWEN_MTP_V1),
           prompt_chars=30000, cells=(1, 2, 4), min_blocks=6, max_blocks=12),
    Family("qwen38-gguf", "llm", "qwen3.8-gguf", V1_STORE, _qwen_gguf(QWEN_GGUF_V1),
           prompt_chars=30000, cells=(1, 2, 4), repeats=1,
           min_blocks=3, max_blocks=6),
    Family("deepseek", "llm", "deepseek", V1_STORE, _deepseek(DSV4_V1, DSPARK_V1),
           prompt_chars=30000, cells=(1, 2, 4), repeats=1,
           min_blocks=3, max_blocks=6),
    Family("image", "image", "image", V1_STORE,
           f'[models.image]\ncomposition = "{IMAGE_V1}"\n',
           min_blocks=6, max_blocks=12),
)}

V1_MODELS = {f.name: f.models_toml for f in FAMILIES.values()}

TIME_METRICS = ("prefill_s", "generation_s", "peak_gib")  # lower is better; the rest are rates


# ---------------------------------------------------------------- inputs


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(path: pathlib.Path) -> str:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CORPUS_SHA256:
        raise SystemExit(f"{path}: not the pinned corpus ({CORPUS_SHA256[:12]}...)")
    return data.decode("utf-8")


def passage(corpus: str, index: int, chars: int) -> str:
    """The index-th passage: `chars` characters from a line start, spread over the corpus."""
    span = len(corpus) - chars - 1
    start = (index * 7919 * 211) % span
    newline = corpus.find("\n", start)
    start = newline + 1 if 0 <= newline < span else start
    return corpus[start:start + chars]


def prompt(corpus: str, index: int, chars: int) -> str:
    # The leading number makes every prompt of a run distinct from its first tokens,
    # so no request reuses another's prefix.
    return f"Request {index}. {INSTRUCTION}\n\n{passage(corpus, index, chars)}"


def config_template(family: Family, label: str = "old") -> str:
    """The family's node configuration for an arm, with {share}, {data} and {port} left to
    fill: the v1 imports (D-118)."""
    _ = label
    return ('schema_version = 2\n[storage]\ndata_dir = "{data}"\n'
            f'installed = "{{share}}/{family.store}"\n'
            '[client]\nbind = ["loopback"]\nport = {port}\n'
            + family.models_toml)


def render_config(template: str, share: pathlib.Path, data: pathlib.Path, port: int) -> str:
    for key in ("{data}", "{port}"):
        if key not in template:
            raise SystemExit(f"the configuration template has no {key}")
    resolved_v1 = resolve_store(V1_STORE, share)
    if resolved_v1 != V1_STORE:
        template = template.replace(f"{{share}}/{V1_STORE}", f"{{share}}/{resolved_v1}")
    return (template.replace("{share}", str(share)).replace("{data}", str(data))
            .replace("{port}", str(port)))


def config_text(family: Family, share: pathlib.Path, data: pathlib.Path, port: int,
                label: str = "old") -> str:
    return render_config(config_template(family, label), share, data, port)


def arm_template(family: Family, path: str | None, label: str = "old") -> str:
    """An arm's configuration template: its label's default, or the file `--config-template` names."""
    return config_template(family, label) if path is None else (
        pathlib.Path(path).expanduser().read_text())


DIAGNOSTIC_PREFIX = "LLMP_DIAGNOSTIC_"


def parse_runtime_env(pairs: list[str] | None) -> dict[str, str]:
    """An arm's diagnostic switches, NAME=VALUE each, sorted by name."""
    env: dict[str, str] = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if not sep or not name.startswith(DIAGNOSTIC_PREFIX) or name in env:
            raise SystemExit(f"--runtime-env {pair!r}: not a single {DIAGNOSTIC_PREFIX}*=VALUE")
        env[name] = value
    return dict(sorted(env.items()))


def unknown_runtime_env(runtime: pathlib.Path, runtime_env: dict[str, str]) -> list[str]:
    """The switches the runtime binary does not name. A switch the runtime never
    reads changes nothing, so an arm given one is not the arm its records claim
    (image-check02's LLMP_DIAGNOSTIC_COMPONENT_SERVING, which no source reads)."""
    if not runtime_env:
        return []
    with open(runtime, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        # The whole name as a token, not a prefix or suffix of a longer one.
        return [name for name in runtime_env
                if re.search(rb"(?<![A-Za-z0-9_])" + re.escape(name.encode()) + rb"(?![A-Za-z0-9_])",
                             data) is None]


def check_runtime_env(runtime: pathlib.Path, runtime_env: dict[str, str]) -> None:
    """Raises ValueError (a run records it as the run's error) for a switch the
    runtime does not name."""
    if unknown := unknown_runtime_env(runtime, runtime_env):
        raise ValueError(f"{runtime}: reads no diagnostic switch named {', '.join(unknown)}")


def arm_env(runtime_env: dict[str, str]) -> dict[str, str]:
    """The runtime's environment: this process's, with no diagnostic but the arm's own."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(DIAGNOSTIC_PREFIX)}
    env.update(runtime_env, CUDA_DISABLE_PTX_JIT="1")
    return env


# What makes an arm: its binary, its configuration and its diagnostic switches
# (a record before switches had none).
ARM_KEYS = ("runtime_sha256", "config_sha256", "runtime_env", "image_pixels_sha256")


def arm_identity(record: dict, key: str) -> str:
    if key == "runtime_env":
        return json.dumps(record.get(key) or {}, sort_keys=True)
    if key == "image_pixels_sha256":
        return str(record.get(key, IMAGE_PIXELS_SHA256 if record.get("family") == "image" else None))
    return str(record.get(key))


def image_pixels_pin(family: Family, label: str, value: str | None) -> str | None:
    """Freeze output repeatability per implementation, not cross-library bits."""
    if family.kind != "image":
        if value is not None:
            raise SystemExit("an image output pin requires the image family")
        return None
    pin = IMAGE_PIXELS_SHA256 if value is None else value
    if len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
        raise SystemExit("image output pin must be a lowercase SHA-256 digest")
    if label == "old" and pin != IMAGE_PIXELS_SHA256:
        raise SystemExit("the frozen old image arm must retain its baseline output pin")
    return pin


def image_output_problems(record: dict) -> list[str]:
    try:
        pin = image_pixels_pin(FAMILIES["image"], record.get("label"),
                               record.get("image_pixels_sha256"))
    except SystemExit as error:
        return [str(error)]
    if (record.get("turn") or {}).get("pixels_sha256") != pin:
        return ["pixels differ from this arm's pinned image"]
    return []


# ---------------------------------------------------------------- memory


def mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("no MemAvailable")


class MemorySampler:
    def __init__(self, period: float = 0.02):
        self.period = period
        self.start = mem_available()
        self.low = self.start
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            self.low = min(self.low, mem_available())
            self.samples += 1

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        return {"mem_available_start": self.start, "mem_available_low": self.low,
                "samples": self.samples, "peak_gib": (self.start - self.low) / (1 << 30)}


# ---------------------------------------------------------------- the service


class Service:
    def __init__(self, runtime: pathlib.Path, config: pathlib.Path, run_dir: pathlib.Path,
                 port: int, runtime_env: dict[str, str]):
        self.port = port
        if port_open(port):
            # A runtime left from a killed run would answer this run's requests.
            raise RuntimeError(f"something already listens on port {port}")
        self.log = open(run_dir / "runtime.log", "wb")
        env = arm_env(runtime_env)
        self.started = time.monotonic()
        # In the harness's own process group, so the supervisor's timeout or
        # kill (which signals the job's group) also stops the runtime.
        self.proc = subprocess.Popen(
            [str(runtime), "--config", str(config), "--anchor", str(run_dir / "enrollment")],
            stdout=self.log, stderr=subprocess.STDOUT, env=env)

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


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def stream_chat(port: int, body: dict, barrier: threading.Barrier | None,
                raw_stream: pathlib.Path | None = None) -> dict:
    """One streamed chat request: client-clock timings, usage and the reply's digest."""
    record: dict = {"error": None}
    payload = json.dumps(body).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    raw_output = None
    try:
        if raw_stream is not None:
            raw_output = raw_stream.open("xb")
        conn.connect()
        if barrier is not None:
            barrier.wait(timeout=600)
        record["t_send"] = time.monotonic()
        conn.request("POST", "/v1/chat/completions", payload,
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            failed_body = response.read()
            if raw_output is not None:
                raw_output.write(failed_body)
            record["error"] = f"HTTP {response.status}: {failed_body[:400]!r}"
            return record
        digest = hashlib.sha256()
        times: list[float] = []
        for raw in response:
            if raw_output is not None:
                raw_output.write(raw)
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
                record["error"] = "SSE error: " + json.dumps(event["error"], ensure_ascii=True)[:400]
                return record
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
    except Exception as e:  # recorded with the cell; the run goes on
        record["error"] = f"{type(e).__name__}: {e}"
        if barrier is not None:
            barrier.abort()  # the cell's other requests stop waiting for this one
    finally:
        conn.close()
        if raw_output is not None:
            raw_output.close()
            record["raw_stream"] = {"file": raw_stream.name,
                                    "sha256": sha256_file(raw_stream)}
    return record


def positive_concurrency(value: str) -> int:
    """An argparse type for a positive request concurrency."""
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def collect_quality_replies(port: int, model: str, max_tokens: int, concurrency: int,
                            retain_streams: bool, quality_dir: pathlib.Path,
                            name: str) -> tuple[list[dict], str | None]:
    """Collect fixed greedy prompts in synchronized batches, preserving prompt order.

    Any request or worker-launch error returns the batch's results and prevents later scoring.
    """
    if concurrency < 1 or concurrency > len(QUALITY_PROMPTS):
        raise ValueError(f"chat concurrency must be between 1 and {len(QUALITY_PROMPTS)}")
    replies: list[dict | None] = [None] * len(QUALITY_PROMPTS)
    for first in range(0, len(QUALITY_PROMPTS), concurrency):
        indexes = list(range(first, min(first + concurrency, len(QUALITY_PROMPTS))))
        barrier = threading.Barrier(len(indexes)) if len(indexes) > 1 else None

        def collect(index: int) -> None:
            body = {"model": model, "stream": True, "temperature": 0,
                    "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": QUALITY_PROMPTS[index]}]}
            raw_stream = (quality_dir / f"{name}-reply-{index}.sse" if retain_streams else None)
            try:
                replies[index] = stream_chat(port, body, barrier, raw_stream)
            except Exception as e:
                replies[index] = {"error": f"{type(e).__name__}: {e}"}
                if barrier is not None:
                    barrier.abort()

        launch_error = None
        started: list[threading.Thread] = []
        try:
            for index in indexes:
                try:
                    thread = threading.Thread(target=collect, args=(index,))
                    thread.start()
                    started.append(thread)
                except Exception as e:
                    replies[index] = {"error": f"{type(e).__name__}: {e}"}
                    launch_error = f"greedy request {index}: {replies[index]['error']}"
                    if barrier is not None:
                        barrier.abort()
                    break
        finally:
            for thread in started:
                thread.join()
        errors = [f"greedy request {index}: {replies[index].get('error')}"
                  for index in indexes if replies[index] and replies[index].get("error")]
        if launch_error and launch_error not in errors:
            errors.append(launch_error)
        if errors:
            return [reply for reply in replies if reply is not None], "; ".join(errors)
    return [reply for reply in replies if reply is not None], None


def decode_rate(requests: list[dict]) -> float | None:
    """Tokens a second while every request decodes: the window after the last first token
    and before the first last token. Each request's rate there is measured between its own
    chunks inside the window (one chunk a decode step, worth its mean tokens a chunk), so a
    chunk landing just inside or outside the window's edge does not count; the cohort's
    rate is their sum."""
    start = max(r["t_first"] for r in requests)
    end = min(r["t_last"] for r in requests)
    if end <= start:
        return None
    rate = 0.0
    for r in requests:
        steps = len(r["times"]) - 1
        inside = [t for t in r["times"] if start <= t <= end]
        if steps < 1 or len(inside) < 2 or inside[-1] <= inside[0]:
            return None
        per_chunk = (r["usage"]["completion_tokens"] - 1) / steps
        rate += per_chunk * (len(inside) - 1) / (inside[-1] - inside[0])
    return rate


def cell_metrics(requests: list[dict]) -> dict:
    release = min(r["t_send"] for r in requests)
    tokens = [r["usage"]["completion_tokens"] for r in requests]
    metrics = {
        "prefill_s": max(r["t_first"] for r in requests) - release,
        "wall_tok_s": sum(tokens) / (max(r["t_end"] for r in requests) - release),
    }
    rate = decode_rate(requests)
    if rate is not None:
        metrics["decode_tok_s"] = rate
    return metrics


def stored_requests(cell: dict) -> list[dict] | None:
    """A recorded cell's requests on the client's clock, as cell_metrics takes them."""
    requests = []
    for kept in cell["requests"]:
        if kept.get("error") or not kept.get("chunk_times") or kept.get("end_time") is None:
            return None
        t0 = kept["send_offset"]
        times = [t0 + t for t in kept["chunk_times"]]
        requests.append({"t_send": t0, "t_first": times[0], "t_last": times[-1],
                         "t_end": t0 + kept["end_time"], "times": times, "usage": kept["usage"]})
    return requests


def run_metrics(record: dict, family: Family) -> dict:
    """A run's metrics from its recorded passes: each cell's metric, its passes' mean.

    The verdict recomputes them from the record, so every run is judged by this code."""
    metrics = {}
    passes = [p for p in record.get("passes", []) if not p["warmup"]]
    widths = list(family.cells)
    for p in passes:
        for k in p.get("cells", {}):
            if k.startswith("C") and k[1:].isdigit():
                w = int(k[1:])
                if w not in widths:
                    widths.append(w)
    for width in sorted(widths):
        cells = [stored_requests(p["cells"][f"C{width}"]) for p in passes if f"C{width}" in p.get("cells", {})]
        found = [cell_metrics(c) for c in cells if c]
        for key in ("prefill_s", "decode_tok_s", "wall_tok_s"):
            values = [m[key] for m in found if key in m]
            if len(values) == family.repeats:
                metrics[f"C{width}.{key}"] = statistics.fmean(values)
    metrics["peak_gib"] = record["memory"]["peak_gib"]
    return metrics


def run_cell(port: int, family: Family, corpus: str, first: int, width: int) -> dict:
    barrier = threading.Barrier(width)
    results: list = [None] * width

    def one(i: int) -> None:
        body = {"model": family.model, "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": 0, "max_tokens": family.max_tokens,
                "messages": [{"role": "user",
                              "content": prompt(corpus, first + i, family.prompt_chars)}]}
        results[i] = stream_chat(port, body, barrier)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(width)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cell: dict = {"width": width, "prompts": list(range(first, first + width)),
                  "requests": []}
    problems = []
    for r in results:
        kept = {k: r.get(k) for k in ("error", "finish", "chunks", "reply_sha256", "usage")}
        if "t_send" in r:
            kept["chunk_times"] = [round(t - r["t_send"], 6) for t in r.get("times", [])]
            kept["end_time"] = round(r["t_end"] - r["t_send"], 6) if "t_end" in r else None
            kept["send_offset"] = r["t_send"]
        cell["requests"].append(kept)
        if r["error"]:
            problems.append(r["error"])
            continue
        u = r["usage"]
        if u.get("completion_tokens") != family.max_tokens:
            problems.append(f"{u.get('completion_tokens')} tokens, not {family.max_tokens}")
        if (u.get("prompt_tokens_details") or {}).get("cached_tokens"):
            problems.append(f"{u['prompt_tokens_details']['cached_tokens']} cached prompt tokens")
    if not any(r["error"] for r in results):
        cell["metrics"] = cell_metrics(results)
        cell["prompt_tokens"] = [r["usage"]["prompt_tokens"] for r in results]
        if "decode_tok_s" not in cell["metrics"]:
            problems.append("no window in which every request decodes")
    cell["problems"] = problems
    return cell


# ---------------------------------------------------------------- runs


def fresh_run_dir(out: pathlib.Path, name: str) -> pathlib.Path:
    run_dir = out / "work" / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    os.chmod(run_dir, 0o700)
    return run_dir


def host_facts() -> dict:
    facts = {"host": os.uname().nodename, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        facts["gpu_compute_apps"] = apps.splitlines() if apps else []
        facts["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        facts["gpu_error"] = str(e)
    return facts


def llm_run(family: Family, runtime: pathlib.Path, share: pathlib.Path, corpus: str,
            run_dir: pathlib.Path, port: int, template: str, runtime_env: dict[str, str],
            record: dict) -> None:
    config = run_dir / "node.toml"
    config.write_text(render_config(template, share, run_dir / "data", port))
    sampler = MemorySampler()
    service = Service(runtime, config, run_dir, port, runtime_env)
    try:
        record["ready_s"] = service.wait_ready()
        index = 0
        passes = []
        for p in range(1 + family.repeats):
            cells = {}
            for width in family.cells:
                cells[f"C{width}"] = run_cell(port, family, corpus, index, width)
                index += width
            passes.append({"warmup": p == 0, "cells": cells})
        record["passes"] = passes
    finally:
        record["stop"] = service.stop()
        record["memory"] = sampler.stop()
    record["metrics"] = run_metrics(record, family)
    record["problems"] = sorted({f"{name}: {problem}" for p in passes
                                 for name, c in p["cells"].items() for problem in c["problems"]})


def image_run(family: Family, runtime: pathlib.Path, share: pathlib.Path,
              run_dir: pathlib.Path, template: str, runtime_env: dict[str, str],
              record: dict) -> None:
    noise = share / IMAGE_NOISE
    if sha256_file(noise) != IMAGE_NOISE_SHA256:
        raise SystemExit(f"{noise}: not the pinned latents")
    config = run_dir / "node.toml"
    config.write_text(render_config(template, share, run_dir / "data", PORT))
    report = run_dir / "report.json"
    sampler = MemorySampler()
    with open(run_dir / "runtime.log", "wb") as log:
        result = subprocess.run(
            [str(runtime), "--config", str(config), "--anchor", str(run_dir / "enrollment"),
             "chat", "--image-prompt", IMAGE_PROMPT, "--image-noise", str(noise),
             "--report", str(report), "--turn", family.model, IMAGE_PROMPT],
            stdout=log, stderr=subprocess.STDOUT, env=arm_env(runtime_env))
    record["memory"] = sampler.stop()
    record["stop"] = {"returncode": result.returncode}
    if result.returncode != 0:
        raise RuntimeError(f"chat exited with {result.returncode}")
    turn = json.loads(report.read_text())["turns"][0]
    ready = turn["swap"]["ready"]
    record["turn"] = {k: turn[k] for k in ("first_step_seconds", "generation_seconds",
                                           "pixels_sha256", "peak_bytes")}
    record["turn"]["ready"] = ready
    record["metrics"] = {"generation_s": turn["generation_seconds"] - ready,
                         "peak_gib": record["memory"]["peak_gib"]}
    record["problems"] = ([] if turn["pixels_sha256"] == record["image_pixels_sha256"]
                          else [f"pixels {turn['pixels_sha256']}, not this arm's pinned image"])


def cmd_run(args: argparse.Namespace) -> int:
    family = FAMILIES[args.family]
    if getattr(args, "cells", None):
        family = dataclasses.replace(
            family, cells=tuple(int(c.strip()) for c in args.cells.split(",")))
    pixels_pin = image_pixels_pin(family, args.label, args.image_pixels_sha256)
    out = pathlib.Path(args.out).resolve()
    share = pathlib.Path(args.share).expanduser().resolve()
    runtime = pathlib.Path(args.runtime).resolve()
    if args.prime:
        # A session's priming run: a whole run, kept apart from the measured ones.
        if not args.session or args.block is not None or args.position is not None:
            raise SystemExit("run --prime takes --session and no --block or --position")
        name = f"{family.name}-prime-{args.session}"
        records = out / "primes"
        path = records / f"{args.session}.json"
    else:
        if args.block is None or args.position is None:
            raise SystemExit("run takes --block and --position (or --prime)")
        name = f"{family.name}-b{args.block:02d}-p{args.position}-{args.label}"
        records = out / "runs"
        path = records / f"{name}.json"
    records.mkdir(parents=True, exist_ok=True)
    run_dir = fresh_run_dir(out, name)
    template = arm_template(family, args.config_template, args.label)
    runtime_env = parse_runtime_env(args.runtime_env)
    record = {"format": "llmp-speed-gate/1", "family": family.name, "label": args.label,
              "block": args.block, "position": args.position,
              "prime": args.prime, "session": args.session, "started_unix": time.time(),
              "runtime": str(runtime), "runtime_sha256": sha256_file(runtime),
              "config_template": template,
              "config_sha256": hashlib.sha256(template.encode()).hexdigest(),
              "runtime_env": runtime_env,
              "image_pixels_sha256": pixels_pin,
              "family_spec": dataclasses.asdict(family), "facts": host_facts(), "error": None}
    began = time.monotonic()
    try:
        check_runtime_env(runtime, runtime_env)
        if family.kind == "llm":
            corpus = load_corpus(pathlib.Path(args.corpus).expanduser())
            llm_run(family, runtime, share, corpus, run_dir, args.port, template, runtime_env,
                    record)
        else:
            image_run(family, runtime, share, run_dir, template, runtime_env, record)
    except Exception as e:  # the record says how the run failed
        record["error"] = f"{type(e).__name__}: {e}"
    if record["facts"].get("gpu_compute_apps"):
        record.setdefault("problems", []).append(
            f"GPU compute processes before the run: {record['facts']['gpu_compute_apps']}")
    record["elapsed_s"] = time.monotonic() - began
    path.write_text(json.dumps(record, indent=1) + "\n")
    if not args.keep_data:
        shutil.rmtree(run_dir / "data", ignore_errors=True)
    summary = {k: round(v, 4) for k, v in (record.get("metrics") or {}).items()}
    print(json.dumps({"run": name, "error": record["error"],
                      "problems": record.get("problems"), "metrics": summary}), flush=True)
    return 1 if record["error"] else 0


def cmd_quality(args: argparse.Namespace) -> int:
    family = FAMILIES[args.family]
    retain_streams = getattr(args, "retain_streams", False)
    requested_concurrency = getattr(args, "chat_concurrency", 1)
    if (not isinstance(requested_concurrency, int) or isinstance(requested_concurrency, bool)
            or requested_concurrency < 1 or requested_concurrency > len(QUALITY_PROMPTS)):
        raise SystemExit(f"chat concurrency must be between 1 and {len(QUALITY_PROMPTS)}")
    effective_concurrency = min(requested_concurrency, len(QUALITY_PROMPTS))
    if family.kind != "llm":
        raise SystemExit(f"{family.name}: quality covers LLM families")
    out = pathlib.Path(args.out).resolve()
    share = pathlib.Path(args.share).expanduser().resolve()
    runtime = pathlib.Path(args.runtime).resolve()
    corpus = load_corpus(pathlib.Path(args.corpus).expanduser())
    try:
        check_runtime_env(runtime, parse_runtime_env(args.runtime_env))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    name = f"{family.name}-quality-{args.label}"
    (out / "quality").mkdir(parents=True, exist_ok=True)
    run_dir = fresh_run_dir(out, name)
    config = run_dir / "node.toml"
    template = arm_template(family, args.config_template, args.label)
    runtime_env = parse_runtime_env(args.runtime_env)
    config.write_text(render_config(template, share, run_dir / "data", args.port))
    record = {"format": "llmp-speed-gate-quality/1", "family": family.name, "label": args.label,
              "runtime": str(runtime), "runtime_sha256": sha256_file(runtime),
              "config_template": template,
              "config_sha256": hashlib.sha256(template.encode()).hexdigest(),
              "runtime_env": runtime_env,
              "facts": host_facts(), "error": None, "replies": [],
              "chat_concurrency": {"requested": requested_concurrency,
                                   "effective": effective_concurrency}}
    service = None
    try:
        service = Service(runtime, config, run_dir, args.port, runtime_env)
        service.wait_ready()
        replies, error = collect_quality_replies(
            args.port, family.model, family.max_tokens, effective_concurrency,
            retain_streams, out / "quality", name)
        record["replies"] = [r if retain_streams else
                             {k: r.get(k) for k in ("error", "finish", "reply_sha256", "usage")}
                             for r in replies]
        if error:
            raise RuntimeError(error)
        text = passage(corpus, 100003, family.score_chars)
        conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=3600)
        conn.request("POST", "/v1/completions", json.dumps(
            {"model": family.model, "prompt": text, "max_tokens": 0, "prompt_logprobs": 0,
             "echo": True, "logprobs": 0, "return_tokens_as_token_ids": True}),
            {"Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"scoring: HTTP {response.status}: {body[:400]!r}")
        # Keep the actual scored IDs, including the unscored first token, and
        # every target loss for reference pairing without another native run.
        prompt_name, response_name = f"{name}-score.txt", f"{name}-score.json"
        (out / "quality" / prompt_name).write_text(text)
        (out / "quality" / response_name).write_bytes(body)
        record["score_prompt"] = {"file": prompt_name,
                                  "sha256": hashlib.sha256(text.encode()).hexdigest()}
        record["score_response"] = {"file": response_name,
                                    "sha256": hashlib.sha256(body).hexdigest()}
        rows = json.loads(body)["choices"][0]["prompt_logprobs"]
        scores = [next(iter(row.values()))["logprob"] for row in rows if row]
        record["score"] = {"tokens": len(scores), "nll": -sum(scores) / len(scores),
                           "scores_sha256": hashlib.sha256(
                               json.dumps(scores).encode()).hexdigest()}
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        if service is not None:
            record["stop"] = service.stop()
    (out / "quality" / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
    shutil.rmtree(run_dir / "data", ignore_errors=True)
    print(json.dumps({"quality": name, "error": record["error"], "score": record.get("score"),
                      "replies": [r.get("reply_sha256") for r in record["replies"]]}), flush=True)
    return 1 if record["error"] else 0


# ---------------------------------------------------------------- the schedule


ORDER = ("old", "new", "new", "old")
# The arm of every session's priming run. A first-run penalty (cold page
# cache, GPU clocks, a fresh build and install) is mostly shared: either
# arm's priming run pays it. What one arm's run leaves cold for the other
# (its binary, an artifact of its own configuration) then falls on the
# other arm's first run. Priming with the old arm puts that residue on the
# new arm (position 1), against it, as a non-inferiority gate should err;
# priming with the new arm would put it on position 0's old run, in the
# new arm's favour; alternating by session leaves it to the sessions'
# count and order.
PRIME_ARM = "old"


def cmd_plan(args: argparse.Namespace) -> int:
    family = FAMILIES[args.family]
    pixels_pin = image_pixels_pin(family, "new", args.new_image_pixels_sha256)
    me = pathlib.Path(__file__).resolve()
    blocks = args.blocks or family.min_blocks
    runs = [(block, position, label) for block in range(args.first, args.first + blocks)
            for position, label in enumerate(ORDER)]
    size = args.job_runs or len(runs)
    if size < 1:
        raise SystemExit("plan: --job-runs is at least 1")
    sessions = [runs[i:i + size] for i in range(0, len(runs), size)]
    if len(sessions) > 1 and not args.steps:
        raise SystemExit("plan: more than one job's steps; write them with --steps PREFIX")

    def command(label: str, which: str) -> str:
        runtime = args.old if label == "old" else args.new
        template = args.old_config if label == "old" else args.new_config
        env = parse_runtime_env(args.old_env if label == "old" else args.new_env)
        if pathlib.Path(runtime).is_file():
            try:
                check_runtime_env(pathlib.Path(runtime), env)
            except ValueError as e:
                raise SystemExit(str(e)) from None
        return (f"python3 -B {me} run --family {family.name} --runtime {runtime} "
                f"--label {label} --out {args.out} {which}"
                + (f" --corpus {args.corpus}" if family.kind == "llm" else "")
                + (f" --config-template {template}" if template else "")
                + (f" --image-pixels-sha256 {pixels_pin}"
                   if label == "new" and args.new_image_pixels_sha256 is not None else "")
                + "".join(f" --runtime-env {k}={v}" for k, v in env.items()))

    texts = []
    for mine in sessions:
        block, position, _ = mine[0]
        session = f"{family.name}-b{block:02d}p{position}-{secrets.token_hex(4)}"
        lines = [command(PRIME_ARM, f"--prime --session {session}")]
        lines += [command(label, f"--block {b} --position {p} --session {session}")
                  for b, p, label in mine]
        texts.append("".join(f"{line}\n" for line in lines))
    if not args.steps:
        sys.stdout.write(texts[0])
        return 0
    for n, text in enumerate(texts, start=1):
        path = pathlib.Path(f"{args.steps}-{n:02d}.steps")
        path.write_text(text)
        print(path)
    return 0


# ---------------------------------------------------------------- the verdict


def load_primes(out: pathlib.Path) -> dict:
    """Sessions' priming runs: {session: record}."""
    primes = {}
    for path in sorted((out / "primes").glob("*.json")):
        r = json.loads(path.read_text())
        primes[r.get("session") or path.stem] = r
    return primes


def load_runs(out: pathlib.Path, family: str | None) -> dict:
    """Complete ABBA blocks: {(family, block): {position: record}}."""
    blocks: dict = {}
    for path in sorted((out / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        if family and r["family"] != family:
            continue
        if r.get("passes") and not r["error"]:
            r["metrics"] = run_metrics(r, FAMILIES[r["family"]])
        blocks.setdefault((r["family"], r["block"]), {})[r["position"]] = r
    return blocks


def block_slowdowns(block: dict, metric: str) -> float | None:
    if sorted(block) != [0, 1, 2, 3]:
        return None
    if any(block[p]["error"] or block[p]["label"] != ORDER[p] or
           metric not in block[p]["metrics"] for p in range(4)):
        return None
    old = statistics.fmean(block[p]["metrics"][metric] for p in (0, 3))
    new = statistics.fmean(block[p]["metrics"][metric] for p in (1, 2))
    if metric == "peak_gib":
        return new - old  # absolute GiB
    ratio = new / old if metric.split(".")[-1] in TIME_METRICS else old / new
    return math.log(ratio)


def judge(values: list[float], margin: float, min_blocks: int, max_blocks: int,
          quantile=t_look) -> dict:
    n = len(values)
    mean = statistics.fmean(values) if values else math.nan
    sd = statistics.stdev(values) if n >= 2 else math.nan
    half = quantile(n - 1) * sd / math.sqrt(n) if n >= 2 else math.inf
    upper, lower = mean + half, mean - half
    if n < max(2, min_blocks):
        verdict = "INCOMPLETE"
    elif upper <= margin:
        verdict = "PASS"
    elif lower > margin:
        verdict = "FAIL"
    elif n < max_blocks:
        verdict = "CONTINUE"
    else:
        verdict = "NOT PASSED"
    return {"blocks": n, "mean": mean, "sd": sd, "half_width": half, "upper": upper,
            "lower": lower, "verdict": verdict}


def run_spec(spec: dict) -> dict:
    return {k: v for k, v in json.loads(json.dumps(spec)).items()
            if k not in ("min_blocks", "max_blocks", "cells")}


def workload_identity(record: dict) -> dict:
    """What workload a run's cells actually paid, as recorded: per pass and cell, the prompts
    drawn, the width and each request's prompt token count. Two arms given the same prompts
    but tokenizing them differently (a v1 against a v0 artifact) did different work."""
    identity = {}
    for index, p in enumerate(record.get("passes") or []):
        for name, cell in sorted((p.get("cells") or {}).items()):
            tokens = cell.get("prompt_tokens")
            if tokens is None:
                tokens = [(r.get("usage") or {}).get("prompt_tokens")
                          for r in cell.get("requests") or []]
                if not tokens or any(t is None for t in tokens):
                    continue
            identity[f"pass {index}{' (warm-up)' if p.get('warmup') else ''} {name}"] = {
                "prompts": cell.get("prompts"), "width": cell.get("width"),
                "prompt_tokens": tokens}
    return identity


def workload_problems(runs: list[dict]) -> list[str]:
    """Cells whose workload identity differs between runs, old against new first. The first
    old run is the reference; any other run that differs from it is one more mismatch."""
    reference = next((r for r in runs if r["label"] == "old" and workload_identity(r)), None)
    if reference is None:
        return []
    base = workload_identity(reference)
    reasons = []
    for r in runs:
        if r is reference:
            continue
        here = workload_identity(r)
        if not here:
            continue
        for cell, mine in sorted(here.items()):
            theirs = base.get(cell)
            if theirs is None:
                continue
            for field in ("prompts", "width", "prompt_tokens"):
                if mine[field] != theirs[field]:
                    reasons.append(
                        f"{r['label']} block {r.get('block')} position {r.get('position')} "
                        f"{cell}: {field} {mine[field]} differ from the {reference['label']} "
                        f"reference's {theirs[field]} (block {reference.get('block')} position "
                        f"{reference.get('position')}): the arms did different work "
                        "(a tokenizer or prompt difference)")
    return reasons


def validity(family: Family, mine: list[dict], primes: dict) -> list[str]:
    """Why a family's counted runs cannot be judged, if they cannot: a failed or
    incomplete run, a recorded problem, a run outside a session whose priming run
    succeeded before it, or arms whose runs differ in binary, configuration or
    protocol constants."""
    reasons = []
    sessions: dict = {}
    # What a run does (the stopping rule's block counts are the verdict's own).
    spec = run_spec(dataclasses.asdict(family))
    for b in mine:
        block = next(iter(b.values()))["block"]
        missing = [p for p in range(4) if p not in b]
        if missing:
            reasons.append(f"block {block}: no record at positions {missing}")
        for p, r in sorted(b.items()):
            where = f"block {block} position {p}"
            if r["label"] != ORDER[p]:
                reasons.append(f"{where}: labelled {r['label']}, not {ORDER[p]}")
            if r["error"]:
                reasons.append(f"{where}: {r['error']}")
            reasons += [f"{where}: {problem}" for problem in r.get("problems") or []]
            if family.kind == "image":
                reasons += [f"{where}: {problem}" for problem in image_output_problems(r)]
            if run_spec(r.get("family_spec") or {}) != spec:
                reasons.append(f"{where}: other protocol constants than the current {family.name}")
            if r.get("session"):
                sessions.setdefault(r["session"], []).append((where, r))
            else:
                reasons.append(f"{where}: no session, so no priming run")
    for session, members in sorted(sessions.items()):
        prime = primes.get(session)
        if prime is None:
            reasons.append(f"session {session}: no priming run recorded")
            continue
        if prime.get("family") != family.name or not prime.get("prime"):
            reasons.append(f"session {session}: its priming record is not {family.name}'s")
        # PRIME_ARM's own binary and configuration, as its measured runs have.
        arm = [r for b in mine for r in b.values() if r["label"] == PRIME_ARM]
        if prime.get("label") != PRIME_ARM or any(
                arm_identity(prime, key) != arm_identity(r, key)
                for r in arm for key in ARM_KEYS):
            reasons.append(f"session {session}: its priming run is not the {PRIME_ARM} arm's "
                           "binary and configuration")
        if prime.get("error"):
            reasons.append(f"session {session}: priming run: {prime['error']}")
        reasons += [f"session {session}: priming run: {problem}"
                    for problem in prime.get("problems") or []]
        if family.kind == "image":
            reasons += [f"session {session}: priming run: {problem}"
                        for problem in image_output_problems(prime)]
        began = prime.get("started_unix")
        for where, r in members:
            if began is None or r.get("started_unix") is None or r["started_unix"] < began:
                reasons.append(f"session {session}: {where} did not start after its priming run")
    # The same prompts must tokenize to the same counts in both arms (and within one).
    reasons += workload_problems([r for b in mine for _, r in sorted(b.items())
                                  if not r["error"]])
    for label in ("old", "new"):
        runs = [r for b in mine for r in b.values() if r["label"] == label]
        for key, what in zip(ARM_KEYS, ("runtimes", "configurations", "diagnostic switch sets",
                                      "image output pins")):
            seen = sorted({arm_identity(r, key) for r in runs})
            if len(seen) > 1:
                reasons.append(f"{label} runs used {len(seen)} {what}: "
                               + ", ".join(x[:12] for x in seen))
    return reasons


def verdicts(out: pathlib.Path, family_name: str | None) -> dict:
    blocks = load_runs(out, family_name)
    primes = load_primes(out)
    report = {}
    for name in sorted({f for f, _ in blocks}):
        family = FAMILIES[name]
        numbers = sorted(k[1] for k in blocks if k[0] == name)
        # Only the first max_blocks blocks count: more would be another look.
        mine = [blocks[(name, n)] for n in numbers[:family.max_blocks]]
        ignored = numbers[family.max_blocks:]
        metrics = sorted({m for b in mine for r in b.values() for m in (r.get("metrics") or {})})
        rows = {}
        for metric in metrics:
            values = [v for v in (block_slowdowns(b, metric) for b in mine) if v is not None]
            memory = metric == "peak_gib"
            margin = 0.0 if memory else math.log1p(GATE)
            row = judge(values, margin, family.min_blocks, family.max_blocks)
            if memory:
                # Keep paired estimates/bounds, but more memory is allowed when
                # the candidate fits and its quality/performance benefit warrants it.
                row["verdict"] = "INFORMATIONAL"
            runs = {label: [r["metrics"][metric] for b in mine for r in b.values()
                            if not r["error"] and r["label"] == label and metric in r["metrics"]]
                    for label in ("old", "new")}
            row["old_mean"] = statistics.fmean(runs["old"]) if runs["old"] else None
            row["new_mean"] = statistics.fmean(runs["new"]) if runs["new"] else None
            row["runs"] = {k: len(v) for k, v in runs.items()}
            row["unit"] = "GiB" if memory else "slowdown"
            rows[metric] = row
        invalid = validity(family, mine, primes)
        states = {row["verdict"] for metric, row in rows.items() if metric != "peak_gib"}
        overall = ("INVALID" if invalid else "INCOMPLETE" if not states
                   else "FAIL" if "FAIL" in states else "NOT PASSED" if "NOT PASSED" in states
                   else "INCOMPLETE" if "INCOMPLETE" in states
                   else "CONTINUE" if "CONTINUE" in states else "PASS")
        report[name] = {"verdict": overall, "metrics": rows, "invalid": invalid,
                        "ignored_blocks": ignored,
                        "runtimes": {label: sorted({str(r.get("runtime_sha256")) for b in mine
                                                    for r in b.values() if r["label"] == label})
                                     for label in ("old", "new")},
                        "diagnostics": {label: sorted({arm_identity(r, "runtime_env")
                                                       for b in mine for r in b.values()
                                                       if r["label"] == label})
                                        for label in ("old", "new")},
                        "problems": sorted({p for b in mine for r in b.values()
                                            for p in (r.get("problems") or [])}),
                        "errors": sorted({r["error"] for b in mine for r in b.values()
                                          if r["error"]})}
    return report


def blocks_needed(sd: float, cap: int = 12) -> int | None:
    """The fewest blocks (at least 3) whose one-sided 95% bound half-width is expected within
    RESOLUTION: the basis of min_blocks (the look bound, at 97.5%, is wider)."""
    for n in range(3, cap + 1):
        if t95(n - 1) * sd / math.sqrt(n) <= math.log1p(RESOLUTION):
            return n
    return None


def spread(out: pathlib.Path, family_name: str | None) -> dict:
    """Same-binary (A-vs-A) spread: run-to-run and block-to-block, a metric of a family.

    run_sd: the standard deviation of a run's log value about its block's mean
    (pooled; drift inside a block counts against it). For ABBA blocks of
    independent runs a block's slowdown has the same standard deviation, so
    the expected sd of block slowdowns is max(run_sd, the observed one).

    It reads runs with or without a primed session (the min_blocks records
    predate priming); `unprimed` counts the runs outside a session whose
    priming run succeeded, whose first-run effects the spread then includes.
    """
    blocks = load_runs(out, family_name)
    primes = load_primes(out)

    def primed(r: dict) -> bool:
        prime = primes.get(r.get("session") or "")
        return (prime is not None and not prime.get("error") and not prime.get("problems")
                and prime.get("family") == r.get("family"))

    report: dict = {}
    for name in sorted({f for f, _ in blocks}):
        mine = [blocks[k] for k in sorted(blocks) if k[0] == name]
        unprimed = sum(not primed(r) for b in mine for r in b.values())
        metrics = sorted({m for b in mine for r in b.values() for m in (r.get("metrics") or {})})
        rows = {}
        for metric in metrics:
            memory = metric == "peak_gib"
            residuals, groups, values = [], 0, []
            for b in mine:
                xs = [r["metrics"][metric] for r in b.values()
                      if not r["error"] and metric in r["metrics"]]
                xs = xs if memory else [math.log(x) for x in xs if x > 0]
                if len(xs) >= 2:
                    mean = statistics.fmean(xs)
                    residuals += [x - mean for x in xs]
                    groups += 1
                    values += xs
            df = len(residuals) - groups
            run_sd = math.sqrt(sum(r * r for r in residuals) / df) if df > 0 else math.nan
            slow = [v for v in (block_slowdowns(b, metric) for b in mine) if v is not None]
            block_sd = statistics.stdev(slow) if len(slow) >= 2 else math.nan
            expected = max(x for x in (run_sd, block_sd) if not math.isnan(x)) if (
                not math.isnan(run_sd) or not math.isnan(block_sd)) else math.nan
            rows[metric] = {
                "runs": len(values), "blocks": len(slow), "run_sd": run_sd, "block_sd": block_sd,
                "block_mean": statistics.fmean(slow) if slow else math.nan,
                "range": (max(values) - min(values)) if values else math.nan,
                "mean": (statistics.fmean(values) if memory else
                         math.exp(statistics.fmean(values))) if values else math.nan,
                "blocks_needed": None if memory or math.isnan(expected) else
                blocks_needed(expected),
                "unprimed": unprimed,
            }
        report[name] = rows
    return report


def cmd_spread(args: argparse.Namespace) -> int:
    report = spread(pathlib.Path(args.out), args.family)
    if args.json:
        print(json.dumps(report, indent=1))
        return 0
    for name, rows in report.items():
        unprimed = next(iter(rows.values()))["unprimed"] if rows else 0
        print(name + (f" ({unprimed} runs outside a primed session)" if unprimed else ""))
        for metric, row in rows.items():
            if metric == "peak_gib":
                print(f"  {metric:18} runs {row['runs']:2}: mean {row['mean']:.3f} GiB, "
                      f"run sd {row['run_sd']:.3f} GiB, range {row['range']:.3f} GiB, "
                      f"block mean {row['block_mean']:+.3f} GiB")
                continue
            print(f"  {metric:18} runs {row['runs']:2}: geomean {row['mean']:.4f}, "
                  f"run sd {100 * row['run_sd']:.3f}%, block sd {100 * row['block_sd']:.3f}% "
                  f"over {row['blocks']} blocks (mean {pct(row['block_mean'])}); "
                  f"blocks needed {row['blocks_needed']}")
    return 0


def pct(x: float) -> str:
    return "n/a" if x is None or math.isnan(x) or math.isinf(x) else f"{100 * math.expm1(x):+.2f}%"


def verdict_status(report: dict) -> int:
    """0 when every family judged passes; 2 while one needs more blocks; 1 otherwise
    (FAIL, NOT PASSED, INVALID, or nothing to judge)."""
    states = {fam["verdict"] for fam in report.values()}
    if not states or states & {"FAIL", "NOT PASSED", "INVALID"}:
        return 1
    return 0 if states == {"PASS"} else 2


def cmd_verdict(args: argparse.Namespace) -> int:
    report = verdicts(pathlib.Path(args.out), args.family)
    if args.json:
        print(json.dumps(report, indent=1))
        return verdict_status(report)
    if not report:
        print("no runs to judge")
    for name, fam in report.items():
        print(f"{name}: {fam['verdict']}")
        for label, hashes in fam["runtimes"].items():
            print(f"  {label} runtime {', '.join(h[:12] for h in hashes)}")
        for label, sets in fam["diagnostics"].items():
            if sets != ["{}"]:
                print(f"  {label} diagnostics {'; '.join(sets)}")
        for reason in fam["invalid"]:
            print(f"  invalid: {reason}")
        if fam["ignored_blocks"]:
            print(f"  ignored blocks past max_blocks: {fam['ignored_blocks']}")
        for metric, row in fam["metrics"].items():
            if row["unit"] == "GiB":
                show = (f"{row['mean']:+.3f} GiB, upper {row['upper']:+.3f}"
                        if row["blocks"] else "n/a")
            else:
                show = f"{pct(row['mean'])}, upper {pct(row['upper'])}, sd {pct(row['sd'])}"
            old = "n/a" if row["old_mean"] is None else f"{row['old_mean']:.4f}"
            new = "n/a" if row["new_mean"] is None else f"{row['new_mean']:.4f}"
            print(f"  {metric:18} {row['verdict']:10} blocks {row['blocks']}: {show}; "
                  f"old {old}, new {new}")
        for p in fam["problems"]:
            print(f"  problem: {p}")
        for e in fam["errors"]:
            print(f"  error: {e}")
    return verdict_status(report)


def operating(min_blocks: int, max_blocks: int, slowdown: float, sd: float,
              quantile=t_look, trials: int = 4000, seed: int = 1) -> dict:
    """The stopping rule's behaviour by simulation: block slowdowns drawn normal with
    this mean and sd, judged after every block as `verdict` judges them, until PASS,
    FAIL or max_blocks. Returns each outcome's share and the mean blocks run."""
    rng = random.Random(seed)
    margin = math.log1p(GATE)
    outcomes: dict = {}
    blocks = 0
    for _ in range(trials):
        values = [rng.gauss(slowdown, sd) for _ in range(max_blocks)]
        for n in range(min_blocks, max_blocks + 1):
            verdict = judge(values[:n], margin, min_blocks, max_blocks, quantile)["verdict"]
            if verdict != "CONTINUE":
                break
        outcomes[verdict] = outcomes.get(verdict, 0) + 1
        blocks += n
    return {**{k: v / trials for k, v in sorted(outcomes.items())}, "blocks": blocks / trials}


def cmd_operating(_: argparse.Namespace) -> int:
    """For each family's block range: how often a true 1% slowdown passes (at the
    look bound, and at 95% a look for comparison), and how often a true 0 passes
    when the block sd is the one min_blocks was chosen for (0.5% half-width at 95%)."""
    margin = math.log1p(GATE)
    for f in FAMILIES.values():
        design_sd = math.log1p(RESOLUTION) * math.sqrt(f.min_blocks) / t95(f.min_blocks - 1)
        edge = operating(f.min_blocks, f.max_blocks, margin, design_sd)
        edge95 = operating(f.min_blocks, f.max_blocks, margin, design_sd, t95)
        zero = operating(f.min_blocks, f.max_blocks, 0.0, design_sd)
        print(f"{f.name:12} blocks {f.min_blocks:2}-{f.max_blocks:2}: true +1.00% passes "
              f"{100 * edge.get('PASS', 0):4.1f}% (95% a look: {100 * edge95.get('PASS', 0):4.1f}%); "
              f"true 0 at sd {100 * design_sd:.2f}% passes {100 * zero.get('PASS', 0):5.1f}% "
              f"in {zero['blocks']:.2f} blocks")
    return 0


def cmd_families(_: argparse.Namespace) -> int:
    for f in FAMILIES.values():
        cells = ",".join(f"C{c}" for c in f.cells) if f.kind == "llm" else "generation"
        print(f"{f.name:12} {f.kind:5} {cells:10} blocks {f.min_blocks}-{f.max_blocks}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("families").set_defaults(func=cmd_families)
    sub.add_parser("operating", help="the stopping rule's error rates, simulated").set_defaults(
        func=cmd_operating)
    for command, func in (("run", cmd_run), ("quality", cmd_quality)):
        p = sub.add_parser(command)
        p.add_argument("--family", required=True, choices=sorted(FAMILIES))
        p.add_argument("--runtime", required=True)
        p.add_argument("--label", required=True, choices=("old", "new"))
        p.add_argument("--out", required=True)
        p.add_argument("--share", default="~/.local/share/llmp")
        p.add_argument("--corpus", default="~/.local/share/llmp/m36-baselines/inputs/"
                       "corpus-1facc957.md")
        p.add_argument("--port", type=int, default=PORT)
        p.add_argument("--config-template",
                       help="this arm's node configuration, with {share}, {data} and {port} "
                       "(default: its --label's: old v0 artifacts, new v1 imports)")
        p.add_argument("--runtime-env", action="append", metavar="NAME=VALUE",
                       help="a diagnostic switch (LLMP_DIAGNOSTIC_*) for this arm's runtime")
        p.set_defaults(func=func)
        if command == "quality":
            p.add_argument("--retain-streams", action="store_true",
                           help="retain raw greedy SSE and full client timing records")
            p.add_argument("--chat-concurrency", type=positive_concurrency, default=1,
                           help="simultaneous greedy quality chat requests (default: 1; any error "
                           "fails quality)")
        if command == "run":
            p.add_argument("--image-pixels-sha256",
                           help="this image arm's output pin, frozen before timing; quality is separate")
            p.add_argument("--block", type=int)
            p.add_argument("--position", type=int, choices=range(4))
            p.add_argument("--prime", action="store_true",
                           help="the session's priming run, kept out of every statistic")
            p.add_argument("--session", help="the session (job) the run belongs to")
            p.add_argument("--cells", help="comma-separated cell widths (e.g. 1,2,4)")
            p.add_argument("--keep-data", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("--family", required=True, choices=sorted(FAMILIES))
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--new-image-pixels-sha256",
                   help="qualified candidate image output pin, frozen before timing")
    p.add_argument("--out", required=True)
    p.add_argument("--blocks", type=int)
    p.add_argument("--old-config", help="the old arm's --config-template")
    p.add_argument("--new-config", help="the new arm's --config-template")
    p.add_argument("--old-env", action="append", metavar="NAME=VALUE",
                   help="the old arm's --runtime-env (repeatable)")
    p.add_argument("--new-env", action="append", metavar="NAME=VALUE",
                   help="the new arm's --runtime-env (repeatable)")
    p.add_argument("--first", type=int, default=1)
    p.add_argument("--job-runs", type=int,
                   help="measured runs a job (session) at most, each job after its priming run")
    p.add_argument("--steps", help="write each job's steps to PREFIX-NN.steps")
    p.add_argument("--corpus", default="~/.local/share/llmp/m36-baselines/inputs/"
                   "corpus-1facc957.md")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("verdict")
    p.add_argument("out")
    p.add_argument("--family", choices=sorted(FAMILIES))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verdict)
    p = sub.add_parser("spread", help="same-binary spread and the blocks it needs")
    p.add_argument("out")
    p.add_argument("--family", choices=sorted(FAMILIES))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_spread)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
