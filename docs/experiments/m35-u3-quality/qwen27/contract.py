#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Strict identities and row pairing for the prospective Qwen27 source adapter."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

IMAGE = "sha256:1010df9519efb5fb2ca8a09c1504e95829206e4fff8c8e8f3b671b4d71a6737a"
SOURCE = "cb2ebf0540f42604e2759b2ddef497861e928248"
REVISION = "70ae7fac63274ff2eac54152031433374cb80f2f"
MANIFEST = "e64f621c146c3b14ad7023eab05a476acaa663f1d9ba97a57a7768986d4d6a66"
CORPUS = "55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb"
METADATA = {
    "config.json": "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}
PHASES = {"short": (1024, 1, 1023, 0), "depth": (9216, 8192, 1024, 1)}
RECIPE = {"context": 12288, "prefill_rows": 512, "head_rows": 16,
          "source_prefill_fp8": False, "source_kv": "bfloat16", "native_kv": "float16",
          "streams": 1, "tp": 1, "max_rows": 16, "draft": False, "behavior_max_tokens": 512,
          "enable_thinking": False}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def metadata(checkpoint: Path) -> None:
    for name, expected in METADATA.items():
        if sha((checkpoint / name).read_bytes()) != expected:
            raise ValueError(f"checkpoint metadata identity changed: {name}")


def valid_ids(ids, vocab: int, *, count: int | None = None) -> None:
    if not isinstance(ids, list) or len(ids) < 2 or (count is not None and len(ids) != count):
        raise ValueError("empty/short or wrong-sized token sequence")
    if type(vocab) is not int or vocab <= 0 or any(type(i) is not int or not 0 <= i < vocab for i in ids):
        raise ValueError("token outside the source vocabulary")


def select_cases(phase, name, behavior):
    if phase == 'behavior':
        cases = [c for c in behavior['cases'] if c['id'] == name]
        if len(cases) != 1:
            raise ValueError('select exactly one frozen behavior case')
        return cases
    if name is not None:
        raise ValueError('case is only valid for behavior')
    return []


def frozen(protocol: Path, inputs: Path, prereg_commit: str):
    raw = protocol.read_bytes()
    spec = json.loads(raw)
    if spec.get("status") != "frozen" or spec.get("format") != "llmp-u3-qwen27-prereg/1":
        raise ValueError("protocol is not frozen; inputs-only preparation must precede preregistration")
    if not re.fullmatch(r"[0-9a-f]{40}", prereg_commit):
        raise ValueError("record the actual reviewed preregistration commit")
    for key in ("ids_sha256", "behavior_sha256", "native_artifact", "native_runtime_sha256",
                "native_manifest_sha256", "native_spec_sha256"):
        if not digest(spec.get(key)):
            raise ValueError(f"unfrozen identity: {key}")
    if spec.get("image") != IMAGE or spec.get("source") != SOURCE or spec.get("recipe") != RECIPE:
        raise ValueError("reference pin or recipe differs from the reviewed collector")
    if spec.get("bounds") != {"cross_ppl_increase_pct": 1.0, "own_abs_ppl_delta_pct": 0.1}:
        raise ValueError("quality bounds differ from the reviewed protocol")
    if spec.get("revision") != REVISION or spec.get("archive_manifest_sha256") != MANIFEST:
        raise ValueError("checkpoint identity differs")
    phases = {k: {"input_ids": v[0], "first_scored_position": v[1], "targets": v[2], "document": v[3]}
              for k, v in PHASES.items()}
    if spec.get("phases") != phases:
        raise ValueError("phase geometry differs")
    data = (inputs / "ids.json").read_bytes()
    if sha(data) != spec["ids_sha256"]:
        raise ValueError("corpus token IDs changed")
    ids = json.loads(data)
    valid_ids(ids, spec["vocab_size"], count=9216)
    behavior = (inputs / "behavior.json").read_bytes()
    if sha(behavior) != spec["behavior_sha256"]:
        raise ValueError("behavior token IDs changed")
    pack = json.loads(behavior)
    if (pack.get("format") != "llmp-u3-qwen27-behavior/1" or pack.get("max_tokens") != 512
            or pack.get("enable_thinking") is not False
            or [c.get("id") for c in pack.get("cases", [])] !=
                ["capital", "arithmetic", "sort", "structured", "context", "fibonacci"]):
        raise ValueError("behavior geometry differs")
    for case in pack["cases"]:
        valid_ids(case.get("ids"), spec["vocab_size"])
        if len(case["ids"]) + 512 > RECIPE["context"]:
            raise ValueError("behavior exceeds context")
    return spec, ids, pack, sha(raw)


def row_groups(ids: list[int], first: int, spans: list[tuple[int, int]], head_rows: int = 16):
    """Each source row at absolute a predicts ids[a+1], including chunk boundaries."""
    if not 1 <= first < len(ids) or head_rows < 1:
        raise ValueError("invalid score interval or projection batch")
    previous = 0
    result = []
    for a, b in spans:
        if type(a) is not int or type(b) is not int or a != previous or not a < b <= len(ids) - 1:
            raise ValueError("chunks must partition the entire teacher prefix exactly")
        groups = []
        start = max(a, first - 1)
        for row in range(start, b, head_rows):
            end = min(row + head_rows, b)
            groups.append((row - a, end - a, list(range(row + 1, end + 1)), ids[row + 1:end + 1]))
        result.append((a, b, groups))
        previous = b
    if previous != len(ids) - 1:
        raise ValueError("missing prefix rows or final target")
    return result


class ActualTargets:
    """The pinned capture API calls add; no alternative tokens or label-only denominator."""
    top = 0
    labels = None

    def __init__(self, ids: list[int], first: int, document: int):
        if not 1 <= first < len(ids):
            raise ValueError("empty target interval")
        self.ids, self.next, self.first, self.document = ids, first, first, document
        self.records = []

    def add(self, positions, tokens, chosen, alternatives, scores):
        lengths = [len(x) for x in (positions, tokens, chosen, alternatives, scores)]
        if not lengths[0] or len(set(lengths)) != 1:
            raise ValueError("capture row count mismatch or empty capture")
        for pos, token, value, alt, score in zip(positions, tokens, chosen, alternatives, scores):
            if type(pos) is not int or pos != self.next or pos >= len(self.ids):
                raise ValueError("missing, reordered, duplicate or extra position")
            if type(token) is not int or token != self.ids[pos]:
                raise ValueError("capture did not score the actual teacher target")
            if type(value) not in (int, float) or not math.isfinite(value) or value > 0:
                raise ValueError("nonfinite, positive or invalid log probability")
            if alt or score:
                raise ValueError("unexpected alternative-token reporting")
            self.records.append({"document": self.document, "position": pos, "target": token, "nll": -value})
            self.next += 1

    def finish(self):
        if self.next != len(self.ids) or len(self.records) != len(self.ids) - self.first:
            raise ValueError("incomplete actual-target capture")
        return self.records


def native_records(response, ids, first, document):
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or response.get("error"):
        raise ValueError("native refusal or invalid completion count")
    rows = choices[0].get("prompt_logprobs")
    if not isinstance(rows, list) or len(rows) != len(ids) or rows[0] is not None:
        raise ValueError("native prompt score geometry differs")
    if response.get("usage", {}).get("prompt_tokens") != len(ids):
        raise ValueError("native consumed different prompt count")
    if choices[0].get("prompt_token_ids", ids) != ids:
        raise ValueError("native consumed different token IDs")
    collector = ActualTargets(ids, first, document)
    for pos in range(first, len(ids)):
        row = rows[pos]
        entry = row.get(str(ids[pos])) if isinstance(row, dict) else None
        value = entry.get("logprob") if isinstance(entry, dict) else None
        collector.add([pos], [ids[pos]], [value], [[]], [[]])
    return collector.finish()
