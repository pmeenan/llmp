# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect bounded, explicit-ID prompt losses from the existing Mia API.

The caller owns server admission, authenticated source and supervised retirement.
This is an external diagnostic; it neither starts a server nor accepts quality.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import urllib.request
import uuid

P = Path(__file__).resolve().parent
CAP = 2 << 20
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(base, endpoint, payload):
    request = urllib.request.Request(base + endpoint, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with OPENER.open(request, timeout=90) as response:
        raw = response.read(CAP + 1)
    if len(raw) > CAP:
        raise ValueError("reference response exceeds 2MiB")
    return json.loads(raw)


def losses(body, ids):
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("one reference choice required")
    rows = choices[0].get("prompt_logprobs")
    if not isinstance(rows, list) or len(rows) != len(ids) or rows[0] is not None:
        raise ValueError("complete explicit prompt coordinates required")
    for position in range(1, len(ids)):
        entries = rows[position]
        if not isinstance(entries, dict) or str(ids[position]) not in entries:
            raise ValueError("actual target log probability missing")
        value = entries[str(ids[position])].get("logprob")
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("target log probability is not numeric")
        if not math.isfinite(value) or value > 0:
            raise ValueError("target log probability is not finite and nonpositive")
        yield position, ids[position], -float(value)


def main(base, model):
    if base != "http://127.0.0.1:18253" or model != "qwen3.8-flash-next":
        raise ValueError("fixed owned endpoint/model required")
    raw = (P / "inputs.json").read_bytes()
    if len(raw) > 64 << 10:
        raise ValueError("input metadata exceeds 64KiB")
    source = json.loads(raw)
    documents = source["documents"]
    if len(documents) != 4 or [d["document"] for d in documents] != list(range(4)):
        raise ValueError("four fixed independent documents required")
    ids = [token for d in documents for token in d["tokens"]]
    if len(ids) != 1020 or any(type(t) is not int or not 0 <= t < 248320 for t in ids):
        raise ValueError("fixed finite token inventory required")
    digest = hashlib.sha256(struct.pack("<1020i", *ids)).hexdigest()
    if digest != "3cd946c3876501eccbe079b9d2700e542e66242be241a8549546df8d9b23e468":
        raise ValueError("authenticated native token IDs changed")
    output = {"status": "INCOMPLETE", "ids_sha256": digest, "documents": []}
    # Exclusive files distinguish partial collection from a completed receipt.
    with (P / "losses.jsonl").open("x") as sink:
        for document in documents:
            number = document["document"]
            tokens = document["tokens"]
            text = (P / f"doc{number}.txt").read_bytes()
            if len(tokens) != 255 or hashlib.sha256(text).hexdigest() != document["text_sha256"]:
                raise ValueError("fixed document/token geometry changed")
            actual = post(base, "/tokenize", {"model": model, "prompt": text.decode(),
                                               "add_special_tokens": False})["tokens"]
            if actual[:255] != tokens:
                raise ValueError("reference token meanings/prefix disagree")
            begin = time.monotonic()
            body = post(base, "/v1/completions", {"model": model, "prompt": tokens,
                        "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1,
                        "cache_salt": str(uuid.uuid4())})
            count = 0
            total = 0.0
            for position, token, nll in losses(body, tokens):
                sink.write(json.dumps({"document": number, "position": position,
                                      "target": token, "nll": nll}) + "\n")
                total += nll
                if not math.isfinite(total):
                    raise ValueError("document loss sum is not finite")
                count += 1
            if count != 254:
                raise ValueError("incomplete document loss coverage")
            output["documents"].append({"document": number, "targets": count,
                                         "nll_sum": total, "seconds": time.monotonic() - begin})
    output["targets"] = sum(d["targets"] for d in output["documents"])
    output["mean_nll"] = sum(d["nll_sum"] for d in output["documents"]) / output["targets"]
    if not math.isfinite(output["mean_nll"]):
        raise ValueError("aggregate mean loss is not finite")
    output["perplexity"] = math.exp(output["mean_nll"])
    if not math.isfinite(output["perplexity"]):
        raise ValueError("aggregate perplexity is not finite")
    output["status"] = "PASS"
    with (P / "result.json").open("x") as report:
        report.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
