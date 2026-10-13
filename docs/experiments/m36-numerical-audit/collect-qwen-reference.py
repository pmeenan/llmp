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
    with (P / "inputs.json").open("rb") as source_file:
        raw = source_file.read((64 << 10) + 1)
    if len(raw) > 64 << 10:
        raise ValueError("input metadata exceeds 64KiB")
    source = json.loads(raw)
    documents = source["documents"]
    if not documents or [d["document"] for d in documents] != list(range(len(documents))):
        raise ValueError("ordered authenticated documents required")
    ids = [token for d in documents for token in d["tokens"]]
    if any(not 2 <= len(d["tokens"]) <= 33792 for d in documents) or any(
            type(t) is not int or not 0 <= t < 248320 for t in ids):
        raise ValueError("finite token inventory within the admitted context required")
    digest = hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()
    if digest != source["ids_sha256"]:
        raise ValueError("authenticated native token IDs changed")
    output = {"status": "INCOMPLETE", "ids_sha256": digest, "documents": []}
    # Exclusive files distinguish partial collection from a completed receipt.
    with (P / "losses.jsonl").open("x") as sink:
        for document in documents:
            number = document["document"]
            tokens = document["tokens"]
            text = (P / f"doc{number}.txt").read_bytes()
            if hashlib.sha256(text).hexdigest() != document["text_sha256"]:
                raise ValueError("authenticated document changed")
            actual = post(base, "/tokenize", {"model": model, "prompt": text.decode(),
                                               "add_special_tokens": source.get(
                                                   "add_special_tokens", False)})["tokens"]
            compared = actual if source.get("complete_tokenization", False) else actual[:len(tokens)]
            if compared != tokens:
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
            if count != len(tokens) - 1:
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


def native_inputs(record_path, destination):
    """Bind the existing native quality response; no tokenization or model call."""
    record_path, destination = Path(record_path), Path(destination)
    with record_path.open("rb") as source:
        raw_record = source.read(CAP + 1)
    if len(raw_record) > CAP:
        raise ValueError("native quality record exceeds 2MiB")
    record = json.loads(raw_record)
    if record.get("error") or record.get("family") != "qwen38":
        raise ValueError("completed native Qwen quality required")
    blobs = {}
    for name in ("score_prompt", "score_response"):
        member = record[name]
        filename = member["file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("native quality member must be a sibling file")
        with (record_path.parent / filename).open("rb") as source:
            raw = source.read(CAP + 1)
        if len(raw) > CAP or hashlib.sha256(raw).hexdigest() != member["sha256"]:
            raise ValueError("bounded native quality member changed")
        blobs[name] = raw
    body = json.loads(blobs["score_response"])
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("one native choice required")
    token_names = choices[0]["logprobs"]["tokens"]
    if not isinstance(token_names, list) or not 2 <= len(token_names) <= 33792:
        raise ValueError("complete native token coordinates required")
    ids = []
    for name in token_names:
        if not isinstance(name, str) or not name.startswith("token_id:"):
            raise ValueError("native token-ID labels required")
        value = name[len("token_id:"):]
        if not value.isascii() or not value.isdecimal() or value != str(int(value)):
            raise ValueError("canonical native token-ID label required")
        token = int(value)
        if not 0 <= token < 248320:
            raise ValueError("native token outside checkpoint vocabulary")
        ids.append(token)
    rows = list(losses(body, ids))
    if body["usage"]["prompt_tokens"] != len(ids) or body["usage"]["completion_tokens"] != 0:
        raise ValueError("native scoring work differs from complete token inventory")
    if len(rows) != record["score"]["tokens"]:
        raise ValueError("native loss count differs from recorded quality")
    scores = [-row[2] for row in rows]
    if hashlib.sha256(json.dumps(scores).encode()).hexdigest() != record["score"]["scores_sha256"]:
        raise ValueError("native target loss array differs from recorded quality")
    if choices[0]["logprobs"]["token_logprobs"] != [None, *scores]:
        raise ValueError("native echo and prompt loss coordinates disagree")
    total = math.fsum(row[2] for row in rows)
    mean = total / len(rows)
    # The recorded mean uses ordered binary64 sum; fsum differs only by
    # accumulation roundoff. Nonnegative losses avoid cancellation. Allow four
    # n*epsilon terms for accumulation and the serialized binary64 endpoints.
    tolerance = 4 * (len(rows) + 1) * sys.float_info.epsilon
    if not math.isfinite(mean) or not math.isclose(
            mean, record["score"]["nll"], rel_tol=tolerance, abs_tol=0.0):
        raise ValueError("native loss aggregate differs from recorded quality")
    perplexity = math.exp(mean)
    if not math.isfinite(perplexity):
        raise ValueError("native perplexity is not finite")
    digest = hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()
    inputs = {"ids_sha256": digest, "complete_tokenization": True,
              "add_special_tokens": True,
              "native_response_sha256": record["score_response"]["sha256"],
              "documents": [{"document": 0, "tokens": ids,
                             "text_sha256": record["score_prompt"]["sha256"]}]}
    # Input and result files are exclusive so a partial collection is never
    # mistaken for the newly completed authority.
    with (destination / "doc0.txt").open("xb") as sink:
        sink.write(blobs["score_prompt"])
    with (destination / "inputs.json").open("x") as sink:
        sink.write(json.dumps(inputs, indent=2, allow_nan=False) + "\n")
    with (destination / "native-losses.jsonl").open("x") as sink:
        for position, token, nll in rows:
            sink.write(json.dumps({"document": 0, "position": position,
                                   "target": token, "nll": nll}, allow_nan=False) + "\n")
    result = {"status": "PASS", "targets": len(rows), "mean_nll": mean,
              "perplexity": perplexity, "ids_sha256": digest,
              "native_response_sha256": record["score_response"]["sha256"]}
    with (destination / "native-result.json").open("x") as sink:
        sink.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--native-quality":
        print(json.dumps(native_inputs(sys.argv[2], sys.argv[3]), allow_nan=False))
    else:
        main(*sys.argv[1:])
