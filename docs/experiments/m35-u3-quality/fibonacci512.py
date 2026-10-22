#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Separate post-truncation completion screen; never revises the 256-token result."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import urllib.error
import urllib.request

INPUT_SHA = "ef8b44d0f97e7813edbc506d3105ec0dfd112de05efc4e4bbbe3de9158e9c960"
RUNTIME_SHA = "32e247723ebe2c00bcdae1cb1aff12884ce7fb7fb6b421e05e47eea9e1c124af"
SERVICE_SHA = "0f1135a2283fe63797e15c8dffe48c9df39b0baae91a57f925b5ba21c3e3252e"
LOADER_SHA = "61732d66793a85ca140cb090a2c8aecad23da141eb4980f3cb8767953f3127a1"
METADATA = {
    "config.json": "e7cbee89103cc488ef55c142c2a2a376ea00ce67b49c3ae9282d944107d97c33",
    "tokenizer.json": "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b",
    "tokenizer_config.json": "5c756ad6048c884974a331963ae3c3e1624b548a11174dadda02a69041c87352",
    "generation_config.json": "b52d7103195cf7520887804dc703c715515dca31c7a2db20afdb2bf8f78203f0",
}
TEMPLATE_SHA = "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65"
# Copied from the authenticated, original input file, not retokenized.
IDS = [128000,128006,9125,128007,271,38766,1303,33025,2696,25,6790,220,
       2366,18,198,15724,2696,25,220,1627,10263,220,2366,19,271,128009,
       128006,882,128007,271,8144,264,13325,734,430,4780,279,308,7716,
       80783,1396,5480,8046,13,128009,128006,78191,128007,271]


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require_sha(path, expected):
    got = sha(path)
    if got != expected:
        raise ValueError(f"registered identity differs: {path}: {got}")
    return got


def inputs(path, claimed=INPUT_SHA):
    if claimed != INPUT_SHA:
        raise ValueError("supplement input hash cannot be overridden")
    require_sha(path, INPUT_SHA)
    obj = json.loads(Path(path).read_bytes())
    if (obj["format"] != "llmp-u3-llama-behavior-inputs/1" or
            [c["id"] for c in obj["cases"]] !=
            ["capital", "haiku", "sky", "fibonacci", "primes", "french"] or
            obj["max_tokens"] != 256 or obj["cases"][3]["ids"] != IDS):
        raise ValueError("original behavior fixture contract differs")
    return obj["cases"][3]


def source_metadata(checkpoint):
    return {name: require_sha(Path(checkpoint) / name, want)
            for name, want in METADATA.items()}


def native_preflight(runtime, config, repo):
    require_sha(runtime, RUNTIME_SHA)
    require_sha(Path(repo) / "tools/speed_gate.py", SERVICE_SHA)
    # Permit only the original recipe and relocatable paths. No drafter,
    # precision mode, sampling override or retained conversation is inherited.
    if set(config) != {"schema_version", "storage", "models", "client"} or config["schema_version"] != 2:
        raise ValueError("supplement native config sections differ")
    if set(config["models"]) != {"llama33"}:
        raise ValueError("supplement requires the single original model")
    model = config["models"]["llama33"]
    expected = {"artifact": "9cca62f1e6582324de2ebce34f38b89a8aa8890cdced957a50782e407ff4d7ee",
                "context": 12288, "max_slots": 1, "prefill_chunk": 512,
                "kv_cache_encoding": "f16", "temperature": 0.0, "top_p": 1.0, "top_k": 0}
    if set(model) != set(expected) | {"tokenizer", "chat_template"} or any(
            model.get(k) != v for k, v in expected.items()):
        raise ValueError("supplement native model recipe differs")
    if config["client"] != {"bind": "127.0.0.1", "port": 18974}:
        raise ValueError("supplement native endpoint differs")
    if set(config["storage"]) != {"installed", "data_dir"}:
        raise ValueError("supplement storage recipe differs")
    if Path(config["storage"]["data_dir"]).exists():
        raise ValueError("supplement requires a fresh native state directory")
    artifact = Path(config["storage"]["installed"]) / expected["artifact"]
    for name, digest in {
        "manifest.json": "6d4b2ebc689516dc8da7a5586dafd7090cfd720452faeeca5f95ec003ec89c8c",
        "index.json": "f7009e3c654ca4664b2bf4413c00348f00c8c106f9c034f5424c73103c4fa46b",
        "spec.json": "9647472757577a29b988466202b27cecad070627d25f627a2210336a1694d912",
    }.items():
        require_sha(artifact / name, digest)
    require_sha(model["tokenizer"], METADATA["tokenizer.json"])
    source_metadata(Path(model["tokenizer"]).parent)
    require_sha(model["chat_template"], TEMPLATE_SHA)


def reference_startup(log):
    """Fail before the request if the pinned image's actual startup differs."""
    engines = [line for line in log.splitlines() if "Initializing a V1 LLM engine" in line]
    expected = ("v0.1.dev20073+g8e685d198", "dtype=torch.float16,",
                "kv_cache_dtype=float16,", "quantization=modelopt_fp4,",
                "speculative_config=None,", "trust_remote_code=False,",
                "enable_prefix_caching=False,")
    if len(engines) != 1 or not all(value in engines[0] for value in expected):
        raise ValueError("reference actual engine precision/configuration differs")
    # Inspect every actual selection, not just existence of the desired one.
    # Engine config contains available-operation names too; those are not
    # selections. The pinned logger emits actual choices as these records.
    architectures = []
    linears = []
    attentions = []
    attention_backends = []
    for line in log.splitlines():
        if "Resolved architecture:" in line:
            architectures.append(line.split("Resolved architecture:", 1)[1].strip())
        using = re.search(r"\bUsing (.+)$", line)
        if using:
            selection = using.group(1).strip()
            if re.search(r"linear|gemm|nvfp4|marlin|humming|w4a16", selection, re.I):
                linears.append(selection)
            backend = re.fullmatch(r"(\S+) attention backend out of potential backends: .+", selection)
            if backend:
                attention_backends.append(backend.group(1))
            elif re.search(r"attention|flashattn|flash_attn", selection, re.I):
                attentions.append(selection)
        if re.search(r"fall(?:ing)? back|fallback", line, re.I) and re.search(
                r"linear|gemm|nvfp4|marlin|humming|w4a16|attention|flash.?attn", line, re.I):
            raise ValueError("reference startup reports a numerical/dispatch fallback")
    for selections, expected_selection in (
            (architectures, "LlamaForCausalLM"),
            (linears, "FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM"),
            (attentions, "FlashAttention version 2"),
            (attention_backends, "FLASH_ATTN")):
        if not selections or set(selections) != {expected_selection}:
            raise ValueError(f"reference actual selections differ: {selections}")
    return {"architecture": "LlamaForCausalLM", "dtype": "float16",
            "kv_cache_dtype": "float16", "quantization": "modelopt_fp4",
            "linear": "FlashInferCutlassNvFp4LinearKernel", "attention": "FlashAttention2"}


def body():
    # Identical to the original Fibonacci body except max_tokens 256 -> 512.
    # EOS remains effective: no ignore_eos, custom stop strings or token filter.
    return {"model": "llama33", "prompt": IDS, "max_tokens": 512,
            "temperature": 0, "top_p": 1, "add_special_tokens": False, "stream": False}


def validate_response(obj):
    choices = obj.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("expected one completion")
    choice = choices[0]
    if not isinstance(choice.get("text"), str) or not choice["text"].strip():
        raise ValueError("missing reply text")
    finish = choice.get("finish_reason")
    if finish not in {"stop", "length"}:
        raise ValueError("missing or unexpected finish reason")
    usage = obj.get("usage", {})
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(key)) is not int:
            raise ValueError("missing integer usage")
    count = usage["completion_tokens"]
    if (usage["prompt_tokens"] != len(IDS) or not 0 < count <= 512 or
            usage["total_tokens"] != len(IDS) + count):
        raise ValueError("request/usage mismatch")
    if finish == "length" and count != 512:
        raise ValueError("length finish before the registered allowance")
    details = usage.get("prompt_tokens_details")
    if details is not None and details.get("cached_tokens", 0) != 0:
        raise ValueError("independent launch unexpectedly reused cached prompt tokens")
    return {"finish_reason": finish, "usage": usage,
            "completion_status": "complete_finish" if finish == "stop" else "truncated",
            "semantic_verdict": "pending_independent_review"}


def collect(url, input_path, out):
    inputs(input_path)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    request_raw = (json.dumps(body(), sort_keys=True) + "\n").encode()
    (out / "request.json").write_bytes(request_raw)
    receipt = {"format": "llmp-u3-fibonacci512/1", "input_sha256": INPUT_SHA,
               "case": "fibonacci", "input_ids": IDS, "max_tokens": 512,
               "request_sha256": hashlib.sha256(request_raw).hexdigest(),
               "collection_complete": False, "semantic_verdict": "pending_independent_review"}
    try:
        request = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                         data=request_raw, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            (out / "response-error.bin").write_bytes(error.read())
            raise
        # Preserve malformed and nonqualifying replies too, before parsing.
        (out / "response.json").write_bytes(raw)
        receipt["response_sha256"] = hashlib.sha256(raw).hexdigest()
        receipt.update(validate_response(json.loads(raw)))
        receipt["collection_complete"] = True
    except BaseException as error:
        receipt["error"] = str(error)
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def main():
    import os
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(args.url, args.inputs, args.out)


if __name__ == "__main__":
    main()
