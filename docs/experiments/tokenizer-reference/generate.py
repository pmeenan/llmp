#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates the tokenizer and chat-template reference fixtures (README.md).

    generate.py --repo REPO --models DIR --work DIR [--only tokens|chat]

Runs on a Spark, in a virtual environment holding requirements.txt, with
sudo docker for the pinned llama.cpp image. Reads the corpus
(tests/unit/data/tokenizer/corpus.json) and writes the expected token IDs
of every configuration and the chat fixtures under tests/unit/data/.

Token references: llama.cpp's llama-tokenize at the pinned image for the
GGUF vocabularies (with and without parsing special tokens), and Hugging
Face tokenizers for tokenizer.json vocabularies (special tokens parsed, and
with encode_special_tokens for the other mode). Ill-formed UTF-8 has no
reference encoding; its items record the tokens of the text Python's
"replace" decoding gives.

Chat references: transformers' apply_chat_template on the pinned template
bytes, and for DeepSeek V4 0731 also DeepSeek's encoding_dsv4.py where it
defines the case. The community GGUF's chat-v2 template needs a from_json
filter, which transformers lacks: json.loads stands in for it.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess


LLAMA_IMAGE = "ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7"

DEEPSEEK_GGUF = "models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf"
QWEN38_GGUF = "reference-models/Q/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
QWEN38_DIR = "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6"
QWEN_IMAGE_TOKENIZER = "models/Qwen/Qwen-Image-2.1@790c9263/processor/tokenizer.json"
# DeepSeek's own files at deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b: tokenizer.json,
# tokenizer_config.json and encoding/ (README.md lists them with their hashes).
DEEPSEEK_DIR = "tokenizer-reference/deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b"
# The community IQ2_XXS GGUF (the ds4 study's), whose embedded template is "chat-v2"; its
# tokenizer is the 0731 GGUF's.
COMMUNITY_GGUF = ("models/antirez/deepseek-v4-gguf@f71f23d5/"
                  "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf")

# Gemma4 approved unsloth/gemma-4-26B-A4B-it-GGUF@c099eb48e663fd284577b04978a94ffccb261841.
GEMMA4_GGUF = "reference-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
GEMMA4_HEADER = "tokenizer-reference/gemma-4-26b/header.gguf"

# (name, kind, path relative to --models)
CONFIGS = [
    ("gemma-4-26b-gguf", "gguf", GEMMA4_HEADER),
    ("deepseek-v4-0731-gguf", "gguf", DEEPSEEK_GGUF),
    ("qwen3.8-gguf", "gguf", QWEN38_GGUF),
    ("qwen3.8-nvfp4", "hf", QWEN38_DIR + "/tokenizer.json"),
    ("qwen-image-2.1", "hf", QWEN_IMAGE_TOKENIZER),
    ("deepseek-v4-0731-hf", "hf", DEEPSEEK_DIR + "/tokenizer.json"),
]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gguf_chat_template(path: pathlib.Path) -> bytes:
    """tokenizer.chat_template's bytes from a GGUF file's metadata."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with path.open("rb") as f:
        def u(n):
            return int.from_bytes(f.read(n), "little")

        def string():
            return f.read(u(8))

        def skip(t):
            if t in sizes:
                f.seek(sizes[t], 1)
            elif t == 8:
                string()
            elif t == 9:
                et, n = u(4), u(8)
                if et in sizes:
                    f.seek(sizes[et] * n, 1)
                else:
                    for _ in range(n):
                        skip(et)
            else:
                raise ValueError(f"GGUF type {t}")

        if f.read(4) != b"GGUF":
            raise ValueError("not GGUF")
        _version, _tensors, keys = u(4), u(8), u(8)
        for _ in range(keys):
            key, t = string(), u(4)
            if key == b"tokenizer.chat_template" and t == 8:
                return string()
            skip(t)
    raise ValueError("no chat template")


def corpus_items(path: pathlib.Path):
    """Yields (name, bytes) for each corpus item."""
    for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
        if "hex" in item:
            yield item["name"], bytes.fromhex(item["hex"])
        elif "repeat" in item:
            yield item["name"], (item["repeat"] * item["count"]).encode("utf-8")
        else:
            yield item["name"], item["text"].encode("utf-8")


def reference_text(data: bytes) -> tuple[bool, bytes]:
    """Whether data is well-formed UTF-8, and the text the references see."""
    try:
        data.decode("utf-8")
        return True, data
    except UnicodeDecodeError:
        return False, data.decode("utf-8", "replace").encode("utf-8")


def llama_tokenize(models: pathlib.Path, gguf: str, texts: dict[str, bytes], work: pathlib.Path) -> dict:
    """Runs llama-tokenize over every text in one container, both modes."""
    work.mkdir(parents=True, exist_ok=True)
    names = sorted(texts)
    for i, name in enumerate(names):
        (work / f"{i:04d}.txt").write_bytes(texts[name])
    script = (
        "cd /app && for f in /work/*.txt; do "
        f"./llama-tokenize -m /models/{gguf} -f $f --no-escape --ids --no-bos --log-disable > $f.parse 2>$f.err1; "
        f"./llama-tokenize -m /models/{gguf} -f $f --no-escape --ids --no-bos --log-disable --no-parse-special"
        " > $f.plain 2>$f.err2; done")
    subprocess.run(["sudo", "-n", "docker", "run", "--rm", "--network", "none", "-e", "CUDA_DISABLE_PTX_JIT=1",
                    "--user", f"{os.getuid()}:{os.getgid()}",
                    "-v", f"{models}:/models:ro", "-v", f"{work}:/work", "--entrypoint", "/bin/sh", LLAMA_IMAGE,
                    "-c", script], check=True)
    out = {}
    for i, name in enumerate(names):
        base = work / f"{i:04d}.txt"
        out[name] = {mode: json.loads((base.parent / f"{base.name}.{mode}").read_text()) for mode in ("parse", "plain")}
    return out


def hf_tokenize(path: pathlib.Path, texts: dict[str, bytes]) -> dict:
    tok = tokenizers.Tokenizer.from_file(str(path))
    out = {}
    for name, data in texts.items():
        text = data.decode("utf-8")
        tok.encode_special_tokens = False
        parse = tok.encode(text, add_special_tokens=False).ids
        tok.encode_special_tokens = True
        plain = tok.encode(text, add_special_tokens=False).ids
        tok.encode_special_tokens = False
        out[name] = {"parse": parse, "plain": plain}
    return out


def write_json(path: pathlib.Path, value) -> None:
    """Writes value with each list element on its own line and anything below that compact."""
    lines = ["{"]
    keys = list(value)
    for i, key in enumerate(keys):
        comma = "," if i + 1 < len(keys) else ""
        v = value[key]
        if isinstance(v, list):
            lines.append(f"  {json.dumps(key)}: [")
            for j, element in enumerate(v):
                lines.append("    " + json.dumps(element, ensure_ascii=False) + ("," if j + 1 < len(v) else ""))
            lines.append("  ]" + comma)
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(v, ensure_ascii=False)}{comma}")
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    license = path.with_name(path.name + ".license")
    license.write_text("SPDX-FileCopyrightText: 2026 jitLLM contributors\nSPDX-License-Identifier: Apache-2.0\n")
    license = path.with_name(path.name + ".license")
    license.write_text("SPDX-FileCopyrightText: 2026 jitLLM contributors\nSPDX-License-Identifier: Apache-2.0\n")


def tokens(args):
    repo, models = pathlib.Path(args.repo), pathlib.Path(args.models)
    corpus = repo / "tests/unit/data/tokenizer/corpus.json"
    items = list(corpus_items(corpus))
    texts, valid = {}, {}
    for name, data in items:
        valid[name], texts[name] = reference_text(data)
    results = {}
    for config, kind, rel in CONFIGS:
        if args.config and config != args.config:
            continue
        path = models / rel
        oracle_rel = rel
        if config == "gemma-4-26b-gguf":
            # Exact metadata prefix of the approved file; avoid loading or
            # hashing 17 GB of weights for a tokenizer-only comparison.
            oracle_rel = GEMMA4_GGUF
            prefix = (models / oracle_rel).open("rb").read(15784342)
            digest = hashlib.sha256(prefix).hexdigest()
            if digest != "9f1f00bb292382747a7035a8a3e5d5f41f31e7d7be80b2aefa044ff03434f990":
                raise ValueError("Gemma4 header differs from the approved checkpoint")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(prefix)
        if kind == "gguf":
            ids = llama_tokenize(models, oracle_rel, texts, pathlib.Path(args.work) / config)
            reference = f"llama.cpp llama-tokenize, image {LLAMA_IMAGE} (b29c606e)"
        else:
            ids = hf_tokenize(path, texts)
            reference = f"Hugging Face tokenizers {tokenizers.__version__}"
        results[config] = ids
        write_json(repo / f"tests/unit/data/tokenizer/{config}.json", {
            "config": config,
            "source": {"path": rel, "sha256": sha256(path)},
            "reference": reference,
            "corpus_sha256": sha256(corpus),
            "items": [{"name": name, "well_formed": valid[name], **ids[name]} for name, _ in items],
        })
    return results


# --- chat -------------------------------------------------------------------

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name, e.g. Paris"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer", "minimum": 1, "maximum": 7.5}},
        "required": ["city"]}}}
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search", "description": "Search the web — returns “snippets”.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "number", "default": 1e-05}}}}}


def call(name, arguments):
    return {"name": name, "arguments": arguments}


COMMON_CASES = [
    {"name": "single-user", "messages": [{"role": "user", "content": "What is 2+2?"}]},
    {"name": "system-user", "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                         {"role": "user", "content": "Hello!"}]},
    {"name": "multi-turn", "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?", "reasoning_content": "The user greets me."},
        {"role": "user", "content": "Capital of France?"}]},
    {"name": "multi-turn-ending-assistant", "add_generation_prompt": False, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "Greeting."}]},
    {"name": "tools-full-round", "tools": [WEATHER_TOOL, SEARCH_TOOL], "messages": [
        {"role": "system", "content": "You can call tools."},
        {"role": "user", "content": "Weather in Paris and Tokyo for 3 days, in celsius?"},
        {"role": "assistant", "content": "", "reasoning_content": "I need two weather calls.",
         "tool_calls": [call("get_weather", {"city": "Paris", "days": 3, "unit": "celsius"}),
                        call("get_weather", {"city": "東京", "days": 3, "extra": {"nested": [1, 2.5, None, True]}})]},
        {"role": "tool", "content": "{\"temp\": 21, \"sky\": \"sunny\"}"},
        {"role": "tool", "content": "{\"temp\": 18.5, \"sky\": \"rain\"}"},
        {"role": "assistant", "content": "Paris: 21°C, sunny. Tokyo: 18.5°C, rain.",
         "reasoning_content": "Summarize both results."},
        {"role": "user", "content": "Thanks! Any search results on umbrellas?"}]},
    {"name": "tools-without-system", "tools": [SEARCH_TOOL], "messages": [
        {"role": "user", "content": "Find news about tokenizers."}]},
    {"name": "tools-empty-system", "tools": [SEARCH_TOOL], "messages": [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Find news."}]},
    {"name": "assistant-content-and-call", "tools": [WEATHER_TOOL], "messages": [
        {"role": "system", "content": "Tools allowed."},
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [call("get_weather", {"city": "Oslo"})]},
        {"role": "tool", "content": "cold"}]},
    {"name": "unicode-content", "messages": [
        {"role": "system", "content": "Réponds en français. 用中文也可以。"},
        {"role": "user", "content": "Tiếng Việt 😀 👨‍👩‍👧 \t tabs and  spaces  "}]},
    {"name": "whitespace-trim", "messages": [
        {"role": "system", "content": "  \n system with spaces 　"},
        {"role": "user", "content": "\n\n  question?  \n\u001c"}]},
    {"name": "content-with-tags", "messages": [
        {"role": "user", "content": "Explain <think> and </think> and <tool_call> tags."}]},
    {"name": "none-content", "messages": [
        {"role": "user", "content": None}]},
]

DEEPSEEK_CASES = [
    {"name": "thinking-single", "options": {"enable_thinking": True}, "messages": [
        {"role": "user", "content": "Prove that sqrt(2) is irrational."}]},
    {"name": "thinking-effort-high", "options": {"enable_thinking": True, "reasoning_effort": "high"}, "messages": [
        {"role": "system", "content": "Think hard."}, {"role": "user", "content": "Solve x^2=2."}]},
    {"name": "thinking-effort-max", "options": {"enable_thinking": True, "reasoning_effort": "max"}, "messages": [
        {"role": "user", "content": "Plan a trip."}]},
    {"name": "chat-effort-ignored", "options": {"enable_thinking": False, "reasoning_effort": "max"}, "messages": [
        {"role": "user", "content": "Hi"}]},
    {"name": "thinking-multi-turn-drops", "options": {"enable_thinking": True}, "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "The user said hello."},
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris.", "reasoning_content": "It is Paris."}],
     "add_generation_prompt": False},
    {"name": "thinking-tools-keeps", "options": {"enable_thinking": True}, "tools": [WEATHER_TOOL], "messages": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Weather in Rome?"},
        {"role": "assistant", "content": "", "reasoning_content": "Call the tool.",
         "tool_calls": [call("get_weather", {"city": "Rome", "unit": "celsius"})]},
        {"role": "tool", "content": "{\"temp\": 25}"},
        {"role": "assistant", "content": "25°C in Rome.", "reasoning_content": "Report it."},
        {"role": "user", "content": "And tomorrow?"}]},
    {"name": "two-system-messages", "messages": [
        {"role": "system", "content": "First."}, {"role": "system", "content": "Second."},
        {"role": "user", "content": "Go."}]},
    {"name": "user-after-user", "messages": [
        {"role": "user", "content": "one"}, {"role": "user", "content": "two"}]},
    {"name": "bad-effort", "options": {"reasoning_effort": "extreme"}, "error": "invalid",
     "messages": [{"role": "user", "content": "x"}]},
]

# The community GGUF's chat-v2 template: DeepSeek's cases (it has no reasoning
# effort, so an unknown one renders rather than fails), and the places where it
# parts from the 0731 template. A tool call's arguments may be the client's
# string, which the template parses with from_json.
DEEPSEEK_V2_CASES = [c for c in DEEPSEEK_CASES if c["name"] != "bad-effort"] + [
    {"name": "effort-unknown-ignored", "options": {"enable_thinking": True, "reasoning_effort": "extreme"},
     "messages": [{"role": "user", "content": "x"}]},
    {"name": "thinking-history-kept", "options": {"enable_thinking": True}, "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "The user said hello."},
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris.", "reasoning_content": "It is Paris."},
        {"role": "user", "content": "And Italy?"}]},
    {"name": "thinking-empty-and-null-reasoning", "options": {"enable_thinking": True}, "messages": [
        {"role": "user", "content": "One"},
        {"role": "assistant", "content": "1", "reasoning_content": ""},
        {"role": "user", "content": "Two"},
        {"role": "assistant", "content": "2", "reasoning_content": None},
        {"role": "user", "content": "Three"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "Four"}]},
    {"name": "thinking-off-history", "options": {"enable_thinking": False}, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "Greeting."},
        {"role": "user", "content": "Bye"}]},
    {"name": "string-arguments", "options": {"enable_thinking": True}, "tools": [WEATHER_TOOL, SEARCH_TOOL],
     "messages": [
        {"role": "user", "content": "Weather in Paris and Tokyo, then search?"},
        {"role": "assistant", "content": "", "reasoning_content": "Three calls.",
         "tool_calls": [
             call("get_weather", "{\"city\": \"Paris\", \"days\": 3}"),
             call("get_weather", "{\"city\":\"\\u6771\\u4eac\",\"days\":2.50,\"unit\":\"celsius\","
                                 "\"extra\":{\"a\":[1,-0,2.0,1E5,null,true,\"\\u00e9\\n\\\"q\\\"\"],"
                                 "\"big\":123456789012345678901234567890}}"),
             call("search", "{}")]},
        {"role": "tool", "content": "sunny"},
        {"role": "tool", "content": "rain"},
        {"role": "tool", "content": ""},
        {"role": "assistant", "content": "Paris sunny, Tokyo rain.", "reasoning_content": "Report."},
        {"role": "user", "content": "Thanks."}]},
    {"name": "string-arguments-not-object", "error": "invalid", "tools": [WEATHER_TOOL], "messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [call("get_weather", "[1, 2]")]},
        {"role": "tool", "content": "x"}]},
    {"name": "tool-after-user", "tools": [WEATHER_TOOL], "messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "tool", "content": "cold"},
        {"role": "tool", "content": "windy"}]},
    {"name": "user-after-tool", "tools": [WEATHER_TOOL], "messages": [
        {"role": "user", "content": "Weather in Oslo?"},
        {"role": "assistant", "content": "Checking.", "tool_calls": [call("get_weather", {"city": "Oslo"})]},
        {"role": "tool", "content": "cold"},
        {"role": "user", "content": "And in Rome?"}]},
    {"name": "tool-first", "messages": [
        {"role": "tool", "content": "orphan"}, {"role": "user", "content": "What was that?"}]},
    {"name": "assistant-first", "messages": [
        {"role": "assistant", "content": "Hello, I am ready.", "reasoning_content": "Opening."},
        {"role": "user", "content": "Hi"}]},
    {"name": "consecutive-assistants", "options": {"enable_thinking": True}, "messages": [
        {"role": "user", "content": "Count."},
        {"role": "assistant", "content": "One.", "reasoning_content": "First."},
        {"role": "assistant", "content": "Two.", "reasoning_content": "Second."},
        {"role": "user", "content": "Go on."}]},
    {"name": "ending-assistant-with-prompt", "add_generation_prompt": True, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"}]},
    {"name": "user-without-prompt", "add_generation_prompt": False, "messages": [
        {"role": "user", "content": "Hi"}]},
    {"name": "system-only", "messages": [{"role": "system", "content": "Just a system."}]},
    {"name": "system-between", "options": {"enable_thinking": True}, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "Late rule."},
        {"role": "assistant", "content": "Hello.", "reasoning_content": "Greet."},
        {"role": "user", "content": "Bye"}]},
    {"name": "tools-two-systems-first-empty", "tools": [SEARCH_TOOL], "messages": [
        {"role": "system", "content": ""}, {"role": "system", "content": "Second."},
        {"role": "user", "content": "Find it."}]},
    {"name": "tools-non-function", "tools": [{"type": "code_interpreter"}], "messages": [
        {"role": "user", "content": "Run it."}]},
    {"name": "thinking-whitespace-reasoning", "options": {"enable_thinking": True}, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello.", "reasoning_content": " "},
        {"role": "user", "content": "Bye"}]},
    {"name": "tool-runs-interleaved", "options": {"enable_thinking": True}, "tools": [WEATHER_TOOL], "messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "reasoning_content": "Call.",
         "tool_calls": [call("get_weather", {"city": "Oslo", "ключ": {"é": [[], {}, -1.5e-7, "</｜DSML｜parameter>"]}})]},
        {"role": "tool", "content": None},
        {"role": "user", "content": "Also Rome."},
        {"role": "tool", "content": "warm"},
        {"role": "tool", "content": "dry"},
        {"role": "user", "content": "Summarize."}]},
    {"name": "assistant-first-then-tool", "options": {"enable_thinking": True}, "messages": [
        {"role": "assistant", "content": "", "reasoning_content": "Dropped.",
         "tool_calls": [call("search", {"query": "x"})]},
        {"role": "tool", "content": "found"}]},
    {"name": "system-last-with-prompt", "options": {"enable_thinking": False}, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "Trailing rule."}]},
]

QWEN_CASES = [
    {"name": "thinking-off", "options": {"enable_thinking": False}, "messages": [
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]},
    {"name": "effort-low", "options": {"reasoning_effort": "low"}, "messages": [
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]},
    {"name": "effort-medium-no-system", "options": {"reasoning_effort": "medium"}, "messages": [
        {"role": "user", "content": "Hi"}]},
    {"name": "preserve-thinking-off", "options": {"preserve_thinking": False}, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "  greet back  "},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "Bye!", "reasoning_content": "farewell"}],
     "add_generation_prompt": False},
    {"name": "tools-thinking-off", "options": {"enable_thinking": False}, "tools": [WEATHER_TOOL], "messages": [
        {"role": "user", "content": "Weather in Oslo?"}]},
    {"name": "tool-response-only-user", "messages": [
        {"role": "user", "content": "Real question"},
        {"role": "user", "content": "<tool_response>\nresult\n</tool_response>"}]},
    {"name": "no-user-query", "error": "invalid", "messages": [
        {"role": "user", "content": "<tool_response>x</tool_response>"}]},
    {"name": "system-not-first", "error": "invalid", "messages": [
        {"role": "user", "content": "Hi"}, {"role": "system", "content": "late"}]},
    {"name": "bad-effort", "options": {"reasoning_effort": "extreme"}, "error": "invalid",
     "messages": [{"role": "user", "content": "x"}]},
]


def template_messages(case):
    """The messages as a template sees them (OpenAI shape; arguments an object, or the client's string)."""
    out = []
    for m in case["messages"]:
        msg = {"role": m["role"], "content": m.get("content")}
        if "reasoning_content" in m:
            msg["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            msg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                 for c in m["tool_calls"]]
        out.append(msg)
    return out


def render_template(tokenizer, template: str, case, **override) -> str:
    kwargs = dict(case.get("options", {}))
    kwargs.update(override)
    return tokenizer.apply_chat_template(
        template_messages(case), tools=case.get("tools") or None, chat_template=template,
        add_generation_prompt=case.get("add_generation_prompt", True), tokenize=False, **kwargs)


def add_from_json_filter():
    """transformers compiles templates without a from_json filter, which the
    chat-v2 template applies to string arguments: add Python's json.loads as
    it to transformers' sandboxed environment (the other templates never use
    it, so their renderings are unchanged)."""
    from transformers.utils import chat_template_utils

    base = chat_template_utils.ImmutableSandboxedEnvironment

    class WithFromJson(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.filters["from_json"] = json.loads

    chat_template_utils.ImmutableSandboxedEnvironment = WithFromJson


def check_v2_options(tok, template: str, case, text: str) -> None:
    """chat-v2 reads `thinking` before `enable_thinking` and has no reasoning
    effort: the same text either way."""
    options = case.get("options", {})

    def without(key):
        return {**case, "options": {k: v for k, v in options.items() if k != key}}

    if "enable_thinking" in options:
        alt = render_template(tok, template, without("enable_thinking"), thinking=options["enable_thinking"])
        assert alt == text, f"{case['name']}: thinking and enable_thinking differ"
    if "reasoning_effort" in options:
        alt = render_template(tok, template, without("reasoning_effort"))
        assert alt == text, f"{case['name']}: reasoning_effort changed the text"


def upstream_deepseek(encoding, case):
    """encoding_dsv4.py's text for a case it defines, else None."""
    messages = copy.deepcopy(template_messages(case))
    tools = case.get("tools")
    last_role = messages[-1]["role"]
    gen = case.get("add_generation_prompt", True)
    if gen != (last_role in ("user", "tool")):
        return None  # the encoder always prompts after a user turn, never after an assistant one
    systems = [m for m in messages if m["role"] == "system"]
    if len(systems) > 1 or (systems and messages[0]["role"] != "system"):
        return None
    if tools:
        if not systems:
            return None
        messages[0]["tools"] = tools
    for m in messages:
        for c in m.get("tool_calls", []):
            c["function"]["arguments"] = json.dumps(c["function"]["arguments"], ensure_ascii=False)
        if m["role"] == "user" and m["content"] is None:
            return None
    options = case.get("options", {})
    mode = "thinking" if options.get("enable_thinking") else "chat"
    try:
        return encoding.encode_messages(messages, thinking_mode=mode, reasoning_effort=options.get("reasoning_effort"))
    except (AssertionError, NotImplementedError):
        return None


def chat(args, results_unused=None):
    repo, models = pathlib.Path(args.repo), pathlib.Path(args.models)
    deepseek_dir = models / DEEPSEEK_DIR
    spec = importlib.util.spec_from_file_location("encoding_dsv4", deepseek_dir / "encoding/encoding_dsv4.py")
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)

    ds_template_path = pathlib.Path(args.work) / "dsv4-0731-gguf.jinja"
    ds_template_path.parent.mkdir(parents=True, exist_ok=True)
    ds_template_path.write_bytes(gguf_chat_template(models / DEEPSEEK_GGUF))
    ds_template = ds_template_path.read_text(encoding="utf-8")
    v2_template_path = pathlib.Path(args.work) / "dsv4-community-chat-v2.jinja"
    v2_template_path.write_bytes(gguf_chat_template(models / COMMUNITY_GGUF))
    v2_template = v2_template_path.read_text(encoding="utf-8")
    qwen_template_path = models / QWEN38_DIR / "chat_template.jinja"
    qwen_template = qwen_template_path.read_text(encoding="utf-8")

    ds_tok = transformers.AutoTokenizer.from_pretrained(str(deepseek_dir))
    qwen_tok = transformers.AutoTokenizer.from_pretrained(str(models / QWEN38_DIR))
    add_from_json_filter()

    for label, tok, template, template_path, cases, gguf, hf_config in (
            ("deepseek-v4-0731", ds_tok, ds_template, ds_template_path, COMMON_CASES + DEEPSEEK_CASES, DEEPSEEK_GGUF,
             "deepseek-v4-0731-hf"),
            ("deepseek-v4-chat-v2", ds_tok, v2_template, v2_template_path, COMMON_CASES + DEEPSEEK_V2_CASES,
             COMMUNITY_GGUF, "deepseek-v4-0731-hf"),
            ("qwen3.8", qwen_tok, qwen_template, qwen_template_path, COMMON_CASES + QWEN_CASES, None, "qwen3.8-nvfp4")):
        out_cases, texts = [], {}
        for case in cases:
            entry = {k: case[k] for k in ("name", "messages") if k in case}
            for k in ("tools", "options", "add_generation_prompt"):
                if k in case:
                    entry[k] = case[k]
            try:
                text = render_template(tok, template, case)
            except Exception as e:  # noqa: BLE001 (the template's raise_exception)
                entry["error"] = "invalid"
                entry["reference_error"] = f"{type(e).__name__}: {e}"
                if case.get("error") != "invalid":
                    print(f"{label}/{case['name']}: unexpected error {e}", file=sys.stderr)
                out_cases.append(entry)
                continue
            if case.get("error"):
                # jitLLM refuses what the template renders anyway (an unknown
                # option value); record the template's text beside the error.
                entry["error"] = case["error"]
                entry["reference_text"] = text
                out_cases.append(entry)
                continue
            if label == "deepseek-v4-chat-v2":
                check_v2_options(tok, template, case, text)
            entry["text"] = text
            entry["hf_ids"] = tok.encode(text, add_special_tokens=False)
            if label == "deepseek-v4-0731":
                up = upstream_deepseek(encoding, case)
                entry["upstream_encoder"] = "n/a" if up is None else ("same" if up == text else "differs")
                if up is not None and up != text:
                    entry["upstream_text"] = up
            texts[case["name"]] = text.encode("utf-8")
            out_cases.append(entry)
        if gguf:
            ids = llama_tokenize(models, gguf, texts, pathlib.Path(args.work) / f"chat-{label}")
            for entry in out_cases:
                if entry["name"] in ids:
                    entry["gguf_ids"] = ids[entry["name"]]["parse"]
        renderer = f"transformers {transformers.__version__} apply_chat_template"
        if label == "deepseek-v4-chat-v2":
            renderer += ", with json.loads as the from_json filter"
        write_json(repo / f"tests/unit/data/chat/{label}.json", {
            "template": label,
            "template_sha256": sha256(template_path),
            "renderer": renderer,
            "tokenizer_config": hf_config,
            "cases": out_cases,
        })

    # Qwen-Image's text-to-image prompt, through its processor's tokenizer.
    image_dir = models / "models/Qwen/Qwen-Image-2.1@790c9263/processor"
    image_tok = transformers.AutoTokenizer.from_pretrained(str(image_dir))
    sys_prompt = "Comprehend and analyze the provided prompt."
    template = ("<|im_start|>system\n" + sys_prompt + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n")
    processor_template = (image_dir / "chat_template.jinja").read_text(encoding="utf-8")
    # As the pipeline counts them: the processor template's system turn.
    sys_ids = image_tok.apply_chat_template([{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}],
                                            chat_template=processor_template, tokenize=True, return_dict=False)
    if sys_ids and isinstance(sys_ids[0], list):
        sys_ids = sys_ids[0]
    cases = []
    for name, prompt in (("simple", "A red fox in the snow, watercolor."), ("empty", ""),
                         ("unicode", "一只在雪地里的红狐狸 🦊, Tiếng"),
                         ("braces", "A sign that says {hello} and <think>")):
        text = template.format(prompt if prompt else " ")
        cases.append({"name": name, "prompt": prompt, "text": text,
                      "hf_ids": image_tok.encode(text, add_special_tokens=False)})
    write_json(repo / "tests/unit/data/chat/qwen-image-2.1.json", {
        "template": "qwen-image-2.1-t2i",
        "pipeline": "diffusers 8b3c707e pipelines/qwenimage21/pipeline_qwenimage21.py",
        "drop_tokens": len(sys_ids),
        "cases": cases,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--models", required=True, help="~/.local/share/jitllm on a Spark")
    parser.add_argument("--work", required=True, help="scratch directory")
    parser.add_argument("--only", choices=("tokens", "chat"))
    parser.add_argument("--config", choices=[x[0] for x in CONFIGS],
                        help="generate one token configuration; requires --only tokens")
    args = parser.parse_args()
    if args.config and args.only != "tokens":
        parser.error("--config requires --only tokens")
    if args.config != "gemma-4-26b-gguf":
        global tokenizers, transformers
        import tokenizers
        import transformers
    if args.only != "chat":
        tokens(args)
    if args.only != "tokens":
        chat(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
