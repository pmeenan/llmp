#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Collects the chat-template corpus and renders its references (README.md).

    collect.py fetch  --out DIR [--models DIR] [--only NAME...]
    collect.py render --out DIR --conversations FILE

`fetch` downloads, read-only over HTTPS, each corpus entry's chat template
at a pinned revision: chat_template.jinja or tokenizer_config.json's
chat_template from a Hugging Face repository, or tokenizer.chat_template
from the metadata at the head of a GGUF file (an HTTP range read; no
weights are fetched). Local GGUF and template files under --models are read
in place. It writes DIR/templates/<name>.jinja and DIR/manifest.json (repo,
revision, file, SHA-256, BOS and EOS texts, the repository's declared
license). With --only it fetches just the named entries and merges them
into the existing manifest, leaving the others' pinned revisions as they
were.

`render` renders every template on every conversation of the committed
conversation corpus with transformers' render_jinja_template (the function
apply_chat_template calls; Jinja2's sandbox, trim_blocks and lstrip_blocks,
transformers' tojson, raise_exception and strftime_now), with the bos_token
and eos_token texts the manifest records, `tools` and `documents` None when
absent, and strftime_now fixed at 2026-10-02 12:34:56. It writes
DIR/references.json: per template and conversation, the text or the
exception. Template texts and renderings stay outside the repository (their
licenses vary); the test reads them from DIR (tests/unit/chat_corpus_test.cc).
"""

import sys

sys.dont_write_bytecode = True

import argparse
import datetime
import hashlib
import json
import pathlib
import urllib.request

HF = "https://huggingface.co"

# (name, repo, kind, file or None). kind: "hf" (chat_template.jinja, else
# tokenizer_config.json), "gguf" (the named file's metadata, over HTTP), or
# "local" (a path under --models: a GGUF or a template file).
CORPUS = [
    ("deepseek-v4-0731-gguf", "unsloth/DeepSeek-V4-Flash-0731-GGUF", "local",
     "models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf"),
    ("deepseek-v4-chat-v2-gguf", "antirez/deepseek-v4-gguf", "local",
     "models/antirez/deepseek-v4-gguf@f71f23d5/"
     "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"),
    ("qwen3.8-flash-next-nvfp4", "Mia-AiLab/Qwen3.8-Flash-Next-NVFP4", "local",
     "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/chat_template.jinja"),
    ("qwen3.8-flash-next-unsloth-gguf", "unsloth/Qwen3.8-Flash-Next-GGUF", "local",
     "reference-models/Q/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"),
    ("qwen3.8-flash-next", "Qwen/Qwen3.8-Flash-Next", "hf", None),
    ("qwen3.8-27b", "Qwen/Qwen3.8-27B", "hf", None),
    ("qwen3.8-27b-unsloth-gguf", "unsloth/Qwen3.8-27B-GGUF", "gguf", None),
    ("qwen3-8b", "Qwen/Qwen3-8B", "hf", None),
    ("qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct", "hf", None),
    ("qwen3-coder-30b-a3b", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "hf", None),
    ("gemma-4-26b-a4b-it", "google/gemma-4-26B-A4B-it", "hf", None),
    ("gemma-4-31b-it", "google/gemma-4-31B-it", "hf", None),
    ("gemma-3-4b-it", "unsloth/gemma-3-4b-it", "hf", None),
    ("gemma-4-e4b-it", "google/gemma-4-E4B-it", "hf", None),
    ("gemma-4-26b-a4b-it-fp8", "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic", "hf", None),
    ("gemma-4-31b-it-nvfp4", "nvidia/Gemma-4-31B-IT-NVFP4", "hf", None),
    ("gemma-4-31b-it-unsloth-gguf", "unsloth/gemma-4-31B-it-GGUF", "gguf", None),
    ("gemma-4-e4b-it-unsloth-gguf", "unsloth/gemma-4-E4B-it-GGUF", "gguf", None),
    ("gemma-4-e4b-it-lmstudio-gguf", "lmstudio-community/gemma-4-E4B-it-GGUF", "gguf", None),
    ("gemma-3-1b-it-gguf", "ggml-org/gemma-3-1b-it-GGUF", "gguf", None),
    ("gemma-3n-e4b-it", "unsloth/gemma-3n-E4B-it", "hf", None),
    ("gemma-3-270m-it", "unsloth/gemma-3-270m-it", "hf", None),
    ("gpt-oss-120b", "openai/gpt-oss-120b", "hf", None),
    ("kimi-linear-48b-a3b", "moonshotai/Kimi-Linear-48B-A3B-Instruct", "hf", None),
    ("kimi-k2-instruct", "moonshotai/Kimi-K2-Instruct", "hf", None),
    ("nemotron-3-super-120b", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "hf", None),
    ("mistral-small-4-119b", "mistralai/Mistral-Small-4-119B-2603", "hf", None),
    ("mistral-small-3.2-24b", "unsloth/Mistral-Small-3.2-24B-Instruct-2506", "hf", None),
    ("llama-4-scout", "unsloth/Llama-4-Scout-17B-16E-Instruct", "hf", None),
    ("llama-3.3-70b", "unsloth/Llama-3.3-70B-Instruct", "hf", None),
    ("llama-3.2-3b", "unsloth/Llama-3.2-3B-Instruct", "hf", None),
    ("glm-4.7-flash", "zai-org/GLM-4.7-Flash", "hf", None),
    ("mimo-v2.6-flash", "XiaomiMiMo/MiMo-V2.6-Flash-RL", "hf", None),
    ("phi-4-mini", "microsoft/Phi-4-mini-instruct", "hf", None),
    ("phi-4", "microsoft/phi-4", "hf", None),
    ("deepseek-v3.1", "deepseek-ai/DeepSeek-V3.1", "hf", None),
    ("smollm3-3b", "HuggingFaceTB/SmolLM3-3B", "hf", None),
    ("granite-4.1-8b", "ibm-granite/granite-4.1-8b", "hf", None),
]

NOW = datetime.datetime(2026, 10, 2, 12, 34, 56)


def get(url: str, headers=None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def repo_info(repo: str) -> dict:
    return json.loads(get(f"{HF}/api/models/{repo}"))


class Reader:
    def __init__(self, data: bytes):
        self.data, self.at = data, 0

    def take(self, n: int) -> bytes:
        if self.at + n > len(self.data):
            raise EOFError("GGUF metadata past the fetched range")
        b = self.data[self.at:self.at + n]
        self.at += n
        return b

    def u(self, n: int) -> int:
        return int.from_bytes(self.take(n), "little")

    def string(self) -> bytes:
        return self.take(self.u(8))


SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_metadata(data: bytes) -> dict:
    """The template, BOS/EOS IDs and token texts from a GGUF file's head."""
    r = Reader(data)
    if r.take(4) != b"GGUF":
        raise ValueError("not GGUF")
    _version, _tensors, keys = r.u(4), r.u(8), r.u(8)
    out = {}

    def value(t):
        if t in SIZES:
            raw = r.take(SIZES[t])
            return int.from_bytes(raw, "little") if t in (0, 2, 4, 10) else raw
        if t == 8:
            return r.string()
        if t == 9:
            et, n = r.u(4), r.u(8)
            if et == 8:
                return [r.string() for _ in range(n)]
            if et in SIZES:
                r.take(SIZES[et] * n)
                return None
            return [value(et) for _ in range(n)]
        raise ValueError(f"GGUF type {t}")

    for _ in range(keys):
        key, t = r.string().decode(), r.u(4)
        v = value(t)
        if key in ("tokenizer.chat_template", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                   "tokenizer.ggml.tokens"):
            out[key] = v
    tokens = out.get("tokenizer.ggml.tokens") or []

    def text(k):
        i = out.get(k)
        return tokens[i].decode("utf-8") if i is not None and i < len(tokens) else None

    if "tokenizer.chat_template" not in out:
        raise ValueError("no chat template")
    return {"template": out["tokenizer.chat_template"], "bos": text("tokenizer.ggml.bos_token_id"),
            "eos": text("tokenizer.ggml.eos_token_id")}


def token_text(v):
    if isinstance(v, dict):
        return v.get("content")
    return v


def fetch(args) -> int:
    out = pathlib.Path(args.out)
    (out / "templates").mkdir(parents=True, exist_ok=True)
    models = pathlib.Path(args.models) if args.models else None
    manifest = {}
    if args.only:  # add or refresh only these entries; keep the rest of the manifest
        manifest = json.loads((out / "manifest.json").read_text())
    for name, repo, kind, file in CORPUS:
        if args.only and name not in args.only:
            continue
        try:
            info = repo_info(repo)
            rev = info["sha"]
            license_ = (info.get("cardData") or {}).get("license")
            bos = eos = None
            if kind == "local":
                path = models / file
                if path.suffix == ".gguf":
                    with path.open("rb") as f:
                        meta = gguf_metadata(f.read(64 << 20))
                    template, bos, eos = meta["template"], meta["bos"], meta["eos"]
                else:
                    template = path.read_bytes()
                    cfg = path.parent / "tokenizer_config.json"
                    if cfg.exists():
                        c = json.loads(cfg.read_text())
                        bos, eos = token_text(c.get("bos_token")), token_text(c.get("eos_token"))
                source = file
                rev = file.split("@")[1].split("/")[0] if "@" in file else rev
            elif kind == "gguf":
                files = sorted(s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".gguf"))
                pick = [f for f in files if "Q4_K_M" in f and "mmproj" not in f] or files
                source = file or pick[0]
                head = get(f"{HF}/{repo}/resolve/{rev}/{source}", {"Range": "bytes=0-67108863"})
                meta = gguf_metadata(head)
                template, bos, eos = meta["template"], meta["bos"], meta["eos"]
            else:
                files = {s["rfilename"] for s in info["siblings"]}
                cfg = {}
                if "tokenizer_config.json" in files:
                    cfg = json.loads(get(f"{HF}/{repo}/resolve/{rev}/tokenizer_config.json"))
                bos, eos = token_text(cfg.get("bos_token")), token_text(cfg.get("eos_token"))
                if "chat_template.jinja" in files:
                    source = "chat_template.jinja"
                    template = get(f"{HF}/{repo}/resolve/{rev}/{source}")
                elif isinstance(cfg.get("chat_template"), str):
                    source = "tokenizer_config.json"
                    template = cfg["chat_template"].encode("utf-8")
                elif isinstance(cfg.get("chat_template"), list):
                    source = "tokenizer_config.json (default)"
                    template = next(t["template"] for t in cfg["chat_template"] if t["name"] == "default").encode()
                elif "chat_template.json" in files:
                    source = "chat_template.json"
                    template = json.loads(get(f"{HF}/{repo}/resolve/{rev}/{source}"))["chat_template"].encode()
                else:
                    raise ValueError("no chat template")
            (out / "templates" / f"{name}.jinja").write_bytes(template)
            manifest[name] = {"repo": repo, "revision": rev, "source": source,
                              "sha256": hashlib.sha256(template).hexdigest(), "bytes": len(template),
                              "bos_token": bos, "eos_token": eos, "license": license_}
            print(f"{name}: {len(template)} bytes from {repo}@{rev[:8]} {source}")
        except Exception as e:  # noqa: BLE001 (a missing entry is recorded and skipped)
            print(f"{name}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return 0


def template_messages(conv):
    """The messages as llmpalooza hands a template (chat/template.cc): role and
    content, reasoning_content when given, OpenAI-shaped tool calls with
    object arguments when any."""
    out = []
    for m in conv["messages"]:
        msg = {"role": m["role"], "content": m.get("content")}
        if "reasoning_content" in m:
            msg["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            msg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                 for c in m["tool_calls"]]
        out.append(msg)
    return out


def render(args) -> int:
    import transformers
    from transformers.utils import chat_template_utils

    class FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    chat_template_utils.datetime = FixedNow
    out = pathlib.Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    conversations = json.loads(pathlib.Path(args.conversations).read_text())["conversations"]
    results = {"renderer": f"transformers {transformers.__version__} render_jinja_template",
               "now": NOW.isoformat(), "templates": {}}
    for name, entry in manifest.items():
        template = (out / "templates" / f"{name}.jinja").read_text(encoding="utf-8")
        cases = {}
        for conv in conversations:
            kwargs = dict(conv.get("options", {}))
            if "enable_thinking" in kwargs:
                kwargs["thinking"] = kwargs["enable_thinking"]  # as llmpalooza passes it (chat/template.cc)
            if entry["bos_token"] is not None:
                kwargs["bos_token"] = entry["bos_token"]
            if entry["eos_token"] is not None:
                kwargs["eos_token"] = entry["eos_token"]
            try:
                rendered, _ = chat_template_utils.render_jinja_template(
                    conversations=[template_messages(conv)], tools=conv.get("tools"), chat_template=template,
                    add_generation_prompt=conv.get("add_generation_prompt", True), **kwargs)
                cases[conv["name"]] = {"text": rendered[0]}
            except Exception as e:  # noqa: BLE001 (the template's own exceptions are results)
                cases[conv["name"]] = {"error": f"{type(e).__name__}: {e}"[:300]}
        ok = sum("text" in c for c in cases.values())
        print(f"{name}: {ok}/{len(cases)} rendered")
        results["templates"][name] = {"sha256": entry["sha256"], "cases": cases}
    (out / "references.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["fetch", "render"])
    p.add_argument("--out", required=True)
    p.add_argument("--models")
    p.add_argument("--conversations")
    p.add_argument("--only", nargs="+", help="fetch: only these entries, merged into the manifest")
    args = p.parse_args()
    return fetch(args) if args.command == "fetch" else render(args)


if __name__ == "__main__":
    sys.exit(main())
