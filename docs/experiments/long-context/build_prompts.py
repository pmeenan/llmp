# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Builds the long-context prompts and the perplexity text (corpus.json).

    build_prompts.py --repo DIR --tokenizer FILE --model deepseek|qwen3.8 \
        --out DIR --rung NAME=TOKENS [--rung ...] [--session NAME=TOKENS] \
        [--retrieval NAME=TOKENS ...] [--ppl-source FILE]

Runs on a Spark, in the pinned PyTorch container (its Python has Hugging
Face `tokenizers`), never on the workstation. Deterministic: the same
repository commit, tokenizer file and rungs give the same bytes, and every
input is checked against corpus.json's pins before anything is written.

A rung's prompt is one user message: the preamble, whole source files of the
pinned repository in corpus.json's order, each that still fits the rung's
budget (the rung less `reserve_tokens`, the preamble, the question and the
notes) until the budget is within 1% (256 tokens for a small rung), the
three notes (the retrieval check's needles) inserted before
the first file at or past their fraction of the files' tokens, and the first
question. A session rung (the turn-reuse check) reserves
`session_reserve_tokens` instead, for its later turns. Counts are the given
tokenizer's, without special tokens or a template; the engines' own counts
are recorded with each run.

A --retrieval rung is the retrieval check's own prompt: the same files,
corpus.json's `retrieval` notes (neutral facts: the first prompts' notes
call their values passphrases, which Qwen3.8 declined to repeat at 256K)
and its one question, answered in a few tokens after any reasoning.

--ppl-source cuts the downloaded perplexity text to its body (between the
Project Gutenberg markers) and writes it as ppl.txt.

Writes OUT/<model>/<rung>.json per rung and OUT/<model>/manifest.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
CORPUS = json.loads((HERE / "corpus.json").read_text())


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fail(message):
    raise SystemExit(f"build_prompts: {message}")


def repository_files(repo):
    spec = CORPUS["repository"]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != spec["commit"]:
        fail(f"{repo} is at {head}, not {spec['commit']}")
    if sha256((repo / spec["license_file"]).read_bytes()) != spec["license_sha256"]:
        fail("the repository's LICENSE is not the pinned one")
    taken, files = set(), []
    for directory in spec["directories"]:
        listed = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--", directory],
                                check=True, capture_output=True).stdout.split(b"\0")
        for raw in sorted(p for p in listed if p):
            path = raw.decode()
            if (path in taken or Path(path).suffix not in spec["extensions"]
                    or any(x in "/" + path for x in spec["exclude_containing"])):
                continue
            data = (repo / path).read_bytes()
            if len(data) > spec["max_file_bytes"]:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            taken.add(path)
            files.append((path, text))
    return files


def count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def block(path, text):
    return CORPUS["file_header"].format(path=path) + text


def build(files, counts, tokenizer, rung, reserve, retrieval=False):
    source = CORPUS["retrieval"] if retrieval else CORPUS
    question = source["question"] if retrieval else CORPUS["questions"][0]
    needles = source["needles"]
    notes = [block(n["path"], n["text"]) for n in needles]
    fixed = (count(tokenizer, CORPUS["preamble"]) + count(tokenizer, question)
             + sum(count(tokenizer, n) for n in notes))
    budget = rung - reserve - fixed
    # Files in order, each that still fits the budget (a larger one is
    # skipped), until the budget is within 1% (or 256 tokens, for a small
    # rung) or the counted files run out.
    full = budget - max(budget / 100, 256)
    chosen, total = [], 0
    for i, n in enumerate(counts):
        if total + n <= budget:
            chosen.append(i)
            total += n
        if total >= full:
            break
    if total < full:
        fail(f"the corpus holds too few tokens for a rung of {rung}")
    # Each note before the first chosen file at or past its fraction (after
    # the last when none is).
    starts, at = [], 0
    for i in chosen:
        starts.append(at)
        at += counts[i]
    before = {}
    for needle, note in zip(needles, notes):
        k = next((k for k, s in enumerate(starts) if s >= needle["fraction"] * total),
                 len(chosen))
        before.setdefault(k, []).append((needle, note))
    parts, placed, offset = [CORPUS["preamble"]], [], count(tokenizer, CORPUS["preamble"])
    for k in range(len(chosen) + 1):
        for needle, note in before.get(k, []):
            placed.append({"id": needle["id"], "passphrase": needle["passphrase"],
                           "before_file": files[chosen[k]][0] if k < len(chosen) else None,
                           "fraction": needle["fraction"], "token_offset": offset})
            parts.append(note)
            offset += count(tokenizer, note)
        if k < len(chosen):
            parts.append(block(*files[chosen[k]]))
            offset += counts[chosen[k]]
    parts.append(question)
    content = "".join(parts)
    return content, {"files": len(chosen), "first_file": files[chosen[0]][0],
                     "last_file": files[chosen[-1]][0], "needles": placed,
                     "files_tokens": total}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(CORPUS["tokenizers"]), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rung", action="append", default=[])
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--retrieval", action="append", default=[])
    parser.add_argument("--ppl-source", type=Path)
    args = parser.parse_args()

    pin = CORPUS["tokenizers"][args.model]
    if sha256(args.tokenizer.read_bytes()) != pin["sha256"]:
        fail(f"{args.tokenizer} is not the pinned {args.model} tokenizer")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rungs = [(r, CORPUS["reserve_tokens"], False) for r in args.rung]
    rungs += [(r, CORPUS["session_reserve_tokens"], True) for r in args.session]
    rungs += [(r, CORPUS["reserve_tokens"], "retrieval") for r in args.retrieval]
    out = args.out / args.model
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"corpus": CORPUS["id"],
                "corpus_json_sha256": sha256((HERE / "corpus.json").read_bytes()),
                "repository_commit": CORPUS["repository"]["commit"],
                "model": args.model, "tokenizer_sha256": pin["sha256"], "prompts": []}

    if rungs:
        files = repository_files(args.repo)
        largest = max(int(r.partition("=")[2]) for r, _, _ in rungs)
        counts, total = [], 0
        # Counted only as far as the rungs need (twice the largest, and at
        # least 1M tokens, so a small rung that skips large files still fills).
        for path, text in files:
            if total > max(2 * largest, 1 << 20):
                break
            counts.append(count(tokenizer, block(path, text)))
            total += counts[-1]
        files = files[:len(counts)]
        for spec, reserve, kind in rungs:
            name, _, tokens = spec.partition("=")
            retrieval = kind == "retrieval"
            session = kind is True
            content, info = build(files, counts, tokenizer, int(tokens), reserve, retrieval)
            data = content.encode()
            record = {"id": f"{args.model}-{name}", "model": args.model, "rung": name,
                      "rung_tokens": int(tokens), "reserve_tokens": reserve,
                      "session": session, "retrieval": retrieval,
                      "content_tokens": count(tokenizer, content),
                      "content_sha256": sha256(data), **info,
                      "messages": [{"role": "user", "content": content}],
                      "followups": CORPUS["questions"][1:] if session else [],
                      "generate_tokens": (CORPUS["retrieval"]["generate_tokens"] if retrieval
                                          else CORPUS["generate_tokens"])}
            (out / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False) + "\n")
            manifest["prompts"].append({k: v for k, v in record.items()
                                        if k not in ("messages", "followups")})
            print(f"{record['id']}: {record['content_tokens']} tokens, {info['files']} files, "
                  f"sha256 {record['content_sha256'][:16]}", flush=True)

    if args.ppl_source:
        spec = CORPUS["perplexity"]
        raw = args.ppl_source.read_bytes()
        if sha256(raw) != spec["sha256"]:
            fail(f"{args.ppl_source} is not the pinned perplexity text")
        text = raw.decode("utf-8-sig")
        body = text.split(spec["start_marker"], 1)[1].split(spec["end_marker"], 1)[0].strip()
        body = body.replace("\r\n", "\n") + "\n"
        (args.out / "ppl.txt").write_text(body)
        manifest["ppl"] = {"sha256": sha256(body.encode()), "bytes": len(body.encode()),
                           "tokens": count(tokenizer, body)}
        print(f"ppl.txt: {manifest['ppl']}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
