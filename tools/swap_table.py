#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""M3's swap table and the Gemma near-8K switch pairs: a parameterized run and its checker.

The table is `llmp-runtime swap-table` (docs/runtime-serving.md, "The commands"; the table's
rules, docs/m3-record.md, "Swap time"): 32 rows over the six ordered pairs of DeepSeek,
Qwen3.8 and the image model, A->B->A, first use and prepared, with 8K and 0 tokens of A's
context. The near-8K cases are one Gemma model against Qwen3.8 (docs/experiments/m36-baselines/
README.md, "The gate's tools"). This tool replaces the baseline's `swaps.sh` and
`swaps-fixed.sh`: the runtime, the artifact store, the artifact IDs and the image pin are
inputs, so the same table runs on the old and the new runtime, or on re-imported artifacts.

    swap_table.py plan  --case CASE --runtime R --store DIR --out DIR [--ids FILE]
                        [--image-pin SHA256] [--pairs-per-job N] [--share DIR] [--inputs DIR]
                        [--config-template FILE] > STEPS
    swap_table.py run   --case CASE --runtime R --store DIR --out DIR --name NAME
                        [--pairs A:B,...] [same options]
    swap_table.py check DIR... --case CASE [--prepared-limit 20] [--first-use-limit 40]
                        [--partial] [--json]

CASE is `m3` or `gemma2`, `gemma3`, `gemma26`, `gemma31` (the near-8K pair, one job each).
`plan` prints a spark-job steps file (`spark-job start --gpu --steps FILE`; one command a
line): the table split into runs of at most PAIRS-PER-JOB ordered pairs (default 2, so each
run stays inside a 600 s job; a run is a whole process, as the baseline's two-run split
was) and the printed commands call `run`. `run` writes OUT/NAME/{node.toml, table.out,
table.log, report.json, runtime.sha256, mem-before.txt}, verifies the pinned inputs first,
refuses an existing OUT/NAME, removes the run's data directory afterwards and exits with the
runtime's status.

The node configuration is rendered from the case with the artifact IDs of `--ids`, a JSON
object over the defaults, by model name. The defaults are the v1 imports serving uses (D-118,
`--store` the store holding them; by default LLMP_TEST_V1_STORE, else `artifacts`). Each entry
may hold `artifact`, `drafter`, `composition`, `tokenizer`, `chat_template` and `settings`
(extra keys for the model's table, e.g. what a newer runtime needs):

    {"deepseek": {"artifact": "...", "drafter": "..."},
     "qwen3.8": {"artifact": "...", "settings": {"depth_cost_ratio": 1.16}},
     "image": {"composition": "..."}, "gemma31": {"artifact": "..."}}

`--config-template FILE` replaces the rendering: a whole configuration with {share}, {data}
and {installed}. `--store` is the installed-artifacts directory. `--share` (default
~/.local/share/llmp) holds the Qwen tokenizer files and the image latents; `--inputs`
(default SHARE/m36-baselines/inputs) the context texts. Their SHA-256 are pinned.

`check` reads the run directories of one table (all the jobs of a case) and fails (exit 1) unless
  * every row of the case is present once, each exact and state_exact, and the report's
    `problems` is empty;
  * every B->A row with context restored A's state with the hash it left with, its
    continuation equal token and logit to the unswapped one (reference and continued logits
    hashes equal, one logit row and one token a continued token), the report's --continue
    tokens (16 for M3, 64 for the Gemma pairs) over the full target vocabulary;
  * every prepared B->A row of an LLM replayed graphs captured before it (graphs kept, none
    captured or run eagerly, with 8K context some replayed);
  * every image control (`control: image generated, pixels X` in table.out) and the image
    A's regenerations equal the image pin, and an image A has its control line;
  * prepared rows take at most --prepared-limit seconds and first-use rows
    --first-use-limit (the 20 s bound and the 40 s first-use target);
  * the runs agree on the runtime binary (runtime.sha256), the context text hash and the
    case's context and continuation lengths.
`--partial` accepts missing rows (checking those present; exit 2 when only rows are missing).
It prints each row's endpoint seconds and tracked peak_bytes, the worst LLM<->LLM row and
the worst prepared one (the table's headline), and the lowest MemAvailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import speed_gate as sg  # noqa: E402  (the artifact IDs, the image pins and the share layout)

DEFAULT_SHARE = "~/.local/share/llmp"
CONTEXT_M3_SHA256 = "6b159ff20d198a3ff825d78b5edc0c2bd8c9cf5af7222875fb88301635e14db5"
CONTEXT_M3 = "decisions-4655685.md"
CONTEXT_GEMMA_SHA256 = "4e41c6aac01454db43f44522999d2dff87098891cd87f7737ab7a0a58d636a01"
CONTEXT_GEMMA = "gemma-context.txt"
SHORT_PROMPT = "What is the capital of France? Answer in one sentence."
M3_PAIRS = ("qwen3.8:deepseek", "qwen3.8:image", "deepseek:image", "deepseek:qwen3.8",
            "image:deepseek", "image:qwen3.8")
PREPARED_LIMIT = 20.0  # seconds: the bound for previously prepared swaps
FIRST_USE_LIMIT = 40.0  # seconds: first use is no worse than about twice the bound
JOB_SECONDS = 600
# The published image composition's pixels, the pin its frozen test holds
# (tests/unit/data/v1_frozen/qwen_image.json, docs/model-support.md): the image
# A's control and regenerations must equal it. speed_gate's IMAGE_PIXELS_SHA256
# stays the M3.6 protocol's frozen old-arm pin.
IMAGE_PIN = json.loads((pathlib.Path(__file__).resolve().parent.parent /
                        "tests/unit/data/v1_frozen/qwen_image.json").read_text())[
    "digests"]["pixels.1024x1024.40"]

QWEN_FILES = {"tokenizer": f"{{share}}/{sg.QWEN_TOKENIZER}/tokenizer.json",
              "chat_template": f"{{share}}/{sg.QWEN_TOKENIZER}/chat_template.jinja"}
# The v1 imports serving uses (D-118).
DEFAULT_IDS = {
    "deepseek": {"artifact": sg.DSV4_V1, "drafter": sg.DSPARK_V1},
    "qwen3.8": {"artifact": sg.QWEN_V1, "drafter": sg.QWEN_MTP_V1, **QWEN_FILES},
    "image": {"composition": sg.IMAGE_V1},
    "gemma2": {"artifact": sg.GEMMA2_V1},
    "gemma3": {"artifact": sg.GEMMA3_V1},
    "gemma26": {"artifact": sg.GEMMA26_V1},
    "gemma31": {"artifact": sg.GEMMA31_V1},
}


class Case:
    def __init__(self, name, models, pairs, context_file, context_sha256, context_tokens,
                 continue_tokens, model_context=None, chunk=None):
        self.name, self.models, self.pairs = name, models, pairs
        self.context_file, self.context_sha256 = context_file, context_sha256
        self.context_tokens, self.continue_tokens = context_tokens, continue_tokens
        self.model_context, self.chunk = model_context, chunk


CASES = {"m3": Case("m3", ("deepseek", "qwen3.8", "image"), M3_PAIRS, CONTEXT_M3,
                    CONTEXT_M3_SHA256, 8192, 16)}
for _name, _context, _chunk, _tokens in (("gemma26", 8192, 1024, 8063),
                                         ("gemma31", 8192, 256, 8063),
                                         ("gemma2", 8192, 128, 8063),
                                         ("gemma3", 8448, 128, 8192)):
    CASES[_name] = Case(_name, (_name, "qwen38"), (f"{_name}:qwen38",), CONTEXT_GEMMA,
                        CONTEXT_GEMMA_SHA256, _tokens, 64, _context, _chunk)


# ------------------------------------------------------------------ the configuration


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise SystemExit(f"a configuration setting must be a string, number or boolean: {value!r}")


def load_ids(path: str | None) -> dict:
    ids = {k: dict(v) for k, v in DEFAULT_IDS.items()}
    if path:
        given = json.loads(pathlib.Path(path).expanduser().read_text())
        if not isinstance(given, dict):
            raise SystemExit(f"{path}: not a JSON object of model entries")
        for model, entry in given.items():
            if not isinstance(entry, dict):
                raise SystemExit(f"{path}: {model}: not an object")
            unknown = set(entry) - {"artifact", "drafter", "composition", "tokenizer",
                                    "chat_template", "settings"}
            if unknown:
                raise SystemExit(f"{path}: {model}: unknown keys {sorted(unknown)}")
            ids.setdefault(model, {}).update(entry)
    return ids


def model_table(name: str, entry: dict, settings: dict) -> str:
    """One [models.NAME] table: the entry's identity keys, then the case's settings, then the
    entry's own `settings` (which win)."""
    head = f'[models."{name}"]\n' if "." in name else f"[models.{name}]\n"
    merged = dict(settings)
    merged.update(entry.get("settings") or {})
    lines = []
    for key in ("artifact", "drafter", "composition"):
        if entry.get(key):
            lines.append((key, entry[key]))
    lines += list(merged.items())
    for key in ("tokenizer", "chat_template"):
        if entry.get(key):
            lines.append((key, entry[key]))
    return head + "".join(f"{k} = {toml_value(v)}\n" for k, v in lines)


def render_config(case: Case, ids: dict, store: str, data: str, share: str,
                  template: str | None = None) -> str:
    if template is not None:
        for key in ("{data}", "{installed}"):
            if key not in template:
                raise SystemExit(f"the configuration template has no {key}")
        return (template.replace("{share}", share).replace("{data}", data)
                .replace("{installed}", store))
    text = f'schema_version = 2\n[storage]\ndata_dir = "{data}"\ninstalled = "{store}"\n'
    if case.name == "m3":
        text += model_table("deepseek", ids["deepseek"],
                            {"wave_form": "speculative", "max_slots": 4, "speculation": True,
                             "context": 262144, "prefill_chunk": 4096})
        text += model_table("qwen3.8", ids["qwen3.8"],
                            {"max_slots": 4, "draft_vocab": 65536, "speculation": True,
                             "context": 33792, "prefill_chunk": 4096})
        text += model_table("image", ids["image"], {})
    else:
        text += model_table(case.name, ids[case.name],
                            {"context": case.model_context, "prefill_chunk": case.chunk,
                             "max_slots": 1})
        qwen = {k: v for k, v in ids["qwen3.8"].items() if k != "drafter"}
        text += model_table("qwen38", qwen, {"context": case.model_context,
                                             "prefill_chunk": 4096, "max_slots": 1})
    return text.replace("{share}", share)


# ------------------------------------------------------------------ plan and run


def job_pairs(case: Case, per_job: int) -> list[list[str]]:
    if per_job < 1:
        raise SystemExit("--pairs-per-job must be at least 1")
    return [list(case.pairs[i:i + per_job]) for i in range(0, len(case.pairs), per_job)]


def common_options(args: argparse.Namespace) -> list[str]:
    options = ["--case", args.case, "--runtime", args.runtime, "--store", args.store,
               "--share", args.share, "--out", args.out]
    for flag, value in (("--ids", args.ids), ("--inputs", args.inputs),
                        ("--image-pin", args.image_pin),
                        ("--config-template", args.config_template)):
        if value:
            options += [flag, value]
    return options


def resolve_store(store: str | None, share: pathlib.Path) -> str:
    """`--store` (relative names are under `share`), else the one standing v1 store:
    LLMP_TEST_V1_STORE, else `artifacts` under `share` (speed_gate.v1_store)."""
    if not store:
        return sg.v1_store(share)
    path = pathlib.Path(store).expanduser()
    return str(path if path.is_absolute() else share / path)


def cmd_plan(args: argparse.Namespace) -> int:
    case = CASES[args.case]
    share = pathlib.Path(args.share).expanduser().resolve()
    args.store = resolve_store(args.store, share)
    me = shlex.quote(str(pathlib.Path(__file__).resolve()))
    options = " ".join(shlex.quote(o) for o in common_options(args))
    jobs = job_pairs(case, args.pairs_per_job)
    for number, pairs in enumerate(jobs, start=1):
        name = args.case if len(jobs) == 1 else f"{args.case}-{number}"
        print(f"python3 -B {me} run {options} --name {shlex.quote(name)} "
              f"--pairs {shlex.quote(','.join(pairs))}")
    return 0


def sha256_file(path: pathlib.Path) -> str:
    return sg.sha256_file(path)


def pinned(path: pathlib.Path, digest: str) -> None:
    if not path.exists() or sha256_file(path) != digest:
        raise SystemExit(f"{path}: not the pinned input ({digest[:12]}...)")


def swap_command(case: Case, args: argparse.Namespace, run_dir: pathlib.Path, pairs: str,
                 share: pathlib.Path, inputs: pathlib.Path) -> list[str]:
    command = [str(args.runtime), "--config", str(run_dir / "node.toml"),
               "--anchor", str(run_dir / "enrollment"), "swap-table", "--pairs", pairs,
               "--context-text", str(inputs / case.context_file),
               "--context-tokens", str(case.context_tokens),
               "--continue", str(case.continue_tokens), "--cycles", "2",
               "--zero-context", "on", "--handoff", "on"]
    if case.name == "m3":
        command += ["--short-prompt", SHORT_PROMPT, "--image-prompt", sg.IMAGE_PROMPT,
                    "--image-noise", str(share / sg.IMAGE_NOISE),
                    "--image-expect", args.image_pin or IMAGE_PIN]
    else:
        command += ["--plain"]
    return command + ["--report", str(run_dir / "report.json")]


def cmd_run(args: argparse.Namespace) -> int:
    case = CASES[args.case]
    share = pathlib.Path(args.share).expanduser().resolve()
    args.store = resolve_store(args.store, share)
    inputs = pathlib.Path(args.inputs).expanduser().resolve() if args.inputs else (
        share / "m36-baselines" / "inputs")
    runtime = pathlib.Path(args.runtime).expanduser().resolve()
    args.runtime = str(runtime)
    pairs = args.pairs or ",".join(case.pairs)
    unknown = [p for p in pairs.split(",") if p not in case.pairs]
    if unknown:
        raise SystemExit(f"{args.case}: pairs {unknown} are not in {list(case.pairs)}")
    pinned(inputs / case.context_file, case.context_sha256)
    if case.name == "m3":
        pinned(share / sg.IMAGE_NOISE, sg.IMAGE_NOISE_SHA256)
    run_dir = pathlib.Path(args.out).expanduser().resolve() / args.name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists")
    run_dir.mkdir(parents=True)
    os.chmod(run_dir, 0o700)
    template = (pathlib.Path(args.config_template).expanduser().read_text()
                if args.config_template else None)
    (run_dir / "node.toml").write_text(render_config(
        case, load_ids(args.ids), str(pathlib.Path(args.store).expanduser().resolve()),
        str(run_dir / "data"), str(share), template))
    (run_dir / "runtime.sha256").write_text(f"{sha256_file(runtime)}  {runtime}\n")
    try:
        meminfo = pathlib.Path("/proc/meminfo").read_text()
        (run_dir / "mem-before.txt").write_text(
            "".join(line + "\n" for line in meminfo.splitlines() if "MemAvailable" in line))
    except OSError:
        pass
    command = swap_command(case, args, run_dir, pairs, share, inputs)
    env = dict(os.environ, CUDA_DISABLE_PTX_JIT="1")
    with open(run_dir / "table.out", "wb") as out, open(run_dir / "table.log", "wb") as log:
        proc = subprocess.Popen(command, stdout=out, stderr=log, env=env, start_new_session=True)
        try:
            status = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            print(f"{args.name}: timed out after {args.timeout} s", file=sys.stderr)
            status = 124
    subprocess.run(["rm", "-rf", str(run_dir / "data")], check=False)
    print(f"{args.name} {'done' if status == 0 else f'failed ({status})'}")
    return status


# ------------------------------------------------------------------ the checker


ROW = re.compile(r"^(A->B|B->A), (first use|prepared) \((?:(\d+) context tokens|(0) context|"
                 r"(image))\)$")
CONTROL = re.compile(r"^control: image generated, pixels ([0-9a-f]+)\s*$", re.M)


def parse_row(name: str):
    m = ROW.match(name)
    if not m:
        return None
    return {"direction": m.group(1), "kind": m.group(2),
            "context": int(m.group(3)) if m.group(3) else 0, "image": bool(m.group(5))}


def expected_rows(case: Case, tokens: int) -> list[tuple[str, str]]:
    """[(pair label, row name)] of the case: six rows for an LLM A (8K and 0 context), four
    for an image A."""
    rows = []
    for pair in case.pairs:
        a, b = pair.split(":")
        label = f"{a} <-> {b}"
        if a == "image":
            end = "(image)"
            names = [f"{d}, {k} {end}" for k in ("first use", "prepared")
                     for d in ("A->B", "B->A")]
            names = [names[0], names[1], names[2], names[3]]
        else:
            end = f"({tokens} context tokens)"
            names = [f"A->B, first use {end}", f"B->A, first use {end}",
                     f"A->B, prepared {end}", f"B->A, prepared {end}",
                     "A->B, prepared (0 context)", "B->A, prepared (0 context)"]
        rows += [(label, name) for name in names]
    return rows


def is_hex(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check_row(row: dict, report: dict, case: Case, prepared_limit: float,
              first_use_limit: float) -> list[str]:
    """The reasons a row fails, [] if it passes."""
    where = f"{row['pair']} | {row['name']}"
    parsed = parse_row(row["name"])
    if parsed is None:
        return [f"{where}: unrecognized row name"]
    problems = []
    if row.get("exact") is not True:
        problems.append(f"{where}: not exact")
    if row.get("state_exact") is not True:
        problems.append(f"{where}: state not exact")
    limit = prepared_limit if parsed["kind"] == "prepared" else first_use_limit
    if not row["total"] <= limit:
        problems.append(f"{where}: {row['total']:.3f} s over the {limit:g} s "
                        f"{parsed['kind']} limit")
    a_llm = not row["pair"].startswith("image")
    graphs = row.get("graphs") or {}
    if parsed["direction"] == "B->A" and a_llm and parsed["context"] > 0:
        continued = report.get("continue")
        before, after = row.get("state_before_sha256"), row.get("state_restored_sha256")
        if not is_hex(before) or before != after:
            problems.append(f"{where}: the restored state hash {str(after)[:12]!r} is not "
                            f"the one A left with {str(before)[:12]!r}")
        if row.get("initialized_state_bytes", 0) <= 0:
            problems.append(f"{where}: no initialized state was restored")
        ref, cont = row.get("reference_logits_sha256"), row.get("continued_logits_sha256")
        if not is_hex(ref) or ref != cont:
            problems.append(f"{where}: the continued logits {str(cont)[:12]!r} differ from the "
                            f"unswapped continuation's {str(ref)[:12]!r}")
        if row.get("continuation_logit_rows") != continued:
            problems.append(f"{where}: {row.get('continuation_logit_rows')} continuation logit "
                            f"rows, not {continued}")
        if not isinstance(row.get("continuation_vocabulary"), int) or \
                row["continuation_vocabulary"] <= 0:
            problems.append(f"{where}: no target-vocabulary logit rows")
        tokens = str(row.get("output", "")).split(",")
        if not all(t.isdigit() for t in tokens) or len(tokens) != continued:
            problems.append(f"{where}: the continuation is not {continued} tokens")
    if parsed["direction"] == "B->A" and a_llm and parsed["kind"] == "prepared":
        if not row.get("graphs_kept", 0) > 0:
            problems.append(f"{where}: no graphs kept through the swap")
        if parsed["context"] > 0 and not (graphs.get("replayed", 0) > 0 and
                                          graphs.get("captured", 0) == 0 and
                                          graphs.get("eager", 0) == 0):
            problems.append(f"{where}: prepared graphs not replayed ({graphs})")
    return problems


def check(dirs: list[str], case_name: str, prepared_limit: float = PREPARED_LIMIT,
          first_use_limit: float = FIRST_USE_LIMIT, image_pin: str | None = None,
          partial: bool = False) -> dict:
    case = CASES[case_name]
    pin = image_pin or IMAGE_PIN
    problems: list[str] = []
    rows: list[dict] = []
    runtimes, contexts = set(), set()
    controls: list[str] = []
    mem_low = None
    for d in dirs:
        base = pathlib.Path(d)
        try:
            report = json.loads((base / "report.json").read_text())
        except (OSError, ValueError) as e:
            problems.append(f"{d}: no readable report.json ({e})")
            continue
        try:
            runtimes.add((base / "runtime.sha256").read_text().split()[0])
        except (OSError, IndexError):
            problems.append(f"{d}: no runtime.sha256")
        # A run whose A is only the image never reads the context text (the
        # runtime hashes what it read: nothing), as the planned job of the
        # image A pairs does; only runs with an LLM A must agree on it.
        if any(not (parse_row(r.get("name", "")) or {}).get("image")
               for r in report.get("rows") or []):
            contexts.add(report.get("context_sha256"))
        if report.get("context_tokens") != case.context_tokens:
            problems.append(f"{d}: {report.get('context_tokens')} context tokens, not "
                            f"{case.context_tokens}")
        if report.get("continue") != case.continue_tokens:
            problems.append(f"{d}: --continue {report.get('continue')}, not "
                            f"{case.continue_tokens}")
        for p in report.get("problems") or []:
            problems.append(f"{d}: runtime problem: {p}")
        if report.get("rows") is None:
            problems.append(f"{d}: no rows")
        for row in report.get("rows") or []:
            row = dict(row, source=d)
            rows.append(row)
            problems += check_row(row, report, case, prepared_limit, first_use_limit)
        low = report.get("mem_available_low")
        if isinstance(low, int):
            mem_low = low if mem_low is None else min(mem_low, low)
        table = base / "table.out"
        if table.exists():
            text = table.read_text()
        else:
            text = ""
            problems.append(f"{d}: no table.out")
        found = CONTROL.findall(text)
        controls += found
        if any(r["pair"].startswith("image") for r in report.get("rows") or []) and not found:
            problems.append(f"{d}: an image A ran but table.out has no image control line")
    for pixels in controls:
        if pixels != pin:
            problems.append(f"image control pixels {pixels[:12]}... are not the pin "
                            f"{pin[:12]}...")
    if len(runtimes) > 1:
        problems.append(f"the runs used {len(runtimes)} runtimes: "
                        + ", ".join(sorted(x[:12] for x in runtimes)))
    if len(contexts) > 1 or (contexts and contexts != {case.context_sha256}):
        problems.append(f"context text hash(es) {sorted(str(c)[:12] for c in contexts)} are not "
                        f"the pinned {case.context_sha256[:12]}")
    want = expected_rows(case, case.context_tokens)
    seen: dict = {}
    for row in rows:
        seen.setdefault((row["pair"], row["name"]), []).append(row)
    missing = [k for k in want if k not in seen]
    extra = [k for k in seen if k not in want]
    duplicated = [k for k, v in seen.items() if len(v) > 1]
    problems += [f"{p} | {n}: more than once" for p, n in duplicated]
    problems += [f"{p} | {n}: not a row of the {case.name} table" for p, n in extra]
    incomplete = [f"{p} | {n}: missing" for p, n in missing]
    if not partial:
        problems += incomplete
    llm_rows = [r for r in rows if "image" not in r["pair"]]
    prepared_llm = [r for r in llm_rows if (parse_row(r["name"]) or {}).get("kind") == "prepared"]

    def summary(r):
        return None if r is None else {"pair": r["pair"], "name": r["name"],
                                       "total": r["total"], "peak_bytes": r["peak_bytes"]}

    worst = max(llm_rows, key=lambda r: r["total"], default=None)
    worst_prepared = max(prepared_llm, key=lambda r: r["total"], default=None)
    status = ("FAIL" if problems else "INCOMPLETE" if partial and incomplete else "PASS")
    return {"case": case.name, "status": status, "problems": problems,
            "missing": incomplete, "rows": [
                {"pair": r["pair"], "name": r["name"], "total": r["total"],
                 "peak_bytes": r["peak_bytes"], "source": r["source"]} for r in rows],
            "expected_rows": len(want), "row_count": len(rows),
            "worst_llm_llm": summary(worst), "worst_prepared_llm_llm": summary(worst_prepared),
            "max_peak_bytes": max((r["peak_bytes"] for r in rows), default=None),
            "mem_available_low": mem_low}


def print_check(report: dict) -> None:
    gib = 1 << 30
    print(f"swap table {report['case']}: {report['row_count']} of {report['expected_rows']} rows")
    for r in report["rows"]:
        print(f"  {r['pair']:22} {r['name']:45} {r['total']:8.3f} s  peak "
              f"{r['peak_bytes'] / gib:7.2f} GiB ({r['peak_bytes']} B)")
    for key, label in (("worst_llm_llm", "worst LLM<->LLM row"),
                       ("worst_prepared_llm_llm", "worst prepared LLM<->LLM row")):
        w = report[key]
        if w:
            print(f"{label}: {w['total']:.3f} s ({w['pair']}, {w['name']}, peak "
                  f"{w['peak_bytes'] / gib:.2f} GiB)")
    if report["max_peak_bytes"] is not None:
        print(f"largest tracked peak: {report['max_peak_bytes'] / gib:.2f} GiB")
    if report["mem_available_low"] is not None:
        print(f"lowest MemAvailable: {report['mem_available_low'] / gib:.2f} GiB")
    for p in report["problems"]:
        print(f"problem: {p}")
    if report["status"] == "INCOMPLETE":
        for m in report["missing"]:
            print(f"missing: {m}")
    print(report["status"])


def cmd_check(args: argparse.Namespace) -> int:
    report = check(args.dirs, args.case, args.prepared_limit, args.first_use_limit,
                   args.image_pin, args.partial)
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print_check(report)
    return {"PASS": 0, "INCOMPLETE": 2}.get(report["status"], 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_run_options(p):
        p.add_argument("--case", required=True, choices=sorted(CASES))
        p.add_argument("--runtime", required=True)
        p.add_argument("--store", help="the installed-artifacts directory")
        p.add_argument("--out", required=True)
        p.add_argument("--ids", help="JSON of artifact IDs by model name, over the defaults")
        p.add_argument("--image-pin", help="the expected pixels SHA-256 (default: the published composition's frozen pin)")
        p.add_argument("--share", default=DEFAULT_SHARE)
        p.add_argument("--inputs", help="the context texts (default SHARE/m36-baselines/inputs)")
        p.add_argument("--config-template",
                       help="a whole node configuration with {share}, {data}, {installed}")

    p = sub.add_parser("plan", help="print a spark-job steps file")
    add_run_options(p)
    p.add_argument("--pairs-per-job", type=int, default=2)
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("run", help="run one process of the table")
    add_run_options(p)
    p.add_argument("--name", required=True)
    p.add_argument("--pairs", help="A:B,... (default: the case's whole table)")
    p.add_argument("--timeout", type=int, default=JOB_SECONDS - 10)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("check", help="check the outputs of a table's runs")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--case", required=True, choices=sorted(CASES))
    p.add_argument("--prepared-limit", type=float, default=PREPARED_LIMIT)
    p.add_argument("--first-use-limit", type=float, default=FIRST_USE_LIMIT)
    p.add_argument("--image-pin")
    p.add_argument("--partial", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
