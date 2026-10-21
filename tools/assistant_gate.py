#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The Gemma4 31B assistant screen's matched A/B (docs/experiments/m36-baselines/README.md,
"The Gemma4 31B assistant", and "The gate's tools").

ABBA blocks of processes: an old and a new `llmp_gemma_engine_greedy` (or reference) binary,
each over its own arm's v1 artifact store. A process's value
is the mean of its two timed rounds' seconds for 32 committed tokens; a block's slowdown is
ln(mean(new) / mean(old)); the verdict is speed_gate.py's rule (non-inferiority at the 1%
margin, one-sided 97.5% t bound at every look from MIN blocks to MAX, the same code).

    assistant_gate.py steps --out DIR --old BIN --new BIN
                      --old-artifacts STORE --new-artifacts STORE
                      [--target ID] [--assistant ID] --inputs FILE
                      [--new-target ID --new-assistant ID]
                      [--first B] [--blocks N] [--profile 31] [--mode unit32] > FILE
    assistant_gate.py verdict DIR [--min 8] [--max 11] [--json]

`steps` prints a supervised job's steps file (one shell command per line, for
`spark-job start --gpu --steps FILE`; a job runs at most 600 s, so call it per
block pair: `--first 1 --blocks 2`, then `--first 3 --blocks 2`, ...): a priming old run
no statistic reads, then blocks first..first+blocks-1, each old/new/new/old. Each arm runs
its own `BIN TARGET ASSISTANT RUNDIR PROFILE INPUTS MODE`, TARGET and ASSISTANT being
`STORE/ID` of that arm's store (defaulting to the published v1 Gemma31 target and assistant;
`--new-target`/`--new-assistant` when the new store holds
the same artifacts under other IDs). BIN is the benchmark executable, STORE the
installed-artifacts directory (`~/.local/share/llmp/artifacts`, or the store holding the
re-imported ones). Each call also records `DIR/plan.json` (arms, artifact paths, the inputs'
SHA-256 when readable) for the verdict to print.

`verdict` reads DIR's logs, `bNN-pP-{old,new}.log`. It is INVALID (exit 1) when a log
is not a complete run of the screen's trajectory, when the retired line's workload
identity (profile, mode, C, P, prefill query, emitted, endpoint, warm, quality and timed
rounds) differs between any two runs, or when a priming log that exists is incomplete.
Exit status: 0 PASS, 2 while more blocks are needed (CONTINUE), 1 for FAIL, NOT PASSED
or INVALID.

The harness is model-specific; its migrated counterpart must pay the same work (target,
assistant, inputs, 32 committed tokens at P=64) to be compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import shlex
import statistics
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import speed_gate  # noqa: E402  (the one implementation of the protocol's statistic)

LIMIT = math.log1p(speed_gate.GATE)
DEFAULT_TARGET = speed_gate.GEMMA31_V1
DEFAULT_ASSISTANT = speed_gate.GEMMA31_ASSISTANT_V1
ORDER = ("old", "new", "new", "old")
RUN = "rm -rf {d} && CUDA_DISABLE_PTX_JIT=1 {bin} {t} {a} {d} {profile} {i} {mode} > {d}.log 2>&1"
COMPLETED = re.compile(
    r"GREEDY32_COMPLETED mode=(?P<mode>\S+) phase=(?P<phase>timed-first|timed-repeat) "
    r"emitted=32 endpoint=96 verified=46 drafted=34 seconds=(?P<seconds>[0-9.]+)")
# The frozen reference harness retires with GREEDY_NATIVE_RETIRED, the engine harness (the
# new arm, benchmarks/gemma_engine_greedy.cc) with GREEDY_ENGINE_RETIRED; the fields are the same.
RETIRED = re.compile(r"GREEDY_(?:NATIVE|ENGINE)_RETIRED (?P<fields>.*)")
IDENTITY_KEYS = ("profile", "mode", "C", "P", "prefill_query", "emitted", "endpoint", "warm",
                 "quality", "timed")


def sha256_or_none(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def step_lines(out, first, blocks, arms, inputs, profile="31", mode="unit32") -> list[str]:
    """arms: {"old": (binary, target_path, assistant_path), "new": (...)}. A priming old run
    no statistic reads, then ABBA blocks."""
    def line(directory: str, arm: str) -> str:
        binary, target, assistant = arms[arm]
        return RUN.format(d=shlex.quote(directory), bin=shlex.quote(binary),
                          t=shlex.quote(target), a=shlex.quote(assistant),
                          profile=profile, i=shlex.quote(inputs), mode=mode)

    lines = [line(f"{out}/prime-b{first:02d}", "old")]
    for b in range(first, first + blocks):
        for pos, arm in enumerate(ORDER):
            lines.append(line(f"{out}/b{b:02d}-p{pos}-{arm}", arm))
    return lines


def parse_log(text: str) -> dict:
    """A complete run's two timed rounds' seconds and its retired line's identity; raises
    ValueError when the log is not a complete run of the screen's trajectory."""
    timed = [float(m.group("seconds")) for m in COMPLETED.finditer(text)]
    retired = RETIRED.search(text)
    if len(timed) != 2 or retired is None:
        raise ValueError("not a complete run with the screen's trajectory")
    fields = dict(kv.split("=", 1) for kv in retired.group("fields").split() if "=" in kv)
    identity = {k: fields.get(k) for k in IDENTITY_KEYS}
    if any(v is None for v in identity.values()):
        raise ValueError("the retired line lacks " +
                         ", ".join(k for k, v in identity.items() if v is None))
    return {"seconds": sum(timed) / 2, "identity": identity}


def read_blocks(out: pathlib.Path) -> tuple[list[tuple], list[str]]:
    """(rows, invalid): rows are (block, old mean, new mean, slowdown) of the complete
    blocks; invalid lists the runs that are missing or cannot be judged."""
    invalid: list[str] = []
    identities: dict = {}
    parsed: dict = {}

    def read(path: pathlib.Path):
        try:
            result = parse_log(path.read_text())
        except (OSError, ValueError) as e:
            invalid.append(f"{path.name}: {e}")
            return None
        identities[path.name] = result["identity"]
        return result["seconds"]

    for prime in sorted(out.glob("prime-b*.log")):
        read(prime)
    rows = []
    numbers = sorted({int(m.group(1)) for path in out.glob("b*-p*-*.log")
                      if (m := re.fullmatch(r"b(\d+)-p[0-3]-(?:old|new)\.log", path.name))})
    for b in range(1, (numbers[-1] if numbers else 0) + 1):
        for p, arm in enumerate(ORDER):
            path = out / f"b{b:02d}-p{p}-{arm}.log"
            if path.exists():
                parsed[(b, p)] = read(path)
            else:
                invalid.append(f"{path.name}: missing")
                parsed[(b, p)] = None
        if all(parsed[(b, p)] is not None for p in range(4)):
            old = [parsed[(b, p)] for p in (0, 3)]
            new = [parsed[(b, p)] for p in (1, 2)]
            rows.append((b, statistics.mean(old), statistics.mean(new),
                         math.log(statistics.mean(new) / statistics.mean(old))))
    reference_name = next(iter(identities), None)
    for name, identity in identities.items():
        if identity != identities[reference_name]:
            diff = {k: (identities[reference_name][k], v) for k, v in identity.items()
                    if v != identities[reference_name][k]}
            invalid.append(f"{name}: workload differs from {reference_name}'s: {diff}")
    return rows, invalid


def verdict_report(out: pathlib.Path, least: int = 8, most: int = 11) -> dict:
    rows, invalid = read_blocks(out)
    counted = rows[:most]  # only the first MAX blocks count: more would be another look
    judged = speed_gate.judge([r[3] for r in counted], LIMIT, least, most)
    verdict = "INVALID" if invalid else judged["verdict"]
    return {"verdict": verdict, "blocks": [dict(zip(("block", "old_s", "new_s", "slowdown"), r))
                                           for r in rows],
            "judged": judged, "invalid": invalid, "limit": LIMIT, "min": least, "max": most}


def status(verdict: str) -> int:
    return {"PASS": 0, "CONTINUE": 2, "INCOMPLETE": 2}.get(verdict, 1)


def cmd_steps(args: argparse.Namespace) -> int:
    new_target = args.new_target or args.target
    new_assistant = args.new_assistant or args.assistant
    arms = {"old": (args.old, f"{args.old_artifacts}/{args.target}",
                    f"{args.old_artifacts}/{args.assistant}"),
            "new": (args.new, f"{args.new_artifacts}/{new_target}",
                    f"{args.new_artifacts}/{new_assistant}")}
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {"arms": {k: {"binary": v[0], "binary_sha256": sha256_or_none(v[0]),
                         "target": v[1], "assistant": v[2]} for k, v in arms.items()},
            "inputs": args.inputs, "inputs_sha256": sha256_or_none(args.inputs),
            "profile": args.profile, "mode": args.mode}
    plan_path = out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise SystemExit(f"{plan_path} records other arms or inputs: one comparison, one plan")
    plan_path.write_text(json.dumps(plan, indent=1) + "\n")
    print("\n".join(step_lines(str(out), args.first, args.blocks, arms, args.inputs,
                               args.profile, args.mode)))
    return 0


def cmd_verdict(args: argparse.Namespace) -> int:
    out = pathlib.Path(args.out)
    report = verdict_report(out, args.min, args.max)
    if args.json:
        print(json.dumps(report, indent=1))
        return status(report["verdict"])
    plan = out / "plan.json"
    if plan.exists():
        p = json.loads(plan.read_text())
        for arm, a in p["arms"].items():
            print(f"{arm}: {a['binary']} ({(a['binary_sha256'] or 'unread')[:12]}), "
                  f"target {a['target']}, assistant {a['assistant']}")
        print(f"inputs {p['inputs']} ({(p['inputs_sha256'] or 'unread')[:12]})")
    for r in report["blocks"]:
        print(f"block {r['block']}: old {r['old_s']:.5f} s, new {r['new_s']:.5f} s, "
              f"slowdown {100 * r['slowdown']:+.3f}%")
    for reason in report["invalid"]:
        print(f"invalid: {reason}")
    j = report["judged"]
    if j["blocks"] >= 2:
        print(f"{j['blocks']} blocks: slowdown {100 * j['mean']:+.3f}%, bound "
              f"[{100 * j['lower']:+.3f}%, {100 * j['upper']:+.3f}%], "
              f"limit {100 * LIMIT:+.3f}%")
    else:
        print(f"{j['blocks']} of at least {args.min} blocks")
    print(report["verdict"])
    return status(report["verdict"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("steps")
    p.add_argument("--out", required=True)
    p.add_argument("--old", required=True, help="the old arm's benchmark executable")
    p.add_argument("--new", required=True, help="the new arm's benchmark executable")
    p.add_argument("--old-artifacts", required=True, help="the old arm's installed store")
    p.add_argument("--new-artifacts", required=True, help="the new arm's installed store")
    p.add_argument("--target", default=DEFAULT_TARGET,
                   help="the target artifact ID (default: the v1 Gemma31 target)")
    p.add_argument("--assistant", default=DEFAULT_ASSISTANT,
                   help="the assistant artifact ID (default: the v1 Gemma31 assistant)")
    p.add_argument("--new-target", help="the new store's target ID, if it differs")
    p.add_argument("--new-assistant", help="the new store's assistant ID, if it differs")
    p.add_argument("--inputs", required=True, help="the canonical query inputs (inputs.i32)")
    p.add_argument("--first", type=int, default=1)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--profile", default="31")
    p.add_argument("--mode", default="unit32")
    p.set_defaults(func=cmd_steps)
    p = sub.add_parser("verdict")
    p.add_argument("out")
    p.add_argument("--min", type=int, default=8)
    p.add_argument("--max", type=int, default=11)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verdict)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
