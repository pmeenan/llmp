# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Judges jitLLM at depth against its oracle (README.md, "Correctness").

    judge.py inputs ORACLE.json OUTDIR NAME
        The oracle record's prompt IDs and greedy tokens as harness input
        lines (NAME.prompt.tsv, NAME.force.tsv) for jitllm_dsv4_exec or
        jitllm_qwen38_exec --prompts ... --force ... --generate N.
    judge.py noise A B NAME --vocab V
        The near-tie bound, recorded before the comparison: two of jitLLM's
        own paths forced on the same tokens (its default fast plan and its
        reference form); at each step the change in A's top-two logit margin
        when B scores the same two tokens; the bound is the 99th percentile.
    judge.py greedy ORACLE.json HARNESS NAME --vocab V --bound B
                    [--reference REF [--tolerance T]]
        jitLLM forced on the oracle's greedy tokens: at each step its argmax
        is the oracle's token, or the oracle's log-probability margin between
        its token and jitLLM's argmax is below B (a near-tie; when jitLLM's
        argmax is outside the oracle's top list, the margin is at least the
        oracle token's lead over the list's last). Every exception is listed.
        `strict_pass`: no step outside (the rule before 2026-10-03). With
        REF, the model's pinned reference run on this history (NAME's
        logits; D-085 names each, and changing one is the owner's decision),
        `pass` applies the tie-aware rule (the owner, 2026-10-03; README.md,
        "Correctness"): every outside step a tie flip (jitLLM's own logit
        margin of its argmax over the oracle's token below B, and its NLL of
        that token less than B above the oracle's own), at most max(2, REF's
        outside steps) of them, and the oracle continuation's conditional
        perplexity ratio to the oracle's own at most T (default 0.005) above
        REF's. A REF with an outside step past the per-step tolerance, or
        more than 2, is refused. Without REF, `pass` is `strict_pass`.
    judge.py repeat A B NAME --vocab V
        Two runs of the same forced prompt: the first step whose logits
        differ, and the largest difference (RE-031).
    judge.py ppl ORACLE HARNESS --ctx L [--within F]
        Perplexity over the window's second half (tokens L/2+1 .. L-1):
        ORACLE is a vLLM ppl-L.nll.json or a number (llama-perplexity's),
        HARNESS the jitLLM harness's ppl.nll.f64. Within F (default 0.03).

Runs on a Spark in a container whose Python has NumPy (the pinned PyTorch
image). Comparisons require complete, nonempty captures: matching row
counts for noise/repeat, every oracle step for greedy, and L-1 NLL values
for a perplexity window. Prints one JSON document; exits 1 when a bound or
repeatability check fails, or a capture is incomplete or has non-finite
logits or NLL values.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_floats(path, dtype):
    path = Path(path)
    if path.stat().st_size % np.dtype(dtype).itemsize:
        raise SystemExit(f"{path}: incomplete floating-point value")
    a = np.fromfile(path, dtype=dtype)
    if not a.size or not np.all(np.isfinite(a)):
        raise SystemExit(f"{path}: empty or non-finite capture")
    return a


def require_count(actual, expected, label):
    if expected < 1 or actual != expected:
        raise SystemExit(f"{label}: expected {expected} nonempty entries, got {actual}")


def load_logits(directory, name, vocab):
    if vocab < 1:
        raise SystemExit("--vocab must be positive")
    a = load_floats(Path(directory) / f"{name}.logits.f32", np.float32)
    if a.size % vocab:
        raise SystemExit(f"{directory}/{name}: {a.size} logits are not whole rows of {vocab}")
    return a.reshape(-1, vocab).astype(np.float64)


def arg(flag, default=None, cast=str):
    if flag not in sys.argv:
        return default
    at = sys.argv.index(flag) + 1
    if at >= len(sys.argv):
        raise SystemExit(f"{flag} needs a value")
    return cast(sys.argv[at])


def inputs(oracle_path, outdir, name):
    record = json.loads(Path(oracle_path).read_text())
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.prompt.tsv").write_text(
        name + "\t" + " ".join(map(str, record["prompt_ids"])) + "\n")
    (out / f"{name}.force.tsv").write_text(name + "\t" + " ".join(map(str, record["ids"])) + "\n")
    return {"prompt_tokens": len(record["prompt_ids"]), "generated": len(record["ids"])}


def noise(a_dir, b_dir, name, vocab):
    if vocab < 2:
        raise SystemExit("noise requires a vocabulary of at least two tokens")
    a, b = load_logits(a_dir, name, vocab), load_logits(b_dir, name, vocab)
    require_count(len(b), len(a), "noise rows")
    steps = len(a)
    moves = []
    for k in range(steps):
        top = np.argsort(a[k])[-2:][::-1]
        margin_a = a[k][top[0]] - a[k][top[1]]
        margin_b = b[k][top[0]] - b[k][top[1]]
        moves.append(abs(margin_a - margin_b))
    moves = np.array(moves)
    return {"steps": steps, "p50": float(np.percentile(moves, 50)),
            "p99": float(np.percentile(moves, 99)), "max": float(moves.max()),
            "argmax_equal": int(sum(np.argmax(a[k]) == np.argmax(b[k]) for k in range(steps)))}


def score(record, logits, bound):
    """One forced run against the oracle: agreement, near-ties, outside steps
    split into tie flips and violations, and the continuation ratio."""
    steps = len(record["steps"])
    require_count(len(logits), steps, "greedy rows")
    agree, near, outside, flips, violations = 0, [], [], [], []
    excess = []  # jitLLM's NLL of the oracle's token minus the oracle's own
    for k in range(steps):
        step = record["steps"][k]
        row = logits[k]
        peak = float(row.max())
        nll = peak + math.log(float(np.exp(row - peak).sum())) - float(row[step["id"]])
        excess.append(nll + float(step["logprob"]))
        mine = int(np.argmax(row))
        if mine == step["id"]:
            agree += 1
            continue
        top = {int(t): float(v) for t, v in step["top"]}
        lead = step["logprob"] - top.get(mine, min(top.values()) if top else -math.inf)
        entry = {"step": k, "oracle": step["id"], "jitllm": mine, "oracle_margin": lead,
                 "in_top": mine in top, "jitllm_margin": float(row[mine] - row[step["id"]]),
                 "nll_excess": excess[-1]}
        if lead < bound and mine in top:
            near.append(entry)
            continue
        outside.append(entry)
        # The per-step tolerance (README.md, "Correctness"): jitLLM itself
        # holds the oracle's token within the bound, and its NLL of that
        # token is within the bound of the oracle's own. An owner-accepted
        # tolerance, not a calibrated test of a tie.
        tie = entry["jitllm_margin"] < bound and entry["nll_excess"] < bound
        (flips if tie else violations).append(entry)
    # The oracle continuation's conditional perplexity against the oracle's
    # own (the oracle's greedy path, so every engine sits above 1).
    continuation = math.exp(sum(excess) / steps)
    return {"steps": steps, "agree": agree, "near_ties": near, "outside": len(outside),
            "tie_flips": flips, "violations": violations, "continuation_ratio": continuation}


def greedy(oracle_path, harness, name, vocab, bound, reference, tolerance):
    record = json.loads(Path(oracle_path).read_text())
    result = score(record, load_logits(harness, name, vocab), bound)
    result.update(bound=bound, strict_pass=result["outside"] == 0)
    if reference is None:
        # Without the model's reference run only the strict rule applies.
        result["pass"] = result["strict_pass"]
        return result
    if not reference or not (Path(reference) / f"{name}.logits.f32").is_file():
        raise SystemExit(f"--reference {reference!r}: no {name}.logits.f32 there")
    ref = score(record, load_logits(reference, name, vocab), bound)
    if ref["violations"] or ref["outside"] > 2:
        raise SystemExit(
            f"the reference run {reference} has {ref['outside']} outside steps, "
            f"{len(ref['violations'])} past the per-step tolerance: a reference past "
            "the tolerance or the default cap of 2 needs an owner decision")
    # The tie-aware rule (the owner, 2026-10-03; D-085 as refined): every
    # outside step a tie flip, no more of them than max(2, the reference's
    # outside steps), and the continuation ratio at most `tolerance` above
    # the reference run's (the model's pinned reference on this history).
    cap = max(2, ref["outside"])
    allowed = ref["continuation_ratio"] + tolerance
    result.update(reference={"outside": ref["outside"],
                             "continuation_ratio": ref["continuation_ratio"]},
                  flip_cap=cap, continuation_allowed=allowed, tolerance=tolerance)
    result["pass"] = (not result["violations"] and result["outside"] <= cap and
                      result["continuation_ratio"] <= allowed)
    return result


def repeat(a_dir, b_dir, name, vocab):
    a, b = load_logits(a_dir, name, vocab), load_logits(b_dir, name, vocab)
    require_count(len(b), len(a), "repeat rows")
    steps = len(a)
    differ = [k for k in range(steps) if not np.array_equal(a[k], b[k])]
    return {"steps": steps, "identical": not differ, "pass": not differ, "first_differing_step":
            differ[0] if differ else None, "steps_differing": len(differ),
            "max_abs_difference": float(np.abs(a[:steps] - b[:steps]).max())}


def ppl(oracle, harness, ctx, within):
    if ctx < 3:
        raise SystemExit("--ctx must leave at least one token in the scored half (at least 3)")
    mine = load_floats(harness, np.float64)
    require_count(len(mine), ctx - 1, "harness NLL values")
    half = mine[ctx // 2:ctx - 1]
    ours = math.exp(float(half.mean()))
    if Path(oracle).exists():
        theirs_nll = np.array(json.loads(Path(oracle).read_text()), dtype=np.float64)
        if theirs_nll.ndim != 1 or not np.all(np.isfinite(theirs_nll)):
            raise SystemExit(f"{oracle}: expected a finite NLL vector")
        require_count(len(theirs_nll), ctx - 1, "oracle NLL values")
        theirs = math.exp(float(theirs_nll[ctx // 2:ctx - 1].mean()))
    else:
        theirs = float(oracle)
    if not math.isfinite(theirs) or theirs <= 0:
        raise SystemExit("oracle perplexity must be finite and positive")
    ratio = ours / theirs
    return {"window": ctx, "scored": int(half.size), "jitllm_ppl": ours, "oracle_ppl": theirs,
            "ratio": ratio, "within": within, "pass": abs(ratio - 1) <= within}


def main():
    command = sys.argv[1]
    vocab = arg("--vocab", 0, int)
    if command == "inputs":
        result = inputs(sys.argv[2], sys.argv[3], sys.argv[4])
    elif command == "noise":
        result = noise(sys.argv[2], sys.argv[3], sys.argv[4], vocab)
    elif command == "greedy":
        result = greedy(sys.argv[2], sys.argv[3], sys.argv[4], vocab, arg("--bound", 1.0, float),
                        arg("--reference"), arg("--tolerance", 0.005, float))
    elif command == "repeat":
        result = repeat(sys.argv[2], sys.argv[3], sys.argv[4], vocab)
    elif command == "ppl":
        result = ppl(sys.argv[2], sys.argv[3], arg("--ctx", 0, int), arg("--within", 0.03, float))
    else:
        raise SystemExit(__doc__)
    print(json.dumps(result, indent=1))
    if result.get("pass") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
