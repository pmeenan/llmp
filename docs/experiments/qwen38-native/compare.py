#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""llmpalooza's Qwen3.8 runs against Mia's vLLM (the oracle), stdlib only.

  python3 compare.py prompts FAST_SWAP_DIR RUN_DIR STEPWISE_DIR [FREE_RUN_DIR]
  python3 compare.py ppl FAST_SWAP_DIR RUN_DIR

prompts: RUN_DIR is a teacher-forced run (llmp_qwen38_exec --prompts
prompts.tsv --force forced.tsv --generate 32), whose logits files hold each
step's row; STEPWISE_DIR the same run with --stepwise (every product on
decode's kernels). The near-tie bound is llmpalooza's own noise: the 95th
percentile, over the 192 steps, of how much its top-1 to top-2 logit
margin moves between the two runs (activations quantized to FP4 or to
8 bits, MXFP8 through BF16 or not), rounded up to the oracle's logprob
resolution of 0.125. At every step, llmpalooza's log-softmax is compared with
the oracle's top-5 logprobs:
  - greedy: llmpalooza's argmax must be the oracle's token, except at a near-tie:
    a step whose oracle top-1 to top-2 margin is at most the bound;
  - dlogprob: |llmpalooza's logprob - the oracle's| over the oracle's top-5
    tokens, reported per prompt (max and RMS), and between the two runs;
  - margin: the oracle's top-1 to top-2 logprob gap at each disagreement.
FREE_RUN_DIR (optional) is a free-running greedy run; its argmax tokens are
compared with the oracle's greedy continuation (reported, not gated).

ppl: RUN_DIR is a --ppl run; its mean NLL and perplexity against the
oracle's, the per-position |dNLL| and top-1 agreement.
"""
import array
import json
import math
import struct
import sys
from pathlib import Path

VOCAB = 248320
RESOLUTION = 0.125  # the oracle's logprob grid near the top (BF16 logits)


def log_softmax_at(row, ids):
    most = max(row)
    total = math.fsum(math.exp(v - most) for v in row)
    lse = most + math.log(total)
    return {i: row[i] - lse for i in ids}


def rows(path):
    data = array.array("f")
    data.frombytes(Path(path).read_bytes())
    return [data[i:i + VOCAB] for i in range(0, len(data), VOCAB)]


def top2(row):
    first = max(range(VOCAB), key=row.__getitem__)
    second = max((i for i in range(VOCAB) if i != first), key=row.__getitem__)
    return first, second


def noise_bound(ref, run_dir, stepwise_dir):
    """The 95th percentile of the margin's move between the two runs, and
    the runs' own |dlogprob| RMS over the oracle's top-5 tokens."""
    moves, diffs = [], []
    for p in ref["prompts"]:
        a = rows(Path(run_dir) / f"{p['id']}.logits.f32")
        b = rows(Path(stepwise_dir) / f"{p['id']}.logits.f32")
        for k, top in enumerate(p["top_logprobs"]):
            i, j = top2(a[k])
            moves.append(abs((a[k][i] - a[k][j]) - (b[k][i] - b[k][j])))
            ids = [t for t, _ in top]
            la, lb = log_softmax_at(a[k], ids), log_softmax_at(b[k], ids)
            diffs += [abs(la[t] - lb[t]) for t in ids]
    moves.sort()
    p95 = moves[min(len(moves) - 1, int(0.95 * len(moves)))]
    return (math.ceil(p95 / RESOLUTION) * RESOLUTION, round(p95, 4),
            round(math.sqrt(math.fsum(d * d for d in diffs) / len(diffs)), 5))


def prompts(ref_dir, run_dir, stepwise_dir, free_dir=None):
    ref = json.loads((Path(ref_dir) / "reference-qwen3.8-nvfp4-vllm.json").read_text())
    bound, p95, self_rms = noise_bound(ref, run_dir, stepwise_dir)
    out = {"near_tie_bound": bound, "self_margin_move_p95": p95, "self_dlogprob_rms": self_rms,
           "prompts": []}
    steps = agree = exceptions_ok = failures = 0
    for p in ref["prompts"]:
        name = p["id"]
        ours = rows(Path(run_dir) / f"{name}.logits.f32")
        tokens, tops = p["greedy_token_ids"], p["top_logprobs"]
        diffs, disagreements = [], []
        for k, (want, top) in enumerate(zip(tokens, tops)):
            row = ours[k]
            got = max(range(VOCAB), key=row.__getitem__)
            ids = [t for t, _ in top]
            lp = log_softmax_at(row, ids + [got])
            diffs += [abs(lp[t] - v) for t, v in top]
            steps += 1
            if got == want:
                agree += 1
                continue
            margin = top[0][1] - top[1][1]
            ok = margin <= bound
            exceptions_ok += ok
            failures += not ok
            disagreements.append({"step": k, "oracle": want, "llmp": got, "oracle_margin": round(margin, 5),
                                  "llmp_logprob_of_oracle": round(lp[want], 5),
                                  "llmp_logprob_of_own": round(lp[got], 5), "near_tie": ok})
        entry = {"name": name, "steps": len(tokens), "agree": len(tokens) - len(disagreements),
                 "dlogprob_max": round(max(diffs), 5),
                 "dlogprob_rms": round(math.sqrt(math.fsum(d * d for d in diffs) / len(diffs)), 5),
                 "disagreements": disagreements}
        if free_dir:
            summary = json.loads((Path(free_dir) / "summary.json").read_text())
            free = next(x for x in summary["prompts"] if x["name"] == name)["argmax"]
            same = 0
            while same < min(len(free), len(tokens)) and free[same] == tokens[same]:
                same += 1
            entry["free_running_identical_prefix"] = same
        out["prompts"].append(entry)
    out.update(steps=steps, agree=agree, near_tie_exceptions=exceptions_ok, failures=failures,
               passed=failures == 0)
    print(json.dumps(out, indent=1))
    return 0 if failures == 0 else 1


def ppl(ref_dir, run_dir):
    ref = json.loads((Path(ref_dir) / "reference-qwen3.8-nvfp4-vllm-ppl.json").read_text())
    data = Path(run_dir, "ppl.nll.f64").read_bytes()
    nll = list(struct.unpack(f"<{len(data) // 8}d", data))
    top = array.array("i")
    top.frombytes(Path(run_dir, "ppl.top1.i32").read_bytes())
    want = [-v for v in ref["target_logprobs"]]
    ids = ref["token_ids"]
    if len(nll) != len(want):
        raise SystemExit(f"{len(nll)} scored positions, the oracle has {len(want)}")
    mean, ref_mean = math.fsum(nll) / len(nll), math.fsum(want) / len(want)
    d = [abs(a - b) for a, b in zip(nll, want)]
    top1_actual = sum(1 for i, t in enumerate(top) if t == ids[i + 1]) / len(top)
    oracle_top1 = ref.get("top1_token_ids")
    out = {"positions": len(nll), "llmp_mean_nll": round(mean, 6), "llmp_ppl": round(math.exp(mean), 4),
           "oracle_mean_nll": round(ref_mean, 6), "oracle_ppl": round(math.exp(ref_mean), 4),
           "ppl_relative_difference": round(math.exp(mean) / math.exp(ref_mean) - 1, 5),
           "dnll_max": round(max(d), 5), "dnll_rms": round(math.sqrt(math.fsum(x * x for x in d) / len(d)), 5),
           "dnll_mean_first_2048": round(math.fsum(d[:2047]) / 2047, 5),
           "dnll_mean_after_2048": round(math.fsum(d[2047:]) / max(1, len(d) - 2047), 5),
           "llmp_top1_is_next_token": round(top1_actual, 4)}
    if oracle_top1 is not None:
        out["top1_agreement_with_oracle"] = round(sum(1 for a, b in zip(top, oracle_top1) if a == b) / len(top), 4)
    print(json.dumps(out, indent=1))
    return 0


def main(argv):
    if len(argv) >= 5 and argv[1] == "prompts":
        return prompts(argv[2], argv[3], argv[4], argv[5] if len(argv) > 5 else None)
    if len(argv) == 4 and argv[1] == "ppl":
        return ppl(argv[2], argv[3])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
