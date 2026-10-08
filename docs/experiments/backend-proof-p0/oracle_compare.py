#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Score exl3_heldout.py runs against the FP64 oracle (oracle.py).

For each run and prefix, the prefill rows are compared with oracle rows
0..p-1 and the 16 single-token steps with rows p..p+15. Per row: the largest
absolute logit error and the RMS logit error. Per section: their maxima and
quantiles, the overall RMS, and top-1 agreement with the oracle. A run made
with --capture is also scored per layer: the residual stream after each
block (per-row RMS error relative to the oracle row's RMS) and each layer's
K and V per position (largest absolute error, and the RMS error relative to
the oracle's RMS at that position, over heads and channels), for the prefix
and, when captured, for the positions the single-token steps wrote.

With several runs, the output ends with the envelope: for every statistic,
the worst value over the runs. The envelope describes these upstream runs;
it is not by itself an acceptance bound.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def rows(err):
    return {"row_max_abs": np.abs(err).max(axis=1), "row_rms": np.sqrt((err * err).mean(axis=1))}


def section(values, oracle):
    err = values.astype(np.float64) - oracle
    r = rows(err)
    return {
        "rows": int(values.shape[0]),
        "finite": bool(np.isfinite(values).all()),
        "max_abs": float(r["row_max_abs"].max()),
        "rms": float(np.sqrt((err * err).mean())),
        "row_rms_max": float(r["row_rms"].max()),
        "row_max_abs_quantiles": {q: float(np.quantile(r["row_max_abs"], float(q))) for q in ("0.5", "0.9", "0.99")},
        "top1_equal_oracle": int(np.count_nonzero(values.argmax(axis=1) == oracle.argmax(axis=1))),
        "row_max_abs": [round(float(x), 6) for x in r["row_max_abs"]],
        "row_rms": [round(float(x), 7) for x in r["row_rms"]],
    }


def layers(capture, oracle, prefix):
    blocks = capture["blocks"].astype(np.float64).reshape(-1, prefix, oracle["blocks"].shape[2])
    ref = oracle["blocks"][:, :prefix]
    scale = np.sqrt((ref * ref).mean(axis=2))
    rel = np.sqrt(((blocks - ref) ** 2).mean(axis=2)) / scale
    kv = capture["kv"].astype(np.float64)
    k_err = np.abs(kv[0::2] - oracle["k"][:, :prefix]).max(axis=(2, 3))
    v_err = np.abs(kv[1::2] - oracle["v"][:, :prefix]).max(axis=(2, 3))

    def relative(values, ref):
        err = np.sqrt(((values - ref) ** 2).mean(axis=(2, 3)))
        return err / np.sqrt((ref * ref).mean(axis=(2, 3)))
    kv_rel = {"k_relative_position_max": relative(kv[0::2], oracle["k"][:, :prefix]).max(axis=1),
           "v_relative_position_max": relative(kv[1::2], oracle["v"][:, :prefix]).max(axis=1)}
    if "kv_after_suffix" in capture:
        after = capture["kv_after_suffix"].astype(np.float64)[:, prefix:]
        end = prefix + after.shape[1]
        kv_rel["k_relative_suffix_max"] = relative(after[0::2], oracle["k"][:, prefix:end]).max(axis=1)
        kv_rel["v_relative_suffix_max"] = relative(after[1::2], oracle["v"][:, prefix:end]).max(axis=1)
    return {**{k: [float(x) for x in v] for k, v in kv_rel.items()},
        "block_row_relative_rms_max": [float(x) for x in rel.max(axis=1)],
        "block_relative_rms": [float(x) for x in np.sqrt((rel * rel).mean(axis=1))],
        "k_position_max_abs": [float(x) for x in k_err.max(axis=1)],
        "v_position_max_abs": [float(x) for x in v_err.max(axis=1)],
        "k_position_max_abs_by_position": [[round(float(x), 5) for x in layer] for layer in k_err],
    }


def score(run, oracle):
    out = {"run": run.name, "prefixes": []}
    logits = oracle["logits"]
    for path in sorted(run.glob("logits-*.npz"), key=lambda p: int(p.stem.split("-")[1])):
        prefix = int(path.stem.split("-")[1])
        data = np.load(path)
        row = {"prefix": prefix,
               "prefill": section(data["prefill"].reshape(-1, logits.shape[1]), logits[:prefix]),
               "suffix": section(data["suffix"].reshape(-1, logits.shape[1]), logits[prefix:prefix + 16])}
        capture = run / f"capture-{prefix}.npz"
        if capture.exists():
            row["layers"] = layers(np.load(capture), oracle, prefix)
        out["prefixes"].append(row)
    return out


STATS = ("max_abs", "rms", "row_rms_max")


def envelope(scores):
    env = {}
    for s in scores:
        for p in s["prefixes"]:
            for part in ("prefill", "suffix"):
                e = env.setdefault(str(p["prefix"]), {}).setdefault(part, {k: 0.0 for k in STATS})
                for k in STATS:
                    e[k] = max(e[k], p[part][k])
                e["top1_equal_oracle_min"] = min(e.get("top1_equal_oracle_min", p[part]["rows"]),
                                                 p[part]["top1_equal_oracle"])
            if "layers" in p:
                lay = env[str(p["prefix"])].setdefault("layers", {})
                for k in (key for key in p["layers"] if not key.endswith("_by_position")):
                    lay[k] = [max(a, b) for a, b in zip(lay.get(k, [0.0] * len(p["layers"][k])), p["layers"][k])]
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    oracle = dict(np.load(args.oracle))
    scores = [score(run, oracle) for run in args.runs]
    print(json.dumps({"oracle": str(args.oracle), "runs": scores, "envelope": envelope(scores),
                      "interpretation": "Error against the FP64 oracle; not an acceptance tolerance by itself"}))


if __name__ == "__main__":
    main()
