#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Derive BP-F1's kernel cases from the FP16 bridge's recorded plan.

  bpf1_cases.py FP16_PLAN_JSON OUT.txt

BP-F1 (docs/backend-proof.md) times the GGML kernels llmpalooza has, at the
held-out trajectory's chunk shapes (1, 16, 17 and 512 rows), on host VMM
against cudaMalloc. The shapes and the launches each case must make come
from fp16-plan.json, the executed plan of the FP16 bridge (llama.cpp
b29c606e2 built with the llmpalooza SDK) for the held-out trajectory:

- the unfused arm gives the plain RMSNorm, the weight multiply, the adds and
  the matrix products (MMVF at 1 row, MMF at 16, GGML's cuBLAS path at 17
  and 512);
- the fused arm gives GGML's fused RMSNorm-multiply.

Each sequence's plan is flattened, cut into operations (a cuBLAS product
is its conversions, pointer setup, memset and GEMM launches, plus the
output conversion when it computes in F16), and labelled by position: nine
products per layer (q, k, v, KQ, KQV, o, gate, up, down), then the output
head. Operations with the same shape in a layer (q and o, k and v, gate
and up, the three norms, the two residual adds) must have launched
identically in every layer; they become one case. Launches llmpalooza has no
implementation of (RoPE, softmax, set_rows, get_rows, the copy, SiLU-gate
and the fused MMVF variants of the fused arm) are not cases.

The output is the harness's case file (benchmarks/ggml_vmm_bench.cc): one
`case` line per case, then its per-invocation launches in order, each a
`kernel` (name with any compiler-internal namespace tag replaced by
`_INTERNAL_`, grid, block, static plus dynamic shared bytes, registers) or
a `memset` (bytes, value), then `end`.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

SET = "qwen2.5-0.5b-f16"
HIDDEN, KV_WIDTH, FFN, VOCAB = 896, 128, 4864, 151936
HEAD_DIM, HEADS, KV_HEADS, KV_SIZE = 64, 14, 2, 1024
LAYERS = 24
PRODUCTS = ("q", "k", "v", "kq", "kqv", "o", "gate", "up", "down")
# The held-out trajectory's sequences (fp16-plan.json): rows and the KV
# cells in use (n_past + rows, padded to 256) at their first occurrence.
SEQUENCES = {0: (16, 256), 1: (17, 256), 2: (1, 256), 3: (512, 768), 4: (1, 768)}
INTERNAL = re.compile(r"(\d+)_INTERNAL_")


def normalize(mangled):
    """Replaces a length-prefixed compiler-internal namespace (a per-TU tag) with `_INTERNAL_`."""
    out, at = [], 0
    for match in INTERNAL.finditer(mangled):
        if match.start() < at:
            continue
        # The digits before _INTERNAL_ are the source name's length, which
        # starts at _INTERNAL_; take the longest digit suffix that fits.
        digits = match.group(1)
        for cut in range(len(digits)):
            length = int(digits[cut:])
            start = match.start() + cut
            end = match.start(1) + len(digits) + length
            if end <= len(mangled) and length >= len("_INTERNAL_"):
                out.append(mangled[at:start] + "_INTERNAL_")
                at = end
                break
    out.append(mangled[at:])
    return "".join(out)


def flatten(plan):
    for entry in plan:
        if isinstance(entry, dict):
            for _ in range(entry["repeat"]):
                yield from flatten(entry["body"])
        elif entry[0] in ("kernel", "memset"):
            yield entry


class Plan:
    def __init__(self, data):
        self.kernels = {k["id"]: k for k in data["kernels"]}
        self.calls = {c["id"]: c for c in data["cublas_calls"]}
        self.arms = {a["arm"]: a for a in data["arms"]}

    def name(self, launch):
        return self.kernels[launch[1]]["demangled"] if launch[0] == "kernel" else "memset"

    def family(self, launch):
        if launch[0] == "memset":
            return "memset"
        name = self.kernels[launch[1]]["demangled"]
        for prefix, family in (("void convert_unary", "convert"), ("k_compute_batched_ptrs", "ptrs"),
                               ("void rms_norm_f32<(int)256, (bool)0", "rms_norm"),
                               ("void rms_norm_f32<(int)256, (bool)1", "rms_norm_mul"),
                               ("void k_bin_bcast<&op_add", "add"), ("void k_bin_bcast<&op_mul", "mul"),
                               ("void mul_mat_vec_f", "mmvf"), ("void mul_mat_f", "mmf")):
            if name.startswith(prefix):
                return family
        return "cublas" if self.kernels[launch[1]]["owner"] == "cublas" else "other"

    def record(self, launch):
        """The harness's line for one recorded launch."""
        if launch[0] == "memset":
            return ("memset", launch[1], launch[2])
        k = self.kernels[launch[1]]
        return ("kernel", normalize(k["mangled"]), *launch[2], *launch[3], k["static_shared"] + launch[4],
                k["registers"])

    def operations(self, sequence):
        """The sequence's launches cut into operations: (family, [launch, ...])."""
        ops, pending = [], []
        launches = list(flatten(sequence["plan"]))
        i = 0
        while i < len(launches):
            launch = launches[i]
            family = self.family(launch)
            tag = launch[5] if launch[0] == "kernel" else (launch[4] if len(launch) > 4 else "")
            if family in ("convert", "ptrs") or (launch[0] == "memset" and str(tag).startswith("cublas#")):
                pending.append(launch)
                i += 1
                continue
            if str(tag).startswith("cublas#"):
                call = self.calls[int(tag.split("#")[1])]
                group = pending + [launch]
                i += 1
                while i < len(launches) and launches[i][0] == "kernel" and launches[i][5] == tag:
                    group.append(launches[i])
                    i += 1
                if call["args"]["computeType"] == "CUBLAS_COMPUTE_16F":
                    # F16 output goes through scratch and is converted back.
                    if self.family(launches[i]) != "convert":
                        raise ValueError(f"no output conversion after {tag}")
                    group.append(launches[i])
                    i += 1
                ops.append(("cublas", group, call))
                pending = []
                continue
            if pending:
                raise ValueError(f"conversions before a non-cuBLAS launch: {self.name(launch)}")
            ops.append((family, [launch], None))
            i += 1
        if pending:
            raise ValueError("trailing conversions")
        return ops


def fused_norms(ops):
    """The fused arm's RMSNorm-multiply launches, one per norm (two per layer and the output's)."""
    norms = [launches for family, launches, _ in ops if family in ("rms_norm", "rms_norm_mul", "mul")]
    if len(norms) != LAYERS * 2 + 1 or any(len(n) != 1 or n[0][0] != "kernel" for n in norms):
        raise ValueError("the fused arm's norms are not one fused launch each")
    return norms


def label(ops):
    """Labels the operations of one unfused sequence; returns {label: [launch lists]}."""
    labels = {}
    products = norms = adds = 0
    per_layer_adds = ("bias_q", "bias_k", "bias_v", "residual", "residual")
    for family, launches, _ in ops:
        if family in ("mmvf", "mmf", "cublas"):
            name = PRODUCTS[products % 9] if products < LAYERS * 9 else "lm_head"
            products += 1
        elif family in ("rms_norm", "rms_norm_mul"):
            name = family
            norms += 1
        elif family == "mul":
            name = "mul"
        elif family == "add":
            name = per_layer_adds[adds % 5] if adds < LAYERS * 5 else "extra_add"
            adds += 1
        else:
            continue
        labels.setdefault(name, []).append(launches)
    if products != LAYERS * 9 + 1 or norms != LAYERS * 2 + 1 or adds != LAYERS * 5:
        raise ValueError(f"unexpected structure: {products} products, {norms} norms, {adds} adds")
    return labels


def same(occurrences, what):
    first = occurrences[0]
    if any(o != first for o in occurrences):
        raise ValueError(f"{what}: launches differ between occurrences")
    return first


# Cases: (case name, labels that must all have launched alike, op, params).
def case_specs(rows, n_kv):
    attention = [n_kv, KV_SIZE, HEAD_DIM, HEADS, KV_HEADS]
    kv = f"kv{n_kv}"
    return [
        ("rms_norm", ["rms_norm"], "rms_norm", [HIDDEN]),
        ("rms_norm_mul", ["rms_norm_mul"], "rms_norm_mul", [HIDDEN]),
        ("mul", ["mul"], "mul", [HIDDEN]),
        # At one row a residual add has the bias add's operands exactly.
        ("add.bias_896", ["bias_q"] + (["residual"] if rows == 1 else []), "add_bias", [HIDDEN]),
        ("add.bias_128", ["bias_k", "bias_v"], "add_bias", [KV_WIDTH]),
        *([] if rows == 1 else [("add.residual", ["residual"], "add", [HIDDEN])]),
        ("linear.q_o", ["q", "o"], "linear", [HIDDEN, HIDDEN]),
        ("linear.k_v", ["k", "v"], "linear", [HIDDEN, KV_WIDTH]),
        ("linear.gate_up", ["gate", "up"], "linear", [HIDDEN, FFN]),
        ("linear.down", ["down"], "linear", [FFN, HIDDEN]),
        ("linear.lm_head", ["lm_head"], "linear", [HIDDEN, VOCAB]),
        (f"attn.kq.{kv}", ["kq"], "kq", attention),
        (f"attn.kqv.{kv}", ["kqv"], "kqv", attention),
    ]


IMPL = {"mmvf": "mmvf", "mmf": "mmf", "cublas": "cublas"}


def derive(data):
    plan = Plan(data)
    unfused = plan.arms["heldout-unfused"]["sequences"]
    fused = plan.arms["heldout-fused"]["sequences"]
    cases = {}
    for sid, (rows, n_kv) in SEQUENCES.items():
        u = {s["id"]: s for s in unfused}[sid]
        f = {s["id"]: s for s in fused}[sid]
        if u["rows"] != rows or f["rows"] != rows:
            raise ValueError(f"sequence {sid} is not {rows} rows")
        u_ops = plan.operations(u)
        f_ops = plan.operations(f)
        labels = label(u_ops)
        labels["rms_norm_mul"] = fused_norms(f_ops)
        families = {}
        for family, launches, _ in u_ops:
            families.setdefault(json.dumps(launches), family)
        for name, sources, op, params in case_specs(rows, n_kv):
            occurrences = [launches for source in sources for launches in labels[source]]
            launches = same(occurrences, f"{name} at {rows} rows")
            family = "rms_norm_mul" if op == "rms_norm_mul" else families.get(json.dumps(launches))
            impl = IMPL.get(family, "-")
            record = [plan.record(launch) for launch in launches]
            key = (name, rows)
            if key in cases and cases[key]["launches"] != record:
                raise ValueError(f"{name} at {rows} rows differs between sequences")
            cases[key] = {"name": name, "rows": rows, "op": op, "impl": impl, "params": params,
                          "launches": record, "sequence": sid}
    return [cases[k] for k in sorted(cases, key=lambda k: (k[1], k[0]))]


def render(raw):
    """The case file for fp16-plan.json's bytes."""
    cases = derive(json.loads(raw))
    spdx = "SPDX"  # split, so that REUSE does not read these lines as this file's own
    # The registered case file predates the rename (D-111) and keeps its bytes.
    lines = [f"# {spdx}-FileCopyrightText: 2026 jitLLM contributors",
             f"# {spdx}-License-Identifier: Apache-2.0",
             "# BP-F1's kernel cases, generated by bpf1_cases.py from fp16-plan.json",
             f"# (SHA-256 {hashlib.sha256(raw).hexdigest()}). Do not edit.",
             f"set {SET}"]
    for c in cases:
        lines.append(" ".join(map(str, ["case", c["name"], c["rows"], c["op"], c["impl"], *c["params"]])))
        lines += [" ".join(map(str, launch)) for launch in c["launches"]]
        lines.append("end")
    return "\n".join(lines) + "\n", len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    text, count = render(args.plan.read_bytes())
    args.out.write_text(text)
    print(f"{count} cases")


if __name__ == "__main__":
    main()
