#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The per-linear sweep's cases and inputs (BP-N5; docs/backend-proof.md), shared by the reference
harness (linear_reference.py, in the ExLlamaV3 reference container), the plan writer
(native_plan.py) and the comparison (compare.py). Reference-only experiment tooling; it does not
implement llmpalooza inference and shares no code with the native launchers.

Cases. For each fixture and arm (EXL3-G: GEMV off; EXL3-O: GEMV on) and each BP-N5 row count, every
real projection of Qwen2.5-0.5B (24 layers × q, k, v, o, gate, up, down, and lm_head) through the
path upstream takes at that row count:
- rows 1 to 32: gate and up together through exl3_mgemm (the gated MLP's fused call), every other
  projection packed (exl3_gemm, which EXL3-O may send to the GEMV kernel at up to eight rows);
- rows 33 to 144: every projection packed;
- rows 145 to 1,023: the reconstruction path; 1,024 and more: the fused reconstruction.

Inputs. Row-major F16 activations, identical on both sides: element i of a case's input is
(top 11 bits of splitmix64(seed + i)) - 1024, divided by 1,024, which F16 holds exactly; the seed is
the first eight bytes (little-endian) of SHA-256 over "p3a|<fixture>|<key>|<rows>", where key names
the linear (or "<layer>.mlp.gate_up" for the pair).
"""

import hashlib

ROWS = (1, 8, 9, 16, 32, 33, 144, 145, 1023, 1024)
FIXTURES = ("4.0bpw", "4.5bpw")
ARMS = ("G", "O")
LAYERS = 24
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MULTI_MAX_ROWS = 32      # exllamav3/modules/mlp.py: fused gate/up up to 32 rows
PACKED_MAX_ROWS = 144    # exllamav3/modules/quant/exl3.py: AUTO_RECONSTRUCT_THRESHOLD
FUSED_MIN_ROWS = 1024    # reconstruct_hgemm: fused from 1,024 rows
SLICE_COLUMNS = 32768    # MAX_RECONSTRUCT_SLICE_N

MASK = (1 << 64) - 1


def linears():
    """Every quantized linear of the model, in the model's order."""
    names = [f"model.layers.{layer}.{projection}" for layer in range(LAYERS) for projection in PROJECTIONS]
    return names + ["lm_head"]


def path(rows):
    if rows > PACKED_MAX_ROWS:
        return "fused" if rows >= FUSED_MIN_ROWS else "recon"
    return "packed"


def cases():
    """Each case: id, rows, path ('packed', 'multi', 'recon', 'fused') and the linears it runs."""
    out = []
    for rows in ROWS:
        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}."
            for projection in PROJECTIONS:
                name = prefix + projection
                if rows <= MULTI_MAX_ROWS and projection == "mlp.up_proj":
                    continue
                if rows <= MULTI_MAX_ROWS and projection == "mlp.gate_proj":
                    out.append({"key": prefix + "mlp.gate_up", "rows": rows, "path": "multi",
                                "linears": [prefix + "mlp.gate_proj", prefix + "mlp.up_proj"]})
                else:
                    out.append({"key": name, "rows": rows, "path": path(rows), "linears": [name]})
        out.append({"key": "lm_head", "rows": rows, "path": path(rows), "linears": ["lm_head"]})
    for case in out:
        case["id"] = f"{case['key']}@{case['rows']}"
    return out


def seed(fixture, key, rows):
    digest = hashlib.sha256(f"p3a|{fixture}|{key}|{rows}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def splitmix64(x):
    """splitmix64's output function on a numpy uint64 array (wrapping arithmetic)."""
    import numpy as np
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def inputs(fixture, key, rows, k):
    """The case's input, rows × k, as numpy float16."""
    import numpy as np
    base = np.uint64(seed(fixture, key, rows))
    with np.errstate(over="ignore"):
        index = base + np.arange(rows * k, dtype=np.uint64)
        bits = splitmix64(index) >> np.uint64(53)
    values = (bits.astype(np.int64) - 1024).astype(np.float32) / np.float32(1024)
    return values.astype(np.float16).reshape(rows, k)


def splitmix64_scalar(x):
    z = (x + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def slices(n):
    return [min(SLICE_COLUMNS, n - start) for start in range(0, n, SLICE_COLUMNS)]


if __name__ == "__main__":
    # A self-check of the generator against its scalar definition, and the case count.
    import numpy as np
    x = inputs("4.0bpw", "lm_head", 1, 896)
    s = seed("4.0bpw", "lm_head", 1)
    for i in (0, 1, 895):
        want = ((splitmix64_scalar((s + i) & MASK) >> 53) - 1024) / 1024
        assert float(x[0, i]) == want, (i, float(x[0, i]), want)
    print(len(cases()), "cases per fixture and arm;", np.unique(x).size, "distinct values in one input")
