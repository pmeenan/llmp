#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""BP-F2's case set, as pre-registered at P3 entry (docs/backend-proof.md, "Performance protocol").

Reference-only experiment tooling: it reads the M0 reference's protocol and writes the case list;
it implements nothing of llmpalooza.

  bpf2_cases.py [--protocol ../exl3-reference/protocol.json] --out bpf2-cases.txt

One line per case, `SET|NAME|ROWS|KIND`, in the timing sessions' key format (SET|NAME|ROWS, as
timing_stats.py and timing_protocol.py key them) plus what upstream runs for it:
- the M0 reference's 176 kernel cases, unchanged (the protocol's kernel_rows for each fixture's
  kernel_real_keys, and for each synthetic shape and rate), KIND `linear`: upstream's
  LinearEXL3.forward under EXL3-O, so its own choice of GEMV, GEMM or reconstruction, with the
  q and k projections' bias through ExLlamaV3's add_kernel_hhh on every path;
- the fused gate/up multi-GEMM (exl3_mgemm, as the gated MLP calls it for up to 32 rows) of each
  fixture's layer 0 at the protocol's kernel rows up to 32 (1, 8, 16, 32), KIND `multi`,
  named model.layers.0.mlp.gate_up.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MULTI_MAX_ROWS = 32  # exllamav3/modules/mlp.py: gate and up fused up to 32 rows


def cases(protocol):
    rows = protocol["kernel_rows"]
    out = []
    for fixture in ("40", "45"):
        for name in protocol["kernel_real_keys"]:
            out += [f"real-{fixture}|{name}|{r}|linear" for r in rows]
        out += [f"real-{fixture}|model.layers.0.mlp.gate_up|{r}|multi" for r in rows if r <= MULTI_MAX_ROWS]
    for k, n in protocol["kernel_synthetic_shapes"]:
        for rate in protocol["kernel_synthetic_rates"]:
            out += [f"synthetic|synthetic-{k}-{n}-K{rate}|{r}|linear" for r in rows]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=HERE.parent / "exl3-reference" / "protocol.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lines = cases(json.loads(args.protocol.read_text()))
    if len(lines) != len(set(lines)):
        raise SystemExit("duplicate cases")
    args.out.write_text("\n".join(lines) + "\n")
    print(len(lines), "cases")


if __name__ == "__main__":
    main()
