<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 repaired ordinary C2 speed

This is a prospective adoption check for the ordinary multirow dispatch repair.
The numerical control uses the unchanged Gemma31 quality pair at 1,070 targets;
the A/R result is byte-pinned in `speed-pins.json`. The full-plan receipts and
their independent review remain the quality gate. The timing workload uses two
concurrent requests per cell, with 128 generated tokens per request.

The executor and six-argv real-parser check are in `run_c2.py` and
`validate_cli.py`. `PROTOCOL.md` records the fixed workload, arm policies,
collection order, and 10% adoption bounds. `registration.json` remains disabled;
only the root may integrate and commit a reviewed registration, then enable its
external envelope after review.
