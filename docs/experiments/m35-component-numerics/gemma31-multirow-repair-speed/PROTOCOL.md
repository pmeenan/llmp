<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 repaired ordinary C2 adoption speed (preparation only)

This bundle is disabled preparation. It does not authorize collection. Root must
integrate and commit the registration, then enable its external envelope after
the original-A versus R quality pair has passed manual and automated review.

Use the same retained R runtime, dependencies, source closure, and Gemma31
artifact for both arms. Both arms disable `owner_padded_reads` and use
`max_decode_rows 16`; A additionally disables `grouped_vecq`, `vecq_glu`, and
`vecq_inputs`. R uses ordinary repaired dispatch. Keep diagnostic capacity and
startup settings equal. No drafter or other new grant/rank setting is admitted.

Run the pinned `speed_gate.py run --family gemma31 --cells 2` helper with the
frozen 8192 context, F16 KV, two slots, 256-token prefill chunk, full corpus,
26000-character prompts, temperature 0, and 128 output tokens. Each fresh
service runs one warm-up C2 cell on prompt indices 0 and 1, then one measured
C2 cell on indices 2 and 3. Preserve all timing, usage, raw workload, memory,
error, and lifecycle records. Require two successful 128-token replies per
cell, unchanged prompt identities, zero cache, finite positive metrics, and
normal service retirement. This is a capped-length timing workload; it makes
no semantic-completion claim.

Run exactly three paired arms in order A/R, R/A, A/R. Use a fresh supervised
exclusive GPU job for every arm, at most 600 seconds with 30 seconds grace.
Stop after pair 1 if any candidate slowdown ratio exceeds 1.10. Otherwise
finish all three pairs and require each median ratio at most 1.10. Compute
R/A for C2 client-cohort TTFT (`prefill_s`) and A/R for C2 `decode_tok_s` and
`wall_tok_s`. Report every absolute value, paired ratio, median, and actual
occupancy. No 1% migration gate, calibration, C4 test, or new quality domain.

Before collection, independently review the completed R/A quality pair,
including its relational plan review, and bind both arms' manual receipts and
the pair report. Run read-only preflight plus all six actual argv through the
real parser with process, service, and network calls forbidden. This bundle's
validation is setup-only; actual collection remains externally disabled.
