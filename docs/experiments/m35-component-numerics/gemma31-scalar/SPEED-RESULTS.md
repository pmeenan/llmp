<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 scalar repair: C1 adoption speed result

**PASS** under the prospectively registered coarse D-085 rule. All three
median slower-candidate ratios are below 1.10. The first pair passed the stop
rule, allowing the fixed two remaining pairs. All six fresh services completed
the prescribed warmup and measured request and retired by TERM with rc 0.
No retries, extra priming, debug plans or selected samples were used.

[SPEED.md](SPEED.md), committed as
`ee88938d83c74b75f6463efa17b4bcc2ccc8e7df`, fixes the workload and bounds;
[speed-results.json](speed-results.json) records all values, counts and receipt
hashes. Both arms used the same retained runtime 5965e83c…, source manifest
d3b32850…, resolved libraries and current artifact09080998… on Spark A
(`spark-c4e2`, driver 580.178.04). Context 8192, chunk 256, two slots, F16 KV and
no drafter were common. A used incumbent MMVQ policy; D used normal repaired
VecQ. This measures competitiveness against the quality-matched incumbent,
not isolated performance of the old inaccurate VecQ schedule.

Each arm used the unchanged helper's Gemma31 C1 workload: two sequential
26,000-character prompts (one warmup, one measured), temperature0 and 128
generated tokens per request. The actual prompt counts were 6,966/7,402 in
every arm, with zero cached tokens. Across the six arms, 12 complete requests
produced 1,536 tokens. The six warmup requests are excluded from reported rates.

| Pair/order | Arm | TTFT s | Decode tok/s | Wall tok/s | Peak occupancy GiB | Collector elapsed s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 / A→D | A | 20.375079 | 9.887125 | 3.852904 | 23.105507 | 73.946 |
| 1 / A→D | D | 20.755610 | 10.092368 | 3.839120 | 22.787651 | 74.138 |
| 2 / D→A | D | 20.742280 | 10.084996 | 3.839593 | 22.389313 | 73.996 |
| 2 / D→A | A | 20.378476 | 9.918208 | 3.857199 | 22.485348 | 73.645 |
| 3 / A→D | A | 20.564191 | 9.939416 | 3.838873 | 22.598343 | 73.766 |
| 3 / A→D | D | 20.747842 | 10.090905 | 3.839711 | 22.686295 | 73.992 |

| Slower-candidate ratio | Pair1 | Pair2 | Pair3 | Median | Bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTFT D/A | 1.018676 | 1.017852 | 1.008931 | 1.017852 | ≤1.10 |
| Decode A/D | 0.979664 | 0.983462 | 0.984988 | 0.983462 | ≤1.10 |
| Wall A/D | 1.003590 | 1.004585 | 0.999782 | 1.003590 | ≤1.10 |

The helper's historical `prefill_s` field is client TTFT, not isolated prefill
compute time. Each reported arm metric is its one prescribed measured pass;
medians are across three paired ratios, independently for each metric. Peak
occupancy is informational; every complete workload fit. This is the coarse
10% adoption comparison, not the M3.6 statistical 1% gate.

The actual executor authenticated the committed registration, runtime,
resolved/runnable library bytes, source and compiled identities, helper,
corpus, config and artifact metadata before and after every arm. Payload stat
identities matched the already verified publication. Every later arm's complete
workload identity was checked against pair1-A before marking it valid. Raw
records, runtime logs, hashes and supervisors remain outside Git under
`/tmp/llmp-u4-gemma31-speed`; exact bindings are in speed-results.json.
The complete parser/preflight validation ran without service or inference;
its synthetic mismatch fixtures refused all five later-arm positions before
writing a valid audit. No source/build changes were made for measurement.

Together with the registered exact 1,070-score scalar transfer and 182 positive
operand controls, this supports adoption of the generic scalar schedule repair
in the tested domain. The implementation has no family/dimension whitelist;
ordinary multirow and routed-ID schedules remain unchanged. C4 is excluded
because its ordinary multirow path is unchanged with owner-row-invariance
disabled. This result does not establish Gemma31 assistant or frozen-state
parity, Gemma26 joined-row quality, broader checkpoint qualification or U3/U4
closure. Prior numerical negatives and the original quality-auditor failures
remain recorded; no frozen expectation or quality bound changed. Final
integrated source checks are recorded by the integration owner separately.
