<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 exact16 MXFP8 products: rejected

The private four-request control passed every equality check, but combining
four four-row target products into one 16-column MXFP8 product made the paid
generation loop 7.469× slower. Retain the paired eight-column implementation.
This is a single diagnostic ABBA sequence with two observations per arm; it
does not change a production kernel, selector, default or API.

The four frozen spec-C4 prompts each contain 8192 literal token IDs. Both
arms use the same target, selected 47172-token drafter, context 33792,
chunk 4096, fixed depth 3 and 256 outputs per request, including the prefill
anchor. The timed loop pays 1020 subsequent outputs, all three activation
concatenations, product and split consumers, ID copies, normal PLE,
cold private plan/graph setup and final independent settlement. Prefill,
state-copy/equality diagnostics and kernel-attribute collection are outside
that clock. Only complete four-slot, four-row target barriers select the
new entry; draft, routed, inactive and ragged products retain their existing
paired paths.

| Chronological arm | Paid loop seconds | Aggregate tokens/s | Draft seconds | Verify seconds | Settlement seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paired 8, first | 21.094487 | 48.353866 | 3.060607 | 17.574709 | 0.454937 |
| Exact 16, first | 158.549307 | 6.433330 | 3.192828 | 154.891646 | 0.448700 |
| Exact 16, second | 157.437719 | 6.478752 | 2.984795 | 153.991872 | 0.445769 |
| Paired 8, second | 21.211643 | 48.086798 | 3.067037 | 17.671857 | 0.455246 |

The two-arm mean times are 21.153065 and 157.993513 seconds. The much larger
Verify interval identifies the changed target-product path as the useful
follow-up scope; these host intervals are not individual kernel timers.
Every arm has the same 101 active waves, all 1024 output IDs, acceptance
trajectory, final initialized pages and pending cursors. The complete new
window also matches full draft inputs/logits, target logits and IDs between
paired/full, paired/IDs, exact16/full and exact16/IDs. All 72 raw arrays,
16 window-output controls, 32 state controls, eight canonical captures and
24 exact page comparisons pass. The four-slot baseline is 2,720,432,128 B;
the separate controls charge 27,204,321,280 B of checkpoint copies and
21,763,457,024 B of state reads.

The device kernel's source body and launch geometry formulas are unchanged
from the checked paired implementation. Instantiating sixteen columns changes
compiled resource use:
the seven larger geometries report 190 registers and zero local bytes for
eight columns, versus 40 registers and 1744 local bytes for 16. The
48×2560 geometry reports 164/zero versus 255/144. Rows per warp, warps per
block and zero static/dynamic shared bytes remain the same. These local-memory
attributes are a concrete concern; no occupancy, local-memory traffic or
sole-cause attribution was measured. Per arm, paid activation packing grows
from 9,392,856,368 to 17,385,954,608 B. Routed launches remain 19,992; the
candidate replaces most paired MXFP8 launches with 36,096 wider launches.

The target-only build completed in 34.676 seconds. The model job took
455.402 seconds, exceeding the provisional 3–5-minute estimate because the
candidate was slow, while remaining inside its existing bounded allocation.
It completed without retry; all five child commands exited zero and were
reaped in the supervised group. Receipt-bound final retirement was
117.026 GiB with no active model. Build receipt `3dce82e1…`, outer control
`2ffc7519…`, native control `06bca230…`, source map `70f3039a…` and binary
`12ed2d41…` bind the actual pre-adoption experiment. The source, executable, SDK-relative libraries, 256 target objects/archives
and all 72 raw arrays are preserved outside Git under
`~/scratch/m3-qwen-mxfp8-sixteen-r1/preserved-r1/` on Spark B, bound by
preservation `d1238386…` and inventory `e288bd51…` (1630 files). The
preservation completed with all children reaped and 116.985 GiB free.

This negative result closes the unchanged-reduction exact16 axis. A separate
ordinary BF16 full-target-head sharing factor is evaluated independently; no
HTTP throughput, comparator parity or quality conclusion follows here.
