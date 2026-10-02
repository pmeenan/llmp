<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 four-request decode waves

Qwen3.8 serving now decodes up to four requests in one wave, up from fixed
pairs. Matched 8K HTTP cells complete **+20% tokens/s at C4** and **+9.8% at
C2**. C1 is unchanged, with a byte-identical reply. Spark B (`spark-56f5`),
2026-10-02.

## What limited concurrency

An nsys trace of the private C2 benchmark (`jitllm_qwen38_batch --mode natural`)
was decisive. A two-request wave cost 92–104 ms against 57 ms for one
request's verify, so pairing saved only ~19%.

| Cause | What the trace showed |
| --- | --- |
| Routed experts | 35% of GPU time. `jitllm.moe.gemv` reads each (token, expert) slot's weights separately, so pairing barely helped (4.00 → 3.66 s an arm). |
| Fixed pairs of at most 8 rows | A 4-request wave read every dense weight twice. The earlier [exact16](../qwen38-mxfp8-sixteen/README.md) rejection was a register spill: the 8-column template was instantiated at 16 and used 1,744 bytes of local memory. It was not a limit of the approach. |
| Graph replay | Wave graph keys hold each slot's KV extent (256-cell alignment), rows and pending draft rows. With four slots the keys seldom repeated: 241K eager launches against 86 graph launches, and the GPU was only ~45% busy for long stretches. |

## Changes

All but the depth policy keep every request's arithmetic. Replies are
byte-identical across old/new binaries at C1, and between 2- and 4-slot builds.

- **Groups of up to four consecutive compatible slots** replace pairs: up
  to 16 rows a joined product, and target heads joined up to 16 columns.
  The preflight, authentication and per-slot state are as before.
- **Wide MXFP8 vector kernels** (9–16 columns): x is staged a chunk at a time
  in shared memory and weights are decoded to F32 once. Each output is
  bit-identical to the 8-column kernel. A standalone microbenchmark of every
  production shape picks the launch: 2 rows a warp with 16 warps for n ≥ 4096,
  2×8 for n ≥ 2048, and a register-only form below that. Large shapes reach
  ~170 GB/s against a ~220 GB/s 1-column ceiling.
- **Expert-major routed GEMV** for products of more than 4 tokens: each
  distinct expert is read once for every slot that chose it. A block leads for
  its expert and loops over 4 (gate/up) or 8 (down) output tiles. Outputs are
  bit-identical to the per-slot kernel.
  - C2 benchmark ABBA, 2 runs each way: +7.0% shared-wave throughput. Within
    one request L2 already dedups, so ≤4 tokens keep the per-slot form.
  - Tiles: down 458 → 312 µs a layer, gate/up 588 → 544 µs.
- **Wave KV alignment 2048 cells** for multi-slot target and draft waves.
  Cells past a row's position are masked, or hidden from selection by position,
  so no result changes. Those cells are still read (vector attention and the
  block scores read every cell), so a wave backs each slot's caches through
  the same alignment: up to 1,792 more cells of state a slot. The wave plan
  caches hold 64 entries.
- **Policy:**
  - 4 funded slots.
  - Each request in a shared wave drafts depth 2. A lone request keeps its
    adaptive 2–3 depth.
  - Past two requests, drafts run per slot on their cached scalar graphs.

## Results

All cells use a fresh service, the frozen common-v2 client and 8,256-token
prompts with 256 outputs each. Rates include prefill and queueing, as in
[serving-concurrent](../serving-concurrent/README.md).

| Cell | Old binary (bookends) | New (two runs) | Change |
| --- | --- | --- | ---: |
| C4 tok/s | 25.13, 27.06 | 32.00, 32.74 | about +20% |
| C2 tok/s | 26.96, 27.01 | 29.66, 29.56 | +9.8% |
| C1 tok/s | 25.65, 25.50 | 25.62, 25.68 | ±0 |

The cleaned production build gives 32.22 / 29.72 / 25.68. Against current
TensorFold 0.6.2 NVFP4 (31.87 / 24.79 / 21.46, the earlier session's
[matched comparison](../serving-concurrent/README.md)) native leads at C2 and C1
and is level at C4: +1.1% across sessions is within the run-to-run spread
above. Against legacy Mia (39.50 / 30.05 / 20.32) it trails 18% at C4 and
about 1% at C2.

### Draft depth

| Policy | C4 | C2 | C1 |
| --- | ---: | ---: | ---: |
| Adaptive 2–3, 4 slots | 26.76 | 26.93 | 25.69 |
| Fixed 3 | 29.70 | 28.25 | 24.68 |
| Fixed 2 | 31.29, 31.57 | 29.08 | 25.10, 25.10 |
| Selected: 2 when shared, adaptive alone | 31.67 | 29.52 | 25.59 |

Fixed depth 3 wins nowhere measured. Adaptive beats fixed 2 alone by about 2%.

Funding four slots raises the fixed budget from 3.89 to 7.45 GiB (workspace
3.25 → 6.80 GiB): the wave workspace is provisioned as four times the largest
scalar activation, including a 4,096-row prefill chunk, although waves run
only verify shapes. A tighter bound is a follow-up.

## State capacity

Four requests can outgrow the conversation-state capacity that two never
reached. Under a 30 GB host-memory hold (about 3.2–3.7 GB of state
capacity at startup), four concurrent ~10.7K-token prompts generating
~4–7K-token stories had all failed with a 500 at "the chunk at 8192". With
the cohort's capacity policy
([runtime-serving](../../runtime-serving.md#state-capacity-in-a-cohort))
the same attacks complete every request: equal 10,660-token prompts finish
in 442 s and staggered 10,480–12,280-token prompts in 430 s, each with
three waits, one preemption whose generation resumed from its rebuilt
state, and one cleared idle cache; the service stays healthy and answers a
follow-up. A matched 8K HTTP set after the change measured C4 / C2 / C1
32.45 / 30.04 / 25.61 completed tok/s, the C1 reply identical to the
earlier cell (spark-b, 2026-10-02; records under `~/scratch/ccqw-fix/`).

## Not adopted

- **Joining plain BF16 products** (draft heads, drafter products, router).
  GGML routes some joined products to cuBLAS, which the authentication
  refuses. A per-product selector probe is owed.
- **jitLLM's own HC kernels on every column count** (a cluster split-K down
  product, batch-invariant by construction). They are neutral solo and slower
  than cuBLAS at 12–16 columns, and they change solo trajectories. Reverted.
- **Grouped drafts past two requests**: 3.6% slower at C4.

## Open

- **Replies under C2/C4 vary run to run.** Wave composition depends on arrival
  timing, and the joined HC (cuBLAS) and head (GGML MMF) products are not
  column-count invariant. Solo runs repeat exactly. Batch invariance would
  need row-invariant head and HC products.
- **Per-wave budget at 12 rows** (~146 ms, ~10% gaps):

  | Component | ms |
  | --- | ---: |
  | Routed experts (near bandwidth) | ~52 |
  | MXFP8 | 19 |
  | Draft heads | 7.5 |
  | HC | 6.7 |
  | Target head | 5.4 |
  | HC prep | 4.4 |
  | GDN commit | 3.9 |

  The C4 gap to Mia is decode rate, about 58 vs ~100 tok/s. Prefill is similar.

Raw records are on spark-b under `~/scratch/cc-http/` and `~/scratch/cc-prof/`.
The microbenchmarks are in `~/scratch/cc-mxbench/`.
