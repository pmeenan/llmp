<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma2 two-owner decode and ring screen

This opt-in extension admits the approved Gemma2 Q8_0 runner with one or two
independent slots. The owner-attention descriptor admits cap 50 only for
D256/H8/GQA2, two actual roots, logical cohort two and offset zero. All earlier
zero-cap descriptors and launch specializations retain their existing defaults.
The original packed MMA and independent-root kernel both query the true
softcap specialization's actual occupancy; the owner launch scales its already
scaled Q by 1/50 before applying cap 50. Scratch and stream-K geometry retain
the original kernel's partitions. No shader arithmetic, import or quantized
storage changes.

The graph option defaults off. When explicitly enabled it views the two packed
Q rows, joins their masks and retains independent K/V roots. Two one-row
segments with equal bounded local/global padded read widths select it. C1,
state-only, ragged and unequal-width segments retain ordinary attention.
Setup funds both one-row owner endpoints as well as the ordinary shapes.
Repeated unchanged selection within a held request reuses its closure, while
changed selection, state growth, Clear and restore still refresh it. Every step
checks that its actual closure is held.

The focused controls pass on Spark B (NVIDIA GB10, driver 580.178.04) on
2026-10-07. Operand controls compare cap50 at
cells256/512/1024/4352 against the original packed true MMA kernel byte for
byte and against independent FP64 attention at the existing bound. Saturating
Q/K values make the cap meaningful. They check actual occupancy/grid, exact
funding, one-byte-short refusal, repeated/fresh captured inputs and closed
metadata/shape/root refusals. Existing zero-cap and softcap primitive controls
remain in the check set.

[The native caller](../../../benchmarks/gemma2_c2_probe.cc) prepares 4,387 IDs
including BOS from each of two distinct externally supplied authored plain-text
inputs. The native and [original-image public caller](llama_c2_probe.cc) use
context8192 per slot, max_rows128, separate F16 K/V and independent 128-row
prefill chunks. A 256-row first screen and a 4,352-row ring screen each add
three supplied scalar rows, then 32 joined supplied rows per owner. The 66
finite full heads include 64 scored transitions and two unscored final rows.
Stock independently retokenizes both complete retained inputs and witnesses
actual C2 FLASH shapes/softcap metadata without downloading operands.

Two native own processes per geometry freeze heads, chosen IDs and initialized
state before stock is read. The small screen also checks eager/captured
identity and an ordinary-attention fallback's own controls without assuming
cross-policy equality. GPU-chosen tokens match the full-head path; malformed
publication leaves state/cursors unchanged. Clear retains backing. Spill and
restore reproduce initialized bytes, and the restored next head/state match
an uninterrupted replay, including wrapped local-ring history.

[Analysis](analyze.py) requires stock own repeats, finite complete heads and the
existing strict greedy/conditional-loss gate for both geometries before timing.
No new numerical allowance is introduced. The short small-geometry RNNR uses
two fresh processes per engine. Both pay independent 256-row prefills and 32
joined decode steps per owner, including a final complete verification head;
three supplied scalar rows remain off the clock. Warmup precedes logical Clear
without a paid full-KV zeroing. Original stock graphs/fusions and public backend
greedy sampling remain enabled. Its unchanged sampler also exports full F32
sampled-logit rows internally, even when the caller asks for tokens. Both
engines' natural histories and final heads must agree before reporting timing.
This is an n=2 engine screen; serving, compatible joint prefill, unequal widths,
departures, broader quality and sustained performance remain separate gates.

## Measured controls

All six installed jobs completed with exit zero: focused checks, small native
own controls, native ring controls, small stock quality, stock ring quality,
and short RNNR. The focused build ran 21 tests: the new cap50 owner operand
control, three legacy attention controls, six Gemma2 graph controls, five
Gemma2 plan controls and six Gemma3 graph controls. No full suite ran.

| Cap50 cells | Original and owner blocks/SM | Stream-K grid | Paid scratch bytes | FP64 NMSE |
| --- | --- | --- | --- | --- |
| 256 | 1 / 1 | 32 | 266,496 | 4.96161e-7 |
| 512 | 1 / 1 | 48 | 399,616 | 5.25146e-7 |
| 1,024 | 1 / 1 | 48 | 399,616 | 5.43137e-7 |
| 4,352 | 1 / 1 | 48 | 399,616 | 7.39125e-7 |

All four owner results are byte-identical to the original true-specialization
packed MMA; each FP64 error is below the existing 5e-4 bound. Repeated eager
and fresh captured inputs, exact funding and one-byte-short refusal pass.
The new zero-cap H8/4,352-cell control and retained zero-cap cases also pass.

Both native own repeats pass at each geometry; small eager/captured heads,
choices and initialized state are exact. Ordinary attention independently
passes the same own controls; its small outputs differ from owner attention,
so no cross-policy byte-equality claim is made. Actual owner decode selects
52 owner nodes, with norm/Mul, quantized GeGLU and norm/ADD also selected;
there are no norm/RoPE selections. Wrapped local-ring restore reproduces the
next head and initialized state of an uninterrupted identical-history replay.

| Prefix rows per owner | Exact native/stock full heads | Strict / positive-margin / tie differences | Relative conditional loss delta |
| --- | --- | --- | --- |
| 256 | 65 / 66 | 0 / 0 / 0 | +0.000589321% |
| 4,352 | 66 / 66 | 0 / 0 / 0 | 0% |

Each row covers 64 scored targets plus two unscored final heads and exact
stock own repeats. The one nonexact small head remains explicit: maximum raw
logit difference is 0.005218505859375, mean total variation 4.44358e-6 and mean
target NLL delta 5.89319e-6. Both existing strict/loss gates pass without a new
allowance. Stock's observed joined decode has Q `[256,1,8,2]`, K/V
`[256,512,4,2]`, mask `[512,1,1,2]` and cap50. These are controller selection
records rather than per-replay kernel counts.

| Short RNNR mean, n=2 per engine | Native | Original stock | Native latency over stock |
| --- | --- | --- | --- |
| Paid independent prefill | 100.23565 ms | 93.78680 ms | +6.876% |
| Paid joined decode | 457.25100 ms | 455.51000 ms | +0.382% |
| Total paid cycle | 557.48665 ms | 549.29680 ms | +1.49097% |

All four natural 64-choice histories and final two full heads are byte-exact.
Both native arms record eight graph captures, 44 replays and 52 selected owner
nodes. Every stock container is absent after checked retirement. This short
representative C2 screen retains a 1.49% total latency gap; it establishes
neither stock parity nor sustained or HTTP performance.

## Provenance and replay

The measured native source is based on `b986092`; SDK
`aarch64-c09daba6ac31edee`, locked GGML v0.6.0/d812 and the unchanged prepared
source lock bind the narrow build. Actual native caller SHA-256 is
`cc178ff13d636bc2593464b487548665e1187a2b7ffb632aa217ba5503689d53`;
receipt `874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89`.
The original stock image is
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`;
its public caller SHA-256 is
`271298ad26337b823977a39d0106a1b3266beb130b22540e0ee7c2a281a165fb`.
Stock uses total context16,384/per-sequence8,192, separate F16 KV,
`swa_full=false`, `kv_unified=false`, ordinary graphs/fusions and unmodified
public greedy samplers. Its full sampled-logit exports remain paid work.

Supply two external UTF-8 plain texts to the native caller's `prepare` mode,
then retain the first 4,387 little-endian i32 IDs per owner. These runs used
distinct authored cache/paging prose, not a broad held-out corpus. Retained
input ID SHA-256 values are
`75e4577e3b2f415a8d89fb09083649ab02d073ccd9b42a77bf726a104b884fb6`
and `b5b56a08ed8fe5bd9b0508837f09801b69e9ec2b09fe726da7370a1386d59845`.
The original caller must independently retokenize both complete texts before
execution. Keep native own freezes before stock reads and both quality gates
before timing; the checked-in callers and analyzer define the bounded modes.

The existing fully authenticated/deeply verified Q8_0 import is reused; its
format, writer and source identity are unchanged. The fixed prepared artifact
is `eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870`.
Source inventories, actual binary/receipt bindings, raw outputs and installed
supervisor receipts remain outside Git under
`/tmp/jitllm-m35-coordination/gemma2-owner2-raw` and Spark B's
`~/scratch/m35-gemma2-owner2/run1`. Each installed GPU job has a 600-second
limit and stops on failure; no full suite is part of this screen.

The per-task TensorFold HEAD observed on 2026-10-07 is
[`041d14a94e951834470fd514ed33e65b8be1059a`](https://github.com/ashhart/TensorFold/tree/041d14a94e951834470fd514ed33e65b8be1059a).
Its native1.0.0 README documents qualified CUDA GB10 only for Nemotron3.5 and
no Gemma2 CUDA/GGUF comparator. No TensorFold timing is claimed; earlier
comparison tasks retain their historical pins.
