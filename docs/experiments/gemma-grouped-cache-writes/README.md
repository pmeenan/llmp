<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rejected grouped independent Gemma cache writes

Neither grouped-write variant demonstrated a paid decode benefit in the 31B
C12 screen. The original variant increased mean time by 7.18%; replacing its
64-bit division with GGML's original fast divider increased mean time by 3.22%
in a later, more variable screen. Both preserve complete native outputs and
state. **Only this experiment report is retained: none of the eighteen
prototype source/test paths, registry entry, option or default changes is
adopted.** No 26B model transfer was run.

## Matched screens

Each screen used one immutable candidate helper on Spark B (`spark-b`,
`spark-56f5`), in off/on/on/off order. The fixed 31B production-suffix manual
recipe uses twelve owners in one ordinary shared-product wave, three real-root
attention quads, context 4096, max_rows 256, head cap 12, 992 untimed prefix tokens
and 32 paid forced steps. Norm, norm/ROPE and norm/ADD are enabled;
rope-store, row-products, shared-Q8 and MoE route/reduce are disabled. CUDA
graphs and fusion remain enabled. The input carrier is 49152 bytes,
SHA `d5844500…`; exact artifact, input, source and binary identities are in
[results.json](results.json).

| Variant / grouping | First (s) | Repeat (s) | Mean (s) | Pair spread (ms) |
| --- | ---: | ---: | ---: | ---: |
| Original / off | 5.67590 | 5.73358 | 5.70474 | 57.68 |
| Original / on | 5.93537 | 6.29283 | 6.11410 | 357.46 |
| Fast divider / off | 5.78521 | 6.23683 | 6.01102 | 451.62 |
| Fast divider / on | 6.04501 | 6.36428 | 6.204645 | 319.27 |

The original mean increases 409.36ms/+7.175787%; the fast-divider mean
increases 193.625ms/+3.221167%. Both on arms in each screen actually select 360
grouped steps; off arms select zero. Fast-divider spread exceeds its mean
difference, and the off times move substantially between screens. These short
results establish no win and do not isolate a dominant division cost or a
benefit from the divider change. There is no new stock/reference comparison,
noise calibration, NLL measurement or parity claim.

For **all four arms of each variant**, all 396 finite complete heads, 24 complete
initialized-state snapshots and their layouts, and 384 forced choices match
one another and the retained `a130902c…` native baseline/`976a0559…` proof.
The checks compare complete initialized bytes, not selected samples.

## Focused proof and contract

The prototype retains four original SET_ROWS outputs and their dependencies
in one arity-four step. A bounded 64-node matcher selects exact nonconsecutive
indices and delays the launch to the last selected writer. Executable cache
consumers, alias writes, overwritten inputs, unsupported descriptors and tagged
lanes retain primitives. Metadata views launch nothing. Both placement passes
preserve original writer edges and account for the extended input lifetimes.

Six focused controls pass without skips for **each** variant on Spark A
(`spark`, `spark-c4e2`): dependency/hazard refusal, descriptor/alias validation,
both placement passes and funding, grouped CUDA operands, existing primitive
row operations, and registry arity/binding. Real 26B C4/C8/C12 plans select
60/120/180 groups; 31B selects120/240/360, retaining all 240/480/720 and
480/960/1440 original writers respectively. This is plan selection evidence,
not a 26B performance transfer.

The CUDA operand proof compares every byte of four initialized padded
destinations, both guards and untouched cells. It covers source-view offsets,
valid I64 ring indices, signed zero and half/subnormal/overflow rounding.
Eager execution and two poisoned captured replays with changed valid indices
match four primitives exactly, using the original `ggml_cuda_cast<half>`.
The fast-divider variant changes one fixture width 512 to 513 to cover legal
non-power-of-two row/column division.

The launcher has zero additional scratch or GPU pointer-table bytes. Its
192-byte original or 224-byte fast-divider argument is passed by value; four
original nodes remain within the existing 512-byte/node host allowance.
**Zero scratch does not establish unchanged peak memory**: delaying writers
can extend activation lifetimes, funded by normal startup sizing and placement.
Grouping is also requested during untimed solo prefill, where unsuccessful
quartet attempts and producer-prefix walks can add planning cost. These paid
shared-decode screens do not establish end-to-end prefill benefit. The primitive
uses PDL helpers and the prototype uses ordinary stream ordering, so launch
count and launch protocol effects were not isolated.

## Compiled argument-copy hypothesis

Targeted resource and SASS inspection of the exact `aca3a70d…` and `76a89f20…`
binaries found **16 registers, zero stack and zero local memory** for both
`SetRowsGroup4Kernel` variants. Neither has local load/store instructions.
The original reads the argument directly through dynamic `LDCU.64` constant
loads; the fast-divider variant uses dynamic `LDC`/`LDC.64` constant loads.
Thus the proposed per-thread argument copy is absent in these binaries; no
`__grid_constant__` variant was implemented.

The actual pinned SDK compiler is **NVCC 13.4.92**
(`aarch64-c09daba6ac31edee`), separate from the cuBLAS 13.8 library version.
The existing `/usr/local/cuda-13.0/bin/cuobjdump` **13.0.85** decoded the exact
`sm_121a` function. An initial invalid CLI inspection failed; the corrected
function-selected inspection completed successfully. Three earlier compile
attempts also failed before tests: validator qualifications/signedness, missing
cast-helper include, and fixture iterator signedness. All failed records remain
external alongside the successful focused builds, screens and exact checks.

## Backward eligibility and limits

Qwen2's V writer targets a reshaped cache view, excluded by this contract
([graph](../../../src/kernels/ggml/qwen2_graph.cc)). Qwen3.8 has individually
compatible F16 K/V roots, but current four-owner waves carry per-slot lane tags
and consume each owner's stores before later stores; its indexer/recurrent
state is F32 ([wave planner](../../../src/engine/qwen38_wave_plan.cc)).
DeepSeek's persisted CSA/LID/HCA state is F32 with I32 indices; compressed
cache destinations are views and normal waves carry lane tags. An individual
raw F16 writer may fit, but no selectable quartet is proved
([graph](../../../src/kernels/ggml/dsv4_graph.cc),
[planner](../../../src/engine/dsv4_plan.cc)). No other-family selection or
performance claim follows.

TensorFold entry observation remains HEAD `609ca419…`, version 0.6.5, with
Gemma 26 documented on MLX and no comparable GB10 CUDA recipe; no TensorFold
inference was run. Existing Gemma strict/reference failures and wider-context
qualification remain separate. Whole-suite checks remain owner-deferred.

The discarded prototypes, frozen source4/source5 receipts, exact queues,
binaries, SDK receipt, logs, outputs and targeted disassembly remain in the
external `gemma-grouped-cache-writes-raw` and
`gemma-grouped-cache-writes-screen-raw` coordination directories. They identify
base `774515e`; the production tree has no grouped-write selector to reproduce
these screens directly. The report retains aggregate results and concise
record identities only.
