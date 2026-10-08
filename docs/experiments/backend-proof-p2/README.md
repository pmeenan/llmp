<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Backend proof P2: resident FP16 — 2026-09-27

This is P2 of the [backend proof](../../backend-proof.md#stages): native
execution of the Qwen2.5-0.5B FP16 fixture from its v0 prepared artifact,
judged by the FP16 Tier E gate (the recorded-plan match, then bit-exact
logits against the toolchain bridge) and by memory. The memory judgment
was the allocation census until D-085 replaced it with a coarse peak check
([below](#memory-check-d-085)). The gate order is pre-registered: the
bridge's `F` cap first, then FP16-U `control`, then FP16-F `control`, then
`heldout` for both, each plan match before its logits.

**Results in brief** (2026-09-27, `spark-b`, rung 3: `cudaMalloc`; peaks on
`spark`):

| Arm | Plan (`plan_compare.py`) | Logits against the bridge | Repeat | Census v1 | Peak against the bridge (D-085) |
| --- | --- | --- | --- | --- | ---: |
| FP16-U `control` | MATCH, exit 0 | bit-identical, `3560d337…` | exact | FAIL (below) | 0.96× |
| FP16-F `control` | MATCH, exit 0 | bit-identical, `bb8ae5e7…` | exact | FAIL (below) | 0.81× |
| FP16-U `heldout` | MATCH, exit 0 | bit-identical, `69ff0821…` | exact | FAIL (below) | 0.95× |
| FP16-F `heldout` | MATCH, exit 0 | bit-identical, `bfb36f19…` | exact | FAIL (below) | 0.94× |

Each arm's plan matched `fp16-plan.json` completely (every chunk of both
evaluations run, with cuBLAS's logs, SASS hashes and an nsys trace of the
whole process), and only then were its logits compared: the first
evaluation's F32 logits have the bridge's recorded SHA-256, and the second
evaluation, run from a cleared cache in the same process, equals the first
bit for bit. BP-S1 holds: the fused RMSNorm-mul implementation is exact
against the fused arms and the unfused one against the unfused arms.

Rungs 4 and 5 (the same fixture paged into device VMM through the landing
zone, D-081, then evicted, restored and relocated) are
[below](#rungs-4-and-5-paged-into-device-vmm-through-the-landing-zone):
on all four arms the plan matches and every evaluation is bit-identical to
the bridge.

## What runs

- **Model description.** M2 has no GGUF reader, so the Qwen2.5-0.5B
  hyperparameters are compiled in (`src/model/qwen2.cc`) and bound to the
  artifact's resources at load, every tensor's type and shape checked
  (`BindQwen2`). [`gguf_profile_check.py`](gguf_profile_check.py) checks
  the profile's scalars against the GGUF's key/values on the reference
  side, with the M0 prototype's GGUF reader (`layout.py read_gguf`; the
  bridge host has no numpy for gguf-py). Run on `spark-b` against the FP16
  GGUF (`8e0ae260…`, header `1bdb17cd…`): every value agrees. Its first run
  caught one error: the profile had RoPE's original context as 32,768
  where the GGUF says 8,192 (`qwen2.context_length`, which the bridge's log
  also prints as `n_ctx_train`). With YaRN off (`ext_factor` 0) that value
  does not reach the arithmetic, and both plans and logits matched before
  and after the fix; the gates above were rerun after it.
- **Weights (rung 3).** `benchmarks/fp16_exec.cc` opens the artifact
  (`b93cdc32…`) with the native reader, plans direct reads of every chunk
  of every group (`Artifact::PlanReads`: coalesced runs of at most 64 MiB),
  reads them with `O_DIRECT` into pinned staging, checks each chunk's
  SHA-256 against the index (490 of 490), and copies each group into one
  `cudaMalloc` region (988,221,440 bytes: the artifact's stored bytes; the
  tensors' own bytes are 988,208,640, so 12,800 bytes are the groups'
  alignment pads). The token table is also kept on the host for the
  embedding lookup, as the bridge keeps its 259.66 MiB `CUDA_Host` copy.
- **Graph.** `src/kernels/ggml/qwen2_graph.cc` builds each chunk's graph
  with GGML's graph functions as llama.cpp builds it: 941 GPU nodes, which
  with the embedding lookup the bridge runs on the CPU are the "graph nodes
  = 942" of its log. The inputs are copied in the scheduler's order
  (embedding rows, positions, K cells, V elements, mask, output rows). n_kv
  is padded to 256 and the cache holds position p in cell p
  (`model/qwen2.h`).
- **Plan.** `graph_plan.cc` walks the nodes as the CUDA backend does. With
  fusion on it asks upstream's gates in `ggml_cuda_try_fuse`'s order
  (`fusion.h`, which gains the RMSNorm-mul gate and a conservative check
  that no pattern llmpalooza lacks can apply) and the device's MMVF, MMF or
  cuBLAS choice (`ops.h SelectMulMat`); with fusion off it runs node by
  node, the RMSNorm-mul pair as the unfused implementation. Activations are
  placed largest first in one region, 128-byte aligned, from a plan made
  with every tensor at a distinct address; the plan is then made again
  over the real addresses and must be the same. Every bound tensor is
  checked for 128-byte alignment.
- **Execution.** Each chunk's plan is resolved against the registry
  (`executor.h`), which now also declares RMSNorm, add, mul and the three
  matrix product families, and run through the K-C launch context on one
  provider stream. The cuBLAS handle and its 32 MiB workspace are created
  lazily, before the first chunk whose plan calls cuBLAS, as the bridge
  creates them.

## Running the gates

[`run_native.sh`](run_native.sh) runs one arm on a Spark: `plan` (the
recorded run under `nsys profile --trace=cuda` with cuBLAS's and
cuBLASLt's logs, the binary's SASS hashes from `cuobjdump -sass`, then
`plan_compare.py convert` and `compare`), `logits` (the same run's logits
against the arm's recorded SHA-256), and `census`. The census runs below
were made under rule version 1, by the `census` stage of the commit that
recorded them (`d2601ad`, judged by `census.py`). Since D-085 the stage is
`peak`, the [memory check](#memory-check-d-085), and `census.py` and
`fp16-f-caps.json` are in Git history only. Conditions of the runs
above: `spark-b` (GB10, kernel
7.0.0-1019-nvidia, driver 580.178.04), the `cross` build (SDK
`x86_64-e0a0c85c42806fb1`) deployed there, `llmp_fp16_exec` SHA-256
`0f4d6331…`, cuBLAS 13.8.0.4 (the libraries' hashes equal the record's),
nsys 2025.3.2, cuobjdump 13.0.85, `CUDA_DISABLE_PTX_JIT=1`. Token IDs: the
bridge's `control` `tokens.txt` (76 IDs, SHA-256 `37e46a23…`) and the
declared held-out IDs (`6dd8da89…`). Raw recordings, traces, logs and
logits stay on the Spark.

| Arm | Chunks recorded | Tokens compared | Distinct kernels | Distinct cuBLAS calls |
| --- | ---: | ---: | ---: | ---: |
| FP16-U `control` | 90 | 55,996 | 24 | 7 |
| FP16-F `control` | 90 | 34,730 | 26 | 7 |
| FP16-U `heldout` | 70 | 44,672 | 37 | 14 |
| FP16-F `heldout` | 70 | 28,874 | 39 | 14 |

Between chunks only the weights' 490 uploads and two cache clears ran.

## The bridge's census: the FP16 `F` cap

Pre-registered in [backend-proof.md](../../backend-proof.md#memory-and-workspace-the-m2-gate-in-exl3-bringupmd)
before any native census result was seen.

- **Harness.** [`fp16_census.cc`](fp16_census.cc) drives llama.cpp's public
  API as [`fp16_reference.cc`](../backend-proof-p0/fp16_reference.cc) does,
  first evaluation only, reading the census counters at every step and
  running the rule's controls after the context and after the evaluation.
  (That is the revision of `d2601ad`, binary `b6bd4f6b…`. The file now
  takes three reads per reading and two evaluations; the memory check
  runs it.)
  [`census_bridge.sh`](census_bridge.sh) compiles it on a Spark and links it
  with the P0 bridge build's own libraries and link line.
  `census.py` (removed under D-085) attributed each interval
  (`bridge RUN... --plan fp16-plan.json --arm ARM`), adding the buffers the
  API cannot report from P0's record, and wrote the caps
  (`fp16-f-caps.json`, `--caps-out`).
- **Conditions.** `spark-b` (`spark-56f5`; GB10, kernel 7.0.0-1019-nvidia,
  driver 580.178.04), no other GPU work, the GGUF (`8e0ae260…`) read into
  the page cache first, P0's environment (`CUDA_DISABLE_PTX_JIT=1`,
  `GGML_CUDA_DISABLE_GRAPHS=1`, and `GGML_CUDA_DISABLE_FUSION=1` unfused), a
  250 ms settle before each reading. Three processes per arm, arms
  interleaved. The binary (`b6bd4f6b…`) was built on `spark` against the
  bridge build (SDK `aarch64-f469d317c88c3044`) and ran with the bridge's
  cuBLAS 13.8.0.4 and cudart 13.4.92. Every run reproduced its arm's
  recorded logits hash.
- **What it found.** `MemAvailable` does not follow host allocations on
  this kernel (RE-024); corrected for the per-CPU page lists, R is about
  1 MiB. The CUDA context costs about 240 MiB of unexplained growth, model
  load and context creation 16 to 20 MiB more, and the first phase that
  calls cuBLAS 67 to 70 MiB. The three processes of a `control` arm agree
  within 0.5 MiB at every step; those of a `heldout` arm within 3.2 MiB
  (one FP16-U `heldout` process sits about 2 MiB above the other two from
  the context on). The fused and unfused arms' caps agree within 2 MiB.
  Runs made before the correction, with a 50 ms or 2.5 s settle, scattered
  by up to 190 MiB and are not used.

## The native census

`run_native.sh census`: a run of its own (no nsys, no logs, no recording),
two evaluations, the rule's controls after the context and after the
evaluations, the harness's own buffers declared as the catalog (weights,
KV, the activation region, the pool scratch, the pinned staging, input and
logits buffers, the host token table, the cuBLAS workspace once created),
judged by `census.py native --caps fp16-f-caps.json`. Each phase's
observed peak is the bytes its plan places: A, the activation region's
extent; S, the plan's pool scratch; I, the input bytes copied; L, the
logits copied. The readings wait 1.5 s each: this harness faults pages in
between readings (the logits it keeps), and at 250 ms `MemAvailable` swung
by up to 150 MiB until every CPU's counter deltas were folded in (1.5 s
exceeds `vm.stat_interval`; the bridge, which touches no new pages between
readings, read steady at 250 ms). A monitor voids a run if another process
of this user holds an NVIDIA device meanwhile.

The 1.5 s wait is a method change made after native census readings had
been seen, not before: the first native run (FP16-U `control`, 250 ms, void
through its host control) showed those swings, and one further
single-evaluation run at 1.5 s, not judged, was read to confirm they went
away. The wait moves no bound, but it changed native's readings after they
were seen, and the bridge's cap was not re-measured at 1.5 s, so whether a
1.5 s bridge reading would move the cap is not known. It is reported here
so the owner can weigh it. The void 250 ms run's charges were larger
(102 MiB over the cap at chunk 0); at 1.5 s the outcome below still fails.

**Outcome: every arm fails the census rule as pre-registered.** No bound
was moved. What each arm shows (binary `0f4d6331…`, 2026-09-27, `spark-b`):

- **Placement and KV pass everywhere.** In every phase of every arm A,
  S, I and L equal the itemized limit's terms exactly (for example the
  512-row prefill: 312,999,936 + 156,499,968 + 3,940,352 + 311,164,928 =
  784,605,184, the limit), so `E` has no slack. KV equals its declared
  layout, and the persistent library workspace llmpalooza allocates is the
  32 MiB cuBLAS workspace; in FP16-U `control` the 0.51 MiB charged to it
  at the controls step (below) takes it over its limit.
- **What is charged, and fails:**

  | Arm | R (MiB) | Charged at warm-up steps | Charged in evaluation 2 |
  | --- | ---: | --- | --- |
  | FP16-U `control` | 1.8 | 0.51 MiB at the controls step (to the persistent workspace); 135.1 MiB at chunk 15 | none |
  | FP16-F `control` | 2.0 | none | 4.57 MiB in chunk 13's interval (−5.79 MiB in the next) |
  | FP16-U `heldout` | 3.8 | 0.30 MiB at chunk 15 | 27.55 MiB before chunk 1 (−27.03 MiB in the next) |
  | FP16-F `heldout` | 4.1 | 0.70 MiB at chunk 0, 0.08 MiB at chunk 1, 0.19 MiB at chunk 11 | none |

  (R in this table is each run's resolution from its own controls.)
- **What the charges are.** Two kinds. Warm-up excesses of 0.08 to 0.70 MiB
  over the bridge's cumulative cap at one step, each below the run's R;
  the rule allows no resolution at warm-up steps. And one-interval growth
  that reverses within the next few intervals (4.6, 27.6 and 137 MiB), with
  no move in SUnreclaim or the catalog, and RssAnon moving only by the
  logits the harness keeps. At the end of the first evaluation native's
  cumulative unexplained growth is below the bridge's cap on all four
  arms: 323.3 against 326.7 MiB and 323.1 against 327.0 (`control`), 347.0
  against 353.0 and 347.3 against 351.6 (`heldout`); at the end of the
  second it is within 1.1 MiB of those figures.
  Native's CUDA context (+240 MiB) and first cuBLAS phase (+69 MiB with the
  handle) match the bridge's.
- **Runs not counted.** Six native census processes ran before the
  counted batch; none is counted.
  - The first (FP16-U `control`, 250 ms) was void: the harness's host
    control never freed its memory (a vector assigned `{}` keeps its
    capacity). Its readings led to the 1.5 s wait (above).
  - One FP16-U `control` evaluation at 1.5 s, to check the wait; not
    judged.
  - A first batch of all four arms at 1.5 s, from a binary before the
    clang-tidy fixes to the harness and planner: FP16-U `control` was void
    (its VMM control moved the wrong way after the evaluations); FP16-U
    `heldout` overlapped another worktree's GPU tests on `spark-b`
    (`check:spark` of `cross`, `cross-asan` and `cross-tsan` at
    06:06–06:07), which moved `MemAvailable` by gigabytes. The other two
    were valid under the rule by the reconstructed timeline: FP16-F
    `control` **passed** the census, and FP16-F `heldout` failed on
    warm-up excesses of 0.61 MiB at chunk 0 and a 4.05 MiB charge in
    evaluation 2. The whole batch was set aside for the final binary and
    the monitor, added after it.

  The runs in the table are the next batch, the first with the monitor,
  which found no other GPU process. FP16-F `control` passing in one valid
  run and failing in the next is itself a result: under this rule a single
  run's census verdict does not reproduce.

## Rungs 4 and 5: paged into device VMM through the landing zone

**Results** (2026-09-27, `spark-b`, the `spark-native` build,
`llmp_fp16_paged` SHA-256 `5b10fd5b…`):

| Arm | Plan (`plan_compare.py`) | Evaluation 1 against the bridge | Evaluations 2–4 against 1 | Bound tensors outside the catalog |
| --- | --- | --- | --- | --- |
| FP16-U `control` | MATCH, exit 0 | bit-identical, `3560d337…` | 0, 0, 0 bit differences | 0 of 443,160 |
| FP16-F `control` | MATCH, exit 0 | bit-identical, `bb8ae5e7…` | 0, 0, 0 | 0 of 443,160 |
| FP16-U `heldout` | MATCH, exit 0 | bit-identical, `69ff0821…` | 0, 0, 0 | 0 of 344,680 |
| FP16-F `heldout` | MATCH, exit 0 | bit-identical, `bfb36f19…` | 0, 0, 0 | 0 of 344,680 |

Evaluation 1 is rung 4 (the weights paged in before it), 2 its repeat from
a cleared cache, and 3 and 4 are rung 5: each evicts every weight at the
profile's restore point (after 32 tokens for `control`, 33 for `heldout`),
releasing its backing, pages it all back in and continues; evaluation 4
brings the weights back at a second reservation (relocation). Rung 4 is
therefore bit-identical to rung 3, and rung 5 to rung 4, on all four arms
(Tier E, approved 2026-09-26).

- **The harness.** [`fp16_paged.cc`](../../../benchmarks/fp16_paged.cc)
  plans each chunk as rung 3's harness does (`fp16_common.cc` carries
  a copy of its planning; the plan gate below checks the result), over memory that the scheduler and its lanes manage:
  - *Weights:* every chunk of every group is an extent of device VMM with
    managed backing (D-033). Group g has a 2 MiB-aligned region and chunk k
    maps at its base + k × 2 MiB: 490 extents, 1,027,604,480 bytes of
    backing for the artifact's 988,221,440 stored bytes (the groups' last
    chunks round up to 2 MiB; the tensors' own bytes are 988,208,640).
  - *The token table's host copy:* its group's 130 chunks again, read in
    place into host VMM (272,273,408 bytes), because the recorded plan
    looks embeddings up on the CPU, as the bridge does from its CUDA_Host
    copy. The tied output head reads the device copy. This is BP-P3's
    duplicated-storage arm, and the only weight bytes held twice.
  - *The landing zone:* 8 slots of 2 MiB of host VMM (2 × depth 4, D-081),
    mapped at setup and cataloged as pinned staging.
  - *Mapped at setup, in 2 MiB extents of device VMM:* the cache (6 MiB for
    `control`, 12 MiB for `heldout`), the activations (20 and 300 MiB), the
    GGML pool scratch (10 and 150 MiB) and the 32 MiB cuBLAS workspace.
    The input staging and the logits are pinned host memory, cataloged.
- **The page-in** is the scheduler's (`scheduler.h`): per extent, the
  device lane creates and maps its backing, the load waits in order for a
  slot, the storage lane reads the chunk into it with `O_DIRECT` (io_uring,
  depth 4), the device lane copies it into place with the copy engine on
  its own stream, and the extent is published, and the slot freed, only
  when that copy's fence has completed. Reads are one per chunk (at most
  2 MiB), not coalesced across chunks.
- **Execution.** Each chunk is one device job on the compute stream,
  holding a lease on every extent above until the fence after it
  completes: the embedding lookup from the host table, the inputs copied in
  the bridge's order, the plan bound through the registry under the K-C
  launch context, and the logits copied back. The cache clears and the
  cuBLAS handle's creation (before the first chunk that calls cuBLAS, as
  the bridge creates it) are jobs too. Eviction takes each weight extent
  out of lease and unmaps and releases it on the device lane (the
  provider's backing count drops by 620); the next load maps fresh
  backing.
- **BP-A1's in-process check.** Before each chunk, every tensor its graph
  binds (each node and each of its sources, cuBLAS's operands among them)
  must lie in resident, cataloged extents of device memory of one class,
  and that class must be the tensor's: weights for weights (52,200 bindings
  in `control`), live state for the cache (69,120) and scratch for the
  activations and inputs (321,840). None fell outside. The GGML pool draws
  only from the scratch region the launch context was given, and cuBLAS's
  workspace is the cataloged workspace region.
- **The plan gate.** [`run_paged.sh`](run_paged.sh) `plan` runs the harness
  with every lane driven from the recording thread (`--lanes inline`: the
  launch recorder sees only its own thread) under `nsys --trace=cuda`, with
  cuBLAS's logs and the binary's SASS, and `plan_compare.py` compared all
  four evaluations, 180 chunks (`control`) and 140 (`heldout`), with
  `fp16-plan.json`. Between chunks ran 1,470 copies (490 per load, three
  loads) and 4 memsets (the cache clears), and no kernel or cuBLAS call.
  `logits` then checked the hashes. The distinct kernels and cuBLAS calls
  are rung 3's (24, 26, 37 and 39 kernels; 7 and 14 calls).
- **With threads.** `run_paged.sh threads` runs each lane on its own
  thread, as a program wires them (no recording): all four arms gave the
  same hashes, zero bit differences and no coverage violation. The
  recorder cannot see the lanes' threads, so the plan gate runs inline;
  to check that inline turns hide no ordering, `control-fused` was also
  run threaded under `nsys --trace=cuda` and its trace compared with the
  inline gate run's (binary `a4e1b759…`): the same 64,168 kernels in the
  same order on the compute stream, the same copies and memsets per
  stream and kind (the 1,470 page-in copies on the copy stream), and no
  page-in copy overlapping a kernel in either.
- **Conditions.** `spark-b` (GB10, kernel 7.0.0-1019-nvidia, driver
  580.178.04), SDK `aarch64-e0a0c85c42806fb1`, cuBLAS 13.8.0.4, nsys
  2025.3.2, cuobjdump 13.0.85, `CUDA_DISABLE_PTX_JIT=1`, the FP16 artifact
  `b93cdc32…` on the NVMe root file system. Another agent's GPU jobs ran on
  the host during the gate runs; the recording and trace are this
  process's alone, and exactness does not depend on timing. The gates and
  threaded runs were made twice, on binaries `832b06c9…` and `5b10fd5b…`
  (the second adds a fence before teardown), with the same results.

**Page-in throughput** (BP-P6, reported, not gated). `run_paged.sh loads`
loads the 490 device weight chunks (988,221,440 bytes, reads of at most
2 MiB, depth 4) and evicts them, five times per process, through the same
scheduler and lanes on their own threads, in four variants interleaved over
three rounds. Each process started once no other process had held the GPU
for 10 s (one start found another process). GB/s of the second to fifth
loads (the first includes the lanes' warm-up and ran 8.6–13.5):

| Variant | Range | Median |
| --- | ---: | ---: |
| Through the zone, backing made on each load (the runtime's path) | 10.65–12.01 | 11.4 |
| Through the zone, backing mapped once at setup | 11.67–12.18 | 11.9 |
| In place into host VMM (D-034's path), backing made on each load | 6.50–7.23 | 6.9 |
| In place into host VMM, backing mapped once | 13.02–13.94 | 13.8 |

- D-033's per-load backing costs little through the zone (the device lane
  maps ahead of the reads) but halves in-place loads: making host backing
  the CPU maps is slow.
- With backing mapped once, the zone's loads run about 14% below in-place
  reads. The zone's size is not the limit: 12, 16 and 32 slots gave no
  more, in runs made while another process's kernels ran. D-081's
  standalone measurement (8 GiB of 2 MiB reads, a tight loop) found no
  cost at depth 4, so the gap lies in this runtime's hand-offs, not
  measured further here. Against D-081's reopen condition, which compares
  loads through the zone with in-place reads, this is a measured shortfall
  with like backing, and a gain with the runtime's managed backing.
- The rung-5 restores in the final threaded gate runs took 94–106 ms for
  the device weights (9.3–10.5 GB/s) and 37–51 ms for the host table's
  272 MB (managed host backing).

**Since closed** ([pagein-perf](../pagein-perf/README.md), 2026-09-27).
The gap was the runtime's, not the copy's: the direct reader started a
load's reads out of file order, which this SSD serves ~18% slower at the
artifact's 4 KiB-aligned offsets (RE-026); the storage and device
submission lanes slept between reads and copies (RE-017); and VMM work
held up the copies on the submission lane. With reads in order, polling
lanes and a VMM lane of its own, this harness's loads through the zone
matched in-place reads, and both, on this artifact's file, run ~11% below
a fresh copy of it (RE-027); the numbers are in that report. The harness
now gives VMM work to that lane. Rungs 4 and 5 were re-run on `spark-b`
with it (binary `a60ec0cd…`): `threads` on all four arms gave the recorded
hashes, zero bit differences and no coverage violation, and `plan` on
`control-fused` matched every chunk.

## Memory check (D-085)

D-085 stopped the census rules and their nsys passes. Memory is judged
loosely: a native run's peak, by ordinary counters, may be at most about
10% above the bridge's. On `spark` (idle, 2026-09-27) each engine ran once
per arm under [`peak_memory.sh`](peak_memory.sh) (`run_native.sh peak`, and
`fp16_census` for the bridge, both with a 50 ms settle). All four arms pass:

| Arm | Native peak (MiB) | Bridge peak (MiB) | Ratio |
| --- | ---: | ---: | ---: |
| FP16-F `control` | 1,886 | 2,321 | 0.81 |
| FP16-U `control` | 2,019 | 2,108 | 0.96 |
| FP16-F `heldout` | 3,716 | 3,973 | 0.94 |
| FP16-U `heldout` | 3,473 | 3,654 | 0.95 |

Method, conditions and the `spark-b` run that was not used are in
[backend-proof.md](../../backend-proof.md#memory-and-workspace-the-m2-gate-in-exl3-bringupmd).
Before D-085, two more census rules were tried; their text and tools are
in Git history:
- **Version 2** added an exact nsys tier to the counters. Its counter tier
  failed a holdout on the bridge itself (a 675 MiB jump with nothing
  API-visible behind it), and it was never registered.
- **Version 3** gated on the nsys tier alone. It passed its bridge holdout
  on all four arms and was dropped before registration.

What the nsys passes found: cuBLAS's handle creation keeps a 64.1 MiB
default workspace pool that `cublasSetWorkspace` does not free (RE-028).

## Not covered here

- **The cache is not evicted at the restore point:** spilling state takes
  the reverse path through the zone (write-back). Rung 5 here evicts and
  restores the weights, as BP-P1 asks. BP-P4's cache eviction came later,
  with write-back
  ([aggregate report](../backend-proof/README.md), `--spill`).
- **Relocation rebuilds every descriptor** because each chunk is planned
  and bound anew. Nothing captures a pointer across chunks here (no CUDA
  graphs, no pointer tables), so BP-P5's rejection of stale ones is not
  exercised.
- **The memory check covers rung 3 only.** The paged harness (rungs 4
  and 5) has not been measured against the bridge. Version 1's census
  charges are recorded as that rule made them.
