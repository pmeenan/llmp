<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Backend proof: aggregate report — 2026-09-27

The early backend integration proof ([scope](../../backend-proof.md))
ran real GGML FP16 and ExLlamaV3 kernels on llmpalooza-owned memory under
llmpalooza's dispatch. This report gathers its stages and the status of every
case, and gives the results of the last stages: P4 (paging), P5 (lifetime
and failure) and P6 (the contract, recorded as D-086).

## Stages

| Stage | Result | Report |
| --- | --- | --- |
| P0 Bridges and controls | Both toolchain bridges bit-exact against their references; profiles, bounds and protocols approved or pre-registered | [P0](../backend-proof-p0/README.md) |
| P1 Substrate probes | GGML launchers under llmpalooza's K-C context; cuBLAS handle and workspace injection; plan selection between implementations; BP-F1 failed on host VMM and passed on device VMM (D-081) | [P1](../backend-proof-p1/README.md) |
| P2 Resident FP16 | Native FP16 matches the bridge bit for bit, plan gate included (rung 3); paged into device VMM through the zone, evicted, restored and relocated, bit-identical (rungs 4, 5) | [P2](../backend-proof-p2/README.md) |
| P3 Resident EXL3 | Every linear byte-equal to upstream at a forced plan (BP-N5); both fixtures end to end, the executed plan equal to the record, Tier E at operation level, Tier C; rungs 4 and 5 bit-identical | [P3](../backend-proof-p3/README.md) |
| P4 Paging | Write-back through the zone; partial evictions, state spill and shared-storage views on both representations, bit-identical | below |
| P5 Lifetime and failure | Cancellation, event permutations and injected failures on the real providers | below |
| P6 Envelopes and contract | Bounds against peaks; the operation contract, registry, patch set, envelopes and memory account | below, D-086 |

BP-F2 (EXL3 kernel timing) and the census protocol did not run: D-085
replaces them with end-to-end parity once serving works and a loose
process-level memory comparison. The owner moved BP-F3 and BP-F4's
per-token half into that comparison (2026-09-27); under D-087 BP-F3 is in
M5 and BP-F4's per-token half in M3, on M3's models. Page-in
performance is in
[pagein-perf](../pagein-perf/README.md).

## P4: paging, on both representations

**What changed** (`scheduler.h`, `pagein.cc`, `catalog.h`,
`direct_reader.h`).
- Evicting live state whose source is its write-back place first copies
  the extent into a landing slot on the zone's stream and fences the copy.
- The storage lane then writes the slot to the place with direct I/O
  (`ReadSpec::kind`).
- Only once that write has moved the whole range is the slot freed and
  the backing unmapped and released.
- The catalog then keeps the content generation and marks the contents
  `preserved`, so a later load restores them through the zone as a page-in
  does.
- A write that fails or stops short abandons the eviction: the backing
  was never touched, so the state is resident again with its contents,
  and nothing is marked preserved. A copy or write whose completion is
  unproven quarantines the extent and its slot.
- A load of a write-back place is refused unless its contents are
  preserved at the current generation.

**The harnesses.** [`fp16_paged.cc`](../../../benchmarks/fp16_paged.cc) and
[`exl3_paged.cc`](../../../benchmarks/exl3_paged.cc) gained:
- `--partial` (BP-P2): one more evaluation, which at the restore point runs
  [`paging_cases.h`](../../../benchmarks/paging_cases.h)'s partial
  evictions. Each evicts exactly its extents, then checks that a phase's
  job submitted without materializing is refused before it runs, and pages
  only those extents back in.
- `--spill premapped|managed` (BP-P4): one more evaluation, which writes the
  cache back to an unnamed direct-I/O spill file and evicts it, then
  restores it. FP16 does this after its first prefill chunk and 8 tokens
  past the restore point. EXL3 does it after every prefix's prefill and its
  eighth step.
  - `premapped` keeps the backing mapped and poisons it with 0xff while
    the cache is nonresident, so the restore must bring back every byte.
  - `managed` releases the backing (D-033) and restores into fresh
    backing.
  - Either way the cache is copied to the host before and after and must
    match byte for byte.
- `--embeddings shared` (BP-P3, FP16): the token table is held once, in
  device VMM. Each chunk's rows are copied from it to pinned staging by a
  job leasing both, and widened on the host as the bridge's CPU lookup
  widens them.
- For EXL3, a setup check that the head is a resource of its own whose
  bytes do not overlap the embedding's (BP-P3).

**Results** (2026-09-27, `spark-b`: GB10, kernel 7.0.0-1019-nvidia, driver
580.178.04, the `spark-native` build, `llmp_fp16_paged` `4bc95949…`,
`llmp_exl3_paged` `87140d73…`, `CUDA_DISABLE_PTX_JIT=1`, lanes on their
own threads):

| Arm | Options | Evaluation 1 | Later evaluations against 1 | Launches refused over an incomplete closure | Cache bytes differing after restore |
| --- | --- | --- | --- | --- | --- |
| FP16-F `control` | `--restores 1 --partial --spill premapped` | bridge's `bb8ae5e7…` | 0, 0, 0, 0 | 5 of 5 | 0 |
| FP16-F `control` | `--spill managed --embeddings shared` | bridge's `bb8ae5e7…` | 0, 0 | — | 0 |
| FP16-U `control` | `--partial --spill managed` | bridge's `3560d337…` | 0, 0, 0 | 5 of 5 | 0 |
| FP16-U `heldout` | `--partial --spill premapped` | bridge's `69ff0821…` | 0, 0, 0 | 5 of 5 | 0 |
| FP16-F `heldout` | `--restores 1 --relocate --partial --spill managed --embeddings shared` | bridge's `bfb36f19…` | 0, 0, 0, 0 | 5 of 5 | 0 |
| EXL3-G 4.0 bpw | `--partial --spill premapped --cancel-in-flight` | rung 3's logits files | 0, 0, 0 | 6 of 6 | 0 |
| EXL3-O 4.0 bpw | `--partial --spill managed` | rung 3's | 0, 0, 0 | 6 of 6 | 0 |
| EXL3-G 4.5 bpw | `--restores 1 --relocate --partial --spill managed --cancel-in-flight` | rung 3's | 0, 0, 0, 0 | 6 of 6 | 0 |
| EXL3-O 4.5 bpw | `--partial --spill premapped` | rung 3's | 0, 0, 0 | 6 of 6 | 0 |

"Rung 3's" means every `.npy` equals the P3 paged run's, which equals
rung 3. Every bound tensor or range of every run lay in cataloged,
resident device memory of its class (BP-A1's in-process check, no
violation).

- **The partial evictions** (extents evicted in each):

  | Case | FP16 | EXL3 (both rates) |
  | --- | ---: | ---: |
  | one layer (the middle one) | 15 | 4 |
  | side vectors and biases (resources of at most 64 KiB) | 97 | 97 |
  | the trellis only (that layer's) | — | 4 |
  | a shared small-tensor chunk | 1 | 1 |
  | padded tails (each group's last chunk) | 25 | 26 |
  | a tensor crossing a chunk boundary | 5 | 2 |

- **Write-back and restore times** (reported, not gated; one run each):
  - FP16's cache (3 extents for `control`, 6 for `heldout`) wrote back in
    3.9–24.5 ms and was restored in 0.7–2.0 ms.
  - EXL3's (24 extents, 48 MiB) wrote back in 11.5–54.3 ms and was
    restored in 3.5–7.8 ms.
  - The writes go to a newly created file and are slower than reads;
    nothing here was tuned (M3 and M6 schedule spill writes).

**GPU unit tests** (`gpu`, `spark-b`; not labeled `gpu-discrete` until
run on the discrete GPU):
- `unit.VmmWork/CudaWriteBack.*`: four extents of state written back to an
  `O_TMPFILE` through io_uring and the zone and restored exactly; the
  premapped pair was poisoned while nonresident, and the managed pair's
  backing was released.
- The fake-backend cases are in `unit.VmmWork/PageInTest.*`:
  - staging and fence order;
  - direct places;
  - failed and short writes (D-050's spill-failed row);
  - invalidated state;
  - unproven copy-outs;
  - slot order;
  - an evictor cancelled mid-write, and one cancelled while its
    write-back still waited for a slot (abandoned, nothing written);
  - rejected places, and a preserved place that no new source may
    rename.

  `unit.CatalogTest.AWriteBackEvictionPreservesTheContentGeneration`
  covers the catalog's side.

## P5: lifetime and failure on the real providers

- **BP-L1, BP-L3** (`llmp_exl3_paged --cancel-in-flight`, both 4.0 bpw
  EXL3-G and 4.5 bpw EXL3-G above):
  - The largest reconstruction phase, the 1,023-row prefill (GGML and EXL3
    work, each reconstruction slice followed by its GEMM), is submitted
    behind a gate that its job queues first: a stream wait on a host
    flag. Its request is then cancelled.
  - While the gate held, every extent the phase touches (494 at 4.0 bpw,
    504 at 4.5 bpw: weights, cache and the region with the reconstruction
    scratch) was still leased and not evictable, and the task had not
    retired.
  - Once the gate opened, the task retired cancelled and no lease remained.
  - The job's launches blocked in the driver until the gate opened
    (RE-029), so the cancellation met the phase part-submitted: some
    slices reconstructed, their GEMMs not yet queued.
- **BP-L2** (`unit.VmmWork/CudaPageIn.RepeatedCancellationsNeverCorruptAReassignedSlot`):
  over 8 rounds, a request is cancelled once 1 to 8 slots are busy. Its
  loads drain, and every extent is evicted and loaded again; every reload
  held exactly the file's bytes.
  - Whether io_uring cancelled a given read or it completed first is not
    observed. A read in flight on NVMe is not recalled; either way the slot
    waits for its completion.
- **BP-L4** (`unit.VmmWork/CudaPermutations.*`), through decorators the
  lanes call in place of the real providers:
  - io_uring submissions reported as of unknown start, and every
    completion handed over twice, change nothing. All 32 extents load
    intact, with no quarantine and no fault.
  - A fence whose queries are of unknown outcome quarantines its extent
    and slot. Both stay charged, the node faults and admission stops. The
    stop reports the fault.
  - Completion before acceptance is a board property, independent of the
    provider (`unit.BoardTest.CompletionBeforeAcceptanceIsKept`).
- **BP-L5:** P3's `unit.Exl3LinearTest.*`. A lock area shared by two
  live contexts is refused, and two contexts' cooperative grids at the
  co-resident limit on two streams both complete.
- **BP-L6:** no I/O buffers are registered (the storage measurement needed
  none). Registrations hold extents like leases
  (`unit.CatalogTest.RegistrationsHoldLikeLeases`). A registration left
  live after a cancelled phase keeps its extent from eviction until it is
  retired (`unit.VmmWork/PageInTest.ALiveRegistrationOutlivesACancelledPhase`).
- **BP-V1:** there is no C++ importer in M2. The loader's checks are
  judged by M0's prototype (`unit.ArtifactCorpusTest.*`). The corpus
  covers both families' mutations, the EXL3 ones included: rate
  (`k_bits`), codebook, `in_features`, shapes, dtypes, closures,
  truncation and hashes.
- **BP-V2** (`unit.VmmWork/CudaPageIn.BackingThatCannotBeMadeOrMappedUnwindsCleanly`):
  - 1 TiB of device backing (`cuMemCreate` refuses it) and a mapping outside
    its reservation (made, refused at the map, released) each fail their
    load with a known outcome. No backing, charge or fault remains.
  - cuBLAS handle refusals and workspace exhaustion return errors
    (`unit.GgmlCublasTest.AHandleMustFitItsContextAndWorkspace`,
    `…WhatThePathCannotRunIsRefusedBeforeLaunch`,
    `unit.GgmlKernelsTest.WhatDoesNotFitIsRefusedAndALaunchErrorIsAFault`).
- **BP-V3** (`unit.ProgramPlanTest.ATightBudgetAdmitsTheLargestReconstructionPhaseOrRefusesThePlan`),
  with the EXL3 fixture's own numbers:
  - At exactly `R_i` plus the 1,023-row prefill's envelope, the 1,024-row
    prefill is admitted, since a prompt may end in a 1,023-row chunk.
  - One byte less and the plan narrows to 145 rows.
  - Below the narrowest prefill the plan is refused. The refusal names
    the phase kind, width 32, the bytes required and a shortfall of one
    byte.
- **BP-A4:** stale tables are refused before launch (P3's
  `Exl3LinearTest`, BP-P5's stale-table case). Each FP16 chunk is planned
  and bound anew, so nothing captures a pointer across chunks. The
  negative control (`unit.GgmlStaleMemoryDeathTest.AKernelOverUnmappedBackingFaults`)
  runs GGML's RMSNorm over device VMM whose backing was unmapped: in a
  child process, the kernel faults and its fence reports the fault. It
  never reads what the memory held.

## P6: bounds against peaks

The harnesses record, for each phase kind, the guaranteed bound and the
observed peak (`paging.json`):
- FP16: the placement's activation extent against the highest activation
  byte a bound tensor reaches, and the plan's pool bound against the
  pool's peak in that chunk.
- EXL3: the plan's region against the highest region byte an operation
  reaches, and the GGML pool's bound against its peak.

| Phase kind | Activations or region: bound | Seen | Pool: bound | Seen |
| --- | ---: | ---: | ---: | ---: |
| FP16 single-token step | 611,328 | 611,328 | 0 | 0 |
| FP16 16-row prefill | 9,781,248 | 9,781,248 | 0 | 0 |
| FP16 17-row prefill | 10,392,576 | 10,392,576 | 5,196,288 | 5,196,288 |
| FP16 32-row prefill | 19,562,496 | 19,562,496 | 9,781,248 | 9,781,248 |
| FP16 512-row prefill | 312,999,936 | 312,999,936 | 156,499,968 | 156,499,968 |
| EXL3 step, Npad 256 | 307,456 | 305,664 | 14,848 | 14,784 |
| EXL3 step, Npad 1,024 or 1,280 | 307,456 | 305,664 | 48,128 | 48,048 |
| EXL3 32-row prefill | 9,838,592 | 9,781,248 | 354,816 | 354,816 |
| EXL3 144-row prefill | 44,273,664 | 44,015,616 | 1,596,672 | 1,596,672 |
| EXL3 145-row prefill | 103,301,376 | 103,041,536 | 1,607,936 | 1,607,760 |
| EXL3 1,023-row prefill | 373,247,744 | 371,414,528 | 11,343,104 | 11,343,024 |
| EXL3 1,024-row prefill | 371,720,192 | 371,720,192 | 11,356,160 | 11,356,160 |

- The same in every FP16 arm (fused and unfused, where the kind occurs) and
  in all four EXL3 arms (both rates, EXL3-G and EXL3-O).
- FP16's input copies and logits are exact by construction: the harness
  copies those bytes.
- The bounds are the pre-registered limits' items
  ([memory and workspace](../../backend-proof.md#memory-and-workspace-the-m2-gate-in-exl3-bringupmd)).
- EXL3's region slots are 256-byte aligned by lifetime, which is why a
  kind can stay below its region.

The memory account and `F` per profile are D-086's. The catalog is exact
for llmpalooza's own bytes. Peak device memory for the whole process, from
ordinary counters, is compared loosely (at most about 1.1×) against each
reference engine. It passes: native is 0.81–0.96× the FP16 bridge on
every arm and 0.82–0.87× ExLlamaV3 on both EXL3 fixtures
([backend-proof.md](../../backend-proof.md), "The memory check").

## BP-S3: FP16 and EXL3 in one process

**The harness.** The paged harnesses became model runners on one node:
- [`paged_node.h`](../../../src/engine/paged_node.h) (then in
  `tests/support/`; the engine's since D-096) holds what the
  models share: the providers, one catalog domain, the scheduler with its
  lanes, the landing zone, and one workspace (the activation region and
  the GGML pool) sized for the larger need of each part.
- Each model has its own stream, launch contexts, cache and staging:
  [`fp16_runner.h`](../../../benchmarks/fp16_runner.h) for FP16 (GGML's
  kernels and cuBLAS), and [`exl3_runner.h`](../../../benchmarks/exl3_runner.h)
  for EXL3 (ExLlamaV3's kernels and GGML's).
- `llmp_fp16_paged` and `llmp_exl3_paged` run one model each, as
  before. The EXL3 launch context is now made before the scheduler runs,
  and EXL3's VMM work moved to the VMM lane, as FP16's already ran.
- The refactored binaries were rerun with P4's options (`spark-b`, same
  build). Each gave rung 3's logits, all refusals, no cache byte
  differing, and for `--cancel-in-flight` the result above:
  - FP16-F `control`: `--restores 1 --relocate --partial --spill managed`;
  - FP16-U `heldout`: `--lanes inline --embeddings shared --spill premapped`;
  - EXL3-G 4.0 bpw: `--restores 1 --relocate --partial --spill premapped
    --cancel-in-flight`;
  - EXL3-O 4.5 bpw: `--lanes inline --spill managed`.
- [`alternate_paged.cc`](../../../benchmarks/alternate_paged.cc) registers
  both models and alternates them: FP16, EXL3, FP16 and so on, for three
  rounds. Each model runs its whole trajectory set per evaluation: 76
  tokens for FP16 `control` and 577 for `heldout`; for EXL3, every
  prefix with its 16 steps.
- **The budget** B is the node's fixed occupancy plus the larger model's
  weights plus half the smaller's. Both models' weights never fit at once.
- **Before each evaluation** the model's whole closure is acquired
  ([`AcquireProgram`](../../../src/scheduler/programs.h), then in
  `tests/support/paged_programs.h`):
  - the memory module plans the materialization against B
    (`PlanMaterialization`), the shared workspace protected;
  - the victims it chooses are evicted and their backing released;
  - what is missing pages in through the zone.
  Its jobs then run under leases, as in a standalone run. The models
  never run at once: each job is waited for until its fence completes, so
  the workspace needs no ordering between their streams.
- **The run checks** that:
  - every evaluation's logits equal that model's first, and rung 3's
    hashes;
  - every acquisition after the first evicts only the other model's
    weights, and at least one;
  - the catalog's occupancy never exceeds B;
  - the scratch class is charged exactly the shared workspace, once;
  - each model's coverage and bounds hold, as in its standalone
    `summary.json` and `paging.json`.

**Results** (2026-09-27, `spark-b`: GB10, kernel 7.0.0-1019-nvidia,
driver 580.178.04, the `spark-native` build, `llmp_alternate_paged`
`1e1467ec…`, `CUDA_DISABLE_PTX_JIT=1`, lanes on their own threads, three
rounds; an earlier build of the same code before formatting, `0bcffb4b…`,
gave the same results, and the timings and memory below are its):

| Pair | Every evaluation against rung 3 | Weight extents (FP16, EXL3) | Evicted from the other model, per acquisition after the first | Own or other extents evicted | B, peak occupancy | Shared workspace (separate) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16-F `control` + EXL3-G 4.0 bpw | FP16 `bb8ae5e7…` ×3; EXL3 all 10 `.npy` ×3 | 620, 292 | 146 | 0 | 2,442,810,496, the same | 385,875,968 (417,333,248) |
| FP16-U `heldout` + EXL3-O 4.5 bpw | FP16 `69ff0821…` ×3; EXL3 all 10 `.npy` ×3 | 620, 302 | 151 | 0 | 2,899,767,936, the same | 530,579,456 (857,735,168) |

- Every bound tensor or range lay in cataloged, resident device memory of
  its class, the model's own or the shared workspace: no violation in
  332,370 (`control`) or 258,510 (`heldout`) FP16 tensors, or in
  542,928 EXL3 ranges per pair.
- Each acquisition took 41–115 ms. Each evaluation took 0.53–0.63 s for
  FP16 `control`, 1.38–1.50 s for `heldout`, and 6.5–7.3 s for EXL3.
  Reported, not gated.
- **Peak memory, loosely (D-085).** This is the drop in `MemAvailable`
  during each run, from system counters (the
  [vmm-counters](../vmm-counters/README.md) finding: VMM backing appears
  in no process counter). Measured on an idle `spark`, same binaries,
  FP16-F `control` and EXL3-G 4.0 bpw, two runs each:

  | Run | Drop |
  | --- | ---: |
  | FP16 alone | 2,069–2,134 MiB |
  | EXL3 alone | 6,457–6,671 MiB |
  | Both, alternating | 7,224–7,858 MiB |

  - The pair costs less than the two alone.
  - About 1.4 GB of EXL3's drop is cataloged (its weights, region, pool,
    cache, zone and pinned staging). The other 5 GB or so is outside the
    catalog. Not investigated.
  - EXL3 alone here drops about 1.7 GB more than native EXL3's 4,813 MiB
    in the memory check (`llmp_exl3_exec`, rung 3, arm O, `cudaMalloc`;
    [backend-proof.md](../../backend-proof.md), "The memory check").
    The harness and arm differ; the gap is not explained.

**Tests.**
- `unit.AcquireTest.*` (fake backend): each acquisition evicts only the
  other model's weights, as many as the shortfall needs, never the shared
  workspace, even from a closure that omits it (unprotected, it would be
  the first victim). A closure that cannot fit is refused as over budget
  and evicts nothing.
- `unit.CudaPagedNodeTest.*` (`gpu`): two synthetic models alternate on
  the real providers, each on its own stream, through one zone and
  workspace. Each acquisition evicts only the other's weights, and the
  bytes read back are the file's. Teardown leaves no backing.

## BP-P1: coalesced reads

Built and correct, but off by default: the A/B below measured it slower
than one read per chunk, so under D-085 the reader keeps one request per
chunk and coalescing is an option.

**What changed** (`direct_reader.h`, `storage.h`, `uring_storage.h`,
`io_uring.h`). D-056's page-in contract, which the owner chose to build
in M2:
- **Coalescing** (`ReaderSettings::span_bytes`; off unless set, and the
  harnesses' `--coalesce on`). When the provider has room, the direct
  reader starts the queued reads that continue one another in a file as
  one vectored request (io_uring `READV`). Each chunk is one segment into
  its own landing slot (or its own host backing, in place).
- **Why vectored, not consecutive slots.** Slots are granted one per load
  and freed one per copy, in any order, so consecutive free slots are
  rare; segments need no change to the scheduler's slot or operation
  model. Every load keeps its own read operation, which completes only
  when its whole span has.
- **Bounds.** A span stops where the file range does not continue (a
  resident chunk, another shard), at a write (write-back never
  coalesces), at 64 MiB (the prototype's run size) and at 1,024 segments
  (`IOV_MAX`). The zone bounds it further: only reads that hold slots
  exist.
- **Nothing waits in order to coalesce.** A read starts as soon as the
  provider takes it; only reads already waiting for room join. Starts
  stay in file order (RE-026).
- **Short or failed spans.** A span's count fills its chunks in order.
  Those it filled are done; the first it left short continues, or ends
  as end of file at an unaligned point or a count of zero; the rest start
  again, ahead of later reads. A retryable error retries each read; any
  other error starts each read again on its own, so it lands on the read
  it belongs to.
- **Cancellation.** A span is cancelled only once every read in it is
  withdrawn. A withdrawn read still ends only when its span completes, so
  its slot is freed only after that.

**Results**, all with coalescing on (the harnesses' default when these
runs were made, before BP-S3's runners; now `--coalesce on`):
- **Rungs 4 and 5** (`spark-b`, 2026-09-27, `llmp_fp16_paged`
  `7b8d4f3d…`): `run_paged.sh threads` on all four FP16 arms gave the
  recorded hashes (`bb8ae5e7…`, `3560d337…`, `bfb36f19…`, `69ff0821…`),
  zero bit differences over four evaluations (two with every weight
  evicted and restored, one relocated) and no coverage violation. In the
  two arms whose counts were read (`control` fused, `heldout` unfused),
  the 490 device chunks took 343–452 requests on the first load and
  328–352 on each restore; the 130 host-table chunks, read in place as
  each one's backing is made, did not coalesce.
- **EXL3** (same host and build): `llmp_exl3_paged --restores 2
  --relocate` on 4.0 bpw G and 4.5 bpw O wrote every logits file equal to
  the P4 run's (rung 3's), with zero bit differences and no violation.
- **A/B** (D-085's quick check): `llmp_fp16_paged --load-only 6`, the
  490 device chunks (988 MB) through the zone at depth 4 with 8 slots,
  coalescing on (64 MiB spans) and off (one request per chunk), on
  `spark`, 2026-09-28 01:52–01:54 UTC. Three interleaved rounds, one
  process per variant per round, each starting after 10 s with no other
  GPU process, no llmpalooza or benchmark process and a 1-minute load under
  1.5 (0.06–0.31 at the starts). The artifact's shard was written on
  2026-09-23, so at rest (RE-027). GB/s over loads 2–6 (n = 15; the
  first includes warm-up, 9.6–11.7):

| Backing | Coalescing on | Off | On against off | Requests per load, on |
| --- | ---: | ---: | ---: | ---: |
| Made per load (the runtime's path) | 12.91 (12.28–13.13) | 13.17 (13.02–13.23) | −2.0% | 331–353 |
| Mapped once | 12.34 (11.80–13.26) | 13.32 (13.21–13.35) | −7.4% | 331–354 |

Neither is more than ~10% slower, which is the question D-085 asks; but
coalescing is not faster here either, and with backing mapped once, where
reads back up most, it is slower and more variable. Plausibly a span
hands its chunks to the copy only once all of them have landed, so the
zone turns over in bursts; not investigated.

**Selection (D-085).** A quick A/B selects the implementation, so the
reader's default is the faster one here: one request per chunk
(`ReaderSettings::span_bytes` defaults to `kNoCoalescing`). Coalescing
stays built and tested as an option: the fake-provider tests
(`unit.CoalesceTest.*`, `unit.VmmWork/PageInTest.AdjacentLoadsCoalesce…`,
`…CancellingOneLoadOfASpanLeavesItsNeighbourWhole`), the kernel's
vectored requests (`unit.UringTest.VectoredRequestsFillEachSegmentInOrder`)
and `unit.VmmWork/CudaCoalescing.*` on the GB10 run with it on; every
other test, and the harnesses unless given `--coalesce on`, run the
default. When on, the span is 64 MiB, a tuning value
(docs/artifact-format.md).

## The case matrix

| Case | Status | Where |
| --- | --- | --- |
| BP-A1 | In-process check passes on every run: all bound tensors and ranges in cataloged extents of their class. The census reconciliation is replaced by D-085's peak check, which passes (0.81–0.96× the bridge) | P2, P3, above |
| BP-A2 | Pool draws within the plan's bound (table above); cuBLAS workspace declared; no allocation after the first launch in P3's traces. Library-internal growth is in `F`, judged loosely | above, P3 |
| BP-A3 | FP16: no F32 conversion or extra buffer types; the table's host copy declared, and shown unnecessary (`--embeddings shared` gives the same logits). EXL3: weights are the artifact's bytes plus the declared F32 norms and tables; reconstruction is phase scratch in the region | P2, P3, above |
| BP-A4 | Passes (above) | above |
| BP-A5 | Counters reconciled with the catalog on the Spark ([vmm-counters](../vmm-counters/README.md)); the per-run census is replaced by D-085's peak check, which passes | P2, [M2 record](../../m2-record.md) |
| BP-N1–N6 | Pass | P0, P1, P3 |
| BP-N7 | Reported diagnostic: the CPU path is not bit-exact across CPU variants | P0 |
| BP-P1 | Passes with coalescing on: every weight evicted and restored through coalesced, vectored chunk reads, bit-identical on all four FP16 arms and both EXL3 fixtures. Slower than one read per chunk in a quick A/B, so off by default (D-085) | above |
| BP-P2 | Passes on both representations | above |
| BP-P3 | Passes: FP16 duplicated and shared give the same logits; the EXL3 head is its own resource | above |
| BP-P4 | Passes on both representations, poisoned and managed | above |
| BP-P5 | Passes: relocation rebuilds descriptors (FP16) and rewrites EXL3's tables; stale tables refused | P2, P3 |
| BP-P6 | Reported | P2, P3, above |
| BP-L1–L6 | Pass (above) | above |
| BP-V1–V3 | Pass (above) | above |
| BP-F1 | Passed on device VMM (rule v2) | P1 |
| BP-F2 | Not run (D-085) | — |
| BP-F3 | Moved to M5 by the owner (2026-09-27): part of M5's end-to-end comparison of each engine with its reference (D-085) | — |
| BP-F4 | Per-launch host cost reported; per-token against upstream's decode moved to M3, on M3's models, by the owner | [launch-overhead](../launch-overhead/README.md) |
| BP-S1, S2, S4 | Pass | P1, P2 |
| BP-S3 | Passes: FP16 and EXL3 alternate in one process, each evicting the other's weights, every evaluation equal to rung 3 | above |

## D-050's adversarial matrix (M2 rows)

Each M2 row, with the tests that carry it
([matrix](../../reservation-policy.md#worked-cases-and-implementation-gates)):
- **Pass on the fake backend:**
  - two phases awaiting weights; same-class models; interactive over
    background with D-069's pauses; a member's change during a pause; a
    feasible queued request against an impossible phase (`unit.Admission.*`,
    `unit.CommitmentLedger.*`, `unit.ShapeScenarioTest.*`);
  - the substitute's change and the second pause
    (`unit.Admission.ASecondPauseIsNotTakenWhileOneIsOpen`, new);
  - a grant over a full useful cache, and lease release without pressure
    (`unit.ShapeScenarioTest.AGrantOverAFullUsefulCacheEvictsNothing`,
    new);
  - spill full or failed (`…AFailedSpillLeavesTheStateResidentAndTheShortfallExplicit`,
    new);
  - a cancelled phase with a registration still live
    (`…ALiveRegistrationOutlivesACancelledPhase`, new);
  - an admitted request suspended indefinitely keeps its allowance
    (`unit.Admission.APausedRequestKeepsItsAllowanceHoweverLongItWaits`,
    new; expiry itself is M6's retention);
  - unknown provider completion and budget reduction
    (`unit.SchedulerTest.*`, `unit.CommitmentLedger.*`).
- **Moved to the milestones that build the feature** (owner, 2026-09-27;
  the matrix marks each part *moved*, and
  [M2's exit criterion](../../m2-record.md#exit-criteria-and-the-gate) covers only
  the rest):
  - state growth through branch or copy-on-write, and a fork's divergent
    growth: M6 (retention's branches and sharing);
  - many suballocation holes: M5 (state blocks, which bring suballocation
    within extents);
  - envelope upgrade racing cached-state promotion: M6 (retention);
  - repeated speculation with prefetch: M9 (prefetch is a deferred
    optimization);
  - a slow or disconnected client's termination: M5 (the front door;
    output limits exist in `OutputBuffer`);
  - a closure or rounded allocation over its bound detected at runtime
    before submission: M7 (the routing boundary; planning refuses it now);
  - faulting on injected capacity loss: M5 (serving the fixtures under a
    real memory budget; owner, 2026-09-27);
  - every queue full at once during cancellation: M5 (the front door's
    output queues; each queue is covered alone now).

## Reproduction

On `spark-b`, with the `spark-native` build, P2's `control-tokens.txt`
and held-out IDs, P3's plans (`p3b-20260927/plans`), and the installed
artifacts (`artifact-layout-20260922/installed`):
- `llmp_fp16_paged --artifact ART --trajectory control|heldout --tokens
  FILE --fusion on|off --out DIR` with the options in the table;
- `llmp_exl3_paged --artifact ART --fixture 4.0bpw|4.5bpw --arm G|O
  --plan PLAN --ids IDS --out DIR` with the options in the table;
- BP-P1's A/B: `llmp_fp16_paged ... --load-only 6 --weights device
  --backing managed|premapped --coalesce on|off`, each load's GB/s being
  `read_bytes / seconds` in `loads.json`, which also counts its requests
  (BP-P1's rung runs: add `--coalesce on` to either harness);
- then compare `summary.json` (logits, bit differences and coverage) and
  `paging.json` (partial evictions, refusals, cache comparison, cancel
  result, bounds against peaks);
- for BP-S3, `llmp_alternate_paged --fp16-artifact ART --trajectory T
  --tokens FILE --fusion F --exl3-artifact ART --fixture X --arm A --plan
  PLAN --ids IDS --out DIR --rounds 3 --fp16-expect SHA256 --exl3-expect
  RUNG3DIR`, where RUNG3DIR is P3's `rung3-*` output. It exits 1 on any
  failed check; `alternation.json` has each evaluation's hashes and
  evictions and each sample's occupancy.

Each run took under 20 s; each alternation, about 40 s. Raw outputs stay
on `spark-b` under `~/.local/share/llmp/m2close-final` and
`~/.local/share/llmp/bps3-20260927`. The memory runs' summaries are on
`spark` under `~/.local/share/llmp/bps3-20260927`.
