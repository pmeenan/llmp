<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M2 record — Resource core and backend proof

*Milestones renumbered (D-087, 2026-09-27): numbers here are those of
their date; the 2026-09-23 ladder's M3–M8 are now M5–M10, and M4a is M6a.
plan.md says which hand-offs moved to the new M3 and M4.*

M2 ran from 2026-09-24, on M1's exit, to 2026-09-27 and exited on the
owner's word on 2026-09-27, after its gate (below). This is its
item-by-item record, moved out of [plan.md](plan.md) at exit so the plan
stays lean: what each item built, the evidence and hosts behind it, and
what it handed to later milestones.

**This is frozen history.** The living documents supersede it where they
differ: [plan.md](plan.md) owns what M2 handed on,
[decisions.md](decisions.md) the settled choices (D-076 to D-086 are M2's),
[backend-proof.md](backend-proof.md) and the
[aggregate report](experiments/backend-proof/README.md) each case's status,
and [architecture.md](architecture.md) how the runtime works now. Later
corrections go in those documents, not here. Commits are cited by short
hash; every experiment report cited carries its hosts and provenance.

Goal: the node-wide catalog, ledgers, reservation and lease state machine
and D-048's task lanes, deterministic on a fake backend and real on a
Spark; alongside them, a backend proof that runs real GGML FP16 and EXL3
kernels on llmpalooza-owned memory and settles the internal operation contract.
M2 added no tokenizer, C++ importer, HTTP or new model shapes.

**Entry:** M1 exited on 2026-09-24.

## Scope

- [x] **Catalog and ledgers** (D-006, D-007). `047bbdf`. Typed,
      generation-checked identities and checked byte counts (`base/`);
      extents with the architecture's load and eviction state machine,
      resources, closures that charge shared extents once, all-or-none
      leases, registrations, unknown allocations pinned and never evictable,
      per-domain occupancy (`catalog/`); the commitment ledger, the
      deterministic LRU victim baseline and materialization planning
      (`memory/`). Tests: `unit.CatalogTest.*`, `unit.CommitmentLedger.*`,
      `unit.VictimTest.*`, `unit.MaterializeTest.*`. *Handed on:* retained
      entries and `M_state` in the victim order (M4, with the retention
      cache); suballocation within extents (M3, with state blocks).
- [x] **Admission and leases** (D-050, D-069). `047bbdf`. One active
      request at a time; D-069's three pause policies with their guards,
      the promised resumption held in the ledger, deadline checks, aging,
      and explicit refusal of queued work that can no longer fit
      (`unit.Admission.*`, including a randomized progress check under all
      three policies). *Handed on:* a pause with several paused cohort peers
      (M4's concurrency).
- [x] **Task lanes** (D-048). `047bbdf` (bounded queue with cleanup
      reserve, wake flag, completion board, lanes, task table),
      `ca8b26e` (the provider lanes and the scheduler's turn loop). The
      threaded cases run the owner with an hour-long tick, so a lost wakeup
      or cancellation hangs. [Measurements](experiments/task-lanes/README.md):
      a sleeping owner wakes in hundreds of microseconds on the Spark, so
      the scheduler polls while a critical completion is imminent, and a
      lane queue takes at most four workers (RE-017). Verified in
      `check:spark` (cross, ASan, TSan) on `spark-b` when it landed, and
      again at the gate under TSan (below); the fake cases also passed on
      the workstation and under qemu-user in `check`, where
      `unit.StorageLaneTest.*` (io_uring) is skipped. *Not yet wired:* victim
      selection on a miss, spill as retention (M4), admission driving task
      starts.
- [x] **Providers** (D-026). `047bbdf`. Device-memory, device-execution
      and storage interfaces with poison-filling fakes; the CUDA VMM
      provider at D-033's 2 MiB extents; direct reads over a raw io_uring
      ring with alignment, short-transfer, retry, cancellation and
      duplicate-load handling. [Storage queue](experiments/storage-queue/README.md):
      about 14.9 GB/s with 8 MiB in flight as 2 MiB requests. The
      workstation's Btrfs quietly buffers direct I/O (RE-018).
- [x] **OS counters for VMM backing.** `047bbdf`,
      [vmm-counters](experiments/vmm-counters/README.md): on `spark`
      (driver 580.178.04) device and host backing leave and return to
      `MemAvailable`, neither is charged to the cgroup (RE-019), and the
      memory breakdown reconciles.
- [x] **Backend integration proof**, stages P0–P6
      ([scope](backend-proof.md), [aggregate report](experiments/backend-proof/README.md)).
  - *P0* (`047bbdf`, [report](experiments/backend-proof-p0/README.md)):
    both toolchain bridges reproduce their references bit for bit; the
    owner approved the numerical profiles, the FP16 gate and the EXL3
    bounds on 2026-09-26, and delegated the rest under D-079 (`7a4b768`).
  - *P1* ([report](experiments/backend-proof-p1/README.md)): GGML enters as
    the locked, narrowed, patched llama.cpp archive (D-077, `db09a9b`); the
    cuBLAS handle and workspace are llmpalooza's, cuBLAS linked dynamically
    (D-076, `d95b000`); plan selection between implementations (D-053,
    `3a2d8f5`); every GGML kernel of the FP16 plan and upstream's fusion
    gates (`1982f4b`); ExLlamaV3's GEMM kernels enter the build (`d2601ad`;
    cleared under D-080). GGML's launchers hold no `throw`, `try` or
    `catch` at the pin (D-066). The test binary's SASS for the FP16
    plan's GGML kernels 3–28 and the copied `k_compute_batched_ptrs`
    (`37c97848…`) has the bridge's recorded hashes, and ExLlamaV3's 14
    recorded GEMM kernels have the reference's in the `native` and
    `cross` archives and the linked binary (cuobjdump 13.0.85). All 160
    traps in the four GEMM units' SASS are `cooperative_groups`' grid
    sync, which `kernels/exl3/launch_contract.h` guards by launching
    cooperatively within the co-resident limit on unshared lock slots;
    the four units cost 75–113 CPU-seconds per CUDA profile. Reported,
    not gated, on the Spark: fused and unfused RMSNorm-mul, the fused bias
    add and the fused K write each equal their unfused forms bit for bit;
    the fused gate/up product (F32 accumulation) differs from the unfused
    one in every element. BP-F1 was calibrated and pre-registered
    (`9576d72`), failed on
    host VMM (`14aaf08`: 41 of 53 cases, matrix products 1.10–4.9×
    slower), was diagnosed ([host-vmm-diagnosis](experiments/host-vmm-diagnosis/README.md),
    `554652b`, RE-022; direct landing in device memory impossible,
    `817a67e`, RE-025) and led to D-081 (`44f681a`): weights and state in
    device VMM behind a host-VMM landing zone. Re-registered for device VMM
    (`72c7c62`), it passed (`7dd9dc8`: every case 0.957–1.037× of
    `cudaMalloc`). BP-F4's per-launch host cost is
    [reported](experiments/launch-overhead/README.md).
  - *P2* ([report](experiments/backend-proof-p2/README.md)): the native v0
    artifact reader, judged by M0's prototype on a mutated corpus
    (`20608ae`: its views of three synthetic artifacts match the
    prototype's line for line, and its verdicts on 188 mutated and 800
    seeded random artifacts match, apart from two documented divergences,
    a string escape and kept `.kv.gguf` metadata it does not parse, each
    checked to its rule; on `spark-b` it opens the M0 fixtures, Qwen2.5
    FP16, both EXL3 rates and Gemma 4, as the prototype does); the FP16
    memory limits and plan comparator (`b550204`; its tests catch and
    locate 22 mutations and never call a fragment, or a run without its
    nsys trace, a complete match);
    native Qwen2.5-0.5B FP16 with a complete plan match and logits
    bit-identical to the bridge on all four arms, rung 3 (`202b14a`); the
    D-081 page-in path, and rungs 4 and 5 (paged, evicted, restored,
    relocated) bit-identical (`c3e441a`); page-in at disk speed, 14.93 GB/s
    through the zone against 14.70 in place (`a38a4d8`,
    [pagein-perf](experiments/pagein-perf/README.md), RE-026, RE-027). The
    pre-registered census failed on sub-R warm-up excesses; D-085 replaced
    it with a coarse peak check, which passes at 0.81–0.96× the bridge
    (`4029bae`).
  - *P3* ([report](experiments/backend-proof-p3/README.md)): every real
    EXL3 projection of both fixtures, 6,280 cases, byte-equal to upstream
    at a forced plan (BP-N5, `d8e1aaf`; challenge fixes `961cc09`); both
    fixtures end to end with the executed plan equal to the record, Tier E
    exact at operation level, Tier C's 750 statistics within bounds, and
    rungs 4 and 5 bit-identical (`582322c`); peak memory 0.82× (4.0 bpw)
    and 0.87× (4.5 bpw) of ExLlamaV3's (`072348c`).
  - *P4–P6* (`ccd3c31`, D-086): write-back of live state through the zone;
    partial evictions, state spill and shared-storage views on both
    representations; cancellation, event permutations and injected
    failures on the real providers (BP-L1–L6, BP-V1–V3); bounds against
    peaks, none exceeded; the operation contract. BP-S3, FP16 and EXL3
    alternating on one paged node, each evicting the other's weights
    (`8145eb3`). BP-P1's coalesced vectored reads restore bit-identically
    but ran 2–7% slower in a quick A/B, so they are an option, off by
    default (`a309f4c`). Clang-tidy findings cleared (`095ffbe`).
  - *Caveats:*
    - BP-F2 (EXL3 kernel timing) did not run, an owner-approved tradeoff
      (D-085): each engine is held to its reference's speed end to end
      once it serves.
    - BP-F3 (resident full-model timings, report-only) and BP-F4's
      per-token comparison are not reported in M2; the owner moved them
      to M3's end-to-end comparison (2026-09-27, D-085).
    - BP-V1 has no C++ importer in M2: the loader's checks are judged by
      the prototype's corpus.
    - BP-L2's io_uring cancellation race is not observed; the slot waits
      for either outcome.
    - The EXL3 memory check compares against ExLlamaV3 with PyTorch's
      context in its peak, so it is not like for like. In the paged
      harness, about 5 GB of EXL3's `MemAvailable` drop lies outside the
      catalog, not investigated.
    - cuBLAS keeps a 64.1 MiB default pool outside the 32 MiB workspace
      (RE-028), which the owner kept outside that figure (2026-09-27);
      launches can block a lane while the stream is busy (RE-029).
    - Spill is process-private (`O_TMPFILE`); no spill format exists yet.
- [x] **Retained-backing comparison.** The swap trace and pre-registered
      criteria (`fb20bd4`), and part (a)'s deterministic replay of D-033
      against 12 slab and hybrid designs (`25dfc1d`,
      [replay](experiments/retained-backing/replay.md)): none meets every
      criterion at every budget. *D-033 is retained* (2026-09-27). The
      timed sessions and part (b) did not run (D-085); the confirmation
      seed stays unread.
- [x] **Shape expressibility** (D-068). `177a20b`. State representations
      with capabilities, composed model contexts, phase kinds and
      `PlanProgram`; `unit.ShapeScenarioTest.*` drives draft rollback, a
      canvas paused across boundaries, block output and a two-artifact
      context through admission, the ledgers and the catalog with no
      special case in the core. Its negative control charges a paused
      canvas to the phase instead of `R_i`: that request is admitted beside
      a substitute whose first phase then cannot materialize (a circular
      wait).
- [x] **Explainable plans.** `177a20b`, `ccd3c31`. A `ProgramPlan`
      itemizes widths, closures, working sets and envelopes, and a
      rejection names the phase kind, width, required bytes and shortfall;
      the harnesses record each phase kind's bound against its peak
      ([P6](experiments/backend-proof/README.md#p6-bounds-against-peaks)).
- [x] **Discrete NVIDIA target** (D-082, `cf552be`). The `native` build
      also targets `sm_86`; `llmp doctor` judges a discrete GPU; the
      `gpu-discrete` tests pass on the workstation's RTX 3080 Ti. No model
      runs there in M2; fast-swap validation is M4's.
- [x] **The operation-contract decision.** D-086 (`ccd3c31`): registry-bound
      implementations run as device jobs over leased closures, itemized
      phase envelopes, a catalog-exact memory account, and `F` judged
      loosely at the process level.

## Process changes during M2

- **D-079** (`7a4b768`, 2026-09-26): the owner delegated the remaining
  backend-proof approvals to the agents with fixed defaults; each
  threshold is pre-registered in backend-proof.md, in its own commit,
  before the native result it judges.
- **D-083** (`e647f02`): test and development builds check libstdc++'s
  preconditions; the package's build does not.
- **D-084** (`ad2cae8`): one check set per slice, `mise run test --
  spark-native --locked` on `spark-b` (about a minute); the workstation
  tiers only at gates and for host-only needs.
- **D-085** (`8a2374b`): anything over 10 minutes runs only when its result
  is needed now, and never by default in a gate. It dropped BP-F2, the
  census protocol and the retained-backing timed sessions, and made
  performance and memory coarse, end-to-end comparisons (within about 10%
  of each engine's reference).
- D-080 set the license policy the kernels entered under: optional
  copyleft modules ship by default with a build-time opt-out, and GEMV is
  core-eligible.

## Exit criteria and the gate

The gate ran on 2026-09-27 on `095ffbe` plus this record's documentation
changes, under D-085: the Spark set and short checks, no workstation tier
in full.

| Check | Host | Result | Wall time |
| --- | --- | --- | ---: |
| `mise run prepare && mise run test -- spark-native --locked` | `spark-b` (GB10, driver 580.178.04, kernel 7.0.0-1019-nvidia) | 611 of 611 passed, 59 on the GPU | 72 s |
| D-048's lane, queue, wake, board, task-tree, scheduler, storage-lane and page-in tests, `spark-native` configured with `LLMP_SANITIZE=thread` | `spark-b` | 174 of 174 passed (165, then 9 GPU page-in cases), no ThreadSanitizer report | 11 s build, 15 s tests |
| `mise run test -- cpu --locked`, no CUDA toolkit, under `hostlock` | workstation | 519 passed; `unit.ArtifactFixtureTest.RealArtifactsMatchThePrototype` skipped (the fixtures live on the Sparks) | 59 s |
| clang-tidy on the three x86-only units (`confine.cc`, `toolchain_contract_test.cc`, `cpu_only_device_probe.cc`), `cpu` database | workstation | no findings | 10 s |
| BP-S3's two alternations on the final build (`llmp_alternate_paged` `6dba3a5e…`) | `spark-b` | both exit 0: FP16-F `control` with EXL3-G 4.0 bpw, FP16-U `heldout` with EXL3-O 4.5 bpw; every evaluation equal to rung 3 over three rounds; budgets reached exactly, never exceeded | 38 s, 40 s |

Not run at the gate (D-085): the x86-64 `native` build and tests, the cross
build under qemu-user, formatting, REUSE, header and tooling checks, the
sanitizer builds (`cpu-asan`, `cross-asan`, `cross-tsan`), the reference
container, the package with its install test, and the confined-job proof.
M2 changed the source lock, toolchain files and packaging; those run
before a package ships.

- **D-050's M2 rows pass on the fake backend with no vendor SDK.** The
  `cpu` run above covers them: `unit.Admission.*` (33),
  `unit.CommitmentLedger.*`, `unit.ShapeScenarioTest.*`,
  `unit.SchedulerTest.*` and the fake `unit.VmmWork/PageInTest.*` (74),
  the new row tests among them
  ([mapping](experiments/backend-proof/README.md#d-050s-adversarial-matrix-m2-rows)).
  Rows needing real allocation pass as BP-L and BP-V cases. Parts moved
  by the owner on 2026-09-27: suballocation holes, stalled-client
  termination, every queue full at once and capacity-loss injection to
  M3; fork and copy-on-write and cached-state promotion to M4; the
  runtime closure-excess check to M5; repeated speculation to M7. *Met.*
- **The BP case matrix passes, and the three fixtures stay within their
  oracle bounds before and after restoration.** Every gated case passes
  ([case matrix](experiments/backend-proof/README.md#the-case-matrix));
  the gate's alternations re-ran two of FP16's four arms and both EXL3
  fixtures, paged, evicted and restored, on the final build. Fake-backend and CPU-only cases ran on the workstation; the rest
  ran in `spark-native` on `spark-b` in place of `check:spark` (D-084,
  D-085). *Met;* BP-F3 and BP-F4's per-token half, report-only cases,
  were moved by the owner on 2026-09-27 to M3's end-to-end comparison of
  each engine with its reference (D-085).
- **BP-F2 passes, or the owner approved a tradeoff.** The owner did, on
  2026-09-27 (D-085). *Met by the approved tradeoff.*
- **D-048's lanes pass their concurrency, lost-wakeup and task-tree
  unwind tests, with ARM memory-ordering stress.** Under ThreadSanitizer
  on `spark-b` at the gate, including `TheOwnerLosesNoWakeupAmongManyPublishers`,
  `ReusedTagsLoseNoCancellationAmongManyThreads`,
  `ThreadedLanesKeepEveryHandoffOrdered`,
  `UringTest.NoWakeIsLostAmongManyProducers` and the threaded page-in
  stress, and `unit.CudaLanes.*` on the GPU. The build was `spark-native`
  with TSan, not `check:spark`'s cross-built `cross-tsan`. *Met.*
- **D-033 retained or amended, and the contract expresses D-068's
  shapes.** D-033 retained (`25dfc1d`); D-086 and
  `unit.ShapeScenarioTest.*`. *Met.*

**Heavy path.** Every blast-radius slice of this cycle was challenged
before it was committed:
- task lanes (`ca8b26e`), three rounds;
- the artifact reader (`20608ae`), three rounds;
- the P2 page-in path (`c3e441a`), one capped round;
- P3 (`961cc09`), `a38a4d8`, `ccd3c31` and `a309f4c`.

The rounds are recorded in the session's handoff notes rather than in Git.
`047bbdf` predates this cycle's records.

**Outside review.** The owner's outside review of the M2 commits found
three defects, all fixed at the gate, each with a regression test that
failed before its fix:
- a job whose first copy or launch had an unknown outcome reported it
  not started, releasing its lease before the fence (the runners, the
  paged-node test and the page-in benchmark). Jobs now report through
  `scheduler::AfterRefusal`, and a vendor-API error counts as unknown
  (`unit.VmmWork/PageInTest.AnUnknownFirstCopyHoldsItsLeaseUntilItsFence/*`);
- a GGML graph ending in a view let the viewed tensor's bytes be reused
  before the end (`unit.Qwen2GraphTest.AFinalViewKeepsItsSourceAlive`);
- the EXL3 binding took `heads % kv_heads` before refusing a zero count
  (`unit.Qwen2Exl3Test.RefusesAProfileWithAZeroCount`; the FP16 binding's
  guard is pinned in `unit.Qwen2Test.RefusesWhatTheProfileDoesNotDescribe`).

Rechecked after the fixes on `spark-b`: the Spark set, 615 of 615 (59 on
the GPU), and the FP16-F `control` with EXL3-G 4.0 bpw alternation, exit
0 in 37 s with every evaluation equal to rung 3.

## Handed on

- To M3: end-to-end parity of each engine with its reference once serving
  works (D-085), with BP-F3's timings and BP-F4's per-token half (owner,
  2026-09-27); the moved D-050 parts above; suballocation with state
  blocks; the importer, which replaces BP-V1's prototype corpus; and M1's
  two hand-offs (D-074, moved by the owner on 2026-09-27), a system-call
  filter admitting io_uring for `llmp.service` and a single reaper for
  jobs started from other threads, which M2 did not need: it neither runs
  the storage lane in the service nor starts jobs.
- To M4: victim selection on a miss and the D-033 handoff of a victim's
  backing, spill as retention with D-055's spill
  format, the cohort pause with several paused peers, the moved D-050
  parts, and fast-swap validation on the discrete GPU.
- To the model-swap work: `ReconGemm` builds and checks its cuBLASLt
  descriptors and pinned algorithm on every `Run`; preparing them once
  per bound GEMM would take that host cost off each launch (outside
  review, optional).
- Open, with no milestone named yet: `spark-native` sanitizer presets
  (D-084's intended follow-up); the unexplained EXL3 memory outside the
  catalog in the paged harness.
