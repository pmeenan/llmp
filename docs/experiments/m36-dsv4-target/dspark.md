<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Composed DeepSeek target and DSpark execution screen

2026-10-12, Spark B, main candidate based on `b8d156d0` plus the exact image
integration later committed as `346e1c4f`. The existing shared factory composes
its prepared target and companion using block drafting, real injection state
and the ordinary complete expert closure. Production serving remains unopted.

Target artifact:
`8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`.
DSpark artifact:
`dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5`.
The closed envelope is one owner, context 3072, prompt chunk 512 and draft
depth three, with the exact two-stage numerical domain. Short draft blocks and
verify tails use their actual rows; no autoregressive catch-up or replacement
orchestrator is added. Target hash IDs are checked and all 256 experts remain
represented. Target steps retain three actual companion injection writes.

Construction inventory totals 3,461,560,256 bytes within the existing prepaid
4 GiB allowance. Source weight extents total 110,930,952,192 bytes; combined
state layout is 486,539,264 bytes. Actual setup accounts:

| Component | Bytes |
| --- | ---: |
| Fixed | 4,356,694,720 |
| Activation | 270,532,608 |
| Pool | 29,360,128 |
| Host inputs | 9,365,504 |
| Plan floor | 31,199,360 |

The complete paid setup footprint is 117,595,233,088 bytes, including the
weights, four conservative state/snapshot planes of 490,733,568 bytes each,
and 4,194,304 bytes of workspace page-rounding margin. It leaves 516,367,552
bytes below the unchanged 110 GiB cap. The existing 6 GiB physical margin
passes as well. The ordinary dynamic budget is 122,793,353,920 bytes; it is not
permission to bypass the node cap. The earlier target-derived fit estimate was
a lower bound; this run establishes composed fit for the stated envelope.

## Observed controls

`dspark-shared-bridge-main01` passes the full locked native build and all 308
selected host controls across 18 targets, with complete raw inventories and no
skips. The two factory envelope controls refuse unsupported/unpaid cases before
artifact reading. Both DS model registrations are listed; only the composed
registration executes in `dspark-shared-bridge-gpu02`:
`Dsv4EngineTargetGpu.TargetAndDsparkFiniteHeadsAndSnapshotContinuation`.

- Prefix 2560 is initialized in five prompt chunks. Two snapshot continuations
  each complete seven generation units and all 17 requested draft proposals;
  one draft is accepted in each. No acceptance threshold is imposed.
- The existing synchronous callbacks/counts require every block's complete
  proposals, so an unnoticed draft-refusal fallback cannot pass.
- Eighteen complete target head rows of width 129,280 are finite. Snapshot
  restore compares nine full heads (1,163,520 values), generated token history
  and every initialized combined-state byte at completed frontier 2568.
- Five draft programs and 15 injected target programs pass the actual lowered
  source/selection checks.

The case passes in 35.309 seconds (40-second supervised job), including setup
and storage. This is not a serving-throughput or matched migration measurement.
Source, binary, artifact-metadata and recipe guards pass before and after; all
memory limit/OOM counters remain zero, the complete child process group retires
and no GPU process remains. The cgroup process/cache peak is 2,787,831,808 bytes;
it is not total GPU residency.

Initialized-state SHA-256:
`99b233e0df9a0c5398f9a51a70f18d369817d8afaa64ea0bf2da90a0f500be5e`.
Selected-plan receipt SHA-256:
`9d70773c9098d62b4eafee44ed1ef521d50ebc55fd4216edb2e22044c83a6118`.
Native recipe SHA-256:
`3a75805e3dcc5f6decd54e5054d5968a5a4edbdc747c58823103d5eb103f44c5`.
Actual recipe manifest SHA-256:
`fbd1d5d54d4435420da1f2d4b0b9c73aae24ad77e12f538eb6f76821ab54910b`.
Raw logs/XML remain outside Git in session scratch. The first sealed invocation
was held at review because its complete two-case native inventory was compared
with a one-case guard; only that guard and the scratch path changed before the
actual invocation. No native rebuild or arithmetic change was needed.

Full draft-head finiteness remains unqualified: complete proposals prove actual
draft execution, while finite-value checks cover target heads. Broader quality
against a reference of the same quantized checkpoint, durable restart,
selected-only expert paging, production serving, throughput and matched family
speed/swap gates remain open. No bit-parity policy is adopted. The absent
original BF16 model limits original-agreement claims, not that quantized
reference baseline.

## Shared Server admission and the remaining C2 capacity wall

2026-10-13, Spark B, checked combined main06. The internal generic component
selector now consumes the ordinary Server/EngineLlm path for target/DSpark.
Permissions derive from the actual component/state contracts and configured
geometry, including block versus repeat-pending drafting. Already migrated
Gemma aliases retain their shared route. Unsupported legacy-only fusion
overrides refuse before source construction; public defaults do not change.

One Server-owned compiler allowance pays the maximum transient construction
across aliases. Actual use is charged and confined to its driver thread; returned
programs retain that payment until destruction or transfer to paid current-plan
or future ownership. Commit-buffer graph construction acquires the same loan.
Source/spec/chat/tokenizer ownership remains separately paid per alias, as do
live plans, graphs, state and futures. The compiler extent remains catalog-paid
through teardown; releasing a use does not pretend that its capacity disappeared.

The actual service uses the same prepared target/DSpark IDs above, context 8448,
prompt chunk 256, two configured slots, draft depth three and fixed speculative
waves. The automatic parser loan retires after 6,350,176,256 retained host bytes
are paid. Complete physical admission passes with 6,716,392,320 fixed bytes,
110,930,952,192 weight bytes, 270,532,608 activation bytes and 29,360,128 scratch
bytes. The diagnostic node cap is 118,338,093,056 bytes, below the fresh ordinary
dynamic budget of 118,574,285,696 bytes. HTTP request memory is 64 MiB; the
6 GiB live physical margin and zero-swap process guards remain enforced.

All three HTTP requests complete 128 outputs each, with prompt lengths
371/341/344 and no cached prompt tokens. The shared composition records 110
speculative units, 323 proposed tokens and 271 accepted drafts. Its prepared
execution identity is
`2e690ae4e57e258b6bf335f8b800ba1372516a09f6af57efe2f32d27e462307a`.
The service shuts down with TERM/exit 0; source guards, zero memory-limit/OOM
events and positive child/GPU retirement pass. This establishes bounded serial
speculative serving, not a throughput or family-quality result.

The full batch remains **FAIL**: the two simultaneously submitted requests
serialize, with zero joined waves/units. The first ends at monotonic
1876326.289819543; the second first output arrives at 1876329.574886926.
The runtime explicitly queues the second request for another 506 MiB of prompt
state beside one active request, then reports 7.988 seconds queued. The logged
conversation room beside the largest weights is only 0.64 GiB. This is the
existing `serve_api.cc` request-slot admission (`PromptStateToCome`/`RoomFor`),
using `EngineLlm::StateBytesThrough` and actual component used-range page demand;
it is not evidence of an incompatible joined kernel or read shape. Idle-state
spill and plan/graph reclaim occur, but they do not establish restored-state or
swap integrity. The next bounded investigation is the complete simultaneous
state-page/physical footprint, preserving the admission guard.

The failed overlap gate stops the batch before fixed-plain HTTP and the planned
DSpark/Gemma 8K/16 two-cycle swap. Those controls, full-head continuation,
process/GPU restart, DSpark reference quality, qualified family cohorts and
matched 1% speed remain open. Existing target-only 256/2,560-loss quantized
reference evidence is reused unchanged; it is not DSpark quality evidence or
an original-model agreement claim. Public migration and legacy deletion remain
blocked by the family gates.

Earlier immutable attempts preserve physical-fit/cap refusals, controller
assertion failures before native execution, and an HTTP rendering-budget refusal.
None is relabeled PASS. `dsv4-serving-actual06` retains the genuine C2 failure:
recipe `490c9701…`, result `2be5392c…`, detailed service result `4378a616…`.
The checked runtime is `2939b1f3…` (276,362,160 bytes), full source 2,551 paths;
`m36-serving-union-main06` result `08daa73b…` passes the full native build and
241 current host controls, with unchanged main02 proofs retained. The actual
required 263-name subset is authenticated through 329 raw cases across eight
targets. The executed binary includes separately reviewed image/native-publication
changes; this service/compiler source is staged independently with those hunks
excluded. Raw receipts remain in `/tmp/llmp-m36-coordination/` until milestone
close. The unmerged author worktree is
`/home/pmeenan/src/llmp/.claude/worktrees/m36-dsv4-serving`.
