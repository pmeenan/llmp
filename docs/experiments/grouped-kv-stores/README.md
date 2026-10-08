<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared physical KV store grouping (2026-10-08)

T86 now has one checked implementation for independent GGML F32→F16
`SET_ROWS` stores. Gemma3 selects it by default: the bounded C2 screen lowers
paid prefill/decode time 0.454% with exact choices, complete heads and initialized
state. Gemma2, Gemma26 and native/GGUF Qwen remain off after neutral whole-paid
screens. Dense31's earlier neutral result is retained without repetition.
These short own-engine results do not establish reference parity or a broad
context/format gain.

## Contract and focused controls

A group contains 2–16 adjacent independent stores, with independent widths,
rows, strides, planes and broadcast indices. Long runs split into bounded
groups. The checked element product fits uint32 before inherited element
arithmetic runs. Every destination span is disjoint from every source/index
span and every other destination; an unknown intersection falls back to the
primitive. Dynamic indices retain the ordinary model-authenticated contract.
Launchless views may intervene; actual read/producer dependencies, kept nodes
and unsupported operands stop grouping. Producers and their arithmetic stay
unchanged, including the primitive's F32→F16 conversion and rounding.

Every original node remains in plan coverage/liveness. Lane assignment splits
groups into maximal contiguous equal lane/region subgroups; tagged/untagged
or different-lane boundaries preserve primitive singletons. Placement and
host-plan funding run after this split. Qwen composition authenticates exact
expanded primitive-store order/count while retaining every other fusion
identity; its off path keeps the original authentication. Stack kernel
parameters are copied into captures, with no scratch or asynchronous host
descriptor lifetime. A 16-store control launches once using 18 registers,
zero local memory and zero static shared memory on the actual GB10.

Nine focused positive cases cover checked product/span/refusal bounds,
long runs, views/read barriers/keeps/aliases, lane boundaries, ragged actual
Gemma2/3/26 roots and placement, native/GGUF Qwen composition, and an actual
DeepSeek raw/compressed graph. The latter has zero groups: intervening
producer/read dependencies must remain. DeepSeek is a compatible open
extension requiring a safe ordering opportunity, rather than a layout-based
exclusion. Qwen MTP/drafter consumers remain unqualified and unselected.
Image has no matching GGML F32→F16 `SET_ROWS` consumer.

The GPU operand case compares complete poisoned destination buffers with the
original primitive for 16 heterogeneous descriptors, including odd width,
nonpacked strides, planes and broadcast indices. Changed values/indices replay
through one captured node. Rounding ties, infinities, NaNs, subnormals and
untouched parent padding agree exactly. The first compile attempt ran no
tests: two fixture declarations required `Implementations()` and an explicit
empty `dense_mmvq_shape`. Only those two failed targets were rebuilt before
the original nine unrun controls. That failed aggregate remains unpooled.

## Five matched recipient screens

All profiles use fixed off/on/on/off, `n=2` per arm, the same executable and
equal total funding within a profile. Only immutable store grouping varies.
All 20 children and both supervisors retire with exit 0, empty GPU processes,
stable boots and no kernel errors. All 32 fetched complete head files are
finite and byte-exact within each profile, together with choices/histories,
initialized state and range/layout identities. Product selections and logical
capture/replay work are unchanged. No cross-format equality is inferred.

| Profile | Prefill change | Decode change | Combined paid change | Off paid bookend movement | Selection |
| --- | ---: | ---: | ---: | ---: | --- |
| Gemma2 | +0.062% | −0.207% | −0.024% | −0.012% | Off; whole paid neutral |
| Gemma3 | −0.517% | −0.313% | −0.454% | +0.036% | On; both endpoint ranges nonoverlap |
| Gemma26 | +0.256% | +0.133% | +0.228% | −0.342% | Off; ranges overlap |
| Qwen native | +0.033% | +0.061% | +0.045% | −0.344% | Off; ranges overlap |
| Qwen GGUF | +0.010% | +0.176% | +0.053% | +0.484% | Off; ranges overlap |

Gemma3 prefill off/on means are 1.103230/1.097530 s, with ranges
[1.102960, 1.103500]/[1.096530, 1.098530]. Decode means are
0.484142/0.482627 s, ranges [0.484128, 0.484155]/[0.481992, 0.483262].
Combined paid means are 1.587372/1.580157 s, ranges
[1.587088, 1.587655]/[1.578522, 1.581792]. Candidate prefill/decode movement
is +0.182%/+0.263%; the modest gain is specific to this recipe. Gemma2's
small decode-only change does not justify whole-path selection. All individual
samples, ranges and drift are reproducible with [compare.py](compare.py) from
[results.json](results.json).

Gemma2 and Gemma3 use ordinary 128-row C2 first-cycle prefill, respectively
context 8192/prefixes 4352,4864/local ring 4352 and context 4096/prefixes
3072,3584/local ring 1280. They execute 33/23 joined groups plus six scalar
groups and 32 greedy decode steps per owner (64 generated choices).
Device masks, actual-root attention, two-future capture, shared Q8 and fresh
state preparation are common. A three-row seed is untimed; Clear/DropPlans
then makes first-traversal planning, growth, capture and execution paid.
The Gemma2 late ring wrap is scalar; this factor does not qualify a joined
wrapped execution. Loading and Setup are excluded.

Gemma26 uses the ordinary serving policy at context 8192, 7680 prompt rows,
1024-row chunks with a final 512-row tail, normmul on, state-only prefill,
lookahead capacity 1, capture-ahead/features off, and preparation on. A fixed
six-row warm remnant is authenticated; growth beyond it is paid. Three anchors
are outside the endpoint; 32 forced steps end at position 7715. Complete
prefill/final 262144-float heads and 581959680 initialized-state bytes agree.

Qwen uses target-only context 2048/chunk 512, distinct 1536/1280 prefixes and
GPU greedy publication in both arms. One fixed warm traversal and loading
are excluded. The retained-plan traversal pays six scalar prefill calls and
32 joined greedy decode calls (38 units, 64 device tokens). Histories include
the initial prefill choice plus 32 decode choices; each owner's complete
182525952-byte state is saved before off-paid full-row continuation heads.
Native MXFP8 and GGUF VecQ selections remain unchanged. Spark B runs the exact
A-built static executable with authenticated existing private SDK cuBLAS
aliases; all payload/input transfer finishes before either timing batch.

These counters describe bound-plan selections, not executed kernel counts:
Gemma2 packages 2444 stores into 728 steps; Gemma3 2516 into 782; Gemma26
each final prefill/decode plan has 60 stores in 30 steps; each Qwen final
wave has 48 stores in 24 steps. Primitive store selections become zero in
these candidates. Actual total budget is 32 GiB for each Gemma profile and
108 GiB for each Qwen format. Complete derived/fixed/activation/scratch/host
receipts happen to match all four arms within each profile; results retain
every receipt. No useful residency or admission policy was relaxed.

## Replay and provenance

The durable results retain nine case names/XML hashes, 30 executed source
bindings, official SDK receipt, eleven executable/library identities, all
20 endpoint samples/selection counts/budget receipts and exact proof hashes.
The final source bridge changes only Gemma3's default from false to its
measured true branch (plus its comment); diagnostic probes still set off/on
explicitly. No new model execution is needed for that literal default carry.

Build the eleven targets listed under `prerequisite.executables` in results
in a normally prepared `spark-native` tree, using the pinned
`aarch64-c09daba6ac31edee` SDK and parallelism 4. Run exactly the nine named
cases, selecting their class/name with `--gtest_filter`, and require completed
positive XML without skips. Inspect the operand resource marker. No full
suite or prior model/reference qualification is repeated by this recipe.

For fresh model replay, use new output directories and run the following
argument vectors from that tree, fixed O1/A1/A2/O2 per profile. `GROUP` is
absent/off in O arms and `group-stores`/`stores-on` in A arms. Run all three
Gemma profiles as one installed Spark A job and both Qwen formats as one
installed Spark B job, `--gpu --timeout 600 --grace 30 --stop-on-fail`, with
145 s per child, and positively wait. Transfer any required executable/inputs
before timing, with no concurrent host work.

```text
llmp_gemma{2,3}_joint_prefill_probe ARTIFACT INPUT0 INPUT1 OUT
  first-cycle bounded-roots device-masks prefill-ahead owner-prefill
  flexible-owner-prefill chunk=128 lookahead-capacity=2 shared-q8
  budget-bytes=34359738368 prepare-state [GROUP]

llmp_gemma_prefill ARTIFACT INPUT OUT 26 serving 1024 normmul-on state-only
  lookahead-on phases-off state-chunked capture-ahead-off features-off
  prepare-on 8192 7680 stores-off|stores-on 34359738368

llmp_plain_token_probe ARTIFACT INPUT0 INPUT1 OUT on [gguf] [GROUP]
  budget=115964116992
```

Executable names above live in `build/spark-native/benchmarks`. Artifact
manifest/index and input size/SHA bindings and standing paths are in results.
Gemma2 inputs are in `references/gemma-prefill-copies/gemma2/input{0,1}/ids.i32`;
Gemma3 uses `references/prefill-lookahead-group/gemma3-long1/input{0,1}/ids.i32`;
Gemma26 uses `gemma-prefill/input1/ids.i32`, authenticating the whole 8227-ID
file while consuming 7715 positions. Qwen inputs derive as little-endian i32
from standing `references/qwen-device-masks/inputs/wave-prompts.json`, entries
`mask-short-1536` and `mask-short-1280`. Paths are under
`~/.local/share/llmp`. Preserve originals; no new corpus or artifact import.

Before/after each arm authenticate source, actual prerequisite, ELF, receipt,
first-resolved pinned cuBLAS/Lt, ID size/SHA and small artifact metadata.
Require exact complete heads/state/history/choices, equal producer/logical
work, positive candidate groups/zero off groups, total budget at least the
actual derived minimum, finite positive endpoints, stable boot and no new
GPU/OOM diagnostics. Stop first failure without retry. Large state payloads
may stay on the execution host; raw outputs are not replay dependencies.

Task-entry TensorFold refs remain main `f8fe17d24629aedabf90bbf78279dd776e6d62e7`
and python-0.6 `ed78d6fc204d89d90b045bf033d6551e7714f3a1`. The native GGML
primitive is the arithmetic oracle; no TensorFold performance claim follows.

Final local light checks pass: REUSE and embedded headers cover 1962 files,
portability boundaries 426 files, changed C++/CUDA formatting 28 files, and
all 15 endpoint aggregates replay. All eight plan unchecked sets match the
96-technique matrix exactly: 129 open family/technique combinations remain
after this batch. The owner requested finishing only this T86 batch and the
concurrent T54 batch, reviewing/committing both, then stopping. Remaining
transfers, shipment tiers and refactoring belong to the later handoff; no
additional batch or cleanup is started here.
