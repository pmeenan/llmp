<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 GGUF router sharing — rejected screen

The four-request, captured GGUF wave screen is neutral: the private router
candidate improves the measured paid-wave rate by **0.245%** against the mean
of its control bookends; the captured bookend rate changes **−0.269%**.
All measured histories,
greedy IDs, full logits and initialized final state agree exactly. The
prototype is not adopted; production code, settings and benchmarks are
unchanged. No expanded quality, unit-suite or context ladder is warranted
by this result.

## Mechanism and actual weight types

The native NVFP4 target's router is BF16 `[2560, 512]`, not F32. On GB10,
its one-column product selects GGML MMVF, but its multirow products use
another implementation. Sharing those multirow BF16 products through a
forced vector kernel would change the baseline arithmetic and was excluded.

The UD-IQ3_XXS GGUF target `5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78`
has 48 F32 router matrices of `[2560, 512]`, each 5,242,880 bytes. GGML's
natural GB10 selector takes MMVF for one to three columns. The prototype
groups whole requests up to eight columns using the same MMVF launcher,
paid F32 concatenations and output views. It checks the actual graph router
descriptor, shared immutable weight identity, geometry, precision parameters
and original vector selection. Joined router barriers are independent of
existing product groups: the eight-column limit does not split their
sixteen-row expert or MXFP8 products. The inherited per-node selector is
preserved, and original four-column routers retain their original path.

This screen exercises four one-row requests, hence four-column joined
routers. It does not qualify every guarded shape. Added private planner
cases, including wider slot groups and three-row originals, were not run;
mixed one-to-three-row groups and wider slot barriers remain unqualified.
Neither the F32 mechanism nor its result applies to native NVFP4/MTP
multirow BF16 routing.

## Frozen comparison

The benchmark is the existing `llmp_qwen38_gguf_wave` with private timing,
hash and `screen` instrumentation. `screen` selects only short context and
width four, retaining all four eager/captured and unpaired/paired cells,
32 fixed-history steps per cell. Both binaries have the same instrumentation
and header; the control retains the original wave composer. Formatting alone
differs between the two harness source files.

Each arm runs one native process with context 4,096, chunk 512, four funded
slots, lanes on and attention read alignment 2,048. No drafter or selected
draft head is present. Prompt IDs are the existing native TSV fixture.
Scalar steps produce the fixed histories; the logit/state reference is the
unpaired eager width-four wave at this alignment. Scalar chunks use a
different attention alignment, so this is not a scalar/wave equivalence
claim. Each arm compares 384 complete rows with that fixed wave reference,
and all three arms' aggregate hashes also agree across binaries.

The measured seconds cover the 32 successful `ChunkWave` calls, including
plan preparation, device execution, capture/replay and complete-logit copies.
Prefill, initial weight loading and subsequent host hashing are outside this
timer. These are paid resident-wave timings, not HTTP completed-token rates
or a serving performance gate.

| Paired cell, 128 output rows | Control 1 | Candidate | Control 2 | Rate gain vs mean bookends |
| --- | ---: | ---: | ---: | ---: |
| Eager, seconds | 2.694667378 | 2.665845568 | 2.670027813 | 0.619% |
| Captured/replayed, seconds | 2.071711733 | 2.069439414 | 2.077303730 | 0.245% |

For equal row counts, the reported rate gain is
`(mean control seconds / candidate seconds − 1) × 100`; it uses the inverse
of the mean control duration, rather than the mean of the two control rates.

The eager bookend rate changes +0.923%. Both apparent gains are smaller than
the absolute change between their respective bookends. Every arm records two wave captures, 62
replays, zero coverage violations and successful completion and retirement.
The candidate records 1,536 executed-plan router groups and 141,557,760
paid concatenation bytes in each paired cell (48 layers × 32 completed
waves); unpaired cells and both controls record zero router groups. These
statistics are read only after successful wave execution, with the original
and joined product implementations authenticated by the composer.

The common aggregate SHA-256 values are:

| Object | SHA-256 |
| --- | --- |
| All conditioned histories | `ee8911edb06e86781d40fde1ac70f4eba8db6791ad43910152a28a12f50c70ca` |
| All greedy wave IDs | `ff2a30bff02586ebabfda9a53bd37c549764dd6cb3c9a57d700415828ad3b12c` |
| All full wave logit rows | `04fb4479f192fe453947ddc58c121c8ca875ed838c2a2b54d6be792196be0981` |
| Initialized final states, every cell/slot | `bec61d72fc094daacd9df7d69ead9482e5125c64ed9c385e1ae702773d2480ec` |

This establishes the measured native-control exactness only. It establishes
no cross-engine parity, quality-bound verdict, maximum-context behavior or
peak-memory claim.

## Provenance and replay

Measured on Spark B, NVIDIA GB10, driver 580.178.04, from private worktree
`codex/qwen-router-wave` based on `e2bc358`. Spark A warm-built only the
required benchmark target, using the existing pinned SDK and source lock
`440f03cdb52921c6c55843819e6ac950a5b2c4aafdc01055e52a0af3117ce32e`;
GGML is b10964 with llmpalooza's locked patches. Frozen binaries and source
inventories were then copied to Spark B scratch. Its production tree was
not modified. Each **src-only** inventory contains 446 files; the two
inventories differ only in `src/engine/qwen38_wave_plan.cc`. These inventories
are not interchangeable with the broader historical 487-file inventories.

| Measured object | SHA-256 |
| --- | --- |
| Control binary | `2c0f4bbc0134dffcaba419bf16a2dde99c8f75e4a8ac15d567fdb96fb45a5bc9` |
| Candidate binary | `d8357f5d1e21d7c88c2e4ebb7095aa714bdeded6cda9a4107d70f44da96352a4` |
| Control harness source | `6933ec4178e3d94c7bba697b09082996b9ab8d2e570fed3ac5ed91a718d14755` |
| Candidate harness source | `323e0e7aac6dd5066f88716d969275e21210670ba79e645d871bb0cb3b8f06f2` |
| Control src-only inventory JSON | `afc915c2db176ae47bfac50323ee929b5e0be168d592d9d6dd109336e97c4e8d` |
| Candidate src-only inventory JSON | `48d5493a74920787c87f2de357f4956d385ac404ae1a5e4781bf827612c1da6f` |
| GGUF artifact index | `d6c69d701ead0cbadc3a64f1f3c354a70ecac7179f61dbac1e1163fca0cd3550` |
| Prompt TSV | `c7cb91c3a5126fa6bc1de9d010b9d0f134ec1d529704656e03e48ad96436105c` |
| Frozen controller | `6e6f87d1a366ce6f2a018e5ed3f09043d7c9b0ef88886b1ecc8f9a55ab133815` |
| Archived prototype patch | `f3343ce1192762db27e3e8bdc6b0a7bb4e098411527f2131ca396ad512058129` |

Spark A jobs `qwen-router-control-freeze` and
`qwen-router-candidate-build2` completed zero and were waited on. Earlier
failed wrappers are retained: the first control wrapper used the wrong
binary-copy path after a successful compile; the first candidate compile
missed an empty designated-field initializer. Neither failure is qualified
as a pass. Spark B `qwen-router-gguf-screen` completed zero and was waited
on. These jobs used the installed supervisor, `--gpu` and timeout 600.

The controller runs control/candidate/control, each preceded by the existing
strong 105 GiB preflight (`3d9f07a48373741a24ef4e9615e99c6bdacb7854ca5cc2356afb73914ddc3ec5`),
then invokes the frozen binary as:

```text
BINARY GGUF_ARTIFACT prompts.tsv NEW_STATE_DIR 2048 on screen
```

After the three arms, the separate installed-GPU job
`qwen-router-terminal-room` (timeout 120) also completed zero and was waited
on. Its strong 105 GiB preflight at 06:12:39 EDT (10:12:39 UTC) reports 117.135 GiB free
and clear GPU/container/native-model probes. There are three pre-arm gates
and this separate terminal gate, not a post-arm gate paired with each arm.
All three model subprocesses were reaped by the controller; completion and
retirement are recorded in their own logs. The terminal probe qualifies the
later cleanup state, rather than the end of each timed cell.

Replay requires the archived private instrumented sources or frozen binaries;
the production benchmark does not expose `screen`. Raw logs, supervisor
receipts, binaries, source inventories, controller and summary are retained
on Spark B under `~/scratch/qwen-router-wave-20261004`, with build records
on Spark A at the same scratch basename. The workstation copy and complete
prototype patch/sources are under
`~/scratch/llmp-m3-qwen-router-wave-2026-10-04`. No production adoption,
broader check cycle or M3 exit-gate closure follows from this rejected screen.
