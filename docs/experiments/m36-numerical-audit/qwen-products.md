<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen BF16 product recovery: bounded replays 46 and 47

These Spark B experiments separate legacy owner-width preservation from
activation precision. They do not qualify original-model quality, serving,
state migration or the matched 1% gate. The original unquantized Qwen3.8
reference remains unavailable. No singleton-bit agreement is an acceptance
requirement for choosing the next implementation.

## Inputs and method

Both runs use authentic native BF16 weights with K=2560 and N=512 (router),
10240 (PLE key), or 2560 (PLE value). One captured owner0 F32 activation is
repeated into eight physical columns; the seven peers are **synthetic**.
The source artifact is
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`.
Its authenticated index and 35 complete 2MiB chunk hashes cover the 65MiB
weight extraction; this is not a new whole-artifact rehash. Inputs came from
the completed Qwen45 shared capture. Prepared files were reused unchanged.

Each role runs four fixed ABBA blocks, 64 captured launches per cell,
16 cells total. GPU event timing includes the complete product path and
excludes input staging and CPU analysis. All cells, including outliers,
are retained. Every full output is finite; same-path repeated execution and
changed-input graph replay pass. Binary64 references use the stored BF16
weights and captured F32 input, with a conservative summation-error enclosure.
This local product reference is not an original-model quality reference.

46 compares A, four products of widths 1/2/3/2 plus three concatenations,
with B, one ordinary eight-column MMF product. 47 separately compares B
with C, one existing eight-column `llmp.mul_mat.mmvf_rows` product. C uses
no owner views or concatenation and introduces no arithmetic kernel.

## Measured results

Descriptive median microseconds per captured path:

| Product | 46 A split | 46 B joined MMF | 47 B joined MMF | 47 C joined MMVF |
| --- | ---: | ---: | ---: | ---: |
| router | 26.626250 | 8.199000 | 8.198000 | 14.340000 |
| PLE key | 931.174992 | 236.614750 | 242.793500 | 235.229000 |
| PLE value | 53.282000 | 16.392000 | 16.393250 | 45.085250 |

Only A/B within46 and B/C within47 are paired comparisons. Historical A/C
is directional evidence, not a paired speed result. In47, C/B is 1.7492,
0.9688 and 2.7502 respectively. Key cell ranges are B238.0525–247.0375µs
and C230.7395–235.625µs; router ranges are B8.194–8.2065µs and
C14.3205–14.3485µs; value ranges are B16.376–18.3215µs and
C45.053–49.1575µs. Router46 includes A30.4315µs/B11.324µs outliers.
No serving speed threshold is inferred from these short replays.

| Product | B owner0 RMS error | C owner0 RMS error |
| --- | ---: | ---: |
| router | 0.0012390226456680657 | 2.4535916282882237e-7 |
| PLE key | 7.620043979146656e-6 | 7.550516020433528e-10 |
| PLE value | 8.802214280111787e-6 | 6.871567673797366e-10 |

C retains this lower error across all eight columns. B reproduced every
prior46 B output byte. All C columns happened to reproduce46 A's singleton
output; this is an observation, not a required cross-implementation oracle.
In46, all B columns already matched A's seven multi-column peers. Removing
owner-width preservation exposed the singleton to their arithmetic.

The actual recorded kernels and prepared GGML tree
`ggml-37947dde9e23668f` explain the precision difference: MMF selects BF162
(`mmf.cu:120–127`), converts loaded F32 activation pairs (`mmf.cuh:208–209`),
and accumulates BF16×BF16 into F32 (`mma.cuh:1325`). NVIDIA conversion uses
the BF16 constructors in `convert.cuh:59`; line57 is the HIP branch.
MMVF keeps F32 activations and accumulation (`mmvf.cu:291–293`). Thus this
is activation precision contraction, not solely a different F32 reduction.
No separate internal conversion buffer was captured.

## Capability boundary and next step

The existing rows implementation supports one to eight physical columns.
A main-based descriptor slice exposes its dense BF16/F32/F32 form only
when existing device tuning facts name an exact rows/K/N/F32-output entry.
Execution compatibility seals the optional output encoding as well as the
shape, distinguishing omitted, F16 and F32 tuning; unknown encodings refuse.
The capability independently checks packed two-dimensional shapes, strides,
alignment and default product parameters. Lowering identifies the actual
selected product and checks its physical binding in measuring and bound
modes. Ordinary products, expert selection and contributions are unchanged;
no runtime caller installs new tuning entries in this slice.

This foundation has no dependency on the 78-path pending Qwen migration,
owner-width splitting or source-format conversion. Wider12/16-column
products retain existing selection or refuse an explicitly unavailable
implementation; they are never silently tiled. The descriptor domain is
not a performance qualification of every shape. Measured47 data covers
only the three eight-column products above; host fixtures are synthetic
contract controls. A chosen production source still needs immutable tuning
ownership, funded plans, same-plan state and broader secondary quality
controls. Original-model and full migration gates remain open.

## Provenance and resource scope

46/47 ran on Spark B's GB10 with the repository-pinned GGML and SDK.
Source, recipe, executable and input guards passed before and after both
runs; each completed one native case with no skips and zero remaining GPU
processes. Native wall times were1.280s/0.870s. Observed cgroup process/cache
peaks were249,516,032B/246,935,552B, all memory events zero; these are not
attributed total GPU/runtime peaks. Each role reused one completion-owned
96MiB device and64MiB host allocation, with no next-role allocation until
successful retirement. Unknown completion retains the entire owner.

The exact numeric receipts are SHA-256
`d2d95c0be974a04e009d4462d0b2cd765759ee86f5ef4b0d84b33249dce8e8bd`
(46) and
`4d6bc093c295d9b457c78957efe748a833fd291f63a689aab2606c7b88daaf08`
(47). Independent audits recomputed all six output hashes and48 column
metric sets per run. Raw records remain outside Git under
`/tmp/llmp-m36-coordination/qwen-recovery46-gpu01` and
`/tmp/llmp-m36-coordination/qwen-recovery47-gpu01`; prepared-source receipt
`3e49155d025eb93335e1a2e3346f0583fecabcad6fd23fafd0586e5b66e4762b`
binds the shared extraction and references. The existing prepared artifact
and captured inputs are required to replay; this report does not authorize
another checkpoint download.

The inactive descriptor foundation subsequently passed a Spark B native build
and all93 focused host controls (20 GGML owner/selection,13 owner-proof,
15 registry,45 planner), zero skips/failures and all source/recipe/binary guards.
This includes measuring and real-address descriptor lowering, exact selection
isolation, protected older owner policies and invalid-binding refusals.
The first attempt compiled but two new controls exposed an incorrectly narrow
parameter-buffer check; canonical zero padding is now accepted while malformed
padding refuses. This was a host eligibility defect, not a numerical result.
The successful receipt is
`773fbb8d0b3bd84e7508034d73fedb8af944ef1817b6824e69b005a9815306ec`.
No additional GPU replay was needed for the unchanged kernel; no source
activation, original-quality or family migration claim follows from these tests.

Main subsequently passed a full Spark B native build and all98 focused controls
(the same93 plus five execution-identity controls), with zero failures/skips
and source/recipe/binary guards passing. Output-typed tuning identities are
stable on repetition, distinguish omitted/F16/F32 output, and reject unknown
encodings. The root receipt is
`50d6607cc2c562345566573244dbcd83d396c44c4348b9f0c4127a7cf1e4ed29`. These checks do not activate runtime tuning or qualify
full-model quality, performance or serving.

## Owned backend configuration

The next shared slice owns the qualified native backend configuration before
a required source can publish its execution identity. It retains typed tuning,
actual device and cuBLAS/Lt settings, paid handle/workspace readiness and all
stages' spec/layout/policy/domain in source, plan-cache, snapshot and kept-state
compatibility. Failed preparation publishes no partial identity. Changed required
configuration refuses; it cannot become an absent optional identity.

Metadata has a bounded pinned allowance before node start and keeps one catalog
payment across setup retries. Direct programs and captured recordings retain the
launch generation, so replacement or recycled addresses cannot revive retired
work. Uncertain completion still retains the complete owner. The newly shared
FP16 adapter keeps its ordinary source and checked lowering; this slice does not
promote that borrowed configuration to required sealed authority.

The implementation uses the qualified pinned installation. Recursive runtime
SDK/library byte attestation, loader scans and selected-SASS extraction were
removed from the design; compiled provenance, real readiness and completion
checks remain. This reduces prospective complexity without changing arithmetic
or selecting a new family policy. Joined qualification and the matched 1% gate
remain separate. Main passes the full native build and all 148 focused
host/refusal cases, including funding limits/retry lifetime and required-source
plan/snapshot/kept refusal before effects. One actual model-free control passes
real handle preparation, capture with changed inputs, old-program refusal,
shifted-pool rebind and deterministic same-address stale-record refusal. The
FP16 adapter also passes all three existing controls; both full-result hashes
and reload properties equal the previous shared runs. Source/recipe/binary
guards pass, no compute process remains and all max/OOM events are zero.
Final format, boundary, header and license checks pass.

The synthetic check retains the existing 64-MiB host and 192-MiB device
inventory, inside a 4-GiB zero-swap process/cache cap with the existing runtime
allowance and physical margin. Its observed cgroup peak is 218,488,832 bytes;
this is not device occupancy. The adapter retains its existing 16-GiB zero-swap
cap and observes 2,174,857,216 bytes of process/cache occupancy. Neither fixture
is a family quality or latency qualification.

| Final main evidence | SHA-256 |
| --- | --- |
| Full native build and 148 focused controls | `b6737ccc0cdaa8ca4261de49e6cee08be8c0e02099cc69ccbc6351da517073b2` |
| Actual model-free backend control | `4cac15cfe32f2ceebc3c08d40607a5208d6a99bbe1e6ed0a98f37e36ffa9aafe` |
| Actual FP16 adapter controls | `4911d5a1e5f209127541e33f4c52db28b185749cd5d59da7e6f09d06757fd20b` |

## Shared phase admission correction — 2026-10-12

A closed score/decode diagnostic source refused its first scoring wave before
model arithmetic: the shared target-wave transport incorrectly required prompt
permission for every phase. It now checks each actual pending phase before
builds or state effects: prefill uses prompt permission, decode uses decode,
and teacher-forced scoring uses score. Draft/verify target waves refuse.
Companion execution retains its existing separate permission; companion
prefill keeps the existing compound-only rule, while companion decode/score
also require their own semantic mode. No domain or numerical policy widens.

Root main passes the full native build and all 60 affected host controls
(12 corpus, 19 companion, 29 decoding), with no failures or skips and all
source/recipe guards passing. Three added controls cover score-only completion,
mixed-phase refusal before effects and companion-permission refusal. The first
root harness omitted its required scratch directory and failed at fixture setup;
that failed receipt is retained. The corrected harness passed in seven seconds
on the same source; the initial full build passed in 25 seconds.

- Independent source review SHA-256: `050d5e4f8c7dea5797c0ba8a33b1724213047d1d72411a1dae89693192e3bc56`.
- Main result receipt SHA-256: `4374b0467df394538255f623a3e4132e09e08390894902073d13818d153d6c64`.

Raw evidence remains outside Git under `qwen-phase-admission-root-main01`
and `qwen-phase-admission-root-main02/results` in the coordination scratch root.
This correction does not adopt Qwen50 arithmetic or establish model quality,
state or speed qualification.

## Checked weight alignment — 2026-10-12

The shared runner now supplies actual single-part weight reservation alignment
by complete WeightRef/part role, capped by the provider's reservation promise.
Component preparation validates power-of-two, nonzero-address and divisibility
promises before lowering, and forwards only matching semantic root facts.
Multipart logical roots remain unknown. Measurement uses actual reservations,
not placeholders. The physical lowerer rechecks the selected binding.

Valid facts alone do not activate the explicitly tuned BF16/F32/F32 product.
An actual component/planner/lower host control covers eleven modes: valid
measurement and bound facts; missing facts; malformed or misaligned promises;
unrelated WeightRef or part role; zero addresses in both modes; and valid facts
with no tuning or wrong-output-type tuning. Unknown facts select ordinary
products; invalid promises refuse before the lower callback. No arithmetic,
capability predicate, owner policy, fusion setting or tuning pins change.

Root's warm main native build passes on Spark B, followed by all 108 affected
host controls: ModelRunner 54, GgmlLowerQwen38 12, GgmlLowerDsv4 29 and
ExecutionContract 13. The eleven modes execute inside one named Qwen control;
they are not eleven independent test cases. Source, recipe and binary guards
pass with no failures or skips. No new GPU inference was necessary for this
source-fact carry. The separate [joined scoring report](qwen-joined-score.md)
retains the completed private diagnostic's measurements and limitations.

- Main build/host receipt SHA-256: `7a87eccda7a89b9861cad245e72f6c6350fe89895d4c229cedad9b2bf8e8b17b`.
- Final source review SHA-256: `36a8419e5ef97d36bc8a0a3a78e16a2ee0895e5477303288bf42821612652479`.
- Current carry/recipe review SHA-256: `7bd58d7d6a35bfcb34ca57927090d13eca1e30f9165bef98411485b8e00a7d18`.

Raw evidence remains outside Git in the coordination root's
`weight-alignment-current-main07/results`. Family admission and migration gates
remain open.
