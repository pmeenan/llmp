<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen routed FP4 products in a small target verify

The existing grouped FP4/BF16 expert chain is slower than the native
Q8/F32 vector chain at a four-row speculative verify. A separate bounded
arm uses the tile/stage/cooperative schedule pair observed in Mia's actual
main-model consumer and is slower still. Both prototypes are removed from
the published source; their checked source is preserved externally for
replay, and serving retains its previous vector choice. These are implementation-factor
tests, with the stage-precision differences below held visible.

## One chain, two product schedules

The archived `llmp_qwen38_spec --verify-experts vector|grouped|grouped-mia` changes
only the target Verify kind. Plain decode, prefill and the MTP drafter
keep their existing choices. The choice enters the plan-cache key and
Setup's workspace probes. Invalid nonverify/reference/GGML-layout choices
refuse; the cooperative pair additionally requires the measured K/N
shapes and at most eight rows.

| Chain | Activation preparation | Intermediate products and activation |
| --- | --- | --- |
| Vector, default | Signed Q8, per-16-value amax/127, roundf | DP4A/F32 FMA and XOR reduction; F32 gate/up, SwiGLU and down |
| Grouped | E2M1 FP4 with per-16 E4M3 scales and dynamic F32 row-global scale | F32 accumulation to BF16 gate/up; BF16 SwiGLU input, FP4 requantization, BF16 down, original F32 weighted combine |

Weights remain the same NVFP4 codes/scales and prepared strides. The
grouped arm runs the existing route/quantize/gate-up/GluQuantize/down/
combine chain, including scale-padding initialization and argument setup.
This changes both activation quantization and intermediate rounding.
Mia's main backend is `FLASHINFER_CUTLASS` with static a1/a2 activation
scales and BF16 output; its drafter uses Marlin. Matching the following
tile pair does not reproduce Mia's full fused-MoE implementation or scales.
The [fresh Mia profile](../qwen38-mtp-speed/README.md) records actual
consumer attribution and two unprofiled controls.

| Grouped schedule | Gate/up M×N×K tile | Down M×N×K tile |
| --- | --- | --- |
| Existing native | 128×128×256, auto stages, pingpong | 128×128×256, auto stages, pingpong |
| Observed Mia pair | 128×64×128, 5 stages, cooperative | 128×32×256, 3 stages, cooperative |

The pinned CUTLASS builder maps the cooperative tag to
`KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<3>`.
Common stride and scale-layout types are checked at compile time; planned
scratch is the maximum of both cooperative product variants. The original
grouped prefill configuration stays unchanged. The prototype registry describes the
explicit op-parameter selection, including each variant's tile/stages.

## First complete-model comparison

Both fresh model processes use the same 128,799 canonical input IDs,
curated 47,172-row drafter head, fixed depth three, context capacity
262,144, 4,096-row production-prefill chunks and CUDA graphs. Each process
runs two fresh generations with 512 output IDs; the benchmark deliberately
ignores stop to fill that budget. The runs are unprofiled and serial.
Own repeats are exact in both modes. They are not cross-mode quality,
rejection or swap evidence.

| Expert chain | First decode tok/s | Fresh repeat tok/s | Accepted/drafted | Verify steps | Draft / verify / complete step ms |
| --- | ---: | ---: | ---: | ---: | --- |
| Vector | 46.289 | 46.220 | 336/520 | 175 | 8.034 / 55.085 / 63.176 |
| Existing grouped | 37.386 | 37.442 | 308/605 | 203 | 7.986 / 59.191 / 67.230 |

The first grouped arm loses 19.0–19.2% decode throughput and its average
verify cost rises 7.45%; acceptance also falls. It is rejected for default
adoption without spending a larger quality/rollback/swap batch on a slower
candidate. No quality-gate pass or universal probability equivalence is
claimed. Whole-runtime cache precision is unchanged: F16 KV and F32
recurrent state. Planned activation/pool/host-input bytes are identical
at 1,417,674,752 / 1,518,338,048 / 10,485,760; peak MemAvailable drops
are 83.01 / 83.15 GiB for vector/grouped respectively.

The follow-up repeats the entire paired protocol with a fresh control,
changing only the grouped tile/stage/schedule pair. The fresh vector's
512 output IDs exactly match the earlier vector control.

| Expert chain | First decode tok/s | Fresh repeat tok/s | Accepted/drafted | Verify steps | Draft / verify / complete step ms |
| --- | ---: | ---: | ---: | ---: | --- |
| Fresh vector | 46.390 | 46.318 | 336/520 | 175 | 8.000 / 54.991 / 63.042 |
| Observed cooperative pair | 30.782 | 30.889 | 308/605 | 203 | 7.973 / 73.471 / 81.492 |

The cooperative pair loses 33.3–33.6% decode throughput against its fresh
vector control; verify cost rises 33.60%. All 512 output IDs and acceptance
match the existing grouped arm, so the new schedule does not recover that
factor's loss. Planned envelopes remain identical; peak drops are
83.02 / 83.15 GiB. The slower branch ends here, without claiming unchanged
model quality. Matching static activation scaling and the full actual
fused consumer is still unmeasured; this schedule result does not predict
that cost.

## Controls and provenance

The prototype graph control proves three separate choices, all-row verify outputs,
full drafter streams and saved recurrent histories; reference/invalid
choices retain their refusal. One- and four-row GPU controls exercise both
grouped schedules with the existing BF16 product-rounding and absolute
FP64 bounds. Wider prefill controls retain their previous MMQ-relative
bounds. The published test retains the useful one-/four-row controls for
the original grouped kernel, without the removed cooperative option.
These numerical controls establish a bounded operator contract,
not full-model quality for changed FP4 activation rounding.

Host is `spark-56f5` (`spark-b`), CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`; native base `af879a0` plus this experiment.
Artifacts: target
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
curated drafter
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
The canonical little-endian I32 file is
`~/scratch/m3-qwen-draft-input/capture-prefix/draft-head.prompt.i32`,
SHA-256 `c9d455d1afc75e1fd4e0f93d0312f1768196609fab8a5836f57c15a72e2e3661`.
Message fixture `~/scratch/m3-qwen-mtp-speed/128k.json` has SHA-256
`b737f8a903c9ecb9257daeeac5d009b507e8b204ce99f89b10d88e88a0435557`;
the controller checks emitted token IDs against the canonical file.

First paired run `qwen-grouped-verify-model-ab` completes rc0 at
09:37:27–09:42:29 EDT on 2026-09-30. Its measured benchmark SHA-256 is
`b19330257d4428fbbf9ba5556ee3ce5609fc2f3fbeda812d01116361a1e4356a`.
Raw commands, full source identities, load gates and summaries are under
`spark-b:~/scratch/m3-qwen-grouped-verify/model-ab/` (`pins.json`,
`vector/spec.json`, `grouped/spec.json`); the external controller is
`~/scratch/m3-qwen-grouped-verify/run.py`. Every load requires at least
105 GiB available and no GPU/model/container process. Raw tokens/logs
remain outside Git.

Follow-up `qwen-grouped-mia-model-ab` completes rc0 at
10:10:52–10:16:00 EDT. Benchmark SHA-256 is
`f135b7f65ff4ccc2028c709acb3606530202a8ef545c30acc7158f0fab56e6ed`;
CUTLASS object SHA-256
`16dc6753d6d767fdba917f974040d8c89b8a2fd255870afa1e80a281ad8e72aa`;
CUDA expert object SHA-256
`090ac00ef662f69d322cbda3163b25924340751c4e733288fee626c753fe5894`.
Its `model-mia-ab/pins.json` stores every source/operand identity and load
gate; controller `run-mia.py` SHA-256 is
`8a84e06017479b7fa8f9e38a6faf53901ce06fcf85965af4443d49ae3a64f966`.
The archived candidate patch is `qwen-grouped-mia-candidate.patch`, SHA-256
`80c2752276fd1a1f891011cd9049f87cc69b10a3484f5530cee7c2f6652f5d19`,
with corresponding sources under `prototype/` in the same scratch root.
Apply it to the recorded base to reconstruct the benchmark choices;
they are deliberately absent from the final production source.

The cooperative prototype passes the locked Spark-native 1,038 tests
(201 GPU) in 63.43 s, SDK format, eight host tidy units, 294 boundaries,
the enum refusal and actual REUSE/1,003 headers. This check belongs to the
measured prototype, distinct from the final source with its options removed.

After removing the prototypes, the final implementation is production `cf63418`
plus the retained small-row controls. Spark prepare/locked
tests pass 1,039 tests (202 GPU) in 64.54 s, SDK format and one host tidy
unit; the checked mirror/export has 295 clean boundaries and actual
REUSE/1,007 headers. Checked runtime / long-swap / receipt SHA-256:
`757c29452d26bfbd0fcc07686d0a13a632eef0850621be68e02ad7144886eaf0` /
`1cc15b612782efc1b52264724ab6cac7b227b4bfb2fb8fb80471bd53d318415c` /
`303028857f8a87cbdf3a7735c4536ebfcb3b4066d340115540228d37cd7e1f8e`.
The warm mirror was updated by owned-file overlays, rather than a complete
source replacement. A subsequent Git-blob audit of 449 implementation,
build and harness files found one mismatch: the unused HTTP helper
`docs/experiments/long-context/longctx.py` was stale. All `src/`,
`benchmarks/`, CMake, source-lock/patch, `tools/` and baseline-helper files
in that audit matched `cf63418`. These native benchmark comparisons do
not use `longctx.py`; their measured source/object pins remain separate
from the later HTTP runtime queue. The stale-helper HTTP run is excluded
from qualification. The replacement HTTP protocol pins the checked helpers to
SHA-256 `340f7d2a3ca6397c807cf0b2502954fe0a22552c3e5aa44997b34c9ba24f5cae`
(`longctx.py`) and
`41c693de64e2d63b32622cb03f29024d76455c0ad7efac4394ea3f5d45f4b1e2`
(baseline helper), with its sibling prompt fixture required as well.
The first replacement attempt refused before model load because that
fixture was absent; this note claims no completed HTTP qualification.
The REUSE/header export proves its license/header
checks, not whole-tree content equality.
Workstation/package checks remain deferred with the owner's current M3
override. No changed-default quality or speed gate is claimed by this unit.
