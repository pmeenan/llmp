<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 ModelOpt v1: native target and MTP drafter (2026-10-15)

The native importer's ModelOpt Qwen3.8 target and MTP drafter, served
through the shared engine, against the v0 artifacts the family's gate uses
(`c4fb47a9…` + `8600a998…`). Their routed experts are relaid out into the
scale-atom layout and the target's n-gram table is row-interleaved
([engine-components](../../engine-components.md#weight-and-state-formats),
"The scale-atom layout and the importer's expert relayout"). The importer's
parameter folds (D-108 as amended; by role, `import/parameter_folds.h`)
store the zero-centered norms' 1 + w and A's -exp(A_log) in F32, as v0's
converter did. The sections up to "Restored parameter folds" record the
interim state without those folds (the coordinator's H1), whose runs
motivated restoring them.

## Artifacts

On `spark-b` (GB10), imported from the staged revision itself,
`~/.local/share/llmp/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`
(its `llmp-archive.json` included: the variant's `amax*.safetensors`, which
the index does not name, are not required).

| Artifact | ID | Import |
| --- | --- | --- |
| target (`llmp import SOURCE`), folded | `d145900e058ff6470405c396affb9879b5c3c3770cc2b586d846e7e148c2058d` | 32 min 4 s |
| drafter with Mia's 47,172-ID list (`--component mtp --draft-head`), folded | `316caa59d577202ea236061ce350e4fa8daa1a9d451586b908cd1e4fedf55eeb` | 47 s |
| target without folds (interim) | `79d092cb…` | 31 min 37 s |
| drafter without folds (interim) | `6892c537…` | 41 s |

The folded target's stored spec digest is `c2e2822e…` (the interim one's
`85d5aa0b…`); the approval rows pin both artifacts
(`runtime/approvals.cc`). Its import takes about
4 minutes longer than a split one: the interleaved table's 2.5 million rows
are written lane by lane.

## How v0 quantizes the experts' activations

v0 never reads the checkpoint's `input_scale` tensors (its converter drops
them). The routed experts' grouped GEMM (`llmp_moe.cu`, after GGML's
`quantize_mmq_nvfp4`) quantizes each activation row on the device every
step: a row scale of amax / (6 · 448), each 16-value block's E4M3 scale
searched over five codes around amax / 6 for the least squared error,
E2M1 codes. Decode rows take `llmp.moe.gemv`, which quantizes each 16-value
block to 8 bits (amax / 127, as `quantize_q8_1`). The same holds for the
v1 tables, which take the same kernels.

## The drafter's default: W4A4 by measurement

The checkpoint quantized the MTP block's experts `W4A16_NVFP4` (no input
scales); the target's are `NVFP4` with input scales. Verification decides
every token, so a drafter's expert numerics change only acceptance and
speed (AGENTS.md rule 11). Two alternating single-arm runs each of the v1
pair (target `1dd4eec6…`, drafter `316caa59…`, the same binary on
`spark-b`), the drafter at W4A4 (v0's policy) and at W4A16 (the format
ladder's grouped product), fixed before the runs: W4A4 becomes the
default if it decodes at least 0.5% faster at equal acceptance.

| Cell | W4A4 prefill s | W4A16 prefill s | W4A4 decode tok/s | W4A16 decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| C1 | 3.168 | 3.192 | 48.77 | 48.21 (W4A4 +1.2%) |
| C2 | 7.036 | 7.030 | 67.35 | 66.49 (+1.3%) |
| C4 | 15.589 | 15.966 | 80.65 | 79.03 (+2.0%) |

Acceptance was 1,655 of 2,088 drafts in all four runs. The drafter mapping
now takes v0's policy for any speculative stage (`spec/mappings/qwen38.cc`,
the rule's comment); the target's W4A4 gate stays the importer's finding of
the experts' input scales.

## Serving

`Qwen38EngineServingGpu`, one slot, context 512, 16-row chunks, depth 3,
7-token prompt, two turns of 8 greedy tokens, each arm a fresh Server.
Bounds fixed before the runs.

| Arms | Bound | Result |
| --- | --- | --- |
| v0 target + v1 drafter (`NativeMtpDrafterMatchesV0`) | bit for bit | heads, tokens, logit rows, histories bit for bit; 4/21 against 4/21; its norms, zero-centered BF16, give exactly v0's (1 + w) |
| the same, the drafter's experts on the format ladder | tokens equal, acceptance at most 2 fewer | tokens equal, 4/21 |
| v1 target + v1 drafter (`NativeModelOptTargetAndDrafterServe`) | prefill argmax equal; tokens equal up to a first divergence whose v0 margin is below 1.0; acceptance at most 2 fewer | pass: argmax equal, prefill head's largest difference 3.60, first divergence at turn 0 token 5 (margin 0.12), 3/20 against 4/21; heads and states not bit for bit |

Quality, the speed gate's pass (`tools/speed_gate.py quality --family
qwen38`, 996 scored tokens), bound |ΔNLL| at most 0.1% fixed by the
coordinator before the run: v1 NLL 2.733237 against v0's 2.756445,
**−0.84%: outside the bound** (lower). With the norms and A folded at
import, as v0 holds them, the v1 target scored v0's NLL exactly (the same
score digest, `9266b500…`) and served bit for bit; the folds were taken
out because they grow resident bytes beyond the source encoding. The norms'
1 is exact either way (the drafter above serves bit for bit with
zero-centered norms), so the difference is A's -exp computed on the device
(fast-math `expf`) instead of in double on the host, which the routed
experts' top-10 selection amplifies.

## Speed

Two alternating single-arm runs each of `tools/speed_gate.py run --family
qwen38`, same binary and host, diagnostics
`LLMP_DIAGNOSTIC_NODE_CAP_BYTES=120259084288` and
`LLMP_DIAGNOSTIC_STARTUP_HOST_BYTES=auto`; v1 at its default policies.

| Cell | v0 prefill s | v1 prefill s | v0 decode tok/s | v1 decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| C1 | 3.185 | 3.160 (−0.8%) | 49.44 | 45.34 (**−8.3%**) |
| C2 | 7.012 | 6.995 (−0.2%) | 68.18 | 67.08 (−1.6%) |
| C4 | 15.896 | 15.807 (−0.6%) | 80.16 | 81.48 (+1.6%) |

Peak `MemAvailable` drop 91.1 GiB (v1), 91.4 GiB (v0). Accepted drafts
equal (1,654).

With the norms and A folded at import (target `1dd4eec6…`, two runs
each), v1 matched v0 within 1.5% in every cell (decode C1 −0.5%, C2
−2.6%, C4 −0.3%). Without the folds every step recomputes the 148 norms'
(1 + w) and the 36 layers' -exp(A_log) in small kernels (an estimated 400
a step, not counted), which a single decode row's step cannot hide: C1
decode is 8% slower.

Before the n-gram table was interleaved (its codes and scales in separate
groups: two direct reads a paged row against v0's one), v1 was 4% slower in
prefill and 2 to 3% in decode at equal kernel time (nsys: 103.6 s against
102.0 s).

## Restored parameter folds: bounds (fixed 2026-10-15, before the runs)

D-108 as amended (owner, 2026-10-15) lets an import store small
per-channel parameters derived in F32 when that removes run-time work. The
importer now folds the target's zero-centered norms and A and the
drafter's norms by role (`import/parameter_folds.h`). Bit-for-bit
agreement with v0 is a diagnostic only, recorded, not required. Acceptance,
fixed before the runs:

- speed: the M3.6 gate, `tools/speed_gate.py` `qwen38` against the frozen
  baseline `1facc957` (v0 artifacts, the family's configuration) at its
  1% non-inferiority margin, the v1 pair at its default policies through
  `--new-config` and the two diagnostics above;
- quality: `speed_gate.py quality`, |ΔNLL| at most 0.1% against the same
  binary's v0 pair (2.756445 above), the frozen baseline's
  (2.7655865) recorded beside it;
- serving: the bounds of the table above, unchanged.

## Why the derived values moved NLL

A probe (deleted after use) replicated, on `spark-b` and with the build's
`-use_fast_math`, the GGML arithmetic the components emit for each derived
parameter (BF16 to F32 conversion, `scale(w, 1, 1)`, `neg(exp(a))`) over
the target's actual tensors, against v0's folded F32 bytes and a host
computation in double:

| Values | Count | Differ from v0's folded bytes |
| --- | ---: | --- |
| conversions (BF16 to F32) | 1,521,856 | 0 |
| norms' 1 + w | 1,033,216 | 0 |
| A's -exp(A_log), on the device | 1,728 | 705, by at most 4 ulp |
| A's -exp(A_log), host double rounded once | 1,728 | 0 |

So 1 + w was never rounded through BF16: it is exact either way. The
0.84% NLL difference comes from A alone: fast-math `expf` is within 4 ulp,
and the routed experts' top-10 selection turns those last bits into other
experts on some tokens. The lower NLL without folds is that perturbation's
luck on this passage, not a systematic gain; the folded numerics (A exact
from double, the converter's and the checkpoint's own definition) are the
better ones, and they are also faster.

## Restored parameter folds: results

Target `d145900e…` with drafter `316caa59…`, same binary on `spark-b`:

| Check (bound fixed above) | Result |
| --- | --- |
| serving against the v0 pair | pass: prefill head difference 0, heads, tokens and state images bit for bit, acceptance 4/21 against 4/21 |
| the drafter's experts on the format ladder (W4A16) | tokens equal, 4/21 |
| quality, \|ΔNLL\| at most 0.1% | NLL 2.756445 against v0's 2.756445: 0 (the same score digest, `9266b500…`); the frozen baseline's 2.7655865 recorded beside it |
| the drafter publication against `8600a998…` | every one of its 34 resources and 2,048 expert slices byte for byte |

The M3.6 speed gate (`tools/speed_gate.py`, `qwen38`, 6 ABBA blocks in 8
primed sessions on `spark-b`): old the frozen baseline `1facc957`
(runtime `28890a54…`, the v0 pair `c4fb47a9…` + `8600a998…` at the family's
configuration), new this build (runtime `52eb47ce…`) with the v1 pair at
its default policies and no diagnostic switch. Slowdown is positive when
v1 is slower; the bound is the one-sided upper bound, the margin 1%.

| Cell | Metric | Old | New | Slowdown | Upper bound | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| C1 | prefill s | 3.2264 | 3.1724 | −1.67% | −1.34% | PASS |
| C1 | decode tok/s | 47.69 | 48.52 | −1.71% | −1.40% | PASS |
| C1 | wall tok/s | 21.69 | 22.07 | −1.70% | −1.40% | PASS |
| C2 | prefill s | 7.1512 | 7.0171 | −1.88% | −1.46% | PASS |
| C2 | decode tok/s | 63.98 | 67.06 | −4.59% | −4.18% | PASS |
| C2 | wall tok/s | 22.81 | 23.37 | −2.39% | −2.08% | PASS |
| C4 | prefill s | 15.8007 | 15.8083 | +0.05% | +0.55% | PASS |
| C4 | decode tok/s | 77.13 | 79.14 | −2.54% | −1.50% | PASS |
| C4 | wall tok/s | 23.35 | 23.77 | −1.73% | −1.39% | PASS |

The family passes. Peak memory (informational) is 8.3 GiB higher than the
frozen baseline's (90.9 against 82.6 GiB); the same build serving the v0
pair peaked at 91.4 GiB above, so the difference is the runtime's since
`1facc957`, not the v1 artifacts'. The same build also serves the pair
with no diagnostic variable set (`llmp-runtime chat`, a one-sentence reply,
acceptance 0.85).

## Open

- The MLX target's v1 artifact `b842fe59…` (its norms and A folded as
  ModelOpt's, rule 11) is approved by its publication (inventory, spec
  `90e6fbbb…`, fold records) and admission, but not qualified for serving:
  no serving comparison or speed gate of it ran here (a later task). A drafter with another list, or without a head, has no
  approval.
- The split NVFP4 experts' own fast products (a grouped vector product for
  decode rows; the block-scaled grouped GEMM with scales swizzled per step)
  are not built: the importer's relayout serves the checkpoints that need
  them, and split experts run the format ladder's grouped tiles.
- MXFP4 and MXFP8 routed experts, and dense NVFP4 linears, are not relaid
  out; another block-scaled encoding joins the relayout's encoding list
  once a grouped product reads its atoms.
- The interleaved table's import writes two small writes a row; buffering
  lanes would recover the import's extra minutes.

Run records: `spark-b:~/scratch/m36-qwenmtp-gate/` (`s6-*` the drafter
policy, `s7-*` and `q4-v1` the interim screen, `q5-v1`/`q6-v1` and `g1`
the restored folds' quality and gate, the nsys summaries
`nsys/w*_cuda_gpu_kern_sum.csv`); raw traces deleted after summarizing.
