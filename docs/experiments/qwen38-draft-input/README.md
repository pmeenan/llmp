<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 draft-head input rounding (2026-09-30)

The real-input comparison found no reason to change the draft head. BF16
round/widen preserved all 12 draft choices per head and added 2–3 µs;
the existing BF16 library product was about 40% slower. Neither arm proceeds
to an acceptance experiment. Serving keeps its original F32 draft-head input.
The retained implementation is a bounded capture diagnostic for future
comparisons on identical real operands. Target/verifier arithmetic, state,
caches, sampling, weights and token maps are unchanged.

## Source difference

The native [MTP graph](../../../src/kernels/ggml/qwen38_graph.cc) computes
one final mixed row, including after multirow catch-up, then calls
`ggml_mul_mat` with BF16 head weights and F32 input. The locked GGML
`mmvf.cu:852–856` selects its GB10 one-column vector product; F32 activation
pairs at `mmvf.cu:281–294` remain F32. The multirow BF16 product instead
rounds activation tiles (`mmf.cuh:208–209`). Locked GGML source identity:
`7dc37d35c542276122a6c0ce7beb72f7a816524873e391591a98ecac0cd0167a`.

Mia recipe `b8439110eec0230facbe4ddf0dffe01b8f769be0`
[`patch_mtp_draft_vocab.py:102`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/b8439110eec0230facbe4ddf0dffe01b8f769be0/files/patch_mtp_draft_vocab.py#L102)
casts the mixed input to the selected weight's BF16 dtype before its linear
product. This proves a precision difference, not an acceptance benefit.
Mia also returns BF16 output; the initial three arms return F32 logits and
do not reproduce the whole Mia head. The follow-up below measures actual
BF16 output separately. No Mia implementation is copied.

External replay compares original MMVF, existing `ToBf16` plus BF16
`get_rows` widening and original MMVF, and existing `GemmBf16` with rounded
BF16 input and F32 output. RN/widen preserves the vector reduction; the
library arm also changes reduction order. Conversion values occupy 5,120
BF16 bytes and, for RN/widen, 10,240 F32 bytes at width 2,560. Head weights
occupy 335,544,320 bytes for prefix 65,536 and 241,520,640 bytes for selected
47,172. All arms use the same full weight/input allocations. The temporary
runner RN option was removed; rounding is external replay only.

## Capture and replay controls

`jitllm_qwen38_spec --check draft-head` retains mixed rows and head logits
only for this diagnostic. It refuses context above 131,072, chunk above
8,192, more than eight steps, depth above three and heads above 65,536.
Pinned destinations remain node-owned through successful device-job
retirement; ordinary CPU vectors receive copies after completion. Retained
tensors must be packed, bound inside the activation allocation and live
through the final copy. Default graph/runner paths allocate no capture.

Both heads passed four independent capture-off/on/on/off prefills with
common forced history. Draft IDs, confidence bytes, full verify-row
fingerprints, target-state tensor and drafter cache/live-stream fingerprints
matched after draft and rollback. Captured inputs/logits/maps repeat byte
for byte. Four depth-three steps exercise catch-up rows 1, 1, 2 and 3 and
yield 12 real inputs per head; both heads' raw inputs are identical. These
are state-fingerprint and operand-byte controls, not universal state proofs.

Every original replay row matches its captured model row byte for byte
before timing. Fresh and CUDA-graph repeats match each arm exactly; RN MMVF
also matches an independently rounded F32-input control. Device mapped
argmax agrees with CPU ranking using original token IDs. A GPU component
control covers nearest-even halfways, signed zero, subnormals and a valid
260-output partial tile. Host graph controls cover both layouts, catch-up,
confidence and headless prefill: retention preserves the entire ordered
operation/type/shape sequence and enters the plan/cache identity.

Each input uses three CUDA-event batches of 64 graph suffix calls. Timing
charges conversion, product, argmax, optional confidence and ID mapping,
excluding host readback and provider completion-proof overhead. Confidence
equality belongs to the model capture control. Original/RN/library arms run
in that order, so the small RN delta is treated as neutral evidence.

| Head | Confidence | Original MMVF ms | RN/widen + MMVF ms | BF16 library/F32 output ms |
| --- | --- | ---: | ---: | ---: |
| Prefix 65,536 | Off | 1.417479 | 1.419455 | 1.997681 |
| Prefix 65,536 | On | 1.426994 | 1.428339 | 2.009374 |
| Selected 47,172 | Off | 1.026919 | 1.029652 | 1.435126 |
| Selected 47,172 | On | 1.035450 | 1.037321 | 1.442258 |

Values are medians of 12 inputs' batch medians. The library arm is 40.9%
slower for prefix and 39.8% slower for selected with confidence off.
Neither altered arm changes any winning token ID. Maximum absolute logit
movement is 0.024738; maximum original top-two margin movement is 0.010500.
Maximum softmax TV is 0.00172025 / 0.00172002 for RN on prefix/selected,
and 0.00172095 / 0.00172071 for the library arm. These are drafter-head
diagnostics, not a target-distribution gate. Twelve greedy rows do not
establish acceptance on another prompt.

No full adaptive A/B, altered-proposal rejection/swap gate or default change
follows. New proposals can change verifier grouping and its numerical path
despite unchanged target source. This does not repeat the earlier neutral
full-target-head/HC cuBLASLt or failed multirow target-verifier BF16 trials.

### Follow-up with BF16 product output

A fourth external arm uses BF16 input/weights and actual cuBLAS BF16
output, widened afterward only for the original argmax/confidence/map.
It charges conversion, product, widening and ranking over the same 12 real
inputs per head/full working set. The algorithm is the existing
`cublasGemmEx`, `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`;
there is no algorithm sweep. Original captured logits, fresh/graph own
repeats and mapped argmax pass, and widened BF16 bits are checked exactly.

| Head | Confidence | Fresh original MMVF ms | BF16-output library plus widening ms |
| --- | --- | ---: | ---: |
| Prefix | Off | 1.420394 | 2.001223 |
| Prefix | On | 1.424258 | 2.010117 |
| Selected | Off | 1.023493 | 1.441080 |
| Selected | On | 1.028103 | 1.447656 |

The BF16-output arm is also about 41% slower and changes no winning IDs.
Every winning maximum is unique: no BF16-induced winning ties occur in
these 12 rows; minimum top-two margin is 0.875. Maximum absolute logit
movement versus original is 0.118084 and maximum original top-two margin
movement is 0.137091. Maximum drafter softmax TV is 0.0135804 prefix /
0.0135433 selected. This matches the head operand/output dtypes, not Mia's
complete MTP state or its chosen library implementation. No output-quality
or full-model acceptance conclusion follows. The branch is not integrated.

Raw records are `spark-b:~/scratch/m3-qwen-draft-bf16-output/`, with exact
commands/operand pins in `pins.json` and all four-arm vectors in
`replay-{prefix,selected}-{on,off}`. Supervised job
`qwen-draft-head-bf16-output` completes rc0 at 09:15:53 EDT after external
SDK compile/format/tidy, replay and analysis. Source/binary/object SHA-256:
`c3ecd43f64d6ed1d2024159414a5de5ae8dcdd3cd94a98306c30f323d8bb28b6` /
`864e5e7bb1b76e6411d49406fa18671c2c5035e00dc1cd4d92ef2279c34aa115` /
`e676202068ad84a4942841551b670415b6a76a1c38aeb3f0a683456cef1937e6`.
Controller SHA-256:
`c73ca7d621491c30fb6a080533f983362b9e178834762172cd9dda3b8aaf5893`.

The retained slice passes the locked Spark-native 1,037 tests (201 GPU),
SDK formatting and six host tidy units, 294 boundary checks and nine early
CLI refusals; actual REUSE/header checks pass for the exported tree.

## Decode-only profile and effective precision

Two subsequent 512-output profiles use the identical 128,799 prompt IDs,
context 262,144, chunk 4,096, runtime prefill and graphs on. Prefix runs
current adaptive depth 2–3; selected 47,172 runs fixed depth three to match
Mia's head/depth configuration. Their phases and policy differ and are not
a head performance A/B. CUDA capture starts after prefill and first argmax,
covering draft, verify, judge and owed recurrence commit/rollback through
final settlement. Loading and prefill are excluded. Profiled rates are not
unprofiled speed comparisons.

Routed gate/up-plus-SwiGLU and down `Gemv<true/false>` account for 30.8% /
32.3% of summed GPU kernel time on prefix/selected. MXFP8 vector products
are also major consumers. Recurrent commit alone is 1.6% in both captures;
this does not establish recurrence as the dominant remaining bottleneck.
Summed kernel time is distinct from elapsed generation time and excludes
CPU waits/copies between kernels.

| Stage | Current native path | Reference distinction |
| --- | --- | --- |
| Routed products, ≤8 rows | NVFP4 E2M1 weights/E4M3 scales; signed Q8 activation per16, amax/127, roundf; DP4A/F32 accumulation and F32 output/SwiGLU | Mia's actual main consumer is FlashInfer CUTLASS with FP4 activations, static a1/a2 scales and BF16 output; its drafter is Marlin |
| Small-row MXFP8 linears | E4M3 weights/E8M0 scales decoded by the vector product; raw F32 activations, F32 FMA and output | Mia quantizes activations for its FP8 product; this is separate from routed Q8/FP4 |
| Wider native routed products | FP4 E2M1 activation with per16 E4M3 and row-global scale; CUTLASS F32 accumulation→BF16 product; BF16 SwiGLU input/FP4 down requantization | Small-row model transfer loses about 19%; Mia's observed cooperative tile/stage pair loses about 33%, with native dynamic scaling retained ([grouped factor](../qwen38-grouped-verify/README.md)) |
| Draft head | BF16 weights, F32 input/accumulation/output | Mia BF16 input/output; isolated input RN was neutral here |
| KV / recurrent state | F16 / F32 | Pinned Mia recipe FP8 KV / BF16 SSM; no causal cache/state quality-loss measurement is inferred from these profiles |

The ≤8-row path covers current target verifies and MTP passes. Existing
grouped FP4 code is not another schedule for identical Q8 arithmetic.
Any direct consumer experiment must charge preparation, product and casts
and pass the existing model-quality gates. Earlier neutral weight-sharing
and slower small-column MXFP8 tensor-core trials remain separate results.

## Provenance and reproduction

Runs are serial on `spark-b`, GB10 `sm_121`, CUDA 13.4.92,
SDK `aarch64-e0a0c85c42806fb1`. Every load passed checked 105-GiB memory,
GPU-process, native-process and Docker-container gates. Artifact identities:

| Role | SHA-256 artifact identity |
| --- | --- |
| Target | `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93` |
| Prefix drafter | `056a750e3a90be3ae6a4b12bb963ce45290aaa6f52b5ba9799e777d491f80aea` |
| Selected drafter | `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40` |

Capture uses context131,072/chunk4,096/fixeddepth3/foursteps. Tokenizer is
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/tokenizer.json`; source messages
are `~/scratch/m3-qwen-mtp-speed/128k.json`, SHA-256
`b737f8a903c9ecb9257daeeac5d009b507e8b204ce99f89b10d88e88a0435557`.
Canonical little-endian I32 prompt file:
`~/scratch/m3-qwen-draft-input/capture-prefix/draft-head.prompt.i32`, SHA-256
`c9d455d1afc75e1fd4e0f93d0312f1768196609fab8a5836f57c15a72e2e3661`.
Both heads' 12×2,560 F32 inputs have SHA-256
`a1f76278fef27af4315f3ca72c8ff458bdd4b89e5955a3f9ab798820b6763884`.

Measured capture/profile binary SHA-256:
`baf6f67de12f57561050dc89a832d81d69531aa2cd731230a50dd37aab420d29`.
MTP graph/runner source SHA-256:
`158887618f277d4ae2af1551adcc66198360495b00945184b7dbac73438b9f3b` /
`b30853b62f81d0b08bea04e372bea8ba30a9e8eb6fb2a68ed3bae6130ecc106c`.
This measured source contained an unused default-false RN branch; its
removal preserves the measured F32 operation sequence.

Clean replay source/binary/object SHA-256:
`60f16ef926e5ecf52799e9b6bdd6dc312a8e6d8dbca9773b9d7b88a42488be79` /
`ac82608bdbd6e3548868d68430f12a5d9fdf2994e2c744d169c5139a058dddfb` /
`afb402f43aabbe9c4bd7b0fa061c4888fdb0816e34161ee355f52e6b6c3db02a`.
Extractor/controller SHA-256:
`ae88f5a219da11db42f53e4aef7c7b3f3071f33174fcf408404db544e5e7b557` /
`e997f565531c50a4acf84c159a9e32657c8c44a130d209f842582c4df82cbe0c`.

Raw records are at `spark-b:~/scratch/m3-qwen-draft-input/`:
`commands.json` has exact commands; `inputs-{prefix,selected}/pins.json`
pins artifacts/chunks/weights/maps/logits; `capture-{prefix,selected}/spec.json`
has passed four-arm controls; clean replay records are
`replay-{prefix,selected}-{on,off}-v2.jsonl`, full-row directories and
`analysis.json`. V1 failed at final provider-stream retirement and remains
failed. V2 records/synchronizes/queries/releases a provider fence before
retirement; it reuses only qualified capture after operand/source hashes
are revalidated, never failed timing data.

Decode profiles are at `spark-b:~/scratch/m3-qwen-current-profile/`:
`pins.json` records commands/gates/input matches, `prefix-adaptive` and
`selected-fixed3` directories have summaries, and matching `.nsys-rep`,
`.sqlite` and `*-kernels.csv` retain traces. Both complete rc0 in the
supervised job from 08:46:01–08:49:07 EDT, 2026-09-30.

Repeat the capture below, substituting selected artifact and vocab47,172
for its arm. External pinned extractor/replay source is beside raw records;
provide the full validated head/map and require original logits to match
the captured model logits before interpreting timing.

```sh
jitllm_qwen38_spec --qwen38-artifact TARGET --drafter PREFIX \
  --tokenizer TOKENIZER --prompts 128k.json --only qwen3.8-128k \
  --context 131072 --prefill-chunk 4096 --check draft-head \
  --tokens 4 --draft 3 --draft-vocab 65536 --adaptive-depth off \
  --runtime-prefill on --graphs on --out CAPTURE
replay HEAD.bf16 INPUTS.f32 IDS.i32 ROWS 12 CAPTURED-LOGITS.f32 OUT on
```
