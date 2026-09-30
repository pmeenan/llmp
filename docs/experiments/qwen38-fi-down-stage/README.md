<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 literal FlashInfer down stage (2026-09-30)

The captured-input comparison rejects this standalone G2 transfer. Its
complete cold latency rises 20.95–27.64%, and warm latency rises
28.77–29.27%, against the original routed vector product and combine.
No model candidate, quality-loss percentage or production change follows
this slower operator. The bounded native capture diagnostic is retained
for another source-backed component comparison.

This is a narrower factor than Mia's complete fused MoE. It starts from
native F32 post-SwiGLU values, changes the G2 activation preparation and
scaled BF16 product, and keeps native ordered route/shared combination.
It neither reproduces Mia's first product nor tests its atomic fused
finalization. Earlier [dynamic grouped and cooperative-tile trials](../qwen38-grouped-verify/README.md)
remain separate negative experiments.

## Actual consumer and factor

The pinned image is
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
recipe `b8439110eec0230facbe4ddf0dffe01b8f769be0`. Its main expert consumer
is vLLM `FlashInferExperts` with `FLASHINFER_CUTLASS`; the drafter uses
Marlin. FlashInfer is 0.6.17, commit
`a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`. The source identities and
Apache notices are recorded in the [upstream note](../../upstream/flashinfer.md).
No FlashInfer code or new imported format is added to the repository.

The actual initial input preparation calls private vLLM
`torch.ops._C.scaled_fp4_quant.out`; FlashInfer subsequently expands those
codes and scales. Replacing that call with FlashInfer's optional
unquantized-input branch would not prove first-product equivalence.
The source-visible G2 stage computes F32 SwiGLU from BF16 first-product
outputs, rounds to BF16, then quantizes using static NVFP4 calibration.

vLLM reduces each layer's 512 calibrated `down_proj.input_scale` values
to one maximum. The activation encode scale is F32 `1/a2`, and each
raw down-weight scale is multiplied by `a2` before the BF16 product
epilogue. This factor uses that exact order. The three quantization
environment knobs `FLASHINFER_NVFP4_4OVER6`,
`FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH` and
`TRTLLM_DISABLE_FP4_QUANT_FAST_MATH` were absent in the pinned-image
probe; its quantizer uses the corresponding default branches.

| Stage | Original replay | Literal G2 factor |
| --- | --- | --- |
| Weights | Original packed NVFP4 codes and E4M3 block scales | Same bytes, no dequantized replica |
| Activation | Native F32 post-SwiGLU, signed Q8 per16 | Same F32 values rounded BF16, static-a2 NVFP4 E2M1/E4M3 |
| Product | Native DP4A/F32 vector product | Existing native ping-pong grouped CUTLASS, F32 accumulation with per-expert `raw_scale2*a2`, BF16 output |
| Output | Raw F32 down plus ordered scale/route/shared combine | BF16 down widened F32, same ordered combine with scale1 |

The external specialization copies the actual BF16/default NVFP4 block
conversion: 16-value maximum, E4M3 RN scale, approximate-FTZ reciprocal,
and E2M1 RN/saturate conversion. It omits GGML's five-scale error search.
The previously rejected cooperative schedules are not reused. Native F16
KV, F32 recurrent state, target, drafter and serving defaults are unchanged.

## Capture and controls

`jitllm_qwen38_spec --check routed-down` is benchmark-only, default off.
It retains existing graph views through graph end, adding no arithmetic
nodes. A mask selects at most three layers and participates in the plan
cache key. It takes fast CUTLASS Verify rows1–4; the harness limits fixed
depth3, at most eight capture steps, context131072 and chunks8192.
Runner-owned bounded pinned staging stays with the device job until its
completion and retirement are proved. Unknown completion uses the runner's
existing settlement/quarantine path.

Four independent capture-off/on/on/off prefills use the same forced draft
history. All draft IDs/probabilities, full verifier logits, complete target
and drafter state, own repeats and continuations agree exactly. Twelve
real records cover layers0/23/47 and four common steps after the canonical
128799-token prompt, with the selected47172 head and fixed depth3.
Captured activation `[640,10,4]`, raw down `[2560,10,4]`, combined FFN,
IDs, route weights, shared output and gate are authenticated externally.

Replay shape is K640/N2560/E512/top10/T4. A T3 control uses each record's
first three tokens. Every original down and combined float byte matches
its captured model output in all 24 T4/T3 controls, before timing. Both
candidate down and combined outputs repeat exactly and remain finite;
256-byte allocation guards and zeroed padded scale rows pass.

A four-second no-model CUDA probe in the actual pinned image separately
exports the three reciprocal scalars and 1536 post-load alpha values.
Torch2.13.0+cu130/CUDA13.0 performs the actual F32 max, reciprocal and
in-place multiply. Native explicit RN reciprocal/multiply match every bit
before the replay can qualify. This avoids substituting Python F64 division
or an assumed CPU consumer for the actual CUDA operations.

The first replay is excluded: an asynchronous guard fill on a nonblocking
stream was unordered against a blocking default-stream upload. It retired
rc1 at the original-byte check without producing timings. The corrected
setup proves the guard fill complete before upload; measured arithmetic
is unchanged. Raw receipts preserve both runs.

## Charged operator measurements

Spark B, GB10, SDK `aarch64-e0a0c85c42806fb1`, sm121a. Each timing graph
contains all 12 real invocations and reports latency per invocation.
Cold ABBA arms use seven samples of one complete graph after a512MiB
device-cache flush outside the charged event. Warm arms are secondary:
seven samples of64 repeats of the same complete graph. Host readback,
setup and completion-proof overhead are excluded from both GPU timings.

The actual captured union is165 expert-layer pairs, touching152064000B
of down weights. The three complete512-expert arrays allocate1415577600B;
that larger allocation is not claimed as the touched working set.
Preparation, padded scale clearing, grouping, conversion, product,
widening and ordered combination are all inside the complete candidate.

| Arm, ABBA order | Cold median ms | Warm median ms |
| --- | ---: | ---: |
| Original A | 0.160480 | 0.147543 |
| Literal G2 B | 0.194101 | 0.189987 |
| Literal G2 B | 0.193960 | 0.190567 |
| Original A | 0.151960 | 0.147417 |

| Candidate phase, cold | Median ms |
| --- | ---: |
| Route, BF16 RN, quantize and padded-scale clear | 0.047445 |
| Grouped scaled BF16 product | 0.137789 |
| Widen and native ordered combine | 0.008680 |

The product alone is close to the complete original stage; preparation
adds another47.4µs. Phase timings are separate graphs and need not add
exactly to the complete graph. This negative does not establish that
Mia's fused whole consumer is slower. It identifies preparation/fusion as
an unmeasured remaining difference rather than another tile-search lead.

Combined FFN maximum absolute differences range0.005183–0.015517,
RMS0.000913–0.003315 and relative L2 0.016515–0.085024. These are operator
differences on24 inputs, not a percentage of model quality loss.
There is no full-model near-tie, PPL or distribution result for this factor.

## Provenance and repeat commands

Local source base7a31985 plus the eight-file native capture delta.
The warm Spark mirror received only owned changed files; it is not an
asserted whole-Git export. The separate license tree is an exact tracked
base export with this delta. Native source, objects, graph operands and
external scripts are pinned in the raw receipts under Spark B
`~/scratch/m3-qwen-fi-static`.

Target artifact:
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`.
Selected drafter:
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
Canonical I32 prompt SHA256:
`c9d455d1afc75e1fd4e0f93d0312f1768196609fab8a5836f57c15a72e2e3661`.
Raw calibration safetensors SHA256:
`49c7d04606ff7c8f103b4d1fc9742b44649a2937b25ecc896a5450ab1020f308`.
Only9.7MB of calibration metadata was transferred; no model or image copy.

Native capture binary SHA256:
`a4376827b4283d91e8783bdd411bf976ce0de77f5c0ab7679137611be239e472`.
Final external replay binary SHA256:
`e13a732b0c9f28ff80c53f49508740a37d655a80808e07b9f40e43e0d969a44e`.
`capture-command.json` SHA256:
`11d7ce09e39a4c5e1faadaf3efdad3ae8fc83b85c98566bf5ca0914e1e8a3417`.
`capture-selected/spec.json` SHA256:
`d5cc057dbcd2b405986717dc0dda417947581e31cf3e3590d9cd5029d7828166`.
`replay/build-receipt.json` SHA256:
`5e95e72781776daf5dda0fbebc1cd53aa5a746b7b84a7a1f726b40ec5397da94`.
`replay-summary.json` SHA256:
`77a70028ffebe2a3fd06030607d42596a7e3d42f402deff019c22b25452d2f60`.
The measured specialization source hash205c8d15… is preserved; its later
notice-only addition of NVIDIA2020–2025 changes no numerical source bytes.

Capture command, with the paths rooted at the identities above:

```sh
jitllm_qwen38_spec --qwen38-artifact TARGET --drafter SELECTED \
  --tokenizer tokenizer.json --prompts 128k.json --only qwen3.8-128k \
  --context 131072 --prefill-chunk 4096 --check routed-down --tokens 4 \
  --draft 3 --draft-vocab 47172 --adaptive-depth off \
  --runtime-prefill on --graphs on --out capture-selected
python3 replay/jitllm-fi-replay-build.py
python3 replay/jitllm-fi-replay-run.py
```

The exact absolute commands and all operand hashes live in the named
receipts. External scripts remain in scratch and are hashed there.
`qwen-fi-capture-check-v2` passes1040 tests/202GPU, SDK format8/tidy5
units,295 boundaries, seven early CLI refusals and actual REUSE/1007
headers. `qwen-fi-routed-capture` completes rc0 in299s;
`qwen-fi-scalar-consumer` completes rc0 in4s;
`qwen-fi-external-replay-build-v5` completes rc0 with SDK host tidy;
`qwen-fi-literal-down-replay-v2` completes rc0 in7s. Post-run retirement
and independent busy/memory/GPU/container/native-process gates are clear.
The final docs-only REUSE check passes with1009 headers; independent
whole-unit review is recorded in the task handoff. Workstation execution
checks remain owner-deferred.
