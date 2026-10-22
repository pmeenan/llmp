<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen27 GDN output-gate diagnosis

After the MLX normalization repair, artifact
`65834c669c64ce29da12260ca5317c9c130707b19eca9c6a956ad5a9c45acfd1` still failed
the original bounded native smoke: the literal capital prompt repeated digits,
and ordinary chat returned empty content with digits and whitespace in reasoning.
Both stopped at the unchanged 64-token limit. The normalization audit proved all
209 relevant vectors matched their source. That negative artifact and receipt
remain preserved; this is a separate execution-semantics defect, not a change to
any quality threshold, prompt, reference pin or precision.

The pinned TensorFold source is `cb2ebf0540f42604e2759b2ddef497861e928248`.
Its Qwen3.5-family CUDA
[`glue.py::_gated_norm`](https://github.com/pmeenan/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/families/qwen3_5/cuda/glue.py)
computes `rmsnorm(y) * weight * z * sigmoid(z)`: a SiLU output gate. The source
prompt and decode paths use this convention. The Qwen4 experimental family instead
uses `rmsnorm(y) * weight * sigmoid(z)` in both
[`gdn.cu`](https://github.com/pmeenan/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/families/qwen4_exp/cuda/gdn.cu)
and `gdn_io.cu`. These contracts differ even at zero: sigmoid gives 0.5 while
SiLU gives zero; at -1 they have opposite signs. Replacing sigmoid globally would
break the Qwen4 contract.

The native `attention.gdn` component, its CPU reference, and the fused output-norm
kernels previously all used sigmoid. Agreement among those implementations could
therefore miss the source mismatch. The repair adds a declarative `output_gate`
choice, `sigmoid` (the existing default) or `silu`. The Qwen3.5 mapping explicitly
selects SiLU; Qwen4 keeps sigmoid. There is no architecture branch in execution.
The choice travels through primitive IR, dedicated fused parameters, decomposition,
capability checks, lowering and the existing F32, BF16 and MXFP8 output kernels.
Unknown selectors are refused.

The existing GDN Q8, BF16 and MXFP8 numerical-site declarations are conditioned on
`output_gate=sigmoid`. Their evidence is not silently granted to SiLU. Supporting
both activations in a kernel does not authorize an arithmetic-changing rewrite;
those rewrites still need applicable evidence. The F32 fused SiLU path implements
the declared primitive formula without changing the model's precision mode.

A separate head-order audit found the existing permutation consistent with the
pinned source: source value head `g*(V/K)+j` reads key head `g`, while native value
head `j*K+g` reads `head % K`. The importer permutes the affected projection rows,
convolution channels, per-head scalars and output columns together. This reasoning
establishes no additional packing defect. A sampled affine payload/decode audit
remains separate work; this report makes no claim that the gate is the final cause
of the failed smoke.

Regression coverage includes the two declared component formulas, mapping
selection, numerical-site separation, signed/zero/large gate values against an
independent formula, fused/decomposed agreement, planner selection, selector
preservation in lowering, invalid-selector refusal, and both output activations
in F32/BF16/MXFP8 kernel tests. The native runtime/importer and seven affected
test targets built on Spark B, and all 314 focused cases passed without skips.
Independent whole-change and adversarial review was clean. The final combined
native build and all 3,466 unique shared checks passed on Spark B without skips.
Six affected real-model controls passed on Spark A: the MTP ladder comparison,
full-logit/default token schedules, plain/frozen-head GGUF state/spill, and owned
joined-MTP restoration/replay. They preserve the existing sigmoid path; no
Qwen27 reference-quality result is inferred from these controls.

## Corrected artifact and unchanged smoke

The approved checkpoint remains
`TensorFold/Qwen3.8-27B-MLX-4bit@70ae7fac63274ff2eac54152031433374cb80f2f`.
Official SSD staging on Spark A verified 17 files totaling 16,081,520,742 bytes;
its archive manifest SHA-256 is
`e64f621c146c3b14ad7023eab05a476acaa663f1d9ba97a57a7768986d4d6a66`.
The corrected import published artifact
`bbf67ee688746792a1fbb06552b59bc0132e74bc79ced2333302b431549d0fa5`
in 227.876 seconds under the exclusive supervisor. The writer's publication
path passed deep artifact validation, stored-spec equality and import-time
`ValidateSpec`. The old artifacts remain preserved.

| Published metadata | SHA-256 |
| --- | --- |
| Manifest | `0e0e325360f02b4a678eab55c4cb9847538a849695f5c1c92c9902369f56b1bd` |
| Index | `0170880976d804a3d19f543bbca78d7ddb4edc3e47100926f76a4e231bcba219` |
| Stored spec | `3930c5bc8c74c5ddb2a3b043e1011c6dc15d82bb04e87e3e359d235dc8a3ec2a` |

The index is byte-identical to norm-corrected artifact `65834c66…`. All four
shard manifest entries, including their hashes and combined 15,133,126,656-byte
payload, match that artifact too. Deep publication validation authenticates
those shard hashes. Its previously audited 209 source-equal normalization
vectors are therefore retained; the gate changes stored execution semantics,
not their bytes.

The same original five literal IDs and ordinary chat request were submitted
with the unchanged 64-token limit, one slot, context 12,288, prefill 512, F16 KV,
temperature zero and top-p one. Both returned HTTP 200 and 64 tokens with
`finish_reason=length`. The literal completion starts with Paris and continues
coherently through other European capitals; all 64 reported token log
probabilities are finite. Chat contains coherent reasoning that identifies
Paris and the requested answer format, but its final content is empty. The
screen no longer reproduces the digit corruption; the truncated ordinary chat
reply remains an incomplete behavioral control. Neither HTTP success nor a
coherent beginning establishes reference quality, full-answer delivery or
serving qualification. No token bound was widened.

Readiness took 1.806 seconds. The supervised smoke ended in 15.468 seconds and the
service retired on TERM with return code zero. A preceding smoke-wrapper
attempt selected the wrong executable basename and failed its hash assertion
before service launch; its empty namespace is preserved. The corrected attempt
used a fresh namespace, with no request or model-setting change.

The frozen importer SHA-256 is
`45d4329d9f26d11ad58c4a9cb251cf1abba44a24aae64882f71caf9c782c2d0c`;
runtime SHA-256
`078a8da5a78242477891757b88675ed685b4ebe6d687db634c8c116cef1f3181`.
Both were built from base
`2683b9070da87d759ccd2b6f3c507f0a5a0287c4` plus source patch
`7a519a525d900ceb34f6ac81d27e6467ef15c649d7712113bddf151b62220d62`;
the remote Git checkout does not identify the compiled source. The native
configuration SHA-256 is
`082c042e4088fc7e50c499c57be486da1dc48caa1f1b0f8266bc0c7d69176750`.
Receipts, responses and logs remain outside Git under
`spark:~/.local/share/llmp/m35-u3-quality/qwen27-gdn-{reimport-r1,smoke-r2}`,
with supervised jobs `u3-qwen27-gdn-reimport-r1` and
`u3-qwen27-gdn-smoke-r2`.

No new Qwen reference-quality comparison has been collected. A separately
reviewed prospective protocol must freeze its source reference, inputs,
precision, actual-target likelihood contract and acceptance bounds first.
