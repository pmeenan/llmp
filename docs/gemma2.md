<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 2 bounded execution

`model/gemma2.h` supplies a closed profile and checked tensor binding for the
approved Gemma 2 2B Q8_0 checkpoint. The bounded execution source adds checked
state/inputs, a native graph/plan and a bounded runner over the shared paged engine.
The approved source is authenticated and deeply imported; a representative C1
control reproduces all 33 stock full heads byte for byte. No serving route is
added. Gemma 2 remains outside the supported execution matrix.

## Actual checkpoint contract

The approved source is
[`bartowski/gemma-2-2b-it-GGUF@855f67caed130e1befc571b52bd181be2e858883`](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/tree/855f67caed130e1befc571b52bd181be2e858883),
`gemma-2-2b-it-Q8_0.gguf`, 2,784,495,456 bytes. The retained exact-revision
primary HF API LFS identity is
`2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe`;
the execution slice locally authenticated the complete file against that SHA-256
and exact length. The earlier foundation used metadata ranges only.

| Metadata or descriptor fact | Value |
| --- | ---: |
| Architecture / GGUF version | `gemma2` / 3 |
| Layers / hidden width / FFN width | 26 / 2,304 / 9,216 |
| Query / KV heads; key / value dimension | 8 / 4; 256 / 256 |
| Vocabulary / context / sliding window | 256,000 / 8,192 / 4,096 |
| RMS epsilon; attention / final logit softcaps | F32 1e-6; 50 / 30 |
| Actual tensor count | 288 |
| Tensor types | Q8_0 ×183; F32 ×105 |

The exact [d812 Gemma2 model](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/models/gemma2.cpp)
selects alternating local/global attention (even/odd layers), scales attention
queries by 1/sqrt(256) = 0.0625, and narrows the final attention/residual rows
before post-normalization and FFN. Its generic loader defaults absent RoPE
base/scaling metadata to 10,000/1. These are source-derived profile facts,
not numerical execution qualification. Unlike Gemma3, there are no Q/K norm
weights. Each layer has four F32 norm vectors and distinct Q/K/V matrices.
The output head ties to the Q8_0 token embedding; there is no `output.weight`.

## Storage and refusal contract

`BindGemma2` accepts only this exact profile, architecture and 288-resource
F32/Q8_0 role/type/shape contract. A second output role may alias the embedding
resource. Missing, repeated, empty or unused roles, independent heads, changed
profiles, other formats/shapes and insufficient readable storage are refused.
`CheckGemma2Binding` rechecks mutable public descriptors with a fixed stack
membership array, without reconstructing resources or allocating on success.
It refuses duplicate/out-of-domain indices, edited shapes/types/storage,
incomplete layers and a head that no longer aliases the embedding.

Stored bytes and readable bytes differ deliberately. Width 2,304 is a multiple
of Q8_0's 32 elements but not GGML's 512-element row padding. The existing
artifact representation contract requires 272 extra readable bytes at the
end of each affected tensor, not per row. There are 131 such tensors: the
embedding and five matrices per layer. Their source payload sizes remain
unchanged; prepared import funds and initializes the tail using the existing
artifact mechanism. Binding raw stored bytes as readable is refused.

The network-free [fixture generator](../tests/unit/data/gemma2/generate.py)
checks the retained metadata prefix and a 65,536-byte range, parses all 288
descriptors with the existing GGUF helper/native type table, checks unique names,
rank/type/block geometry, aligned nonoverlapping file ranges and the final
file boundary, and emits only aggregate metadata and tensor descriptors.
The foundation range response was exactly HTTP 206,
`bytes 6029343-6094878/2784495456`, with the checked content length.
The tensor table ends at byte 6,046,552; data starts at 6,046,560.
Raw prefixes/ranges stay outside Git. Fixture provenance includes their hashes
and explicitly distinguishes the reported whole-file identity from a local hash.

Six focused CPU controls cover actual metadata/storage, role and profile
refusals, every tensor's readable boundary, mutable descriptor identity and
alternating local/global scheduling. All six pass on spark-b alongside
the six existing Gemma3 foundation/state controls. The native ARM focused
build and tests ran under installed supervision on 2026-10-07; no model or
GPU operation ran. Whole-source checksum synchronization, an empty itemized
dry run and the checked source inventory bind the successful build. Local
fixture regeneration, format, license/header and portability checks also pass.

## Bounded execution and remaining gates

The [bounded C1 execution slice](experiments/gemma2-execution/README.md)
adds the deep-verified artifact, checked alternating local/global state and
inputs, native graph/plan and C1 runner. Its seven own controls preserve eager
and captured heads, retained Clear, refusal atomicity, initialized-state
spill/restore and device-greedy choices/state. The optimized norm/Mul, quantized
GeGLU and width-2304 norm/ADD recipe reproduces all 33 stock heads exactly at
context4096, two 128-row prompt chunks and 32 teacher-forced transitions.
All 17 focused model/graph/plan tests, 15 norm CPU checks, one width-2304 GPU
operand control and 26 shared importer controls pass on spark-b.
The [attention-softcap primitive slice](experiments/gemma2-softcap/README.md)
qualifies actual nonzero-specialization occupancy and funded vector/MMA
launches at caps 0/50/25, with FP64 operand checks, byte-exact repeats and fresh
capture replays, exact funding and refusal controls. Both new controls and
eleven focused legacy controls pass on spark-b. The opt-in
[two-owner decode slice](experiments/gemma2-owner2/README.md) now qualifies
cap50 D256/H8/GQA2 owner attention with actual true-specialization occupancy
and funding. Independent-prefill C2 controls cover small and wrapped local-ring
history, eager/capture identity, refusal atomicity, Clear and restore-next
replay. Stock comparison has zero strict choice differences, with 65/66 small
and 66/66 ring full heads byte-identical. Its short n=2 paid C2 cycle is 1.49%
slower than stock with exact natural histories/final heads; no parity claim.
Owner selection defaults off and unequal widths retain ordinary attention.
The C1 control covers final logit softcap30 and width-2304 readable tails;
compatible joint prefill, departures, broader context/quality, serving,
sustained and switch qualification remain open.
The [checkpoint/adoption foundation](experiments/gemma2-checkpoint/README.md)
adds an internal layout-bound kept-state path: ordered whole extents must fund
all initialized logical ranges, pending writes block execution and growth, and
restore completion requires proven contiguous copies of every needed range.
Selected peers remain held while the destination footprint changes. Adoption
requires an empty idle healthy slot and an identical layout identifier. All
19 focused checks pass, including actual wrapped-ring snapshot restoration and
named-file adoption in a fresh node with an exact live-peer next head; actual
serving and compatible joint prefill remain separate work.
The pin's Gemma2 HF-to-GGUF converter already adds one to norm weights;
approved GGUF import must preserve those norm payloads, with no second +1
at import or runtime.
The existing SentencePiece fixture covers tokenizer behavior. Exact Gemma2
chat-template behavior/refusals remain a separate check before serving; this
binding foundation does not select a renderer or claim template compatibility.
