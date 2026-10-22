// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8's architecture mappings (spec/mapping.h), as llama.cpp's qwen4exp
// GGUF names its tensors and keys (b29c606e; conversion at b11254) and as
// the ModelOpt and MLX imports (import/normalize.h) write the same names.
//
// "qwen4exp" (Qwen3.8 Flash Next, its ModelOpt NVFP4/MXFP8 import and its
// GGUF quantizations, through one spec): hyper-connection streams
// (residual.hc; each block between hc.mix and hc.combine); three Gated
// DeltaNet layers to a QSA one (the per-layer compression ratios, nonzero
// at the QSA layers, must agree); the n-gram embedding (side.ngram, read
// by residual.ple at the layers qwen4exp.ple.layers lists) before the
// layer's first mix; every layer's MoE: a softmax top-k router over the
// stream with renormalized weights, routed SwiGLU experts and the shared
// expert behind its sigmoid gate; the head's mix (hc.mix as the final
// norm) and the untied output head.
//
// The imports' tensors differ in form, which the spec records as it is on
// disk: the native ModelOpt import's MXFP8 matrices (codes beside scales,
// bound with their parts), the n-gram table's NVFP4 rows with its tensor
// scale, the routed experts in the scale-atom layout (a stacked gate-up
// table), each projection's per-expert tensor scales, the indexer's fused
// query and key projection, and the hash's constants also as tensors
// (accounted for, unbound: the spec holds them as parameters); a GGUF's
// matrices whole of their types, the indexer's projection split, the
// experts unscaled. The form is given as llmp.qwen4exp.* values (absent: a
// GGUF checkpoint's form). The converters fold norm weights to (1 + w) (but
// linear attention's gated norm), A to -exp(A_log) and the linear
// attention's value heads into tiled order (rows of the QKV value part, Z,
// beta, alpha, dt_bias, A and the convolution's value channels, columns of
// the output projection): the mapping asserts the folds for GGUF sources,
// and the native importer records the ones it applies (attention.gdn
// refuses a spec without them).
//
// "qwen4exp-mtp": the MTP drafter stage "mtp" composed with its target
// (ComposeDrafter): it reads its target's streams after its last layer
// (the tap "streams", every row's: TapPoint::kStreams); its input
// (embedding.mtp) is its target's token table's row beside those streams;
// one QSA layer with its own caches (rewritten by its catch-up: truncated)
// and the MoE, between their mixes and combines; the head's mix returns
// the streams it mixes as its next feature; its head is its target's
// output, or its own selected head with its rows' token ids where the
// artifact has one.
//
// No default planning policy yet: the patterns that reproduce today's
// fast form's kernels are slice G3b's (the hyper-connections' prep, mix
// and combine, Gated DeltaNet's convolution, step and gated norm, QSA's
// prep, pool, top-k and sparse attention, the router, the experts' GLU and
// combine, the MXFP8 and NVFP4 products).

#ifndef LLMP_SPEC_MAPPINGS_QWEN38_H_
#define LLMP_SPEC_MAPPINGS_QWEN38_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "spec/mapping.h"

namespace llmp::spec::mappings {

const Mapping& Qwen38Mapping();
// Closed, byte-preserving GGUF target normalization.
const Mapping& Qwen38GgufMapping();
// Native source-encoded target parts. No historical converter fold assertions
// or execution-policy eligibility; normalization proves the head permutation.
const Mapping& Qwen38NativeMapping();
// Container-neutral canonical recipe for synthetic component/lowering tables.
// Runtime resolves the native variants below from actual source provenance.
const Mapping& Qwen38MtpMapping();
// The native importer's MTP drafter (v1, import/normalize.h): source
// encodings but its norms (the importer's applied norm.plus_one, its
// source's fold), the routed experts' NVFP4 parts by projection, the
// converted drafter's parameters and policy.
const Mapping& Qwen38MtpNativeMapping();

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_QWEN38_H_
