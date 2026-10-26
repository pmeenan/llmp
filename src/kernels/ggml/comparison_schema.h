// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_GGML_COMPARISON_SCHEMA_H_
#define LLMP_KERNELS_GGML_COMPARISON_SCHEMA_H_

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "base/sha256.h"
#include "format/encoding.h"
#include "ggml.h"
#include "kernels/ggml/llmp_ops.h"

namespace llmp::kernels::ggml {

class CanonicalHash {
 public:
  explicit CanonicalHash(std::string_view domain) { Text(domain); }

  void U8(std::uint8_t value) { Add(value); }
  void U16(std::uint16_t value) { Add(value); }
  void U32(std::uint32_t value) { Add(value); }
  void U64(std::uint64_t value) { Add(value); }
  void Text(std::string_view value) {
    U64(value.size());
    hash_.Update(value);
  }
  void Bytes(std::span<const std::byte> value) {
    U64(value.size());
    hash_.Update(value);
  }
  void Digest(const base::Sha256Digest& value) { hash_.Update(std::as_bytes(std::span(value))); }
  base::Sha256Digest Finish() { return hash_.Finish(); }

 private:
  template <class T>
  void Add(T value) {
    std::array<std::byte, sizeof(T)> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<std::byte>((value >> (i * 8U)) & 0xffU);
    hash_.Update(bytes);
  }
  base::Sha256 hash_;
};

inline bool ValidAddressRange(std::uintptr_t begin, std::uint64_t bytes) {
  std::uintptr_t end = 0;
  return begin != 0 && bytes != 0 &&
         !__builtin_add_overflow(begin, static_cast<std::uintptr_t>(bytes), &end);
}

inline std::optional<bool> IntervalsOverlap(std::uintptr_t left, std::uint64_t left_bytes,
                                            std::uintptr_t right, std::uint64_t right_bytes) {
  std::uintptr_t left_end = 0, right_end = 0;
  if (left == 0 || right == 0 || left_bytes == 0 || right_bytes == 0 ||
      __builtin_add_overflow(left, static_cast<std::uintptr_t>(left_bytes), &left_end) ||
      __builtin_add_overflow(right, static_cast<std::uintptr_t>(right_bytes), &right_end))
    return std::nullopt;
  return left < right_end && right < left_end;
}

inline bool IntervalsDisjoint(std::uintptr_t left, std::uint64_t left_bytes, std::uintptr_t right,
                              std::uint64_t right_bytes) {
  const auto overlap = IntervalsOverlap(left, left_bytes, right, right_bytes);
  return overlap.has_value() && !*overlap;
}

inline bool HalfOpenIntervalsDisjoint(std::uint64_t left_offset, std::uint64_t left_bytes,
                                      std::uint64_t right_offset, std::uint64_t right_bytes) {
  std::uint64_t left_end = 0, right_end = 0;
  return !__builtin_add_overflow(left_offset, left_bytes, &left_end) &&
         !__builtin_add_overflow(right_offset, right_bytes, &right_end) &&
         (left_end <= right_offset || right_end <= left_offset);
}

inline bool HasNoExtraSources(const ggml_tensor& tensor, std::size_t count) {
  if (count > GGML_MAX_SRC) return false;
  for (std::size_t i = count; i < GGML_MAX_SRC; ++i)
    if (tensor.src[i] != nullptr) return false;
  return true;
}

inline bool HasZeroOpParamsTail(const ggml_tensor& tensor, std::size_t first_word) {
  constexpr std::size_t kWords = GGML_MAX_OP_PARAMS / sizeof(tensor.op_params[0]);
  if (first_word > kWords) return false;
  for (std::size_t i = first_word; i < kWords; ++i)
    if (tensor.op_params[i] != 0) return false;
  return true;
}

// Only the scalar tail of these pinned native constructors is described here.
// This is a serialization schema, not permission to execute: complete-program
// comparison also requires the registered checker, actual dispatch/resources,
// and owned physical operands. Local cut/remainder eligibility is unchanged.
inline int WholeNativeCustomScalarWords(LlmpOp op) {
  switch (op) {
    case LlmpOp::kMxfp8MulMatVec:
    case LlmpOp::kBf16:
    case LlmpOp::kMoeGlu:
    case LlmpOp::kDsv4HcMix:
    case LlmpOp::kDsv4WeightedReduce:
    case LlmpOp::kDsv4F16Copy:
    case LlmpOp::kDsv4LidTopK:
    case LlmpOp::kRoundedIndexScore:
    case LlmpOp::kSelect:
    case LlmpOp::kVisibleTopK:
    case LlmpOp::kSwiGluBf16:
    case LlmpOp::kAttentionMaskPack:
      return 0;
    case LlmpOp::kGemmBf16:  // vector/wide/pinned mode
    case LlmpOp::kHcNorm:    // epsilon
    case LlmpOp::kHcMix:     // epsilon
    case LlmpOp::kMoeCombine:
    case LlmpOp::kDsv4Combine:
    case LlmpOp::kGdnHistory:
    case LlmpOp::kGdnStep:
    case LlmpOp::kHcLo:
    case LlmpOp::kPrefixOverwrite:  // literal F32 fill bits
    case LlmpOp::kSelectedMask:
      return 1;
    case LlmpOp::kMxfp8Quantize:   // K, rows
    case LlmpOp::kMxfp8Swizzle:    // N, K
    case LlmpOp::kDsv4Compress:    // ratio, overlap
    case LlmpOp::kDsv4SparseMask:  // cells, visible-count mode
    case LlmpOp::kQkAttention:     // epsilon, scale
      return 2;
    case LlmpOp::kMoeRoute:  // experts, used, tokens
    case LlmpOp::kMoeCombineSorted:
    case LlmpOp::kMxfp8Gemm:  // K, N, T
    case LlmpOp::kDsv4HcPre:  // RMS epsilon, HC epsilon, iterations
    case LlmpOp::kQsaGateQuantize:
      return 3;
    case LlmpOp::kGdnConv:      // qk channels, head, epsilon, scale
    case LlmpOp::kMoeQuantize:  // experts, used, tokens, K
    case LlmpOp::kMoeGluQuantize:
    case LlmpOp::kMoeGemv:    // row0, rows, code offset, scale offset
    case LlmpOp::kMoeRouter:  // experts, used, tokens, width
    case LlmpOp::kQsaTopK:
      return 4;
    case LlmpOp::kVecQ:       // tokens, per-slot, GLU, limit, one-token mode
    case LlmpOp::kDsv4Route:  // used, norm, clamp, scale, selection mode
    case LlmpOp::kHcMixBf16:  // HC, width, tokens, MXFP8 flag, BF16 flag
    case LlmpOp::kQsaPool:    // raw dimension, ratio, tokens, epsilon, theta
    case LlmpOp::kQsaAttn:    // heads, KV heads, tokens, row, scale
    case LlmpOp::kFlashAttnOwners:
    case LlmpOp::kRopeTable:  // base, section count, three dimensions
      return 5;
    case LlmpOp::kHcPrep:  // width, HC, tokens, combine, inject, epsilon
      return 6;
    case LlmpOp::kMoeGemm:   // experts, used, tokens, K, N, code/scale offsets
    case LlmpOp::kDsv4OutA:  // context and six RoPE scalars
    case LlmpOp::kCausalMask:
      return 7;
    case LlmpOp::kDsv4QHead:  // epsilon, context and six RoPE scalars
      return 8;
    default:
      return -1;
  }
}

inline bool HasWholeNativeCustomScalarSchema(const ggml_tensor& tensor) {
  if (tensor.op != GGML_OP_CUSTOM) return false;
  const int words = WholeNativeCustomScalarWords(LlmpOpOf(&tensor));
  if (words < 0) return false;
  for (int field = words; field < 8; ++field)
    if (LlmpOpInt(&tensor, field) != 0) return false;
  return true;
}

// Shared canonical encodings for format and pinned GGML semantic parameters.
// The caller separately requires its local or whole-native typed schema before
// accepting a descriptor; reserved bytes never become execution authority.
template <class Hash>
void HashTensorFormat(Hash& hash, const format::TensorFormat& value) {
  hash.U8(static_cast<std::uint8_t>(value.encoding));
  hash.U8(static_cast<std::uint8_t>(value.layout));
  hash.U8(value.stacked ? 1 : 0);
  hash.U8(value.shape.rank);
  for (const auto dim : value.shape.ne) hash.U64(dim);
  hash.U8(value.params.k_bits);
  hash.U8(static_cast<std::uint8_t>(value.params.codebook));
  hash.U8(value.params.tensor_scale ? 1 : 0);
  hash.U8(value.params.bits);
  hash.U32(value.params.group);
  hash.U8(static_cast<std::uint8_t>(value.params.scale));
  hash.U8(value.params.zero_point ? 1 : 0);
  hash.U32(value.params.block_rows);
  hash.U32(value.params.block_cols);
}

// Whole-only scalar schema for the native fill and bounded-index producer.
// CheckFill/CheckTopK and the actual selected schedule remain required.
inline bool HasWholeNativeFillOrTopKSchema(const ggml_tensor& tensor) {
  if (tensor.op != GGML_OP_FILL && tensor.op != GGML_OP_TOP_K) return false;
  // Lower removes FILL's unused shape-only source. TOP_K always reads src0.
  if (tensor.op == GGML_OP_TOP_K && !tensor.src[0]) return false;
  const std::size_t words = tensor.op == GGML_OP_FILL ? 1 : 0;
  return HasNoExtraSources(tensor, 1) && HasZeroOpParamsTail(tensor, words);
}

// Whole-only admission for the existing native ReLU implementation. The
// registered unary checker still validates its actual operands and storage.
inline bool HasWholeNativeReluSchema(const ggml_tensor& tensor) {
  if (tensor.op != GGML_OP_UNARY || tensor.op_params[0] != GGML_UNARY_OP_RELU || !tensor.src[0])
    return false;
  return HasNoExtraSources(tensor, 1) && HasZeroOpParamsTail(tensor, 1);
}

template <class Hash>
void HashTypedOperationParameters(Hash& hash, const ggml_tensor& tensor) {
  const auto word = [&](std::size_t index) {
    return static_cast<std::uint32_t>(tensor.op_params[index]);
  };
  switch (tensor.op) {
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:
    case GGML_OP_L2_NORM:
    case GGML_OP_FILL:
      // Norm epsilon and the fill value each occupy one f32 at op_params[0].
      hash.U32(word(0));
      break;
    case GGML_OP_PERMUTE:
      for (std::size_t i = 0; i < GGML_MAX_DIMS; ++i) hash.U32(word(i));
      break;
    case GGML_OP_ARGSORT:
    case GGML_OP_CONCAT:
      hash.U32(word(0));
      break;
    case GGML_OP_DSV4_HC_COMB:
    case GGML_OP_DSV4_HC_PRE:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_CLAMP:
    case GGML_OP_SCALE:
      hash.U32(word(0));
      hash.U32(word(1));
      break;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
      // Pinned GGML ggml_rope_impl's typed 16-word parameter record.
      for (std::size_t i = 0; i <= 15; ++i) hash.U32(word(i));
      break;
    case GGML_OP_FLASH_ATTN_EXT:
      for (std::size_t i = 0; i < 7; ++i) hash.U32(word(i));
      break;
    case GGML_OP_GATED_DELTA_NET:
      hash.U32(word(0));
      break;
    case GGML_OP_UNARY:
      hash.U32(word(0));
      break;
    case GGML_OP_MUL_MAT:
      hash.U32(word(0));
      hash.U32(word(1));
      break;
    case GGML_OP_GLU:
      hash.U32(word(0));
      hash.U32(word(1));
      break;
    case GGML_OP_CUSTOM: {
      const auto op = LlmpOpOf(&tensor);
      hash.U8(static_cast<std::uint8_t>(op));
      const int whole_words = WholeNativeCustomScalarWords(op);
      if (whole_words >= 0) {
        for (int field = 0; field < whole_words; ++field)
          hash.U32(static_cast<std::uint32_t>(LlmpOpInt(&tensor, field)));
      } else if (op == LlmpOp::kSpatialChannelNorm) {
        for (int field = 0; field < 3; ++field)
          hash.U32(static_cast<std::uint32_t>(LlmpOpInt(&tensor, field)));
      } else if (op == LlmpOp::kNormRopeSections) {
        for (int field = 0; field < 7; ++field)
          hash.U32(static_cast<std::uint32_t>(LlmpOpInt(&tensor, field)));
      } else if (op == LlmpOp::kLayerNormModulateSplit) {
        hash.U32(std::bit_cast<std::uint32_t>(LlmpOpFloat(&tensor, 0)));
        hash.U32(static_cast<std::uint32_t>(LlmpOpInt(&tensor, 1)));
        hash.U32(static_cast<std::uint32_t>(LlmpOpInt(&tensor, 2)));
      } else if (op == LlmpOp::kGatedResidualSplit) {
        hash.U32(static_cast<std::uint32_t>(LlmpOpInt(&tensor, 1)));
      } else if (op == LlmpOp::kGdnNormGate) {
        hash.U32(std::bit_cast<std::uint32_t>(LlmpOpEps(&tensor)));
        hash.U32(static_cast<std::uint32_t>(LlmpOpInt(&tensor, 1)));
      } else if (op == LlmpOp::kDsv4HcNormF16) {
        hash.U32(std::bit_cast<std::uint32_t>(LlmpOpEps(&tensor)));
      } else if (op == LlmpOp::kQsaPrep) {
        for (int field = 0; field < 4; ++field)
          hash.U32(static_cast<std::uint32_t>(LlmpOpInt(&tensor, field)));
        for (int field = 4; field < 6; ++field)
          hash.U32(std::bit_cast<std::uint32_t>(LlmpOpFloat(&tensor, field)));
      }
      break;
    }
    default:
      break;
  }
}

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_COMPARISON_SCHEMA_H_
