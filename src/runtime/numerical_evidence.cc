// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/numerical_evidence.h"

#include <algorithm>
#include <array>
#include <string_view>

namespace llmp::runtime {
namespace {
using format::Encoding;
using ir::NumericalChange;
constexpr std::array kF32{Encoding::kF32};
constexpr std::array kBf16{Encoding::kBF16};
constexpr std::array kMxfp8{Encoding::kMxfp8};
constexpr std::array kPrepared{Encoding::kNvfp4, Encoding::kMxfp8};
constexpr std::array kNvfp4{Encoding::kNvfp4};
// The accepted reports' SHA-256 digests (each report's identity in a plan's
// attributed evidence).
constexpr std::string_view kNativeNumericalEvidence =
    "114d79515017285c44735f85c40795b2444d7c7694064e7f6f1ac8103b21b29a";
constexpr std::string_view kFinalNumericalEvidence =
    "62bcea419709532e745efe404cbe1f938ec869d1657cad0797ce42266b51f07d";
constexpr std::string_view kImageNumericalEvidence =
    "9999175c760fd4a25601978f592daa63dae329ce5efadeae50dc1b3c26a65d6a";
// The accepted qwen38-gguf source checkpoint's stored matrix inventory.
// Its layer 2 gate/up experts are IQ3_S, while the other 47 are IQ2_S;
// the report's tuning examples do not enumerate this one mixed layer.
// Artifact f60d801a's index SHA-256 is
// 40803567725558418f1993e1c50948870d92ea7022fea40151846334fbd03d70.
// Its manifest pins the same three source shards as the report. Keep the
// IQ3_S evidence in selected products, not unrelated dense products.
constexpr std::array kQwenBlock{Encoding::kQ6_K, Encoding::kQ8_0, Encoding::kIQ2_S,
                                Encoding::kIQ4_NL};
constexpr std::array kQwenExpert{Encoding::kQ6_K, Encoding::kQ8_0, Encoding::kIQ2_S,
                                 Encoding::kIQ4_NL, Encoding::kIQ3_S};
constexpr std::array kVectorRows{std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3},
                                 std::uint64_t{4}};
// Explicit inventory from accepted final artifact
// 698b1cd8696f75355d8a130b71369099c2c5d1d241932330565d33480cbc0bfd:
// index.json sha256 4f84772b3cfa5912e6a5ece00f438b9b8ac330679dd848a1896810f7d48a8e80,
// spec.json sha256 ddaaa1d38bd6a501b9ef891bc0f639a70f41b19ff85a85674be5d77470e7a8e8.
// Dense and selected-expert role inventories are deliberately separate.
constexpr std::array kFinalProduct{Encoding::kQ4_K, Encoding::kQ5_K, Encoding::kQ6_K,
                                   Encoding::kQ8_0};
constexpr std::array kFinalExpert{Encoding::kIQ2_XS, Encoding::kIQ3_XXS, Encoding::kMxfp4};
constexpr std::array kFinalRows{std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3},
                                std::uint64_t{4}, std::uint64_t{5}};
constexpr std::string_view kFinalReport = "docs/experiments/m36-final/README.md";
// Report provenance includes 1024x1024 / 25 text rows at width4096.
// Geometry is not a selection condition (D-123); transfer of these component
// arithmetic contracts is subject to U4's additional controls and image run.
constexpr std::string_view kImageReport = "docs/experiments/m36-runtime-fusion/README.md";
constexpr std::string_view kNativeReport =
    "docs/experiments/m36-numerical-audit/qwen-modelopt-v1.md";
constexpr std::string_view kVectorReport = "docs/experiments/qwen38-gguf/README.md";
constexpr std::string_view kVectorDigest =
    "8b9e1298f0e1759cfae0c3f1c792fe79052a1af02c9b9d4e93e9aca137512041";
// Device facts are explicit in every record. Nothing derives evidence from
// an architecture number or treats a missing device as the historical Spark.
constexpr std::array kEvidence{
    ComponentNumericalEvidence{.change = NumericalChange::kBf16HyperConnectionIntermediates,
                               .semantic = "hc.sigmoid_per_stream_norm",
                               .report = kNativeReport,
                               .digest = kNativeNumericalEvidence,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kF32,
                               .rows_per_owner = 4096,
                               .owners = 4,
                               .stream_rows = true},
    ComponentNumericalEvidence{.change = NumericalChange::kBf16ProductInputAndResult,
                               .semantic = "gdn.product_bf16_intermediates",
                               .report = kNativeReport,
                               .digest = kNativeNumericalEvidence,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kMxfp8,
                               .rows_per_owner = 4096,
                               .owners = 4},
    ComponentNumericalEvidence{.change = NumericalChange::kMxfp8ProductInput,
                               .semantic = "product.mxfp8_input",
                               .report = kNativeReport,
                               .digest = kNativeNumericalEvidence,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kPrepared,
                               .rows_per_owner = 4096,
                               .owners = 4},
    ComponentNumericalEvidence{.change = NumericalChange::kNvfp4ExpertIntermediates,
                               .semantic = "experts.nvfp4_intermediates",
                               .report = kNativeReport,
                               .digest = kNativeNumericalEvidence,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kNvfp4,
                               .rows_per_owner = 4096,
                               .owners = 4,
                               .row_parameter = "used"},
    ComponentNumericalEvidence{.change = NumericalChange::kQ8VectorReduction,
                               .semantic = "product.block32_q8",
                               .report = kVectorReport,
                               .digest = kVectorDigest,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kQwenBlock,
                               .rows = kVectorRows},
    ComponentNumericalEvidence{.change = NumericalChange::kQ8VectorReduction,
                               .semantic = "product.selected_block32_q8",
                               .report = kVectorReport,
                               .digest = kVectorDigest,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kQwenExpert,
                               .rows = kVectorRows},
    ComponentNumericalEvidence{.change = NumericalChange::kQ8VectorReduction,
                               .semantic = "product.block32_q8",
                               .report = kFinalReport,
                               .digest = kFinalNumericalEvidence,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kFinalProduct,
                               .rows = kFinalRows,
                               .incumbent_recipe = true},
    ComponentNumericalEvidence{.change = NumericalChange::kQ8VectorReduction,
                               .semantic = "product.selected_block32_q8",
                               .report = kFinalReport,
                               .digest = kFinalNumericalEvidence,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kFinalExpert,
                               .rows = kFinalRows,
                               .incumbent_recipe = true},
    ComponentNumericalEvidence{.change = NumericalChange::kBf16ProductInputAndResult,
                               .semantic = "product.bf16_input_and_result",
                               .report = kImageReport,
                               .digest = kImageNumericalEvidence,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kBf16,
                               .pipeline = NumericalPipeline::kImage},
    ComponentNumericalEvidence{.change = NumericalChange::kBf16SwiGluIntermediates,
                               .semantic = "glu.silu_bf16_intermediates",
                               .report = kImageReport,
                               .digest = kImageNumericalEvidence,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kF32,
                               .pipeline = NumericalPipeline::kImage},
    ComponentNumericalEvidence{.change = NumericalChange::kBf16AttentionIntermediates,
                               .semantic = "attention.stateless_bf16_intermediates",
                               .report = kImageReport,
                               .digest = kImageNumericalEvidence,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kF32,
                               .pipeline = NumericalPipeline::kImage,
                               .attention_head_dim = 128},
    ComponentNumericalEvidence{.change = NumericalChange::kBf16ResidualCarry,
                               .semantic = "residual.bf16_carry",
                               .report = kImageReport,
                               .digest = kImageNumericalEvidence,
                               .device_class = "sm_121",
                               .device_model = "NVIDIA GB10",
                               .arch = 121,
                               .multiprocessors = 48,
                               .encodings = kF32,
                               .pipeline = NumericalPipeline::kImage},
};
}  // namespace

std::span<const ComponentNumericalEvidence> ComponentNumericalEvidenceTable() { return kEvidence; }
bool HasImageAttentionEvidence(const planner::DeviceFacts& device, std::uint64_t head_dim) {
  return std::ranges::any_of(kEvidence, [&](const auto& evidence) {
    return evidence.pipeline == NumericalPipeline::kImage &&
           evidence.change == NumericalChange::kBf16AttentionIntermediates &&
           EvidenceDeviceMatches(evidence, device) && evidence.attention_head_dim == head_dim;
  });
}
bool EvidenceDeviceMatches(const ComponentNumericalEvidence& evidence,
                           const planner::DeviceFacts& device) {
  return !evidence.device_class.empty() && !evidence.device_model.empty() && evidence.arch != 0 &&
         evidence.multiprocessors != 0 && evidence.device_class == device.device_class &&
         evidence.device_model == device.device_model && evidence.arch == device.arch &&
         evidence.multiprocessors == device.multiprocessors;
}
}  // namespace llmp::runtime
