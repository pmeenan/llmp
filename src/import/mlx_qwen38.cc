// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/mlx_qwen38.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/step.h"
#include "import/parameter_folds.h"
#include "spec/mappings/qwen38.h"

namespace llmp::import {
namespace {
using format::Encoding;
using format::Layout;
using format::PartRole;
using Failure = std::unexpected<Error>;
Failure Bad(std::string_view why, Rule rule = Rule::kConfig) {
  return Failure(Error{.rule = rule, .reason = why});
}
enum class Domain : std::uint8_t { kNone, kLayer, kLinear, kFull, kPle, kShard };
enum class HeadRecipe : std::uint8_t { kNone, kScalar, kScalarRows, kConv, kQkv, kValues, kOutput };
struct SourceRule {
  std::string_view pattern, dtype;
  std::array<std::uint64_t, 5> dims;
  std::size_t rank;
  Domain first, second;
  std::string_view canonical;
  PartAssemblyKind assembly;
  HeadRecipe heads;
  PartRole part;
  std::uint32_t count;
};

// Closed source roles: converter semantics for Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80.
// Routed target consists of 3,335 members across 1,212 outputs -> 2,954 canonical tensors.
constexpr std::array kRules = {
    // Globals (5 outputs, 13 members).
    SourceRule{"language_model.lm_head.weight",
               "U32",
               {248320, 320, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "output.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               1},
    SourceRule{"language_model.lm_head.scales",
               "BF16",
               {248320, 80, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "output.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               1},
    SourceRule{"language_model.lm_head.biases",
               "BF16",
               {248320, 80, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "output.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               1},
    SourceRule{"language_model.model.embed_tokens.weight",
               "U32",
               {248320, 320, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "token_embd.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               1},
    SourceRule{"language_model.model.embed_tokens.scales",
               "BF16",
               {248320, 80, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "token_embd.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               1},
    SourceRule{"language_model.model.embed_tokens.biases",
               "BF16",
               {248320, 80, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "token_embd.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               1},
    SourceRule{"language_model.model.hyper_connection_mixer.hc_norm.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kNone,
               Domain::kNone,
               "output_hc_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"language_model.model.hyper_connection_mixer.input_mix_weight_down.weight",
               "U32",
               {320, 1280, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "output_hc_down.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               1},
    SourceRule{"language_model.model.hyper_connection_mixer.input_mix_weight_down.scales",
               "BF16",
               {320, 320, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "output_hc_down.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               1},
    SourceRule{"language_model.model.hyper_connection_mixer.input_mix_weight_down.biases",
               "BF16",
               {320, 320, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "output_hc_down.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               1},
    SourceRule{"language_model.model.hyper_connection_mixer.input_mix_weight_up.weight",
               "U32",
               {10240, 40, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "output_hc_up.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               1},
    SourceRule{"language_model.model.hyper_connection_mixer.input_mix_weight_up.scales",
               "BF16",
               {10240, 10, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "output_hc_up.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               1},
    SourceRule{"language_model.model.hyper_connection_mixer.input_mix_weight_up.biases",
               "BF16",
               {10240, 10, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               "output_hc_up.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               1},

    // Layer 1 PLE dense (6 outputs, 10 members).
    SourceRule{"language_model.model.layers.{index}.ple.conv1d.weight",
               "BF16",
               {10240, 4, 1, 0, 0},
               3,
               Domain::kPle,
               Domain::kNone,
               "blk.{layer}.ple_conv1d.weight",
               PartAssemblyKind::kDenseReshape,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"language_model.model.layers.{index}.ple.key_proj.weight",
               "U32",
               {10240, 320, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kNone,
               "blk.{layer}.ple_key.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               1},
    SourceRule{"language_model.model.layers.{index}.ple.key_proj.scales",
               "BF16",
               {10240, 80, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kNone,
               "blk.{layer}.ple_key.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               1},
    SourceRule{"language_model.model.layers.{index}.ple.key_proj.biases",
               "BF16",
               {10240, 80, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kNone,
               "blk.{layer}.ple_key.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               1},
    SourceRule{"language_model.model.layers.{index}.ple.norm_conv.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kPle,
               Domain::kNone,
               "blk.{layer}.ple_norm_conv.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"language_model.model.layers.{index}.ple.norm_key.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kPle,
               Domain::kNone,
               "blk.{layer}.ple_norm_key.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"language_model.model.layers.{index}.ple.norm_query.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kPle,
               Domain::kNone,
               "blk.{layer}.ple_norm_query.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"language_model.model.layers.{index}.ple.value_proj.weight",
               "U32",
               {2560, 320, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kNone,
               "blk.{layer}.ple_value.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               1},
    SourceRule{"language_model.model.layers.{index}.ple.value_proj.scales",
               "BF16",
               {2560, 80, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kNone,
               "blk.{layer}.ple_value.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               1},
    SourceRule{"language_model.model.layers.{index}.ple.value_proj.biases",
               "BF16",
               {2560, 80, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kNone,
               "blk.{layer}.ple_value.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               1},

    // Layer 1 PLE shards (1 output, 384 members).
    SourceRule{"language_model.model.layers.{index}.ple.ple_embedding.ngram_embedding.shard_{index}"
               ".weight",
               "U32",
               {2500012, 20, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kShard,
               "per_layer_token_embd.weight",
               PartAssemblyKind::kRows,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               128},
    SourceRule{"language_model.model.layers.{index}.ple.ple_embedding.ngram_embedding.shard_{index}"
               ".scales",
               "BF16",
               {2500012, 5, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kShard,
               "per_layer_token_embd.weight",
               PartAssemblyKind::kRows,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               128},
    SourceRule{"language_model.model.layers.{index}.ple.ple_embedding.ngram_embedding.shard_{index}"
               ".biases",
               "BF16",
               {2500012, 5, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kShard,
               "per_layer_token_embd.weight",
               PartAssemblyKind::kRows,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               128},

    // 48 layers: Hyper-connections (8 outputs, 960 members).
    SourceRule{
        "language_model.model.layers.{index}.attn_hyper_connection.block_inject_weight.weight",
        "U32",
        {4, 1280, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_attn_inject.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kCodes,
        48},
    SourceRule{
        "language_model.model.layers.{index}.attn_hyper_connection.block_inject_weight.scales",
        "BF16",
        {4, 320, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_attn_inject.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBlockScales,
        48},
    SourceRule{
        "language_model.model.layers.{index}.attn_hyper_connection.block_inject_weight.biases",
        "BF16",
        {4, 320, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_attn_inject.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBiases,
        48},
    SourceRule{"language_model.model.layers.{index}.attn_hyper_connection.hc_norm.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.hc_attn_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               48},
    SourceRule{
        "language_model.model.layers.{index}.attn_hyper_connection.input_mix_weight_down.weight",
        "U32",
        {320, 1280, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_attn_down.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kCodes,
        48},
    SourceRule{
        "language_model.model.layers.{index}.attn_hyper_connection.input_mix_weight_down.scales",
        "BF16",
        {320, 320, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_attn_down.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBlockScales,
        48},
    SourceRule{
        "language_model.model.layers.{index}.attn_hyper_connection.input_mix_weight_down.biases",
        "BF16",
        {320, 320, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_attn_down.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBiases,
        48},
    SourceRule{
        "language_model.model.layers.{index}.attn_hyper_connection.input_mix_weight_up.weight",
        "U32",
        {10240, 40, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_attn_up.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kCodes,
        48},
    SourceRule{
        "language_model.model.layers.{index}.attn_hyper_connection.input_mix_weight_up.scales",
        "BF16",
        {10240, 10, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_attn_up.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBlockScales,
        48},
    SourceRule{
        "language_model.model.layers.{index}.attn_hyper_connection.input_mix_weight_up.biases",
        "BF16",
        {10240, 10, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_attn_up.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBiases,
        48},
    SourceRule{
        "language_model.model.layers.{index}.mlp_hyper_connection.block_inject_weight.weight",
        "U32",
        {4, 1280, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_ffn_inject.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kCodes,
        48},
    SourceRule{
        "language_model.model.layers.{index}.mlp_hyper_connection.block_inject_weight.scales",
        "BF16",
        {4, 320, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_ffn_inject.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBlockScales,
        48},
    SourceRule{
        "language_model.model.layers.{index}.mlp_hyper_connection.block_inject_weight.biases",
        "BF16",
        {4, 320, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_ffn_inject.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBiases,
        48},
    SourceRule{"language_model.model.layers.{index}.mlp_hyper_connection.hc_norm.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.hc_ffn_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               48},
    SourceRule{
        "language_model.model.layers.{index}.mlp_hyper_connection.input_mix_weight_down.weight",
        "U32",
        {320, 1280, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_ffn_down.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kCodes,
        48},
    SourceRule{
        "language_model.model.layers.{index}.mlp_hyper_connection.input_mix_weight_down.scales",
        "BF16",
        {320, 320, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_ffn_down.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBlockScales,
        48},
    SourceRule{
        "language_model.model.layers.{index}.mlp_hyper_connection.input_mix_weight_down.biases",
        "BF16",
        {320, 320, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_ffn_down.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBiases,
        48},
    SourceRule{
        "language_model.model.layers.{index}.mlp_hyper_connection.input_mix_weight_up.weight",
        "U32",
        {10240, 40, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_ffn_up.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kCodes,
        48},
    SourceRule{
        "language_model.model.layers.{index}.mlp_hyper_connection.input_mix_weight_up.scales",
        "BF16",
        {10240, 10, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_ffn_up.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBlockScales,
        48},
    SourceRule{
        "language_model.model.layers.{index}.mlp_hyper_connection.input_mix_weight_up.biases",
        "BF16",
        {10240, 10, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        "blk.{layer}.hc_ffn_up.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kBiases,
        48},

    // 48 layers: Routers (2 outputs, 192 members).
    SourceRule{"language_model.model.layers.{index}.mlp.gate.weight",
               "BF16",
               {512, 2560, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_gate_inp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert_gate.weight",
               "U32",
               {1, 320, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_gate_inp_shexp.weight",
               PartAssemblyKind::kDenseReshape,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert_gate.scales",
               "BF16",
               {1, 80, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_gate_inp_shexp.weight",
               PartAssemblyKind::kDenseReshape,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert_gate.biases",
               "BF16",
               {1, 80, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_gate_inp_shexp.weight",
               PartAssemblyKind::kDenseReshape,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               48},

    // 48 layers: Shared expert (3 outputs, 432 members).
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert.gate_proj.weight",
               "U32",
               {640, 320, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_gate_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert.gate_proj.scales",
               "BF16",
               {640, 80, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_gate_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert.gate_proj.biases",
               "BF16",
               {640, 80, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_gate_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert.up_proj.weight",
               "U32",
               {640, 320, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_up_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert.up_proj.scales",
               "BF16",
               {640, 80, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_up_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert.up_proj.biases",
               "BF16",
               {640, 80, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_up_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert.down_proj.weight",
               "U32",
               {2560, 80, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_down_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert.down_proj.scales",
               "BF16",
               {2560, 20, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_down_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.shared_expert.down_proj.biases",
               "BF16",
               {2560, 20, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_down_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               48},

    // 48 layers: Routed experts (3 outputs, 432 members).
    SourceRule{"language_model.model.layers.{index}.mlp.switch_mlp.gate_proj.weight",
               "U32",
               {512, 640, 320, 0, 0},
               3,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_gate_exps.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.switch_mlp.gate_proj.scales",
               "BF16",
               {512, 640, 80, 0, 0},
               3,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_gate_exps.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.switch_mlp.gate_proj.biases",
               "BF16",
               {512, 640, 80, 0, 0},
               3,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_gate_exps.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.switch_mlp.up_proj.weight",
               "U32",
               {512, 640, 320, 0, 0},
               3,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_up_exps.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.switch_mlp.up_proj.scales",
               "BF16",
               {512, 640, 80, 0, 0},
               3,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_up_exps.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.switch_mlp.up_proj.biases",
               "BF16",
               {512, 640, 80, 0, 0},
               3,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_up_exps.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.switch_mlp.down_proj.weight",
               "U32",
               {512, 2560, 80, 0, 0},
               3,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_down_exps.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.switch_mlp.down_proj.scales",
               "BF16",
               {512, 2560, 20, 0, 0},
               3,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_down_exps.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               48},
    SourceRule{"language_model.model.layers.{index}.mlp.switch_mlp.down_proj.biases",
               "BF16",
               {512, 2560, 20, 0, 0},
               3,
               Domain::kLayer,
               Domain::kNone,
               "blk.{layer}.ffn_down_exps.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               48},

    // 36 GDN linear attention layers (9 outputs, 684 members).
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_qkv.weight",
               "U32",
               {10240, 320, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.attn_qkv.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kQkv,
               format::PartRole::kCodes,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_qkv.scales",
               "BF16",
               {10240, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.attn_qkv.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kQkv,
               format::PartRole::kBlockScales,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_qkv.biases",
               "BF16",
               {10240, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.attn_qkv.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kQkv,
               format::PartRole::kBiases,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_z.weight",
               "U32",
               {6144, 320, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.attn_gate.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kValues,
               format::PartRole::kCodes,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_z.scales",
               "BF16",
               {6144, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.attn_gate.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kValues,
               format::PartRole::kBlockScales,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_z.biases",
               "BF16",
               {6144, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.attn_gate.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kValues,
               format::PartRole::kBiases,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_b.weight",
               "U32",
               {48, 320, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_beta.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalarRows,
               format::PartRole::kCodes,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_b.scales",
               "BF16",
               {48, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_beta.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalarRows,
               format::PartRole::kBlockScales,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_b.biases",
               "BF16",
               {48, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_beta.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalarRows,
               format::PartRole::kBiases,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_a.weight",
               "U32",
               {48, 320, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_alpha.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalarRows,
               format::PartRole::kCodes,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_a.scales",
               "BF16",
               {48, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_alpha.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalarRows,
               format::PartRole::kBlockScales,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.in_proj_a.biases",
               "BF16",
               {48, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_alpha.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalarRows,
               format::PartRole::kBiases,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.out_proj.weight",
               "U32",
               {2560, 768, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_out.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kOutput,
               format::PartRole::kCodes,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.out_proj.scales",
               "BF16",
               {2560, 192, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_out.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kOutput,
               format::PartRole::kBlockScales,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.out_proj.biases",
               "BF16",
               {2560, 192, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_out.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kOutput,
               format::PartRole::kBiases,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.conv1d.weight",
               "BF16",
               {10240, 4, 1, 0, 0},
               3,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_conv1d.weight",
               PartAssemblyKind::kDenseReshape,
               HeadRecipe::kConv,
               format::PartRole::kData,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.dt_bias",
               "BF16",
               {48, 0, 0, 0, 0},
               1,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_dt.bias",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalar,
               format::PartRole::kData,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.A_log",
               "BF16",
               {48, 0, 0, 0, 0},
               1,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_a",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalar,
               format::PartRole::kData,
               36},
    SourceRule{"language_model.model.layers.{index}.linear_attn.norm.weight",
               "BF16",
               {128, 0, 0, 0, 0},
               1,
               Domain::kLinear,
               Domain::kNone,
               "blk.{layer}.ssm_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               36},

    // 12 QSA full attention layers (9 outputs, 228 members).
    SourceRule{"language_model.model.layers.{index}.self_attn.q_proj.weight",
               "U32",
               {12288, 320, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_q.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.q_proj.scales",
               "BF16",
               {12288, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_q.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.q_proj.biases",
               "BF16",
               {12288, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_q.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.k_proj.weight",
               "U32",
               {512, 320, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_k.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.k_proj.scales",
               "BF16",
               {512, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_k.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.k_proj.biases",
               "BF16",
               {512, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_k.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.v_proj.weight",
               "U32",
               {512, 320, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_v.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.v_proj.scales",
               "BF16",
               {512, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_v.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.v_proj.biases",
               "BF16",
               {512, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_v.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.o_proj.weight",
               "U32",
               {2560, 768, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_output.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.o_proj.scales",
               "BF16",
               {2560, 192, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_output.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.o_proj.biases",
               "BF16",
               {2560, 192, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_output.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.indexer.index_qk_proj.weight",
               "U32",
               {640, 320, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.indexer.qk_proj.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.indexer.index_qk_proj.scales",
               "BF16",
               {640, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.indexer.qk_proj.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.indexer.index_qk_proj.biases",
               "BF16",
               {640, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.indexer.qk_proj.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBiases,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.q_norm.weight",
               "BF16",
               {256, 0, 0, 0, 0},
               1,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_q_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.k_norm.weight",
               "BF16",
               {256, 0, 0, 0, 0},
               1,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.attn_k_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.indexer.q_layernorm.weight",
               "BF16",
               {128, 0, 0, 0, 0},
               1,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.indexer.q_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               12},
    SourceRule{"language_model.model.layers.{index}.self_attn.indexer.k_layernorm.weight",
               "BF16",
               {128, 0, 0, 0, 0},
               1,
               Domain::kFull,
               Domain::kNone,
               "blk.{layer}.indexer.k_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               12},
};

bool InDomain(Domain domain, std::uint32_t n) {
  switch (domain) {
    case Domain::kNone:
      return false;
    case Domain::kLayer:
      return n < 48;
    case Domain::kLinear:
      return n < 48 && (n % 4 != 3);
    case Domain::kFull:
      return n < 48 && (n % 4 == 3);
    case Domain::kPle:
      return n == 1;
    case Domain::kShard:
      return n < 128;
  }
  return false;
}

std::optional<std::array<std::uint32_t, 2>> Match(const SourceRule& rule, std::string_view name) {
  auto pattern = rule.pattern;
  std::array<std::uint32_t, 2> indices{};
  std::size_t field = 0;
  while (true) {
    const auto at = pattern.find("{index}");
    if (at == std::string_view::npos)
      return name == pattern ? std::optional(indices) : std::nullopt;
    if (field == indices.size() || !name.starts_with(pattern.substr(0, at))) return std::nullopt;
    name.remove_prefix(at);
    const auto end = name.find_first_not_of("0123456789");
    const auto digits = name.substr(0, end);
    if (digits.empty() || digits.size() > 3 || (digits.size() > 1 && digits[0] == '0'))
      return std::nullopt;
    auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), indices[field]);
    if (parsed.ec != std::errc{} ||
        !InDomain(field == 0 ? rule.first : rule.second, indices[field]))
      return std::nullopt;
    ++field;
    name.remove_prefix(digits.size());
    pattern.remove_prefix(at + 7);
  }
}

std::string Canonical(std::string_view pattern, std::uint32_t layer) {
  const auto at = pattern.find("{layer}");
  if (at == std::string_view::npos) return std::string(pattern);
  return std::string(pattern.substr(0, at)) + std::to_string(layer) +
         std::string(pattern.substr(at + 7));
}

std::string PartName(std::string_view name, PartRole part) {
  if (part == PartRole::kData || part == PartRole::kCodes) return std::string(name);
  return std::string(name.substr(0, name.size() - 7)) +
         (part == PartRole::kBlockScales ? ".weight_scale"
          : part == PartRole::kBiases    ? ".weight_bias"
                                         : ".weight_scale_2");
}

format::TensorFormat SourceFormat(const SourceRule& r) {
  format::TensorFormat f{};
  f.shape.rank = static_cast<std::uint8_t>(r.rank);
  for (std::size_t i = 0; i < r.rank; ++i) f.shape.ne[i] = r.dims[r.rank - 1 - i];
  if (r.part == PartRole::kData) {
    f.encoding = Encoding::kBF16;
    f.layout = Layout::kDense;
  } else {
    f.layout = Layout::kSplit;
    f.encoding = Encoding::kAffine;
    f.params = {.tensor_scale = false,
                .bits = 4,
                .group = 32,
                .scale = format::AffineScale::kBF16,
                .zero_point = false};
    std::uint64_t k = 0, n = 0;
    if (r.rank == 3) {
      k = r.dims[2] * (r.part == PartRole::kCodes ? 8U : 32U);
      n = r.dims[1];
      f.shape = {.ne = {k, n, r.dims[0], 1}, .rank = 3};
    } else {
      k = r.dims[1] * (r.part == PartRole::kCodes ? 8U : 32U);
      n = r.dims[0];
      f.shape = {.ne = {k, n, 1, 1}, .rank = 2};
    }
  }
  return f;
}

format::TensorFormat OutputFormat(const SourceRule& r) {
  auto f = SourceFormat(r);
  if (r.assembly == PartAssemblyKind::kRows) {
    f.shape.ne[1] *= 128;
  }
  if (r.assembly == PartAssemblyKind::kDenseReshape) {
    if (f.shape.rank == 3)
      f.shape = {.ne = {f.shape.ne[1], f.shape.ne[2], 1, 1}, .rank = 2};
    else
      f.shape = {.ne = {f.shape.ne[0], 1, 1, 1}, .rank = 1};
  }
  return f;
}

std::expected<std::optional<HeadPermutationPlan>, Error> Permutation(
    const SourceRule& r, const format::TensorFormat& f) {
  if (r.heads == HeadRecipe::kNone) return std::nullopt;
  const bool columns = r.heads == HeadRecipe::kScalar || r.heads == HeadRecipe::kOutput;
  const auto head =
      r.heads == HeadRecipe::kScalar || r.heads == HeadRecipe::kScalarRows ? 1U : 128U;
  const auto first = r.heads == HeadRecipe::kQkv || r.heads == HeadRecipe::kConv ? 4096U : 0U;
  auto p = HeadPermutationPlan::Make(f, columns ? HeadAxis::kColumns : HeadAxis::kRows, 16, 48,
                                     head, first);
  if (!p) return std::unexpected(p.error());
  return *p;
}

const MetaValue* Meta(const Checkpoint& source, std::string_view key) {
  if (!key.starts_with("text_config.")) {
    std::string prefixed = "text_config." + std::string(key);
    if (const auto* v = source.FindValue(prefixed)) return v;
  }
  return source.FindValue(key);
}

bool Int(const Checkpoint& s, std::string_view k, std::int64_t expected) {
  const auto* v = Meta(s, k);
  const auto* x = v ? std::get_if<std::int64_t>(v) : nullptr;
  return x && *x == expected;
}
bool Real(const Checkpoint& s, std::string_view k, double expected) {
  const auto* v = Meta(s, k);
  if (!v) return false;
  if (const auto* x = std::get_if<double>(v)) return *x == expected;
  if (const auto* x = std::get_if<std::int64_t>(v)) return static_cast<double>(*x) == expected;
  return false;
}
bool Ints(const Checkpoint& s, std::string_view k, std::span<const std::int64_t> expected) {
  const auto* v = Meta(s, k);
  const auto* x = v ? std::get_if<std::vector<std::int64_t>>(v) : nullptr;
  return x && std::ranges::equal(*x, expected);
}

bool Config(const Checkpoint& s) {
  constexpr std::pair<std::string_view, std::int64_t> ints[] = {
      {"vocab_size", 248320},
      {"hidden_size", 2560},
      {"num_hidden_layers", 48},
      {"num_attention_heads", 24},
      {"num_key_value_heads", 2},
      {"head_dim", 256},
      {"max_position_embeddings", 262144},
      {"linear_conv_kernel_dim", 4},
      {"linear_key_head_dim", 128},
      {"linear_value_head_dim", 128},
      {"linear_num_key_heads", 16},
      {"linear_num_value_heads", 48},
      {"moe_intermediate_size", 640},
      {"shared_expert_intermediate_size", 640},
      {"num_experts_per_tok", 10},
      {"num_experts", 512},
      {"hc_count", 4},
      {"hc_lowrank", 320},
      {"ple_embed_dim", 2560},
      {"ple_conv_kernel_size", 4},
      {"ngram_size", 3},
      {"heads_per_ngram", 8},
      {"ngram_vocab_size_base", 20000000},
      {"make_ngram_vocab_size_divisible_by", 128},
      {"split_ngram_parts", 128},
      {"indexer_n_heads", 4},
      {"indexer_kv_heads", 1},
      {"indexer_head_dim", 128},
      {"indexer_budget", 2048},
      {"indexer_compress_ratio", 4},
      {"eos_token_id", 248044},
      {"bos_token_id", 248044},
      {"mtp_num_hidden_layers", 1}};
  for (const auto& [key, value] : ints)
    if (!Int(s, key, value)) return false;
  constexpr std::array<std::int64_t, 1> ple = {2};
  constexpr std::array<std::int64_t, 3> sections = {11, 11, 10};
  if (!Ints(s, "ple_layer_ids", ple) || !Ints(s, "rope_parameters.mrope_section", sections) ||
      !Real(s, "rope_parameters.rope_theta", 10000000) || !Real(s, "rms_norm_eps", 0.000001))
    return false;
  const auto* layers = Meta(s, "layer_types");
  const auto* list = layers ? std::get_if<std::vector<std::string>>(layers) : nullptr;
  if (!list || list->size() != 48) return false;
  for (std::size_t i = 0; i < 48; ++i)
    if ((*list)[i] != (i % 4 == 3 ? "full_attention" : "linear_attention")) return false;
  return true;
}

constexpr std::uint64_t kPhysicalTensors = 3747, kTargetParts = 3335;
constexpr std::uint64_t kOutputs = 1212, kCanonical = 2954, kComponents = 512;

NormalizationCost Cost() {
  const auto V = [](std::uint64_t n, std::uint64_t z) { return 3 * n * z + 64; };
  const auto Str = [](std::uint64_t n) { return n * 1024; };
  const auto Tree = [](std::uint64_t n) { return n * 256; };
  std::uint64_t bytes = 0;
  bytes += 3 * V(kCanonical, 256) + 2 * Str(kCanonical) + 4 * V(kCanonical, 8);
  bytes += 2 * V(512, 256) + V(64, 256) + 2 * Str(64) + 68 * V(48, 8);
  bytes += 2 * Tree(kCanonical) + Str(kCanonical) + Tree(64) + 8 * Tree(1024) + 4 * Str(1024);
  bytes += V(48, 32) + V(kComponents, 256) + 48 * 64 + Str(kComponents);
  bytes += kComponents * (V(64, 128) + 2 * Str(64));
  bytes += kComponents * (V(32, 160) + 3 * Str(32));
  bytes += V(48, 32) + V(kComponents, 256) + 48 * 64 + 3 * V(kComponents, 8);
  bytes += kComponents * (V(64, 128) + 3 * Str(64) + V(64, 32));
  bytes += V(96, 256) + 2 * Str(96) + V(2, 256) + 6 * V(32, 256) + 4 * Str(32);
  bytes += 4 * V(kPhysicalTensors, 8) + V(kPhysicalTensors, sizeof(SourceDisposition));
  bytes += V(kOutputs, sizeof(TensorAssembly)) + V(4 * kOutputs, sizeof(PartAssembly));
  bytes += V(kTargetParts, 4) + 4 * Str(kOutputs) + Tree(kOutputs);
  bytes += 65536 + 8 * 64 +
           kPhysicalTensors * (2 * sizeof(std::uint32_t) + sizeof(SourceDisposition) + 1);
  bytes +=
      kOutputs * (sizeof(IndexedOutput) + sizeof(std::uint32_t) + 2 * (kMaxNameBytes + 1) + 64);
  bytes += kTargetParts * (sizeof(IndexedMember) + sizeof(std::uint32_t));
  bytes += V(1, sizeof(PlannedPartFold)) + Str(1) + V(kOutputs, sizeof(std::uint32_t));
  bytes += 1024 * 1024;
  constexpr auto work = kPhysicalTensors * 139 * 512 +
                        64 * (kPhysicalTensors + kOutputs) * 32 * 513 + 512ULL * 1024 * 1024 +
                        kComponents * 64 * 64 * 256 + 1024ULL * 1024 * 256;
  return {.metadata_bytes = bytes, .work_units = work, .payload_bytes = 280, .layers = 48};
}

std::vector<spec::SourceValue> Values(const std::array<std::vector<std::int64_t>, 3>& semantic) {
  std::vector<spec::SourceValue> out;
  out.reserve(64);
  const auto integer = [&](std::string_view key, std::int64_t value) {
    out.push_back({std::string(key), value});
  };
  const auto real = [&](std::string_view key, double value) {
    out.push_back({std::string(key), value});
  };
  integer("qwen4exp.block_count", 48);
  integer("qwen4exp.full_attention_interval", 4);
  integer("qwen4exp.context_length", 262144);
  integer("qwen4exp.embedding_length", 2560);
  integer("qwen4exp.hyper_connection.count", 4);
  integer("qwen4exp.hyper_connection.low_rank", 320);
  // The checkpoint's float32 epsilon, widened.
  real("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6F);
  integer("qwen4exp.attention.head_count", 24);
  integer("qwen4exp.attention.head_count_kv", 2);
  integer("qwen4exp.attention.key_length", 256);
  integer("qwen4exp.attention.value_length", 256);
  integer("qwen4exp.rope.dimension_count", 64);
  real("qwen4exp.rope.freq_base", 10000000);
  out.push_back({"qwen4exp.rope.dimension_sections", std::vector<std::int64_t>{11, 11, 10, 0}});
  integer("qwen4exp.attention.indexer.head_count", 4);
  integer("qwen4exp.attention.indexer.key_length", 128);
  integer("qwen4exp.attention.indexer.top_k", 2048);
  std::vector<std::int64_t> ratios(48, 0);
  for (std::size_t i = 3; i < 48; i += 4) ratios[i] = 4;
  out.push_back({"qwen4exp.attention.compress_ratios", std::move(ratios)});
  integer("qwen4exp.expert_count", 512);
  integer("qwen4exp.expert_used_count", 10);
  integer("qwen4exp.expert_feed_forward_length", 640);
  integer("qwen4exp.expert_shared_feed_forward_length", 640);
  integer("qwen4exp.ssm.conv_kernel", 4);
  integer("qwen4exp.ssm.group_count", 16);
  integer("qwen4exp.ssm.time_step_rank", 48);
  integer("qwen4exp.ssm.state_size", 128);
  integer("qwen4exp.ssm.inner_size", 6144);
  integer("qwen4exp.ple.ngram_size", 3);
  integer("qwen4exp.ple.heads_per_ngram", 8);
  integer("qwen4exp.ple.conv_kernel", 4);
  integer("qwen4exp.ple.eos_token_id", 248044);
  integer("qwen4exp.embedding_length_per_layer_input", 160);
  out.push_back({"qwen4exp.ple.layers", std::vector<std::int64_t>{1}});
  out.push_back({"qwen4exp.ple.layer_multipliers", semantic[0]});
  out.push_back({"qwen4exp.ple.head_offsets", semantic[1]});
  out.push_back({"qwen4exp.ple.head_vocab_sizes", semantic[2]});
  out.push_back({"llmp.qwen4exp.indexer", std::string("fused")});
  out.push_back({"llmp.qwen4exp.experts_stacked", false});
  out.push_back({"llmp.qwen4exp.expert_scales", false});
  return out;
}

constexpr std::array<std::string_view, 3> kSemanticSuffixes = {
    "layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"};
constexpr std::array<std::string_view, 3> kSemanticHashes = {
    "37349930c3c77bf4927b2afb6b50f30c404f42f00f5790bda96e7d9589c26477",
    "af1566f7a3be2f4a9ee8e2384aa161b8d45cfcbd92376827662a028679b7d8eb",
    "81879b658d9974287ceecc77463e5fdd183cb4cd66692b061ea6af24ef414d39"};

// The semantic inputs as their pinned hashes fix them (kSemanticHashes).
std::array<std::vector<std::int64_t>, 3> PinnedSemantic() {
  return {std::vector<std::int64_t>{23703573157769, 20109073645365, 8052911324071},
          std::vector<std::int64_t>{0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228,
                                    140000297, 160000374, 180000455, 200000548, 220000655,
                                    240000802, 260000955, 280001114, 300001275},
          std::vector<std::int64_t>{20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
                                    20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
                                    20000153, 20000159, 20000161, 20000171}};
}
// Each output's parts as the canonical tensors they publish.
std::vector<spec::SourceTensor> CanonicalOf(std::span<const TensorAssembly> outputs) {
  std::vector<spec::SourceTensor> tensors;
  for (const auto& out : outputs) {
    const auto parts = format::PartsOf(out.format);
    if (!parts) continue;
    for (const auto& part : parts->parts())
      tensors.push_back({.name = PartName(out.name, part.role),
                         .format = out.format,
                         .part = part.role,
                         .experts = out.format.shape.rank == 3});
  }
  return tensors;
}
constexpr std::string_view kHeadReorder = "gdn.value_heads_tiled";
// The target's parameter folds (import/parameter_folds.h, D-108 as
// amended), as the ModelOpt import's: its spec synthesized from its
// unfolded outputs decides them by role; its zero-centered norms (1 + w)
// and A (-exp(A_log), under the head reorder) in F32, each fold whole.
std::expected<std::vector<PlannedPartFold>, Error> FoldTargetParameters(
    std::vector<TensorAssembly>& outputs) {
  const spec::Source unfolded{.architecture = "qwen4exp",
                              .container = "mlx",
                              .tensors = CanonicalOf(outputs),
                              .folds = {std::string(kHeadReorder)}};
  auto spec =
      spec::Synthesize(spec::mappings::Qwen38NativeMapping(), unfolded, Values(PinnedSemantic()));
  if (!spec) return Bad("native MLX architecture synthesis refused");
  auto applied = FoldDerivedParameters(spec->stages.front(), outputs);
  if (!applied) return std::unexpected(applied.error());
  if (applied->size() != 2 || (*applied)[0].fold != "norm.plus_one" ||
      (*applied)[1].fold != "ssm.a_neg_exp")
    return Bad("the MLX target's norms and A did not fold whole");
  std::vector<PlannedPartFold> planned{{std::string(kHeadReorder), {}}};
  for (std::size_t i = 0; i < outputs.size(); ++i)
    if (outputs[i].permutation) planned.front().outputs.push_back(static_cast<std::uint32_t>(i));
  for (auto& a : *applied) planned.push_back({std::move(a.fold), std::move(a.outputs)});
  return planned;
}
std::vector<std::string> FoldNames(std::span<const PlannedPartFold> planned) {
  std::vector<std::string> names;
  for (const auto& p : planned) names.push_back(p.fold);
  std::ranges::sort(names);
  return names;
}
}  // namespace

std::expected<NormalizationCost, Error> MlxTargetBound(const Checkpoint& source) {
  if (source.container != "mlx" || source.architecture != "qwen4_exp" ||
      source.quantization != "mlx affine" || !source.role.empty() || !source.components.empty() ||
      source.tensors.size() != kPhysicalTensors || source.files.empty() ||
      source.files.size() > 64 || source.metadata.size() > 512)
    return Bad("outside closed MLX target source domain", Rule::kUnsupported);
  for (std::size_t i = 0; i < source.metadata.size(); ++i) {
    const auto& e = source.metadata[i];
    if (e.key.size() > kMaxKeyBytes || (i != 0 && source.metadata[i - 1].key >= e.key))
      return Bad("invalid MLX metadata index", Rule::kBounds);
  }
  for (const auto& tensor : source.tensors)
    if (tensor.name.empty() || tensor.name.size() > kMaxNameBytes || tensor.dtype.size() > 16 ||
        tensor.dims.size() > 5)
      return Bad("MLX source metadata outside fixed inventory", Rule::kBounds);
  if (!Config(source)) return Bad("MLX source configuration differs from target contract");
  const auto& mapping = spec::mappings::Qwen38NativeMapping();
  if (mapping.slots.size() > 16 || mapping.tensors.size() > 1024 || mapping.params.size() > 1024)
    return Bad("native mapping exceeds prepaid inventory", Rule::kBounds);
  std::uint64_t component_count = 0;
  for (const auto& slot : mapping.slots) {
    component_count += slot.site == spec::Site::kLayer ? 48U : 1U;
    const auto* def = components::Find(slot.kind, slot.version);
    if (!def || def->params.size() > 64 || def->roles.size() > 32)
      return Bad("component catalog exceeds prepaid inventory", Rule::kBounds);
    for (const auto& p : def->params) {
      if (p.name.size() > spec::kMaxText || p.choices.size() > 64)
        return Bad("component parameter exceeds prepaid inventory", Rule::kBounds);
      for (const auto choice : p.choices)
        if (choice.size() > spec::kMaxText)
          return Bad("component choice exceeds prepaid inventory", Rule::kBounds);
    }
  }
  if (component_count > kComponents)
    return Bad("component occurrences exceed prepaid inventory", Rule::kBounds);
  return Cost();
}

std::expected<ModelOptTargetPlan, Error> NormalizeMlxTarget(Checkpoint&& source,
                                                            NormalizationLimits prepaid) {
  auto cost = MlxTargetBound(source);
  if (!cost) return std::unexpected(cost.error());
  if (prepaid.metadata_bytes < cost->metadata_bytes || prepaid.work_units < cost->work_units)
    return Bad("MLX target normalization is not prepaid", Rule::kBounds);
  std::vector<TensorAssembly> outputs;
  outputs.reserve(kOutputs);
  std::map<std::string, std::uint32_t, std::less<>> output_names;
  std::vector<SourceDisposition> dispositions(source.tensors.size());
  std::array<std::uint32_t, kRules.size()> counts{};
  std::array<std::uint32_t, 3> semantic_ids{};
  semantic_ids.fill(std::numeric_limits<std::uint32_t>::max());
  std::uint32_t auxiliary_count = 0;

  std::vector<std::uint32_t> names(source.tensors.size());
  std::iota(names.begin(), names.end(), 0U);
  std::ranges::sort(
      names, [&](auto a, auto b) { return source.tensors[a].name < source.tensors[b].name; });
  for (std::size_t i = 1; i < names.size(); ++i)
    if (source.tensors[names[i - 1]].name == source.tensors[names[i]].name)
      return Bad("duplicate MLX source name");

  for (std::size_t ti = 0; ti < source.tensors.size(); ++ti) {
    const auto& t = source.tensors[ti];
    if (t.name.starts_with("language_model.mtp.") || t.name.starts_with("vision_tower.")) {
      dispositions[ti] = {SourceDispositionKind::kAuxiliary, 1U};
      ++auxiliary_count;
      continue;
    }
    bool is_semantic = false;
    for (std::size_t i = 0; i < kSemanticSuffixes.size(); ++i) {
      if (t.name.ends_with(kSemanticSuffixes[i])) {
        if (semantic_ids[i] != std::numeric_limits<std::uint32_t>::max())
          return Bad("duplicate semantic source");
        semantic_ids[i] = static_cast<std::uint32_t>(ti);
        dispositions[ti] = {SourceDispositionKind::kSemantic, static_cast<std::uint32_t>(i + 1)};
        is_semantic = true;
        break;
      }
    }
    if (is_semantic) continue;

    std::size_t ri = 0;
    std::optional<std::array<std::uint32_t, 2>> coordinates;
    for (; ri < kRules.size(); ++ri)
      if ((coordinates = Match(kRules[ri], t.name))) break;
    if (ri == kRules.size()) return Bad("unclassified MLX source tensor");
    const auto& rule = kRules[ri];
    if (t.dtype != rule.dtype || t.dims.size() != rule.rank ||
        !std::ranges::equal(t.dims, std::span(rule.dims).first(rule.rank)) ||
        ++counts[ri] > rule.count)
      return Bad("MLX source role descriptor or multiplicity differs");
    dispositions[ti] = {SourceDispositionKind::kRouted, 0U};
    if (t.format != SourceFormat(rule) || t.part != rule.part ||
        t.experts != (rule.rank == 3 && rule.part != PartRole::kData))
      return Bad("MLX native source format differs from role contract");
    const auto canonical = Canonical(rule.canonical, (*coordinates)[0]);
    auto found = output_names.find(canonical);
    const auto format = OutputFormat(rule);
    if (found == output_names.end()) {
      if (outputs.size() >= kOutputs) return Bad("too many native target outputs", Rule::kBounds);
      auto permutation = Permutation(rule, format);
      if (!permutation) return std::unexpected(permutation.error());
      auto parts = format::PartsOf(format);
      if (!parts) return Bad("invalid target part geometry");
      TensorAssembly out{
          .name = canonical, .format = format, .permutation = std::move(*permutation)};
      out.parts.reserve(parts->size());
      for (const auto& part : parts->parts()) {
        const auto kind = rule.assembly;
        const std::size_t count = kind == PartAssemblyKind::kRows ? 128U : 1U;
        out.parts.push_back({.part = part.role,
                             .kind = kind,
                             .sources = std::vector<std::uint32_t>(
                                 count, std::numeric_limits<std::uint32_t>::max())});
      }
      const auto index = static_cast<std::uint32_t>(outputs.size());
      outputs.push_back(std::move(out));
      found = output_names.emplace(canonical, index).first;
    }
    auto& out = outputs[found->second];
    if (out.format != format) return Bad("paired source parts disagree on native output geometry");
    auto part = std::ranges::find(out.parts, rule.part, &PartAssembly::part);
    if (part == out.parts.end()) return Bad("source part is not a target part");
    const auto ordinal = rule.second == Domain::kShard ? (*coordinates)[1] : 0U;
    if (ordinal >= part->sources.size() ||
        part->sources[ordinal] != std::numeric_limits<std::uint32_t>::max())
      return Bad("two sources select one target member");
    part->sources[ordinal] = static_cast<std::uint32_t>(ti);
  }

  if (auxiliary_count != 409) return Bad("MLX auxiliary tensor count differs");
  for (std::size_t i = 0; i < semantic_ids.size(); ++i)
    if (semantic_ids[i] == std::numeric_limits<std::uint32_t>::max())
      return Bad("missing semantic tensor");
  for (std::size_t i = 0; i < counts.size(); ++i)
    if (counts[i] != kRules[i].count) return Bad("source role set is incomplete");
  auto planned = FoldTargetParameters(outputs);
  if (!planned) return std::unexpected(planned.error());

  auto indexed_cost = IndexedPartPlan::Bound(source, outputs);
  if (!indexed_cost) return std::unexpected(indexed_cost.error());
  if (indexed_cost->members != kTargetParts ||
      indexed_cost->metadata_bytes > cost->metadata_bytes ||
      indexed_cost->work_units > cost->work_units)
    return Bad("indexed plan exceeds prepaid native domain", Rule::kBounds);
  // The published model is the mapping's architecture (its manifest names
  // what serving finds a mapping and an admission profile for), not
  // config.json's model_type.
  source.architecture = "qwen4exp";
  auto parts = IndexedPartPlan::Make(std::move(source), outputs, dispositions, *indexed_cost);
  if (!parts) return std::unexpected(parts.error());

  ModelOptTargetPlan result;
  result.cost_ = *cost;
  result.parts_.emplace(std::move(*parts));
  result.planned_folds_ = std::move(*planned);
  const auto& owned = result.parts_->source();
  std::array<std::vector<std::int64_t>, 3> semantic;
  std::array<std::byte, 128> buffer{};
  for (std::size_t i = 0; i < semantic.size(); ++i) {
    if (semantic_ids[i] >= owned.tensors.size()) return Bad("missing semantic tensor");
    const auto& t = owned.tensors[semantic_ids[i]];
    const auto count = i == 0 ? 3U : 16U;
    if (t.extent.bytes != count * 8 || t.format.encoding != Encoding::kI64 ||
        t.part != PartRole::kData)
      return Bad("semantic input has wrong extent/encoding");
    auto bytes = std::span(buffer).first(count * 8);
    if (auto read = owned.Read(t.extent, 0, bytes); !read) return std::unexpected(read.error());
    const auto digest = base::Sha256{}.Update(bytes).Finish();
    if (base::ToHex(digest) != kSemanticHashes[i])
      return Bad("semantic input differs from source contract");
    result.semantic_[i] = {semantic_ids[i], t.extent, digest};
    semantic[i].reserve(count);
    for (std::size_t v = 0; v < count; ++v) {
      std::uint64_t value = 0;
      for (unsigned b = 0; b < 8; ++b)
        value |= std::to_integer<std::uint64_t>(bytes[v * 8 + b]) << (8 * b);
      if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        return Bad("negative semantic input");
      semantic[i].push_back(static_cast<std::int64_t>(value));
    }
  }
  result.canonical_.architecture = "qwen4exp";
  result.canonical_.container = "mlx";
  result.canonical_.folds = FoldNames(result.planned_folds_);
  result.canonical_.tensors.reserve(kCanonical);
  result.bindings_.reserve(kCanonical);
  std::uint32_t output_index = 0;
  for (const auto& out : result.parts_->outputs()) {
    for (const auto& part : out.parts.parts()) {
      if (result.canonical_.tensors.size() >= kCanonical)
        return Bad("canonical part count exceeds inventory", Rule::kBounds);
      result.bindings_.push_back({output_index, part.role});
      result.canonical_.tensors.push_back({.name = PartName(out.name, part.role),
                                           .format = out.format,
                                           .part = part.role,
                                           .experts = out.format.shape.rank == 3});
    }
    ++output_index;
  }
  result.values_ = Values(semantic);
  auto spec =
      spec::Synthesize(spec::mappings::Qwen38NativeMapping(), result.canonical_, result.values_);
  if (!spec) return Bad("native MLX architecture synthesis refused");
  if (auto validated = components::ValidateSpec(*spec, result.canonical_.tensors); !validated)
    return Bad("native MLX component validation refused");
  result.spec_ = std::move(*spec);
  return result;
}

spec::Source ApprovedMlxTargetSource() {
  spec::Source source;
  source.architecture = "qwen4exp";
  source.container = "mlx";
  // The outputs as the normalization makes them (each part's member count,
  // not its sources), then its parameter folds.
  std::vector<TensorAssembly> outputs;
  std::map<std::string, std::uint32_t, std::less<>> seen;
  for (const auto& rule : kRules) {
    const auto process = [&](std::uint32_t layer) {
      const auto canonical = Canonical(rule.canonical, layer);
      if (!seen.emplace(canonical, 0U).second) return;
      const auto format = OutputFormat(rule);
      const auto parts = format::PartsOf(format);
      if (!parts) return;
      TensorAssembly out{.name = canonical, .format = format};
      for (const auto& part : parts->parts())
        out.parts.push_back({.part = part.role,
                             .kind = rule.assembly,
                             .sources = std::vector<std::uint32_t>(
                                 rule.assembly == PartAssemblyKind::kRows ? 128U : 1U, 0U)});
      outputs.push_back(std::move(out));
    };
    switch (rule.first) {
      case Domain::kNone:
        process(0);
        break;
      case Domain::kPle:
        process(1);
        break;
      case Domain::kLayer:
        for (std::uint32_t l = 0; l < 48; ++l) process(l);
        break;
      case Domain::kLinear:
        for (std::uint32_t l = 0; l < 48; ++l)
          if (l % 4 != 3) process(l);
        break;
      case Domain::kFull:
        for (std::uint32_t l = 0; l < 48; ++l)
          if (l % 4 == 3) process(l);
        break;
      case Domain::kShard:
        break;
    }
  }
  auto planned = FoldTargetParameters(outputs);
  source.folds = planned ? FoldNames(*planned) : std::vector<std::string>{};
  source.tensors = CanonicalOf(outputs);
  return source;
}
}  // namespace llmp::import
