// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/modelopt_qwen38.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>

#include "components/step.h"
#include "model/qwen38.h"
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
enum class Domain : std::uint8_t {
  kNone,
  kLayer,
  kLinear,
  kFull,
  kPle,
  kMtp,
  kVision,
  kExpert,
  kShard
};
enum class HeadRecipe : std::uint8_t { kNone, kScalar, kScalarRows, kConv, kQkv, kValues, kOutput };
struct SourceRule {
  std::string_view pattern, dtype;
  std::array<std::uint64_t, 5> dims;
  std::size_t rank;
  Domain first, second;
  SourceDispositionKind disposition;
  std::string_view canonical;
  PartAssemblyKind assembly;
  HeadRecipe heads;
  PartRole part;
  std::uint32_t count;
};
// Closed source roles: converter semantics in modelopt_qwen38.py, with exact
// raw descriptors independently checked against all 35 pinned source headers.
// Auxiliary entries are retained/classified individually, never prefix-trusted.
constexpr std::array kRules = {
    SourceRule{"mtp.fc_embedding.weight",
               "BF16",
               {2560, 2560, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.fc_hidden.weight",
               "BF16",
               {2560, 2560, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.hyper_connection_mixer.hc_norm.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.hyper_connection_mixer.input_mix_weight_down.weight",
               "BF16",
               {320, 10240, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.hyper_connection_mixer.input_mix_weight_up.weight",
               "BF16",
               {10240, 320, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.attn_hyper_connection.block_inject_weight.weight",
               "BF16",
               {4, 10240, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.attn_hyper_connection.hc_norm.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.attn_hyper_connection.input_mix_weight_down.weight",
               "BF16",
               {320, 10240, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.attn_hyper_connection.input_mix_weight_up.weight",
               "BF16",
               {10240, 320, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.mlp.experts.{index}.down_proj.weight",
               "U8",
               {2560, 320, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               512},
    SourceRule{"mtp.layers.{index}.mlp.experts.{index}.down_proj.weight_scale",
               "F8_E4M3",
               {2560, 40, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               512},
    SourceRule{"mtp.layers.{index}.mlp.experts.{index}.down_proj.weight_scale_2",
               "F32",
               {0, 0, 0, 0, 0},
               0,
               Domain::kMtp,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               512},
    SourceRule{"mtp.layers.{index}.mlp.experts.{index}.gate_proj.weight",
               "U8",
               {640, 1280, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               512},
    SourceRule{"mtp.layers.{index}.mlp.experts.{index}.gate_proj.weight_scale",
               "F8_E4M3",
               {640, 160, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               512},
    SourceRule{"mtp.layers.{index}.mlp.experts.{index}.gate_proj.weight_scale_2",
               "F32",
               {0, 0, 0, 0, 0},
               0,
               Domain::kMtp,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               512},
    SourceRule{"mtp.layers.{index}.mlp.experts.{index}.up_proj.weight",
               "U8",
               {640, 1280, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               512},
    SourceRule{"mtp.layers.{index}.mlp.experts.{index}.up_proj.weight_scale",
               "F8_E4M3",
               {640, 160, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               512},
    SourceRule{"mtp.layers.{index}.mlp.experts.{index}.up_proj.weight_scale_2",
               "F32",
               {0, 0, 0, 0, 0},
               0,
               Domain::kMtp,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               512},
    SourceRule{"mtp.layers.{index}.mlp.gate.weight",
               "BF16",
               {512, 2560, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.mlp.shared_expert.down_proj.weight",
               "BF16",
               {2560, 640, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.mlp.shared_expert.gate_proj.weight",
               "BF16",
               {640, 2560, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.mlp.shared_expert.up_proj.weight",
               "BF16",
               {640, 2560, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.mlp.shared_expert_gate.weight",
               "BF16",
               {1, 2560, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.mlp_hyper_connection.block_inject_weight.weight",
               "BF16",
               {4, 10240, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.mlp_hyper_connection.hc_norm.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.mlp_hyper_connection.input_mix_weight_down.weight",
               "BF16",
               {320, 10240, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.mlp_hyper_connection.input_mix_weight_up.weight",
               "BF16",
               {10240, 320, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.self_attn.indexer.index_qk_proj.weight",
               "BF16",
               {640, 2560, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.self_attn.indexer.k_layernorm.weight",
               "BF16",
               {128, 0, 0, 0, 0},
               1,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.self_attn.indexer.q_layernorm.weight",
               "BF16",
               {128, 0, 0, 0, 0},
               1,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.self_attn.k_norm.weight",
               "BF16",
               {256, 0, 0, 0, 0},
               1,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.self_attn.k_proj.weight",
               "BF16",
               {512, 2560, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.self_attn.o_proj.weight",
               "BF16",
               {2560, 6144, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.self_attn.q_norm.weight",
               "BF16",
               {256, 0, 0, 0, 0},
               1,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.self_attn.q_proj.weight",
               "BF16",
               {12288, 2560, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.layers.{index}.self_attn.v_proj.weight",
               "BF16",
               {512, 2560, 0, 0, 0},
               2,
               Domain::kMtp,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.pre_fc_norm_embedding.weight",
               "BF16",
               {2560, 0, 0, 0, 0},
               1,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"mtp.pre_fc_norm_hidden.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.blocks.{index}.attn.proj.bias",
               "BF16",
               {1152, 0, 0, 0, 0},
               1,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.attn.proj.weight",
               "F8_E4M3",
               {1152, 1152, 0, 0, 0},
               2,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.attn.proj.weight_scale",
               "U8",
               {1152, 36, 0, 0, 0},
               2,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.attn.qkv.bias",
               "BF16",
               {3456, 0, 0, 0, 0},
               1,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.attn.qkv.weight",
               "F8_E4M3",
               {3456, 1152, 0, 0, 0},
               2,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.attn.qkv.weight_scale",
               "U8",
               {3456, 36, 0, 0, 0},
               2,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.mlp.linear_fc1.bias",
               "BF16",
               {4304, 0, 0, 0, 0},
               1,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.mlp.linear_fc1.weight",
               "F8_E4M3",
               {4304, 1152, 0, 0, 0},
               2,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.mlp.linear_fc1.weight_scale",
               "U8",
               {4304, 36, 0, 0, 0},
               2,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.mlp.linear_fc2.bias",
               "BF16",
               {1152, 0, 0, 0, 0},
               1,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.mlp.linear_fc2.weight",
               "U8",
               {1152, 2152, 0, 0, 0},
               2,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.mlp.linear_fc2.weight_scale",
               "F8_E4M3",
               {1152, 269, 0, 0, 0},
               2,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.mlp.linear_fc2.weight_scale_2",
               "F32",
               {0, 0, 0, 0, 0},
               0,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.norm1.bias",
               "BF16",
               {1152, 0, 0, 0, 0},
               1,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.norm1.weight",
               "BF16",
               {1152, 0, 0, 0, 0},
               1,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.norm2.bias",
               "BF16",
               {1152, 0, 0, 0, 0},
               1,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.blocks.{index}.norm2.weight",
               "BF16",
               {1152, 0, 0, 0, 0},
               1,
               Domain::kVision,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               27},
    SourceRule{"model.visual.merger.linear_fc1.bias",
               "BF16",
               {4608, 0, 0, 0, 0},
               1,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.merger.linear_fc1.weight",
               "F8_E4M3",
               {4608, 4608, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.merger.linear_fc1.weight_scale",
               "U8",
               {4608, 144, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.merger.linear_fc2.bias",
               "BF16",
               {2560, 0, 0, 0, 0},
               1,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.merger.linear_fc2.weight",
               "F8_E4M3",
               {2560, 4608, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.merger.linear_fc2.weight_scale",
               "U8",
               {2560, 144, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.merger.norm.bias",
               "BF16",
               {1152, 0, 0, 0, 0},
               1,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.merger.norm.weight",
               "BF16",
               {1152, 0, 0, 0, 0},
               1,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.patch_embed.proj.bias",
               "BF16",
               {1152, 0, 0, 0, 0},
               1,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.patch_embed.proj.weight",
               "BF16",
               {1152, 3, 2, 16, 16},
               5,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.visual.pos_embed.weight",
               "BF16",
               {2304, 1152, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.down_proj.input_scale",
               "F32",
               {0, 0, 0, 0, 0},
               0,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.gate_proj.input_scale",
               "F32",
               {0, 0, 0, 0, 0},
               0,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.up_proj.input_scale",
               "F32",
               {0, 0, 0, 0, 0},
               0,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kAuxiliary,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               24576},
    SourceRule{"model.language_model.layers.{index}.ple.ple_embedding.layer_multipliers",
               "I64",
               {3, 0, 0, 0, 0},
               1,
               Domain::kPle,
               Domain::kNone,
               SourceDispositionKind::kSemantic,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.layers.{index}.ple.ple_embedding.ngram_heads_offsets",
               "I64",
               {16, 0, 0, 0, 0},
               1,
               Domain::kPle,
               Domain::kNone,
               SourceDispositionKind::kSemantic,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.layers.{index}.ple.ple_embedding.ngram_heads_vocab_sizes",
               "I64",
               {16, 0, 0, 0, 0},
               1,
               Domain::kPle,
               Domain::kNone,
               SourceDispositionKind::kSemantic,
               "",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"lm_head.weight",
               "BF16",
               {248320, 2560, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "output.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.embed_tokens.weight",
               "BF16",
               {248320, 2560, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "token_embd.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.hyper_connection_mixer.hc_norm.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "output_hc_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
               "BF16",
               {320, 10240, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "output_hc_down.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
               "BF16",
               {10240, 320, 0, 0, 0},
               2,
               Domain::kNone,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "output_hc_up.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{
        "model.language_model.layers.{index}.attn_hyper_connection.block_inject_weight.weight",
        "BF16",
        {4, 10240, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        SourceDispositionKind::kRouted,
        "blk.{layer}.hc_attn_inject.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kData,
        48},
    SourceRule{"model.language_model.layers.{index}.attn_hyper_connection.hc_norm.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kLayer,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.hc_attn_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               48},
    SourceRule{
        "model.language_model.layers.{index}.attn_hyper_connection.input_mix_weight_down.weight",
        "BF16",
        {320, 10240, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        SourceDispositionKind::kRouted,
        "blk.{layer}.hc_attn_down.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kData,
        48},
    SourceRule{
        "model.language_model.layers.{index}.attn_hyper_connection.input_mix_weight_up.weight",
        "BF16",
        {10240, 320, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        SourceDispositionKind::kRouted,
        "blk.{layer}.hc_attn_up.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kData,
        48},
    SourceRule{"model.language_model.layers.{index}.linear_attn.A_log",
               "BF16",
               {48, 0, 0, 0, 0},
               1,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ssm_a",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalar,
               format::PartRole::kData,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.conv1d.weight",
               "BF16",
               {10240, 1, 4, 0, 0},
               3,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ssm_conv1d.weight",
               PartAssemblyKind::kDenseReshape,
               HeadRecipe::kConv,
               format::PartRole::kData,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.dt_bias",
               "BF16",
               {48, 0, 0, 0, 0},
               1,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ssm_dt.bias",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalar,
               format::PartRole::kData,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.in_proj_a.weight",
               "F8_E4M3",
               {48, 2560, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ssm_alpha.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalarRows,
               format::PartRole::kCodes,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.in_proj_a.weight_scale",
               "U8",
               {48, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ssm_alpha.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalarRows,
               format::PartRole::kBlockScales,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.in_proj_b.weight",
               "F8_E4M3",
               {48, 2560, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ssm_beta.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalarRows,
               format::PartRole::kCodes,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.in_proj_b.weight_scale",
               "U8",
               {48, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ssm_beta.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kScalarRows,
               format::PartRole::kBlockScales,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.in_proj_qkv.weight",
               "F8_E4M3",
               {10240, 2560, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_qkv.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kQkv,
               format::PartRole::kCodes,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.in_proj_qkv.weight_scale",
               "U8",
               {10240, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_qkv.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kQkv,
               format::PartRole::kBlockScales,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.in_proj_z.weight",
               "F8_E4M3",
               {6144, 2560, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_gate.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kValues,
               format::PartRole::kCodes,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.in_proj_z.weight_scale",
               "U8",
               {6144, 80, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_gate.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kValues,
               format::PartRole::kBlockScales,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.norm.weight",
               "BF16",
               {128, 0, 0, 0, 0},
               1,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ssm_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.out_proj.weight",
               "F8_E4M3",
               {2560, 6144, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ssm_out.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kOutput,
               format::PartRole::kCodes,
               36},
    SourceRule{"model.language_model.layers.{index}.linear_attn.out_proj.weight_scale",
               "U8",
               {2560, 192, 0, 0, 0},
               2,
               Domain::kLinear,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ssm_out.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kOutput,
               format::PartRole::kBlockScales,
               36},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.down_proj.weight",
               "U8",
               {2560, 320, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_down_exps.weight",
               PartAssemblyKind::kMatrices,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.down_proj.weight_scale",
               "F8_E4M3",
               {2560, 40, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_down_exps.weight",
               PartAssemblyKind::kMatrices,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.down_proj.weight_scale_2",
               "F32",
               {0, 0, 0, 0, 0},
               0,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_down_exps.weight",
               PartAssemblyKind::kMatrices,
               HeadRecipe::kNone,
               format::PartRole::kTensorScale,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.gate_proj.weight",
               "U8",
               {640, 1280, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_gate_exps.weight",
               PartAssemblyKind::kMatrices,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.gate_proj.weight_scale",
               "F8_E4M3",
               {640, 160, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_gate_exps.weight",
               PartAssemblyKind::kMatrices,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.gate_proj.weight_scale_2",
               "F32",
               {0, 0, 0, 0, 0},
               0,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_gate_exps.weight",
               PartAssemblyKind::kMatrices,
               HeadRecipe::kNone,
               format::PartRole::kTensorScale,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.up_proj.weight",
               "U8",
               {640, 1280, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_up_exps.weight",
               PartAssemblyKind::kMatrices,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.up_proj.weight_scale",
               "F8_E4M3",
               {640, 160, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_up_exps.weight",
               PartAssemblyKind::kMatrices,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.experts.{index}.up_proj.weight_scale_2",
               "F32",
               {0, 0, 0, 0, 0},
               0,
               Domain::kLayer,
               Domain::kExpert,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_up_exps.weight",
               PartAssemblyKind::kMatrices,
               HeadRecipe::kNone,
               format::PartRole::kTensorScale,
               24576},
    SourceRule{"model.language_model.layers.{index}.mlp.gate.weight",
               "BF16",
               {512, 2560, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_gate_inp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               48},
    SourceRule{"model.language_model.layers.{index}.mlp.shared_expert.down_proj.weight",
               "F8_E4M3",
               {2560, 640, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_down_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               48},
    SourceRule{"model.language_model.layers.{index}.mlp.shared_expert.down_proj.weight_scale",
               "U8",
               {2560, 20, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_down_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               48},
    SourceRule{"model.language_model.layers.{index}.mlp.shared_expert.gate_proj.weight",
               "F8_E4M3",
               {640, 2560, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_gate_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               48},
    SourceRule{"model.language_model.layers.{index}.mlp.shared_expert.gate_proj.weight_scale",
               "U8",
               {640, 80, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_gate_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               48},
    SourceRule{"model.language_model.layers.{index}.mlp.shared_expert.up_proj.weight",
               "F8_E4M3",
               {640, 2560, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_up_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               48},
    SourceRule{"model.language_model.layers.{index}.mlp.shared_expert.up_proj.weight_scale",
               "U8",
               {640, 80, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_up_shexp.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               48},
    SourceRule{"model.language_model.layers.{index}.mlp.shared_expert_gate.weight",
               "BF16",
               {1, 2560, 0, 0, 0},
               2,
               Domain::kLayer,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ffn_gate_inp_shexp.weight",
               PartAssemblyKind::kDenseReshape,
               HeadRecipe::kNone,
               format::PartRole::kData,
               48},
    SourceRule{
        "model.language_model.layers.{index}.mlp_hyper_connection.block_inject_weight.weight",
        "BF16",
        {4, 10240, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        SourceDispositionKind::kRouted,
        "blk.{layer}.hc_ffn_inject.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kData,
        48},
    SourceRule{"model.language_model.layers.{index}.mlp_hyper_connection.hc_norm.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kLayer,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.hc_ffn_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               48},
    SourceRule{
        "model.language_model.layers.{index}.mlp_hyper_connection.input_mix_weight_down.weight",
        "BF16",
        {320, 10240, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        SourceDispositionKind::kRouted,
        "blk.{layer}.hc_ffn_down.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kData,
        48},
    SourceRule{
        "model.language_model.layers.{index}.mlp_hyper_connection.input_mix_weight_up.weight",
        "BF16",
        {10240, 320, 0, 0, 0},
        2,
        Domain::kLayer,
        Domain::kNone,
        SourceDispositionKind::kRouted,
        "blk.{layer}.hc_ffn_up.weight",
        PartAssemblyKind::kCopy,
        HeadRecipe::kNone,
        format::PartRole::kData,
        48},
    SourceRule{"model.language_model.layers.{index}.ple.conv1d.weight",
               "BF16",
               {10240, 1, 4, 0, 0},
               3,
               Domain::kPle,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ple_conv1d.weight",
               PartAssemblyKind::kDenseReshape,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.layers.{index}.ple.key_proj.weight",
               "BF16",
               {10240, 2560, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ple_key.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.layers.{index}.ple.norm_conv.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kPle,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ple_norm_conv.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.layers.{index}.ple.norm_key.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kPle,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ple_norm_key.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.layers.{index}.ple.norm_query.weight",
               "BF16",
               {10240, 0, 0, 0, 0},
               1,
               Domain::kPle,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ple_norm_query.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.layers.{index}.ple.ple_embedding.ngram_embedding.shard_{index}"
               ".weight",
               "U8",
               {2500012, 80, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kShard,
               SourceDispositionKind::kRouted,
               "per_layer_token_embd.weight",
               PartAssemblyKind::kRows,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               128},
    SourceRule{"model.language_model.layers.{index}.ple.ple_embedding.ngram_embedding.shard_{index}"
               ".weight_scale",
               "F8_E4M3",
               {2500012, 10, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kShard,
               SourceDispositionKind::kRouted,
               "per_layer_token_embd.weight",
               PartAssemblyKind::kRows,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               128},
    SourceRule{
        "model.language_model.layers.{index}.ple.ple_embedding.ngram_embedding.weight_scale_2",
        "F32",
        {1, 0, 0, 0, 0},
        1,
        Domain::kPle,
        Domain::kNone,
        SourceDispositionKind::kRouted,
        "per_layer_token_embd.weight",
        PartAssemblyKind::kSharedTensorScale,
        HeadRecipe::kNone,
        format::PartRole::kTensorScale,
        1},
    SourceRule{"model.language_model.layers.{index}.ple.value_proj.weight",
               "BF16",
               {2560, 2560, 0, 0, 0},
               2,
               Domain::kPle,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.ple_value.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               1},
    SourceRule{"model.language_model.layers.{index}.self_attn.indexer.index_qk_proj.weight",
               "F8_E4M3",
               {640, 2560, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.indexer.qk_proj.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.indexer.index_qk_proj.weight_scale",
               "U8",
               {640, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.indexer.qk_proj.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.indexer.k_layernorm.weight",
               "BF16",
               {128, 0, 0, 0, 0},
               1,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.indexer.k_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.indexer.q_layernorm.weight",
               "BF16",
               {128, 0, 0, 0, 0},
               1,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.indexer.q_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.k_norm.weight",
               "BF16",
               {256, 0, 0, 0, 0},
               1,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_k_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.k_proj.weight",
               "F8_E4M3",
               {512, 2560, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_k.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.k_proj.weight_scale",
               "U8",
               {512, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_k.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.o_proj.weight",
               "F8_E4M3",
               {2560, 6144, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_output.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.o_proj.weight_scale",
               "U8",
               {2560, 192, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_output.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.q_norm.weight",
               "BF16",
               {256, 0, 0, 0, 0},
               1,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_q_norm.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kData,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.q_proj.weight",
               "F8_E4M3",
               {12288, 2560, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_q.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.q_proj.weight_scale",
               "U8",
               {12288, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_q.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.v_proj.weight",
               "F8_E4M3",
               {512, 2560, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_v.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kCodes,
               12},
    SourceRule{"model.language_model.layers.{index}.self_attn.v_proj.weight_scale",
               "U8",
               {512, 80, 0, 0, 0},
               2,
               Domain::kFull,
               Domain::kNone,
               SourceDispositionKind::kRouted,
               "blk.{layer}.attn_v.weight",
               PartAssemblyKind::kCopy,
               HeadRecipe::kNone,
               format::PartRole::kBlockScales,
               12},
};
bool InDomain(Domain d, std::uint32_t n) {
  switch (d) {
    case Domain::kNone:
      return n == 0;
    case Domain::kLayer:
      return n < 48;
    case Domain::kLinear:
      return n < 48 && n % 4 != 3;
    case Domain::kFull:
      return n < 48 && n % 4 == 3;
    case Domain::kPle:
      return n == 1;
    case Domain::kMtp:
      return n == 0;
    case Domain::kVision:
      return n < 27;
    case Domain::kExpert:
      return n < 512;
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
  if (name.ends_with(".weight")) {
    const auto stem = name.substr(0, name.size() - 7);
    if (part == PartRole::kBlockScales) return std::string(stem) + ".weight_scale";
    if (part == PartRole::kBiases) return std::string(stem) + ".weight_bias";
    if (part == PartRole::kTensorScale) return std::string(stem) + ".weight_scale_2";
  }
  return std::string(name);
}
format::TensorFormat SourceFormat(const SourceRule& r) {
  format::TensorFormat f{.encoding = Encoding::kBF16};
  f.shape.rank = static_cast<std::uint8_t>(r.rank);
  for (std::size_t i = 0; i < r.rank; ++i) f.shape.ne[i] = r.dims[r.rank - 1 - i];
  if (r.part != PartRole::kData) {
    f.layout = Layout::kSplit;
    const bool nv = r.part == PartRole::kTensorScale ||
                    (r.part == PartRole::kCodes ? r.dtype == "U8" : r.dtype == "F8_E4M3");
    f.encoding = nv ? Encoding::kNvfp4 : Encoding::kMxfp8;
    f.params.tensor_scale = nv;
    std::uint64_t k = 0, n = 0;
    if (r.part == PartRole::kTensorScale) {
      if (r.canonical == "per_layer_token_embd.weight") {
        k = 160;
        n = 2500012;
      } else if (r.canonical.find("down_exps") != std::string_view::npos) {
        k = 640;
        n = 2560;
      } else {
        k = 2560;
        n = 640;
      }
    } else {
      k = r.dims[1] * (r.part == PartRole::kCodes ? (nv ? 2U : 1U) : (nv ? 16U : 32U));
      n = r.dims[0];
    }
    f.shape = {.ne = {k, n, 1, 1}, .rank = 2};
  }
  return f;
}
format::TensorFormat OutputFormat(const SourceRule& r) {
  auto f = SourceFormat(r);
  if (r.assembly == PartAssemblyKind::kMatrices) {
    f.shape.rank = 3;
    f.shape.ne[2] = 512;
  }
  if (r.assembly == PartAssemblyKind::kRows || r.assembly == PartAssemblyKind::kSharedTensorScale)
    f.shape.ne[1] *= 128;
  if (r.assembly == PartAssemblyKind::kDenseReshape) {
    if (f.shape.rank == 3)
      f.shape = {.ne = {f.shape.ne[0], f.shape.ne[2], 1, 1}, .rank = 2};
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
bool Text(const Checkpoint& s, std::string_view k, std::string_view expected) {
  const auto* v = Meta(s, k);
  const auto* x = v ? std::get_if<std::string>(v) : nullptr;
  return x && *x == expected;
}
bool Flag(const Checkpoint& s, std::string_view k, bool expected) {
  const auto* v = Meta(s, k);
  const auto* x = v ? std::get_if<bool>(v) : nullptr;
  return x && *x == expected;
}
bool Ints(const Checkpoint& s, std::string_view k, std::span<const std::int64_t> expected) {
  const auto* v = Meta(s, k);
  const auto* x = v ? std::get_if<std::vector<std::int64_t>>(v) : nullptr;
  return x && std::ranges::equal(*x, expected);
}
// Finite flattened-key closure from the pinned config. Only the fields
// checked in Config affect the target spec; the remaining named training,
// output, label and explicitly excluded-stage settings are retained auxiliary
// metadata. A future unknown field is not silently assumed nonsemantic.
constexpr std::array<std::string_view, 121> kConfigKeys = {
    "architectures",
    "chunk_size_feed_forward",
    "dtype",
    "id2label.0",
    "id2label.1",
    "image_token_id",
    "is_encoder_decoder",
    "is_multimodal",
    "label2id.LABEL_0",
    "label2id.LABEL_1",
    "language_model_only",
    "model_type",
    "output_attentions",
    "output_hidden_states",
    "problem_type",
    "return_dict",
    "text_config._name_or_path",
    "text_config.attention_bias",
    "text_config.attention_dropout",
    "text_config.bos_token_id",
    "text_config.chunk_size_feed_forward",
    "text_config.dtype",
    "text_config.eos_token_id",
    "text_config.hc_count",
    "text_config.hc_lowrank",
    "text_config.head_dim",
    "text_config.heads_per_ngram",
    "text_config.hidden_act",
    "text_config.hidden_size",
    "text_config.id2label.0",
    "text_config.id2label.1",
    "text_config.indexer_budget",
    "text_config.indexer_compress_ratio",
    "text_config.indexer_head_dim",
    "text_config.indexer_kv_heads",
    "text_config.indexer_n_heads",
    "text_config.initializer_range",
    "text_config.intermediate_size",
    "text_config.is_encoder_decoder",
    "text_config.label2id.LABEL_0",
    "text_config.label2id.LABEL_1",
    "text_config.layer_types",
    "text_config.linear_conv_kernel_dim",
    "text_config.linear_key_head_dim",
    "text_config.linear_num_key_heads",
    "text_config.linear_num_value_heads",
    "text_config.linear_value_head_dim",
    "text_config.make_ngram_vocab_size_divisible_by",
    "text_config.mamba_ssm_dtype",
    "text_config.max_position_embeddings",
    "text_config.model_type",
    "text_config.moe_intermediate_size",
    "text_config.mtp.hybrid",
    "text_config.mtp.layer_types",
    "text_config.mtp.mtp_use_hidden_state_from_layer",
    "text_config.mtp.num_hidden_layers",
    "text_config.mtp.rope_theta",
    "text_config.mtp_num_hidden_layers",
    "text_config.mtp_use_dedicated_embeddings",
    "text_config.ngram_size",
    "text_config.ngram_vocab_size_base",
    "text_config.num_attention_heads",
    "text_config.num_experts",
    "text_config.num_experts_per_tok",
    "text_config.num_hidden_layers",
    "text_config.num_key_value_heads",
    "text_config.output_attentions",
    "text_config.output_gate_type",
    "text_config.output_hidden_states",
    "text_config.output_router_logits",
    "text_config.pad_token_id",
    "text_config.ple_conv_kernel_size",
    "text_config.ple_embed_dim",
    "text_config.ple_embedding_dtype",
    "text_config.ple_layer_ids",
    "text_config.problem_type",
    "text_config.return_dict",
    "text_config.rms_norm_eps",
    "text_config.rope_parameters.mrope_interleaved",
    "text_config.rope_parameters.mrope_section",
    "text_config.rope_parameters.partial_rotary_factor",
    "text_config.rope_parameters.rope_theta",
    "text_config.rope_parameters.rope_type",
    "text_config.router_aux_loss_coef",
    "text_config.shared_expert_intermediate_size",
    "text_config.split_ngram_parts",
    "text_config.tie_word_embeddings",
    "text_config.use_cache",
    "text_config.vocab_size",
    "tie_word_embeddings",
    "transformers_version",
    "video_token_id",
    "vision_config._name_or_path",
    "vision_config.architectures",
    "vision_config.chunk_size_feed_forward",
    "vision_config.deepstack_visual_indexes",
    "vision_config.depth",
    "vision_config.dtype",
    "vision_config.hidden_act",
    "vision_config.hidden_size",
    "vision_config.id2label.0",
    "vision_config.id2label.1",
    "vision_config.in_channels",
    "vision_config.initializer_range",
    "vision_config.intermediate_size",
    "vision_config.is_encoder_decoder",
    "vision_config.label2id.LABEL_0",
    "vision_config.label2id.LABEL_1",
    "vision_config.model_type",
    "vision_config.num_heads",
    "vision_config.num_position_embeddings",
    "vision_config.out_hidden_size",
    "vision_config.output_attentions",
    "vision_config.output_hidden_states",
    "vision_config.patch_size",
    "vision_config.problem_type",
    "vision_config.return_dict",
    "vision_config.spatial_merge_size",
    "vision_config.temporal_patch_size",
    "vision_end_token_id",
    "vision_start_token_id",
};
bool Config(const Checkpoint& s) {
  for (const auto& e : s.metadata)
    if (!std::ranges::binary_search(kConfigKeys, e.key)) return false;
  constexpr std::pair<std::string_view, std::int64_t> ints[] = {
      {"text_config.vocab_size", 248320},
      {"text_config.hidden_size", 2560},
      {"text_config.num_hidden_layers", 48},
      {"text_config.num_attention_heads", 24},
      {"text_config.num_key_value_heads", 2},
      {"text_config.head_dim", 256},
      {"text_config.max_position_embeddings", 262144},
      {"text_config.linear_conv_kernel_dim", 4},
      {"text_config.linear_key_head_dim", 128},
      {"text_config.linear_value_head_dim", 128},
      {"text_config.linear_num_key_heads", 16},
      {"text_config.linear_num_value_heads", 48},
      {"text_config.moe_intermediate_size", 640},
      {"text_config.shared_expert_intermediate_size", 640},
      {"text_config.num_experts_per_tok", 10},
      {"text_config.num_experts", 512},
      {"text_config.hc_count", 4},
      {"text_config.hc_lowrank", 320},
      {"text_config.ple_embed_dim", 2560},
      {"text_config.ple_conv_kernel_size", 4},
      {"text_config.ngram_size", 3},
      {"text_config.heads_per_ngram", 8},
      {"text_config.ngram_vocab_size_base", 20000000},
      {"text_config.make_ngram_vocab_size_divisible_by", 128},
      {"text_config.split_ngram_parts", 128},
      {"text_config.indexer_n_heads", 4},
      {"text_config.indexer_kv_heads", 1},
      {"text_config.indexer_head_dim", 128},
      {"text_config.indexer_budget", 2048},
      {"text_config.indexer_compress_ratio", 4},
      {"text_config.eos_token_id", 248044},
      {"text_config.bos_token_id", 248044},
      {"text_config.mtp_num_hidden_layers", 1},
      {"text_config.mtp.num_hidden_layers", 1}};
  for (const auto& [key, value] : ints)
    if (!Int(s, key, value)) return false;
  constexpr std::array<std::int64_t, 1> ple = {2};
  constexpr std::array<std::int64_t, 3> sections = {11, 11, 10};
  if (!Ints(s, "text_config.ple_layer_ids", ple) ||
      !Ints(s, "text_config.rope_parameters.mrope_section", sections) ||
      !Text(s, "text_config.model_type", "qwen3_8_flash_next_text") ||
      !Text(s, "text_config.hidden_act", "silu") ||
      !Text(s, "text_config.mamba_ssm_dtype", "float32") ||
      !Text(s, "text_config.dtype", "bfloat16") || !Text(s, "dtype", "bfloat16") ||
      !Flag(s, "text_config.is_encoder_decoder", false) || !Flag(s, "is_encoder_decoder", false) ||
      !Text(s, "text_config.output_gate_type", "sigmoid") ||
      !Text(s, "text_config.ple_embedding_dtype", "nvfp4") ||
      !Text(s, "text_config.rope_parameters.rope_type", "default") ||
      !Real(s, "text_config.rope_parameters.rope_theta", 10000000) ||
      !Real(s, "text_config.rope_parameters.partial_rotary_factor", 0.25) ||
      !Real(s, "text_config.rms_norm_eps", 0.000001) ||
      !Real(s, "text_config.attention_dropout", 0) ||
      !Flag(s, "text_config.rope_parameters.mrope_interleaved", true) ||
      !Flag(s, "text_config.attention_bias", false) ||
      !Flag(s, "text_config.tie_word_embeddings", false) || !Flag(s, "tie_word_embeddings", false))
    return false;
  const auto* layers = Meta(s, "text_config.layer_types");
  const auto* list = layers ? std::get_if<std::vector<std::string>>(layers) : nullptr;
  if (!list || list->size() != 48) return false;
  for (std::size_t i = 0; i < 48; ++i)
    if ((*list)[i] != (i % 4 == 3 ? "full_attention" : "linear_attention")) return false;
  const auto* mtp = Meta(s, "text_config.mtp.layer_types");
  const auto* ml = mtp ? std::get_if<std::vector<std::string>>(mtp) : nullptr;
  if (!ml || ml->size() != 1 || ml->front() != "full_attention") return false;
  constexpr std::pair<std::string_view, std::int64_t> vision[] = {
      {"vision_config.depth", 27},
      {"vision_config.hidden_size", 1152},
      {"vision_config.intermediate_size", 4304},
      {"vision_config.num_heads", 16},
      {"vision_config.in_channels", 3},
      {"vision_config.patch_size", 16},
      {"vision_config.spatial_merge_size", 2},
      {"vision_config.temporal_patch_size", 2},
      {"vision_config.out_hidden_size", 2560},
      {"vision_config.num_position_embeddings", 2304}};
  for (const auto& [key, value] : vision)
    if (!Int(s, key, value)) return false;
  return Text(s, "vision_config.model_type", "qwen3_5_vision");
}
constexpr std::uint64_t kPhysicalTensors = 301730, kTargetParts = 222892;
constexpr std::uint64_t kOutputs = 2048, kCanonical = 4096, kComponents = 512;
static_assert(sizeof(spec::SourceTensor) <= 256 && sizeof(spec::SourceValue) <= 256);
static_assert(sizeof(spec::ComponentRef) <= 256 && sizeof(components::Instance) <= 256);
static_assert(sizeof(spec::Param) <= 128 && sizeof(spec::WeightBinding) <= 160);
static_assert(sizeof(model::StateTensorContract) <= 256);
static_assert(sizeof(std::string) <= 32 && sizeof(std::vector<std::uint32_t>) <= 32);
static_assert(sizeof(IndexedMember) <= 512 && sizeof(IndexedOutput) <= 512 &&
              sizeof(PartAssembly) <= 256);
static_assert(sizeof(TensorAssembly) <= 512 && sizeof(SourceDisposition) <= 8 &&
              sizeof(NormalizedPartBinding) <= 8 && sizeof(PlannedPartFold) <= 64);
static_assert(sizeof(void*) == 8);
NormalizationCost Cost() {
  // Closed domain, not generic IR/compiler limits. The synthesis/catalog
  // ledger is the existing normalization ledger specialized to bounded 48
  // layers, <=512 component occurrences, <=4096 canonical parts and 64 values.
  // Count all overlapping owned/temporary families simultaneously. V includes
  // geometric vector growth, Str includes allocation slack, Tree node overhead.
  const auto V = [](std::uint64_t n, std::uint64_t z) { return 3 * n * z + 64; };
  const auto Str = [](std::uint64_t n) { return n * 1024; };
  const auto Tree = [](std::uint64_t n) { return n * 256; };
  std::uint64_t bytes = 0;
  // Four index-sized vectors include canonical-to-output/part bindings.
  bytes += 3 * V(kCanonical, 256) + 2 * Str(kCanonical) + 4 * V(kCanonical, 8);
  bytes += 2 * V(512, 256) + V(64, 256) + 2 * Str(64) + 68 * V(48, 8);
  bytes += 2 * Tree(kCanonical) + Str(kCanonical) + Tree(64) + 8 * Tree(1024) + 4 * Str(1024);
  bytes += V(48, 32) + V(kComponents, 256) + 48 * 64 + Str(kComponents);
  bytes += kComponents * (V(64, 128) + 2 * Str(64));
  bytes += kComponents * (V(32, 160) + 3 * Str(32));
  bytes += V(48, 32) + V(kComponents, 256) + 48 * 64 + 3 * V(kComponents, 8);
  bytes += kComponents * (V(64, 128) + 3 * Str(64) + V(64, 32));
  bytes += V(96, 256) + 2 * Str(96) + V(2, 256) + 6 * V(32, 256) + 4 * Str(32);
  // Adapter declarations, shared-stem map, source index/classification/counts,
  // all member slots and names coexist with IndexedPartPlan::Make temporaries.
  bytes += 4 * V(kPhysicalTensors, 8) + V(kPhysicalTensors, sizeof(SourceDisposition));
  bytes += V(kOutputs, sizeof(TensorAssembly)) + V(4 * kOutputs, sizeof(PartAssembly));
  bytes += V(kTargetParts, 4) + 4 * Str(kOutputs) + Tree(kOutputs);
  // Upper bound of IndexedPartPlan::Bound at this fixed domain; its exact
  // sizeof-derived request is also checked before calling Make.
  bytes += 65536 + 8 * 64 +
           kPhysicalTensors * (2 * sizeof(std::uint32_t) + sizeof(SourceDisposition) + 1);
  bytes +=
      kOutputs * (sizeof(IndexedOutput) + sizeof(std::uint32_t) + 2 * (kMaxNameBytes + 1) + 64);
  bytes += kTargetParts * (sizeof(IndexedMember) + sizeof(std::uint32_t));
  bytes += V(1, sizeof(PlannedPartFold)) + Str(1) + V(kOutputs, sizeof(std::uint32_t));
  bytes += 1024 * 1024;  // bounded stack, config scalars, 280 B semantics, small keys
  constexpr auto work = kPhysicalTensors * 139 * 512 +
                        64 * (kPhysicalTensors + kOutputs) * 32 * 513 + 512ULL * 1024 * 1024 +
                        kComponents * 64 * 64 * 256 + 1024ULL * 1024 * 256;
  return {.metadata_bytes = bytes, .work_units = work, .payload_bytes = 280, .layers = 48};
}
std::vector<spec::SourceValue> Values(const std::array<std::vector<std::int64_t>, 3>& semantic) {
  const auto& p = model::Qwen38Flash();
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
  real("qwen4exp.attention.layer_norm_rms_epsilon", p.rms_eps);
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
}  // namespace

std::expected<NormalizationCost, Error> ModelOptTargetBound(const Checkpoint& source) {
  if (source.container != "safetensors" || source.architecture != "qwen3_8_flash_next" ||
      source.quantization != "modelopt MIXED_PRECISION" || !source.role.empty() ||
      !source.components.empty() || !source.skipped.empty() ||
      source.tensors.size() != kPhysicalTensors || source.files.empty() ||
      source.files.size() > 64 || source.metadata.size() > 512)
    return Bad("outside closed ModelOpt target source domain", Rule::kUnsupported);
  for (std::size_t i = 0; i < source.metadata.size(); ++i) {
    const auto& e = source.metadata[i];
    if (e.key.size() > kMaxKeyBytes || (i != 0 && source.metadata[i - 1].key >= e.key))
      return Bad("invalid ModelOpt metadata index", Rule::kBounds);
  }
  for (const auto& tensor : source.tensors)
    if (tensor.name.empty() || tensor.name.size() > kMaxNameBytes || tensor.dtype.size() > 16 ||
        tensor.dims.size() > 5)
      return Bad("ModelOpt source metadata outside fixed inventory", Rule::kBounds);
  if (!Config(source)) return Bad("ModelOpt source configuration differs from target contract");
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

std::expected<ModelOptTargetPlan, Error> NormalizeModelOptTarget(Checkpoint&& source,
                                                                 NormalizationLimits prepaid) {
  auto cost = ModelOptTargetBound(source);
  if (!cost) return std::unexpected(cost.error());
  if (prepaid.metadata_bytes < cost->metadata_bytes || prepaid.work_units < cost->work_units)
    return Bad("ModelOpt target normalization is not prepaid", Rule::kBounds);
  std::vector<TensorAssembly> outputs;
  outputs.reserve(kOutputs);
  std::map<std::string, std::uint32_t, std::less<>> output_names;
  std::vector<SourceDisposition> dispositions(source.tensors.size());
  std::array<std::uint32_t, kRules.size()> counts{};
  std::array<std::uint32_t, 3> semantic_ids{};
  semantic_ids.fill(std::numeric_limits<std::uint32_t>::max());
  std::vector<std::uint32_t> names(source.tensors.size());
  std::iota(names.begin(), names.end(), 0U);
  std::ranges::sort(
      names, [&](auto a, auto b) { return source.tensors[a].name < source.tensors[b].name; });
  for (std::size_t i = 1; i < names.size(); ++i)
    if (source.tensors[names[i - 1]].name == source.tensors[names[i]].name)
      return Bad("duplicate ModelOpt source name");
  for (std::size_t ti = 0; ti < source.tensors.size(); ++ti) {
    const auto& t = source.tensors[ti];
    std::size_t ri = 0;
    std::optional<std::array<std::uint32_t, 2>> coordinates;
    for (; ri < kRules.size(); ++ri)
      if ((coordinates = Match(kRules[ri], t.name))) break;
    if (ri == kRules.size()) return Bad("unclassified ModelOpt source tensor");
    const auto& rule = kRules[ri];
    if (t.dtype != rule.dtype || t.dims.size() != rule.rank ||
        !std::ranges::equal(t.dims, std::span(rule.dims).first(rule.rank)) ||
        ++counts[ri] > rule.count)
      return Bad("ModelOpt source role descriptor or multiplicity differs");
    dispositions[ti] = {rule.disposition, rule.disposition == SourceDispositionKind::kRouted
                                              ? 0U
                                              : static_cast<std::uint32_t>(ri + 1)};
    if (rule.disposition == SourceDispositionKind::kSemantic) {
      bool found = false;
      for (std::size_t i = 0; i < kSemanticSuffixes.size(); ++i)
        if (t.name.ends_with(kSemanticSuffixes[i])) {
          if (semantic_ids[i] != std::numeric_limits<std::uint32_t>::max())
            return Bad("duplicate semantic source");
          semantic_ids[i] = static_cast<std::uint32_t>(ti);
          found = true;
        }
      if (!found) return Bad("unknown semantic source");
      continue;
    }
    if (rule.disposition == SourceDispositionKind::kAuxiliary) continue;
    if (t.format != SourceFormat(rule) || t.part != rule.part || t.experts)
      return Bad("ModelOpt native source format differs from role contract");
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
        const bool shared = rule.assembly == PartAssemblyKind::kRows ||
                            rule.assembly == PartAssemblyKind::kSharedTensorScale;
        const auto kind =
            shared ? (part.role == PartRole::kTensorScale ? PartAssemblyKind::kSharedTensorScale
                                                          : PartAssemblyKind::kRows)
                   : rule.assembly;
        const std::size_t count = kind == PartAssemblyKind::kMatrices ? 512U
                                  : kind == PartAssemblyKind::kRows   ? 128U
                                                                      : 1U;
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
    const auto ordinal =
        rule.second == Domain::kExpert || rule.second == Domain::kShard ? (*coordinates)[1] : 0U;
    if (ordinal >= part->sources.size() ||
        part->sources[ordinal] != std::numeric_limits<std::uint32_t>::max())
      return Bad("two sources select one target member");
    part->sources[ordinal] = static_cast<std::uint32_t>(ti);
  }
  for (std::size_t i = 0; i < counts.size(); ++i)
    if (counts[i] != kRules[i].count) return Bad("source role set is incomplete");
  auto indexed_cost = IndexedPartPlan::Bound(source, outputs);
  if (!indexed_cost) return std::unexpected(indexed_cost.error());
  // This exact request is included in Cost's simultaneously charged families.
  if (indexed_cost->members != kTargetParts ||
      indexed_cost->metadata_bytes > cost->metadata_bytes ||
      indexed_cost->work_units > cost->work_units)
    return Bad("indexed plan exceeds prepaid native domain", Rule::kBounds);
  auto parts = IndexedPartPlan::Make(std::move(source), outputs, dispositions, *indexed_cost);
  if (!parts) return std::unexpected(parts.error());
  ModelOptTargetPlan result;
  result.cost_ = *cost;
  result.parts_.emplace(std::move(*parts));
  result.planned_folds_.push_back({"gdn.value_heads_tiled", {}});
  auto& transformed = result.planned_folds_.front().outputs;
  transformed.reserve(outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i)
    if (outputs[i].permutation) transformed.push_back(static_cast<std::uint32_t>(i));
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
  result.canonical_.container = "safetensors";
  result.canonical_.folds = {"gdn.value_heads_tiled"};
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
  if (!spec) return Bad("native ModelOpt architecture synthesis refused");
  if (auto validated = components::ValidateSpec(*spec, result.canonical_.tensors); !validated)
    return Bad("native ModelOpt component validation refused");
  result.spec_ = std::move(*spec);
  return result;
}
}  // namespace llmp::import
