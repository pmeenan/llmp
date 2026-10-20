// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/modelopt_qwen38.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/step.h"
#include "import/expert_relayout.h"
#include "import/files.h"
#include "import/mlx_qwen38.h"
#include "import/normalizers.h"
#include "import/parameter_folds.h"
#include "import/writer.h"
#include "ir/graph.h"
#include "model/state_contract.h"
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
// The n-gram side table (its row-gathered role) row-interleaved
// (import/expert_relayout.h RelayoutRowTable), as the converter wrote it.
std::expected<void, Error> RelayoutSideTable(std::vector<TensorAssembly>& outputs) {
  auto table = std::ranges::find(outputs, "per_layer_token_embd.weight", &TensorAssembly::name);
  if (table == outputs.end()) return Bad("the n-gram table is missing");
  auto relaid = RelayoutRowTable(*table);
  if (!relaid) return std::unexpected(relaid.error());
  if (!*relaid) return Bad("the n-gram table cannot be row-interleaved");
  return {};
}
// The routed experts' gathered tables (blk.L.ffn_{gate,up,down}_exps.weight,
// by their projection's role) into the scale-atom layout
// (import/expert_relayout.h): gate and up stacked as the gate-up table,
// each projection's tensor scales an F32 [experts] vector.
std::expected<ExpertRelayoutSummary, Error> RelayoutRoutedExperts(
    std::vector<TensorAssembly>& outputs, const spec::ModelSpec* spec = nullptr) {
  return RelayoutExpertTables(outputs, spec);
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
// The target's values; `relaid`: its experts in the scale-atom layout,
// gate and up stacked, their tensor scales vectors (import/expert_relayout.h);
// `activation_scales`: its routed experts carry their activations' input
// scales (ModelOpt's NVFP4, W4A4, rather than W4A16_NVFP4).
std::vector<spec::SourceValue> Values(const std::array<std::vector<std::int64_t>, 3>& semantic,
                                      bool relaid, bool activation_scales) {
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
  out.push_back({"llmp.qwen4exp.experts_stacked", relaid});
  out.push_back({"llmp.qwen4exp.expert_scales", relaid});
  out.push_back({"llmp.qwen4exp.expert_activation_scales", activation_scales});
  return out;
}
constexpr std::array<std::string_view, 3> kSemanticSuffixes = {
    "layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"};
// The semantic inputs.
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
std::expected<std::vector<spec::SourceTensor>, Error> CanonicalOf(
    std::span<const TensorAssembly> outputs) {
  std::vector<spec::SourceTensor> tensors;
  for (const auto& out : outputs) {
    const auto parts = format::PartsOf(out.format);
    if (!parts) return Bad("invalid native output geometry");
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
// amended): its spec synthesized from its unfolded outputs decides them by
// role; its zero-centered norms (1 + w) and A (-exp(A_log), under the head
// reorder) in F32, each fold whole.
std::expected<std::vector<PlannedPartFold>, Error> FoldTargetParameters(
    std::vector<TensorAssembly>& outputs, bool activation_scales) {
  auto tensors = CanonicalOf(outputs);
  if (!tensors) return std::unexpected(tensors.error());
  const spec::Source unfolded{.architecture = "qwen4exp",
                              .container = "safetensors",
                              .tensors = std::move(*tensors),
                              .folds = {std::string(kHeadReorder)}};
  auto spec = spec::Synthesize(spec::mappings::Qwen38NativeMapping(), unfolded,
                               Values(PinnedSemantic(), true, activation_scales));
  if (!spec) return Bad("native ModelOpt architecture synthesis refused");
  auto applied = FoldDerivedParameters(spec->stages.front(), outputs);
  if (!applied) return std::unexpected(applied.error());
  if (applied->size() != 2 || (*applied)[0].fold != "norm.plus_one" ||
      (*applied)[1].fold != "ssm.a_neg_exp")
    return Bad("the target's norms and A did not fold whole");
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

std::expected<NormalizationCost, Error> ModelOptTargetBound(const Checkpoint& source) {
  if (source.container != "safetensors" || source.architecture != "qwen3_8_flash_next" ||
      source.quantization != "modelopt MIXED_PRECISION" || !source.role.empty() ||
      !source.components.empty() || source.tensors.size() != kPhysicalTensors ||
      source.files.empty() || source.files.size() > 64 || source.metadata.size() > 512)
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
  // The routed experts' input scales (auxiliary: no product reads them)
  // say the checkpoint quantized their activations too (W4A4).
  bool activation_scales = true;
  for (std::size_t i = 0; i < kRules.size(); ++i)
    if (kRules[i].pattern.contains(".mlp.experts.") && kRules[i].pattern.ends_with(".input_scale"))
      activation_scales = activation_scales && counts[i] == kRules[i].count;
  auto relaid = RelayoutRoutedExperts(outputs);
  if (!relaid) return std::unexpected(relaid.error());
  if (relaid->tables != 96 || relaid->stacked != 48 || relaid->scale_vectors != 144)
    return Bad("the routed experts' relayout does not cover every layer");
  if (auto table = RelayoutSideTable(outputs); !table) return std::unexpected(table.error());
  auto planned = FoldTargetParameters(outputs, activation_scales);
  if (!planned) return std::unexpected(planned.error());
  auto indexed_cost = IndexedPartPlan::Bound(source, outputs);
  if (!indexed_cost) return std::unexpected(indexed_cost.error());
  // This exact request is included in Cost's simultaneously charged families.
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
  result.values_ = Values(semantic, true, activation_scales);
  auto spec =
      spec::Synthesize(spec::mappings::Qwen38NativeMapping(), result.canonical_, result.values_);
  if (!spec) return Bad("native ModelOpt architecture synthesis refused");
  if (auto validated = components::ValidateSpec(*spec, result.canonical_.tensors); !validated)
    return Bad("native ModelOpt component validation refused");
  result.spec_ = std::move(*spec);
  return result;
}

namespace {
// ------------------------------------------------------------- MTP drafter

// The drafter's canonical tensor for each of the checkpoint's mtp.* roles
// (kRules' auxiliary entries, whose descriptors stay pinned there): the
// names the converted drafter has (spec/mappings/qwen38.h), in the target's
// native forms.
struct MtpRole {
  std::string_view source, canonical;
  PartAssemblyKind assembly = PartAssemblyKind::kCopy;
};
constexpr std::array kMtpRoles = {
    MtpRole{"mtp.fc_embedding.weight", "fc_embd.weight"},
    MtpRole{"mtp.fc_hidden.weight", "fc_hidden.weight"},
    MtpRole{"mtp.pre_fc_norm_embedding.weight", "norm_embd.weight"},
    MtpRole{"mtp.pre_fc_norm_hidden.weight", "norm_hidden.weight"},
    MtpRole{"mtp.hyper_connection_mixer.hc_norm.weight", "output_hc_norm.weight"},
    MtpRole{"mtp.hyper_connection_mixer.input_mix_weight_down.weight", "output_hc_down.weight"},
    MtpRole{"mtp.hyper_connection_mixer.input_mix_weight_up.weight", "output_hc_up.weight"},
    MtpRole{"mtp.layers.{index}.attn_hyper_connection.block_inject_weight.weight",
            "blk.{layer}.hc_attn_inject.weight"},
    MtpRole{"mtp.layers.{index}.attn_hyper_connection.hc_norm.weight",
            "blk.{layer}.hc_attn_norm.weight"},
    MtpRole{"mtp.layers.{index}.attn_hyper_connection.input_mix_weight_down.weight",
            "blk.{layer}.hc_attn_down.weight"},
    MtpRole{"mtp.layers.{index}.attn_hyper_connection.input_mix_weight_up.weight",
            "blk.{layer}.hc_attn_up.weight"},
    MtpRole{"mtp.layers.{index}.mlp_hyper_connection.block_inject_weight.weight",
            "blk.{layer}.hc_ffn_inject.weight"},
    MtpRole{"mtp.layers.{index}.mlp_hyper_connection.hc_norm.weight",
            "blk.{layer}.hc_ffn_norm.weight"},
    MtpRole{"mtp.layers.{index}.mlp_hyper_connection.input_mix_weight_down.weight",
            "blk.{layer}.hc_ffn_down.weight"},
    MtpRole{"mtp.layers.{index}.mlp_hyper_connection.input_mix_weight_up.weight",
            "blk.{layer}.hc_ffn_up.weight"},
    MtpRole{"mtp.layers.{index}.self_attn.q_proj.weight", "blk.{layer}.attn_q.weight"},
    MtpRole{"mtp.layers.{index}.self_attn.k_proj.weight", "blk.{layer}.attn_k.weight"},
    MtpRole{"mtp.layers.{index}.self_attn.v_proj.weight", "blk.{layer}.attn_v.weight"},
    MtpRole{"mtp.layers.{index}.self_attn.o_proj.weight", "blk.{layer}.attn_output.weight"},
    MtpRole{"mtp.layers.{index}.self_attn.q_norm.weight", "blk.{layer}.attn_q_norm.weight"},
    MtpRole{"mtp.layers.{index}.self_attn.k_norm.weight", "blk.{layer}.attn_k_norm.weight"},
    MtpRole{"mtp.layers.{index}.self_attn.indexer.index_qk_proj.weight",
            "blk.{layer}.indexer.qk_proj.weight"},
    MtpRole{"mtp.layers.{index}.self_attn.indexer.q_layernorm.weight",
            "blk.{layer}.indexer.q_norm.weight"},
    MtpRole{"mtp.layers.{index}.self_attn.indexer.k_layernorm.weight",
            "blk.{layer}.indexer.k_norm.weight"},
    MtpRole{"mtp.layers.{index}.mlp.gate.weight", "blk.{layer}.ffn_gate_inp.weight"},
    MtpRole{"mtp.layers.{index}.mlp.shared_expert_gate.weight",
            "blk.{layer}.ffn_gate_inp_shexp.weight", PartAssemblyKind::kDenseReshape},
    MtpRole{"mtp.layers.{index}.mlp.shared_expert.gate_proj.weight",
            "blk.{layer}.ffn_gate_shexp.weight"},
    MtpRole{"mtp.layers.{index}.mlp.shared_expert.up_proj.weight",
            "blk.{layer}.ffn_up_shexp.weight"},
    MtpRole{"mtp.layers.{index}.mlp.shared_expert.down_proj.weight",
            "blk.{layer}.ffn_down_shexp.weight"},
    MtpRole{"mtp.layers.{index}.mlp.experts.{index}.gate_proj.weight",
            "blk.{layer}.ffn_gate_exps.weight", PartAssemblyKind::kMatrices},
    MtpRole{"mtp.layers.{index}.mlp.experts.{index}.gate_proj.weight_scale",
            "blk.{layer}.ffn_gate_exps.weight", PartAssemblyKind::kMatrices},
    MtpRole{"mtp.layers.{index}.mlp.experts.{index}.gate_proj.weight_scale_2",
            "blk.{layer}.ffn_gate_exps.weight", PartAssemblyKind::kMatrices},
    MtpRole{"mtp.layers.{index}.mlp.experts.{index}.up_proj.weight",
            "blk.{layer}.ffn_up_exps.weight", PartAssemblyKind::kMatrices},
    MtpRole{"mtp.layers.{index}.mlp.experts.{index}.up_proj.weight_scale",
            "blk.{layer}.ffn_up_exps.weight", PartAssemblyKind::kMatrices},
    MtpRole{"mtp.layers.{index}.mlp.experts.{index}.up_proj.weight_scale_2",
            "blk.{layer}.ffn_up_exps.weight", PartAssemblyKind::kMatrices},
    MtpRole{"mtp.layers.{index}.mlp.experts.{index}.down_proj.weight",
            "blk.{layer}.ffn_down_exps.weight", PartAssemblyKind::kMatrices},
    MtpRole{"mtp.layers.{index}.mlp.experts.{index}.down_proj.weight_scale",
            "blk.{layer}.ffn_down_exps.weight", PartAssemblyKind::kMatrices},
    MtpRole{"mtp.layers.{index}.mlp.experts.{index}.down_proj.weight_scale_2",
            "blk.{layer}.ffn_down_exps.weight", PartAssemblyKind::kMatrices},
};
// 29 tensors and 512 experts' three parts of three projections.
constexpr std::size_t kMtpSources = 29 + 9 * 512;
static_assert(kMtpRoles.size() == 38);
constexpr std::uint64_t kHeadWidth = 2560, kVocab = kModelOptQwen38Vocab;
constexpr std::string_view kHeadSource = "lm_head.weight";

// A routed copy of an mtp.* auxiliary rule: its canonical name, assembly
// and (a quantized expert's) part.
SourceRule Routed(const SourceRule& aux, const MtpRole& role) {
  SourceRule r = aux;
  r.disposition = SourceDispositionKind::kRouted;
  r.canonical = role.canonical;
  r.assembly = role.assembly;
  r.part = PartRole::kData;
  if (role.assembly == PartAssemblyKind::kMatrices)
    r.part = r.pattern.ends_with(".weight_scale_2") ? PartRole::kTensorScale
             : r.pattern.ends_with(".weight_scale") ? PartRole::kBlockScales
                                                    : PartRole::kCodes;
  return r;
}

// The domain the drafter shares with the target: the container and its
// metadata index, and the config contract.
std::expected<void, Error> DrafterDomain(const Checkpoint& source) {
  if (source.container != "safetensors" || source.architecture != "qwen3_8_flash_next" ||
      source.quantization != "modelopt MIXED_PRECISION" || !source.role.empty() ||
      !source.components.empty() || source.files.empty() || source.files.size() > 64 ||
      source.metadata.size() > 512 || source.tensors.size() > kPhysicalTensors)
    return Bad("outside closed ModelOpt drafter source domain", Rule::kUnsupported);
  for (std::size_t i = 0; i < source.metadata.size(); ++i) {
    const auto& e = source.metadata[i];
    if (e.key.size() > kMaxKeyBytes || (i != 0 && source.metadata[i - 1].key >= e.key))
      return Bad("invalid ModelOpt metadata index", Rule::kBounds);
  }
  if (!Config(source)) return Bad("ModelOpt source configuration differs from drafter contract");
  return {};
}

// The native target as NormalizeModelOptTarget gives it (every routed
// rule's outputs, their parts in order) with the n-gram hash's checked
// constants: the target the drafter's spec composes with at import.
using Carrier = ModelOptTargetForm;
std::expected<Carrier, Error> NativeTargetCarrier() {
  Carrier c;
  c.source = {.architecture = "qwen4exp", .container = "safetensors", .tensors = {}, .folds = {}};
  // The target's outputs as its normalization makes them (each part's
  // member count, not its sources), then its experts' relayout.
  std::vector<TensorAssembly> outputs;
  std::map<std::string, std::size_t, std::less<>> made;
  for (const auto& rule : kRules) {
    if (rule.disposition != SourceDispositionKind::kRouted) continue;
    for (std::uint32_t layer = 0; layer < 48; ++layer) {
      if (!InDomain(rule.first, layer)) continue;
      const auto name = Canonical(rule.canonical, layer);
      const auto format = OutputFormat(rule);
      const auto [at, inserted] = made.emplace(name, outputs.size());
      if (!inserted) {
        if (outputs[at->second].format != format)
          return Bad("native target carrier geometry disagrees");
        continue;
      }
      const auto parts = format::PartsOf(format);
      if (!parts) return Bad("invalid native target carrier geometry");
      TensorAssembly out{.name = name, .format = format};
      for (const auto& part : parts->parts()) {
        // As the target's: a row table's tensor scale is shared.
        const bool shared = rule.assembly == PartAssemblyKind::kRows ||
                            rule.assembly == PartAssemblyKind::kSharedTensorScale;
        const auto kind =
            shared ? (part.role == PartRole::kTensorScale ? PartAssemblyKind::kSharedTensorScale
                                                          : PartAssemblyKind::kRows)
                   : rule.assembly;
        out.parts.push_back(
            {.part = part.role,
             .kind = kind,
             .sources = std::vector<std::uint32_t>(kind == PartAssemblyKind::kMatrices ? 512U
                                                   : kind == PartAssemblyKind::kRows   ? 128U
                                                                                       : 1U,
                                                   0U)});
      }
      outputs.push_back(std::move(out));
      if (rule.first == Domain::kNone) break;
    }
  }
  if (auto relaid = RelayoutRoutedExperts(outputs); !relaid) return std::unexpected(relaid.error());
  if (auto table = RelayoutSideTable(outputs); !table) return std::unexpected(table.error());
  auto planned = FoldTargetParameters(outputs, true);
  if (!planned) return std::unexpected(planned.error());
  c.source.folds = FoldNames(*planned);
  auto tensors = CanonicalOf(outputs);
  if (!tensors) return std::unexpected(tensors.error());
  c.source.tensors = std::move(*tensors);
  c.values = Values(PinnedSemantic(), true, true);
  return c;
}

constexpr std::uint64_t kListLineBytes = 12;  // "248319\r\n" and slack, as v0's bound
}  // namespace

std::expected<ModelOptTargetForm, Error> ModelOptNativeTargetForm() {
  return NativeTargetCarrier();
}

std::expected<DraftHeadList, Error> ReadDraftHeadList(const std::filesystem::path& path,
                                                      std::uint32_t vocab) {
  if (vocab == 0 || vocab > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
    return Bad("invalid draft head vocabulary", Rule::kBounds);
  const auto name = path.filename().string();
  if (name.empty() || name.size() > 255 || name == "." || name == ".." ||
      !std::ranges::all_of(name, [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '_' || c == '.' || c == '-';
      }))
    return Bad("the draft head list's file name is outside the artifact vocabulary",
               Rule::kUnsupported);
  auto opened = OpenCheckpointFile(path, 0);
  if (!opened) return std::unexpected(opened.error());
  auto text = ReadDocument(*opened, kListLineBytes * vocab, 0);
  if (!text) return std::unexpected(text.error());
  if (text->empty()) return Bad("the draft head list is empty");
  DraftHeadList out;
  std::string_view rest = *text;
  if (rest.ends_with('\n')) rest.remove_suffix(1);
  while (true) {
    const auto end = rest.find('\n');
    auto line = rest.substr(0, end);
    if (line.ends_with('\r')) line.remove_suffix(1);
    std::uint64_t id = 0;
    const auto parsed = std::from_chars(line.data(), line.data() + line.size(), id);
    if (line.empty() || line.size() > 10 || parsed.ec != std::errc{} ||
        parsed.ptr != line.data() + line.size() ||
        !std::ranges::all_of(line, [](char c) { return c >= '0' && c <= '9'; }))
      return Bad("the draft head list needs one decimal token ID a line");
    if (id >= vocab) return Bad("a draft head token ID is outside the vocabulary");
    if (!out.ids.empty() && static_cast<std::uint64_t>(out.ids.back()) >= id)
      return Bad("draft head token IDs must be strictly ascending");
    if (out.ids.size() == vocab) return Bad("the draft head list has more IDs than the vocabulary");
    out.ids.push_back(static_cast<std::int32_t>(id));
    if (end == std::string_view::npos) break;
    rest.remove_prefix(end + 1);
  }
  out.file = {.name = name,
              .path = opened->path,
              .bytes = opened->bytes,
              .device = opened->device,
              .inode = opened->inode,
              .parsed = {opened->bytes, base::Sha256().Update(*text).Finish()}};
  return out;
}

std::expected<NormalizationCost, Error> ModelOptDrafterBound(const Checkpoint& source,
                                                             std::size_t head_ids) {
  if (auto ok = DrafterDomain(source); !ok) return std::unexpected(ok.error());
  if (head_ids > kVocab) return Bad("a draft head of more rows than the vocabulary", Rule::kBounds);
  // The target's ledger covers the carrier's synthesis and the composition
  // (a closed domain of at most 48 layers); the head adds a source tensor,
  // a member and its name per selected row at most, and its token map.
  auto cost = Cost();
  const std::uint64_t per_row = sizeof(Tensor) + sizeof(IndexedMember) + 4 * sizeof(std::uint32_t) +
                                2 * (kMaxNameBytes + 1) + sizeof(SourceDisposition) + 64 + 8;
  cost.metadata_bytes += head_ids * per_row + kListLineBytes * head_ids + (std::uint64_t{1} << 20);
  cost.work_units += head_ids * 64 * 32;
  cost.payload_bytes = 0;
  cost.layers = 1;
  cost.metadata_bytes += spec::kMaxSpecBytes * 32ULL;
  return cost;
}

std::expected<ModelOptDrafterPlan, Error> NormalizeModelOptDrafter(
    const Checkpoint& source, std::optional<DraftHeadList> head, NormalizationLimits prepaid,
    const NormalizationTarget* actual_target) {
  if (actual_target == nullptr)
    return Bad("drafter import requires --target TARGET_ID in the selected store");
  if (auto valid = ValidateTarget(*actual_target, spec::mappings::Qwen38MtpNativeMapping()); !valid)
    return std::unexpected(valid.error());
  auto target_vocab = spec::Vocabulary(actual_target->model, actual_target->source);
  if (!target_vocab) return Bad("drafter target vocabulary lacks a valid output head");
  if (head) {
    std::int32_t previous = -1;
    for (const auto id : head->ids) {
      if (id <= previous || id < 0 || static_cast<std::uint32_t>(id) >= *target_vocab)
        return Bad("draft head token IDs exceed the actual target vocabulary or are not ascending");
      previous = id;
    }
  }
  auto cost = ModelOptDrafterBound(source, head ? head->ids.size() : 0);
  if (!cost) return std::unexpected(cost.error());
  if (prepaid.metadata_bytes < cost->metadata_bytes || prepaid.work_units < cost->work_units)
    return Bad("ModelOpt drafter normalization is not prepaid", Rule::kBounds);
  if (head && (head->ids.empty() || head->ids.size() > kVocab ||
               head->file.parsed.bytes != head->file.bytes))
    return Bad("the draft head list was not read whole");
  // The drafter's own sources, by role; the head's when it has one.
  std::array<std::uint32_t, kMtpRoles.size()> counts{};
  std::array<const SourceRule*, kMtpRoles.size()> aux{};
  for (std::size_t r = 0; r < kMtpRoles.size(); ++r) {
    const auto at = std::ranges::find(kRules, kMtpRoles[r].source, &SourceRule::pattern);
    if (at == kRules.end() || at->disposition != SourceDispositionKind::kAuxiliary)
      return Bad("drafter role has no pinned source descriptor");
    aux[r] = &*at;
  }
  std::vector<std::pair<std::uint32_t, std::size_t>> selected;  // source tensor, role
  selected.reserve(kMtpSources);
  std::optional<std::uint32_t> head_tensor;
  std::vector<std::uint8_t> files(source.files.size(), 0);
  for (std::size_t ti = 0; ti < source.tensors.size(); ++ti) {
    const auto& t = source.tensors[ti];
    if (t.name == kHeadSource && head) {
      if (head_tensor) return Bad("duplicate ModelOpt source name");
      if (t.dtype != "BF16" || t.dims != std::vector<std::uint64_t>{kVocab, kHeadWidth} ||
          t.part != PartRole::kData || t.experts ||
          t.format != ir::Dense(Encoding::kBF16, {.ne = {kHeadWidth, kVocab, 1, 1}, .rank = 2}) ||
          t.extent.bytes != kVocab * kHeadWidth * 2 || t.extent.file >= source.files.size())
        return Bad("the target head differs from its pinned descriptor");
      head_tensor = static_cast<std::uint32_t>(ti);
      files[t.extent.file] = 1;
      continue;
    }
    if (!t.name.starts_with("mtp.")) continue;
    std::size_t r = 0;
    std::optional<std::array<std::uint32_t, 2>> at;
    for (; r < kMtpRoles.size(); ++r)
      if ((at = Match(*aux[r], t.name))) break;
    if (r == kMtpRoles.size()) return Bad("unclassified ModelOpt drafter tensor");
    const auto routed = Routed(*aux[r], kMtpRoles[r]);
    if (t.dtype != routed.dtype || t.dims.size() != routed.rank ||
        !std::ranges::equal(t.dims, std::span(routed.dims).first(routed.rank)) ||
        ++counts[r] > routed.count)
      return Bad("ModelOpt drafter descriptor or multiplicity differs");
    if (t.format != SourceFormat(routed) || t.part != routed.part || t.experts ||
        t.extent.file >= source.files.size())
      return Bad("ModelOpt drafter source format differs from its role contract");
    files[t.extent.file] = 1;
    selected.emplace_back(static_cast<std::uint32_t>(ti), r);
  }
  for (std::size_t r = 0; r < kMtpRoles.size(); ++r)
    if (counts[r] != aux[r]->count) return Bad("the drafter's source role set is incomplete");
  if (head && !head_tensor) return Bad("a draft head needs the checkpoint's lm_head.weight");
  if (head)
    for (const auto id : head->ids)
      if (std::cmp_greater_equal(id, source.tensors[*head_tensor].format.shape.ne[1]))
        return Bad("a draft head token ID is outside the checkpoint head's actual rows");
  // The publication's sources: the configuration documents, the files
  // holding what it reads, then the head list.
  Checkpoint sub{.role = {},
                 .container = source.container,
                 .architecture = "qwen4exp-mtp",
                 .quantization = source.quantization};
  std::vector<std::uint32_t> remap(source.files.size(), std::numeric_limits<std::uint32_t>::max());
  for (std::size_t f = 0; f < source.files.size(); ++f) {
    if (!files[f] && source.files[f].name.ends_with(".safetensors")) continue;
    if (source.files[f].parsed.bytes == 0)
      return Bad("a drafter source lacks its parsed receipt", Rule::kConfig);
    remap[f] = static_cast<std::uint32_t>(sub.files.size());
    sub.files.push_back(source.files[f]);
  }
  std::optional<std::uint32_t> list_file;
  if (head) {
    // No file of the checkpoint's, read or not, may share its name.
    if (std::ranges::contains(source.files, head->file.name, &SourceFile::name))
      return Bad("the draft head list's name is a checkpoint file's", Rule::kConfig);
    list_file = static_cast<std::uint32_t>(sub.files.size());
    sub.files.push_back(head->file);
  }
  sub.metadata = source.metadata;
  sub.tensors.reserve(selected.size() + (head ? head->ids.size() : 0));
  for (const auto& [ti, r] : selected) {
    auto t = source.tensors[ti];
    t.extent.file = remap[t.extent.file];
    sub.tensors.push_back(std::move(t));
  }
  // The head: one source tensor a run of consecutive selected rows of
  // lm_head.weight, gathered in the list's (ascending) order.
  std::vector<std::uint32_t> runs;
  if (head) {
    const auto& lm = source.tensors[*head_tensor];
    const auto& ids = head->ids;
    for (std::size_t at = 0; at < ids.size();) {
      std::size_t end = at + 1;
      while (end < ids.size() && ids[end] == ids[end - 1] + 1) ++end;
      const std::uint64_t first = static_cast<std::uint64_t>(ids[at]), rows = end - at;
      runs.push_back(static_cast<std::uint32_t>(sub.tensors.size()));
      sub.tensors.push_back(
          {.name = std::format("{}.rows.{}", kHeadSource, first),
           .format = ir::Dense(Encoding::kBF16, {.ne = {kHeadWidth, rows, 1, 1}, .rank = 2}),
           .part = PartRole::kData,
           .experts = false,
           .extent = {.file = remap[lm.extent.file],
                      .offset = lm.extent.offset + first * kHeadWidth * 2,
                      .bytes = rows * kHeadWidth * 2},
           .dtype = lm.dtype,
           .dims = {rows, kHeadWidth}});
      at = end;
    }
  }
  // Outputs: each role's canonical tensor (the experts' parts as the
  // target's), then the head's rows.
  std::vector<TensorAssembly> outputs;
  std::map<std::string, std::uint32_t, std::less<>> names;
  for (std::size_t i = 0; i < selected.size(); ++i) {
    const auto& t = sub.tensors[i];
    const auto routed = Routed(*aux[selected[i].second], kMtpRoles[selected[i].second]);
    const auto coordinates = Match(routed, t.name);
    if (!coordinates) return Bad("drafter source lost its role");
    const auto canonical = Canonical(routed.canonical, (*coordinates)[0]);
    const auto format = OutputFormat(routed);
    auto found = names.find(canonical);
    if (found == names.end()) {
      auto parts = format::PartsOf(format);
      if (!parts) return Bad("invalid drafter part geometry");
      TensorAssembly out{.name = canonical, .format = format};
      for (const auto& part : parts->parts())
        out.parts.push_back({.part = part.role,
                             .kind = routed.assembly,
                             .sources = std::vector<std::uint32_t>(
                                 routed.assembly == PartAssemblyKind::kMatrices ? 512U : 1U,
                                 std::numeric_limits<std::uint32_t>::max())});
      found = names.emplace(canonical, static_cast<std::uint32_t>(outputs.size())).first;
      outputs.push_back(std::move(out));
    }
    auto& out = outputs[found->second];
    if (out.format != format) return Bad("paired drafter parts disagree on output geometry");
    auto part = std::ranges::find(out.parts, routed.part, &PartAssembly::part);
    if (part == out.parts.end()) return Bad("drafter source part is not an output part");
    const auto ordinal = routed.second == Domain::kExpert ? (*coordinates)[1] : 0U;
    if (ordinal >= part->sources.size() ||
        part->sources[ordinal] != std::numeric_limits<std::uint32_t>::max())
      return Bad("two drafter sources select one member");
    part->sources[ordinal] = static_cast<std::uint32_t>(i);
  }
  // Its routed experts in the scale-atom layout, as the target's.
  auto relaid = RelayoutRoutedExperts(outputs, &actual_target->model);
  if (!relaid) return std::unexpected(relaid.error());
  if (relaid->tables != 2 || relaid->stacked != 1 || relaid->scale_vectors != 3)
    return Bad("the drafter's routed experts' relayout is incomplete");
  const std::uint64_t rows = head ? head->ids.size() : 0;
  if (head)
    outputs.push_back(
        {.name = "draft_output.weight",
         .format = ir::Dense(Encoding::kBF16, {.ne = {kHeadWidth, rows, 1, 1}, .rank = 2}),
         .parts = {{.part = PartRole::kData, .kind = PartAssemblyKind::kRows, .sources = runs}}});
  // Its parameter folds, by role (import/parameter_folds.h): its stage
  // composed from its unfolded outputs with the target's carrier decides
  // them; its zero-centered norms' 1 + w in F32, as the target's.
  const auto& target_mapping = *actual_target->mapping;
  const auto* target = &actual_target->model;
  const auto ids_tensor = [&] {
    return spec::SourceTensor{
        .name = "draft_output.ids",
        .format = ir::Dense(Encoding::kI32, {.ne = {1, rows, 1, 1}, .rank = 2}),
        .part = PartRole::kData,
        .experts = false};
  };
  auto value_tensors = CanonicalOf(outputs);
  if (!value_tensors) return std::unexpected(value_tensors.error());
  if (head) value_tensors->push_back(ids_tensor());
  auto values =
      spec::ExtractValues(spec::mappings::Qwen38MtpNativeMapping(), *target, actual_target->source,
                          {.architecture = "qwen4exp-mtp",
                           .container = "safetensors",
                           .tensors = std::move(*value_tensors),
                           .folds = {}});
  if (!values) return Bad("native ModelOpt drafter values cannot be extracted from the target");
  std::vector<PlannedPartFold> planned;
  {
    auto tensors = CanonicalOf(outputs);
    if (!tensors) return std::unexpected(tensors.error());
    if (head) tensors->push_back(ids_tensor());
    const spec::Source unfolded{.architecture = "qwen4exp-mtp",
                                .container = "safetensors",
                                .tensors = std::move(*tensors),
                                .folds = {}};
    auto composed = spec::ComposeDrafter(
        *target, target_mapping, spec::mappings::Qwen38MtpNativeMapping(), unfolded, *values);
    if (!composed) return Bad("native ModelOpt drafter composition refused");
    const auto stage =
        std::ranges::find(composed->stages, spec::StageRole::kDrafter, &spec::Stage::role);
    if (stage == composed->stages.end()) return Bad("native ModelOpt drafter has no stage");
    auto applied = FoldDerivedParameters(*stage, outputs);
    if (!applied) return std::unexpected(applied.error());
    if (applied->size() != 1 || applied->front().fold != "norm.plus_one")
      return Bad("the drafter's norms did not fold whole");
    planned.push_back({std::move(applied->front().fold), std::move(applied->front().outputs)});
  }
  std::vector<SourceDisposition> dispositions(sub.tensors.size());
  auto indexed = IndexedPartPlan::Bound(sub, outputs);
  if (!indexed) return std::unexpected(indexed.error());
  if (indexed->metadata_bytes > cost->metadata_bytes || indexed->work_units > cost->work_units)
    return Bad("indexed drafter plan exceeds its prepaid domain", Rule::kBounds);
  auto parts = IndexedPartPlan::Make(std::move(sub), outputs, dispositions, *indexed);
  if (!parts) return std::unexpected(parts.error());
  ModelOptDrafterPlan result;
  result.cost_ = *cost;
  result.parts_.emplace(std::move(*parts));
  result.planned_folds_ = std::move(planned);
  result.canonical_ = {.architecture = "qwen4exp-mtp",
                       .container = "safetensors",
                       .tensors = {},
                       .folds = FoldNames(result.planned_folds_)};
  std::uint32_t output_index = 0;
  for (const auto& out : result.parts_->outputs()) {
    for (const auto& part : out.parts.parts()) {
      result.bindings_.push_back({output_index, part.role});
      result.canonical_.tensors.push_back({.name = PartName(out.name, part.role),
                                           .format = out.format,
                                           .part = part.role,
                                           .experts = out.format.shape.rank == 3});
    }
    ++output_index;
  }
  if (head) {
    // The token map: I32 [1, rows], little-endian, from the list's bytes.
    GeneratedPart ids{.tensor = static_cast<std::uint32_t>(result.canonical_.tensors.size()),
                      .file = *list_file,
                      .bytes = {}};
    ids.bytes.reserve(rows * 4);
    for (const auto id : head->ids)
      for (unsigned b = 0; b < 4; ++b)
        ids.bytes.push_back(
            static_cast<std::byte>((static_cast<std::uint32_t>(id) >> (8U * b)) & 0xFFU));
    result.canonical_.tensors.push_back(ids_tensor());
    result.generated_.push_back(std::move(ids));
    result.selected_rows_ = rows;
    result.options_ = std::format(R"({{"component":"mtp","draft_head":"{}"}})", head->file.name);
  }
  result.values_ = std::move(*values);
  auto composed =
      spec::ComposeDrafter(*target, target_mapping, spec::mappings::Qwen38MtpNativeMapping(),
                           result.canonical_, result.values_);
  if (!composed) return Bad("native ModelOpt drafter composition refused");
  const std::array<std::span<const spec::SourceTensor>, 2> tables{actual_target->source.tensors,
                                                                  result.canonical_.tensors};
  if (auto valid = components::ValidateSpec(*composed, tables); !valid)
    return Bad("native ModelOpt drafter component validation refused");
  result.spec_ = std::move(*composed);
  return result;
}

std::span<const Normalizer> Normalizers() {
  static const std::array all{
      Normalizer{"safetensors", "qwen3_8_flash_next", "modelopt MIXED_PRECISION",
                 &ModelOptTargetBound, &NormalizeModelOptTarget, &ModelOptDrafterBound,
                 &NormalizeModelOptDrafter},
      Normalizer{"mlx", "qwen4_exp", "mlx affine", &MlxTargetBound, &NormalizeMlxTarget,
                 &MlxDrafterBound, &NormalizeMlxDrafter}};
  return all;
}
const Normalizer* FindNormalizer(const Checkpoint& source) {
  const Normalizer* result = nullptr;
  for (const auto& normalizer : Normalizers()) {
    if (normalizer.container != source.container ||
        normalizer.architecture != source.architecture ||
        normalizer.quantization != source.quantization)
      continue;
    if (result) return nullptr;
    result = &normalizer;
  }
  return result;
}
bool IsIndexedTarget(const Checkpoint& source) { return FindNormalizer(source) != nullptr; }
std::expected<NormalizationCost, Error> IndexedTargetBound(const Checkpoint& source) {
  if (const auto* normalizer = FindNormalizer(source)) return normalizer->target_bound(source);
  return Bad("no indexed normalizer recognizes this checkpoint", Rule::kUnsupported);
}
std::expected<ModelOptTargetPlan, Error> NormalizeIndexedTarget(Checkpoint&& source,
                                                                NormalizationLimits prepaid) {
  if (const auto* normalizer = FindNormalizer(source))
    return normalizer->target(std::move(source), prepaid);
  return Bad("no indexed normalizer recognizes this checkpoint", Rule::kUnsupported);
}

std::expected<WriteResult, Error> WriteArtifact(const ModelOptTargetPlan& plan,
                                                const WriteOptions& options) {
  return publication_detail::WriteIndexed(plan.parts(), plan.canonical(), plan.spec(),
                                          plan.cost().payload_bytes, plan.bindings(),
                                          plan.semantic(), plan.planned_folds(), {}, options);
}

std::expected<WriteResult, Error> WriteArtifact(const ModelOptDrafterPlan& plan,
                                                const WriteOptions& options) {
  if (options.options != plan.options())
    return std::unexpected(Error{.rule = Rule::kConfig,
                                 .reason = "drafter publication options differ from its plan's"});
  return publication_detail::WriteIndexed(plan.parts(), plan.canonical(), plan.spec(),
                                          plan.cost().payload_bytes, plan.bindings(), {},
                                          plan.planned_folds(), plan.generated(), options);
}

}  // namespace llmp::import
