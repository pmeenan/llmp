// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/exl3/qwen2.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "base/sha256.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/exl3/implementations.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/validate.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen2_exl3.h"

namespace llmp::kernels::exl3 {
namespace {

using model::Exl3Op;
using model::Exl3Owner;
using model::Exl3Path;

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

std::unexpected<KernelFailure> From(const ggml::KernelFailure& failure) {
  return std::unexpected(KernelFailure{.error = failure.error == ggml::KernelError::kRejected
                                                    ? KernelError::kRejected
                                                    : KernelError::kUnknown,
                                       .detail = failure.detail});
}

// The registry operation each implementation the plan names performs.
std::optional<execution::Operation> OperationOf(std::string_view implementation) {
  using execution::Operation;
  static const std::map<std::string_view, Operation, std::less<>> kOperations = {
      {"ggml.get_rows", Operation::kGetRows},
      {"ggml.rms_norm_mul.fused", Operation::kRmsNormMul},
      {"ggml.convert", Operation::kConvert},
      {"ggml.rope.neox", Operation::kRope},
      {"ggml.flash_attn_ext.vec", Operation::kFlashAttn},
      {"ggml.add", Operation::kAdd},
      {"ggml.swiglu", Operation::kSwiGlu},
      {"exl3.linear.gemm", Operation::kQuantLinear},
      {"exl3.linear.gemv", Operation::kQuantLinear},
      {"exl3.linear.reconstruct", Operation::kQuantLinear},
      {"exl3.linear.reconstruct_fused", Operation::kQuantLinear},
      {"exl3.multi_linear.mgemm", Operation::kQuantMultiLinear},
      {"exl3.bias_add", Operation::kBiasAdd},
  };
  const auto found = kOperations.find(implementation);
  return found == kOperations.end() ? std::nullopt : std::optional(found->second);
}

constexpr std::uint64_t kRound = 256;
std::uint64_t RoundUp(std::uint64_t bytes) { return (bytes + kRound - 1) / kRound * kRound; }

// IEEE binary16 negative infinity and zero, the mask's two values.
constexpr std::uint16_t kHalfZero = 0x0000;
constexpr std::uint16_t kHalfMinusInf = 0xFC00;

}  // namespace

struct Qwen2Program::Step {
  enum class Kind : std::uint8_t {
    kInputs,
    kGgml,
    kNorm,
    kGemm,
    kGemv,
    kRecon,
    kMulti,
    kBias,
    kCopy
  };
  Kind kind = Kind::kGgml;
  std::size_t op = 0;
  std::optional<ggml::Kernel> ggml_kernel;
  std::optional<ggml::RmsNormMulKernel> norm_kernel;
  std::optional<Kernel> exl3_kernel;
  std::vector<ggml_tensor*> nodes;
  LinearOperands linear;
  GemmPlan gemm;
  GemvPlan gemv;
  MultiLinearOperands multi;
  MultiGemmPlan multi_plan;
  ReconstructedOperands recon;
  std::vector<LtAlgorithm> algorithms;
  BiasOperands bias;
  std::uint64_t destination = 0;  // a copy's
  std::uint64_t source = 0;
  std::uint64_t bytes = 0;
  // The weights it reads, at the addresses bound (BoundWeights).
  std::vector<std::pair<std::string, std::uint64_t>> weights;

  // The kernel its kind runs, bound when the step is made.
  const ggml::Kernel& Ggml() const {
    if (!ggml_kernel) {
      base::Fatal("a GGML step without its kernel");
    }
    return *ggml_kernel;
  }
  const ggml::RmsNormMulKernel& Norm() const {
    if (!norm_kernel) {
      base::Fatal("a norm step without its kernel");
    }
    return *norm_kernel;
  }
  const Kernel& Exl3() const {
    if (!exl3_kernel) {
      base::Fatal("an EXL3 step without its kernel");
    }
    return *exl3_kernel;
  }
};

struct Qwen2Program::State {
  model::Exl3PhasePlan plan;
  Qwen2Memory memory;
  std::optional<execution::BoundPlan> bound;
  std::optional<ggml::TensorArena> arena;
  std::vector<Step> steps;
  HostInputs inputs;
  std::uint64_t cell_bytes = 0;  // one cache cell of one layer's K (or V)
};

Qwen2Program::~Qwen2Program() = default;

const execution::BoundPlan& Qwen2Program::bound() const {
  const std::optional<execution::BoundPlan>& plan = state_->bound;
  if (!plan) {
    base::Fatal("a program without its bound plan");
  }
  return *plan;
}

HostInputs HostInputsLayout(const model::Exl3PhasePlan& plan) {
  const auto rows = static_cast<std::uint64_t>(plan.phase.rows);
  HostInputs out;
  out.ids = 0;
  out.positions = RoundUp(rows * sizeof(std::int32_t));
  out.mask = out.positions + RoundUp(rows * sizeof(std::int32_t));
  out.bytes = out.mask + RoundUp(rows * static_cast<std::uint64_t>(plan.padded) * 2);
  return out;
}

std::expected<void, KernelFailure> WriteHostInputs(const model::Qwen2Profile& profile,
                                                   const model::Exl3PhasePlan& plan,
                                                   std::span<const std::int32_t> tokens,
                                                   std::span<std::byte> out) {
  const HostInputs layout = HostInputsLayout(plan);
  const int rows = plan.phase.rows;
  if (tokens.size() != static_cast<std::size_t>(rows) || out.size() < layout.bytes) {
    return Rejected("host inputs of another phase");
  }
  for (const std::int32_t token : tokens) {
    if (token < 0 || std::cmp_greater_equal(token, profile.vocab)) {
      return Rejected(std::format("token {} is outside the vocabulary", token));
    }
  }
  std::ranges::fill(out, std::byte{0});
  std::memcpy(out.data() + layout.ids, tokens.data(), tokens.size_bytes());
  for (int i = 0; i < rows; ++i) {
    const std::int32_t position = plan.phase.past + i;
    std::memcpy(out.data() + layout.positions + (4 * static_cast<std::size_t>(i)), &position, 4);
    for (int column = 0; column < plan.padded; ++column) {
      const std::uint16_t value = column <= position ? kHalfZero : kHalfMinusInf;
      std::memcpy(out.data() + layout.mask +
                      (2 * ((static_cast<std::size_t>(i) * static_cast<std::size_t>(plan.padded)) +
                            static_cast<std::size_t>(column))),
                  &value, 2);
    }
  }
  return {};
}

std::uint64_t Qwen2Program::Address(std::size_t op, std::string_view tensor) const {
  const State& s = *state_;
  const Exl3Op& o = s.plan.ops.at(op);
  if (tensor == "k_cache" || tensor == "v_cache") {
    if (o.layer < 0) {
      return 0;
    }
    const Qwen2Layer& layer = s.memory.layers.at(static_cast<std::size_t>(o.layer));
    const std::uint64_t base = tensor == "k_cache" ? layer.k_cache : layer.v_cache;
    // A KV write's cells start at the phase's position; attention reads
    // from cell 0.
    return o.name.starts_with("kv_write")
               ? base + (static_cast<std::uint64_t>(s.plan.phase.past) * s.cell_bytes)
               : base;
  }
  if (const auto slot = s.plan.slots.find(tensor); slot != s.plan.slots.end()) {
    return s.memory.region + slot->second.offset;
  }
  for (const auto& [name, address] : BoundWeights(op)) {
    if (name == tensor) {
      return address;
    }
  }
  return 0;
}

std::span<const std::pair<std::string, std::uint64_t>> Qwen2Program::BoundWeights(
    std::size_t op) const {
  const State& s = *state_;
  if (op < s.steps.size() && s.steps[op].op == op) {
    return s.steps[op].weights;
  }
  return {};
}

std::expected<void, KernelFailure> WriteMultiGemmTables(providers::DeviceExecution& execution,
                                                        providers::StreamId stream,
                                                        std::span<std::byte> staging,
                                                        Qwen2Memory& memory) {
  constexpr std::uint64_t kLayerBytes = 48;
  constexpr std::uint64_t kTableBytes = 16;
  for (Qwen2Layer& layer : memory.layers) {
    layer.tables_written = {};
  }
  if (staging.size() < memory.layers.size() * kLayerBytes) {
    return Rejected("the tables' staging is smaller than 48 bytes per layer");
  }
  const auto base = reinterpret_cast<std::uintptr_t>(staging.data());
  for (std::size_t l = 0; l < memory.layers.size(); ++l) {
    Qwen2Layer& layer = memory.layers[l];
    const std::array<std::uint64_t, 6> words =
        MultiGemmTables(layer.gate.weights, layer.up.weights);
    std::memcpy(staging.data() + (l * kLayerBytes), words.data(), kLayerBytes);
    const std::array<std::uint64_t, 3> tables = {layer.trellis_table, layer.suh_table,
                                                 layer.svh_table};
    for (std::size_t t = 0; t < tables.size(); ++t) {
      if (auto copied =
              execution.Copy(stream, tables[t], base + (l * kLayerBytes) + (t * kTableBytes),
                             base::Bytes(kTableBytes));
          !copied) {
        return std::unexpected(KernelFailure{
            .error = KernelError::kUnknown,
            .detail = std::format("layer {}'s multi-GEMM tables: {}", l, copied.error().detail)});
      }
    }
    layer.tables_written = words;
  }
  return {};
}

std::expected<std::unique_ptr<Qwen2Program>, KernelFailure> Qwen2Program::Bind(
    const execution::Registry& registry, const model::Qwen2Profile& p,
    const model::Exl3PhasePlan& plan, const Qwen2Memory& memory, ggml::LaunchContext& ggml_launch,
    LaunchContext& launch) {
  if (memory.layers.size() != p.layers || plan.region > memory.region_bytes ||
      memory.region % kRound != 0 || plan.padded > memory.cells ||
      plan.phase.past + plan.phase.rows > memory.cells) {
    return Rejected(
        "memory of another profile, a region smaller than the plan's, or a cache "
        "too small for the phase");
  }
  std::unique_ptr<Qwen2Program> program(new Qwen2Program());
  program->state_ = std::make_unique<State>();
  State& s = *program->state_;
  s.plan = plan;
  s.memory = memory;
  s.inputs = HostInputsLayout(plan);
  s.cell_bytes = std::uint64_t{p.kv_width()} * 2;

  // The registry plan: every operation that launches a kernel, in order.
  std::vector<execution::Choice> choices;
  for (const Exl3Op& o : s.plan.ops) {
    if (o.implementation.empty()) {
      continue;
    }
    const auto operation = OperationOf(o.implementation);
    if (!operation) {
      return Rejected(std::format("{}: no operation known for {}", o.name, o.implementation));
    }
    choices.push_back({.operation = *operation, .implementation = o.implementation});
  }
  auto registry_plan = execution::Plan::Build(registry, choices);
  if (!registry_plan) {
    return Rejected("the registry refused the plan: " + registry_plan.error().detail);
  }
  auto bound = execution::Resolve(registry, *registry_plan);
  if (!bound) {
    return Rejected("the plan did not resolve: " + bound.error().detail);
  }
  s.bound = std::move(*bound);
  {
    base::Sha256 hash;
    hash.Update("llmp.exl3.qwen2.plan.v0");
    hash.Update(std::as_bytes(std::span(s.bound->identity())));
    hash.Update(std::as_bytes(std::span(s.plan.launch_digest)));
    program->identity_ = hash.Finish();
  }

  const std::size_t tensors = 64 * (static_cast<std::size_t>(p.layers) + 2);
  auto arena = ggml::TensorArena::Create(tensors);
  if (!arena) {
    return From(arena.error());
  }
  s.arena.emplace(std::move(*arena));
  if (auto room = s.arena->Reserve(tensors); !room) {
    return From(room.error());
  }
  ggml_context* c = s.arena->context();
  const auto at = [&](std::string_view tensor) -> std::uint64_t {
    return memory.region + s.plan.slots.at(std::string(tensor)).offset;
  };
  const auto leaf = [&](ggml_type type, std::array<std::int64_t, 3> ne, std::uint64_t address) {
    ggml_tensor* t = ggml_new_tensor_3d(c, type, ne[0], ne[1], ne[2]);
    ggml::TensorArena::Bind(t, address);
    return t;
  };
  const std::int64_t n = plan.phase.rows;
  const std::int64_t width = p.width;
  const std::int64_t head = p.head_dim;
  const std::int64_t heads = p.heads;
  const std::int64_t kv_heads = p.kv_heads;
  const std::int64_t ffn = p.ffn;
  const auto type_of = [&](std::string_view tensor) {
    const model::Exl3Tensor& t = s.plan.tensors.at(std::string(tensor));
    switch (t.dtype) {
      case model::Exl3Dtype::kF32:
        return GGML_TYPE_F32;
      case model::Exl3Dtype::kF16:
        return GGML_TYPE_F16;
      case model::Exl3Dtype::kBF16:
        return GGML_TYPE_BF16;
      case model::Exl3Dtype::kI32:
        return GGML_TYPE_I32;
    }
    return GGML_TYPE_F32;
  };
  // Every region tensor lies inside the region.
  for (const auto& [name, slot] : s.plan.slots) {
    if (slot.offset + slot.bytes > s.plan.region) {
      return Rejected(std::format("{} lies outside the plan's region", name));
    }
  }

  std::size_t choice = 0;
  for (std::size_t index = 0; index < s.plan.ops.size(); ++index) {
    const Exl3Op& o = s.plan.ops[index];
    Step step;
    step.op = index;
    const Qwen2Layer* layer =
        o.layer >= 0 ? &memory.layers.at(static_cast<std::size_t>(o.layer)) : nullptr;
    const execution::Implementation* implementation =
        o.implementation.empty() ? nullptr : &s.bound->at(choice++);
    const auto bind_ggml = [&]() -> std::expected<void, KernelFailure> {
      auto kernel = ggml::Kernel::Bind(*implementation);
      if (!kernel) {
        return From(kernel.error());
      }
      step.ggml_kernel = *kernel;
      return {};
    };
    std::expected<void, KernelFailure> made;
    if (o.owner == Exl3Owner::kHost) {
      step.kind = Step::Kind::kInputs;
    } else if (o.owner == Exl3Owner::kCopy) {
      step.kind = Step::Kind::kCopy;
      const bool k = o.name == "kv_write.k";
      step.source = at(o.inputs.at(0));
      step.destination = (k ? layer->k_cache : layer->v_cache) +
                         (static_cast<std::uint64_t>(plan.phase.past) * s.cell_bytes);
      step.bytes = static_cast<std::uint64_t>(n * kv_heads * head) * 2;
    } else if (o.implementation == "ggml.rms_norm_mul.fused") {
      step.kind = Step::Kind::kNorm;
      auto kernel = ggml::RmsNormMulKernel::Bind(*implementation);
      if (!kernel) {
        return From(kernel.error());
      }
      step.norm_kernel = *kernel;
      const std::uint64_t weight = [&] {
        if (o.name == "attn_norm") {
          return layer->attn_norm;
        }
        return o.name == "mlp_norm" ? layer->mlp_norm : memory.final_norm;
      }();
      ggml_tensor* x = leaf(GGML_TYPE_F32, {width, n, 1}, at(o.inputs.at(0)));
      ggml_tensor* w = leaf(GGML_TYPE_F32, {width, 1, 1}, weight);
      step.weights = {{o.inputs.at(1), weight}};
      ggml_tensor* norm = ggml_rms_norm(c, x, p.rms_eps);
      ggml_tensor* mul = ggml_mul(c, norm, w);
      ggml::TensorArena::Bind(mul, at(o.outputs.at(0)));
      step.nodes = {norm, mul};
      if (auto checked = step.Norm().Check(norm, mul); !checked) {
        return From(checked.error());
      }
    } else if (o.owner == Exl3Owner::kGgml) {
      step.kind = Step::Kind::kGgml;
      made = bind_ggml();
      if (!made) {
        return std::unexpected(made.error());
      }
      ggml_tensor* node = nullptr;
      if (o.implementation == "ggml.get_rows") {
        ggml_tensor* table =
            leaf(GGML_TYPE_BF16, {width, static_cast<std::int64_t>(p.vocab), 1}, memory.embed);
        ggml_tensor* ids = leaf(GGML_TYPE_I32, {n, 1, 1}, at("ids"));
        node = ggml_get_rows(c, table, ids);
        ggml::TensorArena::Bind(node, at(o.outputs.at(0)));
        step.weights = {{o.inputs.at(0), memory.embed}};
      } else if (o.implementation == "ggml.convert") {
        // Both sides as [head, heads-or-rows...]: every cast is between
        // packed tensors of one element count; a row-major [rows, width]
        // tensor is the same memory as [rows, heads, head].
        const std::string& in = o.inputs.at(0);
        const std::string& out = o.outputs.at(0);
        const std::uint64_t elements =
            s.plan.tensors.at(in).bytes() / model::Exl3DtypeBytes(s.plan.tensors.at(in).dtype);
        if (elements !=
            s.plan.tensors.at(out).bytes() / model::Exl3DtypeBytes(s.plan.tensors.at(out).dtype)) {
          return Rejected(std::format("{}: a cast between tensors of different sizes", o.name));
        }
        const auto rows = static_cast<std::int64_t>(elements) / n;
        ggml_tensor* src = leaf(type_of(in), {rows, n, 1}, at(in));
        ggml_tensor* dst = leaf(type_of(out), {rows, n, 1}, at(out));
        node = ggml_cpy(c, src, dst);
      } else if (o.implementation == "ggml.rope.neox") {
        const std::int64_t h = o.name == "rope_q" ? heads : kv_heads;
        ggml_tensor* x = leaf(GGML_TYPE_F32, {head, h, n}, at(o.inputs.at(0)));
        ggml_tensor* positions = leaf(GGML_TYPE_I32, {n, 1, 1}, at("positions"));
        // As llm_graph_context calls it for Qwen2 (the record's rope
        // parameters): NEOX over the whole head, the model's base and
        // original context, no frequency scaling and no YaRN.
        node = ggml_rope_ext(c, x, positions, nullptr, static_cast<int>(head), GGML_ROPE_TYPE_NEOX,
                             static_cast<int>(p.train_context), p.rope_base, 1.0f, 0.0f, 1.0f,
                             32.0f, 1.0f);
        ggml::TensorArena::Bind(node, at(o.outputs.at(0)));
      } else if (o.implementation == "ggml.flash_attn_ext.vec") {
        ggml_tensor* q = leaf(GGML_TYPE_F32, {head, heads, n}, at("q_rope"));
        ggml_tensor* k = leaf(GGML_TYPE_F16, {head, kv_heads, plan.padded}, layer->k_cache);
        ggml_tensor* v = leaf(GGML_TYPE_F16, {head, kv_heads, plan.padded}, layer->v_cache);
        ggml_tensor* mask = leaf(GGML_TYPE_F16, {plan.padded, n, 1}, at("mask"));
        node = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3),
                                   ggml_permute(c, k, 0, 2, 1, 3), ggml_permute(c, v, 0, 2, 1, 3),
                                   mask, 1.0f / std::sqrt(static_cast<float>(head)), 0.0f, 0.0f);
        // As llama.cpp sets it (no CUDA kernel reads it): op_params[3].
        if (!ggml_prec_set_acc(node, GGML_PREC_F32)) {
          return Rejected("attention's precision was not set");
        }
        ggml::TensorArena::Bind(node, at(o.outputs.at(0)));
        auto attention = ggml::PlanFlashAttnVec(ggml_launch, node);
        if (!attention) {
          return From(attention.error());
        }
        program->ggml_scratch_ = std::max(program->ggml_scratch_, attention->scratch);
      } else if (o.implementation == "ggml.add") {
        ggml_tensor* a = leaf(GGML_TYPE_F32, {width, n, 1}, at(o.inputs.at(0)));
        ggml_tensor* b = leaf(GGML_TYPE_F32, {width, n, 1}, at(o.inputs.at(1)));
        node = ggml_add(c, a, b);
        ggml::TensorArena::Bind(node, at(o.outputs.at(0)));
      } else if (o.implementation == "ggml.swiglu") {
        ggml_tensor* gate = leaf(GGML_TYPE_F32, {ffn, n, 1}, at("gate"));
        ggml_tensor* up = leaf(GGML_TYPE_F32, {ffn, n, 1}, at("up"));
        node = ggml_swiglu_split(c, gate, up);
        ggml::TensorArena::Bind(node, at(o.outputs.at(0)));
      } else {
        return Rejected(std::format("{}: no GGML operation for {}", o.name, o.implementation));
      }
      step.nodes = {node};
      const std::array<const ggml_tensor*, 1> nodes = {node};
      if (auto checked = step.Ggml().Check(nodes); !checked) {
        return From(ggml::KernelFailure{
            .error = checked.error().error,
            .detail = std::format("{} (layer {}): {}", o.name, o.layer, checked.error().detail)});
      }
    } else if (o.implementation == "exl3.bias_add") {
      step.kind = Step::Kind::kBias;
      auto kernel = Kernel::Bind(*implementation);
      if (!kernel) {
        return std::unexpected(kernel.error());
      }
      step.exl3_kernel = *kernel;
      const Qwen2Linear& linear = [&]() -> const Qwen2Linear& {
        if (o.name == "q_proj.bias_add") {
          return layer->q;
        }
        return o.name == "k_proj.bias_add" ? layer->k : layer->v;
      }();
      step.bias = {.x = at(o.inputs.at(0)),
                   .bias = linear.bias,
                   .y = at(o.outputs.at(0)),
                   .rows = static_cast<int>(n),
                   .columns = linear.weights.n};
      step.weights = {{o.inputs.at(1), linear.bias}};
      if (auto checked = CheckBias(step.bias); !checked) {
        return std::unexpected(checked.error());
      }
    } else {
      // A linear, or gate and up.
      auto kernel = Kernel::Bind(*implementation);
      if (!kernel) {
        return std::unexpected(kernel.error());
      }
      step.exl3_kernel = *kernel;
      const auto linear_of = [&](std::string_view name) -> const Qwen2Linear* {
        if (layer == nullptr) {
          return &memory.lm_head;
        }
        for (const auto& [suffix, linear] :
             {std::pair{"self_attn.q_proj", &layer->q}, std::pair{"self_attn.k_proj", &layer->k},
              std::pair{"self_attn.v_proj", &layer->v}, std::pair{"self_attn.o_proj", &layer->o},
              std::pair{"mlp.gate_proj", &layer->gate}, std::pair{"mlp.up_proj", &layer->up},
              std::pair{"mlp.down_proj", &layer->down}}) {
          if (name.ends_with(suffix)) {
            return linear;
          }
        }
        return nullptr;
      };
      const Qwen2Linear* linear = linear_of(o.linear);
      if (linear == nullptr) {
        return Rejected(std::format("{}: no weights for {}", o.name, o.linear));
      }
      if (o.launch.path != Exl3Path::kMulti) {
        step.weights = {{"trellis", linear->weights.trellis},
                        {"suh", linear->weights.suh},
                        {"svh", linear->weights.svh}};
      }
      const model::Exl3Tensor& out = s.plan.tensors.at(o.outputs.at(0));
      const Output output = out.dtype == model::Exl3Dtype::kF32 ? Output::kF32 : Output::kF16;
      const int m = static_cast<int>(n);
      switch (o.launch.path) {
        case Exl3Path::kGemm:
        case Exl3Path::kGemv: {
          step.linear = {.weights = linear->weights,
                         .x = at(o.inputs.at(0)),
                         .a_had = at(o.name + ".a_had"),
                         .y = at(o.outputs.at(0)),
                         .output = output,
                         .m = m};
          if (ScratchBytes(step.linear) > s.plan.slots.at(o.name + ".a_had").bytes) {
            return Rejected(
                std::format("{}: the plan's scratch is smaller than the launcher's", o.name));
          }
          auto upstream = launch.UpstreamGemv(linear->weights, output, m);
          if (!upstream) {
            return std::unexpected(upstream.error());
          }
          if (o.launch.path == Exl3Path::kGemm) {
            step.kind = Step::Kind::kGemm;
            step.gemm = {.shape = o.launch.shape, .blocks = o.launch.blocks};
            if (plan.arm == model::Exl3Arm::kO && upstream->has_value()) {
              return Rejected(
                  std::format("{} (layer {}): upstream takes the GEMV here", o.name, o.layer));
            }
            auto coresident = launch.GemmCoresident(linear->weights.bits, o.launch.shape, output);
            if (!coresident) {
              return std::unexpected(coresident.error());
            }
            if (auto checked = CheckGemm(step.linear, step.gemm, *coresident, launch.locks());
                !checked) {
              return std::unexpected(checked.error());
            }
          } else {
            step.kind = Step::Kind::kGemv;
            step.gemv = {.config = o.launch.config, .blocks = o.launch.blocks};
            if (!upstream->has_value() || (*upstream)->config != step.gemv.config ||
                (*upstream)->blocks != step.gemv.blocks) {
              return Rejected(std::format("{} (layer {}): the table's GEMV is not upstream's",
                                          o.name, o.layer));
            }
            auto coresident =
                launch.GemvCoresident(linear->weights.bits, output, m, step.gemv.config);
            if (!coresident) {
              return std::unexpected(coresident.error());
            }
            if (auto checked = CheckGemv(step.linear, step.gemv, *coresident, launch.locks());
                !checked) {
              return std::unexpected(checked.error());
            }
          }
          break;
        }
        case Exl3Path::kMulti: {
          step.kind = Step::Kind::kMulti;
          const Qwen2Linear* second = linear_of(o.second);
          if (second == nullptr || layer == nullptr) {
            return Rejected(std::format("{}: no weights for {}", o.name, o.second));
          }
          step.multi = {.first = linear->weights,
                        .second = second->weights,
                        .trellis_table = layer->trellis_table,
                        .suh_table = layer->suh_table,
                        .svh_table = layer->svh_table,
                        .written = layer->tables_written,
                        .x = at(o.inputs.at(0)),
                        .a_had = at(o.name + ".a_had"),
                        .y = at("gate"),
                        .output = output,
                        .m = m};
          step.weights = {{"gate_proj.trellis", linear->weights.trellis},
                          {"gate_proj.suh", linear->weights.suh},
                          {"gate_proj.svh", linear->weights.svh},
                          {"up_proj.trellis", second->weights.trellis},
                          {"up_proj.suh", second->weights.suh},
                          {"up_proj.svh", second->weights.svh},
                          {"table.trellis", layer->trellis_table},
                          {"table.suh", layer->suh_table},
                          {"table.svh", layer->svh_table}};
          step.multi_plan = {.shape = o.launch.shape,
                             .blocks = o.launch.blocks,
                             .concurrency = o.launch.concurrency};
          if (at("up") != at("gate") + (static_cast<std::uint64_t>(m) *
                                        static_cast<std::uint64_t>(linear->weights.n) * 4) ||
              ScratchBytes(step.multi) > s.plan.slots.at(o.name + ".a_had").bytes) {
            return Rejected(
                std::format("{}: gate and up are not one [2, rows, n] output, or "
                            "the scratch is short",
                            o.name));
          }
          auto coresident =
              launch.MultiGemmCoresident(linear->weights.bits, o.launch.shape, output);
          if (!coresident) {
            return std::unexpected(coresident.error());
          }
          if (auto checked =
                  CheckMultiGemm(step.multi, step.multi_plan, *coresident, launch.locks());
              !checked) {
            return std::unexpected(checked.error());
          }
          break;
        }
        case Exl3Path::kReconstruct:
        case Exl3Path::kReconstructFused: {
          step.kind = Step::Kind::kRecon;
          const bool fused = o.launch.path == Exl3Path::kReconstructFused;
          step.recon = {.weights = linear->weights,
                        .x = at(o.inputs.at(0)),
                        .xh = fused ? 0 : at(o.name + ".xh"),
                        .w = at(o.name + ".w"),
                        .y = at(o.outputs.at(0)),
                        .output = output,
                        .m = m,
                        .bias = 0};
          if (ReconstructScratchBytes(linear->weights) > s.plan.slots.at(o.name + ".w").bytes) {
            return Rejected(std::format("{}: the plan's weight slice is short", o.name));
          }
          for (const model::Exl3Pin& pin : o.launch.pins) {
            step.algorithms.push_back({.config = pin.config,
                                       .m = pin.m,
                                       .k = pin.k,
                                       .n = pin.n,
                                       .ldc = pin.ldc,
                                       .output = pin.f32 ? Output::kF32 : Output::kF16});
          }
          if (auto checked = CheckReconstructedScratch(step.recon, fused); !checked) {
            return std::unexpected(checked.error());
          }
          break;
        }
      }
    }
    s.steps.push_back(std::move(step));
  }
  return program;
}

std::expected<void, KernelFailure> Qwen2Program::Run(ggml::LaunchContext& ggml_launch,
                                                     LaunchContext& launch, ReconGemm& gemm,
                                                     providers::DeviceExecution& execution,
                                                     providers::StreamId stream,
                                                     std::uint64_t host_inputs,
                                                     const Qwen2Hooks& hooks) const {
  const State& s = *state_;
  const auto copy = [&](std::uint64_t destination, std::uint64_t source,
                        std::uint64_t bytes) -> std::expected<void, KernelFailure> {
    if (auto copied = execution.Copy(stream, destination, source, base::Bytes(bytes)); !copied) {
      return std::unexpected(
          KernelFailure{.error = KernelError::kUnknown, .detail = copied.error().detail});
    }
    return {};
  };
  for (const Step& step : s.steps) {
    const Exl3Op& o = s.plan.ops[step.op];
    if (hooks.before) {
      if (auto noted = hooks.before(step.op); !noted) {
        return noted;
      }
    }
    std::expected<void, KernelFailure> done;
    switch (step.kind) {
      case Step::Kind::kInputs: {
        const auto rows = static_cast<std::uint64_t>(s.plan.phase.rows);
        const HostInputs& layout = s.inputs;
        done = copy(Address(step.op, "ids"), host_inputs + layout.ids, rows * 4);
        if (done) {
          done = copy(Address(step.op, "positions"), host_inputs + layout.positions, rows * 4);
        }
        if (done) {
          done = copy(Address(step.op, "mask"), host_inputs + layout.mask,
                      rows * static_cast<std::uint64_t>(s.plan.padded) * 2);
        }
        break;
      }
      case Step::Kind::kCopy:
        done = copy(step.destination, step.source, step.bytes);
        break;
      case Step::Kind::kNorm:
        if (auto ran = step.Norm().Run(ggml_launch, step.nodes[0], step.nodes[1]); !ran) {
          done = From(ran.error());
        }
        break;
      case Step::Kind::kGgml:
        if (auto ran = step.Ggml().Run(ggml_launch, step.nodes); !ran) {
          done = From(ran.error());
        }
        break;
      case Step::Kind::kGemm:
        done = step.Exl3().Run(launch, step.linear, step.gemm, 0);
        break;
      case Step::Kind::kGemv:
        done = step.Exl3().Run(launch, step.linear, step.gemv, 0);
        break;
      case Step::Kind::kMulti:
        done = step.Exl3().Run(launch, step.multi, step.multi_plan);
        break;
      case Step::Kind::kRecon:
        done = step.Exl3().Run(launch, gemm, step.recon, step.algorithms);
        break;
      case Step::Kind::kBias:
        done = step.Exl3().Run(launch, step.bias);
        break;
    }
    if (!done) {
      return std::unexpected(KernelFailure{
          .error = done.error().error,
          .detail = std::format("{} (layer {}): {}", o.name, o.layer, done.error().detail)});
    }
    if (hooks.after) {
      if (auto noted = hooks.after(step.op); !noted) {
        return noted;
      }
    }
  }
  return {};
}

}  // namespace llmp::kernels::exl3
