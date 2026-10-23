// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML lowering (kernels/ggml/lower.h) of Qwen3.8's steps structurally,
// on the CPU: for each import (the ModelOpt one with its experts in the
// CUTLASS layout or in GGML's NVFP4 blocks, and a GGUF quantization's) and
// each shape measured at setup (prefill, decode, its token shape, the widest
// chunk that does not select, a selecting chunk at the context's end,
// state-only prefill, and the boundaries of the fast kernels' row limits),
// plain and with the MTP drafter (each step's streams for the drafter, a
// verify with its saves, the drafter's prefill and draft passes), the step
// planned under the mapping's default policy and lowered launches valid GGML
// nodes under PlanGraph's steps.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compiler_inventory.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "execution/registry.h"
#include "families/qwen.h"
#include "format/encoding.h"
#include "ggml.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/common/lower.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/plan_step.h"
#include "kernels/ggml/tensors.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "qwen38_fixture.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen38.h"
#include "state_layouts.h"

namespace {

namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace ex = llmp::execution;
namespace pl = llmp::planner;
namespace co = llmp::components;
namespace ir = llmp::ir;
namespace sp = llmp::spec;
namespace fx = llmp::test_support::qwen38;
namespace fam = llmp::test_support::families;

constexpr std::uint64_t kExtent = 2ULL << 20U;
constexpr std::uint32_t kGb10 = 121;
constexpr std::uint32_t kContext = 8192;
constexpr std::uint32_t kMaxRows = 512;
constexpr std::uint32_t kDraftRows = 2;  // Qwen38Options::draft_rows

ex::Registry PlanningRegistry() {
  std::vector<ex::Implementation> v;
  for (auto d : kg::PlanningImplementations()) {
    v.push_back(std::move(d));
  }
  return std::move(ex::Registry::Create(std::move(v)).value());
}

const pl::DeviceFacts& Gb10() {
  static const pl::DeviceFacts facts{
      .device_class = "sm_121",
      .arch = kGb10,
      .features = llmp::execution::kBf16Mma | llmp::execution::kInt8Mma | llmp::execution::kFp8Mma |
                  llmp::execution::kFp4Mma | llmp::execution::kBlockScaledMma};
  return facts;
}

std::optional<llmp::format::Encoding> EncodingOf(ggml_type type) {
  for (const auto& info : llmp::format::Encodings()) {
    if (info.ggml && info.ggml->type == static_cast<std::uint32_t>(type)) return info.id;
  }
  return std::nullopt;
}

// The GB10's routing as both sides take it (ggml_lower_test's model): MMVQ
// where upstream's ggml_cuda_should_use_mmvq takes a product's columns (an
// expert product's tokens), MMQ otherwise; cuBLAS for float products.
kg::DeviceChoices Device() {
  kg::DeviceChoices d;
  d.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return kg::MulMatPath::kCublas;
  };
  d.quant = [](const ggml_tensor* n) -> std::expected<kg::QuantMulMatPath, kg::KernelFailure> {
    const auto e = EncodingOf(n->src[0]->type);
    const auto columns =
        static_cast<std::uint64_t>(n->op == GGML_OP_MUL_MAT_ID ? n->ne[2] : n->src[1]->ne[1]);
    return e && kg::MmvqRouted(*e, kGb10, columns) ? kg::QuantMulMatPath::kVector
                                                   : kg::QuantMulMatPath::kTile;
  };
  return d;
}

// ------------------------------------------------------------ the imports

fam::Qwen38PleHash Hash() {
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> vocab;
  for (std::int64_t h = 0; h < 16; ++h) {
    offsets.push_back(h * 20000096);
    vocab.push_back(20000096 - (h * 8));
  }
  return *fam::CheckQwen38PleHash(fam::Qwen38Flash(), std::vector<std::int64_t>{3, 5, 7}, offsets,
                                  vocab, fx::kTableRows);
}

// A fixture's result, or the error it names (and the test aborts).
struct Must {
  explicit Must(int line) : line(line) {}
  int line;
};
template <typename T, typename E>
T operator|(std::expected<T, E> e, Must m) {
  if (!e) {
    std::println(stderr, "the fixture at line {}: {}", m.line, e.error());
    std::abort();
  }
  return std::move(*e);
}

// The CUTLASS layout's arrays packed from each expert group's start, as
// the M3 importer wrote them (each layer's gate-up codes and scales, then
// down's).
void PackCutlassArrays(std::vector<fam::Qwen38Resource>& resources) {
  std::map<std::string, std::uint64_t> at;  // by layer prefix
  for (fam::Qwen38Resource& r : resources) {
    if (!r.expert_array || r.type != "I8") continue;
    const std::string& name = r.roles.front();
    const std::string prefix = name.substr(0, name.find("ffn_"));
    r.group_offset = at[prefix];
    at[prefix] += r.ne[0] * r.ne[1];
  }
}

std::string FormName(fx::Form form) {
  switch (form) {
    case fx::Form::kCutlass:
      return "cutlass";
    case fx::Form::kGgmlExperts:
      return "ggml_experts";
    case fx::Form::kGguf:
      return "gguf";
  }
  return "?";
}

// An import's target and, for the ModelOpt ones, its MTP drafter: the
// specs (the target alone, and with the drafter), the state layouts, the
// policies the mappings give, and the fixture's resources.
struct Import {
  fx::Form form{};
  std::vector<fam::Qwen38Resource> resources;
  std::vector<fam::Qwen38Resource> mtp_resources;
  sp::Source source;
  sp::Source mtp_source;
  std::optional<sp::ModelSpec> spec;                // the target alone
  std::optional<sp::ModelSpec> spec_mtp;            // with the drafter
  std::vector<md::StateLayoutDerived> layouts;      // the target alone's
  std::vector<md::StateLayoutDerived> layouts_mtp;  // by stage
  pl::Policy policy;
  pl::Policy mtp_policy;
  std::uint32_t cells = 8192;
  std::vector<std::uint64_t> stride;  // the CUTLASS slab's, by layer (else none: packed)
  std::uint64_t mtp_stride = 0;
  bool mtp() const { return spec_mtp.has_value(); }
};

const Import& ImportOf(fx::Form form) {
  static std::map<fx::Form, Import> imports;
  if (const auto at = imports.find(form); at != imports.end()) return at->second;
  const auto& p = fam::Qwen38Flash();
  Import im;
  im.form = form;
  const bool gguf = form == fx::Form::kGguf;
  im.resources = fx::Resources(p, form);
  PackCutlassArrays(im.resources);
  im.source = fx::SourceOf("qwen4exp", gguf ? "gguf" : "safetensors", im.resources,
                           gguf ? std::vector<std::string>{} : fx::ModelOptFolds());
  if (!gguf) fx::FixtureNativeTensorNames(im.resources);
  const auto values = fx::TargetValues(p, Hash(), form) | Must(__LINE__);
  im.spec = sp::Synthesize(sp::mappings::Qwen38Mapping(), im.source, values) | Must(__LINE__);
  const sp::Envelope envelope{.context = kContext, .slots = 1};
  im.policy = pl::ParsePolicy(sp::DefaultPolicy(sp::mappings::Qwen38Mapping(), values, envelope,
                                                {.architecture = im.source.architecture,
                                                 .container = im.source.container}) |
                              Must(__LINE__)) |
              Must(__LINE__);
  // Descriptor-only operand controls propose the selected numerical recipe
  // explicitly. They carry no production source receipt or same-quality grant.
  im.policy.numerical_candidates =
      gguf ? std::vector{pl::NumericalChange::kQ8VectorReduction}
           : std::vector{pl::NumericalChange::kBf16HyperConnectionIntermediates,
                         pl::NumericalChange::kBf16ProductInputAndResult,
                         pl::NumericalChange::kMxfp8ProductInput,
                         pl::NumericalChange::kNvfp4ExpertIntermediates};
  pl::Canonicalize(im.policy);
  im.layouts.push_back(co::StageLayoutUniform(*im.spec, 0, kContext, kMaxRows, kExtent) |
                       Must(__LINE__));
  if (form == fx::Form::kCutlass) {
    const kg::moe::ExpertLayout layout{.ffn = p.expert_ffn, .width = p.width};
    im.stride.assign(p.layers, (layout.bytes() + 4095) / 4096 * 4096);
  }
  if (!gguf) {
    im.mtp_resources = fx::MtpResources(p);
    PackCutlassArrays(im.mtp_resources);
    im.mtp_source =
        fx::SourceOf("qwen3_8_flash_next", "safetensors", im.mtp_resources, fx::ModelOptFolds());
    fx::FixtureNativeTensorNames(im.mtp_resources);
    const auto mtp_values = fam::Qwen38MtpValues(p, 0) | Must(__LINE__);
    im.spec_mtp = sp::ComposeDrafter(*im.spec, sp::mappings::Qwen38Mapping(),
                                     sp::mappings::Qwen38MtpMapping(), im.mtp_source, mtp_values) |
                  Must(__LINE__);
    im.mtp_policy =
        pl::ParsePolicy(sp::DefaultPolicy(sp::mappings::Qwen38MtpMapping(), mtp_values, envelope,
                                          {.architecture = im.mtp_source.architecture,
                                           .container = im.mtp_source.container}) |
                        Must(__LINE__)) |
        Must(__LINE__);
    im.mtp_policy.numerical_candidates = im.policy.numerical_candidates;
    pl::Canonicalize(im.mtp_policy);
    for (std::uint16_t s = 0; s < 2; ++s) {
      im.layouts_mtp.push_back(
          co::StageLayoutUniform(*im.spec_mtp, s, kContext, kMaxRows, kExtent) | Must(__LINE__));
    }
    const kg::moe::ExpertLayout layout{.ffn = p.expert_ffn, .width = p.width};
    im.mtp_stride = (layout.bytes() + 4095) / 4096 * 4096;
  }
  return imports.emplace(form, std::move(im)).first->second;
}

// ------------------------------------------------------------ the shapes

enum class Kind : std::uint8_t {
  kHead,
  kToken,
  kStateOnly,
  kStreams,
  kVerify,
  kPrefillPass,
  kDraft
};

struct Case {
  Kind kind = Kind::kHead;
  std::uint32_t past = 0;
  std::uint32_t rows = 0;
  std::string Name() const {
    constexpr std::array kNames{"head",   "token",        "state-only", "streams",
                                "verify", "prefill pass", "draft"};
    return std::format("{} {}@{}", kNames.at(static_cast<std::size_t>(kind)), rows, past);
  }
  bool drafter() const { return kind == Kind::kPrefillPass || kind == Kind::kDraft; }
};

// The runner's measured chunks, and the rows where a fast
// kernel's limit falls (the MXFP8 vector product's 8 columns, the float
// products' 16 rows and the recurrence's lanes, the convolution's
// 32-token windows).
std::vector<Case> Cases(const Import& im) {
  const auto& p = fam::Qwen38Flash();
  const std::uint32_t unselected =
      std::min<std::uint32_t>(kContext, (p.indexer_budget + p.indexer_ratio - 1) / 256U * 256U);
  const std::uint32_t unselected_rows = std::min(kMaxRows, unselected);
  std::vector<Case> out;
  const std::vector<std::pair<std::uint32_t, std::uint32_t>> chunks = {
      {0, kMaxRows},
      {kContext - kMaxRows, kMaxRows},
      {unselected - unselected_rows, unselected_rows},
      {kContext - 1, 1},
      {0, 1},
      {40, 8},
      {40, 9},
      {40, 16},
      {40, 17},
      {3, 33},
      {100, 40}};
  for (const auto& [past, rows] : chunks) {
    out.push_back({.kind = Kind::kHead, .past = past, .rows = rows});
    out.push_back({.kind = Kind::kStateOnly, .past = past, .rows = rows});
    if (rows == 1) out.push_back({.kind = Kind::kToken, .past = past, .rows = rows});
  }
  if (!im.mtp()) return out;
  const std::uint32_t verify = kDraftRows + 1;
  for (const auto& [past, rows] : {std::pair{0U, kMaxRows},
                                   {kContext - kMaxRows, kMaxRows},
                                   {unselected - unselected_rows, unselected_rows},
                                   {40U, 17U}}) {
    out.push_back({.kind = Kind::kStreams, .past = past, .rows = rows});
  }
  for (const std::uint32_t past : {0U, kContext - verify, 40U}) {
    out.push_back({.kind = Kind::kVerify, .past = past, .rows = verify});
  }
  out.push_back({.kind = Kind::kPrefillPass, .past = kContext - kMaxRows - 1, .rows = kMaxRows});
  out.push_back(
      {.kind = Kind::kPrefillPass, .past = unselected - unselected_rows, .rows = unselected_rows});
  out.push_back({.kind = Kind::kDraft, .past = kContext - verify - kDraftRows, .rows = verify});
  out.push_back({.kind = Kind::kDraft, .past = 40, .rows = verify});
  out.push_back({.kind = Kind::kDraft, .past = 40, .rows = 1});
  return out;
}

// ------------------------------------------------------------ the lowering

struct Lowered {
  std::optional<co::StepGraph> step;
  std::optional<pl::StepPlan> plan;
  std::optional<kg::LoweredStep> lowered;
  std::map<const ggml_tensor*, std::string> leaves;
  std::vector<std::string> applied;  // the patterns the plan applied
  std::string refusal;
};

// The step's leaves by what they bind, as the builder's are named.
std::map<const ggml_tensor*, std::string> LeafNames(
    const Import& im, const co::StepGraph& step, const kg::LoweredStep& lowered,
    std::span<const md::StateLayoutDerived> layouts) {
  std::map<std::string, std::uint32_t> target_index;
  for (std::uint32_t i = 0; i < im.resources.size(); ++i) {
    target_index[im.resources[i].roles.front()] = i;
  }
  std::map<std::string, std::uint32_t> mtp_index;
  for (std::uint32_t i = 0; i < im.mtp_resources.size(); ++i) {
    mtp_index[im.mtp_resources[i].roles.front()] = i;
  }
  std::map<std::uint32_t, std::string> by_primitive;
  for (const co::StepInput& in : step.inputs) {
    std::string name = "?";
    switch (in.kind) {
      case co::InputKind::kTokens:
        name = "tokens";
        break;
      case co::InputKind::kPositions:
        name = "positions";
        break;
      case co::InputKind::kOutputRows:
        name = "out_ids";
        break;
      case co::InputKind::kStoreCells:
        name = in.cells.fixed                                        ? "state_row"
               : !in.cells.per_row                                   ? "cells"
               : step.graph.name(in.tensor).starts_with("save_rows") ? "row_ids"
                                                                     : "stream_rows";
        break;
      case co::InputKind::kSideRows:
        name = "ple_rows";
        break;
      case co::InputKind::kFeatures:
        name = "features";
        break;
      case co::InputKind::kCellPositions:
        name = "cell_positions";
        break;
      case co::InputKind::kBlockPositions:
        name = "block_positions";
        break;
      default:
        break;
    }
    by_primitive[in.tensor.v] = name;
  }
  for (const co::StepState& st : step.state) {
    if (st.tap) {
      by_primitive[st.tensor.v] = "streams";
      continue;
    }
    const auto& item = layouts[st.borrowed ? st.source : st.ref.stage].items.at(st.item);
    by_primitive[st.tensor.v] = std::format("{}.{}", item.contract.name, st.ref.layer);
  }
  std::map<std::uint32_t, const co::StepWeight*> weights;
  for (const co::StepWeight& w : step.weights) weights[w.tensor.v] = &w;
  std::map<const ggml_tensor*, std::string> out;
  for (const kg::LoweredLeaf& leaf : lowered.leaves) {
    if (const auto at = weights.find(leaf.primitive.v); at != weights.end()) {
      const co::StepWeight& w = *at->second;
      std::string part = w.parts.front().first;
      if (leaf.part) {
        for (const auto& [name, role] : w.parts) {
          if (role == *leaf.part) part = name;
        }
      }
      const bool target = w.ref.stage == 0;
      const auto& index = target ? target_index : mtp_index;
      const auto found = index.find(part);
      out[leaf.tensor] = found == index.end()
                             ? "w.?" + part
                             : std::format("{}.{}", target ? "w" : "m", found->second);
      continue;
    }
    const auto at = by_primitive.find(leaf.primitive.v);
    out[leaf.tensor] = at == by_primitive.end() ? "?" : at->second;
  }
  for (const kg::SavedLeaf& s : lowered.saves) {
    out[s.tensor] = std::format("save.{}.{}", s.what, s.layer);
  }
  return out;
}

pl::StepShape StepShapeOf(const Import& im, const Case& c) {
  const bool drafter = c.drafter();
  const bool head = c.kind == Kind::kDraft || c.kind == Kind::kHead || c.kind == Kind::kToken ||
                    c.kind == Kind::kStreams || c.kind == Kind::kVerify;
  const std::uint32_t cells = im.cells;
  const std::uint32_t read = std::min<std::uint32_t>((c.past + c.rows + 255) / 256 * 256, cells);
  const std::uint32_t outputs = !head ? 0 : c.kind == Kind::kVerify ? c.rows : 1;
  pl::Phase phase = c.rows == 1 ? pl::Phase::kDecode : pl::Phase::kPrefill;
  if (c.kind == Kind::kVerify) phase = pl::Phase::kVerify;
  if (drafter) phase = pl::Phase::kDraft;
  return {.stage = static_cast<std::uint16_t>(drafter ? 1 : 0),
          .phase = phase,
          .segments = {{.slot = 0,
                        .rows = c.rows,
                        .past = c.past,
                        .read_cells = read,
                        .outputs = outputs,
                        .want_head = head,
                        .greedy = c.kind == Kind::kToken || c.kind == Kind::kDraft ||
                                  c.kind == Kind::kVerify}}};
}

Lowered Lower(const Import& im, const Case& c, const ex::Registry& registry,
              const pl::Policy* policy_override = nullptr,
              const pl::StepShape* shape_override = nullptr,
              const std::function<void(co::StepGraph&)>& mutate = {},
              const std::function<void(co::StepGraph&)>& before_plan = {}) {
  Lowered out;
  const bool with_mtp = c.kind == Kind::kStreams || c.kind == Kind::kVerify || c.drafter();
  const sp::ModelSpec& spec = with_mtp ? *im.spec_mtp : *im.spec;
  const auto& layouts = with_mtp ? im.layouts_mtp : im.layouts;
  std::vector<const md::StateLayoutDerived*> stages;
  for (const auto& l : layouts) stages.push_back(&l);
  std::vector<std::span<const co::BoundTensor>> tables{im.source.tensors};
  if (with_mtp) tables.emplace_back(im.mtp_source.tensors);
  const pl::StepShape shape = shape_override != nullptr ? *shape_override : StepShapeOf(im, c);
  const pl::Policy& policy = policy_override != nullptr ? *policy_override
                             : shape.stage == 0         ? im.policy
                                                        : im.mtp_policy;
  const sp::Stage& s = spec.stages.at(shape.stage);
  auto step = co::BuildStep(spec, shape.stage, shape,
                            {.tensors = tables.at(s.artifact),
                             .state = &layouts.at(shape.stage),
                             .stages = stages,
                             .artifacts = tables},
                            {.narrow = !pl::Disabled(policy, "narrow_final")});
  EXPECT_TRUE(step.has_value()) << step.error();
  if (!step) return out;
  if (before_plan) before_plan(*step);
  std::vector<ir::TensorId> staged;
  for (const co::StepInput& in : step->inputs) {
    if (in.kind != co::InputKind::kCellPositions) staged.push_back(in.tensor);
  }
  auto plan = pl::Plan(step->graph.Clone(), registry, Gb10(), policy,
                       {.capturable = true,
                        .staged = staged,
                        .written = &kg::WrittenBytes,
                        .within = &kg::WithinRoot,
                        .shape = &shape});
  EXPECT_TRUE(plan.has_value()) << (plan ? "" : plan.error().detail);
  if (!plan) return out;
  for (const pl::Applied& a : plan->applied) out.applied.emplace_back(a.pattern);
  // A diagnosis: the planned graph's nodes from LLMP_DUMP_PLAN's first to
  // its last ("first:last").
  if (const char* dump = std::getenv("LLMP_DUMP_PLAN"); dump != nullptr) {
    const std::string range(dump);
    const auto colon = range.find(':');
    const std::size_t first = std::stoul(range.substr(0, colon));
    const std::size_t last =
        colon == std::string::npos ? first + 100 : std::stoul(range.substr(colon + 1));
    const ir::Graph& g = plan->graph;
    for (std::size_t i = first; i < std::min(last, g.nodes().size()); ++i) {
      const ir::Node& n = g.nodes()[i];
      std::string text = std::format("{:5} {:24}", i, ir::OpKeyName(n.key));
      for (const ir::TensorId o : n.outputs()) {
        const ir::TensorDesc& t = g.tensor(o);
        text += std::format(" t{}:{}[{},{},{},{}]", o.v, llmp::format::Info(t.format.encoding).name,
                            t.ne(0), t.ne(1), t.ne(2), t.ne(3));
      }
      text += " <-";
      for (const ir::TensorId from : n.sources()) {
        text += from.valid() ? std::format(" t{}", from.v) : std::string(" -");
      }
      text += std::format("  {}", n.nout > 0 ? g.name(n.out[0]) : "");
      std::println("{}", text);
    }
    std::println("applied:");
    for (const pl::Applied& a : plan->applied) {
      std::string made;
      for (const ir::NodeId m : a.made) made += std::format(" {}", m.v);
      std::println("  {}:{}", a.pattern, made);
    }
  }
  std::map<ir::WeightRef, std::uint64_t> strides;
  for (const co::StepWeight& w : step->weights) {
    if (!w.ref.expert_array) continue;
    if (w.ref.stage == 0) {
      if (w.ref.layer < im.stride.size()) strides[w.ref] = im.stride[w.ref.layer];
    } else {
      strides[w.ref] = im.mtp_stride;
    }
  }
  if (mutate) mutate(*step);
  out.step.emplace(std::move(*step));
  out.plan.emplace(std::move(*plan));
  const kn::Addresses addresses{};
  const kn::LowerInput input{*out.plan, *out.step, shape, addresses, kContext};
  auto lowered =
      kg::LowerStep(input, registry, Device(),
                    {.device_masks = true,
                     .frozen_host_masks = true,
                     .expert_stride = [&strides](const ir::WeightRef& ref) -> std::uint64_t {
                       const auto at = strides.find(ref);
                       return at == strides.end() ? 0 : at->second;
                     }});
  if (!mutate && !before_plan)
    EXPECT_TRUE(lowered.has_value()) << (lowered ? "" : lowered.error().detail);
  if (!lowered) {
    out.refusal = lowered.error().detail;
    return out;
  }
  out.leaves = LeafNames(im, *out.step, *lowered, layouts);
  out.lowered.emplace(std::move(*lowered));
  return out;
}

// The lowering appends its save stores after every planned node. The
// staged row indices must still be intact then: no computed allocation
// may reuse their bytes, even when the IR has no ordinary reader of them.
void ExpectSaveRowsLive(const Lowered& l) {
  for (const co::CommitSave& save : l.step->saves) {
    const auto slot = std::ranges::find(l.lowered->inputs, save.rows, &kn::InputSlot::tensor);
    ASSERT_NE(slot, l.lowered->inputs.end());
    const std::uint64_t start = slot->device_address;
    const std::uint64_t end = start + slot->bytes;
    for (const ggml_tensor* n : l.lowered->nodes) {
      if (n->view_src != nullptr || n->op == GGML_OP_NONE) continue;
      const auto at = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(n->data));
      EXPECT_TRUE(at >= end || at + ggml_nbytes(n) <= start)
          << "save rows overlap " << ggml_op_desc(n) << " for " << save.what;
    }
  }
}

// ------------------------------------------------------------ the tests

class GgmlLowerQwen38 : public testing::TestWithParam<fx::Form> {};

TEST_P(GgmlLowerQwen38, SetupShapesLowerToTheFastFormsGraphs) {
  const Import& im = ImportOf(GetParam());
  const ex::Registry registry = PlanningRegistry();
  std::size_t shapes = 0, launches = 0;
  std::set<std::string> applied;
  for (const Case& c : Cases(im)) {
    SCOPED_TRACE(std::format("{} {}", FormName(im.form), c.Name()));
    const Lowered w = Lower(im, c, registry);
    ASSERT_TRUE(w.lowered.has_value());
    ExpectSaveRowsLive(w);
    applied.insert(w.applied.begin(), w.applied.end());
    ASSERT_FALSE(w.lowered->nodes.empty());
    const bool expect_outputs = c.kind != Kind::kStateOnly && c.kind != Kind::kPrefillPass;
    EXPECT_EQ(!w.lowered->outputs.empty(), expect_outputs);
    EXPECT_GT(w.lowered->plan.steps.size(), 0U);
    launches += w.lowered->plan.steps.size();
    ++shapes;
  }
  std::string patterns;
  for (const std::string& p : applied) patterns += " " + p;
  std::println("{}: {} lowered steps, {} launches; patterns:{}", FormName(im.form), shapes,
               launches, patterns);
  EXPECT_GT(shapes, 0U);
}

INSTANTIATE_TEST_SUITE_P(Qwen38, GgmlLowerQwen38,
                         testing::Values(fx::Form::kCutlass, fx::Form::kGgmlExperts,
                                         fx::Form::kGguf),
                         [](const testing::TestParamInfo<fx::Form>& info) {
                           return FormName(info.param);
                         });

// QSA's custom kernels infer positions, pooling cells and selection masks.
// The lowering must prove their typed staging contract; diagnostic leaf names
// alone cannot make arbitrary caller-supplied values satisfy that contract.
TEST(GgmlLowerQwen38Test, QsaProofOnlyMasksDoNotReceivePhysicalBacking) {
  const Import& im = ImportOf(fx::Form::kCutlass);
  const auto registry = PlanningRegistry();
  const auto lowered = Lower(im, {.kind = Kind::kHead, .past = 7000, .rows = 3}, registry);
  ASSERT_TRUE(lowered.lowered) << lowered.refusal;
  EXPECT_EQ(lowered.lowered->logical_extent, lowered.plan->placement.extent);
  EXPECT_GT(lowered.lowered->unmaterialized_roots, 0U);
  EXPECT_GT(lowered.lowered->unmaterialized_logical_bytes, 0U);
  // Omitted logical sizes overlap in time and space; they are not subtracted
  // from a peak. The program funds its independently placed physical extent.
  EXPECT_GT(lowered.lowered->extent, 0U);
}

TEST(GgmlLowerQwen38Test, QsaRefusesInputsWithoutItsStagingContract) {
  const Import& im = ImportOf(fx::Form::kCutlass);
  const ex::Registry registry = PlanningRegistry();
  const Case c{.kind = Kind::kHead, .past = 7000, .rows = 3};
  for (std::uint32_t mutation = 0; mutation < 18; ++mutation) {
    SCOPED_TRACE(mutation);
    bool changed = false;
    const auto mutate = [&](co::StepGraph& step) {
      auto at = step.inputs.end();
      if (mutation >= 16) {
        at = std::ranges::find_if(step.inputs, [](const co::StepInput& in) {
          return in.kind == co::InputKind::kPositions && in.sections == 1;
        });
      } else if (mutation < 3) {
        at = std::ranges::find_if(step.inputs, [](const co::StepInput& in) {
          return in.kind == co::InputKind::kPositions && in.sections == 4;
        });
      } else if (mutation < 6) {
        at = std::ranges::find(step.inputs, co::InputKind::kBlockPositions, &co::StepInput::kind);
      } else if (mutation < 10) {
        at = std::ranges::find_if(step.inputs, [](const co::StepInput& in) {
          return in.kind == co::InputKind::kStoreCells && in.cells.block != 0;
        });
      } else {
        // Seen-block bias, per-cell causal scores, and the attention's exact
        // causal mask all obtain their positions from typed cell inputs.
        for (const ir::Node& n : step.graph.nodes()) {
          if (n.key != ir::Primitive(ir::Op::kMaskBuild)) continue;
          const bool exact = n.Params<ir::MaskParams>().exact_rows;
          const bool wanted = mutation < 12 ? !exact : mutation >= 14 ? exact : !exact;
          if (!wanted) continue;
          auto candidate = std::ranges::find(step.inputs, n.src[1], &co::StepInput::tensor);
          if (candidate == step.inputs.end()) continue;
          const bool block = candidate->cells.block != 0;
          if (block != (mutation < 12)) continue;
          at = candidate;
          break;
        }
      }
      ASSERT_NE(at, step.inputs.end());
      const std::string name(step.graph.name(at->tensor));
      switch (mutation) {
        case 0:
          at->sections = 1;
          break;
        case 1:
          at->kind = co::InputKind::kFeatures;
          break;
        case 2:
          at->segment = 0;
          break;
        case 3:
          ++at->cells.block;
          break;
        case 4:
          at->segment = ir::kJoinedSegment;
          break;
        case 5:
          at->sections = 1;
          break;
        case 6:
          ++at->cells.block;
          break;
        case 7:
          at->cells.ring = true;
          break;
        case 8:
          at->segment = ir::kJoinedSegment;
          break;
        case 9:
          --at->cells.capacity;
          break;
        case 10:
          ++at->cells.block;
          break;
        case 11:
          at->kind = co::InputKind::kFeatures;
          break;
        case 12:
          at->cells.frozen = true;
          break;
        case 13:
          at->cells.ring = true;
          break;
        case 14:
          --at->cells.capacity;
          break;
        case 15:
          at->segment = ir::kJoinedSegment;
          break;
        case 16:
          // Two declarations cannot supply conflicting staging contracts for
          // one leaf, even when the duplicated declaration is identical.
          EXPECT_EQ(step.graph.name(at->tensor), name);
          step.inputs.push_back(*at);
          changed = true;
          return;
        case 17:
          // A slot declares a leaf, rather than a derived owner-position view.
          for (const ir::Node& n : step.graph.nodes()) {
            if (n.key == ir::Primitive(ir::Op::kView) && n.src[0] == at->tensor) {
              at->tensor = n.out[0];
              changed = true;
              return;
            }
          }
          FAIL() << "the fixture has no owner-position view";
      }
      EXPECT_EQ(step.graph.name(at->tensor), name);
      changed = true;
    };
    const Lowered out = Lower(im, c, registry, nullptr, nullptr, mutate);
    ASSERT_TRUE(changed);
    ASSERT_TRUE(out.plan.has_value());
    EXPECT_FALSE(out.lowered.has_value());
    EXPECT_FALSE(out.refusal.empty());
  }
}

// A small context does not select QSA attention: its exact causal mask
// remains a primitive and must prove its own position substitution.
TEST(GgmlLowerQwen38Test, UnselectedExactMasksProveTheirPositionSource) {
  const Import& im = ImportOf(fx::Form::kCutlass);
  const ex::Registry registry = PlanningRegistry();
  const Case c{.kind = Kind::kHead, .past = 100, .rows = 3};
  for (std::uint32_t mutation = 0; mutation < 9; ++mutation) {
    SCOPED_TRACE(mutation);
    bool changed = false;
    const auto mutate = [&](co::StepGraph& step) {
      auto at = step.inputs.end();
      if (mutation < 3) {
        at = std::ranges::find_if(step.inputs, [](const co::StepInput& in) {
          return in.kind == co::InputKind::kPositions && in.sections == 1;
        });
      } else {
        for (const ir::Node& n : step.graph.nodes()) {
          if (n.key == ir::Primitive(ir::Op::kMaskBuild) && n.Params<ir::MaskParams>().exact_rows) {
            at = std::ranges::find(step.inputs, n.src[1], &co::StepInput::tensor);
            break;
          }
        }
      }
      ASSERT_NE(at, step.inputs.end());
      if (mutation == 0) at->kind = co::InputKind::kFeatures;
      if (mutation == 1) at->sections = 2;
      if (mutation == 2) at->segment = 0;
      if (mutation == 3) at->cells.block = 2;
      if (mutation == 4) at->cells.ring = true;
      if (mutation == 5) ++at->read;
      if (mutation == 6) at->cells.capacity = at->read - 1;
      if (mutation == 7) at->segment = ir::kJoinedSegment;
      if (mutation == 8) at->cells.capacity += std::uint64_t{1} << 32;
      changed = true;
    };
    const Lowered out = Lower(im, c, registry, nullptr, nullptr, mutate);
    ASSERT_TRUE(changed);
    ASSERT_TRUE(out.plan.has_value());
    EXPECT_FALSE(out.lowered.has_value());
    EXPECT_NE(out.refusal.find("exact device mask"), std::string::npos) << out.refusal;
  }

  ir::TensorId extra{};
  bool write_mask = false;
  bool oversized_window = false;
  const auto add_unrelated = [&](co::StepGraph& step) {
    // Clone through the public builder without changing existing tensor
    // identities, then append either unrelated positions or a mask write.
    ir::Builder b;
    for (const ir::Node& n : step.graph.nodes()) {
      b.SetComponent(n.component);
      if (n.batch == ir::Batch::kPerOwner)
        b.SetOwner(step.graph.tensor(n.out[0]).segment);
      else
        b.SetJoined();
      std::array<ir::TensorDesc, ir::kMaxOutputs> requested{};
      for (std::size_t i = 0; i < n.nout; ++i) requested[i] = step.graph.tensor(n.out[i]);
      ir::ParamBytes params = n.params;
      if (oversized_window && n.key == ir::Primitive(ir::Op::kMaskBuild) &&
          n.Params<ir::MaskParams>().exact_rows) {
        auto mp = n.Params<ir::MaskParams>();
        mp.kind = ir::MaskKind::kWindow;
        mp.window = std::uint32_t{1} << 31;
        params = ir::ToBytes(mp);
      }
      const auto made = b.Emit(n.key, params, n.sources(), std::span(requested).first(n.nout),
                               step.graph.name(n.out[0]));
      for (std::size_t i = 0; i < n.nout; ++i) ASSERT_EQ(made[i], n.out[i]);
    }
    for (const ir::TensorId t : step.graph.outputs()) b.Output(t);
    for (const ir::TensorId t : step.graph.keep()) b.Keep(t);
    if (write_mask) {
      for (const ir::Node& n : step.graph.nodes()) {
        if (n.key != ir::Primitive(ir::Op::kMaskBuild) || !n.Params<ir::MaskParams>().exact_rows)
          continue;
        b.SetOwner(0);
        const ir::TensorId rows =
            b.Input(ir::Dense(llmp::format::Encoding::kI64, ir::ShapeOf({1})), "mask_write_row");
        step.inputs.push_back({.tensor = rows,
                               .kind = co::InputKind::kStoreCells,
                               .segment = 0,
                               .cells = {.fixed = true}});
        const ir::TensorId zeros =
            b.Fill(ir::ShapeOf({b.tensor(n.out[0]).ne(0), 1}), 0.0F, llmp::format::Encoding::kF16);
        b.Output(b.CacheStore(n.out[0], zeros, rows));
        break;
      }
    } else if (!oversized_window) {
      b.SetJoined();
      extra = b.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({c.rows})),
                      "unrelated_positions");
      step.inputs.push_back({.tensor = extra, .kind = co::InputKind::kPositions, .sections = 2});
    }
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph.has_value()) << graph.error().detail;
    step.graph = std::move(*graph);
  };
  const Lowered fallback = Lower(im, c, registry, nullptr, nullptr, {}, add_unrelated);
  ASSERT_TRUE(fallback.lowered.has_value()) << fallback.refusal;
  EXPECT_TRUE(std::ranges::any_of(fallback.lowered->inputs, [](const kn::InputSlot& in) {
    return in.kind == kn::InputKind::kPositions && in.sections == 1;
  }));
  EXPECT_FALSE(std::ranges::any_of(fallback.lowered->inputs,
                                   [&](const kn::InputSlot& in) { return in.tensor == extra; }));
  write_mask = true;
  const Lowered overwritten = Lower(im, c, registry, nullptr, nullptr, {}, add_unrelated);
  ASSERT_TRUE(overwritten.plan.has_value());
  EXPECT_FALSE(overwritten.lowered.has_value());
  EXPECT_NE(overwritten.refusal.find("exact device mask"), std::string::npos)
      << overwritten.refusal;
  write_mask = false;
  oversized_window = true;
  const Lowered window = Lower(im, c, registry, nullptr, nullptr, {}, add_unrelated);
  ASSERT_TRUE(window.plan.has_value());
  EXPECT_FALSE(window.lowered.has_value());
  EXPECT_NE(window.refusal.find("exact device mask"), std::string::npos) << window.refusal;
}

// Several owners' steps (a wave: joined products, each owner's own
// attention, recurrence and stores) lower under each import's policy:
// decodes and prefill chunks, one owner selecting above the indexer's
// budget beside one that does not, verifies and the drafter's passes;
// each owner's fast kernels its own (a block pool, a recurrence and a
// convolution each).
TEST(GgmlLowerQwen38Test, SeveralOwnersStepsLower) {
  const ex::Registry registry = PlanningRegistry();
  for (const fx::Form form : {fx::Form::kCutlass, fx::Form::kGgmlExperts, fx::Form::kGguf}) {
    const Import& im = ImportOf(form);
    struct Wave {
      std::vector<std::pair<std::uint32_t, std::uint32_t>> owners;  // (past, rows)
      Kind kind = Kind::kHead;
    };
    std::vector<Wave> waves = {
        {.owners = {{40, 1}, {7000, 1}}},
        {.owners = {{40, 1}, {41, 1}, {42, 1}, {7000, 1}}, .kind = Kind::kToken},
        {.owners = {{0, 37}, {100, 37}}},
        {.owners = {{0, 100}, {3000, 5}}},
        {.owners = {{0, 64}, {64, 64}}, .kind = Kind::kStateOnly},
    };
    if (im.mtp()) {
      waves.push_back({.owners = {{40, 3}, {7000, 3}}, .kind = Kind::kVerify});
      waves.push_back({.owners = {{40, 3}, {500, 1}}, .kind = Kind::kDraft});
      waves.push_back({.owners = {{40, 2}, {500, 7}}, .kind = Kind::kPrefillPass});
      waves.push_back({.owners = {{40, 5}, {500, 5}}, .kind = Kind::kStreams});
    }
    for (const Wave& w : waves) {
      Case c{.kind = w.kind, .past = w.owners[0].first, .rows = w.owners[0].second};
      SCOPED_TRACE(std::format("{} {} owners: {}", FormName(form), w.owners.size(), c.Name()));
      pl::StepShape shape = StepShapeOf(im, c);
      shape.segments.clear();
      const std::uint32_t cells = im.cells;
      for (std::size_t s = 0; s < w.owners.size(); ++s) {
        const auto [past, rows] = w.owners[s];
        const bool head = w.kind != Kind::kStateOnly && w.kind != Kind::kPrefillPass;
        shape.segments.push_back(
            {.slot = static_cast<std::uint8_t>(s),
             .rows = rows,
             .past = past,
             .read_cells = std::min<std::uint32_t>((past + rows + 255) / 256 * 256, cells),
             .outputs = !head                     ? 0
                        : w.kind == Kind::kVerify ? rows
                                                  : 1,
             .want_head = head,
             .greedy =
                 w.kind == Kind::kToken || w.kind == Kind::kDraft || w.kind == Kind::kVerify});
      }
      const Lowered l = Lower(im, c, registry, nullptr, &shape);
      ASSERT_TRUE(l.lowered.has_value());
      ExpectSaveRowsLive(l);
      std::map<kg::LlmpOp, std::size_t> ops;
      for (const ggml_tensor* n : l.lowered->nodes) {
        if (n->op == GGML_OP_CUSTOM) ++ops[kg::LlmpOpOf(n)];
      }
      const std::size_t owners = w.owners.size();
      if (!c.drafter()) {
        EXPECT_EQ(ops[kg::LlmpOp::kQsaPool], 12U * owners);
        EXPECT_EQ(ops[kg::LlmpOp::kGdnConv], 36U * owners);
      } else {
        EXPECT_EQ(ops[kg::LlmpOp::kQsaPool], owners);
      }
      const std::size_t outputs =
          w.kind == Kind::kStateOnly || w.kind == Kind::kPrefillPass ? 0 : owners;
      std::set<std::uint8_t> segments;
      for (const kn::OutputSlot& o : l.lowered->outputs) segments.insert(o.segment);
      EXPECT_EQ(segments.size(), outputs);
    }
  }
}

// A verify's commit (components/step.h BuildCommit) of k of its rows, one
// owner or two, lowers under the target's policy: each Gated DeltaNet
// layer's recurrence over the kept rows' saves in place (llmp.gdn.step,
// written), its convolution history and the n-gram layer's stored, the
// saves read where the verify stored them (Addresses::save).
TEST(GgmlLowerQwen38Test, CommitsOfKeptRowsLower) {
  for (const fx::Form form : {fx::Form::kCutlass, fx::Form::kGgmlExperts}) {
    const Import& im = ImportOf(form);
    const ex::Registry registry = PlanningRegistry();
    std::vector<const md::StateLayoutDerived*> stages;
    for (const auto& l : im.layouts_mtp) stages.push_back(&l);
    const std::vector<std::span<const co::BoundTensor>> tables{im.source.tensors,
                                                               im.mtp_source.tensors};
    for (const std::uint32_t owners : {1U, 2U}) {
      for (const std::uint32_t keep : {1U, 2U, 3U}) {
        SCOPED_TRACE(std::format("{} {} owners, {} kept", FormName(form), owners, keep));
        pl::StepShape kept;
        kept.phase = pl::Phase::kVerify;
        for (std::uint32_t s = 0; s < owners; ++s) {
          kept.segments.push_back({.slot = static_cast<std::uint8_t>(s),
                                   .rows = keep,
                                   .past = 40 + s,
                                   .read_cells = 256,
                                   .outputs = 0,
                                   .want_head = false});
        }
        auto commit = co::BuildCommit(*im.spec_mtp, 0, kept,
                                      {.tensors = im.source.tensors,
                                       .state = &im.layouts_mtp[0],
                                       .stages = stages,
                                       .artifacts = tables});
        ASSERT_TRUE(commit.has_value()) << commit.error();
        auto plan = pl::Plan(commit->graph.Clone(), registry, Gb10(), im.policy,
                             {.capturable = true,
                              .written = &kg::WrittenBytes,
                              .within = &kg::WithinRoot,
                              .shape = &kept});
        ASSERT_TRUE(plan.has_value()) << plan.error().detail;
        const kn::Addresses addresses{};
        const kn::LowerInput input{*plan, *commit, kept, addresses, kContext};
        auto lowered = kg::LowerStep(input, registry, Device(), {.device_masks = true});
        ASSERT_TRUE(lowered.has_value()) << lowered.error().detail;
        std::size_t steps = 0, histories = 0, stores = 0;
        for (const ggml_tensor* n : lowered->nodes) {
          steps += n->op == GGML_OP_CUSTOM && kg::LlmpOpOf(n) == kg::LlmpOp::kGdnStep ? 1 : 0;
          histories +=
              n->op == GGML_OP_CUSTOM && kg::LlmpOpOf(n) == kg::LlmpOp::kGdnHistory ? 1 : 0;
          stores += n->op == GGML_OP_SET_ROWS ? 1 : 0;
        }
        // Each layer's recurrence and history, and the n-gram layer's.
        EXPECT_EQ(steps, 36U * owners);
        EXPECT_EQ(histories, 36U * owners);
        EXPECT_EQ(stores, ((36U + 1U) * owners));
        EXPECT_TRUE(lowered->outputs.empty());
      }
    }
  }
}

// With llmp.gdn.conv explicitly disabled, Gated
// DeltaNet's convolution is GGML's, its input padded to whole 32-token
// windows past 32 tokens (RE-032), as the unfused builder pads it.
TEST(GgmlLowerQwen38Test, GgmlsConvolutionPadsPartialWindows) {
  const Import& im = ImportOf(fx::Form::kCutlass);
  const ex::Registry registry = PlanningRegistry();
  pl::Policy policy = im.policy;
  std::erase(policy.enabled, std::string("gdn_conv"));
  policy.disabled.emplace_back("gdn_conv");
  pl::Canonicalize(policy);
  ASSERT_TRUE(pl::Disabled(policy, "gdn_conv"));
  for (const std::uint32_t rows : {8U, 32U, 40U, 64U}) {
    SCOPED_TRACE(rows);
    const Case c{.kind = Kind::kHead, .past = 100, .rows = rows};
    const Lowered w = Lower(im, c, registry, &policy);
    ASSERT_TRUE(w.lowered.has_value());
    std::size_t convs = 0;
    for (const ggml_tensor* n : w.lowered->nodes) {
      if (n->op != GGML_OP_SSM_CONV) continue;
      ++convs;
      // The window's taps, then the rows (padded to whole windows past 32).
      const std::int64_t padded = rows > 32 && rows % 32 != 0 ? (rows + 31) / 32 * 32 : rows;
      EXPECT_EQ(n->src[0]->ne[0], 3 + padded);
      if (padded != rows) EXPECT_EQ(n->src[0]->src[1]->op, GGML_OP_CONT);  // the copy
      EXPECT_EQ(n->ne[1], padded);
    }
    EXPECT_EQ(convs, 36U);  // each Gated DeltaNet layer's
  }
}

// The default policies: each import's form, and the drafter's.
TEST(GgmlLowerQwen38Test, DefaultPoliciesFollowTheImportsForm) {
  const Import& cutlass = ImportOf(fx::Form::kCutlass);
  const Import& ggml = ImportOf(fx::Form::kGgmlExperts);
  const Import& gguf = ImportOf(fx::Form::kGguf);
  EXPECT_FALSE(pl::Disabled(cutlass.policy, "moe_experts"));
  EXPECT_FALSE(pl::Disabled(cutlass.policy, "moe_glu"));
  EXPECT_FALSE(pl::Disabled(ggml.policy, "moe_experts"));
  EXPECT_FALSE(pl::Disabled(ggml.policy, "moe_glu"));
  EXPECT_FALSE(pl::Disabled(gguf.policy, "vecq_glu"));
  EXPECT_FALSE(pl::Disabled(gguf.policy, "hc_prep"));
  EXPECT_FALSE(pl::Disabled(gguf.policy, "hc_mix"));
  for (const Import* im : {&cutlass, &ggml, &gguf}) {
    EXPECT_FALSE(pl::Disabled(im->policy, "rms_norm_mul"));
    EXPECT_FALSE(pl::Disabled(im->policy, "owner_attention"));
    EXPECT_FALSE(pl::Disabled(im->policy, "block_sparse_attention"));
    // Production QSA selects its own fused attention, so it never enabled
    // the old generic sparse-attention product bundle. Keep that arithmetic.
    EXPECT_EQ(im->policy.products, pl::ProductPolicy{});
    EXPECT_EQ(im->mtp_policy.products, pl::ProductPolicy{});
  }
  EXPECT_FALSE(pl::Disabled(cutlass.mtp_policy, "bf16_inputs"));
  EXPECT_FALSE(pl::Disabled(cutlass.mtp_policy, "mxfp8_inputs"));
  // No step rules: a solo prompt tail of 9 to 16 rows keeps the GGUF
  // form's vector GLU and inputs (no other model's row gates apply).
  for (const Import* im : {&cutlass, &ggml, &gguf}) EXPECT_TRUE(im->policy.rules.empty());
  for (std::uint64_t rows = 1; rows <= 16; ++rows) {
    for (const std::size_t segments : {std::size_t{1}, std::size_t{2}}) {
      const pl::Policy step = pl::ForStep(gguf.policy, rows, segments).value();
      EXPECT_EQ(step, gguf.policy) << rows << " rows";
      EXPECT_FALSE(pl::Disabled(step, "vecq_glu")) << rows << " rows";
      EXPECT_FALSE(pl::Disabled(step, "vecq_inputs")) << rows << " rows";
      EXPECT_EQ(step.max_decode_rows, 16U);
      EXPECT_EQ(step.vector_rows, 0U);
    }
  }
}

TEST(GgmlLowerQwen38Test, KeptNormResultsAreNotReplacedByPreparedRows) {
  for (const auto fn : {ir::NormGateFn::kSigmoid, ir::NormGateFn::kSilu}) {
    for (const std::uint32_t retained : {0U, 1U, 2U}) {
      SCOPED_TRACE(retained);
      ir::Builder b;
      const ir::TensorId x =
          b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({128, 1, 17, 1})));
      const ir::TensorId norm =
          b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({128})));
      const ir::TensorId gate =
          b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({128, 1, 17, 1})));
      const std::array sources{x, norm, gate};
      const ir::TensorId value =
          b.Fused(ir::FusedKind::kNormGate, sources, ir::NormGateParams{.gate = fn})[0];
      const ir::TensorId flat = b.Reshape(value, ir::ShapeOf({128, 17}));
      const ir::TensorId prepared =
          b.Prepare(flat, llmp::format::Encoding::kMxfp8, llmp::format::Layout::kScaleAtoms);
      b.Output(prepared);
      if (retained == 1) b.Keep(value);
      if (retained == 2) b.Keep(flat);
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph.has_value()) << graph.error().detail;
      EXPECT_EQ(kg::WithinRoot(*graph, prepared).valid(), retained == 0);
      co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
      for (const ir::TensorId input : sources) {
        step.inputs.push_back({.tensor = input, .kind = co::InputKind::kFeatures});
      }
      const ex::Registry registry = PlanningRegistry();
      auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), {},
                           {.staged = std::vector<ir::TensorId>(sources.begin(), sources.end()),
                            .written = &kg::WrittenBytes,
                            .within = &kg::WithinRoot});
      ASSERT_TRUE(plan.has_value()) << plan.error().detail;
      const pl::StepShape shape{};
      const kn::Addresses addresses{};
      const kn::LowerInput input{*plan, step, shape, addresses};
      auto lowered = kg::LowerStep(input, registry, Device());
      ASSERT_TRUE(lowered.has_value()) << lowered.error().detail;
      std::size_t separate = 0;
      for (const ggml_tensor* node : lowered->nodes) {
        if (node->op == GGML_OP_CUSTOM && kg::LlmpOpOf(node) == kg::LlmpOp::kGdnNormGate)
          EXPECT_EQ(kg::LlmpOpInt(node, 1), static_cast<std::int32_t>(fn));
        if (node->op == GGML_OP_CUSTOM && kg::LlmpOpOf(node) == kg::LlmpOp::kMxfp8Quantize) {
          ++separate;
        }
      }
      EXPECT_EQ(separate, retained == 0 ? 0U : 1U);
    }
  }
}

TEST(GgmlLowerQwen38Test, FastCapabilitiesPreserveTheOperationsParameters) {
  const ex::Capability* lo = kg::CapabilityOf(kg::kHcLoName);
  ASSERT_NE(lo, nullptr);
  ASSERT_NE(lo->accepts, nullptr);
  ir::ScaledUnaryParams unary;
  ex::CapabilityQuery q{.arch = 121, .params = std::as_bytes(std::span(&unary, 1))};
  for (const std::uint32_t streams : {1U, 2U, 3U, 4U, 8U}) {
    unary.scale = 1.0F / static_cast<float>(streams);
    EXPECT_TRUE(lo->accepts(q));
  }
  for (const float scale : {0.3F, 0.0F, -1.0F, 1.0F / 9.0F}) {
    unary.scale = scale;
    EXPECT_FALSE(lo->accepts(q));
  }

  const ex::Capability* prep = kg::CapabilityOf(kg::kQsaPrepName);
  const ex::Capability* pool = kg::CapabilityOf(kg::kQsaPoolName);
  ASSERT_NE(prep, nullptr);
  ASSERT_NE(pool, nullptr);
  ir::NormRopeParams norm;
  norm.rope.mode = ir::RopeMode::kInterleavedMulti;
  norm.rope.dims = 64;
  const std::array operands{ex::OperandFormat{.operand = 0, .ne = {128, 4, 3, 1}},
                            ex::OperandFormat{.operand = 2}, ex::OperandFormat{.operand = 5}};
  q.operands = operands;
  q.params = std::as_bytes(std::span(&norm, 1));
  EXPECT_TRUE(prep->accepts(q));
  ir::BlockPoolParams pooled{.ratio = 16, .rope = norm.rope};
  q.params = std::as_bytes(std::span(&pooled, 1));
  EXPECT_TRUE(pool->accepts(q));
  const ir::RopeParams plain = norm.rope;
  for (std::uint32_t change = 0; change < 6; ++change) {
    norm.rope = plain;
    switch (change) {
      case 0:
        norm.rope.inverse = true;
        break;
      case 1:
        norm.rope.table = true;
        break;
      case 2:
        norm.rope.offset = 2;
        break;
      case 3:
        norm.rope.scale = 0.5F;
        break;
      case 4:
        norm.rope.ext_factor = 1.0F;
        break;
      case 5:
        norm.rope.attn_factor = 2.0F;
        break;
    }
    q.params = std::as_bytes(std::span(&norm, 1));
    EXPECT_FALSE(prep->accepts(q));
    pooled.rope = norm.rope;
    q.params = std::as_bytes(std::span(&pooled, 1));
    EXPECT_FALSE(pool->accepts(q));
  }
}

// The validator sees the actual unchanged lower result in both address modes.
// A refusal destroys an unsubmitted program; it never provides a replacement.
class ValidationProgram final : public kn::StepProgram {
 public:
  ValidationProgram(std::uint32_t& destroyed, std::uint32_t& positioned)
      : destroyed_(destroyed), positioned_(positioned) {}
  ~ValidationProgram() override { ++destroyed_; }
  std::uint64_t scratch_bytes() const override { return 0; }
  std::uint64_t host_bytes() const override { return sizeof(*this); }
  std::uint64_t launches() const override { return 0; }
  std::span<const kn::InputSlot> inputs() const override { return {}; }
  std::span<const kn::OutputSlot> outputs() const override { return {}; }
  std::span<const kn::TensorUse> tensors() const override { return {}; }
  std::expected<void, kn::StepFailure> Bind(kn::StepLaunch&) override { return {}; }
  std::expected<void, kn::StepFailure> Position(std::span<const std::uint32_t>) override {
    ++positioned_;
    return {};
  }
  std::expected<void, kn::StepFailure> Run(kn::StepLaunch&,
                                           llmp::providers::NativeStream) const override {
    return std::unexpected(kn::StepFailure{.detail = "validation control never executes"});
  }

 private:
  std::uint32_t& destroyed_;
  std::uint32_t& positioned_;
};

struct ValidationFunding {
  std::uint64_t control_bytes = 0, capacity = 0, used = 0;
  std::uint32_t charges = 0, releases = 0;
  bool Charge(std::uint64_t bytes) {
    if (used > capacity || bytes > capacity - used) return false;
    used += bytes;
    ++charges;
    return true;
  }
  void Release(std::uint64_t bytes) {
    EXPECT_LE(control_bytes, used);
    if (used < control_bytes) return;
    EXPECT_LE(bytes, used - control_bytes);
    if (bytes <= used - control_bytes) used -= bytes;
    ++releases;
  }
};

TEST(GgmlLowerQwen38Test, ReadOnlyValidationSeesRealProgramAndRefusesBothAddressModes) {
  const auto& im = ImportOf(fx::Form::kCutlass);
  auto registry = PlanningRegistry();
  std::uint32_t lowers = 0, validations = 0, destroyed = 0, positioned = 0;
  bool refuse = false, lower_refuses = false;
  const kn::StepProgram* produced = nullptr;
  // This fake lowerer allocates no native arena. Its real report loan borrows
  // the reviewed compiler inventory for the actual planned graph; the payer
  // and its shared control blocks have their own explicit fixture payment.
  constexpr auto control_bytes =
      sizeof(ValidationFunding) + sizeof(kn::ConstructionFunder) + 2 * 64;
  static_assert(llmp::test_support::compiler_inventory::GraphBytes(
                    ir::Graph::kMaxNodes, ir::Graph::kMaxTensors) <= UINT64_MAX - control_bytes);
  auto funding = std::make_shared<ValidationFunding>();
  funding->control_bytes = funding->capacity = funding->used = control_bytes;
  auto payer = std::make_shared<kn::ConstructionFunder>(kn::ConstructionFunder{
      .charge = [funding](std::uint64_t bytes) { return funding->Charge(bytes); },
      .release = [funding](std::uint64_t bytes) { funding->Release(bytes); }});
  kn::Addresses addresses{};
  pl::StepShape shape{.phase = pl::Phase::kScore,
                      .segments = {{.rows = 2, .read_cells = 256, .outputs = 2}}};
  auto source = llmp::engine::ComponentStepSource::Create(
      {.spec = &*im.spec,
       .bindings = {.tensors = im.source.tensors, .state = &im.layouts.front()},
       .slots = 1,
       .policy = im.policy,
       .backend = {.registry = &registry,
                   .device = Gb10(),
                   .written = kg::WrittenBytes,
                   .lower = [&](const kn::LowerInput& input)
                       -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
                     ++lowers;
                     if (lower_refuses) return std::unexpected("real lower refused");
                     funding->capacity =
                         control_bytes +
                         llmp::test_support::compiler_inventory::GraphBytes(
                             input.plan.graph.nodes().size(), input.plan.graph.tensor_count());
                     auto result = std::make_unique<ValidationProgram>(destroyed, positioned);
                     produced = result.get();
                     return result;
                   },
                   .construction_funder = payer},
       .validate_program = [&](const kn::LowerInput& input,
                               const kn::StepProgram& program) -> std::expected<void, std::string> {
         ++validations;
         EXPECT_EQ(&program, produced);
         EXPECT_EQ(input.addresses.activations, addresses.activations);
         EXPECT_EQ(input.token->shape, shape);
         EXPECT_EQ(input.token->step.graph.tensor(input.token->step.logits).shape().ne[0],
                   fam::Qwen38Flash().vocab);
         if (refuse) return std::unexpected("full-head observer refused");
         return {};
       }});
  ASSERT_TRUE(source) << (source ? "" : source.error());
  for (const auto base : {std::uint64_t{0}, std::uint64_t{1} << 42U}) {
    addresses.activations = base;
    refuse = false;
    {
      auto accepted = (*source)->Build(shape, addresses);
      ASSERT_TRUE(accepted) << (accepted ? "" : accepted.error());
      EXPECT_GT(funding->used, control_bytes);
      EXPECT_LE(funding->used, funding->capacity);
      // The paid report decorator forwards to that same validated program.
      EXPECT_TRUE((*accepted)->Position({}));
      EXPECT_GT((*accepted)->host_bytes(), produced->host_bytes());
      EXPECT_EQ(destroyed, validations - 1);
    }
    EXPECT_EQ(funding->used, control_bytes);
    refuse = true;
    auto rejected = (*source)->Build(shape, addresses);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error(), "full-head observer refused");
    EXPECT_EQ(funding->used, control_bytes);
  }
  EXPECT_EQ(lowers, 4U);
  EXPECT_EQ(validations, 4U);
  EXPECT_EQ(destroyed, 4U);
  EXPECT_EQ(positioned, 2U);
  lower_refuses = true;
  auto rejected = (*source)->Build(shape, addresses);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error(), "real lower refused");
  EXPECT_EQ(validations, 4U);
  EXPECT_EQ(funding->used, control_bytes);
  EXPECT_EQ(funding->charges, 2U);
  EXPECT_EQ(funding->releases, funding->charges);
}

// Actual component emission and selection, with synthetic reserved addresses:
// no model payload, device allocation or dispatch is needed to prove eligibility.
TEST(GgmlLowerQwen38Test, ReservedWeightFactsSelectTypedProductsAndRejectBrokenPromises) {
  const auto& im = ImportOf(fx::Form::kCutlass);
  auto spec = *im.spec;
  spec.stages.front().layers.resize(fam::Qwen38Flash().ple_layer + 1);
  auto layout = co::StageLayoutUniform(spec, 0, kContext, kMaxRows, kExtent);
  ASSERT_TRUE(layout) << layout.error();
  auto policy = im.policy;
  ASSERT_EQ(std::erase(policy.enabled, "owner_row_invariant"), 1U);
  policy.disabled.emplace_back("owner_row_invariant");
  pl::Canonicalize(policy);
  auto device = Gb10();
  for (const auto n : {512U, 2560U, 10240U})
    device.tuned.push_back({.implementation = kg::kMulMatVecFRows,
                            .rows = 8,
                            .k = 2560,
                            .n = n,
                            .output_encoding = llmp::format::Encoding::kF32});
  const auto registry = PlanningRegistry();
  pl::StepShape shape{.phase = pl::Phase::kScore, .segments = {}, .read_align = 256};
  for (std::uint8_t owner = 0; owner < 4; ++owner)
    shape.segments.push_back({.slot = owner, .rows = 2, .read_cells = 256, .outputs = 2});
  // Positive measuring/bound; unknown; malformed; broken binding; unrelated
  // WeightRef; wrong PartRole; zero actual address in either mode; and valid
  // facts without tuning or with a mismatched output-typed tuning record.
  // No negative changes the selection predicate.
  for (int mode = 0; mode < 11; ++mode) {
    SCOPED_TRACE(mode);
    auto selected_device = device;
    if (mode == 9) selected_device.tuned.clear();
    if (mode == 10)
      for (auto& tune : selected_device.tuned) tune.output_encoding = llmp::format::Encoding::kF16;
    bool lowered = false;
    std::map<std::pair<ir::WeightRef, llmp::format::PartRole>, std::uint64_t> bases;
    kn::Addresses addresses{};
    addresses.activation_alignment = kExtent;
    addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole part) {
      if (mode == 7 || mode == 8) return std::uint64_t{0};
      const auto key = std::pair{ref, part};
      auto [at, inserted] = bases.emplace(key, 0);
      if (inserted) at->second = (std::uint64_t{1} << 45) + bases.size() * (std::uint64_t{1} << 32);
      return at->second + (mode == 4 ? 2U : 0U);
    };
    if (mode != 2) {
      addresses.weight_alignment = [&](const ir::WeightRef& ref, llmp::format::PartRole part) {
        if (mode == 5 && ref.stage != 99) return std::uint64_t{0};
        if (mode == 6 && part != llmp::format::PartRole::kBlockScales) return std::uint64_t{0};
        return mode == 3 ? std::uint64_t{3} : kExtent;
      };
    }
    addresses.state = [](const ir::StateRef& ref) {
      return (std::uint64_t{1} << 52) + std::uint64_t{ref.segment} * (std::uint64_t{1} << 38) +
             std::uint64_t{ref.layer} * (std::uint64_t{1} << 32) +
             std::uint64_t{ref.tensor} * (std::uint64_t{1} << 26);
    };
    if (mode == 1 || mode == 8) {
      addresses.activations = std::uint64_t{1} << 58;
      addresses.activation_bytes = std::uint64_t{1} << 32;
    }
    auto source = llmp::engine::ComponentStepSource::Create(
        {.spec = &spec,
         .bindings = {.tensors = im.source.tensors, .state = &*layout},
         .slots = 4,
         .policy = policy,
         .backend = {
             .registry = &registry,
             .device = selected_device,
             .written = kg::WrittenBytes,
             .within = kg::WithinRoot,
             .lower = [&](const kn::LowerInput& input)
                 -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
               lowered = true;
               std::set<std::uint64_t> selected_widths;
               for (const auto& selected : input.plan.steps) {
                 if (registry.at(selected.implementation).name != kg::kMulMatVecFRows) continue;
                 for (const auto id : selected.nodes) {
                   const auto& node = input.plan.graph.node(id);
                   EXPECT_EQ(node.key, ir::Primitive(ir::Op::kMatMul));
                   const auto& weight = input.plan.graph.tensor(node.src[0]);
                   const auto& x = input.plan.graph.tensor(node.src[1]);
                   const auto& y = input.plan.graph.tensor(node.out[0]);
                   EXPECT_EQ(weight.format.encoding, llmp::format::Encoding::kBF16);
                   EXPECT_EQ(x.format.encoding, llmp::format::Encoding::kF32);
                   EXPECT_EQ(y.format.encoding, llmp::format::Encoding::kF32);
                   EXPECT_EQ(x.ne(1), 8U);
                   EXPECT_EQ(weight.ne(0), 2560U);
                   EXPECT_EQ(node.Params<ir::ProductParams>().accumulate, ir::Accumulate::kDefault);
                   selected_widths.insert(weight.ne(1));
                 }
               }
               const std::set<std::uint64_t> expected =
                   mode < 2 ? std::set<std::uint64_t>{512, 2560, 10240} : std::set<std::uint64_t>{};
               EXPECT_EQ(selected_widths, expected);
               auto physical = kg::LowerStep(
                   input, registry, Device(),
                   {.expert_stride = [&](const ir::WeightRef& ref) -> std::uint64_t {
                     return ref.stage == 0 && ref.expert_array && ref.layer < im.stride.size()
                                ? im.stride[ref.layer]
                                : 0;
                   }});
               EXPECT_TRUE(physical) << (physical ? "" : physical.error().detail);
               if (physical) {
                 const auto count = std::ranges::count_if(
                     physical->plan.steps,
                     [](const auto& step) { return step.implementation == kg::kMulMatVecFRows; });
                 EXPECT_EQ(count, mode < 2 ? fam::Qwen38Flash().ple_layer + 3 : 0);
               }
               return std::unexpected("observed reserved-weight selection");
             }}});
    ASSERT_TRUE(source) << source.error();
    auto result = (*source)->Build(shape, addresses);
    ASSERT_FALSE(result);
    EXPECT_EQ(lowered, mode != 3 && mode != 4 && mode != 7 && mode != 8);
    if (mode == 3 || mode == 4 || mode == 7 || mode == 8)
      EXPECT_NE(result.error().find("alignment"), std::string::npos);
    else
      EXPECT_EQ(result.error(), "observed reserved-weight selection");
  }
}

}  // namespace

TEST(GgmlLowerQwen38Test, GdnGatesCapabilityUsesRepresentableShapes) {
  using llmp::format::Encoding;
  const ex::Capability* cap = kg::CapabilityOf(kg::kGdnGatesName);
  ASSERT_NE(cap, nullptr);
  ASSERT_NE(cap->accepts, nullptr);
  constexpr std::array<std::array<std::uint64_t, 2>, 5> shapes = {
      {{48, 1}, {1, 17}, {37, 19}, {257, 17}, {513, 1}}};
  for (const auto& [heads, rows] : shapes) {
    SCOPED_TRACE(std::format("heads={} rows={}", heads, rows));
    std::array operands{ex::OperandFormat{.operand = 0,
                                          .encoding = Encoding::kF32,
                                          .layout = llmp::format::Layout::kDense,
                                          .ne = {heads, rows, 1, 1}},
                        ex::OperandFormat{.operand = 1,
                                          .encoding = Encoding::kF32,
                                          .layout = llmp::format::Layout::kDense,
                                          .ne = {heads, rows, 1, 1}},
                        ex::OperandFormat{.operand = 2,
                                          .encoding = Encoding::kF32,
                                          .layout = llmp::format::Layout::kDense,
                                          .ne = {heads, 1, 1, 1}},
                        ex::OperandFormat{.operand = 3,
                                          .encoding = Encoding::kF32,
                                          .layout = llmp::format::Layout::kDense,
                                          .ne = {heads, 1, 1, 1}},
                        ex::OperandFormat{.operand = ex::kOutputOperand,
                                          .encoding = Encoding::kF32,
                                          .layout = llmp::format::Layout::kDense,
                                          .ne = {1, heads, rows, 1}},
                        ex::OperandFormat{.operand = ex::kOutputOperand + 1,
                                          .encoding = Encoding::kF32,
                                          .layout = llmp::format::Layout::kDense,
                                          .ne = {1, heads, rows, 1}}};
    ex::CapabilityQuery q{.op = cap->op, .operands = operands, .arch = 121};
    EXPECT_TRUE(cap->accepts(q));
    for (auto& operand : operands) {
      const auto saved = operand;
      for (std::size_t dim = 0; dim < 4; ++dim) {
        for (const auto bad : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
          operand.ne[dim] = bad;
          EXPECT_FALSE(cap->accepts(q));
          operand = saved;
        }
      }
      operand.encoding = Encoding::kBF16;
      EXPECT_FALSE(cap->accepts(q));
      operand = saved;
      operand.ne[0] += 1;
      EXPECT_FALSE(cap->accepts(q));
      operand = saved;
    }
    q.operands = std::span(operands).first(5);
    EXPECT_FALSE(cap->accepts(q));
  }
  constexpr auto max_values =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / (2 * sizeof(float));
  EXPECT_TRUE(kg::GdnGatesShapeFits(1, max_values));
  EXPECT_FALSE(kg::GdnGatesShapeFits(1, max_values + 1));
  EXPECT_FALSE(kg::GdnGatesShapeFits(max_values, 2));
}
