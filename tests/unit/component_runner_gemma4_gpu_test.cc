// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The one model runner (engine/model_runner.h) over the component step
// source (engine/component_source.h) and the GGML module's lowering
// (kernels/ggml/step_program.h), on the real Gemma 4 26B-A4B and 31B
// published v1 artifacts and their assistants (label `models`: their stored
// specs and tensors, each assistant composed with its target from its
// approved values), under the gemma4 mapping's policy for each serving
// envelope (the component source plans under it, gated as the model's
// approval gates it): within the qualified envelope at 4,096 and two owners (every
// row to the head, the fused norm-mul, norm-rope and norm-add, the
// 26B-A4B's fused routing and scaled reduce, the 31B's quantized GeGLU,
// owner attention with bounded roots over unequal reads, device masks), at
// 8,192 and four owners (owner attention over equal reads, its masks joined
// in pairs), and outside it (the frontier head's narrowed final layer, the
// routing and reduce as their primitives). The owners run one script of
// waves: joined prefill chunks, ragged and single-owner chunks, state-only
// chunks, one owner past its sliding layers' ring until it wraps, joined
// one-row decodes returning logits and device-chosen tokens. The repeated
// decode shapes are captured on their second run and replayed after
// (D-090). The verify cases run a speculative verify's waves (four rows an
// owner, every row's outputs, then their Accept), alone, joined and ragged.
//
// The assistant's draft passes (two, joined over the owners, at each
// owner's committed endpoint after the prompt, reading the target's
// cache: one prompt ends at 37 rows, another at 256, where a frozen read
// covers exactly the past) run through the same runner as stage 1, the
// model's step source an engine::StagedStepSource over both stages of the
// composed spec, with the target's state borrowed. The target's steps
// write the assistant's feature tap (its output rows' final-norm rows);
// the first pass reads the tap's feature of the endpoint (the runner's
// device copy), the second its own first pass's next feature (the
// runner's carry), and their tokens are the top tokens of the prompt's
// frontier and of the first pass.
//
// Every output is finite, and a second runner over the same script gives
// every output bit for bit (the replays included). (The script was once
// compared output for output with the M3 runner's and assistant's under
// their serving options, a verify's four-row outputs with its Wave, bit
// for bit, retired with them after the migration's gate.)

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/execution_contract.h"
#include "engine/expert_slab.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "kernels/ggml/step_program.h"
#include "planner/step_shape.h"
#include "platform/host_probe.h"
#include "runtime/approvals.h"
#include "runtime/component_admission.h"
#include "spec/mapping.h"
#include "spec/stored.h"
#include "v1_frozen.h"

namespace {

namespace co = llmp::components;
namespace en = llmp::engine;
namespace ar = llmp::artifact;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace fs = std::filesystem;

constexpr std::uint32_t kMaxRows = 128;
constexpr std::uint32_t kVocab = 262144;

struct Case {
  std::string name;
  std::uint32_t size = 26;
  std::uint32_t context = 4096;
  std::uint32_t owners = 2;
  bool candidate = true;  // within the envelope
  bool assistant = false;
  // The prompt's rows before the drafts (0: 37, or 31 for more owners), in
  // joined waves of at most kMaxRows rows.
  std::uint32_t prompt = 0;
  // A speculative verify's waves instead of the serving script's tail: four
  // rows an owner, every row's outputs, committed by their Accept.
  bool verify = false;
};

constexpr std::uint32_t kVerifyRows = 4;

// The store's v1-ids.json names of the published target and assistant.
std::string_view TargetName(std::uint32_t size) { return size == 26 ? "gemma26" : "gemma31"; }
std::string_view AssistantName(std::uint32_t size) {
  return size == 26 ? "gemma26-assistant" : "gemma31-assistant";
}

// The sliding layers' window: the largest an attention component of the
// stored spec names.
std::uint32_t Window(const llmp::spec::ModelSpec& spec) {
  std::int64_t window = 0;
  for (const llmp::spec::Layer& layer : spec.stages.front().layers) {
    for (const llmp::spec::ComponentRef& c : layer.components) {
      const llmp::spec::Param* p = c.Find("window");
      if (p == nullptr) continue;
      if (const auto* v = std::get_if<std::int64_t>(&p->value)) window = std::max(window, *v);
    }
  }
  return static_cast<std::uint32_t>(window);
}

// The values serving plans the target mapping's policy with: the gates the
// model's approval holds, as flags (runtime/component_admission.cc
// ApprovedRule).
std::vector<llmp::spec::SourceValue> ApprovedGates(const ar::Artifact& artifact) {
  std::vector<llmp::spec::SourceValue> gates;
  const auto* approval =
      llmp::runtime::FindApproval(artifact.model().architecture, artifact.sources());
  EXPECT_NE(approval, nullptr) << "no approval for the published artifact";
  if (approval == nullptr) return gates;
  for (const std::string_view gate : approval->gates)
    gates.push_back({.key = std::string(gate), .value = true});
  return gates;
}

// ------------------------------------------------------------ the script

enum class Head : std::uint8_t { kLogits, kGreedy, kNone, kDraft, kAll };
struct Unit {
  std::uint32_t slot = 0;
  std::uint32_t rows = 0;
};
struct Wave {
  std::vector<Unit> units;
  Head head = Head::kLogits;
};
Wave W(std::vector<Unit> units, Head head = Head::kLogits) {
  return {.units = std::move(units), .head = head};
}
struct Planned {
  Wave wave;
  std::vector<std::uint32_t> past;
  std::vector<std::vector<std::int32_t>> tokens;  // a draft's: filled as it runs
};

std::int32_t TokenAt(std::uint32_t slot, std::uint32_t position) {
  return static_cast<std::int32_t>(100 + (((position * 7919U) + (slot * 104729U) + 13U) % 20000U));
}

std::vector<Unit> All(std::uint32_t owners, std::uint32_t rows) {
  std::vector<Unit> out;
  for (std::uint32_t i = 0; i < owners; ++i) out.push_back({i, rows});
  return out;
}

// `local_cells`: the sliding layers' ring, which owner 0 then wraps.
std::vector<Planned> Script(const Case& c, std::uint32_t local_cells) {
  const std::uint32_t n = c.owners;
  // A joined prefill of equal rows within the chunk bound.
  std::vector<Wave> waves;
  const std::uint32_t prompt = c.prompt != 0 ? c.prompt : n <= 2 ? 37U : 31U;
  // Today's joined waves hold kMaxRows rows in all.
  for (std::uint32_t done = 0; done < prompt; done += kMaxRows / n) {
    waves.push_back(W(All(n, std::min(kMaxRows / n, prompt - done))));
  }
  if (c.assistant) {
    // Two draft passes over every owner's endpoint.
    waves.push_back(W(All(n, 1), Head::kDraft));
    waves.push_back(W(All(n, 1), Head::kDraft));
  }
  const auto more = [&](std::vector<Wave> w) { waves.insert(waves.end(), w.begin(), w.end()); };
  if (c.verify) {
    // One owner's verify three times (the shape's second and third runs),
    // both owners' joined, a ragged pair, then a joined decode over the
    // verified state.
    more({W({{0, kVerifyRows}}, Head::kAll), W({{0, kVerifyRows}}, Head::kAll),
          W({{0, kVerifyRows}}, Head::kAll), W(All(n, kVerifyRows), Head::kAll),
          W(All(n, kVerifyRows), Head::kAll), W({{0, 2}, {1, kVerifyRows}}, Head::kAll),
          W(All(n, 1))});
    std::vector<Planned> out;
    std::vector<std::uint32_t> at(n, 0);
    for (const Wave& w : waves) {
      Planned p;
      p.wave = w;
      for (const Unit& u : w.units) {
        p.past.push_back(at.at(u.slot));
        std::vector<std::int32_t> ids;
        for (std::uint32_t i = 0; i < u.rows; ++i)
          ids.push_back(TokenAt(u.slot, at.at(u.slot) + i));
        at.at(u.slot) += u.rows;
        p.tokens.push_back(std::move(ids));
      }
      out.push_back(std::move(p));
    }
    return out;
  }
  more({W(All(n, 1)),                 // joined decode, equal reads
        W(All(n, 1), Head::kGreedy),  // its tokens chosen on the device
        W({{0, 100}, {1, 5}}),        // ragged: per owner
        W({{0, 1}}),                  // one owner
        W({{1, 3}}, Head::kNone),     // state-only
        W({{0, 128}}, Head::kNone),   // owner 0 past 256: unequal reads
        W(All(n, 1)), W(All(n, 1), Head::kGreedy)});
  std::vector<std::uint32_t> past(n, 0);
  for (const Wave& w : waves) {
    if (w.head == Head::kDraft) continue;
    for (const Unit& u : w.units) past.at(u.slot) += u.rows;
  }
  // Owner 0 past its ring (a state-only chunk at a time), then a chunk
  // returning its logits.
  while (past[0] + kMaxRows <= local_cells + kMaxRows + 64) {
    waves.push_back(W({{0, kMaxRows}}, Head::kNone));
    past[0] += kMaxRows;
  }
  waves.push_back(W({{0, kMaxRows}}));
  // Joined decodes over one wrapped ring and the short caches, each head's
  // shape three times: captured beside its second run, replayed on its
  // third.
  for (const Head head : {Head::kLogits, Head::kGreedy}) {
    for (int i = 0; i < 3; ++i) waves.push_back(W(All(n, 1), head));
  }
  waves.push_back(W(All(n, 9)));  // equal rows, unequal reads: per owner
  std::vector<Planned> out;
  past.assign(n, 0);
  for (const Wave& w : waves) {
    Planned p;
    p.wave = w;
    for (const Unit& u : w.units) {
      p.past.push_back(past.at(u.slot));
      std::vector<std::int32_t> ids;
      if (w.head != Head::kDraft) {
        for (std::uint32_t i = 0; i < u.rows; ++i)
          ids.push_back(TokenAt(u.slot, past.at(u.slot) + i));
        past.at(u.slot) += u.rows;
      }
      p.tokens.push_back(std::move(ids));
    }
    out.push_back(std::move(p));
  }
  return out;
}

std::uint32_t ReadOf(std::uint32_t end, std::uint32_t context) {
  return std::min(context, (end + 255) / 256 * 256);
}

pl::StepShape ShapeOf(const Planned& p, std::uint32_t context) {
  pl::StepShape shape;
  const bool draft = p.wave.head == Head::kDraft;
  shape.stage = draft ? 1 : 0;
  shape.phase = draft ? pl::Phase::kDraft : pl::Phase::kDecode;
  shape.read_align = 256;
  if (p.wave.head == Head::kAll) shape.phase = pl::Phase::kVerify;
  for (std::size_t i = 0; i < p.wave.units.size(); ++i) {
    const Unit& u = p.wave.units[i];
    if (u.rows > 1 && p.wave.head != Head::kAll) shape.phase = pl::Phase::kPrefill;
    // A draft reads its target's committed positions (frozen).
    const std::uint32_t end = p.past[i] + (draft ? 0 : u.rows);
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(u.slot),
                              .rows = u.rows,
                              .past = p.past[i],
                              .read_cells = ReadOf(end, context),
                              .outputs = p.wave.head == Head::kNone  ? 0U
                                         : p.wave.head == Head::kAll ? u.rows
                                                                     : 1U,
                              .group = 0,
                              .want_head = p.wave.head != Head::kNone,
                              .greedy = p.wave.head == Head::kGreedy});
  }
  return shape;
}

// Each wave's outputs by unit: a logits row's or a token's bytes (a
// draft's: its logits, then its next feature); none for a state-only wave.
using Outputs = std::vector<std::vector<std::vector<std::byte>>>;

template <typename T>
std::vector<std::byte> Bytes(std::span<const T> values) {
  const auto bytes = std::as_bytes(values);
  return {bytes.begin(), bytes.end()};
}

std::int32_t Top(std::span<const std::byte> logits) {
  std::vector<float> row(kVocab);
  std::memcpy(row.data(), logits.data(), kVocab * sizeof(float));
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

// ------------------------------------------------------------ the components

struct Component {
  Outputs outputs;
  std::uint64_t replayed = 0;
};

// The kernel source's own memory beside the runner (its cuBLAS handle and
// workspace, which the backend borrows): released by the node's teardown
// after the runner's, the backend first.
class KernelSource final : public en::PagedModel {
 public:
  explicit KernelSource(en::PagedNode& node)
      : resources(std::make_unique<en::RunnerResources>(node, 0, 0)) {}
  std::uint32_t stream() const override { return 0; }
  const llmp::catalog::Closure& fence_closure() const override { return nothing_; }
  std::vector<llmp::catalog::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override {
    backend.reset();
    std::vector<std::string> problems;
    resources->Release(problems);
    return en::support::Joined(problems);
  }
  std::unique_ptr<en::RunnerResources> resources;
  std::unique_ptr<kg::GgmlBackend> backend;

 private:
  llmp::catalog::Closure nothing_;
};

std::uint32_t ArrayIndex(const ar::Artifact& a, std::string_view name) {
  const auto arrays = a.expert_arrays();
  return static_cast<std::uint32_t>(std::ranges::find(arrays, name, &ar::ExpertArray::name) -
                                    arrays.begin());
}

// The model runner over the component sources for the script; the drafts'
// tokens the top tokens of the prompt's frontier and of the first pass, their
// features the runner's (the tap the target's steps wrote, then the carry).
Component Lowered(const Case& c, const fs::path& target, const fs::path& assistant,
                  const fs::path& scratch, std::span<const Planned> script) {
  Component out;
  auto ta = ar::Artifact::Open(target, {.expected_id = {}});
  EXPECT_TRUE(ta.has_value());
  if (!ta) return out;
  std::optional<ar::Artifact> aa;
  if (c.assistant) {
    auto opened = ar::Artifact::Open(assistant, {.expected_id = {}});
    EXPECT_TRUE(opened.has_value());
    if (!opened) return out;
    aa.emplace(std::move(*opened));
  }
  const auto* target_mapping = llmp::spec::FindMapping(ta->model().architecture);
  EXPECT_NE(target_mapping, nullptr);
  if (target_mapping == nullptr) return out;
  const llmp::spec::Mapping& mapping = *target_mapping;
  const auto values = ApprovedGates(*ta);
  const auto target_spec = llmp::spec::StoredSpec(*ta);
  EXPECT_TRUE(target_spec.has_value()) << target_spec.error();
  if (!target_spec) return out;
  const auto ts = llmp::spec::SourceOf(*ta);
  EXPECT_TRUE(ts.has_value()) << (ts ? "" : ts.error());
  if (!ts) return out;
  std::vector<llmp::spec::SourceValue> assistant_values;
  std::optional<llmp::spec::ModelSpec> composed;
  std::optional<llmp::spec::Source> as;
  const llmp::spec::Mapping* assistant_mapping = nullptr;
  if (c.assistant) {
    // The assistant's values as serving composes it: the approved pairing
    // checked, its values from the metadata it keeps.
    auto drafter_values = llmp::runtime::ComponentDrafterValues(*ta, *aa);
    EXPECT_TRUE(drafter_values.has_value()) << (drafter_values ? "" : drafter_values.error());
    if (!drafter_values) return out;
    assistant_values = std::move(*drafter_values);
    assistant_mapping = llmp::runtime::ComposedDrafterMapping(*aa);
    EXPECT_NE(assistant_mapping, nullptr);
    if (assistant_mapping == nullptr) return out;
    auto source = llmp::spec::SourceOf(*aa);
    EXPECT_TRUE(source.has_value()) << (source ? "" : source.error());
    if (!source) return out;
    as.emplace(std::move(*source));
    auto spec = llmp::spec::ComposeDrafter(*target_spec, mapping, *assistant_mapping, *as,
                                           assistant_values);
    EXPECT_TRUE(spec.has_value()) << spec.error();
    if (!spec) return out;
    composed.emplace(std::move(*spec));
  }
  const llmp::spec::ModelSpec& layout_spec = composed ? *composed : *target_spec;
  std::vector<md::StateLayoutDerived> layouts;
  for (std::uint16_t stage = 0; stage < layout_spec.stages.size(); ++stage) {
    auto layout = co::StageLayout(layout_spec, stage, c.context, kMaxRows);
    EXPECT_TRUE(layout.has_value()) << layout.error();
    if (!layout) return out;
    layouts.push_back(std::move(*layout));
  }
  std::vector<const md::StateLayoutDerived*> stage_layouts;
  for (const auto& l : layouts) stage_layouts.push_back(&l);
  std::vector<std::span<const co::BoundTensor>> artifacts{ts->tensors};
  if (as) artifacts.emplace_back(as->tensors);
  std::vector<pl::StepShape> shapes;
  for (const Planned& planned : script) {
    const pl::StepShape shape = ShapeOf(planned, c.context);
    if (std::ranges::find(shapes, shape) == shapes.end()) shapes.push_back(shape);
  }
  // Each weight's place, by what the steps bind: resources and expert
  // arrays of the target, the assistant's resources.
  std::map<llmp::ir::WeightRef, std::pair<bool, std::uint32_t>> place;  // (array, index)
  for (const pl::StepShape& shape : shapes) {
    const bool draft = shape.stage == 1;
    auto step = co::BuildStep(layout_spec, shape.stage, shape,
                              {.tensors = draft ? as->tensors : ts->tensors,
                               .state = &layouts.at(shape.stage),
                               .stages = stage_layouts,
                               .artifacts = artifacts});
    EXPECT_TRUE(step.has_value()) << step.error();
    if (!step) return out;
    for (const co::StepWeight& w : step->weights) {
      const ar::Artifact& a = w.ref.stage == 1 ? *aa : *ta;
      const std::string& name = w.parts.front().first;
      if (w.ref.expert_array) {
        place[w.ref] = {true, ArrayIndex(a, name)};
      } else {
        const auto found = a.FindResource(name);
        EXPECT_TRUE(found.has_value()) << name;
        if (found) place[w.ref] = {false, *found};
      }
    }
  }
  // The slabs and pitches serving takes (a layer's experts in one slab).
  auto slabs = en::ExpertSlabsOf(*ta, 256);
  EXPECT_TRUE(slabs.has_value()) << (slabs ? "" : slabs.error());
  if (!slabs) return out;

  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  auto source = std::make_unique<KernelSource>(node);
  std::unique_ptr<en::ModelRunner> runner;
  std::vector<en::PagedModel*> entered;
  const auto started = [&]() -> en::Status {
    if (auto x = node.Open(); !x) return x;
    if (auto x = source->resources->OpenCublas("lowered cuBLAS workspace"); !x) return x;
    const auto pitch = slabs->pitch;
    auto made = kg::GgmlBackend::Create(
        0, node.execution(), node.stream(0), &source->resources->cublas(),
        {.lower = {
             .expert_stride = [&place, pitch](const llmp::ir::WeightRef& ref) -> std::uint64_t {
               const auto at = place.find(ref);
               if (at == place.end() || !at->second.first || at->second.second >= pitch.size()) {
                 return 0;
               }
               return pitch[at->second.second];
             }}});
    if (!made) return std::unexpected(made.error());
    source->backend = std::move(*made);
    // Native construction has a separate finite fixture allowance. Retained
    // plans still acquire the runner's ordinary paid ownership before transfer.
    const auto available = llmp::platform::AvailableMemoryBytes();
    if (!available) return std::unexpected("native construction physical headroom is unknown");
    auto paid = en::ExecutionMetadataAllowance::Admit(node.catalog(), node.domain(), 128ULL << 20U,
                                                      *available, 6ULL << 30U);
    if (!paid) return std::unexpected(paid.error());
    if (auto funded = source->backend->FundConstruction(
            [paid = *paid](std::uint64_t n) { return paid->Charge(n); },
            [paid = *paid](std::uint64_t n) { paid->Release(n); });
        !funded)
      return std::unexpected(funded.error());
    if (auto x = source->resources->OpenSelectedWorkspace("selected kernel state",
                                                          source->backend->persistent_bytes());
        !x)
      return x;
    auto target_weights = std::make_unique<en::PagedWeights>();
    if (auto x = target_weights->Open(target); !x) return x;
    en::PagedWeights* tw = target_weights.get();
    en::PagedWeights* aw = nullptr;
    std::vector<std::unique_ptr<en::ComponentStepSource>> sources;
    // With the assistant, the target's steps are the composed spec's: they
    // write its feature tap.
    auto stage0 = en::ComponentStepSource::Create({.spec = &layout_spec,
                                                   .stage = 0,
                                                   .bindings = {.tensors = ts->tensors,
                                                                .state = &layouts[0],
                                                                .stages = stage_layouts,
                                                                .artifacts = artifacts},
                                                   .bound_digest = layout_spec.Digest(),
                                                   .mapping = &mapping,
                                                   .values = values,
                                                   .slots = c.owners,
                                                   .policy = std::nullopt,
                                                   .backend = source->backend->Backend()});
    if (!stage0) return std::unexpected(stage0.error());
    sources.push_back(std::move(*stage0));
    en::RunnerConfig config;
    config.slots = {.model = "lowered",
                    .slots = c.owners,
                    .stages = {{.name = "target", .layout = layouts[0]}},
                    .spec = layout_spec.Digest(),
                    .read_align = 256,
                    .spill_place = {},
                    .spill_directory = scratch,
                    .invalidate = {},
                    .drop_plans = {}};
    config.weights.push_back(
        {.weights = std::move(target_weights), .places = slabs->places, .slabs = slabs->slabs});
    if (c.assistant) {
      auto drafter = en::ComponentStepSource::Create({.spec = &*composed,
                                                      .stage = 1,
                                                      .bindings = {.tensors = as->tensors,
                                                                   .state = &layouts[1],
                                                                   .stages = stage_layouts,
                                                                   .artifacts = artifacts},
                                                      .bound_digest = composed->Digest(),
                                                      .mapping = assistant_mapping,
                                                      .values = assistant_values,
                                                      .slots = c.owners,
                                                      .policy = std::nullopt,
                                                      .backend = source->backend->Backend()});
      if (!drafter) return std::unexpected(drafter.error());
      sources.push_back(std::move(*drafter));
      config.slots.stages.push_back({.name = "assistant", .layout = layouts[1]});
      auto weights = std::make_unique<en::PagedWeights>();
      if (auto x = weights->Open(assistant); !x) return x;
      aw = weights.get();
      std::vector<en::GroupPlace> places(aw->artifact().groups().size(), en::GroupPlace::kDevice);
      config.weights.push_back(
          {.weights = std::move(weights), .places = std::move(places), .slabs = {}});
    }
    if (c.verify) config.max_verify_rows = kVerifyRows;
    config.setup_shapes = shapes;
    config.launch =
        [&](std::uint64_t) -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return source->backend->Launch(
          {.base = node.pool().base, .size = llmp::base::Bytes(node.pool().bytes)},
          {.base = source->resources->selected_workspace().base,
           .bytes = source->resources->selected_workspace().bytes});
    };
    config.weight_address = [&place, tw, aw](const llmp::ir::WeightRef& ref,
                                             llmp::format::PartRole) -> std::uint64_t {
      const auto at = place.find(ref);
      if (at == place.end()) return 0;
      const auto [array, index] = at->second;
      if (ref.stage == 1) return aw == nullptr ? 0 : aw->resource_address(index);
      return array ? tw->array_address(index) : tw->resource_address(index);
    };
    std::unique_ptr<en::StepSource> step_source;
    if (sources.size() == 1) {
      step_source = std::move(sources.front());
    } else {
      auto staged = en::StagedStepSource::Create(std::move(sources));
      if (!staged) return std::unexpected(staged.error());
      step_source = std::move(*staged);
    }
    auto created = en::ModelRunner::Create(node, std::move(config), std::move(step_source), 0, 0);
    if (!created) return std::unexpected(created.error());
    runner = std::move(*created);
    entered.push_back(runner.get());
    entered.push_back(source.get());  // released after the runner
    if (auto x = runner->Setup(); !x) return x;
    if (auto x =
            node.MapWorkspace(runner->activations_needed(),
                              std::max<std::uint64_t>(runner->pool_needed(), en::kPagedExtent));
        !x)
      return x;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes() + (16ULL << 20U));
    std::uint64_t extents = tw->extents().size();
    if (aw != nullptr) extents += aw->extents().size();
    if (auto x = node.Start(
            llmp::base::Bytes(fixed + extents * en::kPagedExtent + 4 * node.StateCapacity()));
        !x)
      return x;
    if (auto x = runner->Register(); !x) return x;
    if (auto x = runner->Bind(); !x) return x;
    node.Run();
    std::vector<std::uint32_t> slots;
    for (std::uint32_t i = 0; i < c.owners; ++i) slots.push_back(i);
    if (auto x = runner->slots().SelectSlots(slots); !x) return x;
    return node.WithRequest(0, runner->slots().execution(), "lowered steps", [&]() -> en::Status {
      std::size_t pass = 0;
      std::vector<std::vector<std::byte>> last_draft;  // the first pass's rows
      for (std::size_t w = 0; w < script.size(); ++w) {
        const Planned& planned = script[w];
        auto& row = out.outputs.emplace_back();
        const pl::StepShape shape = ShapeOf(planned, c.context);
        std::vector<en::SegmentWork> work;
        std::vector<std::vector<std::int32_t>> tokens(planned.tokens);
        std::vector<en::FeatureFrom> features(planned.wave.units.size());
        std::vector<en::StateBorrow> borrows;
        const bool draft = planned.wave.head == Head::kDraft;
        if (draft) {
          // The first pass's anchor is the top token of the prompt's
          // frontier (the last wave before the drafts), the second's the
          // first pass's.
          const auto& from = pass == 0 ? out.outputs[w - 1] : last_draft;
          if (from.size() != planned.wave.units.size())
            return en::support::Error("a draft without the inputs it follows");
          for (std::size_t i = 0; i < planned.wave.units.size(); ++i) {
            tokens[i] = {Top(from[i])};
            // The first pass reads the target's tapped feature of the
            // endpoint, a later one its own last pass's.
            features[i] = pass == 0 ? en::FeatureFrom{.kind = en::FeatureFrom::Kind::kTap,
                                                      .position = planned.past[i] - 1}
                                    : en::FeatureFrom{.kind = en::FeatureFrom::Kind::kCarry};
            auto borrow = runner->slots().Borrow(planned.wave.units[i].slot, 0);
            if (!borrow) return en::support::Error(borrow.error());
            borrows.push_back(std::move(*borrow));
          }
        }
        for (std::size_t i = 0; i < tokens.size(); ++i) {
          work.push_back({.tokens = tokens[i], .bytes = {}, .feature = features[i]});
        }
        // A verify leaves its rows to their Accept (all kept here).
        const bool verify = planned.wave.head == Head::kAll;
        auto ran = runner->RunStep(
            {.shape = shape, .work = work, .advance = !draft && !verify, .ahead = {}});
        if (!ran) return std::unexpected(ran.error().what);
        if (verify) {
          for (const Unit& u : planned.wave.units) {
            if (auto x = runner->slots().Accept(u.slot, u.rows); !x) return x;
          }
        }
        if (ran->path == en::RunPath::kReplayed) ++out.replayed;
        pass += draft ? 1 : 0;
        if (planned.wave.head == Head::kNone) continue;
        for (const auto& bytes : ran->outputs) row.push_back(bytes);
        if (draft && pass == 1) last_draft = row;
      }
      return {};
    });
  }();
  EXPECT_TRUE(started) << (started ? "" : started.error());
  if (runner) {
    EXPECT_EQ(runner->coverage().violations, 0U) << runner->coverage().first_violation;
  }
  const auto retired = node.TearDown(entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) {
    std::ignore = runner.release();  // its memory may still be in use
    std::ignore = source.release();
    return out;
  }
  runner.reset();
  source.reset();
  return out;
}

// The published v1 artifact the store's v1-ids.json names `name`.
std::optional<fs::path> Installed(std::string_view name) {
  const auto id = llmp::test_support::V1IdNamed(name);
  if (!id) return std::nullopt;
  return fs::path(llmp::test_support::V1Store()) / *id;
}

class ComponentRunnerGemma4Gpu : public testing::TestWithParam<Case> {};

TEST_P(ComponentRunnerGemma4Gpu, ServingRecipeStepsAreFiniteAndRepeatBitForBit) {
  const Case& c = GetParam();
  const auto target = Installed(TargetName(c.size));
  const auto assistant = Installed(AssistantName(c.size));
  if (!target || (c.assistant && !assistant)) {
    GTEST_FAIL() << "no published v1 Gemma 4 artifacts in " << llmp::test_support::V1Store();
  }
  // The window and the width the target's stored spec names.
  auto opened = ar::Artifact::Open(*target, {.expected_id = {}});
  ASSERT_TRUE(opened.has_value()) << opened.error().ToString();
  const auto stored = llmp::spec::StoredSpec(*opened);
  ASSERT_TRUE(stored.has_value()) << stored.error();
  const std::uint32_t window = Window(*stored);
  ASSERT_GT(window, 0U);
  const std::uint64_t width = stored->stages.front().stream.width;
  // NOLINTNEXTLINE(concurrency-mt-unsafe): the test environment is immutable
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch != nullptr ? scratch : testing::TempDir()) / "lowered4";
  fs::create_directories(out);
  // The sliding layers' ring: pad(min(context, window + rows), 256) cells.
  const std::uint32_t local_cells = (std::min(c.context, window + kMaxRows) + 255) / 256 * 256;
  const std::vector<Planned> script = Script(c, local_cells);
  const Component first = Lowered(c, *target, assistant.value_or(fs::path{}), out, script);
  ASSERT_EQ(first.outputs.size(), script.size());
  // The repeated decode shapes replay (a verify's shapes alone, once).
  EXPECT_GE(first.replayed, c.verify ? 1U : 2U);
  // A second runner over the same script gives every output bit for bit
  // (the replays included).
  const Component second = Lowered(c, *target, assistant.value_or(fs::path{}), out, script);
  ASSERT_EQ(second.outputs.size(), script.size());
  EXPECT_EQ(second.replayed, first.replayed);
  std::size_t compared = 0;
  std::size_t drafted = 0;
  for (std::size_t w = 0; w < script.size(); ++w) {
    const bool draft = script[w].wave.head == Head::kDraft;
    ASSERT_EQ(first.outputs[w].size(), second.outputs[w].size()) << "wave " << w;
    ASSERT_EQ(first.outputs[w].size(),
              script[w].wave.head == Head::kNone ? 0U : script[w].wave.units.size())
        << "wave " << w;
    for (std::size_t u = 0; u < first.outputs[w].size(); ++u) {
      const auto& got = first.outputs[w][u];
      const auto& again = second.outputs[w][u];
      ASSERT_EQ(got.size(), again.size()) << "wave " << w << " owner " << u;
      EXPECT_EQ(std::memcmp(got.data(), again.data(), got.size()), 0)
          << "wave " << w << " owner " << u << ": another output";
      if (script[w].wave.head != Head::kGreedy) {
        // A verify's outputs are every row's logits, a draft's its logits
        // and next feature.
        const std::size_t rows =
            script[w].wave.head == Head::kAll ? script[w].wave.units[u].rows : 1U;
        const std::size_t want = draft ? kVocab + width : rows * kVocab;
        ASSERT_EQ(got.size(), want * sizeof(float)) << "wave " << w << " owner " << u;
        bool finite = true;
        for (std::size_t j = 0; j < want && finite; ++j) {
          float value = 0;
          std::memcpy(&value, got.data() + (j * sizeof(float)), sizeof(float));
          finite = std::isfinite(value);
        }
        EXPECT_TRUE(finite) << "wave " << w << " owner " << u;
      }
      drafted += draft ? 1 : 0;
      ++compared;
    }
  }
  if (c.assistant) EXPECT_EQ(drafted, 2U * c.owners);
  std::printf("%s: %zu waves, %zu owner outputs compared, %llu replayed\n", c.name.c_str(),
              script.size(), compared, static_cast<unsigned long long>(first.replayed));
}

INSTANTIATE_TEST_SUITE_P(
    Gemma4, ComponentRunnerGemma4Gpu,
    testing::Values(
        Case{.name = "gemma26_bounded", .size = 26, .assistant = true},
        Case{.name = "gemma31_bounded", .size = 31, .assistant = true},
        // Drafts at a past of 256: their frozen reads end there.
        Case{.name = "gemma31_bounded_256", .size = 31, .assistant = true, .prompt = 256},
        // A verify's four rows an owner, every row's outputs (the 26B-A4B's
        // four-row routing and reduce, the 31B's).
        Case{.name = "gemma26_verify", .size = 26, .verify = true},
        Case{.name = "gemma31_verify", .size = 31, .verify = true},
        Case{.name = "gemma26_envelope", .size = 26, .context = 8192, .owners = 4},
        Case{.name = "gemma31_outside",
             .size = 31,
             .context = 8448,
             .owners = 2,
             .candidate = false},
        Case{.name = "gemma26_outside",
             .size = 26,
             .context = 8448,
             .owners = 2,
             .candidate = false}),
    [](const testing::TestParamInfo<Case>& info) { return info.param.name; });

}  // namespace
