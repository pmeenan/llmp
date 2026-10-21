// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The one model runner (engine/model_runner.h) over the component step
// source (engine/component_source.h) and the GGML module's lowering
// (kernels/ggml/step_program.h), on the published v1 Qwen3.8 artifacts
// (label `models`, one model at a time): the native ModelOpt import and a
// GGUF quantization, each under its approved policy (as serving selects
// it), in its fast form (the default: fused,
// not exact, device masks and tokens). Two owners run one script: prompt
// chunks of the chunk bound, a ragged chunk and short ones (the fast
// kernels' row limits: 8 and 16 rows, the convolution's 32-token windows),
// a state-only chunk, one owner's prompt past the indexer's budget (its
// steps then select their cells), joined one-row decodes returning logits
// and device-chosen tokens (one owner selecting beside one that does not),
// repeated so that they are captured and replayed (D-090). Each owner's
// outputs are checked for finiteness and repeat bit for bit across runs, and
// at the end each owner's state tensor by tensor (each cache through its
// positions, the block keys through its complete blocks, the recurrent and
// convolution state whole).
//
// The n-gram rows each row reads (the step's side rows, components/
// component.h side_rows) are the test's: the hash of the token history
// (tests/support/families/qwen.h Qwen38PleRows, with the hash constants
// the stored spec's side.ngram component carries), read on demand into
// bounded row slots. (The v0 GGML-experts ModelOpt import, 67617f87, has no
// v1 counterpart but the native one.)
// Speculation (a verify's saves and its commit) runs through the engine in
// G3c; its steps' lowering is validated on the CPU (ggml_lower_qwen38_test).

#include <gtest/gtest.h>

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
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/decoding.h"
#include "engine/execution_contract.h"
#include "engine/expert_slab.h"
#include "engine/live_state.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "families/qwen.h"
#include "kernels/ggml/qwen38_commit.h"
#include "kernels/ggml/step_program.h"
#include "model/state_contract.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"
#include "runtime/component_admission.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen38.h"
#include "spec/spec.h"
#include "spec/stored.h"
#include "v1_frozen.h"

namespace {

namespace co = llmp::components;
namespace en = llmp::engine;
namespace ar = llmp::artifact;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace fm = llmp::test_support::families;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace rt = llmp::runtime;
namespace sp = llmp::spec;
namespace fs = std::filesystem;

constexpr std::uint32_t kContext = 8192;
constexpr std::uint32_t kMaxRows = 512;
constexpr std::uint32_t kOwners = 2;
// The slabs' offset in their first page.
constexpr std::uint64_t kSlabAlignment = 16;

struct Case {
  std::string name;
  std::string_view v1_name;  // the store's name (v1_frozen.h V1IdNamed)
};

// ------------------------------------------------------------ the script

enum class Head : std::uint8_t { kLogits, kGreedy, kNone };
struct Unit {
  std::uint32_t slot = 0;
  std::uint32_t rows = 0;
  std::uint32_t keep = 0;  // nonzero: verify, then accept only this prefix
};
struct Wave {
  std::vector<Unit> units;
  Head head = Head::kLogits;
};
struct Planned {
  Wave wave;
  std::vector<std::uint32_t> past;
};

std::int32_t TokenAt(std::uint32_t slot, std::uint32_t position) {
  return static_cast<std::int32_t>(100 + (((position * 7919U) + (slot * 104729U) + 13U) % 150000U));
}

// Each owner's tokens from position 0 through `end`.
std::vector<std::int32_t> History(std::uint32_t slot, std::uint32_t end) {
  std::vector<std::int32_t> out;
  for (std::uint32_t p = 0; p < end; ++p) out.push_back(TokenAt(slot, p));
  return out;
}

std::vector<Planned> Script(bool verify = false) {
  // Each owner's prompt, its short chunks and lone decodes first, then
  // joined decodes.
  std::vector<Wave> waves = {
      {.units = {{0, kMaxRows}}},
      {.units = {{1, 300}}},
      {.units = {{1, 37}}},
      {.units = {{1, 8}}},
      {.units = {{1, 9}}},
      {.units = {{1, 16}}},
      {.units = {{1, 17}}},
      {.units = {{0, 1}}},
      {.units = {{1, 1}}},
      {.units = {{0, 1}, {1, 1}}},
      {.units = {{0, 1}, {1, 1}}, .head = Head::kGreedy},
      {.units = {{0, kMaxRows}}, .head = Head::kNone},
  };
  // Owner 0 past the indexer's budget (2,048 kept cells), a chunk at a time.
  std::uint32_t past0 = 0;
  for (const Wave& w : waves) {
    for (const Unit& u : w.units) past0 += u.slot == 0 ? u.rows : 0;
  }
  while (past0 + kMaxRows <= 2600) {
    waves.push_back({.units = {{0, kMaxRows}}});
    past0 += kMaxRows;
  }
  waves.push_back({.units = {{0, 100}}});
  // Its selecting decode alone, then beside owner 1's, each head's shape
  // three times: captured beside its second run, replayed on its third.
  waves.push_back({.units = {{0, 1}}});
  for (const Head head : {Head::kLogits, Head::kGreedy}) {
    for (int i = 0; i < 3; ++i) waves.push_back({.units = {{0, 1}, {1, 1}}, .head = head});
  }
  waves.push_back({.units = {{1, 33}}});
  if (verify) {
    std::erase_if(waves, [](const Wave& w) { return w.head == Head::kGreedy; });
    // One owner leaves a commit pending while another verifies. Joined
    // saves must retain each owner's rows even when later waves regroup.
    waves.push_back({.units = {{0, 3, 1}}});
    waves.push_back({.units = {{1, 3, 2}}});
    waves.push_back({.units = {{0, 1}, {1, 1}}});
    for (std::uint32_t keep = 1; keep <= 3; ++keep) {
      for (int repeat = 0; repeat < 3; ++repeat) {
        waves.push_back({.units = {{0, 3, keep}, {1, 3, 4 - keep}}});
      }
    }
    waves.push_back({.units = {{0, 1}, {1, 1}}});
  }
  std::vector<Planned> out;
  std::array<std::uint32_t, kOwners> past{};
  for (const Wave& w : waves) {
    Planned p{.wave = w, .past = {}};
    for (const Unit& u : w.units) {
      p.past.push_back(past.at(u.slot));
      past.at(u.slot) += u.keep == 0 ? u.rows : u.keep;
    }
    out.push_back(std::move(p));
  }
  return out;
}

std::uint32_t ReadOf(std::uint32_t end) { return std::min(kContext, (end + 255) / 256 * 256); }

pl::StepShape ShapeOf(const Planned& p) {
  pl::StepShape shape;
  shape.phase = pl::Phase::kDecode;
  shape.read_align = 256;
  for (std::size_t i = 0; i < p.wave.units.size(); ++i) {
    const Unit& u = p.wave.units[i];
    if (u.rows > 1) shape.phase = pl::Phase::kPrefill;
    if (u.keep != 0) shape.phase = pl::Phase::kVerify;
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(u.slot),
                              .rows = u.rows,
                              .past = p.past[i],
                              .read_cells = ReadOf(p.past[i] + u.rows),
                              .outputs = p.wave.head == Head::kNone ? 0U
                                         : u.keep != 0              ? u.rows
                                                                    : 1U,
                              .group = 0,
                              .want_head = p.wave.head != Head::kNone,
                              .greedy = p.wave.head == Head::kGreedy});
  }
  return shape;
}

// Each wave's outputs by unit (a logits row's or a token's bytes), and at
// the end each owner's state tensors by name and layer.
using Outputs = std::vector<std::vector<std::vector<std::byte>>>;
using States = std::vector<std::map<std::string, std::vector<std::byte>>>;

template <typename T>
std::vector<std::byte> Bytes(std::span<const T> values) {
  const auto bytes = std::as_bytes(values);
  return {bytes.begin(), bytes.end()};
}

// ------------------------------------------------------------ the artifact

struct Target {
  fs::path root;
  ar::Artifact artifact;
  fm::Qwen38PleHash hash;
  std::uint32_t table = 0;  // the n-gram table's resource
  sp::Source source;
  sp::ModelSpec spec;
  std::optional<pl::Policy> policy;  // the approved one, as serving selects it
};

std::vector<std::uint64_t> Unsigned(const sp::ComponentRef& c, std::string_view name) {
  const sp::Param* param = c.Find(name);
  EXPECT_NE(param, nullptr) << name;
  if (param == nullptr) return {};
  const auto* list = std::get_if<std::vector<std::int64_t>>(&param->value);
  EXPECT_NE(list, nullptr) << name;
  if (list == nullptr) return {};
  return {list->begin(), list->end()};
}

// The v1 artifact the store names `name`, its stored spec and source as the
// runtime reads them (spec/stored.h), its n-gram hash as its side.ngram
// component carries it, and its approved policy; nullopt when absent.
std::optional<Target> Open(std::string_view name) {
  const auto id = llmp::test_support::V1IdNamed(name);
  if (!id) return std::nullopt;
  const fs::path root = fs::path(llmp::test_support::V1Store()) / *id;
  auto artifact = ar::Artifact::Open(root, {.expected_id = *id});
  EXPECT_TRUE(artifact.has_value());
  if (!artifact) return std::nullopt;
  auto spec = sp::StoredSpec(*artifact);
  EXPECT_TRUE(spec.has_value()) << (spec ? "" : spec.error());
  auto source = sp::SourceOf(*artifact);
  EXPECT_TRUE(source.has_value()) << (source ? "" : source.error());
  if (!source || !spec) return std::nullopt;
  const auto side = std::ranges::find(spec->stages.front().side, std::string("side.ngram"),
                                      [](const sp::SideInput& in) { return in.source.kind; });
  EXPECT_NE(side, spec->stages.front().side.end());
  if (side == spec->stages.front().side.end()) return std::nullopt;
  fm::Qwen38PleHash hash{.multipliers = Unsigned(side->source, "multipliers"),
                         .offsets = Unsigned(side->source, "offsets"),
                         .vocab = Unsigned(side->source, "vocab")};
  const auto& p = fm::Qwen38Flash();
  EXPECT_EQ(hash.multipliers.size(), p.ngram);
  EXPECT_EQ(hash.offsets.size(), p.ple_heads());
  EXPECT_EQ(hash.vocab.size(), p.ple_heads());
  std::optional<std::uint32_t> table;
  for (const sp::WeightBinding& w : side->source.weights) {
    if (w.role == "table" && w.part == llmp::format::PartRole::kData)
      table = artifact->FindResource(w.tensor);
  }
  EXPECT_TRUE(table.has_value()) << "no n-gram table resource";
  if (!table) return std::nullopt;
  auto rule = rt::ComponentAdmissionPolicy(*spec, *source, artifact->sources(),
                                           {.context = kContext, .slots = kOwners});
  EXPECT_TRUE(rule.has_value()) << (rule ? "" : rule.error());
  if (!rule) return std::nullopt;
  std::optional<pl::Policy> policy;
  if (!rule->text.empty()) {
    auto parsed = pl::ParsePolicy(rule->text);
    EXPECT_TRUE(parsed.has_value()) << (parsed ? "" : parsed.error());
    if (!parsed) return std::nullopt;
    policy = std::move(*parsed);
  }
  return Target{.root = root,
                .artifact = std::move(*artifact),
                .hash = std::move(hash),
                .table = *table,
                .source = std::move(*source),
                .spec = std::move(*spec),
                .policy = std::move(policy)};
}

// The bytes of a tensor the owner's positions have written: a cache's
// cells through them, the block keys' complete blocks, the rest whole.
std::uint64_t UsedBytes(std::string_view name, std::uint64_t cell_bytes, std::uint64_t whole,
                        std::uint32_t positions) {
  const auto& p = fm::Qwen38Flash();
  if (name == "k" || name == "v" || name == "indexer_k") return positions * cell_bytes;
  if (name == "indexer_blocks") return (positions / p.indexer_ratio) * cell_bytes;
  return whole;
}

// ------------------------------------------------------------ the components

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

// The model's weight places (engine/expert_slab.h ExpertSlabsOf, as
// serving places them): every dense group on the device except the
// row-paged n-gram table, each layer's expert arrays in one slab.
std::expected<en::LayerSlabs, std::string> PlacesOf(const Target& t) {
  auto out = en::ExpertSlabsOf(t.artifact, kSlabAlignment);
  if (!out) return out;
  out->places.at(t.artifact.resources()[t.table].group) = en::GroupPlace::kNone;
  return out;
}

// The n-gram table's rows a step's rows read, in order.
std::vector<std::int32_t> SideRows(const Target& t, const Planned& p) {
  const auto& prof = fm::Qwen38Flash();
  std::vector<std::int32_t> out;
  for (std::size_t i = 0; i < p.wave.units.size(); ++i) {
    const Unit& u = p.wave.units[i];
    const std::vector<std::int32_t> history = History(u.slot, p.past[i] + u.rows);
    for (std::uint32_t r = 0; r < u.rows; ++r) {
      const auto rows = fm::Qwen38PleRows(prof, t.hash, history, p.past[i] + r);
      out.insert(out.end(), rows.begin(), rows.end());
    }
  }
  return out;
}

struct Component {
  Outputs outputs;
  States states;
  std::uint64_t replayed = 0;
  std::uint64_t checked_commits = 0;
};

Component Lowered(const Target& t, const fs::path& scratch, std::span<const Planned> script,
                  bool check_commits = false) {
  Component out;
  const auto& mapping = sp::mappings::Qwen38Mapping();
  auto layout = co::StageLayout(t.spec, 0, kContext, kMaxRows);
  EXPECT_TRUE(layout.has_value()) << layout.error();
  if (!layout) return out;
  // The independent legacy commit consumes the actual verify saves on a
  // private copy of the unchanged fixed state. Its recurrence is separately
  // proven bitexact to the verifier by Qwen38CommitTest. Plain target rows
  // are not this oracle: Qwen3.8's fast verify is not row-invariant.
  struct Fixed {
    const md::StateLayoutDerived::Item* item;
    std::uint64_t offset;
  };
  std::vector<Fixed> fixed;
  std::vector<en::LiveState::Range> fixed_ranges;
  std::uint64_t fixed_bytes = 0;
  for (const auto& item : layout->items) {
    if (item.contract.rollback != md::Rollback::kCommit) continue;
    fixed.push_back({.item = &item, .offset = fixed_bytes});
    fixed_ranges.push_back({.region = 0, .offset = item.offset, .bytes = item.bytes});
    fixed_bytes += item.bytes;
  }
  std::map<std::tuple<std::uint32_t, std::string, std::uint32_t>, std::uint64_t> saves;
  std::uint64_t oracle_base = 0;
  std::byte* oracle_host = nullptr;
  kg::GgmlStepLaunch* launch = nullptr;
  std::vector<pl::StepShape> shapes;
  for (const Planned& planned : script) {
    const pl::StepShape shape = ShapeOf(planned);
    if (std::ranges::find(shapes, shape) == shapes.end()) shapes.push_back(shape);
  }
  // Each weight part's place: a resource or an expert array.
  struct Place {
    bool array = false;
    std::uint32_t index = 0;
  };
  std::map<std::pair<llmp::ir::WeightRef, llmp::format::PartRole>, Place> place;
  std::map<llmp::ir::WeightRef, std::uint64_t> pitch_of;
  std::optional<en::RunnerConfig::RowTable> table;
  auto places = PlacesOf(t);
  EXPECT_TRUE(places.has_value()) << (places ? "" : places.error());
  if (!places) return out;
  for (const pl::StepShape& shape : shapes) {
    auto step = co::BuildStep(t.spec, 0, shape, {.tensors = t.source.tensors, .state = &*layout});
    EXPECT_TRUE(step.has_value()) << step.error();
    if (!step) return out;
    for (const co::StepWeight& w : step->weights) {
      for (const auto& [name, role] : w.parts) {
        if (w.ref.expert_array) {
          const auto found = t.artifact.FindExpertArray(name);
          EXPECT_TRUE(found.has_value()) << name;
          if (!found) continue;
          place[{w.ref, role}] = {.array = true, .index = *found};
          pitch_of[w.ref] = places->pitch.at(*found);
          continue;
        }
        const auto found = t.artifact.FindResource(name);
        EXPECT_TRUE(found.has_value()) << name;
        if (!found) continue;
        place[{w.ref, role}] = {.array = false, .index = *found};
        if (*found == t.table) {
          table = {.weight = w.ref,
                   .resource = *found,
                   .format = step->graph.tensor(w.tensor).format,
                   .part = role,
                   .rows_per_token = fm::Qwen38Flash().ple_heads()};
        }
      }
    }
  }
  std::vector<std::vector<std::int32_t>> side;
  for (const Planned& planned : script) side.push_back(SideRows(t, planned));

  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  auto source = std::make_unique<KernelSource>(node);
  std::unique_ptr<en::ModelRunner> runner;
  std::vector<en::PagedModel*> entered;
  const auto started = [&]() -> en::Status {
    if (auto x = node.Open(); !x) return x;
    if (auto x = source->resources->OpenLibrary("lowered cuBLAS workspace"); !x) return x;
    auto made = kg::GgmlBackend::Create(
        0, node.execution(), node.stream(0), &source->resources->library(),
        {.lower = {.expert_stride = [&pitch_of](const llmp::ir::WeightRef& ref) -> std::uint64_t {
           const auto at = pitch_of.find(ref);
           return at == pitch_of.end() ? 0 : at->second;
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
    auto weights = std::make_unique<en::PagedWeights>();
    if (auto x = weights->Open(t.root); !x) return x;
    en::PagedWeights* tw = weights.get();
    std::vector<std::unique_ptr<en::ComponentStepSource>> sources;
    auto backend = source->backend->Backend();
    const auto lower = backend.lower;
    backend.lower = [&](const kn::LowerInput& input) {
      if (!input.addresses.measuring() && input.token->shape.phase == pl::Phase::kVerify) {
        for (const auto& save : input.token->step.saves) {
          for (std::size_t segment = 0; segment < input.token->shape.segments.size(); ++segment) {
            saves[{save.layer, save.what, input.token->shape.segments[segment].slot}] =
                input.addresses.save(save.layer, save.component, save.what,
                                     static_cast<std::uint8_t>(segment));
          }
        }
      }
      return lower(input);
    };
    auto stage0 = en::ComponentStepSource::Create(
        {.spec = &t.spec,
         .stage = 0,
         .bindings = {.tensors = t.source.tensors, .state = &*layout},
         .bound_digest = t.spec.Digest(),
         .mapping = &mapping,
         .values = {},
         .slots = kOwners,
         .policy = t.policy,
         .backend = std::move(backend)});
    if (!stage0) return std::unexpected(stage0.error());
    en::RunnerConfig config;
    config.slots = {.model = "lowered",
                    .slots = kOwners,
                    .stages = {{.name = "target", .layout = *layout}},
                    .spec = t.spec.Digest(),
                    .read_align = 256,
                    .spill_place = {},
                    .spill_directory = scratch,
                    .invalidate = {},
                    .drop_plans = {}};
    config.split_expert_parts = true;
    config.weights.push_back(
        {.weights = std::move(weights), .places = places->places, .slabs = places->slabs});
    config.setup_shapes = shapes;
    config.max_verify_rows = 3;
    if (!table) return en::support::Error("the n-gram table was not bound");
    config.row_tables.push_back(*table);
    // Prompt chunks captured on their second run, as decodes (D-090).
    config.capture_phases = 0x1F;
    if (check_commits) {
      config.kernel_memory.push_back({.name = "commit oracle", .bytes = fixed_bytes});
      config.prepare = [&](std::span<const std::uint64_t> bases) -> en::Status {
        oracle_base = bases.front();
        return {};
      };
      auto host = source->resources->Pinned(2 * fixed_bytes);
      if (!host) return std::unexpected(host.error());
      oracle_host = static_cast<std::byte*>(*host);
    }
    config.launch =
        [&](std::uint64_t) -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      auto made_launch = source->backend->Launch(
          {.base = node.pool().base, .size = llmp::base::Bytes(node.pool().bytes)},
          {.base = source->resources->selected_workspace().base,
           .bytes = source->resources->selected_workspace().bytes});
      if (made_launch) launch = static_cast<kg::GgmlStepLaunch*>(made_launch->get());
      return made_launch;
    };
    config.weight_address = [&place, tw](const llmp::ir::WeightRef& ref,
                                         llmp::format::PartRole role) -> std::uint64_t {
      const auto at = place.find({ref, role});
      if (at == place.end()) return 0;
      return at->second.array ? tw->array_address(at->second.index)
                              : tw->resource_address(at->second.index);
    };
    auto created = en::ModelRunner::Create(node, std::move(config), std::move(*stage0), 0, 0);
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
    if (auto x = node.Start(llmp::base::Bytes(std::uint64_t{110} << 30U)); !x) return x;
    if (auto x = runner->Register(); !x) return x;
    if (auto x = runner->Bind(); !x) return x;
    node.Run();
    std::vector<en::LoadStats> loads;
    if (auto x = node.Load(runner->weights(), "lowered weights", loads); !x) return x;
    std::vector<std::uint32_t> slots;
    for (std::uint32_t i = 0; i < kOwners; ++i) slots.push_back(i);
    if (auto x = runner->slots().SelectSlots(slots); !x) return x;
    return node.WithRequest(0, runner->slots().execution(), "lowered steps", [&]() -> en::Status {
      std::array<std::uint32_t, kOwners> past{};
      for (std::size_t w = 0; w < script.size(); ++w) {
        const Planned& planned = script[w];
        auto& row = out.outputs.emplace_back();
        const pl::StepShape shape = ShapeOf(planned);
        std::vector<std::vector<std::int32_t>> tokens, histories;
        for (std::size_t i = 0; i < planned.wave.units.size(); ++i) {
          const Unit& u = planned.wave.units[i];
          const auto& history = histories.emplace_back(History(u.slot, planned.past[i] + u.rows));
          tokens.emplace_back(history.end() - u.rows, history.end());
          past.at(u.slot) = planned.past[i] + (u.keep == 0 ? u.rows : u.keep);
        }
        std::vector<en::SegmentWork> work;
        for (const auto& ids : tokens) work.push_back({.tokens = ids, .bytes = {}});
        if (w == 0) {
          // Missing or partial side-table indices must refuse before any
          // model work. Zero-filling them would silently read wrong rows.
          const auto initialized = runner->slots().live(0, 0).used_bytes();
          const auto positions = runner->slots().positions(0);
          for (const std::size_t count : {std::size_t{0}, side[w].size() - 1}) {
            auto refused = runner->RunStep({.shape = shape,
                                            .work = work,
                                            .side_rows = std::span(side[w]).first(count),
                                            .ahead = {}});
            if (refused || !refused.error().clean ||
                runner->slots().live(0, 0).used_bytes() != initialized ||
                runner->slots().positions(0) != positions) {
              return en::support::Error(
                  std::format("incomplete side rows were not cleanly refused: {}",
                              refused ? "accepted" : refused.error().what));
            }
          }
          auto invalid_rows = side[w];
          invalid_rows.back() = -1;
          const auto invalid = runner->RunStep(
              {.shape = shape, .work = work, .side_rows = invalid_rows, .ahead = {}});
          if (invalid || !invalid.error().clean ||
              runner->slots().live(0, 0).used_bytes() != initialized ||
              runner->slots().positions(0) != positions)
            return en::support::Error("an invalid side row changed initialized state");
        }
        for (std::size_t i = 0; i < work.size(); ++i) work[i].history = histories[i];
        // Alternate caller-supplied indices with the component callback.
        // Both must preserve the same exact outputs and state on every form.
        auto ran = runner->RunStep({.shape = shape,
                                    .work = work,
                                    .step_bytes = {},
                                    .side_rows = w % 2 == 0 ? std::span<const std::int32_t>(side[w])
                                                            : std::span<const std::int32_t>{},
                                    .advance = shape.phase != pl::Phase::kVerify,
                                    .ahead = {}});
        if (!ran) return std::unexpected(std::format("wave {}: {}", w, ran.error().what));
        for (const Unit& u : planned.wave.units) {
          if (u.keep != 0) {
            if (check_commits) {
              const auto ptr = [](std::uint64_t address) {
                return static_cast<float*>(en::support::Pointer(address));
              };
              const auto state = [&](std::uint32_t layer, std::string_view name) {
                for (const Fixed& f : fixed)
                  if (f.item->layer == layer && f.item->contract.name == name)
                    return ptr(oracle_base + f.offset);
                return static_cast<float*>(nullptr);
              };
              const auto saved = [&](std::uint32_t layer, std::string_view name) {
                const auto at = saves.find({layer, std::string(name), u.slot});
                return at == saves.end() ? nullptr : ptr(at->second);
              };
              const auto& profile = fm::Qwen38Flash();
              kg::Qwen38CommitArgs args;
              args.keep = static_cast<int>(u.keep);
              args.channels = static_cast<int>(profile.conv_channels());
              args.qk_heads = static_cast<int>(profile.lin_k_heads);
              args.v_heads = static_cast<int>(profile.lin_v_heads);
              args.taps = static_cast<int>(profile.conv - 1);
              for (const Fixed& f : fixed) {
                if (f.item->contract.name != "recurrent") continue;
                const auto layer = f.item->layer;
                args.layer[args.layers++] = {.state = state(layer, "recurrent"),
                                             .history = state(layer, "conv"),
                                             .conv = saved(layer, "conv"),
                                             .qkv = saved(layer, "qkv"),
                                             .gate = saved(layer, "gate"),
                                             .beta = saved(layer, "beta")};
              }
              args.ple_history = state(profile.ple_layer, "ple_conv");
              args.ple_rows = saved(profile.ple_layer, "ple");
              args.ple_width = static_cast<int>(profile.hc_width());
              args.ple_taps = static_cast<int>(profile.ple_history());
              const auto oracle = node.Job(
                  runner->slots().execution(),
                  [&](llmp::providers::NativeStream native) {
                    namespace pr = llmp::providers;
                    namespace sc = llmp::scheduler;
                    for (const Fixed& f : fixed) {
                      const auto copied = pr::CopyAsync(
                          native, ptr(oracle_base + f.offset),
                          ptr(runner->slots().live(u.slot, 0).base(0) + f.item->offset),
                          f.item->bytes, pr::CopyKind::kDeviceToDevice);
                      if (!copied.ok()) return sc::JobResult::kUnknown;
                    }
                    const auto committed = kg::Qwen38Commit(launch->context(), args);
                    if (!committed) return sc::JobResult::kUnknown;
                    return sc::JobResult::kQueued;
                  },
                  "legacy commit oracle", 0);
              if (!oracle) return oracle;
            }
            if (auto x = runner->slots().Accept(u.slot, u.keep); !x) return x;
            if (check_commits) {
              if (auto x = runner->slots().Settle(u.slot); !x) return x;
              if (auto x = node.Copy(0, runner->slots().execution(), oracle_base, oracle_host,
                                     fixed_bytes, true, "read commit oracle");
                  !x)
                return x;
              if (auto x = runner->slots().CopyState(u.slot, oracle_host + fixed_bytes,
                                                     fixed_ranges, true);
                  !x)
                return x;
              for (const Fixed& f : fixed) {
                EXPECT_EQ(std::memcmp(oracle_host + f.offset, oracle_host + fixed_bytes + f.offset,
                                      f.item->bytes),
                          0)
                    << "wave " << w << " slot " << u.slot << " keep " << u.keep << " "
                    << f.item->contract.name << "." << f.item->layer;
              }
              ++out.checked_commits;
            }
          }
        }
        if (ran->path == en::RunPath::kReplayed) ++out.replayed;
        if (planned.wave.head == Head::kNone) continue;
        for (const auto& bytes : ran->outputs) row.emplace_back(bytes.begin(), bytes.end());
      }
      // Each owner's state, by tensor.
      for (std::uint32_t s = 0; s < kOwners; ++s) {
        std::vector<en::LiveState::Range> ranges;
        std::vector<std::pair<std::string, std::uint64_t>> names;
        std::uint64_t total = 0;
        for (const auto& item : layout->items) {
          const std::string name = std::format("{}.{}", item.contract.name, item.layer);
          const std::uint64_t used =
              UsedBytes(item.contract.name, item.contract.cell_bytes, item.bytes, past.at(s));
          ranges.push_back({.region = 0, .offset = item.offset, .bytes = used});
          names.emplace_back(name, used);
          total += used;
        }
        std::vector<llmp::catalog::ExtentId> host_staging;
        auto host = node.Pinned(std::max<std::uint64_t>(total, 4096), 0, host_staging);
        if (!host) return std::unexpected(host.error());
        if (auto x = runner->slots().CopyState(s, *host, ranges, true); !x) return x;
        auto& state = out.states.emplace_back();
        const auto* bytes = static_cast<const std::byte*>(*host);
        for (const auto& [name, used] : names) {
          state[name] = {bytes, bytes + used};
          bytes += used;
        }
        if (auto x = node.FreePinned(*host); !x) return x;
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

void CompareStates(const States& got, const States& want) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t owner = 0; owner < want.size(); ++owner) {
    ASSERT_EQ(got[owner].size(), want[owner].size());
    for (const auto& [name, bytes] : want[owner]) {
      const auto found = got[owner].find(name);
      ASSERT_NE(found, got[owner].end());
      ASSERT_EQ(found->second.size(), bytes.size());
      const auto mismatch = std::memcmp(found->second.data(), bytes.data(), bytes.size());
      EXPECT_EQ(mismatch, 0) << "owner " << owner << " state " << name;
    }
  }
}

class ComponentRunnerQwen38Gpu : public testing::TestWithParam<Case> {};

TEST_P(ComponentRunnerQwen38Gpu, FastFormStepsAreFiniteAndRepeatBitForBit) {
  const Case& c = GetParam();
  if (!llmp::test_support::V1IdNamed(c.v1_name))
    GTEST_FAIL() << "no Qwen3.8 v1 artifact " << c.v1_name << " in "
                 << llmp::test_support::V1Store();
  const auto target = Open(c.v1_name);
  ASSERT_TRUE(target.has_value());
  // NOLINTNEXTLINE(concurrency-mt-unsafe): the test environment is immutable
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch != nullptr ? scratch : testing::TempDir()) / "lowered38";
  fs::create_directories(out);
  const std::vector<Planned> script = Script();
  const Component first = Lowered(*target, out, script);
  ASSERT_EQ(first.outputs.size(), script.size());
  const Component second = Lowered(*target, out, script);
  ASSERT_EQ(second.outputs.size(), script.size());
  EXPECT_GE(first.replayed, 2U);
  EXPECT_GE(second.replayed, 2U);
  std::size_t compared = 0;
  for (std::size_t w = 0; w < script.size(); ++w) {
    ASSERT_EQ(first.outputs[w].size(), second.outputs[w].size()) << "wave " << w;
    for (std::size_t u = 0; u < first.outputs[w].size(); ++u) {
      const auto& a = first.outputs[w][u];
      const auto& b = second.outputs[w][u];
      ASSERT_EQ(a.size(), b.size()) << "wave " << w << " owner " << u;
      EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size()), 0)
          << "wave " << w << " owner " << u << ": output mismatch";
      if (script[w].wave.head == Head::kLogits) {
        for (std::size_t j = 0; j < a.size() / sizeof(float); ++j) {
          float v = 0;
          std::memcpy(&v, a.data() + (j * sizeof(float)), sizeof(float));
          EXPECT_TRUE(std::isfinite(v))
              << "wave " << w << " owner " << u << " logit " << j << " is not finite";
        }
      }
      ++compared;
    }
  }
  CompareStates(second.states, first.states);
  std::printf("%s: %zu waves, %zu owner outputs and states compared, %llu replayed\n",
              c.name.c_str(), script.size(), compared,
              static_cast<unsigned long long>(first.replayed));
}

// Every format's saved inputs drive an independent legacy commit oracle.
// This is exact against the verify's recurrence, without pretending the
// fast verify has the arithmetic of a shorter plain target step.
TEST_P(ComponentRunnerQwen38Gpu, VerifiedPrefixesMatchTheSavedRecurrence) {
  if (!llmp::test_support::V1IdNamed(GetParam().v1_name))
    GTEST_FAIL() << "no Qwen3.8 v1 artifact " << GetParam().v1_name << " in "
                 << llmp::test_support::V1Store();
  const auto target = Open(GetParam().v1_name);
  ASSERT_TRUE(target);
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const fs::path out = fs::path(scratch != nullptr ? scratch : testing::TempDir()) / "verified38";
  fs::create_directories(out);
  const auto script = Script(true);
  const auto lowered = Lowered(*target, out, script, true);
  ASSERT_EQ(lowered.outputs.size(), script.size());
  EXPECT_EQ(lowered.checked_commits, 20U);
  EXPECT_GE(lowered.replayed, 2U);
}

INSTANTIATE_TEST_SUITE_P(Qwen38, ComponentRunnerQwen38Gpu,
                         testing::Values(Case{.name = "modelopt", .v1_name = "qwen38-modelopt"},
                                         Case{.name = "gguf", .v1_name = "qwen38-gguf"}),
                         [](const testing::TestParamInfo<Case>& info) { return info.param.name; });

}  // namespace
