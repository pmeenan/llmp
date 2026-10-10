// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The one model runner (engine/model_runner.h) over the component step
// source (engine/component_source.h) and the GGML module's lowering
// (kernels/ggml/step_program.h), on the real Qwen3.8 artifacts (label
// `models`, one model at a time): the ModelOpt import with its experts in
// the CUTLASS layout and in GGML's NVFP4 blocks, and a GGUF quantization,
// each under its mapping's default policy, against today's runner
// (engine/qwen38_runner.h) in its fast form (the default: fused, not
// exact, device masks and tokens). Two owners run one script: prompt
// chunks of the chunk bound, a ragged chunk and short ones (the fast
// kernels' row limits: 8 and 16 rows, the convolution's 32-token windows),
// a state-only chunk, one owner's prompt past the indexer's budget (its
// steps then select their cells), joined one-row decodes returning logits
// and device-chosen tokens (one owner selecting beside one that does not),
// repeated so that they are captured and replayed (D-090). Each owner's
// outputs are compared with today's bit for bit, and at the end each
// owner's state tensor by tensor (each cache through its positions, the
// block keys through its complete blocks, the recurrent and convolution
// state whole): the two layouts place the tensors differently.
//
// The n-gram rows each row reads (the step's side rows, components/
// component.h side_rows) are the test's: the hash of the token history
// (model/qwen38.h Qwen38PleRows) into the whole table, resident (today's
// runner reads the rows it needs on demand into row slots; the engine's
// binding of side rows is slice G3c's). Speculation (a verify's
// saves and its commit, the MTP drafter's passes and their features) runs
// through the engine in G3c; its steps' lowering is compared with today's
// builder on the CPU (ggml_lower_qwen38_test).

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
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/expert_slab.h"
#include "engine/live_state.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/qwen38_runner.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "kernels/ggml/step_program.h"
#include "model/qwen38.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen38.h"
#include "spec/spec.h"
#include "spec/stored.h"
#include "tokenizer_fixtures.h"

namespace {

namespace co = llmp::components;
namespace en = llmp::engine;
namespace ar = llmp::artifact;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace sp = llmp::spec;
namespace fs = std::filesystem;

constexpr std::uint32_t kContext = 8192;
constexpr std::uint32_t kMaxRows = 512;
constexpr std::uint32_t kOwners = 2;
// The slabs' offset in their first page (qwen38_runner.cc kSlabAlignment).
constexpr std::uint64_t kSlabAlignment = 16;

struct Case {
  std::string name;
  std::string_view id;
};

// ------------------------------------------------------------ the script

enum class Head : std::uint8_t { kLogits, kGreedy, kNone };
struct Unit {
  std::uint32_t slot = 0;
  std::uint32_t rows = 0;
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

std::vector<Planned> Script() {
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
  std::vector<Planned> out;
  std::array<std::uint32_t, kOwners> past{};
  for (const Wave& w : waves) {
    Planned p{.wave = w, .past = {}};
    for (const Unit& u : w.units) {
      p.past.push_back(past.at(u.slot));
      past.at(u.slot) += u.rows;
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
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(u.slot),
                              .rows = u.rows,
                              .past = p.past[i],
                              .read_cells = ReadOf(p.past[i] + u.rows),
                              .outputs = p.wave.head == Head::kNone ? 0U : 1U,
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

std::optional<fs::path> Installed(std::string_view id) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / id;
  if (std::error_code error; !fs::exists(root, error)) return std::nullopt;
  return root;
}

// A resource's bytes (from `offset` within it), read from its shard.
std::vector<std::byte> ReadResource(const fs::path& root, const ar::Artifact& a,
                                    std::uint32_t index, std::uint64_t offset,
                                    std::uint64_t bytes) {
  const auto placement = a.ResourcePlacement(index);
  EXPECT_TRUE(placement.has_value());
  if (!placement) return {};
  const ar::Group& g = a.groups()[placement->group];
  const ar::Shard& shard = a.shards()[g.shard];
  std::ifstream in(root / shard.path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(shard.data_offset.value() + g.offset.value() +
                                       placement->offset.value() + offset));
  std::vector<std::byte> out(bytes);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
  EXPECT_TRUE(in.good());
  return out;
}

std::vector<std::int64_t> ReadI64(const fs::path& root, const ar::Artifact& a,
                                  const md::Qwen38Tensor& t) {
  const auto bytes = ReadResource(root, a, t.index, 0, t.ne.at(0) * sizeof(std::int64_t));
  std::vector<std::int64_t> out(t.ne.at(0));
  std::memcpy(out.data(), bytes.data(), std::min(bytes.size(), out.size() * sizeof(out[0])));
  return out;
}

struct Target {
  fs::path root;
  ar::Artifact artifact;
  md::Qwen38Binding binding;
  md::Qwen38PleHash hash;
  std::vector<sp::SourceValue> values;
  sp::Source source;
  sp::ModelSpec spec;
};

// The artifact's spec as the runtime reads it (spec/stored.h), its hash as
// today's runner reads it.
std::optional<Target> Open(const fs::path& root) {
  auto artifact = ar::Artifact::Open(root, {.expected_id = {}});
  EXPECT_TRUE(artifact.has_value());
  if (!artifact) return std::nullopt;
  const auto& p = md::Qwen38Flash();
  auto binding = md::BindQwen38(p, *artifact);
  EXPECT_TRUE(binding.has_value()) << (binding ? "" : binding.error());
  if (!binding) return std::nullopt;
  const std::uint64_t rows = binding->ple_table.ne.back();
  std::expected<md::Qwen38PleHash, std::string> hash;
  if (binding->gguf()) {
    hash = md::ReadQwen38GgufHash(p, *artifact, rows);
  } else {
    const md::Qwen38Layer& l = binding->layers.at(p.ple_layer);
    hash = md::CheckQwen38PleHash(p, ReadI64(root, *artifact, l.ple_multipliers),
                                  ReadI64(root, *artifact, l.ple_head_offsets),
                                  ReadI64(root, *artifact, l.ple_head_vocab), rows);
  }
  EXPECT_TRUE(hash.has_value()) << (hash ? "" : hash.error());
  if (!hash) return std::nullopt;
  auto values = sp::mappings::Qwen38V0Values(p, *binding, *hash);
  EXPECT_TRUE(values.has_value()) << (values ? "" : values.error());
  if (!values) return std::nullopt;
  const auto& mapping = sp::mappings::Qwen38Mapping();
  auto source = sp::SourceOf(*artifact, &mapping);
  EXPECT_TRUE(source.has_value()) << (source ? "" : source.error());
  auto spec = sp::SynthesizeV0(*artifact, mapping, *values);
  EXPECT_TRUE(spec.has_value()) << (spec ? "" : spec.error());
  if (!source || !spec) return std::nullopt;
  return Target{.root = root,
                .artifact = std::move(*artifact),
                .binding = std::move(*binding),
                .hash = std::move(*hash),
                .values = std::move(*values),
                .source = std::move(*source),
                .spec = std::move(*spec)};
}

// Today's state tensor names (the components' contracts').
std::string KindName(md::Qwen38StateTensor::Kind kind) {
  using K = md::Qwen38StateTensor::Kind;
  switch (kind) {
    case K::kK:
      return "k";
    case K::kV:
      return "v";
    case K::kIndexerK:
      return "indexer_k";
    case K::kIndexerBlocks:
      return "indexer_blocks";
    case K::kConv:
      return "conv";
    case K::kRecurrent:
      return "recurrent";
    case K::kPleConv:
      return "ple_conv";
  }
  return "?";
}

// The bytes of a tensor the owner's positions have written: a cache's
// cells through them, the block keys' complete blocks, the rest whole.
std::uint64_t UsedBytes(std::string_view name, std::uint64_t cell_bytes, std::uint64_t whole,
                        std::uint32_t positions) {
  const auto& p = md::Qwen38Flash();
  if (name == "k" || name == "v" || name == "indexer_k") return positions * cell_bytes;
  if (name == "indexer_blocks") return (positions / p.indexer_ratio) * cell_bytes;
  return whole;
}

// ------------------------------------------------------------ today's runner

struct Production {
  Outputs outputs;
  States states;
};

Production Today(const Target& t, const fs::path& out, std::span<const Planned> script) {
  Production result;
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  const en::Qwen38Options options{.artifact = t.root,
                                  .out = out,
                                  .context = kContext,
                                  .max_rows = kMaxRows,
                                  .drafter = {},
                                  .wave_slots = kOwners,
                                  .request_slots = kOwners,
                                  .wave_read_align = 256,
                                  .spill_place = {}};
  auto runner = std::make_unique<en::Qwen38Runner>(node, options, 0, 0);
  std::vector<en::PagedModel*> entered;
  const auto started = [&]() -> en::Status {
    if (auto x = node.Open(); !x) return x;
    entered.push_back(runner.get());
    if (auto x = runner->Setup(); !x) return x;
    if (auto x = node.MapWorkspace(runner->activations_needed(), runner->pool_needed()); !x)
      return x;
    if (auto x = node.Start(llmp::base::Bytes(std::uint64_t{110} << 30U)); !x) return x;
    if (auto x = runner->Register(); !x) return x;
    if (auto x = runner->Bind(); !x) return x;
    node.Run();
    std::vector<en::LoadStats> loads;
    if (auto x = node.Load(runner->weights(), "today's weights", loads); !x) return x;
    if (auto x = runner->ReadPleHash(); !x) return x;
    std::vector<en::Qwen38Runner::Slot*> slots;
    for (std::uint32_t i = 0; i < kOwners; ++i) {
      auto slot = runner->request_slot(i);
      if (!slot) return std::unexpected(slot.error());
      slots.push_back(*slot);
    }
    if (auto x = runner->SelectSlots(slots); !x) return x;
    return node.WithRequest(0, runner->execution_closure(), "today's script", [&]() -> en::Status {
      std::array<std::uint32_t, kOwners> past{};
      for (const Planned& p : script) {
        auto& row = result.outputs.emplace_back();
        const std::size_t n = p.wave.units.size();
        std::vector<std::vector<std::int32_t>> histories;
        for (std::size_t i = 0; i < n; ++i) {
          const Unit& u = p.wave.units[i];
          histories.push_back(History(u.slot, p.past[i] + u.rows));
        }
        std::vector<std::vector<float>> logits(n);
        std::vector<std::int32_t> tokens(n, -1);
        const bool joined = n > 1;
        if (joined) {
          std::vector<en::Qwen38Runner::ChunkWork> work;
          for (std::size_t i = 0; i < n; ++i) {
            const bool greedy = p.wave.head == Head::kGreedy;
            work.push_back({.slot = slots[p.wave.units[i].slot],
                            .history = histories[i],
                            .n_past = p.past[i],
                            .logits = greedy ? nullptr : &logits[i],
                            .token = greedy ? &tokens[i] : nullptr});
          }
          if (auto x = runner->ChunkWave(work); !x) return x;
        } else {
          en::Qwen38Runner::Slot& slot = *slots[p.wave.units[0].slot];
          if (p.wave.head == Head::kGreedy) {
            if (auto x = slot.GreedyChunk(histories[0], p.past[0], tokens[0]); !x) return x;
          } else if (auto x = slot.PrefillChunk(histories[0], p.past[0], logits[0], false, {},
                                                p.wave.head != Head::kNone);
                     !x) {
            return x;
          }
        }
        for (std::size_t i = 0; i < n; ++i) {
          past.at(p.wave.units[i].slot) = p.past[i] + p.wave.units[i].rows;
          if (p.wave.head == Head::kLogits) {
            row.push_back(Bytes(std::span<const float>(logits[i])));
          } else if (p.wave.head == Head::kGreedy) {
            row.push_back(Bytes(std::span<const std::int32_t>(&tokens[i], 1)));
          }
        }
      }
      // Each owner's state, by tensor.
      for (std::uint32_t s = 0; s < kOwners; ++s) {
        std::vector<std::byte> target, drafter;
        if (auto x = slots[s]->ReadState(target, drafter); !x) return x;
        auto& state = result.states.emplace_back();
        for (const md::Qwen38StateTensor& tensor : runner->state_layout().tensors) {
          const std::string name = std::format("{}.{}", KindName(tensor.kind), tensor.layer);
          const std::uint64_t cell = tensor.ne0 * (tensor.f16 ? 2 : 4);
          const std::uint64_t used = UsedBytes(KindName(tensor.kind), cell, tensor.bytes, past[s]);
          if (tensor.offset + used > target.size()) return en::support::Error("today's state read");
          state[name] = {target.begin() + static_cast<std::ptrdiff_t>(tensor.offset),
                         target.begin() + static_cast<std::ptrdiff_t>(tensor.offset + used)};
        }
      }
      return {};
    });
  }();
  EXPECT_TRUE(started) << (started ? "" : started.error());
  EXPECT_EQ(runner->coverage_violations(), 0U) << runner->first_violation();
  const auto retired = node.TearDown(entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) std::ignore = runner.release();  // its memory may still be in use
  return result;
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

// Today's places (Qwen38Runner::ReserveWeights): every dense group on the
// device (and here the n-gram table's too), each layer's expert arrays in
// one slab.
struct Places {
  std::vector<en::GroupPlace> places;
  std::vector<en::SlabSpec> slabs;
  std::map<std::uint32_t, std::uint64_t> pitch;  // by expert array
};
std::expected<Places, std::string> PlacesOf(const Target& t) {
  const auto& p = md::Qwen38Flash();
  const ar::Artifact& a = t.artifact;
  Places out;
  out.places.assign(a.groups().size(), en::GroupPlace::kNone);
  for (std::size_t g = 0; g < a.groups().size(); ++g) {
    if (a.groups()[g].kind != ar::GroupKind::kExpert) out.places[g] = en::GroupPlace::kDevice;
  }
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    std::vector<std::pair<std::uint32_t, std::string_view>> arrays;
    for (const md::Qwen38Tensor* x : t.binding.layers[il].expert_arrays(t.binding.cutlass())) {
      arrays.emplace_back(x->index, x->type);
    }
    auto slab = en::ExpertSlab(a, arrays, p.experts, kSlabAlignment, il);
    if (!slab) return std::unexpected(slab.error());
    for (const auto& [index, type] : arrays) out.pitch[index] = slab->stride;
    out.slabs.push_back(*slab);
  }
  return out;
}

// The n-gram table's rows a step's rows read, in order.
std::vector<std::int32_t> SideRows(const Target& t, const Planned& p) {
  const auto& prof = md::Qwen38Flash();
  std::vector<std::int32_t> out;
  for (std::size_t i = 0; i < p.wave.units.size(); ++i) {
    const Unit& u = p.wave.units[i];
    const std::vector<std::int32_t> history = History(u.slot, p.past[i] + u.rows);
    for (std::uint32_t r = 0; r < u.rows; ++r) {
      const auto rows = md::Qwen38PleRows(prof, t.hash, history, p.past[i] + r);
      out.insert(out.end(), rows.begin(), rows.end());
    }
  }
  return out;
}

struct Component {
  Outputs outputs;
  States states;
  std::uint64_t replayed = 0;
};

Component Lowered(const Target& t, const fs::path& scratch, std::span<const Planned> script) {
  Component out;
  const auto& mapping = sp::mappings::Qwen38Mapping();
  auto layout = co::StageLayout(t.spec, 0, kContext, kMaxRows);
  EXPECT_TRUE(layout.has_value()) << layout.error();
  if (!layout) return out;
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
    if (auto x = source->resources->OpenCublas("lowered cuBLAS workspace"); !x) return x;
    auto made = kg::GgmlBackend::Create(
        0, node.execution(), node.stream(0), &source->resources->cublas(),
        {.lower = {.expert_stride = [&pitch_of](const llmp::ir::WeightRef& ref) -> std::uint64_t {
           const auto at = pitch_of.find(ref);
           return at == pitch_of.end() ? 0 : at->second;
         }}});
    if (!made) return std::unexpected(made.error());
    source->backend = std::move(*made);
    auto weights = std::make_unique<en::PagedWeights>();
    if (auto x = weights->Open(t.root); !x) return x;
    en::PagedWeights* tw = weights.get();
    std::vector<std::unique_ptr<en::ComponentStepSource>> sources;
    auto stage0 = en::ComponentStepSource::Create(
        {.spec = &t.spec,
         .stage = 0,
         .bindings = {.tensors = t.source.tensors, .state = &*layout},
         .bound_digest = t.spec.Digest(),
         .mapping = &mapping,
         .values = t.values,
         .slots = kOwners,
         .policy = std::nullopt,
         .backend = source->backend->Backend()});
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
    config.weights.push_back(
        {.weights = std::move(weights), .places = places->places, .slabs = places->slabs});
    config.setup_shapes = shapes;
    // Prompt chunks captured on their second run, as decodes (D-090).
    config.capture_phases = 0x1F;
    config.launch = [&]() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return source->backend->Launch(
          {.base = node.pool().base, .size = llmp::base::Bytes(node.pool().bytes)});
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
        std::vector<std::vector<std::int32_t>> tokens;
        for (std::size_t i = 0; i < planned.wave.units.size(); ++i) {
          const Unit& u = planned.wave.units[i];
          const auto history = History(u.slot, planned.past[i] + u.rows);
          tokens.emplace_back(history.end() - u.rows, history.end());
          past.at(u.slot) = planned.past[i] + u.rows;
        }
        std::vector<en::SegmentWork> work;
        for (const auto& ids : tokens) work.push_back({.tokens = ids, .bytes = {}});
        if (w == 0) {
          // Missing or partial side-table indices must refuse before any
          // model work. Zero-filling them would silently read wrong rows.
          for (const std::size_t count : {std::size_t{0}, side[w].size() - 1}) {
            auto refused = runner->RunStep({.shape = shape,
                                            .work = work,
                                            .side_rows = std::span(side[w]).first(count),
                                            .ahead = {}});
            if (refused || refused.error().what != "the side rows must fill their slot") {
              return en::support::Error("incomplete side rows were not cleanly refused");
            }
          }
        }
        auto ran = runner->RunStep({.shape = shape,
                                    .work = work,
                                    .step_bytes = {},
                                    .side_rows = side[w],
                                    .advance = true,
                                    .ahead = {}});
        if (!ran) return std::unexpected(std::format("wave {}: {}", w, ran.error().what));
        if (ran->path == en::RunPath::kReplayed) ++out.replayed;
        if (planned.wave.head == Head::kNone) continue;
        for (const auto& bytes : ran->outputs) row.push_back(bytes);
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

class ComponentRunnerQwen38Gpu : public testing::TestWithParam<Case> {};

TEST_P(ComponentRunnerQwen38Gpu, FastFormStepsAreTodaysBitForBit) {
  const Case& c = GetParam();
  const auto root = Installed(c.id);
  if (!root) GTEST_SKIP() << "no Qwen3.8 artifact " << c.id << " installed";
  const auto target = Open(*root);
  ASSERT_TRUE(target.has_value());
  // NOLINTNEXTLINE(concurrency-mt-unsafe): the test environment is immutable
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch != nullptr ? scratch : testing::TempDir()) / "lowered38";
  fs::create_directories(out);
  const std::vector<Planned> script = Script();
  const Production today = Today(*target, out, script);
  ASSERT_EQ(today.outputs.size(), script.size());
  const Component lowered = Lowered(*target, out, script);
  ASSERT_EQ(lowered.outputs.size(), script.size());
  EXPECT_GE(lowered.replayed, 2U);
  std::size_t compared = 0;
  for (std::size_t w = 0; w < script.size(); ++w) {
    const auto& want_row = today.outputs[w];
    ASSERT_EQ(lowered.outputs[w].size(), want_row.size()) << "wave " << w;
    for (std::size_t u = 0; u < want_row.size(); ++u) {
      const auto& want = want_row[u];
      const auto& got = lowered.outputs[w][u];
      ASSERT_EQ(got.size(), want.size()) << "wave " << w << " owner " << u;
      if (script[w].wave.head == Head::kGreedy) {
        EXPECT_EQ(std::memcmp(got.data(), want.data(), want.size()), 0)
            << "wave " << w << " owner " << u << ": another token";
      } else {
        std::size_t differ = 0;
        double worst = 0.0;
        for (std::size_t j = 0; j < want.size() / sizeof(float); ++j) {
          float a = 0;
          float b = 0;
          std::memcpy(&a, want.data() + (j * sizeof(float)), sizeof(float));
          std::memcpy(&b, got.data() + (j * sizeof(float)), sizeof(float));
          differ += std::memcmp(&a, &b, sizeof(float)) != 0;
          worst = std::max(worst, std::fabs(static_cast<double>(a) - b));
        }
        if (differ != 0) {
          std::printf(
              "%s wave %zu owner %zu: %zu values differ in their bits, the largest by %.6g\n",
              c.name.c_str(), w, u, differ, worst);
        }
        EXPECT_EQ(differ, 0U) << "wave " << w << " owner " << u;
      }
      ++compared;
    }
  }
  // Each owner's state, tensor by tensor.
  ASSERT_EQ(today.states.size(), lowered.states.size());
  std::size_t tensors = 0;
  for (std::size_t s = 0; s < today.states.size(); ++s) {
    for (const auto& [name, want] : today.states[s]) {
      const auto at = lowered.states[s].find(name);
      ASSERT_NE(at, lowered.states[s].end()) << "owner " << s << " has no " << name;
      ASSERT_EQ(at->second.size(), want.size()) << "owner " << s << " " << name;
      const bool same = std::memcmp(at->second.data(), want.data(), want.size()) == 0;
      if (!same) {
        std::size_t differ = 0;
        for (std::size_t j = 0; j < want.size(); ++j) differ += at->second[j] != want[j];
        std::printf("%s owner %zu %s: %zu of %zu bytes differ\n", c.name.c_str(), s, name.c_str(),
                    differ, want.size());
      }
      EXPECT_TRUE(same) << "owner " << s << " " << name;
      ++tensors;
    }
  }
  std::printf("%s: %zu waves, %zu owner outputs and %zu state tensors compared, %llu replayed\n",
              c.name.c_str(), script.size(), compared, tensors,
              static_cast<unsigned long long>(lowered.replayed));
}

INSTANTIATE_TEST_SUITE_P(
    Qwen38, ComponentRunnerQwen38Gpu,
    testing::Values(
        Case{.name = "cutlass",
             .id = "m3-artifacts/c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93"},
        Case{.name = "ggml_experts",
             .id = "m3-artifacts/67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6"},
        Case{.name = "gguf",
             .id = "qgguf-artifacts/"
                   "5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78"}),
    [](const testing::TestParamInfo<Case>& info) { return info.param.name; });

}  // namespace
