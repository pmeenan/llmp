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
// (model/qwen38.h Qwen38PleRows), read on demand into bounded row slots.
// Speculation (a verify's
// saves and its commit, the MTP drafter's passes and their features) runs
// through the engine in G3c; its steps' lowering is compared with today's
// builder on the CPU (ggml_lower_qwen38_test).

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
#include "engine/decoding.h"
#include "engine/draft_companions.h"
#include "engine/expert_slab.h"
#include "engine/live_state.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/qwen38_runner.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "engine/wave_composer.h"
#include "kernels/ggml/qwen38_commit.h"
#include "kernels/ggml/step_program.h"
#include "model/qwen38.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "platform/kept_files.h"
#include "providers/device_runtime.h"
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
constexpr std::string_view kMtp =
    "m3-artifacts/8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40";

struct Case {
  std::string name;
  std::string_view id;
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
  const bool verifying =
      std::ranges::any_of(script, [](const Planned& p) { return p.wave.units.front().keep != 0; });
  const en::Qwen38Options options{
      .artifact = t.root,
      .out = out,
      .context = kContext,
      .max_rows = kMaxRows,
      .drafter = verifying ? fs::path(llmp::test_support::ModelsDir()) / kMtp : fs::path{},
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
        if (p.wave.units.front().keep != 0) {
          std::vector<std::vector<std::int32_t>> picks(n);
          if (joined) {
            std::vector<en::Qwen38Runner::VerifyWork> work;
            for (std::size_t i = 0; i < n; ++i) {
              work.push_back({.slot = slots[p.wave.units[i].slot],
                              .history = histories[i],
                              .n_past = p.past[i],
                              .argmax = &picks[i],
                              .logits = &logits[i]});
            }
            if (auto x = runner->VerifyWave(work); !x) return x;
          } else if (auto x = slots[p.wave.units[0].slot]->Verify(histories[0], p.past[0], picks[0],
                                                                  &logits[0]);
                     !x)
            return x;
          for (const Unit& u : p.wave.units) {
            if (auto x = slots[u.slot]->Accept(u.keep); !x) return x;
          }
        } else if (joined) {
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
          const Unit& u = p.wave.units[i];
          past.at(u.slot) = p.past[i] + (u.keep == 0 ? u.rows : u.keep);
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
// device except the row-paged n-gram table, each layer's expert arrays in
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
  out.places[a.resources()[t.binding.ple_table.index].group] = en::GroupPlace::kNone;
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
        if (*found == t.binding.ple_table.index) {
          table = {.weight = w.ref,
                   .resource = *found,
                   .format = step->graph.tensor(w.tensor).format,
                   .part = role,
                   .rows_per_token = md::Qwen38Flash().ple_heads()};
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
    auto backend = source->backend->Backend();
    const auto lower = backend.lower;
    backend.lower = [&](const kn::LowerInput& input) {
      if (!input.addresses.measuring() && input.shape.phase == pl::Phase::kVerify) {
        for (const auto& save : input.step.saves) {
          for (std::size_t segment = 0; segment < input.shape.segments.size(); ++segment) {
            saves[{save.layer, save.what, input.shape.segments[segment].slot}] =
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
         .values = t.values,
         .slots = kOwners,
         .policy = std::nullopt,
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
    config.launch = [&]() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      auto made_launch = source->backend->Launch(
          {.base = node.pool().base, .size = llmp::base::Bytes(node.pool().bytes)});
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
              const auto& profile = md::Qwen38Flash();
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
void CompareLogits(std::span<const std::byte> got, std::span<const std::byte> want,
                   std::size_t wave, std::size_t owner) {
  ASSERT_EQ(got.size(), want.size());
  std::size_t different = 0;
  double worst = 0;
  for (std::size_t at = 0; at < got.size(); at += sizeof(float)) {
    float a = 0, b = 0;
    std::memcpy(&a, got.data() + at, sizeof(a));
    std::memcpy(&b, want.data() + at, sizeof(b));
    different += std::memcmp(&a, &b, sizeof(a)) != 0;
    worst = std::max(worst, std::abs(static_cast<double>(a) - b));
  }
  EXPECT_EQ(different, 0U) << "wave " << wave << " owner " << owner << " worst " << worst;
}

// Every format's saved inputs drive an independent legacy commit oracle.
// This is exact against the verify's recurrence, without pretending the
// fast verify has the arithmetic of a shorter plain target step.
TEST_P(ComponentRunnerQwen38Gpu, VerifiedPrefixesMatchTheSavedRecurrence) {
  const auto root = Installed(GetParam().id);
  if (!root) GTEST_SKIP() << "no Qwen3.8 artifact installed";
  const auto target = Open(*root);
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

TEST(ComponentRunnerQwen38VerifyGpu, NativeVerifyMatchesLegacySavedRowsAndCommits) {
  const auto root =
      Installed("m3-artifacts/c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93");
  if (!root || !Installed(kMtp)) GTEST_SKIP() << "no native target and MTP fixtures installed";
  const auto target = Open(*root);
  ASSERT_TRUE(target);
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const fs::path out =
      fs::path(scratch != nullptr ? scratch : testing::TempDir()) / "verify-native38";
  fs::create_directories(out);
  const auto script = Script(true);
  const auto today = Today(*target, out, script);
  ASSERT_EQ(today.outputs.size(), script.size());
  const auto lowered = Lowered(*target, out, script);
  ASSERT_EQ(lowered.outputs.size(), script.size());
  for (std::size_t w = 0; w < script.size(); ++w) {
    ASSERT_EQ(lowered.outputs[w].size(), today.outputs[w].size());
    for (std::size_t u = 0; u < today.outputs[w].size(); ++u)
      CompareLogits(lowered.outputs[w][u], today.outputs[w][u], w, u);
  }
  CompareStates(lowered.states, today.states);
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

struct MtpAction {
  enum Kind { kPrompt, kUnit, kRefusedPrompt, kSave, kRestore, kSpill } kind;
  std::uint32_t slot = 0, rows = 0;
  std::int32_t anchor_delta = 0;
};
std::vector<MtpAction> MtpActions() {
  using A = MtpAction;
  return {{A::kPrompt, 0, 7},   {A::kPrompt, 1, 5}, {A::kPrompt, 0, 4},        {A::kUnit, 0, 3},
          {A::kUnit, 1, 3},     {A::kSave, 0},      {A::kRefusedPrompt, 0, 2}, {A::kUnit, 0, 0},
          {A::kUnit, 0, 1, 29}, {A::kUnit, 1, 0},   {A::kPrompt, 0, 5},        {A::kUnit, 0, 2},
          {A::kUnit, 1, 2},     {A::kRestore, 0},   {A::kUnit, 0, 0, 71},      {A::kSpill, 0},
          {A::kUnit, 0, 2},     {A::kUnit, 1, 2},   {A::kUnit, 0, 2},          {A::kUnit, 1, 2},
          {A::kUnit, 0, 2},     {A::kUnit, 1, 2},   {A::kUnit, 0, 2},          {A::kUnit, 1, 2}};
}
using MtpState = std::map<std::string, std::vector<std::byte>>;
struct MtpRecord {
  MtpAction action;
  std::vector<std::int32_t> history, drafts;
  std::vector<float> target_logits;
  en::Qwen38DraftHeadCapture head;
  std::uint32_t positions = 0, pending = 0;
  MtpState state;
};

en::Status ReadMtpBytes(en::PagedNode& node, const llmp::catalog::Closure& closure,
                        std::uint64_t base, std::uint64_t bytes, std::vector<std::byte>& out) {
  if (bytes == 0) {
    out.clear();
    return {};
  }
  std::vector<llmp::catalog::ExtentId> staging;
  auto host = node.Pinned(bytes, 0, staging);
  if (!host) return std::unexpected(host.error());
  auto copied = node.Copy(0, closure, base, *host, bytes, true, "MTP numerical state");
  if (!copied) {
    // Failed copies do not prove retirement of a possible DMA into this host.
    node.KeepPinned(*host);
    return copied;
  }
  const auto* begin = static_cast<const std::byte*>(*host);
  out.assign(begin, begin + bytes);
  return node.FreePinned(*host);
}

std::vector<MtpRecord> MtpToday(const Target& t, const fs::path& scratch) {
  const auto actions = MtpActions();
  std::vector<MtpRecord> records(actions.size());
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  const en::Qwen38Options options{.artifact = t.root,
                                  .out = scratch,
                                  .context = 512,
                                  .max_rows = 16,
                                  .drafter = fs::path(llmp::test_support::ModelsDir()) / kMtp,
                                  .draft_rows = 2,
                                  .draft_vocab = 65536,
                                  .draft_head_capture = true,
                                  .wave_slots = 2,
                                  .request_slots = 2,
                                  .wave_read_align = 2048,
                                  .spill_place = {}};
  auto runner = std::make_unique<en::Qwen38Runner>(node, options, 0, 0);
  std::vector<en::PagedModel*> entered;
  const auto ran = [&]() -> en::Status {
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
    if (auto x = node.Load(runner->weights(), "legacy MTP weights", loads); !x) return x;
    if (auto x = runner->ReadPleHash(); !x) return x;
    std::vector<en::Qwen38Runner::Slot*> slots;
    for (std::uint32_t i = 0; i < 2; ++i) {
      auto slot = runner->request_slot(i);
      if (!slot) return std::unexpected(slot.error());
      slots.push_back(*slot);
    }
    if (auto x = runner->SelectSlots(slots); !x) return x;
    return node.WithRequest(
        0, runner->execution_closure(), "legacy MTP script", [&]() -> en::Status {
          std::array<std::vector<std::int32_t>, 2> histories;
          std::vector<std::int32_t> saved_history;
          std::vector<en::LiveState::Range> saved_ranges;
          void* saved = nullptr;
          std::uint32_t saved_pending = 0;
          for (std::size_t index = 0; index < actions.size(); ++index) {
            const auto action = actions[index];
            auto& record = records[index];
            record.action = action;
            auto& history = histories[action.slot];
            auto& slot = *slots[action.slot];
            const auto past = static_cast<std::uint32_t>(history.size());
            switch (action.kind) {
              case MtpAction::kPrompt:
                for (std::uint32_t i = 0; i < action.rows; ++i)
                  history.push_back(TokenAt(action.slot, past + i));
                record.history = history;
                if (auto x = slot.PrefillChunk(history, past, record.target_logits, true); !x)
                  return x;
                break;
              case MtpAction::kUnit: {
                record.history = history;
                record.history.push_back(TokenAt(action.slot, past) + action.anchor_delta);
                if (auto x = slot.Draft(record.history, record.drafts, nullptr, 2, &record.head);
                    !x)
                  return x;
                if (record.head.catch_up_rows != slot.pending_rows())
                  return std::unexpected("legacy head did not execute its retained K-row schedule");
                auto verify = record.history;
                verify.insert(verify.end(), record.drafts.begin(), record.drafts.end());
                std::vector<std::int32_t> picks;
                if (auto x = slot.Verify(verify, past, picks, &record.target_logits); !x) return x;
                if (action.rows == 0) {
                  if (auto x = slot.DiscardVerify(); !x) return x;
                } else {
                  if (auto x = slot.Accept(action.rows); !x) return x;
                  if (auto x = slot.Rollback(); !x) return x;
                  history.assign(verify.begin(), verify.begin() + past + action.rows);
                }
                break;
              }
              case MtpAction::kRefusedPrompt:
                record.history = history;
                for (std::uint32_t i = 0; i < action.rows; ++i)
                  record.history.push_back(TokenAt(action.slot, past + i));
                break;  // The shared-only clean-refusal witness must leave this
                        // logical schedule.
              case MtpAction::kSave: {
                saved_history = history;
                saved_pending = slot.pending_rows();
                saved_ranges = slot.used_state_ranges();
                std::uint64_t bytes = 0;
                for (const auto& r : saved_ranges) bytes += r.bytes;
                std::vector<llmp::catalog::ExtentId> extents;
                auto host = node.Pinned(bytes, 0, extents);
                if (!host) return std::unexpected(host.error());
                saved = *host;
                if (auto x = slot.SaveUsedState(saved, saved_ranges); !x) return x;
                break;
              }
              case MtpAction::kRestore:
                if (auto x = slot.PrepareRestoreState(saved_ranges); !x) return x;
                if (auto x = slot.RestoreUsedState(saved, saved_ranges); !x) return x;
                slot.set_pending_rows(saved_pending);
                history = saved_history;
                break;
              case MtpAction::kSpill:
                if (auto x = slot.Spill(); !x) return x;
                if (auto x = slot.Restore(); !x) return x;
                break;
            }
            record.positions = static_cast<std::uint32_t>(history.size());
            record.pending = slot.pending_rows();
            const auto& p = md::Qwen38Flash();
            const auto& layout = runner->mtp_state();
            struct Part {
              const char* name;
              std::uint64_t offset, bytes;
            };
            const std::array parts{
                Part{"k", layout.k, std::uint64_t{record.positions} * p.kv_heads * p.head_dim * 2},
                Part{"v", layout.v, std::uint64_t{record.positions} * p.kv_heads * p.head_dim * 2},
                Part{"indexer_k", layout.indexer,
                     std::uint64_t{record.positions} * p.indexer_head_dim * 4},
                Part{"indexer_blocks", layout.blocks,
                     std::uint64_t{record.positions / p.indexer_ratio} * p.indexer_head_dim * 2},
                Part{"streams", layout.hidden + std::uint64_t{p.hc_width()} * 4,
                     std::uint64_t{record.pending} * p.hc_width() * 4}};
            for (const auto& part : parts)
              if (auto x = ReadMtpBytes(node, runner->execution_closure(),
                                        slot.drafter_state_base() + part.offset, part.bytes,
                                        record.state[part.name]);
                  !x)
                return x;
          }
          return {};
        });
  }();
  EXPECT_TRUE(ran) << (ran ? "" : ran.error());
  const auto retired = node.TearDown(entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) std::ignore = runner.release();
  return records;
}

// Test-only delegation keeps the real GGML program and CUDA recording. It
// identifies a draft-head graph when recorded, then counts later launches of
// that same graph after the script supplied a different anchor and features.
struct MtpReplayProbe {
  std::array<std::int32_t, 2> anchors{-1, -1};
  std::uint64_t head_captures = 0, changed_anchor_replays = 0, catch_up_replays = 0;
};
class MtpRecorded final : public kn::Recorded {
 public:
  MtpRecorded(const kn::StepLaunch* owner, std::unique_ptr<kn::Recorded> inner,
              std::uint32_t owners, bool catch_up, std::array<std::int32_t, 2> anchors)
      : Recorded(owner),
        inner_(std::move(inner)),
        owners_(owners),
        catch_up_(catch_up),
        anchors_(anchors) {}
  std::size_t nodes() const override { return inner_->nodes(); }
  double capture_seconds() const override { return inner_->capture_seconds(); }
  double instantiate_seconds() const override { return inner_->instantiate_seconds(); }
  std::unique_ptr<kn::Recorded> inner_;
  std::uint32_t owners_;
  bool catch_up_;
  std::array<std::int32_t, 2> anchors_;
};
class MtpLaunch final : public kn::StepLaunch {
 public:
  MtpLaunch(std::unique_ptr<kn::StepLaunch> inner, MtpReplayProbe& probe)
      : inner_(std::move(inner)), probe_(probe) {}
  std::expected<std::unique_ptr<kn::Recorded>, kn::StepFailure> Capture(
      llmp::providers::NativeStream stream, const Body& body) override {
    auto made = inner_->Capture(stream, body);
    if (!made) return std::unexpected(made.error());
    return std::make_unique<MtpRecorded>(this, std::move(*made), 0, false, probe_.anchors);
  }
  std::expected<void, kn::StepFailure> Launch(llmp::providers::NativeStream stream,
                                              const kn::Recorded& recorded) override {
    const auto& graph = static_cast<const MtpRecorded&>(recorded);
    auto queued = inner_->Launch(stream, *graph.inner_);
    bool changed = false;
    for (std::uint32_t slot = 0; slot < 2; ++slot)
      changed |=
          (graph.owners_ & (1U << slot)) != 0 && graph.anchors_[slot] != probe_.anchors[slot];
    if (queued && changed) {
      ++probe_.changed_anchor_replays;
      if (graph.catch_up_) ++probe_.catch_up_replays;
    }
    return queued;
  }
  bool faulted() const override { return inner_->faulted(); }
  kn::StepLaunch& inner() { return *inner_; }
  MtpReplayProbe& probe() { return probe_; }

 private:
  std::unique_ptr<kn::StepLaunch> inner_;
  MtpReplayProbe& probe_;
};
class MtpProgram final : public kn::StepProgram {
 public:
  MtpProgram(std::unique_ptr<kn::StepProgram> inner, std::uint32_t owners, bool catch_up)
      : inner_(std::move(inner)), owners_(owners), catch_up_(catch_up) {}
  std::uint64_t scratch_bytes() const override { return inner_->scratch_bytes(); }
  std::uint64_t pool_bytes() const override { return inner_->pool_bytes(); }
  std::uint64_t host_bytes() const override { return inner_->host_bytes() + sizeof(*this); }
  std::uint64_t launches() const override { return inner_->launches(); }
  std::span<const kn::InputSlot> inputs() const override { return inner_->inputs(); }
  std::span<const kn::OutputSlot> outputs() const override { return inner_->outputs(); }
  std::span<const kn::TensorUse> tensors() const override { return inner_->tensors(); }
  const llmp::execution::BoundPlan* bound() const override { return inner_->bound(); }
  bool capturable() const override { return inner_->capturable(); }
  std::expected<void, kn::StepFailure> Bind(kn::StepLaunch& launch) override {
    return inner_->Bind(static_cast<MtpLaunch&>(launch).inner());
  }
  std::expected<void, kn::StepFailure> Position(std::span<const std::uint32_t> past) override {
    return inner_->Position(past);
  }
  std::expected<void, kn::StepFailure> Run(kn::StepLaunch& launch,
                                           llmp::providers::NativeStream stream) const override {
    return inner_->Run(static_cast<MtpLaunch&>(launch).inner(), stream);
  }
  std::expected<std::unique_ptr<kn::Recorded>, kn::StepFailure> Record(
      kn::StepLaunch& launch, llmp::providers::NativeStream stream,
      const kn::StepLaunch::Body& body) const override {
    auto& outer = static_cast<MtpLaunch&>(launch);
    auto made = inner_->Record(outer.inner(), stream, body);
    if (!made) return std::unexpected(made.error());
    if (owners_ != 0) ++outer.probe().head_captures;
    return std::make_unique<MtpRecorded>(&launch, std::move(*made), owners_, catch_up_,
                                         outer.probe().anchors);
  }

 private:
  std::unique_ptr<kn::StepProgram> inner_;
  std::uint32_t owners_;
  bool catch_up_;
};

struct MtpKept {
  int dir = -1;
  std::array<std::array<llmp::platform::FileIdentity, 2>, 2> identities{};
  std::array<std::vector<en::LiveState::Range>, 2> footprint{};
  std::array<std::uint32_t, 2> positions{};
  std::array<md::DraftProgress, 2> progress{};
  std::string layout{};
  std::uint64_t head_captures = 0, changed_head_replays = 0, catch_up_replays = 0;
};

void MtpLowered(const Target& t, const fs::path& scratch, std::span<const MtpRecord> records,
                MtpKept& kept, bool resume) {
  auto artifact =
      ar::Artifact::Open(fs::path(llmp::test_support::ModelsDir()) / kMtp, {.expected_id = {}});
  ASSERT_TRUE(artifact);
  auto binding = md::BindQwen38Mtp(md::Qwen38Flash(), *artifact);
  ASSERT_TRUE(binding);
  const auto& mapping = sp::mappings::Qwen38MtpMapping();
  auto values = sp::mappings::Qwen38MtpV0Values(md::Qwen38Flash(), *binding);
  ASSERT_TRUE(values);
  auto mtp_source = sp::SourceOf(*artifact, &mapping);
  ASSERT_TRUE(mtp_source);
  auto spec =
      sp::ComposeDrafter(t.spec, sp::mappings::Qwen38Mapping(), mapping, *mtp_source, *values);
  ASSERT_TRUE(spec) << (spec ? "" : spec.error());
  spec->stages[1].embedding->version = 2;
  spec->stages[1].embedding->params.push_back({.name = "token_offset", .value = std::int64_t{1}});
  std::ranges::sort(spec->stages[1].embedding->params, {}, &sp::Param::name);
  for (auto& layer : spec->stages[1].layers)
    for (auto& component : layer.components)
      if (component.kind == "attention.qsa") component.version = 2;
  std::array<md::StateLayoutDerived, 2> layouts;
  for (std::uint16_t stage = 0; stage < 2; ++stage) {
    auto layout = co::StageLayout(*spec, stage, 512, 16);
    ASSERT_TRUE(layout) << (layout ? "" : layout.error());
    layouts[stage] = std::move(*layout);
  }
  const std::array<const md::StateLayoutDerived*, 2> layout_ptr{&layouts[0], &layouts[1]};
  const std::array<std::span<const co::BoundTensor>, 2> tables{t.source.tensors,
                                                               mtp_source->tensors};
  const std::array<const ar::Artifact*, 2> artifacts{&t.artifact, &*artifact};
  std::array<en::LayerSlabs, 2> placements;
  for (std::uint16_t stage = 0; stage < 2; ++stage) {
    auto placement = en::ExpertSlabsOf(*artifacts[stage], kSlabAlignment);
    ASSERT_TRUE(placement) << (placement ? "" : placement.error());
    placements[stage] = std::move(*placement);
  }
  placements[0].places[t.artifact.resources()[t.binding.ple_table.index].group] =
      en::GroupPlace::kNone;
  std::vector<pl::StepShape> shapes;
  for (std::uint16_t stage = 0; stage < 2; ++stage)
    for (const std::uint32_t rows : {1U, 3U, 16U})
      for (const bool head : {false, true}) {
        shapes.push_back({.stage = stage,
                          .phase = stage == 0 ? pl::Phase::kPrefill : pl::Phase::kDraft,
                          .segments = {{.rows = rows,
                                        .past = 512 - rows,
                                        .read_cells = 512,
                                        .outputs = head ? 1U : 0U,
                                        .want_head = head}}});
      }
  shapes.push_back({.phase = pl::Phase::kVerify,
                    .segments = {{.rows = 3, .past = 509, .read_cells = 512, .outputs = 3}}});
  struct Place {
    std::uint16_t artifact;
    bool array;
    std::uint32_t index;
  };
  std::map<std::pair<llmp::ir::WeightRef, llmp::format::PartRole>, Place> places;
  std::map<llmp::ir::WeightRef, std::uint64_t> pitch;
  std::optional<en::RunnerConfig::RowTable> table;
  for (const auto& shape : shapes) {
    auto step = co::BuildStep(*spec, shape.stage, shape,
                              {.tensors = tables[shape.stage],
                               .state = &layouts[shape.stage],
                               .stages = layout_ptr,
                               .artifacts = tables});
    ASSERT_TRUE(step) << (step ? "" : step.error());
    for (const auto& weight : step->weights)
      for (const auto& [name, role] : weight.parts) {
        const auto& a = *artifacts[weight.ref.stage];
        const auto found = weight.ref.expert_array ? a.FindExpertArray(name) : a.FindResource(name);
        ASSERT_TRUE(found) << name;
        places[{weight.ref, role}] = {weight.ref.stage, weight.ref.expert_array, *found};
        if (weight.ref.expert_array) pitch[weight.ref] = placements[weight.ref.stage].pitch[*found];
        if (weight.ref.stage == 0 && !weight.ref.expert_array &&
            *found == t.binding.ple_table.index)
          table = {.weight = weight.ref,
                   .resource = *found,
                   .format = step->graph.tensor(weight.tensor).format,
                   .part = role,
                   .rows_per_token = md::Qwen38Flash().ple_heads()};
      }
  }
  ASSERT_TRUE(table);
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  auto source = std::make_unique<KernelSource>(node);
  std::unique_ptr<en::ModelRunner> runner;
  std::vector<en::PagedModel*> entered;
  bool refuse_target = false;
  MtpReplayProbe replay;
  const auto ran = [&]() -> en::Status {
    if (auto x = node.Open(); !x) return x;
    if (auto x = source->resources->OpenCublas("MTP comparison cuBLAS"); !x) return x;
    auto backend =
        kg::GgmlBackend::Create(0, node.execution(), node.stream(0), &source->resources->cublas(),
                                {.lower = {.expert_stride = [&](const llmp::ir::WeightRef& ref) {
                                   return pitch.contains(ref) ? pitch.at(ref) : 0;
                                 }}});
    if (!backend) return std::unexpected(backend.error());
    source->backend = std::move(*backend);
    std::vector<std::unique_ptr<en::ComponentStepSource>> sources;
    for (std::uint16_t stage = 0; stage < 2; ++stage) {
      auto own_backend = source->backend->Backend();
      const auto lower = own_backend.lower;
      own_backend.lower = [&, lower](const kn::LowerInput& input)
          -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
        if (refuse_target && input.shape.stage == 0 && input.shape.phase == pl::Phase::kPrefill)
          return std::unexpected("injected clean target planning refusal");
        auto program = lower(input);
        if (!program) return std::unexpected(program.error());
        const bool head = input.shape.stage == 1 && input.shape.phase == pl::Phase::kDraft &&
                          std::ranges::any_of(input.shape.segments, [](const auto& segment) {
                            return segment.want_head && segment.outputs != 0;
                          });
        std::uint32_t owners = 0;
        bool catch_up = false;
        if (head)
          for (const auto& segment : input.shape.segments) {
            owners |= 1U << segment.slot;
            catch_up |= segment.rows > 1;
          }
        return std::make_unique<MtpProgram>(std::move(*program), owners, catch_up);
      };
      auto made = en::ComponentStepSource::Create(
          {.spec = &*spec,
           .stage = stage,
           .bindings = {.tensors = tables[stage],
                        .state = &layouts[stage],
                        .stages = layout_ptr,
                        .artifacts = tables},
           .bound_digest = spec->Digest(),
           .mapping = stage == 0 ? &sp::mappings::Qwen38Mapping() : &mapping,
           .values = stage == 0 ? std::span<const sp::SourceValue>(t.values)
                                : std::span<const sp::SourceValue>(*values),
           .slots = 2,
           .policy = std::nullopt,
           .backend = std::move(own_backend)});
      if (!made) return std::unexpected(made.error());
      sources.push_back(std::move(*made));
    }
    auto staged = en::StagedStepSource::Create(std::move(sources));
    if (!staged) return std::unexpected(staged.error());
    en::RunnerConfig config;
    config.slots = {
        .model = "MTP comparison",
        .slots = 2,
        .stages = {{.name = "target", .layout = layouts[0]}, {.name = "mtp", .layout = layouts[1]}},
        .spec = spec->Digest(),
        .spill_place = {},
        .spill_directory = scratch,
        .invalidate = {},
        .drop_plans = {}};
    config.slots.spill_place = [&](std::uint32_t slot, std::uint32_t stage) {
      return en::LiveState::SpillPlace{
          .directory = scratch,
          .dir = kept.dir,
          .name = std::format("owner-{}.stage-{}.state", slot, stage),
          .keep = resume,
          .expect = resume ? std::optional(kept.identities[slot][stage]) : std::nullopt};
    };
    std::array<en::PagedWeights*, 2> paged{};
    for (std::uint16_t stage = 0; stage < 2; ++stage) {
      auto weights = std::make_unique<en::PagedWeights>();
      if (auto x =
              weights->Open(stage == 0 ? t.root : fs::path(llmp::test_support::ModelsDir()) / kMtp);
          !x)
        return x;
      paged[stage] = weights.get();
      config.weights.push_back({.weights = std::move(weights),
                                .places = placements[stage].places,
                                .slabs = placements[stage].slabs});
    }
    config.weight_address = [&](const auto& ref, auto role) -> std::uint64_t {
      const auto at = places.find({ref, role});
      if (at == places.end()) return 0;
      const auto& p = at->second;
      return p.array ? paged[p.artifact]->array_address(p.index)
                     : paged[p.artifact]->resource_address(p.index);
    };
    config.setup_shapes = shapes;
    config.max_verify_rows = 3;
    config.row_tables.push_back(*table);
    config.launch = [&]() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      auto launched = source->backend->Launch(
          {.base = node.pool().base, .size = llmp::base::Bytes(node.pool().bytes)});
      if (!launched) return std::unexpected(launched.error());
      return std::make_unique<MtpLaunch>(std::move(*launched), replay);
    };
    auto made = en::ModelRunner::Create(node, std::move(config), std::move(*staged), 0, 0);
    if (!made) return std::unexpected(made.error());
    runner = std::move(*made);
    entered = {runner.get(), source.get()};
    if (auto x = runner->Setup(); !x) return x;
    if (auto x = node.MapWorkspace(runner->activations_needed(),
                                   std::max(runner->pool_needed(), en::kPagedExtent));
        !x)
      return x;
    if (auto x = node.Start(llmp::base::Bytes(std::uint64_t{110} << 30U)); !x) return x;
    if (auto x = runner->Register(); !x) return x;
    if (auto x = runner->Bind(); !x) return x;
    node.Run();
    std::vector<en::LoadStats> loads;
    if (auto x = node.Load(runner->weights(), "shared MTP weights", loads); !x) return x;
    if (resume) {
      auto& slots = runner->slots();
      for (std::uint32_t slot = 0; slot < 2; ++slot) {
        for (std::size_t stage = 0; stage < 2; ++stage)
          if (slots.live(slot, stage).spill_identity() != kept.identities[slot][stage])
            return std::unexpected("restarted MTP opened another spill file");
        if (auto x = slots.Adopt(slot, kept.positions[slot], kept.footprint[slot], kept.layout); !x)
          return x;
        if (slots.usable(slot) || slots.kept_whole(slot) || slots.BeginDraftUnit(slot, 1))
          return std::unexpected("adopted MTP became ready before control hydration");
        if (auto x = slots.Restore(slot); !x) return x;
        if (slots.progress(slot, 1)->completed() != kept.progress[slot])
          return std::unexpected("restarted MTP changed its pending schedule");
      }
    }
    if (auto x = runner->slots().SelectSlots(std::array{0U, 1U}); !x) return x;
    auto caps = co::StageBatching(*spec, 0);
    if (!caps) return std::unexpected(caps.error());
    auto composer = en::WaveComposer::For(*caps, runner->slots());
    if (!composer) return std::unexpected(composer.error());
    std::array<const MtpRecord*, 2> current{};
    std::array<std::uint32_t, 2> pass{}, judgements{};
    en::DecodingPolicy policy;
    policy.propose = [&](std::uint32_t slot, std::size_t, std::span<const std::byte> bytes,
                         std::uint32_t) -> std::expected<std::vector<std::int32_t>, std::string> {
      const auto& record = *current[slot];
      const auto count = record.head.head_rows;
      const auto logits =
          std::span(record.head.logits).subspan(std::size_t{pass[slot]} * count, count);
      const auto expected = std::as_bytes(logits);
      if (bytes.size() < expected.size())
        return std::unexpected("MTP head output is shorter than its legacy logits");
      if (!std::ranges::equal(bytes.first(expected.size()), expected)) {
        const auto mismatch = std::mismatch(expected.begin(), expected.end(), bytes.begin());
        const auto word =
            static_cast<std::size_t>(mismatch.first - expected.begin()) / sizeof(float);
        float got = 0;
        std::memcpy(&got, bytes.data() + word * sizeof(float), sizeof(float));
        return std::unexpected(
            std::format("MTP head logits differ for owner{} pass{} at {}: {} versus {}", slot,
                        pass[slot], word, got, logits[word]));
      }
      const auto best = static_cast<std::size_t>(std::ranges::max_element(logits) - logits.begin());
      if (record.head.token_ids.at(best) != record.drafts.at(pass[slot]))
        return std::unexpected("MTP logits do not select the legacy draft token");
      return std::vector<std::int32_t>{record.drafts.at(pass[slot]++)};
    };
    policy.judge =
        [&](std::uint32_t slot, std::span<const std::int32_t>,
            std::span<const std::byte> bytes) -> std::expected<en::Verdict, std::string> {
      ++judgements[slot];
      const auto& record = *current[slot];
      if (!std::ranges::equal(bytes, std::as_bytes(std::span(record.target_logits))))
        return std::unexpected("target verify logits differ after MTP");
      if (record.action.rows == 0) return std::unexpected("whole discard witness");
      return en::Verdict{.keep = record.action.rows};
    };
    auto decoder = en::DecodingOrchestrator::Make(
        *runner, *composer,
        {.kind = en::DecodingMode::Kind::kSpeculative,
         .drafters = {{.stage = 1, .max_depth = 2, .catch_up = true, .features = true}}},
        std::move(policy));
    if (!decoder) return std::unexpected(decoder.error());
    return node.WithRequest(
        0, runner->slots().execution(), "shared MTP script", [&]() -> en::Status {
          auto& slots = runner->slots();
          void* saved = nullptr;
          std::vector<en::LiveState::Range> pages, footprint;
          std::uint32_t saved_positions = 0;
          std::vector<bool> completed(records.size(), false);
          for (std::size_t index = 0; index < records.size(); ++index) {
            const auto& record = records[index];
            const auto a = record.action;
            const auto past = slots.positions(a.slot);
            switch (a.kind) {
              case MtpAction::kRefusedPrompt:
                refuse_target = true;
                runner->DropPlans();
                [[fallthrough]];
              case MtpAction::kPrompt: {
                const en::PendingWork pending{.slot = a.slot,
                                              .phase = pl::Phase::kPrefill,
                                              .past = past,
                                              .rows = a.rows,
                                              .outputs = 1};
                const en::SegmentWork work{.tokens =
                                               std::span(record.history).subspan(past, a.rows)};
                const std::array<std::span<const std::int32_t>, 1> histories{record.history};
                auto report = en::RunPromptWave(*runner, *composer, std::span(&pending, 1),
                                                std::span(&work, 1), histories);
                if (refuse_target) {
                  refuse_target = false;
                  if (!report.outcomes[0].refusal || !report.outcomes[0].refusal->clean ||
                      slots.positions(a.slot) != past)
                    return std::unexpected(
                        "pre-companion target refusal did not "
                        "preserve the original schedule");
                } else {
                  if (report.outcomes[0].refusal)
                    return std::unexpected(report.outcomes[0].refusal->what);
                  if (!std::ranges::equal(report.outcomes[0].outputs,
                                          std::as_bytes(std::span(record.target_logits))))
                    return std::unexpected(
                        std::format("MTP prompt logits differ at action {}", index));
                }
                break;
              }
              case MtpAction::kUnit: {
                if (completed[index]) break;
                constexpr std::size_t width = 1;
                std::vector<en::DecodeWork> work;
                for (std::size_t j = 0; j < width; ++j) {
                  const auto& own = records[index + j];
                  const auto slot = own.action.slot;
                  current[slot] = &own;
                  pass[slot] = judgements[slot] = 0;
                  replay.anchors[slot] = own.history.back();
                  work.push_back({.slot = slot, .history = own.history});
                }
                auto report = decoder->Step(work);
                for (std::size_t j = 0; j < width; ++j) {
                  const auto& own = records[index + j];
                  const auto slot = own.action.slot;
                  const auto& out = report.outcomes[j];
                  const bool verdict_ok =
                      own.action.rows == 0
                          ? (out.discarded && out.refusal &&
                             out.refusal->what == "whole discard witness")
                          : (!out.discarded && !out.refusal && out.kept == own.action.rows);
                  if (pass[slot] != 2 || judgements[slot] != 1 || !verdict_ok)
                    return std::unexpected(
                        std::format("MTP unit{} owner{}: {}; draft: {} ({} passes)", index + j,
                                    slot, out.refusal ? out.refusal->what : "incomplete",
                                    !out.drafts.empty() && out.drafts.back().refusal
                                        ? out.drafts.back().refusal->what
                                        : "none",
                                    pass[slot]));
                  if (auto x = slots.Settle(slot); !x) return x;
                  completed[index + j] = true;
                }
                break;
              }
              case MtpAction::kSave: {
                saved_positions = past;
                footprint.clear();
                for (std::size_t stage = 0; stage < slots.stages(); ++stage)
                  for (auto r : slots.live(a.slot, stage).used_ranges()) {
                    r.region = stage;
                    footprint.push_back(r);
                  }
                auto ranges = slots.CheckpointRanges(a.slot, past);
                if (!ranges) return std::unexpected(ranges.error());
                pages = std::move(*ranges);
                std::uint64_t bytes = 0;
                for (const auto& p : pages) bytes += p.bytes;
                std::vector<llmp::catalog::ExtentId> extents;
                auto host = node.Pinned(bytes, 0, extents);
                if (!host) return std::unexpected(host.error());
                saved = *host;
                if (auto x = slots.CopyState(a.slot, saved, pages, true); !x) return x;
                break;
              }
              case MtpAction::kRestore:
                if (auto x = slots.PrepareRestore(a.slot, saved_positions, footprint,
                                                  slots.kept_layout_id());
                    !x)
                  return x;
                if (auto x = slots.CopyState(a.slot, saved, pages, false); !x) return x;
                if (auto x = slots.CompleteRestore(a.slot, saved_positions); !x) return x;
                decoder->Reset(a.slot);
                break;
              case MtpAction::kSpill:
                if (auto x = slots.Spill(a.slot); !x) return x;
                if (auto x = slots.Restore(a.slot); !x) return x;
                decoder->Reset(a.slot);
                break;
            }
            const auto& progress = slots.progress(a.slot, 1)->completed();
            if (progress.positions != record.positions ||
                progress.positions - progress.pending_first != record.pending)
              return std::unexpected(
                  std::format("MTP pending schedule differs at action {}", index));
            if (a.kind == MtpAction::kRefusedPrompt)
              continue;  // physical pre-work is replayed by the following exact
                         // K-row pass
            for (const auto& [name, expected] : record.state) {
              const auto item = std::ranges::find(layouts[1].items, name,
                                                  [](const auto& i) { return i.contract.name; });
              if (item == layouts[1].items.end()) return std::unexpected("MTP state item missing");
              std::uint64_t offset = item->offset;
              if (name == "streams")
                offset +=
                    (item->contract.window_or_ratio + progress.pending_first - progress.tap_first) *
                    item->contract.cell_bytes;
              std::vector<std::byte> got;
              if (auto x =
                      ReadMtpBytes(node, slots.execution(), slots.live(a.slot, 1).base(0) + offset,
                                   expected.size(), got);
                  !x)
                return x;
              if (got != expected)
                return std::unexpected(
                    std::format("MTP state {} differs at action {}", name, index));
            }
          }
          if (!resume) {
            kept.layout = slots.kept_layout_id();
            for (std::uint32_t slot = 0; slot < 2; ++slot) {
              if (auto x = slots.Settle(slot); !x) return x;
              kept.positions[slot] = slots.positions(slot);
              kept.progress[slot] = slots.progress(slot, 1)->completed();
              for (std::size_t stage = 0; stage < 2; ++stage) {
                for (auto range : slots.live(slot, stage).used_ranges()) {
                  range.region = stage;
                  kept.footprint[slot].push_back(range);
                }
              }
              if (auto x = slots.Spill(slot); !x) return x;
              if (!slots.kept_whole(slot)) return std::unexpected("MTP spill was not whole");
              for (std::size_t stage = 0; stage < 2; ++stage) {
                const auto& identity = slots.live(slot, stage).spill_identity();
                if (!identity) return std::unexpected("MTP spill has no named identity");
                kept.identities[slot][stage] = *identity;
              }
            }
          }
          return {};
        });
  }();
  EXPECT_TRUE(ran) << (ran ? "" : ran.error());
  if (runner) {
    EXPECT_EQ(runner->coverage().violations, 0U) << runner->coverage().first_violation;
    EXPECT_GT(runner->graph_stats().captured, 0U);
    EXPECT_GT(runner->graph_stats().replayed, 0U);
    kept.head_captures += replay.head_captures;
    kept.changed_head_replays += replay.changed_anchor_replays;
    kept.catch_up_replays += replay.catch_up_replays;
    std::printf("MTP multirow catch-up graph changed-anchor replays=%llu\n",
                static_cast<unsigned long long>(replay.catch_up_replays));
    std::printf("MTP head graph captures=%llu changed-anchor replays=%llu\n",
                static_cast<unsigned long long>(replay.head_captures),
                static_cast<unsigned long long>(replay.changed_anchor_replays));
    std::printf("MTP script: %zu actions, shared captured=%llu replayed=%llu\n", records.size(),
                static_cast<unsigned long long>(runner->graph_stats().captured),
                static_cast<unsigned long long>(runner->graph_stats().replayed));
  }
  const auto retired = node.TearDown(entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) {
    std::ignore = runner.release();
    std::ignore = source.release();
  }
}

void CheckMtp() {
  const auto root = Installed(
      "m3-artifacts/"
      "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93");
  if (!root || !Installed(kMtp)) GTEST_SKIP() << "no native target and MTP fixtures installed";
  const auto target = Open(*root);
  ASSERT_TRUE(target);
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  ASSERT_NE(scratch, nullptr);
  fs::create_directories(scratch);
  const auto records = MtpToday(*target, scratch);
  ASSERT_EQ(records.size(), MtpActions().size());
  const auto kept_path = fs::path(scratch) / "shared-mtp-kept";
  fs::remove_all(kept_path);
  const auto directory = llmp::platform::OpenPrivateDirectory(-1, kept_path.c_str());
  ASSERT_TRUE(directory);
  MtpKept kept{.dir = *directory};
  // The first runner covers checkpoint branching and the failed-prompt retry.
  // The second has fresh host metadata, plan caches and device reservations.
  const auto first_spill = std::ranges::find_if(
      records, [](const auto& record) { return record.action.kind == MtpAction::kSpill; });
  ASSERT_NE(first_spill, records.end());
  const auto boundary = static_cast<std::size_t>(first_spill - records.begin()) + 1;
  MtpLowered(*target, scratch, std::span(records).first(boundary), kept, false);
  if (!testing::Test::HasFailure())
    MtpLowered(*target, scratch, std::span(records).subspan(boundary), kept, true);
  EXPECT_GT(kept.head_captures, 0U);
  EXPECT_GT(kept.changed_head_replays, 0U);
  EXPECT_GT(kept.catch_up_replays, 0U);
  (void)::close(*directory);
  fs::remove_all(kept_path);
}

TEST(ComponentRunnerQwen38MtpGpu, PendingSchedulesAndFeaturesMatchLegacyAcrossRestoreAndRetry) {
  CheckMtp();
}

}  // namespace
