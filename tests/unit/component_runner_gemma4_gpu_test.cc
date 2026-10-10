// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The one model runner (engine/model_runner.h) over the component step
// source (engine/component_source.h) and the GGML module's lowering
// (kernels/ggml/step_program.h), on the real Gemma 4 26B-A4B and 31B
// artifacts and their assistants (label `models`), against today's runner
// (engine/gemma4_runner.h, gemma4_assistant.h) under its serving recipe
// (runtime/serving.cc's Gemma options; the component source plans under
// the gemma4 mapping's policy for the same envelope): within the
// qualified envelope at 4,096 and two owners (every row to the head, the
// fused norm-mul, norm-rope and norm-add, the 26B-A4B's fused routing and
// scaled reduce, the 31B's quantized GeGLU, owner attention with bounded
// roots over unequal reads, device masks), at 8,192 and four owners (owner
// attention over equal reads, its masks joined in pairs), and outside it
// (the frontier head's narrowed final layer, the routing and reduce as
// their primitives). The owners run one script of waves: joined prefill
// chunks, ragged and single-owner chunks, state-only chunks, one owner
// past its sliding layers' ring until it wraps, joined one-row decodes
// returning logits and device-chosen tokens; each owner's outputs are bit
// for bit today's. The repeated decode shapes are captured on their second
// run and replayed after (D-090), the replays compared too.
//
// The assistant's draft passes (two, joined over the owners, at each
// owner's committed endpoint after the first prompt chunk, reading the
// target's cache) run through the same runner as stage 1 (its own step
// source over the composed spec) with the target's state borrowed, their
// first feature the one today's runner retains for the endpoint and their
// tokens today's anchor and proposal; each pass's logits and next
// features are today's assistant's bit for bit. The target's own steps
// here write no tap (the target-only spec: the engine stages no tap cells
// of a frontier row yet, G2c).

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
#include <vector>

#include "artifact/artifact.h"
#include "artifact/gguf_metadata.h"
#include "base/bytes.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/expert_slab.h"
#include "engine/gemma4_assistant.h"
#include "engine/gemma4_runner.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "kernels/ggml/step_program.h"
#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma4.h"
#include "tokenizer/gguf.h"
#include "tokenizer_fixtures.h"

namespace {

namespace co = llmp::components;
namespace en = llmp::engine;
namespace ar = llmp::artifact;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace tk = llmp::tokenizer;
namespace fs = std::filesystem;

constexpr std::uint32_t kMaxRows = 128;
constexpr std::uint32_t kVocab = 262144;
const fs::path kStore = "m3-artifacts";

struct Case {
  std::string name;
  std::uint32_t size = 26;
  std::uint32_t context = 4096;
  std::uint32_t owners = 2;
  bool candidate = true;  // within the envelope
  bool assistant = false;
};

std::string_view TargetId(std::uint32_t size) {
  return size == 26 ? "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3"
                    : "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08";
}
std::string_view AssistantId(std::uint32_t size) {
  return size == 26 ? "1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42"
                    : "447a5c20a0a25632bf35e118d5dde1867a182cd209b9ccd3afe93866c3696120";
}
std::string_view TargetVocabulary(std::uint32_t size) {
  return size == 26 ? "gemma-4-26B-A4B-it-UD-Q4_K_M.kv.gguf" : "gemma-4-31B-it-UD-Q4_K_XL.kv.gguf";
}
std::string_view AssistantVocabulary(std::uint32_t size) {
  return size == 26 ? "mtp-gemma-4-26B-A4B-it.kv.gguf" : "mtp-gemma-4-31B-it.kv.gguf";
}
const md::Gemma4Profile& Profile(std::uint32_t size) {
  return size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
}

// ------------------------------------------------------------ the script

enum class Head : std::uint8_t { kLogits, kGreedy, kNone, kDraft };
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
  std::vector<Wave> waves = {W(All(n, n <= 2 ? 37U : 31U))};
  if (c.assistant) {
    // Two draft passes over every owner's endpoint.
    waves.push_back(W(All(n, 1), Head::kDraft));
    waves.push_back(W(All(n, 1), Head::kDraft));
  }
  const auto more = [&](std::vector<Wave> w) { waves.insert(waves.end(), w.begin(), w.end()); };
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
  for (std::size_t i = 0; i < p.wave.units.size(); ++i) {
    const Unit& u = p.wave.units[i];
    if (u.rows > 1) shape.phase = pl::Phase::kPrefill;
    // A draft reads its target's committed positions (frozen), today's
    // assistant's width.
    const std::uint32_t end = p.past[i] + (draft ? 0 : u.rows);
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(u.slot),
                              .rows = u.rows,
                              .past = p.past[i],
                              .read_cells = ReadOf(end, context),
                              .outputs = p.wave.head == Head::kNone ? 0U : 1U,
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

// ------------------------------------------------------------ today's runner

struct Vocabulary {
  tk::GgufTokenizer tokenizer;
  ar::GgufMetadata raw;
  md::Gemma4AssistantVocabulary view() const {
    return {tokenizer.spec.tokens, tokenizer.spec.merges, raw.at("tokenizer.ggml.scores").reals,
            raw.at("tokenizer.ggml.token_type").integers};
  }
};
std::expected<Vocabulary, std::string> Decode(const ar::Artifact& artifact,
                                              std::string_view filename) {
  auto bytes = artifact.ReadMetadata(filename);
  if (!bytes) return en::support::Error(bytes.error().ToString());
  auto input = std::as_bytes(std::span(bytes->data(), bytes->size()));
  auto tokenizer = tk::ReadGgufTokenizer(input);
  constexpr std::array<std::string_view, 2> keys{"tokenizer.ggml.scores",
                                                 "tokenizer.ggml.token_type"};
  auto raw = ar::ReadGgufMetadata(input, keys);
  if (!tokenizer || !raw || !raw->contains(keys[0]) || !raw->contains(keys[1]))
    return en::support::Error("vocabulary parse failed");
  return Vocabulary{std::move(*tokenizer), std::move(*raw)};
}

en::Gemma4Options Options(const Case& c, const fs::path& target, const fs::path& out) {
  const bool bounded = c.candidate && c.context <= 4096 && c.owners == 2;
  return {.artifact = target,
          .out = out,
          .variant = c.size == 26 ? en::Gemma4Variant::k26BA4B : en::Gemma4Variant::k31B,
          .context = c.context,
          .max_rows = kMaxRows,
          .slots = c.owners,
          .max_head_rows = c.owners,
          .frontier_head = !c.candidate,
          .fuse_norm_rope = c.candidate,
          .fuse_norm_add = c.candidate,
          .fuse_gemma_route = c.candidate && c.size == 26,
          .fuse_gemma_reduce = c.candidate && c.size == 26,
          .fuse_quant_glu = c.candidate && c.size == 31,
          .owner_attention = c.candidate,
          .common_owner_reads = bounded,
          .bounded_owner_roots = bounded};
}

// The production run of the target waves (drafts skipped).
Outputs Production(const Case& c, const fs::path& target, const fs::path& out,
                   std::span<const Planned> script) {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  auto runner = std::make_unique<en::Gemma4Runner>(node, Options(c, target, out), 0, 0);
  std::vector<en::PagedModel*> entered;
  Outputs outputs;
  const auto started = [&]() -> en::Status {
    if (auto x = node.Open(); !x) return x;
    entered.push_back(runner.get());
    if (auto x = runner->Setup(); !x) return x;
    if (auto x = node.MapWorkspace(runner->activations_needed(), runner->pool_needed()); !x)
      return x;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes() + (16ULL << 20U));
    if (auto x = node.Start(llmp::base::Bytes(fixed + runner->weights().size() * en::kPagedExtent +
                                              4 * node.StateCapacity()));
        !x)
      return x;
    if (auto x = runner->Register(); !x) return x;
    if (auto x = runner->Bind(); !x) return x;
    node.Run();
    std::vector<std::uint32_t> slots;
    for (std::uint32_t i = 0; i < c.owners; ++i) slots.push_back(i);
    if (auto x = runner->SelectSlots(slots); !x) return x;
    return node.WithRequest(0, runner->closure(), "serving recipe control", [&]() -> en::Status {
      for (const Planned& p : script) {
        auto& row = outputs.emplace_back();
        if (p.wave.head == Head::kDraft) continue;
        const std::size_t n = p.wave.units.size();
        std::vector<std::vector<float>> logits(n);
        std::vector<std::int32_t> tokens(n, -1);
        std::vector<en::Gemma4Runner::Work> work;
        for (std::size_t i = 0; i < n; ++i) {
          const bool greedy = p.wave.head == Head::kGreedy;
          work.push_back({p.wave.units[i].slot, p.past[i], p.tokens[i],
                          greedy ? nullptr : &logits[i], greedy ? &tokens[i] : nullptr});
        }
        if (p.wave.head == Head::kNone) {
          if (auto x = runner->WavePrefill(work, false); !x) return x;
        } else if (auto x = runner->Wave(work); !x) {
          return x;
        }
        for (std::size_t i = 0; i < n; ++i) {
          if (p.wave.head == Head::kLogits) {
            row.push_back(Bytes(std::span<const float>(logits[i])));
          } else if (p.wave.head == Head::kGreedy) {
            row.push_back(Bytes(std::span<const std::int32_t>(&tokens[i], 1)));
          }
        }
      }
      return {};
    });
  }();
  EXPECT_TRUE(started) << (started ? "" : started.error());
  const auto retired = node.TearDown(entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) std::ignore = runner.release();  // its memory may still be in use
  return outputs;
}

// Today's assistant over the script's first wave: its outputs (by owner:
// the logits, then the next feature, of each pass) and the target's
// retained final feature at each owner's endpoint.
struct Drafts {
  std::vector<std::vector<std::byte>> first_features;  // by owner
  std::vector<std::vector<std::int32_t>> tokens;       // by pass, by owner
  std::vector<std::vector<std::vector<std::byte>>> passes;
};

Drafts ProductionDrafts(const Case& c, const fs::path& target, const fs::path& assistant,
                        const fs::path& out, const Planned& prompt) {
  Drafts drafts;
  const md::Gemma4Profile& p = Profile(c.size);
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  en::Gemma4Options options = Options(c, target, out);
  options.retain_features = true;
  auto runner = std::make_unique<en::Gemma4Runner>(node, options, 0, 0);
  std::vector<en::PagedModel*> entered;
  en::Gemma4Assistant* drafter = nullptr;
  const auto started = [&]() -> en::Status {
    if (auto x = node.Open(); !x) return x;
    entered.push_back(runner.get());
    if (auto x = runner->Setup(); !x) return x;
    {
      constexpr auto admission_bytes = 512ULL << 20U;
      if (!node.ChargeHost(admission_bytes, false)) return en::support::Error("no host charge");
      struct Grant {
        en::PagedNode& node;
        ~Grant() { node.UnchargeHost(admission_bytes); }
      } grant{node};
      const auto ta = ar::Artifact::Open(target);
      const auto aa = ar::Artifact::Open(assistant);
      if (!ta || !aa) return en::support::Error("artifacts do not open");
      auto t = Decode(*ta, TargetVocabulary(c.size));
      auto a = Decode(*aa, AssistantVocabulary(c.size));
      if (!t || !a) return en::support::Error("vocabularies do not read");
      auto created = runner->SetupAssistant(assistant, t->view(), a->view());
      if (!created) return en::support::Error(created.error());
      drafter = *created;
    }
    if (auto x = node.MapWorkspace(runner->activations_needed(), runner->pool_needed()); !x)
      return x;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes() + (16ULL << 20U));
    if (auto x = node.Start(llmp::base::Bytes(fixed + runner->weights().size() * en::kPagedExtent +
                                              4 * node.StateCapacity()));
        !x)
      return x;
    if (auto x = runner->Register(); !x) return x;
    if (auto x = runner->Bind(); !x) return x;
    node.Run();
    const std::uint64_t host = std::uint64_t{c.owners} * 2 * (kVocab + p.width) * sizeof(float);
    if (!node.ChargeHost(host, false)) return en::support::Error("no host charge");
    struct Grant {
      en::PagedNode& node;
      std::uint64_t bytes;
      ~Grant() { node.UnchargeHost(bytes); }
    } grant{node, host};
    std::vector<std::uint32_t> slots;
    for (std::uint32_t i = 0; i < c.owners; ++i) slots.push_back(i);
    if (auto x = runner->SelectSlots(slots); !x) return x;
    return node.WithRequest(0, runner->closure(), "assistant control", [&]() -> en::Status {
      const std::size_t n = prompt.wave.units.size();
      std::vector<std::vector<float>> logits(n);
      std::vector<en::Gemma4Runner::Work> work;
      for (std::size_t i = 0; i < n; ++i) {
        work.push_back({prompt.wave.units[i].slot, prompt.past[i], prompt.tokens[i], &logits[i]});
      }
      if (auto x = runner->Wave(work); !x) return x;
      std::vector<std::int32_t> anchors;
      for (const auto& row : logits) {
        anchors.push_back(Top(std::as_bytes(std::span<const float>(row))));
      }
      // The retained feature at each endpoint.
      std::vector<llmp::catalog::ExtentId> staging;
      auto pinned = node.Pinned(p.width * sizeof(float), 0, staging);
      if (!pinned) return en::support::Error(pinned.error());
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t end = prompt.past[i] + prompt.wave.units[i].rows;
        if (auto x = runner->CopyFeatures(prompt.wave.units[i].slot, end - 1, 1, *pinned); !x)
          return x;
        const auto* f = static_cast<const std::byte*>(*pinned);
        drafts.first_features.emplace_back(f, f + (p.width * sizeof(float)));
      }
      if (auto freed = node.FreePinned(*pinned); !freed) return freed;
      std::vector<en::Gemma4Runner::FrozenBorrow> borrows;
      for (std::size_t i = 0; i < n; ++i) {
        auto borrow = runner->BorrowFrozen(prompt.wave.units[i].slot, anchors[i]);
        if (!borrow) return en::support::Error(borrow.error());
        borrows.push_back(std::move(*borrow));
      }
      std::vector<const en::Gemma4Runner::FrozenBorrow*> owners;
      for (const auto& b : borrows) owners.push_back(&b);
      std::vector<std::vector<float>> heads(n), features(n);
      std::vector<std::vector<float>*> head_out, feature_out;
      for (std::size_t i = 0; i < n; ++i) {
        heads[i].reserve(kVocab);
        features[i].reserve(p.width);
        head_out.push_back(&heads[i]);
        feature_out.push_back(&features[i]);
      }
      std::vector<std::int32_t> tokens = anchors;
      for (std::uint32_t pass = 0; pass < 2; ++pass) {
        drafts.tokens.push_back(tokens);
        if (auto x = drafter->Step(owners, tokens, pass == 0, head_out, feature_out); !x) return x;
        auto& outputs = drafts.passes.emplace_back();
        for (std::size_t i = 0; i < n; ++i) {
          std::vector<std::byte> both = Bytes(std::span<const float>(heads[i]));
          const auto f = Bytes(std::span<const float>(features[i]));
          both.insert(both.end(), f.begin(), f.end());
          outputs.push_back(std::move(both));
          tokens[i] = Top(std::as_bytes(std::span<const float>(heads[i])));
        }
      }
      return {};
    });
  }();
  EXPECT_TRUE(started) << (started ? "" : started.error());
  const auto retired = node.TearDown(entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) std::ignore = runner.release();
  return drafts;
}

// ------------------------------------------------------------ the components

// Each stage's step source, by the shape's stage.
class Stages final : public en::StepSource {
 public:
  explicit Stages(std::vector<std::unique_ptr<en::ComponentStepSource>> sources)
      : sources_(std::move(sources)) {}
  pl::PlanKey KeyBase() const override { return sources_.front()->KeyBase(); }
  std::expected<std::unique_ptr<kn::StepProgram>, std::string> Build(
      const pl::StepShape& shape, const kn::Addresses& addresses) override {
    if (shape.stage >= sources_.size()) return std::unexpected("no source of the stage");
    return sources_[shape.stage]->Build(shape, addresses);
  }

 private:
  std::vector<std::unique_ptr<en::ComponentStepSource>> sources_;
};

struct Component {
  Outputs outputs;
  std::uint64_t replayed = 0;
};

// Gemma4Runner::ReserveWeights' slabs and pitches: each layer's expert
// arrays in one slab whose pitch keeps 256-byte bases and whole blocks.
struct Slabs {
  std::vector<en::GroupPlace> places;
  std::vector<en::SlabSpec> slabs;
  std::map<std::uint32_t, std::uint64_t> pitch;  // by expert array
};
std::expected<Slabs, std::string> SlabsOf(const ar::Artifact& a, const md::Gemma4Profile& p) {
  Slabs out;
  out.places.assign(a.groups().size(), en::GroupPlace::kDevice);
  std::vector<bool> covered(a.groups().size(), false);
  const auto binding = md::BindGemma4(p, a);
  if (!binding) return std::unexpected(binding.error());
  for (std::uint32_t layer = 0; layer < p.layers; ++layer) {
    const auto& l = binding->layers[layer];
    std::vector<std::pair<std::uint32_t, std::string_view>> list;
    for (const auto* t : {&l.gate_up_exps, &l.gate_exps, &l.up_exps, &l.down_exps}) {
      if (*t) list.emplace_back((**t).index, (**t).type);
    }
    if (list.empty()) continue;
    auto slab = en::ExpertSlab(a, list, p.experts, 256, layer);
    if (!slab) return std::unexpected(slab.error());
    std::vector<std::string_view> types;
    for (const auto& [index, type] : list) types.push_back(type);
    auto pitch = en::Gemma4ExpertPitch(slab->stride, types);
    if (!pitch) return std::unexpected(pitch.error());
    slab->stride = *pitch;
    for (const auto& [index, type] : list) out.pitch[index] = *pitch;
    for (std::uint32_t e = 0; e < slab->count; ++e) covered[slab->first_group + e] = true;
    out.slabs.push_back(*slab);
  }
  for (std::size_t i = 0; i < out.places.size(); ++i) {
    if (a.groups()[i].kind != ar::GroupKind::kExpert) continue;
    if (!covered[i]) return std::unexpected("an expert group in no slab");
    out.places[i] = en::GroupPlace::kNone;
  }
  return out;
}

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

// The model runner over the component sources for the same script; the
// drafts' inputs from today's (its tokens and the first feature).
Component Lowered(const Case& c, const fs::path& target, const fs::path& assistant,
                  const fs::path& scratch, std::span<const Planned> script, const Drafts* drafts) {
  Component out;
  const md::Gemma4Profile& p = Profile(c.size);
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
  const auto& mapping = llmp::spec::mappings::Gemma4Mapping();
  const auto values = *llmp::spec::mappings::Gemma4V0Values(p);
  const auto target_spec = llmp::spec::SynthesizeV0(*ta, mapping, values);
  EXPECT_TRUE(target_spec.has_value()) << target_spec.error();
  if (!target_spec) return out;
  const auto ts = llmp::spec::SourceOfV0(*ta);
  EXPECT_TRUE(ts.has_value());
  if (!ts) return out;
  std::vector<llmp::spec::SourceValue> assistant_values;
  std::optional<llmp::spec::ModelSpec> composed;
  std::optional<llmp::spec::Source> as;
  if (c.assistant) {
    assistant_values = *llmp::spec::mappings::Gemma4AssistantV0Values(
        c.size == 26 ? md::Gemma4Assistant26() : md::Gemma4Assistant31(), p);
    auto spec = llmp::spec::ComposeDrafterV0(*target_spec, mapping, *aa,
                                             llmp::spec::mappings::Gemma4AssistantMapping(),
                                             assistant_values);
    EXPECT_TRUE(spec.has_value()) << spec.error();
    if (!spec) return out;
    composed.emplace(std::move(*spec));
    auto source = llmp::spec::SourceOfV0(*aa);
    EXPECT_TRUE(source.has_value());
    if (!source) return out;
    as.emplace(std::move(*source));
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
    auto step = co::BuildStep(draft ? *composed : *target_spec, shape.stage, shape,
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
  auto slabs = SlabsOf(*ta, p);
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
               if (at == place.end() || !at->second.first) return 0;
               const auto p = pitch.find(at->second.second);
               return p == pitch.end() ? 0 : p->second;
             }}});
    if (!made) return std::unexpected(made.error());
    source->backend = std::move(*made);
    auto target_weights = std::make_unique<en::PagedWeights>();
    if (auto x = target_weights->Open(target); !x) return x;
    en::PagedWeights* tw = target_weights.get();
    en::PagedWeights* aw = nullptr;
    std::vector<std::unique_ptr<en::ComponentStepSource>> sources;
    auto stage0 =
        en::ComponentStepSource::Create({.spec = &*target_spec,
                                         .stage = 0,
                                         .bindings = {.tensors = ts->tensors, .state = &layouts[0]},
                                         .bound_digest = target_spec->Digest(),
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
      auto drafter = en::ComponentStepSource::Create(
          {.spec = &*composed,
           .stage = 1,
           .bindings = {.tensors = as->tensors,
                        .state = &layouts[1],
                        .stages = stage_layouts,
                        .artifacts = artifacts},
           .bound_digest = composed->Digest(),
           .mapping = &llmp::spec::mappings::Gemma4AssistantMapping(),
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
    config.setup_shapes = shapes;
    config.launch = [&]() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return source->backend->Launch(
          {.base = node.pool().base, .size = llmp::base::Bytes(node.pool().bytes)});
    };
    config.weight_address = [&place, tw, aw](const llmp::ir::WeightRef& ref,
                                             llmp::format::PartRole) -> std::uint64_t {
      const auto at = place.find(ref);
      if (at == place.end()) return 0;
      const auto [array, index] = at->second;
      if (ref.stage == 1) return aw == nullptr ? 0 : aw->resource_address(index);
      return array ? tw->array_address(index) : tw->resource_address(index);
    };
    auto created = en::ModelRunner::Create(node, std::move(config),
                                           std::make_unique<Stages>(std::move(sources)), 0, 0);
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
      for (const Planned& planned : script) {
        auto& row = out.outputs.emplace_back();
        const pl::StepShape shape = ShapeOf(planned, c.context);
        std::vector<en::SegmentWork> work;
        std::vector<std::vector<std::int32_t>> tokens(planned.tokens);
        std::vector<std::byte> features;
        std::vector<en::StateBorrow> borrows;
        const bool draft = planned.wave.head == Head::kDraft;
        if (draft) {
          if (drafts == nullptr || pass >= drafts->passes.size())
            return en::support::Error("a draft without today's inputs");
          for (std::size_t i = 0; i < planned.wave.units.size(); ++i) {
            tokens[i] = {drafts->tokens[pass][i]};
            // The first pass reads the target's retained feature, a later
            // one its own last (today's, equal where the pass was).
            const auto& f =
                pass == 0 ? drafts->first_features[i]
                          : std::vector<std::byte>(drafts->passes[pass - 1][i].begin() + kVocab * 4,
                                                   drafts->passes[pass - 1][i].end());
            features.insert(features.end(), f.begin(), f.end());
            auto borrow = runner->slots().Borrow(planned.wave.units[i].slot, 0);
            if (!borrow) return en::support::Error(borrow.error());
            borrows.push_back(std::move(*borrow));
          }
        }
        for (const auto& ids : tokens) work.push_back({.tokens = ids, .bytes = {}});
        auto ran = runner->RunStep(
            {.shape = shape, .work = work, .step_bytes = features, .advance = !draft, .ahead = {}});
        if (!ran) return std::unexpected(ran.error().what);
        if (ran->path == en::RunPath::kReplayed) ++out.replayed;
        pass += draft ? 1 : 0;
        if (planned.wave.head == Head::kNone) continue;
        for (const auto& bytes : ran->outputs) row.push_back(bytes);
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

std::optional<fs::path> Installed(std::string_view id) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / kStore / id;
  if (std::error_code error; !fs::exists(root, error)) return std::nullopt;
  return root;
}

class ComponentRunnerGemma4Gpu : public testing::TestWithParam<Case> {};

TEST_P(ComponentRunnerGemma4Gpu, ServingRecipeStepsAreTodaysBitForBit) {
  const Case& c = GetParam();
  const auto target = Installed(TargetId(c.size));
  const auto assistant = Installed(AssistantId(c.size));
  if (!target || (c.assistant && !assistant)) GTEST_SKIP() << "no Gemma 4 artifacts installed";
  // NOLINTNEXTLINE(concurrency-mt-unsafe): the test environment is immutable
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch != nullptr ? scratch : testing::TempDir()) / "lowered4";
  fs::create_directories(out);
  const auto layout = md::Gemma4State(Profile(c.size), c.context, kMaxRows);
  ASSERT_TRUE(layout.has_value());
  const std::vector<Planned> script = Script(c, layout->local_cells);
  const Outputs production = Production(c, *target, out, script);
  ASSERT_EQ(production.size(), script.size());
  std::optional<Drafts> drafts;
  if (c.assistant) {
    drafts = ProductionDrafts(c, *target, *assistant, out, script.front());
    ASSERT_EQ(drafts->passes.size(), 2U);
  }
  const Component lowered =
      Lowered(c, *target, assistant.value_or(fs::path{}), out, script, drafts ? &*drafts : nullptr);
  ASSERT_EQ(lowered.outputs.size(), production.size());
  EXPECT_GE(lowered.replayed, 2U);
  std::size_t compared = 0;
  std::size_t pass = 0;
  for (std::size_t w = 0; w < script.size(); ++w) {
    const bool draft = script[w].wave.head == Head::kDraft;
    const auto& want_row = draft ? drafts->passes.at(pass++) : production[w];
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
  std::printf("%s: %zu waves, %zu owner outputs compared, %llu replayed\n", c.name.c_str(),
              script.size(), compared, static_cast<unsigned long long>(lowered.replayed));
}

INSTANTIATE_TEST_SUITE_P(
    Gemma4, ComponentRunnerGemma4Gpu,
    testing::Values(Case{.name = "gemma26_bounded", .size = 26, .assistant = true},
                    Case{.name = "gemma31_bounded", .size = 31, .assistant = true},
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
