// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Gemma 4 components end to end on the real artifacts (label
// `models`): the specs synthesized from the installed v0 artifacts
// (SynthesizeV0, the assistant composed with its target by
// ComposeDrafterV0) and validated against their tensors, built into steps
// (components/step.h BuildStep) and run by the reference executor (double
// arithmetic over the decoded weights, F16 KV state), against today's
// production runners on the GPU (engine/gemma4_runner.h, gemma4_assistant.h)
// for the same prompt and the tokens they pick:
//
//   - 26B-A4B: a two-token prompt and one decode on the target; between
//     them, from the prompt's state, two of the assistant's draft passes
//     (its first feature the target's final-norm tap for the prompt's last
//     position, which the target's reference step writes into the
//     assistant's tap ring; its second its own last feature);
//   - 31B: the two-token prompt.
//
// The production path quantizes activations for its quantized products,
// rounds attention's operands to F16 and fuses norms, so the logits agree
// to a tolerance, not bit for bit: the same top token at every step (or two
// tokens within kTieMargin of each other on both sides) and every logit
// within its tolerance of the reference's; the assistant's drafts are its
// top tokens, so they agree the same way. Each case prints its measured
// differences.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
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
#include "engine/gemma4_assistant.h"
#include "engine/gemma4_runner.h"
#include "engine/paged_node.h"
#include "engine/support.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/reference/exec.h"
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
namespace md = llmp::model;
namespace ref = llmp::kernels::reference;
namespace tk = llmp::tokenizer;
namespace fs = std::filesystem;
namespace pl = llmp::planner;

// The largest logit difference allowed between a production runner and
// the reference (the targets' logits are capped at 30, the assistant's are
// not), set over the differences measured on spark-b (2026-10-09; each
// case prints its report) with headroom, as components_models_gpu_test's.
// The dense 31B's prompt: 2.11 at most, 0.29 on average. The routed
// 26B-A4B's prompt 6.20 and 1.91, its decode 2.97 and 0.73. Where the
// 26B's differ (review, 2026-10-09, spark-b, the reference run again with
// each row's expert ids forced to the production's): routing explains the
// decode's excess (forced, 2.31 and 0.33: the production chose otherwise
// at near ties, probabilities 1.002 to 1.011 apart, each case printing its
// closest), not the prompt's (forced, 6.24 and 1.95). The prompt's is the
// production's arithmetic (quantized activations, F16 operands) gathered
// over the layers (0.3 to 2.5% of layer 0's output, the 31B's 0.25 to
// 0.37%; 6% by the last layer's attention), then grown by that
// layer's cancellation in the frontier row (its output's norm 38 from an
// input of 92: 13% of it). The assistant's two passes: 3.17 and 2.94 at
// most, 1.00 and 0.82 on average. Every top token, and so every draft,
// agrees.
constexpr double kDenseTolerance = 3.0;
constexpr double kRoutedTolerance = 8.0;
constexpr double kDraftTolerance = 8.0;
// Two tokens whose logits are this close on both sides are a tie the two
// arithmetics may break differently.
constexpr double kTieMargin = 0.25;
constexpr std::array<std::int32_t, 2> kPrompt{2, 818};
constexpr std::uint32_t kContext = 512;
constexpr std::uint32_t kRows = 64;
constexpr std::uint32_t kVocab = 262144;

const fs::path kStore = fs::path("m3-artifacts");
constexpr std::string_view k26 = "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3";
constexpr std::string_view k31 = "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08";
constexpr std::string_view kAssistant26 =
    "1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42";

std::int32_t Top(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

// ---------------------------------------------------------------- production

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

// What today's runners give: the prompt's frontier logits, the
// assistant's two draft passes' logits (26B), the decode's logits.
struct Production {
  std::vector<float> prompt, decode;
  std::vector<std::vector<float>> drafts;
  std::vector<std::int32_t> proposed;  // the draft passes' tokens
};

Production RunProduction(const fs::path& target, const fs::path& assistant, bool is31) {
  Production out;
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  // NOLINTNEXTLINE(concurrency-mt-unsafe): the test environment is immutable
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path root = scratch != nullptr ? scratch : testing::TempDir();
  auto runner = std::make_unique<en::Gemma4Runner>(
      node,
      en::Gemma4Options{.artifact = target,
                        .out = root / "components-gemma4",
                        .variant = is31 ? en::Gemma4Variant::k31B : en::Gemma4Variant::k26BA4B,
                        .max_rows = 128,
                        .slots = 1,
                        .frontier_head = false,
                        .retain_features = !is31},
      0, 0);
  std::vector<en::PagedModel*> entered;
  en::Gemma4Assistant* drafter = nullptr;
  const auto started = [&]() -> en::Status {
    if (auto x = node.Open(); !x) return x;
    entered.push_back(runner.get());
    if (auto x = runner->Setup(); !x) return x;
    if (!is31) {
      constexpr auto admission_bytes = 512ULL << 20U;
      if (!node.ChargeHost(admission_bytes, false)) return en::support::Error("no host charge");
      struct Grant {
        en::PagedNode& node;
        ~Grant() { node.UnchargeHost(admission_bytes); }
      } grant{node};
      const auto ta = ar::Artifact::Open(target);
      const auto aa = ar::Artifact::Open(assistant);
      if (!ta || !aa) return en::support::Error("artifacts do not open");
      auto t = Decode(*ta, "gemma-4-26B-A4B-it-UD-Q4_K_M.kv.gguf");
      auto a = Decode(*aa, "mtp-gemma-4-26B-A4B-it.kv.gguf");
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
    const auto bytes = std::uint64_t{4} * (kVocab + 5376U) * sizeof(float);
    if (!node.ChargeHost(bytes, false)) return en::support::Error("no host charge");
    struct Grant {
      en::PagedNode& node;
      std::uint64_t bytes;
      ~Grant() { node.UnchargeHost(bytes); }
    } grant{node, bytes};
    if (auto x = runner->SelectSlots(std::array<std::uint32_t, 1>{0}); !x) return x;
    return node.WithRequest(0, runner->closure(), "components control", [&]() -> en::Status {
      const en::Gemma4Runner::Work prefill{0, 0, kPrompt, &out.prompt};
      if (auto x = runner->Wave(std::span(&prefill, 1)); !x) return x;
      const std::int32_t anchor = Top(out.prompt);
      if (drafter != nullptr) {
        auto borrow = runner->BorrowFrozen(0, anchor);
        if (!borrow) return en::support::Error(borrow.error());
        const std::array<const en::Gemma4Runner::FrozenBorrow*, 1> owners{&*borrow};
        std::vector<float> heads;
        std::vector<float> features;
        heads.reserve(kVocab);
        features.reserve(5376);
        const std::array<std::vector<float>*, 1> head_out{&heads};
        const std::array<std::vector<float>*, 1> feature_out{&features};
        std::int32_t token = anchor;
        for (std::uint32_t pass = 0; pass < 2; ++pass) {
          const std::array<std::int32_t, 1> anchors{token};
          if (auto x = drafter->Step(owners, anchors, pass == 0, head_out, feature_out); !x)
            return x;
          out.drafts.push_back(heads);
          token = Top(heads);
          out.proposed.push_back(token);
        }
      }  // the borrow ends before the target writes again
      const std::array<std::int32_t, 1> next{anchor};
      const en::Gemma4Runner::Work decode{0, static_cast<std::uint32_t>(kPrompt.size()), next,
                                          &out.decode};
      return runner->Wave(std::span(&decode, 1));
    });
  }();
  EXPECT_TRUE(started) << (started ? "" : started.error());
  const auto retired = node.TearDown(entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) std::ignore = runner.release();  // its memory may still be in use
  return out;
}

// ----------------------------------------------------------------- reference

// One weight's bytes: a resource, or an expert array's slices one after
// the other (the IR's [K, N, experts]).
std::vector<std::byte> ReadWeight(const fs::path& root, const ar::Artifact& a,
                                  const std::string& name) {
  const auto read = [&](const ar::Placement& p, std::vector<std::byte>& out) {
    const ar::Group& g = a.groups()[p.group];
    const ar::Shard& shard = a.shards()[g.shard];
    std::ifstream in(root / shard.path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(shard.data_offset.value() + g.offset.value() +
                                         p.offset.value()));
    const std::size_t at = out.size();
    out.resize(at + p.bytes.value());
    in.read(reinterpret_cast<char*>(out.data() + at),
            static_cast<std::streamsize>(p.bytes.value()));
    EXPECT_TRUE(in.good()) << name;
  };
  std::vector<std::byte> out;
  if (const auto r = a.FindResource(name); r) {
    read(a.ResourcePlacement(*r).value(), out);
    return out;
  }
  const auto arrays = a.expert_arrays();
  const auto at = std::ranges::find(arrays, name, &ar::ExpertArray::name);
  EXPECT_NE(at, arrays.end()) << name;
  if (at == arrays.end()) return out;
  const auto index = static_cast<std::uint32_t>(at - arrays.begin());
  for (std::uint32_t e = 0; e < at->count; ++e) read(a.SlicePlacement(index, e).value(), out);
  return out;
}

struct Model {
  fs::path root;
  ar::Artifact* artifact;
};

// The reference's runs over one composed spec: its weights by (stage,
// name), each stage's state by (stage, layer, tensor) for slot 0.
class Reference {
 public:
  Reference(const llmp::spec::ModelSpec& spec, std::vector<Model> models,
            std::vector<llmp::spec::Source> sources)
      : spec_(spec), models_(std::move(models)), sources_(std::move(sources)) {
    for (const auto& s : sources_) tables_.emplace_back(s.tensors);
    for (std::size_t i = 0; i < spec_.stages.size(); ++i) {
      auto layout = co::StageLayout(spec_, i, kContext, kRows);
      EXPECT_TRUE(layout.has_value()) << (layout ? "" : layout.error());
      layouts_.push_back(std::move(*layout));
    }
    for (const auto& l : layouts_) stage_layouts_.push_back(&l);
  }

  const llmp::model::StateLayoutDerived& layout(std::size_t stage) const {
    return layouts_.at(stage);
  }
  // A drafter's tap cell for `position`, F32.
  std::vector<float> TapFeature(std::uint16_t stage, std::uint32_t position) const {
    const auto& item = layouts_.at(stage).items.at(0);
    const auto& bytes = state_.at({stage, item.layer, static_cast<std::uint16_t>(item.tensor)});
    const std::uint64_t width = item.contract.cell_bytes / 4;
    const std::uint64_t cell = position % item.cells;
    std::vector<float> out(width);
    std::memcpy(out.data(), bytes.data() + (cell * item.contract.cell_bytes),
                item.contract.cell_bytes);
    return out;
  }

  // One step: its logits (and a drafter head's next feature).
  std::pair<std::vector<float>, std::vector<float>> Run(const pl::StepShape& shape,
                                                        std::span<const std::int32_t> tokens,
                                                        std::span<const float> features) {
    const auto stage = shape.stage;
    const llmp::spec::Stage& s = spec_.stages.at(stage);
    auto graph = co::BuildStep(spec_, stage, shape,
                               {.tensors = tables_.at(s.artifact),
                                .state = &layouts_.at(stage),
                                .stages = stage_layouts_,
                                .artifacts = tables_});
    EXPECT_TRUE(graph.has_value()) << (graph ? "" : graph.error());
    if (!graph) return {};
    auto exec = ref::Executor::Create(graph->graph);
    EXPECT_TRUE(exec.has_value()) << (exec ? "" : exec.error().detail);
    if (!exec) return {};
    for (const co::StepWeight& w : graph->weights) {
      const std::string& name = w.parts.front().first;
      auto& bytes = weights_[{w.ref.stage, name}];
      if (bytes.empty()) {
        const Model& m = models_.at(spec_.stages.at(w.ref.stage).artifact);
        bytes = ReadWeight(m.root, *m.artifact, name);
      }
      const ref::PartView part{.role = w.parts.front().second, .bytes = bytes};
      const auto bound = exec->Bind(w.tensor, std::span(&part, 1));
      EXPECT_TRUE(bound.has_value()) << name << ": " << (bound ? "" : bound.error().detail);
    }
    for (const co::StepState& st : graph->state) {
      // A borrowed tensor is the target's item it reads.
      Key key{st.ref.stage, st.ref.layer, st.ref.tensor};
      if (st.borrowed) {
        const auto& item = layouts_.at(0).items.at(st.item);
        key = {0, item.layer, static_cast<std::uint16_t>(item.tensor)};
      }
      auto& bytes = state_[key];
      if (bytes.empty()) {
        const std::vector<float> zeros(llmp::ir::Elements(graph->graph.tensor(st.tensor).shape()),
                                       0.0F);
        EXPECT_TRUE(exec->BindValues(st.tensor, zeros));
      } else {
        const ref::PartView part{.role = llmp::format::PartRole::kData, .bytes = bytes};
        EXPECT_TRUE(exec->Bind(st.tensor, std::span(&part, 1)));
      }
    }
    for (const co::StepInput& input : graph->inputs) {
      if (input.kind == co::InputKind::kFeatures) {
        EXPECT_TRUE(exec->BindValues(input.tensor, features));
        continue;
      }
      const auto values = co::InputValues(input, shape, tokens);
      EXPECT_TRUE(values.has_value()) << (values ? "" : values.error());
      EXPECT_TRUE(exec->BindInts(input.tensor, *values));
    }
    const auto ran = exec->Run();
    EXPECT_TRUE(ran.has_value()) << (ran ? "" : ran.error().detail);
    if (!ran) return {};
    // How close each routed selection came to another (the probability of
    // its last chosen expert over the first not chosen): a production whose
    // arithmetic differs may choose otherwise where it is near 1.
    double closest = INFINITY;
    std::size_t selections = 0;
    for (const llmp::ir::Node& n : graph->graph.nodes()) {
      if (n.key.op != llmp::ir::Op::kRoute) continue;
      const auto p = n.Params<llmp::ir::RouteParams>();
      const auto logits = exec->Values(n.src[0]);
      if (!logits) continue;
      for (std::size_t t = 0; t + p.experts <= logits->size(); t += p.experts) {
        std::vector<float> row(logits->begin() + static_cast<std::ptrdiff_t>(t),
                               logits->begin() + static_cast<std::ptrdiff_t>(t + p.experts));
        std::ranges::sort(row, std::greater<>());
        closest = std::min(closest, std::exp(static_cast<double>(row[p.used - 1] - row[p.used])));
        ++selections;
      }
    }
    if (selections != 0) {
      std::printf("routing: %zu selections, the closest a ratio of %.6f\n", selections, closest);
    }
    std::pair<std::vector<float>, std::vector<float>> out;
    out.first = *exec->Values(graph->logits);
    if (graph->features.valid()) out.second = *exec->Values(graph->features);
    for (const co::StepState& st : graph->state) {
      if (st.borrowed) continue;  // read only
      const auto bytes = exec->Bytes(st.tensor);
      state_[{st.ref.stage, st.ref.layer, st.ref.tensor}].assign(bytes.begin(), bytes.end());
    }
    return out;
  }

 private:
  using Key = std::tuple<std::uint16_t, std::uint32_t, std::uint16_t>;
  const llmp::spec::ModelSpec& spec_;
  std::vector<Model> models_;
  std::vector<llmp::spec::Source> sources_;
  std::vector<std::span<const co::BoundTensor>> tables_;
  std::vector<llmp::model::StateLayoutDerived> layouts_;
  std::vector<const llmp::model::StateLayoutDerived*> stage_layouts_;
  std::map<std::pair<std::uint16_t, std::string>, std::vector<std::byte>> weights_;
  std::map<Key, std::vector<std::byte>> state_;
};

// The production's row against the reference's: the same top token (or a
// near tie) and every logit within `tolerance`; prints the differences.
void ExpectClose(std::string_view what, std::span<const float> reference,
                 std::span<const float> production, double tolerance) {
  ASSERT_EQ(reference.size(), production.size()) << what;
  double worst = 0.0;
  double sum = 0.0;
  for (std::size_t j = 0; j < reference.size(); ++j) {
    const double d = std::fabs(static_cast<double>(reference[j]) - production[j]);
    worst = std::max(worst, d);
    sum += d;
  }
  const auto top = static_cast<std::size_t>(Top(reference));
  const auto chosen = static_cast<std::size_t>(Top(production));
  const double reference_margin = static_cast<double>(reference[top] - reference[chosen]);
  const double production_margin = static_cast<double>(production[chosen] - production[top]);
  std::printf(
      "%.*s: top %zu (production %zu, margins %.4f and %.4f), max |logit difference| %.4f, "
      "mean %.5f\n",
      static_cast<int>(what.size()), what.data(), top, chosen, reference_margin, production_margin,
      worst, sum / static_cast<double>(reference.size()));
  EXPECT_TRUE(top == chosen || (reference_margin <= kTieMargin && production_margin <= kTieMargin))
      << what;
  EXPECT_LE(worst, tolerance) << what;
}

std::optional<fs::path> Installed(std::string_view id) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / kStore / id;
  if (std::error_code error; !fs::exists(root, error)) return std::nullopt;
  return root;
}

TEST(ComponentsGemma4ModelsGpu, Gemma26AndItsAssistantMatchTheProductionRunners) {
  const auto target = Installed(k26);
  const auto assistant = Installed(kAssistant26);
  if (!target || !assistant) GTEST_SKIP() << "no Gemma 4 26B target and assistant installed";
  const Production production = RunProduction(*target, *assistant, false);
  ASSERT_EQ(production.drafts.size(), 2U);
  ASSERT_FALSE(production.decode.empty());
  auto ta = ar::Artifact::Open(*target, {.expected_id = {}});
  auto aa = ar::Artifact::Open(*assistant, {.expected_id = {}});
  ASSERT_TRUE(ta.has_value() && aa.has_value());
  const auto& p = md::Gemma4_26BA4B();
  const auto& mapping = llmp::spec::mappings::Gemma4Mapping();
  ASSERT_EQ(llmp::spec::FindMapping(ta->model().architecture), &mapping);
  const auto target_spec =
      llmp::spec::SynthesizeV0(*ta, mapping, *llmp::spec::mappings::Gemma4V0Values(p));
  ASSERT_TRUE(target_spec.has_value()) << target_spec.error();
  const auto spec = llmp::spec::ComposeDrafterV0(
      *target_spec, mapping, *aa, llmp::spec::mappings::Gemma4AssistantMapping(),
      *llmp::spec::mappings::Gemma4AssistantV0Values(md::Gemma4Assistant26(), p));
  ASSERT_TRUE(spec.has_value()) << spec.error();
  auto ts = llmp::spec::SourceOfV0(*ta);
  auto as = llmp::spec::SourceOfV0(*aa);
  ASSERT_TRUE(ts.has_value() && as.has_value());
  const std::array<std::span<const co::BoundTensor>, 2> artifacts{ts->tensors, as->tensors};
  const auto valid = co::ValidateSpec(*spec, artifacts);
  ASSERT_TRUE(valid.has_value()) << valid.error();
  Reference reference(*spec, {{*target, &*ta}, {*assistant, &*aa}}, {*ts, *as});
  const auto past = static_cast<std::uint32_t>(kPrompt.size());
  // The prompt, its tap written into the assistant's ring.
  const auto prompt = reference.Run(
      {.stage = 0,
       .phase = pl::Phase::kPrefill,
       .segments = {{.slot = 0, .rows = past, .past = 0, .read_cells = 256, .outputs = 1}}},
      kPrompt, {});
  ExpectClose("gemma26 prompt", prompt.first, production.prompt, kRoutedTolerance);
  // Two draft passes at the target's committed endpoint, from the
  // production's anchor and proposal.
  const std::int32_t anchor = Top(production.prompt);
  std::vector<float> feature = reference.TapFeature(1, past - 1);
  const pl::StepShape draft{
      .stage = 1,
      .phase = pl::Phase::kDraft,
      .segments = {{.slot = 0, .rows = 1, .past = past, .read_cells = 256, .outputs = 1}}};
  std::int32_t token = anchor;
  for (std::uint32_t pass = 0; pass < 2; ++pass) {
    const std::array<std::int32_t, 1> tokens{token};
    auto drafted = reference.Run(draft, tokens, feature);
    ASSERT_EQ(drafted.second.size(), p.width);
    ExpectClose(std::format("gemma26 assistant pass {}", pass), drafted.first,
                production.drafts[pass], kDraftTolerance);
    feature = std::move(drafted.second);
    token = production.proposed[pass];
  }
  // The decode of the anchor.
  const std::array<std::int32_t, 1> next{anchor};
  const auto decode = reference.Run(
      {.stage = 0,
       .phase = pl::Phase::kDecode,
       .segments = {{.slot = 0, .rows = 1, .past = past, .read_cells = 256, .outputs = 1}}},
      next, {});
  ExpectClose("gemma26 decode", decode.first, production.decode, kRoutedTolerance);
}

TEST(ComponentsGemma4ModelsGpu, Gemma31MatchesTheProductionRunner) {
  const auto target = Installed(k31);
  if (!target) GTEST_SKIP() << "no Gemma 4 31B installed";
  const Production production = RunProduction(*target, {}, true);
  ASSERT_FALSE(production.prompt.empty());
  auto ta = ar::Artifact::Open(*target, {.expected_id = {}});
  ASSERT_TRUE(ta.has_value());
  const auto spec =
      llmp::spec::SynthesizeV0(*ta, llmp::spec::mappings::Gemma4Mapping(),
                               *llmp::spec::mappings::Gemma4V0Values(md::Gemma4_31B()));
  ASSERT_TRUE(spec.has_value()) << spec.error();
  auto ts = llmp::spec::SourceOfV0(*ta);
  ASSERT_TRUE(ts.has_value());
  const auto valid = co::ValidateSpec(*spec, ts->tensors);
  ASSERT_TRUE(valid.has_value()) << valid.error();
  Reference reference(*spec, {{*target, &*ta}}, {*ts});
  const auto prompt =
      reference.Run({.stage = 0,
                     .phase = pl::Phase::kPrefill,
                     .segments = {{.slot = 0,
                                   .rows = static_cast<std::uint32_t>(kPrompt.size()),
                                   .past = 0,
                                   .read_cells = 256,
                                   .outputs = 1}}},
                    kPrompt, {});
  ExpectClose("gemma31 prompt", prompt.first, production.prompt, kDenseTolerance);
}

}  // namespace
