// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4 Flash served by the shared engine (D-107; runtime/engine_llm.h
// over the component step source) on the installed artifacts (label
// `models`), through the runtime's server as the service drives it: the
// coverage that the retired Dsv4Runner's serving tests held
// (plain_tokens_serving_gpu_test, dsv4_prefill_plan_test's lookahead cases)
// on the one engine every model now shares.
//
// - the plain-token oracle (plain_token_serving_checks.h): device tokens
//   and logit rows agree, alone and joined, mixed and through an idle
//   spill, target alone and with DSpark;
// - a logit row is the model's width, and a continuation's rows complete;
// - DSpark: after a joined plain wave of two requests whose prompts pass
//   the 256-cell ring, one of them leaves, and the other's draft/verify
//   consumes the features the token-only waves injected, exactly as after
//   waves that returned rows (complete state, tokens, drafts, heads);
//   joined draft/verify waves return the same with rows and with tokens;
// - four slots' speculative waves at CSA and HCA block alignments with
//   mixed verify rows, inside the startup graph and metadata allowance;
// - prefill lookahead on and off: the same heads and states at prompts
//   crossing CSA and HCA blocks and the raw ring;
// - the fixed prompt's head and greedy tokens against what the engine
//   gave on the M3.6 reference run.
//
// DeepSeek's serving admission is the paid diagnostic tuple (the service's
// LLMP_DIAGNOSTIC_COMPONENT_ENGINE, _STARTUP_HOST_BYTES=auto and
// _NODE_CAP_BYTES, applied as the service applies them); without it the
// build refuses. Every test needs roughly 116 GiB of headroom on a quiet
// host. A missing artifact skips with its reason.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <vector>

#include "base/sha256.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "plain_token_serving_checks.h"
#include "runtime/commands.h"
#include "runtime/engine_llm.h"
#include "runtime/serving.h"
#include "tokenizer_fixtures.h"

namespace {

namespace rt = llmp::runtime;
namespace en = llmp::engine;
namespace cfg = llmp::config;
namespace fs = std::filesystem;

constexpr const char* kTarget = "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234";
constexpr const char* kDrafter = "dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5";
constexpr std::uint32_t kVocab = 129280;
// The paid diagnostic envelope the M3.6 gates ran under: the node's cap
// (112 GiB, 2 MiB aligned) and the automatic startup host payment.
constexpr const char* kNodeCap = "120259084288";

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool Finite(const std::vector<float>& row, std::size_t width = kVocab) {
  return row.size() == width && std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
}

std::int32_t Argmax(const std::vector<float>& row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

// The prompt of `owner`, `rows` tokens long: the serving fixtures' seed
// tokens, shifted a step by owner and (slowly) by position.
std::vector<std::int32_t> Tokens(std::uint32_t owner, std::size_t rows, std::size_t salt = 0) {
  constexpr std::array<std::int32_t, 7> seed{2, 818, 5279, 529, 7001, 563, 42};
  std::vector<std::int32_t> out(rows);
  for (std::size_t i = 0; i < rows; ++i)
    out[i] = seed[(i + (owner * 3) + (i / 7) + salt) % seed.size()];
  return out;
}

// A host observation's funding: the state hashes copy through pinned host
// memory the node charges beside its catalog.
class HostFunding {
 public:
  explicit HostFunding(en::PagedNode& node)
      : node_(node), funded_(node.ChargeHost(kBytes, false)) {}
  HostFunding(const HostFunding&) = delete;
  HostFunding& operator=(const HostFunding&) = delete;
  ~HostFunding() {
    if (funded_) node_.UnchargeHost(kBytes);
  }
  bool funded() const { return funded_; }

 private:
  static constexpr std::uint64_t kBytes = 64ULL << 20U;
  en::PagedNode& node_;
  bool funded_;
};

// Sessions close even when a later check fails.
template <class Session, std::size_t N>
struct SessionSet {
  std::array<std::unique_ptr<Session>, N> owned;
  SessionSet() = default;
  SessionSet(const SessionSet&) = delete;
  SessionSet& operator=(const SessionSet&) = delete;
  ~SessionSet() { Close(); }
  void Close() {
    for (auto& s : owned) {
      if (!s) continue;
      if (!s->done()) s->Cancel();
      const auto r = s->Finish();
      EXPECT_TRUE(r) << (r ? "" : r.error());
      s.reset();
    }
  }
};

class Dsv4EngineBase : public testing::Test {
 protected:
  // One server's shape: the models-tests' ordinary DeepSeek settings.
  struct Shape {
    bool drafter = false;
    std::string wave_form;  // "plain", "speculative"; empty: the default
    std::uint32_t context = 8704;
    std::uint32_t chunk = 4096;
    std::uint32_t slots = 2;
    bool lookahead = true;
  };
  struct Lifetime {
    cfg::NodeConfig config;
    cfg::RuntimeRoles roles;
    rt::ServingOptions options;
    std::unique_ptr<rt::Server> server;
  };
  std::unique_ptr<Lifetime> life = std::make_unique<Lifetime>();
  rt::Llm* model = nullptr;
  fs::path scratch;
  std::FILE* log = nullptr;
  bool retirement_failed = false;
  bool needs_drafter = false;

  static fs::path Installed() { return fs::path(llmp::test_support::ModelsDir()) / "m3-artifacts"; }

  void SetUp() override {
    const fs::path installed = Installed();
    std::error_code error;
    if (!fs::exists(installed / kTarget, error))
      GTEST_SKIP() << "no DeepSeek V4 Flash artifact " << kTarget << " in " << installed;
    if (needs_drafter && !fs::exists(installed / kDrafter, error))
      GTEST_SKIP() << "no DSpark drafter artifact " << kDrafter << " in " << installed;
    // Spill files need direct I/O: scratch in the build tree.
    const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const fs::path root = base != nullptr ? base : ::testing::TempDir();
    fs::create_directories(root, error);
    std::string name = (root / "dsv4-engine-XXXXXX").string();
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    scratch = name;
    life->roles.installed = installed;
    life->roles.spill = scratch / "spill";
    life->roles.state = scratch / "state";
    for (const auto& path : {life->roles.spill, life->roles.state}) {
      ASSERT_TRUE(fs::create_directory(path));
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
    log = std::tmpfile();
    ASSERT_NE(log, nullptr);
  }

  // A server with DeepSeek on the engine and its first `owners` request
  // slots selected.
  rt::Status Start(const Shape& shape, std::size_t owners = 2) {
    life->config.models.clear();
    cfg::ModelEntry entry;
    entry.name = "deepseek";
    entry.artifact = kTarget;
    if (shape.drafter) entry.drafter = kDrafter;
    if (!shape.wave_form.empty()) entry.overrides["wave_form"] = shape.wave_form;
    entry.overrides["context"] = std::int64_t{shape.context};
    entry.overrides["prefill_chunk"] = std::int64_t{shape.chunk};
    entry.overrides["max_slots"] = std::int64_t{shape.slots};
    life->config.models.push_back(entry);
    life->options = {};
    // The paid diagnostic tuple, as the service applies its environment.
    if (auto r = rt::ApplyServiceDiagnostics(life->options, "all", "auto", kNodeCap, std::nullopt);
        !r)
      return std::unexpected(r.error());
    life->options.dsv4_prefill_lookahead = shape.lookahead;
    life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, log);
    if (auto r = life->server->Start(true); !r) return r;
    model = dynamic_cast<rt::Llm*>(life->server->Find("deepseek"));
    if (model == nullptr) return std::unexpected("configured DeepSeek model missing");
    if (dynamic_cast<rt::EngineLlm*>(model) == nullptr)
      return std::unexpected("DeepSeek is not on the component engine");
    rt::SwapParts parts;
    if (auto r = life->server->Activate(*model, parts); !r) return r;
    return Select(owners);
  }
  rt::Status Select(std::size_t count) {
    std::array<rt::Llm::Branch*, 4> branches{};
    for (std::size_t id = 0; id < count; ++id) {
      auto branch = model->branch(static_cast<std::uint32_t>(id));
      if (!branch) return std::unexpected(branch.error());
      branches[id] = *branch;
    }
    return life->server->SelectRequestBranches(*model, std::span(branches).first(count));
  }
  rt::Status Retire() {
    if (retirement_failed) return std::unexpected("prior server retirement unproven");
    if (!life->server) return {};
    const auto r = life->server->TearDown();
    retirement_failed = !r;
    if (r) {
      life->server.reset();
      model = nullptr;
    }
    return r;
  }
  void TearDown() override {
    const auto r = Retire();
    EXPECT_TRUE(r) << (r ? "" : r.error());
    if (!r) {
      std::ignore = life.release();
      return;
    }
    if (log != nullptr) {
      if (HasFailure()) {
        std::array<char, 4096> buffer{};
        (void)std::fflush(log);
        std::rewind(log);
        std::string text;
        for (std::size_t n = 0; (n = std::fread(buffer.data(), 1, buffer.size(), log)) > 0;)
          text.append(buffer.data(), n);
        std::println(stderr, "--- server log ---\n{}", text);
      }
      (void)std::fclose(log);
    }
    if (!scratch.empty()) fs::remove_all(scratch);
  }
  rt::Llm::Branch& Branch(std::uint32_t id) { return **model->branch(id); }
  rt::EngineLlm& Engine() { return dynamic_cast<rt::EngineLlm&>(*model); }

  // Every stage's used state of `slot`, hashed (the engine's slots).
  std::expected<llmp::base::Sha256Digest, std::string> StateHash(std::uint32_t slot) {
    en::SlotLifecycle& slots = Engine().runner().slots();
    auto& node = life->server->node();
    constexpr std::uint64_t kPart = 1ULL << 20U;
    std::vector<llmp::catalog::ExtentId> staging;
    auto buffer = node.Pinned(kPart, 0, staging);
    if (!buffer) return std::unexpected(buffer.error());
    llmp::base::Sha256 hash;
    std::uint64_t bytes = 0;
    for (std::size_t stage = 0; stage < slots.stages(); ++stage) {
      for (const en::LiveState::Range& range : slots.live(slot, stage).used_ranges()) {
        for (std::uint64_t at = 0; at < range.bytes; at += kPart) {
          const en::LiveState::Range part{.region = range.region,
                                          .offset = range.offset + at,
                                          .bytes = std::min(kPart, range.bytes - at)};
          en::LiveState::CopyRetirement retired = en::LiveState::CopyRetirement::kUnproven;
          const auto copied = slots.CopyState(slot, *buffer, std::span(&part, 1), true, &retired);
          if (retired == en::LiveState::CopyRetirement::kUnproven) node.KeepPinned(*buffer);
          if (!copied) {
            if (retired == en::LiveState::CopyRetirement::kProven) (void)node.FreePinned(*buffer);
            return std::unexpected(copied.error());
          }
          hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
          bytes += part.bytes;
        }
      }
    }
    if (auto r = node.FreePinned(*buffer); !r) return std::unexpected(r.error());
    if (bytes == 0) return std::unexpected("no state");
    return hash.Finish();
  }

  // Two owners' prompts in joined prompt waves (as the service's driver
  // runs them, the second owner first): each owner's head.
  rt::Status JoinedPrompts(std::array<std::uint32_t, 2> rows,
                           std::array<std::vector<float>, 2>& heads) {
    std::array<std::vector<std::int32_t>, 2> tokens{Tokens(0, rows[0]), Tokens(1, rows[1])};
    SessionSet<rt::Llm::PromptSession, 2> sessions;
    for (std::uint32_t id = 0; id < 2; ++id) {
      if (auto r = Branch(id).Clear(); !r) return r;
      auto begun = Branch(id).BeginPrompt(tokens[id]);
      if (!begun) return std::unexpected(begun.error());
      sessions.owned[id] = std::move(*begun);
      if (auto r = sessions.owned[id]->Advance(); !r) return r;
    }
    while (!sessions.owned[0]->done() || !sessions.owned[1]->done()) {
      std::array<rt::Llm::PromptSession*, 2> active{};
      std::size_t count = 0;
      for (const auto id : {1U, 0U}) {
        if (!sessions.owned[id]->done() && (count == 0 || active[0]->CanJoin(*sessions.owned[id])))
          active[count++] = sessions.owned[id].get();
      }
      const std::array<rt::PrefillGoOn, 2> callbacks{};
      if (auto r = model->RunPromptWave(std::span(active).first(count),
                                        std::span(callbacks).first(count));
          !r)
        return r;
      for (const auto* session : std::span(active).first(count))
        if (!session->last_unit_result()) return session->last_unit_result();
    }
    for (std::uint32_t id = 0; id < 2; ++id) {
      heads[id] = sessions.owned[id]->last();
      if (Branch(id).history() != tokens[id]) return std::unexpected("a joined history differs");
      if (auto r = sessions.owned[id]->Finish(); !r) return r;
      sessions.owned[id].reset();
    }
    return {};
  }
};

// The target alone: no drafter, so no test of the drafter's skips beside it.
class Dsv4EngineTarget : public Dsv4EngineBase {};

// The target with DSpark (a suite of its own: it skips without the drafter).
class Dsv4EngineDspark : public Dsv4EngineBase {
 protected:
  Dsv4EngineDspark() { needs_drafter = true; }
};

// What a generation held after its run: tokens, work and rows.
struct Observed {
  std::vector<std::int32_t> tokens;
  std::uint64_t drafted = 0, accepted = 0;
  std::vector<std::vector<float>> rows;
  std::vector<std::int32_t> history;
  llmp::base::Sha256Digest state{};
};

// ---------------------------------------------------------------- 1, 2

// The plain-token oracle on the engine: a lone owner, two joined, one
// leaving, a joined row or sampling peer, and an idle spill, device tokens
// against rows, complete state against complete state.
TEST_F(Dsv4EngineTarget, PlainTokensAndRowsAgreeOnTheEngine) {
  const auto started = Start({});
  ASSERT_TRUE(started) << (started ? "" : started.error());
  EXPECT_EQ(model->logit_width(), kVocab);
  const auto r = llmp::test::CheckPlainServing(
      *life->server, *model, kVocab, [&](std::uint32_t id) { return StateHash(id); },
      [&] { return Engine().counts().greedy_tokens; });
  ASSERT_TRUE(r) << (r ? "" : r.error());
  EXPECT_EQ(Engine().runner().coverage().violations, 0U);
}

// The same with DSpark configured and its joined waves forced plain (the
// token-only C2 waves): a lone owner speculates, so only the joined waves
// and a speculative step with one row left (plain, EngineLlm::
// RunPreparedGenerationWave) choose on the device. The lone owner's
// first step keeps two of its three rows on this prompt (greedy decoding is
// deterministic, so is the count): one device token; the joined pair's two
// waves of two and the owner left alone for its last row: five.
TEST_F(Dsv4EngineDspark, PlainTokensAndRowsAgreeWithDspark) {
  Shape shape;
  shape.drafter = true;
  shape.wave_form = "plain";
  const auto started = Start(shape);
  ASSERT_TRUE(started) << (started ? "" : started.error());
  EXPECT_EQ(model->logit_width(), kVocab);
  const auto r = llmp::test::CheckPlainServing(
      *life->server, *model, kVocab, [&](std::uint32_t id) { return StateHash(id); },
      [&] { return Engine().counts().greedy_tokens; }, {1, 5});
  ASSERT_TRUE(r) << (r ? "" : r.error());
  EXPECT_EQ(Engine().runner().coverage().violations, 0U);
}

// A logit row is the model's logit width (the swap table's continuation
// check, rt::CompleteContinuationRows), not the tokenizer's.
TEST_F(Dsv4EngineTarget, LogitRowsAreTheModelsWidthAndContinuationRowsComplete) {
  const auto started = Start({}, 1);
  ASSERT_TRUE(started) << (started ? "" : started.error());
  ASSERT_EQ(model->logit_width(), kVocab);
  HostFunding funding(life->server->node());
  ASSERT_TRUE(funding.funded());
  constexpr std::array<std::int32_t, 6> seed{2, 818, 5279, 529, 7001, 563};
  std::vector<float> last;
  std::array<rt::Generation, 2> generated;
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  options.keep_logits = true;
  for (auto& out : generated) {
    ASSERT_TRUE(Branch(0).Clear());
    ASSERT_TRUE(Branch(0).Prefill(seed, last));
    const auto r = Branch(0).Generate(last, options, out);
    ASSERT_TRUE(r) << (r ? "" : r.error());
  }
  EXPECT_TRUE(
      rt::CompleteContinuationRows(last, generated[0], generated[1], 3, model->logit_width()));
  // Two runs of one prompt give one answer.
  EXPECT_EQ(generated[0].tokens, generated[1].tokens);
}

// ---------------------------------------------------------------- 3

// One owner's generation after `first` tokens, run alone to its end: the
// ordinary scalar path (a lone request speculates).
rt::Status RunToEnd(rt::EngineLlm& llm, rt::Llm::GenerationSession& session) {
  for (unsigned guard = 0; !session.done(); ++guard) {
    if (guard > 64) return std::unexpected("a lone generation made no progress");
    std::array<rt::Llm::GenerationSession*, 1> one{&session};
    if (auto r = llm.RunGenerationWave(one); !r) return r;
  }
  return {};
}

// DSpark joined, then one request leaves. Two requests whose prompts pass
// the drafter's 256-cell ring decode three joined waves, once keeping rows
// and once choosing tokens on the device; their complete states (the
// target's and the drafter's) must match. The waves' features (the taps of
// a token-only wave are injected into the drafter) are then consumed by
// each owner's own draft/verify step: its tokens, drafts, heads and settled
// state must be what the row waves' gave.
TEST_F(Dsv4EngineDspark, JoinedTokenWavesInjectTheFeaturesRowWavesDo) {
  Shape shape;
  shape.drafter = true;
  shape.wave_form = "plain";
  const auto started = Start(shape);
  ASSERT_TRUE(started) << (started ? "" : started.error());
  HostFunding funding(life->server->node());
  ASSERT_TRUE(funding.funded());
  ASSERT_TRUE(Engine().runner().slots().stages() >= 2U);
  constexpr std::array<std::uint32_t, 2> kPrompt{257, 258};
  std::array<Observed, 2> joined_rows, joined_tokens, alone_rows, alone_tokens;
  for (const bool device : {false, true}) {
    auto& joined = device ? joined_tokens : joined_rows;
    auto& alone = device ? alone_tokens : alone_rows;
    std::array<std::vector<float>, 2> last;
    std::array<rt::GenerateOptions, 2> options;
    std::array<rt::Generation, 2> out;
    for (std::uint32_t i = 0; i < 2; ++i) {
      ASSERT_TRUE(Branch(i).Clear());
      ASSERT_TRUE(Branch(i).Prefill(Tokens(i, kPrompt[i]), last[i]));
      ASSERT_TRUE(Finite(last[i]));
      // The head's token and three joined decode waves; then the same
      // sessions' bound is raised for the owners' own draft/verify steps.
      options[i].max_tokens = 4;
      options[i].stop = false;
      options[i].keep_logits = !device;
    }
    SessionSet<rt::Llm::GenerationSession, 2> sessions;
    for (std::uint32_t i = 0; i < 2; ++i) {
      auto opened = Branch(i).BeginGeneration(last[i], options[i], out[i]);
      ASSERT_TRUE(opened) << (opened ? "" : opened.error());
      sessions.owned[i] = std::move(*opened);
    }
    const auto before = Engine().counts();
    const auto graphs = Engine().runner().graph_stats();
    for (unsigned wave = 0; wave < 3; ++wave) {
      const std::array<rt::Llm::GenerationSession*, 2> both{sessions.owned[0].get(),
                                                            sessions.owned[1].get()};
      const auto ran = Engine().RunGenerationWave(both);
      ASSERT_TRUE(ran) << "wave " << wave << ": " << (ran ? "" : ran.error());
    }
    const auto after = Engine().counts();
    EXPECT_EQ(after.joined_waves - before.joined_waves, 3U);
    EXPECT_EQ(after.greedy_tokens - before.greedy_tokens, device ? 6U : 0U);
    EXPECT_EQ(after.speculative_units, before.speculative_units);
    EXPECT_GT(Engine().runner().graph_stats().replayed + Engine().runner().graph_stats().captured,
              graphs.replayed + graphs.captured);
    for (std::uint32_t i = 0; i < 2; ++i) {
      ASSERT_TRUE(sessions.owned[i]->done());
      ASSERT_TRUE(sessions.owned[i]->Finish());
      sessions.owned[i].reset();
      ASSERT_EQ(out[i].tokens.size(), 4U);
      ASSERT_EQ(out[i].logits.size(), device ? 0U : 4U);
      joined[i].tokens = out[i].tokens;
      joined[i].history = Branch(i).history();
      for (const auto& row : out[i].logits) ASSERT_TRUE(Finite(row));
      const auto state = StateHash(i);
      ASSERT_TRUE(state) << state.error();
      joined[i].state = *state;
    }
    // Both owners continue from the joined waves; one leaves first, as a
    // finished request does, and the other speculates alone. Each owner's
    // own draft/verify step: rows kept (the heads compared) in both runs.
    std::array<rt::GenerateOptions, 2> next;
    for (std::uint32_t i = 0; i < 2; ++i) {
      next[i].max_tokens = 8;
      next[i].stop = false;
      next[i].keep_logits = true;
      const std::size_t kept_rows = out[i].logits.size();
      auto resumed = Branch(i).ResumeGeneration({}, next[i], out[i]);
      ASSERT_TRUE(resumed) << (resumed ? "" : resumed.error());
      std::unique_ptr<rt::Llm::GenerationSession> session = std::move(*resumed);
      const auto drafted = out[i].drafted;
      const auto ran = RunToEnd(Engine(), *session);
      const auto finished = session->Finish();
      ASSERT_TRUE(ran) << (ran ? "" : ran.error());
      ASSERT_TRUE(finished) << (finished ? "" : finished.error());
      ASSERT_EQ(out[i].tokens.size(), 8U);
      EXPECT_GT(out[i].drafted, drafted) << "owner " << i << " did not draft";
      alone[i].tokens = out[i].tokens;
      alone[i].drafted = out[i].drafted - drafted;
      alone[i].accepted = out[i].accepted;
      alone[i].rows.assign(out[i].logits.begin() + static_cast<std::ptrdiff_t>(kept_rows),
                           out[i].logits.end());
      ASSERT_EQ(alone[i].rows.size(), 4U);
      for (const auto& row : alone[i].rows) ASSERT_TRUE(Finite(row));
      alone[i].history = Branch(i).history();
      const auto state = StateHash(i);
      ASSERT_TRUE(state) << state.error();
      alone[i].state = *state;
    }
  }
  for (std::uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(joined_tokens[i].tokens, joined_rows[i].tokens) << "owner " << i;
    EXPECT_EQ(joined_tokens[i].history, joined_rows[i].history) << "owner " << i;
    EXPECT_EQ(joined_tokens[i].state, joined_rows[i].state) << "owner " << i;
    EXPECT_EQ(alone_tokens[i].tokens, alone_rows[i].tokens) << "owner " << i;
    EXPECT_EQ(alone_tokens[i].drafted, alone_rows[i].drafted) << "owner " << i;
    EXPECT_EQ(alone_tokens[i].accepted, alone_rows[i].accepted) << "owner " << i;
    EXPECT_EQ(alone_tokens[i].history, alone_rows[i].history) << "owner " << i;
    ASSERT_EQ(alone_tokens[i].rows.size(), alone_rows[i].rows.size());
    for (std::size_t row = 0; row < alone_rows[i].rows.size(); ++row)
      EXPECT_TRUE(Same(alone_tokens[i].rows[row], alone_rows[i].rows[row]))
          << "owner " << i << " head " << row;
    EXPECT_EQ(alone_tokens[i].state, alone_rows[i].state) << "owner " << i;
  }
  EXPECT_EQ(Engine().runner().coverage().violations, 0U);
}

// Joined draft/verify waves (both forms forced speculative) past the ring:
// rows kept and tokens chosen on the device verify the same drafts, tokens,
// complete state and history; the heads (rows) of the first agree with the
// continuation's.
TEST_F(Dsv4EngineDspark, JoinedDraftVerifyWavesAgreeWithRowsAndWithTokens) {
  Shape shape;
  shape.drafter = true;
  shape.wave_form = "speculative";
  const auto started = Start(shape);
  ASSERT_TRUE(started) << (started ? "" : started.error());
  HostFunding funding(life->server->node());
  ASSERT_TRUE(funding.funded());
  constexpr std::array<std::uint32_t, 2> kPrompt{257, 258};
  std::array<Observed, 2> by_rows, by_tokens;
  for (const bool device : {false, true}) {
    auto& seen = device ? by_tokens : by_rows;
    std::array<std::vector<float>, 2> last;
    std::array<rt::GenerateOptions, 2> options;
    std::array<rt::Generation, 2> out;
    for (std::uint32_t i = 0; i < 2; ++i) {
      ASSERT_TRUE(Branch(i).Clear());
      ASSERT_TRUE(Branch(i).Prefill(Tokens(i, kPrompt[i]), last[i]));
      options[i].max_tokens = 24;
      options[i].stop = false;
      options[i].keep_logits = !device;
    }
    SessionSet<rt::Llm::GenerationSession, 2> sessions;
    for (std::uint32_t i = 0; i < 2; ++i) {
      auto opened = Branch(i).BeginGeneration(last[i], options[i], out[i]);
      ASSERT_TRUE(opened) << (opened ? "" : opened.error());
      sessions.owned[i] = std::move(*opened);
    }
    const auto before = Engine().counts();
    for (unsigned wave = 0; wave < 48 && (!sessions.owned[0]->done() || !sessions.owned[1]->done());
         ++wave) {
      std::array<rt::Llm::GenerationSession*, 2> active{};
      std::size_t count = 0;
      for (auto& s : sessions.owned)
        if (!s->done()) active[count++] = s.get();
      const auto ran = Engine().RunGenerationWave(std::span(active).first(count));
      ASSERT_TRUE(ran) << "wave " << wave << ": " << (ran ? "" : ran.error());
    }
    const auto after = Engine().counts();
    EXPECT_GT(after.joined_waves, before.joined_waves);
    EXPECT_GT(after.speculative_units, before.speculative_units);
    EXPECT_GT(after.drafted, before.drafted);
    EXPECT_EQ(after.greedy_tokens, before.greedy_tokens);
    for (std::uint32_t i = 0; i < 2; ++i) {
      ASSERT_TRUE(sessions.owned[i]->done());
      ASSERT_TRUE(sessions.owned[i]->Finish());
      sessions.owned[i].reset();
      ASSERT_EQ(out[i].tokens.size(), 24U);
      ASSERT_EQ(out[i].logits.size(), device ? 0U : 24U);
      for (const auto& row : out[i].logits) ASSERT_TRUE(Finite(row));
      seen[i].tokens = out[i].tokens;
      seen[i].drafted = out[i].drafted;
      seen[i].accepted = out[i].accepted;
      seen[i].history = Branch(i).history();
      const auto state = StateHash(i);
      ASSERT_TRUE(state) << state.error();
      seen[i].state = *state;
      seen[i].rows = std::move(out[i].logits);
    }
  }
  for (std::uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(by_tokens[i].tokens, by_rows[i].tokens) << "owner " << i;
    EXPECT_EQ(by_tokens[i].drafted, by_rows[i].drafted) << "owner " << i;
    EXPECT_EQ(by_tokens[i].accepted, by_rows[i].accepted) << "owner " << i;
    EXPECT_EQ(by_tokens[i].history, by_rows[i].history) << "owner " << i;
    EXPECT_EQ(by_tokens[i].state, by_rows[i].state) << "owner " << i;
    // The rows kept name the tokens chosen: each row's argmax.
    ASSERT_EQ(by_rows[i].rows.size(), by_rows[i].tokens.size());
    for (std::size_t row = 0; row < by_rows[i].rows.size(); ++row)
      EXPECT_EQ(Argmax(by_rows[i].rows[row]), by_rows[i].tokens[row])
          << "owner " << i << " row " << row;
  }
  EXPECT_EQ(Engine().runner().coverage().violations, 0U);
}

// ---------------------------------------------------------------- 4

// Four slots' DSpark verify waves (and their joined drafts) at every
// alignment of the compressors' blocks and mixed rows a slot, in a context
// whose end is past HCA's 256 compressed cells. A wave of 9 to 16 rows
// plans more graph nodes than the start's widest waves; each must fit the
// startup graph and metadata allowance, as the chat route's four-request
// speculative waves need, with no admission refusal.
TEST_F(Dsv4EngineDspark, FourSlotSpeculativeWavesFitTheStartupAllowance) {
  Shape shape;
  shape.drafter = true;
  shape.wave_form = "speculative";
  shape.context = 65536;
  shape.slots = 4;
  const auto started = Start(shape, 4);
  ASSERT_TRUE(started) << (started ? "" : started.error());
  ASSERT_EQ(Engine().generation_wave_capacity(), 4U);
  HostFunding funding(life->server->node());
  ASSERT_TRUE(funding.funded());
  constexpr std::size_t kSlots = 4;
  // Whole-chunk prompts ending at each alignment of a CSA block, one at an
  // HCA block's end.
  constexpr std::array<std::uint32_t, kSlots> kPast{4093, 4094, 4095, 4096};
  std::array<std::vector<float>, kSlots> heads;
  for (std::uint32_t i = 0; i < kSlots; ++i) {
    ASSERT_TRUE(Branch(i).Clear());
    const auto prefilled = Branch(i).Prefill(Tokens(i, kPast[i]), heads[i]);
    ASSERT_TRUE(prefilled) << "slot " << i << ": " << (prefilled ? "" : prefilled.error());
    ASSERT_TRUE(Finite(heads[i]));
  }
  constexpr std::array<std::array<std::uint32_t, kSlots>, 3> kRows{
      {{4, 4, 4, 4}, {1, 2, 3, 4}, {3, 1, 4, 2}}};
  std::array<rt::GenerateOptions, kSlots> options;
  std::array<rt::Generation, kSlots> out;
  const auto before = Engine().counts();
  const auto graphs = Engine().runner().graph_stats();
  std::uint64_t speculative = 0;
  for (std::uint32_t wave = 0; wave < 12; ++wave) {
    const auto& rows = kRows[wave % kRows.size()];
    SessionSet<rt::Llm::GenerationSession, kSlots> sessions;
    std::array<std::size_t, kSlots> had{};
    for (std::uint32_t i = 0; i < kSlots; ++i) {
      had[i] = out[i].tokens.size();
      options[i].stop = false;
      // The step's rows: the anchor and what is left of the bound.
      options[i].max_tokens = static_cast<std::uint32_t>(had[i]) + rows[i] + (wave == 0 ? 1U : 0U);
      auto opened = wave == 0 ? Branch(i).BeginGeneration(heads[i], options[i], out[i])
                              : Branch(i).ResumeGeneration({}, options[i], out[i]);
      ASSERT_TRUE(opened) << "wave " << wave << ": " << (opened ? "" : opened.error());
      sessions.owned[i] = std::move(*opened);
      if (wave == 0) had[i] = out[i].tokens.size();  // the head's token
    }
    std::array<rt::Llm::GenerationSession*, kSlots> active{};
    for (std::size_t i = 0; i < kSlots; ++i) active[i] = sessions.owned[i].get();
    const auto units = Engine().counts().speculative_units;
    const auto ran = Engine().RunGenerationWave(active);
    ASSERT_TRUE(ran) << "wave " << wave << ": " << (ran ? "" : ran.error());
    speculative += Engine().counts().speculative_units - units;
    for (std::uint32_t i = 0; i < kSlots; ++i) {
      const std::size_t gained = out[i].tokens.size() - had[i];
      EXPECT_GE(gained, 1U) << "wave " << wave << " slot " << i;
      EXPECT_LE(gained, rows[i]) << "wave " << wave << " slot " << i;
    }
    // A step that kept fewer than its rows leaves its session open: it
    // ends there (its anchor pending), as a finished request does.
    sessions.Close();
  }
  const auto after = Engine().counts();
  EXPECT_EQ(speculative, 12U * kSlots);
  EXPECT_EQ(after.speculative_units - before.speculative_units, 12U * kSlots);
  EXPECT_EQ(after.joined_waves - before.joined_waves, 12U);
  EXPECT_EQ(after.joined_units - before.joined_units, 12U * kSlots);
  EXPECT_GT(after.drafted, before.drafted);
  const auto graphs_after = Engine().runner().graph_stats();
  EXPECT_GT(graphs_after.eager + graphs_after.captured + graphs_after.replayed,
            graphs.eager + graphs.captured + graphs.replayed);
  EXPECT_EQ(graphs_after.refused, graphs.refused) << graphs_after.first_refusal;
  EXPECT_EQ(Engine().runner().coverage().violations, 0U);
  for (std::uint32_t i = 0; i < kSlots; ++i) {
    EXPECT_GT(out[i].drafted, 0U) << "slot " << i;
    ASSERT_TRUE(StateHash(i));
  }
}

// ---------------------------------------------------------------- 5

// Prefill lookahead on and off give one answer: the same heads and complete
// states at prompts that cross CSA and HCA blocks (a block's end, and
// five past it) and the raw ring (4,352 cells at the 4,096-row chunk), as
// two owners' prompts and as one that passes the ring twice; and the
// lookahead really ran when on, and did not when off. (DeepSeek's prompts
// are not joined: its steps run each owner's chunks alone.)
TEST_F(Dsv4EngineTarget, PrefillLookaheadOnAndOffGiveTheSameHeadsAndState) {
  struct Arm {
    std::array<std::vector<float>, 2> joined_aligned, joined_odd;
    std::vector<float> lone;
    std::array<llmp::base::Sha256Digest, 2> state_aligned{}, state_odd{};
    llmp::base::Sha256Digest state_lone{};
    std::uint64_t built = 0, captured_ahead = 0;
  };
  std::array<Arm, 2> arms;
  for (const bool lookahead : {true, false}) {
    Arm& arm = arms[lookahead ? 0 : 1];
    Shape shape;
    shape.lookahead = lookahead;
    const auto started = Start(shape);
    ASSERT_TRUE(started) << (started ? "" : started.error());
    HostFunding funding(life->server->node());
    ASSERT_TRUE(funding.funded());
    // 4,608 is 36 HCA blocks and 4,736 is 37; each pair's second chunk is
    // 512 and 640 rows, the odd pair's five more.
    for (const bool aligned : {true, false}) {
      const std::array<std::uint32_t, 2> rows = aligned ? std::array<std::uint32_t, 2>{4608, 4736}
                                                        : std::array<std::uint32_t, 2>{4613, 4741};
      auto& heads = aligned ? arm.joined_aligned : arm.joined_odd;
      const auto waves = JoinedPrompts(rows, heads);
      ASSERT_TRUE(waves) << (waves ? "" : waves.error());
      for (std::uint32_t id = 0; id < 2; ++id) {
        ASSERT_TRUE(Finite(heads[id])) << "owner " << id;
        EXPECT_EQ(Engine().runner().slots().positions(id), rows[id]);
        const auto state = StateHash(id);
        ASSERT_TRUE(state) << state.error();
        (aligned ? arm.state_aligned : arm.state_odd)[id] = *state;
      }
    }
    // One owner's prompt of two full chunks and five rows more.
    ASSERT_TRUE(Select(1));
    ASSERT_TRUE(Branch(0).Clear());
    ASSERT_TRUE(Branch(0).Prefill(Tokens(0, 8453), arm.lone));
    ASSERT_TRUE(Finite(arm.lone));
    const auto state = StateHash(0);
    ASSERT_TRUE(state) << state.error();
    arm.state_lone = *state;
    arm.built = Engine().runner().lookahead_stats().built;
    arm.captured_ahead = Engine().runner().lookahead_stats().captured_ahead;
    EXPECT_EQ(Engine().runner().coverage().violations, 0U);
    ASSERT_TRUE(Retire());
  }
  EXPECT_GT(arms[0].built, 0U) << "lookahead on built no step ahead";
  EXPECT_EQ(arms[1].built, 0U) << "lookahead off built a step ahead";
  EXPECT_EQ(arms[1].captured_ahead, 0U);
  for (std::uint32_t id = 0; id < 2; ++id) {
    EXPECT_TRUE(Same(arms[0].joined_aligned[id], arms[1].joined_aligned[id])) << "aligned " << id;
    EXPECT_TRUE(Same(arms[0].joined_odd[id], arms[1].joined_odd[id])) << "odd " << id;
    EXPECT_EQ(arms[0].state_aligned[id], arms[1].state_aligned[id]) << "aligned " << id;
    EXPECT_EQ(arms[0].state_odd[id], arms[1].state_odd[id]) << "odd " << id;
  }
  EXPECT_TRUE(Same(arms[0].lone, arms[1].lone));
  EXPECT_EQ(arms[0].state_lone, arms[1].state_lone);
}

// KNOWN DEFECT, found writing this file (reported, not fixed here): a
// prefill_chunk override below the default's makes the first chunk of a
// prompt fail at its bind, "the step's scratch (36700160 bytes) exceeds the
// pool (34603008 bytes)", at every chunk from 256 to 1,024 rows, one and
// two slots, 3,072 and 8,704 context, always the same two numbers, so the
// pool is not sized by the chunk (ModelRunner::Setup sizes it from the
// setup shapes, component_build.cc: a chunk's rows at the end of the
// context, a decode row, ...). The default 4,096-row chunk's pool is larger
// than the step needs. Enable when the pool covers it.
TEST_F(Dsv4EngineTarget, DISABLED_ASmallPrefillChunkServesItsFirstChunk) {
  Shape shape;
  shape.chunk = 512;
  const auto started = Start(shape, 1);
  ASSERT_TRUE(started) << (started ? "" : started.error());
  std::vector<float> last;
  const auto r = Branch(0).Prefill(Tokens(0, 1285), last);
  ASSERT_TRUE(r) << (r ? "" : r.error());
  EXPECT_TRUE(Finite(last));
}

// ---------------------------------------------------------------- 6

struct TopLogit {
  std::int32_t id;
  float logit;
};

// The whole-model reference of one fixed prompt: Tokens(0, 1285) (a prompt
// that crosses CSA and HCA blocks), its head and seven greedy steps' rows
// (eight rows, the head's first), at the default 4,096-row chunk and 8,704
// context on the artifact 8a355bfb....
//
// kLegacy: the frozen M3 runner's (the baseline build
// ~/.local/share/llmp/m36-baseline/1facc957-fixed, llmp_dsv4_exec --context
// 8704 --max-rows 4096 --outa-prefill --ds4-hca --outa-partial --generate
// 8): each row's two largest logits. kEngine: the engine's own four
// largest on this tree's reference run (M3.6, 2026-10-15, Spark B), a
// snapshot that a change to the arithmetic re-baselines.
constexpr std::array<std::int32_t, 8> kLegacyTokens{563, 42, 509, 818, 5279, 529, 7001, 42};
constexpr std::array<std::array<TopLogit, 2>, 8> kLegacy{{
    {{{563, 30.9269F}, {42, 19.6843F}}},
    {{{42, 38.8520F}, {62684, 22.4839F}}},
    {{{509, 19.9194F}, {818, 19.4967F}}},
    {{{818, 23.4615F}, {5279, 21.1966F}}},
    {{{5279, 28.0338F}, {529, 26.3023F}}},
    {{{529, 27.3918F}, {7001, 19.5456F}}},
    {{{7001, 26.3718F}, {17960, 19.5754F}}},
    {{{42, 26.4342F}, {563, 22.9392F}}},
}};
constexpr std::array<std::array<TopLogit, 4>, 8> kEngine{{
    {{{563, 30.8123F}, {42, 20.0247F}, {840, 17.8198F}, {1862, 15.6124F}}},
    {{{42, 38.6271F}, {93533, 22.1704F}, {62684, 21.9064F}, {47, 21.6156F}}},
    {{{509, 20.8542F}, {818, 20.5476F}, {304, 18.5027F}, {16, 18.4503F}}},
    {{{818, 23.9802F}, {5279, 22.1220F}, {30, 20.0510F}, {28527, 17.3587F}}},
    {{{5279, 28.7194F}, {529, 27.2951F}, {8679, 21.7651F}, {28527, 19.8762F}}},
    {{{529, 28.1108F}, {7001, 20.4166F}, {42, 19.3092F}, {82, 18.7747F}}},
    {{{7001, 27.4798F}, {17960, 20.4821F}, {31803, 18.6722F}, {42, 18.6383F}}},
    {{{42, 27.3389F}, {563, 23.5059F}, {41138, 17.7831F}, {64784, 17.2868F}}},
}};
// The bounds, from the two runs' measured differences and fixed here: the
// engine's run-to-run rows are bit-identical (so 0.05 is slack for a
// toolchain's rounding, not noise); the engine against the M3 runner differs
// by an RMS of 0.2 to 0.65 over every logit and at most 1.49 at each row's
// eight largest values, 1.11 at its two largest (fusion and reduction
// order; the logits' own RMS is 4 to 5.6), so 2.0 bounds a value against
// the M3 runner's. A token the M3
// runner chose by a margin of this or less may be its runner-up here
// (kNearTie; the smallest margin of these rows is 0.42, and the engine
// chose the same token on every row).
constexpr float kSnapshotBound = 0.05F;
constexpr float kLegacyBound = 2.0F;
constexpr float kNearTie = 0.5F;

TEST_F(Dsv4EngineTarget, FixedPromptHeadAndGreedyTokensMatchTheReferences) {
  const auto started = Start({}, 1);
  ASSERT_TRUE(started) << (started ? "" : started.error());
  std::vector<float> head;
  const auto prefilled = Branch(0).Prefill(Tokens(0, 1285), head);
  ASSERT_TRUE(prefilled) << (prefilled ? "" : prefilled.error());
  rt::GenerateOptions options;
  options.max_tokens = kLegacyTokens.size();
  options.stop = false;
  options.keep_logits = true;
  rt::Generation out;
  const auto generated = Branch(0).Generate(head, options, out);
  ASSERT_TRUE(generated) << (generated ? "" : generated.error());
  ASSERT_EQ(out.tokens.size(), kLegacyTokens.size());
  ASSERT_EQ(out.logits.size(), kLegacyTokens.size());
  ASSERT_TRUE(Same(head, out.logits[0]));
  for (std::size_t row = 0; row < kLegacyTokens.size(); ++row) {
    const auto& logits = out.logits[row];
    ASSERT_TRUE(Finite(logits)) << "row " << row;
    EXPECT_EQ(out.tokens[row], Argmax(logits)) << "row " << row;
    // The M3 runner's tokens, tie-aware.
    const auto& legacy = kLegacy[row];
    const float margin = legacy[0].logit - legacy[1].logit;
    EXPECT_TRUE(out.tokens[row] == kLegacyTokens[row] ||
                (margin <= kNearTie && out.tokens[row] == legacy[1].id))
        << "row " << row << ": token " << out.tokens[row] << ", the M3 runner's "
        << kLegacyTokens[row] << " by " << margin;
    for (const TopLogit& want : legacy)
      EXPECT_NEAR(logits[static_cast<std::size_t>(want.id)], want.logit, kLegacyBound)
          << "row " << row << " token " << want.id << " against the M3 runner";
    // The engine's own reference.
    EXPECT_EQ(Argmax(logits), kEngine[row][0].id) << "row " << row;
    for (const TopLogit& want : kEngine[row])
      EXPECT_NEAR(logits[static_cast<std::size_t>(want.id)], want.logit, kSnapshotBound)
          << "row " << row << " token " << want.id << " against the engine's reference";
    llmp::base::Sha256 digest;
    digest.Update(std::as_bytes(std::span(logits)));
    RecordProperty(std::format("row{}_sha256", row), llmp::base::ToHex(digest.Finish()));
  }
  EXPECT_EQ(Engine().runner().coverage().violations, 0U);
}

}  // namespace
