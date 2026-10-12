// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Default-off single-slot migration screen through the real Server and Llm.
// Both arms draft and judge their own outputs. No graph/source wrapper or
// teacher-forced proposals. This small screen is not the 1% migration gate.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/json.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "engine/component_source.h"
#include "engine/paged_weights.h"
#include "execution/registry.h"
#include "kernels/common/step_program.h"
#include "kernels/ggml/graph_plan.h"
#include "model/qwen38.h"
#include "planner/plan.h"
#include "runtime/component_build.h"
#include "runtime/engine_llm.h"
#include "runtime/serving.h"
#include "tokenizer_fixtures.h"

namespace {
namespace fs = std::filesystem;
namespace rt = llmp::runtime;
namespace cfg = llmp::config;
namespace en = llmp::engine;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
// Fixed before the first model screen: both arms retain the same payment
// inside this cap; normal fresh physical-headroom guards still apply.
constexpr std::uint64_t kBudget = std::uint64_t{104} << 30U;
constexpr std::uint64_t kControlBudget = std::uint64_t{16} << 30U;
// The source package inventories compiler phases at the existing IR hard
// caps. The serving delta includes artifact/binding owners, maps, integer
// metadata, server state, and both retained results. ABI guards below check
// the coefficients; native allocations remain cataloged by the normal node.
constexpr std::uint64_t kCompiler = 3584 * kMiB;
constexpr std::uint64_t kResults = 128 * kMiB;
constexpr std::uint64_t kAllowance = kCompiler + kResults;
static_assert(sizeof(void*) == 8 && sizeof(std::string) <= 32 && sizeof(std::vector<void*>) <= 24);
static_assert(llmp::ir::Graph::kMaxNodes == 65536 && llmp::ir::Graph::kMaxTensors == 131072);
static_assert(sizeof(llmp::ir::Node) <= 160 && sizeof(llmp::ir::TensorDesc) <= 160);
static_assert(sizeof(llmp::ir::Binding) <= 24 && sizeof(llmp::format::TensorFormat) <= 72);
static_assert(sizeof(llmp::planner::Selected) <= 128 && sizeof(llmp::planner::Applied) <= 64);
static_assert(sizeof(llmp::planner::Match) + sizeof(void*) <= 256);
static_assert(sizeof(llmp::planner::RootAlignmentFact) <= 16);
static_assert(sizeof(llmp::kernels::InputSlot) <= 512 && sizeof(llmp::kernels::TensorUse) <= 64);
static_assert(sizeof(llmp::kernels::ggml::PlanStep) <= 64);
static_assert(sizeof(llmp::execution::OperandMemory) <= 48 &&
              sizeof(llmp::execution::OperandFormat) <= 112 &&
              sizeof(llmp::execution::CapabilityQuery) <= 256);
static_assert(sizeof(llmp::ir::WeightRef) <= 16 && sizeof(llmp::ir::StateRef) <= 16);
static_assert(llmp::execution::Registry::kMaxImplementations == 256);
static_assert(llmp::ir::kMaxSources == 10 && llmp::ir::kMaxOutputs == 4);
// Inherited hard-cap compiler peak plus the source-inventoried serving
// owners below. The fixture fingerprints are checked before either arm;
// this fixed allowance is not a general arbitrary-artifact memory claim.
constexpr std::uint64_t kInheritedCompilerMaximum = 2528028992ULL;
constexpr std::uint64_t kArtifactValues = 530000;
constexpr std::uint64_t kTokenizerValues = 750000;
constexpr std::uint64_t kTokens = 250000;
// Artifact parser: three-capacity growth overlap for nodes/kids/pending,
// sorted-key views and input text, plus its ordered validation records.
constexpr std::uint64_t kArtifactParse =
    3 * kArtifactValues * (24 + 4 + 4) + 16 * kArtifactValues + 3 * 9 * kMiB + 3 * 26000 * 80;
// Four conservative artifact owners (actual target+drafter, temporary
// settings/build opener and shared registration owner). Each resource's
// 4096 includes its record, <=9 role/name strings, rank vectors and nodes.
constexpr std::uint64_t kArtifacts =
    4 * (3 * 1840 * 4096 + 3 * 26000 * 64 + 70000 * 32 + 3 * 4096 * 64);
// Tokenizer spec + final tokens/decoded/merge strings, all growth overlap;
// map nodes/bucket rehashes; parser and GGUF string/index views. Parse and
// Create are serial, but this sum deliberately pays both alongside final.
constexpr std::uint64_t kTokenizer =
    3 * 4 * kTokens * 32 + 3 * 2 * 9 * kMiB + 2 * kTokens * (64 + 3 * 8) + 3 * 16 * kMiB +
    kTokenizerValues * (12 + 4 + 4 + 16) + 3 * 2 * kTokens * 16 + 2 * kMiB;
// 70k bounds dense/slab pages plus all context-512 state/runtime entries.
// Extent slot, three-capacity PagedWeights source array, scheduler tree,
// slab-page construction, node address/coverage arrays and keyed records.
constexpr std::uint64_t kCatalogOwners = 70000 * (3 * 192 + 3 * 288 + 320 + 3 * 128 + 3 * 64 + 128);
// Binding/spec/source tables: six simultaneous bounded 1840-record sets;
// multipart places/strides, selected IDs, two row-table metadata owners,
// known Server/request/template/controller headers and allocator rounding.
constexpr std::uint64_t kServingBindings =
    6 * 1840 * 2048 + 3 * 4096 * 256 + 3 * 248320 * 4 + 2 * 512 * 256 + 4 * kMiB;
// Artifact directory entry vector/string growth, including an early
// malformed file-set refusal up to the existing directory-entry cap.
constexpr std::uint64_t kDirectoryNames = 3 * 65536 * (32 + 2 * 256);
static_assert(llmp::artifact::kMaxDirectoryEntries == 65536);
constexpr std::uint64_t kServingDelta =
    kDirectoryNames + kArtifactParse + kArtifacts + kTokenizer + kCatalogOwners + kServingBindings;
static_assert(kInheritedCompilerMaximum + kServingDelta <= kCompiler);
static_assert(sizeof(llmp::artifact::Resource) <= 512 &&
              sizeof(llmp::artifact::ExpertArray) <= 512 && sizeof(llmp::artifact::Group) <= 64);
static_assert(sizeof(llmp::engine::SlabPage) <= 128 && sizeof(llmp::scheduler::PageSource) <= 256 &&
              sizeof(llmp::catalog::ExtentView) <= 144);
static_assert(sizeof(llmp::spec::SourceTensor) <= 128 && sizeof(llmp::model::Qwen38Tensor) <= 96);
// Exact private-field mirrors used by the inherited and new inventories.
struct JsonNodeAbi {
  llmp::artifact::json::Kind kind{};
  bool fits = false;
  std::uint32_t size = 0, first = 0;
  std::int64_t integer = 0;
};
struct RankedAbi {
  std::size_t index = 0;
  std::uint8_t rung = 0;
  std::uint64_t measured = 0;
  std::uint16_t priority = 0;
  std::string_view name{};
};
struct RootIdentityAbi {
  llmp::ir::TensorId id{};
  llmp::ir::Binding binding{};
  std::uint32_t input = 0, computation = 0;
  std::uint8_t output = 0;
  llmp::ir::Role role{};
  llmp::format::TensorFormat format{};
  std::array<std::uint64_t, 4> nb{};
  std::uint8_t segment = 0;
};
static_assert(sizeof(JsonNodeAbi) <= 24 && sizeof(RankedAbi) <= 64 &&
              sizeof(RootIdentityAbi) <= 160);
static_assert(sizeof(std::pair<llmp::ir::TensorDesc, std::uint64_t>) <= 176);
constexpr std::uint32_t kVocab = 248320;
constexpr std::uint64_t kRetainedRows = 2 * (2 + 8 + 8);
static_assert(2 * kRetainedRows * kVocab * sizeof(float) + 18 * kMiB <= kResults);

constexpr std::string_view kCutlass =
    "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93";
constexpr std::string_view kGgml =
    "67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6";
constexpr std::string_view kGguf =
    "5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78";
constexpr std::string_view kMtp =
    "8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40";

// A streaming, fixed-stack identity preflight authenticates the metadata
// counts used above without constructing JSON, artifacts or tokenizers.
// The actual Server installs its catalog payment before reading them again.
struct FixtureFile {
  std::string_view artifact, path;
  std::uint64_t bytes;
  std::string_view digest;
};
constexpr std::array kFixtureFiles{
    FixtureFile{
        "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93",
        "m3-artifacts/c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93/index.json",
        8152364ULL, "5b65dcce8169374e638264d7ebdcb5cca517234b3dec26f1272a5f4f9c0e4c89"},
    FixtureFile{"c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93",
                "m3-artifacts/c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93/"
                "manifest.json",
                23037ULL, "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93"},
    FixtureFile{"c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93",
                "m3-artifacts/c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93/"
                "meta/config.json",
                109897ULL, "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e"},
    FixtureFile{
        "67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6",
        "m3-artifacts/67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6/index.json",
        8142763ULL, "ac9f3705caea91987cb517a84e70d16b3f9159385db2d0d4b24132805846a54f"},
    FixtureFile{"67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6",
                "m3-artifacts/67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6/"
                "manifest.json",
                19159ULL, "67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6"},
    FixtureFile{"67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6",
                "m3-artifacts/67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6/"
                "meta/config.json",
                109897ULL, "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e"},
    FixtureFile{"5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78",
                "qgguf-artifacts/5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78/"
                "index.json",
                6330819ULL, "d6c69d701ead0cbadc3a64f1f3c354a70ecac7179f61dbac1e1163fca0cd3550"},
    FixtureFile{"5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78",
                "qgguf-artifacts/5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78/"
                "manifest.json",
                14289ULL, "5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78"},
    FixtureFile{"5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78",
                "qgguf-artifacts/5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78/"
                "meta/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.kv.gguf",
                10946618ULL, "7c1460a95eb2e95377c32b14deb92cf74fe504d4feb5a62a40d48f78e9e0dcb8"},
    FixtureFile{"5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78",
                "qgguf-artifacts/5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78/"
                "meta/Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.kv.gguf",
                106ULL, "0bd485a7f050b5ac258c3722f25f2be38a7ef3e597696f3e4d7e09ec461ff978"},
    FixtureFile{"5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78",
                "qgguf-artifacts/5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78/"
                "meta/Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.kv.gguf",
                106ULL, "2fd68e0dc3770fdd9e4d52a5e9570394855926807e0e020b879651ded2b004c6"},
    FixtureFile{
        "8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40",
        "m3-artifacts/8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40/index.json",
        157885ULL, "5cd45fc5354ab224d281c2416027f224c61e32e2ae0acf8e7580d063fc274d99"},
    FixtureFile{"8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40",
                "m3-artifacts/8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40/"
                "manifest.json",
                1861ULL, "8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40"},
    FixtureFile{"8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40",
                "m3-artifacts/8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40/"
                "meta/config.json",
                109897ULL, "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e"},
    FixtureFile{"", "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/tokenizer.json",
                12809320ULL, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"},
    FixtureFile{"", "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/chat_template.jinja",
                8952ULL, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"},
};

rt::Status CheckFixture(std::string_view id, bool mtp) {
  for (const auto& file : kFixtureFiles) {
    if (!file.artifact.empty() && file.artifact != id && !(mtp && file.artifact == kMtp)) continue;
    const auto path = fs::path(llmp::test_support::ModelsDir()) / file.path;
    std::FILE* input = std::fopen(path.c_str(), "rb");
    if (input == nullptr) return std::unexpected("opening fixed serving metadata");
    struct stat st{};
    bool valid = fstat(fileno(input), &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 &&
                 static_cast<std::uint64_t>(st.st_size) == file.bytes;
    llmp::base::Sha256 hash;
    std::array<std::byte, 4096> block{};
    std::uint64_t total = 0;
    while (valid && total < file.bytes) {
      const auto want =
          static_cast<std::size_t>(std::min<std::uint64_t>(block.size(), file.bytes - total));
      const auto got = std::fread(block.data(), 1, want, input);
      if (got != want) {
        valid = false;
        break;
      }
      hash.Update(std::span(block).first(got));
      total += got;
    }
    valid = valid && std::fgetc(input) == EOF && std::ferror(input) == 0;
    valid = std::fclose(input) == 0 && valid;
    if (!valid || llmp::base::ToHex(hash.Finish()) != file.digest)
      return std::unexpected("fixed serving metadata identity changed: " + path.string());
  }
  return {};
}

struct Result {
  std::array<std::vector<float>, 2> heads;
  std::array<rt::Generation, 2> generations;
  std::array<std::vector<std::int32_t>, 2> history;
};

// Everything callbacks and native work can borrow has one heap owner. A
// faulted operation or failed terminal teardown leaks this complete owner,
// and the enclosing test stops before another arm is admitted.
struct Lifetime {
  cfg::NodeConfig config;
  cfg::RuntimeRoles roles;
  rt::ServingOptions options;
  std::unique_ptr<rt::Server> server;
  Result result;
  fs::path scratch;
  std::FILE* log = nullptr;
  ~Lifetime() {
    if (log != nullptr) std::fclose(log);
  }
};
static_assert(sizeof(Lifetime) <= 16384 && sizeof(rt::Server) <= 65536 &&
              sizeof(rt::EngineLlm) <= 65536);

std::vector<std::int32_t> Prompt() {
  std::vector<std::int32_t> out;
  out.reserve(7);
  for (std::uint32_t p = 0; p < 7; ++p)
    out.push_back(static_cast<std::int32_t>(100 + ((p * 7919U + 13U) % 150000U)));
  return out;
}

std::expected<std::unique_ptr<Lifetime>, std::string> Run(std::string_view id, bool mtp,
                                                          bool component, bool rows,
                                                          std::uint32_t depth,
                                                          Lifetime* previous = nullptr) {
  auto life = std::make_unique<Lifetime>();
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  if (scratch == nullptr) return std::unexpected("no test scratch directory");
  fs::create_directories(scratch);
  std::string name = (fs::path(scratch) / "qwen-solo-XXXXXX").string();
  if (mkdtemp(name.data()) == nullptr) return std::unexpected("creating test scratch");
  life->scratch = name;
  life->roles.installed = fs::path(llmp::test_support::ModelsDir()) /
                          (id == kGguf ? "qgguf-artifacts" : "m3-artifacts");
  life->roles.spill = life->scratch / "spill";
  life->roles.state = life->scratch / "state";
  for (const auto& path : {life->roles.spill, life->roles.state}) {
    fs::create_directory(path);
    if (chmod(path.c_str(), 0700) != 0) return std::unexpected("protecting test state");
  }
  cfg::ModelEntry entry;
  entry.name = "qwen";
  entry.artifact = id;
  const auto checkpoint = fs::path(llmp::test_support::ModelsDir()) /
                          "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6";
  entry.tokenizer = checkpoint / "tokenizer.json";
  entry.chat_template = checkpoint / "chat_template.jinja";
  if (mtp) entry.drafter = kMtp;
  entry.overrides["context"] = std::int64_t{512};
  entry.overrides["prefill_chunk"] = std::int64_t{16};
  entry.overrides["max_slots"] = std::int64_t{1};
  entry.overrides["draft_rows"] = static_cast<std::int64_t>(depth);
  entry.overrides["wave_read_align"] = std::int64_t{256};
  life->config.models.push_back(std::move(entry));
  life->options.diagnostic_component_solo = component;
  life->options.diagnostic_startup_host_bytes = kAllowance;
  life->options.diagnostic_budget_cap_bytes = kBudget;
  life->log = std::tmpfile();
  if (!life->log) return std::unexpected("opening test log");
  life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, life->log);
  rt::Llm* model = nullptr;
  auto run = [&]() -> rt::Status {
    if (auto s = life->server->Start(true); !s) return s;
    if (life->server->node().catalog().OccupancyOf(life->server->node().domain()).Total().value() <
        kAllowance)
      return std::unexpected("startup compiler allowance was not cataloged");
    // The old stopped Server pays its retained results until this allowance
    // is actually admitted. Only then transfer that payment and destroy its
    // remaining model metadata, before activating any new model work.
    if (previous != nullptr) previous->server.reset();
    model = dynamic_cast<rt::Llm*>(life->server->Find("qwen"));
    if (!model || (dynamic_cast<rt::EngineLlm*>(model) != nullptr) != component)
      return std::unexpected("wrong serving pipeline selected");
    if (model->generation_wave_capacity() != 1 || model->logit_width() != kVocab)
      return std::unexpected("wrong serving envelope");
    rt::SwapParts parts;
    if (auto s = life->server->Activate(*model, parts); !s) return s;
    auto b = model->branch(0);
    if (!b) return std::unexpected(b.error());
    std::array<rt::Llm::Branch*, 1> active{*b};
    if (auto s = life->server->SelectRequestBranches(*model, active); !s) return s;
    if (auto s = (*b)->Prefill(Prompt(), life->result.heads[0]); !s) return s;
    for (std::size_t turn = 0; turn < 2; ++turn) {
      rt::GenerateOptions options;
      options.max_tokens = 8;
      options.stop = false;
      options.keep_logits = rows;
      auto& result = life->result.generations[turn];
      if (auto s = (*b)->Generate(life->result.heads[turn], options, result); !s) return s;
      if (result.tokens.size() != 8 || (rows && result.logits.size() != 8))
        return std::unexpected("serving returned incomplete generation");
      if (mtp && result.drafted == 0) return std::unexpected("the serving unit did not speculate");
      life->result.history[turn] = (*b)->history();
      if (turn == 0) {
        // Idle spill retains the actual pending MTP state. Restore through
        // the normal prompt interface, then process the last sampled token.
        if (auto s = life->server->EndRequestBranches(*model); !s) return s;
        if (auto s = model->SpillIdle(**b); !s) return s;
        if (auto s = life->server->SelectRequestBranches(*model, active); !s) return s;
        auto next = (*b)->history();
        next.push_back(result.tokens.back());
        std::uint32_t reused = 0;
        if (auto s = (*b)->PreparePrompt(next, 0, life->result.heads[1], reused); !s) return s;
        if (reused + 1 != next.size()) return std::unexpected("spill resume lost its prefix");
      }
    }
    if (component) {
      const auto& engine = *static_cast<rt::EngineLlm*>(model);
      if (engine.runner().coverage().violations != 0)
        return std::unexpected("shared program bound uncovered storage");
      if (engine.runner().slots().stages() > 1 && engine.runner().slots().progress(0, 1) == nullptr)
        return std::unexpected("the serving drafter has no durable replay contract");
    }
    return life->server->EndRequestBranches(*model);
  }();
  // Any model-path failure is conservatively terminal for this test. Do
  // not destroy or partially tear down an environment with unknown work.
  if (!run) {
    const std::string reason = run.error();
    std::ignore = life.release();
    return std::unexpected(reason + "; complete arm retained");
  }
  const auto retired = life->server->TearDown();
  if (!retired) {
    const std::string reason = retired.error();
    std::ignore = life.release();
    return std::unexpected(reason + "; complete arm retained");
  }
  // TearDown retired GPU resources; retain the stopped Server/catalog
  // payment with its result through metadata checks and the next admission,
  // or through Compare for the last arm. Unknown owners never reach here.
  return life;
}

void Compare(const Result& got, const Result& want, bool rows) {
  for (std::size_t turn = 0; turn < 2; ++turn) {
    ASSERT_EQ(got.heads[turn].size(), kVocab);
    ASSERT_EQ(want.heads[turn].size(), kVocab);
    EXPECT_EQ(std::memcmp(got.heads[turn].data(), want.heads[turn].data(), kVocab * sizeof(float)),
              0);
    EXPECT_EQ(got.generations[turn].tokens, want.generations[turn].tokens);
    EXPECT_EQ(got.generations[turn].drafted, want.generations[turn].drafted);
    EXPECT_EQ(got.generations[turn].accepted, want.generations[turn].accepted);
    EXPECT_EQ(got.history[turn], want.history[turn]);
    EXPECT_EQ(got.generations[turn].steps, want.generations[turn].steps);
    if (rows) {
      ASSERT_EQ(got.generations[turn].logits.size(), 8U);
      ASSERT_EQ(want.generations[turn].logits.size(), 8U);
      for (std::size_t r = 0; r < 8; ++r) {
        const auto& a = got.generations[turn].logits[r];
        const auto& b = want.generations[turn].logits[r];
        ASSERT_EQ(a.size(), kVocab);
        ASSERT_EQ(b.size(), kVocab);
        EXPECT_EQ(std::memcmp(a.data(), b.data(), kVocab * sizeof(float)), 0) << turn << ':' << r;
      }
    }
  }
}

void Screen(std::string_view id, bool mtp, bool rows, std::uint32_t depth) {
  const auto store = fs::path(llmp::test_support::ModelsDir()) /
                     (id == kGguf ? "qgguf-artifacts" : "m3-artifacts");
  const auto checkpoint = fs::path(llmp::test_support::ModelsDir()) /
                          "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6";
  if (!fs::exists(store / id) || (mtp && !fs::exists(store / kMtp)) ||
      !fs::exists(checkpoint / "tokenizer.json") || !fs::exists(checkpoint / "chat_template.jinja"))
    GTEST_SKIP() << "the established Qwen artifact is absent";
  auto fixed = CheckFixture(id, mtp);
  ASSERT_TRUE(fixed) << (fixed ? "" : fixed.error());
  auto legacy = Run(id, mtp, false, rows, depth);
  ASSERT_TRUE(legacy) << (legacy ? "" : legacy.error());
  fixed = CheckFixture(id, mtp);
  ASSERT_TRUE(fixed) << (fixed ? "" : fixed.error());
  auto shared = Run(id, mtp, true, rows, depth, legacy->get());
  ASSERT_TRUE(shared) << (shared ? "" : shared.error());
  fixed = CheckFixture(id, mtp);
  if (!fixed) legacy->reset();  // release under the still-live final payment
  ASSERT_TRUE(fixed) << (fixed ? "" : fixed.error());
  Compare((*shared)->result, (*legacy)->result, rows);
  fs::remove_all((*legacy)->scratch);
  fs::remove_all((*shared)->scratch);
  // Release the earlier result while the final stopped Server still pays
  // both results. Lifetime destroys its own Result before its Server.
  legacy->reset();
}

// No artifact, runner registration, model binding or device work is needed
// to discriminate these early refusal/payment paths.
TEST(Qwen38EngineServingGpu, StartupPaymentPrecedesFailedSetupAndStaysOwned) {
  auto life = std::make_unique<Lifetime>();
  cfg::ModelEntry entry;
  entry.name = "missing";
  entry.artifact = std::string(64, '0');
  life->config.models.push_back(entry);
  life->roles.installed = "/a-nonexistent-llmp-solo-control-store";
  life->options.diagnostic_startup_host_bytes = kAllowance;
  life->options.diagnostic_budget_cap_bytes = kControlBudget;
  life->log = std::tmpfile();
  ASSERT_NE(life->log, nullptr);
  life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, life->log);
  const auto failed = life->server->Start(true);
  EXPECT_FALSE(failed);
  if (!failed) {
    EXPECT_EQ(failed.error(), "model missing: artifact " + *entry.artifact +
                                  ": file-type: the artifact root must be a directory");
  }
  EXPECT_EQ(life->server->Find("missing"), nullptr);
  const auto occupied =
      life->server->node().catalog().OccupancyOf(life->server->node().domain()).Total().value();
  EXPECT_GE(occupied, kAllowance) << (failed ? "unexpected successful setup" : failed.error());
  const auto retired = life->server->TearDown();
  if (!retired) {
    std::ignore = life.release();
    FAIL() << "failed setup owner retained after unproven teardown";
  }
  EXPECT_GE(
      life->server->node().catalog().OccupancyOf(life->server->node().domain()).Total().value(),
      kAllowance);
}

TEST(Qwen38EngineServingGpu, BadPaymentAndSlotCountRefuseBeforeModelEffects) {
  auto life = std::make_unique<Lifetime>();
  cfg::ModelEntry entry;
  entry.name = "missing";
  entry.artifact = std::string(64, '0');
  life->config.models.push_back(entry);
  life->roles.installed = "/a-nonexistent-llmp-solo-control-store";
  life->options.diagnostic_startup_host_bytes = kAllowance + 1;
  life->options.diagnostic_budget_cap_bytes = kControlBudget;
  life->log = std::tmpfile();
  ASSERT_NE(life->log, nullptr);
  life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, life->log);
  const auto failed = life->server->Start(true);
  ASSERT_FALSE(failed);
  EXPECT_NE(failed.error().find("startup host allowance"), std::string::npos);
  EXPECT_EQ(life->server->Find("missing"), nullptr);
  EXPECT_EQ(
      life->server->node().catalog().OccupancyOf(life->server->node().domain()).Total().value(),
      0U);
  for (const auto count : {0U, 2U}) {
    rt::ModelSettings settings;
    settings.architecture = "qwen4exp";
    settings.max_slots.value = count;
    auto build = rt::ComponentLlmBuild(life->server->node(), life->roles.installed, entry, 0,
                                       {.target_policy = {}, .diagnostic_solo = true});
    const auto refused = build(settings);
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error(),
              "the Qwen component qualification requires exactly one configured slot");
    EXPECT_EQ(life->server->Find("missing"), nullptr);
    EXPECT_EQ(
        life->server->node().catalog().OccupancyOf(life->server->node().domain()).Total().value(),
        0U);
  }
  const auto retired = life->server->TearDown();
  if (!retired) {
    std::ignore = life.release();
    FAIL() << "unproven empty server teardown";
  }
}

// No provider open or artifact read on malformed envelopes. The positive
// request reaches ordinary artifact authentication rather than claiming success.
TEST(Qwen38EngineServingGpu, JoinedScoringRequiresExactOwnedTargetEnvelope) {
  en::PagedNode node{{}};
  cfg::ModelEntry entry;
  entry.name = "missing";
  entry.artifact = std::string(64, '0');
  rt::ModelSettings original;
  original.architecture = "qwen4exp";
  original.max_slots.value = 4;
  original.context.value = 512;
  original.prefill_chunk.value = 2;
  original.wave_read_align.value = 256;
  original.speculation.value = false;
  en::NumericalStage target{.stage = 0,
                            .name = "target",
                            .uses = {en::NumericalUse::kDecode, en::NumericalUse::kScore},
                            .phases = {{.phase = llmp::planner::Phase::kPrefill,
                                        .max_owners = 4,
                                        .max_owner_rows = 2,
                                        .max_total_rows = 8},
                                       {.phase = llmp::planner::Phase::kDecode,
                                        .max_owners = 4,
                                        .max_owner_rows = 1,
                                        .max_total_rows = 4},
                                       {.phase = llmp::planner::Phase::kScore,
                                        .max_owners = 4,
                                        .max_owner_rows = 2,
                                        .max_total_rows = 8}}};
  for (int mutation = 0; mutation < 12; ++mutation) {
    SCOPED_TRACE(mutation);
    auto settings = original;
    auto declared = target;
    auto model = entry;
    rt::ComponentBuildOptions options{.target_policy = std::nullopt, .diagnostic_solo = true};
    switch (mutation) {
      case 1:
        settings.max_slots.value = 3;
        break;
      case 2:
        settings.context.value = 513;
        break;
      case 3:
        settings.prefill_chunk.value = 8;
        break;
      case 4:
        settings.wave_read_align.value = 512;
        break;
      case 5:
        settings.speculation.value = true;
        break;
      case 6:
        model.drafter = std::string(64, '1');
        break;
      case 7:
        declared.name = "other";
        break;
      case 8:
        declared.phases.back().max_owner_rows = 3;
        break;
      case 9:
        declared.phases.back().row_invariant = true;
        break;
      case 10:
        options.diagnostic_solo = false;
        break;
      default:
        break;
    }
    if (mutation != 11) {
      auto domain = en::NumericalDomain::Create({std::move(declared)});
      ASSERT_TRUE(domain);
      options.execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*domain));
    }
    auto build = rt::ComponentLlmBuild(node, "/nonexistent-qwen-scoring-control", model, 0,
                                       std::move(options));
    auto result = build(settings);
    ASSERT_FALSE(result);
    if (mutation == 0)
      EXPECT_NE(result.error().find("artifact"), std::string::npos);
    else
      EXPECT_EQ(result.error(),
                "the Qwen component qualification requires exactly one configured slot");
  }
}

// Only the complete semantic declaration reaches the existing artifact opener.
// No provider/model payload or GPU is touched by these early negatives.
TEST(Qwen38EngineServingGpu, JoinedMtpRequiresOwnedWholeCompositionBeforeArtifacts) {
  en::PagedNode node{{}};
  cfg::ModelEntry entry;
  entry.name = "missing";
  entry.artifact = std::string(64, '0');
  entry.drafter = std::string(64, '1');
  rt::ModelSettings original;
  original.architecture = "qwen4exp";
  original.max_slots.value = 2;
  original.context.value = 512;
  original.prefill_chunk.value = 16;
  original.draft_rows.value = 2;
  original.speculation.value = true;
  for (int mutation = 0; mutation != 7; ++mutation) {
    auto settings = original;
    auto model = entry;
    std::vector<en::NumericalStage> stages;
    for (std::uint16_t stage = 0; stage != 2; ++stage)
      stages.push_back(
          {.stage = stage,
           .name = stage == 0 ? "target" : "mtp",
           .uses = {en::NumericalUse::kPromptCompanions, en::NumericalUse::kSpeculative,
                    en::NumericalUse::kJoinedRepeatPending},
           .phases = {{.phase = llmp::planner::Phase::kDraft,
                       .max_owners = 2,
                       .max_owner_rows = 16,
                       .max_total_rows = 32}}});
    // Actual phase/layout eligibility is checked by the real source seal;
    // the factory's first guard checks operation scope before opening files.
    if (mutation == 1) stages[1].uses.pop_back();
    if (mutation == 2) stages.pop_back();
    if (mutation == 3) settings.speculation.value = false;
    if (mutation == 4) model.drafter.reset();
    if (mutation == 5) settings.draft_rows.value = 0;
    auto domain = en::NumericalDomain::Create(std::move(stages));
    ASSERT_TRUE(domain);
    rt::ComponentBuildOptions options{
        .target_policy = std::nullopt,
        .diagnostic_solo = mutation != 6,
        .execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*domain))};
    auto build = rt::ComponentLlmBuild(node, "/nonexistent-joined-mtp-control", model, 0,
                                       std::move(options));
    auto result = build(settings);
    ASSERT_FALSE(result);
    if (mutation == 0)
      EXPECT_NE(result.error().find("artifact"), std::string::npos);
    else
      EXPECT_EQ(result.error(),
                "the Qwen component qualification requires exactly one configured slot");
  }
}

TEST(Qwen38EngineServingGpu, NativeMtpDepthTwoFullLogits) { Screen(kCutlass, true, true, 2); }
TEST(Qwen38EngineServingGpu, NativeMtpDefaultTokenSchedule) { Screen(kCutlass, true, false, 3); }
TEST(Qwen38EngineServingGpu, NativeGgmlExpertsPlain) { Screen(kGgml, false, true, 2); }
TEST(Qwen38EngineServingGpu, GgufPlain) { Screen(kGguf, false, true, 2); }

}  // namespace
