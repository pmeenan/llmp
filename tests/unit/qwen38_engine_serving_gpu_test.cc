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
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
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
#include "engine/plan_cache.h"
#include "execution/registry.h"
#include "kernels/common/step_program.h"
#include "kernels/ggml/graph_plan.h"
#include "model/qwen38.h"
#include "planner/plan.h"
#include "platform/host_probe.h"
#include "platform/kept_files.h"
#include "runtime/component_build.h"
#include "runtime/engine_llm.h"
#include "runtime/memory_guard.h"
#include "runtime/serving.h"
#include "runtime/snapshot_store.h"
#include "runtime/state_keeper.h"
#include "spec/mappings/qwen38.h"
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
  for (const bool prompt : {false, true}) {
    SCOPED_TRACE(prompt);
    for (int mutation = 0; mutation < 14; ++mutation) {
      SCOPED_TRACE(mutation);
      auto settings = original;
      auto declared = target;
      if (prompt) declared.uses.insert(declared.uses.begin(), en::NumericalUse::kPrompt);
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
        case 12:
          declared.uses.push_back(en::NumericalUse::kPromptCompanions);
          break;
        case 13:
          std::erase(declared.uses, en::NumericalUse::kDecode);
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

TEST(Qwen38EngineServingGpu, DerivedServingDomainRequiresAnExplicitComposedRequest) {
  en::PagedNode node{{}};
  cfg::ModelEntry entry;
  entry.name = "missing";
  entry.artifact = std::string(64, '0');
  entry.drafter = std::string(64, '1');
  for (int mutation = 0; mutation != 8; ++mutation) {
    auto model = entry;
    rt::ModelSettings settings;
    settings.architecture = "qwen4exp";
    settings.max_slots.value = 2;
    settings.context.value = 512;
    settings.prefill_chunk.value = 16;
    settings.draft_rows.value = 2;
    settings.speculation.value = true;
    rt::ComponentBuildOptions options{
        .target_policy = {}, .diagnostic_solo = true, .derive_execution_domain = true};
    switch (mutation) {
      case 1:
        options.derive_execution_domain = false;
        break;
      case 2:
        options.diagnostic_solo = false;
        break;
      case 3:
        model.drafter.reset();
        break;
      case 4:
        settings.speculation.value = false;
        break;
      case 5:
        settings.context.value = 0;
        break;
      case 6:
        settings.prefill_chunk.value = 0;
        break;
      case 7:
        settings.draft_rows.value = 0;
        break;
      default:
        break;
    }
    auto build = rt::ComponentLlmBuild(node, "/nonexistent-derived-serving-control", model, 0,
                                       std::move(options));
    const auto result = build(settings);
    ASSERT_FALSE(result);
    if (mutation == 0)
      EXPECT_NE(result.error().find("artifact"), std::string::npos);
    else
      EXPECT_EQ(result.error().find("artifact"), std::string::npos);
    EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), 0U);
  }
}

TEST(Qwen38EngineServingGpu, NativeMtpDepthTwoFullLogits) { Screen(kCutlass, true, true, 2); }
TEST(Qwen38EngineServingGpu, NativeMtpDefaultTokenSchedule) { Screen(kCutlass, true, false, 3); }
TEST(Qwen38EngineServingGpu, NativeGgmlExpertsPlain) { Screen(kGgml, false, true, 2); }
TEST(Qwen38EngineServingGpu, GgufPlain) { Screen(kGguf, false, true, 2); }

// A manually enabled durable C2 control in the existing real EngineLlm fixture.
// It uses the same selected implementation in two sequential whole owners.
namespace c2 {
namespace pl = llmp::planner;
namespace kn = llmp::kernels;
namespace kept = rt::kept;
using Digest = llmp::base::Sha256Digest;
constexpr std::uint64_t kN = llmp::ir::Graph::kMaxNodes;
constexpr std::uint64_t kT = llmp::ir::Graph::kMaxTensors;
constexpr std::uint64_t kSaves = 5 * 48 * 2;
constexpr std::uint64_t kDescriptors = kT + 16 + 31 * kN + 4 * kSaves;
// Current shared compiler graph/rewrite/placement/descriptor overlap, plus
// the existing authenticated artifact/tokenizer/catalog owners above. Retained
// programs are measured and separately admitted; no launch-count guess.
constexpr std::uint64_t kCompile =
    12 * (kN * 480 + kT * 192) + 4096 * (kN + kT) + kDescriptors * (512 + 1024 + 32) + 64 * kMiB;
constexpr std::uint64_t kObserve = 128 * kMiB;
constexpr std::uint64_t kStartup = kCompile + kServingDelta + kObserve;
static_assert(kCompile == 4946022912ULL);
static_assert(3 * 2 * 8 * kVocab * sizeof(float) + 32 * kMiB < kObserve);

struct Plan {
  pl::StepShape shape;
  std::vector<std::uint64_t> parameters;
  Digest recipe{};
  std::uint64_t host = 0, launches = 0, lowered = 0;
};
using StateRanges = std::vector<std::array<std::uint64_t, 4>>;
struct Evidence {
  std::array<rt::Generation, 2> prefix, final;
  std::array<std::vector<std::int32_t>, 2> history;
  std::array<Digest, 2> boundary{}, end{};
  std::array<StateRanges, 2> boundary_ranges, end_ranges;
  std::array<llmp::model::DraftProgress, 2> progress{};
  std::array<kept::Record, 2> records;
  kept::Identity identity;
};
struct Owner {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  rt::RequestMemory tokens{kMiB};
  std::shared_ptr<en::ExecutionMetadataAllowance> startup, plans;
  cfg::RuntimeRoles roles;
  cfg::ModelEntry entry;
  std::unique_ptr<rt::StateKeeper> keeper;
  std::unique_ptr<rt::EngineLlm> llm;
  en::StepSource* source = nullptr;
  std::vector<Plan> measured;
  std::vector<pl::StepShape> classes, runnable;
  std::array<std::unique_ptr<rt::SnapshotStore>, 2> snapshots;
  std::array<rt::StateSnapshot, 2> saved;
  std::unique_ptr<rt::SnapshotStore> digest_store;
  std::vector<en::PagedModel*> entered;
  std::shared_ptr<Evidence> evidence;
  std::size_t keeper_model = 0;
  int directory = -1;
  std::uint64_t snapshot_bound = 0, plan_bytes = 0, physical = 0;
  bool admitted = false, attempted = false, retired = false;
  bool Retire() {
    if (attempted) return retired;
    attempted = true;
    if (node.InRequest(0) && !node.EndRequest(0)) return false;
    if (keeper && !keeper->Drain(rt::Clock::now() + std::chrono::seconds(30))) return false;
    snapshots = {};
    digest_store.reset();
    if (!node.TearDown(entered)) return false;
    retired = true;
    return true;
  }
  ~Owner() {
    llm.reset();
    keeper.reset();
    if (directory >= 0) ::close(directory);
  }
  rt::Llm::Branch& Branch(std::size_t i) { return **llm->branch(i); }
};
struct RetainUnknown {
  void operator()(Owner* owner) const {
    if (owner->Retire())
      delete owner;
    else
      ADD_FAILURE() << "retaining the complete unknown C2 owner; no next owner is admitted";
  }
};
using Owned = std::unique_ptr<Owner, RetainUnknown>;

bool AddBytes(std::uint64_t& sum, std::uint64_t bytes) {
  if (bytes > kBudget - sum) return false;
  sum += bytes;
  return true;
}
rt::Status Observe(Owner& owner, const kn::LowerInput& in, const kn::StepProgram& program) {
  if (!in.token || in.token->context != 512 || in.token->shape.stage > 1 ||
      !in.plan.owner_products.empty() || in.token->step.saves.size() > kSaves ||
      in.plan.owner_rows_policy_enabled != (in.token->shape.stage == 1))
    return std::unexpected("C2 source/policy left its declared complete domain");
  const auto& shape = in.token->shape;
  if (shape.segments.empty() || shape.segments.size() > 2)
    return std::unexpected("C2 owner geometry changed");
  auto parameters = owner.source->ShapeParameters(shape);
  llmp::base::Sha256 hash;
  const auto graph = in.plan.graph.Digest();
  hash.Update(std::as_bytes(std::span(graph)));
  for (const auto& selected : in.plan.steps) {
    hash.Update(std::as_bytes(std::span(&selected.implementation, 1)));
    hash.Update(std::as_bytes(std::span(selected.nodes)));
  }
  const auto recipe = hash.Finish();
  auto at = std::ranges::find(owner.measured, shape, &Plan::shape);
  if (in.addresses.measuring()) {
    if (at == owner.measured.end()) {
      // One record per explicit setup key, plus ModelRunner's three target
      // commit keys. Other shapes cannot grow an unbounded observer table.
      if (owner.measured.size() == owner.classes.size() + 3)
        return std::unexpected("C2 setup emitted an undeclared extra class");
      owner.measured.push_back({.shape = shape,
                                .parameters = std::move(parameters),
                                .recipe = recipe,
                                .host = program.host_bytes(),
                                .launches = program.launches()});
      return {};
    }
  } else if (!owner.admitted || at == owner.measured.end() ||
             std::ranges::find(owner.runnable, shape) == owner.runnable.end()) {
    return std::unexpected("C2 bound program lacks measured admission");
  }
  if (at == owner.measured.end() || at->parameters != parameters || at->recipe != recipe ||
      program.host_bytes() > at->host || program.launches() > at->launches)
    return std::unexpected("C2 actual program differs from its paid complete key/selection");
  if (!in.addresses.measuring()) ++at->lowered;
  return {};
}

rt::Status AddClasses(rt::EngineLlmParts& parts, Owner& owner) {
  auto composer = en::WaveComposer::Make(
      parts.caps, {{.context = 512, .read_align = 256}, {.context = 512, .read_align = 256}});
  if (!composer) return std::unexpected(composer.error());
  owner.classes = parts.config.setup_shapes;
  const auto add = [&](const pl::StepShape& shape) {
    if (std::ranges::find(owner.classes, shape) == owner.classes.end())
      owner.classes.push_back(shape);
    if (std::ranges::find(owner.runnable, shape) == owner.runnable.end())
      owner.runnable.push_back(shape);
  };
  const auto compose = [&](std::uint16_t stage, pl::Phase phase, std::uint32_t a, std::uint32_t b,
                           int outputs, bool greedy) -> rt::Status {
    std::vector<en::PendingWork> work;
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      const auto rows = slot == 0 ? a : b;
      if (rows)
        work.push_back({.slot = slot,
                        .stage = stage,
                        .phase = phase,
                        .past = 0,
                        .rows = rows,
                        .outputs = outputs < 0 ? rows : static_cast<std::uint32_t>(outputs),
                        .want_head = outputs != 0,
                        .greedy = greedy});
    }
    auto composed = composer->Compose(work);
    if (!composed.refused.empty()) return std::unexpected("C2 declared class cannot compose");
    for (const auto& step : composed.steps) add(step.shape);
    return {};
  };
  // Only the fixed initial prompt uses large rows. Generation's depth two
  // permits one-to-three verify/replay rows, single or joined as owners finish.
  for (const auto [a, b] : {std::pair{7U, 0U}, {0U, 5U}})
    if (auto s = compose(0, pl::Phase::kPrefill, a, b, 1, false); !s) return s;
  for (const auto [a, b] : {std::pair{6U, 0U}, {0U, 4U}})
    if (auto s = compose(1, pl::Phase::kDraft, a, b, 0, false); !s) return s;
  for (std::uint32_t a = 0; a <= 3; ++a)
    for (std::uint32_t b = 0; b <= 3; ++b) {
      if (!a && !b) continue;
      if (auto s = compose(0, pl::Phase::kVerify, a, b, -1, false); !s) return s;
      if (auto s = compose(1, pl::Phase::kDraft, a, b, 1, true); !s) return s;
      if (auto s = compose(1, pl::Phase::kDraft, a, b, 0, false); !s) return s;
      if (a <= 1 && b <= 1)
        if (auto s = compose(0, pl::Phase::kDecode, a, b, 1, false); !s) return s;
    }
  for (std::uint8_t slot = 0; slot < 2; ++slot)
    for (std::uint32_t rows = 1; rows <= 3; ++rows)
      add({.stage = 0,
           .phase = pl::Phase::kCommit,
           .segments = {{.slot = slot, .rows = rows, .read_cells = 512, .want_head = false}}});
  parts.config.setup_shapes = owner.classes;
  return {};
}

std::expected<rt::ComponentBuildOptions, std::string> Options(Owner& owner) {
  auto policy = pl::ParsePolicy(llmp::spec::mappings::Qwen38Mapping().policies.front().text);
  if (!policy) return std::unexpected(policy.error());
  if (std::erase(policy->enabled, "owner_row_invariant") != 1)
    return std::unexpected("qualified target mapping policy changed");
  policy->disabled.emplace_back("owner_row_invariant");
  pl::Canonicalize(*policy);
  std::vector<en::NumericalStage> stages;
  for (std::uint16_t stage = 0; stage < 2; ++stage) {
    en::NumericalStage s{
        .stage = stage,
        .name = stage ? "mtp" : "target",
        .uses = {en::NumericalUse::kPrompt, en::NumericalUse::kDecode,
                 en::NumericalUse::kPromptCompanions, en::NumericalUse::kSpeculative,
                 en::NumericalUse::kJoinedRepeatPending},
        .phases = {}};
    const auto add = [&](pl::Phase phase, std::uint32_t rows) {
      s.phases.push_back(
          {.phase = phase, .max_owners = 2, .max_owner_rows = rows, .max_total_rows = 2ULL * rows});
    };
    if (stage)
      add(pl::Phase::kDraft, 16);
    else {
      add(pl::Phase::kPrefill, 16);
      add(pl::Phase::kDecode, 1);
      add(pl::Phase::kVerify, 3);
      add(pl::Phase::kCommit, 3);
    }
    stages.push_back(std::move(s));
  }
  auto domain = en::NumericalDomain::Create(std::move(stages));
  if (!domain) return std::unexpected(domain.error());
  rt::ComponentBuildOptions options{
      .target_policy = pl::ToText(*policy),
      .diagnostic_solo = true,
      .prefill_lookahead = false,
      .prepare_state = false,
      .state_only_prefill = false,
      .execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*domain))};
  options.validate_program = [&owner](const auto& in, const auto& program) {
    return Observe(owner, in, program);
  };
  testing::Test::RecordProperty("c2_target_policy", *options.target_policy);
  testing::Test::RecordProperty(
      "c2_drafter_policy",
      std::string(llmp::spec::mappings::Qwen38MtpReplayMapping().policies.front().text));
  return options;
}

rt::Status Setup(Owner& owner, const fs::path& directory, Evidence& evidence, bool reopen,
                 Owned* previous = nullptr) {
  owner.node.SetHostFloor(0);
  if (auto s = owner.node.Open(); !s) return s;
  auto prepaid = en::ExecutionMetadataAllowance::Admit(
      owner.node.catalog(), owner.node.domain(),
      kStartup + en::ExecutionMetadataAllowance::OverheadBytes(),
      llmp::platform::AvailableMemoryBytes().value_or(0), rt::kUncountedMargin);
  if (!prepaid) return std::unexpected(prepaid.error());
  owner.startup = std::move(*prepaid);
  if (!owner.startup->Charge(kStartup)) return std::unexpected("C2 startup payment refused");
  if (previous != nullptr) previous->reset();
  if (auto s = CheckFixture(kCutlass, true); !s) return s;
  owner.roles.installed = fs::path(llmp::test_support::ModelsDir()) / "m3-artifacts";
  owner.roles.spill = directory;
  owner.roles.state = directory;
  owner.entry.name = "qwen-c2-durable";
  owner.entry.artifact = kCutlass;
  owner.entry.drafter = kMtp;
  const auto checkpoint = fs::path(llmp::test_support::ModelsDir()) /
                          "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6";
  owner.entry.tokenizer = checkpoint / "tokenizer.json";
  owner.entry.chat_template = checkpoint / "chat_template.jinja";
  rt::ModelSettings settings;
  settings.name = owner.entry.name;
  settings.architecture = "qwen4exp";
  settings.max_slots.value = 2;
  settings.context.value = 512;
  settings.prefill_chunk.value = 16;
  settings.wave_read_align.value = 256;
  settings.speculation.value = true;
  settings.draft_rows.value = 2;
  settings.draft_vocab.value = 65536;
  auto options = Options(owner);
  if (!options) return std::unexpected(options.error());
  auto build =
      rt::ComponentLlmBuild(owner.node, owner.roles.installed, owner.entry, 0, std::move(*options));
  auto inspect = [&owner, build = std::move(build)](const rt::ModelSettings& actual) mutable
      -> std::expected<rt::EngineLlmParts, std::string> {
    auto parts = build(actual);
    if (!parts) return std::unexpected(parts.error());
    if (parts->config.slots.stages.size() != 2 || parts->mode.drafters.size() != 1 ||
        parts->mode.row_invariant || parts->logit_width != kVocab)
      return std::unexpected("C2 factory changed stages/mode/head");
    const auto width = en::DrafterHeadRows(parts->mode.drafters.front(), kVocab);
    if (!width || *width != 47172 || !parts->mode.drafters.front().device_tokens)
      return std::unexpected("C2 factory changed the actual selected device head");
    owner.source = parts->source.get();
    if (auto s = AddClasses(*parts, owner); !s) return std::unexpected(s.error());
    return std::move(*parts);
  };
  owner.llm = std::make_unique<rt::EngineLlm>(owner.node, owner.entry, settings, owner.roles, 0,
                                              std::move(inspect));
  owner.llm->SetTokenMemory(owner.tokens);
  owner.llm->set_turn_checkpoints(false);
  if (auto s = owner.llm->Setup(); !s) {
    if (owner.llm->has_paged()) owner.entered.push_back(&owner.llm->paged());
    return s;
  }
  owner.entered.push_back(&owner.llm->paged());
  const auto identity = owner.llm->ExecutionIdentity();
  if (!identity || !*identity || !owner.source->JoinedRepeatPending(0) ||
      !owner.source->JoinedRepeatPending(1))
    return std::unexpected("C2 source lacks prepared complete pending identity");
  std::uint64_t largest = 0;
  for (const auto& plan : owner.measured) {
    if (std::ranges::find(owner.runnable, plan.shape) == owner.runnable.end()) continue;
    if (plan.host > kBudget || plan.launches > (kBudget - plan.host) / en::kGraphNodeHostBytes)
      return std::unexpected("C2 measured capture payment overflows domain");
    const auto bytes = plan.host + plan.launches * en::kGraphNodeHostBytes;
    if (!AddBytes(owner.plan_bytes, bytes))
      return std::unexpected("C2 measured plans exceed domain");
    largest = std::max(largest, bytes);
  }
  if (!AddBytes(owner.plan_bytes, largest))
    return std::unexpected("C2 replacement payment overflow");
  auto paid_plans = en::ExecutionMetadataAllowance::Admit(
      owner.node.catalog(), owner.node.domain(),
      owner.plan_bytes + en::ExecutionMetadataAllowance::OverheadBytes(),
      llmp::platform::AvailableMemoryBytes().value_or(0), rt::kUncountedMargin);
  if (!paid_plans) return std::unexpected(paid_plans.error());
  owner.plans = std::move(*paid_plans);
  if (!owner.plans->Charge(owner.plan_bytes)) return std::unexpected("C2 measured payment refused");
  // EngineLlm::UsedStateRangesFor forwards LiveState::used_ranges: at
  // most one descriptor per actual 2 MiB extent, including each final partial
  // extent. SnapshotStore::Prepare adds its nine-u64 header, cursor, optional
  // execution digest, adaptive words, ranges, history and actual layout text.
  // Four max-align paddings cover all header/internal/following alignments.
  std::uint64_t ranges = 0;
  for (const auto bytes : owner.llm->KeptLayouts()) {
    if (!AddBytes(owner.snapshot_bound, bytes)) return std::unexpected("C2 snapshot overflow");
    ranges += (bytes + en::kPagedExtent - 1) / en::kPagedExtent;
  }
  const auto metadata = 9 * sizeof(std::uint64_t) + sizeof(std::uint32_t) +
                        sizeof(std::optional<Digest>) +
                        sizeof(std::array<std::uint64_t, llmp::execution::AdaptiveDepth::kWords>) +
                        4 * alignof(std::max_align_t) + ranges * sizeof(en::LiveState::Range) +
                        512 * sizeof(std::int32_t) + owner.llm->KeptLayout().size();
  if (!AddBytes(owner.snapshot_bound, metadata))
    return std::unexpected("C2 snapshot metadata overflow");
  const auto snapshot_physical =
      (owner.snapshot_bound + en::kPagedExtent - 1) / en::kPagedExtent * en::kPagedExtent;
  for (const auto& [id, extent] : owner.node.catalog().ExtentsOf(owner.node.domain())) {
    (void)id;
    if (!AddBytes(owner.physical, extent.descriptor.size.value()))
      return std::unexpected("C2 full catalog exceeds node");
  }
  const auto pool = std::max(owner.llm->pool_needed(), en::kPagedExtent);
  for (const auto bytes :
       {owner.llm->activations_needed(), pool, owner.plan_bytes, snapshot_physical,
        snapshot_physical, snapshot_physical, 2 * en::kPagedExtent})
    if (!AddBytes(owner.physical, bytes))
      return std::unexpected("C2 complete physical closure does not fit");
  const auto available = llmp::platform::AvailableMemoryBytes().value_or(0);
  if (!available || owner.physical > available || rt::kUncountedMargin > available - owner.physical)
    return std::unexpected("C2 complete future closure lacks physical headroom");
  owner.admitted = true;
  if (auto s = owner.node.MapWorkspace(owner.llm->activations_needed(), pool); !s) return s;
  if (auto s = owner.node.Start(llmp::base::Bytes(kBudget)); !s) return s;
  auto dir = llmp::platform::OpenPrivateDirectory(-1, directory.c_str());
  if (!dir) return std::unexpected(std::format("C2 private directory: errno {}", dir.error()));
  owner.directory = *dir;
  owner.keeper = std::make_unique<rt::StateKeeper>(1, rt::StateKeeper::Log{});
  const int worker_dir = ::dup(owner.directory);
  if (worker_dir < 0) return std::unexpected("C2 keeper directory duplication failed");
  owner.keeper_model = owner.keeper->AddModel(worker_dir);
  kept::Identity expected{.build = "qwen-c2-owned-durable-v1",
                          .artifact = std::string(kCutlass),
                          .drafter = std::string(kMtp),
                          .layout = owner.llm->KeptLayout()};
  rt::Llm::Kept state{.keeper = owner.keeper.get(),
                      .model = owner.keeper_model,
                      .directory = owner.directory,
                      .identity = expected,
                      .adopt = {}};
  if (reopen) {
    if (expected != evidence.identity)
      return std::unexpected("C2 fresh execution/layout identity differs");
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      auto& record = evidence.records[slot];
      kept::Expected check{.version = owner.llm->KeptVersion(),
                           .identity = expected,
                           .slot = slot,
                           .regions = owner.llm->KeptRegions(),
                           .layouts = owner.llm->KeptLayouts(),
                           .context = 512,
                           .vocabulary = kVocab,
                           .now_unix_ms = record.used_unix_ms,
                           .retention_ms = 60000};
      if (auto s = kept::Check(record, check); !s) return s;
      if (const auto why = rt::VerifyKeptState(owner.directory, record); !why.empty())
        return std::unexpected(why);
      if (record.files.size() != 2) return std::unexpected("C2 record omitted a stage file");
      state.adopt.push_back(slot);
      auto& files = state.files.emplace_back();
      for (const auto& file : record.files) files.push_back(file.id);
    }
  } else
    evidence.identity = expected;
  owner.llm->set_kept(std::move(state));
  if (auto s = owner.llm->Register(); !s) return s;
  owner.node.Run();
  if (auto s = owner.llm->Bind(); !s) return s;
  owner.digest_store = std::make_unique<rt::SnapshotStore>(owner.node, owner.snapshot_bound);
  testing::Test::RecordProperty(reopen ? "c2_reopened_plans_bytes" : "c2_initial_plans_bytes",
                                std::to_string(owner.plan_bytes));
  testing::Test::RecordProperty(reopen ? "c2_reopened_physical_bytes" : "c2_initial_physical_bytes",
                                std::to_string(owner.physical));
  return {};
}

// Sessions must close even when a later owner/session or wave refuses.
// These callbacks return only at completed units; production owns retirement.
template <class Session>
struct FinishSession {
  void operator()(Session* session) const {
    if (!session->done()) session->Cancel();
    (void)session->Finish();
    delete session;
  }
};
template <class Session>
using SessionOwner = std::unique_ptr<Session, FinishSession<Session>>;
rt::Status Request(Owner& owner, const std::function<rt::Status()>& body) {
  const std::array<rt::Llm::Branch*, 2> branches{&owner.Branch(0), &owner.Branch(1)};
  if (auto s = owner.llm->SelectBranches(branches); !s) return s;
  return owner.node.WithRequest(0, owner.llm->request_closure(), "C2 durable control", body);
}
std::expected<Digest, std::string> StateDigest(Owner& owner, std::size_t slot) {
  auto snapshot = owner.Branch(slot).CaptureState(*owner.digest_store);
  if (!snapshot) return std::unexpected(snapshot.error());
  return owner.digest_store->Digest(*snapshot);
}
StateRanges Coordinates(Owner& owner, std::size_t slot) {
  StateRanges result;
  for (std::size_t stage = 0; stage < 2; ++stage)
    for (const auto& range :
         owner.llm->runner().slots().live(static_cast<std::uint32_t>(slot), stage).used_ranges())
      result.push_back({stage, range.region, range.offset, range.bytes});
  return result;
}
rt::Status Complete(const rt::Generation& out, std::size_t count, bool keep_logits = true) {
  if (out.tokens.size() != count ||
      (keep_logits ? out.logits.size() != count : !out.logits.empty()) || !out.drafted ||
      out.stopped || out.cancelled)
    return std::unexpected("C2 generation omitted target rows or actual MTP work");
  for (std::size_t i = 0; i < count; ++i) {
    if (out.tokens[i] < 0 || std::cmp_greater_equal(out.tokens[i], kVocab) ||
        (keep_logits &&
         (out.logits[i].size() != kVocab ||
          !std::ranges::all_of(out.logits[i], [](float x) { return std::isfinite(x); }))))
      return std::unexpected("C2 completed target head is not finite and complete");
  }
  return {};
}
template <class OwnerLike>
rt::Status Generate(OwnerLike& owner, const std::array<std::vector<float>, 2>& heads,
                    std::array<rt::Generation, 2>& out, std::uint32_t count, bool resume,
                    bool keep_logits = true) {
  rt::GenerateOptions options;
  options.max_tokens = count;
  options.stop = false;
  options.keep_logits = keep_logits;
  std::array<SessionOwner<rt::Llm::GenerationSession>, 2> sessions;
  for (std::size_t i = 0; i < 2; ++i) {
    auto session = resume ? owner.Branch(i).ResumeGeneration(heads[i], options, out[i])
                          : owner.Branch(i).BeginGeneration(heads[i], options, out[i]);
    if (!session) return std::unexpected(session.error());
    sessions[i].reset(session->release());
  }
  // Every successful generation unit publishes at least one next token.
  for (std::uint32_t wave = 0; wave < 2 * count; ++wave) {
    std::vector<rt::Llm::GenerationSession*> active;
    for (auto& session : sessions)
      if (!session->done()) active.push_back(session.get());
    if (active.empty()) break;
    if (auto s = owner.llm->RunGenerationWave(active); !s) return s;
  }
  for (std::size_t i = 0; i < 2; ++i) {
    if (!sessions[i]->done()) return std::unexpected("C2 generation made no bounded progress");
    if (auto s = sessions[i]->Finish(); !s) return s;
    if (auto s = Complete(out[i], count, keep_logits); !s) return s;
  }
  return {};
}
rt::Status Initial(Owner& owner, Evidence& evidence) {
  return Request(owner, [&]() -> rt::Status {
    auto first = Prompt(), second = Prompt();
    second.resize(5);
    for (auto& token : second) token += 17;
    const std::array<std::vector<std::int32_t>, 2> prompts{std::move(first), std::move(second)};
    std::array<SessionOwner<rt::Llm::PromptSession>, 2> sessions;
    std::array<std::vector<float>, 2> heads;
    for (std::size_t i = 0; i < 2; ++i) {
      auto session = owner.Branch(i).BeginPrompt(prompts[i]);
      if (!session) return std::unexpected(session.error());
      sessions[i].reset(session->release());
      while (!sessions[i]->done()) {
        const auto unit = sessions[i]->NextUnit();
        if (!unit) return std::unexpected(unit.error());
        if (unit->phase == rt::Llm::PromptSession::Phase::kChunk) break;
        if (auto s = sessions[i]->Advance(); !s) return s;
      }
    }
    // Llm's prompt-cohort API is plain-only. The ordinary session advance
    // owns speculative companion prefill; joined generation follows below.
    for (auto& session : sessions)
      if (!session->done())
        if (auto s = session->Advance(); !s) return s;
    for (std::size_t i = 0; i < 2; ++i) {
      if (!sessions[i]->done()) return std::unexpected("C2 bounded prompt did not finish");
      heads[i] = sessions[i]->last();
      if (auto s = sessions[i]->Finish(); !s) return s;
      sessions[i].reset();
    }
    if (auto s = Generate(owner, heads, evidence.prefix, 4, false); !s) return s;
    for (std::size_t i = 0; i < 2; ++i) {
      owner.snapshots[i] = std::make_unique<rt::SnapshotStore>(owner.node, owner.snapshot_bound);
      auto saved = owner.Branch(i).CaptureState(*owner.snapshots[i]);
      if (!saved) return std::unexpected(saved.error());
      owner.saved[i] = std::move(*saved);
      auto digest = owner.snapshots[i]->Digest(owner.saved[i]);
      if (!digest) return std::unexpected(digest.error());
      evidence.boundary[i] = *digest;
      evidence.boundary_ranges[i] = Coordinates(owner, i);
      evidence.history[i] = owner.Branch(i).history();
      const auto* progress = owner.llm->runner().slots().progress(static_cast<std::uint32_t>(i), 1);
      if (!progress || !progress->settled())
        return std::unexpected("C2 draft progress is not settled");
      evidence.progress[i] = progress->completed();
      if (evidence.progress[i].pending_first >= evidence.progress[i].positions)
        return std::unexpected("C2 boundary did not preserve a real repeat-pending interval");
    }
    evidence.final = evidence.prefix;
    if (auto s = Generate(owner, {}, evidence.final, 8, true); !s) return s;
    for (std::size_t i = 0; i < 2; ++i) {
      auto digest = StateDigest(owner, i);
      if (!digest) return std::unexpected(digest.error());
      evidence.end[i] = *digest;
      evidence.end_ranges[i] = Coordinates(owner, i);
      if (auto s = owner.Branch(i).RestoreState(*owner.snapshots[i], owner.saved[i]); !s) return s;
      if (owner.Branch(i).history() != evidence.history[i])
        return std::unexpected("C2 snapshot lost its history boundary");
      digest = StateDigest(owner, i);
      if (!digest || *digest != evidence.boundary[i] ||
          Coordinates(owner, i) != evidence.boundary_ranges[i])
        return std::unexpected("C2 restored boundary initialized state differs");
      const auto* progress = owner.llm->runner().slots().progress(static_cast<std::uint32_t>(i), 1);
      if (!progress || progress->completed() != evidence.progress[i])
        return std::unexpected("C2 restored pending progress differs");
    }
    if (!owner.llm->counts().joined_waves)
      return std::unexpected("C2 did not join generation owners");
    return {};
  });
}
rt::Status Keep(Owner& owner, Evidence& evidence) {
  owner.snapshots = {};
  owner.digest_store.reset();
  for (std::size_t i = 0; i < 2; ++i)
    if (auto s = owner.llm->SpillIdle(owner.Branch(i)); !s) return s;
  if (!owner.keeper->Drain(rt::Clock::now() + std::chrono::seconds(30)))
    return std::unexpected("C2 keeper did not retire record writes");
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    if (!owner.keeper->Kept(owner.keeper_model, slot))
      return std::unexpected("C2 keeper omitted a complete owner record");
    const auto name = kept::RecordFileName(slot);
    auto text =
        llmp::platform::ReadPrivateFile(owner.directory, name.c_str(), kept::kMostRecordBytes);
    if (!text) return std::unexpected(std::format("C2 kept record read: errno {}", text.error()));
    auto record = kept::Decode(*text);
    if (!record) return std::unexpected(record.error());
    if (record->files.size() != 2 || record->tokens != evidence.history[slot])
      return std::unexpected("C2 kept record lost a stage/history");
    if (const auto why = rt::VerifyKeptState(owner.directory, *record); !why.empty())
      return std::unexpected(why);
    evidence.records[slot] = std::move(*record);
  }
  owner.llm->PreserveKeptFiles();
  return {};
}
rt::Status Reopen(Owner& owner, Evidence& evidence) {
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    rt::MemoryCharge charge;
    if (!charge.Add(owner.tokens, evidence.records[slot].tokens.capacity() * sizeof(std::int32_t)))
      return std::unexpected("C2 adopted token payment refused");
    if (auto s = owner.llm->Adopt(owner.Branch(slot), evidence.records[slot], charge); !s) return s;
    const auto& slots = owner.llm->runner().slots();
    if (!slots.restorable(slot) || slots.usable(slot) || slots.CheckActive(slot))
      return std::unexpected("C2 cold adopted state became executable before hydration");
  }
  return Request(owner, [&]() -> rt::Status {
    const auto before = owner.llm->counts();
    const auto graphs = owner.llm->runner().graph_stats();
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      auto begun = owner.Branch(slot).BeginPrompt(evidence.history[slot], 0, false, true);
      if (!begun) return std::unexpected(begun.error());
      SessionOwner<rt::Llm::PromptSession> session(begun->release());
      const auto unit = session->NextUnit();
      if (!unit || unit->phase != rt::Llm::PromptSession::Phase::kReuse)
        return std::unexpected("C2 cold history did not begin with reuse");
      if (auto s = session->Advance({}, true); !s) return s;
      if (!session->done() || session->reused() != evidence.history[slot].size())
        return std::unexpected("C2 cold hydration attempted prefix recomputation");
      if (auto s = session->Finish(); !s) return s;
      session.reset();
      auto digest = StateDigest(owner, slot);
      if (!digest || *digest != evidence.boundary[slot] ||
          Coordinates(owner, slot) != evidence.boundary_ranges[slot])
        return std::unexpected("C2 durable hydrated initialized state differs");
      const auto* progress = owner.llm->runner().slots().progress(slot, 1);
      if (!progress || !progress->settled() || progress->completed() != evidence.progress[slot])
        return std::unexpected("C2 durable pending state differs");
    }
    const auto after = owner.llm->counts();
    const auto now = owner.llm->runner().graph_stats();
    if (before.joined_prefill_steps != after.joined_prefill_steps ||
        before.speculative_units != after.speculative_units || graphs.eager != now.eager ||
        graphs.replayed != now.replayed)
      return std::unexpected("C2 hydrate executed a model prefix");
    auto continued = evidence.prefix;
    if (auto s = Generate(owner, {}, continued, 8, true); !s) return s;
    for (std::size_t slot = 0; slot < 2; ++slot) {
      const auto& got = continued[slot];
      const auto& want = evidence.final[slot];
      if (got.tokens != want.tokens || got.drafted != want.drafted || got.accepted != want.accepted)
        return std::unexpected("same-selected C2 durable continuation differs");
      for (std::size_t row = 0; row < got.logits.size(); ++row)
        if (std::memcmp(got.logits[row].data(), want.logits[row].data(), kVocab * sizeof(float)))
          return std::unexpected("same-selected C2 durable complete target head differs");
      auto digest = StateDigest(owner, slot);
      if (!digest || *digest != evidence.end[slot] ||
          Coordinates(owner, slot) != evidence.end_ranges[slot])
        return std::unexpected("same-selected C2 final initialized state differs");
    }
    return {};
  });
}

class DurableCase : public testing::Test {
  void TestBody() override {
    const char* root = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    ASSERT_NE(root, nullptr);
    fs::create_directories(root);
    auto path = (fs::path(root) / "qwen-c2-XXXXXX").string();
    ASSERT_NE(::mkdtemp(path.data()), nullptr);
    auto observations = std::make_shared<Evidence>();
    auto& evidence = *observations;
    Owned first(new Owner);
    first->evidence = observations;
    auto result = ::c2::Setup(*first, path, evidence, false);
    ASSERT_TRUE(result) << (result ? "" : result.error());
    result = Initial(*first, evidence);
    ASSERT_TRUE(result) << (result ? "" : result.error());
    result = Keep(*first, evidence);
    ASSERT_TRUE(result) << (result ? "" : result.error());
    ASSERT_TRUE(first->Retire()) << "unknown first-owner completion; reopening is forbidden";
    // Retain the old host payment until the fresh startup allowance exists.
    // Its GPU/source domain is positively retired before the second opens.
    Owned second(new Owner);
    second->evidence = observations;
    result = ::c2::Setup(*second, path, evidence, true, &first);
    ASSERT_TRUE(result) << (result ? "" : result.error());
    result = Reopen(*second, evidence);
    ASSERT_TRUE(result) << (result ? "" : result.error());
    ASSERT_TRUE(second->Retire());
    RecordProperty("durable_owners", 2);
    RecordProperty("durable_stage_files", 4);
    RecordProperty("durable_hydrated_without_prefix", 2);
    RecordProperty("durable_complete_target_heads", 16);
    RecordProperty("durable_same_selected_final_states", 2);
    RecordProperty("durable_snapshot_bytes", std::to_string(second->snapshot_bound));
    evidence = {};
    second.reset();
    fs::remove_all(path);
  }
};
// Opt-in manual case: ordinary CTest discovery does not register this captured
// two-owner experiment or turn a missing fixture into a skipped qualification.
[[maybe_unused]] const auto* durable_case = []() -> testing::TestInfo* {
  if (std::getenv("LLMP_QWEN_C2_DURABLE") == nullptr)
    return nullptr;  // NOLINT(concurrency-mt-unsafe)
  return testing::RegisterTest("Qwen38EngineServingGpu", "NativeC2DurablePendingReopen", nullptr,
                               nullptr, __FILE__, __LINE__,
                               []() -> testing::Test* { return new DurableCase; });
}();

}  // namespace c2

}  // namespace

// The actual Server route uses its ordinary options and keeper. The detailed
// within-selected state oracle above remains shared evidence, not a second
// persistence implementation in this adapter control.
namespace c2_server {
constexpr std::uint64_t kStartupExtent = ((c2::kStartup + 2 * kMiB - 1) / (2 * kMiB)) * (2 * kMiB);
static_assert(kStartupExtent >= c2::kStartup && kStartupExtent - c2::kStartup < 2 * kMiB);

TEST(Qwen38EngineServingGpu, DerivedServerStartupExtentPaysCompleteConstruction) {
  EXPECT_TRUE(
      rt::CheckStartupHostAllowance(kStartupExtent, kBudget, kBudget + rt::kUncountedMargin));
  EXPECT_GE(kStartupExtent, c2::kStartup);
  EXPECT_LT(kStartupExtent - c2::kStartup, 2 * kMiB);
  EXPECT_FALSE(
      rt::CheckStartupHostAllowance(kStartupExtent - 1, kBudget, kBudget + rt::kUncountedMargin));
}
struct Evidence {
  std::array<rt::Generation, 2> generations;
  std::array<std::vector<std::int32_t>, 2> history;
  std::array<llmp::model::DraftProgress, 2> progress{};
  llmp::base::Sha256Digest identity{};
};
struct Owner {
  Lifetime life;
  std::shared_ptr<Evidence> evidence;
  rt::Llm* llm = nullptr;
  bool attempted = false, retired = false;
  rt::Llm::Branch& Branch(std::size_t i) { return **llm->branch(i); }
  bool Retire() {
    if (attempted) return retired;
    attempted = true;
    if (life.server) {
      if (llm && life.server->node().InRequest(0) && !life.server->EndRequestBranches(*llm))
        return false;
      if (!life.server->DrainKept(rt::Clock::now() + std::chrono::seconds(30))) return false;
      if (!life.server->TearDown()) return false;
    }
    retired = true;
    return true;
  }
};
struct RetainUnknown {
  void operator()(Owner* owner) const {
    if (owner->Retire())
      delete owner;
    else
      ADD_FAILURE() << "retaining the complete unknown Server owner";
  }
};
using Owned = std::unique_ptr<Owner, RetainUnknown>;

rt::Status Setup(Owner& owner, const fs::path& root, bool reopen, Owned* previous,
                 bool component = true) {
  if (auto s = CheckFixture(kCutlass, true); !s) return s;
  auto& life = owner.life;
  life.scratch = root;
  life.roles.installed = fs::path(llmp::test_support::ModelsDir()) / "m3-artifacts";
  life.roles.spill = root / "spill";
  life.roles.state = root / "state";
  for (const auto& path : {life.roles.spill, life.roles.state}) {
    fs::create_directories(path);
    if (::chmod(path.c_str(), 0700) != 0) return std::unexpected("protecting Server state");
  }
  cfg::ModelEntry entry;
  entry.name = "qwen-server-c2";
  entry.artifact = kCutlass;
  entry.drafter = kMtp;
  const auto checkpoint = fs::path(llmp::test_support::ModelsDir()) /
                          "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6";
  entry.tokenizer = checkpoint / "tokenizer.json";
  entry.chat_template = checkpoint / "chat_template.jinja";
  entry.overrides["context"] = std::int64_t{512};
  entry.overrides["prefill_chunk"] = std::int64_t{16};
  entry.overrides["max_slots"] = std::int64_t{2};
  entry.overrides["draft_rows"] = std::int64_t{2};
  entry.overrides["wave_read_align"] = std::int64_t{256};
  life.config.models.push_back(std::move(entry));
  life.options.diagnostic_component_solo = component;
  life.options.keep_conversations = true;
  // Current compiler is transient once: the two futures build sequentially
  // while the device runs. Their retained programs are separately charged by
  // PrefillLookaheadGroup/PlanAccount, as the ordinary Server's current plan.
  life.options.diagnostic_startup_host_bytes = kStartupExtent;
  life.options.diagnostic_budget_cap_bytes = kBudget;
  life.log = std::tmpfile();
  if (!life.log) return std::unexpected("opening Server log");
  life.server = std::make_unique<rt::Server>(life.config, life.roles, life.options, life.log);
  if (auto s = life.server->Start(false); !s) return s;
  if (previous != nullptr) previous->reset();
  owner.llm = dynamic_cast<rt::Llm*>(life.server->Find("qwen-server-c2"));
  auto* shared = dynamic_cast<rt::EngineLlm*>(owner.llm);
  if (!owner.llm || owner.llm->generation_wave_capacity() != 2 ||
      (shared != nullptr) != component || (shared && shared->runner().slots().stages() != 2))
    return std::unexpected("Server did not build the declared C2 implementation");
  if (shared) {
    auto identity = shared->ExecutionIdentity();
    if (!identity || !*identity) return std::unexpected("Server omitted its prepared identity");
    if (reopen && **identity != owner.evidence->identity)
      return std::unexpected("Server reopened another selected execution");
    owner.evidence->identity = **identity;
    testing::Test::RecordProperty("server_execution_identity", llmp::base::ToHex(**identity));
  }
  if (life.server->budget() != kBudget)
    return std::unexpected("Server changed the admitted budget");
  testing::Test::RecordProperty("server_plan_floor_bytes",
                                std::to_string(life.server->plan_floor_bytes()));
  return {};
}

rt::Status Run(Owner& owner, bool reopen) {
  auto& server = *owner.life.server;
  auto& llm = *dynamic_cast<rt::EngineLlm*>(owner.llm);
  auto& evidence = *owner.evidence;
  const std::array<rt::Llm::Branch*, 2> branches{&owner.Branch(0), &owner.Branch(1)};
  if (reopen) {
    for (std::uint32_t slot = 0; slot < 2; ++slot)
      if (owner.Branch(slot).history() != evidence.history[slot] ||
          !llm.runner().slots().restorable(slot) || llm.runner().slots().usable(slot) ||
          owner.Branch(slot).turn_checkpoints() != 1)
        return std::unexpected("Server did not adopt both pending stages and the turn checkpoint");
  }
  rt::SwapParts parts;
  if (auto s = server.Activate(llm, parts); !s) return s;
  if (auto s = server.SelectRequestBranches(llm, branches); !s) return s;
  std::array<std::vector<float>, 2> heads;
  const auto before = llm.runner().graph_stats();
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    std::vector<std::int32_t> prompt;
    if (reopen)
      prompt = evidence.history[slot];
    else {
      const auto rows = slot == 0 ? 55U : 53U;
      for (std::uint32_t p = 0; p < rows; ++p)
        prompt.push_back(
            static_cast<std::int32_t>(100 + ((p * 7919U + 13U + slot * 17U) % 150000U)));
    }
    auto begun = owner.Branch(slot).BeginPrompt(prompt, reopen ? 0 : 16, false, reopen);
    if (!begun) return std::unexpected(begun.error());
    c2::SessionOwner<rt::Llm::PromptSession> session(begun->release());
    while (!session->done()) {
      if (auto s = session->Advance(); !s) return s;
    }
    if (reopen && session->reused() != prompt.size())
      return std::unexpected("Server durable hydration recomputed its prefix");
    heads[slot] = session->last();
    if (auto s = session->Finish(); !s) return s;
    if (owner.Branch(slot).turn_checkpoints() != 1)
      return std::unexpected("default Server turn checkpoint was not saved");
    if (reopen) {
      const auto* progress = llm.runner().slots().progress(slot, 1);
      if (!progress || progress->completed() != evidence.progress[slot])
        return std::unexpected("Server lost the hydrated pending-drafter boundary");
    }
  }
  const auto after = llm.runner().graph_stats();
  if (reopen && (after.eager != before.eager || after.replayed != before.replayed))
    return std::unexpected("Server reopened with new prefix execution");
  if (auto s = c2::Generate(owner, heads, evidence.generations, reopen ? 8 : 4, reopen); !s)
    return s;
  if (!llm.counts().joined_waves || llm.runner().coverage().violations)
    return std::unexpected("Server did not execute covered joined generations");
  if (!reopen) {
    const auto ahead = llm.runner().lookahead_stats();
    if (!ahead.built || !(ahead.captured_ahead + ahead.captured_first))
      return std::unexpected("default Server prompt futures were not built and captured");
    testing::Test::RecordProperty("server_future_built", std::to_string(ahead.built));
    testing::Test::RecordProperty("server_future_captured",
                                  std::to_string(ahead.captured_ahead + ahead.captured_first));
    // These short prompts share read256 pages; prepare can correctly need no
    // additional backing. Record the observed count, not a manufactured gain.
    testing::Test::RecordProperty("server_future_prepared", std::to_string(ahead.prepared));
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      evidence.history[slot] = owner.Branch(slot).history();
      const auto* progress = llm.runner().slots().progress(slot, 1);
      if (!progress || !progress->settled() ||
          progress->completed().pending_first >= progress->completed().positions)
        return std::unexpected("Server did not leave a real pending-drafter interval");
      evidence.progress[slot] = progress->completed();
    }
  }
  if (auto s = server.EndRequestBranches(llm); !s) return s;
  for (auto* branch : branches)
    if (auto s = llm.SpillIdle(*branch); !s) return s;
  if (!server.DrainKept(rt::Clock::now() + std::chrono::seconds(30)))
    return std::unexpected("Server record writes did not retire");
  return {};
}

class ServerCase : public testing::Test {
  void TestBody() override {
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    ASSERT_NE(scratch, nullptr);
    fs::create_directories(scratch);
    auto root = (fs::path(scratch) / "qwen-server-c2-XXXXXX").string();
    ASSERT_NE(::mkdtemp(root.data()), nullptr);
    Owned first(new Owner);
    first->evidence = std::make_shared<Evidence>();
    auto result = ::c2_server::Setup(*first, root, false, nullptr);
    ASSERT_TRUE(result) << (result ? "" : result.error());
    result = ::c2_server::Run(*first, false);
    ASSERT_TRUE(result) << (result ? "" : result.error());
    ASSERT_TRUE(first->Retire()) << "unknown Server completion forbids another owner";
    Owned second(new Owner);
    second->evidence = first->evidence;
    result = ::c2_server::Setup(*second, root, true, &first);
    ASSERT_TRUE(result) << (result ? "" : result.error());
    result = ::c2_server::Run(*second, true);
    ASSERT_TRUE(result) << (result ? "" : result.error());
    ASSERT_TRUE(second->Retire());
    RecordProperty("server_owners", 2);
    RecordProperty("server_hydrated_conversations", 2);
    RecordProperty("server_kept_turn_checkpoints", 2);
    RecordProperty("server_fresh_continuation_heads", 8);
    second.reset();
    fs::remove_all(root);
  }
};
const bool registered = [] {
  if (std::getenv("LLMP_QWEN_C2_SERVER") == nullptr) return false;  // NOLINT(concurrency-mt-unsafe)
  testing::RegisterTest("Qwen38EngineServingGpu", "NativeC2ServerDefaultOptionsReopen", nullptr,
                        nullptr, __FILE__, __LINE__,
                        []() -> testing::Test* { return new ServerCase; });
  return true;
}();

// One coarse whole-implementation timing screen, using the same Server owner
// and public sessions. The existing durable/default tests remain unchanged.
constexpr std::string_view kSpeedIdsSha =
    "3cd946c3876501eccbe079b9d2700e542e66242be241a8549546df8d9b23e468";
using SpeedIds = std::array<std::array<std::int32_t, 255>, 4>;
rt::Status ReadSpeedIds(SpeedIds& ids) {
  static_assert(std::endian::native == std::endian::little && sizeof(ids) == 4080);
  const char* path = std::getenv("LLMP_QWEN_C2_SPEED_IDS");  // NOLINT(concurrency-mt-unsafe)
  if (!path) return std::unexpected("missing authenticated natural speed inputs");
  auto* file = std::fopen(path, "rb");
  if (!file) return std::unexpected("opening natural speed inputs");
  struct stat st{};
  bool valid = fstat(fileno(file), &st) == 0 && S_ISREG(st.st_mode) &&
               st.st_size == static_cast<off_t>(sizeof(ids));
  valid = valid && std::fread(ids.data(), 1, sizeof(ids), file) == sizeof(ids) &&
          std::fgetc(file) == EOF && std::ferror(file) == 0;
  valid = std::fclose(file) == 0 && valid;
  llmp::base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(ids)));
  if (!valid || llmp::base::ToHex(hash.Finish()) != kSpeedIdsSha)
    return std::unexpected("natural speed input identity mismatch");
  for (const auto& document : ids)
    for (const auto id : document)
      if (id < 0 || std::cmp_greater_equal(id, kVocab))
        return std::unexpected("natural speed token outside source vocabulary");
  return {};
}
rt::Status SpeedPass(Owner& owner, const SpeedIds& ids, bool measured) {
  auto& server = *owner.life.server;
  auto& llm = *owner.llm;
  const std::array<rt::Llm::Branch*, 2> branches{&owner.Branch(0), &owner.Branch(1)};
  if (auto selected = server.SelectRequestBranches(llm, branches); !selected) return selected;
  // Clear invalidates kept records and checkpoints through the ordinary
  // branch API. Retain only warmed program/weight residency between passes.
  for (auto* branch : branches) {
    if (auto cleared = branch->Clear(); !cleared) return cleared;
    if (!branch->history().empty() || branch->turn_checkpoints())
      return std::unexpected("speed pass retained conversation reuse");
  }
  std::array<std::vector<float>, 2> heads;
  std::uint32_t prefilled = 0, reused = 0, chunks = 0;
  const auto started = rt::Clock::now();
  for (std::size_t slot = 0; slot < 2; ++slot) {
    const auto& prompt = ids[(measured ? 2U : 0U) + slot];
    auto begun = owner.Branch(slot).BeginPrompt(prompt, 16, false, false);
    if (!begun) return std::unexpected(begun.error());
    c2::SessionOwner<rt::Llm::PromptSession> session(begun->release());
    while (!session->done())
      if (auto advanced = session->Advance(); !advanced) return advanced;
    reused += session->reused();
    prefilled += session->run().end - session->reused();
    chunks += session->run().chunks;
    heads[slot] = session->last();
    if (auto finished = session->Finish(); !finished) return finished;
  }
  const auto prompted = rt::Clock::now();
  if (reused || prefilled != 510)
    return std::unexpected("speed pass omitted fresh natural prefill");
  std::uint32_t checkpoints = 0;
  for (const auto* branch : branches) {
    if (branch->turn_checkpoints() != 1)
      return std::unexpected("speed pass omitted a default turn checkpoint");
    ++checkpoints;
  }
  std::array<rt::Generation, 2> output;
  // stop=false fixes completed work; greedy token-only is the normal fast
  // path. Full-head observation is confined to the separate quality controls.
  const auto generation_started = rt::Clock::now();
  if (auto generated = c2::Generate(owner, heads, output, 64, false, false); !generated)
    return generated;
  const auto generated = rt::Clock::now();
  const auto prefix = measured ? "measured" : "warm";
  const auto property = [&](std::string_view key, auto value) {
    testing::Test::RecordProperty(std::format("{}_{}", prefix, key), std::to_string(value));
  };
  property("prefill_ns",
           std::chrono::duration_cast<std::chrono::nanoseconds>(prompted - started).count());
  property(
      "generation_ns",
      std::chrono::duration_cast<std::chrono::nanoseconds>(generated - generation_started).count());
  property("prefilled", prefilled);
  property("reused", reused);
  property("chunks", chunks);
  property("turn_checkpoints", checkpoints);
  std::uint64_t tokens = 0, drafted = 0, accepted = 0, steps = 0;
  for (std::size_t slot = 0; slot < 2; ++slot) {
    const auto& result = output[slot];
    tokens += result.tokens.size();
    drafted += result.drafted;
    accepted += result.accepted;
    steps += result.steps;
    llmp::base::Sha256 hash;
    hash.Update(std::as_bytes(std::span(result.tokens)));
    testing::Test::RecordProperty(std::format("{}_tokens{}_sha256", prefix, slot),
                                  llmp::base::ToHex(hash.Finish()));
  }
  property("tokens", tokens);
  property("drafted", drafted);
  property("accepted", accepted);
  property("branch_steps", steps);
  const auto graphs = llm.graphs();
  property("graphs_captured", graphs.captured);
  property("graphs_replayed", graphs.replayed);
  if (auto* shared = dynamic_cast<rt::EngineLlm*>(&llm)) {
    const auto ahead = shared->runner().lookahead_stats();
    property("future_built", ahead.built);
    property("future_captured", ahead.captured_ahead + ahead.captured_first);
    property("future_prepared", ahead.prepared);
  }
  if (auto ended = server.EndRequestBranches(llm); !ended) return ended;
  if (!server.DrainKept(rt::Clock::now() + std::chrono::seconds(30)))
    return std::unexpected("speed pass keeper did not retire");
  return {};
}
class SpeedCase : public testing::Test {
 public:
  explicit SpeedCase(bool component) : component_(component) {}

 private:
  void TestBody() override {
    SpeedIds ids{};
    auto read = ReadSpeedIds(ids);
    ASSERT_TRUE(read) << (read ? "" : read.error());
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    ASSERT_NE(scratch, nullptr);
    fs::create_directories(scratch);
    auto root = (fs::path(scratch) / "qwen-c2-speed-XXXXXX").string();
    ASSERT_NE(::mkdtemp(root.data()), nullptr);
    Owned owner(new Owner);
    owner->evidence = std::make_shared<Evidence>();
    const auto started = rt::Clock::now();
    auto status = ::c2_server::Setup(*owner, root, false, nullptr, component_);
    ASSERT_TRUE(status) << (status ? "" : status.error());
    const auto setup = rt::Clock::now();
    rt::SwapParts parts;
    status = owner->life.server->Activate(*owner->llm, parts);
    ASSERT_TRUE(status) << (status ? "" : status.error());
    const auto ready = rt::Clock::now();
    RecordProperty("speed_arm", component_ ? "shared" : "legacy");
    RecordProperty("speed_inputs_sha256", std::string(kSpeedIdsSha));
    RecordProperty(
        "speed_setup_ns",
        std::to_string(
            std::chrono::duration_cast<std::chrono::nanoseconds>(setup - started).count()));
    RecordProperty(
        "speed_materialize_ns",
        std::to_string(
            std::chrono::duration_cast<std::chrono::nanoseconds>(ready - setup).count()));
    status = SpeedPass(*owner, ids, false);
    ASSERT_TRUE(status) << (status ? "" : status.error());
    status = SpeedPass(*owner, ids, true);
    ASSERT_TRUE(status) << (status ? "" : status.error());
    ASSERT_TRUE(owner->Retire());
    owner.reset();
    fs::remove_all(root);
  }
  bool component_;
};
const bool speed_registered = [] {
  if (std::getenv("LLMP_QWEN_C2_SPEED") == nullptr) return false;  // NOLINT(concurrency-mt-unsafe)
  testing::RegisterTest("Qwen38EngineServingGpu", "NativeC2SpeedLegacy", nullptr, nullptr, __FILE__,
                        __LINE__, []() -> testing::Test* { return new SpeedCase(false); });
  testing::RegisterTest("Qwen38EngineServingGpu", "NativeC2SpeedShared", nullptr, nullptr, __FILE__,
                        __LINE__, []() -> testing::Test* { return new SpeedCase(true); });
  return true;
}();
}  // namespace c2_server
