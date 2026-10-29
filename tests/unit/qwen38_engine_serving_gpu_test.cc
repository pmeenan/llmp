// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Single-slot screens through the real Server and Llm on the published v1
// Qwen3.8 artifacts (tests/support/v1_frozen.h V1Store: the native ModelOpt
// target 02354af1 with its MTP drafter bb49b7da, and the GGUF import
// b31c469e). Two arms of one binary draft and judge their own outputs and
// must agree bit for bit; the GGUF publication arm is asserted against its
// frozen digests (tests/unit/data/v1_frozen/qwen38_gguf.json). No
// graph/source wrapper or teacher-forced proposals. This small screen is
// not the 1% migration gate. Each screen fails when its artifacts or the
// tokenizer assets are absent.
//
// Also (label `models`, the ordinary Server with its default options and
// no diagnostic tuple): the native target and MTP drafter's two kept
// conversations adopted by a second Server and continued
// (NativeC2ServerDefaultOptionsReopen), and the conversations kept across a
// restart (D-105, D-112) continuing exactly, of the GGUF import
// (Qwen38GgufKeptGpu, below; kept_restart_serving_checks.h) and of the native
// target with its MTP drafter (Qwen38MtpKeptGpu). They fail, naming the
// artifact, when the artifacts or the tokenizer assets are absent.

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
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/json.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "compiler_inventory.h"
#include "engine/paged_weights.h"
#include "execution/registry.h"
#include "kept_restart_serving_checks.h"
#include "kernels/common/step_program.h"
#include "kernels/ggml/plan_step.h"
#include "planner/plan.h"
#include "platform/host_probe.h"
#include "runtime/component_build.h"
#include "runtime/memory_guard.h"
#include "runtime/serving.h"
#include "spec/mappings/qwen38.h"
#include "spec/stored.h"
#include "tokenizer_fixtures.h"
#include "v1_frozen.h"

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
constexpr std::uint64_t kCompilerMinimum = 3584 * kMiB;
constexpr std::uint64_t kResults = 128 * kMiB;
static_assert(sizeof(void*) == 8 && sizeof(std::string) <= 32 && sizeof(std::vector<void*>) <= 24);
static_assert(llmp::ir::Graph::kMaxNodes == 65536 && llmp::ir::Graph::kMaxTensors == 131072);
static_assert(sizeof(llmp::ir::Node) <= 160 && sizeof(llmp::ir::TensorDesc) <= 160);
static_assert(sizeof(llmp::ir::Binding) <= 24 && sizeof(llmp::format::TensorFormat) <= 72);
static_assert(sizeof(llmp::planner::Selected) <= 128);
static_assert(sizeof(llmp::planner::Match) + sizeof(void*) <= 256);
static_assert(sizeof(llmp::planner::RootAlignmentFact) <= 16);
static_assert(sizeof(llmp::kernels::InputSlot) <= 512 && sizeof(llmp::kernels::TensorUse) <= 64);
static_assert(sizeof(llmp::kernels::ggml::PlanStep) <= 64);
// Exact logical rank adds eight ABI bytes per transient operand descriptor.
// These bounded selection vectors remain within the serving metadata allowance.
static_assert(sizeof(llmp::execution::OperandMemory) <= 48 &&
              sizeof(llmp::execution::OperandFormat) <= 120 &&
              sizeof(llmp::execution::CapabilityQuery) <= 256);
static_assert(sizeof(llmp::ir::WeightRef) <= 16 && sizeof(llmp::ir::StateRef) <= 16);
static_assert(llmp::execution::Registry::kMaxImplementations == 512);
static_assert(llmp::ir::kMaxSources == 10 && llmp::ir::kMaxOutputs == 4);
// Inherited hard-cap compiler peak plus the source-inventoried serving
// owners below. The fixture fingerprints are checked before either arm;
// this fixed allowance is not a general arbitrary-artifact memory claim.
constexpr std::uint64_t kInheritedCompilerMaximum =
    2528028992ULL + llmp::test_support::compiler_inventory::kLegacyCompilerGrowth;
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
// Exact legacy phase inventory plus current owned serving metadata. Retain the
// prior minimum; provenance/site payloads increase payment only when needed.
constexpr std::uint64_t kCompilerRequired = kInheritedCompilerMaximum + kServingDelta;
constexpr std::uint64_t kCompiler =
    ((std::max(kCompilerMinimum, kCompilerRequired) + 2 * kMiB - 1) / (2 * kMiB)) * (2 * kMiB);
constexpr std::uint64_t kAllowance = kCompiler + kResults;
static_assert(kCompilerRequired <= kCompiler);
static_assert(sizeof(llmp::artifact::Resource) <= 512 &&
              sizeof(llmp::artifact::ExpertArray) <= 512 && sizeof(llmp::artifact::Group) <= 64);
static_assert(sizeof(llmp::engine::SlabPage) <= 128 && sizeof(llmp::scheduler::PageSource) <= 256 &&
              sizeof(llmp::catalog::ExtentView) <= 144);
// Alias storage and fold provenance stay within the existing 2 KiB per-record
// serving allowance; this assertion anchors the source record itself.
static_assert(sizeof(llmp::spec::SourceTensor) <= 256);
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
// The ModelOpt target's device choices read its tokenizer.json's 248,077 of
// those rows (RE-050); the GGUF's tokenizer lists every row.
constexpr std::uint32_t kModelOptTokens = 248077;
constexpr std::uint64_t kRetainedRows = 2 * (2 + 8 + 8);
static_assert(2 * kRetainedRows * kVocab * sizeof(float) + 18 * kMiB <= kResults);

// The published v1 artifacts (D-118): the native ModelOpt target, the GGUF
// import, and the ModelOpt MTP drafter.
constexpr std::string_view kModelOpt = llmp::test_support::v1_store::kQwen38ModelOptTarget;
constexpr std::string_view kGguf = llmp::test_support::v1_store::kQwen38Gguf;
constexpr std::string_view kMtp = llmp::test_support::v1_store::kQwen38ModelOptDrafter;

// The published v1 store, where all three are installed.
fs::path Store() { return fs::path(llmp::test_support::V1Store()); }

// A streaming, fixed-stack identity preflight authenticates the metadata
// counts used above without constructing JSON, artifacts or tokenizers.
// The actual Server installs its catalog payment before reading them again.
// An artifact's files are under the v1 store, the tokenizer assets under the
// models directory.
struct FixtureFile {
  std::string_view artifact, path;
  std::uint64_t bytes;
  std::string_view digest;
};
constexpr std::array kFixtureFiles{
    FixtureFile{kModelOpt,
                "02354af162f30b6ec6a117d5311da7d642c5be3128867750be55eeb7b6cd38b9/index.json",
                8236769ULL, "2b29dc0f905bd670522abda9d8ef5e17f0db328c1d4fcd9f8dae63e9079dc5dd"},
    FixtureFile{kModelOpt,
                "02354af162f30b6ec6a117d5311da7d642c5be3128867750be55eeb7b6cd38b9/manifest.json",
                48492ULL, kModelOpt},
    FixtureFile{kModelOpt,
                "02354af162f30b6ec6a117d5311da7d642c5be3128867750be55eeb7b6cd38b9/meta/config.json",
                109897ULL, "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e"},
    FixtureFile{kGguf,
                "b31c469e0fc129765fa91ce541cc1639661aa9bcb1ee5b885b8ffc33b6cc21c8/index.json",
                6391531ULL, "40803567725558418f1993e1c50948870d92ea7022fea40151846334fbd03d70"},
    FixtureFile{kGguf,
                "b31c469e0fc129765fa91ce541cc1639661aa9bcb1ee5b885b8ffc33b6cc21c8/manifest.json",
                26385ULL, "61827c2fdd87aabd0ee6371054358b6f90433d9452ceef1910edcbe130268be1"},
    FixtureFile{kGguf,
                "b31c469e0fc129765fa91ce541cc1639661aa9bcb1ee5b885b8ffc33b6cc21c8/"
                "meta/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.kv.gguf",
                10946618ULL, "7c1460a95eb2e95377c32b14deb92cf74fe504d4feb5a62a40d48f78e9e0dcb8"},
    FixtureFile{kGguf,
                "b31c469e0fc129765fa91ce541cc1639661aa9bcb1ee5b885b8ffc33b6cc21c8/"
                "meta/Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.kv.gguf",
                106ULL, "0bd485a7f050b5ac258c3722f25f2be38a7ef3e597696f3e4d7e09ec461ff978"},
    FixtureFile{kGguf,
                "b31c469e0fc129765fa91ce541cc1639661aa9bcb1ee5b885b8ffc33b6cc21c8/"
                "meta/Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.kv.gguf",
                106ULL, "2fd68e0dc3770fdd9e4d52a5e9570394855926807e0e020b879651ded2b004c6"},
    FixtureFile{kMtp, "bb49b7da7edb78202a64f1f9ab049412a10f2a9f73eb4eef66608e81b011f263/index.json",
                159859ULL, "c20f71e3c79ad0505f24a5f822a8a644d53a3cdfc4e962e4e34533a508cee316"},
    FixtureFile{kMtp,
                "bb49b7da7edb78202a64f1f9ab049412a10f2a9f73eb4eef66608e81b011f263/manifest.json",
                4092ULL, "f4d84ed99ca105dfa420b362c39551f73ccb10a730aa6a893b21ff577e83add0"},
    FixtureFile{kMtp,
                "bb49b7da7edb78202a64f1f9ab049412a10f2a9f73eb4eef66608e81b011f263/meta/config.json",
                109897ULL, "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e"},
    FixtureFile{"", "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/tokenizer.json",
                12809320ULL, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"},
    FixtureFile{"", "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/chat_template.jinja",
                8952ULL, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"},
};

rt::Status CheckFixture(std::string_view id, bool mtp) {
  for (const auto& file : kFixtureFiles) {
    if (!file.artifact.empty() && file.artifact != id && !(mtp && file.artifact == kMtp)) continue;
    const auto path =
        (file.artifact.empty() ? fs::path(llmp::test_support::ModelsDir()) : Store()) / file.path;
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

struct StateImage {
  std::vector<std::array<std::uint64_t, 3>> ranges;
  llmp::base::Sha256Digest digest{};
  std::uint32_t positions = 0;
  bool operator==(const StateImage&) const = default;
};
struct Result {
  std::array<StateImage, 3> states;
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
  std::shared_ptr<en::ExecutionMetadataAllowance> observation_payment;
  std::unique_ptr<rt::Llm::PromptSession> restoration;
  Result result;
  fs::path scratch;
  std::FILE* log = nullptr;
  ~Lifetime() {
    if (log != nullptr) std::fclose(log);
  }
};
static_assert(sizeof(Lifetime) <= 16384 && sizeof(rt::Server) <= 65536 && sizeof(rt::Llm) <= 65536);

std::vector<std::int32_t> Prompt() {
  std::vector<std::int32_t> out;
  out.reserve(7);
  for (std::uint32_t p = 0; p < 7; ++p)
    out.push_back(static_cast<std::int32_t>(100 + ((p * 7919U + 13U) % 150000U)));
  return out;
}

// Same completed-state reader used by the native Gemma publication proof:
// bounded pinned staging, complete initialized coordinates and streamed bytes.
std::expected<StateImage, std::string> ReadState(Lifetime& life, rt::Llm& model) {
  StateImage state;
  auto& slots = model.runner().slots();
  auto& node = life.server->node();
  state.positions = slots.positions(0);
  for (std::size_t stage = 0; stage < slots.stages(); ++stage)
    for (const auto& range : slots.live(0, stage).used_ranges()) {
      if (state.ranges.size() == 4096)
        return std::unexpected("state observation exceeds prepaid coordinates");
      state.ranges.push_back({stage, range.offset, range.bytes});
    }
  if (state.ranges.empty()) return std::unexpected("state observation is empty");
  constexpr std::uint64_t part_bytes = 1ULL << 20U;
  std::vector<llmp::catalog::ExtentId> extents;
  auto buffer = node.Pinned(part_bytes, 0, extents);
  if (!buffer) return std::unexpected(buffer.error());
  llmp::base::Sha256 hash;
  for (const auto& range : state.ranges)
    for (std::uint64_t at = 0; at < range[2]; at += part_bytes) {
      const en::LiveState::Range part{.region = range[0],
                                      .offset = range[1] + at,
                                      .bytes = std::min(part_bytes, range[2] - at)};
      auto retired = en::LiveState::CopyRetirement::kUnproven;
      auto copied = slots.CopyState(0, *buffer, std::span(&part, 1), true, &retired);
      if (retired == en::LiveState::CopyRetirement::kUnproven) {
        node.KeepPinned(*buffer);
        return std::unexpected("state copy retirement is unproven");
      }
      if (!copied) {
        (void)node.FreePinned(*buffer);
        return std::unexpected(copied.error());
      }
      hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
    }
  if (auto freed = node.FreePinned(*buffer); !freed) return std::unexpected(freed.error());
  state.digest = hash.Finish();
  return state;
}

std::expected<std::unique_ptr<Lifetime>, std::string> Run(
    std::string_view id, bool mtp, bool component, bool rows, std::uint32_t depth,
    Lifetime* previous = nullptr, const fs::path& stored_root = {}, bool publication = false,
    std::string_view drafter = kMtp, std::optional<std::string> drafter_policy = std::nullopt) {
  auto life = std::make_unique<Lifetime>();
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  if (scratch == nullptr) return std::unexpected("no test scratch directory");
  fs::create_directories(scratch);
  std::string name = (fs::path(scratch) / "qwen-solo-XXXXXX").string();
  if (mkdtemp(name.data()) == nullptr) return std::unexpected("creating test scratch");
  life->scratch = name;
  life->roles.installed = stored_root.empty() ? Store() : stored_root;
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
  if (mtp) entry.drafter = std::string(drafter);
  entry.overrides["context"] = std::int64_t{512};
  entry.overrides["prefill_chunk"] = std::int64_t{16};
  entry.overrides["max_slots"] = std::int64_t{1};
  entry.overrides["draft_rows"] = static_cast<std::int64_t>(depth);
  entry.overrides["wave_read_align"] = std::int64_t{256};
  life->config.models.push_back(std::move(entry));
  life->options.diagnostic_drafter_policy = std::move(drafter_policy);
  life->options.diagnostic_startup_host_bytes = publication ? 0 : kAllowance;
  life->options.diagnostic_auto_startup = publication;
  life->options.diagnostic_budget_cap_bytes = kBudget;
  life->log = std::tmpfile();
  if (!life->log) return std::unexpected("opening test log");
  life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, life->log);
  rt::Llm* model = nullptr;
  auto run = [&]() -> rt::Status {
    if (auto s = life->server->Start(true); !s) return s;
    if (publication) {
      auto& node = life->server->node();
      const auto available = llmp::platform::AvailableMemoryBytes().value_or(0);
      auto paid = node.Call(
          [&]() -> en::Status {
            std::uint64_t charge = 0;
            const auto occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
            const auto budget = node.budget().value();
            if (__builtin_add_overflow(kResults, en::ExecutionMetadataAllowance::OverheadBytes(),
                                       &charge) ||
                occupied > budget || charge > budget - occupied)
              return std::unexpected("publication observations exceed the node budget");
            auto payment = en::ExecutionMetadataAllowance::Admit(
                node.catalog(), node.domain(), charge, available, rt::kUncountedMargin);
            if (!payment) return std::unexpected(payment.error());
            life->observation_payment = std::move(*payment);
            if (!life->observation_payment->Charge(kResults))
              return std::unexpected("publication observations lack complete payment");
            return {};
          },
          "pay native publication observations");
      if (!paid) return paid;
    } else if (life->server->node()
                   .catalog()
                   .OccupancyOf(life->server->node().domain())
                   .Total()
                   .value() < kAllowance)
      return std::unexpected("startup compiler allowance was not cataloged");
    // The old stopped Server pays its retained results until this allowance
    // is actually admitted. Only then transfer that payment and destroy its
    // remaining model metadata, before activating any new model work.
    if (previous != nullptr) previous->server.reset();
    model = dynamic_cast<rt::Llm*>(life->server->Find("qwen"));
    if (!model || (dynamic_cast<rt::Llm*>(model) != nullptr) != component)
      return std::unexpected("wrong serving pipeline selected");
    if (model->generation_wave_capacity() != 1 || model->logit_width() != kVocab ||
        model->choice_vocabulary() != (id == kGguf ? 0 : kModelOptTokens))
      return std::unexpected("wrong serving envelope");
    rt::SwapParts parts;
    if (auto s = life->server->Activate(*model, parts); !s) return s;
    auto b = model->branch(0);
    if (!b) return std::unexpected(b.error());
    std::array<rt::Llm::Branch*, 1> active{*b};
    if (auto s = life->server->SelectRequestBranches(*model, active); !s) return s;
    auto prompt = Prompt();
    if (auto s = (*b)->Prefill(prompt, life->result.heads[0]); !s) return s;
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
      if (publication) {
        auto state = ReadState(*life, *static_cast<rt::Llm*>(model));
        if (!state) return std::unexpected(state.error());
        life->result.states[turn == 0 ? 0 : 2] = std::move(*state);
      }
      if (turn == 0) {
        // Idle spill retains the actual pending MTP state. Restore through
        // the normal prompt interface, then process the last sampled token.
        if (auto s = life->server->EndRequestBranches(*model); !s) return s;
        if (auto s = model->SpillIdle(**b); !s) return s;
        if (auto s = life->server->SelectRequestBranches(*model, active); !s) return s;
        if (publication) {
          if (!(*b)->spilled()) return std::unexpected("native publication did not spill state");
          const auto before = model->graphs();
          auto begun = (*b)->BeginPrompt(life->result.history[0], 0, false, true);
          if (!begun) return std::unexpected(begun.error());
          life->restoration = std::move(*begun);
          while (!life->restoration->done())
            if (auto step = life->restoration->Advance(); !step) return step;
          if (life->restoration->reused() != life->result.history[0].size())
            return std::unexpected("native publication restoration recomputed its history");
          if (auto finished = life->restoration->Finish(); !finished) return finished;
          life->restoration.reset();
          const auto after = model->graphs();
          if (before.eager != after.eager || before.captured != after.captured ||
              before.replayed != after.replayed || (*b)->spilled())
            return std::unexpected("native publication restoration executed model work");
          auto state = ReadState(*life, *static_cast<rt::Llm*>(model));
          if (!state) return std::unexpected(state.error());
          life->result.states[1] = std::move(*state);
          if (life->result.states[0] != life->result.states[1])
            return std::unexpected("native publication restored different initialized state");
        }
        auto next = (*b)->history();
        next.push_back(result.tokens.back());
        std::uint32_t reused = 0;
        if (auto s = (*b)->PreparePrompt(next, 0, life->result.heads[1], reused); !s) return s;
        if (reused + 1 != next.size()) return std::unexpected("spill resume lost its prefix");
      }
    }
    if (component) {
      const auto& engine = *static_cast<rt::Llm*>(model);
      if (engine.runner().coverage().violations != 0)
        return std::unexpected("shared program bound uncovered storage");
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
  const auto store = Store();
  const auto checkpoint = fs::path(llmp::test_support::ModelsDir()) /
                          "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6";
  if (!fs::exists(store / id) || (mtp && !fs::exists(store / kMtp)) ||
      !fs::exists(checkpoint / "tokenizer.json") || !fs::exists(checkpoint / "chat_template.jinja"))
    GTEST_FAIL() << "the published Qwen3.8 v1 artifacts or the tokenizer assets are absent ("
                 << store << ")";
  auto fixed = CheckFixture(id, mtp);
  ASSERT_TRUE(fixed) << (fixed ? "" : fixed.error());
  auto first = Run(id, mtp, true, rows, depth);
  ASSERT_TRUE(first) << (first ? "" : first.error());
  fixed = CheckFixture(id, mtp);
  ASSERT_TRUE(fixed) << (fixed ? "" : fixed.error());
  auto second = Run(id, mtp, true, rows, depth, first->get());
  ASSERT_TRUE(second) << (second ? "" : second.error());
  fixed = CheckFixture(id, mtp);
  if (!fixed) first->reset();  // release under the still-live final payment
  ASSERT_TRUE(fixed) << (fixed ? "" : fixed.error());
  Compare((*second)->result, (*first)->result, rows);
  fs::remove_all((*first)->scratch);
  fs::remove_all((*second)->scratch);
  // Release the earlier result while the final stopped Server still pays
  // both results. Lifetime destroys its own Result before its Server.
  first->reset();
}

// The GGUF import's publication arm (heads, greedy tokens with their logit
// rows, the state after each turn and after a spill and restore) against its
// frozen digests. LLMP_TEST_QWEN_V1_STORE (or LLMP_TEST_V1_STORE) and
// LLMP_TEST_QWEN_V1_ID name another store and artifact.
TEST(Qwen38EngineServingGpu, PublishedGgufMatchesFrozenHeadsStateAndSpill) {
  const char* root = std::getenv("LLMP_TEST_QWEN_V1_STORE");  // NOLINT(concurrency-mt-unsafe)
  const char* given = std::getenv("LLMP_TEST_QWEN_V1_ID");    // NOLINT(concurrency-mt-unsafe)
  const fs::path store = root != nullptr ? fs::path(root) : Store();
  const std::string id = given != nullptr ? given : std::string(kGguf);
  ASSERT_EQ(id.size(), 64U);
  ASSERT_TRUE(std::ranges::all_of(std::string_view(id), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  }));
  const auto checkpoint = fs::path(llmp::test_support::ModelsDir()) /
                          "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6";
  std::error_code error;
  if (!fs::exists(store / id, error) || !fs::exists(checkpoint / "tokenizer.json", error) ||
      !fs::exists(checkpoint / "chat_template.jinja", error))
    GTEST_FAIL() << "no Qwen3.8 GGUF v1 artifact " << id << " in " << store
                 << " or no tokenizer assets";
  if (id == kGguf && root == nullptr) {
    auto fixed = CheckFixture(kGguf, false);
    ASSERT_TRUE(fixed) << (fixed ? "" : fixed.error());
  }
  auto native = ::Run(id, false, true, true, 2, nullptr, store, true);
  ASSERT_TRUE(native) << (native ? "" : native.error());
  llmp::test_support::FrozenDigests frozen("qwen38_gguf");
  const auto& r = (*native)->result;
  using llmp::test_support::HexOfFloats;
  using llmp::test_support::HexOfInts;
  using llmp::test_support::HexOfRows;
  const auto state_text = [](const StateImage& state) {
    return llmp::base::ToHex(state.digest) + ':' + std::to_string(state.positions) + ':' +
           std::to_string(state.ranges.size());
  };
  for (std::size_t turn = 0; turn < 2; ++turn) {
    const auto t = std::to_string(turn);
    frozen.Expect("heads." + t, HexOfFloats(r.heads[turn]));
    frozen.Expect("tokens." + t, llmp::test_support::JoinTokens(r.generations[turn].tokens));
    frozen.Expect("logits." + t, HexOfRows(r.generations[turn].logits));
    frozen.Expect("history." + t, HexOfInts(r.history[turn]));
  }
  // States: 0 after the first turn, 1 after spill and restore, 2 after the second.
  for (std::size_t i = 0; i < r.states.size(); ++i)
    frozen.Expect("state." + std::to_string(i), state_text(r.states[i]));
  for (const auto& head : r.heads)
    EXPECT_TRUE(std::ranges::all_of(head, [](float value) { return std::isfinite(value); }));
  for (const auto& generation : r.generations)
    for (const auto& head : generation.logits)
      EXPECT_TRUE(std::ranges::all_of(head, [](float value) { return std::isfinite(value); }));
  testing::Test::RecordProperty("complete_head_rows", 18);
  testing::Test::RecordProperty("head_values_per_row", kVocab);
  testing::Test::RecordProperty("complete_state_images", 3);
  testing::Test::RecordProperty("restored_owners", 1);
  testing::Test::RecordProperty("native_artifact", id);
  fs::remove_all((*native)->scratch);
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
  for (const auto count : {0U, static_cast<std::uint32_t>(rt::Llm::kMaxBranches + 1)}) {
    rt::ModelSettings settings;
    settings.architecture = "qwen4exp";
    settings.container = "safetensors";
    settings.max_slots.value = count;
    settings.context.value = 512;
    settings.prefill_chunk.value = 16;
    auto build = rt::ComponentLlmBuild(life->server->node(), life->roles.installed, entry, 0,
                                       {
                                           .target_policy = {},
                                       });
    const auto refused = build(settings);
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error(),
              "component serving needs positive context/chunk and a bounded owner count");
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

// Geometry is checked against the authenticated source rather than a historical
// checkpoint experiment. No missing source can manufacture execution authority.
TEST(Qwen38EngineServingGpu, ServingGeometryRequiresAnAuthenticatedSource) {
  en::PagedNode node{{}};
  cfg::ModelEntry entry;
  entry.name = "missing";
  entry.artifact = std::string(64, '0');
  for (const auto owners : {1U, 2U, 3U, 4U}) {
    for (const auto context : {512U, 8192U}) {
      rt::ModelSettings settings;
      settings.architecture = "qwen4exp";
      settings.container = "safetensors";
      settings.max_slots.value = owners;
      settings.context.value = context;
      settings.prefill_chunk.value = 16;
      auto build = rt::ComponentLlmBuild(node, "/nonexistent-serving-source", entry, 0,
                                         {.target_policy = {}});
      const auto refused = build(settings);
      ASSERT_FALSE(refused);
      EXPECT_NE(refused.error().find("artifact"), std::string::npos);
      EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), 0U);
    }
  }
}

TEST(Qwen38EngineServingGpu, NativeMtpDepthTwoFullLogits) { Screen(kModelOpt, true, true, 2); }
TEST(Qwen38EngineServingGpu, NativeMtpDefaultTokenSchedule) { Screen(kModelOpt, true, false, 3); }
TEST(Qwen38EngineServingGpu, GgufPlain) { Screen(kGguf, false, true, 2); }

// The native drafter mapping's policy (W4A4: its moe_experts), and
// the same without the block-scaled GEMM: the format ladder decodes the
// scale-atom experts on chip (W4A16), as on a device without FP4.
std::expected<std::string, std::string> LadderDrafterPolicy() {
  const auto rules = llmp::spec::PolicyRules(llmp::spec::mappings::Qwen38MtpMapping(),
                                             {"qwen4exp-mtp", "safetensors"});
  if (!rules) return std::unexpected(rules.error());
  if (rules->empty()) return std::unexpected("native drafter policy is empty");
  auto policy = llmp::planner::ParsePolicy(rules->front().text);
  if (!policy) return std::unexpected(policy.error());
  std::erase(policy->enabled, "moe_experts");
  if (!std::ranges::contains(policy->disabled, "moe_experts"))
    policy->disabled.emplace_back("moe_experts");
  llmp::planner::Canonicalize(*policy);
  if (auto valid = llmp::planner::Validate(*policy); !valid) return std::unexpected(valid.error());
  return llmp::planner::ToText(*policy);
}
std::pair<std::uint64_t, std::uint64_t> Acceptance(const Result& r) {
  std::uint64_t accepted = 0, drafted = 0;
  for (const auto& g : r.generations) {
    accepted += g.accepted;
    drafted += g.drafted;
  }
  return {accepted, drafted};
}
void ExpectSameTokens(const Result& was, const Result& now, std::string_view arm) {
  SCOPED_TRACE(arm);
  for (std::size_t turn = 0; turn < 2; ++turn) {
    EXPECT_EQ(now.generations[turn].tokens, was.generations[turn].tokens) << turn;
    EXPECT_EQ(now.history[turn], was.history[turn]) << turn;
    EXPECT_GT(now.generations[turn].drafted, 0U);
  }
  const auto [old_accepted, old_drafted] = Acceptance(was);
  const auto [accepted, drafted] = Acceptance(now);
  testing::Test::RecordProperty(std::format("{}_base_accepted_drafted", arm),
                                std::format("{}/{}", old_accepted, old_drafted));
  testing::Test::RecordProperty(std::format("{}_accepted_drafted", arm),
                                std::format("{}/{}", accepted, drafted));
  EXPECT_GE(accepted + 2, old_accepted);
}

// The native ModelOpt target and its MTP drafter speculating through the
// shared engine at their default policies, state observed, then the same
// pair with the drafter's scale-atom experts on the format ladder (W4A16,
// as a device without FP4): the default arm's tokens, accepting at most 2
// fewer drafts.
TEST(Qwen38EngineServingGpu, NativeMtpLadderDrafterKeepsTokens) {
  const auto checkpoint = fs::path(llmp::test_support::ModelsDir()) /
                          "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6";
  std::error_code error;
  if (!fs::exists(Store() / kModelOpt, error) || !fs::exists(Store() / kMtp, error) ||
      !fs::exists(checkpoint / "tokenizer.json", error) ||
      !fs::exists(checkpoint / "chat_template.jinja", error))
    GTEST_FAIL() << "the published Qwen3.8 v1 pair or the tokenizer assets are absent (" << Store()
                 << ")";
  auto fixed = CheckFixture(kModelOpt, true);
  ASSERT_TRUE(fixed) << (fixed ? "" : fixed.error());
  auto default_arm = ::Run(kModelOpt, true, true, true, 3, nullptr, {}, true);
  ASSERT_TRUE(default_arm) << (default_arm ? "" : default_arm.error());
  const auto& now = (*default_arm)->result;
  for (const auto& head : now.heads) {
    ASSERT_EQ(head.size(), kVocab);
    EXPECT_TRUE(std::ranges::all_of(head, [](float value) { return std::isfinite(value); }));
  }
  for (const auto& generation : now.generations) EXPECT_GT(generation.drafted, 0U);
  const auto ladder_policy = LadderDrafterPolicy();
  ASSERT_TRUE(ladder_policy) << (ladder_policy ? "" : ladder_policy.error());
  auto ladder_arm =
      ::Run(kModelOpt, true, true, true, 3, default_arm->get(), {}, true, kMtp, *ladder_policy);
  ASSERT_TRUE(ladder_arm) << (ladder_arm ? "" : ladder_arm.error());
  ExpectSameTokens(now, (*ladder_arm)->result, "w4a16");
  for (auto* arm : {&*default_arm, &*ladder_arm}) {
    fs::remove_all((*arm)->scratch);
    arm->reset();
  }
}

// The two-owner MTP Server control's startup payment and its shared session
// helpers.
namespace c2 {
constexpr std::uint64_t kN = llmp::ir::Graph::kMaxNodes;
constexpr std::uint64_t kT = llmp::ir::Graph::kMaxTensors;
constexpr std::uint64_t kSaves = 5 * 48 * 2;
constexpr std::uint64_t kDescriptors = kT + 16 + 31 * kN + 4 * kSaves;
// Current shared compiler graph/rewrite/placement/descriptor overlap, plus
// the existing authenticated artifact/tokenizer/catalog owners above. Retained
// programs are measured and separately admitted; no launch-count guess.
constexpr std::uint64_t kCompile =
    llmp::test_support::compiler_inventory::GraphBytes(kN, kT) + 4096 * (kN + kT) +
    llmp::test_support::compiler_inventory::NativeBytes(kDescriptors, kT) + 64 * kMiB;
constexpr std::uint64_t kObserve = 128 * kMiB;
constexpr std::uint64_t kStartup = kCompile + kServingDelta + kObserve;
// Preserve the reviewed baseline identity while deriving all added record
// storage from actual types, including per-recipient vector payloads.
static_assert(kCompile == 4946022912ULL +
                              12 * kN * (llmp::planner::kConstructionGraphNodeBytes - 480) +
                              llmp::planner::NumericalConstructionBytes(kN).value() +
                              2 * kDescriptors * sizeof(llmp::kernels::ggml::PhysicalTensor) +
                              2 * kT * sizeof(llmp::kernels::ggml::PhysicalRoot));
static_assert(3 * 2 * 8 * kVocab * sizeof(float) + 32 * kMiB < kObserve);

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
}  // namespace c2

}  // namespace

// The native ModelOpt target and its MTP drafter, two owners, through the
// ordinary Server with its default options and keeper: both conversations
// spilled and kept, then adopted by a second Server and continued with no
// prefix recomputation (NativeC2ServerDefaultOptionsReopen).
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

rt::Status Setup(Owner& owner, const fs::path& root, bool reopen, Owned* previous) {
  if (auto s = CheckFixture(kModelOpt, true); !s) return s;
  auto& life = owner.life;
  life.scratch = root;
  life.roles.installed = Store();
  life.roles.spill = root / "spill";
  life.roles.state = root / "state";
  for (const auto& path : {life.roles.spill, life.roles.state}) {
    fs::create_directories(path);
    if (::chmod(path.c_str(), 0700) != 0) return std::unexpected("protecting Server state");
  }
  cfg::ModelEntry entry;
  entry.name = "qwen-server-c2";
  entry.artifact = kModelOpt;
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
  if (!owner.llm || owner.llm->generation_wave_capacity() != 2 ||
      owner.llm->runner().slots().stages() != 2)
    return std::unexpected("Server did not build the declared C2 implementation");
  auto identity = owner.llm->ExecutionIdentity();
  if (!identity || !*identity) return std::unexpected("Server omitted its prepared identity");
  if (reopen && **identity != owner.evidence->identity)
    return std::unexpected("Server reopened another selected execution");
  owner.evidence->identity = **identity;
  testing::Test::RecordProperty("server_execution_identity", llmp::base::ToHex(**identity));
  if (life.server->budget() != kBudget)
    return std::unexpected("Server changed the admitted budget");
  testing::Test::RecordProperty("server_plan_floor_bytes",
                                std::to_string(life.server->plan_floor_bytes()));
  return {};
}

rt::Status Run(Owner& owner, bool reopen) {
  auto& server = *owner.life.server;
  auto& llm = *owner.llm;
  auto& evidence = *owner.evidence;
  const std::array<rt::Llm::Branch*, 2> branches{&owner.Branch(0), &owner.Branch(1)};
  if (reopen) {
    // Adopted cold: its spill files hold the conversation, healthy and
    // restorable, until the prompt below hydrates it.
    for (std::uint32_t slot = 0; slot < 2; ++slot)
      if (owner.Branch(slot).history() != evidence.history[slot] || !owner.Branch(slot).spilled() ||
          !llm.runner().slots().restorable(slot) || owner.Branch(slot).turn_checkpoints() != 1)
        return std::unexpected(std::format(
            "Server did not adopt both pending stages and the turn checkpoint (slot {}: history "
            "{}/{} tokens, spilled {}, restorable {}, turn checkpoints {})",
            slot, owner.Branch(slot).history().size(), evidence.history[slot].size(),
            owner.Branch(slot).spilled(), llm.runner().slots().restorable(slot),
            owner.Branch(slot).turn_checkpoints()));
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
    }
  }
  if (auto s = server.EndRequestBranches(llm); !s) return s;
  for (auto* branch : branches)
    if (auto s = llm.SpillIdle(*branch); !s) return s;
  if (!server.DrainKept(rt::Clock::now() + std::chrono::seconds(30)))
    return std::unexpected("Server record writes did not retire");
  return {};
}

TEST(Qwen38EngineServingGpu, NativeC2ServerDefaultOptionsReopen) {
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
}  // namespace c2_server

// The GGUF import's conversations kept across a restart (D-105, D-112),
// through the ordinary Server with its default options (the service's, with
// conversations kept): two owners' prompts, a turn checkpoint each and a
// first turn, spilled and kept; the server stopped and another started on
// the same directories adopts, restores and continues them exactly as an
// uninterrupted run does (kept_restart_serving_checks.h).
class Qwen38GgufKeptGpu : public testing::Test {
 public:  // the harness of the check reaches the fixture
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
  void CheckKeptRestartContinuesExactly(bool expect_drafts);
  bool mtp = false;  // the native target with its MTP drafter (a per-row tap)

  void SetUp() override {
    const auto checkpoint = fs::path(llmp::test_support::ModelsDir()) /
                            "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6";
    const std::string_view id = mtp ? kModelOpt : kGguf;
    std::error_code error;
    if (!fs::exists(Store() / id, error))
      GTEST_FAIL() << "no Qwen3.8 artifact " << id << " in " << Store();
    if (!fs::exists(checkpoint / "tokenizer.json", error) ||
        !fs::exists(checkpoint / "chat_template.jinja", error))
      GTEST_FAIL() << "no Qwen3.8 tokenizer assets in " << checkpoint;
    const auto fixed = CheckFixture(id, mtp);
    ASSERT_TRUE(fixed) << (fixed ? "" : fixed.error());
    // Spill files need direct I/O: scratch in the build tree.
    const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const fs::path root = base != nullptr ? base : ::testing::TempDir();
    fs::create_directories(root, error);
    std::string name = (root / "qwen-gguf-kept-XXXXXX").string();
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    scratch = name;
    life->roles.installed = Store();
    life->roles.spill = scratch / "spill";
    life->roles.state = scratch / "state";
    for (const auto& path : {life->roles.spill, life->roles.state}) {
      ASSERT_TRUE(fs::create_directory(path));
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
    cfg::ModelEntry entry;
    entry.name = "qwen";
    entry.artifact = id;
    if (mtp) entry.drafter = kMtp;
    entry.tokenizer = checkpoint / "tokenizer.json";
    entry.chat_template = checkpoint / "chat_template.jinja";
    entry.overrides["context"] = std::int64_t{2048};
    entry.overrides["max_slots"] = std::int64_t{2};
    life->config.models.push_back(std::move(entry));
    log = std::tmpfile();
    ASSERT_NE(log, nullptr);
  }

  // A server on the ordinary defaults (conversations kept, as the service
  // keeps them) with both owners selected.
  rt::Status Start() {
    life->options = {};
    life->options.keep_conversations = true;
    life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, log);
    if (auto r = life->server->Start(true); !r) return r;
    model = dynamic_cast<rt::Llm*>(life->server->Find("qwen"));
    if (model == nullptr) return std::unexpected("configured Qwen3.8 model missing");
    if (dynamic_cast<rt::Llm*>(model) == nullptr)
      return std::unexpected("Qwen3.8 is not on the component engine");
    if (model->logit_width() != kVocab || model->generation_wave_capacity() != 2 ||
        model->choice_vocabulary() != (mtp ? kModelOptTokens : 0))
      return std::unexpected("wrong serving envelope");
    rt::SwapParts parts;
    if (auto r = life->server->Activate(*model, parts); !r) return r;
    std::array<rt::Llm::Branch*, 2> branches{};
    for (std::uint32_t id = 0; id < 2; ++id) {
      auto branch = model->branch(id);
      if (!branch) return std::unexpected(branch.error());
      branches[id] = *branch;
    }
    return life->server->SelectRequestBranches(*model, branches);
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
        std::fprintf(stderr, "--- server log ---\n%s\n", text.c_str());
      }
      (void)std::fclose(log);
    }
    if (!scratch.empty()) fs::remove_all(scratch);
  }
  rt::Llm::Branch& Branch(std::uint32_t id) { return **model->branch(id); }
  rt::Llm& Engine() { return dynamic_cast<rt::Llm&>(*model); }
  std::string Log() const {
    std::string text;
    (void)std::fflush(log);
    std::rewind(log);
    std::array<char, 4096> buffer{};
    for (std::size_t n = 0; (n = std::fread(buffer.data(), 1, buffer.size(), log)) > 0;)
      text.append(buffer.data(), n);
    (void)std::fseek(log, 0, SEEK_END);  // the server appends after this read
    return text;
  }

  // Every stage's used state of `slot`, hashed. A copied range's region is
  // the stage (each stage's state is one region).
  std::expected<llmp::base::Sha256Digest, std::string> StateHash(std::uint32_t slot) {
    auto& slots = Engine().runner().slots();
    auto& node = life->server->node();
    constexpr std::uint64_t kPart = 1ULL << 20U;
    std::vector<llmp::catalog::ExtentId> staging;
    auto buffer = node.Pinned(kPart, 0, staging);
    if (!buffer) return std::unexpected(buffer.error());
    llmp::base::Sha256 hash;
    std::uint64_t bytes = 0;
    for (std::size_t stage = 0; stage < slots.stages(); ++stage) {
      for (const auto& range : slots.live(slot, stage).used_ranges()) {
        for (std::uint64_t at = 0; at < range.bytes; at += kPart) {
          const en::LiveState::Range part{.region = stage,
                                          .offset = range.offset + at,
                                          .bytes = std::min(kPart, range.bytes - at)};
          auto retired = en::LiveState::CopyRetirement::kUnproven;
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
};

// A conversation kept across a graceful restart continues exactly
// (kept_restart_serving_checks.h), with or without the drafter's drafts.
void Qwen38GgufKeptGpu::CheckKeptRestartContinuesExactly(bool expect_drafts) {
  struct Harness {
    Qwen38GgufKeptGpu& test;
    rt::Server& server() { return *test.life->server; }
    rt::Llm& model() { return *test.model; }
    rt::Llm::Branch& Branch(std::uint32_t id) { return test.Branch(id); }
    std::expected<llmp::base::Sha256Digest, std::string> StateHash(std::uint32_t slot) {
      return test.StateHash(slot);
    }
    std::string Log() const { return test.Log(); }
    rt::Status Restart() {
      // The graceful stop's order: settle and write the records, then stop;
      // the next server adopts what validates.
      test.life->server->Persist(rt::Clock::now() + std::chrono::seconds(120), {});
      if (auto r = test.Retire(); !r) return r;
      return test.Start();
    }
  } harness{*this};
  const auto started = Start();
  ASSERT_TRUE(started) << (started ? "" : started.error());
  llmp::test::KeptRestartPlan plan;
  plan.vocab = kVocab;
  plan.expect_drafts = expect_drafts;
  plan.tokens = [](std::uint32_t owner, std::size_t count) {
    std::vector<std::int32_t> out;
    out.reserve(count);
    for (std::size_t p = 0; p < count; ++p)
      out.push_back(static_cast<std::int32_t>(100 + ((p * 7919U + 13U + owner * 17U) % 150000U)));
    return out;
  };
  const auto r = llmp::test::CheckKeptRestart(harness, plan);
  ASSERT_TRUE(r) << (r ? "" : r.error());
  EXPECT_EQ(Engine().runner().coverage().violations, 0U);
}

TEST_F(Qwen38GgufKeptGpu, ConversationKeptAcrossARestartContinuesExactly) {
  CheckKeptRestartContinuesExactly(false);
}

// The same with the native target and its MTP drafter, whose feature tap is
// a cell per chunk row (its held positions are host bookkeeping beside the
// kept state).
class Qwen38MtpKeptGpu : public Qwen38GgufKeptGpu {
 protected:
  Qwen38MtpKeptGpu() { mtp = true; }
};
TEST_F(Qwen38MtpKeptGpu, ConversationKeptAcrossARestartContinuesExactly) {
  CheckKeptRestartContinuesExactly(true);
}
