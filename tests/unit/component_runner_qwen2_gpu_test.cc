// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The real FP16 fixture through the shared model runner. External oracles
// are the frozen P0 bridge's 76-row control and 577-row heldout outputs,
// checked by their recorded hashes. Full controls also exercise checkpoint
// rewind and in-process spill. Repeated controls also evict every weight
// extent and reload its backing before continuing. EXL3, address relocation
// and kept restart adoption remain separate gates.
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "components/state_encoding.h"
#include "components/step.h"
#include "corpus_score.h"
#include "engine/component_source.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/plan_cache.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/step_program.h"
#include "planner/pattern.h"
#include "platform/host_probe.h"
#include "runtime/memory_guard.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen2.h"
#include "spec/stored.h"
#include "v1_frozen.h"

namespace {
namespace fs = std::filesystem;
namespace ar = llmp::artifact;
namespace co = llmp::components;
namespace en = llmp::engine;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
namespace sp = llmp::spec;
namespace ts = llmp::test_support;

// Qwen2.5 0.5B Instruct's vocabulary (the fixture's: its logit rows).
constexpr std::uint32_t kVocab = 151936;

// llama.cpp's n_kv (llama_kv_cache::get_n_kv at b29c606e2): the attended
// cells, the used cells padded to 256 and at most the cache's size.
std::uint32_t PaddedKv(std::uint32_t used_cells, std::uint32_t cells) {
  constexpr std::uint32_t kPad = 256;
  const std::uint64_t padded = (std::uint64_t{used_cells} + kPad - 1) / kPad * kPad;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(cells, std::max<std::uint64_t>(kPad, padded)));
}

std::string Env(const char* name) {
  const char* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe)
  return value ? value : "";
}
// Validate the opened regular file before allocating. Reading is bounded
// even if it changes after fstat; a short read or an extra byte is refused.
std::expected<std::string, std::string> Read(const fs::path& path, std::size_t limit,
                                             bool exact = false) {
  struct File {
    int fd;
    ~File() {
      if (fd >= 0) (void)::close(fd);
    }
  } file{::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK)};
  struct stat st{};
  if (file.fd < 0 || ::fstat(file.fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
    return std::unexpected("fixture is not a readable regular file");
  const auto size = static_cast<std::uint64_t>(st.st_size);
  if (size > limit || (exact && size != limit))
    return std::unexpected("fixture file size differs from its bound");
  std::string bytes(static_cast<std::size_t>(size), '\0');
  std::size_t used = 0;
  while (used < bytes.size()) {
    const auto count = ::read(file.fd, bytes.data() + used, bytes.size() - used);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return std::unexpected("fixture read ended before its checked size");
    used += static_cast<std::size_t>(count);
  }
  char extra = 0;
  ssize_t count = 0;
  do {
    count = ::read(file.fd, &extra, 1);
  } while (count < 0 && errno == EINTR);
  if (count != 0) return std::unexpected("fixture changed or failed after its checked size");
  return bytes;
}
// Bound the prepared safetensors headers too: Artifact::Open checks their
// canonical bytes by bounded streaming comparison within the paid metadata peak.
en::Status HeaderBounds(const fs::path& root) {
  std::error_code error;
  std::size_t files = 0;
  std::uint64_t total = 0;
  fs::directory_iterator at(root / "data", error), end;
  if (error) return std::unexpected("cannot inspect fixture shard headers");
  for (; at != end; at.increment(error)) {
    if (error || ++files > 8) return std::unexpected("too many fixture shards");
    const int fd = ::open(at->path().c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) return std::unexpected("cannot inspect fixture shard");
    struct stat st{};
    std::array<unsigned char, 8> header{};
    const bool regular = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode);
    ssize_t got = -1;
    if (regular) {
      do {
        got = ::pread(fd, header.data(), header.size(), 0);
      } while (got < 0 && errno == EINTR);
    }
    (void)::close(fd);
    if (got != 8) return std::unexpected("fixture shard has no bounded header");
    std::uint64_t bytes = 0;
    for (std::size_t i = 0; i < header.size(); ++i) bytes |= std::uint64_t{header[i]} << (8 * i);
    if (bytes > 256 * 1024U || total > 1024 * 1024U - bytes)
      return std::unexpected("fixture shard headers exceed paid parser bound");
    total += bytes;
  }
  if (error || files == 0) return std::unexpected("cannot inspect fixture shard inventory");
  return {};
}
std::string Hash(const std::string& bytes) {
  llmp::base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(bytes)));
  return llmp::base::ToHex(hash.Finish());
}

// Released after the runner, once the node has fenced its work.
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

pl::StepShape Shape(std::uint32_t past, std::uint32_t rows) {
  return {.phase = rows == 1 ? pl::Phase::kDecode : pl::Phase::kPrefill,
          .segments = {{.slot = 0, .rows = rows, .past = past, .read_cells = 256, .outputs = rows}},
          .read_align = 256};
}

// Same reviewed hard-cap inventory as the DeepSeek first screen: 2,560
// MiB compiler, 1,024 MiB metadata/retained native programs, 512 MiB for
// the retained initial IR and observations. The external funding note names
// each phase/owner, including BoundGraph::Bind's transient copies. These are
// source bounds, not measured peaks. Guard metadata before parsing and
// reproduce Lowering::Run's descriptor reservation before TensorArena.
constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
constexpr std::uint64_t kHostAllowance = 4096 * kMiB;
constexpr std::uint64_t kStateCopyLimit = 16 * kMiB;
static_assert(sizeof(en::SlotLifecycle::Range) <= 24);
constexpr std::uint64_t kStateMetadata = 3 * 4 * 512 * 24;
constexpr std::uint64_t kProgramHost = 64 * kMiB;
constexpr std::uint64_t kLaunches = 4096;
constexpr std::uint64_t kRuntimePlans = 2 * (kProgramHost + kLaunches * en::kGraphNodeHostBytes);
static_assert(llmp::ir::Graph::kMaxNodes == 65536 && llmp::ir::Graph::kMaxTensors == 131072);
static_assert(sizeof(void*) == 8 && sizeof(std::string) <= 32 && sizeof(std::vector<void*>) <= 24);
static_assert(sizeof(llmp::ir::Node) <= 160 && sizeof(llmp::ir::TensorDesc) <= 160);
static_assert(sizeof(llmp::ir::Binding) <= 24 && sizeof(llmp::format::TensorFormat) <= 72);
static_assert(sizeof(pl::Selected) <= 128 && sizeof(pl::Applied) <= 64);
static_assert(sizeof(pl::Match) + sizeof(void*) <= 256);
static_assert(sizeof(pl::RootAlignmentFact) <= 16 && sizeof(kn::InputSlot) <= 512);
static_assert(sizeof(std::pair<llmp::ir::TensorDesc, std::uint64_t>) <= 176);
static_assert(sizeof(kg::PlanStep) <= 64 && sizeof(kn::TensorUse) <= 64);
static_assert(llmp::execution::Registry::kMaxImplementations == 256);
static_assert(llmp::ir::kMaxSources == 10 && llmp::ir::kMaxOutputs == 4);
static_assert(sizeof(llmp::execution::OperandMemory) <= 48 &&
              sizeof(llmp::execution::OperandFormat) <= 112 &&
              sizeof(llmp::execution::CapabilityQuery) <= 256);
// The inherited inventory's largest overlapping phase (seven graphs;
// this target uses fewer) and the fixed screen's additional live owners.
constexpr std::uint64_t kInheritedCompiler = 2526477744ULL;
struct BoundStepAbi {
  std::variant<kg::Kernel, kg::RmsNormMulKernel> kernel;
  std::vector<void*> nodes;
  std::uint8_t lane = 0;
};
static_assert(sizeof(BoundStepAbi) <= 128 && sizeof(llmp::execution::Choice) <= 40 &&
              sizeof(llmp::execution::Plan::Step) <= 80);
constexpr std::uint64_t kBindCopies =
    3 * kLaunches * (40 + 80 + 2 * 129 + 128 + 10 * 8 + 8 + 128) + 256 * 1024 + kMiB;
static_assert(kBindCopies == 10182656);
constexpr std::uint64_t kMetadataOwners =
    3 * ar::kMaxDirectoryEntries * (32 + 2 * 256) +  // directory names/growth
    192 * kMiB +                                     // overlapping bounded JSON documents
    128 * kMiB +                                     // artifact/spec/bindings/catalog owners
    2 * 192 * kMiB + kBindCopies;                    // retained programs + active Bind copies
constexpr std::uint64_t kObservationOwners =
    209977600 +                                       // one extra hard-cap IR graph envelope
    std::uint64_t{76} * 151936 * sizeof(float) +      // complete oracle
    3 * std::uint64_t{32} * 151936 * sizeof(float) +  // outputs, joined row copy, growth overlap
    16 * kMiB +                                       // histories, script, headers, rounding
    2 * kStateCopyLimit + kStateMetadata;             // state copies and bounded range lists
static_assert(kInheritedCompiler <= 2560 * kMiB);
static_assert(kMetadataOwners <= 1024 * kMiB);
static_assert(kObservationOwners <= 512 * kMiB);
static_assert(kInheritedCompiler + kMetadataOwners + kObservationOwners <= kHostAllowance);
constexpr std::uint64_t kDomainCap = std::uint64_t{8} << 30U;
constexpr std::size_t kTokenLimit = 1024;
constexpr std::size_t kOracleBytes = std::size_t{76} * 151936 * sizeof(float);
constexpr std::size_t kHeldoutOracleBytes = std::size_t{577} * 151936 * sizeof(float);
constexpr std::uint64_t kHeldoutHostAllowance = 8192 * kMiB;
constexpr std::uint64_t kHeldoutDomainCap = std::uint64_t{16} << 30U;
constexpr std::uint64_t kHeldoutRuntimePlans =
    5 * (kProgramHost + kLaunches * en::kGraphNodeHostBytes);
constexpr std::uint64_t kHeldoutMetadataOwners = kMetadataOwners + 3 * 192 * kMiB;
constexpr std::uint64_t kHeldoutObservationOwners =
    209977600 + kHeldoutOracleBytes + 3 * std::uint64_t{512} * 151936 * sizeof(float) + 16 * kMiB +
    2 * kStateCopyLimit + kStateMetadata;
static_assert(kHeldoutMetadataOwners <= 2048 * kMiB);
static_assert(kHeldoutObservationOwners <= 2048 * kMiB);
static_assert(kInheritedCompiler + kHeldoutMetadataOwners + kHeldoutObservationOwners <=
              kHeldoutHostAllowance);

en::Status NativeReservation(const kn::LowerInput& input) {
  const auto& p = input.plan.graph;
  const auto& g = input.token->step.graph;
  if (input.token->shape.stage != 0 || input.token->step.inputs.size() > 256 ||
      input.token->step.weights.size() > 4096 || input.token->step.state.size() > 4096 ||
      !input.token->step.saves.empty())
    return std::unexpected("screen bindings exceed paid single-stage inventory");
  std::uint64_t descriptors = p.tensor_count() + 16;
  for (const auto& applied : input.plan.applied)
    for (const auto replaced : applied.replaced) {
      if (replaced.v >= g.nodes().size())
        return std::unexpected("screen malformed original replacement map");
      descriptors += g.node(replaced).nout;
    }
  using O = llmp::ir::Op;
  for (const auto& n : p.nodes()) {
    if (n.key == llmp::ir::Primitive(O::kRoute)) descriptors += 10;
    if (n.key == llmp::ir::Primitive(O::kFill) || n.key == llmp::ir::Primitive(O::kConvert))
      ++descriptors;
    if (n.key == llmp::ir::Primitive(O::kExpertReduce)) {
      if (n.nsrc == 0 || p.tensor(n.src[0]).ne(1) > 10)
        return std::unexpected("screen expert reduction bound");
      descriptors += 3 + 2 * p.tensor(n.src[0]).ne(1);
    }
    if (n.key.op == O::kFused || n.key.op == O::kMatMul || n.key.op == O::kWeight ||
        n.key.op == O::kGatedDeltaNet)
      descriptors += 8;
  }
  if (descriptors > 65536) return std::unexpected("screen native descriptors exceed paid bound");
  std::uint64_t traversal = 2;
  while (traversal / 2 < descriptors) traversal *= 2;
  if (descriptors * ggml_tensor_overhead() + traversal * sizeof(void*) + 4096 > (32U << 20U))
    return std::unexpected("screen native arena exceeds paid bound");
  return {};
}

// Bounded metadata of the programs actually bound and entered. A completed
// scorer run is required before these records are emitted as evidence.
static_assert(kInheritedCompiler + kMetadataOwners + kObservationOwners + 8 * kMiB <=
              kHostAllowance);  // Selected-plan name arrays, output text and sink records.
constexpr std::size_t kPlanTextLimit = 2 * kMiB;
constexpr std::uint64_t kPlanRecordsBound = 5 * 4096 * (sizeof(std::string) + 129);
static_assert(kPlanRecordsBound + 2 * kPlanTextLimit + 128 * 1024 <= 8 * kMiB);
static_assert(kInheritedCompiler + kHeldoutMetadataOwners + kHeldoutObservationOwners + 8 * kMiB <=
              kHeldoutHostAllowance);

struct Lifetime {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  std::string tokens_bytes, oracle;
  std::vector<std::int32_t> tokens;
  std::optional<ar::Artifact> artifact;
  std::vector<sp::SourceValue> values;
  std::optional<sp::Source> source;
  std::optional<sp::ModelSpec> spec;
  std::optional<llmp::model::StateLayoutDerived> layout;
  std::optional<co::StepGraph> step;
  std::vector<pl::StepShape> shapes;
  std::map<llmp::ir::WeightRef, std::uint32_t> resource;
  std::unique_ptr<KernelSource> kernel;
  std::unique_ptr<en::ModelRunner> runner;
  std::vector<en::PagedModel*> entered;
  // Pinned buffers belong to node through teardown, including an uncertain copy.
  void* saved_state = nullptr;
  void* state_readback = nullptr;
  std::vector<llmp::catalog::ExtentId> state_copy_extents;
  std::vector<en::SlotLifecycle::Range> saved_pages, saved_footprint;
  std::uint32_t saved_positions = 0;
  std::uint64_t saved_bytes = 0;
  std::uint64_t domain_cap = kDomainCap;
  std::optional<llmp::base::Sha256Digest> checkpoint_reference;
  std::array<ts::CorpusPlanRecord, 5> corpus_plans{};
  std::array<std::optional<pl::StepShape>, 5> recorded_shapes{};
  std::optional<en::WaveComposer> corpus_composer;
  std::unique_ptr<ts::CorpusScore> corpus_scorer;
  std::array<ts::CorpusTargetScore, 256> corpus_scores{};
  std::size_t corpus_score_count = 0;
};

en::Status SaveState(Lifetime& life, bool checkpoint) {
  auto& slots = life.runner->slots();
  if (slots.stages() != 1 || slots.layout(0).bytes > kStateCopyLimit)
    return std::unexpected("FP16 state exceeds the paid single-stage copy bound");
  life.saved_positions = slots.positions(0);
  life.saved_footprint = slots.live(0, 0).used_ranges();
  if (checkpoint) {
    auto pages = slots.CheckpointRanges(0, life.saved_positions);
    if (!pages) return std::unexpected(pages.error());
    life.saved_pages = std::move(*pages);
  } else {
    life.saved_pages = life.saved_footprint;
  }
  if (life.saved_pages.empty() || life.saved_pages.size() > 512 ||
      life.saved_footprint.size() > 512)
    return std::unexpected("FP16 state copy range inventory exceeds its bound");
  life.saved_bytes = 0;
  for (const auto& range : life.saved_pages) {
    if (range.region != 0 || range.bytes == 0 || range.bytes > kStateCopyLimit - life.saved_bytes)
      return std::unexpected("FP16 state copy bytes exceed their bound");
    life.saved_bytes += range.bytes;
  }
  if (!life.saved_state) {
    const auto occupied = life.node.catalog().OccupancyOf(life.node.domain()).Total().value();
    if (occupied > life.domain_cap || 2 * kStateCopyLimit > life.domain_cap - occupied)
      return std::unexpected("FP16 state buffers exceed the checked domain");
    auto saved = life.node.Pinned(kStateCopyLimit, 0, life.state_copy_extents);
    if (!saved) return std::unexpected(saved.error());
    life.saved_state = *saved;
    auto readback = life.node.Pinned(kStateCopyLimit, 0, life.state_copy_extents);
    if (!readback) return std::unexpected(readback.error());
    life.state_readback = *readback;
  }
  return slots.CopyState(0, life.saved_state, life.saved_pages, true);
}

en::Status CheckSavedState(Lifetime& life) {
  auto& slots = life.runner->slots();
  const auto footprint = slots.live(0, 0).used_ranges();
  if (slots.positions(0) != life.saved_positions || slots.spilled(0) || slots.restoring(0) ||
      footprint.size() != life.saved_footprint.size())
    return std::unexpected("FP16 restored state counters or footprint differ");
  for (std::size_t i = 0; i < footprint.size(); ++i)
    if (footprint[i].region != life.saved_footprint[i].region ||
        footprint[i].offset != life.saved_footprint[i].offset ||
        footprint[i].bytes != life.saved_footprint[i].bytes)
      return std::unexpected("FP16 restored whole-extent footprint differs");
  if (auto copied = slots.CopyState(0, life.state_readback, life.saved_pages, true); !copied)
    return copied;
  if (std::memcmp(life.saved_state, life.state_readback,
                  static_cast<std::size_t>(life.saved_bytes)) != 0)
    return std::unexpected("FP16 restored state bytes differ");
  return {};
}

// Covers assertion returns and every setup failure too. Teardown is once;
// any refusal retains the complete graph of owners, including the node and
// catalog payment. No provider, borrowed spec or callback target is freed
// while a pending state preparation or device operation may still use it.
struct Retire {
  void operator()(Lifetime* life) const {
    const auto retired = life->node.TearDown(life->entered);
    EXPECT_TRUE(retired.has_value()) << (retired ? "" : retired.error());
    if (retired) delete life;
  }
};
using Owner = std::unique_ptr<Lifetime, Retire>;

bool FitsDomain(std::uint64_t available, std::uint64_t cap = kDomainCap) {
  return available >= cap && available - cap >= llmp::runtime::kUncountedMargin;
}

TEST(ComponentRunnerQwen2Inputs, ChecksTheWholeDomainBeforeStartup) {
  EXPECT_FALSE(FitsDomain(0));
  EXPECT_FALSE(FitsDomain(kDomainCap - 1));
  EXPECT_FALSE(FitsDomain(kDomainCap + llmp::runtime::kUncountedMargin - 1));
  EXPECT_TRUE(FitsDomain(kDomainCap + llmp::runtime::kUncountedMargin));
  EXPECT_TRUE(FitsDomain(std::numeric_limits<std::uint64_t>::max()));
  EXPECT_FALSE(FitsDomain(kDomainCap + llmp::runtime::kUncountedMargin, kHeldoutDomainCap));
  EXPECT_FALSE(
      FitsDomain(kHeldoutDomainCap + llmp::runtime::kUncountedMargin - 1, kHeldoutDomainCap));
  EXPECT_TRUE(FitsDomain(kHeldoutDomainCap + llmp::runtime::kUncountedMargin, kHeldoutDomainCap));
}

TEST(ComponentRunnerQwen2Inputs, RefusesOversizeAndWrongLengthFilesBeforeReading) {
  struct Temporary {
    std::string path = testing::TempDir() + "/qwen2-input-XXXXXX";
    int fd = ::mkstemp(path.data());
    ~Temporary() {
      if (fd >= 0) {
        (void)::close(fd);
        (void)::unlink(path.c_str());
      }
    }
  } file;
  ASSERT_GE(file.fd, 0);
  ASSERT_EQ(::write(file.fd, "abcde", 5), 5);
  const auto exact = Read(file.path, 5, true);
  ASSERT_TRUE(exact.has_value()) << exact.error();
  EXPECT_EQ(*exact, "abcde");
  EXPECT_FALSE(Read(file.path, 4));
  EXPECT_FALSE(Read(file.path, 6, true));
  ASSERT_EQ(::ftruncate(file.fd, static_cast<off_t>(kOracleBytes + 1)), 0);
  const auto oversized = Read(file.path, kOracleBytes, true);
  ASSERT_FALSE(oversized.has_value());
  EXPECT_EQ(oversized.error(), "fixture file size differs from its bound");
  EXPECT_FALSE(Read(testing::TempDir(), kTokenLimit));
}

// SHA-256 over the saved checkpoint's page coordinates and bytes.
llmp::base::Sha256Digest CheckpointDigestRaw(const Lifetime& life) {
  llmp::base::Sha256 digest;
  for (const auto& range : life.saved_pages) {
    digest.Update(std::as_bytes(std::span(&range.region, 1)));
    digest.Update(std::as_bytes(std::span(&range.offset, 1)));
    digest.Update(std::as_bytes(std::span(&range.bytes, 1)));
  }
  digest.Update(std::span(static_cast<const std::byte*>(life.saved_state),
                          static_cast<std::size_t>(life.saved_bytes)));
  return digest.Finish();
}
std::string CheckpointDigest(const Lifetime& life) {
  return llmp::base::ToHex(CheckpointDigestRaw(life)) + ':' + std::to_string(life.saved_bytes) +
         ':' + std::to_string(life.saved_pages.size());
}

// (fusion, full trajectory, heldout, repeated). Every output row is requested so narrowing
// cannot change the legacy chunk's matrix shapes.
class ComponentRunnerQwen2Gpu : public testing::TestWithParam<std::tuple<bool, bool, bool, bool>> {
};
void RunQwen2Control(bool fusion, bool full, bool heldout, bool repeated, bool corpus = false,
                     bool same_plan = false, bool q8_cache = false) {
  ASSERT_FALSE(q8_cache && (heldout || (!corpus && !same_plan)));
  ASSERT_FALSE(same_plan && (corpus || !full || !repeated));
  const auto domain_cap = heldout ? kHeldoutDomainCap : kDomainCap;
  const auto host_allowance = heldout ? kHeldoutHostAllowance : kHostAllowance;
  const auto runtime_plans = heldout ? kHeldoutRuntimePlans : kRuntimePlans;
  std::uint32_t token_rows = heldout ? 577U : 76U;
  const auto checkpoint_positions = heldout ? 33U : 32U;
  const auto oracle_bytes = heldout ? kHeldoutOracleBytes : kOracleBytes;
  const std::string tokens_path =
      Env((corpus || (same_plan && !heldout))
              ? "LLMP_QWEN2_CORPUS_IDS"
              : (heldout ? "LLMP_QWEN2_FP16_HELDOUT_IDS" : "LLMP_QWEN2_FP16_TOKENS"));
  const std::string oracle_path = Env(
      heldout ? (fusion ? "LLMP_QWEN2_FP16_HELDOUT_ORACLE_F" : "LLMP_QWEN2_FP16_HELDOUT_ORACLE_U")
              : (fusion ? "LLMP_QWEN2_FP16_ORACLE_F" : "LLMP_QWEN2_FP16_ORACLE_U"));
  if (tokens_path.empty() || (!corpus && !same_plan && oracle_path.empty()))
    GTEST_SKIP() << "requires the frozen FP16 trajectory IDs and fused/unfused oracle";
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available.has_value()) << "unknown physical availability";
  ASSERT_TRUE(FitsDomain(*available, domain_cap))
      << "the complete domain and driver margin do not fit";
  Owner life(new Lifetime);
  life->domain_cap = domain_cap;
  auto& node = life->node;
  node.SetHostFloor(0);
  const auto opened_node = node.Open();
  ASSERT_TRUE(opened_node.has_value()) << opened_node.error();
  const auto occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_LE(occupied, domain_cap - host_allowance);
  const auto payment =
      node.catalog().AddExtent({.domain = node.domain(),
                                .memory_class = llmp::catalog::MemoryClass::kRuntime,
                                .recovery = llmp::catalog::Recovery::kPinned,
                                .size = llmp::base::Bytes(host_allowance),
                                .content = {}},
                               true);
  ASSERT_TRUE(payment.has_value());
  ASSERT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), occupied + host_allowance);
  auto token_read = Read(tokens_path, heldout ? 8320 : kTokenLimit, heldout);
  ASSERT_TRUE(token_read.has_value()) << token_read.error();
  life->tokens_bytes = std::move(*token_read);
  auto& tokens = life->tokens;
  if (corpus || (same_plan && !heldout)) {
    const auto seal = Env("LLMP_QWEN2_CORPUS_IDS_SHA256");
    ASSERT_EQ(seal.size(), 64U);
    ASSERT_EQ(Hash(life->tokens_bytes), seal);
    ASSERT_EQ(life->tokens_bytes.size() % 4, 0U);
    token_rows = static_cast<std::uint32_t>(life->tokens_bytes.size() / 4);
    ASSERT_GE(token_rows, 2U);
    ASSERT_LE(token_rows, 256U);
    if (same_plan) {
      ASSERT_EQ(token_rows, 76U);
      life->oracle.resize(kOracleBytes);  // Existing prepaid full-row buffer, filled by pass1.
    }
    for (std::size_t i = 0; i < token_rows; ++i) {
      std::uint32_t id = 0;
      for (int byte = 3; byte >= 0; --byte)
        id = (id << 8U) |
             static_cast<unsigned char>(life->tokens_bytes[i * 4 + static_cast<std::size_t>(byte)]);
      ASSERT_LT(id, 151936U);
      tokens.push_back(static_cast<std::int32_t>(id));
    }
  } else {
    ASSERT_EQ(Hash(life->tokens_bytes),
              heldout ? "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c"
                      : "37e46a2304b5ab5648d0f2b7fdb94224596ed690caaddc1726df27110b585e21");
    if (heldout) {
      for (std::size_t i = 0; i < token_rows; ++i) {
        std::uint64_t id = 0;
        for (int byte = 7; byte >= 0; --byte)
          id = (id << 8U) | static_cast<unsigned char>(
                                life->tokens_bytes[i * 8 + static_cast<std::size_t>(byte)]);
        ASSERT_LT(id, 151936U);
        tokens.push_back(static_cast<std::int32_t>(id));
      }
    } else {
      std::istringstream ids(life->tokens_bytes);
      for (std::int32_t id = 0; ids >> id;) tokens.push_back(id);
    }
    ASSERT_EQ(tokens.size(), token_rows);
    if (same_plan) {
      life->oracle.resize(oracle_bytes);  // Reuse the sole paid heldout oracle for pass1.
    } else {
      auto oracle_read = Read(oracle_path, oracle_bytes, true);
      ASSERT_TRUE(oracle_read.has_value()) << oracle_read.error();
      life->oracle = std::move(*oracle_read);
      ASSERT_EQ(
          Hash(life->oracle),
          heldout ? (fusion ? "bfb36f192317c4a32723e65753dbe6a75f4c5e45d4d885736d2b26e92431e9d5"
                            : "69ff08210fb2450388e32401091eed2bb02df7325ca966268a9076009a407ae0")
                  : (fusion ? "bb8ae5e7e3ac6da734173edb1111160a0c80a55c4279b94e67a8f90b142e7571"
                            : "3560d3370b167ce87d4134b0be70c0a59aa230db469f1bbbc5fb4a6b5829d6b1"));
    }
  }
  const auto& oracle = life->oracle;
  const std::size_t row_bytes = std::size_t{kVocab} * sizeof(float);
  if (!corpus) ASSERT_EQ(oracle.size(), token_rows * row_bytes);
  // The published v1 import of the fixture's GGUF (LLMP_TEST_V1_STORE and its
  // v1-ids.json), held to the sealed bridge oracle and the frozen checkpoint
  // digests (which its v0 import also met, bit for bit).
  const std::string store = ts::V1Store();
  const auto mapped = ts::V1IdNamed("qwen2-fp16");
  if (!mapped) GTEST_FAIL() << "no published v1 FP16 fixture in " << store;
  const std::string artifact_id = *mapped;
  const fs::path root = fs::path(store) / artifact_id;
  if (!fs::is_directory(root)) GTEST_FAIL() << "fixture not installed: " << root;
  // These pure filesystem bounds precede both artifact parser calls. The
  // checked manifest identity fixes the index's declared length/hash too.
  std::error_code metadata_error;
  const auto manifest_bytes = fs::file_size(root / "manifest.json", metadata_error);
  ASSERT_FALSE(metadata_error);
  ASSERT_LE(manifest_bytes, 64 * 1024U);
  const auto index_bytes = fs::file_size(root / "index.json", metadata_error);
  ASSERT_FALSE(metadata_error);
  ASSERT_LE(index_bytes, kMiB);
  const auto headers = HeaderBounds(root);
  ASSERT_TRUE(headers.has_value()) << headers.error();
  auto opened = ar::Artifact::Open(root, {.expected_id = artifact_id});
  ASSERT_TRUE(opened.has_value()) << opened.error().ToString();
  ASSERT_EQ(opened->format_version(), 1U);
  life->artifact.emplace(std::move(*opened));
  auto& artifact = life->artifact;
  ASSERT_LE(artifact->resources().size(), 512U);
  ASSERT_LE(artifact->groups().size(), 64U);
  ASSERT_LE(artifact->chunk_sha256().size(), 1024U);
  const auto& mapping = sp::mappings::Qwen2Mapping();
  // The stored spec and the artifact's own tensor formats execute unchanged;
  // the mapping's policy has no gate, so its plans take no values.
  auto bound_source = sp::SourceOf(*artifact);
  ASSERT_TRUE(bound_source.has_value()) << bound_source.error();
  life->source.emplace(std::move(*bound_source));
  auto stored = sp::StoredSpec(*artifact);
  ASSERT_TRUE(stored.has_value()) << stored.error();
  life->spec.emplace(std::move(*stored));
  auto& source = life->source;
  ASSERT_LE(source->tensors.size(), 512U);
  for (const auto& tensor : source->tensors) ASSERT_LE(tensor.name.size(), 128U);
  auto& spec = life->spec;
  // Qwen2.5 0.5B: 24 layers of width 896.
  ASSERT_EQ(spec->stages.front().stream.width, 896U);
  if (q8_cache) {
    const auto bytes = co::KvCacheEncodingBytes(*spec, llmp::format::Encoding::kQ8_0);
    ASSERT_TRUE(bytes) << bytes.error();
    // One MiB of the existing 128 MiB artifact/spec/binding owner category;
    // no second allowance or unbounded copied model.
    ASSERT_LE(*bytes, kMiB);
    ASSERT_TRUE(co::ApplyKvCacheEncoding(*spec, llmp::format::Encoding::kQ8_0, *bytes));
  }
  testing::Test::RecordProperty("kv_cache_encoding", q8_cache ? "q8_0" : "f16");
  // Frozen Bridge files describe the historical explicit-attention recipe,
  // independently of the selected production default. Never rewrite their oracle.
  if (!corpus && !same_plan) {
    std::uint32_t historical = 0;
    for (auto& layer : spec->stages.front().layers)
      for (auto& component : layer.components)
        if (component.kind == "attention.dense")
          for (auto& parameter : component.params)
            if (parameter.name == "explicit_scores") {
              ASSERT_TRUE(std::holds_alternative<bool>(parameter.value));
              parameter.value = true;
              ++historical;
            }
    ASSERT_EQ(historical, 24U);
    testing::Test::RecordProperty("attention_recipe", "historical-explicit-scores");
  } else {
    testing::Test::RecordProperty("attention_recipe", "selected-mapping-default");
  }
  ASSERT_EQ(spec->stages.size(), 1U);
  ASSERT_EQ(spec->stages.front().layers.size(), 24U);
  std::uint64_t components = 0, parameters = 0, weights_bound = 0;
  const auto count_component = [&](const sp::ComponentRef& component) {
    ++components;
    parameters += component.params.size();
    weights_bound += component.weights.size();
  };
  const auto& stage = spec->stages.front();
  ASSERT_TRUE(stage.side.empty());
  ASSERT_FALSE(stage.stream.mixing.has_value());
  ASSERT_TRUE(stage.heads.mtp.empty());
  if (stage.embedding) count_component(*stage.embedding);
  for (const auto& layer : stage.layers)
    for (const auto& component : layer.components) count_component(component);
  if (stage.heads.final_norm) count_component(*stage.heads.final_norm);
  if (stage.heads.output) count_component(*stage.heads.output);
  ASSERT_EQ(components, 51U);
  ASSERT_LE(parameters, 2048U);
  ASSERT_LE(weights_bound, 512U);
  auto derived = co::StageLayout(*spec, 0, heldout ? 1024 : 512, heldout ? 512 : 32);
  ASSERT_TRUE(derived.has_value()) << derived.error();
  life->layout.emplace(std::move(*derived));
  auto& layout = life->layout;
  std::uint64_t logical_state = 0;
  for (const auto& item : layout->items) logical_state += item.bytes;
  testing::Test::RecordProperty("state_logical_bytes", std::to_string(logical_state));
  testing::Test::RecordProperty("state_reserved_bytes", std::to_string(layout->bytes));
  auto& shapes = life->shapes;
  const auto shape_of = [heldout](std::uint32_t past, std::uint32_t rows) {
    auto shape = Shape(past, rows);
    if (heldout) shape.segments.front().read_cells = PaddedKv(past + rows, 1024);
    return shape;
  };
  if (corpus) {
    for (std::uint32_t past = 0; past < token_rows; past += 8) {
      auto shape = shape_of(past, std::min(8U, token_rows - past));
      shape.phase = pl::Phase::kScore;
      life->corpus_plans[shape.rows() == 8 ? 0U : 1U].rows = shape.rows();
      shapes.push_back(std::move(shape));
    }
  } else if (heldout) {
    shapes.push_back(shape_of(0, 16));
    shapes.push_back(shape_of(16, 17));
    for (std::uint32_t past = 33; past < 49; ++past) shapes.push_back(shape_of(past, 1));
    shapes.push_back(shape_of(49, 512));
    for (std::uint32_t past = 561; past < 577; ++past) shapes.push_back(shape_of(past, 1));
  } else {
    shapes.push_back(shape_of(0, 32));
    for (std::uint32_t past = 32; past < (full ? 76U : 33U); ++past)
      shapes.push_back(shape_of(past, 1));
  }
  if (same_plan) {
    std::size_t recorded = 0;
    for (const auto& shape : shapes) {
      const auto found = std::find_if(life->recorded_shapes.begin(), life->recorded_shapes.end(),
                                      [&](const auto& key) { return key && *key == shape; });
      if (found != life->recorded_shapes.end()) continue;
      ASSERT_LT(recorded, life->recorded_shapes.size());
      life->recorded_shapes[recorded] = shape;
      life->corpus_plans[recorded++].rows = shape.rows();
    }
    ASSERT_EQ(recorded, heldout ? 5U : 2U);
  }
  auto& resource = life->resource;
  auto built_step =
      co::BuildStep(*spec, 0, shapes.front(), {.tensors = source->tensors, .state = &*layout});
  ASSERT_TRUE(built_step.has_value()) << built_step.error();
  life->step.emplace(std::move(*built_step));
  for (const co::StepWeight& weight : life->step->weights) {
    ASSERT_EQ(weight.parts.size(), 1U);
    const auto found = artifact->FindResource(weight.parts.front().first);
    ASSERT_TRUE(found.has_value()) << weight.parts.front().first;
    resource[weight.ref] = *found;
  }
  pl::Policy policy;
  policy.max_decode_rows = 1;
  if (fusion) {
    policy.disabled = {"owner_attention", "rope_store"};
    policy.enabled = {"float_mul_mat_add", "float_mul_mat_glu"};
    pl::Canonicalize(policy);
  } else {
    for (const auto& pattern : pl::Patterns()) policy.disabled.emplace_back(pattern.name);
    pl::Canonicalize(policy);
  }
  ts::FrozenDigests frozen("qwen2_fp16");
  std::string scratch = Env("LLMP_TEST_SCRATCH");
  if (scratch.empty()) scratch = testing::TempDir();
  const fs::path spill = fs::path(scratch) / "qwen2-shared";
  fs::create_directories(spill);
  life->kernel = std::make_unique<KernelSource>(node);
  auto& kernel = life->kernel;
  auto& runner = life->runner;
  auto& entered = life->entered;
  const auto ran = [&]() -> en::Status {
    entered.push_back(kernel.get());
    if (auto x = kernel->resources->OpenCublas("FP16 shared cuBLAS workspace"); !x) return x;
    auto backend =
        kg::GgmlBackend::Create(0, node.execution(), node.stream(0), &kernel->resources->cublas(),
                                {.lower = {.index_scratch_limit = 32 * kMiB}});
    if (!backend) return std::unexpected(backend.error());
    kernel->backend = std::move(*backend);
    auto weights = std::make_unique<en::PagedWeights>();
    if (auto x = weights->Open(root); !x) return x;
    en::PagedWeights* paged = weights.get();
    auto checked_backend = kernel->backend->Backend();
    const auto lower = checked_backend.lower;
    checked_backend.lower = [lower](const kn::LowerInput& input)
        -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
      if (auto checked = NativeReservation(input); !checked)
        return std::unexpected(checked.error());
      auto program = lower(input);
      if (!program) return std::unexpected(program.error());
      // Before cache installation, required charges or graph capture. The
      // temporary program is still covered by the prepaid compiler row.
      if ((*program)->host_bytes() > kProgramHost || (*program)->launches() > kLaunches)
        return std::unexpected("program exceeds paid retained-plan bound");
      return program;
    };
    if (corpus || same_plan) {
      auto guarded = corpus ? ts::GuardCorpusLower(std::move(checked_backend.lower), kVocab)
                            : std::move(checked_backend.lower);
      checked_backend.lower = [held = life.get(), same_plan,
                               guarded = std::move(guarded)](const kn::LowerInput& input)
          -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
        auto program = guarded(input);
        if (!program || input.addresses.measuring()) return program;
        const auto rows = input.token->shape.rows();
        auto* record = &held->corpus_plans[rows == 8U ? 0U : 1U];
        if (same_plan) {
          const auto found =
              std::find_if(held->recorded_shapes.begin(), held->recorded_shapes.end(),
                           [&](const auto& key) { return key && *key == input.token->shape; });
          if (found == held->recorded_shapes.end())
            return std::unexpected("same-plan program has an unrecorded shape/read class");
          record = &held->corpus_plans[static_cast<std::size_t>(
              std::distance(held->recorded_shapes.begin(), found))];
        }
        if (record->rows != rows ||
            (*program)->host_bytes() > kProgramHost - sizeof(ts::CorpusProgram))
          return std::unexpected("corpus exceeded measured shape or retained program bound");
        return std::make_unique<ts::CorpusProgram>(std::move(*program), *record);
      };
    }
    auto component = en::ComponentStepSource::Create(
        {.spec = &*spec,
         .stage = 0,
         .bindings = {.tensors = source->tensors, .state = &*layout},
         .bound_digest = spec->Digest(),
         .mapping = &mapping,
         .values = life->values,
         .slots = 1,
         .policy = policy,
         .backend = std::move(checked_backend)});
    if (!component) return std::unexpected(component.error());
    // Exactly two control keys, or five historical heldout keys. Position
    // parameters must agree within each shape class before cache admission.
    std::vector<pl::StepShape> keys;
    std::vector<std::vector<std::uint64_t>> key_parameters;
    const auto paid_keys = heldout ? 5U : 2U;
    for (const auto& shape : shapes) {
      const auto found = std::find(keys.begin(), keys.end(), shape);
      const auto parameters = (*component)->ShapeParameters(shape);
      if (found == keys.end()) {
        if (keys.size() >= paid_keys)
          return std::unexpected("shape exceeds the paid trajectory key inventory");
        keys.push_back(shape);
        key_parameters.push_back(parameters);
      } else if (key_parameters[static_cast<std::size_t>(found - keys.begin())] != parameters) {
        return std::unexpected("position changes the paid plan key");
      }
    }
    if ((!corpus && keys.size() != paid_keys) || keys.empty())
      return std::unexpected("trajectory key inventory differs");
    en::RunnerConfig config;
    config.slots = {.model = "qwen2-fp16-shared",
                    .slots = 1,
                    .stages = {{.name = "target", .layout = *layout}},
                    .spec = spec->Digest(),
                    .read_align = 256,
                    .spill_place = {},
                    .spill_directory = spill,
                    .invalidate = {},
                    .drop_plans = {}};
    std::vector<en::GroupPlace> places(paged->artifact().groups().size(), en::GroupPlace::kDevice);
    config.weights.push_back(
        {.weights = std::move(weights), .places = std::move(places), .slabs = {}});
    config.setup_shapes = shapes;
    config.launch =
        [held = life.get()](
            std::uint64_t) -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return held->kernel->backend->Launch(
          {.base = held->node.pool().base, .size = llmp::base::Bytes(held->node.pool().bytes)});
    };
    config.weight_address = [held = life.get(), paged](const llmp::ir::WeightRef& ref,
                                                       llmp::format::PartRole) -> std::uint64_t {
      const auto found = held->resource.find(ref);
      return found == held->resource.end() ? 0 : paged->resource_address(found->second);
    };
    auto made = en::ModelRunner::Create(node, std::move(config), std::move(*component), 0, 0);
    if (!made) return std::unexpected(made.error());
    runner = std::move(*made);
    entered.insert(entered.begin(), runner.get());
    if (auto x = runner->Setup(); !x) return x;
    const auto before_workspace = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto pool = std::max<std::uint64_t>(runner->pool_needed(), en::kPagedExtent);
    // Include both regions' page rounding before MapWorkspace allocates.
    if (before_workspace > domain_cap || 2 * en::kPagedExtent > domain_cap - before_workspace ||
        runner->activations_needed() > domain_cap - before_workspace - 2 * en::kPagedExtent ||
        pool > domain_cap - before_workspace - 2 * en::kPagedExtent - runner->activations_needed())
      return std::unexpected("workspace exceeds the checked domain");
    if (auto x =
            node.MapWorkspace(runner->activations_needed(),
                              std::max<std::uint64_t>(runner->pool_needed(), en::kPagedExtent));
        !x)
      return x;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    if (fixed > domain_cap) return std::unexpected("startup footprint exceeds the checked domain");
    if (paged->extents().size() > (domain_cap - fixed) / en::kPagedExtent)
      return std::unexpected("weight extents exceed the checked domain");
    const auto weights_bytes = paged->extents().size() * en::kPagedExtent;
    if (weights_bytes > domain_cap - fixed || runtime_plans > domain_cap - fixed - weights_bytes ||
        node.StateCapacity() > (domain_cap - fixed - weights_bytes - runtime_plans) / 4)
      return std::unexpected("weights and state exceed the checked domain");
    if (auto x = node.Start(llmp::base::Bytes(domain_cap)); !x) return x;
    if (auto x = runner->Register(); !x) return x;
    if (auto x = runner->Bind(); !x) return x;
    node.Run();
    if (auto x = runner->slots().SelectSlots(std::vector<std::uint32_t>{0}); !x) return x;
    if (corpus) {
      llmp::model::Batching batching;
      batching.max_segments = 1;
      auto composer = en::WaveComposer::For(batching, runner->slots());
      if (!composer) return std::unexpected(composer.error());
      life->corpus_composer.emplace(std::move(*composer));
      const std::array documents{ts::CorpusDocument{.slot = 0, .tokens = tokens}};
      auto scorer = ts::CorpusScore::Make(
          node, *runner, *life->corpus_composer, documents, {kVocab, 8},
          [held = life.get()](const ts::CorpusTargetScore& score) -> en::Status {
            if (held->corpus_score_count == held->corpus_scores.size())
              return std::unexpected("corpus NLL report exceeds prepaid rows");
            held->corpus_scores[held->corpus_score_count++] = score;
            return {};
          });
      if (!scorer) return std::unexpected(scorer.error());
      life->corpus_scorer = std::move(*scorer);
      auto scored = node.WithRequest(
          0, runner->slots().execution(), "fixed teacher corpus", [&]() -> en::Status {
            for (std::size_t i = 0; i < shapes.size(); ++i) {
              auto done = life->corpus_scorer->Step();
              if (!done) return std::unexpected(done.error());
              if (*done != (i + 1 == shapes.size()))
                return std::unexpected("corpus completion differs from fixed chunk schedule");
            }
            return {};
          });
      if (!scored) return scored;
      const auto& result = life->corpus_scorer->result();
      if (!life->corpus_scorer->complete() || result.inputs != token_rows ||
          result.targets != token_rows - 1 || life->corpus_score_count != token_rows - 1 ||
          result.steps != shapes.size() || runner->slots().positions(0) != token_rows)
        return std::unexpected("corpus target coverage or state frontier differs");
      std::ostringstream report;
      report << std::setprecision(17);
      for (std::size_t i = 0; i < life->corpus_score_count; ++i) {
        const auto& score = life->corpus_scores[i];
        if (score.document || score.slot || score.position != i + 1 ||
            score.target != tokens[i + 1])
          return std::unexpected("corpus NLL coordinates differ from fixed teacher IDs");
        report << score.document << ',' << score.slot << ',' << score.position << ','
               << score.target << ',' << score.nll << '\n';
      }
      testing::Test::RecordProperty("corpus_nll_rows", report.str());
      testing::Test::RecordProperty("corpus_ids_sha256", Hash(life->tokens_bytes));
      testing::Test::RecordProperty("corpus_targets", std::to_string(result.targets));
      report.str("");
      report.clear();
      report << std::setprecision(17) << result.total_nll;
      testing::Test::RecordProperty("corpus_total_nll", report.str());
      testing::Test::RecordProperty("corpus_policy",
                                    fusion ? "existing-fused" : "existing-unfused");
      report.str("");
      report.clear();
      for (const auto& plan : life->corpus_plans) {
        if (!plan.rows) continue;
        if (!plan.runs || plan.implementations.empty())
          return std::unexpected("corpus selected program never entered execution");
        if (q8_cache &&
            std::ranges::count(plan.implementations, "ggml.flash_attn_ext.vec_q8_0") != 24)
          return std::unexpected("Q8 corpus did not select all quantized attention operations");
        report << "rows=" << plan.rows << " bound_plan=" << llmp::base::ToHex(plan.identity)
               << " run_or_record=" << plan.runs << '\n';
        for (const auto& name : plan.implementations) report << name << '\n';
      }
      testing::Test::RecordProperty("corpus_selected_programs", report.str());
      return {};
    }
    // Only the repeated arms add this bounded generation inventory. The
    // source-bound allowance includes 128 KiB for it and operation scratch.
    // The artifact's own weight extents, bounded.
    const std::size_t weight_extents = paged->extents().size();
    constexpr std::uint64_t kRepeatObservation = 128 * 1024;
    static_assert(kInheritedCompiler + kMetadataOwners + kObservationOwners + kRepeatObservation <=
                  kHostAllowance);
    static_assert(kInheritedCompiler + kHeldoutMetadataOwners + kHeldoutObservationOwners +
                      kRepeatObservation <=
                  kHeldoutHostAllowance);
    if (repeated && (weight_extents == 0 || weight_extents > 1024))
      return std::unexpected("repeated fixture weight extent inventory differs");
    std::vector<std::pair<std::uint64_t, std::uint64_t>> generations;
    if (repeated) generations.resize(weight_extents);
    bool reload_pending = false;
    std::size_t backing_before = 0;
    std::uint32_t evictions = 0;
    std::uint32_t restored_extents = 0;
    std::uint32_t evaluations = 0;
    std::uint64_t compared_rows = 0;
    std::uint64_t reference_rows = 0;
    const auto evaluate = [&](std::uint32_t evaluation) -> en::Status {
      const auto run_step = [&](const pl::StepShape& shape) -> en::Status {
        const auto& segment = shape.segments.front();
        const std::vector<en::SegmentWork> work{
            {.tokens = std::span(tokens).subspan(segment.past, segment.rows), .bytes = {}}};
        const auto result = runner->RunStep(
            {.shape = shape, .work = work, .step_bytes = {}, .advance = true, .ahead = {}});
        if (!result) return std::unexpected(result.error().what);
        if (result->outputs.size() != 1 ||
            result->outputs.front().size() != segment.rows * row_bytes)
          return std::unexpected("wrong output count or bytes");
        const auto& got = result->outputs.front();
        for (std::size_t i = 0; i < got.size(); i += sizeof(float)) {
          float value = 0;
          std::memcpy(&value, got.data() + i, sizeof(value));
          if (!std::isfinite(value)) return std::unexpected("nonfinite FP16 control head");
        }
        if (same_plan && evaluation == 1) {
          if (segment.past != reference_rows)
            return std::unexpected("live reference is not one uninterrupted ordered trajectory");
          std::memcpy(life->oracle.data() + segment.past * row_bytes, got.data(), got.size());
          reference_rows += segment.rows;
        } else {
          compared_rows += segment.rows;
          const char* expected = oracle.data() + segment.past * row_bytes;
          std::size_t different = 0;
          for (std::size_t i = 0; i < got.size(); i += sizeof(float))
            different += std::memcmp(got.data() + i, expected + i, sizeof(float)) != 0;
          EXPECT_EQ(different, 0U) << "past=" << segment.past << " rows=" << segment.rows;
          if (different != 0)
            return std::unexpected(
                same_plan ? "FP16 replay differs from uninterrupted same-plan reference"
                          : "FP16 shared logits differ from frozen bridge");
        }
        if (runner->slots().positions(0) != segment.past + segment.rows)
          return std::unexpected("FP16 completed positions differ from the control");
        return {};
      };
      for (const auto& shape : shapes) {
        const auto chunk = [&]() -> en::Status {
          if (reload_pending) {
            if (node.memory().backings() != backing_before)
              return std::unexpected("full weight reload backing count differs");
            for (std::size_t i = 0; i < paged->extents().size(); ++i) {
              auto extent = node.catalog().Describe(paged->extents()[i]);
              if (!extent || extent->state != llmp::catalog::ExtentState::kResident ||
                  extent->backing_generation <= generations[i].first ||
                  extent->content_generation != generations[i].second)
                return std::unexpected("full weight reload generations differ");
            }
            // Before any new decode write, the state still equals the
            // complete checkpoint saved at the original restore point.
            if (auto x = CheckSavedState(*life); !x) return x;
            restored_extents += static_cast<std::uint32_t>(paged->extents().size());
            reload_pending = false;
          }
          if (full && (!same_plan || evaluation > 1) &&
              shape.segments.front().past == checkpoint_positions + 8) {
            auto& slots = runner->slots();
            // Eight real decode writes must change the saved checkpoint pages.
            if (auto x = slots.CopyState(0, life->state_readback, life->saved_pages, true); !x)
              return x;
            if (std::memcmp(life->saved_state, life->state_readback,
                            static_cast<std::size_t>(life->saved_bytes)) == 0)
              return std::unexpected("FP16 checkpoint control made no changed state");
            if (auto x = slots.PrepareRestore(0, life->saved_positions, life->saved_footprint,
                                              slots.kept_layout_id());
                !x)
              return x;
            if (!slots.restoring(0))
              return std::unexpected("FP16 checkpoint restore did not enter preparation");
            if (auto x = slots.CopyState(0, life->saved_state, life->saved_pages, false); !x)
              return x;
            if (auto x = slots.CompleteRestore(0, life->saved_positions); !x) return x;
            if (auto x = CheckSavedState(*life); !x) return x;
            testing::Test::RecordProperty("checkpoint_bytes_restored",
                                          std::to_string(life->saved_bytes));
            // The same frozen full-vocabulary rows must survive the rewind.
            for (std::uint32_t past = checkpoint_positions; past < checkpoint_positions + 8; ++past)
              if (auto x = run_step(shape_of(past, 1)); !x) return x;
          }
          if (full && (!same_plan || evaluation > 1) &&
              shape.segments.front().past == (heldout ? 49U : 48U)) {
            if (auto x = SaveState(*life, false); !x) return x;
            auto& slots = runner->slots();
            if (auto x = slots.Spill(0); !x) return x;
            if (!slots.spilled(0) || !slots.kept_whole(0))
              return std::unexpected("FP16 spill did not keep the whole state");
            if (auto x = slots.Restore(0); !x) return x;
            if (auto x = CheckSavedState(*life); !x) return x;
            testing::Test::RecordProperty("spill_bytes_restored",
                                          std::to_string(life->saved_bytes));
          }
          if (auto x = run_step(shape); !x) return x;
          if (full &&
              shape.segments.front().past + shape.segments.front().rows == checkpoint_positions) {
            if (auto x = SaveState(*life, true); !x) return x;
            if (life->saved_positions != checkpoint_positions)
              return std::unexpected("FP16 checkpoint starts at the wrong completed position");
            // The complete checkpoint state at position 32, frozen (the v0
            // import met the same digest, as the same sealed logit rows).
            if (!corpus)
              frozen.Expect(std::string(fusion ? "fused" : "unfused") +
                                (heldout ? ".heldout" : ".control") + (q8_cache ? ".q8" : "") +
                                ".checkpoint_state",
                            CheckpointDigest(*life));
            if (same_plan) {
              const auto now = CheckpointDigestRaw(*life);
              if (evaluation == 1)
                life->checkpoint_reference = now;
              else if (!life->checkpoint_reference || *life->checkpoint_reference != now)
                return std::unexpected("same-plan checkpoint differs from uninterrupted pass");
            }
          }
          return {};
        };
        const auto completed = repeated ? node.WithRequest(0, runner->slots().execution(),
                                                           "FP16 repeated chunk", chunk)
                                        : chunk();
        if (!completed) return completed;
        if (repeated && evaluation > 2 &&
            shape.segments.front().past + shape.segments.front().rows == checkpoint_positions) {
          // The completed chunk's request has ended: no weight lease
          // remains. State stays resident; only immutable weights leave.
          backing_before = node.memory().backings();
          for (std::size_t i = 0; i < paged->extents().size(); ++i) {
            auto extent = node.catalog().Describe(paged->extents()[i]);
            if (!extent || extent->state != llmp::catalog::ExtentState::kResident ||
                extent->leases != 0 || extent->registrations != 0)
              return std::unexpected("weight eviction begins with held or absent backing");
            generations[i] = {extent->backing_generation, extent->content_generation};
          }
          if (auto x = node.Evict(paged->extents()); !x) return x;
          if (node.memory().backings() + weight_extents != backing_before)
            return std::unexpected("weight eviction did not release all backing");
          for (const auto id : paged->extents()) {
            auto extent = node.catalog().Describe(id);
            if (!extent || extent->state != llmp::catalog::ExtentState::kNonresident ||
                extent->leases != 0 || extent->registrations != 0)
              return std::unexpected("weight extent remains resident or held after eviction");
          }
          ++evictions;
          reload_pending = true;
        }
      }
      if (reload_pending) return std::unexpected("evaluation ended before full weight reload");
      return {};
    };
    const auto count = repeated ? 4U : 1U;
    for (std::uint32_t evaluation = 1; evaluation <= count; ++evaluation) {
      if (evaluation > 1) {
        if (auto x = node.WithRequest(0, runner->slots().execution(), "clearing FP16 repeat",
                                      [&]() { return runner->slots().Clear(0); });
            !x)
          return x;
        const auto& slots = runner->slots();
        if (slots.positions(0) != 0 || slots.spilled(0) || slots.restoring(0) ||
            slots.verify_pending(0) || slots.borrowed(0))
          return std::unexpected("FP16 repeat did not begin with cleared state");
        life->saved_pages.clear();
        life->saved_bytes = 0;
        life->saved_positions = 0;
        life->saved_footprint = {};
      }
      const auto completed =
          repeated ? evaluate(evaluation)
                   : node.WithRequest(0, runner->slots().execution(), "FP16 bridge comparison",
                                      [&]() { return evaluate(evaluation); });
      if (!completed) return completed;
      ++evaluations;
    }
    if (same_plan) {
      if (reference_rows != token_rows) return std::unexpected("incomplete live reference");
      testing::Test::RecordProperty("same_plan_reference_rows", std::to_string(reference_rows));
      testing::Test::RecordProperty("same_plan_reference_sha256", Hash(life->oracle));
      testing::Test::RecordProperty("same_plan_spec_sha256", llmp::base::ToHex(spec->Digest()));
      std::string records;
      records.reserve(kPlanTextLimit);
      const auto append = [&](std::string_view value) {
        if (value.size() > kPlanTextLimit - records.size()) return false;
        records.append(value);
        return true;
      };
      for (std::size_t i = 0; i < life->recorded_shapes.size(); ++i) {
        const auto& shape = life->recorded_shapes[i];
        if (!shape) continue;
        const auto& plan = life->corpus_plans[i];
        if (!plan.runs || plan.implementations.empty())
          return std::unexpected("same-plan program never entered execution");
        if (q8_cache &&
            std::ranges::count(plan.implementations, "ggml.flash_attn_ext.vec_q8_0") != 24)
          return std::unexpected("Q8 replay did not select all quantized attention operations");
        const auto header = "rows=" + std::to_string(plan.rows) +
                            " read_cells=" + std::to_string(shape->segments.front().read_cells) +
                            " phase=" + std::to_string(static_cast<unsigned>(shape->phase)) +
                            " bound_plan=" + llmp::base::ToHex(plan.identity) +
                            " run_or_record=" + std::to_string(plan.runs) + "\n";
        if (!append(header)) return std::unexpected("same-plan report exceeds paid text bound");
        for (const auto& name : plan.implementations)
          if (!append(name) || !append("\n"))
            return std::unexpected("same-plan report exceeds paid text bound");
      }
      testing::Test::RecordProperty("same_plan_selected_programs", records);
    }
    if (repeated) {
      if (evaluations != 4 || evictions != 2 || restored_extents != 2 * weight_extents ||
          compared_rows != (same_plan ? 3U : 4U) * (token_rows + 8U))
        return std::unexpected("FP16 repeated control witnesses are incomplete");
      testing::Test::RecordProperty("evaluations", std::to_string(evaluations));
      testing::Test::RecordProperty("full_weight_evictions", std::to_string(evictions));
      testing::Test::RecordProperty("restored_weight_extents", std::to_string(restored_extents));
      testing::Test::RecordProperty("compared_rows", std::to_string(compared_rows));
    }
    return {};
  }();
  EXPECT_TRUE(ran.has_value()) << (ran ? "" : ran.error());
  EXPECT_EQ(node.host_overcharges(), 0U);
  // Retire also runs on every earlier ASSERT return. Failure retains the
  // whole Lifetime, not only the runner and its kernel resources.
  life.reset();
}
TEST_P(ComponentRunnerQwen2Gpu, FrozenFp16ControlIsBitExact) {
  const auto [fusion, full, heldout, repeated] = GetParam();
  RunQwen2Control(fusion, full, heldout, repeated);
}
TEST(ComponentRunnerQwen2StateGpu, SamePlanCheckpointSpillAndEviction) {
  RunQwen2Control(true, true, false, true, false, true);
}
TEST(ComponentRunnerQwen2StateGpu, SamePlanHeldoutCheckpointSpillAndEviction) {
  RunQwen2Control(true, true, true, true, false, true);
}
TEST(ComponentRunnerQwen2Q8StateGpu, SamePlanCheckpointSpillAndEviction) {
  RunQwen2Control(true, true, false, true, false, true, true);
}
TEST(ComponentRunnerQwen2Q8CorpusGpu, FixedTeacherCorpusScoresEveryTarget) {
  RunQwen2Control(true, false, false, false, true, false, true);
}
class ComponentRunnerQwen2CorpusGpu : public testing::TestWithParam<bool> {};
TEST_P(ComponentRunnerQwen2CorpusGpu, FixedTeacherCorpusScoresEveryTarget) {
  RunQwen2Control(GetParam(), false, false, false, true);
}
INSTANTIATE_TEST_SUITE_P(Corpus, ComponentRunnerQwen2CorpusGpu, testing::Bool(),
                         [](const auto& info) { return info.param ? "Fused" : "Unfused"; });
INSTANTIATE_TEST_SUITE_P(
    Bridge, ComponentRunnerQwen2Gpu,
    testing::Values(std::tuple{false, false, false, false}, std::tuple{false, true, false, false},
                    std::tuple{true, false, false, false}, std::tuple{true, true, false, false},
                    std::tuple{false, true, true, false}, std::tuple{true, true, true, false},
                    std::tuple{false, true, false, true}, std::tuple{true, true, false, true},
                    std::tuple{false, true, true, true}, std::tuple{true, true, true, true}),
    ([](const auto& info) {
      const auto [fusion, full, heldout, repeated] = info.param;
      return std::string(fusion ? "Fused" : "Unfused") + (repeated ? "Repeated" : "") +
             (heldout ? "Heldout" : (full ? "Full" : "Screen"));
    }));
}  // namespace
