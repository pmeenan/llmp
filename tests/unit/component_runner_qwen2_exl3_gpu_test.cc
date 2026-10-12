// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Actual approved EXL3 artifact, one sealed held-out 145-row reconstruction
// prefill, with a separate control for its 16 decode suffix steps. The oracle is produced
// by the legacy native implementation and sealed before this candidate runs.
// Separate same-implementation controls create their own uninterrupted reference
// and exercise checkpoint/spill continuation for both approved G fixtures.
// This diagnostic is not the historical1570-linear or750-statistic TierC gate.
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
#include <tuple>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "components/step.h"
#include "corpus_score.h"
#include "engine/component_source.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/plan_cache.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "kernels/exl3/selected_source.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/step_program.h"
#include "model/qwen2.h"
#include "model/qwen2_exl3.h"
#include "planner/pattern.h"
#include "platform/host_probe.h"
#include "runtime/memory_guard.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen2.h"

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
// MiB compiler, 1,024 MiB metadata/retained native programs, 768 MiB for
// the retained initial IR and observations; their exact sum fits 4 GiB. The external funding note
// names each phase/owner, including BoundGraph::Bind's transient copies. These are source bounds,
// not measured peaks. Guard metadata before parsing and reproduce Lowering::Run's descriptor
// reservation before TensorArena.
constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
constexpr std::uint64_t kHostAllowance = 4096 * kMiB;
constexpr std::uint64_t kProgramHost = 64 * kMiB;
constexpr std::uint64_t kLaunches = 4096;
constexpr std::uint64_t kRuntimePlans = 1 * (kProgramHost + kLaunches * en::kGraphNodeHostBytes);
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
static_assert(sizeof(llmp::execution::TunedShape) <= 48);
static_assert(llmp::ir::kMaxSources == 10 && llmp::ir::kMaxOutputs == 4);
static_assert(sizeof(llmp::execution::OperandMemory) <= 48 &&
              sizeof(llmp::execution::OperandFormat) <= 112 &&
              sizeof(llmp::execution::CapabilityQuery) <= 256);
// The inherited inventory's largest overlapping phase (seven graphs;
// this target uses fewer) and the fixed screen's additional live owners.
constexpr std::uint64_t kInheritedCompiler = 2526477744ULL;
struct BoundStepAbi {
  std::variant<kg::Kernel, kg::RmsNormMulKernel, std::unique_ptr<kg::BoundSelectedCall>> kernel;
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
    2 * 192 * kMiB + kBindCopies + 4096 * 16384ULL +
    4 * kMiB;  // selected calls, profile facts and transient operands
constexpr std::uint64_t kObservationOwners =
    209977600 +                                        // one extra hard-cap IR graph envelope
    std::uint64_t{145} * 151936 * sizeof(float) +      // complete oracle
    3 * std::uint64_t{145} * 151936 * sizeof(float) +  // outputs, joined row copy, growth overlap
    16 * kMiB;                                         // histories, script, headers, rounding
static_assert(kInheritedCompiler <= 2560 * kMiB);
static_assert(kMetadataOwners <= 1024 * kMiB);
static_assert(kObservationOwners <= 768 * kMiB);
static_assert(kInheritedCompiler + kMetadataOwners + kObservationOwners <= kHostAllowance);
constexpr std::uint64_t kDomainCap = std::uint64_t{8} << 30U;
constexpr std::size_t kTokenLimit = 1040 * 8;
constexpr std::size_t kOracleBytes = std::size_t{145} * 151936 * sizeof(float);
constexpr std::size_t kFullOracleBytes = std::size_t{161} * 151936 * sizeof(float);
constexpr std::uint64_t kFullRuntimePlans = 2 * kRuntimePlans;
constexpr std::uint64_t kStateCopyLimit = 16 * kMiB;
constexpr std::uint64_t kStateCopies = 2 * kStateCopyLimit;
constexpr std::uint64_t kStateMetadata = kMiB;
static_assert(sizeof(en::SlotLifecycle::Range) <= 32);
// Three persistent range vectors plus temporary returned/copy lists and growth;
// two1024-entry generation capacities and fixed owner/report bookkeeping.
static_assert(12 * 512 * 32 + 2 * 1024 * 16 + 64 * 1024 <= kStateMetadata);
static_assert(kStateCopyLimit <= kOracleBytes);
constexpr std::uint64_t kFullMetadataOwners = kMetadataOwners + kProgramHost;
constexpr std::uint64_t kFullObservationOwners =
    kObservationOwners + kFullOracleBytes - kOracleBytes;
static_assert(kFullMetadataOwners <= 1024 * kMiB);
static_assert(kFullObservationOwners <= 768 * kMiB);
static_assert(kInheritedCompiler + kFullMetadataOwners + kFullObservationOwners <= kHostAllowance);

std::expected<std::vector<std::int32_t>, std::string> HeldOutPrefix(const std::string& bytes,
                                                                    std::size_t count = 145) {
  // Identical full-file authority to benchmarks/exl3_common.cc::LoadIds.
  if (bytes.size() != kTokenLimit ||
      Hash(bytes) != "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c")
    return std::unexpected("not the sealed legacy held-out IDs");
  std::vector<std::int32_t> tokens;
  if (count != 145 && count != 161) return std::unexpected("unpaid EXL3 trajectory length");
  tokens.reserve(count);
  for (std::size_t i = 0; i < 1040; ++i) {
    std::uint64_t id = 0;
    for (std::size_t b = 0; b < 8; ++b)
      id |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i * 8 + b])) << (8 * b);
    if (id >= 151936) return std::unexpected("held-out ID is outside the vocabulary");
    if (i < count) tokens.push_back(static_cast<std::int32_t>(id));
  }
  return tokens;
}

en::Status NativeReservation(const kn::LowerInput& input) {
  const auto& p = input.plan.graph;
  const auto& g = input.step.graph;
  if (input.shape.stage != 0 || p.nodes().size() > 4096 || p.tensor_count() > 16384 ||
      input.step.inputs.size() > 256 || input.step.weights.size() > 4096 ||
      input.step.state.size() > 4096 || !input.step.saves.empty())
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

// Same fixed plan-name/report inventory as the FP16 scorer. CorpusScore owns
// and charges its additional bounded IDs/history/output/row buffers separately.
static_assert(kInheritedCompiler + kFullMetadataOwners + kFullObservationOwners + 8 * kMiB +
                  kStateMetadata <=
              kHostAllowance);

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
  std::map<std::pair<llmp::ir::WeightRef, llmp::format::PartRole>, std::uint32_t> resource;
  std::unique_ptr<KernelSource> kernel;
  std::unique_ptr<en::ModelRunner> runner;
  std::vector<en::PagedModel*> entered;
  std::array<ts::CorpusPlanRecord, 2> corpus_plans{};
  std::optional<en::WaveComposer> corpus_composer;
  std::unique_ptr<ts::CorpusScore> corpus_scorer;
  std::array<ts::CorpusTargetScore, 256> corpus_scores{};
  std::size_t corpus_score_count = 0;
  void* saved_state = nullptr;
  void* state_readback = nullptr;
  std::vector<llmp::catalog::ExtentId> state_copy_extents;
  std::vector<en::SlotLifecycle::Range> saved_pages, saved_footprint, final_pages;
  std::uint32_t saved_positions = 0;
  std::uint64_t saved_bytes = 0, final_bytes = 0;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> weight_generations;
};

// A baseline prefix may be reused only after all its output rows were compared.
// The suffix reference remains live through the end of both restore controls.
bool CanStoreFinalState(std::uint32_t compared_prefix, std::uint64_t bytes) {
  return compared_prefix == 145 && bytes > 0 && bytes <= kStateCopyLimit && bytes <= kOracleBytes;
}

en::Status SaveState(Lifetime& life, bool checkpoint) {
  auto& slots = life.runner->slots();
  if (slots.stages() != 1 || slots.layout(0).bytes > kStateCopyLimit)
    return std::unexpected("EXL3 state exceeds paid single-stage copy bound");
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
    return std::unexpected("EXL3 state range inventory exceeds its bound");
  life.saved_bytes = 0;
  for (const auto& range : life.saved_pages) {
    if (range.region != 0 || range.bytes == 0 || range.bytes > kStateCopyLimit - life.saved_bytes)
      return std::unexpected("EXL3 state copy bytes exceed their bound");
    life.saved_bytes += range.bytes;
  }
  if (!life.saved_state) {
    const auto occupied = life.node.catalog().OccupancyOf(life.node.domain()).Total().value();
    if (occupied > kDomainCap || kStateCopies > kDomainCap - occupied)
      return std::unexpected("EXL3 state buffers exceed checked domain");
    auto saved = life.node.Pinned(kStateCopyLimit, 0, life.state_copy_extents);
    if (!saved) return std::unexpected(saved.error());
    life.saved_state = *saved;
    auto readback = life.node.Pinned(kStateCopyLimit, 0, life.state_copy_extents);
    if (!readback) return std::unexpected(readback.error());
    life.state_readback = *readback;
  }
  return slots.CopyState(0, life.saved_state, life.saved_pages, true);
}

bool SameRanges(std::span<const en::SlotLifecycle::Range> a,
                std::span<const en::SlotLifecycle::Range> b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].region != b[i].region || a[i].offset != b[i].offset || a[i].bytes != b[i].bytes)
      return false;
  return true;
}

en::Status CheckSavedState(Lifetime& life) {
  auto& slots = life.runner->slots();
  if (slots.positions(0) != life.saved_positions || slots.spilled(0) || slots.restoring(0) ||
      !SameRanges(slots.live(0, 0).used_ranges(), life.saved_footprint))
    return std::unexpected("EXL3 restored state counters or footprint differ");
  if (auto copied = slots.CopyState(0, life.state_readback, life.saved_pages, true); !copied)
    return copied;
  if (std::memcmp(life.saved_state, life.state_readback,
                  static_cast<std::size_t>(life.saved_bytes)) != 0)
    return std::unexpected("EXL3 restored state bytes differ");
  return {};
}

TEST(ComponentRunnerExl3Inputs, FinalStateStorageRequiresConsumedPrefixAndBoundedBytes) {
  EXPECT_FALSE(CanStoreFinalState(0, 1));
  EXPECT_FALSE(CanStoreFinalState(144, 1));
  EXPECT_FALSE(CanStoreFinalState(146, 1));
  EXPECT_FALSE(CanStoreFinalState(145, 0));
  EXPECT_FALSE(CanStoreFinalState(145, kStateCopyLimit + 1));
  EXPECT_TRUE(CanStoreFinalState(145, kStateCopyLimit));
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

bool FitsDomain(std::uint64_t available) {
  return available >= kDomainCap && available - kDomainCap >= llmp::runtime::kUncountedMargin;
}

TEST(ComponentRunnerExl3Inputs, ChecksTheWholeDomainBeforeStartup) {
  EXPECT_FALSE(FitsDomain(0));
  EXPECT_FALSE(FitsDomain(kDomainCap - 1));
  EXPECT_FALSE(FitsDomain(kDomainCap + llmp::runtime::kUncountedMargin - 1));
  EXPECT_TRUE(FitsDomain(kDomainCap + llmp::runtime::kUncountedMargin));
  EXPECT_TRUE(FitsDomain(std::numeric_limits<std::uint64_t>::max()));
}

TEST(ComponentRunnerExl3Inputs, RefusesOversizeAndWrongLengthFilesBeforeReading) {
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
  EXPECT_FALSE(HeldOutPrefix("abcde"));
  EXPECT_FALSE(HeldOutPrefix(std::string(kTokenLimit, '\0')));
}

// Every output row is requested so narrowing
// cannot change the legacy chunk's matrix shapes.
void RunControl(bool full, bool corpus = false, bool mixed_rate = false, bool lifecycle = false,
                bool optimized = false) {
  ASSERT_FALSE(optimized && (!corpus || lifecycle));
  ASSERT_FALSE(mixed_rate && !corpus && !lifecycle);
  ASSERT_FALSE(lifecycle && (!full || corpus));
  std::uint32_t token_rows = full ? 161 : 145;
  const auto oracle_bytes = full ? kFullOracleBytes : kOracleBytes;
  const auto runtime_plans = (full || corpus) ? kFullRuntimePlans : kRuntimePlans;
  const auto ids_path = Env(corpus ? "LLMP_QWEN2_CORPUS_IDS" : "LLMP_EXL3_SCREEN_IDS");
  const auto oracle_path = Env(full ? "LLMP_EXL3_TRAJECTORY_ORACLE" : "LLMP_EXL3_SCREEN_ORACLE");
  const auto oracle_hash =
      Env(full ? "LLMP_EXL3_TRAJECTORY_ORACLE_SHA256" : "LLMP_EXL3_SCREEN_ORACLE_SHA256");
  if (ids_path.empty() || (!corpus && !lifecycle && (oracle_path.empty() || oracle_hash.empty())))
    GTEST_SKIP() << "sealed legacy EXL3 IDs and oracle required";
  if (!corpus && !lifecycle) {
    ASSERT_EQ(oracle_hash.size(), 64U);
    ASSERT_TRUE(std::ranges::all_of(
        oracle_hash, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }));
  }
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available.has_value()) << "unknown physical availability";
  ASSERT_TRUE(FitsDomain(*available)) << "the complete domain and driver margin do not fit";
  Owner life(new Lifetime);
  auto& node = life->node;
  node.SetHostFloor(0);
  const auto opened_node = node.Open();
  ASSERT_TRUE(opened_node.has_value()) << opened_node.error();
  const auto occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_LE(occupied, kDomainCap - kHostAllowance);
  const auto payment =
      node.catalog().AddExtent({.domain = node.domain(),
                                .memory_class = llmp::catalog::MemoryClass::kRuntime,
                                .recovery = llmp::catalog::Recovery::kPinned,
                                .size = llmp::base::Bytes(kHostAllowance),
                                .content = {}},
                               true);
  ASSERT_TRUE(payment.has_value());
  ASSERT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), occupied + kHostAllowance);
  auto ids_read = Read(ids_path, corpus ? 1024 : kTokenLimit, !corpus);
  ASSERT_TRUE(ids_read.has_value()) << ids_read.error();
  life->tokens_bytes = std::move(*ids_read);
  if (corpus) {
    const auto seal = Env("LLMP_QWEN2_CORPUS_IDS_SHA256");
    ASSERT_EQ(seal.size(), 64U);
    ASSERT_EQ(Hash(life->tokens_bytes), seal);
    ASSERT_EQ(life->tokens_bytes.size() % 4, 0U);
    token_rows = static_cast<std::uint32_t>(life->tokens_bytes.size() / 4);
    ASSERT_GE(token_rows, 2U);
    ASSERT_LE(token_rows, 256U);
    for (std::size_t i = 0; i < token_rows; ++i) {
      std::uint32_t id = 0;
      for (int byte = 3; byte >= 0; --byte)
        id = (id << 8U) |
             static_cast<unsigned char>(life->tokens_bytes[i * 4 + static_cast<std::size_t>(byte)]);
      ASSERT_LT(id, 151936U);
      life->tokens.push_back(static_cast<std::int32_t>(id));
    }
  } else {
    // Preserve full-file authority and the historical quantized legacy control.
    auto prefix = HeldOutPrefix(life->tokens_bytes, token_rows);
    ASSERT_TRUE(prefix.has_value()) << prefix.error();
    life->tokens = std::move(*prefix);
    if (lifecycle) {
      // The same allocation holds the uninterrupted candidate reference.
      life->oracle.resize(kFullOracleBytes);
    } else {
      auto oracle_read = Read(oracle_path, oracle_bytes, true);
      ASSERT_TRUE(oracle_read.has_value()) << oracle_read.error();
      life->oracle = std::move(*oracle_read);
      ASSERT_EQ(Hash(life->oracle), oracle_hash);
    }
  }
  auto& tokens = life->tokens;
  const auto& oracle = life->oracle;
  const auto& profile = llmp::model::Qwen25Instruct05BExl3();
  const std::size_t row_bytes = std::size_t{profile.vocab} * sizeof(float);
  std::uint64_t corpus_charge = 0;
  if (corpus) {
    const std::array documents{ts::CorpusDocument{.slot = 0, .tokens = tokens}};
    auto bound = ts::CorpusScore::Bound(documents, {profile.vocab, 7});
    ASSERT_TRUE(bound.has_value()) << bound.error();
    corpus_charge = *bound;
    ASSERT_LE(corpus_charge, 21108224U);
  }
  if (!corpus) ASSERT_EQ(oracle.size(), token_rows * row_bytes);
  std::string store = Env("LLMP_EXL3_FIXTURE_STORE");
  if (store.empty()) store = Env("HOME") + "/.local/share/llmp/artifact-layout-20260922/installed";
  const auto artifact_id = mixed_rate
                               ? "00d77caf056c45b6fa05ebdbeff8b38da79d0556a2fd4f1b661e7e21ef9ed83a"
                               : "6e96e499ea9b859ea726327d4f82fb1739cb116ce4e9d3541d1ba67506efb4b4";
  const fs::path root = fs::path(store) / artifact_id;
  if (!fs::is_directory(root)) GTEST_SKIP() << "fixture not installed: " << root;
  // These pure filesystem bounds precede both artifact parser calls. The
  // checked manifest identity fixes the index's declared length/hash too.
  std::error_code metadata_error;
  const auto manifest_bytes = fs::file_size(root / "manifest.json", metadata_error);
  ASSERT_FALSE(metadata_error);
  ASSERT_LE(manifest_bytes, 16 * 1024U);
  const auto index_bytes = fs::file_size(root / "index.json", metadata_error);
  ASSERT_FALSE(metadata_error);
  ASSERT_LE(index_bytes, kMiB);
  const auto headers = HeaderBounds(root);
  ASSERT_TRUE(headers.has_value()) << headers.error();
  auto opened = ar::Artifact::Open(root, {.expected_id = artifact_id});
  ASSERT_TRUE(opened.has_value()) << opened.error().ToString();
  life->artifact.emplace(std::move(*opened));
  auto& artifact = life->artifact;
  ASSERT_EQ(profile.layers, 24U);
  ASSERT_EQ(profile.width, 896U);
  ASSERT_EQ(profile.vocab, 151936U);
  ASSERT_EQ(profile.ffn, 4864U);
  ASSERT_EQ(profile.heads, 14U);
  ASSERT_EQ(profile.kv_heads, 2U);
  ASSERT_EQ(profile.head_dim, 64U);
  ASSERT_LE(artifact->resources().size(), 1024U);
  ASSERT_LE(artifact->groups().size(), 64U);
  ASSERT_LE(artifact->chunk_sha256().size(), 1024U);
  ASSERT_TRUE(llmp::model::BindQwen2Exl3(profile, *artifact));
  const auto& mapping = sp::mappings::Qwen2Mapping();
  auto mapped_values = sp::mappings::Qwen2V0Values(profile);
  ASSERT_TRUE(mapped_values.has_value()) << mapped_values.error();
  life->values = std::move(*mapped_values);
  auto bound_source = sp::SourceOfV0(*artifact, &mapping);
  ASSERT_TRUE(bound_source.has_value()) << bound_source.error();
  life->source.emplace(std::move(*bound_source));
  auto& source = life->source;
  ASSERT_LE(source->tensors.size(), 1024U);
  if (corpus || lifecycle) {
    // Count each logical product once. Every physical part retains the source
    // encoding; the normal binding/lowering verifies its exact extent and role.
    std::array<std::uint32_t, 9> rates{};
    std::array<std::uint32_t, 4> parts{};
    for (const auto& tensor : source->tensors) {
      if (tensor.format.encoding != llmp::format::Encoding::kExl3Trellis) continue;
      const auto bits = tensor.format.params.k_bits;
      ASSERT_LT(bits, rates.size());
      switch (tensor.part) {
        case llmp::format::PartRole::kCodes:
          ++rates[bits];
          ++parts[0];
          break;
        case llmp::format::PartRole::kInputSigns:
          ++parts[1];
          break;
        case llmp::format::PartRole::kOutputSigns:
          ++parts[2];
          break;
        case llmp::format::PartRole::kCodebookMarker:
          ++parts[3];
          break;
        default:
          FAIL() << "unexpected EXL3 physical weight part";
      }
    }
    const std::array<std::uint32_t, 9> expected =
        mixed_rate ? std::array<std::uint32_t, 9>{0, 0, 0, 0, 42, 118, 8, 0, 1}
                   : std::array<std::uint32_t, 9>{0, 0, 0, 0, 168, 0, 0, 0, 1};
    ASSERT_EQ(rates, expected);
    ASSERT_EQ(parts, (std::array<std::uint32_t, 4>{169, 169, 169, 169}));
    testing::Test::RecordProperty("corpus_artifact_id", artifact_id);
    testing::Test::RecordProperty("corpus_exl3_rates",
                                  mixed_rate ? "4:42,5:118,6:8,8:1" : "4:168,8:1");
    testing::Test::RecordProperty("corpus_exl3_parts",
                                  "codes:169,input_signs:169,output_signs:169,marker:169");
  }
  for (const auto& tensor : source->tensors) ASSERT_LE(tensor.name.size(), 128U);
  auto synthesized = sp::Synthesize(mapping, *source, life->values);
  ASSERT_TRUE(synthesized.has_value()) << synthesized.error();
  life->spec.emplace(std::move(*synthesized));
  auto& spec = life->spec;
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
  ASSERT_LE(weights_bound, 1024U);
  auto derived = co::StageLayout(*spec, 0, 512, 145);
  ASSERT_TRUE(derived.has_value()) << derived.error();
  life->layout.emplace(std::move(*derived));
  auto& layout = life->layout;
  auto bound_identity = spec->Digest();
  if (lifecycle) {
    const std::array<std::string_view, 1> artifacts{artifact->id()};
    auto bound = sp::BoundDigest(*spec, artifacts);
    ASSERT_TRUE(bound.has_value()) << bound.error();
    bound_identity = *bound;
    ASSERT_LE(layout->bytes, kStateCopyLimit);
    testing::Test::RecordProperty("lifecycle_bound_model", llmp::base::ToHex(bound_identity));
    life->corpus_plans[0].rows = 145;
    life->corpus_plans[1].rows = 1;
  }
  auto& shapes = life->shapes;
  if (corpus) {
    for (std::uint32_t past = 0; past < token_rows; past += 7) {
      auto shape = Shape(past, std::min(7U, token_rows - past));
      shape.phase = pl::Phase::kScore;
      life->corpus_plans[shape.rows() == 7 ? 0U : 1U].rows = shape.rows();
      shapes.push_back(std::move(shape));
    }
  } else {
    shapes.push_back(Shape(0, 145));
    if (full)
      for (std::uint32_t past = 145; past < 161; ++past) shapes.push_back(Shape(past, 1));
  }
  auto& resource = life->resource;
  auto built_step =
      co::BuildStep(*spec, 0, shapes.front(), {.tensors = source->tensors, .state = &*layout});
  ASSERT_TRUE(built_step.has_value()) << built_step.error();
  life->step.emplace(std::move(*built_step));
  for (const co::StepWeight& weight : life->step->weights) {
    ASSERT_TRUE(weight.parts.size() == 1 || weight.parts.size() == 4);
    for (const auto& [name, role] : weight.parts) {
      const auto found = artifact->FindResource(name);
      ASSERT_TRUE(found.has_value()) << name;
      ASSERT_TRUE(resource.emplace(std::pair{weight.ref, role}, *found).second);
    }
  }
  const auto parsed_policy = pl::ParsePolicy(mapping.policies.front().text);
  ASSERT_TRUE(parsed_policy.has_value()) << parsed_policy.error();
  const pl::Policy policy = *parsed_policy;
  std::string scratch = Env("LLMP_TEST_SCRATCH");
  if (scratch.empty()) scratch = testing::TempDir();
  const fs::path spill = fs::path(scratch) / "qwen2-exl3-screen";
  fs::create_directories(spill);
  life->kernel = std::make_unique<KernelSource>(node);
  auto& kernel = life->kernel;
  auto& runner = life->runner;
  auto& entered = life->entered;
  const auto ran = [&]() -> en::Status {
    entered.push_back(kernel.get());
    if (auto x = kernel->resources->OpenCublas("EXL3 shared cuBLAS workspace"); !x) return x;
    std::optional<llmp::kernels::exl3::GemvCapacity> gemv;
    if (optimized) {
      auto queried = llmp::kernels::exl3::QueryGemvCapacity(0);
      if (!queried) return std::unexpected(queried.error());
      gemv = *queried;
      std::ostringstream receipt;
      for (const auto& output : gemv->blocks)
        for (const auto& rows : output)
          for (const auto blocks : rows) receipt << blocks << '\n';
      testing::Test::RecordProperty("corpus_gemv_capacity", receipt.str());
    }
    const auto cache = optimized    ? (mixed_rate ? llmp::kernels::exl3::FrozenCache::k45O
                                                  : llmp::kernels::exl3::FrozenCache::k40O)
                       : mixed_rate ? llmp::kernels::exl3::FrozenCache::k45G
                                    : llmp::kernels::exl3::FrozenCache::k40G;
    auto selected = llmp::kernels::exl3::SelectedSource(cache, gemv);
    if (!selected) return std::unexpected(selected.error());
    auto backend = kg::GgmlBackend::Create(
        0, node.execution(), node.stream(0), &kernel->resources->cublas(),
        {.lower = {.index_scratch_limit = 32 * kMiB}, .selected = std::move(*selected)});
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
    if (corpus || lifecycle) {
      auto guarded = std::move(checked_backend.lower);
      if (corpus) guarded = ts::GuardCorpusLower(std::move(guarded), profile.vocab);
      checked_backend.lower = [held = life.get(), lifecycle, vocabulary = profile.vocab,
                               guarded = std::move(guarded)](const kn::LowerInput& input)
          -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
        auto program = guarded(input);
        if (!program) return program;
        if (lifecycle) {
          const auto head = input.step.logits;
          if (!head.valid() || head.v >= input.step.graph.tensor_count() ||
              input.step.greedy.valid() || input.step.features.valid() ||
              input.shape.segments.size() != 1 || input.shape.outputs() != input.shape.rows())
            return std::unexpected("lifecycle requires one complete target head");
          const auto& desc = input.step.graph.tensor(head);
          const auto outputs = (*program)->outputs();
          if (desc.format.encoding != llmp::format::Encoding::kF32 ||
              desc.shape() != llmp::ir::ShapeOf({vocabulary, input.shape.rows()}) ||
              !llmp::ir::IsContiguous(desc) || outputs.size() != 1 ||
              outputs.front().tensor != head || outputs.front().segment != 0 ||
              outputs.front().features ||
              outputs.front().bytes !=
                  std::uint64_t{vocabulary} * input.shape.rows() * sizeof(float))
            return std::unexpected("lifecycle lowered head identity or geometry differs");
        }
        if (input.addresses.measuring()) return program;
        const auto rows = input.shape.rows();
        auto* record = &held->corpus_plans[rows == held->corpus_plans[0].rows ? 0U : 1U];
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
         .bound_digest = bound_identity,
         .mapping = &mapping,
         .values = life->values,
         .slots = 1,
         .policy = policy,
         .backend = std::move(checked_backend),
         .require_capture = false});
    if (!component) return std::unexpected(component.error());
    std::vector<pl::StepShape> keys;
    if (corpus) {
      std::vector<std::vector<std::uint64_t>> parameters;
      for (const auto& shape : shapes) {
        const auto values = (*component)->ShapeParameters(shape);
        if (values.size() > 1024) return std::unexpected("corpus shape parameter bound");
        const auto found = std::find(keys.begin(), keys.end(), shape);
        if (found == keys.end()) {
          if (keys.size() == 2) return std::unexpected("corpus exceeds two paid score keys");
          keys.push_back(shape);
          parameters.push_back(values);
        } else if (parameters[static_cast<std::size_t>(found - keys.begin())] != values) {
          return std::unexpected("corpus position changes paid score key");
        }
      }
      if (keys.empty()) return std::unexpected("corpus has no score shape");
    } else {
      // Both canonical shape keys and every position's baked parameters
      // are authenticated before Setup, which can allocate native programs.
      if (shapes.size() != (full ? 17U : 1U) || shapes.front() != Shape(0, 145))
        return std::unexpected("EXL3 trajectory exceeds its paid shape inventory");
      const auto prefill_parameters = (*component)->ShapeParameters(shapes.front());
      if (prefill_parameters.size() > 1024)
        return std::unexpected("shape parameters exceed paid metadata");
      if (full) {
        const auto decode = Shape(145, 1);
        const auto decode_parameters = (*component)->ShapeParameters(decode);
        if (decode == shapes.front() || decode_parameters.size() > 1024)
          return std::unexpected("EXL3 decode key exceeds paid metadata");
        for (std::size_t i = 1; i < shapes.size(); ++i) {
          if (shapes[i].segments.front().past != 144 + i || shapes[i] != decode ||
              (*component)->ShapeParameters(shapes[i]) != decode_parameters)
            return std::unexpected("EXL3 suffix changes the paid decode key");
        }
      }
      keys = {shapes.front()};
      if (full) keys.push_back(shapes[1]);
    }
    en::RunnerConfig config;
    config.graphs = false;
    config.slots = {.model = "qwen2-exl3-shared-screen",
                    .slots = 1,
                    .stages = {{.name = "target", .layout = *layout}},
                    .spec = bound_identity,
                    .read_align = 256,
                    .spill_place = {},
                    .spill_directory = spill,
                    .invalidate = {},
                    .drop_plans = {}};
    std::vector<en::GroupPlace> places(paged->artifact().groups().size(), en::GroupPlace::kDevice);
    config.weights.push_back(
        {.weights = std::move(weights), .places = std::move(places), .slabs = {}});
    config.setup_shapes = keys;
    config.launch =
        [held = life.get()]() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return held->kernel->backend->Launch(
          {.base = held->node.pool().base, .size = llmp::base::Bytes(held->node.pool().bytes)});
    };
    config.weight_address = [held = life.get(), paged](
                                const llmp::ir::WeightRef& ref,
                                llmp::format::PartRole role) -> std::uint64_t {
      const auto found = held->resource.find({ref, role});
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
    if (before_workspace > kDomainCap || 2 * en::kPagedExtent > kDomainCap - before_workspace ||
        runner->activations_needed() > kDomainCap - before_workspace - 2 * en::kPagedExtent ||
        pool > kDomainCap - before_workspace - 2 * en::kPagedExtent - runner->activations_needed())
      return std::unexpected("workspace exceeds the checked domain");
    if (auto x =
            node.MapWorkspace(runner->activations_needed(),
                              std::max<std::uint64_t>(runner->pool_needed(), en::kPagedExtent));
        !x)
      return x;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    if (fixed > kDomainCap) return std::unexpected("startup footprint exceeds the checked domain");
    if (paged->extents().size() > (kDomainCap - fixed) / en::kPagedExtent)
      return std::unexpected("weight extents exceed the checked domain");
    const auto weights_bytes = paged->extents().size() * en::kPagedExtent;
    if (weights_bytes > kDomainCap - fixed || runtime_plans > kDomainCap - fixed - weights_bytes ||
        corpus_charge > kDomainCap - fixed - weights_bytes - runtime_plans ||
        (lifecycle ? kStateCopies : 0) >
            kDomainCap - fixed - weights_bytes - runtime_plans - corpus_charge ||
        node.StateCapacity() > (kDomainCap - fixed - weights_bytes - runtime_plans - corpus_charge -
                                (lifecycle ? kStateCopies : 0)) /
                                   4)
      return std::unexpected("weights and state exceed the checked domain");
    if (auto x = node.Start(llmp::base::Bytes(kDomainCap)); !x) return x;
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
          node, *runner, *life->corpus_composer, documents, {profile.vocab, 7},
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
      testing::Test::RecordProperty(
          "corpus_policy", optimized    ? (mixed_rate ? "existing-exl3-45o" : "existing-exl3-40o")
                           : mixed_rate ? "existing-exl3-45g"
                                        : "existing-exl3-40g");
      report.str("");
      report.clear();
      for (const auto& plan : life->corpus_plans) {
        if (!plan.rows) continue;
        if (!plan.runs || plan.implementations.empty())
          return std::unexpected("corpus selected program never entered execution");
        const std::string optimized_prefix = mixed_rate ? "exl3.pinned.45o." : "exl3.pinned.40o.";
        if (optimized &&
            (std::ranges::count(plan.implementations, optimized_prefix + "linear.gemv") !=
                 (mixed_rate ? 14 : 120) ||
             std::ranges::count(plan.implementations, optimized_prefix + "linear.gemm") !=
                 (mixed_rate ? 107 : 1) ||
             std::ranges::count(plan.implementations, optimized_prefix + "multi_linear.mgemm") !=
                 24 ||
             std::ranges::count(plan.implementations, optimized_prefix + "bias_add") != 72 ||
             std::ranges::count_if(plan.implementations, [](const std::string& name) {
               return name.starts_with("exl3.");
             }) != 217))
          return std::unexpected("optimized corpus selected GEMV inventory differs");
        if (mixed_rate && !optimized &&
            (std::ranges::count(plan.implementations, "exl3.pinned.45g.linear.gemm") != 121 ||
             std::ranges::count(plan.implementations, "exl3.pinned.45g.multi_linear.mgemm") != 24 ||
             std::ranges::count(plan.implementations, "exl3.pinned.45g.bias_add") != 72 ||
             std::ranges::count_if(plan.implementations, [](const std::string& name) {
               return name.starts_with("exl3.");
             }) != 217))
          return std::unexpected("mixed-rate corpus selected packed inventory differs");
        report << "rows=" << plan.rows << " bound_plan=" << llmp::base::ToHex(plan.identity)
               << " run_or_record=" << plan.runs << '\n';
        for (const auto& name : plan.implementations) report << name << '\n';
      }
      testing::Test::RecordProperty("corpus_selected_programs", report.str());
      return {};
    }
    if (lifecycle) {
      if (paged->extents().size() > 1024)
        return std::unexpected("EXL3 immutable weight witness bound");
      std::uint32_t compared_rows = 0, executed_steps = 0;
      const auto run_step = [&](const pl::StepShape& shape, bool baseline) -> en::Status {
        const auto& segment = shape.segments.front();
        const std::vector<en::SegmentWork> work{
            {.tokens = std::span(tokens).subspan(segment.past, segment.rows), .bytes = {}}};
        const auto result = runner->RunStep(
            {.shape = shape, .work = work, .step_bytes = {}, .advance = true, .ahead = {}});
        if (!result) return std::unexpected(result.error().what);
        if (result->outputs.size() != 1 ||
            result->outputs.front().size() != segment.rows * row_bytes)
          return std::unexpected("EXL3 lifecycle output geometry differs");
        const auto& got = result->outputs.front();
        for (std::size_t i = 0; i < got.size(); i += sizeof(float)) {
          float value = 0;
          std::memcpy(&value, got.data() + i, sizeof(value));
          if (!std::isfinite(value)) return std::unexpected("EXL3 lifecycle nonfinite logit");
        }
        char* expected = life->oracle.data() + segment.past * row_bytes;
        if (baseline) {
          std::memcpy(expected, got.data(), got.size());
        } else {
          if (std::memcmp(expected, got.data(), got.size()) != 0)
            return std::unexpected("EXL3 lifecycle differs from uninterrupted candidate");
          compared_rows += segment.rows;
        }
        if (runner->slots().positions(0) != segment.past + segment.rows)
          return std::unexpected("EXL3 lifecycle position differs");
        ++executed_steps;
        return {};
      };
      return node.WithRequest(
          0, runner->slots().execution(), "EXL3 same-plan state control", [&]() -> en::Status {
            for (const auto& shape : shapes)
              if (auto x = run_step(shape, true); !x) return x;
            if (auto x = SaveState(*life, false); !x) return x;
            if (life->saved_positions != 161) return std::unexpected("baseline state position");
            life->final_pages = life->saved_pages;
            life->final_bytes = life->saved_bytes;
            for (const auto id : paged->extents()) {
              const auto extent = node.catalog().Describe(id);
              if (!extent || extent->state != llmp::catalog::ExtentState::kResident)
                return std::unexpected("baseline weight not resident");
              life->weight_generations.emplace_back(extent->content_generation,
                                                    extent->backing_generation);
            }
            if (auto x = runner->slots().Clear(0); !x) return x;
            auto& slots = runner->slots();
            if (slots.positions(0) || slots.spilled(0) || slots.restoring(0) ||
                slots.verify_pending(0) || slots.borrowed(0))
              return std::unexpected("EXL3 lifecycle clear incomplete");
            for (const auto& shape : shapes) {
              const auto past = shape.segments.front().past;
              if (past == 153) {
                if (auto x = slots.CopyState(0, life->state_readback, life->saved_pages, true); !x)
                  return x;
                if (std::memcmp(life->saved_state, life->state_readback,
                                static_cast<std::size_t>(life->saved_bytes)) == 0)
                  return std::unexpected("EXL3 checkpoint saw no changed state");
                if (auto x = slots.PrepareRestore(0, life->saved_positions, life->saved_footprint,
                                                  slots.kept_layout_id());
                    !x)
                  return x;
                if (!slots.restoring(0)) return std::unexpected("EXL3 checkpoint not preparing");
                if (auto x = slots.CopyState(0, life->saved_state, life->saved_pages, false); !x)
                  return x;
                if (auto x = slots.CompleteRestore(0, life->saved_positions); !x) return x;
                if (auto x = CheckSavedState(*life); !x) return x;
                testing::Test::RecordProperty("checkpoint_bytes_restored",
                                              std::to_string(life->saved_bytes));
                for (std::uint32_t replay = 145; replay < 153; ++replay)
                  if (auto x = run_step(Shape(replay, 1), false); !x) return x;
              }
              if (past == 157) {
                if (auto x = SaveState(*life, false); !x) return x;
                if (auto x = slots.Spill(0); !x) return x;
                if (!slots.spilled(0) || !slots.kept_whole(0))
                  return std::unexpected("EXL3 spill did not retain whole state");
                if (auto x = slots.Restore(0); !x) return x;
                if (auto x = CheckSavedState(*life); !x) return x;
                testing::Test::RecordProperty("spill_bytes_restored",
                                              std::to_string(life->saved_bytes));
              }
              if (auto x = run_step(shape, false); !x) return x;
              if (past == 0) {
                if (!CanStoreFinalState(compared_rows, life->final_bytes))
                  return std::unexpected("EXL3 baseline state cannot reuse live oracle bytes");
                std::memcpy(life->oracle.data(), life->saved_state,
                            static_cast<std::size_t>(life->final_bytes));
                if (auto x = SaveState(*life, true); !x) return x;
                if (life->saved_positions != 145)
                  return std::unexpected("EXL3 checkpoint position");
              }
            }
            if (slots.positions(0) != 161 || compared_rows != 169 || executed_steps != 42 ||
                !SameRanges(slots.live(0, 0).used_ranges(), life->final_pages))
              return std::unexpected("EXL3 lifecycle final inventory differs");
            if (auto x = slots.CopyState(0, life->state_readback, life->final_pages, true); !x)
              return x;
            if (std::memcmp(life->oracle.data(), life->state_readback,
                            static_cast<std::size_t>(life->final_bytes)) != 0)
              return std::unexpected("EXL3 final state differs from uninterrupted candidate");
            for (std::size_t i = 0; i < paged->extents().size(); ++i) {
              const auto extent = node.catalog().Describe(paged->extents()[i]);
              if (!extent || extent->state != llmp::catalog::ExtentState::kResident ||
                  life->weight_generations[i] !=
                      std::pair{extent->content_generation, extent->backing_generation})
                return std::unexpected("EXL3 weight identity changed across state operations");
            }
            std::ostringstream report;
            for (const auto& record : life->corpus_plans) {
              if (record.implementations.empty() || record.runs != (record.rows == 145 ? 2U : 40U))
                return std::unexpected("EXL3 lifecycle selected execution inventory differs");
              report << "rows=" << record.rows
                     << " bound_plan=" << llmp::base::ToHex(record.identity)
                     << " run_or_record=" << record.runs << '\n';
              for (const auto& name : record.implementations) report << name << '\n';
            }
            testing::Test::RecordProperty("lifecycle_selected_programs", report.str());
            testing::Test::RecordProperty("lifecycle_compared_rows", "169");
            testing::Test::RecordProperty("lifecycle_executed_steps", "42");
            testing::Test::RecordProperty("lifecycle_final_positions", "161");
            testing::Test::RecordProperty("lifecycle_final_state_bytes",
                                          std::to_string(life->final_bytes));
            testing::Test::RecordProperty("lifecycle_immutable_weight_extents",
                                          std::to_string(paged->extents().size()));
            return {};
          });
    }
    std::uint32_t completed_steps = 0;
    std::uint32_t compared_rows = 0;
    return node.WithRequest(
        0, runner->slots().execution(), "EXL3 reconstruction comparison", [&]() -> en::Status {
          for (const auto& shape : shapes) {
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
            const char* expected = oracle.data() + segment.past * row_bytes;
            std::size_t different = 0;
            for (std::size_t i = 0; i < got.size(); i += sizeof(float))
              different += std::memcmp(got.data() + i, expected + i, sizeof(float)) != 0;
            EXPECT_EQ(different, 0U) << "past=" << segment.past << " rows=" << segment.rows;
            if (different != 0)
              return std::unexpected("EXL3 shared logits differ from sealed legacy");
            if (runner->slots().positions(0) != segment.past + segment.rows)
              return std::unexpected("EXL3 completed positions differ from the trajectory");
            ++completed_steps;
            compared_rows += segment.rows;
          }
          if (completed_steps != (full ? 17U : 1U) || compared_rows != token_rows ||
              runner->slots().positions(0) != token_rows)
            return std::unexpected("EXL3 complete trajectory witnesses differ");
          ::testing::Test::RecordProperty("completed_steps", std::to_string(completed_steps));
          ::testing::Test::RecordProperty("compared_rows", std::to_string(compared_rows));
          ::testing::Test::RecordProperty("completed_positions",
                                          std::to_string(runner->slots().positions(0)));
          return {};
        });
  }();
  EXPECT_TRUE(ran.has_value()) << (ran ? "" : ran.error());
  EXPECT_EQ(node.host_overcharges(), 0U);
  // Retire also runs on every earlier ASSERT return. Failure retains the
  // whole Lifetime, not only the runner and its kernel resources.
  life.reset();
}
TEST(ComponentRunnerExl3Gpu, Reconstruction145MatchesSealedLegacy) { RunControl(false); }
TEST(ComponentRunnerExl3Gpu, Full161TrajectoryMatchesSealedLegacy) { RunControl(true); }
TEST(ComponentRunnerExl3Gpu, FixedTeacherCorpusScoresEveryTarget) { RunControl(false, true); }
TEST(ComponentRunnerExl3Gpu, MixedRate45GFixedTeacherCorpusScoresEveryTarget) {
  RunControl(false, true, true);
}
TEST(ComponentRunnerExl3Gpu, SameImplementation40GCheckpointSpillContinuation) {
  RunControl(true, false, false, true);
}
TEST(ComponentRunnerExl3Gpu, SameImplementation45GCheckpointSpillContinuation) {
  RunControl(true, false, true, true);
}
TEST(ComponentRunnerExl3Gpu, Optimized40OFixedTeacherCorpusScoresEveryTarget) {
  RunControl(false, true, false, false, true);
}
TEST(ComponentRunnerExl3Gpu, Optimized45OFixedTeacherCorpusScoresEveryTarget) {
  RunControl(false, true, true, false, true);
}
}  // namespace
