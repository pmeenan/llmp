// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Actual approved EXL3 artifact, one sealed held-out 145-row reconstruction
// prefill, with a separate control for its 16 decode suffix steps. The oracle is produced
// by the legacy native implementation and sealed before this candidate runs.
// This diagnostic is not the historical1570-linear or750-statistic TierC gate.
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
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
};

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
void RunControl(bool full) {
  const std::size_t token_rows = full ? 161 : 145;
  const auto oracle_bytes = full ? kFullOracleBytes : kOracleBytes;
  const auto runtime_plans = full ? kFullRuntimePlans : kRuntimePlans;
  const auto ids_path = Env("LLMP_EXL3_SCREEN_IDS");
  const auto oracle_path = Env(full ? "LLMP_EXL3_TRAJECTORY_ORACLE" : "LLMP_EXL3_SCREEN_ORACLE");
  const auto oracle_hash =
      Env(full ? "LLMP_EXL3_TRAJECTORY_ORACLE_SHA256" : "LLMP_EXL3_SCREEN_ORACLE_SHA256");
  if (ids_path.empty() || oracle_path.empty() || oracle_hash.empty())
    GTEST_SKIP() << "sealed legacy EXL3 IDs and oracle required";
  ASSERT_EQ(oracle_hash.size(), 64U);
  ASSERT_TRUE(std::ranges::all_of(
      oracle_hash, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }));
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
  // Both arms authenticate the same complete1040-ID file. The legacy
  // trajectory uses its first161 IDs; the original prefill compares first145.
  auto ids_read = Read(ids_path, kTokenLimit, true);
  ASSERT_TRUE(ids_read.has_value()) << ids_read.error();
  life->tokens_bytes = std::move(*ids_read);
  auto prefix = HeldOutPrefix(life->tokens_bytes, token_rows);
  ASSERT_TRUE(prefix.has_value()) << prefix.error();
  life->tokens = std::move(*prefix);
  auto& tokens = life->tokens;
  auto oracle_read = Read(oracle_path, oracle_bytes, true);
  ASSERT_TRUE(oracle_read.has_value()) << oracle_read.error();
  life->oracle = std::move(*oracle_read);
  const auto& oracle = life->oracle;
  ASSERT_EQ(Hash(oracle), oracle_hash);
  const auto& profile = llmp::model::Qwen25Instruct05BExl3();
  const std::size_t row_bytes = std::size_t{profile.vocab} * sizeof(float);
  ASSERT_EQ(oracle.size(), token_rows * row_bytes);
  std::string store = Env("LLMP_EXL3_FIXTURE_STORE");
  if (store.empty()) store = Env("HOME") + "/.local/share/llmp/artifact-layout-20260922/installed";
  constexpr auto artifact_id = "6e96e499ea9b859ea726327d4f82fb1739cb116ce4e9d3541d1ba67506efb4b4";
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
  auto& shapes = life->shapes;
  shapes.push_back(Shape(0, 145));
  if (full)
    for (std::uint32_t past = 145; past < 161; ++past) shapes.push_back(Shape(past, 1));
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
    auto selected = llmp::kernels::exl3::SelectedSource(llmp::kernels::exl3::FrozenCache::k40G);
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
    auto component = en::ComponentStepSource::Create(
        {.spec = &*spec,
         .stage = 0,
         .bindings = {.tensors = source->tensors, .state = &*layout},
         .bound_digest = spec->Digest(),
         .mapping = &mapping,
         .values = life->values,
         .slots = 1,
         .policy = policy,
         .backend = std::move(checked_backend),
         .require_capture = false});
    if (!component) return std::unexpected(component.error());
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
    en::RunnerConfig config;
    config.graphs = false;
    config.slots = {.model = "qwen2-exl3-shared-screen",
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
    config.setup_shapes = {shapes.front()};
    if (full) config.setup_shapes.push_back(shapes[1]);
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
        node.StateCapacity() > (kDomainCap - fixed - weights_bytes - runtime_plans) / 4)
      return std::unexpected("weights and state exceed the checked domain");
    if (auto x = node.Start(llmp::base::Bytes(kDomainCap)); !x) return x;
    if (auto x = runner->Register(); !x) return x;
    if (auto x = runner->Bind(); !x) return x;
    node.Run();
    if (auto x = runner->slots().SelectSlots(std::vector<std::uint32_t>{0}); !x) return x;
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
}  // namespace
