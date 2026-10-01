// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "ds4_complete/runner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <expected>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "base/json.h"
#include "base/sha256.h"
#include "engine/support.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"

namespace jitllm::benchmarks::ds4_complete {
namespace {
namespace en = engine;
namespace ca = catalog;
namespace pr = providers;
namespace sc = scheduler;
using Clock = std::chrono::steady_clock;
using Kind = model::Ds4BaselineStateKind;
constexpr std::uint32_t kContext = 8192;
constexpr std::uint64_t kGuard = std::uint64_t{6} << 30U;
constexpr std::uint64_t kOutputBCaptureBytes = std::uint64_t{kRows} * 8192 * 4;

void* Pointer(std::uint64_t address) { return en::support::Pointer(address); }

bool Add(std::uint64_t& total, std::uint64_t value) {
  return !__builtin_add_overflow(total, value, &total);
}
}  // namespace

Result Runner::Setup(const std::filesystem::path& artifact, const std::filesystem::path& scratch,
                     PreparedModelWeights& prepared, bool output_b_study, bool routed_ffn_study) {
  if (node_.threaded() || device_sms_ == 0 || prepared.plan.artifact_id != kCommunityArtifact ||
      !mapped_.empty() || raw_.opened() || (output_b_study && routed_ffn_study)) {
    return std::unexpected(
        "complete runner setup requires a fresh node and prepared community set");
  }
  if (auto opened = raw_.Open(artifact); !opened) return opened;
  if (raw_.artifact().id() != prepared.plan.artifact_id) {
    return std::unexpected("complete runner canonical and prepared identities differ");
  }
  const auto& profile = model::Dsv4Flash();
  auto binding = model::BindDsv4(profile, raw_.artifact());
  if (!binding) return std::unexpected(binding.error());
  native_binding_ = std::move(*binding);
  auto layout =
      model::LayoutDs4BaselineState(profile, kContext, kRows, node_.memory().Granularity().value());
  if (!layout) return std::unexpected(layout.error());
  state_layout_ = std::move(*layout);
  if (auto state =
          state_.AddGrowing(node_, "temporary ds4 original state", state_layout_.virtual_bytes, 0);
      !state)
    return state;
  // Replace every canonical expert group; no raw+aligned full expert copy.
  std::vector<en::GroupPlace> places(raw_.artifact().groups().size(), en::GroupPlace::kDevice);
  for (auto group : prepared.plan.replaced_groups) {
    if (group >= places.size()) return std::unexpected("invalid replaced raw group");
    places[group] = en::GroupPlace::kNone;
  }
  if (auto reserved = raw_.Reserve(node_, places, {}); !reserved) return reserved;
  if (auto reserved = aligned_.Reserve(node_, prepared.plan, prepared.files); !reserved) {
    return reserved;
  }
  prepared_plan_ = prepared.plan;
  auto scratch_plan = PlanScratch(profile, kContext, device_sms_, output_b_study, routed_ffn_study);
  if (!scratch_plan) return std::unexpected(scratch_plan.error());
  scratch_plan_ = std::move(*scratch_plan);
  // RunnerResources stores pointers, so reserve the final number before mapping.
  mapped_.reserve(scratch_plan_.ranges.size());
  named_.reserve(scratch_plan_.ranges.size());
  for (const auto& range : scratch_plan_.ranges) {
    if (range.name.starts_with("workspace.")) continue;
    auto& mapped = mapped_.emplace_back();
    if (auto status = resources_.Map(mapped, range.name, range.bytes, ca::MemoryClass::kScratch);
        !status)
      return status;
    named_.push_back(
        {.name = range.name, .storage = {.address = mapped.base, .bytes = range.bytes}});
  }
  if (auto workspace = node_.MapWorkspace(en::kPagedExtent, scratch_plan_.launch_workspace_bytes);
      !workspace) {
    return workspace;
  }
  if (auto cublas = resources_.OpenCublas("temporary ds4 cuBLAS"); !cublas) return cublas;
  if (resources_.cublas_bytes() < scratch_plan_.cublas_workspace_bytes) {
    return std::unexpected("native cuBLAS workspace is smaller than original reference");
  }
  named_.push_back(
      {.name = "workspace.launch",
       .storage = {.address = node_.pool().base, .bytes = scratch_plan_.launch_workspace_bytes}});
  named_.push_back({.name = "workspace.cublas",
                    .storage = {.address = resources_.cublas_workspace().base,
                                .bytes = resources_.cublas_bytes()}});
  if (auto launch = resources_.BindLaunch(scratch_plan_.launch_workspace_bytes); !launch) {
    return launch;
  }
  auto pinned = resources_.Pinned(std::uint64_t{kRows} * sizeof(std::int32_t));
  if (!pinned) return std::unexpected(pinned.error());
  host_tokens_ = *pinned;
  hash_bytes_ = std::uint64_t{profile.vocab} * profile.experts_used * sizeof(std::int32_t);
  pinned = resources_.Pinned(hash_bytes_ * profile.hash_layers);
  if (!pinned) return std::unexpected(pinned.error());
  host_hashes_ = *pinned;
  pinned = resources_.Pinned(std::uint64_t{profile.vocab} * sizeof(float));
  if (!pinned) return std::unexpected(pinned.error());
  host_logits_ = *pinned;
  pinned = resources_.Pinned(sizeof(std::array<float, 128>));
  if (!pinned) return std::unexpected(pinned.error());
  host_decode_table_ = *pinned;
  if (output_b_study || routed_ffn_study) {
    pinned = resources_.Pinned(kOutputBCaptureBytes);
    if (!pinned) return std::unexpected(pinned.error());
    host_output_b_capture_ = *pinned;
  }
  output_b_study_ = output_b_study;
  routed_ffn_study_ = routed_ffn_study;
  const auto table = kg::Ds4CacheDecodeTable();
  std::memcpy(host_decode_table_, table.data(), sizeof(table));
  // State virtual capacity is not resident occupancy. Budget it separately
  // for this fixed8K model control; ordinary growing-state serving is unchanged.
  budget_bytes_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  const auto state_budget = en::support::Round(state_layout_.virtual_bytes, en::kPagedExtent);
  if (!Add(budget_bytes_, raw_.bytes()) || !Add(budget_bytes_, aligned_.bytes()) ||
      !Add(budget_bytes_, state_budget) || !Add(budget_bytes_, 2 * en::kPagedExtent)) {
    return std::unexpected("complete runner execution budget overflows");
  }
  auto memory = pr::QueryDeviceMemory();
  if (!memory) return std::unexpected(memory.error().text());
  const auto future = raw_.bytes() + aligned_.bytes() + state_budget;
  if (future > memory->free || memory->free - future < kGuard) {
    return std::unexpected("complete reference needs its full native budget plus6GiB headroom");
  }
  if (auto started = node_.Start(base::Bytes(budget_bytes_)); !started) return started;
  if (auto registered = raw_.Register(node_, 0); !registered) return registered;
  if (auto registered = aligned_.Register(node_, 0); !registered) return registered;
  if (auto registered = state_.RegisterSpill(node_, scratch); !registered) return registered;
  node_.Run();
  return {};
}

std::vector<ca::ExtentId> Runner::managed_extents() const {
  auto extents = raw_.extents();
  extents.insert(extents.end(), aligned_.extents().begin(), aligned_.extents().end());
  const auto state = state_.extents();
  extents.insert(extents.end(), state.begin(), state.end());
  return extents;
}

Result Runner::RefreshClosure() {
  auto extents = managed_extents();
  const auto owned = resources_.extents();
  extents.insert(extents.end(), owned.begin(), owned.end());
  for (const auto* workspace : {&node_.activations(), &node_.pool()}) {
    extents.insert(extents.end(), workspace->extents.begin(), workspace->extents.end());
  }
  return node_.Call(
      [&]() -> Result {
        auto closure = node_.catalog().ClosureOfExtents(extents);
        if (!closure) return std::unexpected("complete runner closure was refused");
        everything_ = std::move(*closure);
        return {};
      },
      "complete runner closure");
}

Result Runner::Load() {
  if (!node_.threaded() || loaded_) return std::unexpected("complete runner load out of order");
  auto weights = raw_.extents();
  weights.insert(weights.end(), aligned_.extents().begin(), aligned_.extents().end());
  if (auto loaded =
          node_.Load(std::move(weights), "complete original weight representations", loads_);
      !loaded)
    return loaded;
  if (auto used = UseState(); !used) return used;
  if (auto hashes = ReadHashTables(); !hashes) return hashes;
  loaded_ = true;
  return {};
}

Result Runner::UseState() {
  auto ranges = model::Ds4BaselineStateThrough(state_layout_, kContext);
  if (!ranges) return std::unexpected(ranges.error());
  std::vector<en::LiveState::Range> state_ranges;
  state_ranges.reserve(ranges->size());
  for (const auto& range : *ranges) {
    state_ranges.push_back({.region = 0, .offset = range.offset, .bytes = range.bytes});
  }
  auto used = state_.Use(node_, state_ranges);
  if (!used) return std::unexpected(used.error());
  return RefreshClosure();
}

Result Runner::ReadHashTables() {
  std::string problem;
  auto job = [&, this](pr::NativeStream stream) -> sc::JobResult {
    for (std::uint32_t layer = 0; layer < model::Dsv4Flash().hash_layers; ++layer) {
      const auto index = native_binding_.layers[layer].tid2eid.index;
      const auto address = raw_.resource_address(index);
      if (address == 0 || !node_.Covered(address, hash_bytes_, 0)) {
        problem = "original hash table is not mapped/charged";
        return layer == 0 ? sc::JobResult::kNotStarted : sc::JobResult::kFailed;
      }
      auto* host = static_cast<std::byte*>(host_hashes_) + (layer * hash_bytes_);
      auto copied =
          pr::CopyAsync(stream, host, Pointer(address), hash_bytes_, pr::CopyKind::kDeviceToHost);
      if (!copied.ok()) {
        problem = copied.text();
        return sc::JobResult::kUnknown;
      }
    }
    return sc::JobResult::kQueued;
  };
  if (auto read = node_.Job(everything_, std::move(job), "authenticate original hash routing", 0);
      !read)
    return std::unexpected(problem.empty() ? read.error() : problem);
  hash_tables_.clear();
  for (std::uint32_t layer = 0; layer < model::Dsv4Flash().hash_layers; ++layer) {
    const auto index = native_binding_.layers[layer].tid2eid.index;
    const auto* host = reinterpret_cast<const std::int32_t*>(
        static_cast<const std::byte*>(host_hashes_) + (layer * hash_bytes_));
    hash_tables_.push_back(
        {.layer = layer,
         .device = {.address = raw_.resource_address(index), .bytes = hash_bytes_},
         .entries = {host, static_cast<std::size_t>(hash_bytes_ / sizeof(std::int32_t))}});
  }
  return {};
}

kg::Ds4CacheBuffer Runner::State(Kind kind, std::uint32_t layer) const {
  for (const auto& tensor : state_layout_.tensors) {
    if (tensor.kind == kind && tensor.layer == layer) {
      return {.address = state_.base(0) + tensor.offset, .bytes = tensor.bytes};
    }
  }
  return {};
}

const NamedScratch* Runner::FindScratch(std::string_view name) const {
  const auto found = std::ranges::find(named_, name, &NamedScratch::name);
  return found == named_.end() ? nullptr : &*found;
}

Result Runner::Initialize() {
  if (!loaded_) return std::unexpected("complete state setup requires loaded weights");
  initialized_ = false;
  if (auto cleared = state_.Clear(node_, everything_, 0, "original fresh state"); !cleared) {
    return cleared;
  }
  // Growing-state Clear discards occupancy. Reload its registered zero
  // sources and rebuild the generation-bearing closure before any write.
  if (auto used = UseState(); !used) return used;
  const auto* attention_diagnostics = FindScratch("attention.diagnostics");
  const auto* indexer_diagnostics = FindScratch("indexer.diagnostics");
  if (attention_diagnostics == nullptr || indexer_diagnostics == nullptr) {
    return std::unexpected("complete reference diagnostic scratch missing");
  }
  std::string problem;
  auto job = [&, this](pr::NativeStream stream) -> sc::JobResult {
    for (const auto* diagnostics : {attention_diagnostics, indexer_diagnostics}) {
      const auto zeroed = pr::FillAsync(stream, Pointer(diagnostics->storage.address), 0,
                                        diagnostics->storage.bytes);
      if (!zeroed.ok()) {
        problem = zeroed.text();
        return sc::JobResult::kUnknown;
      }
    }
    const auto table = State(Kind::kDecodeTable, 0);
    auto copied = pr::CopyAsync(stream, Pointer(table.address), host_decode_table_, 512,
                                pr::CopyKind::kHostToDevice);
    if (!copied.ok()) {
      problem = copied.text();
      return sc::JobResult::kUnknown;
    }
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
      const auto ratio = model::Dsv4Flash().compress_ratios[layer];
      if (ratio == 0) continue;
      kg::Ds4CompState frontier{.kv = State(Kind::kAttentionKv, layer),
                                .score = State(Kind::kAttentionScore, layer),
                                .kind = kg::Ds4CacheKind::kKv512,
                                .ratio = ratio};
      auto initialized = kg::RunDs4CompInitialize(resources_.launch(), frontier);
      if (!initialized) {
        problem = initialized.error().detail;
        return resources_.launch().faulted() ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
      }
      if (ratio == 4) {
        frontier = {.kv = State(Kind::kIndexerKv, layer),
                    .score = State(Kind::kIndexerScore, layer),
                    .kind = kg::Ds4CacheKind::kIndexer128,
                    .ratio = 4};
        initialized = kg::RunDs4CompInitialize(resources_.launch(), frontier);
        if (!initialized) {
          problem = initialized.error().detail;
          return resources_.launch().faulted() ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
        }
      }
    }
    return sc::JobResult::kQueued;
  };
  if (auto status =
          node_.Job(everything_, std::move(job), "original finite-score frontier boot", 0);
      !status)
    return std::unexpected(problem.empty() ? status.error() : problem);
  if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
    return std::unexpected("complete runner state generation exhausted");
  }
  ++generation_;
  initialized_ = true;
  return {};
}

Result Runner::PrepareProfile() {
  if (!loaded_ || profile_mark_count_ != 0)
    return std::unexpected("diagnostic marks require loaded weights and a fresh mark owner");
  auto memory = pr::QueryDeviceMemory();
  if (!memory) return std::unexpected(memory.error().text());
  profile_setup_free_bytes_[0] = memory->free;
  for (auto& mark : profile_.marks) {
    auto created = pr::CreateTimingMark();
    if (!created) return std::unexpected(created.error().text());
    if (created->handle == nullptr)
      return std::unexpected("diagnostic provider created an empty timing mark");
    mark = *created;
    ++profile_mark_count_;
  }
  memory = pr::QueryDeviceMemory();
  if (!memory) return std::unexpected(memory.error().text());
  profile_setup_free_bytes_[1] = memory->free;
  return {};
}

std::expected<Pass, std::string> Runner::Prefill(std::span<const std::int32_t> tokens, bool profile,
                                                 OutputBConsumer output_b,
                                                 const std::filesystem::path& capture,
                                                 RoutedFfnTier routed_ffn) {
  if (!initialized_ || tokens.size() != kContext ||
      (profile && profile_mark_count_ != profile_.marks.size()) ||
      std::ranges::any_of(tokens, [](auto token) { return token < 0 || token >= 129280; }) ||
      (output_b != OutputBConsumer::kOriginal && output_b != OutputBConsumer::kNativeMmq) ||
      (!output_b_study_ && output_b != OutputBConsumer::kOriginal) ||
      (routed_ffn != RoutedFfnTier::kDirect && routed_ffn != RoutedFfnTier::kMaterialized) ||
      (!routed_ffn_study_ && routed_ffn != RoutedFfnTier::kDirect) ||
      (profile && (output_b_study_ || routed_ffn_study_)) ||
      (!capture.empty() &&
       (profile || output_b != OutputBConsumer::kOriginal || routed_ffn != RoutedFfnTier::kDirect ||
        (!output_b_study_ && !routed_ffn_study_) || host_output_b_capture_ == nullptr))) {
    return std::unexpected("complete reference needs exactly8192 validated current token IDs");
  }
  // Every pass consumes fresh state; failure never permits a suffix retry.
  initialized_ = false;
  Pass result;
  result.output_b_consumer = output_b;
  result.routed_ffn_tier = routed_ffn;
  result.operand_capture = !capture.empty();
  // All sample storage is allocated before the profiled interval. The
  // sixteen handles were allocated once before ordinary warmup.
  if (profile) result.profile.reserve((std::size_t{2} * kLayers * kProfileChains) + 3);
  auto* marks = profile ? &profile_ : nullptr;
  std::optional<Progress> previous;
  const auto run = [&]() -> Result {
    for (std::uint32_t chunk_index = 0; chunk_index < 2; ++chunk_index) {
      const auto first = chunk_index * kRows;
      auto chunk = BindChunk(
          model::Dsv4Flash(),
          {.artifact = &raw_.artifact(),
           .raw_weights = &raw_,
           .prepared_plan = &prepared_plan_,
           .aligned_weights = &aligned_,
           .state = &state_layout_,
           .state_storage = {.address = state_.base(0), .bytes = state_layout_.virtual_bytes},
           .scratch = named_,
           .hash_tables = hash_tables_,
           .first = first,
           .device_sms = device_sms_,
           .storage_generation = generation_,
           .model_generation = 1,
           .output_b_consumer = output_b,
           .output_b_study = output_b_study_,
           .capture_output_b = output_b_study_ && !capture.empty() && first == kRows,
           .routed_ffn_tier = routed_ffn,
           .routed_ffn_study = routed_ffn_study_,
           .capture_routed_ffn = routed_ffn_study_ && !capture.empty() && first == kRows});
      if (!chunk) return std::unexpected(chunk.error());
      std::memcpy(host_tokens_, tokens.data() + first, std::uint64_t{kRows} * sizeof(std::int32_t));
      const auto* input = FindScratch("tokens");
      if (input == nullptr) return std::unexpected("complete reference token scratch missing");
      std::string problem;
      auto upload = [&, this](pr::NativeStream stream) -> sc::JobResult {
        const auto status =
            pr::CopyAsync(stream, Pointer(input->storage.address), host_tokens_,
                          std::uint64_t{kRows} * sizeof(std::int32_t), pr::CopyKind::kHostToDevice);
        if (!status.ok()) {
          problem = status.text();
          return sc::JobResult::kUnknown;
        }
        return sc::JobResult::kQueued;
      };
      if (auto status =
              node_.Job(everything_, std::move(upload), "original current token upload", 0);
          !status)
        return std::unexpected(problem.empty() ? status.error() : problem);
      if (auto resolved = Resolve(node_, resources_.launch(), everything_, 0, model::Dsv4Flash(),
                                  *chunk, hash_tables_);
          !resolved)
        return resolved;
      auto dispatch = ReadResolvedDispatch(*chunk);
      if (!dispatch) return std::unexpected(dispatch.error());
      result.dispatch[chunk_index] = std::move(*dispatch);
      auto progress = Begin(model::Dsv4Flash(), *chunk, previous ? &*previous : nullptr);
      if (!progress) return std::unexpected(progress.error());
      const auto started = Clock::now();
      if (auto embedded =
              RunEmbedding(node_, resources_.launch(), everything_, 0, *chunk, *progress, marks);
          !embedded)
        return embedded;
      if (profile)
        result.profile.push_back({chunk_index, 0, 0, "embedding", profile_.milliseconds[0], true});
      for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        if (auto status = RunLayer(node_, resources_.launch(), everything_, 0, *chunk, layer,
                                   *progress, marks);
            !status)
          return status;
        if (output_b_study_ && !capture.empty() && CaptureOutputBAt(first, layer)) {
          if (auto copied = CaptureOutputB(*chunk, layer, capture); !copied) {
            Poison(*progress);
            return copied;
          }
        }
        if (routed_ffn_study_ && !capture.empty() && CaptureRoutedFfnAt(first, layer)) {
          if (auto copied = CaptureRoutedFfn(*chunk, layer, capture); !copied) {
            Poison(*progress);
            return copied;
          }
        }
        if (profile) {
          for (std::size_t chain = 0; chain < kProfileChains; ++chain)
            result.profile.push_back({chunk_index, layer, chunk->layers[layer].ratio,
                                      kProfileNames[chain], profile_.milliseconds[chain],
                                      profile_.active[chain]});
        }
      }
      if (chunk->frontier) {
        if (auto status =
                RunFrontier(node_, resources_.launch(), everything_, 0, *chunk, *progress, marks);
            !status)
          return status;
        if (profile)
          result.profile.push_back(
              {chunk_index, kLayers, 0, "frontier-full-head", profile_.milliseconds[0], true});
      }
      result.chunks[chunk_index] = en::support::Seconds(Clock::now() - started);
      result.seconds += result.chunks[chunk_index];
      previous = std::move(*progress);
    }
    return {};
  };
  (void)node_.TakeTimes(0);
  const auto wall_started = Clock::now();
  auto status = node_.WithRequest(0, everything_, "complete original8K prefill", run);
  result.wall_seconds = en::support::Seconds(Clock::now() - wall_started);
  result.steps = node_.TakeTimes(0);
  if (!status) {
    state_.Quarantine();
    return std::unexpected(status.error());
  }
  const auto* logits = FindScratch("head.logits");
  if (logits == nullptr) return std::unexpected("complete reference logits scratch missing");
  std::string problem;
  const auto copy_started = Clock::now();
  auto read = [&, this](pr::NativeStream stream) -> sc::JobResult {
    const auto copied = pr::CopyAsync(stream, host_logits_, Pointer(logits->storage.address),
                                      std::uint64_t{model::Dsv4Flash().vocab} * sizeof(float),
                                      pr::CopyKind::kDeviceToHost);
    if (!copied.ok()) {
      problem = copied.text();
      return sc::JobResult::kUnknown;
    }
    return sc::JobResult::kQueued;
  };
  if (auto copied = node_.Job(everything_, std::move(read), "complete original final logits", 0);
      !copied)
    return std::unexpected(problem.empty() ? copied.error() : problem);
  const auto* host = static_cast<const float*>(host_logits_);
  result.logits.assign(host, host + model::Dsv4Flash().vocab);
  result.result_copy_seconds = en::support::Seconds(Clock::now() - copy_started);
  return result;
}

Result Runner::CaptureOutputB(const Chunk& chunk, std::uint32_t layer,
                              const std::filesystem::path& directory) {
  if (!output_b_study_ || host_output_b_capture_ == nullptr ||
      chunk.output_b_consumer != OutputBConsumer::kOriginal ||
      !CaptureOutputBAt(chunk.first, layer) || layer >= chunk.layers.size() ||
      chunk.output_b_control.address == 0)
    return std::unexpected("output-B capture requires completed original upstream operands");
  const auto& product = chunk.layers[layer].output_b;
  if (!product.prepared || product.generation != chunk.storage_generation ||
      product.quantized.generation != chunk.storage_generation ||
      product.quantized.source != product.input.storage.data)
    return std::unexpected("output-B capture lost the current original producer identity");
  const auto buffer = [](auto storage) {
    return kg::Ds4CacheBuffer{
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(storage.data)), storage.bytes};
  };
  struct File {
    std::string_view name;
    kg::Ds4CacheBuffer source;
    bool floating;
  };
  const std::array files{File{"input.f32", buffer(product.input.storage), true},
                         File{"input.d4", buffer(product.quantized.storage), false},
                         File{"weights.q8", buffer(product.weights.raw), false},
                         File{"original.f32", buffer(product.output.storage), true},
                         File{"native.f32", chunk.output_b_control, true}};
  std::error_code problem;
  const auto output = directory / ("chunk1-layer" + std::to_string(layer));
  if (!std::filesystem::create_directory(output, problem) || problem)
    return std::unexpected("output-B capture directory must be fresh and writable");
  std::array<std::string, files.size()> digests;
  for (std::size_t index = 0; index < files.size(); ++index) {
    const auto& file = files[index];
    if (file.source.address == 0 || file.source.bytes == 0 ||
        file.source.bytes > kOutputBCaptureBytes ||
        !node_.Covered(file.source.address, file.source.bytes, 0))
      return std::unexpected("output-B capture range is outside retained mapped operands");
    std::string error;
    auto copied = node_.Job(
        everything_,
        [&](pr::NativeStream stream) {
          auto status = pr::CopyAsync(stream, host_output_b_capture_, Pointer(file.source.address),
                                      file.source.bytes, pr::CopyKind::kDeviceToHost);
          if (!status.ok()) {
            error = status.text();
            return sc::JobResult::kUnknown;
          }
          return sc::JobResult::kQueued;
        },
        "diagnostic output-B current original operand capture", 0);
    if (!copied) return std::unexpected(error.empty() ? copied.error() : error);
    // Owned host staging is consumed only after this copy's Job fence.
    const auto bytes = std::span(static_cast<const std::byte*>(host_output_b_capture_),
                                 static_cast<std::size_t>(file.source.bytes));
    if (file.floating) {
      if (file.source.bytes % sizeof(float) != 0 ||
          std::ranges::any_of(
              std::span(static_cast<const float*>(host_output_b_capture_),
                        static_cast<std::size_t>(file.source.bytes / sizeof(float))),
              [](float value) { return !std::isfinite(value); }))
        return std::unexpected("captured output-B input/output contains nonfinite values");
    }
    if (file.name == "input.d4") {
      const auto payload = std::uint64_t{kRows} * (8192 / 128) * 144;
      if (bytes.size() != payload + (256ULL * 144) ||
          std::ranges::any_of(bytes.subspan(payload),
                              [](std::byte value) { return value != std::byte{0}; }))
        return std::unexpected("captured original D4 does not retain its complete zero guard");
      for (std::uint64_t block = 0; block < payload / 144; ++block) {
        for (std::uint64_t scale = 0; scale < 4; ++scale) {
          float value = 0;
          std::memcpy(&value, bytes.data() + (block * 144) + (scale * 4), sizeof(value));
          if (!std::isfinite(value) || value < 0)
            return std::unexpected("captured original D4 has an invalid F32 scale");
        }
      }
    }
    base::Sha256 hash;
    hash.Update(bytes);
    digests[index] = base::ToHex(hash.Finish());
    std::ofstream stored(output / file.name, std::ios::binary | std::ios::noreplace);
    stored.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    stored.close();
    if (!stored) return std::unexpected("cannot persist complete output-B captured bytes");
  }
  std::ofstream metadata(output / "capture.json", std::ios::out | std::ios::noreplace);
  metadata << R"({"complete":true,"diagnostic":true,"original_upstream":true,)"
           << R"("quality_override":false,"rows":4096,"input_width":8192,"output_width":4096,)"
           << R"("weight_format":"Q8_0","input_format":"D4-F32x4-I8x128",)"
           << R"("guard_blocks":256,"chunk":1,"layer":)" << layer << R"(,"source_generation":)"
           << product.quantized.generation << R"(,"current_generation":)"
           << chunk.storage_generation << R"(,"source_identity_matches":)"
           << (product.quantized.source == product.input.storage.data ? "true" : "false")
           << R"(,"original_prepared":)" << (product.prepared ? "true" : "false")
           << R"(,"artifact":)";
  std::string quoted;
  base::json::AppendQuoted(chunk.artifact_id, quoted);
  metadata << quoted << R"(,"files":[)";
  for (std::size_t index = 0; index < files.size(); ++index) {
    if (index != 0) metadata << ',';
    quoted.clear();
    base::json::AppendQuoted(files[index].name, quoted);
    metadata << R"({"name":)" << quoted << R"(,"bytes":)" << files[index].source.bytes
             << R"(,"sha256":")" << digests[index] << R"(","finite_f32":)"
             << (files[index].floating ? "true" : "false") << '}';
  }
  metadata << "]}\n";
  metadata.close();
  if (!metadata) return std::unexpected("cannot complete output-B capture metadata");
  return {};
}

Result Runner::CaptureRoutedFfn(const Chunk& chunk, std::uint32_t layer,
                                const std::filesystem::path& directory) {
  if (!routed_ffn_study_ || host_output_b_capture_ == nullptr ||
      chunk.routed_ffn_tier != RoutedFfnTier::kDirect || !CaptureRoutedFfnAt(chunk.first, layer) ||
      layer >= chunk.layers.size())
    return std::unexpected("routed-FFN capture requires completed ORIGINAL Direct operands");
  const auto& recipe = chunk.layers[layer];
  const auto& probe = recipe.routed_control;
  if (!probe)
    return std::unexpected("routed-FFN capture requires completed ORIGINAL Direct operands");
  const auto& original = recipe.routed;
  const auto& control = *probe;
  if (original.producer.generation != chunk.storage_generation ||
      original.producer.source_address != original.input.address ||
      original.producer.kind != kg::Ds4MoeQuant::kD4)
    return std::unexpected("routed-FFN capture lost the original token-D4 producer identity");
  enum class Format : std::uint8_t { kBytes, kFloat, kD4, kD2s6 };
  struct File {
    std::string_view name;
    kg::Ds4CacheBuffer source;
    Format format = Format::kBytes;
    std::uint64_t payload = 0;
  };
  constexpr std::uint64_t token_payload = std::uint64_t{kRows} * (4096 / 128) * 144;
  constexpr std::uint64_t pair_payload = 6 * token_payload;
  constexpr std::uint64_t down_payload = 6ULL * kRows * (2048 / 128) * 144;
  const std::array files{
      File{"input.f32", original.input, Format::kFloat},
      File{"token.d4", original.producer.storage, Format::kD4, token_payload},
      File{"selected.i32", original.selected},
      File{"weights.f32", original.weights, Format::kFloat},
      File{"gate.aligned-iq2", original.gate_weights},
      File{"up.aligned-iq2", original.up_weights},
      File{"down.aligned-q2", original.down_weights},
      File{"direct.ids-source.i32", original.ids_source},
      File{"direct.ids-destination.i32", original.ids_destination},
      File{"direct.bounds.i32", original.expert_bounds},
      File{"direct.work.i32", original.work},
      File{"direct.d2s6", original.down_quant, Format::kD2s6, down_payload},
      File{"direct.down.f32", original.down, Format::kFloat},
      File{"materialized.ids-source.i32", control.ids_source},
      File{"materialized.ids-destination.i32", control.ids_destination},
      File{"materialized.bounds.i32", control.expert_bounds},
      File{"materialized.work.i32", control.work},
      File{"materialized.input.d4", control.input_quant, Format::kD4, pair_payload},
      File{"materialized.gate.f32", control.gate, Format::kFloat},
      File{"materialized.up.f32", control.up, Format::kFloat},
      File{"materialized.middle.f32", control.middle, Format::kFloat},
      File{"materialized.d2s6", control.down_quant, Format::kD2s6, down_payload},
      File{"materialized.down.f32", control.down, Format::kFloat}};
  std::error_code problem;
  const auto output = directory / ("chunk1-layer" + std::to_string(layer));
  if (!std::filesystem::create_directory(output, problem) || problem)
    return std::unexpected("routed-FFN capture directory must be fresh and writable");
  std::array<std::string, files.size()> digests;
  // This quantum is block-aligned for both quant layouts and F32. A file
  // can exceed staging capacity; consume each slice only after its Job
  // fence, before reusing the same 128MiB owner for the next slice.
  constexpr auto quantum = (kOutputBCaptureBytes / 144) * 144;
  for (std::size_t index = 0; index < files.size(); ++index) {
    const auto& file = files[index];
    if (file.source.address == 0 || file.source.bytes == 0 ||
        file.source.bytes > (std::uint64_t{4} << 30U) ||
        file.source.address > std::numeric_limits<std::uint64_t>::max() - file.source.bytes ||
        !node_.Covered(file.source.address, file.source.bytes, 0))
      return std::unexpected("routed-FFN capture range is outside retained mapped operands");
    if (file.payload != 0 &&
        (file.payload % 144 != 0 ||
         file.source.bytes != file.payload + ((file.name == "token.d4" ? 256ULL : 128ULL) * 144)))
      return std::unexpected("routed-FFN quantized capture has incomplete guard capacity");
    if (file.format == Format::kFloat && file.source.bytes % sizeof(float) != 0)
      return std::unexpected("routed-FFN F32 capture has a partial scalar");
    std::ofstream stored(output / file.name, std::ios::binary | std::ios::noreplace);
    if (!stored) return std::unexpected("routed-FFN operand output must be a fresh file");
    base::Sha256 hash;
    for (std::uint64_t offset = 0; offset < file.source.bytes;) {
      const auto count = std::min(quantum, file.source.bytes - offset);
      std::string error;
      auto copied = node_.Job(
          everything_,
          [&](pr::NativeStream stream) {
            auto status =
                pr::CopyAsync(stream, host_output_b_capture_, Pointer(file.source.address + offset),
                              count, pr::CopyKind::kDeviceToHost);
            if (!status.ok()) {
              error = status.text();
              return sc::JobResult::kUnknown;
            }
            return sc::JobResult::kQueued;
          },
          "diagnostic routed-FFN original-input operand slice", 0);
      if (!copied) return std::unexpected(error.empty() ? copied.error() : error);
      const auto bytes = std::span(static_cast<const std::byte*>(host_output_b_capture_),
                                   static_cast<std::size_t>(count));
      if (file.format == Format::kFloat &&
          std::ranges::any_of(std::span(static_cast<const float*>(host_output_b_capture_),
                                        static_cast<std::size_t>(count / sizeof(float))),
                              [](float value) { return !std::isfinite(value); }))
        return std::unexpected("routed-FFN captured F32 operand contains nonfinite values");
      if (file.payload != 0) {
        const auto payload_count =
            offset < file.payload ? std::min(count, file.payload - offset) : 0;
        for (std::uint64_t block = 0; block < payload_count / 144; ++block) {
          if (file.format == Format::kD4) {
            for (std::uint64_t scale = 0; scale < 4; ++scale) {
              float value = 0;
              std::memcpy(&value, bytes.data() + (block * 144) + (scale * 4), sizeof(value));
              if (!std::isfinite(value) || value < 0)
                return std::unexpected("routed-FFN D4 capture has an invalid F32 scale");
            }
          } else {
            for (std::uint64_t scalar = 0; scalar < 8; ++scalar) {
              std::uint16_t value = 0;
              std::memcpy(&value, bytes.data() + (block * 144) + (scalar * 2), sizeof(value));
              if ((value & 0x7c00U) == 0x7c00U || (scalar < 2 && (value & 0x8000U) != 0))
                return std::unexpected("routed-FFN D2S6 capture has a nonfinite scale/sum");
            }
          }
        }
        if (std::ranges::any_of(bytes.subspan(static_cast<std::size_t>(payload_count)),
                                [](std::byte value) { return value != std::byte{0}; }))
          return std::unexpected("routed-FFN capture lost its complete initialized quant guard");
      }
      hash.Update(bytes);
      stored.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(count));
      if (!stored) return std::unexpected("cannot persist routed-FFN operand slice");
      offset += count;
    }
    stored.close();
    if (!stored) return std::unexpected("cannot finish routed-FFN operand file");
    digests[index] = base::ToHex(hash.Finish());
  }
  std::ofstream metadata(output / "capture.json", std::ios::out | std::ios::noreplace);
  std::string quoted;
  base::json::AppendQuoted(chunk.artifact_id, quoted);
  metadata << R"({"complete":true,"diagnostic_only":true,"same_original_upstream":true,)"
           << R"("original_outputs_retained":true,"axis":"producer/gather+fusion/storage",)"
           << R"("artifact":)" << quoted << R"(,"first":)" << chunk.first << R"(,"layer":)" << layer
           << R"(,"generation":)" << chunk.storage_generation
           << R"(,"rows":4096,"input":4096,"middle":2048,"output":4096,"pairs":24576,)"
           << R"("experts":256,"top_k":6,"selected_stride":6,"weight_stride":6,)"
           << R"("intermediate_bytes":603979776,"host_staging_bytes":)" << kOutputBCaptureBytes
           << R"(,"copy_quantum":)" << quantum << R"(,"files":[)";
  for (std::size_t index = 0; index < files.size(); ++index) {
    if (index != 0) metadata << ',';
    quoted.clear();
    base::json::AppendQuoted(files[index].name, quoted);
    metadata << R"({"name":)" << quoted << R"(,"bytes":)" << files[index].source.bytes
             << R"(,"sha256":)";
    quoted.clear();
    base::json::AppendQuoted(digests[index], quoted);
    metadata << quoted << R"(,"format":)" << static_cast<unsigned>(files[index].format)
             << R"(,"payload_bytes":)" << files[index].payload << '}';
  }
  metadata << "]}\n";
  metadata.close();
  if (!metadata) return std::unexpected("cannot complete routed-FFN capture metadata");
  return {};
}

Result Runner::Release() {
  // PagedNode has fenced every consumer and evicted every managed extent.
  std::vector<std::string> problems;
  // Destroy only here, after fenced teardown; partial creation and unknown
  // recordings retain the same owner until this proof exists.
  for (auto& mark : profile_.marks) {
    if (mark.handle != nullptr) pr::DestroyTimingMark(mark);
    mark = {};
  }
  profile_mark_count_ = 0;
  resources_.Release(problems);
  host_output_b_capture_ = nullptr;
  output_b_study_ = false;
  routed_ffn_study_ = false;
  state_.Release(node_.memory(), problems);
  if (auto status = raw_.Release(node_.memory()); !status) problems.push_back(status.error());
  if (auto status = aligned_.Release(node_.memory()); !status) problems.push_back(status.error());
  named_.clear();
  hash_tables_.clear();
  mapped_.clear();
  return en::support::Joined(problems);
}

}  // namespace jitllm::benchmarks::ds4_complete
