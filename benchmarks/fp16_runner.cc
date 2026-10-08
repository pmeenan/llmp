// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "fp16_runner.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <expected>
#include <format>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <system_error>
#include <utility>

#include "artifact/layout.h"
#include "base/sha256.h"
#include "ggml.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "paging_cases.h"
#include "plan_record.h"
#include "providers/direct_reader.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"

namespace llmp::benchmarks {

namespace {

namespace kg = llmp::kernels::ggml;
namespace sc = llmp::scheduler;
namespace ts = llmp::test_support;
using base::Bytes;
using catalog::ExtentId;
using catalog::MemoryClass;
using catalog::Recovery;
using providers::BackingKind;
using Status = test_support::Status;

constexpr std::uint64_t kExtent = test_support::kPagedExtent;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

Status Cuda(cudaError_t result, std::string_view what) {
  if (result != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(result)));
  }
  return {};
}

void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

// Each load, with the requests its reads took and the chunks they carried
// (BP-P1).
std::string LoadsJson(std::span<const ts::LoadStats> stats) {
  std::string loads;
  for (const ts::LoadStats& load : stats) {
    loads +=
        std::format(R"({}{{"what":"{}","extents":{},"seconds":{:.6f},"requests":{},"pieces":{}}})",
                    loads.empty() ? "" : ",", load.what, load.extents, load.seconds, load.requests,
                    load.pieces);
  }
  return loads;
}

}  // namespace

std::uint64_t Fp16Runner::WeightAddress(std::uint32_t resource) const {
  const auto& r = artifact_->resources()[resource];
  return weights_base_.at(place_) + group_region_[r.group] + r.offset.value();
}

std::vector<ExtentId> Fp16Runner::weights() const {
  std::vector<ExtentId> all = device_weights_;
  all.insert(all.end(), table_extents_.begin(), table_extents_.end());
  return all;
}

std::uint64_t Fp16Runner::weight_read_bytes() const {
  std::uint64_t bytes = 0;
  const auto groups = artifact_->groups();
  for (const auto& group : groups) {
    bytes += group.stored.value();
  }
  if (!table_extents_.empty()) {
    bytes += groups[artifact_->resources()[binding_.token_embd].group].stored.value();
  }
  return bytes;
}

std::vector<ExtentId> Fp16Runner::managed_extents() const {
  std::vector<ExtentId> all = weights();
  if (o_.spill == "managed") {
    // Its backing is the VMM lane's to release: written back and evicted.
    all.insert(all.end(), kv_.extents.begin(), kv_.extents.end());
  }
  return all;
}

// The weights' places, and an extent per chunk (and per host table chunk).
Status Fp16Runner::ReserveWeights() {
  auto& memory = node_.memory();
  const auto groups = artifact_->groups();
  group_region_.resize(groups.size());
  chunk_extents_.resize(groups.size());
  for (std::size_t g = 0; g < groups.size(); ++g) {
    group_region_[g] = weights_bytes_;
    weights_bytes_ += std::uint64_t{groups[g].chunks} * kExtent;
    stored_bytes_ += groups[g].stored.value();
  }
  for (std::size_t which = 0; which < weights_.size(); ++which) {
    if (which > 0 && o_.load_only > 0) {
      break;  // nothing relocates
    }
    auto reservation = memory.Reserve(Bytes(weights_bytes_));
    if (!reservation) {
      return Error(std::format("reserving the weights: {}", reservation.error().detail));
    }
    weights_.at(which) = *reservation;
    weights_base_.at(which) = memory.RangeOf(*reservation).value().base;
  }
  // The artifact's identity orders victim ties (catalog.h ContentKey).
  for (std::size_t i = 0; i < id_.size() && (2 * i) + 1 < artifact_->id().size(); ++i) {
    (void)std::from_chars(artifact_->id().data() + (2 * i), artifact_->id().data() + (2 * i) + 2,
                          id_.at(i), 16);
  }
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      auto extent =
          node_.catalog().AddExtent({.domain = node_.domain(),
                                     .memory_class = MemoryClass::kWeights,
                                     .recovery = Recovery::kFromArtifact,
                                     .size = Bytes(kExtent),
                                     .content = {.artifact = id_, .group = g, .chunk = c}});
      if (!extent) {
        return Error("cataloging a weight chunk");
      }
      chunk_extents_[g].push_back(*extent);
      device_weights_.push_back(*extent);
    }
  }
  if (o_.premapped) {
    // Every extent's backing, mapped once for the whole run.
    const std::size_t allocation_class =
        o_.weights_host ? node_.host_class() : node_.device_class();
    for (std::uint64_t at = 0; at < weights_bytes_; at += kExtent) {
      auto backing = memory.Create(allocation_class, Bytes(kExtent));
      if (!backing || !memory.Map(weights_[0], Bytes(at), *backing)) {
        return Error("premapping the weights");
      }
      premapped_.push_back(*backing);
    }
    if (!memory.SetAccess(weights_[0], Bytes(0), Bytes(weights_bytes_),
                          providers::Access::kReadWrite)) {
      return Error("access to the premapped weights");
    }
  }
  if (o_.shared_embeddings) {
    return {};  // one copy of the table, in device VMM (BP-P3)
  }
  // The token table's host copy: its group's chunks again, read directly
  // into host VMM the CPU maps.
  const std::uint32_t table_group = artifact_->resources()[binding_.token_embd].group;
  auto host = memory.Reserve(Bytes(std::uint64_t{groups[table_group].chunks} * kExtent));
  if (!host) {
    return Error(std::format("reserving the host table: {}", host.error().detail));
  }
  table_ = *host;
  table_base_ = memory.RangeOf(*host).value().base;
  for (std::uint32_t c = 0; c < groups[table_group].chunks; ++c) {
    auto extent =
        node_.catalog().AddExtent({.domain = node_.domain(),
                                   .memory_class = MemoryClass::kWeights,
                                   .recovery = Recovery::kFromArtifact,
                                   .size = Bytes(kExtent),
                                   .content = {.artifact = id_, .group = table_group, .chunk = c}});
    if (!extent) {
      return Error("cataloging a host table chunk");
    }
    table_extents_.push_back(*extent);
  }
  return {};
}

Status Fp16Runner::Register() {
  if (!o_.shared_embeddings) {
    const std::uint32_t table_group = artifact_->resources()[binding_.token_embd].group;
    for (std::uint32_t c = 0; c < table_extents_.size(); ++c) {
      const auto range =
          artifact::ChunkRangeOf(artifact_->layout(), {.group = table_group, .chunk = c});
      if (!range) {
        return Error("the table's chunk range");
      }
      const std::uint64_t address = table_base_ + (std::uint64_t{c} * kExtent);
      auto set = node_.scheduler().SetSource(
          table_extents_[c],
          sc::PageSource{.read = {.fd = shards_.at(range->shard).get(),
                                  .offset = range->file_offset.value(),
                                  // NOLINTNEXTLINE(performance-no-int-to-ptr): host VMM, CPU-mapped
                                  .memory = reinterpret_cast<std::byte*>(address),
                                  .length = range->length.value()},
                         .landed = false,
                         .destination = 0,
                         .backing = sc::BackingPlace{.reservation = table_,
                                                     .offset = Bytes(std::uint64_t{c} * kExtent),
                                                     .size = Bytes(kExtent),
                                                     .allocation_class = node_.host_class()}});
      if (!set) {
        return Error(std::format("the table's source: {}", sc::ToString(set.error())));
      }
      node_.AddSpan({.base = address,
                     .size = kExtent,
                     .extent = table_extents_[c],
                     .memory_class = MemoryClass::kWeights,
                     .device = false,
                     .owner = owner_});
    }
  }
  if (auto r = Place(0); !r) {
    return r;
  }
  if (!o_.spill.empty()) {
    return RegisterCache();
  }
  return {};
}

// Registers every device weight chunk's source at place `which`.
Status Fp16Runner::Place(std::size_t which) {
  const auto groups = artifact_->groups();
  node_.EraseSpans([&](const ts::Span& span) {
    return span.owner == owner_ && span.memory_class == MemoryClass::kWeights && span.device;
  });
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      const auto range = artifact::ChunkRangeOf(artifact_->layout(), {.group = g, .chunk = c});
      if (!range) {
        return Error("a chunk's range");
      }
      const std::uint64_t offset = group_region_[g] + (std::uint64_t{c} * kExtent);
      const std::uint64_t address = weights_base_.at(which) + offset;
      const ExtentId extent = chunk_extents_[g][c];
      std::optional<sc::BackingPlace> backing;
      if (!o_.premapped) {
        backing = sc::BackingPlace{
            .reservation = weights_.at(which),
            .offset = Bytes(offset),
            .size = Bytes(kExtent),
            .allocation_class = o_.weights_host ? node_.host_class() : node_.device_class()};
      }
      auto set = node_.scheduler().SetSource(
          extent,
          sc::PageSource{
              .read = {.fd = shards_.at(range->shard).get(),
                       .offset = range->file_offset.value(),
                       // In place, for --weights host (D-034).
                       // NOLINTNEXTLINE(performance-no-int-to-ptr)
                       .memory = o_.weights_host ? reinterpret_cast<std::byte*>(address) : nullptr,
                       .length = range->length.value()},
              .landed = !o_.weights_host,
              .destination = o_.weights_host ? 0 : address,
              .backing = backing});
      if (!set) {
        return Error(std::format("a chunk's source: {}", sc::ToString(set.error())));
      }
      node_.AddSpan({.base = address,
                     .size = kExtent,
                     .extent = extent,
                     .memory_class = MemoryClass::kWeights,
                     .device = !o_.weights_host,
                     .owner = owner_});
    }
  }
  node_.SortSpans();
  place_ = which;
  return {};
}

// BP-A1's in-process check: every tensor the chunk's plan binds lies in
// cataloged, resident extents of device memory of one class, and each has
// the class it should: weights, the cache (live state) or the activations
// (scratch).
void Fp16Runner::Check(const kg::Qwen2Graph& graph) {
  const auto expect = [&](const ggml_tensor* t, MemoryClass memory_class) {
    ++coverage_.tensors;
    const std::optional<MemoryClass> covered =
        node_.Covered(Address(t->data), ggml_nbytes(t), owner_);
    if (covered) {
      ++coverage_.by_class.at(static_cast<std::size_t>(*covered));
    }
    if (covered != memory_class) {
      if (coverage_.violations++ == 0) {
        coverage_.first =
            std::format("{} ({} bytes at {:#x})", t->name, ggml_nbytes(t), Address(t->data));
      }
    }
  };
  const auto kind_of = [&](const ggml_tensor* t) {
    const ggml_tensor* base = t->view_src != nullptr ? t->view_src : t;
    for (const auto& layer : graph.layers) {
      if (base == layer.k_cache || base == layer.v_cache) {
        return MemoryClass::kLiveState;
      }
    }
    // The other leaves are the weights; everything computed, and the
    // inputs, live in the activations.
    const auto inputs = graph.inputs();
    return base->op == GGML_OP_NONE && std::ranges::find(inputs, base) == inputs.end()
               ? MemoryClass::kWeights
               : MemoryClass::kScratch;
  };
  for (const ggml_tensor* node : graph.nodes) {
    expect(node, kind_of(node));
    for (const ggml_tensor* src : node->src) {
      if (src != nullptr) {
        expect(src, kind_of(src));
      }
    }
  }
}

Status Fp16Runner::Job(const catalog::Closure& closure, sc::DeviceJob job, std::string_view what) {
  return node_.Job(closure, std::move(job), what, stream_);
}

// ------------------------------------------------------------------ setup

Status Fp16Runner::Setup() {
  auto trajectory = LoadTrajectory(o_.trajectory, o_.tokens);
  if (!trajectory) {
    return std::unexpected(trajectory.error());
  }
  t_ = std::move(*trajectory);
  auto artifact = artifact::Artifact::Open(o_.artifact);
  if (!artifact) {
    return Error(std::format("the artifact was refused: {}", artifact.error().reason));
  }
  artifact_ = std::make_unique<artifact::Artifact>(std::move(*artifact));
  auto binding = model::BindQwen2(profile_, *artifact_);
  if (!binding) {
    return std::unexpected(binding.error());
  }
  binding_ = std::move(*binding);
  for (std::uint32_t s = 0; s < artifact_->shards().size(); ++s) {
    auto fd = artifact_->OpenShardForDirectRead(s);
    if (!fd) {
      return Error(std::format("shard {} cannot be opened for direct reads", s));
    }
    shards_.push_back(std::move(*fd));
  }

  // Measure every chunk's activations, scratch and inputs on a context with
  // no workspace, as rung 3 does; the node sizes the workspace for the
  // largest.
  auto measure = kg::LaunchContext::Create(0, node_.execution(), node_.stream(stream_),
                                           {.base = 0, .size = Bytes(0)});
  if (!measure) {
    return Error(measure.error().detail);
  }
  const std::uint64_t layer_cache = std::uint64_t{profile_.kv_width()} * t_.cells * 2;
  const std::uint64_t kv_bytes = layer_cache * 2 * profile_.layers;
  kv_used_ = kv_bytes;
  {
    const kg::DeviceChoices choices = kg::DeviceChoicesOf(**measure);
    const ChunkMemory placeless{.weight = [](std::uint32_t) { return std::uint64_t{1} << 44U; },
                                .kv = std::uint64_t{1} << 45U,
                                .activations = 0,
                                .activation_bytes = 0};
    std::uint32_t n_past = 0;
    for (const std::uint32_t rows : t_.chunks) {
      const std::uint32_t n_kv = model::PaddedKv(n_past + rows, t_.cells);
      auto planned =
          PlanChunk(profile_, binding_, placeless, t_.cells, rows, n_kv, o_.fusion, choices);
      if (!planned) {
        return Error(std::format("chunk at {}: {}", n_past, planned.error()));
      }
      most_activations_ = std::max(most_activations_, planned->placement.extent);
      auto scratch = kg::PlanScratch(**measure, planned->plan);
      if (!scratch) {
        return Error(scratch.error().detail);
      }
      most_scratch_ = std::max(most_scratch_, *scratch);
      std::uint64_t inputs = 0;
      for (const ggml_tensor* input : planned->graph.inputs()) {
        inputs += RoundUp(ggml_nbytes(input), 128);
      }
      input_bytes_ = std::max(input_bytes_, inputs);
      most_rows_ = std::max(most_rows_, rows);
      n_past += rows;
    }
  }
  measure->reset();
  int major = 0;
  int minor = 0;
  (void)cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  (void)cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
  cublas_bytes_ = kg::CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor)).value();
  if (auto r = node_.MapResident(kv_, "the FP16 cache", kv_bytes, BackingKind::kDevice,
                                 MemoryClass::kLiveState, Recovery::kPreserve, owner_);
      !r) {
    return r;
  }
  if (auto r =
          node_.MapResident(workspace_, "the cuBLAS workspace", cublas_bytes_, BackingKind::kDevice,
                            MemoryClass::kRuntime, Recovery::kPinned, owner_);
      !r) {
    return r;
  }
  logits_bytes_ = std::uint64_t{most_rows_} * profile_.vocab * sizeof(float);
  auto inputs = node_.Pinned(input_bytes_, owner_, staging_);
  auto logits = node_.Pinned(logits_bytes_, owner_, staging_);
  if (!inputs || !logits) {
    return std::unexpected(!inputs ? inputs.error() : logits.error());
  }
  inputs_ = *inputs;
  logits_ = *logits;
  if (o_.shared_embeddings) {
    rows_bytes_ = std::uint64_t{most_rows_} * profile_.width * 2;
    auto rows = node_.Pinned(rows_bytes_, owner_, staging_);
    if (!rows) {
      return std::unexpected(rows.error());
    }
    rows_ = *rows;
  }
  return ReserveWeights();
}

Status Fp16Runner::Bind() {
  auto& catalog = node_.catalog();
  if (o_.shared_embeddings) {
    // What a gather leases: the device table's chunks and the row staging.
    std::vector<ExtentId> gather =
        chunk_extents_.at(artifact_->resources()[binding_.token_embd].group);
    gather.push_back(staging_.back());
    gather_ = catalog.ClosureOfExtents(gather).value();
  }
  // What a chunk leases: every weight (both copies of the table), the
  // cache, the activations, the scratch, the workspace and the staging.
  std::vector<ExtentId> all = weights();
  for (const ts::Mapped* mapped : std::initializer_list<const ts::Mapped*>{
           &kv_, &node_.activations(), &node_.pool(), &workspace_}) {
    all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
  }
  all.insert(all.end(), staging_.begin(), staging_.end());
  everything_ = catalog.ClosureOfExtents(all).value();
  cache_ = catalog.ClosureOfExtents(kv_.extents).value();

  auto launch =
      kg::LaunchContext::Create(0, node_.execution(), node_.stream(stream_),
                                {.base = node_.pool().base, .size = Bytes(most_scratch_)});
  if (!launch) {
    return Error(launch.error().detail);
  }
  launch_ = std::move(*launch);
  auto registry = execution::Registry::Create(kg::Implementations());
  if (!registry) {
    return Error(registry.error().detail);
  }
  registry_ = std::make_unique<execution::Registry>(std::move(*registry));
  return {};
}

// ------------------------------------------------------------------ evaluating

Status Fp16Runner::Evaluate(int evaluation, std::vector<float>& result) {
  // A fresh cache, as the bridge's new context has: a job on the compute
  // stream, leasing the cache.
  const std::uint64_t kv_bytes = kv_used_;
  const std::uint64_t kv = kv_.base;
  if (auto cleared = Job(
          cache_,
          [kv, kv_bytes](providers::NativeStream stream) {
            return cudaMemsetAsync(Pointer(kv), 0, kv_bytes,
                                   static_cast<cudaStream_t>(stream.handle)) == cudaSuccess
                       ? sc::JobResult::kQueued
                       : sc::JobResult::kUnknown;
          },
          "clearing the cache");
      !cleared) {
    return cleared;
  }
  const ts::Mapped& activations = node_.activations();
  result.clear();
  result.reserve(t_.tokens.size() * profile_.vocab);
  std::uint32_t n_past = 0;
  for (std::size_t k = 0; k < t_.chunks.size(); ++k) {
    const std::uint32_t rows = t_.chunks[k];
    auto inputs = model::Qwen2ChunkInputs(profile_, t_.cells, n_past, rows);
    if (!inputs) {
      return std::unexpected(inputs.error());
    }
    const ChunkMemory memory{
        .weight = [this](std::uint32_t resource) { return WeightAddress(resource); },
        .kv = kv_.base,
        .activations = activations.base,
        .activation_bytes = most_activations_};
    auto planned = PlanChunk(profile_, binding_, memory, t_.cells, rows, inputs->n_kv, o_.fusion,
                             kg::DeviceChoicesOf(*launch_));
    if (!planned) {
      return Error(std::format("chunk {}: {}", k, planned.error()));
    }
    const bool calls_cublas = std::ranges::any_of(
        planned->plan.steps, [](const auto& s) { return s.implementation == kg::kMulMatCublas; });
    if (calls_cublas && !cublas_) {
      // The bridge creates its handle here, lazily: a job on the compute
      // stream, leasing the workspace and scratch it lends.
      std::vector<ExtentId> lent = workspace_.extents;
      lent.insert(lent.end(), node_.pool().extents.begin(), node_.pool().extents.end());
      Status made;
      if (auto created = Job(
              node_.catalog().ClosureOfExtents(lent).value(),
              [this, &made](providers::NativeStream) {
                launch_.reset();
                auto handle = kg::CublasHandle::Create(
                    0, node_.execution(), node_.stream(stream_),
                    {.base = workspace_.base, .size = Bytes(cublas_bytes_)});
                if (!handle) {
                  made = Error(handle.error().detail);
                  return sc::JobResult::kNotStarted;
                }
                cublas_ = std::move(*handle);
                auto relaunch = kg::LaunchContext::Create(
                    0, node_.execution(), node_.stream(stream_),
                    {.base = node_.pool().base, .size = Bytes(most_scratch_)}, cublas_.get());
                if (!relaunch) {
                  made = Error(relaunch.error().detail);
                  return sc::JobResult::kNotStarted;
                }
                launch_ = std::move(*relaunch);
                return sc::JobResult::kQueued;
              },
              "creating the cuBLAS handle");
          !created || !made) {
        return !made ? made : created;
      }
      // The plan's device choices follow the new context.
      planned = PlanChunk(profile_, binding_, memory, t_.cells, rows, inputs->n_kv, o_.fusion,
                          kg::DeviceChoicesOf(*launch_));
      if (!planned) {
        return Error(std::format("chunk {}: {}", k, planned.error()));
      }
    }
    auto bound = kg::BoundGraph::Bind(*registry_, planned->plan);
    if (!bound) {
      return Error(std::format("chunk {}: {}", k, bound.error().detail));
    }
    auto step_scratch = kg::PlanScratch(*launch_, planned->plan);
    if (!step_scratch) {
      return Error(step_scratch.error().detail);
    }
    Check(planned->graph);
    const std::uint64_t chunk_logits = std::uint64_t{rows} * profile_.vocab * sizeof(float);
    Status ran;
    const std::span<const std::int32_t> tokens = std::span(t_.tokens).subspan(n_past, rows);
    std::span<const std::uint16_t> host_table;
    std::vector<float> gathered;
    if (o_.shared_embeddings) {
      // BP-P3's shared arm: the rows come from the device table.
      if (auto r = GatherRows(tokens, gathered); !r) {
        return r;
      }
    } else {
      const auto& table = artifact_->resources()[binding_.token_embd];
      host_table =
          std::span(reinterpret_cast<const std::uint16_t*>(  // NOLINT(performance-no-int-to-ptr)
                        table_base_ + table.offset.value()),
                    std::size_t{profile_.vocab} * profile_.width);
    }
    launch_->ResetScratchPeak();
    auto job = [&, rows, chunk = static_cast<int>(k),
                n_past](providers::NativeStream native) -> sc::JobResult {
      if (recording_ != nullptr) {
        for (const auto& event : recording_->Take()) {
          record_ += test_support::EventLine(event);
        }
        record_ += test_support::ChunkLine(
            {.evaluation = evaluation, .chunk = chunk, .rows = rows, .n_past = n_past});
      }
      // The embedding rows, from the host table this job's lease holds (or
      // gathered from the device table before it).
      std::vector<float> embd = gathered;
      if (!o_.shared_embeddings) {
        embd.assign(std::size_t{rows} * profile_.width, 0.0F);
        if (auto r = model::EmbedRows(host_table, profile_.width, profile_.vocab, tokens, embd);
            !r) {
          ran = std::unexpected(r.error());
          return sc::JobResult::kNotStarted;
        }
      }
      auto* const stream = static_cast<cudaStream_t>(native.handle);
      const kg::Qwen2Graph& g = planned->graph;
      const std::array<std::pair<ggml_tensor*, const void*>, 6> sources = {
          {{g.embd, embd.data()},
           {g.positions, inputs->positions.data()},
           {g.k_idxs, inputs->k_idxs.data()},
           {g.v_idxs, inputs->v_idxs.data()},
           {g.mask, inputs->mask.data()},
           {g.out_ids, inputs->out_ids.data()}}};
      std::uint64_t staged = 0;
      for (const auto& [tensor, source] : sources) {
        auto* at = static_cast<std::byte*>(inputs_) + staged;
        std::memcpy(at, source, ggml_nbytes(tensor));
        if (auto r = Cuda(cudaMemcpyAsync(tensor->data, at, ggml_nbytes(tensor),
                                          cudaMemcpyHostToDevice, stream),
                          "an input copy");
            !r) {
          ran = r;
          return sc::JobResult::kUnknown;  // a runtime error, even the first: its effect is unknown
        }
        staged += RoundUp(ggml_nbytes(tensor), 128);
      }
      if (auto r = bound->Run(*launch_); !r) {
        ran = Error(std::format("chunk {}: {}", chunk, r.error().detail));
        return r.error().error == kg::KernelError::kUnknown ? sc::JobResult::kUnknown
                                                            : sc::JobResult::kFailed;
      }
      if (auto r = Cuda(cudaMemcpyAsync(logits_, g.logits->data, chunk_logits,
                                        cudaMemcpyDeviceToHost, stream),
                        "the logits copy");
          !r) {
        ran = r;
        return sc::JobResult::kUnknown;
      }
      if (recording_ != nullptr) {
        for (const auto& event : recording_->Take()) {
          record_ += test_support::EventLine(event);
        }
        record_ += test_support::EndChunkLine();
      }
      return sc::JobResult::kQueued;
    };
    if (auto r = Job(everything_, std::move(job), "a chunk"); !r || !ran) {
      return !ran ? ran : Error(std::format("chunk {}: {}", k, r.error()));
    }
    if (launch_->faulted()) {
      return Error(std::format("chunk {}: the launch context faulted", k));
    }
    // The job's fence has completed: the logits are in.
    const auto* values = static_cast<const float*>(logits_);
    result.insert(result.end(), values, values + (chunk_logits / sizeof(float)));
    if (k == 0) {
      first_chunk_done_ = std::chrono::steady_clock::now();
    }
    // The chunk shape's guaranteed bound against what it reached.
    Peak& peak = peaks_[rows];
    ++peak.chunks;
    peak.activations_bound = std::max(peak.activations_bound, planned->placement.extent);
    peak.scratch_bound = std::max(peak.scratch_bound, *step_scratch);
    peak.scratch_seen = std::max(peak.scratch_seen, launch_->scratch_peak().value());
    const auto reach = [&](const ggml_tensor* t) {
      const std::uint64_t at = t != nullptr ? Address(t->data) : 0;
      if (at >= activations.base && at < activations.base + activations.bytes) {
        peak.activations_seen =
            std::max(peak.activations_seen, at + ggml_nbytes(t) - activations.base);
      }
    };
    for (const ggml_tensor* node : planned->graph.nodes) {
      reach(node);
      for (const ggml_tensor* src : node->src) {
        reach(src);
      }
    }
    n_past += rows;
    const int partial_at = o_.partial ? 3 + o_.restores : -1;
    const int spill_at = o_.spill.empty() ? -1 : 3 + o_.restores + (o_.partial ? 1 : 0);
    if (evaluation > 2 && evaluation <= 2 + o_.restores && n_past == t_.restore_after) {
      if (auto r = Restore(evaluation); !r) {
        return r;
      }
    }
    if (evaluation == partial_at && n_past == t_.restore_after) {
      if (auto r = Partial(); !r) {
        return r;
      }
    }
    if (evaluation == spill_at && (k == 0 || n_past == t_.restore_after + 8)) {
      if (auto r = Spill(k == 0 ? "after the first prefill chunk" : "mid-decode"); !r) {
        return r;
      }
    }
  }
  return {};
}

// BP-P2: each partial eviction in turn, at the restore point.
Status Fp16Runner::Partial() {
  auto& memory = node_.memory();
  auto& catalog = node_.catalog();
  const std::string layer =
      std::format("blk.{}.", profile_.layers / 2);  // a middle layer's resources
  const auto cases = PartialCases(*artifact_, layer, false);
  for (const auto& partial : cases) {
    std::vector<ExtentId> extents;
    extents.reserve(partial.chunks.size());
    for (const auto& [group, chunk] : partial.chunks) {
      extents.push_back(chunk_extents_.at(group).at(chunk));
    }
    const std::size_t backings = memory.backings();
    const auto start = std::chrono::steady_clock::now();
    if (auto r = node_.Evict(extents); !r) {
      return r;
    }
    const double evicted = Seconds(std::chrono::steady_clock::now() - start);
    std::size_t resident = 0;
    for (const ExtentId extent : device_weights_) {
      resident += catalog.Describe(extent).value().state == catalog::ExtentState::kResident ? 1 : 0;
    }
    if (resident + extents.size() != device_weights_.size() ||
        memory.backings() + extents.size() != backings) {
      return Error(std::format("partial eviction '{}' did not evict exactly its {} extents",
                               partial.name, extents.size()));
    }
    // An incomplete closure refuses the launch: nothing may run.
    std::atomic<bool> ran{false};
    ts::Done refused;
    const Status submitted =
        node_.Post(std::make_unique<ts::LaunchOnlyProgram>(refused, everything_, ran, stream_),
                   refused, "a launch over an incomplete closure");
    if (submitted || ran.load() ||
        refused.error.load() != static_cast<int>(sc::WorkError::kNotResident)) {
      return Error(
          std::format("partial eviction '{}': a launch over an incomplete closure was "
                      "not refused before it ran",
                      partial.name));
    }
    ++refusals_;
    events_.push_back(PagingEvent{.what = "partial eviction",
                                  .extents = extents.size(),
                                  .seconds = evicted,
                                  .detail = partial.name});
    // Only the missing extents page in; the next chunk binds against them.
    if (auto r = node_.Load(extents, std::format("partial restore: {}", partial.name), loads_);
        !r) {
      return r;
    }
  }
  return {};
}

// BP-P4: the cache written back through the zone and evicted, then
// restored; its bytes before and after must match.
Status Fp16Runner::Spill(std::string_view where) {
  auto& memory = node_.memory();
  std::vector<std::byte> before;
  if (auto r = Snapshot(before); !r) {
    return r;
  }
  const std::size_t backings = memory.backings();
  auto start = std::chrono::steady_clock::now();
  if (auto r = node_.Evict(kv_.extents); !r) {
    return r;
  }
  events_.push_back(PagingEvent{.what = "state write-back",
                                .extents = kv_.extents.size(),
                                .seconds = Seconds(std::chrono::steady_clock::now() - start),
                                .detail = std::string(where)});
  for (const ExtentId extent : kv_.extents) {
    const auto view = node_.catalog().Describe(extent).value();
    if (view.state != catalog::ExtentState::kNonresident || !view.preserved) {
      return Error("the cache is not nonresident and preserved after its write-back");
    }
  }
  const bool managed = o_.spill == "managed";
  if (memory.backings() + (managed ? kv_.extents.size() : 0) != backings) {
    return Error("the write-back did not release (managed) or keep (premapped) the backing");
  }
  if (!managed) {
    // Poisoned while nonresident: the restore must bring back every byte.
    if (auto r = Cuda(cudaMemset(Pointer(kv_.base), 0xff, kv_.bytes), "poisoning the cache"); !r) {
      return r;
    }
    if (auto r = Cuda(cudaDeviceSynchronize(), "poisoning the cache"); !r) {
      return r;
    }
  }
  start = std::chrono::steady_clock::now();
  if (auto r = node_.Load(kv_.extents, std::format("state restore ({})", where), loads_); !r) {
    return r;
  }
  events_.push_back(PagingEvent{.what = "state restore",
                                .extents = kv_.extents.size(),
                                .seconds = Seconds(std::chrono::steady_clock::now() - start),
                                .detail = std::string(where)});
  std::vector<std::byte> after;
  if (auto r = Snapshot(after); !r) {
    return r;
  }
  for (std::size_t i = 0; i < before.size(); ++i) {
    kv_mismatches_ += before[i] != after.at(i) ? 1 : 0;
  }
  return {};
}

// The cache's bytes, copied to the host by a job that leases it.
Status Fp16Runner::Snapshot(std::vector<std::byte>& out) {
  const std::uint64_t kv = kv_.base;
  const std::uint64_t bytes = kv_used_;
  void* copy = kv_copy_;
  if (auto r = Job(
          cache_,
          [kv, bytes, copy](providers::NativeStream stream) {
            return cudaMemcpyAsync(copy, Pointer(kv), bytes, cudaMemcpyDeviceToHost,
                                   static_cast<cudaStream_t>(stream.handle)) == cudaSuccess
                       ? sc::JobResult::kQueued
                       : sc::JobResult::kUnknown;
          },
          "copying the cache out");
      !r) {
    return r;
  }
  const auto* bytes_in = static_cast<const std::byte*>(kv_copy_);
  out.assign(bytes_in, bytes_in + bytes);
  return {};
}

// BP-P3's shared arm: a chunk's embedding rows copied from the device
// table to pinned staging by a job leasing both, then widened on the host
// as the bridge's CPU lookup widens them.
Status Fp16Runner::GatherRows(std::span<const std::int32_t> tokens, std::vector<float>& embd) {
  const std::uint64_t row_bytes = std::uint64_t{profile_.width} * 2;
  const std::uint64_t table = WeightAddress(binding_.token_embd);
  for (const std::int32_t token : tokens) {
    if (token < 0 || std::cmp_greater_equal(token, profile_.vocab)) {
      return Error(std::format("token {} is outside the vocabulary", token));
    }
  }
  void* rows = rows_;
  if (auto r = Job(
          gather_,
          [&, rows](providers::NativeStream stream) {
            for (std::size_t i = 0; i < tokens.size(); ++i) {
              if (cudaMemcpyAsync(
                      static_cast<std::byte*>(rows) + (i * row_bytes),
                      Pointer(table + (static_cast<std::uint64_t>(tokens[i]) * row_bytes)),
                      row_bytes, cudaMemcpyDeviceToHost,
                      static_cast<cudaStream_t>(stream.handle)) != cudaSuccess) {
                return sc::JobResult::kUnknown;  // a runtime error, even the first
              }
            }
            return sc::JobResult::kQueued;
          },
          "gathering the embedding rows");
      !r) {
    return r;
  }
  std::vector<std::int32_t> local(tokens.size());
  for (std::size_t i = 0; i < local.size(); ++i) {
    local[i] = static_cast<std::int32_t>(i);
  }
  embd.assign(tokens.size() * profile_.width, 0.0F);
  return model::EmbedRows(
      std::span(static_cast<const std::uint16_t*>(rows_), tokens.size() * profile_.width),
      profile_.width, static_cast<std::uint32_t>(tokens.size()), local, embd);
}

// --spill: the cache's write-back places, one 2 MiB range of an unnamed
// direct-I/O file per extent, with its backing managed (released on
// eviction) or kept mapped (premapped).
Status Fp16Runner::RegisterCache() {
  std::filesystem::create_directories(o_.out);
  spill_fd_ = ::open(o_.out.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (spill_fd_ < 0) {
    return Error(std::format("the spill file in {}: {}", o_.out.string(),
                             std::generic_category().message(errno)));
  }
  if (auto r = Cuda(cudaMallocHost(&kv_copy_, kv_used_), "the cache's host copy"); !r) {
    return r;
  }
  const bool managed = o_.spill == "managed";
  for (std::size_t i = 0; i < kv_.extents.size(); ++i) {
    std::optional<sc::BackingPlace> backing;
    if (managed) {
      backing = sc::BackingPlace{.reservation = kv_.reservation,
                                 .offset = Bytes(i * kExtent),
                                 .size = Bytes(kExtent),
                                 .allocation_class = node_.device_class()};
    }
    auto set = node_.scheduler().SetSource(
        kv_.extents[i],
        sc::PageSource{
            .read = {.fd = spill_fd_, .offset = i * kExtent, .memory = nullptr, .length = kExtent},
            .landed = true,
            .destination = kv_.base + (i * kExtent),
            .backing = backing,
            .write_back = true});
    if (!set) {
      return Error(std::format("the cache's write-back place: {}", sc::ToString(set.error())));
    }
  }
  if (managed) {
    kv_.backings.clear();  // the VMM lane releases them on eviction (D-033)
  }
  return {};
}

// Rung 5: every weight extent evicted, its backing released, then paged
// back in through the zone (and, with relocate, every second time at the
// other place).
Status Fp16Runner::Restore(int evaluation) {
  auto& memory = node_.memory();
  const std::vector<ExtentId> all = weights();
  const std::size_t backings = memory.backings();
  if (auto r = node_.Evict(all); !r) {
    return r;
  }
  for (const ExtentId extent : all) {
    if (node_.catalog().Describe(extent).value().state != catalog::ExtentState::kNonresident) {
      return Error("a weight extent is still resident after the eviction");
    }
  }
  if (memory.backings() + all.size() != backings) {
    return Error("the eviction did not release every weight's backing");
  }
  if (o_.relocate && evaluation % 2 == 0) {
    // On the scheduler thread: it owns the sources.
    if (auto r = node_.Call([this] { return Place(1 - place_); }, "relocating the weights"); !r) {
      return r;
    }
  }
  if (auto r =
          node_.Load(device_weights_, std::format("restore {} (device)", evaluation - 2), loads_);
      !r) {
    return r;
  }
  return node_.Load(table_extents_, std::format("restore {} (host table)", evaluation - 2), loads_);
}

Status Fp16Runner::RunAlone() {
  if (o_.load_only > 0) {
    for (int i = 1; i <= o_.load_only; ++i) {
      if (auto r = node_.Load(device_weights_, std::format("load {}", i), loads_); !r) {
        return r;
      }
      if (auto r = node_.Evict(device_weights_); !r) {
        return r;
      }
    }
    return WriteLoads();
  }
  if (auto r = node_.Load(device_weights_, "initial (device)", loads_); !r) {
    return r;
  }
  if (auto r = node_.Load(table_extents_, "initial (host table)", loads_); !r) {
    return r;
  }
  const int evaluations = 2 + o_.restores + (o_.partial ? 1 : 0) + (o_.spill.empty() ? 0 : 1);
  std::vector<std::vector<float>> results(static_cast<std::size_t>(evaluations));
  for (int e = 1; e <= evaluations; ++e) {
    if (auto r = Evaluate(e, results[static_cast<std::size_t>(e - 1)]); !r) {
      return r;
    }
  }
  return Write(results);
}

Status Fp16Runner::WriteLoads() {
  const std::string loads = LoadsJson(loads_);
  std::uint64_t bytes = 0;
  for (const auto& group : artifact_->groups()) {
    bytes += group.stored.value();
  }
  std::filesystem::create_directories(o_.out);
  std::ofstream file(o_.out / "loads.json");
  file << std::format(
              R"({{"weights":"{}","backing":"{}","lanes":"{}","extents":{},"read_bytes":{},)"
              R"("zone_slots":{},"depth":{},"coalesce":{},"loads":[{}]}})",
              o_.weights_host ? "host" : "device", o_.premapped ? "premapped" : "managed",
              node_.inline_lanes() ? "inline" : "threads", device_weights_.size(), bytes,
              node_.slots(), test_support::kPagedDepth, node_.coalesce() ? "true" : "false", loads)
       << "\n";
  std::println("wrote {}", o_.out.string());
  return {};
}

Status Fp16Runner::Write(const std::vector<std::vector<float>>& results) {
  const std::vector<float>& first = results.front();
  std::string differences;
  std::size_t differing_total = 0;
  for (std::size_t e = 1; e < results.size(); ++e) {
    std::size_t differing = results[e].size() == first.size() ? 0 : first.size();
    for (std::size_t i = 0; i < std::min(first.size(), results[e].size()); ++i) {
      differing +=
          std::bit_cast<std::uint32_t>(first[i]) != std::bit_cast<std::uint32_t>(results[e][i]);
    }
    differences += std::format("{}{}", differences.empty() ? "" : ",", differing);
    differing_total += differing;
  }
  base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(first)));
  const std::string digest = base::ToHex(hash.Finish());
  std::filesystem::create_directories(o_.out);
  {
    std::ofstream raw(o_.out / "logits.f32le", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(first.data()),
              static_cast<std::streamsize>(first.size() * sizeof(float)));
    if (!raw) {
      return Error("the logits could not be written");
    }
  }
  const std::string loads = LoadsJson(loads_);
  std::string by_class;
  for (std::size_t c = 0; c < coverage_.by_class.size(); ++c) {
    by_class += std::format("{}{}", c == 0 ? "" : ",", coverage_.by_class.at(c));
  }
  std::string chunks;
  for (const std::uint32_t c : t_.chunks) {
    chunks += std::format("{}{}", chunks.empty() ? "" : ",", c);
  }
  std::uint64_t device_bytes = 0;
  std::uint64_t table_bytes = 0;
  const auto groups = artifact_->groups();
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    device_bytes += groups[g].stored.value();
    if (g == artifact_->resources()[binding_.token_embd].group) {
      table_bytes = groups[g].stored.value();
    }
  }
  const std::string summary = std::format(
      R"({{"trajectory":"{}","fusion":{},"lanes":"{}","tokens":{},"chunks":[{}],"cells":{},)"
      R"("evaluations":{},"restores":{},"restore_after":{},"relocate":{},"logits_sha256":"{}",)"
      R"("bit_differences_from_first":[{}],"artifact":"{}","device_weight_extents":{},)"
      R"("device_weight_read_bytes":{},"host_table_extents":{},"host_table_read_bytes":{},)"
      R"("weights_backing_bytes":{},"stored_bytes":{},"zone_slots":{},"zone_bytes":{},)"
      R"("kv_bytes":{},"activations":{},"scratch":{},"cublas_workspace":{},"loads":[{}],)"
      R"("coverage":{{"tensors":{},"violations":{},"first":"{}","by_class":[{}]}}}})",
      t_.name, o_.fusion ? "true" : "false", node_.inline_lanes() ? "inline" : "threads",
      t_.tokens.size(), chunks, t_.cells, results.size(), o_.restores, t_.restore_after,
      o_.relocate ? "true" : "false", digest, differences, artifact_->id(), device_weights_.size(),
      device_bytes, table_extents_.size(), table_bytes, device_weights_.size() * kExtent,
      stored_bytes_, node_.slots(), node_.zone().bytes, kv_.bytes, node_.activations().bytes,
      node_.pool().bytes, workspace_.bytes, loads, coverage_.tensors, coverage_.violations,
      coverage_.first, by_class);
  // BP-P2, BP-P4 and BP-P3's runs, and each chunk shape's bound against
  // its peak.
  std::string events;
  for (const PagingEvent& event : events_) {
    events += std::format(R"({}{{"what":"{}","extents":{},"seconds":{:.6f},"detail":"{}"}})",
                          events.empty() ? "" : ",", event.what, event.extents, event.seconds,
                          event.detail);
  }
  std::string peaks;
  bool within = true;
  for (const auto& [rows, peak] : peaks_) {
    within = within && peak.activations_seen <= peak.activations_bound &&
             peak.scratch_seen <= peak.scratch_bound;
    peaks +=
        std::format(R"({}{{"rows":{},"chunks":{},"activations_bound":{},"activations_seen":{},)"
                    R"("scratch_bound":{},"scratch_seen":{}}})",
                    peaks.empty() ? "" : ",", rows, peak.chunks, peak.activations_bound,
                    peak.activations_seen, peak.scratch_bound, peak.scratch_seen);
  }
  {
    std::ofstream file(o_.out / "paging.json");
    file << std::format(R"({{"partial":{},"spill":"{}","embeddings":"{}","refused_launches":{},)"
                        R"("kv_bytes_differing":{},"events":[{}],"peaks":[{}]}})",
                        o_.partial ? "true" : "false", o_.spill,
                        o_.shared_embeddings ? "shared" : "duplicated", refusals_, kv_mismatches_,
                        events, peaks)
         << "\n";
  }
  {
    std::ofstream file(o_.out / "summary.json");
    file << summary << "\n";
  }
  std::println("wrote {}", o_.out.string());
  if (kv_mismatches_ != 0) {
    return Error(std::format("{} bytes of the cache differ after its restore", kv_mismatches_));
  }
  if (!within) {
    return Error("a chunk reached past its guaranteed bound (paging.json)");
  }
  if (coverage_.violations != 0) {
    return Error(
        std::format("{} bound tensors lie outside cataloged extents of their class; "
                    "first: {}",
                    coverage_.violations, coverage_.first));
  }
  if (differing_total != 0) {
    return Error(std::format("a later evaluation differs from the first: [{}]", differences));
  }
  return {};
}

Status Fp16Runner::Release() {
  if (released_) {
    return {};
  }
  released_ = true;
  std::vector<std::string> problems;
  launch_.reset();
  cublas_.reset();
  auto& memory = node_.memory();
  for (ts::Mapped* mapped : {&kv_, &workspace_}) {
    if (!ts::ReleaseMapped(memory, *mapped)) {
      problems.push_back(std::format("{} could not be released", mapped->name));
    }
  }
  if (!premapped_.empty()) {
    bool released =
        memory.Unmap(weights_[0], Bytes(0), Bytes(premapped_.size() * kExtent)).has_value();
    for (const auto backing : premapped_) {
      released = memory.Release(backing).has_value() && released;
    }
    if (!released) {
      problems.emplace_back("the premapped weights could not be released");
    }
  }
  for (const providers::ReservationId reservation : {weights_[0], weights_[1], table_}) {
    if (reservation.valid() && !memory.Free(reservation)) {
      problems.emplace_back("a weights reservation still has mappings");
    }
  }
  (void)cudaFreeHost(kv_copy_);
  if (spill_fd_ >= 0) {
    (void)::close(spill_fd_);  // unnamed: nothing outlives the process
  }
  if (problems.empty()) {
    return {};
  }
  std::string all;
  for (const std::string& problem : problems) {
    all += (all.empty() ? "" : "; ") + problem;
  }
  return Error(all);
}

}  // namespace llmp::benchmarks
