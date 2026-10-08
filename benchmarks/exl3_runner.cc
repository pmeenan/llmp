// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "exl3_runner.h"

#include <cuda.h>
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
#include <print>
#include <span>
#include <system_error>
#include <thread>
#include <tuple>

#include "artifact/layout.h"
#include "base/sha256.h"
#include "exl3_common.h"
#include "kernels/exl3/implementations.h"
#include "kernels/exl3/validate.h"
#include "kernels/ggml/implementations.h"
#include "paging_cases.h"
#include "plan_record.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"

namespace llmp::benchmarks {

namespace {

namespace exl3 = llmp::kernels::exl3;
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
// Where planning binds the region before the node maps the shared
// workspace: an aligned address nothing is mapped at (the pool's size
// depends on the shapes alone).
constexpr std::uint64_t kPlanningRegion = std::uint64_t{1} << 44U;

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

}  // namespace

std::vector<ExtentId> Exl3Runner::managed_extents() const {
  std::vector<ExtentId> all = device_weights_;
  if (o_.spill == "managed") {
    // Its backing is the VMM lane's to release: written back and evicted.
    all.insert(all.end(), kv_.extents.begin(), kv_.extents.end());
  }
  return all;
}

std::uint64_t Exl3Runner::WeightAddress(std::uint32_t resource) const {
  const auto& r = artifact_->resources()[resource];
  return weights_base_.at(place_) + group_region_[r.group] + r.offset.value();
}

exl3::Qwen2Linear Exl3Runner::Linear(const model::Exl3LinearBinding& l) const {
  return {.weights = {.trellis = WeightAddress(l.trellis),
                      .suh = WeightAddress(l.suh),
                      .svh = WeightAddress(l.svh),
                      .k = l.k,
                      .n = l.n,
                      .bits = l.bits},
          .bias = l.bias ? WeightAddress(*l.bias) : 0};
}

// The weights' addresses at the current place: every linear and the
// embedding.
void Exl3Runner::MapWeights() {
  memory_map_.embed = WeightAddress(binding_.embed);
  memory_map_.lm_head = Linear(binding_.lm_head);
  for (std::uint32_t l = 0; l < profile_.layers; ++l) {
    const model::Exl3LayerBinding& b = binding_.layers[l];
    exl3::Qwen2Layer& layer = memory_map_.layers[l];
    layer.q = Linear(b.q);
    layer.k = Linear(b.k);
    layer.v = Linear(b.v);
    layer.o = Linear(b.o);
    layer.gate = Linear(b.gate);
    layer.up = Linear(b.up);
    layer.down = Linear(b.down);
  }
}

Status Exl3Runner::ReserveWeights() {
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
    auto reservation = memory.Reserve(Bytes(weights_bytes_));
    if (!reservation) {
      return Error(std::format("reserving the weights: {}", reservation.error().detail));
    }
    weights_.at(which) = *reservation;
    weights_base_.at(which) = memory.RangeOf(*reservation).value().base;
  }
  std::array<std::uint8_t, 32> id{};
  for (std::size_t i = 0; i < id.size() && (2 * i) + 1 < artifact_->id().size(); ++i) {
    (void)std::from_chars(artifact_->id().data() + (2 * i), artifact_->id().data() + (2 * i) + 2,
                          id.at(i), 16);
  }
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      auto extent =
          node_.catalog().AddExtent({.domain = node_.domain(),
                                     .memory_class = MemoryClass::kWeights,
                                     .recovery = Recovery::kFromArtifact,
                                     .size = Bytes(kExtent),
                                     .content = {.artifact = id, .group = g, .chunk = c}});
      if (!extent) {
        return Error("cataloging a weight chunk");
      }
      chunk_extents_[g].push_back(*extent);
      device_weights_.push_back(*extent);
    }
  }
  return {};
}

Status Exl3Runner::Register() {
  if (auto r = Place(0); !r) {
    return r;
  }
  if (!o_.spill.empty()) {
    return RegisterCache();
  }
  return {};
}

// Registers every weight chunk's source at place `which`.
Status Exl3Runner::Place(std::size_t which) {
  const auto groups = artifact_->groups();
  node_.EraseSpans([&](const ts::Span& span) {
    return span.owner == owner_ && span.memory_class == MemoryClass::kWeights && span.device &&
           std::ranges::find(device_weights_, span.extent) != device_weights_.end();
  });
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      const auto range = artifact::ChunkRangeOf(artifact_->layout(), {.group = g, .chunk = c});
      if (!range) {
        return Error("a chunk's range");
      }
      const std::uint64_t offset = group_region_[g] + (std::uint64_t{c} * kExtent);
      const std::uint64_t address = weights_base_.at(which) + offset;
      auto set = node_.scheduler().SetSource(
          chunk_extents_[g][c],
          sc::PageSource{.read = {.fd = shards_.at(range->shard).get(),
                                  .offset = range->file_offset.value(),
                                  .memory = nullptr,
                                  .length = range->length.value()},
                         .landed = true,
                         .destination = address,
                         .backing = sc::BackingPlace{.reservation = weights_.at(which),
                                                     .offset = Bytes(offset),
                                                     .size = Bytes(kExtent),
                                                     .allocation_class = node_.device_class()}});
      if (!set) {
        return Error(std::format("a chunk's source: {}", sc::ToString(set.error())));
      }
      node_.AddSpan({.base = address,
                     .size = kExtent,
                     .extent = chunk_extents_[g][c],
                     .memory_class = MemoryClass::kWeights,
                     .device = true,
                     .owner = owner_});
    }
  }
  node_.SortSpans();
  place_ = which;
  return {};
}

void Exl3Runner::Expect(std::string_view what, std::uint64_t address, std::uint64_t bytes,
                        MemoryClass memory_class) {
  ++coverage_.ranges;
  const std::optional<MemoryClass> covered = node_.Covered(address, bytes, owner_);
  if (covered) {
    ++coverage_.by_class.at(static_cast<std::size_t>(*covered));
  }
  if (covered != memory_class && coverage_.violations++ == 0) {
    coverage_.first = std::format("{} ({} bytes at {:#x})", what, bytes, address);
  }
}

// BP-A1's in-process check: every range the bound program reads or writes
// lies in cataloged, resident extents of device memory of its class.
void Exl3Runner::Check(const exl3::Qwen2Program& program, const PhaseRun& phase) {
  const model::Exl3PhasePlan& plan = phase.plan;
  for (std::size_t op = 0; op < plan.ops.size(); ++op) {
    const model::Exl3Op& o = plan.ops[op];
    for (const auto& names : {o.inputs, o.outputs}) {
      for (const std::string& name : names) {
        if (name == "k_cache" || name == "v_cache") {
          const std::uint64_t cells = o.name.starts_with("kv_write")
                                          ? static_cast<std::uint64_t>(plan.phase.rows)
                                          : static_cast<std::uint64_t>(plan.padded);
          Expect(name, program.Address(op, name), cells * profile_.kv_width() * 2,
                 MemoryClass::kLiveState);
        } else if (const auto t = plan.tensors.find(name); t != plan.tensors.end()) {
          Expect(name, program.Address(op, name), t->second.bytes(), MemoryClass::kScratch);
        }
      }
    }
  }
  const auto weights = [&](const exl3::Qwen2Linear& l) {
    Expect("trellis", l.weights.trellis, exl3::TrellisBytes(l.weights), MemoryClass::kWeights);
    Expect("suh", l.weights.suh, static_cast<std::uint64_t>(l.weights.k) * 2,
           MemoryClass::kWeights);
    Expect("svh", l.weights.svh, static_cast<std::uint64_t>(l.weights.n) * 2,
           MemoryClass::kWeights);
    if (l.bias != 0) {
      Expect("bias", l.bias, static_cast<std::uint64_t>(l.weights.n) * 2, MemoryClass::kWeights);
    }
  };
  Expect("embed", memory_map_.embed, std::uint64_t{profile_.vocab} * profile_.width * 2,
         MemoryClass::kWeights);
  Expect("final norm", memory_map_.final_norm, std::uint64_t{profile_.width} * 4,
         MemoryClass::kWeights);
  weights(memory_map_.lm_head);
  for (const exl3::Qwen2Layer& layer : memory_map_.layers) {
    for (const exl3::Qwen2Linear* l :
         {&layer.q, &layer.k, &layer.v, &layer.o, &layer.gate, &layer.up, &layer.down}) {
      weights(*l);
    }
    Expect("norm", layer.attn_norm, std::uint64_t{profile_.width} * 4, MemoryClass::kWeights);
    Expect("norm", layer.mlp_norm, std::uint64_t{profile_.width} * 4, MemoryClass::kWeights);
    Expect("tables", layer.trellis_table, 48, MemoryClass::kRuntime);
  }
}

Status Exl3Runner::Job(const catalog::Closure& closure, sc::DeviceJob job, std::string_view what) {
  return node_.Job(closure, std::move(job), what, stream_);
}

// ------------------------------------------------------------------ setup

Status Exl3Runner::Setup() {
  auto ids = LoadIds(o_.ids);
  if (!ids) {
    return std::unexpected(ids.error());
  }
  ids_ = std::move(*ids);
  auto artifact = artifact::Artifact::Open(o_.artifact);
  if (!artifact) {
    return Error(std::format("the artifact was refused: {}", artifact.error().reason));
  }
  artifact_ = std::make_unique<artifact::Artifact>(std::move(*artifact));
  auto binding = model::BindQwen2Exl3(profile_, *artifact_);
  if (!binding) {
    return Error("the artifact does not bind: " + binding.error());
  }
  binding_ = std::move(*binding);
  auto table = LoadTable(o_.plan, binding_);
  if (!table) {
    return std::unexpected(table.error());
  }
  table_.emplace(std::move(*table));
  for (std::uint32_t s = 0; s < artifact_->shards().size(); ++s) {
    auto fd = artifact_->OpenShardForDirectRead(s);
    if (!fd) {
      return Error(std::format("shard {} cannot be opened for direct reads", s));
    }
    shards_.push_back(std::move(*fd));
  }

  // Every phase, planned as rung 3 plans it.
  for (const int prefix : o_.prefixes) {
    if (prefix + kSuffix > static_cast<int>(ids_.size())) {
      return Error(std::format("prefix {} and its steps exceed the held-out IDs", prefix));
    }
    for (int index = 0; index <= kSuffix; ++index) {
      const model::Exl3Phase phase = index == 0
                                         ? model::Exl3Phase{.rows = prefix, .past = 0}
                                         : model::Exl3Phase{.rows = 1, .past = prefix + index - 1};
      auto plan = model::PlanPhase(profile_, binding_, *table_, o_.arm, phase);
      if (!plan) {
        return Error(std::format("prefix {} phase {}: {}", prefix, index, plan.error()));
      }
      region_bytes_ = std::max(region_bytes_, plan->region);
      inputs_bytes_ = std::max(inputs_bytes_, exl3::HostInputsLayout(*plan).bytes);
      logits_bytes_ = std::max(logits_bytes_, plan->tensors.at("logits").bytes());
      phases_.push_back({.prefix = prefix, .index = index, .plan = std::move(*plan)});
    }
  }

  kv_bytes_ = std::uint64_t{profile_.layers} * 2 * kCells * profile_.kv_width() * 2;
  const std::uint64_t norm_bytes = std::uint64_t{profile_.width} * 4;
  const std::uint64_t norms_bytes = norm_bytes * ((2 * std::uint64_t{profile_.layers}) + 1);
  for (const auto& [mapped, name, bytes, cls, recovery] :
       {std::tuple{&kv_, "the EXL3 cache", kv_bytes_, MemoryClass::kLiveState, Recovery::kPreserve},
        std::tuple{&locks_, "the lock area", exl3::kLockBytes, MemoryClass::kRuntime,
                   Recovery::kPinned},
        std::tuple{&norms_, "the F32 norms", norms_bytes, MemoryClass::kWeights, Recovery::kPinned},
        std::tuple{&tables_, "the multi-GEMM tables", std::uint64_t{48} * profile_.layers,
                   MemoryClass::kRuntime, Recovery::kPinned}}) {
    if (auto r =
            node_.MapResident(*mapped, name, bytes, BackingKind::kDevice, cls, recovery, owner_);
        !r) {
      return r;
    }
  }
  auto inputs = node_.Pinned(inputs_bytes_, owner_, staging_);
  auto logits = node_.Pinned(logits_bytes_, owner_, staging_);
  auto derive = node_.Pinned(norms_bytes + (std::uint64_t{48} * profile_.layers), owner_, staging_);
  if (!inputs) {
    return std::unexpected(inputs.error());
  }
  if (!logits) {
    return std::unexpected(logits.error());
  }
  if (!derive) {
    return std::unexpected(derive.error());
  }
  inputs_ = *inputs;
  logits_ = *logits;
  derive_ = *derive;

  // The registry and the reconstruction GEMM.
  std::vector<execution::Implementation> implementations = kg::Implementations();
  for (auto& implementation : exl3::Implementations()) {
    implementations.push_back(std::move(implementation));
  }
  auto registry = execution::Registry::Create(std::move(implementations));
  if (!registry) {
    return Error("the registry was refused: " + registry.error().detail);
  }
  registry_ = std::make_unique<execution::Registry>(std::move(*registry));
  auto gemm = exl3::ReconGemm::Create();
  if (!gemm) {
    return Error("the reconstruction GEMM was refused: " + gemm.error().detail);
  }
  gemm_ = std::move(*gemm);
  if (auto r = ReserveWeights(); !r) {
    return r;
  }
  if (auto r = HeadIsItsOwn(); !r) {
    return r;
  }

  // The memory map at place 0 (the weights' addresses, whether resident or
  // not); the region is the node's, once it is mapped.
  memory_map_.cells = kCells;
  memory_map_.region = kPlanningRegion;
  memory_map_.region_bytes = region_bytes_;
  memory_map_.final_norm = norms_.base;
  const std::uint64_t per_layer = std::uint64_t{2} * kCells * profile_.kv_width() * 2;
  for (std::uint32_t l = 0; l < profile_.layers; ++l) {
    exl3::Qwen2Layer layer;
    layer.attn_norm = norms_.base + (norm_bytes * (1 + (2 * std::uint64_t{l})));
    layer.mlp_norm = layer.attn_norm + norm_bytes;
    layer.trellis_table = tables_.base + (std::uint64_t{48} * l);
    layer.suh_table = layer.trellis_table + 16;
    layer.svh_table = layer.trellis_table + 32;
    layer.k_cache = kv_.base + (per_layer * l);
    layer.v_cache = layer.k_cache + (per_layer / 2);
    memory_map_.layers.push_back(layer);
  }
  MapWeights();
  // The EXL3 launch context: it queues its lock area's zeroing on the
  // model's stream, which nothing else uses before the scheduler runs, so
  // every job there follows it.
  auto launch =
      exl3::LaunchContext::Create(0, node_.execution(), node_.stream(stream_), locks_.base);
  if (!launch) {
    return Error("the EXL3 launch context was refused: " + launch.error().detail);
  }
  launch_ = std::move(*launch);
  // The GGML pool's size: every phase bound on a planning context (no
  // workspace; it queues nothing), for the largest attention scratch.
  auto planning = kg::LaunchContext::Create(0, node_.execution(), node_.stream(stream_),
                                            {.base = 0, .size = Bytes(0)});
  if (!planning) {
    return Error(planning.error().detail);
  }
  // The tables are written only by Derive: planning binds a copy that
  // records them as they will be written, and runs nothing.
  exl3::Qwen2Memory planned = memory_map_;
  for (exl3::Qwen2Layer& layer : planned.layers) {
    layer.tables_written = exl3::MultiGemmTables(layer.gate.weights, layer.up.weights);
  }
  for (const PhaseRun& phase : phases_) {
    auto program =
        exl3::Qwen2Program::Bind(*registry_, profile_, phase.plan, planned, **planning, *launch_);
    if (!program) {
      return Error(std::format("prefix {} phase {} did not bind: {}", phase.prefix, phase.index,
                               program.error().detail));
    }
    pool_bytes_ = std::max(pool_bytes_, (*program)->ggml_scratch());
  }
  return {};
}

Status Exl3Runner::Bind() {
  memory_map_.region = node_.activations().base;
  std::vector<ExtentId> all = device_weights_;
  for (const ts::Mapped* mapped : std::initializer_list<const ts::Mapped*>{
           &kv_, &node_.activations(), &node_.pool(), &locks_, &norms_, &tables_}) {
    all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
  }
  all.insert(all.end(), staging_.begin(), staging_.end());
  everything_ = node_.catalog().ClosureOfExtents(all).value();
  cache_ = node_.catalog().ClosureOfExtents(kv_.extents).value();
  auto ggml = kg::LaunchContext::Create(0, node_.execution(), node_.stream(stream_),
                                        {.base = node_.pool().base, .size = Bytes(pool_bytes_)});
  if (!ggml) {
    return Error("the GGML launch context was refused: " + ggml.error().detail);
  }
  ggml_ = std::move(*ggml);
  return {};
}

// The norms widened to F32 from the device weights, and the tables of the
// weights' current place: copied down to staging, widened on the host, and
// copied up, in two jobs.
Status Exl3Runner::Derive() {
  const std::uint64_t norm_bytes = std::uint64_t{profile_.width} * 4;
  const std::uint64_t half = std::uint64_t{profile_.width} * 2;
  std::vector<std::uint32_t> sources = {binding_.final_norm};
  for (const model::Exl3LayerBinding& b : binding_.layers) {
    sources.push_back(b.attn_norm);
    sources.push_back(b.mlp_norm);
  }
  auto* staged = static_cast<std::byte*>(derive_);
  auto& execution = node_.execution();
  const providers::StreamId stream = node_.stream(stream_);
  Status copied;
  if (auto r = Job(
          everything_,
          [&](providers::NativeStream) {
            for (std::size_t i = 0; i < sources.size(); ++i) {
              if (auto c = execution.Copy(stream, Address(staged) + (i * half),
                                          WeightAddress(sources[i]), Bytes(half));
                  !c) {
                copied = Error(c.error().detail);
                return sc::AfterRefusal(c.error().error, i > 0);
              }
            }
            return sc::JobResult::kQueued;
          },
          "copying the norms down");
      !r || !copied) {
    return !copied ? copied : r;
  }
  // Widen in place, from the back (F32 is twice F16's size).
  const std::size_t count = sources.size() * profile_.width;
  std::vector<std::uint16_t> bf16(count);
  std::memcpy(bf16.data(), staged, count * 2);
  for (std::size_t i = 0; i < count; ++i) {
    const float value = Bf16ToFloat(bf16[i]);
    std::memcpy(staged + (i * 4), &value, 4);
  }
  if (auto r = Job(
          everything_,
          [&](providers::NativeStream) {
            if (auto c = execution.Copy(stream, norms_.base, Address(staged),
                                        Bytes(sources.size() * norm_bytes));
                !c) {
              copied = Error(c.error().detail);
              return sc::AfterRefusal(c.error().error, false);
            }
            // Each layer's record is set only once its tables' copies are
            // queued: after a failure here, Bind refuses a stale table.
            if (auto c =
                    exl3::WriteMultiGemmTables(execution, stream,
                                               std::span(staged + (sources.size() * norm_bytes),
                                                         std::uint64_t{48} * profile_.layers),
                                               memory_map_);
                !c) {
              copied = Error(c.error().detail);
              return c.error().error == exl3::KernelError::kUnknown ? sc::JobResult::kUnknown
                                                                    : sc::JobResult::kFailed;
            }
            return sc::JobResult::kQueued;
          },
          "uploading the norms and tables");
      !r || !copied) {
    return !copied ? copied : r;
  }
  return {};
}

// ------------------------------------------------------------------ evaluating

Status Exl3Runner::RunPhase(const PhaseRun& phase, int evaluation, std::vector<float>& out) {
  auto program =
      exl3::Qwen2Program::Bind(*registry_, profile_, phase.plan, memory_map_, *ggml_, *launch_);
  if (!program) {
    return Error(std::format("prefix {} phase {} did not bind: {}", phase.prefix, phase.index,
                             program.error().detail));
  }
  identities_[std::format("{}/{}/{}", evaluation, phase.prefix, phase.index)] =
      base::ToHex((*program)->identity());
  Check(**program, phase);
  const model::Exl3PhasePlan& plan = phase.plan;
  const ts::Mapped& region = node_.activations();
  // The phase kind's guaranteed bound, and the highest region byte its
  // bound operations reach.
  Peak& peak = peaks_[{plan.phase.rows, plan.padded}];
  ++peak.phases;
  peak.region_bound = std::max(peak.region_bound, plan.region);
  peak.pool_bound = std::max(peak.pool_bound, (*program)->ggml_scratch());
  for (std::size_t op = 0; op < plan.ops.size(); ++op) {
    for (const auto& names : {plan.ops[op].inputs, plan.ops[op].outputs}) {
      for (const std::string& name : names) {
        const auto t = plan.tensors.find(name);
        const std::uint64_t at = (*program)->Address(op, name);
        if (t != plan.tensors.end() && at >= region.base && at < region.base + region.bytes) {
          peak.region_seen = std::max(peak.region_seen, at + t->second.bytes() - region.base);
        }
      }
    }
  }
  ggml_->ResetScratchPeak();
  const std::span<const std::int32_t> tokens(ids_.data() + plan.phase.past,
                                             static_cast<std::size_t>(plan.phase.rows));
  if (auto written = exl3::WriteHostInputs(
          profile_, plan, tokens,
          std::span(static_cast<std::byte*>(inputs_), exl3::HostInputsLayout(plan).bytes));
      !written) {
    return Error(written.error().detail);
  }
  const std::uint64_t logits_bytes = plan.tensors.at("logits").bytes();
  const std::uint64_t logits_at = region.base + plan.slots.at("logits").offset;
  Status ran;
  const bool record = recording_ != nullptr && evaluation == 1;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    if (record) {
      for (const auto& event : recording_->Take()) {
        record_ += test_support::EventLine(event);
      }
      record_ += test_support::ChunkLine({.evaluation = evaluation,
                                          .chunk = phase.index,
                                          .rows = plan.phase.rows,
                                          .n_past = plan.phase.past});
    }
    exl3::Qwen2Hooks hooks;
    if (record) {
      hooks.before = [&](std::size_t op) -> std::expected<void, exl3::KernelFailure> {
        for (const auto& event : recording_->Take()) {
          record_ += test_support::EventLine(event);
        }
        record_ += test_support::OpLine(plan.ops[op].name, plan.ops[op].layer);
        return {};
      };
    }
    if (auto r = (*program)->Run(*ggml_, *launch_, *gemm_, node_.execution(), node_.stream(stream_),
                                 Address(inputs_), hooks);
        !r) {
      ran =
          Error(std::format("prefix {} phase {}: {}", phase.prefix, phase.index, r.error().detail));
      return r.error().error == exl3::KernelError::kUnknown ? sc::JobResult::kUnknown
                                                            : sc::JobResult::kFailed;
    }
    if (record) {
      for (const auto& event : recording_->Take()) {
        record_ += test_support::EventLine(event);
      }
      record_ += test_support::OpLine("outputs", -1);
    }
    if (auto r =
            Cuda(cudaMemcpyAsync(logits_, Pointer(logits_at), logits_bytes, cudaMemcpyDeviceToHost,
                                 static_cast<cudaStream_t>(native.handle)),
                 "the logits copy");
        !r) {
      ran = r;
      return sc::JobResult::kUnknown;
    }
    if (record) {
      for (const auto& event : recording_->Take()) {
        record_ += test_support::EventLine(event);
      }
      record_ += test_support::EndChunkLine();
    }
    return sc::JobResult::kQueued;
  };
  if (auto r = Job(everything_, std::move(job), "a phase"); !r || !ran) {
    return !ran
               ? ran
               : Error(std::format("prefix {} phase {}: {}", phase.prefix, phase.index, r.error()));
  }
  if (launch_->faulted() || ggml_->faulted()) {
    return Error(
        std::format("prefix {} phase {}: a launch context faulted", phase.prefix, phase.index));
  }
  peak.pool_seen = std::max(peak.pool_seen, ggml_->scratch_peak().value());
  const auto* half = static_cast<const std::uint16_t*>(logits_);
  for (std::size_t i = 0; i < logits_bytes / 2; ++i) {
    out.push_back(model::HalfToFloat(half[i]));
  }
  return {};
}

Status Exl3Runner::Evaluate(int evaluation, std::map<int, std::vector<float>>& result) {
  for (const PhaseRun& phase : phases_) {
    if (phase.index == 0) {
      const std::uint64_t kv = kv_.base;
      const std::uint64_t bytes = kv_bytes_;
      if (auto cleared = Job(
              cache_,
              [kv, bytes](providers::NativeStream stream) {
                return cudaMemsetAsync(Pointer(kv), 0, bytes,
                                       static_cast<cudaStream_t>(stream.handle)) == cudaSuccess
                           ? sc::JobResult::kQueued
                           : sc::JobResult::kUnknown;
              },
              "clearing the cache");
          !cleared) {
        return cleared;
      }
      result[phase.prefix].clear();
    }
    if (auto r = RunPhase(phase, evaluation, result[phase.prefix]); !r) {
      return r;
    }
    const int partial_at = o_.partial ? 3 + o_.restores : -1;
    const int spill_at = o_.spill.empty() ? -1 : 3 + o_.restores + (o_.partial ? 1 : 0);
    if (evaluation > 2 && evaluation <= 2 + o_.restores && phase.index == 0) {
      if (auto r = Restore(evaluation); !r) {
        return r;
      }
    }
    if (evaluation == partial_at && phase.index == 0 && phase.prefix == phases_.front().prefix) {
      if (auto r = Partial(); !r) {
        return r;
      }
    }
    if (evaluation == spill_at && (phase.index == 0 || phase.index == 8)) {
      if (auto r = Spill(std::format("prefix {} {}", phase.prefix,
                                     phase.index == 0 ? "after the prefill" : "mid-decode"));
          !r) {
        return r;
      }
    }
  }
  return {};
}

// BP-P2: each partial eviction in turn.
Status Exl3Runner::Partial() {
  auto& memory = node_.memory();
  const std::string layer = std::format("model.layers.{}.", profile_.layers / 2);
  const auto cases = PartialCases(*artifact_, layer, true);
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
      resident +=
          node_.catalog().Describe(extent).value().state == catalog::ExtentState::kResident ? 1 : 0;
    }
    if (resident + extents.size() != device_weights_.size() ||
        memory.backings() + extents.size() != backings) {
      return Error(std::format("partial eviction '{}' did not evict exactly its {} extents",
                               partial.name, extents.size()));
    }
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
    if (auto r = node_.Load(extents, std::format("partial restore: {}", partial.name), loads_);
        !r) {
      return r;
    }
  }
  return {};
}

// BP-P4: the cache written back through the zone and evicted, then
// restored; its bytes before and after must match.
Status Exl3Runner::Spill(std::string_view where) {
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

Status Exl3Runner::Snapshot(std::vector<std::byte>& out) {
  const std::uint64_t kv = kv_.base;
  const std::uint64_t bytes = kv_bytes_;
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

// --spill: the cache's write-back places (fp16_runner.cc's).
Status Exl3Runner::RegisterCache() {
  std::filesystem::create_directories(o_.out);
  spill_fd_ = ::open(o_.out.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (spill_fd_ < 0) {
    return Error(std::format("the spill file in {}: {}", o_.out.string(),
                             std::generic_category().message(errno)));
  }
  if (auto r = Cuda(cudaMallocHost(&kv_copy_, kv_bytes_), "the cache's host copy"); !r) {
    return r;
  }
  pinned_.push_back(kv_copy_);
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

// BP-P3's EXL3 half: the head is its own representation, never the
// embedding's storage.
Status Exl3Runner::HeadIsItsOwn() const {
  const auto embed = artifact_->ResourcePlacement(binding_.embed);
  const auto head = artifact_->ResourcePlacement(binding_.lm_head.trellis);
  if (!embed || !head) {
    return Error("the embedding or the head has no placement");
  }
  const auto& resources = artifact_->resources();
  const bool shared_name =
      std::ranges::find(resources[binding_.embed].roles, binding_.lm_head.name + ".trellis") !=
      resources[binding_.embed].roles.end();
  const bool overlap = embed->group == head->group &&
                       embed->offset.value() < head->offset.value() + head->readable.value() &&
                       head->offset.value() < embed->offset.value() + embed->readable.value();
  if (binding_.embed == binding_.lm_head.trellis || shared_name || overlap) {
    return Error("the EXL3 head shares the embedding's storage");
  }
  return {};
}

// BP-L1 and BP-L3: a phase cancelled with its GGML and EXL3 work queued
// behind a gate keeps its lease until the fence after that work completes.
Status Exl3Runner::CancelInFlight() {
  const PhaseRun* chosen = nullptr;
  for (const PhaseRun& phase : phases_) {
    if (phase.index == 0 && (chosen == nullptr || phase.plan.region > chosen->plan.region)) {
      chosen = &phase;  // the largest reconstruction phase
    }
  }
  if (chosen == nullptr) {
    return Error("no prefill to cancel");
  }
  const model::Exl3PhasePlan& plan = chosen->plan;
  auto program =
      exl3::Qwen2Program::Bind(*registry_, profile_, plan, memory_map_, *ggml_, *launch_);
  if (!program) {
    return Error(std::format("the cancelled phase did not bind: {}", program.error().detail));
  }
  const std::span<const std::int32_t> tokens(ids_.data() + plan.phase.past,
                                             static_cast<std::size_t>(plan.phase.rows));
  if (auto written = exl3::WriteHostInputs(
          profile_, plan, tokens,
          std::span(static_cast<std::byte*>(inputs_), exl3::HostInputsLayout(plan).bytes));
      !written) {
    return Error(written.error().detail);
  }
  void* gate = nullptr;
  if (auto r = Cuda(cudaHostAlloc(&gate, sizeof(std::uint32_t), cudaHostAllocMapped), "the gate");
      !r) {
    return r;
  }
  pinned_.push_back(gate);
  std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(gate)).store(0);
  void* device_gate = nullptr;
  if (auto r = Cuda(cudaHostGetDevicePointer(&device_gate, gate, 0), "the gate's device address");
      !r) {
    return r;
  }
  std::atomic<bool> started{false};  // the gate is queued: what follows waits behind it
  std::atomic<bool> queued{false};   // the job has queued everything (more launches than
                                     // the stream's pending queue holds wait in the driver)
  Status ran;
  auto job = [&, device_gate](providers::NativeStream native) -> sc::JobResult {
    if (cuStreamWaitValue32(static_cast<CUstream>(native.handle),
                            reinterpret_cast<CUdeviceptr>(device_gate), 1,
                            CU_STREAM_WAIT_VALUE_GEQ) != CUDA_SUCCESS) {
      ran = Error("cuStreamWaitValue32 was refused");
      return sc::JobResult::kUnknown;  // a driver error: its effect is unknown
    }
    started.store(true);
    if (auto r = (*program)->Run(*ggml_, *launch_, *gemm_, node_.execution(), node_.stream(stream_),
                                 Address(inputs_), exl3::Qwen2Hooks{});
        !r) {
      ran = Error(r.error().detail);
      return r.error().error == exl3::KernelError::kUnknown ? sc::JobResult::kUnknown
                                                            : sc::JobResult::kFailed;
    }
    queued.store(true);
    return sc::JobResult::kQueued;
  };
  ts::Done done;
  const std::uint64_t request =
      node_.Submit(std::make_unique<ts::RunProgram>(done, everything_, std::move(job), stream_));
  // Until the job has queued what it can; the lane may still be inside it,
  // its later launches waiting in the driver behind the gate.
  const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (!started.load() && !done.gone.load() && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const bool was_started = started.load();
  const bool was_queued = queued.load();
  node_.Cancel(request);
  // On the scheduler thread, with the gate still shut: what the lease holds.
  const auto held = [&] {
    std::vector<ExtentId> touched = node_.activations().extents;
    touched.insert(touched.end(), kv_.extents.begin(), kv_.extents.end());
    touched.insert(touched.end(), device_weights_.begin(), device_weights_.end());
    std::size_t holding = 0;
    for (const ExtentId extent : touched) {
      const auto view = node_.catalog().Describe(extent).value();
      holding += view.leases > 0 && !catalog::Catalog::Evictable(view) ? 1 : 0;
    }
    return holding == touched.size() ? touched.size() : 0;
  };
  // Not returned early on failure: the gate must open before this frame,
  // which the job refers to, can end (a failed probe leaves `holding` 0).
  std::size_t holding = 0;
  std::ignore = node_.Call(
      [&]() -> Status {
        holding = held();
        return {};
      },
      "probing the cancelled phase's lease");
  const bool retired_early = done.gone.load();
  std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(gate)).store(1);
  std::ignore = node_.Await(done, "the cancelled phase", request);  // its outcome is checked below
  const bool cancelled = done.outcome.load() == static_cast<int>(sc::TaskOutcome::kCancelled);
  std::size_t after = 1;
  if (auto r = node_.Call(
          [&]() -> Status {
            after = 0;
            for (const ExtentId extent : node_.activations().extents) {
              after += node_.catalog().Describe(extent).value().leases;
            }
            return {};
          },
          "probing after the fence");
      !r) {
    return r;
  }
  cancel_result_ = std::format(
      R"({{"phase_rows":{},"gate_queued_before_cancel":{},"all_queued_before_cancel":{},)"
      R"("extents_held_while_gated":{},"retired_before_the_gate_opened":{},)"
      R"("outcome_cancelled":{},"leases_after_fence":{}}})",
      plan.phase.rows, was_started ? "true" : "false", was_queued ? "true" : "false", holding,
      retired_early ? "true" : "false", cancelled ? "true" : "false", after);
  if (!ran || !was_started || holding == 0 || retired_early || !cancelled || after != 0) {
    return Error(
        std::format("cancelling in flight: {}{}", cancel_result_, ran ? "" : ": " + ran.error()));
  }
  return {};
}

// Rung 5: every weight extent evicted, its backing released, then paged
// back in through the zone (with relocate, every second time at the other
// place, and the tables rewritten for it).
Status Exl3Runner::Restore(int evaluation) {
  auto& memory = node_.memory();
  const std::size_t backings = memory.backings();
  if (auto r = node_.Evict(device_weights_); !r) {
    return r;
  }
  for (const ExtentId extent : device_weights_) {
    if (node_.catalog().Describe(extent).value().state != catalog::ExtentState::kNonresident) {
      return Error("a weight extent is still resident after the eviction");
    }
  }
  if (memory.backings() + device_weights_.size() != backings) {
    return Error("the eviction did not release every weight's backing");
  }
  if (o_.relocate && evaluation % 2 == 0) {
    if (auto r = node_.Call([this] { return Place(1 - place_); }, "relocating the weights"); !r) {
      return r;
    }
  }
  if (auto r = node_.Load(device_weights_, std::format("restore {}", evaluation - 2), loads_); !r) {
    return r;
  }
  MapWeights();
  return Derive();
}

Status Exl3Runner::RunAlone() {
  if (auto r = node_.Load(device_weights_, "initial", loads_); !r) {
    return r;
  }
  if (auto r = Derive(); !r) {
    return r;
  }
  if (o_.record) {
    recording_ = std::make_unique<test_support::Recording>();
    (void)recording_->Take();
  }
  const int evaluations = 2 + o_.restores + (o_.partial ? 1 : 0) + (o_.spill.empty() ? 0 : 1);
  std::vector<std::map<int, std::vector<float>>> results(static_cast<std::size_t>(evaluations));
  for (int e = 1; e <= evaluations; ++e) {
    if (auto r = Evaluate(e, results[static_cast<std::size_t>(e - 1)]); !r) {
      return r;
    }
    if (e == 1 && recording_) {
      for (const auto& event : recording_->Take()) {
        record_ += test_support::EventLine(event);
      }
      recording_.reset();
    }
  }
  if (o_.cancel) {
    if (auto r = CancelInFlight(); !r) {
      return r;
    }
  }
  return Write(results);
}

Status Exl3Runner::Write(const std::vector<std::map<int, std::vector<float>>>& results) {
  std::filesystem::create_directories(o_.out);
  const auto& first = results.front();
  std::string differences;
  std::size_t differing_total = 0;
  for (std::size_t e = 1; e < results.size(); ++e) {
    std::size_t differing = 0;
    for (const auto& [prefix, values] : first) {
      const auto& other = results[e].at(prefix);
      differing += values.size() == other.size() ? 0 : values.size();
      for (std::size_t i = 0; i < std::min(values.size(), other.size()); ++i) {
        differing +=
            std::bit_cast<std::uint32_t>(values[i]) != std::bit_cast<std::uint32_t>(other[i]);
      }
    }
    differences += std::format("{}{}", differences.empty() ? "" : ",", differing);
    differing_total += differing;
  }
  std::string logits;
  const auto vocab = static_cast<std::int64_t>(profile_.vocab);
  for (const auto& [prefix, values] : first) {
    const std::size_t prefill = static_cast<std::size_t>(prefix) * profile_.vocab;
    const auto bytes = std::as_bytes(std::span(values));
    if (auto r = WriteNpy(o_.out / std::format("logits-{}.prefill.npy", prefix), "<f4",
                          {prefix, vocab}, bytes.first(prefill * 4));
        !r) {
      return r;
    }
    if (auto r = WriteNpy(o_.out / std::format("logits-{}.suffix.npy", prefix), "<f4",
                          {kSuffix, vocab}, bytes.subspan(prefill * 4));
        !r) {
      return r;
    }
    logits += std::format(R"({}"{}": {{"prefill_sha256": "{}", "suffix_sha256": "{}"}})",
                          logits.empty() ? "" : ", ", prefix, Hex(bytes.first(prefill * 4)),
                          Hex(bytes.subspan(prefill * 4)));
  }
  std::string loads;
  for (const ts::LoadStats& load : loads_) {
    loads += std::format(
        R"({}{{"what": "{}", "extents": {}, "seconds": {:.6f}, "requests": {}, "pieces": {}}})",
        loads.empty() ? "" : ", ", load.what, load.extents, load.seconds, load.requests,
        load.pieces);
  }
  std::string by_class;
  for (std::size_t c = 0; c < coverage_.by_class.size(); ++c) {
    by_class += std::format("{}{}", c == 0 ? "" : ", ", coverage_.by_class.at(c));
  }
  std::string identities;
  for (const auto& [key, identity] : identities_) {
    identities += std::format(R"({}"{}": "{}")", identities.empty() ? "" : ", ", key, identity);
  }
  std::ofstream summary(o_.out / "summary.json");
  summary << std::format(
      "{{\"harness\": \"llmp_exl3_paged\", \"fixture\": \"{}\", \"arm\": \"{}\", \"artifact\": "
      "\"{}\", \"plan_file_sha256\": \"{}\", \"memory\": \"device VMM through the landing zone\", "
      "\"lanes\": \"{}\", \"evaluations\": {}, \"restores\": {}, \"relocate\": {}, "
      "\"coalesce\": {},\n "
      "\"bit_differences_from_first\": [{}],\n \"logits\": {{{}}},\n \"weight_extents\": {}, "
      "\"stored_bytes\": {}, \"zone_bytes\": {}, \"kv_bytes\": {}, \"region_bytes\": {}, "
      "\"pool_bytes\": {}, \"loads\": [{}],\n \"coverage\": {{\"ranges\": {}, \"violations\": {}, "
      "\"first\": \"{}\", \"by_class\": [{}]}},\n \"plan_identities\": {{{}}}}}\n",
      o_.fixture, o_.arm == model::Exl3Arm::kG ? "G" : "O", artifact_->id(), HexFile(o_.plan),
      node_.inline_lanes() ? "inline" : "threads", results.size(), o_.restores,
      o_.relocate ? "true" : "false", node_.coalesce() ? "true" : "false", differences, logits,
      device_weights_.size(), stored_bytes_, node_.zone().bytes, kv_.bytes,
      node_.activations().bytes, node_.pool().bytes, loads, coverage_.ranges, coverage_.violations,
      coverage_.first, by_class, identities);
  if (!record_.empty()) {
    std::ofstream file(o_.out / "record.jsonl");
    file << test_support::HeaderLine(
                std::format("llmp_exl3_paged {} {}", o_.fixture,
                            o_.arm == model::Exl3Arm::kG ? "EXL3-G" : "EXL3-O"),
                test_support::LoadedCublas())
         << record_;
  }
  // BP-P2 and BP-P4's runs, and each phase kind's bound against its peak.
  std::string events;
  for (const PagingEvent& event : events_) {
    events += std::format(R"({}{{"what":"{}","extents":{},"seconds":{:.6f},"detail":"{}"}})",
                          events.empty() ? "" : ",", event.what, event.extents, event.seconds,
                          event.detail);
  }
  std::string peaks;
  bool within = true;
  for (const auto& [kind, peak] : peaks_) {
    within = within && peak.region_seen <= peak.region_bound && peak.pool_seen <= peak.pool_bound;
    peaks +=
        std::format(R"({}{{"rows":{},"padded":{},"phases":{},"region_bound":{},"region_seen":{},)"
                    R"("pool_bound":{},"pool_seen":{}}})",
                    peaks.empty() ? "" : ",", kind.first, kind.second, peak.phases,
                    peak.region_bound, peak.region_seen, peak.pool_bound, peak.pool_seen);
  }
  {
    std::ofstream file(o_.out / "paging.json");
    file << std::format(
                R"({{"partial":{},"spill":"{}","head_is_its_own":true,"refused_launches":{},)"
                R"("kv_bytes_differing":{},"cancel_in_flight":{},"events":[{}],"peaks":[{}]}})",
                o_.partial ? "true" : "false", o_.spill, refusals_, kv_mismatches_,
                cancel_result_.empty() ? std::string("null") : cancel_result_, events, peaks)
         << "\n";
  }
  std::println("wrote {}", o_.out.string());
  if (kv_mismatches_ != 0) {
    return Error(std::format("{} bytes of the cache differ after its restore", kv_mismatches_));
  }
  if (!within) {
    return Error("a phase reached past its guaranteed bound (paging.json)");
  }
  if (coverage_.violations != 0) {
    return Error(std::format("{} ranges lie outside cataloged extents of their class; first: {}",
                             coverage_.violations, coverage_.first));
  }
  if (differing_total != 0) {
    return Error(std::format("a later evaluation differs from the first: [{}]", differences));
  }
  return {};
}

Status Exl3Runner::Release() {
  if (released_) {
    return {};
  }
  released_ = true;
  std::vector<std::string> problems;
  launch_.reset();
  ggml_.reset();
  auto& memory = node_.memory();
  for (ts::Mapped* mapped : {&kv_, &locks_, &norms_, &tables_}) {
    if (!ts::ReleaseMapped(memory, *mapped)) {
      problems.push_back(std::format("{} could not be released", mapped->name));
    }
  }
  for (const providers::ReservationId reservation : weights_) {
    if (reservation.valid() && !memory.Free(reservation)) {
      problems.emplace_back("a weights reservation still has mappings");
    }
  }
  for (void* pointer : pinned_) {
    (void)cudaFreeHost(pointer);
  }
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
