// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/batch_resources.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <new>
#include <utility>

#include "base/check.h"
#include "engine/batch_execution.h"
#include "engine/step_transaction.h"
#include "engine/support.h"
#include "kernels/common/lower.h"

namespace llmp::engine {
namespace {
namespace kn = kernels;
namespace pl = planner;
constexpr std::uint64_t kAllocationMetadata = 64;
std::unexpected<kn::ResourceFailure> Refused(kn::ResourceError error, std::string what) {
  return std::unexpected(kn::ResourceFailure{error, std::move(what)});
}
bool Add(std::uint64_t& bytes, std::uint64_t count, std::uint64_t size) {
  std::uint64_t added = 0;
  return !__builtin_mul_overflow(count, size, &added) &&
         !__builtin_add_overflow(bytes, added, &bytes) && bytes <= PTRDIFF_MAX;
}
bool SameShape(const pl::StepShape& a, const pl::StepShape& b) {
  // Segment equality deliberately ignores past for ordinary cache reuse. A
  // private prepaid execution inventory authenticates the actual coordinates.
  if (a != b) return false;
  for (std::size_t i = 0; i < a.segments.size(); ++i)
    if (a.segments[i].past != b.segments[i].past) return false;
  return true;
}
}  // namespace

void PromptBatch::Data::Backing::Destroy() {
  base::Check(!open && !activations.reservation.valid() && !pool.reservation.valid(),
              "private backing must retire on its node before consumer metadata destruction");
  auto paid = std::move(payment);
  auto paid_inputs = std::move(host_inputs_payment);
  delete this;
  paid_inputs.Reset();
  paid.Reset();
}
PromptBatch::Data::Data(HostPayment paid, ModelRunner& runner, PromptBatchInventory preview)
    : payment(std::move(paid)),
      owner(runner.RetainedOwner()),
      inventory(std::move(preview)),
      node(runner.node_),
      stream_index(runner.stream()) {
  registered.model = this;
  runner.batch_active_ = this;
}
void PromptBatch::Data::Destroy() {
  base::Check(closed, "a private batch metadata destructor is not retirement authority");
  auto paid = std::move(payment);
  delete this;
  paid.Reset();
}
bool PromptBatch::Data::ready() const {
  return published && !closed && !closing && owner && !owner->released_ && !owner->closing_ &&
         backing && backing->open && !unproven.active();
}
std::uint32_t PromptBatch::Data::stream() const { return stream_index; }
std::expected<std::size_t, std::string> PromptBatch::Data::Find(const ModelRunner& runner,
                                                                const pl::StepShape& shape) const {
  if (!ready() || owner.get() != &runner)
    return std::unexpected("private batch is closed, unproven or belongs to another runner");
  if (auto current = inventory.CheckSource(); !current) return std::unexpected(current.error());
  const auto shapes = inventory.shapes();
  for (std::size_t i = 0; i < shapes.size(); ++i)
    if (!entries[i].consumed && entries[i].transaction && SameShape(shapes[i], shape)) return i;
  return std::unexpected("step has no unused exact prepaid batch plan");
}
bool PromptBatch::Data::Matches(std::size_t index, const pl::StepShape& shape) const {
  return ready() && inventory.CheckSource().has_value() && index < inventory.shapes().size() &&
         SameShape(inventory.shapes()[index], shape);
}
Status PromptBatch::Data::Close() {
  if (closed) return {};
  if (pending || unproven.active())
    return std::unexpected("private batch retains work without positive retirement");
  for (const auto& view : views) {
    if (!view.launch()) continue;
    if (auto drained = view.launch()->DrainGeneratedInputs(); !drained)
      return std::unexpected(drained.error().detail);
  }
  closing = true;
  // Keep every plan/view/backing intact if the canonical reverse exchange
  // refuses. A copied token never evicts or reopens its request to get around it.
  if (extension) {
    if (auto removed = node.CloseRequestExtension(extension); !removed) return removed;
  }
  if (owns_request && node.InRequest(stream_index)) {
    if (auto ended = node.EndRequest(stream_index); !ended) return ended;
  }
  // All bound programs and their launch-dependent metadata die before views,
  // private backing, the parent environment and the original host refunds.
  for (auto& entry : entries) {
    entry.route.reset();
    entry.transaction.reset();
    entry.built_program.reset();
    entry.plan.planned.reset();
    for (auto& flight_runs : entry.plan.runs) flight_runs.DropGraph();
  }
  views.clear();
  entries.clear();
  runs.SetLaunch(nullptr);
  runs.SetStaging(nullptr, 0);
  if (backing) {
    if (auto retired = backing->memory.Retire(); !retired) return retired;
    backing->open = false;
    backing_keep = {};
    backing = nullptr;
  }
  pinned = nullptr;
  inventory = {};
  if (owner && owner->batch_active_ == this) owner->batch_active_ = nullptr;
  retained_parent = {};
  owner.reset();
  closed = true;
  auto keep = registered.keep;  // keep embedded node-registration storage live
  node.RetireModelRoot(registered);
  return {};
}
Status PromptBatch::Data::RetireAfterFence() {
  // Called only by the node after actual stream fences. It can now break a
  // retained flight cycle; a public capability's Close cannot claim this fact.
  if (closed) return {};
  if (owner) {
    owner->unproven_.ClearAfterFence();
  }
  unproven.ClearAfterFence();
  pending = 0;
  return Close();
}
Status PromptBatch::Data::Release() {
  return closed ? Status{} : Status{std::unexpected("private batch needs its node's drain")};
}
bool PromptBatch::ready() const { return keep_ && data_->ready(); }
std::uint64_t PromptBatch::paid_bytes() const {
  if (!keep_) return 0;
  std::uint64_t total = 0;
  const auto add = [&](std::uint64_t bytes) {
    base::Check(!__builtin_add_overflow(total, bytes, &total),
                "live paid batch metadata cannot exceed its physical budget");
  };
  add(data_->payment.bytes());
  add(data_->inventory.paid_bytes());
  if (data_->backing) {
    add(data_->backing->payment.bytes());
    add(data_->backing->host_inputs_payment.bytes());
  }
  for (const auto& entry : data_->entries) {
    add(entry.payment.bytes());
    add(entry.frame_payment.bytes());
  }
  for (const auto& view : data_->views) add(view.host_bytes());
  return total;
}
Status PromptBatch::Close() { return keep_ ? data_->Close() : Status{}; }

StepAhead PromptBatch::Ahead(const planner::StepShape& shape) const {
  if (!ready() || data_->private_resources) return {};
  const auto found = data_->Find(*data_->owner, shape);
  return found ? data_->entries[*found].ahead : StepAhead{};
}

bool PromptBatch::DeclaredFor(const ModelRunner& runner) {
  for (std::size_t stage = 0; stage < runner.slots_.stages(); ++stage)
    if (!runner.source_->ResourceBackend(static_cast<std::uint16_t>(stage))) return false;
  return runner.slots_.stages() != 0;
}

std::expected<PromptBatch, kn::ResourceFailure> PromptBatch::Prepare(
    ModelRunner& runner, const WaveComposer& composer, std::span<const PendingWork> pending,
    std::span<const SegmentWork> work, std::span<const std::span<const std::int32_t>> histories,
    std::span<const std::byte> step_bytes, const BatchAhead& ahead) {
  if (runner.batch_active_)
    return Refused(kn::ResourceError::kOwnership, "runner already has an owned complete batch");
  auto preview = PromptBatchInventory::Preview(runner, composer, pending);
  if (!preview) return std::unexpected(preview.error());
  const auto count = preview->shapes().size();
  std::uint64_t bytes = sizeof(Data) + kAllocationMetadata;
  if (!Add(bytes, count, sizeof(Data::Entry)) ||
      !Add(bytes, composer.stages(), sizeof(kn::ExecutionResources)) ||
      !Add(bytes, 2, kAllocationMetadata))
    return Refused(kn::ResourceError::kBounds, "private batch descriptors overflow");
  auto paid = runner.node_.PrepayHost(bytes);
  if (!paid) return Refused(kn::ResourceError::kCapacity, paid.error());
  auto* data = new (std::nothrow) Data(std::move(*paid), runner, std::move(*preview));
  if (!data) return Refused(kn::ResourceError::kCapacity, "allocating paid batch owner failed");
  PromptBatch result(data->Publish(), data);
  data->registered.keep = data->Keep();
  runner.node_.KeepModelRoot(data->registered);
  // Before publication every failure closes known untouched resources. A
  // failed/unknown cleanup leaves its complete node-owned root registered.
  const auto fail = [&](kn::ResourceError kind,
                        std::string what) -> std::expected<PromptBatch, kn::ResourceFailure> {
    if (auto closed = data->Close(); !closed) {
      what += "; retained cleanup: " + closed.error();
      kind = kn::ResourceError::kFault;
    }
    return Refused(kind, std::move(what));
  };
  data->entries.resize(count);
  data->views.resize(composer.stages());
  ProgramEnvelope maximum;
  std::uint64_t main_pool = 0, inputs = 0;
  bool private_route = false;
  for (std::size_t i = 0; i < count; ++i) {
    if (data->inventory.fallback(i)) continue;
    const auto& shape = data->inventory.shapes()[i];
    if (auto allowed = runner.CheckShape(shape); !allowed)
      return fail(kn::ResourceError::kBounds, allowed.error());
    if (!runner.source_->ResourceBackend(shape.stage))
      return fail(kn::ResourceError::kUnsupported,
                  "actual source backend has no producer-owned private batch resources");
    // A cache hit already carries the exact selected implementation/envelope.
    // Reuse its paid key buffers without allocating or recompiling the graph.
    auto& entry = data->entries[i];
    if (runner.FillKey(shape, runner.lookup_key_, true))
      entry.cached = runner.plans_.Find(runner.lookup_key_);
    std::unique_ptr<kn::StepProgram> measured;
    bool built_for_shared = false;
    double build_seconds = 0;
    if (!entry.cached) {
      if (runner.activation_bytes() != 0) {
        const auto t0 = std::chrono::steady_clock::now();
        auto built = runner.source_->Build(shape, runner.AddressesFor(shape, false));
        if (built && *built) {
          build_seconds = support::Seconds(std::chrono::steady_clock::now() - t0);
          measured = std::move(*built);
          built_for_shared = true;
        } else if (!built && built.error().find("activation") == std::string::npos) {
          return fail(kn::ResourceError::kBounds, built.error());
        }
      }
      if (!measured) {
        auto addresses = runner.AddressesFor(shape, true);
        auto built = runner.source_->Build(shape, addresses);
        if (!built || !*built)
          return fail(kn::ResourceError::kBounds,
                      built ? "batch measurement returned no program" : built.error());
        measured = std::move(*built);
      }
    }
    const auto& candidate = entry.cached ? *entry.cached->planned->program : *measured;
    auto envelope = ProgramEnvelope::Measure(candidate);
    if (!envelope) return fail(kn::ResourceError::kBounds, envelope.error());
    entry.measured = *envelope;
    maximum.Include(*envelope);
    main_pool = std::max(main_pool, candidate.private_pool_bytes());
    if (const auto* route = candidate.routed()) {
      data->route_bytes = std::max(data->route_bytes, route->pinned_bytes());
      private_route = private_route || !runner.CheckRouting(candidate);
    }
    if (built_for_shared && !private_route && envelope->scratch <= runner.activation_bytes() &&
        runner.config_.launch_pool_reserve <= runner.pool_bytes() &&
        envelope->pool <= runner.pool_bytes() - runner.config_.launch_pool_reserve &&
        envelope->inputs <= runner.programs_.runs().staging_bytes() &&
        envelope->outputs <= runner.outputs_staging_bytes_) {
      bool has_later_unbuilt = false;
      for (std::size_t j = i + 1; j < count; ++j)
        if (!data->inventory.fallback(j)) {
          has_later_unbuilt = true;
          break;
        }
      if (has_later_unbuilt) {
        auto temp_paid = runner.node_.PrepayHost(measured->host_bytes());
        if (!temp_paid) return fail(kn::ResourceError::kCapacity, temp_paid.error());
        entry.payment = std::move(*temp_paid);
        measured->ConstructionPaid();
      }
      entry.built_program = std::move(measured);
      entry.build_seconds = build_seconds;
    }
    data->entries[i].inputs_first = inputs;
    if (!Add(inputs, 1, envelope->inputs))
      return fail(kn::ResourceError::kBounds, "private batch immutable input extent overflows");
  }
  data->private_resources = private_route || maximum.scratch > runner.activation_bytes() ||
                            maximum.pool > runner.pool_bytes() ||
                            maximum.inputs > runner.programs_.runs().staging_bytes() ||
                            maximum.outputs > runner.outputs_staging_bytes_;
  if (data->private_resources) {
    for (auto& entry : data->entries) {
      entry.built_program.reset();
      entry.payment.Reset();
    }
  }
  auto retained = runner.KeepResources();
  if (!retained) return fail(retained.error().error, retained.error().what);
  data->retained_parent = std::move(*retained);
  data->output_first = inputs;
  if (data->private_resources && !Add(inputs, 1, maximum.outputs))
    return fail(kn::ResourceError::kBounds, "private batch DMA extent overflows");
  data->route_first = inputs;
  if (data->private_resources && !Add(inputs, 1, data->route_bytes))
    return fail(kn::ResourceError::kBounds, "private batch route DMA extent overflows");
  data->pinned_bytes = inputs;
  auto descriptors = RunnerMemory::MetadataBytes(2, inputs ? 1 : 0);
  if (!descriptors) return fail(kn::ResourceError::kBounds, descriptors.error());
  auto backing_paid =
      runner.node_.PrepayHost(sizeof(Data::Backing) + kAllocationMetadata + *descriptors);
  if (!backing_paid) return fail(kn::ResourceError::kCapacity, backing_paid.error());
  data->backing = new (std::nothrow)
      Data::Backing(std::move(*backing_paid), runner.node_, runner.owner_, runner.stream_);
  if (!data->backing)
    return fail(kn::ResourceError::kCapacity, "allocating paid private backing owner failed");
  data->backing_keep = data->backing->Publish();
  data->backing->memory.ReserveMetadata(2, inputs ? 1 : 0);
  bool over_budget = false;
  if (data->private_resources) {
    if (auto mapped =
            data->backing->memory.Map(data->backing->activations, "batch act", maximum.scratch,
                                      catalog::MemoryClass::kScratch, &over_budget);
        !mapped)
      return fail(over_budget ? kn::ResourceError::kCapacity : kn::ResourceError::kFault,
                  mapped.error());
    if (auto mapped = main_pool
                          ? data->backing->memory.Map(data->backing->pool, "batch pool", main_pool,
                                                      catalog::MemoryClass::kScratch, &over_budget)
                          : Status{};
        !mapped)
      return fail(over_budget ? kn::ResourceError::kCapacity : kn::ResourceError::kFault,
                  mapped.error());
  }
  if (inputs) {
    if (data->private_resources) {
      auto staging = data->backing->memory.Pinned(inputs, &over_budget);
      if (!staging)
        return fail(over_budget ? kn::ResourceError::kCapacity : kn::ResourceError::kFault,
                    staging.error());
      data->pinned = static_cast<std::byte*>(*staging);
    } else {
      std::uint64_t host_bytes = inputs;
      if (!Add(host_bytes, 1, kAllocationMetadata))
        return fail(kn::ResourceError::kBounds, "private batch immutable input extent overflows");
      auto host_paid = runner.node_.PrepayHost(host_bytes);
      if (!host_paid) {
        over_budget = true;
        return fail(kn::ResourceError::kCapacity, host_paid.error());
      }
      data->backing->host_inputs_payment = std::move(*host_paid);
      data->backing->host_inputs.reset(new (std::nothrow) std::byte[inputs]);
      if (!data->backing->host_inputs)
        return fail(kn::ResourceError::kCapacity, "allocating paid batch host inputs failed");
      data->pinned = data->backing->host_inputs.get();
    }
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (data->inventory.fallback(i)) continue;
    const auto& shape = data->inventory.shapes()[i];
    if (!data->private_resources) {
      auto& entry = data->entries[i];
      if (entry.cached) continue;
      if (entry.built_program) {
        if (auto identity = runner.source_->ExecutionIdentity(); !identity)
          return fail(kn::ResourceError::kBounds, identity.error());
        if (auto current = runner.CheckRoutingSource(); !current)
          return fail(kn::ResourceError::kBounds, current.error());
        if (auto filled = runner.FillKey(shape, runner.lookup_key_); !filled)
          return fail(kn::ResourceError::kBounds, filled.error());
        entry.cached = runner.plans_.Find(runner.lookup_key_);
        if (entry.cached) {
          entry.built_program.reset();
          entry.payment.Reset();
          continue;
        }
        auto planned = std::make_unique<StepPlanned>();
        planned->program = std::move(entry.built_program);
        auto installed = runner.Install(shape, std::move(planned), entry.build_seconds,
                                        [&] { entry.payment.Reset(); });
        if (!installed) return fail(kn::ResourceError::kBounds, installed.error());
        entry.cached = *installed;
        continue;
      }
      auto cached = runner.Planned(shape);
      if (!cached) return fail(kn::ResourceError::kBounds, cached.error());
      entry.cached = *cached;
      continue;
    }
    data->entries[i].cached = nullptr;  // private bindings never borrow ordinary entries
    auto& view = data->views[shape.stage];
    if (!view) {
      auto parent = runner.KeepResources();
      if (!parent) return fail(parent.error().error, parent.error().what);
      auto backend = runner.source_->ResourceBackend(shape.stage);
      auto made = backend->Resources(backend, {data->backing->pool.base, data->backing->pool.bytes},
                                     std::move(*parent), data->backing_keep);
      if (!made) return fail(made.error().error, made.error().what);
      view = std::move(*made);
    }
    auto addresses = runner.AddressesFor(shape, false);
    addresses.activations = data->backing->activations.base;
    addresses.activation_bytes = data->backing->activations.bytes;
    auto program = runner.source_->BuildPrivate(shape, addresses, view);
    if (!program || !*program)
      return fail(kn::ResourceError::kBounds,
                  program ? "private lowering returned no program" : program.error());
    auto actual = ProgramEnvelope::Measure(**program);
    if (!actual) return fail(kn::ResourceError::kBounds, actual.error());
    const auto& measured = data->entries[i].measured;
    if (actual->scratch > maximum.scratch || (**program).private_pool_bytes() > main_pool ||
        actual->inputs > measured.inputs || actual->outputs > measured.outputs ||
        ((**program).routed() && (**program).routed()->pinned_bytes() > data->route_bytes))
      return fail(kn::ResourceError::kBounds, "private bound plan exceeds its measured resources");
    // The source's construction owner remains intact until an independent
    // installed payment covers all actual returned program/control storage.
    std::uint64_t host = (**program).host_bytes();
    if (!Add(host, 1, sizeof(StepPlanned) + kAllocationMetadata))
      return fail(kn::ResourceError::kBounds, "private installed plan extent overflows");
    auto installed = runner.node_.PrepayHost(host);
    if (!installed) return fail(kn::ResourceError::kCapacity, installed.error());
    auto& entry = data->entries[i];
    entry.payment = std::move(*installed);
    entry.plan.planned = std::make_unique<StepPlanned>();
    entry.plan.planned->program = std::move(*program);
    auto& bound = *entry.plan.planned->program;
    if (!bound.UsesResources(view))
      return fail(kn::ResourceError::kOwnership, "private program lost its actual producer view");
    auto linked = runner.programs_.BindPrepared(*entry.plan.planned, *view.launch(),
                                                [&](const kn::StepProgram& actual_program) {
                                                  return runner.CheckPorts(shape, actual_program);
                                                });
    if (!linked) return fail(kn::ResourceError::kBounds, linked.error());
    runner.coverage_.tensors += linked->tensors;
    bound.ConstructionPaid();
    entry.plan.runs[0].uncapturable = true;  // private eager-only launch identity
  }
  // Actual input/control/result preflight and all state capacity admission
  // are filled by the canonical transaction machinery, before any compound
  // Begin or numeric child can consume the published batch capability.
  if (auto prepared = StepTransaction::PrepareBatch(result, pending, work, histories, step_bytes,
                                                    ahead, &over_budget);
      !prepared)
    return fail(over_budget ? kn::ResourceError::kCapacity : kn::ResourceError::kBounds,
                prepared.error());
  data->published = true;
  return result;
}
}  // namespace llmp::engine
