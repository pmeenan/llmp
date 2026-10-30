// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/supplemental_run.h"

#include <atomic>
#include <limits>
#include <utility>
#include <vector>

#include "base/process_retention.h"

namespace llmp::engine {
namespace sc = scheduler;

struct SupplementalCapacity::Data final : kernels::ResourceDependency {
  kernels::ResourceKeep Publish() { return Adopt(); }
  void Destroy() override {
    auto paid = std::move(allowance);
    const auto bytes = charge;
    delete this;
    paid->Release(bytes);
  }
  std::shared_ptr<ExecutionMetadataAllowance> allowance;
  std::shared_ptr<std::atomic<std::uint32_t>> external;
  catalog::DomainId domain;
  const catalog::Catalog* catalog = nullptr;
  std::uint64_t charge = 0, max_extents = 0;
  // Only the serialized scheduler/catalog owner accesses busy. Payment can be
  // released on another thread, without touching the catalog or a dead node.
  bool busy = false;
};

struct SupplementalRun::State {
  sc::ProgramDone done;
  catalog::Closure closure;
  std::vector<catalog::ExtentId> ids;
  sc::DeviceJob job;
  std::uint32_t stream = 0;
  std::shared_ptr<const void> payload;
  std::shared_ptr<ExecutionMetadataAllowance> allowance;
  std::shared_ptr<std::atomic<std::uint32_t>> external;
  std::uint64_t charge = 0;
  SupplementalCapacity capacity;
  bool offered = false, started = false, held = false;
  ~State() {
    if (held) external->fetch_sub(1, std::memory_order_acq_rel);
  }
};

class SupplementalRun::ProgramImpl final : public sc::TaskProgram {
 public:
  explicit ProgramImpl(std::shared_ptr<State> state)
      : state_(std::move(state)),
        run_(state_->done, state_->closure, std::move(state_->job), state_->stream) {}
  sc::Step Advance(sc::TaskContext& context) override {
    state_->started = true;
    return run_.Advance(context);
  }
  void Finished(sc::TaskOutcome outcome) override { run_.Finished(outcome); }
  void Retired() override { run_.Retired(); }

 private:
  // run_ is destroyed first: gone is its final touch of State::done.
  std::shared_ptr<State> state_;
  sc::RunProgram run_;
};

std::expected<std::uint64_t, std::string> SupplementalRun::Bound(std::uint64_t extents) {
  // State+wrapper+program+control blocks and one domain-map node <=4096.
  // Two closure vectors, scoped IDs, and scheduler materialization's per-entry
  // transient descriptors fit 256 bytes/extent. Scheduler queues/operations
  // and payload storage remain in the node's separately admitted inventories.
  static_assert(sizeof(State) + sizeof(ProgramImpl) + sizeof(SupplementalRun) + 1024 <= 4096);
  static_assert(
      2 * sizeof(std::pair<catalog::ExtentId, std::uint64_t>) + sizeof(catalog::ExtentId) <= 128);
  if (extents == 0 || extents > (UINT64_MAX - 4096) / 256 || extents > SIZE_MAX)
    return std::unexpected("invalid supplemental closure descriptor bound");
  return 4096 + extents * 256;
}

std::expected<SupplementalCapacity, std::string> SupplementalRun::Prepare(
    const catalog::Catalog& catalog, catalog::DomainId domain, const sc::RequestChannel& base,
    const sc::ProgramDone& base_done, std::uint64_t max_extents,
    std::shared_ptr<ExecutionMetadataAllowance> allowance) {
  auto bytes = Bound(max_extents);
  if (!bytes) return std::unexpected(bytes.error());
  if (!allowance || !allowance->HeldBy(catalog, domain) || !base.external ||
      !base.held.load(std::memory_order_acquire) || base_done.gone.load() ||
      base_done.outcome.load() != -1 || base.end)
    return std::unexpected("supplemental prepayment needs its live authenticated request");
  // Actual receipt storage, its shared control and allocator overhead, plus
  // the node's serialized validation/retirement controls. The operation's
  // State, Program and selected closure are already in Bound(max_extents).
  constexpr std::uint64_t receipt_bytes = sizeof(SupplementalCapacity::Data) + 64;
  std::uint64_t total = 0;
  if (__builtin_add_overflow(*bytes, receipt_bytes + kControlBytes, &total) ||
      !allowance->Charge(total))
    return std::unexpected("supplemental prepaid capacity was not funded");
  auto* data = new SupplementalCapacity::Data;
  data->allowance = std::move(allowance);
  data->external = base.external;
  data->domain = domain;
  data->catalog = &catalog;
  data->charge = total;
  data->max_extents = max_extents;
  return SupplementalCapacity(data->Publish(), data);
}

bool SupplementalRun::Covers(const SupplementalCapacity& capacity, const sc::RequestChannel& base,
                             catalog::DomainId domain,
                             const std::shared_ptr<ExecutionMetadataAllowance>& allowance) {
  return capacity && capacity.data_->external == base.external &&
         capacity.data_->domain == domain && capacity.data_->allowance == allowance &&
         !capacity.data_->busy;
}

std::expected<std::unique_ptr<SupplementalRun>, std::string> SupplementalRun::Create(
    const catalog::Catalog& catalog, catalog::DomainId domain, sc::RequestChannel& base,
    const sc::ProgramDone& base_done, const catalog::Closure& selected, sc::DeviceJob job,
    std::uint32_t stream, std::shared_ptr<const void> payload,
    std::shared_ptr<ExecutionMetadataAllowance> allowance, const SupplementalCapacity& capacity) {
  auto bytes = Bound(selected.extents.size());
  if (!bytes) return std::unexpected(bytes.error());
  if (!payload || !job || !allowance || !base.external ||
      !base.held.load(std::memory_order_acquire) || base_done.gone.load() ||
      base_done.outcome.load() != -1 || base.end)
    return std::unexpected("supplemental work needs a live base request and owned payload");
  if (capacity &&
      (!Covers(capacity, base, domain, allowance) || capacity.data_->catalog != &catalog ||
       selected.extents.size() > capacity.data_->max_extents))
    return std::unexpected("supplemental capacity is busy, foreign or too small");
  std::uint64_t physical = 0;
  std::optional<catalog::ExtentId> previous;
  for (const auto& [id, generation] : selected.extents) {
    const auto view = catalog.Describe(id);
    if ((previous && id <= *previous) || !view || view->descriptor.domain != domain ||
        view->content_generation != generation || view->discarded ||
        view->state == catalog::ExtentState::kQuarantined ||
        __builtin_add_overflow(physical, view->descriptor.size.value(), &physical))
      return std::unexpected("supplemental closure is unsorted, foreign, stale or quarantined");
    previous = id;
  }
  // No caller map/capacity is copied before payment; use the actual catalog's
  // single-domain sizes. Ownership of the caller's original closure stays put.
  if (!capacity && !allowance->Charge(*bytes))
    return std::unexpected("supplemental metadata was not funded");
  auto state = std::shared_ptr<State>(new State, [](State* s) {
    // A dropped ticket is not a retirement certificate. Keep the entire source
    // owner, counter and charge if an offered operation still may touch them.
    if (s->held && s->offered) {
      base::RetainForProcess(s);
      return;
    }
    auto funder = std::move(s->allowance);
    auto saved_capacity = std::move(s->capacity);
    const auto charge = s->charge;
    delete s;
    if (saved_capacity) saved_capacity.data_->busy = false;
    if (charge != 0) funder->Release(charge);
  });
  state->allowance = std::move(allowance);
  state->charge = capacity ? 0 : *bytes;
  state->capacity = capacity;
  if (capacity) capacity.data_->busy = true;
  state->closure.extents.assign(selected.extents.begin(), selected.extents.end());
  state->closure.bytes_by_domain.emplace(domain, base::Bytes(physical));
  state->ids.reserve(selected.extents.size());
  for (const auto& [id, generation] : selected.extents) {
    (void)generation;
    state->ids.push_back(id);
  }
  state->job = std::move(job);
  state->stream = stream;
  state->payload = std::move(payload);
  state->external = base.external;
  auto count = state->external->load(std::memory_order_acquire);
  do {
    if (count == std::numeric_limits<std::uint32_t>::max())
      return std::unexpected("the request's external step count is exhausted");
  } while (!state->external->compare_exchange_weak(count, count + 1, std::memory_order_acq_rel));
  state->held = true;
  return std::unique_ptr<SupplementalRun>(new SupplementalRun(std::move(state)));
}

SupplementalRun::~SupplementalRun() = default;
sc::ProgramDone& SupplementalRun::done() { return state_->done; }
std::unique_ptr<sc::TaskProgram> SupplementalRun::Program() {
  if (state_->offered || !state_->held) return {};
  state_->offered = true;
  return std::make_unique<ProgramImpl>(state_);
}
std::expected<bool, std::string> SupplementalRun::Retire(const sc::Scheduler& scheduler) {
  if (!state_->held) return true;
  if (state_->offered) {
    if (!state_->done.gone.load(std::memory_order_acquire)) return false;
    if (state_->started) {
      if (!state_->done.retired.load()) return false;
      const auto ended = scheduler.PageInsRetired(state_->ids);
      if (!ended) return std::unexpected("supplemental page-in retirement is unproven");
      if (!*ended) return false;
    }
  }
  state_->external->fetch_sub(1, std::memory_order_acq_rel);
  state_->held = false;
  return true;
}
}  // namespace llmp::engine
