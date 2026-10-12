// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/supplemental_run.h"

#include <atomic>
#include <limits>
#include <utility>
#include <vector>

namespace llmp::engine {
namespace sc = scheduler;

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
  bool offered = false, started = false, held = false;
  ~State() {
    if (held) external->fetch_sub(1, std::memory_order_acq_rel);
    allowance->Release(charge);
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

std::expected<std::unique_ptr<SupplementalRun>, std::string> SupplementalRun::Create(
    const catalog::Catalog& catalog, catalog::DomainId domain, sc::RequestChannel& base,
    const sc::ProgramDone& base_done, const catalog::Closure& selected, sc::DeviceJob job,
    std::uint32_t stream, std::shared_ptr<const void> payload,
    std::shared_ptr<ExecutionMetadataAllowance> allowance) {
  auto bytes = Bound(selected.extents.size());
  if (!bytes) return std::unexpected(bytes.error());
  if (!payload || !job || !allowance || !base.external ||
      !base.held.load(std::memory_order_acquire) || base_done.gone.load() ||
      base_done.outcome.load() != -1 || base.end)
    return std::unexpected("supplemental work needs a live base request and owned payload");
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
  if (!allowance->Charge(*bytes)) return std::unexpected("supplemental metadata was not funded");
  auto state = std::shared_ptr<State>(new State, [](State* s) {
    // A dropped ticket is not a retirement certificate. Keep the entire source
    // owner, counter and charge if an offered operation still may touch them.
    if (s->held && s->offered) return;
    delete s;
  });
  state->allowance = std::move(allowance);
  state->charge = *bytes;
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
