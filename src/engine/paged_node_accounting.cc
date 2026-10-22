// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/paged_node_internal.h"

namespace llmp::engine {

bool PagedNode::Recharge(std::uint64_t total, bool force, std::uint64_t* shortfall) {
  const std::uint64_t past = total > host_floor_ ? total - host_floor_ : 0;
  return ChargeRuntime(host_extent_, host_charged_, past, force, shortfall, "plans and graphs");
}

bool PagedNode::SetTokenCharge(std::uint64_t bytes, std::uint64_t* shortfall) {
  return ChargeRuntime(token_extent_, token_charged_, bytes, bytes <= token_charged_, shortfall,
                       "native token histories");
}

bool PagedNode::SetRequestCharge(std::uint64_t bytes, std::uint64_t* shortfall) {
  return ChargeRuntime(request_extent_, request_charged_, bytes, bytes <= request_charged_,
                       shortfall, "request memory");
}

bool PagedNode::ChargeRuntime(catalog::ExtentId& held, std::uint64_t& charged, std::uint64_t bytes,
                              bool force, std::uint64_t* shortfall, std::string_view what) {
  const std::uint64_t charge = (bytes + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  if (charge == charged) {
    return true;
  }
  // Before the scheduler runs, or once it stopped, there is nothing to
  // charge against: only the count changes.
  if (scheduler_ == nullptr || torn_down_) {
    charged = charge;
    return true;
  }
  bool fits = true;
  auto changed = RegionCatalog(
      [&]() -> Status {
        const std::uint64_t occupancy = catalog_.OccupancyOf(domain_).Total().value();
        const std::uint64_t others = occupancy - std::min(occupancy, charged);
        if (charge > charged && !force &&
            (others > budget().value() || charge > budget().value() - others)) {
          fits = false;
          if (shortfall != nullptr) {
            *shortfall = others + charge - budget().value();
          }
          return {};
        }
        if (charge != 0) {
          auto extent = catalog_.AddExtent({.domain = domain_,
                                            .memory_class = MemoryClass::kRuntime,
                                            .recovery = catalog::Recovery::kPinned,
                                            .size = Bytes(charge),
                                            .content = {}},
                                           true);
          if (!extent) {
            return Error(std::format("charging the {}", what));
          }
          if (held.valid()) {
            if (!catalog_.ReleasePinned(held) || !catalog_.RemoveExtent(held)) {
              (void)catalog_.ReleasePinned(*extent);
              (void)catalog_.RemoveExtent(*extent);
              return Error(std::format("forgetting the {}' charge", what));
            }
          }
          held = *extent;
        } else if (held.valid()) {
          if (!catalog_.ReleasePinned(held) || !catalog_.RemoveExtent(held)) {
            return Error(std::format("forgetting the {}' charge", what));
          }
          held = {};
        }
        return {};
      },
      std::format("charging {}", what));
  if (!changed) {
    // Only the count is kept; the catalog's charge is what it was.
    return force;
  }
  if (fits) {
    charged = charge;
  }
  return fits;
}

bool PagedNode::ChargeHost(std::uint64_t bytes, bool required) {
  CollectHostReleases();
  if (bytes > UINT64_MAX - host_total_ || host_total_ + bytes > UINT64_MAX - (kPagedExtent - 1))
    return false;
  if (bytes == 0) {
    return true;
  }
  std::uint64_t shortfall = 0;
  if (Recharge(host_total_ + bytes, false, &shortfall)) {
    host_total_ += bytes;
    return true;
  }
  // What the charge lacks (all of it or nothing), from other plans and
  // graphs only (a graph's only from what costs less to restore than a
  // graph). The charge moves in whole extents past the floor, so the ask is
  // rounded up by one and the room checked again after each reclaim.
  const ReclaimFor what = required ? ReclaimFor::kPlan : ReclaimFor::kGraph;
  for (int attempt = 0; attempt < 3 && shortfall != 0; ++attempt) {
    const std::uint64_t ask =
        ((shortfall + kPagedExtent - 1) / kPagedExtent * kPagedExtent) + kPagedExtent;
    if (AskReclaim(ask, what) == 0) {
      break;
    }
    shortfall = 0;
    if (Recharge(host_total_ + bytes, false, &shortfall)) {
      host_total_ += bytes;
      return true;
    }
  }
  if (!required) {
    return false;
  }
  ++host_overcharges_;
  (void)Recharge(host_total_ + bytes, true, nullptr);
  host_total_ += bytes;
  return true;
}

bool PagedNode::TryChargeHost(std::uint64_t bytes) {
  if (job_active_ || torn_down_) return false;
  CollectHostReleases();
  if (bytes > UINT64_MAX - host_total_ || host_total_ + bytes > UINT64_MAX - (kPagedExtent - 1))
    return false;
  if (!Recharge(host_total_ + bytes, false, nullptr)) return false;
  host_total_ += bytes;
  return true;
}

std::uint64_t PagedNode::AskReclaim(std::uint64_t needed, ReclaimFor what,
                                    std::span<const ExtentId> protect) {
  if (!reclaimer_ || reclaiming_ || needed == 0) {
    return 0;
  }
  reclaiming_ = true;
  const std::uint64_t freed = reclaimer_(needed, what, protect);
  reclaiming_ = false;
  return freed;
}

std::uint64_t PagedNode::Shortfall(std::uint64_t bytes) const {
  const std::uint64_t occupancy = catalog_.OccupancyOf(domain_).Total().value();
  const std::uint64_t current_budget = budget().value();
  return occupancy > current_budget || bytes > current_budget - occupancy
             ? occupancy + bytes - current_budget
             : 0;
}

void PagedNode::CollectHostReleases() {
  if (!host_releases_) return;
  const auto bytes = host_releases_->bytes.exchange(0, std::memory_order_acq_rel);
  if (bytes != 0) UnchargeHost(bytes);
}

std::expected<HostPayment, std::string> PagedNode::PrepayHost(std::uint64_t bytes) {
  if (job_active_ || torn_down_ || !memory_)
    return Error("host result payment needs a live node outside a job");
  if (bytes == 0) return HostPayment{};
  if (!host_releases_) {
    // Allocate once, on the first nonempty result. Its original metadata
    // receipt stays with this shared owner even when a result outlives the
    // node. No node methods are needed to destroy that residual paid account.
    constexpr std::uint64_t kAllocationMetadataBytes = 64;
    constexpr auto kMetadata = sizeof(HostPayment::Releases) + kAllocationMetadataBytes;
    std::uint64_t charge = 0;
    if (__builtin_add_overflow(bytes, std::uint64_t{kMetadata}, &charge) ||
        !ChargeHost(charge, false))
      return Error("owned host result bytes do not fit");
    auto* counter = new (std::nothrow) HostPayment::Releases(kMetadata);
    if (!counter) {
      UnchargeHost(charge);
      return Error("allocating paid host result release metadata failed");
    }
    host_releases_.reset(counter);
  } else if (!ChargeHost(bytes, false)) {
    return Error("owned host result bytes do not fit");
  }
  return HostPayment(host_releases_.get(), bytes);
}

void PagedNode::UnchargeHost(std::uint64_t bytes) {
  host_total_ -= std::min(bytes, host_total_);
  (void)Recharge(host_total_, true, nullptr);
}

std::expected<std::uint64_t, std::string> PagedNode::FreeBytes() {
  CollectHostReleases();
  std::uint64_t free = 0;
  auto read = Call(
      [&]() -> Status {
        const std::uint64_t occupancy = catalog_.OccupancyOf(domain_).Total().value();
        free = occupancy < budget().value() ? budget().value() - occupancy : 0;
        return {};
      },
      "reading the free budget");
  if (!read) {
    return std::unexpected(read.error());
  }
  return free;
}

Status PagedNode::Load(std::vector<ExtentId> extents, std::string what,
                       std::vector<LoadStats>& log) {
  if (auto files = SyncWeightFiles(); !files) {
    return files;
  }
  LoadStats stats{.what = std::move(what), .extents = 0, .seconds = 0, .requests = 0, .pieces = 0};
  // The catalog is the scheduler's: read on its thread, where no eviction
  // or page-in changes it meanwhile.
  std::optional<catalog::Closure> closure;
  if (auto described = Call(
          [&]() -> Status {
            for (const ExtentId extent : extents) {
              const auto view = catalog_.Describe(extent);
              if (!view) return Error("a load names an extent the catalog does not hold");
              if (view->state != catalog::ExtentState::kResident) ++stats.extents;
            }
            auto of = catalog_.ClosureOfExtents(extents);
            if (!of) return Error("a load's closure");
            closure = std::move(*of);
            return {};
          },
          "describing a load");
      !described) {
    return described;
  }
  Done done;
  const std::uint64_t requests = counting_->requests.load(std::memory_order_relaxed);
  const std::uint64_t pieces = counting_->pieces.load(std::memory_order_relaxed);
  const auto start = std::chrono::steady_clock::now();
  if (auto posted =
          Post(std::make_unique<RunProgram>(done, *closure, sc::DeviceJob{}), done, stats.what);
      !posted) {
    return posted;
  }
  stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  // The load's reads have all completed, so their counts were made before.
  stats.requests = counting_->requests.load(std::memory_order_relaxed) - requests;
  stats.pieces = counting_->pieces.load(std::memory_order_relaxed) - pieces;
  log.push_back(std::move(stats));
  return {};
}

std::expected<std::shared_ptr<sc::EvictionReceipt>, std::string> PagedNode::NewEvictionReceipt(
    std::uint64_t observer_bytes) {
  struct PaidReceipt final : sc::EvictionReceipt {
    explicit PaidReceipt(HostPayment paid) : payment(std::move(paid)) {}
    HostPayment payment;
  };
  // Receipt/control block and observer allocation overhead, in addition to
  // the concrete observer payload. Its payment survives task/node teardown.
  constexpr std::uint64_t kMetadata = sizeof(PaidReceipt) + 128;
  if (observer_bytes > UINT64_MAX - kMetadata) return Error("eviction receipt metadata overflows");
  auto paid = PrepayHost(observer_bytes + kMetadata);
  if (!paid) return std::unexpected(paid.error());
  return std::make_shared<PaidReceipt>(std::move(*paid));
}

Status PagedNode::Evict(std::vector<ExtentId> extents, sc::EvictOptions options) {
  if (!options.receipt) options.receipt = eviction_receipt_;
  if (auto ended = EndRequestsOver(extents); !ended) {
    return std::unexpected(ended.error());
  }
  Done done;
  return Post(std::make_unique<EvictProgram>(done, std::move(extents), options), done,
              "an eviction");
}

Status PagedNode::Swap(std::vector<ExtentId> out, const catalog::Closure& in, bool handoff,
                       SwapReport& report, std::vector<ExtentId> release,
                       catalog::Closure weights) {
  if (auto files = SyncWeightFiles(); !files) {
    return files;
  }
  // Asked for between a request's steps: the requests holding what goes
  // out end first, since the swap would wait for their leases.
  if (!weights.extents.empty()) {
    auto checked = Call(
        [&]() -> Status {
          for (const auto& [extent, generation] : weights.extents) {
            const auto view = catalog_.Describe(extent);
            if (!view || view->content_generation != generation ||
                !catalog::Catalog::Evictable(*view))
              return Error("a selected idle weight is stale or held");
          }
          return {};
        },
        "validating selected idle weights before request retirement");
    if (!checked) return checked;
  }
  auto ended = EndRequestsOver(out);
  if (!ended) {
    return std::unexpected(ended.error());
  }
  Done done;
  auto swapped =
      Post(std::make_unique<SwapProgram>(done, std::move(out), in, handoff, report,
                                         std::move(release), std::move(weights), eviction_receipt_),
           done, "a swap");
  report.requests_ended = *ended;
  return swapped;
}

std::expected<sc::SchedulerStats, std::string> PagedNode::Stats() {
  sc::SchedulerStats stats;
  if (auto r = Call(
          [&]() -> Status {
            stats = scheduler_->stats();
            stats.catalog_occupancy_bytes = catalog_.OccupancyOf(domain_).Total().value();
            return {};
          },
          "reading the scheduler's counters");
      !r) {
    return std::unexpected(r.error());
  }
  stats.backing_creates = backing_create_stats();
  return stats;
}

std::expected<std::uint64_t, std::string> PagedNode::EndRequestsOver(
    std::span<const ExtentId> extents) {
  std::vector<std::uint32_t> holding;
  for (const auto& [stream, open] : requests_) {
    const auto& held = open->closure.extents;  // sorted by extent
    if (std::ranges::any_of(extents, [&held](ExtentId extent) {
          const auto at = std::ranges::lower_bound(held, extent, {},
                                                   &std::pair<ExtentId, std::uint64_t>::first);
          return at != held.end() && at->first == extent;
        })) {
      holding.push_back(stream);
    }
  }
  for (const std::uint32_t stream : holding) {
    if (auto ended = EndRequest(stream); !ended) {
      return std::unexpected(ended.error());
    }
  }
  return holding.size();
}

}  // namespace llmp::engine
