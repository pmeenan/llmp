// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The paged node (docs/runtime-serving.md; first built for the backend
// proof's P2-P5 and BP-S3, docs/backend-proof.md, then M3's swap path):
// device 0 with its providers (opened through the device runtime,
// providers/device_runtime.h), a stream per model plus the copy stream,
// a storage ring (io_uring on Linux, providers/storage.h's OpenStorage),
// one catalog domain, the landing zone (D-081), the scheduler and
// its lanes (the zone's copies on a copy lane of their own by default,
// RE-029), and the workspace the models share. Model runners
// (engine/dsv4_runner.h, qwen38_runner.h, qwen_image_runner.h; the
// harnesses' benchmarks/fp16_runner.h, exl3_runner.h) register their memory
// and sources with it and post their work through it, so several models
// share one catalog, one scheduler and one zone. jitllm-runtime drives one
// (runtime/serving.h), and so do the paged harnesses. CUDA builds only (the
// build's device backend, docs/portability.md).
//
// One thread drives the node (the driver): every call below but the
// accessors is the driver's, and none may be made concurrently. The node
// starts its lanes' threads in Run and joins them in TearDown or its
// destructor.
//
// The order of use:
// 1. Open: the device, providers, streams, storage, domain and zone.
// 2. Each model maps its own memory (MapResident, Pinned) and reserves its
//    weights' places, then MapWorkspace maps the shared workspace at the
//    largest request.
// 3. Start(B) builds the scheduler with the execution budget B; each model
//    registers its sources (scheduler().SetSource).
// 4. Run puts the scheduler and every lane on its own thread (unless the
//    lanes are driven inline); from then on only the scheduler's thread
//    touches the scheduler and the catalog (Call runs a function there).
// 5. TearDown: a fence on each model's stream, the models' weights
//    evicted, the scheduler stopped and the lanes drained (or, if setup
//    ended before Start, a fence on each compute stream); then each
//    model's Release (launch contexts, its memory); then the streams, the
//    zone and the workspace, with every backing checked released.
//
// The shared workspace orders nothing between the models' streams: leases
// are shared, so two models' jobs over it would race. They never overlap
// because Post (and Load, Evict, Job, Call, Acquire) returns only once its
// program is gone, a job's only after its fence completed. A caller that
// Submits without waiting runs no other model's job until that request is
// gone.
//
// Requests (M3's lease per request; scheduler.h): BeginRequest opens one
// on a model's stream, whose task (RequestProgram) materializes the
// model's closure and leases it once. Until EndRequest, every Job on that
// stream is a step of the request: handed to its task and run under that
// lease (its closure must lie within the request's), with no closure
// walked and nothing leased or released per step; Job still returns only
// once the step's fence completed. Ending it releases the lease; the
// extents stay resident. A swap or an eviction asked for between a
// request's steps ends the requests holding what it evicts first (the
// swap would otherwise wait for their release), and TearDown ends every
// one. Every Job's time is noted per stream (StepTimes).
//
// Memory is registered as spans for BP-A1's in-process check: each has an
// owner (a model's index, or kShared for the zone and the workspace), and
// Covered accepts a model's own spans and the shared ones.

#ifndef JITLLM_ENGINE_PAGED_NODE_H_
#define JITLLM_ENGINE_PAGED_NODE_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/device_runtime.h"
#include "providers/storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/programs.h"
#include "scheduler/scheduler.h"
#include "scheduler/services.h"

namespace jitllm::engine {

using Status = std::expected<void, std::string>;

// Where a stream's jobs (PagedNode::Job, a request's steps or not) spent
// their time, summed: the wall from the call to its return; within it,
// until the job began on the device lane (dispatch), the job itself (its
// host time: inputs staged, work queued, waits for room in the stream
// included), and from its end to the return (after); and the device's
// span of the job's work, by timing marks on the stream around it
// (providers/device_runtime.h). The round trip a step adds to the device's
// time is wall - device.
struct StepTimes {
  std::uint64_t steps = 0;
  double wall = 0;
  double dispatch = 0;
  double job = 0;
  double after = 0;
  double device = 0;
};

inline constexpr std::uint64_t kPagedExtent = std::uint64_t{2} << 20U;  // D-033, D-056's chunk
inline constexpr std::size_t kPagedDepth = 4;                           // D-034's bulk depth
inline constexpr std::size_t kPagedSlots = 2 * kPagedDepth;             // D-081's zone
inline constexpr int kShared = -1;  // the owner of the zone and the shared workspace

// Pinned host memory (providers/device_runtime.h) of `bytes` at `pointer`,
// unless it holds some already. False, leaving it null, if the allocation
// failed. A runner's own, uncataloged (its state's host copy); the node's
// staging comes from PagedNode::Pinned.
bool HavePinned(void*& pointer, std::uint64_t bytes);

// A cataloged range of memory, for the coverage check.
struct Span {
  std::uint64_t base = 0;
  std::uint64_t size = 0;
  catalog::ExtentId extent;
  catalog::MemoryClass memory_class = catalog::MemoryClass::kUnknown;
  bool device = true;
  int owner = kShared;
};

// Memory mapped at setup and released at teardown: one reservation, one
// 2 MiB backing and one resident extent per 2 MiB.
struct Mapped {
  std::string name;
  providers::ReservationId reservation;
  std::uint64_t base = 0;
  std::uint64_t bytes = 0;  // rounded to extents
  std::vector<providers::BackingId> backings;
  std::vector<catalog::ExtentId> extents;
};

// Unmaps and releases `mapped`'s backing (it holds none once the VMM lane
// manages it, D-033) and frees its reservation; false if any step failed.
// A mapped that holds nothing is left as it is.
bool ReleaseMapped(providers::VmmProvider& memory, Mapped& mapped);

// A model's places checked still where it registered them and pinned there
// (D-090), on the scheduler's thread: how many are not, and the first.
struct PlaceCheck {
  std::size_t moved = 0;
  std::string first;
  void Check(const scheduler::Scheduler& scheduler, catalog::ExtentId extent,
             const scheduler::PageSource& registered);
  // Something registered is missing (`what`).
  void Missing(std::string what);
};

struct LoadStats {
  std::string what;
  std::uint64_t extents = 0;  // nonresident when it was posted
  double seconds = 0;
  std::uint64_t requests = 0;  // direct-I/O requests the load's reads took
  std::uint64_t pieces = 0;    // the chunks (segments) they carried
};

struct NodeSettings {
  std::size_t compute_streams = 1;  // one per model: streams 0..n-1; the copy stream is n
  std::size_t slots = kPagedSlots;
  bool inline_lanes = false;
  // BP-P1's coalesced reads (64 MiB spans): the reader's option, off by
  // default as the reader's own default is.
  bool coalesce = false;
  // The zone's copies on a lane of their own (RE-029; scheduler.h), on
  // the copy stream, so a model's long job never holds them up; otherwise
  // on the device lane, as in M2.
  bool copy_lane = true;
  // Each slot's bytes (and the reader's largest request): 2 MiB, or more
  // for reads longer than their extent (a DeepSeek expert slab's pages,
  // paged_weights.h kSlabSlotBytes). A multiple of 4 KiB.
  std::uint64_t slot_bytes = kPagedExtent;
  // Told of each page-in's progress, on the scheduler's thread; outlives
  // the node. Optional.
  scheduler::PageInObserver* observer = nullptr;
  // A diagnostic only: how long the scheduler (while a critical operation
  // is in flight or a request holds its lease) and the device lane's
  // submission thread keep polling after their last progress before they
  // sleep (RE-017). Unset, the node runs the runtime's own wake (its
  // defaults, docs/experiments/runtime-wake/), which is what the harness
  // measures. The harness once set 100 ms, longer than a decode step, so
  // that neither slept between a request's steps; its figures are labelled
  // harness-polled (docs/experiments/fast-swap/swap.md).
  // Initialized so callers may designate only the fields they change.
  // NOLINTNEXTLINE(readability-redundant-member-init)
  std::optional<std::chrono::microseconds> poll_window = {};
  // A diagnostic only: how long before a step's likely end the device lane
  // (DeviceSettings::spin_ahead) and a request's driver start to spin.
  // Unset, the runtime's default (1 ms).
  // NOLINTNEXTLINE(readability-redundant-member-init)
  std::optional<std::chrono::microseconds> spin_ahead = {};
};

// What the storage lane hands io_uring (BP-P1): requests, and the pieces
// they carry (a segment each; a plain request is one). Counted on the
// lane's thread, read between loads.
class CountingStorage final : public providers::Storage {
 public:
  explicit CountingStorage(providers::Storage& inner) : inner_(inner) {}
  std::size_t depth() const override { return inner_.depth(); }
  std::size_t in_flight() const override { return inner_.in_flight(); }
  providers::Submission Submit(const providers::IoRequest& request) override;
  providers::Submission Cancel(std::uint64_t token) override { return inner_.Cancel(token); }
  std::size_t Harvest(std::span<providers::IoCompletion> out, bool wait) override {
    return inner_.Harvest(out, wait);
  }
  void Wake() override { inner_.Wake(); }

  std::atomic<std::uint64_t> requests{0};
  std::atomic<std::uint64_t> pieces{0};

 private:
  providers::Storage& inner_;
};

// What a model gives the node's teardown.
class PagedModel {
 public:
  PagedModel() = default;
  PagedModel(const PagedModel&) = delete;
  PagedModel& operator=(const PagedModel&) = delete;
  PagedModel(PagedModel&&) = delete;
  PagedModel& operator=(PagedModel&&) = delete;
  virtual ~PagedModel() = default;
  // Its compute stream's index (TearDown's fence there leases nothing),
  // and a closure a fence job there may lease: its own clears, copies and
  // reads use it, and it may be nonresident while the model is swapped out.
  virtual std::uint32_t stream() const = 0;
  virtual const catalog::Closure& fence_closure() const = 0;
  // What must be evicted before the scheduler stops: every extent whose
  // backing the VMM lane manages.
  virtual std::vector<catalog::ExtentId> managed_extents() const = 0;
  // After the scheduler stopped cleanly: its launch contexts, memory and
  // files.
  virtual Status Release() = 0;
};

class PagedNode {
 public:
  explicit PagedNode(NodeSettings settings) : settings_(settings) {}
  PagedNode(const PagedNode&) = delete;
  PagedNode& operator=(const PagedNode&) = delete;
  PagedNode(PagedNode&&) = delete;
  PagedNode& operator=(PagedNode&&) = delete;
  // Stops the scheduler and joins the threads if TearDown did not.
  ~PagedNode();

  Status Open();
  // The shared workspace (scratch, discardable): the activations and the
  // GGML pool, each at the largest size a model asked for (at least one
  // extent). Once, before Start.
  Status MapWorkspace(std::uint64_t activations, std::uint64_t pool);
  Status Start(base::Bytes budget);
  void Run();
  // Every problem in one error. Once.
  Status TearDown(std::span<PagedModel* const> models);

  providers::VmmProvider& memory() { return *memory_; }
  providers::DeviceExecution& execution() { return *execution_; }
  providers::StreamId stream(std::uint32_t index) const { return streams_.at(index); }
  std::size_t device_class() const { return device_class_; }
  std::size_t host_class() const { return host_class_; }
  catalog::Catalog& catalog() { return catalog_; }
  const catalog::Catalog& catalog() const { return catalog_; }
  catalog::DomainId domain() const { return domain_; }
  scheduler::Scheduler& scheduler() { return *scheduler_; }
  base::Bytes budget() const { return budget_; }
  // The physical bytes needed if all registered live-state ceilings were
  // used. A harness may budget this without mapping any unused state.
  std::uint64_t StateCapacity() const;
  bool threaded() const { return !threads_.empty(); }
  bool inline_lanes() const { return settings_.inline_lanes; }
  bool coalesce() const { return settings_.coalesce; }
  const Mapped& zone() const { return zone_; }
  const Mapped& activations() const { return activations_; }
  const Mapped& pool() const { return pool_; }
  std::size_t slots() const { return settings_.slots; }

  // Maps `bytes` (rounded up to extents) of device or host VMM with
  // access, one resident extent of `memory_class` per 2 MiB. Before Run.
  Status MapResident(Mapped& mapped, std::string name, std::uint64_t bytes,
                     providers::BackingKind kind, catalog::MemoryClass memory_class,
                     catalog::Recovery recovery, int owner);
  // Reserves a live-state region and catalogs its extents without physical
  // backing. Each extent is materialized through its registered page source
  // before a job reads or writes it. Before Start.
  Status ReserveState(Mapped& mapped, std::string name, std::uint64_t bytes, int owner);
  // Pinned host memory, cataloged as staging. After Run, added on the
  // scheduler's thread within the execution budget: what does not fit has
  // the reclaimer free it first (as ChargeHost does), and is refused only
  // when it still does not.
  std::expected<void*, std::string> Pinned(std::uint64_t bytes, int owner,
                                           std::vector<catalog::ExtentId>& staging);
  // Pinned staging set apart once, at the runtime's start, for one user at
  // a time (a turn checkpoint's capture or restore, engine/
  // checkpoint_file.h), so a budget full of reclaimable plans, graphs and
  // idle state never starves it. TakeStaging returns it if it is at least
  // `bytes` and free (nullptr otherwise: the caller allocates its own);
  // ReturnStaging gives it back, or, `proven` false (a copy's completion
  // unknown), keeps it out of use to the process's end.
  Status ReserveStaging(std::uint64_t bytes);
  void* TakeStaging(std::uint64_t bytes);
  void ReturnStaging(void* pointer, bool proven);
  // Caller proves no access is in flight. Removes its staging extent.
  Status FreePinned(void* pointer);
  // Unknown completion: retain the allocation until process exit.
  void KeepPinned(void* pointer);
  // A model's own storage ring, at its end. With reads still in flight
  // (stalled ones, which may yet land), the ring and the pinned memory
  // (from Pinned) those reads write are kept to the process's end, neither
  // destroyed nor freed, since only a read's completion retires its memory
  // (storage.h); returns true then. Otherwise the ring is destroyed.
  bool RetireRing(std::unique_ptr<providers::Storage> ring, std::span<void* const> landings);
  // Pinned allocations RetireRing kept from Close's frees.
  std::size_t kept_pinned() const { return kept_pinned_.size(); }

  // Memory the node counts beside the catalog's extents (D-090 as amended
  // 2026-10-02): the models' plans and graphs (engine/planned.h
  // PlanAccount), host heap and driver memory. The first `floor` bytes of
  // them are what the start's guard sets apart outside the budget (what
  // one step of the largest model holds at once); the rest is charged
  // inside the budget, as one pinned runtime extent of the domain in whole
  // 2 MiB steps, so every materialization's check against the budget sees
  // it and growing state can take it back through the reclaim order.
  // Unset (a harness), nothing is charged: plans and graphs are only
  // counted, as before the runtime set its floor.
  void SetHostFloor(std::uint64_t floor) { host_floor_ = floor; }
  // What a charge or a pinned allocation that does not fit asks first: free
  // at least `needed` bytes through the node's one reclaim order (the
  // runtime's), between completed jobs, sparing what the step under way
  // holds; all of it or nothing. What asks: a plan (a cache charge a step
  // needs) or a graph's capture (an optional cache charge, which takes
  // only what costs less to restore than a graph) displace only other
  // plans and graphs, never conversation state; pinned staging (a turn
  // checkpoint's, a snapshot's) may spill idle conversations too.
  // Returns what it freed. Unset, nothing is reclaimed.
  enum class ReclaimFor : std::uint8_t { kPlan, kGraph, kStaging };
  using Reclaimer = std::function<std::uint64_t(std::uint64_t needed, ReclaimFor what)>;
  void SetReclaimer(Reclaimer reclaimer) { reclaimer_ = std::move(reclaimer); }
  // Charges `bytes` more (on the driver's thread): false, charging
  // nothing, when they do not fit beside the occupancy even after the
  // reclaimer ran, unless `required` (what a step needs to run at all),
  // which is always taken: the excess then refuses later growth until a
  // reclaim gives it back.
  bool ChargeHost(std::uint64_t bytes, bool required);
  void UnchargeHost(std::uint64_t bytes);
  // Every model's counted bytes, and the part charged inside the budget.
  std::uint64_t host_counted() const { return host_total_; }
  std::uint64_t host_charged() const { return host_charged_; }
  // How often a required charge went past the budget (the floor too small
  // for a step's plans), for the runtime's log.
  std::uint64_t host_overcharges() const { return host_overcharges_; }
  // The budget less the domain's occupancy, now (0 if it is past it).
  std::expected<std::uint64_t, std::string> FreeBytes();

  void AddSpan(const Span& span) { spans_.push_back(span); }
  void EraseSpans(const std::function<bool(const Span&)>& which) { std::erase_if(spans_, which); }
  void SortSpans();
  // Every byte lies in cataloged spans of device memory of one class,
  // each `owner`'s or shared; returns that class. Normally also requires
  // residency. A live-state tensor may describe its entire virtual cache:
  // resident=false checks that reservation, while its model materializes
  // the actual read/write ranges before dispatch. Read without concurrent
  // page-in or eviction.
  std::optional<catalog::MemoryClass> Covered(std::uint64_t address, std::uint64_t bytes, int owner,
                                              bool resident = true) const;

  // Posts a program and waits for it to be destroyed. It refers to `done`,
  // and a job it queues may refer to the caller's frame, so this never
  // returns while either may still run: past the patience it cancels the
  // request and waits for the drain, and aborts the process if even that
  // does not come.
  Status Post(std::unique_ptr<scheduler::TaskProgram> program, scheduler::ProgramDone& done,
              std::string_view what);
  Status Await(scheduler::ProgramDone& done, std::string_view what, std::uint64_t request);
  // Posts without waiting (with threads): the request, to cancel or await.
  std::uint64_t Submit(std::unique_ptr<scheduler::TaskProgram> program);
  void Cancel(std::uint64_t request);

  Status Load(std::vector<catalog::ExtentId> extents, std::string what,
              std::vector<LoadStats>& log);
  Status Evict(std::vector<catalog::ExtentId> extents, scheduler::EvictOptions options = {});
  // A full swap (SwapProgram): `out` evicted, with their backing handed
  // to `in`'s loads if `handoff`, then `in` materialized.
  Status Swap(std::vector<catalog::ExtentId> out, const catalog::Closure& in, bool handoff,
              scheduler::SwapReport& report);
  // The scheduler's counters, read on its thread.
  std::expected<scheduler::SchedulerStats, std::string> Stats();
  // Requests and pieces the storage lane has handed io_uring so far.
  std::uint64_t requests() const { return counting_->requests.load(); }
  // One job on `stream` over `closure`: a step of the request open there
  // (refused unless every entry of its closure, extent and contents, is
  // the request's), or else a program of its own that materializes and
  // leases the closure for it (RunProgram). Returns once the job's fence
  // completed. The node has one driver: a swap or eviction cannot be
  // asked for while a step is in flight.
  Status Job(const catalog::Closure& closure, scheduler::DeviceJob job, std::string_view what,
             std::uint32_t stream);
  // Opens a request on `stream` (the header's Requests): returns once its
  // task holds its lease on `closure` (copied). Refused if one is open
  // there already.
  Status BeginRequest(std::uint32_t stream, const catalog::Closure& closure, std::string_view what);
  // If a request is open, replaces its lease between completed steps.
  // Growing state calls this only when its set of initialized extents changes.
  Status RefreshRequest(std::uint32_t stream, const catalog::Closure& closure);
  // Ends it: its lease released (no step is in flight between Jobs), its
  // extents resident. Its task's failure, if it failed.
  Status EndRequest(std::uint32_t stream);
  bool InRequest(std::uint32_t stream) const { return requests_.contains(stream); }
  // `body` run as one request on `stream` over `closure` (BeginRequest,
  // then EndRequest whatever `body` returned): its failure, else the end's.
  Status WithRequest(std::uint32_t stream, const catalog::Closure& closure, std::string_view what,
                     const std::function<Status()>& body);
  // The stream's StepTimes since the last call, which resets them.
  StepTimes TakeTimes(std::uint32_t stream);
  // The scheduler's wake flag and the device lane, for tests of the
  // runtime wake (whether they are told to poll ahead of a step's end).
  const base::WakeFlag& wake() const { return wake_; }
  const scheduler::DeviceService& device_lane() const { return *device_lane_; }
  // Runs `call` on the scheduler's thread.
  Status Call(std::function<Status()> call, std::string_view what);
  // Makes room for `closure` under the budget and materializes it
  // (AcquireProgram), never evicting the shared workspace. `over_budget`,
  // if given, says whether a refusal was the budget's (WorkError::
  // kOverBudget: the closure does not fit beside what is leased), a clean
  // refusal before any victim was chosen.
  Status Acquire(const catalog::Closure& closure, scheduler::AcquireReport& report,
                 std::string_view what, bool* over_budget = nullptr);
  // Copies `bytes` between device memory at `device` and pinned host memory
  // at `host` (to the host if `to_host`), as one job on `stream` over
  // `closure`, which must hold the device range (a step of the request
  // open there, if one is). Returns once the copy's fence completed.
  Status Copy(std::uint32_t stream, const catalog::Closure& closure, std::uint64_t device,
              void* host, std::uint64_t bytes, bool to_host, std::string_view what);

 private:
  // A request open on a stream: its task's channel and ProgramDone, and a
  // sorted copy of its closure.
  Status MapRegion(Mapped& mapped, std::string name, std::uint64_t bytes,
                   providers::BackingKind kind, catalog::MemoryClass memory_class,
                   catalog::Recovery recovery, int owner, bool resident);
  struct OpenRequest {
    std::string what;
    catalog::Closure closure;
    std::uint64_t request = 0;
    scheduler::ProgramDone done;
    scheduler::RequestChannel channel;
    base::Expectation walls;  // its steps' walls, for the driver's wait (Step)
  };
  // A job's times on the device lane's thread, read once it has retired.
  struct Timing {
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point queued;
    bool ran = false;
  };

  void Round();
  // `job`, timed into `timing` and between the stream's events.
  scheduler::DeviceJob Timed(scheduler::DeviceJob job, std::uint32_t stream, Timing& timing);
  void Note(std::uint32_t stream, std::chrono::steady_clock::time_point called,
            const Timing& timing);
  Status Step(std::uint32_t stream, OpenRequest& open, const catalog::Closure& closure,
              scheduler::DeviceJob job, std::string_view what);
  void Signal(std::uint64_t request);
  // Ends every open request holding any of `extents`; how many.
  std::expected<std::uint64_t, std::string> EndRequestsOver(
      std::span<const catalog::ExtentId> extents);

  NodeSettings settings_;
  std::unique_ptr<providers::VmmProvider> memory_;
  std::unique_ptr<providers::DeviceExecution> execution_;
  std::unique_ptr<providers::Storage> storage_;
  std::unique_ptr<CountingStorage> counting_;  // over storage_: what the storage lane calls
  std::vector<providers::StreamId> streams_;   // compute streams, then the copy stream
  std::size_t device_class_ = 0;
  std::size_t host_class_ = 0;

  catalog::Catalog catalog_;
  catalog::DomainId domain_;
  std::vector<Span> spans_;  // sorted by base once setup ends
  base::Bytes budget_;

  Mapped zone_;
  Mapped activations_;
  Mapped pool_;
  std::vector<void*> pinned_;
  // Left to the process's end on purpose (RetireRing): never freed.
  std::vector<void*> kept_pinned_;
  std::vector<providers::Storage*> kept_rings_;

  // The host memory charged beside the catalog (ChargeHost).
  bool Recharge(std::uint64_t total, bool force, std::uint64_t* shortfall);
  std::uint64_t host_floor_ = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t host_total_ = 0;
  std::uint64_t host_charged_ = 0;  // the extent's size: whole extents past the floor
  std::uint64_t host_overcharges_ = 0;
  catalog::ExtentId host_extent_;
  Reclaimer reclaimer_;
  bool reclaiming_ = false;
  // The reclaimer asked for `needed` (ChargeHost, Pinned); what it freed.
  std::uint64_t AskReclaim(std::uint64_t needed, ReclaimFor what);
  // The bytes past the budget an allocation of `bytes` would take now (0:
  // it fits), on the scheduler's thread.
  std::uint64_t Shortfall(std::uint64_t bytes) const;

  // ReserveStaging's buffer.
  void* staging_ = nullptr;
  std::uint64_t staging_bytes_ = 0;
  bool staging_taken_ = false;

  // Before the scheduler, so it outlives the programs that refer to it.
  std::map<std::uint32_t, std::unique_ptr<OpenRequest>> requests_;  // by stream
  base::WakeFlag wake_;
  std::unique_ptr<scheduler::CompletionBoard> board_;
  std::unique_ptr<scheduler::StorageService> storage_lane_;
  std::unique_ptr<scheduler::DeviceService> device_lane_;
  std::unique_ptr<scheduler::BackingService> backing_lane_;
  std::unique_ptr<scheduler::DeviceService> copy_lane_;  // settings_.copy_lane
  std::unique_ptr<scheduler::Scheduler> scheduler_;
  std::vector<std::jthread> threads_;  // the scheduler first
  std::optional<std::expected<void, scheduler::Fault>> stopped_;
  std::uint64_t request_ = 0;
  bool torn_down_ = false;
  std::vector<StepTimes> times_;  // by compute stream
  // By compute stream: the marks before and after its job.
  std::vector<std::pair<providers::TimingMark, providers::TimingMark>> events_;
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_PAGED_NODE_H_
