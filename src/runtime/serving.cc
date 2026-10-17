// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/serving.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "base/build_info.h"
#include "base/check.h"
#include "base/report.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/paged_weights.h"
#include "execution/adaptive_depth.h"
#include "execution/adaptive_wave_mode.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/step_program.h"
#include "memory/reclaim.h"
#include "planner/step_shape.h"
#include "platform/crash_policy.h"
#include "platform/files.h"
#include "platform/host_probe.h"
#include "platform/kept_files.h"
#include "platform/memory_pressure.h"
#include "providers/device_runtime.h"
#include "runtime/component_build.h"
#include "runtime/component_image_build.h"
#include "runtime/engine_image.h"
#include "runtime/engine_llm.h"
#include "runtime/intake_limits.h"
#include "runtime/memory_guard.h"
#include "runtime/prefill.h"
#include "runtime/runtime.h"
#include "runtime/serving_support.h"
#include "runtime/swap_room.h"
#include "scheduler/programs.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::runtime {
namespace {

using serving_support::Errno;
using serving_support::Error;
using serving_support::KeptRegionsOf;
using serving_support::kExtent;
using serving_support::NowUnixMs;
using serving_support::Seconds;

namespace fs = std::filesystem;
namespace ja = llmp::artifact;

// Extents the page-in observer tracks (the M3 models use about 100,000).
constexpr std::size_t kObservedExtents = std::size_t{1} << 18U;
// Each model's settings (prefill chunk, request slots, drafting, waves,
// floors, sampling) resolve at registration in model_settings.h's three
// layers (D-103); their fallbacks are its constants, which name what the
// engine does too.
static_assert(Llm::kMaxBranches == engine::kMaxRequestSlots);
static_assert(config::kMaxModelSlots == engine::kMaxRequestSlots);
static_assert(config::kMaxWaveCostWidths + 1 == execution::AdaptiveWaveMode::kMaxWidth);

// The reclaim order's cost of idle conversation state (Server::Reclaim):
// its write-back and its restore, at the node's rates measured so far, and
// before any at the rates measured on a GB10 (spark-b, DeepSeek's four
// slots' state, 1.42 GB written back at 11.0 GB/s and read again at 14.5
// GB/s; docs/experiments/memory-pressure). Dropping it instead (no spill
// budget) costs its recomputation: its tokens at the model's
// recompute_ms_per_token.
constexpr double kSpillBytesPerSecond = 11.0e9;
constexpr double kRestoreBytesPerSecond = 14.5e9;
// Prepared weight page-in measured on the GB10 (D-055, memory-pressure).
constexpr double kWeightReadBytesPerSecond = 13.3e9;
// A graph's capture and instantiation a GiB counted, measured on a GB10
// (0.33–0.45 s in the services' logs): what a graph's capture weighs
// against before any graph is held (Server::Reclaim).
constexpr double kGraphSecondsPerGiB = 0.36;
// How often the driver's maintenance looks (between units and when idle).
constexpr auto kMaintenanceInterval = std::chrono::milliseconds(250);

// Diagnostic caps alone enable these clocks; ordinary serving adds no clock
// reads or scheduler calls. The driver owns every accumulator.
class SwapPhaseTimer {
 public:
  explicit SwapPhaseTimer(double* total)
      : total_(total), begin_(total != nullptr ? Clock::now() : Clock::time_point{}) {}
  SwapPhaseTimer(const SwapPhaseTimer&) = delete;
  SwapPhaseTimer& operator=(const SwapPhaseTimer&) = delete;
  ~SwapPhaseTimer() {
    if (total_ != nullptr) *total_ += Seconds(Clock::now() - begin_);
  }

 private:
  double* total_;
  Clock::time_point begin_;
};

std::uint64_t WeightHandle(catalog::ExtentId extent) {
  return (static_cast<std::uint64_t>(extent.generation()) << 32U) | extent.index();
}

catalog::ExtentId WeightExtent(std::uint64_t handle) {
  return {static_cast<std::uint32_t>(handle), static_cast<std::uint32_t>(handle >> 32U)};
}

// ------------------------------------------- conversations kept (D-105)

// What this build is, for a kept record: its version and commit, and the
// executable file's identity, so another build, or this one rebuilt or
// reinstalled, never adopts what it did not write.
std::string KeptBuild() {
  const base::BuildInfo& info = base::GetBuildInfo();
  const auto stamp = platform::RunningExecutableStamp();
  const std::string exe =
      stamp ? std::format("{}:{}:{}:{}:{}", stamp->device, stamp->inode,
                          stamp->generation ? std::to_string(*stamp->generation) : "-", stamp->size,
                          stamp->modified_ns)
            : std::string("unknown");
  return std::format("{};{}{};{};{};exe={}", info.version, info.commit,
                     info.modified ? "+modified" : "", info.sdk, info.target, exe);
}

}  // namespace

// ---------------------------------------------------------------- memory

MemorySampler::MemorySampler() : start_(Available()) {
  low_ = start_;
  all_ = start_;
  thread_ = std::jthread([this] {
    if (auto stack = platform::InstallThreadSignalStack(); !stack) {
      std::println(stderr, "the memory sampler runs without a signal stack: {}", stack.error());
    }
    Loop();
  });
}

MemorySampler::~MemorySampler() {
  stop_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

std::uint64_t MemorySampler::Available() { return platform::AvailableMemoryBytes().value_or(0); }

void MemorySampler::Reset() { low_ = Available(); }

void MemorySampler::Loop() {
  const auto lower = [](std::atomic<std::uint64_t>& to, std::uint64_t now) {
    std::uint64_t seen = to.load();
    while (now < seen && !to.compare_exchange_weak(seen, now)) {
    }
  };
  while (!stop_) {
    const std::uint64_t now = Available();
    if (now != 0) {
      lower(low_, now);
      lower(all_, now);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void ResidentTimes::Staged(catalog::ExtentId extent, scheduler::PageInEvent event) {
  events_.fetch_add(1, std::memory_order_relaxed);
  if (event == scheduler::PageInEvent::kResident && extent.index() < at_.size()) {
    at_[extent.index()] = Clock::now();
  }
}

Clock::time_point ResidentTimes::Latest(std::span<const catalog::ExtentId> extents) const {
  Clock::time_point latest{};
  for (const catalog::ExtentId extent : extents) {
    if (extent.index() < at_.size()) {
      latest = std::max(latest, at_[extent.index()]);
    }
  }
  return latest;
}

// ---------------------------------------------------------------- the server

Server::Server(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               const ServingOptions& options, std::FILE* log)
    : config_(config),
      roles_(roles),
      options_(options),
      log_(log),
      times_(kObservedExtents),
      node_({.compute_streams = std::max<std::size_t>(config.models.size(), 1),
             .slots = engine::kPagedSlots,
             .inline_lanes = false,
             .coalesce = false,
             .copy_lane = true,
             .lazy_handoff = options.lazy_handoff,
             .zero_state = options.zero_state,
             .handle_reserve = options.handle_reserve.value_or(engine::kHandleReserve),
             .slot_bytes = engine::kSlabSlotBytes,
             .observer = &times_,
             .poll_window = std::nullopt,
             .spin_ahead = std::nullopt,
             .quiet = std::chrono::minutes(10),
             .hold_reads = &hold_reads_,
             .hold_cancellable = options.hold_cancellable}),
      retention_(std::chrono::hours(config.memory.retention_hours)),
      spill_budget_(std::uint64_t{config.memory.spill_budget_gib} << 30U) {}

Server::~Server() {
  if (started_ && !torn_down_) {
    if (auto stopped = TearDown(); !stopped) {
      Log("stopping: " + stopped.error());
    }
  }
}

void Server::Log(std::string_view text) {
  const std::string line = std::format("llmp-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log_);
  (void)std::fflush(log_);
}

Status Server::Make(const config::ModelEntry& entry, const ModelSettings& settings, int index) {
  // A model on the component engine: the adapter of its spec's pipeline
  // kind (its settings', from its stored spec). Its architecture's serving
  // recipe (spec/mapping.h) says whether it serves on the derived shared
  // route (its domain derived from the actual spec, state and owners);
  // every LLM serves under the automatic startup payment by default
  // (Start).
  if (!entry.composition) {
    const spec::ServingRecipe* recipe = RecipeOf(settings.architecture);
    if (recipe == nullptr)
      return Error(std::format("model {}: no runner for architecture {}", entry.name,
                               settings.architecture));
    const bool shared = recipe->shared_route;
    switch (settings.pipeline) {
      case spec::PipelineKind::kLlm: {
        if (AutomaticStartup() && !compiler_)
          compiler_ = std::make_shared<engine::SharedCompilerAllowance>();
        auto llm = std::make_unique<EngineLlm>(
            node_, entry, settings, roles_, index,
            ComponentLlmBuild(node_, roles_.installed, entry, index,
                              {.target_policy = {},
                               .drafter_policy = options_.diagnostic_drafter_policy,
                               .shared_route = shared,
                               .capture_first_verify = options_.diagnostic_capture_first_verify,
                               .automatic_startup_cap = AutomaticStartup() ? startup_cap_ : 0,
                               .shared_compiler = compiler_,
                               .prefill_lookahead = options_.prefill_lookahead,
                               .prepare_state = options_.prepare_state,
                               .state_only_prefill = options_.state_only_prefill,
                               // A route without qualified envelopes serves only derived;
                               // others derive their joined and plain envelopes.
                               .derive_execution_domain =
                                   shared && (!recipe->qualified_envelopes ||
                                              settings.max_slots.value > 1 || !entry.drafter)}));
        if (options_.diagnostic_capture_first_verify)
          Log(std::format("model {}: experimental Verify capture beside first eager run",
                          entry.name));
        if (options_.diagnostic_skip_turn_checkpoints) {
          llm->set_turn_checkpoints(false);
          Log(std::format("model {}: saves no turn checkpoint (a speed gate's diagnostic)",
                          entry.name));
        }
        models_.push_back(std::move(llm));
        return {};
      }
      case spec::PipelineKind::kImage:
        // An image pipeline is served from its composition (below), never
        // from one of its component artifacts.
        return Error(
            std::format("model {}: an image pipeline is served from its composition; configure the "
                        "composition's ID, not a component artifact's",
                        entry.name));
      default:
        return Error(std::format("model {}: no serving adapter for the {} pipeline kind yet",
                                 entry.name, spec::PipelineName(settings.pipeline)));
    }
  }
  // Every other entry is a composition: an image pipeline.
  const auto prepared = std::ranges::find_if(
      prepared_images_, [&](const auto& source) { return source->alias == entry.name; });
  if (prepared != prepared_images_.end()) {
    if ((*prepared)->owner != index)
      return Error("prepared image owner differs from its registered Server index");
    models_.push_back(std::make_unique<EngineImage>(
        node_, entry, settings, index,
        ComponentImageGenerationBuild(node_, index, (*prepared)->source,
                                      (*prepared)->construction)));
    return {};
  }
  if (options_.image_noise.empty()) {
    Log(
        std::format("model {}: not registered: an image pipeline needs --image-noise (its "
                    "initial latents) in this build",
                    entry.name));
    return {};
  }
  // Start prepaid the readers' loan (automatic, the default, or a
  // diagnostic's explicit bytes) under its execution ceiling.
  if (startup_parser_loan_ == 0 || startup_cap_ == 0)
    return Error("image serving requires paid startup host memory");
  models_.push_back(std::make_unique<EngineImage>(
      node_, entry, settings, index,
      ComponentImageServingBuild(node_, roles_.installed, *entry.composition, index,
                                 options_.image_prompt, options_.image_noise, startup_parser_loan_,
                                 startup_cap_)));
  return {};
}

Status Server::SaveSnapshot(Llm& model) {
  if (!snapshot_enabled_) return Error("the diagnostic snapshot is disabled");
  if (!snapshot_store_) {
    snapshot_store_ =
        std::make_unique<SnapshotStore>(node_, node_.budget().value(), [this](std::uint64_t bytes) {
          return Reclaim(bytes, true, "a state snapshot's staging") != 0;
        });
  }
  snapshot_ = {};
  snapshot_model_ = nullptr;
  auto saved = model.default_branch().CaptureState(*snapshot_store_);
  if (!saved) return Error(saved.error());
  snapshot_ = std::move(*saved);
  snapshot_model_ = &model;
  return {};
}

Status Server::RestoreSnapshot(Llm& model) {
  if (!snapshot_enabled_ || !snapshot_store_ || snapshot_model_ != &model) {
    return Error("the diagnostic snapshot is disabled or belongs to another model");
  }
  return model.default_branch().RestoreState(*snapshot_store_, snapshot_);
}

std::expected<base::Sha256Digest, std::string> Server::SnapshotDigest() const {
  if (!snapshot_enabled_ || !snapshot_store_) {
    return std::unexpected("the diagnostic snapshot is disabled");
  }
  return snapshot_store_->Digest(snapshot_);
}

Status Server::Start(bool snapshot) {
  if (started_) {
    return Error("the server started already");
  }
  started_ = true;
  if (config_.models.empty()) {
    return Error("the configuration names no models ([models.<name>], D-096)");
  }
  std::optional<catalog::ExtentId> bootstrap;
  std::uint64_t startup_allowance = options_.diagnostic_startup_host_bytes;
  if (AutomaticStartup()) {
    if (options_.diagnostic_startup_host_bytes != 0)
      return Error("automatic startup takes no explicit startup host bytes");
    // The already parsed configuration is caller-owned. Its actual capacities
    // bound the copied entries/settings; no reopening or file-size estimate.
    std::uint64_t configured = 0;
    const auto add = [&](std::uint64_t count) {
      return !__builtin_add_overflow(configured, count, &configured);
    };
    for (const auto& entry : config_.models) {
      if (!add(sizeof(entry)) || !add(entry.name.capacity() + 1))
        return Error("startup configuration inventory overflows");
      for (const auto* text : {&entry.artifact, &entry.composition, &entry.drafter})
        if (*text && !add((*text)->capacity() + 1))
          return Error("startup identity inventory overflows");
      for (const auto* path : {&entry.tokenizer, &entry.chat_template})
        if (*path && !add((*path)->native().capacity() + 1))
          return Error("startup path inventory overflows");
      for (const auto& [name, value] : entry.overrides) {
        if (!add(name.capacity() + 1 + sizeof(value) + 128))
          return Error("startup setting inventory overflows");
        if (const auto* text = std::get_if<std::string>(&value); text && !add(text->capacity() + 1))
          return Error("startup text setting inventory overflows");
        if (const auto* list = std::get_if<std::vector<double>>(&value)) {
          std::uint64_t storage = 0;
          if (__builtin_mul_overflow(static_cast<std::uint64_t>(list->capacity()),
                                     std::uint64_t{sizeof(double)}, &storage) ||
              !add(storage))
            return Error("startup list setting inventory overflows");
        }
      }
    }
    auto bytes = StartupBootstrapBytes(config_.models.size(), configured);
    if (!bytes) return Error(bytes.error());
    startup_allowance = *bytes;
  }
  const auto& prepared = options_.diagnostic_prepared_images;
  if (!prepared.empty()) {
    if (!node_.open_for_setup() || AutomaticStartup() || startup_allowance == 0 ||
        prepared.size() > config_.models.size())
      return Error("prepared images require an opened stopped node and separate startup payment");
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      const auto& input = prepared[i];
      if (!input || input->node != &node_ || !input->source || !input->source->composition ||
          !input->source->keep || !input->construction || input->construction->used() == 0 ||
          !input->construction->HeldBy(node_.catalog(), node_.domain()))
        return Error("prepared image source has no held payment on this Server node");
      const auto entry = std::ranges::find(config_.models, input->alias, &config::ModelEntry::name);
      if (entry == config_.models.end() || !entry->composition ||
          *entry->composition != input->source->composition->id() || input->owner < 0 ||
          static_cast<std::size_t>(input->owner) !=
              static_cast<std::size_t>(std::distance(config_.models.begin(), entry)))
        return Error("prepared image alias/composition/owner differs from configured source");
      for (std::size_t j = 0; j < i; ++j)
        if (prepared[j]->alias == input->alias) return Error("duplicate prepared image alias");
    }
  }
  if (!prepared.empty()) {
    for (const auto& entry : config_.models)
      if (entry.composition && std::ranges::none_of(prepared, [&](const auto& source) {
            return source->alias == entry.name;
          }))
        return Error("prepared image mode has no source for a configured image");
    if (prepared.size() > startup_allowance / sizeof(prepared.front()))
      return Error("prepared image owner snapshot exceeds startup payment");
  }
  bool prepaid = false;
  if (const auto bytes = startup_allowance; bytes != 0) {
    const auto available = MemorySampler::Available();
    // The ordinary route's ceiling is the node's measured headroom (D-102);
    // a diagnostic cap only lowers it. Unknown availability refuses.
    const auto ceiling = MeasuredStartupCeiling(available);
    if (AutomaticStartup() && ceiling == 0)
      return Error("startup needs known physical memory past the uncounted margin");
    const auto cap = options_.diagnostic_budget_cap_bytes
                         ? std::min(*options_.diagnostic_budget_cap_bytes, ceiling)
                         : (AutomaticStartup() ? ceiling : 0);
    if (auto check = CheckStartupHostAllowance(bytes, cap, available); !check) return check;
    startup_parser_loan_ = bytes;
    startup_cap_ = cap;
    if (prepared.empty()) {
      if (auto opened = node_.Open(); !opened) return opened;
    }
    const auto before = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    if (before > cap || bytes > cap - before)
      return Error("startup host allowance exceeds the execution budget");
    // A real pinned catalog extent, not ChargeHost's pre-Start counter.
    // It stays alive with the complete server on failed setup/retirement.
    auto extent = node_.catalog().AddExtent({.domain = node_.domain(),
                                             .memory_class = catalog::MemoryClass::kRuntime,
                                             .recovery = catalog::Recovery::kPinned,
                                             .size = base::Bytes(bytes),
                                             .content = {}},
                                            true);
    if (!extent) return Error("could not catalog the startup host allowance");
    const auto after = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    if (after < before || after - before != bytes)
      return Error("startup host allowance has no physical occupancy");
    if (AutomaticStartup()) bootstrap = *extent;
    prepaid = true;
  }
  // The parser loan above also covers this bounded alias/owner snapshot. Shared
  // source and construction owners then survive every failed Start/teardown.
  prepared_images_ = prepared;
  // Every model's settings (D-103: derived from its artifacts, calibrated on
  // this machine, or overridden), its checkpoint's ceiling among them,
  // resolved before opening the device node or constructing runners. An
  // invalid model later in the list must not cause any earlier model's
  // resources to be allocated either.
  std::vector<ModelSettings> settings;
  std::vector<Llm::CalibrationRecord> calibrations;
  settings.reserve(config_.models.size());
  if (const std::string note = CalibrationDirectoryNote(roles_.state); !note.empty()) {
    Log(std::format("warning: the calibration directory {}/{}: {}; used as found",
                    roles_.state.string(), config::kCalibrationDirectory, note));
  }
  for (const config::ModelEntry& entry : config_.models) {
    auto facts = ReadArtifactFacts(entry, roles_.installed);
    if (!facts) {
      return Error(std::format("model {}: {}", entry.name, facts.error()));
    }
    // Its calibration on this machine, if one is recorded for this device,
    // driver and build (calibration.h); a reading costs a small file.
    Llm::CalibrationRecord record;
    if (!entry.composition) {
      // Keyed by the settings its measurements depend on, as they resolve
      // without calibration.
      auto uncalibrated =
          ResolveSettings(entry, *facts, nullptr, options_.plain, options_.trained_max_context);
      if (!uncalibrated) {
        return Error(std::format("model {}: {}", entry.name, uncalibrated.error()));
      }
      record.key = CalibrationKeyOf(entry, *uncalibrated);
      const CalibrationRead read = ReadCalibration(roles_.state, record.key);
      Log(std::format("model {}: calibration {}", entry.name, read.note));
      if (read.calibration) {
        record.known = *read.calibration;
      }
    }
    auto resolved = ResolveSettings(entry, *facts, record.known.empty() ? nullptr : &record.known,
                                    options_.plain, options_.trained_max_context);
    if (!resolved) {
      return Error(std::format("model {}: {}", entry.name, resolved.error()));
    }
    for (const std::string& ignored : resolved->ignored) {
      Log(std::format("model {}: ignored {}", entry.name, ignored));
    }
    for (const std::string& warning : resolved->warnings) {
      Log(std::format("model {}: warning: {}", entry.name, warning));
    }
    settings.push_back(std::move(*resolved));
    calibrations.push_back(std::move(record));
  }
  if (!prepaid) {
    if (auto r = node_.Open(); !r) return r;
  }
  // Each model's stream and owner: its place among the registered.
  for (std::size_t i = 0; i < config_.models.size(); ++i) {
    const std::size_t before = models_.size();
    if (auto r = Make(config_.models[i], settings[i], static_cast<int>(models_.size())); !r) {
      return r;
    }
    if (models_.size() > before && models_.back()->llm()) {
      static_cast<Llm&>(*models_.back()).calibration_record() = std::move(calibrations[i]);
    }
  }
  if (models_.empty()) {
    return Error("no configured model could be registered");
  }
  std::uint64_t activations = 0;
  std::uint64_t pool = 0;
  std::uint64_t largest = 0;
  std::uint64_t largest_weights = 0;  // the largest weights alone
  std::uint64_t host_inputs = 0;
  std::uint64_t plans = 0;
  for (const auto& m : models_) {
    const auto started = Clock::now();
    StartProgress();
    if (auto r = m->Setup(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
    activations = std::max(activations, m->activations_needed());
    host_inputs = std::max(host_inputs, m->host_input_bytes());
    plans = std::max(plans, m->plan_floor_bytes());
    pool = std::max(pool, m->pool_needed());
    // What it holds resident while it runs: its weights and its own prefix
    // of the shared workspace, which pages with it (MapWorkspace below), not
    // the largest weights beside the largest workspace of any model.
    const std::uint64_t footprint =
        ActiveFootprint(m->weights().size() * kExtent, m->activations_needed(), m->pool_needed());
    largest = std::max(largest, footprint);
    largest_weights = std::max<std::uint64_t>(largest_weights, m->weights().size() * kExtent);
    std::string chunks;
    if (m->llm()) {
      const auto& l = static_cast<Llm&>(*m);
      chunks = std::format("; prefill chunks of {} rows", l.max_rows());
      if (l.configured_rows() && *l.configured_rows() != l.max_rows()) {
        chunks += std::format(
            " (prefill_chunk {}: capped at what the model allows below its context of {}, in "
            "whole 8-row tiles)",
            *l.configured_rows(), l.context());
      }
    }
    Log(std::format(
        "model {}: set up in {:.2f} s, {} weight extents ({:.2f} GB read a load){}; workspace "
        "{:.2f} GiB of activations and {:.2f} GiB of pool, {:.2f} GiB resident while active",
        m->name(), Seconds(Clock::now() - started), m->weights().size(),
        static_cast<double>(m->weight_read_bytes()) / 1e9, chunks,
        static_cast<double>(m->activations_needed()) / (1ULL << 30U),
        static_cast<double>(m->pool_needed()) / (1ULL << 30U),
        static_cast<double>(footprint) / (1ULL << 30U)));
    // Each effective setting and its source (D-103; `llmp-runtime
    // settings` lists them with what each came from).
    Log(std::format("model {}: settings {}", m->name(), m->settings().Summary()));
    if (const auto report = m->allocation_report(); !report.empty()) {
      Log(std::format("allocation model {}: {}", m->name(), report));
    }
    if (const auto report = m->plan_report(); !report.empty()) {
      Log(std::format("model {}: plans {}", m->name(), report));
    }
    if (const auto reason = m->serial_reason(); !reason.empty()) {
      Log(std::format("model {}: serves one request at a time ({})", m->name(), reason));
    } else if (const auto slots = m->slots_report(); !slots.empty()) {
      Log(std::format("model {}: {}", m->name(), slots));
    }
  }
  if (bootstrap) {
    // Every alias reports only retained source/spec/chat bytes. The shared
    // transient compiler extent is counted once, not once per borrowing source.
    std::uint64_t retained = compiler_ ? compiler_->paid_bytes() : 0;
    for (const auto& model : models_) {
      if (model->startup_host_bytes() == 0 ||
          __builtin_add_overflow(retained, model->startup_host_bytes(), &retained))
        return Error("automatic startup has no complete retained metadata payment");
    }
    // All Artifact/Open/JSON/Setup locals have died. The models now own paid
    // source/compiler/chat metadata. Settings/calibration copies are released
    // before repaying the parser loan, while the scheduler is still stopped.
    std::vector<ModelSettings>().swap(settings);
    std::vector<Llm::CalibrationRecord>().swap(calibrations);
    if (!node_.catalog().ReleasePinned(*bootstrap) || !node_.catalog().RemoveExtent(*bootstrap))
      return Error("could not retire the startup parser loan");
    startup_allowance = retained;
    Log(std::format("automatic startup: parser loan retired; {} retained host bytes", retained));
  }
  const auto measured_startup_available = MemorySampler::Available();
  const auto allowance = startup_allowance;
  if (allowance && measured_startup_available <= allowance)
    return Error("startup host allowance exhausts physical headroom");
  const auto startup_available = measured_startup_available - allowance;
  const MemoryGuard startup_guard{.largest = largest,
                                  .host_inputs = host_inputs,
                                  .plans = plans,
                                  .available = startup_available,
                                  .fixed = 0,
                                  .requests = RequestFloor(config_.client)};
  // The workspace's mode, per node (2026-10-17): resident for the node's life,
  // mapped as one region before anything pages, when the whole workspace
  // fits beside the largest model's weights under the guard (with 1 GiB to
  // spare for what the guard measures again below); paged with the active
  // model otherwise. Backed piecemeal during a page-in, while the kernel
  // reclaims the page cache for the weights, the workspace ran the image's
  // denoiser products up to 5% slower on a node whose page cache was full;
  // mapped resident at the start it did not (docs/runtime-serving.md).
  const std::uint64_t resident_largest = ActiveFootprint(largest_weights, activations, pool);
  const std::uint64_t resident_need = GuardReserve(startup_guard) + resident_largest;
  const bool fits =
      startup_available == 0 || resident_need + (std::uint64_t{1} << 30U) <= startup_available;
  const bool pageable = options_.pageable_workspace.value_or(!fits);
  Log(std::format(
      "workspace: {} ({:.2f} GiB of activations and {:.2f} GiB of pool); {}",
      pageable ? "paged with the active model" : "resident for the node's life",
      static_cast<double>(activations) / (1ULL << 30U), static_cast<double>(pool) / (1ULL << 30U),
      options_.pageable_workspace
          ? "forced by the serving options"
          : std::format("the whole of it beside the largest weights needs {:.2f} GiB of the "
                        "{:.2f} GiB available{}",
                        static_cast<double>(resident_need) / (1ULL << 30U),
                        static_cast<double>(startup_available) / (1ULL << 30U),
                        fits ? "" : ", which does not fit")));
  if (!pageable) largest = resident_largest;
  // The largest footprint holds its model's workspace already (a resident
  // workspace counted whole).
  const auto startup_reserve = GuardReserve(startup_guard) + largest;
  // Unknown availability (0) bounds nothing here, as the guard skips it;
  // the execution budget bounds what is adopted once it is set (Start).
  startup_token_capacity_ = startup_available == 0 ? std::numeric_limits<std::uint64_t>::max()
                            : startup_available > startup_reserve
                                ? startup_available - startup_reserve
                                : 0;
  token_memory_.SetDriver(
      std::this_thread::get_id(),
      [this](std::uint64_t incoming) { return SetTokenMemory(incoming); }, 0,
      std::chrono::milliseconds{0});
  // Conversations the process before kept (D-105): read, checked and
  // their files named before the node registers them.
  if (auto kept = PrepareKept(); !kept) {
    return kept;
  }
  // Pageable: virtually reserved and cataloged at the largest need, each model's
  // closure naming only its own prefix, so a swap brings in what the
  // incoming model uses and evicts what only the outgoing one did. Nothing
  // of it is resident before the first activation. Resident: mapped whole
  // now, and counted in the fixed memory below.
  if (auto r = node_.MapWorkspace(activations, pool, startup_available, pageable); !r) {
    return r;
  }
  // The budget's largest term: with a resident workspace (now in the fixed
  // memory) the largest weights alone.
  if (!pageable) largest = largest_weights;
  workspace_ = activations + pool;
  snapshot_enabled_ = snapshot;
  // The budget (D-050's B): everything mapped for the node's life (the zone,
  // each model's own memory, the staging) and the largest active footprint
  // (a model's weights and its workspace), so a full swap is the only way in.
  fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  budget_ = fixed_ + largest;
  // A chunk's host-built inputs are the node's memory too, though allocated
  // per chunk and outside the catalog: counted here beside the margin, not
  // in the budget the catalog enforces. One model runs chunks at a time, so
  // the most any model's chunk builds, not their sum.
  host_inputs_ = host_inputs;
  // So are its plans and graphs, but only the floor is
  // set apart here: what one step of any model holds at once. Every plan
  // and graph past it is charged inside the budget (PagedNode::ChargeHost)
  // and given back through the reclaim order when the budget is needed.
  plans_ = plans;
  // The startup payments are catalog occupancy now, inside fixed_, which
  // the budget below counts beside this availability: counted once, there,
  // not subtracted from the measured availability as well.
  const std::uint64_t available = MemorySampler::Available();
  MemoryGuard bounds = {.largest = largest,
                        .host_inputs = host_inputs,
                        .plans = plans,
                        .available = available,
                        .fixed = fixed_,
                        .requests = 0};
  // Requests' host memory (intake_limits.h, D-102): a small floor set apart
  // here, out of the catalog's budget, so the requests a route admits never
  // take the margin the driver and the models' state count on; past it the
  // route charges what requests hold to the budget as it goes
  // (SetRequestMemory), through the reclaim order.
  request_memory_ = RequestFloor(config_.client);
  bounds.requests = request_memory_;
  const auto guard = CheckMemoryGuard(bounds);
  Log(
      std::format("allocation guard: {{\"format\":\"llmp-wave-startup-guard-v1\","
                  "\"fixed_catalog_bytes\":{},\"shared_activation_bytes\":{},"
                  "\"shared_scratch_bytes\":{},\"largest_active_footprint_bytes\":{},"
                  "\"host_input_bytes\":{},\"plan_floor_bytes\":{},\"uncounted_margin_bytes\":{},"
                  "\"available_after_fixed_bytes\":{},\"available_known\":{},"
                  "\"guard_passed\":{},\"registered_state_virtual_extent_bytes\":{},"
                  "\"request_memory_bytes\":{},\"native_token_history_bytes\":{}}}",
                  fixed_, activations, pool, largest, host_inputs, plans, kUncountedMargin,
                  available, available != 0, guard.has_value(), node_.StateCapacity(),
                  request_memory_, token_memory_.used()));
  if (!guard) {
    return std::unexpected(guard.error());
  }
  // State is charged as it grows. Keep the physical execution cap below
  // the measured available memory, leaving host-built inputs, the plans and
  // the uncounted margin outside it. A virtual context ceiling need not fit
  // before it is used; a growth that cannot fit its active closure fails.
  if (available != 0) {
    budget_ = fixed_ + ((available - GuardReserve(bounds)) / kExtent * kExtent);
  } else {
    budget_ += node_.StateCapacity();
  }
  dynamic_budget_ = budget_;
  if (options_.diagnostic_budget_cap_bytes) {
    const auto reserve = options_.handle_reserve.value_or(engine::kHandleReserve);
    if (reserve > std::numeric_limits<std::uint64_t>::max() / kExtent)
      return Error("diagnostic budget cap handle reserve overflows");
    const auto requested = *options_.diagnostic_budget_cap_bytes;
    // An automatic startup ceiling cannot enlarge ordinary admission. Keep
    // extent alignment when current measured headroom selects a smaller budget.
    const auto cap =
        AutomaticStartup() ? std::min(requested, dynamic_budget_ / kExtent * kExtent) : requested;
    if (auto checked = CheckDiagnosticBudgetCap(bounds, dynamic_budget_, cap, reserve * kExtent);
        !checked)
      return std::unexpected(checked.error());
    budget_ = cap;
    Log(std::format("diagnostic budget cap: dynamic={} selected={} bytes; requested={} bytes",
                    dynamic_budget_, budget_, requested));
  }
  if (auto r = node_.Start(base::Bytes(budget_)); !r) {
    return r;
  }
  StartProgress();
  // No model is active until its activation: none of its weight files
  // opens before.
  DeactivateIdleModels();
  for (const auto& m : models_) {
    if (auto r = m->Register(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
  }
  for (const auto& m : models_) {
    if (auto r = m->Bind(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
  }
  node_.Run();
  // Plans and graphs past the floor are charged inside the budget; a charge
  // that does not fit has the reclaim order make room first.
  node_.SetHostFloor(plans);
  node_.SetReclaimer([this](std::uint64_t needed, engine::PagedNode::ReclaimFor what,
                            std::span<const catalog::ExtentId> protect) {
    // Cache charges displace only other plans and graphs, never
    // conversation state (a graph's only what costs less to restore than a
    // graph); pinned staging (a checkpoint's, a snapshot's) is a state need
    // and may spill idle conversations.
    using For = engine::PagedNode::ReclaimFor;
    switch (what) {
      case For::kPlan:
        return Reclaim(needed, false, "a plan past the free budget", nullptr, std::nullopt, false,
                       nullptr, 0, {}, nullptr, false);
      case For::kGraph:
        return Reclaim(needed, false, "a graph past the free budget", nullptr,
                       memory::ReclaimKind::kGraph, false, nullptr, 0, {}, nullptr, false);
      case For::kStaging:
        break;
      case For::kMaterialize:
        return Reclaim(needed, false, "state materialization past the free budget", nullptr,
                       std::nullopt, false, nullptr, 0, protect);
    }
    return Reclaim(needed, true, "pinned staging past the free budget");
  });
  // The turn checkpoints' staging, set apart once so a budget full of
  // reclaimable plans, graphs and idle state never starves a capture or a
  // restore (they use it one at a time).
  if (std::ranges::any_of(models_, [](const auto& m) { return m->llm(); })) {
    if (auto r = node_.ReserveStaging(engine::CheckpointStagingBytes()); !r) {
      return Error("setting apart the turn checkpoints' staging: " + r.error());
    }
  }
  // The state room: what the budget leaves the conversations beside the
  // fixed memory and the largest active footprint.
  const std::uint64_t room = budget_ > fixed_ + largest ? budget_ - fixed_ - largest : 0;
  state_room_ = room;
  Log(std::format(
      "serving {} models; budget {:.2f} GiB ({:.2f} GiB fixed; the workspace {:.2f} GiB {}; "
      "host-built "
      "chunk inputs {:.2f} GiB, a step's plans {:.2f} GiB and {:.2f} GiB of request memory "
      "beside it); conversation state room {:.2f} GiB beside the largest weights and workspace; "
      "{:.2f} GiB "
      "available; idle conversations spill, kept {} h within {} GiB",
      models_.size(), static_cast<double>(budget_) / (1ULL << 30U),
      static_cast<double>(fixed_) / (1ULL << 30U), static_cast<double>(workspace_) / (1ULL << 30U),
      node_.workspace_pageable()
          ? "virtually reserved, each model's part of it resident while that model is active"
          : "resident, in the fixed memory",
      static_cast<double>(host_inputs_) / (1ULL << 30U),
      static_cast<double>(plans_) / (1ULL << 30U),
      static_cast<double>(request_memory_) / (1ULL << 30U),
      static_cast<double>(room) / (1ULL << 30U), static_cast<double>(available) / (1ULL << 30U),
      config_.memory.retention_hours, config_.memory.spill_budget_gib));
  for (const auto& m : models_) {
    if (m->llm()) {
      static_cast<Llm&>(*m).SetTokenMemory(token_memory_);
      static_cast<Llm&>(*m).set_retention(retention_);
      static_cast<Llm&>(*m).set_checkpoint_admission(
          [this](std::uint64_t bytes) { return TrimTurnCheckpoints(bytes); });
      static_cast<Llm&>(*m).set_log([this](std::string_view text) { Log(text); });
    } else {
      static_cast<Image&>(*m).set_log([this](std::string_view text) { Log(text); });
    }
  }
  token_catalog_ready_ = true;
  if (!SetTokenMemory(0)) {
    // Only when availability was unknown at the start can what was decoded
    // exceed the budget: those conversations are not adopted (their records
    // removed, their files emptied), rather than every start failing.
    Log("kept conversations' token histories do not fit the execution budget; none is adopted");
    for (PendingAdoption& pending : adoptions_) {
      if (auto branch = pending.model->branch(pending.record.slot); branch) {
        std::ignore = (*branch)->ReleaseIdleState();
      }
    }
    adoptions_.clear();
    if (!SetTokenMemory(0)) {
      return Error("kept token histories do not fit the execution budget");
    }
  }
  AdoptKept();
  return {};
}

Status Server::PrepareKept() {
  if (!options_.keep_conversations) {
    return {};  // a command: what the service kept is left alone
  }
  const bool any_llm = std::ranges::any_of(models_, [](const auto& m) { return m->llm(); });
  const auto stamp = platform::RunningExecutableStamp();
  const bool wanted = config_.memory.keep_across_restart && spill_budget_ != 0;
  const bool keep = any_llm && wanted && stamp.has_value();
  const int spill = ::open(roles_.spill.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (spill < 0) {
    return Error(std::format("the spill directory {}: {}", roles_.spill.string(), Errno(errno)));
  }
  const std::string root_name(kept::kDirectory);
  if (!keep) {
    // Nothing is kept with a spill budget of 0 (idle state is dropped) or
    // with [memory] keep_across_restart false, and nothing kept before
    // stays (D-014: its retention is explicit).
    if (any_llm && wanted) {
      Log("conversations are not kept across a restart: the executable's identity is unknown");
    } else if (any_llm && !config_.memory.keep_across_restart) {
      Log("conversations are not kept across a restart ([memory] keep_across_restart false)");
    }
    std::ignore = platform::RemovePrivate(spill, root_name.c_str());
    (void)::close(spill);
    return {};
  }
  auto opened = platform::OpenPrivateDirectory(spill, root_name.c_str());
  (void)::close(spill);
  if (!opened) {
    Log(std::format("conversations are not kept across a restart: {}/{}: {}", roles_.spill.string(),
                    root_name, Errno(opened.error())));
    return {};
  }
  const int root = *opened;
  // Found with other owners or modes it is used as it is (D-105 as amended
  // on 2026-10-09), said once.
  if (const std::string note = platform::PrivacyNote(root); !note.empty()) {
    Log(std::format("warning: the kept conversations directory {}/{}: {}; used as found",
                    roles_.spill.string(), root_name, note));
  }
  // Hashing a record's files: a few cores, off the driver's path.
  keeper_ = std::make_unique<StateKeeper>(4, [this](std::string_view line) { Log(line); });
  const std::string build = KeptBuild();
  const std::int64_t now = NowUnixMs();
  const std::int64_t retention_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(retention_).count();
  std::vector<std::string> kept_dirs;
  std::size_t adopted_count = 0;
  std::uint64_t adopted_bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    const auto entry = std::ranges::find(config_.models, l.name(), &config::ModelEntry::name);
    const std::string artifact = entry != config_.models.end() ? entry->artifact.value_or("") : "";
    auto execution = l.ExecutionIdentity();
    if (!execution) {
      Log(std::format("model {}: kept execution identity unavailable: {}", l.name(),
                      execution.error()));
      continue;
    }
    const std::string model_build =
        *execution ? build + ";execution-v1=" + base::ToHex(**execution) : build;
    const std::string layout = l.KeptLayout();
    if (!kept::ValidDirectoryName(artifact) || layout.empty()) {
      Log(
          std::format("model {}: its conversations are not kept across a restart (its artifact "
                      "ID names no directory)",
                      l.name()));
      continue;
    }
    auto dir = platform::OpenPrivateDirectory(root, artifact.c_str());
    if (!dir) {
      Log(std::format("model {}: its conversations are not kept across a restart: {}: {}", l.name(),
                      artifact, Errno(dir.error())));
      continue;
    }
    if (const std::string note = platform::PrivacyNote(*dir); !note.empty()) {
      Log(
          std::format("warning: model {}: its kept conversations directory {}/{}/{}: {}; used as "
                      "found",
                      l.name(), roles_.spill.string(), root_name, artifact, note));
    }
    kept_dirs.push_back(artifact);
    // One found that the runtime cannot write (another user's, or its modes
    // changed) keeps nothing and is left as it is: its slots' named spill
    // files could not be made, and the model would not start.
    if (::faccessat(*dir, ".", W_OK | X_OK, AT_EACCESS) != 0) {
      const int error = errno;
      Log(std::format("model {}: its conversations are not kept across a restart: {}/{}/{}: {}",
                      l.name(), roles_.spill.string(), root_name, artifact, Errno(error)));
      (void)::close(*dir);
      continue;
    }
    const std::size_t index = keeper_->AddModel(*dir);
    const kept::Identity identity{
        .build = model_build,
        .artifact = artifact,
        .drafter = l.speculative() && entry->drafter ? *entry->drafter : std::string(),
        .layout = layout};
    kept::Expected expected{.version = l.KeptVersion(),
                            .identity = identity,
                            .slot = 0,
                            .regions = l.KeptRegions(),
                            .layouts = l.KeptLayouts(),
                            .context = l.usable_context(),
                            .vocabulary = static_cast<std::uint32_t>(l.tokenizer().size()),
                            .now_unix_ms = now,
                            .retention_ms = retention_ms};
    // Each slot's record, checked whole and its files hashed; a record
    // that does not validate is refused and its files emptied.
    std::vector<std::uint32_t> adopt;
    std::vector<std::vector<kept::FileId>> adopt_files;
    std::vector<std::string> keep_files;
    std::vector<PendingAdoption> pending;
    bool deferred = false;
    for (std::uint32_t slot = 0; slot < l.branches(); ++slot) {
      const std::string record_name = kept::RecordFileName(slot);
      auto text = platform::ReadPrivateFile(*dir, record_name.c_str(), kept::kMostRecordBytes);
      if (!text) {
        if (text.error() != ENOENT) {
          Log(std::format("model {}: slot {}'s kept conversation was refused: its record: {}",
                          l.name(), slot, Errno(text.error())));
        }
        continue;
      }
      expected.slot = slot;
      std::vector<std::string> dropped;
      MemoryCharge token_charge;
      bool memory_refused = false;
      auto record = kept::Decode(*text, &token_memory_, &token_charge, &memory_refused);
      std::string why;
      if (memory_refused) {
        // Memory this start lacks, not necessarily a bad record: checked
        // uncharged (bounded as its text is), one this process would adopt
        // keeps the model's records and files for a later start (below).
        std::vector<std::string> unused;
        auto plain = kept::Decode(*text);
        if (!plain) {
          why = plain.error();
        } else if (auto checked = kept::Check(*plain, expected, &unused); !checked) {
          why = checked.error();
        } else {
          Log(std::format("model {}: slot {}'s kept conversation does not fit this start: {}",
                          l.name(), slot, record.error()));
          deferred = true;
          break;
        }
      } else if (!record) {
        why = record.error();
      } else if (auto checked = kept::Check(*record, expected, &dropped); !checked) {
        why = checked.error();
      }
      // The files: the very ones the record names, whole, hashing as it says
      // (version 2: one a stage).
      if (why.empty()) {
        why = VerifyKeptState(*dir, *record);
      }
      // Every extent the record does not list reads as zeros from here on,
      // as a fresh slot's do: a record removed before a crash can come back
      // (its removal not yet on disk), its listed extents still matching
      // while the file's others hold what a later conversation wrote there,
      // which this one would read when it grows into them (D-105). Each
      // adoption empties them again, so their emptying need not be synced.
      if (why.empty()) {
        why = EmptyUnlisted(*dir, *record);
      }
      if (!why.empty()) {
        Log(std::format("model {}: slot {}'s kept conversation was refused: {}", l.name(), slot,
                        why));
        continue;
      }
      std::erase_if(record->checkpoints, [&](const kept::Checkpoint& c) {
        const std::string bad =
            VerifyKeptFile(*dir, c.file, c.id, c.file_bytes, kept::CheckpointPlaces(c), c.digests);
        if (!bad.empty()) {
          dropped.push_back(bad);
        }
        return !bad.empty();
      });
      for (const std::string& line : dropped) {
        Log(std::format("model {}: slot {}: a kept turn checkpoint was left out: {}", l.name(),
                        slot, line));
      }
      adopt.push_back(slot);
      adopt_files.emplace_back();
      for (const kept::StateFile& file : record->files) {
        keep_files.push_back(file.name);
        adopt_files.back().push_back(file.id);
      }
      keep_files.push_back(record_name);
      for (const kept::Checkpoint& c : record->checkpoints) {
        keep_files.push_back(c.file);
      }
      pending.push_back(
          {.model = &l, .token_charge = std::move(token_charge), .record = std::move(*record)});
      StartProgress();
    }
    if (deferred) {
      // Nothing of this model's is adopted, removed or kept this run: its
      // slots spill to unnamed files, so its directory stays as it was
      // until a start with the memory adopts it or its retention ends.
      Log(std::format(
          "model {}: its kept conversations stay for a later start; this run keeps none of its "
          "conversations across a restart",
          l.name()));
      continue;
    }
    for (PendingAdoption& p : pending) {
      adopted_bytes += p.record.extents.size() * kExtent;
      ++adopted_count;
      adoptions_.push_back(std::move(p));
    }
    // Nothing else stays: unadopted slots' files, other records, stale
    // temporary files (D-014: what is kept is explicit).
    if (auto entries = platform::ListPrivateDirectory(*dir); entries) {
      for (const platform::DirectoryEntry& listed : *entries) {
        if (!std::ranges::contains(keep_files, listed.name)) {
          std::ignore = platform::RemovePrivate(*dir, listed.name.c_str());
        }
      }
    }
    l.set_kept({.keeper = keeper_.get(),
                .model = index,
                .directory = *dir,
                .identity = identity,
                .adopt = std::move(adopt),
                .files = std::move(adopt_files)});
  }
  // Models no longer configured keep nothing.
  if (auto entries = platform::ListPrivateDirectory(root); entries) {
    for (const platform::DirectoryEntry& entry : *entries) {
      if (!std::ranges::contains(kept_dirs, entry.name)) {
        std::ignore = platform::RemovePrivate(root, entry.name.c_str());
      }
    }
  }
  (void)::close(root);
  Log(
      std::format("conversations are kept across a restart in {}/{} (D-105): {} adopted ({:.1f} "
                  "MiB of state)",
                  roles_.spill.string(), root_name, adopted_count,
                  static_cast<double>(adopted_bytes) / (1U << 20U)));
  return {};
}

void Server::AdoptKept() {
  // Slots whose spill files could not be named after the start's checks
  // (their directory no longer writable): unnamed, nothing of theirs
  // outlives this run.
  for (const auto& m : models_) {
    if (m->llm()) {
      const auto& l = static_cast<const Llm&>(*m);
      for (const std::string& note : l.UnkeptSlots()) {
        Log(std::format("model {}: {}; its conversations are not kept across a restart", l.name(),
                        note));
      }
    }
  }
  for (PendingAdoption& pending : adoptions_) {
    Llm& l = *pending.model;
    const std::uint32_t slot = pending.record.slot;
    auto branch = l.branch(slot);
    Status adopted = branch ? l.Adopt(**branch, pending.record, pending.token_charge)
                            : Status(std::unexpected(branch.error()));
    if (adopted) {
      Log(std::format(
          "model {}: slot {} adopted a kept conversation of {} tokens ({} turn "
          "checkpoints); its next turn restores it",
          l.name(), slot, (*branch)->history().size(), pending.record.checkpoints.size()));
      continue;
    }
    Log(std::format("model {}: slot {}'s kept conversation was not adopted: {}", l.name(), slot,
                    adopted.error()));
    // Its file was kept as it is: emptied (a slot's fresh state reads zeros
    // from it), and its record removed.
    if (branch) {
      std::ignore = (*branch)->ReleaseIdleState();
    }
  }
  adoptions_.clear();
  token_memory_.Settle(0, std::chrono::milliseconds{0});
}

void Server::Persist(Clock::time_point deadline, const std::function<void()>& progress) {
  if (keeper_ == nullptr || !started_ || torn_down_) {
    return;
  }
  std::size_t spilled = 0;
  std::uint64_t bytes = 0;
  auto told = Clock::now();
  if (resident_ != nullptr && resident_->llm()) {
    auto& l = static_cast<Llm&>(*resident_);
    // The most recently used first, so those that do not fit are the least
    // recently used.
    std::vector<Llm::Branch*> idle;
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i);
          b && !(*b)->history().empty() && l.BranchIdle(**b) && l.ResidentStateBytes(**b) != 0) {
        idle.push_back(*b);
      }
    }
    std::ranges::stable_sort(idle, [&l](const Llm::Branch* a, const Llm::Branch* b) {
      return l.LastUsed(*a) > l.LastUsed(*b);
    });
    for (Llm::Branch* branch : idle) {
      if (Clock::now() >= deadline) {
        break;
      }
      const std::uint64_t held = l.ResidentStateBytes(*branch);
      // Within the spill budget, as every spill, but deleting only spilled
      // state used before it (the least recently used first): one that
      // fits only by displacing a more recent conversation is not kept.
      // Its whole state counts once spilled, not only what the spill
      // writes.
      if (held == 0 || held > spill_budget_ ||
          !KeepWithinSpillBudget(held, {}, l.LastUsed(*branch))) {
        continue;
      }
      if (auto r = l.SpillIdle(*branch); r) {
        ++spilled;
        bytes += held;
      }
      if (progress && Clock::now() - told >= std::chrono::seconds(1)) {
        progress();
        told = Clock::now();
      }
    }
  }
  // The models a swap wrote back have their records already queued. The
  // keeper writes them while it makes progress (bytes hashed, records
  // written), however long that takes; a minute with nothing moving ends
  // the wait (a hung drive), as does the deadline.
  bool drained = false;
  const auto moved = [this] {
    const StateKeeper::Stats s = keeper_->stats();
    return s.hashed_bytes + s.written + s.failed + s.stale;
  };
  std::uint64_t seen = moved();
  auto quiet_until = Clock::now() + std::chrono::minutes(1);
  while (!(drained = keeper_->Drain(std::min(deadline, Clock::now() + std::chrono::seconds(1))))) {
    const auto now = Clock::now();
    if (const std::uint64_t at = moved(); at != seen) {
      seen = at;
      quiet_until = now + std::chrono::minutes(1);
    }
    if (now >= deadline || now >= quiet_until) {
      break;
    }
    if (progress) {
      progress();
    }
  }
  const StateKeeper::Stats stats = keeper_->stats();
  Log(
      std::format("kept for the next start: {} conversations spilled ({:.1f} MiB); records {} "
                  "written in all, {} pending{}; {:.1f} MiB hashed in {:.2f} s",
                  spilled, static_cast<double>(bytes) / (1U << 20U), stats.written,
                  keeper_->pending(), drained ? "" : " (the stop's time ran out)",
                  static_cast<double>(stats.hashed_bytes) / (1U << 20U), stats.hash_seconds));
}

bool Server::DrainKept(Clock::time_point deadline) {
  return keeper_ == nullptr || keeper_->Drain(deadline);
}

std::uint64_t Server::context_body_bytes() const {
  std::uint64_t most = 0;
  for (const auto& m : models_) {
    if (m->llm()) {
      const auto& l = static_cast<const Llm&>(*m);
      most = std::max(most, ContextBodyBytes(l.usable_context(), l.longest_token()));
    }
  }
  return most;
}

std::uint64_t Server::request_capacity() const {
  if (config_.client.request_memory_bytes) {
    return *config_.client.request_memory_bytes;
  }
  return request_memory_ + state_room_;
}

bool Server::SetRequestMemory(std::uint64_t to) {
  if (!started_ || torn_down_) {
    return to <= request_memory_;
  }
  // Past the floor, charged inside the budget; what does not fit has the
  // reclaim order free it (plans, graphs, idle conversations spilled; the
  // request memory itself is never reclaimed), all of it or nothing.
  const std::uint64_t past = to > request_memory_ ? to - request_memory_ : 0;
  std::uint64_t shortfall = 0;
  if (node_.SetRequestCharge(past, &shortfall)) {
    return true;
  }
  for (int attempt = 0; attempt < 3 && shortfall != 0; ++attempt) {
    const std::uint64_t ask = ((shortfall + kExtent - 1) / kExtent * kExtent) + kExtent;
    if (Reclaim(ask, true, "request memory") == 0) {
      break;
    }
    shortfall = 0;
    if (node_.SetRequestCharge(past, &shortfall)) {
      return true;
    }
  }
  return false;
}

bool Server::SetTokenMemory(std::uint64_t incoming) {
  if (!started_ || torn_down_) {
    return false;
  }
  if (!token_catalog_ready_) {
    const auto used = token_memory_.used();
    return used <= startup_token_capacity_ && incoming <= startup_token_capacity_ - used;
  }
  std::uint64_t shortfall = 0;
  for (int attempt = 0; attempt < 4; ++attempt) {
    // Reclaim may have retired histories since this grow began.
    const auto target = token_memory_.used() + incoming;
    if (node_.SetTokenCharge(target, &shortfall)) {
      return true;
    }
    if (attempt == 3 || Reclaim(shortfall, true, "native token history", nullptr, std::nullopt,
                                false, nullptr, incoming) == 0) {
      return false;
    }
  }
  return false;
}

Status Server::TearDown() {
  if (torn_down_) {
    return {};
  }
  if (started_) {
    RecordCalibrations();
    // What the incremental spills and swaps left unwritten, and the
    // reclaims that found too little to take, or ended short when a victim
    // gave back less than it counted (counts and bytes only).
    if (auto stats = node_.Stats(); stats) {
      Log(std::format(
          "write-backs released unchanged (nothing written): {} extents, {:.1f} MiB; reclaims "
          "that took nothing (too little to take): {}; that ended short (a victim gave back "
          "less): {}; weight reads refused with their file closed (an inactive model's): {}",
          stats->unchanged_writebacks,
          static_cast<double>(stats->unchanged_writeback_bytes) / (1U << 20U), reclaims_short_,
          reclaims_cut_short_, stats->closed_file_reads));
    }
    // What each model's plans and graphs held at the end, and what bringing
    // them back would have cost (the reclaim order's measured inputs):
    // counts, bytes and seconds only.
    for (std::uint32_t i = 0; i < models_.size(); ++i) {
      std::vector<memory::ReclaimCandidate> held;
      models_[i]->ReclaimCandidates(i, false, held);
      std::array<std::uint64_t, memory::kReclaimKinds> count{};
      std::array<std::uint64_t, memory::kReclaimKinds> bytes{};
      std::array<double, memory::kReclaimKinds> seconds{};
      for (const memory::ReclaimCandidate& c : held) {
        const auto k = static_cast<std::size_t>(c.kind);
        ++count[k];
        bytes[k] += c.bytes;
        seconds[k] += c.restore_seconds;
      }
      const auto p = static_cast<std::size_t>(memory::ReclaimKind::kPlan);
      const auto g = static_cast<std::size_t>(memory::ReclaimKind::kGraph);
      if (count[p] + count[g] != 0) {
        // A plan's count includes its graphs'; a graph may have no plan
        // candidate (the image pipeline's recorded step).
        const std::array<double, memory::kReclaimKinds> cost = memory::KindCosts(held);
        const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / (1U << 20U); };
        std::uint64_t plan_bytes = 0;
        double plan_seconds = 0;
        for (const memory::ReclaimCandidate& c : held) {
          if (c.kind != memory::ReclaimKind::kPlan) {
            continue;
          }
          std::uint64_t own = c.bytes;
          double own_seconds = c.restore_seconds;
          for (const memory::ReclaimCandidate& graph : held) {
            if (graph.kind == memory::ReclaimKind::kGraph && graph.id == c.id) {
              own -= std::min(own, graph.bytes);
              own_seconds -= std::min(own_seconds, graph.restore_seconds);
            }
          }
          plan_bytes += own;
          plan_seconds += own_seconds;
        }
        Log(std::format(
            "model {}: {} plans held {:.1f} MiB (planned in {:.3f} s, {:.2f} s a GiB), "
            "{} graphs {:.1f} MiB counted, {:.1f} MiB measured at their captures (captured in "
            "{:.3f} s, {:.2f} s a GiB)",
            models_[i]->name(), count[p], mib(plan_bytes), plan_seconds, cost[p], count[g],
            mib(bytes[g]), mib(models_[i]->graph_measured_bytes()), seconds[g], cost[g]));
      }
    }
  }
  torn_down_ = true;
  std::vector<engine::PagedModel*> models;
  models.reserve(models_.size());
  for (const auto& m : models_) {
    // A start that failed before a model's Setup made its runner leaves
    // nothing of it on the node.
    if (m->has_paged()) {
      models.push_back(&m->paged());
    }
    // Kept checkpoints' files stay for the next start (D-105): those no
    // record names it removes.
    if (m->llm()) {
      static_cast<Llm&>(*m).PreserveKeptFiles();
    }
  }
  // Expire capabilities and release the single completed snapshot while
  // the accounting node still exists. Unknown copies stay in KeepPinned.
  snapshot_ = {};
  snapshot_model_ = nullptr;
  snapshot_store_.reset();
  const auto stopped = node_.TearDown(models);
  // Every frontend can stack-own Server. A failed fence is not proof that
  // any submitted device/storage work stopped borrowing these owners.
  RetireServiceOrExit(stopped, log_);
  if (options_.diagnostic_budget_cap_bytes) {
    const auto c = node_.backing_create_stats();
    const auto h = node_.retired_harvest_stats();
    const auto harvest =
        h ? std::format("{{\"indices\":{},\"pop_locks\":{}}}", h->indices, h->pop_locks) : "null";
    Log(std::format(
        "diagnostic final backing creates: {{\"retired\":{},\"reserve_attempts\":{},"
        "\"reserve_failures\":{},\"ordinary_attempts\":{},\"ordinary_failures\":{},"
        "\"last_failure_monotonic_ns\":{},\"harvest\":{}}}",
        stopped.has_value(), c.reserve_attempts, c.reserve_failures, c.ordinary_attempts,
        c.ordinary_failures, c.last_failure_monotonic_ns, harvest));
  }
  return stopped;
}

Served* Server::Find(std::string_view name) {
  for (const auto& m : models_) {
    if (m->name() == name) {
      return m.get();
    }
  }
  return nullptr;
}

Status Server::InRequest(Served& m, const std::function<Status()>& body) {
  if (auto selected = m.PrepareDefaultRequest(); !selected) {
    return selected;
  }
  auto ran = node_.WithRequest(m.paged().stream(), m.request_closure(),
                               std::format("{}'s request", m.name()), body);
  if (ran) return RecordWeightUse(m);
  return ran;
}

Status Server::RecordWeightUse(Served& m) {
  const auto weights = m.weights();
  return node_.Call(
      [&]() -> Status {
        std::uint64_t tick = 0;
        for (const auto extent : weights) {
          if (const auto view = node_.catalog().Describe(extent))
            tick = std::max(tick, view->last_use);
        }
        m.RecordWeightsUsed(tick);
        return {};
      },
      "recording a completed model's weight use");
}

Status Server::SelectRequestBranches(Llm& model, std::span<Llm::Branch* const> active) {
  if (!started_ || torn_down_ || resident_ != &model || active.empty() ||
      std::ranges::none_of(models_, [&](const auto& owned) { return owned.get() == &model; })) {
    return Error("a cooperative request needs its resident model and active branches");
  }
  if (auto selected = model.SelectBranches(active); !selected) {
    return selected;
  }
  const auto stream = model.paged().stream();
  if (node_.InRequest(stream)) {
    return node_.RefreshRequest(stream, model.request_closure());
  }
  return node_.BeginRequest(stream, model.request_closure(),
                            std::format("{}'s request cohort", model.name()));
}

Status Server::EndRequestBranches(Llm& model) {
  if (!started_ || torn_down_ || resident_ != &model) {
    return Error("a retiring request cohort does not own the resident model");
  }
  const auto stream = model.paged().stream();
  // A proven failed job or state helper can already have ended its request.
  // The caller must separately prove that no work still borrows its owners.
  if (node_.InRequest(stream)) {
    if (auto ended = node_.EndRequest(stream); !ended) return ended;
  }
  return RecordWeightUse(model);
}

Server::ReferenceRetirement Server::RetireRequestBranches(Llm& model, bool last_owner) {
  if (!started_ || torn_down_ || resident_ != &model || !model.generation_cohort_usable()) {
    return {.result = Error("the native cohort cannot prove reference retirement"),
            .references_retired = false};
  }
  const auto stream = model.paged().stream();
  bool queued = false;
  auto fenced = node_.Job(
      model.request_closure(),
      [&queued](providers::NativeStream) {
        queued = true;
        // kQueued requires the DeviceLane to record and await its completion
        // fence, even though this job queues no arithmetic or copies itself.
        return scheduler::JobResult::kQueued;
      },
      "retiring native request-owner references", stream);
  if (!fenced || !queued) {
    return {.result = !fenced ? std::move(fenced) : Error("the retirement fence was not queued"),
            .references_retired = false};
  }
  // The known-complete stream fence is the proof. An EndRequest error does
  // not revoke it: EndRequest itself awaits ProgramDone::gone before return.
  return {.result = last_owner ? EndRequestBranches(model) : Status{}, .references_retired = true};
}

Status Server::WaitReleased() {
  // Progress-based (D-102): only evictions that stop finishing for the
  // node's quiet time are given up on, however long the whole takes.
  auto give_up = Clock::now() + node_.quiet();
  std::size_t before = std::numeric_limits<std::size_t>::max();
  for (;;) {
    std::size_t left = 0;
    if (auto r = node_.Call(
            [&]() -> Status {
              left = node_.scheduler().evictions();
              return {};
            },
            "counting evictions");
        !r) {
      return r;
    }
    if (left == 0) {
      return {};
    }
    if (left < before) {
      before = left;
      give_up = Clock::now() + node_.quiet();
    } else if (Clock::now() > give_up) {
      return Error(
          "backing no load took was not released: no eviction finished for the node's "
          "quiet time");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

Status Server::FinishSwap(SwapParts& parts) {
  if (auto r = WaitReleased(); !r) {
    return r;
  }
  parts.release = Seconds(Clock::now() - parts.ready);
  auto after = node_.Stats();
  if (!after) {
    return std::unexpected(after.error());
  }
  parts.handed_off = after->handed_off - swap_before_.handed_off;
  parts.released_unused = after->released_unused - swap_before_.released_unused;
  parts.occupancy_released = after->catalog_occupancy_bytes;
  if (parts.diagnostic.enabled) {
    parts.diagnostic.finished = swap_host_phases_;
    parts.diagnostic.creates_finished = node_.backing_create_stats();
  }
  return {};
}

std::uint64_t Server::SpilledBytes(SpillView view) {
  std::uint64_t bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    // A model a swap wrote back holds all its state on disk until it returns.
    const bool written_back =
        m.get() != view.in &&
        (m.get() == view.out || std::ranges::find(spilled_, m.get()) != spilled_.end());
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i); b) {
        bytes += written_back ? l.StateBytes(**b) : l.SpilledStateBytes(**b);
      }
    }
  }
  return bytes;
}

void Server::ForEachDeletableSpill(
    SpillView view, std::optional<Clock::time_point> older_than,
    const std::function<void(Llm& model, Llm::Branch& branch, std::uint64_t counted)>& each) {
  for (const auto& m : models_) {
    if (!m->llm() || m.get() == view.in) {
      continue;  // the incoming model's stop counting once it is resident
    }
    auto& l = static_cast<Llm&>(*m);
    const bool written_back =
        m.get() == view.out || std::ranges::find(spilled_, m.get()) != spilled_.end();
    for (std::size_t i = 0; i < l.branches(); ++i) {
      auto b = l.branch(i);
      if (!b || !l.BranchIdle(**b) || (*b)->HeldByContinuation() ||
          (older_than && l.LastUsed(**b) >= *older_than)) {
        continue;
      }
      const std::uint64_t counted = written_back ? l.StateBytes(**b) : l.SpilledStateBytes(**b);
      if (counted != 0) {
        each(l, **b, counted);
      }
    }
  }
}

bool Server::SpillBudgetFits(std::uint64_t extra, SpillView view,
                             std::optional<Clock::time_point> older_than) {
  const std::uint64_t held = SpilledBytes(view);
  std::uint64_t deletable = 0;
  ForEachDeletableSpill(view, older_than,
                        [&](Llm&, Llm::Branch&, std::uint64_t counted) { deletable += counted; });
  const std::uint64_t left = held - std::min(held, deletable);
  // Eligible checkpoints may go before spilled conversations, but those
  // a request holds cannot: they consume disk at either admission point.
  return TurnCheckpointFits(spill_budget_, left, TurnCheckpointBytes(true), extra);
}

std::uint64_t Server::TurnCheckpointBytes(bool held_only) {
  std::uint64_t checkpoints = 0;
  for (const auto& m : models_) {
    if (m->llm()) {
      auto& l = static_cast<Llm&>(*m);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b && (!held_only || !l.OldestTurnCheckpoint(**b))) {
          const auto bytes = (*b)->turn_checkpoint_bytes();
          if (bytes > std::numeric_limits<std::uint64_t>::max() - checkpoints)
            return std::numeric_limits<std::uint64_t>::max();
          checkpoints += bytes;
        }
      }
    }
  }
  return checkpoints;
}

bool Server::TrimTurnCheckpoints(std::uint64_t extra, SpillView view) {
  std::size_t dropped = 0;
  std::uint64_t dropped_bytes = 0;
  const auto drop = [&]() {
    // The oldest checkpoint of any idle conversation no request holds.
    Llm* model = nullptr;
    Llm::Branch* branch = nullptr;
    std::optional<Clock::time_point> oldest;
    for (const auto& m : models_) {
      if (!m->llm()) {
        continue;
      }
      auto& l = static_cast<Llm&>(*m);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        auto b = l.branch(i);
        if (!b) {
          continue;
        }
        if (const auto at = l.OldestTurnCheckpoint(**b); at && (!oldest || *at < *oldest)) {
          oldest = at;
          model = &l;
          branch = *b;
        }
      }
    }
    if (branch == nullptr) {
      return false;  // what is left, requests hold
    }
    const std::uint64_t bytes = model->DropOldestTurnCheckpoint(*branch);
    if (bytes == 0) {
      return false;
    }
    ++dropped;
    dropped_bytes += bytes;
    return true;
  };
  const bool fits = TrimTurnCheckpointBudget(
      spill_budget_, extra, [this, view] { return SpilledBytes(view); },
      [this] { return TurnCheckpointBytes(); }, drop);
  if (dropped != 0) {
    Log(std::format(
        "deleted {} turn checkpoints ({:.1f} MiB), the oldest, past the spill budget of {} GiB "
        "beside the spilled conversations",
        dropped, static_cast<double>(dropped_bytes) / (1U << 20U), spill_budget_ >> 30U));
  }
  return fits;
}

bool Server::KeepWithinSpillBudget(std::uint64_t extra, SpillView view,
                                   std::optional<Clock::time_point> older_than) {
  // Nothing deleted for a conversation (or a swap) that would not fit even
  // with every deletable one gone.
  return SpillBudgetFits(extra, view, older_than) && DeleteSpilledPast(extra, view, older_than);
}

bool Server::DeleteSpilledPast(std::uint64_t extra, SpillView view,
                               std::optional<Clock::time_point> older_than) {
  // Optional caches yield first, before deleting any conversation; held
  // checkpoints remain charged and may leave too little room for a spill.
  (void)TrimTurnCheckpoints(extra, view);
  std::size_t deleted = 0;
  std::uint64_t deleted_bytes = 0;
  bool fits = false;
  for (;;) {
    const std::uint64_t held = SpilledBytes(view);
    if (TurnCheckpointFits(spill_budget_, held, TurnCheckpointBytes(), extra)) {
      fits = true;
      break;
    }
    // The least recently used spilled conversation no request holds.
    Llm* oldest = nullptr;
    Llm::Branch* branch = nullptr;
    ForEachDeletableSpill(view, older_than, [&](Llm& l, Llm::Branch& b, std::uint64_t) {
      if (branch == nullptr || l.LastUsed(b) < oldest->LastUsed(*branch)) {
        oldest = &l;
        branch = &b;
      }
    });
    if (branch == nullptr) {
      break;  // nothing more to delete: still short
    }
    const std::uint64_t bytes = oldest->StateBytes(*branch);
    if (auto released = branch->ReleaseIdleState(); !released) {
      Log(std::format("{}'s spilled conversation could not be deleted: {}", oldest->name(),
                      released.error()));
      break;
    }
    ++deleted;
    deleted_bytes += bytes;
  }
  if (deleted != 0) {
    Log(std::format(
        "deleted {} spilled conversations ({:.1f} MiB), the least recently used, "
        "past the spill budget of {} GiB",
        deleted, static_cast<double>(deleted_bytes) / (1U << 20U), spill_budget_ >> 30U));
  }
  return fits;
}

void AddIdleStateCandidates(Llm& model, std::uint32_t owner, bool running, const Llm::Branch* spare,
                            const IdleStateRates& rates,
                            std::vector<memory::ReclaimCandidate>& out) {
  for (std::size_t slot = 0; slot < model.branches(); ++slot) {
    auto b = model.branch(slot);
    if (!b || *b == spare || !model.BranchIdle(**b) || (*b)->history().empty()) {
      continue;
    }
    const std::uint64_t bytes = model.ResidentStateBytes(**b);
    if (bytes == 0) {
      continue;
    }
    // Spilled whole within the budget (Server::Reclaim); past all of it,
    // dropped, which a continuation holding it refuses.
    const bool dropped = bytes > rates.spill_budget;
    if (dropped && (*b)->HeldByContinuation()) {
      continue;
    }
    // A spill writes only what changed since its last one; a restore reads
    // it all.
    const auto size = static_cast<double>(bytes);
    const auto writes = static_cast<double>(model.SpillWriteBytes(**b));
    out.push_back(
        {.kind = memory::ReclaimKind::kIdleState,
         .owner = owner,
         .id = slot,
         .bytes = bytes,
         .last_use = static_cast<std::uint64_t>(model.LastUsed(**b).time_since_epoch().count()),
         .restore_seconds = dropped ? static_cast<double>((*b)->history().size()) *
                                          model.settings().recompute_ms_per_token.value / 1000.0
                                    : (writes / rates.spill_rate) + (size / rates.restore_rate),
         .running = running});
    memory::SetUse(out.back(), model.LastStamp(**b));
  }
}

bool RoomFor(std::uint64_t needed, std::optional<std::uint64_t> free,
             const std::function<std::uint64_t(std::uint64_t)>& reclaim) {
  if (needed == 0 || !free || *free >= needed) {
    return true;
  }
  return reclaim(needed - *free) >= needed - *free;
}

bool Server::RoomFor(std::uint64_t needed, std::string_view why, const Llm::Branch* spare) {
  if (needed == 0 || !started_ || torn_down_) {
    return true;
  }
  auto free = node_.FreeBytes();
  return runtime::RoomFor(
      needed, free ? std::optional(*free) : std::nullopt, [&](std::uint64_t shortfall) {
        return Reclaim(shortfall, true, why, nullptr, std::nullopt, false, spare);
      });
}

std::uint64_t Server::Reclaim(std::uint64_t needed, bool states, std::string_view why,
                              const Served* running, std::optional<memory::ReclaimKind> below_kind,
                              bool partial, const Llm::Branch* spare, std::uint64_t token_incoming,
                              std::span<const catalog::ExtentId> protect,
                              PendingWeightEvictions* pending, bool allow_weights) {
  if (reclaiming_ || !started_ || torn_down_ || needed == 0) {
    return 0;
  }
  reclaiming_ = true;
  const bool diagnostic = options_.diagnostic_budget_cap_bytes.has_value();
  if (diagnostic) ++swap_host_phases_.reclaim_calls;
  if (running == nullptr) {
    running = resident_;
  }
  // Kept zeroed state backing holds nothing: it goes first, before the
  // order prices anything (other models' before the running one's).
  std::set<catalog::ExtentId> protected_ids(activation_protect_.begin(), activation_protect_.end());
  protected_ids.insert(protect.begin(), protect.end());
  const std::vector<catalog::ExtentId> protected_extents(protected_ids.begin(),
                                                         protected_ids.end());
  const std::uint64_t kept = ReleaseKept(needed, running, protected_extents);
  if (kept >= needed) {
    reclaiming_ = false;
    return kept;
  }
  // Idle state's cost at the rates measured so far (once a GiB has moved).
  double spill_rate = kSpillBytesPerSecond;
  double restore_rate = kRestoreBytesPerSecond;
  {
    Llm::SpillStats all;
    for (const auto& m : models_) {
      if (m->llm()) {
        const Llm::SpillStats& s = static_cast<Llm&>(*m).spill_stats();
        all.spilled_bytes += s.spilled_bytes;
        all.spill_seconds += s.spill_seconds;
        all.restored_bytes += s.restored_bytes;
        all.restore_seconds += s.restore_seconds;
      }
    }
    constexpr std::uint64_t kEnough = std::uint64_t{1} << 30U;
    if (all.spilled_bytes >= kEnough && all.spill_seconds > 0) {
      spill_rate = static_cast<double>(all.spilled_bytes) / all.spill_seconds;
    }
    if (all.restored_bytes >= kEnough && all.restore_seconds > 0) {
      restore_rate = static_cast<double>(all.restored_bytes) / all.restore_seconds;
    }
  }
  const auto k = [](memory::ReclaimKind kind) { return static_cast<std::size_t>(kind); };
  std::array<std::uint64_t, memory::kReclaimKinds> count{};
  std::array<std::uint64_t, memory::kReclaimKinds> freed_by{};
  std::array<double, memory::kReclaimKinds> cost{};
  std::size_t dropped = 0;
  bool took = false;
  Llm* idle_owner = nullptr;
  struct HistoryVictim {
    Llm* model;
    Llm::Branch* branch;
    std::uint32_t owner;
    std::uint64_t slot;
  };
  std::vector<HistoryVictim> histories;
  std::vector<TokenReclaimGroup> history_groups;
  struct WeightView {
    std::uint32_t owner = 0;
    std::uint64_t generation = 0;
    std::uint64_t bytes = 0;
  };
  std::map<catalog::ExtentId, WeightView> weight_views;
  const double weight_rate = page_read_bytes_ >= (std::uint64_t{1} << 30U) && page_read_seconds_ > 0
                                 ? static_cast<double>(page_read_bytes_) / page_read_seconds_
                                 : kWeightReadBytesPerSecond;
  // The candidates afresh each round (memory::RunReclaim): every model's
  // plans and graphs, and with `states` the resident model's idle state.
  const auto gather = [&](std::vector<memory::ReclaimCandidate>& candidates, double& below) {
    const SwapPhaseTimer timer(diagnostic ? &swap_host_phases_.gather_seconds : nullptr);
    idle_owner = nullptr;
    histories.clear();
    history_groups.clear();
    weight_views.clear();
    std::uint32_t resident_index = 0;
    for (std::uint32_t i = 0; i < models_.size(); ++i) {
      models_[i]->ReclaimCandidates(i, models_[i].get() == running, candidates);
      if (states && models_[i]->llm()) {
        auto& model = static_cast<Llm&>(*models_[i]);
        for (std::size_t slot = 0; slot < model.branches(); ++slot) {
          auto branch = model.branch(slot);
          if (!branch || *branch == spare || !model.HistoryReclaimable(**branch)) {
            continue;
          }
          const auto bytes = model.IdleHistoryBytes(**branch);
          if (bytes == 0) {
            continue;
          }
          histories.push_back({.model = &model, .branch = *branch, .owner = i, .slot = slot});
        }
      }
      if (models_[i].get() == resident_) {
        resident_index = i;
      }
    }
    if (options_.partial_weight_eviction && allow_weights) {
      std::set<catalog::ExtentId> pending_ids;
      if (pending != nullptr) {
        pending_ids = pending->rejected;
        for (const auto& selected : pending->selected) pending_ids.insert(selected.extent);
      }
      std::vector<std::vector<catalog::ExtentId>> weights;
      weights.reserve(models_.size());
      for (const auto& model : models_) {
        weights.push_back(model.get() == running ? std::vector<catalog::ExtentId>{}
                                                 : model->weights());
      }
      const auto listed = node_.Call(
          [&]() -> Status {
            for (std::uint32_t i = 0; i < weights.size(); ++i) {
              for (const auto extent : weights[i]) {
                if (protected_ids.contains(extent) || pending_ids.contains(extent)) continue;
                const auto view = node_.catalog().Describe(extent);
                const auto* source = node_.scheduler().SourceOf(extent);
                if (!view || !catalog::Catalog::Evictable(*view) || source == nullptr ||
                    view->descriptor.memory_class != catalog::MemoryClass::kWeights ||
                    view->descriptor.recovery != catalog::Recovery::kFromArtifact)
                  continue;
                if (!weight_views
                         .try_emplace(extent, WeightView{.owner = i,
                                                         .generation = view->content_generation,
                                                         .bytes = view->descriptor.size.value()})
                         .second) {
                  return Error("a prepared weight extent has more than one owner");
                }
                candidates.push_back(
                    {.kind = memory::ReclaimKind::kIdleWeights,
                     .owner = i,
                     .id = WeightHandle(extent),
                     .bytes = view->descriptor.size.value(),
                     .last_use = view->last_use,
                     .restore_seconds = static_cast<double>(source->read.length) / weight_rate});
                memory::SetUse(candidates.back(), models_[i]->weight_use());
              }
            }
            return {};
          },
          "gathering inactive weight reclaim candidates");
      if (!listed) {
        Log(std::format("listing inactive weights: {}", listed.error()));
        std::erase_if(candidates,
                      [](const auto& c) { return c.kind == memory::ReclaimKind::kIdleWeights; });
        weight_views.clear();
      }
    }
    std::ranges::sort(histories, [](const HistoryVictim& a, const HistoryVictim& b) {
      return a.model->LastUsed(*a.branch) < b.model->LastUsed(*b.branch);
    });
    std::vector<std::uint64_t> capacities;
    capacities.reserve(histories.size());
    for (const auto& history : histories) {
      capacities.push_back(history.model->IdleHistoryBytes(*history.branch));
    }
    history_groups =
        GroupTokenReclaim(capacities, token_memory_.used() + token_incoming, kExtent, needed);
    std::size_t first = 0;
    for (const auto& group : history_groups) {
      const auto& oldest = histories[first];
      double seconds = 0;
      bool group_running = false;
      for (std::size_t i = first; i < group.end; ++i) {
        const auto& h = histories[i];
        seconds += static_cast<double>(h.branch->history().size()) *
                   h.model->settings().recompute_ms_per_token.value / 1000.0;
        group_running = group_running || h.model == running;
      }
      candidates.push_back({.kind = memory::ReclaimKind::kTokenHistory,
                            .owner = oldest.owner,
                            .id = oldest.slot,
                            .bytes = group.catalog_bytes,
                            .last_use = static_cast<std::uint64_t>(
                                oldest.model->LastUsed(*oldest.branch).time_since_epoch().count()),
                            .restore_seconds = seconds,
                            .running = group_running});
      memory::SetUse(candidates.back(), oldest.model->LastStamp(*oldest.branch));
      first = group.end;
    }
    // The running model's next step finds what its last ones used.
    memory::ProtectFloor(candidates, running != nullptr ? running->plan_floor_bytes() : 0);
    if (states && resident_ != nullptr && resident_->llm()) {
      idle_owner = static_cast<Llm*>(resident_);
      AddIdleStateCandidates(
          *idle_owner, resident_index, resident_ == running, spare,
          {.spill_rate = spill_rate, .restore_rate = restore_rate, .spill_budget = spill_budget_},
          candidates);
    }
    cost = memory::KindCosts(candidates);
    if (below_kind) {
      // What the charge would cost to restore a GiB: its kind's measured
      // cost, or the GB10's when none is held.
      double charged = cost[k(*below_kind)];
      if (charged == 0) {
        charged = *below_kind == memory::ReclaimKind::kGraph ? kGraphSecondsPerGiB : 0;
      }
      below = memory::ReclaimInflation() + charged;
    }
  };
  const auto take = [&](const memory::ReclaimCandidate& c) -> std::uint64_t {
    const SwapPhaseTimer timer(diagnostic ? &swap_host_phases_.take_seconds : nullptr);
    std::uint64_t got = 0;
    if (c.kind == memory::ReclaimKind::kTokenHistory) {
      std::size_t first = 0;
      const auto prospective = [&]() {
        const auto target = token_memory_.used() + token_incoming;
        return (target / kExtent + (target % kExtent != 0 ? 1U : 0U)) * kExtent;
      };
      const auto before = prospective();
      const auto charged_before = node_.token_charged();
      const auto free_before = node_.FreeBytes();
      for (const auto& group : history_groups) {
        const auto& oldest = histories[first];
        if (oldest.owner == c.owner && oldest.slot == c.id) {
          for (std::size_t i = first; i < group.end; ++i) {
            auto& h = histories[i];
            if (h.branch != spare && h.model->HistoryReclaimable(*h.branch)) {
              std::ignore = h.branch->ReleaseIdleState();
            }
          }
          break;
        }
        first = group.end;
      }
      const auto after = prospective();
      const auto free_after = node_.FreeBytes();
      const auto actual_tokens = charged_before - std::min(charged_before, node_.token_charged());
      // Dropping a history can also release its resident native state. Count
      // the measured catalog gain once, including overlapping idle victims.
      got = free_before && free_after && *free_after >= *free_before ? *free_after - *free_before
                                                                     : actual_tokens;
      if (token_incoming != 0) {
        const auto target_drop = before - std::min(before, after);
        // The incoming allocation can change which rounded boundary matters.
        // Replace actual token release with the prospective target decrease.
        got = TokenReclaimCredit(got, actual_tokens, target_drop);
      }
    } else if (c.kind == memory::ReclaimKind::kIdleState) {
      if (idle_owner == nullptr) {
        return 0;  // only the resident model's idle state is a candidate
      }
      auto b = idle_owner->branch(c.id);
      if (!b) {
        return 0;
      }
      // Spill, not clear: within the spill budget (the least recently used
      // spilled state deleted first; once spilled its whole state counts);
      // past it, or with none, dropped. A state larger than the whole
      // budget deletes nothing else first.
      const std::uint64_t held = idle_owner->ResidentStateBytes(**b);
      if (held == 0) {
        return 0;  // gone meanwhile: nothing to give back
      }
      if (held <= spill_budget_ && KeepWithinSpillBudget(held)) {
        if (auto spilled = idle_owner->SpillIdle(**b); spilled) {
          got = c.bytes;
        } else {
          Log(std::format("{}'s idle conversation in slot {}: {}", idle_owner->name(), c.id,
                          spilled.error()));
          got = idle_owner->ResidentStateBytes(**b) == 0 ? c.bytes : 0;
        }
      } else if (auto released = (*b)->ReleaseIdleState(); released) {
        got = c.bytes;
        ++dropped;
      }
    } else {
      std::optional<std::uint64_t> before_charge;
      if (options_.partial_weight_eviction) {
        std::ignore = node_.Call(
            [&]() -> Status {
              before_charge = node_.catalog().OccupancyOf(node_.domain()).Total().value();
              return {};
            },
            "sizing a cache reclaim's catalog charge");
      }
      got = models_[c.owner]->Reclaim(c.kind, c.id);
      if (before_charge) {
        const auto measured = node_.Call(
            [&]() -> Status {
              const auto after = node_.catalog().OccupancyOf(node_.domain()).Total().value();
              got = *before_charge - std::min(*before_charge, after);
              return {};
            },
            "measuring a cache reclaim's catalog gain");
        if (!measured) got = 0;
      }
    }
    if (got != 0) {
      took = true;
      ++count[k(c.kind)];
      freed_by[k(c.kind)] += got;
    }
    return got;
  };
  const auto take_weights = [&](std::span<const memory::ReclaimCandidate> victims,
                                std::span<std::uint64_t> freed) {
    const SwapPhaseTimer timer(diagnostic ? &swap_host_phases_.take_seconds : nullptr);
    std::vector<catalog::ExtentId> eligible;
    const auto listed = node_.Call(
        [&]() -> Status {
          for (const auto& c : victims) {
            const auto extent = WeightExtent(c.id);
            const auto known = weight_views.find(extent);
            const auto view = node_.catalog().Describe(extent);
            if (known != weight_views.end() && known->second.owner == c.owner && view &&
                view->content_generation == known->second.generation &&
                view->descriptor.size.value() == c.bytes && catalog::Catalog::Evictable(*view) &&
                !protected_ids.contains(extent))
              eligible.push_back(extent);
          }
          return {};
        },
        "rechecking selected idle weights");
    if (!listed || eligible.empty()) return;
    const auto evicted = node_.Evict(eligible);
    if (!evicted) Log(std::format("reclaiming idle weights: {}", evicted.error()));
    const std::set<catalog::ExtentId> attempted(eligible.begin(), eligible.end());
    const auto reconciled = node_.Call(
        [&]() -> Status {
          for (std::size_t i = 0; i < victims.size(); ++i) {
            const auto extent = WeightExtent(victims[i].id);
            const auto view = node_.catalog().Describe(extent);
            if (attempted.contains(extent) && view &&
                view->state == catalog::ExtentState::kNonresident)
              freed[i] = victims[i].bytes;
          }
          return {};
        },
        "reconciling completed idle weight releases");
    if (!reconciled) return;
    for (const auto bytes : freed) {
      if (bytes != 0) {
        took = true;
        ++count[k(memory::ReclaimKind::kIdleWeights)];
        freed_by[k(memory::ReclaimKind::kIdleWeights)] += bytes;
      }
    }
  };
  // All of it or nothing (memory::RunReclaim): a selection that cannot
  // cover what is needed reclaims nothing (the caller waits or refuses);
  // another round, without it, only when a victim gave back less than its
  // count (held, or gone meanwhile), and what the rounds before took then
  // stays taken even when the rest cannot be covered.
  memory::ReclaimRun run;
  if (pending == nullptr) {
    const auto begin = diagnostic ? Clock::now() : Clock::time_point{};
    const auto gather_before = swap_host_phases_.gather_seconds;
    const auto take_before = swap_host_phases_.take_seconds;
    run = memory::RunReclaim(needed - kept, partial, gather, take, take_weights);
    if (diagnostic) {
      swap_host_phases_.selection_bookkeeping_seconds += std::max(
          0.0, Seconds(Clock::now() - begin) - (swap_host_phases_.gather_seconds - gather_before) -
                   (swap_host_phases_.take_seconds - take_before));
    }
  } else {
    std::vector<memory::ReclaimCandidate> candidates;
    double below = std::numeric_limits<double>::infinity();
    gather(candidates, below);
    memory::ReclaimPlan plan;
    {
      const SwapPhaseTimer timer(diagnostic ? &swap_host_phases_.selection_bookkeeping_seconds
                                            : nullptr);
      plan = memory::SelectReclaim(candidates, needed - kept, below);
    }
    if (!plan.sufficient) {
      run.short_of_need = true;
    } else {
      std::uint64_t pending_credit = 0;
      for (std::size_t i = 0; i < plan.victims.size() && run.freed + pending_credit < needed - kept;
           ++i) {
        const auto& c = candidates[plan.victims[i]];
        if (c.kind == memory::ReclaimKind::kIdleWeights) {
          const auto extent = WeightExtent(c.id);
          const auto known = weight_views.find(extent);
          if (known != weight_views.end()) {
            pending->selected.push_back({.extent = extent,
                                         .generation = known->second.generation,
                                         .candidate = c,
                                         .priority = plan.priorities[i]});
            pending_credit += c.bytes;
          }
        } else {
          const auto got = take(c);
          if (got != 0) {
            memory::RaiseReclaimInflation(plan.priorities[i]);
            run.freed += got;
          }
        }
      }
    }
  }
  const std::uint64_t freed = run.freed + kept;
  if (freed == 0 && run.short_of_need) {
    ++reclaims_short_;
  } else if (pending == nullptr && !partial && freed != 0 && freed < needed) {
    ++reclaims_cut_short_;
  }
  if (freed != 0) {
    const SwapPhaseTimer timer(diagnostic ? &swap_host_phases_.heap_release_seconds : nullptr);
    if (diagnostic) ++swap_host_phases_.heap_release_calls;
    platform::ReleaseFreeHeap();
  }
  if (took) {
    const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / (1U << 20U); };
    using K = memory::ReclaimKind;
    // Counts, bytes and measured costs only (D-014).
    Log(std::format(
        "reclaimed {:.1f} MiB of {:.1f} MiB needed for {}: {} graphs ({:.1f} MiB), {} plans "
        "({:.1f} MiB), {} idle weights ({:.1f} MiB), {} idle conversations {} ({:.1f} MiB); "
        "measured s a GiB: graphs {:.2f}, "
        "plans {:.2f}, idle state {:.2f}; {} history groups discarded ({:.1f} MiB capacity)",
        mib(freed), mib(needed), why, count[k(K::kGraph)], mib(freed_by[k(K::kGraph)]),
        count[k(K::kPlan)], mib(freed_by[k(K::kPlan)]), count[k(K::kIdleWeights)],
        mib(freed_by[k(K::kIdleWeights)]), count[k(K::kIdleState)],
        dropped != 0 ? "dropped" : "spilled", mib(freed_by[k(K::kIdleState)]), cost[k(K::kGraph)],
        cost[k(K::kPlan)], cost[k(K::kIdleState)], count[k(K::kTokenHistory)],
        mib(freed_by[k(K::kTokenHistory)])));
  }
  reclaiming_ = false;
  return freed;
}

Status Server::MakeRoomForSwap(Served& m, std::span<const catalog::ExtentId> out,
                               PendingWeightEvictions* pending) {
  const SwapPhaseTimer timer(options_.diagnostic_budget_cap_bytes ? &swap_host_phases_.room_seconds
                                                                  : nullptr);
  // Size physical occupancy afresh, including required charges above B.
  const auto add = [](std::uint64_t& value, std::uint64_t bytes) {
    if (bytes > UINT64_MAX - value) return false;
    value += bytes;
    return true;
  };
  // Round after round while each makes progress (RoomProgress): the bytes
  // still short shrink every round, so this ends.
  RoomProgress progress;
  for (;;) {
    std::uint64_t need = 0;
    std::uint64_t credit = 0;
    auto sized = node_.Call(
        [&]() -> Status {
          const auto& catalog = node_.catalog();
          std::uint64_t after = catalog.OccupancyOf(node_.domain()).Total().value();
          for (const auto& [extent, generation] : m.everything().extents) {
            const auto view = catalog.Describe(extent);
            if (!view || view->descriptor.domain != node_.domain() ||
                view->content_generation != generation || view->discarded ||
                view->state == catalog::ExtentState::kQuarantined)
              return Error("a swap's incoming closure is stale or invalid");
            if (view->state != catalog::ExtentState::kResident &&
                !add(after, view->descriptor.size.value()))
              return Error("a swap's incoming occupancy overflows");
          }
          std::set<catalog::ExtentId> counted;
          for (const auto extent : out) {
            const auto view = catalog.Describe(extent);
            if (counted.insert(extent).second && view &&
                view->state == catalog::ExtentState::kResident &&
                !add(credit, view->descriptor.size.value()))
              return Error("a swap's outgoing credit overflows");
          }
          need = after > budget_ ? after - budget_ : 0;
          if (pending != nullptr) {
            std::erase_if(pending->selected, [&](const auto& selected) {
              const auto view = catalog.Describe(selected.extent);
              const bool valid = view && view->content_generation == selected.generation &&
                                 view->descriptor.size.value() == selected.candidate.bytes &&
                                 catalog::Catalog::Evictable(*view) &&
                                 !counted.contains(selected.extent);
              if (!valid) pending->rejected.insert(selected.extent);
              return !valid;
            });
            // Keep only the policy-ordered prefix still needed after actual
            // cache-charge rounding. A single final extent may overshoot.
            std::size_t keep = 0;
            while (keep < pending->selected.size() && credit < need) {
              const auto& selected = pending->selected[keep++];
              if (!add(credit, selected.candidate.bytes))
                return Error("a swap's selected weight credit overflows");
            }
            pending->selected.resize(keep);
          }
          return {};
        },
        "sizing actual missing incoming and selected swap extents");
    if (!sized) return sized;
    if (credit >= need) return {};
    if (!progress.Shrank(need - credit)) break;
    const auto selected_before = pending == nullptr ? 0 : pending->selected.size();
    const auto freed = Reclaim(need - credit, false, "a swap's incoming model", &m, std::nullopt,
                               false, nullptr, 0, {}, pending);
    if (freed == 0 && (pending == nullptr || pending->selected.size() == selected_before)) break;
  }
  return Error(std::format("{} to {}: the incoming closure cannot fit without held extents",
                           resident_ != nullptr ? resident_->name() : "a swap", m.name()));
}

bool Server::NodeHealthy() {
  if (!started_ || torn_down_) {
    return false;
  }
  auto checked = node_.Call(
      [&]() -> Status {
        return node_.scheduler().fault() ? Error("the node's scheduler faulted") : Status{};
      },
      "checking the node's health");
  return checked.has_value();
}

std::uint64_t Server::ReleaseKept(std::uint64_t needed, const Served* last,
                                  std::span<const catalog::ExtentId> protect) {
  const std::set<catalog::ExtentId> protected_ids(protect.begin(), protect.end());
  std::vector<catalog::ExtentId> kept;
  for (const bool running : {false, true}) {
    for (const auto& m : models_) {
      if ((m.get() == last) != running) {
        continue;
      }
      const std::vector<catalog::ExtentId> own = m->kept_state();
      kept.insert(kept.end(), own.begin(), own.end());
    }
  }
  if (kept.empty()) {
    return 0;
  }
  std::vector<catalog::ExtentId> resident;
  std::uint64_t bytes = 0;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const catalog::ExtentId extent : kept) {
          if (protected_ids.contains(extent)) continue;
          if (bytes >= needed) {
            break;
          }
          if (const auto view = catalog.Describe(extent);
              view && view->state == catalog::ExtentState::kResident && view->discarded &&
              view->leases == 0 && view->registrations == 0) {
            resident.push_back(extent);
            bytes += view->descriptor.size.value();
          }
        }
        return {};
      },
      "listing kept state backing");
  if (!listed || resident.empty() || !node_.Evict(resident)) {
    return 0;
  }
  return bytes;
}

Status Server::EvictPaged(Served& m, bool weights) {
  // Only what a swap moves: its weights and its conversation state (the
  // shared workspace and its own pinned runtime memory stay where they
  // are), with the zeroed backing its clears kept. State is written back
  // to its places.
  std::vector<catalog::ExtentId> own = weights ? m.weights() : std::vector<catalog::ExtentId>{};
  const std::vector<catalog::ExtentId> state = m.state();
  own.insert(own.end(), state.begin(), state.end());
  const std::vector<catalog::ExtentId> kept = m.kept_state();
  own.insert(own.end(), kept.begin(), kept.end());
  std::vector<catalog::ExtentId> resident;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const catalog::ExtentId extent : own) {
          if (const auto view = catalog.Describe(extent);
              view && view->state == catalog::ExtentState::kResident) {
            resident.push_back(extent);
          }
        }
        return {};
      },
      "listing a model's resident extents");
  if (!listed) {
    return listed;
  }
  return resident.empty() ? Status{} : node_.Evict(resident);
}

std::expected<std::vector<catalog::ExtentId>, std::string> Server::SpareWorkspace(
    const Served& in) {
  std::vector<catalog::ExtentId> spare;
  if (!node_.workspace_pageable()) return spare;
  std::set<catalog::ExtentId> named;
  for (const auto& [extent, generation] : in.everything().extents) {
    (void)generation;
    named.insert(extent);
  }
  const std::vector<catalog::ExtentId> workspace = node_.workspace_extents();
  auto listed = node_.Call(
      [&]() -> Status {
        for (const catalog::ExtentId extent : workspace) {
          if (named.contains(extent)) continue;
          if (const auto view = node_.catalog().Describe(extent);
              view && view->state == catalog::ExtentState::kResident)
            spare.push_back(extent);
        }
        return {};
      },
      "listing the workspace the incoming model does not use");
  if (!listed) return std::unexpected(listed.error());
  return spare;
}

Status Server::FenceModel(Served& m) {
  const std::uint32_t stream = m.paged().stream();
  if (node_.InRequest(stream)) {
    // Its task may have ended with the cancellation: ending it only frees
    // what it held.
    std::ignore = node_.EndRequest(stream);
  }
  return node_.Job(
      catalog::Closure{}, [](providers::NativeStream) { return scheduler::JobResult::kQueued; },
      "fencing a model's stream after a hang", stream);
}

Status Server::RecoverModel(Served& m) {
  if (!NodeHealthy()) {
    return Error("the node's scheduler faulted");
  }
  if (auto fenced = FenceModel(m); !fenced) {
    return Error(std::format("its stream could not be fenced: {}", fenced.error()));
  }
  if (auto recovered = m.RecoverInPlace(); !recovered) {
    return recovered;
  }
  if (m.llm()) {
    // The hung unit may have been timed: no calibration counts it (D-103).
    static_cast<Llm&>(m).calibration_samples().Forget();
  }
  // Evicted, so its next activation loads it whole: its weights, and the
  // state it kept (idle conversations) written back.
  if (auto evicted = EvictPaged(m); !evicted) {
    return Error(std::format("evicting it: {}", evicted.error()));
  }
  if (resident_ == &m) {
    resident_ = nullptr;
  }
  DeactivateIdleModels();  // its weight files close: it pages in again only once activated
  if (m.llm() && m.HasRetainedState() && std::ranges::find(spilled_, &m) == spilled_.end()) {
    // As a swap's write-back: the conversations it kept are wholly on disk,
    // and their records follow (D-105).
    m.StateWrittenBack(true);
    spilled_.push_back(&m);
    auto& l = static_cast<Llm&>(m);
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i); b && !(*b)->spilled()) {
        l.KeepBranch(**b);
      }
    }
  }
  return {};
}

Status Server::ActivateFiles(const Served& m) {
  for (std::size_t i = 0; i < models_.size(); ++i) {
    if (models_[i].get() == &m) {
      if (auto opened = node_.SetActive(static_cast<int>(i), true); !opened) {
        return Error(
            std::format("model {}: its weights cannot be read: {}", m.name(), opened.error()));
      }
    }
  }
  return {};
}

void Server::DeactivateIdleModels() {
  for (std::size_t i = 0; i < models_.size(); ++i) {
    if (models_[i].get() != resident_) {
      // Never refused: a file a read still uses closes at a later sync.
      std::ignore = node_.SetActive(static_cast<int>(i), false);
    }
  }
}

Status Server::UndoSwap(Served& out, Served& in, std::span<const catalog::ExtentId> preexisting) {
  const std::set<catalog::ExtentId> kept(preexisting.begin(), preexisting.end());
  std::vector<catalog::ExtentId> newly_resident;
  if (auto listed = node_.Call(
          [&]() -> Status {
            for (const auto& [extent, generation] : in.everything().extents) {
              (void)generation;
              const auto view = node_.catalog().Describe(extent);
              if (!kept.contains(extent) && view && view->state == catalog::ExtentState::kResident)
                newly_resident.push_back(extent);
            }
            return {};
          },
          "listing only newly loaded extents of a failed swap");
      !listed)
    return listed;
  if (!newly_resident.empty())
    if (auto evicted = node_.Evict(newly_resident); !evicted) return evicted;
  for (const auto& [extent, generation] : out.everything().extents) {
    (void)generation;
    activation_protect_.push_back(extent);
  }
  if (auto room = MakeRoomForSwap(out, {}); !room) return room;
  std::vector<catalog::ExtentId> again;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const auto& [extent, generation] : out.everything().extents) {
          (void)generation;
          if (const auto view = catalog.Describe(extent);
              view && view->state != catalog::ExtentState::kResident) {
            again.push_back(extent);
          }
        }
        return {};
      },
      "listing a failed swap's extents");
  if (!listed) {
    return listed;
  }
  if (!again.empty()) {
    std::vector<engine::LoadStats> log;
    if (auto r = node_.Load(again, std::format("{} back after a failed swap", out.name()), log);
        !r) {
      return r;
    }
  }
  return out.AfterLoad();  // its checks after a load, as after a swap
}

void Server::RecordCalibrations() {
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    Llm::CalibrationRecord& record = l.calibration_record();
    auto measured = l.calibration_samples().Take(record.known);
    if (!measured) {
      continue;
    }
    // Recorded for the next registration; this service keeps its settings,
    // so a running schedule never changes with wall time (calibration.h).
    if (auto written = WriteCalibration(roles_.state, record.key, *measured); !written) {
      Log(std::format("model {}: calibration not recorded: {}", m->name(), written.error()));
      continue;
    }
    record.known = *measured;
    Log(std::format("model {}: calibration recorded, in force from the next start: {}", m->name(),
                    FormatCalibration(*measured, record.key)));
  }
}

void Server::Maintain() {
  if (!started_ || torn_down_ || reclaiming_) {
    return;
  }
  token_memory_.Settle(0, std::chrono::milliseconds{0});
  const auto now = Clock::now();
  if (now < next_maintenance_) {
    return;
  }
  next_maintenance_ = now + kMaintenanceInterval;
  // Idle conversations past their retention, resident or spilled.
  std::size_t expired = 0;
  std::uint64_t expired_bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    for (std::size_t i = 0; i < l.branches(); ++i) {
      auto b = l.branch(i);
      if (b) {
        (void)l.ExpireTurnCheckpoints(**b, now);
      }
      if (!b || !l.BranchIdle(**b) || (*b)->HeldByContinuation() || (*b)->history().empty() ||
          now - l.LastUsed(**b) < retention_) {
        continue;
      }
      const std::uint64_t bytes = l.StateBytes(**b);
      if (auto released = (*b)->ReleaseIdleState(); released) {
        ++expired;
        expired_bytes += bytes;
      }
    }
  }
  if (expired != 0) {
    Log(std::format("deleted {} idle conversations ({:.1f} MiB) past their retention of {} h",
                    expired, static_cast<double>(expired_bytes) / (1U << 20U),
                    config_.memory.retention_hours));
  }
  // As far as it can: what a request holds stays, the rest still goes.
  (void)DeleteSpilledPast(0);
  RecordCalibrations();
  // Pressure from outside the runtime: other processes share the node's
  // memory (D-004). One reclaim of what would restore the target headroom,
  // spaced by a back-off while the pressure persists (PressureTrim), never
  // the running model's in-use floor (Reclaim).
  const platform::MemoryPressure pressure = platform::ReadMemoryPressure();
  const PressureTrim::Trim trim = pressure_.Observe(pressure, now);
  if (!trim.now) {
    return;
  }
  const std::uint64_t freed = Reclaim(trim.needed, true, trim.why, nullptr, std::nullopt, true);
  if (pressure_.Trimmed(freed, trim.needed, now)) {
    const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / (1U << 20U); };
    Log(std::format(
        "memory pressure from outside ({:.0f} MiB available, a full stall {:.1f}% of the last 10 "
        "s): gave back {:.1f} MiB of {:.1f} MiB, too little left to give without the running "
        "model's floor; looking again in {} s ({} trims, {} short)",
        mib(pressure.available.value_or(0)), pressure.stall ? pressure.stall->full_avg10 : 0.0,
        mib(freed), mib(trim.needed),
        std::chrono::duration_cast<std::chrono::seconds>(pressure_.backoff()).count(),
        pressure_.trims(), pressure_.short_trims()));
  }
}

Status Server::Activate(Served& m, SwapParts& parts, std::optional<bool> spill_state) {
  // Records' hashing waits while a swap moves the models (StateKeeper::
  // Quiet): their reads and the swap's share the drive.
  const StateKeeper::Quiet quiet(keeper_.get());
  parts = SwapParts{};
  parts.to = m.name();
  if (resident_ == &m) {
    return {};
  }
  parts.requested = Clock::now();
  // The activation is the incoming model's page-in need: its weight files
  // open now (inside the swap's timing), and whichever model is not
  // resident when this returns has its closed (the outgoing one's once its
  // weights are out).
  if (auto files = ActivateFiles(m); !files) {
    return files;
  }
  struct IdleFiles {
    Server& server;
    ~IdleFiles() { server.DeactivateIdleModels(); }
  } idle_files{*this};
  if (options_.diagnostic_budget_cap_bytes) {
    parts.diagnostic.enabled = true;
    parts.diagnostic.requested = swap_host_phases_;
    parts.diagnostic.creates_requested = node_.backing_create_stats();
    parts.diagnostic.requested_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(parts.requested.time_since_epoch())
            .count());
  }
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  swap_before_ = *before;
  parts.occupancy_requested = before->catalog_occupancy_bytes;
  Served* const outgoing_model = resident_;
  const auto protected_before = activation_protect_.size();
  struct ProtectedActivation {
    std::vector<catalog::ExtentId>& ids;
    std::size_t before;
    ~ProtectedActivation() { ids.resize(before); }
  } protection{activation_protect_, protected_before};
  for (const auto& [extent, generation] : m.everything().extents) {
    (void)generation;
    activation_protect_.push_back(extent);
  }
  std::vector<catalog::ExtentId> preexisting;
  catalog::Closure saved_state;
  bool state_was_spilled = true;
  if (auto snapshot = node_.Call(
          [&]() -> Status {
            for (const auto& [extent, generation] : m.everything().extents) {
              const auto view = node_.catalog().Describe(extent);
              if (!view || view->content_generation != generation || view->discarded ||
                  view->state == catalog::ExtentState::kQuarantined)
                return Error("the incoming activation closure is stale or invalid");
              if (view->state == catalog::ExtentState::kResident) preexisting.push_back(extent);
            }
            for (const auto extent : m.state()) {
              const auto view = node_.catalog().Describe(extent);
              if (!view) return Error("the incoming state extent is absent");
              saved_state.extents.push_back({extent, view->content_generation});
              state_was_spilled =
                  state_was_spilled && view->state == catalog::ExtentState::kNonresident;
            }
            const std::set<catalog::ExtentId> cached(preexisting.begin(), preexisting.end());
            for (const auto extent : m.weights()) {
              if (cached.contains(extent))
                parts.retained_incoming_weight_bytes +=
                    node_.catalog().Describe(extent)->descriptor.size.value();
            }
            return {};
          },
          "snapshotting incoming cache and saved state before activation");
      !snapshot)
    return snapshot;
  PendingWeightEvictions pending;
  const auto completed_before = node_.counting().read_completed_bytes.load();
  const auto submitted_before = node_.counting().read_submitted_bytes.load();
  const bool restoring = std::ranges::find(spilled_, &m) != spilled_.end();
  // The incoming model's conversations a swap wrote back come back live:
  // their records go before anything of it loads (D-105), not before a
  // refusal that leaves them on disk. Those the reclaim order spilled stay
  // on disk, and kept, until a turn restores them.
  const auto unkeep_incoming = [&m]() {
    if (m.llm()) {
      auto& in = static_cast<Llm&>(m);
      for (std::size_t i = 0; i < in.branches(); ++i) {
        if (auto b = in.branch(i); b && !(*b)->spilled()) {
          in.Unkeep(**b);
        }
      }
    }
  };
  const auto rekeep_incoming = [&]() {
    if (!restoring || !state_was_spilled || saved_state.extents.empty() || !m.llm()) return;
    bool intact = true;
    const auto checked = node_.Call(
        [&]() -> Status {
          for (const auto& [extent, generation] : saved_state.extents) {
            const auto view = node_.catalog().Describe(extent);
            intact = intact && view && view->content_generation == generation &&
                     view->state == catalog::ExtentState::kNonresident && !view->discarded;
          }
          return {};
        },
        "proving a failed incoming load still has whole saved generations");
    if (!checked || !intact) return;
    m.StateWrittenBack(true);
    auto& in = static_cast<Llm&>(m);
    for (std::size_t i = 0; i < in.branches(); ++i)
      if (auto branch = in.branch(i); branch && !(*branch)->spilled()) in.KeepBranch(**branch);
  };
  const auto discard_new_incoming = [&]() -> Status {
    std::vector<catalog::ExtentId> newly_loaded;
    const std::set<catalog::ExtentId> cached(preexisting.begin(), preexisting.end());
    if (auto listed = node_.Call(
            [&]() -> Status {
              for (const auto& [extent, generation] : m.everything().extents) {
                (void)generation;
                const auto view = node_.catalog().Describe(extent);
                if (!cached.contains(extent) && view &&
                    view->state == catalog::ExtentState::kResident)
                  newly_loaded.push_back(extent);
              }
              return {};
            },
            "listing only additions to the incoming activation cache");
        !listed)
      return listed;
    return newly_loaded.empty() ? Status{} : node_.Evict(newly_loaded);
  };
  const auto account_pending = [&]() {
    std::vector<double> priorities;
    std::ignore = node_.Call(
        [&]() -> Status {
          for (const auto& selected : pending.selected) {
            const auto view = node_.catalog().Describe(selected.extent);
            if (view && view->content_generation == selected.generation &&
                (view->state == catalog::ExtentState::kNonresident ||
                 node_.scheduler().EvictionUnmapped(selected.extent))) {
              parts.evicted_weight_bytes += selected.candidate.bytes;
              priorities.push_back(selected.priority);
            }
          }
          return {};
        },
        "accounting only completed or parked selected weight evictions");
    for (const auto priority : priorities) memory::RaiseReclaimInflation(priority);
  };
  bool outgoing_published = false;
  bool outgoing_transfer_complete = false;
  catalog::Closure saved_outgoing;
  const auto publish_outgoing = [&]() {
    if (outgoing_published || outgoing_model == nullptr || !parts.with_state) return;
    outgoing_published = true;
    outgoing_model->StateWrittenBack(true);
    if (std::ranges::find(spilled_, outgoing_model) == spilled_.end())
      spilled_.push_back(outgoing_model);
    auto& out = static_cast<Llm&>(*outgoing_model);
    for (std::size_t i = 0; i < out.branches(); ++i)
      if (auto branch = out.branch(i); branch && !(*branch)->spilled()) out.KeepBranch(**branch);
  };
  const auto keep_outgoing_after_failed_undo = [&]() {
    if (!outgoing_transfer_complete || outgoing_model == nullptr || !parts.with_state ||
        saved_outgoing.extents.empty())
      return;
    std::vector<catalog::ExtentId> restored;
    auto proved = node_.Call(
        [&]() -> Status {
          for (const auto& [extent, generation] : saved_outgoing.extents) {
            const auto view = node_.catalog().Describe(extent);
            const auto* source = node_.scheduler().SourceOf(extent);
            if (!view || view->content_generation != generation ||
                view->saved_generation != generation || view->discarded || source == nullptr ||
                !source->write_back ||
                (view->state != catalog::ExtentState::kResident &&
                 view->state != catalog::ExtentState::kNonresident))
              return Error("outgoing saved generations cannot be proven after failed undo");
            if (view->state == catalog::ExtentState::kResident)
              restored.push_back(extent);
            else if (!view->preserved)
              return Error("outgoing saved state is not preserved");
          }
          return {};
        },
        "proving the completed outgoing spill after failed undo");
    // The successful original transfer saved every extent. The undo only
    // restored it; no outgoing request ran. Release that restored subset
    // unchanged before advertising the model as wholly spilled again.
    if (proved && !restored.empty()) proved = node_.Evict(restored, {.unchanged = true});
    if (proved)
      proved = node_.Call(
          [&]() -> Status {
            for (const auto& [extent, generation] : saved_outgoing.extents) {
              const auto view = node_.catalog().Describe(extent);
              if (!view || view->content_generation != generation ||
                  view->saved_generation != generation || !view->preserved ||
                  view->state != catalog::ExtentState::kNonresident)
                return Error("outgoing state did not retire wholly to its saved generations");
            }
            return {};
          },
          "checking whole outgoing spill retirement after failed undo");
    if (proved)
      publish_outgoing();
    else
      Log(std::format("failed undo's outgoing state was not kept: {}", proved.error()));
  };
  const auto release_subset = [&](std::span<const catalog::ExtentId> selected,
                                  std::vector<catalog::ExtentId>& release) -> Status {
    if (!options_.partial_weight_eviction || !handoff_) return {};
    const SwapPhaseTimer timer(
        options_.diagnostic_budget_cap_bytes ? &swap_host_phases_.partition_seconds : nullptr);
    std::vector<SwapBackingExtent> missing;
    std::vector<SwapBackingExtent> donors;
    std::uint64_t occupancy = 0;
    const auto listed = node_.Call(
        [&]() -> Status {
          const auto describe =
              [&](catalog::ExtentId extent) -> std::expected<SwapBackingExtent, std::string> {
            const auto view = node_.catalog().Describe(extent);
            const auto* source = node_.scheduler().SourceOf(extent);
            if (!view || source == nullptr)
              return Error("a swap extent has no descriptor or source");
            SwapBackingExtent result{
                .extent = extent, .bytes = view->descriptor.size.value(), .backing = std::nullopt};
            if (source->backing && source->backing->size.value() == result.bytes)
              result.backing = SwapBackingKey{.domain = view->descriptor.domain,
                                              .allocation_class = source->backing->allocation_class,
                                              .size = source->backing->size.value()};
            return result;
          };
          for (const auto& [extent, generation] : m.everything().extents) {
            (void)generation;
            const auto view = node_.catalog().Describe(extent);
            if (view && view->state != catalog::ExtentState::kResident) {
              auto entry = describe(extent);
              if (!entry) return std::unexpected(entry.error());
              missing.push_back(*entry);
            }
          }
          for (const auto extent : selected) {
            const auto view = node_.catalog().Describe(extent);
            if (view && view->state == catalog::ExtentState::kResident) {
              auto entry = describe(extent);
              if (!entry) return std::unexpected(entry.error());
              donors.push_back(*entry);
            }
          }
          occupancy = node_.catalog().OccupancyOf(node_.domain()).Total().value();
          return {};
        },
        "partitioning selected donor backing by actual class and size");
    if (!listed) return listed;
    const auto partition = ReleaseForHandoff(node_.domain(), occupancy, budget_, missing, donors);
    if (!partition) return std::unexpected(partition.error());
    release = *partition;
    return {};
  };
  const auto selected_weights = [&]() {
    catalog::Closure closure;
    for (const auto& selected : pending.selected)
      closure.extents.push_back({selected.extent, selected.generation});
    return closure;
  };
  const auto record_eviction_split = [&](const scheduler::SwapReport& report) {
    if (!parts.diagnostic.enabled || report.started == Clock::time_point{}) return;
    parts.diagnostic.eviction_split_available = true;
    const auto ns = [](Clock::time_point at) {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(at.time_since_epoch()).count());
    };
    parts.diagnostic.scheduler_started_ns = ns(report.started);
    parts.diagnostic.scheduler_evicted_ns = ns(report.evicted);
    parts.diagnostic.scheduler_loaded_ns = ns(report.loaded);
    parts.diagnostic.eviction_prepare_seconds = Seconds(report.started - parts.requested);
    parts.diagnostic.eviction_retire_seconds = Seconds(report.evicted - report.started);
  };
  Clock::time_point evicted = parts.requested;
  Clock::time_point loaded;
  if (resident_ == nullptr) {
    // Nothing resident: the first load of its weights (its state is
    // resident since setup, or written back by a swap that failed and
    // could not be undone: restored with them).
    // Whatever another model still holds from a swap that failed and could
    // not be undone goes out first (state written back).
    for (const auto& other : models_) {
      if (other.get() != &m) {
        if (auto r = EvictPaged(*other, !options_.partial_weight_eviction); !r) {
          return r;
        }
        if (other->llm() && other->HasRetainedState() &&
            std::ranges::find(spilled_, other.get()) == spilled_.end()) {
          spilled_.push_back(other.get());
        }
      }
    }
    // And the workspace only they used: the incoming closure names its own.
    auto spare = SpareWorkspace(m);
    if (!spare) return std::unexpected(spare.error());
    if (!spare->empty()) {
      if (auto r = node_.Evict(std::move(*spare)); !r) return r;
    }
    // Every model's plans and graphs stay charged inside the budget: room
    // for the whole load is made through the reclaim order first, as a
    // swap's is, or a load after a failed swap (or a recovery) could be
    // refused for room again and again with nothing to free it.
    if (auto room = MakeRoomForSwap(m, {}, options_.partial_weight_eviction ? &pending : nullptr);
        !room) {
      return room;
    }
    std::vector<engine::LoadStats> log;
    std::vector<catalog::ExtentId> all;
    for (const auto& [extent, generation] : m.everything().extents) {
      (void)generation;
      all.push_back(extent);
    }
    Status loaded_result;
    if (options_.partial_weight_eviction) {
      std::vector<catalog::ExtentId> victims;
      for (const auto& selected : pending.selected) victims.push_back(selected.extent);
      std::vector<catalog::ExtentId> release;
      if (auto partition = release_subset(victims, release); !partition) return partition;
      scheduler::SwapReport report;
      unkeep_incoming();
      loaded_result = node_.Swap(std::move(victims), m.everything(), handoff_, report,
                                 std::move(release), selected_weights());
      account_pending();
      if (loaded_result) record_eviction_split(report);
      evicted = report.evicted == Clock::time_point{} ? evicted : report.evicted;
      parts.evicted = report.evictions;
      parts.loaded = report.loads;
    } else {
      unkeep_incoming();
      loaded_result = node_.Load(all, std::format("{}'s first load", m.name()), log);
      parts.loaded = log.empty() ? 0 : log.front().extents;
    }
    if (!loaded_result) {
      if (auto cleaned = discard_new_incoming(); !cleaned)
        Log(std::format("failed first-load additions could not be cleaned: {}", cleaned.error()));
      rekeep_incoming();
      return loaded_result;
    }
    loaded = Clock::now();
  } else {
    Served& out = *resident_;
    parts.from = out.name();
    // The caller yields only at a completed unit. Release its request even
    // when the incoming closure is cached and no weight victim is needed.
    if (node_.InRequest(out.paged().stream())) {
      if (auto ended = node_.EndRequest(out.paged().stream()); !ended) return ended;
    }
    if (auto used = RecordWeightUse(out); !used) return used;
    const bool conversation = out.llm() && out.HasRetainedState();
    parts.with_state = out.llm() && spill_state.value_or(conversation);
    std::vector<catalog::ExtentId> extents;
    std::vector<catalog::ExtentId> unchanged;
    const SpillView view{.out = &out, .in = &m};
    if (parts.with_state) {
      auto& l = static_cast<Llm&>(out);
      // The spill budget holds a swap's write-back too, counted as the swap
      // leaves it: all the outgoing model's state on disk, the incoming
      // model's resident (its conversations, which then stop counting, are
      // never deleted for it). Past it, the least recently used spilled
      // conversations are deleted first, the outgoing model's among them,
      // but only once the swap's room is made below: one a request holds
      // is not, and a swap that would not fit even with every other one
      // deleted is refused here, before anything moves or is deleted.
      if (!SpillBudgetFits(0, view)) {
        return Error("the swap's retained request state exceeds the spill budget");
      }
      // Kept across a restart (D-105): each conversation's last verify's
      // owed restore runs first (no request holds the stream during a
      // swap), so the write-back holds it whole. Before the room is sized:
      // whatever it charges is counted there.
      if (l.keeps()) {
        for (std::size_t i = 0; i < l.branches(); ++i) {
          if (auto b = l.branch(i); b && l.ResidentStateBytes(**b) != 0) {
            (void)l.SettleIdle(**b);
          }
        }
      }
    }
    if (options_.partial_weight_eviction && parts.with_state) {
      if (auto saved = node_.Call(
              [&]() -> Status {
                for (const auto extent : out.state()) {
                  const auto view = node_.catalog().Describe(extent);
                  if (!view) return Error("the settled outgoing state extent is absent");
                  saved_outgoing.extents.push_back({extent, view->content_generation});
                }
                return {};
              },
              "snapshotting settled outgoing state generations");
          !saved)
        return saved;
    }
    // Its plans and graphs stay (D-090 as amended 2026-10-02): charged
    // inside the budget, they go only when the reclaim order needs their
    // room, here first if the incoming model does not fit beside them. The
    // room is checked again after each reclaim, and a swap it cannot make
    // room for is refused before anything moves. What goes out is its
    // weights and, with its state, all of that (how much a write-back
    // writes, and whether the budget deletes some of it, does not change
    // what the swap frees; the reclaim order spills no idle state here).
    const std::vector<catalog::ExtentId> weights = out.weights();
    const std::vector<catalog::ExtentId> kept = out.kept_state();
    // What of the shared workspace only the outgoing model used goes out
    // with its weights (scratch: nothing is written back), so the incoming
    // model's closure, which names its own prefix, fits beside the rest.
    auto spare = SpareWorkspace(m);
    if (!spare) return std::unexpected(spare.error());
    const std::uint64_t graphs_before = out.graphs().kept + m.graphs().kept;
    {
      std::vector<catalog::ExtentId> going =
          options_.partial_weight_eviction ? std::vector<catalog::ExtentId>{} : weights;
      going.insert(going.end(), kept.begin(), kept.end());
      going.insert(going.end(), spare->begin(), spare->end());
      if (parts.with_state) {
        const std::vector<catalog::ExtentId> state = out.state();
        going.insert(going.end(), state.begin(), state.end());
        for (const catalog::ExtentId extent : out.unchanged_state()) {
          if (std::ranges::find(state, extent) == state.end()) {
            going.push_back(extent);
          }
        }
      }
      if (auto room =
              MakeRoomForSwap(m, going, options_.partial_weight_eviction ? &pending : nullptr);
          !room) {
        return room;
      }
    }
    if (parts.with_state) {
      auto& l = static_cast<Llm&>(out);
      // The room is made: the budget's deletions now, which the check above
      // found enough (nothing since spilled or restored; a deletion that
      // fails still refuses the swap before anything moves).
      if (!KeepWithinSpillBudget(0, view)) {
        return Error("the swap's retained request state exceeds the spill budget");
      }
      // Its state first, so its write-backs start first; what nothing wrote
      // since its slot's spill file last held it is released without
      // writing (an incremental spill), before the swap.
      unchanged = out.unchanged_state();
      for (const catalog::ExtentId extent : out.state()) {
        if (std::ranges::find(unchanged, extent) == unchanged.end()) {
          extents.push_back(extent);
        }
      }
      // What it writes back: its resident state (a conversation the reclaim
      // order spilled already is on disk, and passed over).
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b) {
          parts.spilled_bytes += l.ResidentStateBytes(**b);
        }
      }
    }
    extents.insert(extents.end(), kept.begin(), kept.end());
    if (options_.partial_weight_eviction) {
      for (const auto& selected : pending.selected) extents.push_back(selected.extent);
    } else {
      extents.insert(extents.end(), weights.begin(), weights.end());
    }
    extents.insert(extents.end(), spare->begin(), spare->end());
    const std::uint64_t graphs_after = out.graphs().kept + m.graphs().kept;
    parts.dropped_graphs = graphs_before > graphs_after ? graphs_before - graphs_after : 0;
    unkeep_incoming();
    if (!unchanged.empty()) {
      if (auto r = node_.Evict(unchanged, {.unchanged = true}); !r) {
        out.StateWrittenBack(false);
        if (auto undone = UndoSwap(out, m, preexisting); !undone) {
          resident_ = nullptr;  // the next activation loads one whole
        }
        rekeep_incoming();
        return r;
      }
    }
    scheduler::SwapReport report;
    std::vector<catalog::ExtentId> release;
    if (auto partition = release_subset(extents, release); !partition) {
      if (auto undone = UndoSwap(out, m, preexisting); !undone) resident_ = nullptr;
      rekeep_incoming();
      return partition;
    }
    const auto swapped = node_.Swap(std::move(extents), m.everything(), handoff_, report,
                                    std::move(release), selected_weights());
    account_pending();
    if (!swapped) {
      const auto& r = swapped;
      if (parts.with_state) {
        out.StateWrittenBack(false);
      }
      // Never the process: whatever of the incoming model came in goes out
      // again and the outgoing one comes back whole, so both stay usable and
      // only the request that needed the swap fails.
      if (auto undone = UndoSwap(out, m, preexisting); undone) {
        Log(std::format("swap {} -> {} failed and was undone: {}", out.name(), m.name(),
                        r.error()));
      } else {
        Log(
            std::format("swap {} -> {} failed ({}) and could not be undone ({}): no model is "
                        "resident until the next one loads whole",
                        out.name(), m.name(), r.error(), undone.error()));
        if (parts.with_state && std::ranges::find(spilled_, &out) == spilled_.end()) {
          spilled_.push_back(&out);
        }
        resident_ = nullptr;
      }
      rekeep_incoming();
      return r;
    }
    outgoing_transfer_complete = true;
    record_eviction_split(report);
    if (!options_.partial_weight_eviction) publish_outgoing();
    evicted = report.evicted;
    loaded = report.loaded;
    parts.evicted = report.evictions;
    parts.loaded = report.loads;
  }
  if (parts.diagnostic.enabled) parts.diagnostic.creates_loaded = node_.backing_create_stats();
  parts.evict = Seconds(evicted - parts.requested);
  Clock::time_point restored = evicted;
  if (restoring) {
    restored = std::max(restored, times_.Latest(m.state()));
    parts.restore = Seconds(restored - evicted);
    if (!options_.partial_weight_eviction) std::erase(spilled_, &m);
  }
  parts.page_in = Seconds(loaded - restored);
  parts.read_bytes = node_.counting().read_completed_bytes.load() - completed_before;
  parts.read_submitted_bytes = node_.counting().read_submitted_bytes.load() - submitted_before;
  if (outgoing_model != nullptr) {
    std::ignore = node_.Call(
        [&]() -> Status {
          for (const auto extent : outgoing_model->weights()) {
            const auto view = node_.catalog().Describe(extent);
            if (view && view->state == catalog::ExtentState::kResident)
              parts.retained_outgoing_weight_bytes += view->descriptor.size.value();
            else if (!options_.partial_weight_eviction && view &&
                     (view->state == catalog::ExtentState::kNonresident ||
                      node_.scheduler().EvictionUnmapped(extent)))
              parts.evicted_weight_bytes += view->descriptor.size.value();
          }
          return {};
        },
        "measuring actual retained outgoing weights");
  }
  resident_ = &m;
  // A model whose checks after its load fail is not left resident: its
  // kernels index what those checks cover unchecked (DeepSeek's hash
  // routing, Qwen3.8's n-gram hash) or replay graphs at its places (D-090).
  // Evicted (its state written back), its next activation loads it whole
  // and checks it again; only the requests that needed it fail (D-102).
  const auto unload = [&](std::string error) -> Status {
    if (options_.partial_weight_eviction && outgoing_model != nullptr) {
      if (auto undone = UndoSwap(*outgoing_model, m, preexisting); undone)
        resident_ = outgoing_model;
      else {
        error += std::format("; undoing it: {}", undone.error());
        resident_ = nullptr;
        keep_outgoing_after_failed_undo();
      }
      rekeep_incoming();
      return Error(std::move(error));
    }
    if (options_.partial_weight_eviction) {
      if (auto cleaned = discard_new_incoming(); !cleaned)
        error += std::format("; cleaning its new extents: {}", cleaned.error());
      resident_ = nullptr;
      rekeep_incoming();
      return Error(std::move(error));
    }
    if (auto evicted = EvictPaged(m); !evicted) {
      error += std::format("; evicting it: {}", evicted.error());
      m.StateWrittenBack(false);
    } else if (m.llm() && m.HasRetainedState() &&
               std::ranges::find(spilled_, &m) == spilled_.end()) {
      // As a swap's write-back: its conversations are wholly on disk, and
      // their records follow (D-105).
      m.StateWrittenBack(true);
      spilled_.push_back(&m);
      auto& l = static_cast<Llm&>(m);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b && !(*b)->spilled()) {
          l.KeepBranch(**b);
        }
      }
    }
    resident_ = nullptr;
    return Error(std::move(error));
  };
  if (auto r = m.AfterLoad(); !r) {
    return unload(std::format("{} after its load: {}", m.name(), r.error()));
  }
  // Its places still pinned where it registered them (D-090).
  std::size_t unpinned = 0;
  const std::vector<catalog::ExtentId> managed = m.paged().managed_extents();
  if (auto r = node_.Call(
          [&]() -> Status {
            for (const catalog::ExtentId extent : managed) {
              unpinned += node_.scheduler().PlacePinned(extent) ? 0 : 1;
            }
            return {};
          },
          "checking places");
      !r) {
    return unload(std::format("checking incoming places failed: {}", r.error()));
  }
  if (unpinned != 0) {
    return unload(std::format("{} of {}'s {} extents are not pinned at their places", unpinned,
                              m.name(), managed.size()));
  }
  if (auto r = m.CheckPlaces(); !r) {
    return unload(std::format("{}'s places: {}", m.name(), r.error()));
  }
  publish_outgoing();
  if (restoring) std::erase(spilled_, &m);
  page_read_bytes_ += parts.read_bytes;
  page_read_seconds_ += parts.page_in + parts.restore;
  parts.ready = Clock::now();
  parts.setup = Seconds(parts.ready - loaded);
  parts.total = Seconds(parts.ready - parts.requested);
  if (parts.diagnostic.enabled) {
    parts.diagnostic.ready = swap_host_phases_;
    parts.diagnostic.creates_ready = node_.backing_create_stats();
  }
  return {};
}

}  // namespace llmp::runtime
