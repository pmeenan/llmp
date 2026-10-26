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
constexpr auto kMaintenanceInterval = std::chrono::milliseconds(250);

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

namespace {
engine::NodeSettings MakeNodeSettings(const config::NodeConfig& config,
                                      const ServingOptions& options,
                                      scheduler::PageInObserver* observer) {
  const auto mode = config.memory.mode.value;
  const bool preallocate =
      (mode == config::MemoryMode::kDedicated && !options.handle_reserve.has_value());
  const std::size_t reserve = options.handle_reserve.value_or(0);
  return engine::NodeSettings{
      .compute_streams = std::max<std::size_t>(config.models.size(), 1),
      .slots = engine::kPagedSlots,
      .inline_lanes = false,
      .coalesce = false,
      .copy_lane = true,
      .lazy_handoff = options.lazy_handoff,
      .zero_state = options.zero_state,
      .handle_reserve = reserve,
      .memory_mode = (mode == config::MemoryMode::kShared) ? engine::MemoryMode::kShared
                                                           : engine::MemoryMode::kDedicated,
      .preallocate_pool = preallocate,
      .slot_bytes = engine::kSlabSlotBytes,
      .observer = observer,
      .poll_window = std::nullopt,
      .spin_ahead = std::nullopt,
      .quiet = std::chrono::minutes(10),
  };
}
}  // namespace

Server::Server(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               const ServingOptions& options, std::FILE* log)
    : config_(config),
      roles_(roles),
      options_(options),
      log_(log),
      times_(kObservedExtents),
      node_(MakeNodeSettings(config, options, &times_)),
      retention_(std::chrono::hours(config.memory.retention_hours)),
      spill_budget_(std::uint64_t{config.memory.spill_budget_gib} << 30U),
      pressure_({
          .low_bytes = config.memory.pressure_low_mib.value << 20U,
          .headroom_bytes = config.memory.pressure_headroom_mib.value << 20U,
          .full_stall_percent = config.memory.pressure_stall_percent.value,
          .first_backoff = std::chrono::milliseconds(config.memory.pressure_first_backoff_ms.value),
          .most_backoff = std::chrono::milliseconds(config.memory.pressure_max_backoff_ms.value),
          .log_interval = std::chrono::seconds(config.memory.pressure_log_interval_seconds.value),
      }) {}

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

Status Server::Make(const config::ModelEntry& entry, const ModelSettings& settings, int index,
                    std::shared_ptr<OpenedModelArtifacts> opened) {
  // A model on the component engine: the adapter of its spec's pipeline
  // kind (its settings', from its stored spec). Every LLM uses one declared
  // backend route, with an execution envelope from its actual source/state
  // and automatic startup payment by default (Start).
  if (!entry.composition) {
    switch (settings.pipeline) {
      case spec::PipelineKind::kLlm: {
        if (AutomaticStartup() && !compiler_)
          compiler_ = std::make_shared<engine::SharedCompilerAllowance>();
        auto llm = std::make_unique<Llm>(
            node_, entry, settings, roles_, index,
            ComponentLlmBuild(node_, roles_.installed, entry, index,
                              {
                                  .target_policy = options_.diagnostic_target_policy,
                                  .drafter_policy = options_.diagnostic_drafter_policy,
                                  .capture_first_verify = options_.diagnostic_capture_first_verify,
                                  .trial_mixed_outputs = options_.diagnostic_mixed_outputs,
                                  .trial_mixed_causal = options_.diagnostic_mixed_causal,
                                  .automatic_startup_cap = AutomaticStartup() ? startup_cap_ : 0,
                                  .shared_compiler = compiler_,
                                  .prefill_lookahead = options_.prefill_lookahead,
                                  .prepare_state = options_.prepare_state,
                                  .state_only_prefill = options_.state_only_prefill,
                                  .prepared = std::move(opened),
                              }),
            startup_reader_);
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
    startup_reader_lifetime_ = std::make_shared<const std::byte>();
    startup_reader_ = StartupReaderLoan(node_.catalog(), node_.domain(), *extent, bytes,
                                        startup_reader_lifetime_);
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
  std::vector<std::shared_ptr<OpenedModelArtifacts>> opened_artifacts;
  settings.reserve(config_.models.size());
  opened_artifacts.reserve(config_.models.size());
  if (const std::string note = CalibrationDirectoryNote(roles_.state); !note.empty()) {
    Log(std::format("warning: the calibration directory {}/{}: {}; used as found",
                    roles_.state.string(), config::kCalibrationDirectory, note));
  }
  for (const config::ModelEntry& entry : config_.models) {
    auto opened = std::make_shared<OpenedModelArtifacts>();
    const StartupArtifactFunder funder = [this](const artifact::Artifact& target,
                                                const artifact::Artifact* drafter)
        -> std::expected<std::unique_ptr<StartupArtifactPayment>, std::string> {
      auto bytes = StartupModelBytes(target, drafter);
      if (!bytes) return Error(bytes.error());
      return StartupArtifactPayment::Admit(node_.catalog(), node_.domain(), *bytes, startup_cap_,
                                           MemorySampler::Available());
    };
    auto facts = ReadArtifactFacts(entry, roles_.installed, opened.get(), funder);
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
    opened_artifacts.push_back(std::move(opened));
  }
  if (!prepaid) {
    if (auto r = node_.Open(); !r) return r;
  }
  // Each model's stream and owner: its place among the registered.
  for (std::size_t i = 0; i < config_.models.size(); ++i) {
    const std::size_t before = models_.size();
    if (auto r = Make(config_.models[i], settings[i], static_cast<int>(models_.size()),
                      std::move(opened_artifacts[i]));
        !r) {
      return r;
    }
    if (models_.size() > before && models_.back()->llm()) {
      static_cast<Llm&>(*models_.back()).calibration_record() = std::move(calibrations[i]);
    }
  }
  std::vector<std::shared_ptr<OpenedModelArtifacts>>().swap(opened_artifacts);
  if (models_.empty()) {
    return Error("no configured model could be registered");
  }
  std::uint64_t activations = 0;
  std::uint64_t pool = 0;
  std::uint64_t largest = 0;
  std::uint64_t largest_weights = 0;  // the largest weights alone
  std::uint64_t total_weights = 0;
  bool has_image = false;
  std::uint64_t host_inputs = 0;
  std::uint64_t plans = 0;
  for (const auto& m : models_) {
    if (options_.debug_plans)
      m->set_plan_report_sink(
          [this, model = m.get()](kernels::PlanReportEvent event, std::string_view record) {
            // Records are compiler-produced JSON; stream the paid typed inventory.
            // Escape the user-controlled model label without copying the record
            // through Log/Printable/format or allocating a prefix string.
            if (event == kernels::PlanReportEvent::kBegin) {
              (void)std::fputs("llmp-runtime: debug plan model \"", log_);
              constexpr char hex[] = "0123456789abcdef";
              for (const char byte : model->name()) {
                const auto c = static_cast<unsigned char>(byte);
                if (c == '\\' || c == '\"') {
                  (void)std::fputc('\\', log_);
                  (void)std::fputc(c, log_);
                } else if (c < 0x20 || c == 0x7f) {
                  const char escaped[] = {'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
                  (void)std::fwrite(escaped, 1, sizeof(escaped), log_);
                } else {
                  (void)std::fputc(c, log_);
                }
              }
              (void)std::fputs("\": ", log_);
              return;
            }
            if (event == kernels::PlanReportEvent::kFragment) {
              while (!record.empty()) {
                const auto count = std::min<std::size_t>(record.size(), 4096);
                (void)std::fwrite(record.data(), 1, count, log_);
                record.remove_prefix(count);
              }
              return;
            }
            (void)std::fputc('\n', log_);
            (void)std::fflush(log_);
          });
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
    const std::uint64_t model_weights = m->weights().size() * kExtent;
    largest = std::max(largest, footprint);
    largest_weights = std::max<std::uint64_t>(largest_weights, model_weights);
    if (__builtin_add_overflow(total_weights, model_weights, &total_weights)) {
      total_weights = std::numeric_limits<std::uint64_t>::max();
    }
    has_image = has_image || !m->llm();
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
    startup_reader_lifetime_.reset();
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
  const bool shared_mode = config_.memory.mode.value == config::MemoryMode::kShared;
  const bool pageable = options_.pageable_workspace.value_or(shared_mode ? true : !fits);
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
  // The turn checkpoints' staging, set apart once so a budget full of
  // reclaimable plans, graphs and idle state never starves a capture or a
  // restore (they use it one at a time).
  if (std::ranges::any_of(models_, [](const auto& m) { return m->llm(); })) {
    if (auto r = node_.ReserveStaging(engine::CheckpointStagingBytes()); !r) {
      return Error("setting apart the turn checkpoints' staging: " + r.error());
    }
  }
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
  if (config_.memory.budget_gib.value > 0 && !options_.diagnostic_budget_cap_bytes.has_value()) {
    budget_ = std::min(budget_, std::uint64_t{config_.memory.budget_gib.value} << 30U);
  }
  if (budget_ < fixed_ + largest) {
    return std::unexpected(std::format(
        "configured memory budget ({} bytes) is less than the required footprint ({} bytes fixed "
        "+ {} bytes largest model)",
        budget_, fixed_, largest));
  }
  if (options_.diagnostic_budget_cap_bytes) {
    const auto requested = *options_.diagnostic_budget_cap_bytes;
    // An automatic startup ceiling cannot enlarge ordinary admission. Keep
    // extent alignment when current measured headroom selects a smaller budget.
    const auto cap =
        AutomaticStartup() ? std::min(requested, dynamic_budget_ / kExtent * kExtent) : requested;
    if (auto checked = CheckDiagnosticBudgetCap(bounds, dynamic_budget_, cap, 0); !checked)
      return std::unexpected(checked.error());
    budget_ = cap;
    Log(std::format("diagnostic budget cap: dynamic={} selected={} bytes; requested={} bytes",
                    dynamic_budget_, budget_, requested));
  }
  const auto add_sat = [](std::uint64_t a, std::uint64_t b) {
    std::uint64_t out = 0;
    return __builtin_add_overflow(a, b, &out) ? std::numeric_limits<std::uint64_t>::max() : out;
  };
  const std::uint64_t image_growth = has_image ? (std::uint64_t{8} << 30U) : 0;
  const std::uint64_t max_device_bytes = add_sat(add_sat(total_weights, pageable ? workspace_ : 0),
                                                 add_sat(node_.StateCapacity(), image_growth));
  const std::uint64_t avail_pool =
      available != 0 ? (available > GuardReserve(bounds)
                            ? (available - GuardReserve(bounds)) / kExtent * kExtent
                            : 0)
                     : max_device_bytes;
  const std::uint64_t pool_ceiling = std::min(max_device_bytes, avail_pool);
  if (auto r = node_.Start(base::Bytes(budget_), base::Bytes(pool_ceiling)); !r) {
    return r;
  }
  configured_budget_ = budget_;
  last_activity_ = Clock::now();
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
          !KeepWithinSpillBudget(held, {.spilling = branch}, l.LastUsed(*branch))) {
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
  if (config_.memory.mode.value == config::MemoryMode::kShared &&
      config_.memory.idle_release_seconds.value > 0 && !HasActiveTurnWork()) {
    const auto idle_limit = std::chrono::seconds(config_.memory.idle_release_seconds.value);
    if (now >= last_activity_ + idle_limit && (resident_ != nullptr || AnyResidentState())) {
      (void)ReleaseWorkingSet("idle timeout");
    }
  }
  // Pressure from outside the runtime: other processes share the node's
  // memory (D-004). One reclaim of what would restore the target headroom,
  // spaced by a back-off while the pressure persists (PressureTrim), never
  // the running model's in-use floor (Reclaim).
  const platform::MemoryPressure pressure = platform::ReadMemoryPressure();
  const PressureTrim::Trim trim = pressure_.Observe(pressure, now);
  if (trim.now) {
    std::uint64_t freed = Reclaim(trim.needed, true, trim.why, nullptr, std::nullopt, true);
    if (config_.memory.mode.value == config::MemoryMode::kShared && freed < trim.needed &&
        resident_ != nullptr && !HasActiveTurnWork()) {
      auto rel = ReleaseWorkingSet("memory pressure");
      if (rel) {
        freed += rel->reclaimed_bytes;
      }
    }
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
  if (config_.memory.mode.value == config::MemoryMode::kShared && configured_budget_ > 0) {
    const auto avail =
        trim.now ? platform::AvailableMemoryBytes().or_else(
                       [&]() -> std::optional<std::uint64_t> { return pressure.available; })
                 : pressure.available;
    const bool pressed = trim.now || pressure_.pressed();
    std::uint64_t current_occupancy = 0;
    if (pressed && avail.has_value() && *avail < pressure_.target_bytes()) {
      (void)node_.Call(
          [&]() -> Status {
            current_occupancy = node_.catalog().OccupancyOf(node_.domain()).Total().value();
            return {};
          },
          "reading occupancy for shared-mode dynamic budget");
    }
    const std::uint64_t new_budget =
        AdjustSharedBudget(avail, pressed, current_occupancy, fixed_, node_.budget().value(),
                           configured_budget_, pressure_.limits());
    if (new_budget != node_.budget().value()) {
      node_.SetBudget(base::Bytes(new_budget));
      budget_ = new_budget;
    }
  }
}

}  // namespace llmp::runtime
