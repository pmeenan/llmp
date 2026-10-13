// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Default-off complete source generation through the shared typed pipeline.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact/composition.h"
#include "base/json.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "engine/expert_slab.h"
#include "engine/qwen_image_runner.h"
#include "expected_error.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/cublas.h"
#include "kernels/image/pipeline.h"
#include "platform/host_probe.h"
#include "platform/kept_files.h"
#include "providers/device_runtime.h"
#include "runtime/component_image_build.h"
#include "runtime/memory_guard.h"
#include "tokenizer/hf.h"

namespace {
namespace en = llmp::engine;
namespace rt = llmp::runtime;
namespace fs = std::filesystem;
using en::Status;
using llmp::test_support::Failed;
constexpr std::uint64_t kMiB = 1ULL << 20U, kGiB = 1ULL << 30U;
constexpr std::uint64_t kCap = 48 * kGiB, kConstruction = 4 * kGiB;
// This fixed source/parser/caller phase remains locally charged while the
// factory additionally charges its derived mapping/graph/compiler inventory.
// Includes the authenticated <=12MiB tokenizer, <=2MiB source indices,
// composition directory, all retained artifact/catalog descriptors, prompt,
// source noise and the actual diagnostic reports. No reference weight copy.
constexpr std::uint64_t kStartup = kGiB;
constexpr std::uint64_t kSelectedWeights = 30530338816ULL;
constexpr std::string_view kComposition =
    "eca21baad38229e471a44cb2479d392ffcf745fb812e8a41668f336139fa1acd";
constexpr std::array<std::string_view, 3> kIds{
    "ed89ed270cf840965565d42eb61220b07399ff147e4f1de9ffcb7cbfd5ae86cf",
    "d1184efda09c12175b7bae6e9acf6cf3aeafb765f4f2ab6192cbf284c37eb3a4",
    "44c1a20a202ce1078588cbd4b0c594bbadf365a01f26091c715fd16d5b2eb34d"};
constexpr std::array<std::string_view, 3> kIndexHashes{
    "05cdbb6903545d79ffdc0bcc713721710267fd60e1be790a81028657edaf4e72",
    "13e91555cab81aaf627a0e32e6590733518c8c03b02abf65dbce59367a970a4f",
    "bb8c26b04b8d9bd4011a2cbb5c25e7630db8e2cd68b3bf09bb5350be0caca77a"};
constexpr std::array<std::uint64_t, 3> kIndexBytes{751744, 531405, 99639};
constexpr std::string_view kPrompt =
    "A red ceramic teapot on a plain wooden table, soft daylight, no text.";
Status Error(std::string message) { return std::unexpected(std::move(message)); }
std::expected<std::string, std::string> ReadExact(const fs::path& path, std::uint64_t bytes) {
  std::error_code ec;
  if (fs::file_size(path, ec) != bytes || ec || bytes > 12 * kMiB)
    return std::unexpected("source metadata/noise file extent differs");
  std::ifstream file(path, std::ios::binary);
  std::string out(static_cast<std::size_t>(bytes), '\0');
  if (!file.read(out.data(), static_cast<std::streamsize>(out.size())) || file.peek() != EOF)
    return std::unexpected("source metadata/noise read differs");
  return out;
}
struct Lifetime {
  en::PagedNode node{{.compute_streams = 1, .slots = 1, .slot_bytes = en::kSlabSlotBytes}};
  std::shared_ptr<en::ExecutionMetadataAllowance> allowance;
  std::shared_ptr<rt::ImageGenerationSource> source;
  en::QwenImageOptions legacy_options;  // before the runner borrowing it
  std::unique_ptr<en::QwenImageRunner> legacy;
  std::unique_ptr<rt::EngineImage> image;
  bool legacy_arm = false, decoder_times = false, tiled = false;
  std::shared_ptr<const llmp::planner::OwnedDeviceFacts> tiled_facts;
  std::uint64_t tuning_credit = 0;
  std::array<en::StepTimes, 2> decoder_job_times{};
  fs::path output;
  std::vector<std::byte> exported;
  double setup_seconds = 0;
  std::array<double, 2> first_seconds{}, finish_seconds{}, generation_seconds{};
  std::uint64_t legacy_preflight = 0, legacy_staging = 0, export_credit = 0;
  std::uint64_t legacy_weight_capacity = 0;
  std::string native_report;
  static double Seconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
  }
  std::vector<en::PagedModel*> entered;
  bool teardown_attempted = false;
  std::uint64_t footprint = 0, fixed = 0, activation = 0, pool = 0, staging = 0, plans = 0;
  std::uint64_t startup_parser = 0, selected_weight_bytes = 0;
  std::array<std::string, 2> first{}, final{};
  Status Load(const fs::path& store, const fs::path& noise) {
    if (!allowance->Charge(kStartup)) return Error("image startup owner is unpaid");
    source = std::make_shared<rt::ImageGenerationSource>();
    // Retain the source-construction payment with the source owner. The whole
    // enclosing node/provider lifetime is retained separately on failed drain.
    source->keep = allowance;
    auto composition = llmp::artifact::OpenComposition(store / kComposition);
    if (!composition) return Error(composition.error().ToString());
    source->composition =
        std::make_shared<const llmp::artifact::Composition>(std::move(*composition));
    for (std::size_t i = 0; i < kIds.size(); ++i) {
      auto index = ReadExact(store / kIds[i] / "index.json", kIndexBytes[i]);
      if (!index ||
          llmp::base::ToHex(llmp::base::Sha256{}.Update(*index).Finish()) != kIndexHashes[i])
        return Error("image source index identity differs");
      source->weights[i] = std::make_unique<en::PagedWeights>();
      llmp::artifact::OpenOptions options;
      options.expected_id = std::string(kIds[i]);
      if (auto opened = source->weights[i]->Open(store / kIds[i], options); !opened) return opened;
    }
    const auto json = source->composition->Metadata("tokenizer.json");
    if (!json || json->size() != 11422654) return Error("image tokenizer source extent differs");
    startup_parser = llmp::base::json::ParseWorkingBytes(*json);
    // The exact authenticated source parser plus its simultaneously retained
    // vocabulary/merge/trie copies must fit the prepaid startup phase.
    const auto companion = 32 * std::uint64_t{json->size()} + 65536ULL * 1024 + 128 * kMiB;
    if (startup_parser > kStartup || companion > kStartup - startup_parser)
      return Error("image tokenizer/parser exceeds the paid startup phase");
    auto spec = llmp::tokenizer::ReadHfTokenizer(*json);
    if (!spec) return Error(spec.error().ToString());
    auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(*spec));
    if (!tokenizer) return Error(tokenizer.error().ToString());
    auto prompt = llmp::chat::RenderQwenImagePrompt(kPrompt, *tokenizer);
    if (!prompt) return Error(prompt.error().ToString());
    if (auto encoded = tokenizer->EncodeMarked(prompt->rendered.text, prompt->rendered.specials, {},
                                               source->prompt);
        !encoded)
      return Error(encoded.error().ToString());
    if (source->prompt.size() != 39 || prompt->drop_tokens != 14)
      return Error("native source prompt geometry differs");
    source->drop = static_cast<std::uint32_t>(prompt->drop_tokens);
    auto initial = ReadExact(noise, 524288);
    if (!initial) return Error(initial.error());
    source->noise.resize(initial->size() / sizeof(std::uint16_t));
    std::memcpy(source->noise.data(), initial->data(), initial->size());
    return {};
  }
  // This preflight runs before QwenImageRunner::Setup maps any own storage.
  // Artifact/tokenizer copies and source-derived small layout vectors remain
  // in the separately charged startup construction phase.
  Status MeasureLegacy() {
    namespace md = llmp::model;
    namespace ki = llmp::kernels::image;
    const auto& profile = md::QwenImage21();
    const std::array tensors{md::TextEncoderTensors(profile.text),
                             md::DenoiserTensors(profile.denoiser),
                             md::VaeDecoderTensors(profile.vae)};
    const std::array architectures{profile.text_architecture, profile.denoiser_architecture,
                                   profile.vae_architecture};
    std::uint64_t weight_bytes = 0;
    std::vector<std::uint64_t> vae_sizes;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto& artifact = source->weights[i]->artifact();
      auto bound = md::BindQwenImageComponent(tensors[i], architectures[i], artifact);
      if (!bound) return Error(bound.error());
      std::vector<bool> used(artifact.groups().size());
      for (auto index : *bound) {
        const auto& resource = artifact.resources()[index];
        used[resource.group] = true;
        if (i == 2) vae_sizes.push_back(resource.bytes.value());
      }
      for (std::size_t g = 0; g < used.size(); ++g)
        if (used[g]) weight_bytes += std::uint64_t{artifact.groups()[g].chunks} * en::kPagedExtent;
    }
    legacy_weight_capacity = weight_bytes;
    std::uint64_t own = 0, text = 0, flow = 0, vae = 0;
    std::vector<std::uint64_t> converted;
    (void)ki::OwnLayout(0, profile.denoiser, 25, 4096, own);
    (void)ki::TextLayout(0, profile.text, 39, text);
    (void)ki::DitLayout(0, profile.denoiser, 4121, flow);
    (void)ki::VaeLayout(0, profile.vae, 64, vae_sizes, true, converted, vae);
    auto facts = llmp::providers::QueryDeviceFacts(0);
    if (!facts) return Error("legacy preflight device facts are unavailable");
    const auto blas =
        llmp::kernels::ggml::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture))
            .value();
    const auto freqs = md::QwenImageDenoiserRotary(profile.denoiser, 25, 64);
    const auto align = [](std::uint64_t bytes) { return (bytes + 255) / 256 * 256; };
    const auto text_in = align(39 * 4) + 2 * align(39 * 256);
    const auto step_in = align(freqs.size() * 4) + align(4096 * 64 * 2) +
                         align(2 * std::uint64_t{profile.denoiser.timestep_dim} * 2) + 256;
    const auto input = std::max<std::uint64_t>({text_in, step_in, 4096});
    const auto output_bytes =
        std::max<std::uint64_t>(std::uint64_t{profile.vae.out_channels} * 1024 * 1024 * 2,
                                4096 * std::uint64_t{profile.denoiser.out_channels} * 2);
    legacy_staging = input + output_bytes;
    // Existing legacy Step captures only index>=2; this exact two-step
    // workload never creates a denoising graph in either generation.
    constexpr std::uint64_t graph_floor = 0;
    const auto rounded = [](std::uint64_t bytes) {
      return (bytes + en::kPagedExtent - 1) / en::kPagedExtent * en::kPagedExtent;
    };
    std::uint64_t left = kCap;
    // Count the actual opening grant and mapped landing zone together.
    const auto opening = node.catalog().OccupancyOf(node.domain()).Total().value();
    for (const auto bytes :
         {opening, weight_bytes, rounded(own), rounded(std::max({text, flow, vae})), rounded(blas),
          legacy_staging, graph_floor, 3 * en::kPagedExtent}) {
      if (bytes > left) return Error("source-derived legacy generation exceeds 48GiB");
      left -= bytes;
    }
    legacy_preflight = kCap - left;
    const auto available = llmp::platform::AvailableMemoryBytes().value_or(0);
    if (legacy_preflight > available || rt::kUncountedMargin > available - legacy_preflight)
      return Error(
          "legacy source-derived footprint plus margin lacks physical "
          "headroom");
    return {};
  }
  Status Export(std::string_view expected) {
    if (output.empty() || legacy_arm) return {};
    constexpr std::size_t pixels = 1024 * 1024;
    if (exported.size() != pixels * 16) return Error("completed export extent differs");
    std::vector<std::byte> rgba(pixels * 4);
    for (std::size_t c = 0; c < 4; ++c)
      for (std::size_t i = 0; i < pixels; ++i) {
        float value = 0;
        std::memcpy(&value, exported.data() + (c * pixels + i) * 4, 4);
        if (!std::isfinite(value)) return Error("completed export is not finite");
        const float scaled = (std::clamp(value, -1.F, 1.F) * .5F + .5F) * 255.F;
        const auto lo = static_cast<unsigned>(std::floor(scaled));
        const auto rounded =
            lo + static_cast<unsigned>(scaled - static_cast<float>(lo) > .5F ||
                                       (scaled - static_cast<float>(lo) == .5F && lo % 2));
        rgba[i * 4 + c] = static_cast<std::byte>(rounded);
      }
    if (llmp::base::ToHex(llmp::base::Sha256{}.Update(rgba).Finish()) != expected)
      return Error("exported pixels differ from the completed output digest");
    auto opened = llmp::platform::OpenPrivateDirectory(-1, output.c_str());
    if (!opened) return Error("opening shared image export directory failed");
    const llmp::artifact::FileDescriptor directory(*opened);
    auto written = llmp::platform::ReplacePrivateFile(
        directory.get(), "image.rgba8",
        std::string_view(reinterpret_cast<const char*>(rgba.data()), rgba.size()));
    return written ? Status{} : Error("writing completed shared image failed");
  }
  Status Run(const fs::path& store, const fs::path& noise) {
    const auto setup_started = std::chrono::steady_clock::now();
    const auto before = llmp::platform::AvailableMemoryBytes().value_or(0);
    if (auto paid = rt::CheckStartupHostAllowance(kConstruction, kCap, before); !paid) return paid;
    if (auto opened = node.Open(); !opened) return opened;
    auto paid = en::ExecutionMetadataAllowance::Admit(
        node.catalog(), node.domain(), kConstruction,
        llmp::platform::AvailableMemoryBytes().value_or(0), rt::kUncountedMargin);
    if (!paid) return Error(paid.error());
    allowance = std::move(*paid);
    if (auto loaded = Load(store, noise); !loaded) return loaded;
    std::uint64_t weights = 0;
    if (legacy_arm) {
      // A second artifact/tokenizer construction is simultaneous with Load's
      // verified source owners. Its separate 1GiB credit stays inside the one
      // held 4GiB allowance, not another catalog reservation.
      if (!allowance->Charge(kStartup)) return Error("legacy startup metadata is unpaid");
      if (auto measured = MeasureLegacy(); !measured) return measured;
      legacy_options.store = store;
      legacy_options.composition = kComposition;
      legacy_options.noise = noise;
      legacy_options.prompt = kPrompt;
      legacy_options.steps = 2;
      legacy_options.out = output;
      legacy = std::make_unique<en::QwenImageRunner>(node, legacy_options, 0, 0);
      entered.push_back(legacy.get());  // partial Setup resources retire too
      if (auto setup = legacy->Setup(); !setup) return setup;
      selected_weight_bytes = legacy->weight_read_bytes();
      weights = legacy_weight_capacity;
      activation = legacy->activations_needed();
      pool = legacy->pool_needed();
      plans = 0;  // no legacy denoising graph at two steps
      staging = legacy_staging;
    } else {
      llmp::config::ModelEntry entry;
      entry.name = "shared-image-generation";
      entry.artifact = std::string(kComposition);
      rt::ModelSettings settings;
      settings.name = entry.name;
      settings.architecture = "qwen_image_pipeline";
      if (tiled) {
        // Closed source inventory:113 VaeStep entries and128 tensor names/shapes
        // produced together by VaeDecoderPlan. Double capacities and512B per
        // source descriptor include strings, dimensions and bookkeeping. Pay
        // this temporary phase before any vector construction; no payloads.
        constexpr std::uint64_t preparation = 2 * 113 * sizeof(llmp::model::VaeStep) +
                                              2 * 128 * 512 +
                                              2 * 113 * sizeof(llmp::execution::TunedShape) + 4096;
        if (!allowance->Charge(preparation)) return Error("tiled selection preparation unpaid");
        tuning_credit = preparation;
        const auto stages = llmp::model::VaeDecoderPlan(llmp::model::QwenImage21().vae, 64, 64);
        if (stages.size() != 113) return Error("actual decoder tuning inventory changed");
        llmp::planner::DeviceFacts facts{.device_class = "sm_121"};
        facts.tuned.reserve(stages.size());
        for (const auto& stage : stages) {
          if (stage.kind != llmp::model::VaeStep::Kind::kConv3x3) continue;
          llmp::execution::TunedShape key{.implementation = llmp::kernels::ggml::kConv2dTiledF32,
                                          .rows = std::uint64_t{stage.width} * stage.height,
                                          .k = 9ULL * stage.in_channels,
                                          .n = stage.out_channels,
                                          .output_encoding = llmp::format::Encoding::kF32};
          if (std::ranges::none_of(facts.tuned, [&](const auto& before) {
                return before.rows == key.rows && before.k == key.k && before.n == key.n;
              }))
            facts.tuned.push_back(key);
        }
        auto bytes = llmp::planner::OwnedDeviceFacts::Bound(facts);
        if (!bytes || !allowance->Charge(*bytes)) return Error("owned tiled selection unpaid");
        tuning_credit += *bytes;
        auto owned = llmp::planner::OwnedDeviceFacts::Create(facts, *bytes);
        if (!owned) return Error(owned.error());
        tiled_facts = std::make_shared<llmp::planner::OwnedDeviceFacts>(std::move(*owned));
      }
      auto build = rt::ComponentImageGenerationBuild(node, 0, source, allowance, tiled_facts);
      if (tiled) {
        build = [original = std::move(build), facts = tiled_facts](const rt::ModelSettings& current)
            -> std::expected<rt::EngineImageParts, std::string> {
          auto parts = original(current);
          if (!parts) return std::unexpected(parts.error());
          if (!parts->backend || !parts->backend->owner)
            return std::unexpected(std::string("tiled source lost its actual backend owner"));
          const auto& actual = parts->backend->owner->backend().device.tuned;
          const auto& expected = facts->facts().tuned;
          if (actual.size() != expected.size())
            return std::unexpected(std::string("owned tiled tuple count changed"));
          for (std::size_t i = 0; i < expected.size(); ++i)
            if (actual[i].implementation != expected[i].implementation ||
                actual[i].rows != expected[i].rows || actual[i].k != expected[i].k ||
                actual[i].n != expected[i].n ||
                actual[i].output_encoding != expected[i].output_encoding)
              return std::unexpected(std::string("owned tiled tuple identity changed"));
          // Explicit diagnostic candidate, not a serving rank claim for all shapes.
          parts->policy.ranks.push_back(
              {"sm_121", std::string(llmp::kernels::ggml::kConv2dTiledF32), 0});
          return parts;
        };
      }
      if (!output.empty()) {
        export_credit = 20 * kMiB;
        if (!allowance->Charge(export_credit)) return Error("quality export credit is unavailable");
        exported.resize(16 * kMiB);
        build = [this, original = std::move(build)](const rt::ModelSettings& current)
            -> std::expected<rt::EngineImageParts, std::string> {
          auto parts = original(current);
          if (!parts) return std::unexpected(parts.error());
          auto finish = std::move(parts->finish);
          parts->finish =
              [this, finish = std::move(finish)](
                  std::span<const std::byte> bytes) -> std::expected<std::string, std::string> {
            auto digest = finish(bytes);
            if (!digest) return digest;
            if (bytes.size() != exported.size())
              return std::unexpected(std::string("quality observation extent differs"));
            std::ranges::copy(bytes, exported.begin());
            return digest;
          };
          return parts;
        };
      }
      if (decoder_times) {
        build = [original = std::move(build)](const rt::ModelSettings& current)
            -> std::expected<rt::EngineImageParts, std::string> {
          auto parts = original(current);
          if (!parts) return std::unexpected(parts.error());
          if (parts->sequence.size() != 4 || parts->first_output != 1 ||
              parts->sequence[0].stage != 0 || parts->sequence[1].stage != 1 ||
              parts->sequence[2].stage != 1 || parts->sequence[3].stage != 2)
            return std::unexpected(
                std::string("decoder observation needs the complete four-stage sequence"));
          // Test-only boundary: same four programs and always-present copies.
          // FirstOutput now completes both flow iterations; Finish is decoder
          // only.
          parts->first_output = 2;
          return parts;
        };
      }
      image = std::make_unique<rt::EngineImage>(node, entry, settings, 0, std::move(build));
      const auto setup = image->Setup();
      if (image->has_paged()) entered.push_back(&image->paged());
      if (!setup) return setup;
      weights = image->weight_read_bytes();
      selected_weight_bytes = weights;
      if (weights != kSelectedWeights) return Error("complete source closure page union differs");
      activation = image->activations_needed();
      pool = std::max<std::uint64_t>(image->pool_needed(), en::kPagedExtent);
      staging = image->host_input_bytes();
      plans = image->plan_floor_bytes();
    }
    fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    std::uint64_t left = kCap;
    const auto take = [&left](std::uint64_t bytes) {
      if (bytes > left) return false;
      left -= bytes;
      return true;
    };
    // All selected weights can remain cached across stages, while actual
    // materialization still leases only the current stage. Fixed includes
    // construction allowance, stable ports and host staging already owned.
    if (!take(fixed) || !take(weights) || !take(activation) || !take(pool) || !take(plans) ||
        !take(2 * en::kPagedExtent))
      return Error("measured complete generation exceeds the 48GiB paid node envelope");
    footprint = kCap - left;
    if (legacy_arm && footprint > legacy_preflight)
      return Error("actual legacy allocation exceeds the source-derived preflight");
    const auto available = llmp::platform::AvailableMemoryBytes().value_or(0);
    if (footprint > available || rt::kUncountedMargin > available - footprint)
      return Error("actual generation footprint plus physical margin lacks headroom");
    if (auto mapped = node.MapWorkspace(activation, pool); !mapped) return mapped;
    const auto current_fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto now = llmp::platform::AvailableMemoryBytes().value_or(0);
    rt::MemoryGuard guard{.largest = weights,
                          .host_inputs = staging,
                          .plans = plans,
                          .available = now,
                          .fixed = current_fixed,
                          .requests = 0};
    if (auto valid = rt::CheckMemoryGuard(guard); !valid) return valid;
    const auto reserve = rt::GuardReserve(guard);
    if (now < reserve || current_fixed > kCap) return Error("image dynamic budget is absent");
    const auto dynamic = current_fixed + (now - reserve) / en::kPagedExtent * en::kPagedExtent;
    if (auto valid = rt::CheckDiagnosticBudgetCap(guard, dynamic, kCap, 0); !valid) return valid;
    if (auto started = node.Start(llmp::base::Bytes(kCap)); !started) return started;
    if (auto registered = legacy_arm ? legacy->Register() : image->Register(); !registered)
      return registered;
    if (auto bound = legacy_arm ? legacy->Bind() : image->Bind(); !bound) return bound;
    node.Run();
    node.SetHostFloor(0);
    setup_seconds = Seconds(std::chrono::steady_clock::now() - setup_started);
    // Repeated generation uses the same retained typed plans/captures and
    // restores the original latent port only after a completed prior Finish.
    for (std::size_t run = 0; run < first.size(); ++run) {
      const auto started = std::chrono::steady_clock::now();
      auto first_done = started;
      auto complete = node.WithRequest(
          0, legacy_arm ? legacy->everything() : image->request_closure(), "image migration screen",
          [&]() -> Status {
            if (decoder_times) (void)node.TakeTimes(0);
            auto step =
                legacy_arm ? legacy->FirstOutput(first[run]) : image->FirstOutput(first[run]);
            first_done = std::chrono::steady_clock::now();
            if (!step) return step;
            if (decoder_times) (void)node.TakeTimes(0);
            auto output_done = legacy_arm ? legacy->Finish(final[run]) : image->Finish(final[run]);
            if (decoder_times) decoder_job_times[run] = node.TakeTimes(0);
            return output_done;
          });
      const auto finished = std::chrono::steady_clock::now();
      if (!complete) return complete;
      first_seconds[run] = Seconds(first_done - started);
      finish_seconds[run] = Seconds(finished - first_done);
      generation_seconds[run] = Seconds(finished - started);
      if (auto saved = Export(final[run]); !saved) return saved;
      if (first[run].size() != 64 || final[run].size() != 64)
        return Error(
            "complete generation did not publish both completed "
            "output boundaries");
    }
    if (legacy_arm)
      native_report = legacy->Report();
    else
      native_report = image->extra();
    if (first[0] != first[1] || final[0] != final[1])
      return Error("same selected complete generation differs on retained replay");
    return {};
  }
  Status Retire() {
    if (teardown_attempted) return Error("whole generation retirement was already attempted");
    teardown_attempted = true;
    return node.TearDown(entered);
  }
};

TEST(ImageGenerationGpu, CompleteTextFlowAndDecoderRetainAcrossTwoGenerations) {
  if (std::getenv("LLMP_RUN_SHARED_IMAGE_GENERATION") == nullptr)
    GTEST_SKIP() << "explicit bounded source generation diagnostic required";
  const char* store = std::getenv("LLMP_IMAGE_GENERATION_STORE");
  const char* noise = std::getenv("LLMP_IMAGE_GENERATION_NOISE");
  ASSERT_NE(store, nullptr);
  ASSERT_NE(noise, nullptr);
  auto life = std::make_unique<Lifetime>();
  const auto result = life->Run(store, noise);
  const auto retired = life->Retire();
  if (!retired) {
    // Keep the complete enclosing provider/node/factory/model owner on the
    // first failed drain. Never retry TearDown's already-torn-down fast path.
    std::ignore = life.release();
    FAIL() << Failed(retired).value_or("");
  }
  ASSERT_TRUE(result) << Failed(result).value_or("");
  if (life->tiled) {
    RecordProperty("tiled_tuning_keys", std::to_string(life->tiled_facts->facts().tuned.size()));
    RecordProperty("tiled_tuning_credit_bytes", std::to_string(life->tuning_credit));
  }
  RecordProperty("completed_generations", 2);
  RecordProperty("completed_stages", 8);
  RecordProperty("complete_pixel_values_per_generation", 4194304);
  RecordProperty("source_selected_weight_bytes", std::to_string(kSelectedWeights));
  RecordProperty("setup_paid_footprint_bytes", std::to_string(life->footprint));
  RecordProperty("fixed_bytes", std::to_string(life->fixed));
  RecordProperty("activation_bytes", std::to_string(life->activation));
  RecordProperty("pool_bytes", std::to_string(life->pool));
  RecordProperty("staging_bytes", std::to_string(life->staging));
  RecordProperty("retained_plan_floor_bytes", std::to_string(life->plans));
  RecordProperty("tokenizer_parser_bound_bytes", std::to_string(life->startup_parser));
  RecordProperty("first_flow_sha256", life->first[0]);
  RecordProperty("pixels_sha256", life->final[0]);
}
// One driver, two existing implementations; no serving selection or kernel
// changes. Output mode is a quality observation, never a timing sample.
TEST(ImageGenerationGpu, SameSourceMigrationScreen) {
  const char* arm = std::getenv("LLMP_IMAGE_SCREEN_ARM");
  if (arm == nullptr) GTEST_SKIP() << "explicit same-source migration screen required";
  ASSERT_TRUE(std::string_view(arm) == "legacy" || std::string_view(arm) == "shared");
  const char* store = std::getenv("LLMP_IMAGE_GENERATION_STORE");
  const char* noise = std::getenv("LLMP_IMAGE_GENERATION_NOISE");
  ASSERT_NE(store, nullptr);
  ASSERT_NE(noise, nullptr);
  auto life = std::make_unique<Lifetime>();
  life->legacy_arm = std::string_view(arm) == "legacy";
  if (const char* output = std::getenv("LLMP_IMAGE_SCREEN_OUTPUT")) {
    std::error_code error;
    ASSERT_TRUE(fs::is_directory(output, error));
    ASSERT_FALSE(error);
    life->output = output;
  }
  life->tiled = std::getenv("LLMP_IMAGE_SCREEN_TILED_CONV") != nullptr;
  if (life->tiled) ASSERT_FALSE(life->legacy_arm);
  life->decoder_times = std::getenv("LLMP_IMAGE_SCREEN_DECODER_TIMES") != nullptr;
  if (life->decoder_times) {
    ASSERT_FALSE(life->legacy_arm);
    ASSERT_TRUE(life->output.empty());
  }
  const auto result = life->Run(store, noise);
  const auto retired = life->Retire();
  if (!retired) {
    std::ignore = life.release();
    FAIL() << Failed(retired).value_or("");
  }
  ASSERT_TRUE(result) << Failed(result).value_or("");
  RecordProperty("arm", arm);
  RecordProperty("mode", life->decoder_times ? "decoder_diagnostic"
                                             : (life->output.empty() ? "timing" : "quality"));
  RecordProperty("first_output_invocation", life->decoder_times ? 2 : 1);
  if (life->decoder_times) {
    ASSERT_EQ(life->final[0], "6dd66c48649d75057a996f3dd7bed9a0151ffaa22beac81f942641b16b1fc4e1");
    for (std::size_t i = 0; i < 2; ++i) {
      const auto& times = life->decoder_job_times[i];
      ASSERT_EQ(times.steps, 1U);
      ASSERT_GT(times.device, 0.0);
      RecordProperty(std::format("decoder_jobs_{}", i), std::to_string(times.steps));
      RecordProperty(std::format("decoder_job_wall_{}", i), std::format("{:.9f}", times.wall));
      RecordProperty(std::format("decoder_device_{}", i), std::format("{:.9f}", times.device));
      RecordProperty(std::format("decoder_dispatch_{}", i), std::format("{:.9f}", times.dispatch));
      RecordProperty(std::format("decoder_host_job_{}", i), std::format("{:.9f}", times.job));
      RecordProperty(std::format("decoder_after_{}", i), std::format("{:.9f}", times.after));
    }
  }
  if (life->tiled) {
    RecordProperty("tiled_tuning_keys", std::to_string(life->tiled_facts->facts().tuned.size()));
    RecordProperty("tiled_tuning_credit_bytes", std::to_string(life->tuning_credit));
  }
  RecordProperty("completed_generations", 2);
  RecordProperty("setup_seconds", std::format("{:.9f}", life->setup_seconds));
  for (std::size_t i = 0; i < 2; ++i) {
    RecordProperty(std::format("first_seconds_{}", i),
                   std::format("{:.9f}", life->first_seconds[i]));
    RecordProperty(std::format("finish_seconds_{}", i),
                   std::format("{:.9f}", life->finish_seconds[i]));
    RecordProperty(std::format("generation_seconds_{}", i),
                   std::format("{:.9f}", life->generation_seconds[i]));
  }
  RecordProperty("setup_paid_footprint_bytes", std::to_string(life->footprint));
  RecordProperty("legacy_preflight_bytes", std::to_string(life->legacy_preflight));
  RecordProperty("legacy_weight_capacity_bytes", std::to_string(life->legacy_weight_capacity));
  RecordProperty("selected_weight_bytes", std::to_string(life->selected_weight_bytes));
  RecordProperty("legacy_denoising_captures", 0);
  RecordProperty("quality_counters_are_timing_samples", "false");
  RecordProperty("export_credit_bytes", std::to_string(life->export_credit));
  RecordProperty("construction_used_bytes", std::to_string(life->allowance->used()));
  RecordProperty("first_flow_sha256", life->first[0]);
  RecordProperty("pixels_sha256", life->final[0]);
  RecordProperty("native_report", life->native_report);
}
}  // namespace
