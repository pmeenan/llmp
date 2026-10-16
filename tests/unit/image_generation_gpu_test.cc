// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Default-off complete source generation through the shared typed pipeline.
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

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
#include "catalog/catalog.h"
#include "chat/chat.h"
#include "engine/expert_slab.h"
#include "engine/paged_node.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/cublas.h"
#include "model/qwen_image.h"
#include "platform/host_probe.h"
#include "platform/kept_files.h"
#include "providers/device_runtime.h"
#include "runtime/component_image_build.h"
#include "runtime/engine_llm.h"
#include "runtime/memory_guard.h"
#include "tokenizer/hf.h"
#include "v1_frozen.h"

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
constexpr std::string_view kRectangularPrompt =
    "A blue glass vase beside a yellow book, viewed from above.";
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
  std::unique_ptr<en::PagedNode> owned_node;
  en::PagedNode& node;
  Lifetime()
      : owned_node(std::make_unique<en::PagedNode>(
            en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kSlabSlotBytes})),
        node(*owned_node) {}
  explicit Lifetime(en::PagedNode& server_node) : node(server_node) {}
  std::shared_ptr<en::ExecutionMetadataAllowance> allowance;
  std::shared_ptr<rt::ImageGenerationSource> source;
  std::unique_ptr<rt::EngineImage> image;
  bool decoder_times = false, tiled = false, masked_attention = false;
  bool rectangular_request = false, replace_requests = false;
  rt::ImageGenerationRequest seed;
  std::vector<std::int32_t> alternate_prompt;
  std::weak_ptr<void> request_owner;
  llmp::base::Sha256Digest request_layout{}, initial_request_identity{};
  std::string changed_pixels;
  std::uint64_t replacement_credit = 0, construction_peak = 0;
  std::uint64_t eager_occupancy = 0, warmed_occupancy = 0;
  std::uint32_t completed_requests = 0, completed_request_stages = 0, released_requests = 0;
  std::uint32_t prompt_rows = 0, dropped_rows = 0;
  std::uint64_t request_sequence = 0, request_pixel_values = 0;
  std::string request_identity;
  std::shared_ptr<const llmp::planner::OwnedDeviceFacts> tuning_facts;
  std::uint64_t tuning_credit = 0, attention_source_nodes = 0;
  std::array<en::StepTimes, 2> decoder_job_times{};
  fs::path output;
  std::vector<std::byte> exported;
  double setup_seconds = 0;
  std::array<double, 2> first_seconds{}, finish_seconds{}, generation_seconds{};
  std::uint64_t export_credit = 0;
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
    auto prompt = llmp::chat::RenderQwenImagePrompt(
        rectangular_request ? kRectangularPrompt : kPrompt, *tokenizer);
    if (!prompt) return Error(prompt.error().ToString());
    if (auto encoded = tokenizer->EncodeMarked(prompt->rendered.text, prompt->rendered.specials, {},
                                               source->prompt);
        !encoded)
      return Error(encoded.error().ToString());
    if (!rectangular_request && (source->prompt.size() != 39 || prompt->drop_tokens != 14))
      return Error("native source prompt geometry differs");
    source->drop = static_cast<std::uint32_t>(prompt->drop_tokens);
    prompt_rows = static_cast<std::uint32_t>(source->prompt.size());
    dropped_rows = source->drop;
    if (rectangular_request) {
      if (prompt_rows == 39 || prompt_rows <= dropped_rows)
        return Error("rectangular request must exercise a different native prompt length");
      source->width = 2048;
      source->height = 512;
    }
    if (replace_requests) {
      // Extra retained seed/alternate prompt and one simultaneously prepared
      // source copy, with doubled vector capacities; original Load is startup-paid.
      replacement_credit = 4 * 524288ULL + 8 * source->prompt.size() * sizeof(std::int32_t) + 4096;
      if (!allowance->Charge(replacement_credit)) return Error("request seed copies are unpaid");
      auto other = llmp::chat::RenderQwenImagePrompt(
          "A blue ceramic teapot on a plain wooden table, soft daylight, no text.", *tokenizer);
      if (!other) return Error(other.error().ToString());
      if (auto encoded = tokenizer->EncodeMarked(other->rendered.text, other->rendered.specials, {},
                                                 alternate_prompt);
          !encoded)
        return Error(encoded.error().ToString());
      if (alternate_prompt.size() != source->prompt.size() || other->drop_tokens != source->drop ||
          alternate_prompt == source->prompt)
        return Error("native replacement prompt must change content within the retained layout");
    }
    auto initial = ReadExact(noise, 524288);
    if (!initial) return Error(initial.error());
    source->noise.resize(initial->size() / sizeof(std::uint16_t));
    std::memcpy(source->noise.data(), initial->data(), initial->size());
    if (replace_requests) {
      seed.prompt = source->prompt;
      seed.noise = source->noise;
      seed.drop = source->drop;
      seed.keep = allowance;
    }
    return {};
  }
  Status Export(std::string_view expected) {
    if (output.empty()) return {};
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
    llmp::config::ModelEntry entry;
    entry.name = "shared-image-generation";
    entry.artifact = std::string(kComposition);
    rt::ModelSettings settings;
    settings.image_size.value = 1024;
    settings.image_steps.value = rectangular_request || replace_requests ? 1 : 2;
    settings.name = entry.name;
    settings.architecture = "qwen_image_pipeline";
    if (tiled || masked_attention) {
      // Closed source inventory:113 VaeStep entries and128 tensor names/shapes
      // produced together by VaeDecoderPlan. Double capacities and512B per
      // source descriptor include strings, dimensions and bookkeeping. Pay
      // this temporary phase before any vector construction; no payloads.
      constexpr std::uint64_t tiled_preparation =
          2 * 113 * sizeof(llmp::model::VaeStep) + 2 * 128 * 512 +
          2 * 113 * sizeof(llmp::execution::TunedShape) + 4096;
      // One temporary tuple, class name, shared owner and callable controls;
      // the immutable copy and backend declaration are charged separately.
      constexpr std::uint64_t attention_preparation =
          2 * sizeof(llmp::execution::TunedShape) + 4096;
      const auto preparation = tiled ? tiled_preparation : attention_preparation;
      if (!allowance->Charge(preparation)) return Error("candidate selection preparation unpaid");
      tuning_credit = preparation;
      llmp::planner::DeviceFacts facts{.device_class = "sm_121"};
      if (tiled) {
        const auto stages = llmp::model::VaeDecoderPlan(llmp::model::QwenImage21().vae, 64, 64);
        if (stages.size() != 113) return Error("actual decoder tuning inventory changed");
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
      } else {
        // Explicit approximate F16 transient-operand candidate. This does
        // not qualify arbitrary F32 ranges or change the dense default.
        facts.tuned.push_back({.implementation = llmp::kernels::ggml::kAttentionF32,
                               .rows = 4121,
                               .k = 128,
                               .n = 128,
                               .output_encoding = llmp::format::Encoding::kF32});
      }
      auto bytes = llmp::planner::OwnedDeviceFacts::Bound(facts);
      if (!bytes || !allowance->Charge(*bytes)) return Error("owned candidate selection unpaid");
      tuning_credit += *bytes;
      auto owned = llmp::planner::OwnedDeviceFacts::Create(facts, *bytes);
      if (!owned) return Error(owned.error());
      tuning_facts = std::make_shared<llmp::planner::OwnedDeviceFacts>(std::move(*owned));
    }
    auto build = rt::ComponentImageGenerationBuild(node, 0, source, allowance, tuning_facts);
    if (tiled || masked_attention) {
      build = [this, original = std::move(build),
               facts = tuning_facts](const rt::ModelSettings& current)
          -> std::expected<rt::EngineImageParts, std::string> {
        auto parts = original(current);
        if (!parts) return std::unexpected(parts.error());
        if (!parts->backend || !parts->backend->owner)
          return std::unexpected(std::string("candidate source lost its actual backend owner"));
        const auto& actual = parts->backend->owner->backend().device.tuned;
        const auto& expected = facts->facts().tuned;
        if (actual.size() != expected.size())
          return std::unexpected(std::string("owned candidate tuple count changed"));
        for (std::size_t i = 0; i < expected.size(); ++i)
          if (actual[i].implementation != expected[i].implementation ||
              actual[i].rows != expected[i].rows || actual[i].k != expected[i].k ||
              actual[i].n != expected[i].n ||
              actual[i].output_encoding != expected[i].output_encoding)
            return std::unexpected(std::string("owned candidate tuple identity changed"));
        // Explicit diagnostic candidate, not a serving rank claim for all shapes.
        if (tiled)
          parts->policy.ranks.push_back(
              {"sm_121", std::string(llmp::kernels::ggml::kConv2dTiledF32), 0});
        if (masked_attention) {
          // Source witness, not a runtime counter. The existing physical
          // host control checks selection of these32 identical shapes.
          attention_source_nodes = 0;
          for (const auto& stage : parts->stages) {
            const auto& graph = stage.graph->graph();
            for (const auto& operation : graph.nodes()) {
              if (operation.key != llmp::ir::Primitive(llmp::ir::Op::kAttention)) continue;
              const auto& query = graph.tensor(operation.source(0)).format;
              if (query.encoding == llmp::format::Encoding::kF32 &&
                  query.shape.ne == std::array<std::uint64_t, 4>{128, 4121, 32, 1})
                ++attention_source_nodes;
            }
          }
          if (attention_source_nodes != 32)
            return std::unexpected(std::string("actual flow attention source count differs"));
        }
        return parts;
      };
    }
    if (rectangular_request) {
      build = [this, original = std::move(build)](const rt::ModelSettings& current)
          -> std::expected<rt::EngineImageParts, std::string> {
        auto parts = original(current);
        if (!parts) return std::unexpected(parts.error());
        if (parts->request.sequence.size() != 3 || parts->stages.size() != 3 ||
            parts->request.first_output != 1 || parts->request.sequence[0].stage != 0 ||
            parts->request.sequence[1].stage != 1 || parts->request.sequence[1].iteration != 0 ||
            parts->request.sequence[2].stage != 2 ||
            parts->stages[0].graph->outputs()[0].shape !=
                llmp::components::SpatialShape{1, prompt_rows - dropped_rows, 4096, 1} ||
            parts->stages[2].graph->outputs()[0].shape !=
                llmp::components::SpatialShape{2048, 512, 4, 1})
          return std::unexpected(std::string("actual rectangular request composition differs"));
        request_sequence = parts->request.sequence.size();
        request_pixel_values = 2048ULL * 512 * 4;
        request_identity = llmp::base::ToHex(parts->request.identity);
        return parts;
      };
    }
    if (replace_requests) {
      build = [this, original = std::move(build)](const rt::ModelSettings& current)
          -> std::expected<rt::EngineImageParts, std::string> {
        auto parts = original(current);
        if (!parts) return std::unexpected(parts.error());
        if (parts->request.sequence.size() != 3 || parts->stages.size() != 3)
          return std::unexpected(std::string("initial replacement request stage inventory"));
        request_layout = parts->request.layout;
        initial_request_identity = parts->request.identity;
        request_owner = parts->request.keep;
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
        auto finish = std::move(parts->request.finish);
        parts->request.finish =
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
        if (parts->request.sequence.size() != 4 || parts->request.first_output != 1 ||
            parts->request.sequence[0].stage != 0 || parts->request.sequence[1].stage != 1 ||
            parts->request.sequence[2].stage != 1 || parts->request.sequence[3].stage != 2)
          return std::unexpected(
              std::string("decoder observation needs the complete four-stage sequence"));
        // Test-only boundary: same four programs and always-present copies.
        // FirstOutput now completes both flow iterations; Finish is decoder
        // only.
        parts->request.first_output = 2;
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
    if (auto registered = image->Register(); !registered) return registered;
    if (auto bound = image->Bind(); !bound) return bound;
    node.Run();
    node.SetHostFloor(0);
    setup_seconds = Seconds(std::chrono::steady_clock::now() - setup_started);
    if (replace_requests) return RunRequests();
    // Repeated generation uses the same retained typed plans/captures and
    // restores the original latent port only after a completed prior Finish.
    for (std::size_t run = 0; run < first.size(); ++run) {
      const auto started = std::chrono::steady_clock::now();
      auto first_done = started;
      auto complete =
          node.WithRequest(0, image->request_closure(), "image migration screen", [&]() -> Status {
            if (decoder_times) (void)node.TakeTimes(0);
            auto step = image->FirstOutput(first[run]);
            first_done = std::chrono::steady_clock::now();
            if (!step) return step;
            if (decoder_times) (void)node.TakeTimes(0);
            auto output_done = image->Finish(final[run]);
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
    native_report = image->extra();
    if (first[0] != first[1] || final[0] != final[1])
      return Error("same selected complete generation differs on retained replay");
    return {};
  }
  Status RunRequests() {
    const auto original_construction = allowance->used();
    construction_peak = original_construction;
    const auto original_staging = image->host_input_bytes();
    const auto original_plans = image->plan_floor_bytes();
    // Three serial requests keep only one output string plus the first pin;
    // this diagnostic count is not an EngineImage count limit.
    for (std::uint32_t run = 0; run < 3; ++run) {
      const auto steps = run == 1 ? 2U : 1U;
      if (run) {
        rt::ImageGenerationRequest input{.prompt = run == 1 ? alternate_prompt : seed.prompt,
                                         .drop = seed.drop,
                                         .noise = seed.noise,
                                         .keep = allowance};
        if (run == 1)
          for (auto& value : input.noise) value ^= 0x8000U;  // exact finite BF16 sign change
        rt::ModelSettings settings;
        settings.image_size.value = 1024;
        settings.image_steps.value = steps;
        auto request = rt::ComponentImageGenerationRequest(std::move(input), settings, allowance);
        if (!request) return Error(request.error());
        construction_peak = std::max(construction_peak, allowance->used());
        if (request->layout != request_layout || request->sequence.size() != steps + 2 ||
            (request->identity == initial_request_identity) != (run == 2))
          return Error("replacement content identity or fixed typed layout differs");
        const auto old = request_owner;
        auto next_owner = std::weak_ptr<void>(request->keep);
        const auto occupancy = node.catalog().OccupancyOf(node.domain()).Total();
        if (auto replaced = image->ReplaceRequest(std::move(*request)); !replaced) return replaced;
        if (node.catalog().OccupancyOf(node.domain()).Total() != occupancy)
          return Error("request replacement mutated catalog occupancy");
        if (!old.expired()) return Error("completed old request owner remains retained");
        ++released_requests;
        request_owner = next_owner;
      }
      std::string first_output, pixels;
      auto complete = node.WithRequest(
          0, image->request_closure(), "image replacement screen", [&]() -> Status {
            if (auto started = image->FirstOutput(first_output); !started) return started;
            return image->Finish(pixels);
          });
      if (!complete) return complete;
      if (first_output.size() != 64 || pixels.size() != 64)
        return Error("replacement did not complete both finite output boundaries");
      if (run == 0) {
        first[0] = first_output;
        final[0] = pixels;
        eager_occupancy = node.catalog().OccupancyOf(node.domain()).Total().value();
      } else if (run == 1) {
        if (first_output == first[0] || pixels == final[0])
          return Error("changed request did not change both completed outputs");
        changed_pixels = pixels;
        // Second use captures stages under the existing paid PlanAccount.
        // Capture growth is legitimate; the next request reuses these graphs.
        warmed_occupancy = node.catalog().OccupancyOf(node.domain()).Total().value();
      } else if (first_output != first[0] || pixels != final[0]) {
        return Error("restored request did not restore its own completed outputs");
      }
      if (image->host_input_bytes() != original_staging ||
          image->plan_floor_bytes() != original_plans ||
          (run == 2 &&
           node.catalog().OccupancyOf(node.domain()).Total().value() != warmed_occupancy) ||
          image->activations_needed() != activation || image->pool_needed() > pool)
        return Error("same-layout replacement grew retained execution resources");
      ++completed_requests;
      completed_request_stages += steps + 2;
    }
    if (allowance->used() != original_construction)
      return Error("restored request retained extra construction payment");
    return {};
  }
  Status Retire() {
    if (teardown_attempted) return Error("whole generation retirement was already attempted");
    teardown_attempted = true;
    return node.TearDown(entered);
  }
};

TEST(ImageRequestInputs, PaymentAndLayoutRemainIndependentOfRequestContent) {
  llmp::catalog::Catalog catalog;
  const auto domain = catalog.AddDomain("request metadata");
  auto paid = en::ExecutionMetadataAllowance::Admit(catalog, domain, kMiB, 8 * kMiB, kMiB);
  ASSERT_TRUE(paid) << Failed(paid).value_or("");
  ASSERT_TRUE((*paid)->Charge(4096));  // caller input/control vectors, before construction
  const auto input = [&] {
    return rt::ImageGenerationRequest{.prompt = {1, 2, 3},
                                      .drop = 1,
                                      .width = 16,
                                      .height = 16,
                                      .noise = std::vector<std::uint16_t>(64, 0),
                                      .keep = *paid};
  };
  rt::ModelSettings settings;
  settings.image_steps.value = 1;
  llmp::base::Sha256Digest layout, identity;
  {
    auto first = rt::ComponentImageGenerationRequest(input(), settings, *paid);
    ASSERT_TRUE(first) << Failed(first).value_or("");
    layout = first->layout;
    identity = first->identity;
    EXPECT_EQ(first->sequence.size(), 3U);
    std::array<std::byte, 4> staged{};
    ASSERT_TRUE(first->fill({1, 0}, 5, staged));
    float delta = 0;
    std::memcpy(&delta, staged.data(), sizeof(delta));
    EXPECT_EQ(delta, -1.F);
    auto changed = input();
    changed.prompt[0] = 4;
    changed.noise[0] = llmp::model::ToBf16(1.F);
    settings.image_steps.value = 3;
    auto next = rt::ComponentImageGenerationRequest(std::move(changed), settings, *paid);
    ASSERT_TRUE(next) << Failed(next).value_or("");
    EXPECT_EQ(next->layout, layout);
    EXPECT_NE(next->identity, identity);
    EXPECT_EQ(next->sequence.size(), 5U);
    EXPECT_NE(next->ports[1].initial, first->ports[1].initial);
    auto rectangle = input();
    rectangle.height = 32;
    rectangle.noise.resize(128);
    auto different = rt::ComponentImageGenerationRequest(std::move(rectangle), settings, *paid);
    ASSERT_TRUE(different) << Failed(different).value_or("");
    EXPECT_NE(different->layout, layout);
  }
  EXPECT_EQ((*paid)->used(), 4096U);
  auto too_small = en::ExecutionMetadataAllowance::Admit(
      catalog, domain, en::ExecutionMetadataAllowance::OverheadBytes() + 64, 8 * kMiB, kMiB);
  ASSERT_TRUE(too_small);
  auto unpaid = rt::ComponentImageGenerationRequest(input(), settings, *too_small);
  EXPECT_FALSE(unpaid);
  EXPECT_EQ((*too_small)->used(), 0U);
  auto invalid = input();
  invalid.noise[0] = 0x7F80;  // source BF16 infinity
  auto refused = rt::ComponentImageGenerationRequest(std::move(invalid), settings, *paid);
  EXPECT_FALSE(refused);
  EXPECT_EQ((*paid)->used(), 4096U);
}

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
    RecordProperty("tiled_tuning_keys", std::to_string(life->tuning_facts->facts().tuned.size()));
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
// One differently sized request; repeated execution reuses that same request,
// not request replacement. Source noise has the same area as the square screen.
TEST(ImageGenerationGpu, RectangularRequestUsesPromptAndOneStep) {
  if (std::getenv("LLMP_RUN_SHARED_IMAGE_REQUEST") == nullptr)
    GTEST_SKIP() << "explicit bounded rectangular request diagnostic required";
  const char* store = std::getenv("LLMP_IMAGE_GENERATION_STORE");
  const char* noise = std::getenv("LLMP_IMAGE_GENERATION_NOISE");
  ASSERT_NE(store, nullptr);
  ASSERT_NE(noise, nullptr);
  auto life = std::make_unique<Lifetime>();
  life->rectangular_request = true;
  const auto result = life->Run(store, noise);
  const auto retired = life->Retire();
  if (!retired) {
    std::ignore = life.release();
    FAIL() << Failed(retired).value_or("");
  }
  ASSERT_TRUE(result) << Failed(result).value_or("");
  EXPECT_NE(life->prompt_rows, 39U);
  ASSERT_GT(life->prompt_rows, life->dropped_rows);
  EXPECT_EQ(life->request_sequence, 3U);
  EXPECT_EQ(life->request_pixel_values, 4194304U);
  EXPECT_EQ(life->request_identity.size(), 64U);
  RecordProperty("request_width", 2048);
  RecordProperty("request_height", 512);
  RecordProperty("request_steps", 1);
  RecordProperty("prompt_rows", std::to_string(life->prompt_rows));
  RecordProperty("dropped_rows", std::to_string(life->dropped_rows));
  RecordProperty("request_identity", life->request_identity);
  RecordProperty("completed_generations", 2);
  RecordProperty("completed_stages", 6);
  RecordProperty("complete_pixel_values_per_generation", 4194304);
  RecordProperty("source_selected_weight_bytes", std::to_string(kSelectedWeights));
  RecordProperty("setup_paid_footprint_bytes", std::to_string(life->footprint));
  RecordProperty("fixed_bytes", std::to_string(life->fixed));
  RecordProperty("activation_bytes", std::to_string(life->activation));
  RecordProperty("pool_bytes", std::to_string(life->pool));
  RecordProperty("staging_bytes", std::to_string(life->staging));
  RecordProperty("retained_plan_floor_bytes", std::to_string(life->plans));
  RecordProperty("construction_used_bytes", std::to_string(life->allowance->used()));
  RecordProperty("tokenizer_parser_bound_bytes", std::to_string(life->startup_parser));
  RecordProperty("first_flow_sha256", life->first[0]);
  RecordProperty("pixels_sha256", life->final[0]);
}
TEST(ImageGenerationGpu, CompletedRequestsReplaceContentAndReuseRetainedCaptures) {
  if (std::getenv("LLMP_RUN_SHARED_IMAGE_REPLACEMENT") == nullptr)
    GTEST_SKIP() << "explicit bounded completed-request replacement diagnostic required";
  const char* store = std::getenv("LLMP_IMAGE_GENERATION_STORE");
  const char* noise = std::getenv("LLMP_IMAGE_GENERATION_NOISE");
  ASSERT_NE(store, nullptr);
  ASSERT_NE(noise, nullptr);
  auto life = std::make_unique<Lifetime>();
  life->replace_requests = true;
  const auto result = life->Run(store, noise);
  const auto retired = life->Retire();
  if (!retired) {
    std::ignore = life.release();
    FAIL() << Failed(retired).value_or("");
  }
  ASSERT_TRUE(result) << Failed(result).value_or("");
  EXPECT_EQ(life->completed_requests, 3U);
  EXPECT_EQ(life->completed_request_stages, 10U);
  EXPECT_EQ(life->released_requests, 2U);
  RecordProperty("completed_requests", life->completed_requests);
  RecordProperty("completed_stages", life->completed_request_stages);
  RecordProperty("released_request_owners", life->released_requests);
  RecordProperty("request_width", 1024);
  RecordProperty("request_height", 1024);
  RecordProperty("prompt_rows", life->prompt_rows);
  RecordProperty("dropped_rows", life->dropped_rows);
  RecordProperty("request_steps", "1,2,1");
  RecordProperty("complete_pixel_values_per_request", 4194304);
  RecordProperty("source_selected_weight_bytes", std::to_string(kSelectedWeights));
  RecordProperty("setup_paid_footprint_bytes", std::to_string(life->footprint));
  RecordProperty("fixed_bytes", std::to_string(life->fixed));
  RecordProperty("activation_bytes", std::to_string(life->activation));
  RecordProperty("pool_bytes", std::to_string(life->pool));
  RecordProperty("staging_bytes", std::to_string(life->staging));
  RecordProperty("retained_plan_floor_bytes", std::to_string(life->plans));
  RecordProperty("eager_catalog_bytes", std::to_string(life->eager_occupancy));
  RecordProperty("warmed_catalog_bytes", std::to_string(life->warmed_occupancy));
  RecordProperty("replacement_input_credit_bytes", std::to_string(life->replacement_credit));
  RecordProperty("construction_peak_bytes", std::to_string(life->construction_peak));
  RecordProperty("construction_retained_bytes", std::to_string(life->allowance->used()));
  RecordProperty("request_layout", llmp::base::ToHex(life->request_layout));
  RecordProperty("original_request_identity", llmp::base::ToHex(life->initial_request_identity));
  RecordProperty("first_flow_sha256", life->first[0]);
  RecordProperty("original_restored_pixels_sha256", life->final[0]);
  RecordProperty("changed_pixels_sha256", life->changed_pixels);
}
// Direct fixture runs the shared pipeline against the verified source.
// Output mode is a quality observation, never a timing sample.
TEST(ImageGenerationGpu, SameSourceMigrationScreen) {
  const char* arm = std::getenv("LLMP_IMAGE_SCREEN_ARM");
  if (arm == nullptr) GTEST_SKIP() << "explicit same-source migration screen required";
  ASSERT_EQ(std::string_view(arm), "shared");
  const char* store = std::getenv("LLMP_IMAGE_GENERATION_STORE");
  const char* noise = std::getenv("LLMP_IMAGE_GENERATION_NOISE");
  ASSERT_NE(store, nullptr);
  ASSERT_NE(noise, nullptr);
  auto life = std::make_unique<Lifetime>();
  if (const char* output = std::getenv("LLMP_IMAGE_SCREEN_OUTPUT")) {
    std::error_code error;
    ASSERT_TRUE(fs::is_directory(output, error));
    ASSERT_FALSE(error);
    life->output = output;
  }
  life->tiled = std::getenv("LLMP_IMAGE_SCREEN_TILED_CONV") != nullptr;
  life->masked_attention = std::getenv("LLMP_IMAGE_SCREEN_MASKED_ATTENTION") != nullptr;
  if (life->masked_attention) {
    ASSERT_FALSE(life->tiled);  // one changed implementation per screen
  }
  life->decoder_times = std::getenv("LLMP_IMAGE_SCREEN_DECODER_TIMES") != nullptr;
  if (life->decoder_times) {
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
    // Run checks both own-arm pins repeat; the invocation seals the current
    // candidate pin. A historical implementation's pixels are not a requirement.
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
    RecordProperty("tiled_tuning_keys", std::to_string(life->tuning_facts->facts().tuned.size()));
    RecordProperty("tiled_tuning_credit_bytes", std::to_string(life->tuning_credit));
  }
  if (life->masked_attention) {
    RecordProperty("attention_tuning_keys",
                   std::to_string(life->tuning_facts->facts().tuned.size()));
    RecordProperty("attention_tuning_credit_bytes", std::to_string(life->tuning_credit));
    RecordProperty("attention_source_nodes", std::to_string(life->attention_source_nodes));
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
  RecordProperty("selected_weight_bytes", std::to_string(life->selected_weight_bytes));
  RecordProperty("quality_counters_are_timing_samples", "false");
  RecordProperty("export_credit_bytes", std::to_string(life->export_credit));
  RecordProperty("construction_used_bytes", std::to_string(life->allowance->used()));
  RecordProperty("first_flow_sha256", life->first[0]);
  RecordProperty("pixels_sha256", life->final[0]);
  RecordProperty("native_report", life->native_report);
}
// The production Server adapter consumes the same prepared source owner as the
// direct fixture. Four serial images cover initial geometry, larger geometry,
// its repeated output, and original geometry again; Gemma state crosses both
// workspace growth and a real Server swap. No pixels accumulate across requests.
struct ServerGrowthLifetime {
  llmp::config::NodeConfig config;
  llmp::config::RuntimeRoles roles;
  rt::ServingOptions options;
  std::unique_ptr<rt::Server> server;
  std::unique_ptr<Lifetime> loader;
  rt::EngineImage* image = nullptr;
  rt::Llm* llm = nullptr;
  rt::ImageGenerationRequest seed;
  fs::path scratch;
  std::FILE* log = nullptr;
  bool ordinary = false, quality = false;
  bool retirement_attempted = false;
  fs::path small_noise, grown_noise, output;
  Status retirement;
  std::array<std::string, 4> pixels;
  llmp::base::Sha256Digest state_digest{};
  std::uint64_t initial_activation = 0, grown_activation = 0, initial_pool = 0, grown_pool = 0;
  std::uint64_t activation_base = 0, pool_base = 0, peak_construction = 0, peak_catalog = 0;
  std::uint32_t completed_images = 0, swaps = 0, retired_geometries = 0;
  std::weak_ptr<void> grown_owner;

  Status Retire() {
    if (retirement_attempted) return retirement;
    retirement_attempted = true;
    if (server) retirement = server->TearDown();
    return retirement;
  }
  ~ServerGrowthLifetime() {
    // Caller retains this complete owner on any unproven retirement.
    if (log) (void)std::fclose(log);
  }
  void Observe() {
    peak_construction = std::max(
        peak_construction, ordinary ? image->startup_host_bytes() : loader->allowance->used());
    peak_catalog =
        std::max(peak_catalog,
                 server->node().catalog().OccupancyOf(server->node().domain()).Total().value());
  }
  Status Activate(rt::Served& model) {
    rt::SwapParts parts;
    if (auto r = server->Activate(model, parts); !r) return r;
    if (auto r = server->FinishSwap(parts); !r) return r;
    ++swaps;
    return {};
  }
  Status Image(std::size_t index) {
    if (auto r = Activate(*image); !r) return r;
    std::string first;
    const auto result = server->InRequest(*image, [&]() -> Status {
      if (auto r = image->FirstOutput(first); !r) return r;
      return image->Finish(pixels[index]);
    });
    if (!result) return result;
    if (first.size() != 64 || pixels[index].size() != 64)
      return Error("Server image did not produce complete output identities");
    ++completed_images;
    Observe();
    return {};
  }
  Status Geometry(std::uint32_t width, std::uint32_t height) {
    if (ordinary) {
      if (auto r = image->PrepareRequest({.prompt = width == 1024 ? kRectangularPrompt : kPrompt,
                                          .width = width,
                                          .height = height,
                                          .steps = width == 1024 ? 2U : 1U,
                                          .noise = width == 1024 ? grown_noise : small_noise});
          !r)
        return r;
      ++retired_geometries;
      Observe();
      return {};
    }
    // Seed and candidate copies are paid before the copy below; the factory
    // separately pays generated graphs/request data before allocating them.
    auto request = seed;
    request.width = width;
    request.height = height;
    request.noise.resize(static_cast<std::size_t>(width / 16) * (height / 16) * 64);
    auto geometry = rt::ComponentImageGenerationGeometry(loader->source, std::move(request),
                                                         image->settings(), loader->allowance);
    if (!geometry) return Error(geometry.error());
    Observe();
    const auto previous = grown_owner;
    auto next = std::weak_ptr<void>(geometry->keep);
    if (auto r = image->ReplaceGeometry(std::move(*geometry)); !r) return r;
    if (!previous.expired()) return Error("retired image geometry still retains its owner");
    if (width == 1024) grown_owner = next;
    ++retired_geometries;
    Observe();
    return {};
  }
  Status Run(const fs::path& store, const fs::path& noise, const fs::path& gemma_store,
             bool installed_source = false, bool upstream_quality = false) {
    ordinary = installed_source;
    quality = upstream_quality;
    // Gemma2 2B's published v1 artifact (D-118).
    constexpr std::string_view gemma =
        "f176f912bc7761beee21cda0c22aff2080b8b84ed99faa6b962846c7e1379076";
    const char* temporary = std::getenv("LLMP_TEST_SCRATCH");
    if (!temporary) return Error("explicit private scratch required");
    std::string pattern = (fs::path(temporary) / "image-growth-XXXXXX").string();
    if (mkdtemp(pattern.data()) == nullptr) return Error("private image scratch failed");
    scratch = pattern;
    roles.installed = scratch / "installed";
    roles.spill = scratch / "spill";
    roles.state = scratch / "state";
    std::error_code ec;
    for (const auto& path : {roles.installed, roles.spill, roles.state}) {
      if (!fs::create_directory(path, ec) || ec || chmod(path.c_str(), 0700) != 0)
        return Error("private image directory failed");
    }
    for (auto id : kIds) {
      fs::create_directory_symlink(store / id, roles.installed / id, ec);
      if (ec) return Error("prepared artifact link failed");
    }
    fs::create_directory_symlink(store / kComposition, roles.installed / kComposition, ec);
    if (ec) return Error("prepared composition link failed");
    fs::create_directory_symlink(gemma_store / gemma, roles.installed / gemma, ec);
    if (ec) return Error("prepared Gemma link failed");
    llmp::config::ModelEntry image_entry;
    image_entry.name = "image";
    image_entry.composition = kComposition;
    image_entry.overrides["image_size"] = std::int64_t{quality ? 1024 : 512};
    image_entry.overrides["image_steps"] = std::int64_t{quality ? 40 : 1};
    config.models.push_back(std::move(image_entry));
    llmp::config::ModelEntry gemma_entry;
    gemma_entry.name = "gemma";
    gemma_entry.artifact = gemma;
    gemma_entry.overrides["context"] = std::int64_t{512};
    gemma_entry.overrides["prefill_chunk"] = std::int64_t{32};
    gemma_entry.overrides["max_slots"] = std::int64_t{1};
    if (!quality) config.models.push_back(std::move(gemma_entry));
    options.plain = true;
    options.diagnostic_budget_cap_bytes = kCap;
    // The existing Server parser/settings phase is separate from the held
    // image construction owner. Both live charges are included in node admission.
    options.diagnostic_startup_host_bytes = ordinary ? 0 : kStartup;
    if (ordinary) {
      options.diagnostic_auto_startup = true;
      options.image_prompt = kPrompt;
      // The bounded caller preparation is <2 MiB, separately covered by the
      // diagnostic process envelope before Server's own source-derived loan.
      auto bytes = ReadExact(noise, 524288);
      if (!bytes) return Error(bytes.error());
      auto directory = llmp::platform::OpenPrivateDirectory(-1, scratch.c_str());
      if (!directory)
        return Error("private request directory: " + std::to_string(directory.error()));
      llmp::artifact::FileDescriptor fd(*directory);
      for (const auto& [name, length] : {std::pair{"small.bf16", std::size_t{32 * 32 * 64 * 2}},
                                         std::pair{"grown.bf16", std::size_t{64 * 32 * 64 * 2}}})
        if (auto written = llmp::platform::ReplacePrivateFile(
                fd.get(), name, std::string_view(*bytes).substr(0, length));
            !written)
          return Error("private request noise write: " + std::to_string(written.error()));
      small_noise = scratch / "small.bf16";
      grown_noise = scratch / "grown.bf16";
      options.image_noise = quality ? noise : small_noise;
      if (quality) {
        const char* export_path = std::getenv("LLMP_IMAGE_SCREEN_OUTPUT");
        if (!export_path) return Error("quality run requires explicit completed output directory");
        output = export_path;
      }
    }
    log = std::tmpfile();
    if (!log) return Error("private Server diagnostic log failed");
    server = std::make_unique<rt::Server>(config, roles, options, log);
    auto& node = server->node();
    if (!ordinary) {
      if (auto r = rt::CheckStartupHostAllowance(
              kConstruction + kStartup, kCap, llmp::platform::AvailableMemoryBytes().value_or(0));
          !r)
        return r;
      if (auto r = node.Open(); !r) return r;
      loader = std::make_unique<Lifetime>(node);
      auto paid = en::ExecutionMetadataAllowance::Admit(
          node.catalog(), node.domain(), kConstruction,
          llmp::platform::AvailableMemoryBytes().value_or(0), rt::kUncountedMargin);
      if (!paid) return Error(paid.error());
      loader->allowance = std::move(*paid);
      if (auto r = loader->Load(store, noise); !r) return r;
      // Original 1024 source noise plus seed/candidate capacities and one Gemma
      // vocabulary row; no copied weight payload or retained rendered images.
      const auto request_credit = 4 * 524288ULL +
                                  8 * loader->source->prompt.size() * sizeof(std::int32_t) +
                                  256000ULL * sizeof(float) + 16384;
      if (!loader->allowance->Charge(request_credit)) return Error("swap request copies unpaid");
      seed = *loader->source;
      seed.keep = loader->allowance;
      loader->source->width = 512;
      loader->source->height = 512;
      loader->source->noise.resize(32 * 32 * 64);
      auto prepared = std::make_shared<rt::PreparedImageGeneration>();
      prepared->alias = "image";
      prepared->node = &node;
      prepared->owner = 0;
      prepared->source = loader->source;
      prepared->construction = loader->allowance;
      options.diagnostic_prepared_images.push_back(std::move(prepared));
    }
    if (auto r = server->Start(true); !r) return r;
    // Server owns its snapshot; clearing the caller's handoff cannot release it.
    options.diagnostic_prepared_images.clear();
    image = dynamic_cast<rt::EngineImage*>(server->Find("image"));
    llm = dynamic_cast<rt::EngineLlm*>(server->Find("gemma"));
    if (!image || (!quality && !llm))
      return Error("Server did not construct the shared image/LLM adapters");
    if (ordinary) {
      if (!options.diagnostic_prepared_images.empty() || !image->startup_host_bytes())
        return Error("ordinary image unexpectedly used a prepared-source handoff");
      if (auto r = image->PrepareRequest({.prompt = kPrompt, .output = output}); !r) return r;
    }
    activation_base = node.activations().base;
    pool_base = node.pool().base;
    initial_activation = node.activations().bytes;
    initial_pool = node.pool().bytes;
    const auto activation_extent = node.activations().extents.front();
    const auto pool_extent = node.pool().extents.front();
    if (auto r = Image(0); !r) return r;
    if (quality) return {};
    if (auto r = Activate(*llm); !r) return r;
    constexpr std::array<std::int32_t, 6> tokens{2, 818, 5279, 529, 7001, 563};
    std::vector<float> head;
    if (auto r = server->InRequest(*llm, [&]() { return llm->Prefill(tokens, head); }); !r)
      return r;
    if (head.size() != 256000 ||
        !std::ranges::all_of(head, [](float x) { return std::isfinite(x); }))
      return Error("Gemma full head is not finite");
    if (auto r = server->SaveSnapshot(*llm); !r) return r;
    auto digest = server->SnapshotDigest();
    if (!digest) return Error(digest.error());
    state_digest = *digest;
    if (auto r = Geometry(1024, 512); !r) return r;
    grown_activation = node.activations().bytes;
    grown_pool = node.pool().bytes;
    if (grown_activation <= initial_activation && grown_pool <= initial_pool)
      return Error("larger geometry did not exercise backed workspace growth");
    if (node.activations().base != activation_base || node.pool().base != pool_base ||
        node.activations().extents.front() != activation_extent ||
        node.pool().extents.front() != pool_extent)
      return Error("growth changed another retained model's workspace addresses or IDs");
    // Prepared while Gemma is resident: the pageable workspace's growth is
    // cataloged, not paged in beside the resident model; the image's own
    // activation brings it in, the outgoing model's weights making room.
    const auto grown_tail = [&](llmp::catalog::ExtentState want) -> Status {
      return node.Call(
          [&]() -> Status {
            for (const auto& [mapped, from] : {std::pair{&node.activations(), initial_activation},
                                               std::pair{&node.pool(), initial_pool}})
              for (std::size_t i = from / en::kPagedExtent; i < mapped->extents.size(); ++i)
                if (node.catalog().Describe(mapped->extents[i])->state != want)
                  return Error("the grown workspace is not where the active model puts it");
            return {};
          },
          "checking the grown workspace's residency");
    };
    if (!node.workspace_pageable()) return Error("the Server's workspace is not pageable");
    if (auto r = grown_tail(llmp::catalog::ExtentState::kNonresident); !r) return r;
    if (auto r = Image(1); !r) return r;
    if (auto r = grown_tail(llmp::catalog::ExtentState::kResident); !r) return r;
    if (auto r = Image(2); !r) return r;  // existing second-use capture policy
    if (pixels[1] != pixels[2]) return Error("grown geometry repetition changed own pixels");
    if (auto r = Activate(*llm); !r) return r;
    if (auto r = server->SaveSnapshot(*llm); !r) return r;
    digest = server->SnapshotDigest();
    if (!digest || *digest != state_digest)
      return Error("Gemma state changed across image growth and Server swap");
    if (auto r = Geometry(512, 512); !r) return r;
    if (node.activations().bytes != grown_activation || node.pool().bytes != grown_pool)
      return Error("retiring image geometry shrank shared workspace backing");
    if (auto r = Image(3); !r) return r;
    if (pixels[0] != pixels[3]) return Error("restored geometry changed its own pixels");
    return {};
  }
};
// The store holding Gemma2's v1 artifact: LLMP_IMAGE_GROWTH_GEMMA_STORE, else
// the published v1 store.
std::string GemmaStore() {
  const char* gemma = std::getenv("LLMP_IMAGE_GROWTH_GEMMA_STORE");
  return gemma != nullptr ? std::string(gemma) : llmp::test_support::V1Store();
}
TEST(ImageGenerationGpu, ServerSwapsRetainStateAcrossImageGeometryGrowth) {
  if (!std::getenv("LLMP_RUN_SHARED_IMAGE_GROWTH"))
    GTEST_SKIP() << "explicit bounded Server geometry/swap diagnostic required";
  const char* store = std::getenv("LLMP_IMAGE_GENERATION_STORE");
  const char* noise = std::getenv("LLMP_IMAGE_GENERATION_NOISE");
  const std::string gemma = GemmaStore();
  ASSERT_NE(store, nullptr);
  ASSERT_NE(noise, nullptr);
  ASSERT_FALSE(gemma.empty());
  auto life = std::make_unique<ServerGrowthLifetime>();
  const auto run = life->Run(store, noise, gemma);
  const auto retired = life->Retire();
  if (!retired) {
    std::ignore = life.release();
    FAIL() << Failed(retired).value_or("");
  }
  ASSERT_TRUE(run) << Failed(run).value_or("");
  EXPECT_EQ(life->completed_images, 4U);
  EXPECT_EQ(life->retired_geometries, 2U);
  RecordProperty("completed_images", life->completed_images);
  RecordProperty("completed_image_stages", 12);
  RecordProperty("server_activations", life->swaps);
  RecordProperty("retired_geometries", life->retired_geometries);
  RecordProperty("image_geometries", "512x512,1024x512,1024x512,512x512");
  RecordProperty("initial_activation_bytes", std::to_string(life->initial_activation));
  RecordProperty("grown_activation_bytes", std::to_string(life->grown_activation));
  RecordProperty("initial_pool_bytes", std::to_string(life->initial_pool));
  RecordProperty("grown_pool_bytes", std::to_string(life->grown_pool));
  RecordProperty("peak_catalog_bytes", std::to_string(life->peak_catalog));
  RecordProperty("peak_construction_bytes", std::to_string(life->peak_construction));
  RecordProperty("gemma_state_sha256", llmp::base::ToHex(life->state_digest));
  RecordProperty("original_restored_pixels_sha256", life->pixels[0]);
  RecordProperty("grown_repeated_pixels_sha256", life->pixels[1]);
  fs::remove_all(life->scratch);
}
TEST(ImageGenerationGpu, OrdinaryServerConsumesInstalledSourceAndRequests) {
  if (!std::getenv("LLMP_RUN_SHARED_IMAGE_SERVICE"))
    GTEST_SKIP() << "explicit paid ordinary-source service screen required";
  const char* store = std::getenv("LLMP_IMAGE_GENERATION_STORE");
  const char* noise = std::getenv("LLMP_IMAGE_GENERATION_NOISE");
  const std::string gemma = GemmaStore();
  ASSERT_NE(store, nullptr);
  ASSERT_NE(noise, nullptr);
  ASSERT_FALSE(gemma.empty());
  const bool quality = std::getenv("LLMP_IMAGE_SERVICE_UPSTREAM_QUALITY") != nullptr;
  ASSERT_TRUE(rt::CheckStartupHostAllowance(2 * kMiB, kCap,
                                            llmp::platform::AvailableMemoryBytes().value_or(0)));
  auto life = std::make_unique<ServerGrowthLifetime>();
  const auto run = life->Run(store, noise, gemma, true, quality);
  const auto retired = life->Retire();
  if (!retired) {
    std::ignore = life.release();
    FAIL() << Failed(retired).value_or("");
  }
  ASSERT_TRUE(run) << Failed(run).value_or("");
  EXPECT_EQ(life->completed_images, quality ? 1U : 4U);
  EXPECT_EQ(life->retired_geometries, quality ? 0U : 2U);
  RecordProperty("ordinary_installed_source", 1);
  RecordProperty("completed_images", life->completed_images);
  RecordProperty("completed_image_stages", quality ? 42 : 14);
  RecordProperty("server_activations", life->swaps);
  RecordProperty("request_steps", quality ? "40" : "1,2,2,1");
  RecordProperty("image_geometries", quality ? "1024x1024" : "512x512,1024x512,1024x512,512x512");
  RecordProperty("initial_activation_bytes", std::to_string(life->initial_activation));
  RecordProperty("grown_activation_bytes", std::to_string(life->grown_activation));
  RecordProperty("initial_pool_bytes", std::to_string(life->initial_pool));
  RecordProperty("grown_pool_bytes", std::to_string(life->grown_pool));
  RecordProperty("observed_catalog_max_bytes", std::to_string(life->peak_catalog));
  RecordProperty("construction_reserved_max_bytes", std::to_string(life->peak_construction));
  RecordProperty("gemma_state_sha256", llmp::base::ToHex(life->state_digest));
  RecordProperty("original_pixels_sha256", life->pixels[0]);
  RecordProperty("grown_repeated_pixels_sha256", life->pixels[1]);
  fs::remove_all(life->scratch);
}
}  // namespace
