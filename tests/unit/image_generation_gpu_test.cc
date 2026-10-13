// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Default-off complete source generation through the shared typed pipeline.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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
#include "expected_error.h"
#include "platform/host_probe.h"
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
  std::unique_ptr<rt::EngineImage> image;
  std::vector<en::PagedModel*> entered;
  bool teardown_attempted = false;
  std::uint64_t footprint = 0, fixed = 0, activation = 0, pool = 0, staging = 0, plans = 0;
  std::uint64_t startup_parser = 0;
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
  Status Run(const fs::path& store, const fs::path& noise) {
    const auto before = llmp::platform::AvailableMemoryBytes().value_or(0);
    if (auto paid = rt::CheckStartupHostAllowance(kConstruction, kCap, before); !paid) return paid;
    if (auto opened = node.Open(); !opened) return opened;
    auto paid = en::ExecutionMetadataAllowance::Admit(
        node.catalog(), node.domain(), kConstruction,
        llmp::platform::AvailableMemoryBytes().value_or(0), rt::kUncountedMargin);
    if (!paid) return Error(paid.error());
    allowance = std::move(*paid);
    if (auto loaded = Load(store, noise); !loaded) return loaded;
    llmp::config::ModelEntry entry;
    entry.name = "shared-image-generation";
    entry.artifact = std::string(kComposition);
    rt::ModelSettings settings;
    settings.name = entry.name;
    settings.architecture = "qwen_image_pipeline";
    image = std::make_unique<rt::EngineImage>(
        node, entry, settings, 0, rt::ComponentImageGenerationBuild(node, 0, source, allowance));
    const auto setup = image->Setup();
    if (image->has_paged()) entered.push_back(&image->paged());
    if (!setup) return setup;
    const auto weights = image->weight_read_bytes();
    if (weights != kSelectedWeights) return Error("complete source closure page union differs");
    fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    activation = image->activations_needed();
    pool = std::max<std::uint64_t>(image->pool_needed(), en::kPagedExtent);
    staging = image->host_input_bytes();
    plans = image->plan_floor_bytes();
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
    // Repeated generation uses the same retained typed plans/captures and
    // restores the original latent port only after a completed prior Finish.
    for (std::size_t run = 0; run < first.size(); ++run) {
      auto complete =
          node.WithRequest(0, image->request_closure(), "shared image generation", [&]() -> Status {
            if (auto step = image->FirstOutput(first[run]); !step) return step;
            return image->Finish(final[run]);
          });
      if (!complete) return complete;
      if (first[run].size() != 64 || final[run].size() != 64)
        return Error("complete generation did not publish both completed output boundaries");
    }
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
}  // namespace
