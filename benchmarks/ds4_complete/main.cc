// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Explicit temporary reference, never a serving default or a CTest model load.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <print>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "ds4_complete/runner.h"

namespace {
namespace dc = jitllm::benchmarks::ds4_complete;
namespace en = jitllm::engine;
namespace base = jitllm::base;
using Clock = std::chrono::steady_clock;
inline constexpr std::string_view kProfileHeadSha =
    "499a05df44162d26dd151f44003388a68a494c86265ea756fecdabed85ae13b8";
inline constexpr std::string_view kProfileTokensSha =
    "60329b1e4ff5d19d40082666e8c08b86655d4b1173486aecbc4a752bb6aa3ac7";

std::string Quoted(std::string_view value) {
  std::string result;
  base::json::AppendQuoted(value, result);
  return result;
}
template <typename T>
std::string Digest(std::span<const T> values) {
  base::Sha256 hash;
  hash.Update(std::as_bytes(values));
  return base::ToHex(hash.Finish());
}
double Seconds(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

// This mode inspects authenticated metadata and plans only. It never opens a
// native node, queries CUDA, opens payload shards or creates prepared weights.
inline constexpr std::uint64_t kPreparationPayload = std::uint64_t{64} << 20U;
struct MetadataPlan {
  en::Ds4PreparedWeightSet weights;
  std::vector<std::uint64_t> tensor_chunk_counts;
  std::uint64_t expert_tensors = 0;
  std::uint64_t raw_bytes = 0;
  std::uint64_t stored_bytes = 0;
  std::uint64_t direct_read_bytes = 0;
  std::uint64_t chunk_count = 0;
};

std::expected<MetadataPlan, std::string> PlanMetadata(const jitllm::artifact::Artifact& artifact) {
  const auto& profile = jitllm::model::Dsv4Flash();
  auto binding = jitllm::model::BindDsv4(profile, artifact);
  if (!binding) return std::unexpected(binding.error());
  const auto format = [&](const jitllm::model::Dsv4Tensor& tensor,
                          std::string_view type) -> dc::Result {
    if (tensor.type != type)
      return std::unexpected(artifact.resources()[tensor.index].name +
                             ": original recipe requires " + std::string(type));
    return {};
  };
  for (const auto* tensor : {&binding->token_embd, &binding->hc_head_fn})
    if (auto checked = format(*tensor, "F16"); !checked) return std::unexpected(checked.error());
  if (auto checked = format(binding->output, "Q8_0"); !checked)
    return std::unexpected(checked.error());
  for (const auto& layer : binding->layers) {
    for (const auto* tensor : {&layer.hc_attn_fn, &layer.hc_ffn_fn, &layer.router})
      if (auto checked = format(*tensor, "F16"); !checked) return std::unexpected(checked.error());
    for (const auto* tensor : {&layer.q_a, &layer.q_b, &layer.kv, &layer.out_a, &layer.out_b,
                               &layer.gate_shexp, &layer.up_shexp, &layer.down_shexp})
      if (auto checked = format(*tensor, "Q8_0"); !checked) return std::unexpected(checked.error());
    if (layer.ratio != 0) {
      for (const auto* tensor : {&layer.comp_kv, &layer.comp_gate})
        if (auto checked = format(*tensor, "F16"); !checked)
          return std::unexpected(checked.error());
    }
    if (layer.ratio == 4) {
      for (const auto* tensor :
           {&layer.idx_q_b, &layer.idx_proj, &layer.idx_comp_kv, &layer.idx_comp_gate})
        if (auto checked = format(*tensor, "F16"); !checked)
          return std::unexpected(checked.error());
    }
  }
  auto planned = en::PlanDs4PreparedWeights(artifact);
  if (!planned) return std::unexpected(planned.error());
  MetadataPlan result;
  result.weights = std::move(*planned);
  if (result.weights.artifact_id != dc::kCommunityArtifact || result.weights.tensors.empty() ||
      result.weights.tensors.size() > 1024)
    return std::unexpected("invalid complete prepared metadata inventory");
  std::set<std::pair<bool, std::uint32_t>> identities;
  std::uint64_t experts = 0;
  std::uint64_t dense = 0;
  std::uint64_t largest = 0;
  for (const auto& tensor : result.weights.tensors) {
    if (!identities.emplace(tensor.expert_array, tensor.index).second ||
        tensor.sources.size() != (tensor.expert_array ? profile.experts : 1U) ||
        tensor.shape.groups != tensor.sources.size())
      return std::unexpected(tensor.name + ": duplicate identity or incomplete source coverage");
    if (tensor.expert_array) ++result.expert_tensors;
    const auto layout = jitllm::kernels::ggml::Ds4AlignedLayoutOf(tensor.shape);
    if (!layout) return std::unexpected(tensor.name + ": invalid physical preparation layout");
    auto chunks = en::PlanDs4WeightChunks(tensor, kPreparationPayload);
    if (!chunks) return std::unexpected(tensor.name + ": " + chunks.error());
    std::uint64_t next_block = 0;
    std::uint64_t payload = 0;
    for (const auto& chunk : *chunks) {
      if (chunk.first_block != next_block || chunk.blocks == 0 ||
          chunk.payload_bytes != chunk.blocks * layout->raw_block_bytes ||
          __builtin_add_overflow(next_block, chunk.blocks, &next_block) ||
          __builtin_add_overflow(payload, chunk.payload_bytes, &payload) ||
          __builtin_add_overflow(result.direct_read_bytes, chunk.read_bytes,
                                 &result.direct_read_bytes))
        return std::unexpected(tensor.name + ": chunks do not cover the complete original plane");
    }
    std::uint64_t rounded = 0;
    auto& total = tensor.expert_array ? experts : dense;
    if (next_block != layout->blocks || payload != layout->raw_bytes.value() ||
        __builtin_add_overflow(result.raw_bytes, payload, &result.raw_bytes) ||
        __builtin_add_overflow(total, layout->packed_bytes.value(), &total) ||
        __builtin_add_overflow(layout->packed_bytes.value(), jitllm::artifact::kFileAlignment - 1,
                               &rounded) ||
        __builtin_add_overflow(
            result.stored_bytes,
            (rounded / jitllm::artifact::kFileAlignment) * jitllm::artifact::kFileAlignment,
            &result.stored_bytes) ||
        __builtin_add_overflow(result.chunk_count, chunks->size(), &result.chunk_count))
      return std::unexpected(tensor.name + ": complete metadata totals overflow or differ");
    largest = std::max(largest, layout->packed_bytes.value());
    result.tensor_chunk_counts.push_back(chunks->size());
  }
  if (result.expert_tensors != std::uint64_t{profile.layers} * 3 ||
      result.weights.replaced_groups.size() != std::uint64_t{profile.layers} * profile.experts ||
      experts != result.weights.aligned_expert_bytes ||
      dense != result.weights.additive_dense_bytes ||
      largest != result.weights.largest_tensor_bytes ||
      result.tensor_chunk_counts.size() != result.weights.tensors.size())
    return std::unexpected("complete expert coverage or inventory byte totals differ");
  return result;
}

dc::Result SaveMetadata(const std::filesystem::path& path, const MetadataPlan& plan) {
  // C++23 noreplace refuses an existing output atomically, including links.
  std::ofstream output(path, std::ios::out | std::ios::noreplace);
  if (!output) return std::unexpected("metadata output must be a fresh writable file");
  output << R"({"complete":true,"model_loaded":false,"cuda_initialized":false,"artifact":)"
         << Quoted(plan.weights.artifact_id)
         << R"(,"payload_read":false,"layers":43,"experts_per_layer":256,"expert_arrays":)"
         << plan.expert_tensors << R"(,"tensor_count":)" << plan.weights.tensors.size()
         << R"(,"chunk_count":)" << plan.chunk_count << R"(,"payload_limit":)"
         << kPreparationPayload << R"(,"raw_bytes":)" << plan.raw_bytes
         << R"(,"aligned_expert_bytes":)" << plan.weights.aligned_expert_bytes
         << R"(,"additive_dense_bytes":)" << plan.weights.additive_dense_bytes
         << R"(,"largest_tensor_bytes":)" << plan.weights.largest_tensor_bytes
         << R"(,"stored_bytes":)" << plan.stored_bytes << R"(,"direct_read_bytes":)"
         << plan.direct_read_bytes << R"(,"replaced_groups":[)";
  for (std::size_t i = 0; i < plan.weights.replaced_groups.size(); ++i) {
    if (i != 0) output << ',';
    output << plan.weights.replaced_groups[i];
  }
  output << R"(],"tensors":[)";
  for (std::size_t i = 0; i < plan.weights.tensors.size(); ++i) {
    if (i != 0) output << ',';
    const auto& tensor = plan.weights.tensors[i];
    output << R"({"name":)" << Quoted(tensor.name) << R"(,"index":)" << tensor.index
           << R"(,"expert_array":)" << (tensor.expert_array ? "true" : "false") << R"(,"kind":)"
           << static_cast<unsigned>(tensor.shape.kind) << R"(,"input":)" << tensor.shape.input
           << R"(,"output":)" << tensor.shape.output << R"(,"groups":)" << tensor.shape.groups
           << R"(,"raw_bytes":)" << tensor.layout.raw_bytes.value() << R"(,"packed_bytes":)"
           << tensor.layout.packed_bytes.value() << R"(,"chunk_count":)"
           << plan.tensor_chunk_counts[i] << R"(,"blocks":)" << tensor.layout.blocks
           << R"(,"scales_offset":)" << tensor.layout.scales_offset << R"(,"codes_offset":)"
           << tensor.layout.codes_offset << R"(,"sources":[)";
    for (std::size_t j = 0; j < tensor.sources.size(); ++j) {
      if (j != 0) output << ',';
      const auto& source = tensor.sources[j];
      output << R"({"shard":)" << source.shard << R"(,"offset":)" << source.file_offset
             << R"(,"bytes":)" << source.bytes << R"(,"validated_end":)" << source.file_end << '}';
    }
    output << "]}";
  }
  output << "]}\n";
  output.close();
  if (!output) return std::unexpected("metadata output write failed");
  return {};
}

std::expected<std::vector<std::int32_t>, std::string> Tokens(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) return std::unexpected("cannot open fixed token input");
  std::vector<std::int32_t> result;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    if (!result.empty()) return std::unexpected("fixed comparison accepts exactly one token row");
    const auto tab = line.find('\t');
    const std::string_view text =
        tab == std::string::npos ? std::string_view(line) : std::string_view(line).substr(tab + 1);
    std::istringstream ids{std::string(text)};
    std::string spelling;
    while (ids >> spelling) {
      std::int64_t id = 0;
      const auto parsed = std::from_chars(spelling.data(), spelling.data() + spelling.size(), id);
      if (parsed.ec != std::errc{} || parsed.ptr != spelling.data() + spelling.size()) {
        return std::unexpected("fixed input has malformed or overflowing token text");
      }
      if (id < 0 || id >= 129280 || result.size() == 8192) {
        return std::unexpected("fixed input has an invalid ID or exceeds8192 tokens");
      }
      result.push_back(static_cast<std::int32_t>(id));
    }
    if (ids.bad()) return std::unexpected("fixed input read failed");
  }
  if (file.bad() || result.size() != 8192) {
    return std::unexpected("fixed input must contain exactly8192 current token IDs");
  }
  return result;
}

dc::Result Save(const std::filesystem::path& path, std::span<const float> values) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size_bytes()));
  output.close();
  if (!output) return std::unexpected("cannot persist complete final logits");
  return {};
}

void Dispatch(std::ostream& output, const dc::ResolvedDispatch& dispatch) {
  output << "{\"device\":" << dispatch.device << ",\"architecture\":" << dispatch.architecture
         << ",\"stream\":" << dispatch.stream << ",\"sms\":" << dispatch.planned_sms
         << ",\"pool_bytes\":" << dispatch.native_pool_bytes
         << ",\"cublas_bytes\":" << dispatch.cublas_bytes << ",\"stages\":[";
  bool comma = false;
  for (const auto& stage : dispatch.stages) {
    if (comma) output << ',';
    comma = true;
    output << "{\"layer\":" << stage.layer << ",\"stage\":" << Quoted(stage.stage)
           << ",\"kind\":" << Quoted(stage.kind) << ",\"scratch_bytes\":" << stage.scratch_bytes
           << '}';
  }
  output << "]}";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 4 && std::string_view(argv[1]) == "--plan-only") {
    auto artifact = jitllm::artifact::Artifact::Open(argv[2]);
    if (!artifact || artifact->id() != dc::kCommunityArtifact) {
      std::println(stderr, "fixed community artifact was refused");
      return 2;
    }
    auto planned = PlanMetadata(*artifact);
    if (!planned) {
      std::println(stderr, "{}", planned.error());
      return 1;
    }
    if (auto saved = SaveMetadata(argv[3], *planned); !saved) {
      std::println(stderr, "{}", saved.error());
      return 1;
    }
    return 0;
  }
  const bool profile_study = argc == 7 && std::string_view(argv[6]) == "--profile";
  if (argc != 6 && !profile_study) {
    std::println(stderr,
                 "usage: jitllm_ds4_complete ARTIFACT SCRATCH TOKENS OUTPUT REPEATS(1..3) "
                 "[--profile]");
    return 2;
  }
  const std::string_view repeat_text = argv[5];
  unsigned repeats = 0;
  const auto parsed =
      std::from_chars(repeat_text.data(), repeat_text.data() + repeat_text.size(), repeats);
  if (parsed.ec != std::errc{} || parsed.ptr != repeat_text.data() + repeat_text.size() ||
      repeats == 0 || repeats > 3 || (profile_study && repeats != 3))
    return 2;
  const std::filesystem::path artifact_path(argv[1]);
  const std::filesystem::path scratch(argv[2]);
  const std::filesystem::path output_path(argv[4]);
  auto tokens = Tokens(argv[3]);
  if (!tokens) {
    std::println(stderr, "{}", tokens.error());
    return 2;
  }
  if (profile_study && Digest<std::int32_t>(*tokens) != kProfileTokensSha) {
    std::println(stderr, "diagnostic study requires the pinned complete-reference token IDs");
    return 2;
  }
  auto artifact = jitllm::artifact::Artifact::Open(artifact_path);
  if (!artifact || artifact->id() != dc::kCommunityArtifact) {
    std::println(stderr, "fixed community artifact was refused");
    return 2;
  }
  if (auto planned = PlanMetadata(*artifact); !planned) {
    std::println(stderr, "{}", planned.error());
    return 1;
  }
  std::error_code filesystem_error;
  std::filesystem::create_directories(output_path, filesystem_error);
  if (filesystem_error) return 2;
  std::filesystem::create_directories(scratch, filesystem_error);
  if (filesystem_error) return 2;
  int sms = 0;
  int major = 0;
  int minor = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess ||
      sms <= 0 || major != 12 || minor != 1) {
    std::println(stderr, "temporary matched scope requires the actual GB10 device");
    return 2;
  }
  std::println(stderr, "preparing original aligned weight representations");
  auto prepared = dc::PrepareModelWeights(*artifact, scratch);
  if (!prepared) {
    std::println(stderr, "{}", prepared.error());
    return 1;
  }
  std::vector<dc::Pass> passes;
  en::PagedNode node({.compute_streams = 1});
  dc::Runner runner(node, static_cast<std::uint32_t>(sms));
  const auto setup_started = Clock::now();
  double setup_seconds = 0;
  double load_seconds = 0;
  double warmup_seconds = 0;
  std::uint64_t budget = 0;
  std::uint64_t state = 0;
  std::uint64_t raw_reads = 0;
  std::uint64_t aligned_reads = 0;
  std::uint32_t profile_mark_count = 0;
  std::array<std::uint64_t, 2> profile_setup_free_bytes{};
  std::string warmup_sha;
  std::vector<en::Ds4AlignedWeightView> representations;
  auto run = [&]() -> dc::Result {
    if (auto opened = node.Open(); !opened) return opened;
    if (auto setup = runner.Setup(artifact_path, scratch, *prepared); !setup) return setup;
    setup_seconds = Seconds(setup_started);
    budget = runner.budget_bytes();
    raw_reads = runner.raw_read_bytes();
    aligned_reads = runner.aligned().read_bytes();
    const auto load_started = Clock::now();
    if (auto loaded = runner.Load(); !loaded) return loaded;
    load_seconds = Seconds(load_started);
    if (profile_study) {
      if (auto ready = runner.PrepareProfile(); !ready) return ready;
      profile_mark_count = runner.profile_mark_count();
      profile_setup_free_bytes = runner.profile_setup_free_bytes();
    }
    state = runner.physical_state_bytes();
    representations.assign(runner.aligned().views().begin(), runner.aligned().views().end());
    if (auto fresh = runner.Initialize(); !fresh) return fresh;
    std::println(stderr, "warmup complete original8K pipeline");
    auto warmup = runner.Prefill(*tokens);
    if (!warmup) return std::unexpected(warmup.error());
    warmup_seconds = warmup->wall_seconds;
    if (profile_study) {
      warmup_sha = Digest<float>(warmup->logits);
      if (warmup->logits.size() != 129280 || warmup_sha != kProfileHeadSha ||
          std::ranges::any_of(warmup->logits, [](float value) { return !std::isfinite(value); }))
        return std::unexpected("diagnostic warmup differs from the pinned original full head");
      if (auto saved = Save(output_path / "warmup.f32", warmup->logits); !saved) return saved;
    }
    const auto pass_count = profile_study ? 7U : repeats;
    passes.reserve(pass_count);
    for (unsigned repeat = 0; repeat < pass_count; ++repeat) {
      if (auto fresh = runner.Initialize(); !fresh) return fresh;
      std::println(stderr, "timed complete original8K pass{}", repeat);
      auto pass = runner.Prefill(*tokens, profile_study && repeat == 3);
      if (!pass) return std::unexpected(pass.error());
      if (profile_study &&
          pass->profile.size() !=
              (repeat == 3 ? (std::size_t{2} * dc::kLayers * dc::kProfileChains) + 3 : 0))
        return std::unexpected("diagnostic pass has incomplete or unexpected chain samples");
      if (std::ranges::any_of(pass->logits, [](float value) { return !std::isfinite(value); })) {
        return std::unexpected("complete reference produced nonfinite final logits");
      }
      if (pass->logits.size() != warmup->logits.size() ||
          std::memcmp(pass->logits.data(), warmup->logits.data(),
                      pass->logits.size() * sizeof(float)) != 0) {
        return std::unexpected("complete reference own-repeat logits are not byte equal");
      }
      if (profile_study && Digest<float>(pass->logits) != kProfileHeadSha)
        return std::unexpected("diagnostic pass differs from the pinned original full head");
      if (auto saved = Save(output_path / ("pass" + std::to_string(repeat) + ".f32"), pass->logits);
          !saved) {
        return saved;
      }
      passes.push_back(std::move(*pass));
    }
    return {};
  }();
  const auto retirement_started = Clock::now();
  const std::array<en::PagedModel*, 1> models{&runner};
  auto retired = node.TearDown(models);
  const auto retirement_seconds = Seconds(retirement_started);
  if (!retired) {
    std::println(stderr, "complete reference retirement failed: {}", retired.error());
    std::abort();
  }
  if (!run) {
    std::println(stderr, "complete reference failed: {}", run.error());
    return 1;
  }
  std::ofstream receipt(output_path / "receipt.json");
  receipt << std::setprecision(17)
          << R"({"complete":true,"retired":true,"own_repeat_byte_equal":true,)"
          << R"("original_engine_parity":false,"artifact":)" << Quoted(artifact->id())
          << ",\"tokens_sha256\":" << Quoted(Digest<std::int32_t>(*tokens))
          << R"(,"context_capacity":8192,"prompt_tokens":8192,"chunk_rows":4096,"layers":43,)"
          << R"("head_rows":1,"vocab":129280,"preparation_seconds":)"
          << prepared->preparation_seconds
          << ",\"preparation_setup_seconds\":" << prepared->setup_seconds
          << ",\"preparation_retirement_seconds\":" << prepared->teardown_seconds
          << ",\"preparation_read_bytes\":" << prepared->direct_read_bytes
          << ",\"prepared_stored_bytes\":" << prepared->stored_bytes
          << ",\"preparation_physical_bytes\":" << prepared->preparation_physical_bytes
          << ",\"preparation_jobs\":" << prepared->preparation_jobs
          << ",\"setup_seconds\":" << setup_seconds << ",\"load_seconds\":" << load_seconds
          << ",\"warmup_seconds\":" << warmup_seconds
          << ",\"retirement_seconds\":" << retirement_seconds
          << ",\"execution_budget_bytes\":" << budget << ",\"physical_state_bytes\":" << state
          << ",\"raw_pagein_read_bytes\":" << raw_reads
          << ",\"aligned_pagein_read_bytes\":" << aligned_reads
          << ",\"profile_study\":" << (profile_study ? "true" : "false")
          << ",\"profile_mark_count\":" << profile_mark_count << ",\"profile_setup_free_bytes\":["
          << profile_setup_free_bytes[0] << ',' << profile_setup_free_bytes[1] << ']'
          << ",\"profile_pinned_head_sha256\":" << Quoted(profile_study ? kProfileHeadSha : "")
          << ",\"warmup_logits_sha256\":" << Quoted(warmup_sha)
          << ",\"profile_wall_is_diagnostic\":" << (profile_study ? "true" : "false")
          << ",\"representations\":[";
  bool comma = false;
  for (const auto& view : representations) {
    if (comma) receipt << ',';
    comma = true;
    receipt << "{\"index\":" << view.index
            << ",\"expert_array\":" << (view.expert_array ? "true" : "false")
            << ",\"bytes\":" << view.layout.packed_bytes.value()
            << ",\"stored_sha256\":" << Quoted(base::ToHex(view.stored_sha256))
            << ",\"content_key\":" << Quoted(base::ToHex(view.content_key)) << '}';
  }
  receipt << "],\"passes\":[";
  comma = false;
  for (std::size_t pass_index = 0; pass_index < passes.size(); ++pass_index) {
    const auto& pass = passes[pass_index];
    if (comma) receipt << ',';
    comma = true;
    std::string_view role = "original";
    if (profile_study) {
      if (pass_index < 3)
        role = "original-before";
      else if (pass_index == 3)
        role = "profiled-original";
      else
        role = "original-after";
    }
    receipt << "{\"role\":" << Quoted(role)
            << ",\"diagnostic\":" << (!pass.profile.empty() ? "true" : "false")
            << ",\"numerical_seconds\":" << pass.seconds
            << ",\"prefill_wall_seconds\":" << pass.wall_seconds
            << ",\"result_copy_seconds\":" << pass.result_copy_seconds
            << ",\"tok_s\":" << 8192.0 / pass.wall_seconds << ",\"chunks\":[" << pass.chunks[0]
            << ',' << pass.chunks[1] << ']'
            << ",\"logits_sha256\":" << Quoted(Digest<float>(pass.logits))
            << ",\"native_jobs\":" << pass.steps.steps
            << ",\"device_seconds\":" << pass.steps.device
            << ",\"dispatch_seconds\":" << pass.steps.dispatch
            << ",\"job_host_seconds\":" << pass.steps.job
            << ",\"after_seconds\":" << pass.steps.after << ",\"resolved_dispatch\":[";
    Dispatch(receipt, pass.dispatch[0]);
    receipt << ',';
    Dispatch(receipt, pass.dispatch[1]);
    receipt << "],\"profile_samples\":[";
    for (std::size_t sample_index = 0; sample_index < pass.profile.size(); ++sample_index) {
      if (sample_index != 0) receipt << ',';
      const auto& sample = pass.profile[sample_index];
      receipt << "{\"chunk\":" << sample.chunk << ",\"layer\":" << sample.layer
              << ",\"ratio\":" << sample.ratio << ",\"chain\":" << Quoted(sample.chain)
              << ",\"active\":" << (sample.active ? "true" : "false")
              << ",\"milliseconds\":" << sample.milliseconds << '}';
    }
    receipt << "]}";
  }
  receipt << "]}\n";
  receipt.close();
  return receipt ? 0 : 1;
}
