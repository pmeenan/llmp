// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "cli/import.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <format>
#include <optional>
#include <vector>

#include "base/build_info.h"
#include "cli/cli.h"
#include "import/archive_input.h"
#include "import/mlx_qwen38.h"
#include "import/modelopt_qwen38.h"
#include "import/qwen_image.h"
#include "import/reader.h"
#include "import/writer.h"

namespace llmp::cli {
namespace {
bool Print(std::FILE* stream, std::string_view text) {
  return std::fwrite(text.data(), 1, text.size(), stream) == text.size() &&
         std::fflush(stream) == 0;
}
int Failure(std::FILE* err, const import::Error& error) {
  Print(err, "llmp import: " + error.ToString() + '\n');
  return kExitFailure;
}
}  // namespace
int Import(std::span<const std::string_view> args, std::FILE* out, std::FILE* err) {
  const auto usage = [&]() {
    Print(err, "Usage: llmp import SOURCE --store DIRECTORY [--metadata-mib N]\n");
    return kExitUsage;
  };
  if (args.empty() || args[0].empty() || args[0].starts_with('-')) return usage();
  std::filesystem::path source(args[0]), store;
  // Caller-owned checkpoint storage is bounded by the container readers.
  // This separate budget covers normalization and publication metadata.
  std::uint64_t metadata = std::uint64_t{4} << 30U;
  bool limit_seen = false;
  for (std::size_t i = 1; i < args.size(); i += 2) {
    if (i + 1 == args.size() || args[i + 1].empty()) return usage();
    if (args[i] == "--store" && store.empty())
      store = args[i + 1];
    else if (args[i] == "--metadata-mib" && !limit_seen) {
      std::uint64_t mib = 0;
      const auto parsed =
          std::from_chars(args[i + 1].data(), args[i + 1].data() + args[i + 1].size(), mib);
      if (parsed.ec != std::errc{} || parsed.ptr != args[i + 1].data() + args[i + 1].size() ||
          mib == 0 || mib > 65536)
        return usage();
      metadata = mib << 20U;
      limit_seen = true;
    } else
      return usage();
  }
  if (store.empty()) return usage();
  auto archive = import::FindArchiveInput(source);
  if (!archive) return Failure(err, archive.error());
  if (*archive) {
    auto selected = import::SelectArchiveSource(**archive, source);
    if (!selected) return Failure(err, selected.error());
    source = std::move(*selected);
  }
  if (source.extension() == ".safetensors") source = source.parent_path();
  auto input = import::ReadCheckpoint(source);
  if (!input) return Failure(err, input.error());
  std::vector<std::optional<base::Sha256Digest>> expected;
  if (*archive) {
    auto identities = import::ArchiveIdentities(**archive, *input);
    if (!identities) return Failure(err, identities.error());
    expected = std::move(*identities);
  }
  // Both owned plan kinds use the same publication and archive receipt path.
  const auto publish = [&](const auto& plan, const import::Checkpoint& checkpoint,
                           std::uint64_t used) {
    const auto& info = base::GetBuildInfo();
    const std::string build = info.commit.empty()
                                  ? "unknown"
                                  : std::string(info.commit) + (info.modified ? ".dirty" : "");
    auto result = import::WriteArtifact(
        plan,
        {.store = store, .build = build, .expected = expected, .metadata_bytes = metadata - used});
    if (!result) return Failure(err, result.error());
    if (!Print(out,
               std::format("{} {}\n", result->reused ? "verified" : "published", result->id))) {
      Print(err, "llmp import: artifact published; cannot write result to standard output\n");
      return kExitFailure;
    }
    if (*archive) {
      auto opened = artifact::Artifact::Open(result->path);
      if (!opened) {
        Print(err,
              "llmp import: artifact published, but it could not be reopened for catalog "
              "recording\n");
        return kExitFailure;
      }
      if (auto recorded = import::RecordArchiveImport(**archive, checkpoint, *opened); !recorded) {
        Print(err, "llmp import: artifact published; catalog event failed: " +
                       recorded.error().ToString() + '\n');
        return kExitFailure;
      }
    }
    return kExitOk;
  };
  const auto funded = [&](const import::NormalizationCost& bound) {
    return bound.metadata_bytes <= metadata;
  };
  if (input->container == "safetensors" && input->architecture == "qwen3_8_flash_next" &&
      input->quantization == "modelopt MIXED_PRECISION") {
    auto bound = import::ModelOptTargetBound(*input);
    if (!bound) return Failure(err, bound.error());
    if (!funded(*bound))
      return Failure(err, {.rule = import::Rule::kBounds,
                           .reason = "normalization exceeds the requested metadata budget"});
    auto plan = import::NormalizeModelOptTarget(std::move(*input),
                                                {bound->metadata_bytes, bound->work_units});
    if (!plan) return Failure(err, plan.error());
    return publish(*plan, plan->parts().source(), bound->metadata_bytes);
  }
  if (input->container == "mlx" && input->architecture == "qwen4_exp" &&
      input->quantization == "mlx affine") {
    auto bound = import::MlxTargetBound(*input);
    if (!bound) return Failure(err, bound.error());
    if (!funded(*bound))
      return Failure(err, {.rule = import::Rule::kBounds,
                           .reason = "normalization exceeds the requested metadata budget"});
    auto plan =
        import::NormalizeMlxTarget(std::move(*input), {bound->metadata_bytes, bound->work_units});
    if (!plan) return Failure(err, plan.error());
    return publish(*plan, plan->parts().source(), bound->metadata_bytes);
  }
  if (input->container == "diffusers" && input->architecture == "QwenImage21Pipeline") {
    auto bound = import::QwenImagePipelineBound(*input);
    if (!bound) return Failure(err, bound.error());
    if (!funded(*bound))
      return Failure(err, {.rule = import::Rule::kBounds,
                           .reason = "normalization exceeds the requested metadata budget"});
    auto plan = import::PlanQwenImagePipeline(*input);
    if (!plan) return Failure(err, plan.error());

    const auto& info = base::GetBuildInfo();
    const std::string build = info.commit.empty()
                                  ? "unknown"
                                  : std::string(info.commit) + (info.modified ? ".dirty" : "");
    std::vector<artifact::CompositionComponent> published_components;
    published_components.reserve(3);

    for (std::size_t i = 0; i < 3; ++i) {
      const auto& comp_plan = plan->components[i];
      std::vector<std::optional<base::Sha256Digest>> comp_expected;
      if (*archive) {
        auto comp_identities = import::ArchiveFileIdentities(**archive, comp_plan.checkpoint.files);
        if (!comp_identities) return Failure(err, comp_identities.error());
        comp_expected = std::move(*comp_identities);
      }
      auto result =
          import::WriteArtifact(comp_plan, {.store = store,
                                            .build = build,
                                            .expected = comp_expected,
                                            .metadata_bytes = metadata - bound->metadata_bytes});
      if (!result) return Failure(err, result.error());
      published_components.push_back({
          .role = comp_plan.checkpoint.role,
          .artifact = result->id,
          .architecture = comp_plan.checkpoint.architecture,
      });
    }

    std::ranges::sort(published_components, {}, &artifact::CompositionComponent::role);

    auto comp_result = import::WriteComposition({
        .store = store,
        .build = build,
        .architecture = input->architecture,
        .components = std::move(published_components),
        .files = input->files,
        .expected = expected,
    });
    if (!comp_result) return Failure(err, comp_result.error());

    if (!Print(out, std::format("{} {}\n", comp_result->reused ? "verified" : "published",
                                comp_result->id))) {
      Print(err, "llmp import: composition published; cannot write result to standard output\n");
      return kExitFailure;
    }

    if (*archive) {
      auto opened = artifact::OpenComposition(comp_result->path);
      if (!opened) {
        Print(err,
              "llmp import: composition published, but it could not be reopened for catalog "
              "recording\n");
        return kExitFailure;
      }
      if (auto recorded = import::RecordArchiveImport(**archive, *input, *opened); !recorded) {
        Print(err, "llmp import: composition published; catalog event failed: " +
                       recorded.error().ToString() + '\n');
        return kExitFailure;
      }
    }
    return kExitOk;
  }
  auto bound = import::NormalizationBound(*input);
  if (!bound) return Failure(err, bound.error());
  if (!funded(*bound))
    return Failure(err, {.rule = import::Rule::kBounds,
                         .reason = "normalization exceeds the requested metadata budget"});
  auto plan = import::Normalize(std::move(*input), {bound->metadata_bytes, bound->work_units});
  if (!plan) return Failure(err, plan.error());
  return publish(*plan, plan->source(), bound->metadata_bytes);
}
}  // namespace llmp::cli
