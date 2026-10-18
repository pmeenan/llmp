// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "cli/import.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base/build_info.h"
#include "base/json.h"
#include "cli/cli.h"
#include "import/archive_input.h"
#include "import/image_pipeline.h"
#include "import/normalizers.h"
#include "import/reader.h"
#include "import/writer.h"
#include "spec/stored.h"

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
    Print(err,
          "Usage: llmp import SOURCE --store DIRECTORY [--metadata-mib N]\n"
          "                   [--target TARGET_ID] [--component mtp [--draft-head IDS]]\n"
          "  --target ID       validate a drafter against this artifact in --store\n"
          "  --component mtp   import the checkpoint's MTP drafter (Qwen3.8 Flash Next's\n"
          "                    ModelOpt checkpoint) instead of its target\n"
          "  --draft-head IDS  give the drafter a selected draft head: IDS lists one\n"
          "                    decimal token ID a line, strictly ascending; their rows\n"
          "                    of the checkpoint's lm_head.weight become its head\n");
    return kExitUsage;
  };
  if (args.empty() || args[0].empty() || args[0].starts_with('-')) return usage();
  std::filesystem::path source(args[0]), store, draft_head;
  std::string_view component, target_id;
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
    } else if (args[i] == "--component" && component.empty() && args[i + 1] == "mtp")
      component = args[i + 1];
    else if (args[i] == "--target" && target_id.empty())
      target_id = args[i + 1];
    else if (args[i] == "--draft-head" && draft_head.empty())
      draft_head = args[i + 1];
    else
      return usage();
  }
  if (store.empty() || (!draft_head.empty() && component != "mtp")) return usage();
  if (!target_id.empty() && (target_id.size() != 64 || !std::ranges::all_of(target_id, [](char c) {
                               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                             })))
    return usage();
  std::optional<import::NormalizationTarget> target;
  if (!target_id.empty()) {
    auto opened = artifact::Artifact::Open(store / target_id);
    if (!opened || opened->id() != target_id)
      return Failure(err, {.rule = import::Rule::kUnsupported,
                           .reason = "--target does not name a readable artifact in --store"});
    auto model = spec::StoredSpec(*opened);
    auto tensors = spec::SourceOf(*opened);
    if (!model || !tensors)
      return Failure(err, {.rule = import::Rule::kUnsupported,
                           .reason = "--target requires a valid stored version-1 target spec"});
    const auto* mapping = spec::FindMapping(tensors->architecture, tensors->container);
    target.emplace(import::NormalizationTarget{mapping, std::move(*model), std::move(*tensors)});
  }
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
  const auto& info = base::GetBuildInfo();
  const std::string build =
      info.commit.empty() ? "unknown" : std::string(info.commit) + (info.modified ? ".dirty" : "");
  const auto warn = [&](std::string_view name, std::string_view reason) {
    std::string quoted;
    base::json::AppendQuoted(name, quoted);
    Print(err, "llmp import: warning: " + quoted + ": " + std::string(reason) + '\n');
  };
  const auto warn_skipped = [&](const import::Checkpoint& checkpoint) {
    for (const auto& skipped : checkpoint.skipped) warn(skipped.name, skipped.reason);
    for (const auto& skipped : checkpoint.skipped_tensors) warn(skipped.name, skipped.reason);
    if (const auto* mapping =
            import::FindSourceMapping(checkpoint.container, checkpoint.architecture))
      for (const auto prefix : mapping->ignored_tensor_prefixes)
        if (std::ranges::any_of(checkpoint.tensors, [&](const auto& tensor) {
              return tensor.name.starts_with(prefix);
            }))
          warn(prefix, "component excluded by this text-only architecture mapping");
  };
  warn_skipped(*input);
  for (const auto& child : input->components) warn_skipped(child);
  const auto finish = [&](const import::WriteResult& result, const import::Checkpoint& checkpoint,
                          bool composition, bool drafter,
                          const std::optional<std::string>& list_name = std::nullopt) {
    if (!Print(out, std::format("{} {}\n", result.reused ? "verified" : "published", result.id))) {
      Print(err, "llmp import: published; cannot write result to standard output\n");
      return kExitFailure;
    }
    if (*archive) {
      std::expected<void, import::Error> recorded;
      if (composition) {
        auto opened = artifact::OpenComposition(result.path);
        if (!opened)
          return Failure(
              err, {.rule = import::Rule::kIo,
                    .reason = "published composition could not reopen for catalog recording"});
        recorded = import::RecordArchiveImport(**archive, checkpoint, *opened);
      } else {
        auto opened = artifact::Artifact::Open(result.path);
        if (!opened)
          return Failure(err,
                         {.rule = import::Rule::kIo,
                          .reason = "published artifact could not reopen for catalog recording"});
        recorded = drafter ? import::RecordArchiveComponentImport(**archive, checkpoint, *opened,
                                                                  list_name)
                           : import::RecordArchiveImport(**archive, checkpoint, *opened);
      }
      if (!recorded) {
        Print(err, "llmp import: published; catalog event failed: " + recorded.error().ToString() +
                       '\n');
        return kExitFailure;
      }
    }
    return kExitOk;
  };
  const auto publish = [&](const auto& plan, const import::Checkpoint& checkpoint,
                           std::uint64_t used) {
    if constexpr (requires { plan.metadata(); }) {
      for (const auto& use : plan.metadata()) {
        if (use.classification == import::MetadataClass::kAuxiliary &&
            plan.auxiliary_namespaces()[use.destination].prefix.empty())
          warn(checkpoint.metadata[use.source].key,
               "unread metadata retained as source provenance");
      }
    }
    if constexpr (requires { plan.metadata(); }) {
      if (const auto* table =
              import::FindSourceMapping(checkpoint.container, checkpoint.architecture))
        for (const auto& rule : table->lists) {
          const auto* value = checkpoint.FindValue(rule.key);
          const auto* labels = value ? std::get_if<std::vector<std::int64_t>>(value) : nullptr;
          if (rule.allow_unstated_extra && labels && labels->size() > plan.cost().layers)
            warn(rule.key, "trailing unimported layer labels retained as source provenance");
        }
    }
    auto result = import::WriteArtifact(
        plan,
        {.store = store, .build = build, .expected = expected, .metadata_bytes = metadata - used});
    if (!result) return Failure(err, result.error());
    return finish(*result, checkpoint, false, false);
  };
  const auto funded = [&](const import::NormalizationCost& bound) {
    return bound.metadata_bytes <= metadata;
  };
  if (component == "mtp") {
    const auto* normalizer = import::FindNormalizer(*input);
    if (!normalizer || !normalizer->drafter || !normalizer->drafter_bound)
      return Failure(err, {.rule = import::Rule::kUnsupported,
                           .reason = "no registered normalizer recognizes this drafter component"});
    if (!target)
      return Failure(err, {.rule = import::Rule::kUnsupported,
                           .reason = "drafter import requires --target TARGET_ID in --store"});
    auto vocabulary = spec::Vocabulary(target->model, target->source);
    if (!vocabulary)
      return Failure(err, {.rule = import::Rule::kUnsupported,
                           .reason = "drafter target vocabulary lacks a valid output head"});
    std::optional<import::DraftHeadList> head;
    std::optional<std::string> list_name;  // the caller's own source
    if (!draft_head.empty()) {
      auto list = import::ReadDraftHeadList(draft_head, *vocabulary);
      if (!list) return Failure(err, list.error());
      list_name = list->file.name;
      head = std::move(*list);
    }
    auto bound = normalizer->drafter_bound(*input, head ? head->ids.size() : 0);
    if (!bound) return Failure(err, bound.error());
    if (!funded(*bound))
      return Failure(err, {.rule = import::Rule::kBounds,
                           .reason = "normalization exceeds the requested metadata budget"});
    auto plan = normalizer->drafter(*input, std::move(head),
                                    {bound->metadata_bytes, bound->work_units}, &*target);
    if (!plan) return Failure(err, plan.error());
    // The archive's identities of the files it reads (the head list is
    // the caller's, not the archive's).
    std::vector<std::optional<base::Sha256Digest>> drafter_expected;
    if (*archive) {
      for (const auto& file : plan->parts().source().files) {
        const auto at = std::ranges::find(input->files, file.name, &import::SourceFile::name);
        drafter_expected.push_back(
            at == input->files.end()
                ? std::nullopt
                : expected[static_cast<std::size_t>(at - input->files.begin())]);
      }
    }
    auto result = import::WriteArtifact(*plan, {.store = store,
                                                .build = build,
                                                .expected = drafter_expected,
                                                .metadata_bytes = metadata - bound->metadata_bytes,
                                                .options = plan->options()});
    if (!result) return Failure(err, result.error());
    return finish(*result, *input, false, true, list_name);
  }
  if (import::IsIndexedTarget(*input)) {
    if (target) return usage();
    auto bound = import::IndexedTargetBound(*input);
    if (!bound) return Failure(err, bound.error());
    if (!funded(*bound))
      return Failure(err, {.rule = import::Rule::kBounds,
                           .reason = "normalization exceeds the requested metadata budget"});
    auto plan = import::NormalizeIndexedTarget(std::move(*input),
                                               {bound->metadata_bytes, bound->work_units});
    if (!plan) return Failure(err, plan.error());
    return publish(*plan, plan->parts().source(), bound->metadata_bytes);
  }
  if (import::IsImagePipeline(*input)) {
    if (target) return usage();
    auto bound = import::ImagePipelineBound(*input);
    if (!bound) return Failure(err, bound.error());
    if (!funded(*bound))
      return Failure(err, {.rule = import::Rule::kBounds,
                           .reason = "normalization exceeds the requested metadata budget"});
    auto plan = import::PlanImagePipeline(*input);
    if (!plan) return Failure(err, plan.error());

    if (auto valid = import::ValidateCompositionFiles(input->files); !valid)
      return Failure(err, valid.error());
    std::vector<artifact::CompositionComponent> published_components;
    published_components.reserve(plan->components.size());

    for (std::size_t i = 0; i < plan->components.size(); ++i) {
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
        .spec = plan->family,
    });
    if (!comp_result) return Failure(err, comp_result.error());

    return finish(*comp_result, *input, true, false);
  }
  auto bound = import::NormalizationBound(*input);
  if (!bound) {
    if (bound.error().kind == import::ItemKind::kTensor &&
        bound.error().item < input->tensors.size())
      warn(input->tensors[bound.error().item].name, bound.error().reason);
    if (bound.error().kind == import::ItemKind::kKey && bound.error().item < input->metadata.size())
      warn(input->metadata[bound.error().item].key, bound.error().reason);
    return Failure(err, bound.error());
  }
  if (!funded(*bound))
    return Failure(err, {.rule = import::Rule::kBounds,
                         .reason = "normalization exceeds the requested metadata budget"});
  auto plan = import::Normalize(std::move(*input), {bound->metadata_bytes, bound->work_units},
                                target ? &*target : nullptr);
  if (!plan) return Failure(err, plan.error());
  return publish(*plan, plan->source(), bound->metadata_bytes);
}
}  // namespace llmp::cli
