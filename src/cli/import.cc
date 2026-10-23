// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "cli/import.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base/build_info.h"
#include "base/json.h"
#include "cli/cli.h"
#include "import/archive_input.h"
#include "import/image_pipeline.h"
#include "import/image_reuse.h"
#include "import/normalize.h"
#include "import/reader.h"
#include "import/writer.h"
#include "spec/mapping.h"
#include "spec/names.h"
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
          "                   [--target TARGET_ID] [--component NAME [--draft-head IDS]]\n"
          "                   [--reuse-component ROLE=ARTIFACT_ID]...\n"
          "  --reuse-component explicitly reuse prepared image weights with matching\n"
          "                    spec, inventory and complete archive source identities;\n"
          "                    authenticates metadata, without rehashing payloads\n"
          "  --target ID       validate a drafter against this artifact in --store\n"
          "  --component NAME select a registered component mapping (mtp for a drafter,\n"
          "                    transformer or vae for a configured spatial component)\n"
          "  --draft-head IDS  give the MTP drafter a selected draft head: IDS lists one\n"
          "                    decimal token ID a line, strictly ascending; their rows\n"
          "                    of the checkpoint's lm_head.weight become its head\n");
    return kExitUsage;
  };
  if (args.empty() || args[0].empty() || args[0].starts_with('-')) return usage();
  std::filesystem::path source(args[0]), store, draft_head;
  std::string_view component, target_id;
  std::map<std::string_view, std::string_view, std::less<>> reuse;
  const auto artifact_id = [](std::string_view id) {
    return id.size() == 64 && std::ranges::all_of(id, [](char c) {
             return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
  };
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
    } else if (args[i] == "--component" && component.empty() && spec::names::IsName(args[i + 1]))
      component = args[i + 1];
    else if (args[i] == "--target" && target_id.empty())
      target_id = args[i + 1];
    else if (args[i] == "--reuse-component") {
      const auto split = args[i + 1].find('=');
      if (split == std::string_view::npos) return usage();
      const auto role = args[i + 1].substr(0, split);
      const auto id = args[i + 1].substr(split + 1);
      if (!spec::names::IsName(role) || !artifact_id(id) || !reuse.emplace(role, id).second)
        return usage();
    } else if (args[i] == "--draft-head" && draft_head.empty())
      draft_head = args[i + 1];
    else
      return usage();
  }
  if (store.empty() || (!draft_head.empty() && component != "mtp") ||
      (!reuse.empty() && (!component.empty() || !target_id.empty() || !draft_head.empty())))
    return usage();
  if (!component.empty() && component != "mtp" &&
      std::ranges::none_of(spec::Mappings(), [&](const auto* mapping) {
        return mapping->stage_recipe != nullptr && mapping->stage == component;
      }))
    return usage();
  if (!target_id.empty() && !artifact_id(target_id)) return usage();
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
  const auto resolve_mapping = [&](const import::Checkpoint& checkpoint)
      -> std::expected<std::optional<spec::ResolvedMapping>, import::Error> {
    const auto* canonical =
        import::FindImportMapping(checkpoint.container, checkpoint.architecture);
    if (!canonical) return std::optional<spec::ResolvedMapping>{};
    auto selected =
        spec::ResolveSourceMapping(*canonical, checkpoint.architecture, checkpoint.container);
    if (!selected)
      return std::unexpected(import::Error{.rule = import::Rule::kUnsupported,
                                           .reason = "checkpoint source dialect is unsupported"});
    return std::move(*selected);
  };
  const auto warn_skipped =
      [&](const import::Checkpoint& checkpoint) -> std::expected<void, import::Error> {
    for (const auto& skipped : checkpoint.skipped) warn(skipped.name, skipped.reason);
    for (const auto& skipped : checkpoint.skipped_tensors) warn(skipped.name, skipped.reason);
    auto selected = resolve_mapping(checkpoint);
    if (!selected) return std::unexpected(selected.error());
    if (!*selected) return {};
    for (const auto prefix : (**selected).mapping.ignored_tensor_prefixes)
      if (std::ranges::any_of(checkpoint.tensors,
                              [&](const auto& tensor) { return tensor.name.starts_with(prefix); }))
        warn(prefix, "component excluded by this text-only architecture mapping");
    return {};
  };
  if (auto warned = warn_skipped(*input); !warned) return Failure(err, warned.error());
  for (const auto& child : input->components)
    if (auto warned = warn_skipped(child); !warned) return Failure(err, warned.error());
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
  const auto publish = [&](const import::NormalizedPlan& plan, const import::Checkpoint& checkpoint,
                           std::uint64_t used) {
    auto selected = resolve_mapping(checkpoint);
    if (!selected) return Failure(err, selected.error());
    if (!*selected)
      return Failure(err, {.rule = import::Rule::kUnsupported,
                           .reason = "checkpoint has no text import mapping"});
    const auto& mapping = (**selected).mapping;
    for (const auto& use : plan.metadata()) {
      if (use.classification == import::MetadataClass::kAuxiliary &&
          plan.auxiliary_namespaces()[use.destination].prefix.empty())
        warn(checkpoint.metadata[use.source].key, "unread metadata retained as source provenance");
    }
    for (const auto& rule : mapping.lists) {
      const auto canonical = spec::CanonicalKey(mapping, rule.key);
      const auto* value = checkpoint.FindValue(canonical);
      const auto* labels = value ? std::get_if<std::vector<std::int64_t>>(value) : nullptr;
      if (rule.allow_unstated_extra && labels && labels->size() > plan.cost().layers)
        warn(canonical, "trailing unimported layer labels retained as source provenance");
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
    if (auto prebound = import::NormalizationBound(*input, {.component = component}); !prebound)
      return Failure(err, prebound.error());
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
    auto bound = import::NormalizationBound(
        *input, {.component = component, .draft_head_rows = head ? head->ids.size() : 0});
    if (!bound) return Failure(err, bound.error());
    if (!funded(*bound))
      return Failure(err, {.rule = import::Rule::kBounds,
                           .reason = "normalization exceeds the requested metadata budget"});
    auto plan =
        import::Normalize(*input, {bound->metadata_bytes, bound->work_units}, &*target,
                          {.component = std::string(component), .draft_head = std::move(head)});
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
  if (!component.empty()) {
    const import::Checkpoint* spatial_source = &*input;
    if (input->container == "diffusers") {
      const auto child = std::ranges::find(input->components, component, &import::Checkpoint::role);
      if (child == input->components.end())
        return Failure(err, {.rule = import::Rule::kConfig,
                             .reason = "--component names no source component role"});
      spatial_source = &*child;
    }
    const auto* mapping = import::FindImportMapping(spatial_source->container,
                                                    spatial_source->architecture, component);
    if (!mapping || mapping->stage_recipe == nullptr)
      return Failure(err,
                     {.rule = import::Rule::kUnsupported,
                      .reason = "no configured spatial component mapping matches --component"});
    if (target || !reuse.empty()) return usage();
    auto bound = import::NormalizationBound(*spatial_source, {.component = component});
    if (!bound) return Failure(err, bound.error());
    if (!funded(*bound))
      return Failure(err, {.rule = import::Rule::kBounds,
                           .reason = "normalization exceeds the requested metadata budget"});
    auto plan = import::Normalize(*spatial_source, {bound->metadata_bytes, bound->work_units},
                                  nullptr, {.component = std::string(component)});
    if (!plan) return Failure(err, plan.error());
    std::vector<std::optional<base::Sha256Digest>> component_expected;
    if (*archive) {
      auto identities = import::ArchiveFileIdentities(**archive, plan->source().files);
      if (!identities) return Failure(err, identities.error());
      component_expected = std::move(*identities);
    }
    auto result =
        import::WriteArtifact(*plan, {.store = store,
                                      .build = build,
                                      .expected = component_expected,
                                      .metadata_bytes = metadata - bound->metadata_bytes});
    if (!result) return Failure(err, result.error());
    return finish(*result, *spatial_source, false, true);
  }
  if (target && (input->container == "safetensors" || input->container == "mlx")) return usage();
  if (import::IsImagePipeline(*input)) {
    if (target) return usage();
    for (const auto& [role, id] : reuse) {
      (void)id;
      if (std::ranges::none_of(input->components,
                               [&](const auto& child) { return child.role == role; }))
        return Failure(err, {.rule = import::Rule::kConfig,
                             .reason = "--reuse-component names an unknown image component role"});
    }
    auto bound = import::ImagePipelineBound(*input);
    if (!bound) return Failure(err, bound.error());
    if (!funded(*bound))
      return Failure(err, {.rule = import::Rule::kBounds,
                           .reason = "normalization exceeds the requested metadata budget"});
    auto plan = import::PlanImagePipeline(*input);
    if (!plan) return Failure(err, plan.error());

    const auto& comp_files = plan->files.empty() ? input->files : plan->files;
    if (auto valid = import::ValidateCompositionFiles(comp_files); !valid)
      return Failure(err, valid.error());
    // Validate every explicitly selected artifact before publishing any component.
    // Archive receipts identify the source; this path deliberately does not
    // re-read unchanged source or prepared payload bytes.
    std::vector<std::vector<std::optional<base::Sha256Digest>>> component_expected(
        plan->components.size());
    for (std::size_t i = 0; i < plan->components.size(); ++i) {
      if (*archive) {
        auto identities =
            import::ArchiveFileIdentities(**archive, plan->components[i].checkpoint.files);
        if (!identities) return Failure(err, identities.error());
        component_expected[i] = std::move(*identities);
      }
    }
    for (const auto& [role, id] : reuse) {
      const auto at = std::ranges::find_if(
          plan->components, [&](const auto& planned) { return planned.checkpoint.role == role; });
      if (at == plan->components.end())
        return Failure(err, {.rule = import::Rule::kConfig,
                             .reason = "--reuse-component names an unknown image component role"});
      auto opened = artifact::Artifact::Open(store / id);
      if (!opened || opened->id() != id)
        return Failure(
            err, {.rule = import::Rule::kConfig,
                  .reason = "--reuse-component does not name a readable artifact in --store"});
      const auto i = static_cast<std::size_t>(at - plan->components.begin());
      auto valid = import::ValidateImageComponentReuse(*at, *opened, component_expected[i],
                                                       metadata - bound->metadata_bytes);
      if (!valid) return Failure(err, valid.error());
    }
    std::vector<artifact::CompositionComponent> published_components;
    published_components.reserve(plan->components.size());
    for (std::size_t i = 0; i < plan->components.size(); ++i) {
      const auto& comp_plan = plan->components[i];
      if (const auto selected = reuse.find(comp_plan.checkpoint.role); selected != reuse.end()) {
        published_components.push_back({.role = comp_plan.checkpoint.role,
                                        .artifact = std::string(selected->second),
                                        .architecture = comp_plan.checkpoint.architecture});
        continue;
      }
      auto result =
          import::WriteArtifact(comp_plan, {.store = store,
                                            .build = build,
                                            .expected = component_expected[i],
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
        .files = comp_files,
        .expected = expected,
        .spec = plan->family,
    });
    if (!comp_result) return Failure(err, comp_result.error());

    return finish(*comp_result, *input, true, false);
  }
  if (!reuse.empty()) return usage();
  if (const auto* mapping = import::FindImportMapping(input->container, input->architecture);
      mapping && mapping->stage_recipe != nullptr) {
    if (target) return usage();
    auto bound = import::NormalizationBound(*input);
    if (!bound) return Failure(err, bound.error());
    if (!funded(*bound))
      return Failure(err, {.rule = import::Rule::kBounds,
                           .reason = "normalization exceeds the requested metadata budget"});
    auto plan = import::Normalize(*input, {bound->metadata_bytes, bound->work_units});
    if (!plan) return Failure(err, plan.error());
    std::vector<std::optional<base::Sha256Digest>> component_expected;
    if (*archive) {
      auto identities = import::ArchiveFileIdentities(**archive, plan->source().files);
      if (!identities) return Failure(err, identities.error());
      component_expected = std::move(*identities);
    }
    auto result =
        import::WriteArtifact(*plan, {.store = store,
                                      .build = build,
                                      .expected = component_expected,
                                      .metadata_bytes = metadata - bound->metadata_bytes});
    if (!result) return Failure(err, result.error());
    return finish(*result, *input, false, true);
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
