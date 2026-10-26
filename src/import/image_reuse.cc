// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/image_reuse.h"

#include <algorithm>
#include <map>
#include <set>
#include <string_view>

#include "spec/stored.h"

namespace llmp::import {
namespace {
std::unexpected<Error> Refuse(std::string_view reason, Rule rule = Rule::kConfig) {
  return std::unexpected(Error{.rule = rule, .reason = reason});
}
}  // namespace
std::expected<void, Error> ValidateImageComponentReuse(
    const ComponentPlan& plan, const artifact::Artifact& artifact,
    std::span<const std::optional<base::Sha256Digest>> expected, std::uint64_t metadata_bytes) {
  const auto& source = plan.checkpoint;
  if (artifact.format_version() != 1 || source.role.empty() || source.files.empty() ||
      expected.size() != source.files.size() || artifact.sources().size() != source.files.size() ||
      std::ranges::any_of(expected, [](const auto& digest) { return !digest; }))
    return Refuse("image component reuse requires complete declared source identities");
  if (artifact.model().architecture != source.architecture ||
      artifact.model().container != source.container ||
      plan.canonical.architecture != source.architecture ||
      plan.canonical.container != source.container ||
      plan.spec.pipeline != spec::PipelineKind::kImage)
    return Refuse("reused image component architecture or container differs from its plan");
  const auto text = artifact.spec_json();
  if (!text) return Refuse("reused image component has no stored spec");
  // The parser's nodes/indices and strings are bounded by document bytes.
  // Object records, normalized tensor names and lookup nodes are separate;
  // all are priced before StoredSpec/SourceOf or copied lookup inventories.
  std::uint64_t objects = 0;
  bool quoted = false;
  bool escaped = false;
  for (const char c : *text) {
    if (quoted) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        quoted = false;
      }
      continue;
    }
    if (c == '"')
      quoted = true;
    else if (c == '{')
      ++objects;
  }
  const auto record = std::max({sizeof(spec::Stage), sizeof(spec::ComponentRef),
                                sizeof(spec::SideInput), sizeof(spec::StageScalarBinding)});
  std::uint64_t bound = 4096;
  const auto add = [&](std::uint64_t count, std::uint64_t unit) {
    std::uint64_t bytes = 0;
    return !__builtin_mul_overflow(count, unit, &bytes) &&
           !__builtin_add_overflow(bound, bytes, &bound) && bound <= metadata_bytes;
  };
  if (!add(text->size(), 128) || !add(objects, 8 * record) || !add(source.files.size(), 4096) ||
      !add(artifact.resources().size(), 4096) || !add(artifact.expert_arrays().size(), 4096))
    return Refuse("image component reuse metadata is not funded", Rule::kBounds);
  for (const auto& resource : artifact.resources())
    if (!add(resource.roles.size(), 4096))
      return Refuse("image component reuse tensor aliases are not funded", Rule::kBounds);
  std::set<std::string_view> names;
  for (std::size_t i = 0; i < source.files.size(); ++i) {
    const auto& file = source.files[i];
    if (!names.insert(file.name).second)
      return Refuse("image component reuse source names are duplicated");
    const auto prior = std::ranges::find(artifact.sources(), file.name, &artifact::Source::name);
    if (prior == artifact.sources().end() || prior->bytes.value() != file.bytes ||
        prior->sha256 != *expected[i])
      return Refuse("reused image component source receipt differs from the declared source");
  }
  names.clear();
  for (const auto& prior : artifact.sources())
    if (!names.insert(prior.name).second)
      return Refuse("reused image component source receipts are duplicated");
  auto stored = spec::StoredSpec(artifact);
  if (!stored || *stored != plan.spec)
    return Refuse("reused image component stored spec differs from the planned configuration");
  auto tensors = spec::SourceOf(artifact);
  if (!tensors || tensors->folds != plan.canonical.folds ||
      tensors->tensors.size() != plan.canonical.tensors.size())
    return Refuse("reused image component tensor inventory or folds differ from the plan");
  std::map<std::string_view, const spec::SourceTensor*, std::less<>> wanted;
  for (const auto& tensor : plan.canonical.tensors)
    if (!wanted.emplace(tensor.name, &tensor).second)
      return Refuse("image component reuse planned tensor names are duplicated");
  for (const auto& tensor : tensors->tensors) {
    const auto at = wanted.find(tensor.name);
    if (at == wanted.end() || tensor.format != at->second->format ||
        tensor.part != at->second->part || tensor.experts != at->second->experts)
      return Refuse("reused image component tensor format or role differs from the plan");
    wanted.erase(at);
  }
  if (!wanted.empty()) return Refuse("reused image component is missing a planned tensor");
  return {};
}
}  // namespace llmp::import
