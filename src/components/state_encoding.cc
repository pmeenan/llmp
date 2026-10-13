// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "components/state_encoding.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>

#include "components/component.h"
#include "components/step.h"

namespace llmp::components {
namespace {

// Two serial dense parameter copies (candidate + checked Params), each bounded
// by the catalog's scalar-only dense schema. No weight or model copy.
constexpr std::uint64_t kValidationBytes =
    2 * spec::kMaxParams * (sizeof(spec::Param) + 2 * (spec::kMaxText + 1)) + 4096;

void SetEncoding(spec::ComponentRef& component, std::string_view name) {
  const auto at = std::ranges::lower_bound(component.params, std::string_view("kv_encoding"), {},
                                           &spec::Param::name);
  if (at != component.params.end() && at->name == "kv_encoding") {
    at->value = std::string(name);
  } else {
    component.params.reserve(component.params.size() + 1);
    const auto insert = std::ranges::lower_bound(component.params, std::string_view("kv_encoding"),
                                                 {}, &spec::Param::name);
    component.params.insert(insert, {.name = "kv_encoding", .value = std::string(name)});
  }
}

// Include non-layer references: an unexpected stateful embedding or side
// source must not silently keep another format under an alias-wide setting.
template <typename F>
std::expected<void, std::string> Visit(const spec::Stage& stage, const F& f) {
  for (const auto* optional :
       {&stage.embedding, &stage.stream.mixing, &stage.heads.final_norm, &stage.heads.output})
    if (*optional)
      if (auto result = f(**optional, false); !result) return result;
  for (const auto& side : stage.side)
    if (auto result = f(side.source, false); !result) return result;
  for (const auto& layer : stage.layers)
    for (const auto& component : layer.components)
      if (auto result = f(component, true); !result) return result;
  return {};
}

}  // namespace

std::expected<std::uint64_t, std::string> KvCacheEncodingBytes(const spec::ModelSpec& model,
                                                               format::Encoding encoding) {
  if (encoding == format::Encoding::kF16) return 0;
  if (encoding != format::Encoding::kQ8_0 || model.pipeline != spec::PipelineKind::kLlm ||
      model.stages.size() != 1 || model.stages.front().role != spec::StageRole::kTarget ||
      !model.stages.front().reads.empty() || !model.stages.front().taps.empty())
    return std::unexpected("Q8_0 KV requires a single independent LLM target");
  const auto& stage = model.stages.front();
  if (stage.layers.size() > kMaxStageLayers || stage.stream.streams != 1 ||
      !stage.heads.mtp.empty())
    return std::unexpected("Q8_0 KV source is outside the dense stage bounds");
  std::uint64_t bytes = kValidationBytes;
  std::size_t count = 0;
  auto visited = Visit(
      stage,
      [&](const spec::ComponentRef& component, bool layer) -> std::expected<void, std::string> {
        if (component.kind == "attention.dense") {
          if (!layer || component.params.size() >= spec::kMaxParams ||
              ++count > spec::kMaxComponents)
            return std::unexpected("Q8_0 KV parameter inventory exceeds the catalog bound");
          // Current vector and its replacement can coexist during reserve; copied
          // scalar parameter strings are in the fixed serial validation allowance.
          const auto added = (component.params.size() + 1) * sizeof(spec::Param) + 512;
          if (__builtin_add_overflow(bytes, added, &bytes))
            return std::unexpected("Q8_0 KV parameter payment overflows");
        }
        return {};
      });
  if (!visited) return std::unexpected(visited.error());
  if (count == 0) return std::unexpected("Q8_0 KV source has no owned dense attention");
  return bytes;
}

std::expected<void, std::string> ApplyKvCacheEncoding(spec::ModelSpec& model,
                                                      format::Encoding encoding,
                                                      std::uint64_t metadata_bytes) {
  auto bytes = KvCacheEncodingBytes(model, encoding);
  if (!bytes) return std::unexpected(bytes.error());
  if (metadata_bytes < *bytes) return std::unexpected("Q8_0 KV metadata is not admitted");
  if (encoding == format::Encoding::kF16) {
    for (const auto& stage : model.stages) {
      auto checked = Visit(
          stage, [](const spec::ComponentRef& component, bool) -> std::expected<void, std::string> {
            const auto* p = component.Find("kv_encoding");
            if (p && p->value != spec::ParamValue(std::string("f16")))
              return std::unexpected("the source's KV encoding differs from the F16 setting");
            return {};
          });
      if (!checked) return checked;
    }
    return {};
  }
  const auto& stage = model.stages.front();
  const StageFacts facts{
      .width = stage.stream.width, .context = stage.context, .folds = stage.folds};
  auto checked = Visit(
      stage,
      [&](const spec::ComponentRef& component, bool layer) -> std::expected<void, std::string> {
        const auto* definition = Find(component.kind, component.version);
        if (!definition) return std::unexpected("KV encoding encountered an unknown component");
        if (component.kind != "attention.dense") {
          if (definition->state != nullptr || definition->inject != nullptr)
            return std::unexpected(
                "Q8_0 KV does not cover a borrowed or specialized state consumer");
          return {};
        }
        // The dense catalog has only scalar parameters. Reject list inputs before
        // copying so malformed values cannot exceed the scalar validation budget.
        std::string_view previous;
        for (const auto& p : component.params) {
          const auto* text = std::get_if<std::string>(&p.value);
          if (p.name.empty() || p.name.size() > spec::kMaxText ||
              (!previous.empty() && previous >= p.name) ||
              (text && text->size() > spec::kMaxText) ||
              std::holds_alternative<std::vector<std::int64_t>>(p.value) ||
              std::holds_alternative<std::vector<double>>(p.value))
            return std::unexpected("dense KV parameters must be bounded canonical scalars");
          previous = p.name;
        }
        spec::ComponentRef candidate{
            .kind = component.kind, .version = component.version, .params = component.params};
        SetEncoding(candidate, "q8_0");
        auto instance =
            MakeInstance(candidate, facts, layer ? Site::kLayer : Site::kEmbedding, 0, 0);
        if (!instance) return std::unexpected(instance.error());
        return {};
      });
  if (!checked) return checked;
  for (auto& layer : model.stages.front().layers)
    for (auto& component : layer.components)
      if (component.kind == "attention.dense") SetEncoding(component, "q8_0");
  return {};
}

}  // namespace llmp::components
