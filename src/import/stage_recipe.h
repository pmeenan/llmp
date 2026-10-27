// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Stage recipes (spec/stage_recipe.h) applied at import: one visitor over a
// recipe's steps, run in four policies. Check walks every step and expression
// once, statically (shapes, recursion, lexical templates, request scalars);
// Collect gathers the environment names a recipe reads; Count runs the
// expansion over the actual source without building it, counting what it
// would allocate and compare; Expand builds the stage, charging each node what
// Count counted, against the caller's prepaid bound. Values are evaluated by
// the mappings' one evaluator (spec/expr.h) over one environment: the
// family's parameters, the source's config keys and tensors, and the bound
// repeat indices.

#ifndef LLMP_IMPORT_STAGE_RECIPE_H_
#define LLMP_IMPORT_STAGE_RECIPE_H_

#include <cstdint>
#include <expected>
#include <set>
#include <span>
#include <string>
#include <string_view>

#include "import/checkpoint.h"
#include "spec/stage_recipe.h"

namespace llmp::import {

struct StageRecipePreflight {
  std::uint64_t expanded_nodes = 0;
  std::uint64_t tensor_bindings = 0;
  // Expanded raw source-axis expressions materialized and compared against
  // checkpoint header dimensions.
  std::uint64_t raw_dimension_axes = 0;
  // Expanded tensor bindings with a raw source-dimension contract.
  std::uint64_t raw_dimension_contracts = 0;
  // The expanded source-name strings consulted while binding roles.
  std::uint64_t tensor_name_bytes = 0;
  std::uint64_t max_tensor_name_bytes = 0;
  std::uint64_t max_weight_role_bytes = 0;
  std::uint64_t max_source_key_bytes = 0;
  std::uint64_t visited_steps = 0;
  // The environment's value reads across counting and expansion (each a
  // scan of the source's metadata: normalization's comparison-work bound).
  std::uint64_t source_value_lookups = 0;
  // The expansion's own name comparisons (shared-tensor identity and
  // evaluated-weight lookups), in byte-comparison units.
  std::uint64_t expansion_work = 0;
  std::uint64_t metadata_bytes = 0;
  std::uint64_t stream_width = 0;
};

std::expected<StageRecipePreflight, std::string> PreflightStageRecipe(
    const spec::StageRecipe& recipe, const spec::ModelSpec& family, std::uint64_t metadata_bytes,
    std::span<const MetaEntry> source_values = {}, std::span<const Tensor> source_tensors = {},
    std::string_view source_container = {});

// Expand a mapping-owned finite recipe using the shared component catalog to
// derive each bound role's source shape. All expansion is import-time; only
// the resulting ordinary Stage is stored in an artifact. The source's tensors
// are its checkpoint's (their raw dimensions in `source_container`'s order:
// a GGUF's innermost first, any other's outermost first).
std::expected<spec::RecipeExpansion, std::string> ExpandStageRecipe(
    const spec::StageRecipe& recipe, const spec::ModelSpec& family, std::uint64_t metadata_bytes,
    std::span<const MetaEntry> source_values = {}, std::span<const Tensor> source_tensors = {},
    std::string_view source_container = {});

// A source metadata value as the value language holds it (spec/expr.h): a
// JSON null absent, a list of strings refused.
std::expected<spec::Datum, std::string> DatumOf(const MetaValue& value, std::string_view key);

// The environment names a recipe's expressions read (spec::ExprKind::kValue),
// each once; refused where its structure is invalid.
std::expected<std::set<std::string_view>, std::string> RecipeValueNames(
    const spec::StageRecipe& recipe);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_STAGE_RECIPE_H_
