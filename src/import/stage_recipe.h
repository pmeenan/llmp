// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_IMPORT_STAGE_RECIPE_H_
#define LLMP_IMPORT_STAGE_RECIPE_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>

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
  // Bounds on the expanded source-name strings consulted while binding roles.
  std::uint64_t tensor_name_bytes = 0;
  std::uint64_t max_tensor_name_bytes = 0;
  std::uint64_t max_weight_role_bytes = 0;
  std::uint64_t max_source_key_bytes = 0;
  std::uint64_t visited_steps = 0;
  // Conservative number of typed source-metadata lookups across validation,
  // preflight and expansion (used by normalization's comparison-work bound).
  std::uint64_t source_value_lookups = 0;
  std::uint64_t metadata_bytes = 0;
  std::uint64_t stream_width = 0;
};

std::expected<StageRecipePreflight, std::string> PreflightStageRecipe(
    const spec::StageRecipe& recipe, const spec::ModelSpec& family, std::uint64_t metadata_bytes,
    std::span<const MetaEntry> source_values = {});

// Expand a mapping-owned finite recipe using the shared component catalog to
// derive each bound role's source shape. All expansion is import-time; only
// the resulting ordinary Stage is stored in an artifact.
std::expected<spec::RecipeExpansion, std::string> ExpandStageRecipe(
    const spec::StageRecipe& recipe, const spec::ModelSpec& family, std::uint64_t metadata_bytes,
    std::span<const MetaEntry> source_values = {});

}  // namespace llmp::import

#endif  // LLMP_IMPORT_STAGE_RECIPE_H_
