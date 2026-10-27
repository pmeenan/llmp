// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_SPEC_STAGE_RECIPE_H_
#define LLMP_SPEC_STAGE_RECIPE_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "spec/expr.h"
#include "spec/spec.h"

namespace llmp::spec {

// Import-time, mapping-owned instructions for expanding one stage into the
// ordinary stored graph. Recipes are static data, never serialized or run by
// serving. A recipe's values are expressions of the mappings' one value
// language (spec/expr.h): typed references to the environment's values (the
// source's config keys and, in a pipeline, its family's parameters), bounded
// indices, and checked arithmetic.
using RecipeValue = Expr;
using RecipeValueKind = ExprKind;

// Import-only, mapping-owned geometry contracts for spatial graph ports.
// These expressions are evaluated before Stage publication and never stored.
struct RecipeSpatialShape {
  RecipeValue width{}, height{}, channels{}, images{};
};
struct RecipeSpatialOperand {
  RecipeSpatialShape shape{};
  format::Encoding encoding = format::Encoding::kF32;
};

struct RecipeParam {
  std::string_view name{};
  RecipeValue value{};
  // A checked request integer binding. Non-empty source emits a
  // StageScalarBinding for the expanded component using the value as default.
  std::string_view request_scalar{};
  // A parameter only some sources' stages have: emitted where this flag holds.
  const RecipeValue* when = nullptr;
};

struct RecipeWeight {
  std::string_view role{};
  std::string_view tensor{};  // substitutions use `{index-name}`
  format::PartRole part = format::PartRole::kData;
  bool experts = false;
  std::string_view stage{};  // empty means this recipe's stage
  // Optional exact dimensions in source-container order. This import-only
  // assertion preserves raw header semantics alongside the folded role shape;
  // with `source_dims_when`, only where that flag holds.
  std::span<const RecipeValue> source_dims{};
  const RecipeValue* source_dims_when = nullptr;
};

enum class RecipeGroupKind : std::uint8_t { kWhole, kTable, kLayer };

enum class RecipeRefKind : std::uint8_t { kNamedValue, kStageInput, kHostPort };
struct RecipeRef {
  RecipeRefKind kind = RecipeRefKind::kNamedValue;
  std::string_view name{};
  std::uint16_t port = 0;
};

struct RecipeNode {
  std::string_view name{};  // stable symbolic output name, may be templated
  std::string_view component_kind{};
  std::uint16_t version = 1;
  std::span<const RecipeParam> params{};
  std::span<const RecipeWeight> weights{};
  RecipeRef input{};
  RecipeRef auxiliary{};
  RecipeRef carry{};
  bool has_auxiliary = false;
  bool has_carry = false;
  RecipeGroupKind group = RecipeGroupKind::kWhole;
  RecipeValue group_index{};
};

struct RecipeStep;
struct RecipeRepeat {
  std::string_view index{};
  RecipeValue count{};  // non-negative integer, checked before expansion
  std::span<const RecipeStep> body{};
  RecipeRef input{};
  std::string_view input_name{};
  std::string_view body_yield{};
  std::string_view yield{};
};

struct RecipeIf {
  RecipeValue condition{};  // bool, checked at import
  std::span<const RecipeStep> then_steps{};
  std::span<const RecipeStep> else_steps{};
  RecipeRef input{};
  std::string_view then_yield{};
  std::string_view else_yield{};
  std::string_view yield{};
  std::string_view input_name{};
};

struct RecipeStep {
  enum class Kind : std::uint8_t { kNode, kRepeat, kIf };
  Kind kind = Kind::kNode;
  const RecipeNode* node = nullptr;
  const RecipeRepeat* repeat = nullptr;
  const RecipeIf* branch = nullptr;
};

struct StageRecipe {
  std::string_view name{};
  StageRole role = StageRole::kVae;
  std::uint32_t artifact = 0;
  RecipeValue stream_width{};
  RecipeSpatialOperand primary_input{};
  std::span<const RecipeSpatialOperand> host_ports{};
  std::span<const RecipeStep> steps{};
  std::span<const RecipeRef> exports{};
  // If exports are empty, this describes the final component output.
  std::span<const RecipeSpatialShape> expected_outputs{};
};

struct RecipeExpansion {
  Stage stage;
  // Expected tensors and their role-derived shapes, in first-use order.
  // A physical source tensor may be consumed by several graph components;
  // import accounting deduplicates those consumers by source identity.
  struct Tensor {
    std::string name;
    std::vector<std::uint64_t> shape;
    RecipeGroupKind group = RecipeGroupKind::kWhole;
    std::uint32_t layer = 0;
    format::PartRole part = format::PartRole::kData;
    std::vector<std::uint64_t> source_dims;
    bool experts = false;
    std::string stage;
  };
  std::vector<Tensor> tensors;
};

}  // namespace llmp::spec

#endif  // LLMP_SPEC_STAGE_RECIPE_H_
