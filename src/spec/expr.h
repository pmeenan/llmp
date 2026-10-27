// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The mappings' one value language (D-109; docs/engine-components.md#the-value-language):
// one expression AST, one typed environment and one evaluator for every value
// a mapping computes at import. An LLM mapping's parameter rules
// (mapping.h ParamRule) lower into it (LowerParamRule, mapping.cc), an image
// pipeline's family parameters (image_family.h ImageParamRule) are written in
// it, and a stage recipe's values (stage_recipe.h) are it. Nothing here is
// stored or run by serving: an expression is static mapping data, evaluated
// at import against the source it is applied to.
//
// The environment is one namespace of named values: a source's metadata keys
// (a GGUF's, a flattened config.json's), the family parameters of an image
// pipeline's family spec ("<kind>.<param>": "image.vae_decoder.base"), the
// bound indices (a recipe repeat's, the layer a parameter is read for and the
// stage's layer count, "layer" and "layers"), the source's tensors (their raw
// header dimensions) and the folds the source carries. A name the environment
// does not hold is absent: kFirst falls through to its alternative, kPresent
// says so, an LLM rule resolves nothing and its parameter's next rule is
// tried, and a strict evaluation (a recipe's) refuses it. A JSON null is
// absent.
//
// Every operation is checked: integer arithmetic refuses overflow, divisions
// their domains, reals what is not finite where the operation says so, list
// accesses indices outside their list, and nesting is bounded (kMaxDepth).

#ifndef LLMP_SPEC_EXPR_H_
#define LLMP_SPEC_EXPR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "spec/spec.h"

namespace llmp::spec {

enum class ExprKind : std::uint8_t {
  // Leaves.
  kLiteral,        // `literal`
  kText,           // the string `name` (borrowed: a literal string without an allocation)
  kTemplate,       // `literal` (a string), each "{index}" the environment's index
  kAbsent,         // no value
  kValue,          // the environment's value `name`
  kIndex,          // the environment's index `name`
  kTensorPresent,  // whether source tensor `name` (a template) is present
  kTensorRank,     // its raw rank (header dimensions); absent without it
  kTensorDim,      // its raw dimension `fixed_index` (counted from the last when negative)
  kFold,           // whether the source carries fold `name`
  // Lists: `left` a list. An index is `index_name`'s plus `fixed_index`, or
  // without `index_name`, `fixed_index`.
  kElement,
  kElementReverse,  // counted from the last entry
  kSize,
  kReverse,
  // `left` one value a layer (index "layer" of "layers" entries) or one value
  // for every layer (llama.cpp's get_key_or_arr).
  kPerLayer,
  // `left` a list of integers; one of exactly `fixed_index` entries gains
  // `literal` (an integer) as its last.
  kPadTo,
  // Integers, checked.
  kAdd,
  kSubtract,
  kMultiply,
  kDivideExact,
  kDivideFloor,
  kLess,
  // Numbers (integers or reals).
  kRealDivide,    // a real; a zero divisor or a result not finite refused
  kRealMultiply,  // a real
  kQuotient,      // an integer: an exact quotient of integers, or an integral real one
  kProduct,       // an integer: a checked product of integers, or an integral real one
  kSqrt,          // reals: a result not finite refused
  kRsqrt,
  kReciprocal,
  kFloat32,  // the real rounded to float32 (one it cannot hold refused)
  kFinite,   // the real, finite, a negative zero as zero
  // Comparisons and logic.
  kEqual,  // the same type and value
  kNotEqual,
  kAnd,
  kOr,
  kNot,
  kSelect,   // `left` a flag: `right` where it holds, else `alternate`
  kFirst,    // `left` where present, else `right`
  kPresent,  // whether `left` is present
  // Typed reads of `left`.
  kAsInt,
  kAsReal,  // an integer read as a real
  kAsBool,
  kAsName,  // a string that is a name (names.h)
  kAsText,  // any string
  kAsInts,
  // A source string selecting a flag: `left` a string; `name` gives the
  // flag `literal`; the strings `names` mean another rule decides (absent);
  // any other is refused.
  kMatchText,
  // A flag, or a string alias for it: `left` a flag, or the string `name`
  // giving `literal` (another string refused).
  kFlagText,
};

struct Expr {
  ExprKind kind = ExprKind::kLiteral;
  ParamValue literal = std::int64_t{};
  std::string_view name{};
  std::string_view index_name{};
  std::int64_t fixed_index = 0;
  const Expr* left = nullptr;
  const Expr* right = nullptr;
  const Expr* alternate = nullptr;
  std::span<const std::string_view> names{};
};

inline constexpr std::size_t kMaxExprDepth = kMaxParams;

// A value the environment holds, borrowed: absent (monostate), an integer, a
// real, a flag, a string, or a list of integers or reals.
using Datum = std::variant<std::monostate, std::int64_t, double, bool, std::string_view,
                           std::span<const std::int64_t>, std::span<const double>>;

// A source tensor's raw (header) dimensions.
struct TensorDims {
  std::array<std::uint64_t, 8> dims{};
  std::uint8_t rank = 0;
};

class Env {
 public:
  virtual ~Env() = default;
  // The named value; monostate where absent; an error where the name is
  // ambiguous or its value of a type the language does not hold.
  virtual std::expected<Datum, std::string> Value(std::string_view name) const = 0;
  virtual std::optional<std::uint64_t> Index(std::string_view name) const;
  virtual std::expected<std::optional<TensorDims>, std::string> Tensor(std::string_view name) const;
  virtual bool Fold(std::string_view name) const;
};

// A whole evaluation: strict refuses an absent value where one is read
// (kFirst's and kPresent's operands excepted). `lookups` counts the
// environment's value reads (a work bound).
struct EvalOptions {
  bool strict = true;
  std::uint64_t* lookups = nullptr;
};

std::expected<std::optional<ParamValue>, std::string> Evaluate(const Expr& expr, const Env& env,
                                                               EvalOptions options = {});
// A strict evaluation's value, an integer; absent, another type or outside
// [least, most] refused with `what`.
std::expected<std::int64_t, std::string> EvaluateInt(const Expr& expr, const Env& env,
                                                     std::string_view what, std::int64_t least,
                                                     std::int64_t most, EvalOptions options = {});
// A flag, absent false where not strict.
std::expected<bool, std::string> EvaluateFlag(const Expr& expr, const Env& env,
                                              EvalOptions options = {});

// The bytes of string and list payload an evaluation may hold (Select's
// largest branch), from literals and the environment's values, without
// evaluating it; an absent value holds none.
std::expected<std::uint64_t, std::string> PayloadBound(const Expr& expr, const Env& env);

// Every node of `expr` in pre-order; a callback's error stops the walk. The
// walk refuses a cycle and nesting deeper than kMaxExprDepth.
std::expected<void, std::string> VisitExpr(
    const Expr& expr, const std::function<std::expected<void, std::string>(const Expr&)>& visit);
// Its shape: each kind's operands present and the others empty, names given
// where a kind reads one, templates closed; no cycle; bounded depth.
std::expected<void, std::string> CheckExpr(const Expr& expr);
// Whether it reads the environment's values, indices, tensors or folds.
bool ReadsEnvironment(const Expr& expr);

// Substitutes each "{index}" of `pattern` with the environment's index.
std::expected<std::string, std::string> SubstituteIndices(std::string_view pattern, const Env& env);

// The view of an owned value.
Datum View(const ParamValue& value);

// Building expressions in mapping tables (each operand a named static Expr).
namespace ex {
constexpr Expr I(std::int64_t v) { return {.literal = v}; }
constexpr Expr R(double v) { return {.literal = v}; }
constexpr Expr B(bool v) { return {.literal = v}; }
constexpr Expr S(std::string_view v) { return {.kind = ExprKind::kText, .name = v}; }
constexpr Expr Absent() {
  Expr absent;
  absent.kind = ExprKind::kAbsent;
  return absent;
}
constexpr Expr V(std::string_view name) { return {.kind = ExprKind::kValue, .name = name}; }
constexpr Expr Idx(std::string_view name) { return {.kind = ExprKind::kIndex, .name = name}; }
constexpr Expr TensorRank(std::string_view tensor) {
  return {.kind = ExprKind::kTensorRank, .name = tensor};
}
constexpr Expr TensorDim(std::string_view tensor, std::int64_t axis) {
  return {.kind = ExprKind::kTensorDim, .name = tensor, .fixed_index = axis};
}
constexpr Expr At(const Expr& list, std::int64_t index) {
  return {.kind = ExprKind::kElement, .fixed_index = index, .left = &list};
}
constexpr Expr At(const Expr& list, std::string_view index, std::int64_t offset = 0) {
  return {.kind = ExprKind::kElement, .index_name = index, .fixed_index = offset, .left = &list};
}
constexpr Expr AtBack(const Expr& list, std::int64_t index) {
  return {.kind = ExprKind::kElementReverse, .fixed_index = index, .left = &list};
}
constexpr Expr AtBack(const Expr& list, std::string_view index, std::int64_t offset = 0) {
  return {
      .kind = ExprKind::kElementReverse, .index_name = index, .fixed_index = offset, .left = &list};
}
constexpr Expr Size(const Expr& list) { return {.kind = ExprKind::kSize, .left = &list}; }
constexpr Expr Reverse(const Expr& list) { return {.kind = ExprKind::kReverse, .left = &list}; }
constexpr Expr E(ExprKind kind, const Expr& left, const Expr& right) {
  return {.kind = kind, .left = &left, .right = &right};
}
constexpr Expr U(ExprKind kind, const Expr& operand) { return {.kind = kind, .left = &operand}; }
constexpr Expr Select(const Expr& condition, const Expr& yes, const Expr& no) {
  return {.kind = ExprKind::kSelect, .left = &condition, .right = &yes, .alternate = &no};
}
constexpr Expr First(const Expr& value, const Expr& otherwise) {
  return {.kind = ExprKind::kFirst, .left = &value, .right = &otherwise};
}
constexpr Expr Present(const Expr& value) { return {.kind = ExprKind::kPresent, .left = &value}; }
}  // namespace ex

}  // namespace llmp::spec

#endif  // LLMP_SPEC_EXPR_H_
