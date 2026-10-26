// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/stage_recipe.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "components/spatial.h"
#include "components/step.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::import {
namespace {

using namespace llmp::spec;

// Index names belong to the mapping-owned recipe AST for the full expansion
// call. Borrow those names in a flat stack so preflight and repeat iterations
// never allocate per-iteration map nodes.
struct Indices {
  using Entry = std::pair<std::string_view, std::uint64_t>;
  std::vector<Entry> entries{};

  const Entry* begin() const { return entries.data(); }
  const Entry* end() const { return entries.data() + entries.size(); }
  Entry* begin() { return entries.data(); }
  Entry* end() { return entries.data() + entries.size(); }

  const Entry* find(std::string_view key) const {
    for (const auto& entry : entries) {
      if (entry.first == key) return &entry;
    }
    return end();
  }
  Entry* find(std::string_view key) {
    for (auto& entry : entries) {
      if (entry.first == key) return &entry;
    }
    return end();
  }
  std::uint64_t& operator[](std::string_view key) {
    if (auto* it = find(key); it != end()) return it->second;
    entries.emplace_back(key, 0);
    return entries.back().second;
  }
  void erase(std::string_view key) {
    for (std::size_t i = 0; i < entries.size(); ++i) {
      if (entries[i].first == key) {
        entries[i] = entries.back();
        entries.pop_back();
        return;
      }
    }
  }
  void clear() { entries.clear(); }
};
struct BoundValue {
  std::uint16_t id = 0;
  std::optional<std::uint16_t> host_port{};
};
using Values = std::map<std::string, BoundValue, std::less<>>;

std::unexpected<std::string> Bad(std::string why) {
  return std::unexpected("stage recipe: " + std::move(why));
}

std::expected<const ParamValue*, std::string> FamilyValue(const ModelSpec& family,
                                                          std::string_view kind,
                                                          std::string_view param) {
  const ParamValue* result = nullptr;
  for (const auto& stage : family.stages)
    for (const auto& layer : stage.layers)
      for (const auto& component : layer.components)
        if (component.kind == kind)
          if (const auto* value = component.Find(param)) {
            if (result) return std::unexpected("family recipe reference is ambiguous");
            result = &value->value;
          }
  if (!result) return std::unexpected("family recipe reference is absent");
  return result;
}

std::expected<const MetaValue*, std::string> SourceValueOf(std::span<const MetaEntry> values,
                                                           std::string_view key) {
  const MetaValue* result = nullptr;
  for (const auto& value : values) {
    if (value.key != key) continue;
    if (result) return std::unexpected("source recipe key is ambiguous");
    result = &value.value;
  }
  if (!result) return std::unexpected("source recipe key is absent");
  return result;
}

std::expected<ParamValue, std::string> Own(const MetaValue& value) {
  if (const auto* item = std::get_if<std::int64_t>(&value)) return ParamValue(*item);
  if (const auto* item = std::get_if<double>(&value)) return ParamValue(*item);
  if (const auto* item = std::get_if<bool>(&value)) return ParamValue(*item);
  return std::unexpected("source recipe scalar must be an integer, real or boolean");
}

std::expected<std::int64_t, std::string> Integer(const ParamValue& value) {
  if (const auto* result = std::get_if<std::int64_t>(&value)) return *result;
  return std::unexpected("expected an integer recipe value");
}

std::expected<bool, std::string> Boolean(const ParamValue& value) {
  if (const auto* result = std::get_if<bool>(&value)) return *result;
  return std::unexpected("expected a boolean recipe value");
}

std::expected<ParamValue, std::string> Eval(const RecipeValue& expr, const ModelSpec& family,
                                            std::span<const MetaEntry> source_values,
                                            const Indices& indices, std::size_t depth = 0) {
  if (depth > kMaxParams) return std::unexpected("recipe expression exceeds nesting bounds");
  switch (expr.kind) {
    case RecipeValueKind::kLiteral:
      return expr.literal;
    case RecipeValueKind::kFamilyValue: {
      auto value = FamilyValue(family, expr.family_kind, expr.family_param);
      if (!value) return std::unexpected(value.error());
      return **value;
    }
    case RecipeValueKind::kFamilyF32Value: {
      auto value = FamilyValue(family, expr.family_kind, expr.family_param);
      if (!value) return std::unexpected(value.error());
      const auto* real = std::get_if<double>(*value);
      if (!real || !std::isfinite(*real) || std::abs(*real) > std::numeric_limits<float>::max())
        return std::unexpected("family F32 recipe value is not representable");
      return ParamValue(static_cast<double>(static_cast<float>(*real)));
    }
    case RecipeValueKind::kSourceValue: {
      auto value = SourceValueOf(source_values, expr.source_key);
      if (!value) return std::unexpected(value.error());
      return Own(**value);
    }
    case RecipeValueKind::kSourceListElement:
    case RecipeValueKind::kSourceListElementReverse:
    case RecipeValueKind::kSourceListSize: {
      auto value = SourceValueOf(source_values, expr.source_key);
      const auto* list = value ? std::get_if<std::vector<std::int64_t>>(*value) : nullptr;
      if (!list) return std::unexpected("source recipe list is absent or has the wrong type");
      if (expr.kind == RecipeValueKind::kSourceListSize)
        return ParamValue(static_cast<std::int64_t>(list->size()));
      std::uint64_t index = 0;
      if (expr.index_name.empty()) {
        if (expr.fixed_index < 0) return std::unexpected("source list index is negative");
        index = static_cast<std::uint64_t>(expr.fixed_index);
      } else {
        const auto it = indices.find(expr.index_name);
        if (it == indices.end()) return std::unexpected("source list index is unbound");
        if (expr.fixed_index < 0) {
          const auto magnitude = static_cast<std::uint64_t>(-(expr.fixed_index + 1)) + 1;
          if (magnitude > it->second) return std::unexpected("source list index offset underflows");
          index = it->second - magnitude;
        } else if (__builtin_add_overflow(it->second, static_cast<std::uint64_t>(expr.fixed_index),
                                          &index)) {
          return std::unexpected("source list index offset overflows");
        }
      }
      if (index >= list->size())
        return std::unexpected("source list index is outside its metadata list");
      const auto at =
          expr.kind == RecipeValueKind::kSourceListElement ? index : list->size() - index - 1;
      return ParamValue((*list)[at]);
    }
    case RecipeValueKind::kIndex: {
      const auto it = indices.find(expr.index_name);
      if (it == indices.end() || it->second > static_cast<std::uint64_t>(INT64_MAX))
        return std::unexpected("recipe index is absent or too large");
      return ParamValue(static_cast<std::int64_t>(it->second));
    }
    case RecipeValueKind::kListElement:
    case RecipeValueKind::kListElementReverse:
    case RecipeValueKind::kListSize: {
      auto value = FamilyValue(family, expr.family_kind, expr.family_param);
      const auto* list = value ? std::get_if<std::vector<std::int64_t>>(*value) : nullptr;
      if (!list) return std::unexpected("recipe list value is absent or has the wrong type");
      if (expr.kind == RecipeValueKind::kListSize)
        return ParamValue(static_cast<std::int64_t>(list->size()));
      std::uint64_t index = 0;
      if (expr.index_name.empty()) {
        if (expr.fixed_index < 0) return std::unexpected("recipe list index is negative");
        index = static_cast<std::uint64_t>(expr.fixed_index);
      } else {
        const auto it = indices.find(expr.index_name);
        if (it == indices.end()) return std::unexpected("recipe list index is unbound");
        if (expr.fixed_index < 0) {
          const auto magnitude = static_cast<std::uint64_t>(-(expr.fixed_index + 1)) + 1;
          if (magnitude > it->second) return std::unexpected("recipe list index offset underflows");
          index = it->second - magnitude;
        } else if (__builtin_add_overflow(it->second, static_cast<std::uint64_t>(expr.fixed_index),
                                          &index)) {
          return std::unexpected("recipe list index offset overflows");
        }
      }
      if (index >= list->size())
        return std::unexpected("recipe list index is outside its family list");
      const auto at = expr.kind == RecipeValueKind::kListElement ? index : list->size() - index - 1;
      return ParamValue((*list)[at]);
    }
    case RecipeValueKind::kAdd:
    case RecipeValueKind::kSubtract:
    case RecipeValueKind::kMultiply:
    case RecipeValueKind::kDivideExact:
    case RecipeValueKind::kDivideFloor:
    case RecipeValueKind::kRealDivide:
    case RecipeValueKind::kEqual:
    case RecipeValueKind::kNotEqual:
    case RecipeValueKind::kLess:
    case RecipeValueKind::kAnd:
    case RecipeValueKind::kOr:
    case RecipeValueKind::kSelect: {
      if (!expr.left || !expr.right)
        return std::unexpected("binary recipe expression is incomplete");
      auto left = Eval(*expr.left, family, source_values, indices, depth + 1);
      if (!left) return std::unexpected(left.error());
      if (expr.kind == RecipeValueKind::kSelect) {
        if (!expr.alternate) return std::unexpected("select recipe expression is incomplete");
        auto condition = Boolean(*left);
        if (!condition) return std::unexpected(condition.error());
        return Eval(*(*condition ? expr.right : expr.alternate), family, source_values, indices,
                    depth + 1);
      }
      auto right = Eval(*expr.right, family, source_values, indices, depth + 1);
      if (!right) return std::unexpected(right.error());
      if (expr.kind == RecipeValueKind::kAnd || expr.kind == RecipeValueKind::kOr) {
        auto a = Boolean(*left);
        auto b = Boolean(*right);
        if (!a || !b) return std::unexpected("logical recipe operands must be boolean");
        return ParamValue(expr.kind == RecipeValueKind::kAnd ? *a && *b : *a || *b);
      }
      if (expr.kind == RecipeValueKind::kEqual || expr.kind == RecipeValueKind::kNotEqual) {
        const bool equal = *left == *right;
        return ParamValue(expr.kind == RecipeValueKind::kEqual ? equal : !equal);
      }
      if (expr.kind == RecipeValueKind::kRealDivide) {
        const auto number = [](const ParamValue& value) -> std::optional<double> {
          if (const auto* real = std::get_if<double>(&value)) return *real;
          if (const auto* integer = std::get_if<std::int64_t>(&value))
            return static_cast<double>(*integer);
          return std::nullopt;
        };
        const auto a = number(*left);
        const auto b = number(*right);
        if (!a || !b || *b == 0) return std::unexpected("real recipe division is invalid");
        const double result = *a / *b;
        if (!std::isfinite(result)) return std::unexpected("real recipe division overflows");
        return ParamValue(result);
      }
      auto a = Integer(*left);
      auto b = Integer(*right);
      if (!a || !b) return std::unexpected("arithmetic recipe operands must be integers");
      if (expr.kind == RecipeValueKind::kLess) return ParamValue(*a < *b);
      if (expr.kind == RecipeValueKind::kDivideExact ||
          expr.kind == RecipeValueKind::kDivideFloor) {
        if (*b == 0 || (*a == INT64_MIN && *b == -1))
          return std::unexpected(expr.kind == RecipeValueKind::kDivideExact
                                     ? "exact integer recipe division is invalid or non-integral"
                                     : "floor integer recipe division is invalid");
        const auto quotient = *a / *b;
        const auto remainder = *a % *b;
        if (expr.kind == RecipeValueKind::kDivideExact && remainder != 0)
          return std::unexpected("exact integer recipe division is invalid or non-integral");
        const auto floor_quotient =
            expr.kind == RecipeValueKind::kDivideFloor && remainder != 0 && ((*a < 0) != (*b < 0))
                ? quotient - 1
                : quotient;
        return ParamValue(floor_quotient);
      }
      std::int64_t result = 0;
      bool overflow = false;
      switch (expr.kind) {
        case RecipeValueKind::kAdd:
          overflow = __builtin_add_overflow(*a, *b, &result);
          break;
        case RecipeValueKind::kSubtract:
          overflow = __builtin_sub_overflow(*a, *b, &result);
          break;
        case RecipeValueKind::kMultiply:
          overflow = __builtin_mul_overflow(*a, *b, &result);
          break;
        default:
          break;
      }
      if (overflow) return std::unexpected("recipe arithmetic overflows");
      return ParamValue(result);
    }
    case RecipeValueKind::kSqrt: {
      if (!expr.left) return std::unexpected("sqrt recipe expression is incomplete");
      auto operand = Eval(*expr.left, family, source_values, indices, depth + 1);
      if (!operand) return std::unexpected(operand.error());
      double value = 0;
      if (const auto* real = std::get_if<double>(&*operand))
        value = *real;
      else if (const auto* integer = std::get_if<std::int64_t>(&*operand))
        value = static_cast<double>(*integer);
      else
        return std::unexpected("sqrt recipe operand must be numeric");
      if (value < 0 || !std::isfinite(value))
        return std::unexpected("sqrt recipe operand is invalid");
      return ParamValue(std::sqrt(value));
    }
    case RecipeValueKind::kNot: {
      if (!expr.left) return std::unexpected("unary recipe expression is incomplete");
      auto operand = Eval(*expr.left, family, source_values, indices, depth + 1);
      if (!operand) return std::unexpected(operand.error());
      auto value = Boolean(*operand);
      if (!value) return std::unexpected(value.error());
      return ParamValue(!*value);
    }
  }
  return std::unexpected(
      std::format("unknown recipe expression kind {}", static_cast<unsigned>(expr.kind)));
}

std::expected<std::string, std::string> Substitute(std::string_view pattern,
                                                   const Indices& indices) {
  std::string out;
  out.reserve(pattern.size());
  for (std::size_t at = 0; at < pattern.size();) {
    if (pattern[at] != '{') {
      out.push_back(pattern[at++]);
      continue;
    }
    const auto end = pattern.find('}', at + 1);
    if (end == std::string_view::npos)
      return std::unexpected("recipe template has an unclosed substitution");
    const auto key = pattern.substr(at + 1, end - at - 1);
    const auto it = indices.find(key);
    if (it == indices.end()) return std::unexpected("recipe template names an unbound index");
    out += std::to_string(it->second);
    at = end + 1;
  }
  return out;
}

std::expected<BoundValue, std::string> Resolve(const RecipeRef& ref, const Values& values) {
  if (ref.kind == RecipeRefKind::kStageInput) return BoundValue{};
  if (ref.kind == RecipeRefKind::kHostPort) return BoundValue{.host_port = ref.port};
  const auto it = values.find(ref.name);
  if (it == values.end()) return std::unexpected("recipe link refers to an unavailable value");
  return it->second;
}

bool AddBound(std::uint64_t amount, std::uint64_t& total) {
  return !__builtin_add_overflow(total, amount, &total);
}

std::expected<std::uint64_t, std::string> TemplateBound(std::string_view value) {
  std::uint64_t bytes = value.size();
  for (std::size_t at = 0; (at = value.find('{', at)) != std::string_view::npos; ++at)
    if (!AddBound(19, bytes)) return std::unexpected("recipe template bound overflows");
  return bytes;
}

std::uint64_t ParamPayload(const ParamValue& value) {
  return std::visit(
      [](const auto& item) -> std::uint64_t {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::string>)
          return item.size();
        else if constexpr (requires {
                             item.size();
                             typename T::value_type;
                           }) {
          std::uint64_t bytes = 0;
          if constexpr (std::is_same_v<typename T::value_type, std::string>) {
            for (const auto& element : item) bytes += element.size();
          } else {
            bytes = item.size() * sizeof(typename T::value_type);
          }
          return bytes;
        } else
          return 0;
      },
      value);
}

std::expected<std::uint64_t, std::string> ExpressionPayload(
    const RecipeValue& expr, const ModelSpec& family, std::span<const MetaEntry> source_values,
    std::size_t depth = 0) {
  if (depth > kMaxParams) return std::unexpected("recipe expression exceeds nesting bounds");
  if (expr.kind == RecipeValueKind::kLiteral) return ParamPayload(expr.literal);
  if (expr.kind == RecipeValueKind::kFamilyValue) {
    auto value = FamilyValue(family, expr.family_kind, expr.family_param);
    if (!value) return std::unexpected(value.error());
    return ParamPayload(**value);
  }
  if (expr.kind == RecipeValueKind::kSourceValue) {
    auto value = SourceValueOf(source_values, expr.source_key);
    if (!value) return std::unexpected(value.error());
    if (!std::holds_alternative<std::int64_t>(**value) &&
        !std::holds_alternative<double>(**value) && !std::holds_alternative<bool>(**value))
      return std::unexpected("source recipe scalar has unsupported metadata type");
    return 0;
  }
  std::uint64_t result = 0;
  for (const RecipeValue* child : {expr.left, expr.right, expr.alternate}) {
    if (!child) continue;
    auto bytes = ExpressionPayload(*child, family, source_values, depth + 1);
    if (!bytes) return std::unexpected(bytes.error());
    if (expr.kind == RecipeValueKind::kSelect) {
      result = std::max(result, *bytes);
    } else if (!AddBound(*bytes, result)) {
      return std::unexpected("recipe expression metadata bound overflows");
    }
  }
  return result;
}

struct EvaluatedWeightStorage {
  std::string role;
  std::string tensor;
  format::PartRole part = format::PartRole::kData;
  std::vector<std::uint64_t> source_dims;
};

std::expected<std::uint64_t, std::string> NodeMetadataBound(
    const RecipeNode& node, const ModelSpec& family, std::span<const MetaEntry> source_values) {
  std::uint64_t bound = sizeof(ComponentRef) + sizeof(GraphLink) + 512;
  const auto add_string = [&](std::string_view value, std::uint64_t copies,
                              std::uint64_t& total) -> bool {
    auto bytes = TemplateBound(value);
    std::uint64_t repeated = 0;
    return bytes && !__builtin_mul_overflow(*bytes, copies, &repeated) && AddBound(repeated, total);
  };
  if (!add_string(node.name, 3, bound) || !add_string(node.component_kind, 1, bound))
    return std::unexpected("recipe node metadata bound overflows");
  const auto add_reference = [&](const RecipeRef& ref) {
    return ref.kind != RecipeRefKind::kNamedValue || add_string(ref.name, 1, bound);
  };
  if (!add_reference(node.input) || (node.has_auxiliary && !add_reference(node.auxiliary)) ||
      (node.has_carry && !add_reference(node.carry)))
    return std::unexpected("recipe graph-reference metadata bound overflows");
  if (node.params.size() > kMaxParams)
    return std::unexpected("recipe node exceeds the component parameter bound");
  for (const auto& parameter : node.params) {
    if (!AddBound(sizeof(Param) + sizeof(ParamValue) + 128, bound) ||
        !add_string(parameter.name, 1, bound))
      return std::unexpected("recipe parameter metadata bound overflows");
    auto payload = ExpressionPayload(parameter.value, family, source_values);
    if (!payload || !AddBound(*payload, bound))
      return std::unexpected(payload ? "recipe parameter metadata bound overflows"
                                     : payload.error());
    if (!parameter.request_scalar.empty() &&
        (!AddBound(sizeof(StageScalarBinding) + 64, bound) ||
         !add_string(parameter.name, 1, bound) || !add_string(parameter.request_scalar, 1, bound)))
      return std::unexpected("recipe scalar metadata bound overflows");
  }
  for (const auto& weight : node.weights) {
    if (!AddBound(sizeof(WeightBinding) + sizeof(RecipeExpansion::Tensor) + 256 +
                      4 * sizeof(std::uint64_t),
                  bound) ||
        !add_string(weight.role, 2, bound) || !add_string(weight.tensor, 3, bound) ||
        !add_string(weight.stage, 1, bound))
      return std::unexpected("recipe tensor metadata bound overflows");
    if (!AddBound(sizeof(EvaluatedWeightStorage), bound))
      return std::unexpected("recipe evaluated weight metadata bound overflows");
    std::uint64_t axes_bytes = 0;
    if (__builtin_mul_overflow(static_cast<std::uint64_t>(weight.source_dims.size()),
                               std::uint64_t{2 * sizeof(std::uint64_t)}, &axes_bytes) ||
        !AddBound(axes_bytes, bound))
      return std::unexpected("recipe raw source-dimension metadata bound overflows");
    for (const auto& dimension : weight.source_dims) {
      auto payload = ExpressionPayload(dimension, family, source_values);
      if (!payload || *payload != 0 || !AddBound(*payload, bound))
        return std::unexpected("recipe raw source dimension must be a scalar expression");
    }
  }
  return bound;
}

std::expected<void, std::string> CheckExpression(const RecipeValue& expr, const ModelSpec& family,
                                                 std::span<const MetaEntry> source_values,
                                                 std::vector<const RecipeValue*>& active,
                                                 std::size_t depth);

std::expected<components::SpatialShape, std::string> EvalSpatialShape(
    const RecipeSpatialShape& expression, const ModelSpec& family,
    std::span<const MetaEntry> source_values, const Indices& indices) {
  const auto eval_dimension =
      [&](const RecipeValue& value) -> std::expected<std::uint64_t, std::string> {
    auto result = Eval(value, family, source_values, indices);
    if (!result) return std::unexpected(result.error());
    auto integer = Integer(*result);
    if (!integer || *integer <= 0)
      return std::unexpected("spatial recipe dimensions must be positive integers");
    return static_cast<std::uint64_t>(*integer);
  };
  auto width = eval_dimension(expression.width);
  auto height = eval_dimension(expression.height);
  auto channels = eval_dimension(expression.channels);
  auto images = eval_dimension(expression.images);
  if (!width || !height || !channels || !images)
    return std::unexpected(!width      ? width.error()
                           : !height   ? height.error()
                           : !channels ? channels.error()
                                       : images.error());
  components::SpatialShape shape{*width, *height, *channels, *images};
  if (!components::SpatialElements(shape))
    return std::unexpected("spatial recipe shape exceeds representable storage");
  return shape;
}

std::expected<void, std::string> CheckSpatialShape(const RecipeSpatialShape& shape,
                                                   const ModelSpec& family,
                                                   std::span<const MetaEntry> source_values,
                                                   std::vector<const RecipeValue*>& active) {
  for (const auto* dimension : {&shape.width, &shape.height, &shape.channels, &shape.images}) {
    if (auto valid = CheckExpression(*dimension, family, source_values, active, 0); !valid)
      return valid;
    auto payload = ExpressionPayload(*dimension, family, source_values);
    if (!payload || *payload != 0)
      return std::unexpected("spatial recipe dimensions must be scalar expressions");
  }
  return {};
}

std::expected<void, std::string> CheckSpatialOperand(const RecipeSpatialOperand& operand,
                                                     const ModelSpec& family,
                                                     std::span<const MetaEntry> source_values,
                                                     std::vector<const RecipeValue*>& active) {
  if (auto valid = CheckSpatialShape(operand.shape, family, source_values, active); !valid)
    return valid;
  const auto parts = format::PartsOf({.encoding = operand.encoding,
                                      .layout = format::Layout::kDense,
                                      .shape = ir::ShapeOf({1, 1, 1, 1})});
  if (!parts) return std::unexpected("spatial recipe port encoding is unknown");
  return {};
}

// Alias stores the resolved name twice: once in `created` so a lexical scope
// can erase it, and once as the key in `values`. Include both string payloads
// and the map node before Steps() begins allocating aliases.
bool AliasMetadataBound(std::string_view pattern, std::uint64_t copies, std::uint64_t& total) {
  auto bytes = TemplateBound(pattern);
  std::uint64_t one_alias_strings = 0;
  std::uint64_t strings = 0;
  constexpr std::uint64_t kMapNode = sizeof(std::pair<const std::string, BoundValue>) + 64;
  // A growing vector can retain both old and new backing storage during a
  // geometric reallocation; three string slots per final alias cover that peak.
  constexpr std::uint64_t kVectorStrings = 3 * sizeof(std::string);
  constexpr std::uint64_t kVectorNode = kMapNode + kVectorStrings;
  std::uint64_t nodes = 0;
  return bytes && !__builtin_mul_overflow(*bytes, std::uint64_t{2}, &one_alias_strings) &&
         !__builtin_mul_overflow(kVectorNode, copies, &nodes) &&
         !__builtin_mul_overflow(one_alias_strings, copies, &strings) && AddBound(strings, total) &&
         AddBound(nodes, total);
}

bool RefMetadataBound(const RecipeRef& ref, std::uint64_t& total) {
  if (ref.kind != RecipeRefKind::kNamedValue) return true;
  const auto bytes = TemplateBound(ref.name);
  return bytes && AddBound(*bytes, total);
}

std::expected<void, std::string> CheckExpression(const RecipeValue& expr, const ModelSpec& family,
                                                 std::span<const MetaEntry> source_values,
                                                 std::vector<const RecipeValue*>& active,
                                                 std::size_t depth = 0) {
  if (depth > kMaxParams) return std::unexpected("recipe expression exceeds nesting bounds");
  if (std::ranges::contains(active, &expr))
    return std::unexpected("stage recipe contains a recursive expression reference");
  active.push_back(&expr);
  const bool has_child = expr.left || expr.right || expr.alternate;
  const auto visit = [&](const RecipeValue* child) -> std::expected<void, std::string> {
    return child ? CheckExpression(*child, family, source_values, active, depth + 1)
                 : std::expected<void, std::string>{};
  };
  std::expected<void, std::string> result;
  switch (expr.kind) {
    case RecipeValueKind::kLiteral:
      if (has_child) result = std::unexpected("literal recipe expression has child pointers");
      break;
    case RecipeValueKind::kFamilyValue:
    case RecipeValueKind::kFamilyF32Value: {
      auto value = FamilyValue(family, expr.family_kind, expr.family_param);
      if (has_child)
        result = std::unexpected("family recipe reference has child pointers");
      else if (!value)
        result = std::unexpected(value.error());
      break;
    }
    case RecipeValueKind::kSourceValue: {
      auto value = SourceValueOf(source_values, expr.source_key);
      if (has_child || !value || std::holds_alternative<std::monostate>(**value) ||
          std::holds_alternative<std::nullptr_t>(**value) ||
          (!std::holds_alternative<std::int64_t>(**value) &&
           !std::holds_alternative<double>(**value) && !std::holds_alternative<bool>(**value)))
        result =
            std::unexpected("source recipe value is absent, unsupported or has child pointers");
      break;
    }
    case RecipeValueKind::kIndex:
      if (has_child || expr.index_name.empty())
        result = std::unexpected("recipe index name is invalid");
      break;
    case RecipeValueKind::kListElement:
    case RecipeValueKind::kListElementReverse:
    case RecipeValueKind::kListSize: {
      auto value = FamilyValue(family, expr.family_kind, expr.family_param);
      if (has_child || !value || !std::holds_alternative<std::vector<std::int64_t>>(**value))
        result = std::unexpected("recipe list reference is absent or not an integer list");
      else if (expr.kind != RecipeValueKind::kListSize && !expr.index_name.empty() &&
               expr.fixed_index < -static_cast<std::int64_t>(kMaxComponents))
        result = std::unexpected("recipe list index offset exceeds graph bounds");
      break;
    }
    case RecipeValueKind::kSourceListElement:
    case RecipeValueKind::kSourceListElementReverse:
    case RecipeValueKind::kSourceListSize: {
      auto value = SourceValueOf(source_values, expr.source_key);
      if (has_child || !value || !std::holds_alternative<std::vector<std::int64_t>>(**value))
        result = std::unexpected("source recipe list is absent or not an integer list");
      else if (expr.kind != RecipeValueKind::kSourceListSize && !expr.index_name.empty() &&
               expr.fixed_index < -static_cast<std::int64_t>(kMaxComponents))
        result = std::unexpected("source list index offset exceeds graph bounds");
      break;
    }
    case RecipeValueKind::kSqrt:
    case RecipeValueKind::kNot:
      if (!expr.left || expr.right || expr.alternate)
        result = std::unexpected("unary recipe expression has invalid operands");
      else
        result = visit(expr.left);
      break;
    case RecipeValueKind::kSelect:
      if (!expr.left || !expr.right || !expr.alternate)
        result = std::unexpected("select recipe expression is incomplete");
      else if (auto checked_condition = visit(expr.left); !checked_condition)
        result = checked_condition;
      else if (auto checked_yes = visit(expr.right); !checked_yes)
        result = checked_yes;
      else
        result = visit(expr.alternate);
      break;
    case RecipeValueKind::kAdd:
    case RecipeValueKind::kSubtract:
    case RecipeValueKind::kMultiply:
    case RecipeValueKind::kDivideExact:
    case RecipeValueKind::kDivideFloor:
    case RecipeValueKind::kRealDivide:
    case RecipeValueKind::kEqual:
    case RecipeValueKind::kNotEqual:
    case RecipeValueKind::kLess:
    case RecipeValueKind::kAnd:
    case RecipeValueKind::kOr:
      if (!expr.left || !expr.right || expr.alternate)
        result = std::unexpected("binary recipe expression has invalid operands");
      else if (auto checked_left = visit(expr.left); !checked_left)
        result = checked_left;
      else
        result = visit(expr.right);
      break;
    default:
      result = std::unexpected("recipe expression has an unknown kind");
      break;
  }
  active.pop_back();
  return result;
}

std::expected<void, std::string> CheckTemplate(std::string_view pattern,
                                               std::span<const std::string_view> indices) {
  for (std::size_t at = 0; at < pattern.size();) {
    if (pattern[at] != '{') {
      if (pattern[at] == '}') return std::unexpected("recipe template has an unmatched brace");
      ++at;
      continue;
    }
    const auto end = pattern.find('}', at + 1);
    if (end == std::string_view::npos)
      return std::unexpected("recipe template has an unclosed substitution");
    const auto key = pattern.substr(at + 1, end - at - 1);
    if (key.empty() || !std::ranges::contains(indices, key))
      return std::unexpected("recipe template uses an index outside its lexical repeat");
    at = end + 1;
  }
  return {};
}

std::expected<void, std::string> CheckRefTemplate(const RecipeRef& ref,
                                                  std::span<const std::string_view> indices) {
  if (ref.kind == RecipeRefKind::kNamedValue) return CheckTemplate(ref.name, indices);
  if (ref.kind == RecipeRefKind::kStageInput || ref.kind == RecipeRefKind::kHostPort) return {};
  return std::unexpected("recipe reference has an unknown kind");
}

std::expected<void, std::string> CheckRecipeSteps(std::span<const RecipeStep> steps,
                                                  const ModelSpec& family,
                                                  std::span<const MetaEntry> source_values,
                                                  std::vector<const RecipeStep*>& active_steps,
                                                  std::vector<const RecipeValue*>& active_values,
                                                  std::vector<std::string_view>& indices,
                                                  std::uint64_t& visited, std::size_t depth = 0) {
  if (depth > kMaxParams) return std::unexpected("recipe control flow exceeds nesting bounds");
  if (steps.empty()) return {};
  if (std::ranges::contains(active_steps, steps.data()))
    return std::unexpected("stage recipe contains a recursive block reference");
  active_steps.push_back(steps.data());
  for (const auto& step : steps) {
    if (++visited > kMaxComponents)
      return std::unexpected("stage recipe syntax exceeds the component bound");
    std::expected<void, std::string> result;
    switch (step.kind) {
      case RecipeStep::Kind::kNode:
        if (!step.node) return std::unexpected("recipe node instruction is empty");
        for (const auto pattern : {step.node->name, step.node->component_kind})
          if (auto checked = CheckTemplate(pattern, indices); !checked) return checked;
        if (auto checked = CheckRefTemplate(step.node->input, indices); !checked) return checked;
        if (step.node->has_auxiliary)
          if (auto checked = CheckRefTemplate(step.node->auxiliary, indices); !checked)
            return checked;
        if (step.node->has_carry)
          if (auto checked = CheckRefTemplate(step.node->carry, indices); !checked) return checked;
        for (const auto& param : step.node->params) {
          if (auto checked = CheckTemplate(param.name, indices); !checked) return checked;
          if (!param.request_scalar.empty() && !spec::IsStageScalarName(param.request_scalar))
            return std::unexpected(std::format(
                "recipe request scalar {} is not one the runtime supplies", param.request_scalar));
          if (auto checked_value =
                  CheckExpression(param.value, family, source_values, active_values);
              !checked_value)
            return checked_value;
        }
        for (const auto& weight : step.node->weights) {
          if (weight.role.empty() || weight.tensor.empty())
            return std::unexpected("recipe weight binding is incomplete");
          for (const auto pattern : {weight.role, weight.tensor, weight.stage})
            if (auto checked = CheckTemplate(pattern, indices); !checked) return checked;
          for (const auto& dimension : weight.source_dims)
            if (auto checked = CheckExpression(dimension, family, source_values, active_values);
                !checked)
              return checked;
        }
        if (step.node->group == RecipeGroupKind::kLayer)
          result = CheckExpression(step.node->group_index, family, source_values, active_values);
        break;
      case RecipeStep::Kind::kRepeat:
        if (!step.repeat || step.repeat->index.empty() || step.repeat->body.empty() ||
            step.repeat->input_name.empty() || step.repeat->body_yield.empty() ||
            step.repeat->yield.empty())
          return std::unexpected("recipe repeat instruction is incomplete");
        if (auto checked =
                CheckExpression(step.repeat->count, family, source_values, active_values);
            !checked)
          return checked;
        if (auto checked = CheckRefTemplate(step.repeat->input, indices); !checked) return checked;
        if (auto checked = CheckTemplate(step.repeat->yield, indices); !checked) return checked;
        indices.push_back(step.repeat->index);
        for (const auto pattern : {step.repeat->input_name, step.repeat->body_yield})
          if (auto checked = CheckTemplate(pattern, indices); !checked) return checked;
        result = CheckRecipeSteps(step.repeat->body, family, source_values, active_steps,
                                  active_values, indices, visited, depth + 1);
        indices.pop_back();
        break;
      case RecipeStep::Kind::kIf:
        if (!step.branch || step.branch->input_name.empty() || step.branch->yield.empty() ||
            step.branch->then_yield.empty() || step.branch->else_yield.empty())
          return std::unexpected("recipe conditional instruction is incomplete");
        if (auto checked =
                CheckExpression(step.branch->condition, family, source_values, active_values);
            !checked)
          return checked;
        for (const auto pattern : {step.branch->input_name, step.branch->then_yield,
                                   step.branch->else_yield, step.branch->yield})
          if (auto checked = CheckTemplate(pattern, indices); !checked) return checked;
        if (auto checked = CheckRefTemplate(step.branch->input, indices); !checked) return checked;
        if (auto checked =
                CheckRecipeSteps(step.branch->then_steps, family, source_values, active_steps,
                                 active_values, indices, visited, depth + 1);
            !checked)
          return checked;
        result = CheckRecipeSteps(step.branch->else_steps, family, source_values, active_steps,
                                  active_values, indices, visited, depth + 1);
        break;
      default:
        return std::unexpected("recipe instruction has an unknown kind");
    }
    if (!result) return result;
  }
  active_steps.pop_back();
  return {};
}

void Attach(const BoundValue& value, GraphLink& link, unsigned slot) {
  if (slot == 0) {
    link.input = value.id;
    link.input_port = value.host_port;
  } else if (slot == 1) {
    if (value.host_port)
      link.auxiliary_port = value.host_port;
    else
      link.auxiliary = value.id;
  } else if (value.host_port) {
    link.carry_port = value.host_port;
  } else {
    link.carry = value.id;
  }
}

struct Expander {
  const ModelSpec& family;
  std::span<const MetaEntry> source_values;
  std::uint64_t paid;
  Stage stage;
  Values values;
  std::vector<components::SpatialOperand> geometry;
  std::vector<components::SpatialOperand> host_ports;
  std::vector<std::string> created;
  std::vector<RecipeExpansion::Tensor> tensors;
  std::vector<const RecipeStep*> active_blocks;
  std::uint64_t estimate = 0;

  bool Charge(std::uint64_t amount) {
    return !__builtin_add_overflow(estimate, amount, &estimate) && estimate <= paid;
  }

  std::expected<void, std::string> Alias(std::string name, BoundValue value) {
    if (values.contains(name))
      return std::unexpected("recipe value name is defined more than once");
    created.push_back(name);
    values.emplace(std::move(name), value);
    return {};
  }

  void ClearCreatedFrom(std::size_t mark) {
    for (std::size_t i = mark; i < created.size(); ++i) values.erase(created[i]);
    created.resize(mark);
  }

  std::expected<BoundValue, std::string> ResolveRef(const RecipeRef& ref,
                                                    const Indices& indices) const {
    if (ref.kind != RecipeRefKind::kNamedValue) return Resolve(ref, values);
    auto name = Substitute(ref.name, indices);
    if (!name) return std::unexpected(name.error());
    const auto it = values.find(*name);
    if (it == values.end()) return std::unexpected("recipe link refers to an unavailable value");
    return it->second;
  }

  std::expected<components::SpatialOperand, std::string> ResolveGeometry(
      const BoundValue& value) const {
    if (value.host_port) {
      if (*value.host_port >= host_ports.size())
        return std::unexpected("recipe spatial host port is unavailable");
      return host_ports[*value.host_port];
    }
    if (value.id >= geometry.size())
      return std::unexpected("recipe spatial value geometry is unavailable");
    return geometry[value.id];
  }

  std::expected<std::vector<std::uint64_t>, std::string> EvalSourceDims(
      std::span<const RecipeValue> expressions, const Indices& indices) const {
    std::vector<std::uint64_t> dimensions;
    dimensions.reserve(expressions.size());
    for (const auto& expression : expressions) {
      auto value = Eval(expression, family, source_values, indices);
      if (!value) return std::unexpected(value.error());
      const auto integer = Integer(*value);
      if (!integer || *integer <= 0)
        return std::unexpected("recipe raw source dimensions must be positive integers");
      dimensions.push_back(static_cast<std::uint64_t>(*integer));
    }
    return dimensions;
  }

  std::expected<void, std::string> Node(const RecipeNode& node, const Indices& indices) {
    if (stage.layers.empty()) stage.layers.emplace_back();
    auto name = Substitute(node.name, indices);
    if (!name) return std::unexpected(std::string(name.error()));
    if (name->empty()) return std::unexpected("recipe node name is empty");
    if (values.contains(*name)) return std::unexpected("recipe defines a value more than once");
    if (stage.layers[0].components.size() >= kMaxComponents ||
        stage.layers[0].components.size() >= std::numeric_limits<std::uint16_t>::max())
      return std::unexpected("recipe exceeds the stage component/value-id bound");
    ComponentRef component{.kind = std::string(node.component_kind), .version = node.version};
    component.params.reserve(node.params.size());
    const auto scalars_before = stage.scalar_bindings.size();
    for (const auto& parameter : node.params) {
      auto key = Substitute(parameter.name, indices);
      auto value = Eval(parameter.value, family, source_values, indices);
      if (!key) return std::unexpected(std::string(key.error()));
      if (!value) return std::unexpected(value.error());
      component.params.push_back({.name = std::move(*key), .value = std::move(*value)});
      if (!parameter.request_scalar.empty()) {
        const auto* integer = std::get_if<std::int64_t>(&component.params.back().value);
        if (!integer)
          return std::unexpected("request-bound recipe parameter must have an integer default");
        // Stage scalar component IDs are one-based output value IDs.
        stage.scalar_bindings.push_back(
            {.component = static_cast<std::uint16_t>(stage.layers[0].components.size() + 1),
             .parameter = component.params.back().name,
             .source = std::string(parameter.request_scalar)});
      }
    }
    std::uint64_t raw_axis_count = 0;
    for (const auto& weight : node.weights)
      if (__builtin_add_overflow(raw_axis_count,
                                 static_cast<std::uint64_t>(weight.source_dims.size()),
                                 &raw_axis_count))
        return std::unexpected("recipe raw source-dimension count overflows");
    std::uint64_t raw_axis_bytes = 0;
    if (__builtin_mul_overflow(raw_axis_count, std::uint64_t{sizeof(std::uint64_t)},
                               &raw_axis_bytes) ||
        !Charge(raw_axis_bytes))
      return std::unexpected("recipe exceeds its paid metadata bound");
    std::vector<EvaluatedWeightStorage> evaluated_weights;
    evaluated_weights.reserve(node.weights.size());
    component.weights.reserve(node.weights.size());
    for (const auto& weight : node.weights) {
      auto role = Substitute(weight.role, indices);
      auto tensor = Substitute(weight.tensor, indices);
      if (!role) return std::unexpected(std::string(role.error()));
      if (!tensor) return std::unexpected(std::string(tensor.error()));
      auto stage_name = Substitute(weight.stage, indices);
      if (!stage_name) return std::unexpected(std::string(stage_name.error()));
      auto source_dims = EvalSourceDims(weight.source_dims, indices);
      if (!source_dims) return std::unexpected(source_dims.error());
      evaluated_weights.push_back({.role = *role,
                                   .tensor = *tensor,
                                   .part = weight.part,
                                   .source_dims = std::move(*source_dims)});
      component.weights.push_back({.role = std::move(*role),
                                   .tensor = *tensor,
                                   .part = weight.part,
                                   .experts = weight.experts,
                                   .stage = std::move(*stage_name)});
    }
    std::ranges::sort(component.params, {}, &Param::name);
    std::ranges::sort(component.weights, {}, [](const WeightBinding& weight) {
      return std::pair(weight.role, weight.part);
    });
    if (std::adjacent_find(component.weights.begin(), component.weights.end(),
                           [](const WeightBinding& a, const WeightBinding& b) {
                             return a.role == b.role && a.part == b.part;
                           }) != component.weights.end())
      return std::unexpected("recipe assigns one component weight role more than once");
    const auto component_index = static_cast<std::uint16_t>(stage.layers[0].components.size());
    const components::StageFacts facts{
        .width = stage.stream.width, .context = stage.context, .streams = stage.stream.streams};
    auto instance =
        components::MakeInstance(component, facts, components::Site::kLayer, 0, component_index);
    if (!instance) return std::unexpected(instance.error());
    auto role_shapes = instance->def->shapes(*instance);
    if (!role_shapes) return std::unexpected(role_shapes.error());
    for (const auto& weight : component.weights) {
      const auto role = components::RoleIndex(*instance->def, weight.role);
      if (!role) return std::unexpected("recipe binds an unknown component weight role");
      const auto shape = std::ranges::find(*role_shapes, *role, &components::RoleShape::role);
      if (shape == role_shapes->end())
        return std::unexpected("recipe binds a component role with no source shape");
      std::vector<std::uint64_t> dimensions;
      dimensions.reserve(shape->rank);
      for (std::size_t d = shape->rank; d > 0; --d) dimensions.push_back(shape->ne[d - 1]);
      std::uint32_t group_index = 0;
      if (node.group == RecipeGroupKind::kLayer) {
        auto group_value = Eval(node.group_index, family, source_values, indices);
        if (!group_value) return std::unexpected(group_value.error());
        auto index = Integer(*group_value);
        if (!index || *index < 0 || *index > UINT32_MAX)
          return std::unexpected("recipe component group index is invalid");
        group_index = static_cast<std::uint32_t>(*index);
      }
      const auto previous = std::ranges::find_if(tensors, [&](const RecipeExpansion::Tensor& t) {
        return t.name == weight.tensor && t.part == weight.part;
      });
      const auto declared =
          std::ranges::find_if(evaluated_weights, [&](const EvaluatedWeightStorage& w) {
            return w.role == weight.role && w.tensor == weight.tensor && w.part == weight.part;
          });
      if (declared == evaluated_weights.end())
        return std::unexpected("recipe weight lost its evaluated source-dimension contract");
      if (previous == tensors.end())
        tensors.push_back({.name = weight.tensor,
                           .shape = std::move(dimensions),
                           .group = node.group,
                           .layer = group_index,
                           .part = weight.part,
                           .source_dims = std::move(declared->source_dims),
                           .experts = weight.experts,
                           .stage = std::string(weight.stage)});
      else if (previous->shape != dimensions || previous->group != node.group ||
               previous->layer != group_index || previous->experts != weight.experts ||
               previous->stage != weight.stage)
        return std::unexpected("shared recipe tensor has inconsistent role-derived binding");
      else if (!declared->source_dims.empty()) {
        if (!previous->source_dims.empty() && previous->source_dims != declared->source_dims)
          return std::unexpected("shared recipe tensor has conflicting raw source dimensions");
        if (previous->source_dims.empty()) previous->source_dims = std::move(declared->source_dims);
      }
    }
    GraphLink link;
    auto input = ResolveRef(node.input, indices);
    if (!input) return std::unexpected(input.error());
    Attach(*input, link, 0);
    std::optional<BoundValue> auxiliary_value;
    if (node.has_auxiliary) {
      auto auxiliary = ResolveRef(node.auxiliary, indices);
      if (!auxiliary) return std::unexpected(auxiliary.error());
      auxiliary_value = *auxiliary;
      Attach(*auxiliary, link, 1);
    }
    std::optional<BoundValue> carry_value;
    if (node.has_carry) {
      auto carry = ResolveRef(node.carry, indices);
      if (!carry) return std::unexpected(carry.error());
      carry_value = *carry;
      Attach(*carry, link, 2);
    }
    auto input_geometry = ResolveGeometry(*input);
    if (!input_geometry) return std::unexpected(input_geometry.error());
    std::optional<components::SpatialOperand> auxiliary_geometry;
    if (auxiliary_value) {
      auto resolved = ResolveGeometry(*auxiliary_value);
      if (!resolved) return std::unexpected(resolved.error());
      auxiliary_geometry = *resolved;
    }
    std::optional<components::SpatialOperand> carry_geometry;
    if (carry_value) {
      auto resolved = ResolveGeometry(*carry_value);
      if (!resolved) return std::unexpected(resolved.error());
      carry_geometry = *resolved;
    }
    auto output_geometry = components::InferSpatialOutput(*instance, *input_geometry,
                                                          auxiliary_geometry, carry_geometry);
    if (!output_geometry) return std::unexpected(output_geometry.error());
    const auto node_scalars = stage.scalar_bindings.size() - scalars_before;
    if (!Charge(sizeof(ComponentRef) + sizeof(GraphLink) + name->size() +
                component.params.size() * sizeof(Param) +
                component.weights.size() * sizeof(WeightBinding) +
                node_scalars * sizeof(StageScalarBinding)))
      return std::unexpected("recipe exceeds its paid metadata bound");
    stage.layers[0].components.push_back(std::move(component));
    stage.links.push_back(link);
    geometry.push_back(*output_geometry);
    if (auto alias =
            Alias(std::move(*name),
                  BoundValue{.id = static_cast<std::uint16_t>(stage.layers[0].components.size())});
        !alias)
      return alias;
    return {};
  }

  std::expected<void, std::string> Steps(std::span<const RecipeStep> steps, Indices& indices) {
    if (steps.empty()) return {};
    if (std::ranges::contains(active_blocks, steps.data()))
      return std::unexpected("stage recipe contains a recursive block reference");
    active_blocks.push_back(steps.data());
    for (const auto& step : steps) {
      switch (step.kind) {
        case RecipeStep::Kind::kNode:
          if (!step.node) return std::unexpected("recipe node instruction is empty");
          if (auto result = Node(*step.node, indices); !result) return result;
          break;
        case RecipeStep::Kind::kRepeat: {
          if (!step.repeat || step.repeat->index.empty())
            return std::unexpected("recipe repeat instruction is incomplete");
          auto count_value = Eval(step.repeat->count, family, source_values, indices);
          if (!count_value) return std::unexpected(count_value.error());
          auto count = Integer(*count_value);
          if (!count || *count < 0) return std::unexpected("recipe repeat count is invalid");
          auto carried = ResolveRef(step.repeat->input, indices);
          if (!carried) return std::unexpected(carried.error());
          if (step.repeat->input_name.empty() || step.repeat->yield.empty() ||
              step.repeat->body_yield.empty())
            return std::unexpected("recipe repeat has incomplete carried value names");
          for (std::int64_t i = 0; i < *count; ++i) {
            auto previous_index = indices.find(step.repeat->index);
            const auto old_index = previous_index == indices.end()
                                       ? std::optional<std::uint64_t>{}
                                       : std::optional(previous_index->second);
            indices[step.repeat->index] = static_cast<std::uint64_t>(i);
            auto input_name = Substitute(step.repeat->input_name, indices);
            if (!input_name) return std::unexpected(input_name.error());
            const auto scope = created.size();
            if (auto alias = Alias(std::move(*input_name), *carried); !alias) return alias;
            if (auto result = Steps(step.repeat->body, indices); !result) return result;
            auto yield_name = Substitute(step.repeat->body_yield, indices);
            if (!yield_name) return std::unexpected(yield_name.error());
            const auto found = values.find(*yield_name);
            if (found == values.end()) return std::unexpected("recipe repeat yield is unavailable");
            carried = found->second;
            ClearCreatedFrom(scope);
            if (old_index)
              indices[step.repeat->index] = *old_index;
            else
              indices.erase(step.repeat->index);
          }
          auto repeat_yield = Substitute(step.repeat->yield, indices);
          if (!repeat_yield) return std::unexpected(repeat_yield.error());
          if (auto alias = Alias(std::move(*repeat_yield), *carried); !alias) return alias;
          break;
        }
        case RecipeStep::Kind::kIf: {
          if (!step.branch) return std::unexpected("recipe conditional is empty");
          auto predicate = Eval(step.branch->condition, family, source_values, indices);
          if (!predicate) return std::unexpected(predicate.error());
          auto condition = Boolean(*predicate);
          if (!condition) return std::unexpected(condition.error());
          const auto& body = *condition ? step.branch->then_steps : step.branch->else_steps;
          const auto yield_pattern = *condition ? step.branch->then_yield : step.branch->else_yield;
          if (step.branch->input_name.empty() || step.branch->yield.empty() ||
              yield_pattern.empty())
            return std::unexpected("recipe conditional has incomplete input or yield");
          auto input = ResolveRef(step.branch->input, indices);
          if (!input) return std::unexpected(input.error());
          auto input_name = Substitute(step.branch->input_name, indices);
          if (!input_name) return std::unexpected(input_name.error());
          const auto scope = created.size();
          if (auto alias = Alias(std::move(*input_name), *input); !alias) return alias;
          if (auto result = Steps(body, indices); !result) return result;
          BoundValue value;
          {
            auto name = Substitute(yield_pattern, indices);
            if (!name) return std::unexpected(name.error());
            const auto found = values.find(*name);
            if (found == values.end())
              return std::unexpected("recipe conditional yield is unavailable");
            value = found->second;
          }
          ClearCreatedFrom(scope);
          auto output_name = Substitute(step.branch->yield, indices);
          if (!output_name) return std::unexpected(output_name.error());
          if (auto alias = Alias(std::move(*output_name), value); !alias) return alias;
          break;
        }
      }
    }
    active_blocks.pop_back();
    return {};
  }
};

std::expected<std::uint64_t, std::string> CountSteps(
    std::span<const RecipeStep> steps, const ModelSpec& family,
    std::span<const MetaEntry> source_values, Indices& indices,
    std::vector<const RecipeStep*>& active, std::uint64_t& visited, std::uint64_t& tensor_bindings,
    std::uint64_t& tensor_name_bytes, std::uint64_t& max_tensor_name_bytes,
    std::uint64_t& max_weight_role_bytes, std::uint64_t& raw_dimension_axes,
    std::uint64_t& raw_dimension_contracts, std::uint64_t& metadata_bound) {
  if (steps.empty()) return std::uint64_t{0};
  if (std::ranges::contains(active, steps.data()))
    return std::unexpected("stage recipe contains a recursive block reference");
  active.push_back(steps.data());
  std::uint64_t count = 0;
  for (const auto& step : steps) {
    if (__builtin_add_overflow(visited, std::uint64_t{1}, &visited) || visited > kMaxComponents)
      return std::unexpected("stage recipe instruction expansion exceeds spec bounds");
    if (step.kind == RecipeStep::Kind::kNode) {
      if (!step.node) return std::unexpected("recipe node instruction is empty");
      auto bytes = NodeMetadataBound(*step.node, family, source_values);
      if (!bytes || !AddBound(*bytes, metadata_bound))
        return std::unexpected(bytes ? "recipe metadata bound overflows" : bytes.error());
      if (__builtin_add_overflow(tensor_bindings,
                                 static_cast<std::uint64_t>(step.node->weights.size()),
                                 &tensor_bindings))
        return std::unexpected("recipe tensor binding count overflows");
      for (const auto& weight : step.node->weights) {
        auto name_bytes = TemplateBound(weight.tensor);
        if (!name_bytes || !AddBound(*name_bytes, tensor_name_bytes))
          return std::unexpected("recipe tensor-name byte bound overflows");
        max_tensor_name_bytes = std::max(max_tensor_name_bytes, *name_bytes);
        auto role_bytes = TemplateBound(weight.role);
        if (!role_bytes) return std::unexpected("recipe weight-role name bound overflows");
        max_weight_role_bytes = std::max(max_weight_role_bytes, *role_bytes);
        if (!weight.source_dims.empty() &&
            __builtin_add_overflow(raw_dimension_contracts, std::uint64_t{1},
                                   &raw_dimension_contracts))
          return std::unexpected("recipe raw source-dimension contract count overflows");
        if (__builtin_add_overflow(raw_dimension_axes,
                                   static_cast<std::uint64_t>(weight.source_dims.size()),
                                   &raw_dimension_axes))
          return std::unexpected("recipe raw source-dimension count overflows");
      }
      ++count;
    } else if (step.kind == RecipeStep::Kind::kRepeat) {
      if (!step.repeat || step.repeat->index.empty() || step.repeat->body.empty())
        return std::unexpected("recipe repeat instruction is incomplete or empty");
      if (!RefMetadataBound(step.repeat->input, metadata_bound))
        return std::unexpected("recipe repeat reference metadata bound overflows");
      auto control_payload = ExpressionPayload(step.repeat->count, family, source_values);
      if (!control_payload || *control_payload != 0)
        return std::unexpected("recipe repeat count must be a scalar expression");
      auto value = Eval(step.repeat->count, family, source_values, indices);
      if (!value) return std::unexpected(value.error());
      auto iterations = Integer(*value);
      if (!iterations || *iterations < 0 ||
          static_cast<std::uint64_t>(*iterations) > kMaxComponents)
        return std::unexpected("recipe repeat count exceeds the graph component bound");
      for (std::int64_t i = 0; i < *iterations; ++i) {
        if (!AliasMetadataBound(step.repeat->input_name, 1, metadata_bound))
          return std::unexpected("recipe repeat alias metadata bound overflows");
        const auto yield_bytes = TemplateBound(step.repeat->body_yield);
        if (!yield_bytes || !AddBound(*yield_bytes, metadata_bound))
          return std::unexpected("recipe repeat yield temporary bound overflows");
        // The index is inserted and erased for every iteration when this
        // lexical name is not already bound by an outer repeat.
        constexpr std::uint64_t kIndexMapNode =
            sizeof(std::pair<const std::string, std::uint64_t>) + 64;
        std::uint64_t index_bytes = 0;
        if (__builtin_add_overflow(kIndexMapNode,
                                   static_cast<std::uint64_t>(step.repeat->index.size()),
                                   &index_bytes) ||
            !AddBound(index_bytes, metadata_bound))
          return std::unexpected("recipe repeat index metadata bound overflows");
        const auto old = indices.find(step.repeat->index);
        const auto previous =
            old == indices.end() ? std::optional<std::uint64_t>{} : std::optional(old->second);
        indices[step.repeat->index] = static_cast<std::uint64_t>(i);
        auto nested = CountSteps(step.repeat->body, family, source_values, indices, active, visited,
                                 tensor_bindings, tensor_name_bytes, max_tensor_name_bytes,
                                 max_weight_role_bytes, raw_dimension_axes, raw_dimension_contracts,
                                 metadata_bound);
        if (previous)
          indices[step.repeat->index] = *previous;
        else
          indices.erase(step.repeat->index);
        if (!nested) return std::unexpected(nested.error());
        if (__builtin_add_overflow(count, *nested, &count) || count > kMaxComponents)
          return std::unexpected("stage recipe exceeds the graph component bound");
      }
      if (!AliasMetadataBound(step.repeat->yield, 1, metadata_bound))
        return std::unexpected("recipe repeat yield metadata bound overflows");
    } else {
      if (!step.branch) return std::unexpected("recipe conditional is empty");
      if (!RefMetadataBound(step.branch->input, metadata_bound))
        return std::unexpected("recipe conditional reference metadata bound overflows");
      auto control_payload = ExpressionPayload(step.branch->condition, family, source_values);
      if (!control_payload || *control_payload != 0)
        return std::unexpected("recipe predicate must be a scalar expression");
      auto value = Eval(step.branch->condition, family, source_values, indices);
      if (!value) return std::unexpected(value.error());
      auto condition = Boolean(*value);
      if (!condition) return std::unexpected(condition.error());
      const auto& selected_yield = *condition ? step.branch->then_yield : step.branch->else_yield;
      const auto yield_bytes = TemplateBound(selected_yield);
      if (!yield_bytes || !AddBound(*yield_bytes, metadata_bound))
        return std::unexpected("recipe conditional yield temporary bound overflows");
      auto nested = CountSteps(*condition ? step.branch->then_steps : step.branch->else_steps,
                               family, source_values, indices, active, visited, tensor_bindings,
                               tensor_name_bytes, max_tensor_name_bytes, max_weight_role_bytes,
                               raw_dimension_axes, raw_dimension_contracts, metadata_bound);
      if (!nested) return std::unexpected(nested.error());
      if (!AliasMetadataBound(step.branch->input_name, 1, metadata_bound) ||
          !AliasMetadataBound(step.branch->yield, 1, metadata_bound))
        return std::unexpected("recipe conditional alias metadata bound overflows");
      if (__builtin_add_overflow(count, *nested, &count))
        return std::unexpected("stage recipe instruction count overflows");
    }
    if (count > kMaxComponents) return std::unexpected("stage recipe exceeds graph bounds");
  }
  active.pop_back();
  return count;
}

std::expected<std::uint64_t, std::string> SourceReferences(const RecipeValue& value,
                                                           std::vector<const RecipeValue*>& active,
                                                           std::uint64_t& max_source_key_bytes,
                                                           std::size_t depth = 0) {
  if (depth > kMaxParams || std::ranges::contains(active, &value))
    return std::unexpected("stage recipe source-reference walk is recursive or too deep");
  active.push_back(&value);
  std::uint64_t count = value.kind == RecipeValueKind::kSourceValue ||
                                value.kind == RecipeValueKind::kSourceListElement ||
                                value.kind == RecipeValueKind::kSourceListElementReverse ||
                                value.kind == RecipeValueKind::kSourceListSize
                            ? 1
                            : 0;
  if (count != 0)
    max_source_key_bytes =
        std::max(max_source_key_bytes, static_cast<std::uint64_t>(value.source_key.size()));
  for (const auto* child : {value.left, value.right, value.alternate}) {
    if (!child) continue;
    auto nested = SourceReferences(*child, active, max_source_key_bytes, depth + 1);
    if (!nested || __builtin_add_overflow(count, *nested, &count))
      return std::unexpected("stage recipe source-reference count overflows");
  }
  active.pop_back();
  return count;
}

std::expected<std::uint64_t, std::string> RecipeSourceReferences(
    const StageRecipe& recipe, std::vector<const RecipeValue*>& active,
    std::uint64_t& max_source_key_bytes) {
  std::uint64_t count = 0;
  const auto add = [&](const RecipeValue& expression) -> bool {
    auto refs = SourceReferences(expression, active, max_source_key_bytes);
    if (!refs || __builtin_add_overflow(count, *refs, &count)) return false;
    return true;
  };
  const auto shape = [&](const RecipeSpatialShape& value) {
    return add(value.width) && add(value.height) && add(value.channels) && add(value.images);
  };
  if (!add(recipe.stream_width) || !shape(recipe.primary_input.shape))
    return std::unexpected("stage recipe source-reference count overflows");
  for (const auto& port : recipe.host_ports)
    if (!shape(port.shape)) return std::unexpected("stage recipe source-reference count overflows");
  for (const auto& output : recipe.expected_outputs)
    if (!shape(output)) return std::unexpected("stage recipe source-reference count overflows");

  std::vector<const RecipeStep*> blocks;
  const auto walk = [&](auto&& self, std::span<const RecipeStep> steps, std::size_t depth) -> bool {
    if (depth > kMaxComponents || std::ranges::contains(blocks, steps.data())) return false;
    blocks.push_back(steps.data());
    for (const auto& step : steps) {
      if (step.kind == RecipeStep::Kind::kNode) {
        if (!step.node) return false;
        for (const auto& param : step.node->params)
          if (!add(param.value)) return false;
        for (const auto& weight : step.node->weights)
          for (const auto& dimension : weight.source_dims)
            if (!add(dimension)) return false;
        if (step.node->group == RecipeGroupKind::kLayer && !add(step.node->group_index))
          return false;
      } else if (step.kind == RecipeStep::Kind::kRepeat) {
        if (!step.repeat || !add(step.repeat->count) || !self(self, step.repeat->body, depth + 1))
          return false;
      } else {
        if (!step.branch || !add(step.branch->condition) ||
            !self(self, step.branch->then_steps, depth + 1) ||
            !self(self, step.branch->else_steps, depth + 1))
          return false;
      }
    }
    blocks.pop_back();
    return true;
  };
  if (!walk(walk, recipe.steps, 0))
    return std::unexpected("stage recipe source-reference tree is invalid");
  return count;
}

}  // namespace

std::expected<StageRecipePreflight, std::string> PreflightStageRecipe(
    const spec::StageRecipe& recipe, const spec::ModelSpec& family, std::uint64_t metadata_bytes,
    std::span<const MetaEntry> source_values) {
  if (recipe.name.empty() || metadata_bytes == 0)
    return Bad("stage recipe header or prepaid metadata bound is invalid");
  std::vector<const RecipeValue*> active_values;
  if (auto valid = CheckExpression(recipe.stream_width, family, source_values, active_values);
      !valid)
    return Bad(valid.error());
  std::vector<const RecipeStep*> ast_blocks;
  std::vector<std::string_view> lexical_indices;
  std::uint64_t ast_steps = 0;
  if (auto valid = CheckRecipeSteps(recipe.steps, family, source_values, ast_blocks, active_values,
                                    lexical_indices, ast_steps);
      !valid)
    return Bad(valid.error());
  for (const auto& output : recipe.exports)
    if (output.name.empty() && output.kind == RecipeRefKind::kNamedValue)
      return Bad("stage export has an empty recipe name");
  auto width_payload = ExpressionPayload(recipe.stream_width, family, source_values);
  if (!width_payload || *width_payload != 0)
    return Bad("stage stream width must be a scalar expression");
  auto width_value = Eval(recipe.stream_width, family, source_values, {});
  if (!width_value) return Bad(width_value.error());
  auto width = Integer(*width_value);
  if (!width || *width <= 0 || static_cast<std::uint64_t>(*width) > kMaxWidth)
    return Bad("stage recipe stream width is invalid");
  Indices indices;
  std::vector<const RecipeStep*> active;
  std::uint64_t visited = 0;
  std::uint64_t tensor_bindings = 0;
  std::uint64_t tensor_name_bytes = 0;
  std::uint64_t max_tensor_name_bytes = 0;
  std::uint64_t max_weight_role_bytes = 0;
  std::uint64_t raw_dimension_axes = 0;
  std::uint64_t raw_dimension_contracts = 0;
  std::uint64_t metadata_bound = 4096;
  auto node_count =
      CountSteps(recipe.steps, family, source_values, indices, active, visited, tensor_bindings,
                 tensor_name_bytes, max_tensor_name_bytes, max_weight_role_bytes,
                 raw_dimension_axes, raw_dimension_contracts, metadata_bound);
  if (!node_count) return Bad(node_count.error());
  std::vector<const RecipeValue*> reference_path;
  std::uint64_t max_source_key_bytes = 0;
  auto source_references = RecipeSourceReferences(recipe, reference_path, max_source_key_bytes);
  if (!source_references) return Bad(source_references.error());
  if (auto valid = CheckSpatialOperand(recipe.primary_input, family, source_values, active_values);
      !valid)
    return Bad(valid.error());
  for (const auto& port : recipe.host_ports)
    if (auto valid = CheckSpatialOperand(port, family, source_values, active_values); !valid)
      return Bad(valid.error());
  for (const auto& output : recipe.expected_outputs) {
    if (auto valid = CheckSpatialShape(output, family, source_values, active_values); !valid)
      return Bad(valid.error());
  }
  const auto output_count = recipe.exports.empty() ? std::size_t{1} : recipe.exports.size();
  if (!recipe.expected_outputs.empty() && recipe.expected_outputs.size() != output_count)
    return Bad("expected spatial output count differs from recipe exports");
  std::uint64_t geometry_slots = 0, geometry_bytes = 0;
  if (__builtin_add_overflow(*node_count, std::uint64_t{1}, &geometry_slots) ||
      __builtin_mul_overflow(geometry_slots, std::uint64_t{sizeof(components::SpatialOperand)},
                             &geometry_bytes) ||
      !AddBound(geometry_bytes, metadata_bound) ||
      __builtin_mul_overflow(static_cast<std::uint64_t>(recipe.host_ports.size()),
                             std::uint64_t{sizeof(components::SpatialOperand)}, &geometry_bytes) ||
      !AddBound(geometry_bytes, metadata_bound) ||
      __builtin_mul_overflow(static_cast<std::uint64_t>(recipe.expected_outputs.size()),
                             std::uint64_t{sizeof(components::SpatialShape)}, &geometry_bytes) ||
      !AddBound(geometry_bytes, metadata_bound))
    return Bad("spatial recipe geometry metadata bound overflows");
  auto recipe_name = TemplateBound(recipe.name);
  if (!recipe_name || !AddBound(*recipe_name, metadata_bound))
    return Bad("stage recipe metadata bound overflows");
  for (const auto& output : recipe.exports) {
    auto bytes = TemplateBound(output.name);
    if (!bytes || !AddBound(sizeof(std::uint16_t) + *bytes + 128, metadata_bound))
      return Bad("stage export metadata bound overflows");
  }
  if (metadata_bound > metadata_bytes)
    return Bad("stage recipe exceeds its prepaid metadata bound");
  std::uint64_t lookup_multiplier = 0, source_value_lookups = 0;
  if (__builtin_add_overflow(*node_count, visited, &lookup_multiplier) ||
      __builtin_mul_overflow(lookup_multiplier, std::uint64_t{5}, &lookup_multiplier) ||
      __builtin_add_overflow(lookup_multiplier, std::uint64_t{5}, &lookup_multiplier) ||
      __builtin_mul_overflow(*source_references, lookup_multiplier, &source_value_lookups))
    return Bad("stage recipe source lookup work overflows");
  return StageRecipePreflight{.expanded_nodes = *node_count,
                              .tensor_bindings = tensor_bindings,
                              .raw_dimension_axes = raw_dimension_axes,
                              .raw_dimension_contracts = raw_dimension_contracts,
                              .tensor_name_bytes = tensor_name_bytes,
                              .max_tensor_name_bytes = max_tensor_name_bytes,
                              .max_weight_role_bytes = max_weight_role_bytes,
                              .max_source_key_bytes = max_source_key_bytes,
                              .visited_steps = visited,
                              .source_value_lookups = source_value_lookups,
                              .metadata_bytes = metadata_bound,
                              .stream_width = static_cast<std::uint64_t>(*width)};
}

std::expected<spec::RecipeExpansion, std::string> ExpandStageRecipe(
    const spec::StageRecipe& recipe, const spec::ModelSpec& family, std::uint64_t metadata_bytes,
    std::span<const MetaEntry> source_values) {
  auto preflight = PreflightStageRecipe(recipe, family, metadata_bytes, source_values);
  if (!preflight) return Bad(preflight.error());
  Indices indices;
  std::uint64_t node_count = preflight->expanded_nodes;
  const auto width = preflight->stream_width;
  Expander expander{.family = family,
                    .source_values = source_values,
                    .paid = metadata_bytes,
                    .stage = {.name = std::string(recipe.name),
                              .role = recipe.role,
                              .artifact = recipe.artifact,
                              .stream = {.width = width}},
                    .values = {},
                    .geometry = {},
                    .host_ports = {},
                    .created = {},
                    .tensors = {},
                    .active_blocks = {},
                    .estimate = 0};
  indices.clear();
  expander.geometry.reserve(static_cast<std::size_t>(node_count + 1));
  expander.host_ports.reserve(recipe.host_ports.size());
  auto primary_shape = EvalSpatialShape(recipe.primary_input.shape, family, source_values, indices);
  if (!primary_shape) return Bad(primary_shape.error());
  expander.geometry.push_back({.shape = *primary_shape, .encoding = recipe.primary_input.encoding});
  for (const auto& host : recipe.host_ports) {
    auto shape = EvalSpatialShape(host.shape, family, source_values, indices);
    if (!shape) return Bad(shape.error());
    expander.host_ports.push_back({.shape = *shape, .encoding = host.encoding});
  }
  if (auto expanded = expander.Steps(recipe.steps, indices); !expanded)
    return Bad(expanded.error());
  if (expander.stage.layers.empty() || expander.stage.layers[0].components.empty())
    return Bad("stage recipe produced no components");
  for (const auto& output : recipe.exports) {
    auto value = expander.ResolveRef(output, indices);
    if (!value) return Bad(value.error());
    if (value->host_port || value->id == 0)
      return Bad("a stage export must name a computed graph value");
    expander.stage.exports.push_back(value->id);
  }
  if (!recipe.expected_outputs.empty()) {
    const auto output_count = recipe.expected_outputs.size();
    for (std::size_t i = 0; i < output_count; ++i) {
      const auto value_id =
          recipe.exports.empty()
              ? static_cast<std::uint16_t>(expander.stage.layers[0].components.size())
              : expander.stage.exports[i];
      auto expected = EvalSpatialShape(recipe.expected_outputs[i], family, source_values, indices);
      if (!expected) return Bad(expected.error());
      if (value_id >= expander.geometry.size() || expander.geometry[value_id].shape != *expected)
        return Bad("spatial recipe output geometry differs from its declared contract");
    }
  }
  spec::ModelSpec check{.pipeline = spec::PipelineKind::kImage, .stages = {expander.stage}};
  spec::Canonicalize(check);
  if (auto valid = spec::Validate(check); !valid) return Bad(valid.error());
  expander.stage = std::move(check.stages[0]);
  return spec::RecipeExpansion{.stage = std::move(expander.stage),
                               .tensors = std::move(expander.tensors)};
}

}  // namespace llmp::import
