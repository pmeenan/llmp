// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/expr.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "spec/names.h"

namespace llmp::spec {

std::optional<std::uint64_t> Env::Index(std::string_view) const { return std::nullopt; }
std::expected<std::optional<TensorDims>, std::string> Env::Tensor(std::string_view) const {
  return std::optional<TensorDims>();
}
bool Env::Fold(std::string_view) const { return false; }

Datum View(const ParamValue& value) {
  return std::visit(
      [](const auto& item) -> Datum {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::string>)
          return std::string_view(item);
        else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>)
          return std::span<const std::int64_t>(item);
        else if constexpr (std::is_same_v<T, std::vector<double>>)
          return std::span<const double>(item);
        else
          return item;
      },
      value);
}

std::expected<std::string, std::string> SubstituteIndices(std::string_view pattern,
                                                          const Env& env) {
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
    const auto index = env.Index(pattern.substr(at + 1, end - at - 1));
    if (!index) return std::unexpected("recipe template names an unbound index");
    out += std::to_string(*index);
    at = end + 1;
  }
  return out;
}

namespace {

// An evaluation's value: absent, a borrowed or owned scalar, string or list.
// Views borrow the environment's values or the expression's literals, both
// alive for the whole evaluation.
using Value = std::variant<std::monostate, std::int64_t, double, bool, std::string_view,
                           std::span<const std::int64_t>, std::span<const double>, std::string,
                           std::vector<std::int64_t>, std::vector<double>>;
using Result = std::expected<Value, std::string>;

std::unexpected<std::string> Bad(std::string why) { return std::unexpected(std::move(why)); }

bool Absent(const Value& v) { return std::holds_alternative<std::monostate>(v); }
const std::int64_t* Int(const Value& v) { return std::get_if<std::int64_t>(&v); }
const bool* Flag(const Value& v) { return std::get_if<bool>(&v); }
std::optional<double> Number(const Value& v) {
  if (const auto* i = std::get_if<std::int64_t>(&v)) return static_cast<double>(*i);
  if (const auto* d = std::get_if<double>(&v)) return *d;
  return std::nullopt;
}
std::optional<std::string_view> Text(const Value& v) {
  if (const auto* s = std::get_if<std::string_view>(&v)) return *s;
  if (const auto* s = std::get_if<std::string>(&v)) return std::string_view(*s);
  return std::nullopt;
}
std::optional<std::span<const std::int64_t>> Ints(const Value& v) {
  if (const auto* s = std::get_if<std::span<const std::int64_t>>(&v)) return *s;
  if (const auto* s = std::get_if<std::vector<std::int64_t>>(&v))
    return std::span<const std::int64_t>(*s);
  return std::nullopt;
}
std::optional<std::span<const double>> Reals(const Value& v) {
  if (const auto* s = std::get_if<std::span<const double>>(&v)) return *s;
  if (const auto* s = std::get_if<std::vector<double>>(&v)) return std::span<const double>(*s);
  return std::nullopt;
}

Value FromDatum(const Datum& d) {
  return std::visit([](const auto& item) -> Value { return item; }, d);
}

Value FromLiteral(const ParamValue& literal) { return FromDatum(View(literal)); }

ParamValue Own(const Value& v) {
  if (const auto* i = Int(v)) return *i;
  if (const auto* d = std::get_if<double>(&v)) return *d;
  if (const auto* b = Flag(v)) return *b;
  if (const auto t = Text(v)) return std::string(*t);
  if (const auto l = Ints(v)) return std::vector<std::int64_t>(l->begin(), l->end());
  if (const auto l = Reals(v)) return std::vector<double>(l->begin(), l->end());
  return std::int64_t{};
}

// Same type class and value (a borrowed string is a string).
bool Same(const Value& a, const Value& b) {
  if (const auto* x = Int(a)) return Int(b) && *x == *Int(b);
  if (const auto* x = std::get_if<double>(&a)) {
    const auto* y = std::get_if<double>(&b);
    return y && *x == *y;
  }
  if (const auto* x = Flag(a)) return Flag(b) && *x == *Flag(b);
  if (const auto x = Text(a)) return Text(b) && *x == *Text(b);
  if (const auto x = Ints(a)) return Ints(b) && std::ranges::equal(*x, *Ints(b));
  if (const auto x = Reals(a)) return Reals(b) && std::ranges::equal(*x, *Reals(b));
  return false;
}

std::uint64_t Payload(const Datum& d) {
  if (const auto* s = std::get_if<std::string_view>(&d)) return s->size();
  if (const auto* l = std::get_if<std::span<const std::int64_t>>(&d))
    return l->size() * sizeof(std::int64_t);
  if (const auto* l = std::get_if<std::span<const double>>(&d)) return l->size() * sizeof(double);
  return 0;
}

std::uint64_t TemplateBytes(std::string_view value) {
  std::uint64_t bytes = value.size();
  for (std::size_t at = 0; (at = value.find('{', at)) != std::string_view::npos; ++at) bytes += 19;
  return bytes;
}

enum class Arity : std::uint8_t { kLeaf, kUnary, kBinary, kTernary, kUnknown };

Arity ArityOf(ExprKind kind) {
  switch (kind) {
    case ExprKind::kLiteral:
    case ExprKind::kText:
    case ExprKind::kTemplate:
    case ExprKind::kAbsent:
    case ExprKind::kValue:
    case ExprKind::kIndex:
    case ExprKind::kTensorPresent:
    case ExprKind::kTensorRank:
    case ExprKind::kTensorDim:
    case ExprKind::kFold:
      return Arity::kLeaf;
    case ExprKind::kElement:
    case ExprKind::kElementReverse:
    case ExprKind::kSize:
    case ExprKind::kReverse:
    case ExprKind::kPerLayer:
    case ExprKind::kPadTo:
    case ExprKind::kSqrt:
    case ExprKind::kRsqrt:
    case ExprKind::kReciprocal:
    case ExprKind::kFloat32:
    case ExprKind::kFinite:
    case ExprKind::kNot:
    case ExprKind::kPresent:
    case ExprKind::kAsInt:
    case ExprKind::kAsReal:
    case ExprKind::kAsBool:
    case ExprKind::kAsName:
    case ExprKind::kAsText:
    case ExprKind::kAsInts:
    case ExprKind::kMatchText:
    case ExprKind::kFlagText:
      return Arity::kUnary;
    case ExprKind::kAdd:
    case ExprKind::kSubtract:
    case ExprKind::kMultiply:
    case ExprKind::kDivideExact:
    case ExprKind::kDivideFloor:
    case ExprKind::kLess:
    case ExprKind::kRealDivide:
    case ExprKind::kRealMultiply:
    case ExprKind::kQuotient:
    case ExprKind::kProduct:
    case ExprKind::kEqual:
    case ExprKind::kNotEqual:
    case ExprKind::kAnd:
    case ExprKind::kOr:
    case ExprKind::kFirst:
      return Arity::kBinary;
    case ExprKind::kSelect:
      return Arity::kTernary;
  }
  return Arity::kUnknown;
}

bool NamesSomething(ExprKind kind) {
  return kind == ExprKind::kValue || kind == ExprKind::kIndex || kind == ExprKind::kTensorPresent ||
         kind == ExprKind::kTensorRank || kind == ExprKind::kTensorDim || kind == ExprKind::kFold ||
         kind == ExprKind::kMatchText || kind == ExprKind::kFlagText;
}

// Rounds a finite real to float32 (refusing one float32 cannot hold).
std::expected<double, std::string> ToFloat32(double x) {
  if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<float>::max())
    return Bad("a value outside its transform's domain");
  return static_cast<double>(static_cast<float>(x));
}

class Evaluator {
 public:
  Evaluator(const Env& env, EvalOptions options) : env_(env), options_(options) {}

  Result Eval(const Expr& e, bool strict, std::size_t depth) {
    if (depth > kMaxExprDepth) return Bad("recipe expression exceeds nesting bounds");
    switch (ArityOf(e.kind)) {
      case Arity::kLeaf:
        return Leaf(e, strict);
      case Arity::kUnary:
        if (e.left == nullptr) return Bad("unary recipe expression is incomplete");
        return Unary(e, strict, depth);
      case Arity::kBinary:
      case Arity::kTernary:
        if (e.left == nullptr || e.right == nullptr ||
            (e.kind == ExprKind::kSelect && e.alternate == nullptr))
          return Bad("binary recipe expression is incomplete");
        return Binary(e, strict, depth);
      case Arity::kUnknown:
        break;
    }
    return Bad(std::format("unknown recipe expression kind {}", static_cast<unsigned>(e.kind)));
  }

 private:
  Result Leaf(const Expr& e, bool strict) {
    switch (e.kind) {
      case ExprKind::kLiteral:
        return FromLiteral(e.literal);
      case ExprKind::kText:
        return Value(e.name);
      case ExprKind::kTemplate: {
        const auto* text = std::get_if<std::string>(&e.literal);
        if (!text) return Bad("a template is not a string");
        auto out = SubstituteIndices(*text, env_);
        if (!out) return Bad(out.error());
        return Value(std::move(*out));
      }
      case ExprKind::kAbsent:
        return Value();
      case ExprKind::kValue: {
        if (options_.lookups != nullptr) ++*options_.lookups;
        auto d = env_.Value(e.name);
        if (!d) return Bad(d.error());
        if (std::holds_alternative<std::monostate>(*d) && strict)
          return Bad(std::format("value '{}' is absent", e.name));
        return FromDatum(*d);
      }
      case ExprKind::kIndex: {
        const auto index = env_.Index(e.name);
        if (!index || *index > static_cast<std::uint64_t>(INT64_MAX))
          return Bad("recipe index is absent or too large");
        return Value(static_cast<std::int64_t>(*index));
      }
      case ExprKind::kTensorPresent:
      case ExprKind::kTensorRank:
      case ExprKind::kTensorDim: {
        auto name = SubstituteIndices(e.name, env_);
        if (!name) return Bad(name.error());
        auto dims = env_.Tensor(*name);
        if (!dims) return Bad(dims.error());
        if (e.kind == ExprKind::kTensorPresent) return Value(dims->has_value());
        if (!*dims) {
          if (strict) return Bad(std::format("tensor '{}' is absent", *name));
          return Value();
        }
        const auto rank = static_cast<std::int64_t>((*dims)->rank);
        if (e.kind == ExprKind::kTensorRank) return Value(rank);
        const auto axis = e.fixed_index < 0 ? rank + e.fixed_index : e.fixed_index;
        if (axis < 0 || axis >= rank) {
          if (rank == 0) return Bad(std::format("tensor '{}' has rank 0", *name));
          return Bad(std::format("tensor '{}' has no dimension {}", *name, e.fixed_index));
        }
        const auto dim = (*dims)->dims[static_cast<std::size_t>(axis)];
        if (dim > static_cast<std::uint64_t>(INT64_MAX))
          return Bad(std::format("tensor '{}' has a dimension too large", *name));
        return Value(static_cast<std::int64_t>(dim));
      }
      case ExprKind::kFold:
        return Value(env_.Fold(e.name));
      default:
        break;
    }
    return Bad("unknown recipe leaf");
  }

  // The list index an element access reads (before reversal).
  std::expected<std::uint64_t, std::string> ListIndex(const Expr& e, std::size_t size) {
    std::uint64_t index = 0;
    if (e.index_name.empty()) {
      if (e.fixed_index < 0) return Bad("recipe list index is negative");
      index = static_cast<std::uint64_t>(e.fixed_index);
    } else {
      const auto bound = env_.Index(e.index_name);
      if (!bound) return Bad("recipe list index is unbound");
      if (e.fixed_index < 0) {
        const auto magnitude = static_cast<std::uint64_t>(-(e.fixed_index + 1)) + 1;
        if (magnitude > *bound) return Bad("recipe list index offset underflows");
        index = *bound - magnitude;
      } else if (__builtin_add_overflow(*bound, static_cast<std::uint64_t>(e.fixed_index),
                                        &index)) {
        return Bad("recipe list index offset overflows");
      }
    }
    if (index >= size) return Bad("recipe list index is outside its list");
    return e.kind == ExprKind::kElementReverse ? size - index - 1 : index;
  }

  Result Unary(const Expr& e, bool strict, std::size_t depth) {
    const bool lenient = e.kind == ExprKind::kPresent;
    auto operand = Eval(*e.left, strict && !lenient, depth + 1);
    if (!operand) return operand;
    Value& v = *operand;
    if (e.kind == ExprKind::kPresent) return Value(!Absent(v));
    if (Absent(v)) return Value();
    switch (e.kind) {
      case ExprKind::kElement:
      case ExprKind::kElementReverse:
      case ExprKind::kSize: {
        const auto ints = Ints(v);
        const auto reals = Reals(v);
        if (!ints && !reals) return Bad("recipe list value is absent or has the wrong type");
        const auto size = ints ? ints->size() : reals->size();
        if (e.kind == ExprKind::kSize) return Value(static_cast<std::int64_t>(size));
        auto at = ListIndex(e, size);
        if (!at) return Bad(at.error());
        if (ints) return Value((*ints)[*at]);
        return Value((*reals)[*at]);
      }
      case ExprKind::kReverse: {
        if (const auto l = Ints(v)) return Value(std::vector<std::int64_t>(l->rbegin(), l->rend()));
        if (const auto l = Reals(v)) return Value(std::vector<double>(l->rbegin(), l->rend()));
        return Bad("a reversed value that is not a list");
      }
      case ExprKind::kPerLayer: {
        const auto ints = Ints(v);
        const auto reals = Reals(v);
        if (!ints && !reals) return std::move(v);
        const auto size = ints ? ints->size() : reals->size();
        const auto layer = env_.Index("layer");
        const auto layers = env_.Index("layers");
        if (!layer || !layers || size != *layers || *layer >= size)
          return Bad(std::format("a list of {} values for {} layers", size, layers ? *layers : 0));
        if (ints) return Value((*ints)[*layer]);
        return Value((*reals)[*layer]);
      }
      case ExprKind::kPadTo: {
        const auto ints = Ints(v);
        const auto* pad = std::get_if<std::int64_t>(&e.literal);
        if (!ints || !pad) return Bad("not a list of integers");
        if (ints->size() != static_cast<std::uint64_t>(e.fixed_index)) return std::move(v);
        std::vector<std::int64_t> padded(ints->begin(), ints->end());
        padded.push_back(*pad);
        return Value(std::move(padded));
      }
      case ExprKind::kSqrt:
      case ExprKind::kRsqrt:
      case ExprKind::kReciprocal:
      case ExprKind::kFloat32:
      case ExprKind::kFinite: {
        const auto x = Number(v);
        if (!x) return Bad("a transform of a value that is not a number");
        double y = *x;
        switch (e.kind) {
          case ExprKind::kSqrt:
            y = *x >= 0.0 ? std::sqrt(*x) : std::nan("");
            break;
          case ExprKind::kRsqrt:
            y = *x > 0.0 ? 1.0 / std::sqrt(*x) : std::nan("");
            break;
          case ExprKind::kReciprocal:
            y = 1.0 / *x;
            break;
          case ExprKind::kFloat32: {
            auto f = ToFloat32(*x);
            if (!f) return Bad(f.error());
            y = *f;
            break;
          }
          default:
            break;
        }
        if (!std::isfinite(y)) return Bad("a value outside its transform's domain");
        if (e.kind == ExprKind::kFinite && y == 0.0) y = 0.0;
        return Value(y);
      }
      case ExprKind::kNot: {
        const auto* b = Flag(v);
        if (!b) return Bad("expected a boolean recipe value");
        return Value(!*b);
      }
      case ExprKind::kAsInt:
        if (Int(v)) return std::move(v);
        return Bad("not an integer");
      case ExprKind::kAsReal:
        if (const auto x = Number(v)) return Value(*x);
        return Bad("not a number");
      case ExprKind::kAsBool:
        if (Flag(v)) return std::move(v);
        return Bad("not a flag");
      case ExprKind::kAsName:
        if (const auto t = Text(v); t && names::IsName(*t)) return std::move(v);
        return Bad("not a name");
      case ExprKind::kAsText:
        if (Text(v)) return std::move(v);
        return Bad("not a string");
      case ExprKind::kAsInts:
        if (Ints(v)) return std::move(v);
        return Bad("not a list of integers");
      case ExprKind::kMatchText: {
        const auto t = Text(v);
        if (!t) return Bad("not a string");
        if (*t == e.name) return FromLiteral(e.literal);
        if (std::ranges::contains(e.names, *t)) return Value();
        return Bad(std::format("'{}' is not a value the mapping's components implement", *t));
      }
      case ExprKind::kFlagText: {
        if (const auto t = Text(v)) {
          if (*t == e.name) return FromLiteral(e.literal);
          return Bad("not the value the mapping's components implement");
        }
        if (Flag(v)) return std::move(v);
        return Bad("not a flag");
      }
      default:
        break;
    }
    return Bad("unknown unary recipe expression");
  }

  Result Binary(const Expr& e, bool strict, std::size_t depth) {
    if (e.kind == ExprKind::kFirst) {
      auto first = Eval(*e.left, false, depth + 1);
      if (!first || !Absent(*first)) return first;
      return Eval(*e.right, strict, depth + 1);
    }
    auto left = Eval(*e.left, strict, depth + 1);
    if (!left) return left;
    if (e.kind == ExprKind::kSelect) {
      if (Absent(*left)) return Value();
      const auto* condition = Flag(*left);
      if (!condition) return Bad("expected a boolean recipe value");
      return Eval(*(*condition ? e.right : e.alternate), strict, depth + 1);
    }
    auto right = Eval(*e.right, strict, depth + 1);
    if (!right) return right;
    const Value& a = *left;
    const Value& b = *right;
    if (Absent(a) || Absent(b)) return Value();
    switch (e.kind) {
      case ExprKind::kAnd:
      case ExprKind::kOr: {
        const auto* x = Flag(a);
        const auto* y = Flag(b);
        if (!x || !y) return Bad("logical recipe operands must be boolean");
        return Value(e.kind == ExprKind::kAnd ? *x && *y : *x || *y);
      }
      case ExprKind::kEqual:
        return Value(Same(a, b));
      case ExprKind::kNotEqual:
        return Value(!Same(a, b));
      case ExprKind::kRealDivide: {
        const auto x = Number(a);
        const auto y = Number(b);
        if (!x || !y || *y == 0) return Bad("real recipe division is invalid");
        const double q = *x / *y;
        if (!std::isfinite(q)) return Bad("real recipe division overflows");
        return Value(q);
      }
      case ExprKind::kRealMultiply: {
        const auto x = Number(a);
        const auto y = Number(b);
        if (!x || !y) return Bad("a product of values that are not numbers");
        return Value(*x * *y);
      }
      case ExprKind::kQuotient:
      case ExprKind::kProduct:
        return Ratio(e.kind == ExprKind::kProduct, a, b);
      default:
        break;
    }
    const auto* x = Int(a);
    const auto* y = Int(b);
    if (!x || !y) return Bad("arithmetic recipe operands must be integers");
    if (e.kind == ExprKind::kLess) return Value(*x < *y);
    if (e.kind == ExprKind::kDivideExact || e.kind == ExprKind::kDivideFloor) {
      const bool exact = e.kind == ExprKind::kDivideExact;
      if (*y == 0 || (*x == INT64_MIN && *y == -1))
        return Bad(exact ? "exact integer recipe division is invalid or non-integral"
                         : "floor integer recipe division is invalid");
      const auto quotient = *x / *y;
      const auto remainder = *x % *y;
      if (exact && remainder != 0)
        return Bad("exact integer recipe division is invalid or non-integral");
      return Value(!exact && remainder != 0 && ((*x < 0) != (*y < 0)) ? quotient - 1 : quotient);
    }
    std::int64_t result = 0;
    bool overflow = false;
    switch (e.kind) {
      case ExprKind::kAdd:
        overflow = __builtin_add_overflow(*x, *y, &result);
        break;
      case ExprKind::kSubtract:
        overflow = __builtin_sub_overflow(*x, *y, &result);
        break;
      case ExprKind::kMultiply:
        overflow = __builtin_mul_overflow(*x, *y, &result);
        break;
      default:
        return Bad("unknown binary recipe expression");
    }
    if (overflow) return Bad("recipe arithmetic overflows");
    return Value(result);
  }

  // An integer quotient or product: of integers, exact and checked; of
  // numbers otherwise, an integral real within the integers.
  static Result Ratio(bool product, const Value& a, const Value& b) {
    if (!Number(a)) return Bad("numerator is not a number");
    if (!Number(b)) return Bad("divisor is not a number");
    const auto* x = Int(a);
    const auto* y = Int(b);
    if (x && y) {
      if (product) {
        std::int64_t out = 0;
        if (__builtin_mul_overflow(*x, *y, &out)) return Bad("product overflows");
        return Value(out);
      }
      if (*y == 0) return Bad("division by zero");
      if (*x == std::numeric_limits<std::int64_t>::min() && *y == -1) return Bad("ratio overflows");
      if (*x % *y != 0) return Bad("ratio is not an integer");
      return Value(*x / *y);
    }
    const double n = *Number(a);
    const double d = *Number(b);
    if (!product && d == 0.0) return Bad("division by zero");
    const double q = product ? n * d : n / d;
    if (!std::isfinite(q) || std::floor(q) != q ||
        q < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
        q >= 9223372036854775808.0)
      return Bad(product ? "product is not an integer" : "ratio is not an integer");
    return Value(static_cast<std::int64_t>(q));
  }

  const Env& env_;
  EvalOptions options_;
};

}  // namespace

std::expected<std::optional<ParamValue>, std::string> Evaluate(const Expr& expr, const Env& env,
                                                               EvalOptions options) {
  Evaluator evaluator(env, options);
  auto value = evaluator.Eval(expr, options.strict, 0);
  if (!value) return std::unexpected(value.error());
  if (Absent(*value)) return std::optional<ParamValue>();
  return std::optional<ParamValue>(Own(*value));
}

std::expected<std::int64_t, std::string> EvaluateInt(const Expr& expr, const Env& env,
                                                     std::string_view what, std::int64_t least,
                                                     std::int64_t most, EvalOptions options) {
  options.strict = true;
  Evaluator evaluator(env, options);
  auto value = evaluator.Eval(expr, true, 0);
  if (!value) return std::unexpected(value.error());
  const auto* integer = Int(*value);
  if (!integer || *integer < least || *integer > most) return std::unexpected(std::string(what));
  return *integer;
}

std::expected<bool, std::string> EvaluateFlag(const Expr& expr, const Env& env,
                                              EvalOptions options) {
  Evaluator evaluator(env, options);
  auto value = evaluator.Eval(expr, options.strict, 0);
  if (!value) return std::unexpected(value.error());
  if (Absent(*value)) return false;
  const auto* flag = Flag(*value);
  if (!flag) return std::unexpected("expected a boolean recipe value");
  return *flag;
}

std::expected<void, std::string> VisitExpr(
    const Expr& expr, const std::function<std::expected<void, std::string>(const Expr&)>& visit) {
  std::vector<const Expr*> active;
  const auto walk = [&](auto&& self, const Expr& e,
                        std::size_t depth) -> std::expected<void, std::string> {
    if (depth > kMaxExprDepth) return Bad("recipe expression exceeds nesting bounds");
    if (std::ranges::contains(active, &e))
      return Bad("stage recipe contains a recursive expression reference");
    if (auto ok = visit(e); !ok) return ok;
    active.push_back(&e);
    for (const Expr* child : {e.left, e.right, e.alternate})
      if (child != nullptr)
        if (auto ok = self(self, *child, depth + 1); !ok) return ok;
    active.pop_back();
    return {};
  };
  return walk(walk, expr, 0);
}

std::expected<void, std::string> CheckExpr(const Expr& expr) {
  return VisitExpr(expr, [](const Expr& e) -> std::expected<void, std::string> {
    const auto arity = ArityOf(e.kind);
    const bool l = e.left != nullptr, r = e.right != nullptr, a = e.alternate != nullptr;
    switch (arity) {
      case Arity::kLeaf:
        if (l || r || a) return Bad("literal recipe expression has child pointers");
        break;
      case Arity::kUnary:
        if (!l || r || a) return Bad("unary recipe expression has invalid operands");
        break;
      case Arity::kBinary:
        if (!l || !r || a) return Bad("binary recipe expression has invalid operands");
        break;
      case Arity::kTernary:
        if (!l || !r || !a) return Bad("select recipe expression is incomplete");
        break;
      case Arity::kUnknown:
        return Bad("recipe expression has an unknown kind");
    }
    if (NamesSomething(e.kind) && e.name.empty()) return Bad("recipe reference name is invalid");
    if (e.kind == ExprKind::kTemplate && !std::holds_alternative<std::string>(e.literal))
      return Bad("a template is not a string");
    if ((e.kind == ExprKind::kElement || e.kind == ExprKind::kElementReverse) &&
        !e.index_name.empty() && e.fixed_index < -static_cast<std::int64_t>(kMaxComponents))
      return Bad("recipe list index offset exceeds graph bounds");
    if (e.kind == ExprKind::kPadTo &&
        (!std::holds_alternative<std::int64_t>(e.literal) || e.fixed_index < 0))
      return Bad("a list padding needs an integer and a length");
    return {};
  });
}

bool ReadsEnvironment(const Expr& expr) {
  bool reads = false;
  (void)VisitExpr(expr, [&](const Expr& e) -> std::expected<void, std::string> {
    reads = reads || e.kind == ExprKind::kValue || e.kind == ExprKind::kIndex ||
            e.kind == ExprKind::kTemplate || e.kind == ExprKind::kTensorPresent ||
            e.kind == ExprKind::kTensorRank || e.kind == ExprKind::kTensorDim ||
            e.kind == ExprKind::kFold || e.kind == ExprKind::kPerLayer ||
            ((e.kind == ExprKind::kElement || e.kind == ExprKind::kElementReverse) &&
             !e.index_name.empty());
    return {};
  });
  return reads;
}

std::expected<std::uint64_t, std::string> PayloadBound(const Expr& expr, const Env& env) {
  const auto walk = [&](auto&& self, const Expr& e,
                        std::size_t depth) -> std::expected<std::uint64_t, std::string> {
    if (depth > kMaxExprDepth) return Bad("recipe expression exceeds nesting bounds");
    const auto child = [&](const Expr* c) -> std::expected<std::uint64_t, std::string> {
      if (c == nullptr) return std::uint64_t{0};
      return self(self, *c, depth + 1);
    };
    switch (e.kind) {
      case ExprKind::kLiteral:
        return Payload(View(e.literal));
      case ExprKind::kText:
        return static_cast<std::uint64_t>(e.name.size());
      case ExprKind::kTemplate: {
        const auto* text = std::get_if<std::string>(&e.literal);
        return text ? TemplateBytes(*text) : 0;
      }
      case ExprKind::kValue: {
        auto d = env.Value(e.name);
        if (!d) return std::unexpected(d.error());
        return Payload(*d);
      }
      case ExprKind::kReverse:
      case ExprKind::kAsName:
      case ExprKind::kAsText:
      case ExprKind::kAsInts:
      case ExprKind::kPerLayer:
        return child(e.left);
      case ExprKind::kPadTo: {
        auto bytes = child(e.left);
        if (!bytes) return bytes;
        return *bytes + sizeof(std::int64_t);
      }
      case ExprKind::kSelect:
      case ExprKind::kFirst: {
        auto yes = child(e.kind == ExprKind::kSelect ? e.right : e.left);
        auto no = child(e.kind == ExprKind::kSelect ? e.alternate : e.right);
        if (!yes) return yes;
        if (!no) return no;
        return std::max(*yes, *no);
      }
      case ExprKind::kMatchText:
      case ExprKind::kFlagText: {
        // A flag, or the literal it gives.
        return Payload(View(e.literal));
      }
      default:
        return std::uint64_t{0};
    }
  };
  return walk(walk, expr, 0);
}

}  // namespace llmp::spec
