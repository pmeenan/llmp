// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/stage_recipe.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "components/spatial.h"
#include "components/step.h"
#include "spec/expr.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::import {
namespace {

using namespace llmp::spec;
using Status = std::expected<void, std::string>;

std::unexpected<std::string> Bad(std::string why) {
  return std::unexpected("stage recipe: " + std::move(why));
}
std::unexpected<std::string> Fail(std::string why) { return std::unexpected(std::move(why)); }

bool AddBound(std::uint64_t amount, std::uint64_t& total) {
  return !__builtin_add_overflow(total, amount, &total);
}

// Index names belong to the mapping-owned recipe AST for the full expansion
// call. Borrow those names in a flat stack so counting and repeat iterations
// never allocate per-iteration map nodes.
struct Indices {
  std::vector<std::pair<std::string_view, std::uint64_t>> entries{};

  std::optional<std::uint64_t> Find(std::string_view key) const {
    for (const auto& [name, value] : entries)
      if (name == key) return value;
    return std::nullopt;
  }
  // Binds `key` to `value`, returning what it was bound to before.
  std::optional<std::uint64_t> Bind(std::string_view key, std::uint64_t value) {
    for (auto& [name, bound] : entries)
      if (name == key) return std::exchange(bound, value);
    entries.emplace_back(key, value);
    return std::nullopt;
  }
  void Restore(std::string_view key, std::optional<std::uint64_t> previous) {
    for (std::size_t i = 0; i < entries.size(); ++i) {
      if (entries[i].first != key) continue;
      if (previous) {
        entries[i].second = *previous;
      } else {
        entries[i] = entries.back();
        entries.pop_back();
      }
      return;
    }
  }
};

}  // namespace

std::expected<Datum, std::string> DatumOf(const MetaValue& value, std::string_view key) {
  return std::visit(
      [&](const auto& item) -> std::expected<Datum, std::string> {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::monostate> || std::is_same_v<T, std::nullptr_t>)
          return Datum();
        else if constexpr (std::is_same_v<T, std::string>)
          return Datum(std::string_view(item));
        else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>)
          return Datum(std::span<const std::int64_t>(item));
        else if constexpr (std::is_same_v<T, std::vector<double>>)
          return Datum(std::span<const double>(item));
        else if constexpr (std::is_same_v<T, std::vector<std::string>>)
          return Fail(std::format("source recipe value '{}' has an unsupported type", key));
        else
          return Datum(item);
      },
      value);
}

namespace {

// One environment: the family's parameters ("<kind>.<param>"), the source's
// config keys and tensors (raw dimensions, outermost first), the bound
// indices. A name both a family parameter and a source key is ambiguous.
class RecipeEnv final : public Env {
 public:
  RecipeEnv(const ModelSpec& family, std::span<const MetaEntry> values,
            std::span<const import::Tensor> tensors, std::string_view container,
            const Indices& indices)
      : family_(family),
        values_(values),
        tensors_(tensors),
        container_(container),
        indices_(indices) {}

  std::expected<Datum, std::string> Value(std::string_view name) const override {
    const ParamValue* family = nullptr;
    if (const auto dot = name.rfind('.'); dot != std::string_view::npos) {
      const auto kind = name.substr(0, dot);
      const auto param = name.substr(dot + 1);
      for (const auto& stage : family_.stages)
        for (const auto& layer : stage.layers)
          for (const auto& component : layer.components)
            if (component.kind == kind)
              if (const auto* p = component.Find(param)) {
                if (family) return Fail("family recipe reference is ambiguous");
                family = &p->value;
              }
    }
    const MetaValue* source = nullptr;
    for (const auto& entry : values_) {
      if (entry.key != name) continue;
      if (source) return Fail("source recipe key is ambiguous");
      source = &entry.value;
    }
    if (family && source)
      return Fail(std::format("'{}' is both a family parameter and a source key", name));
    if (family) return View(*family);
    if (!source) return Datum();
    return DatumOf(*source, name);
  }
  std::optional<std::uint64_t> Index(std::string_view name) const override {
    return indices_.Find(name);
  }
  std::expected<std::optional<TensorDims>, std::string> Tensor(
      std::string_view name) const override {
    const import::Tensor* found = nullptr;
    for (const auto& tensor : tensors_) {
      if (tensor.name != name || tensor.part != format::PartRole::kData) continue;
      if (found) return Fail(std::format("source tensor '{}' is ambiguous", name));
      found = &tensor;
    }
    if (!found) return std::optional<TensorDims>();
    TensorDims dims;
    if (found->dims.size() > dims.dims.size())
      return Fail(std::format("source tensor '{}' has too many dimensions", name));
    dims.rank = static_cast<std::uint8_t>(found->dims.size());
    const bool innermost_first = container_ == "gguf";
    for (std::size_t i = 0; i < found->dims.size(); ++i)
      dims.dims[i] = found->dims[innermost_first ? found->dims.size() - 1 - i : i];
    return std::optional(dims);
  }

 private:
  const ModelSpec& family_;
  std::span<const MetaEntry> values_;
  std::span<const import::Tensor> tensors_;
  std::string_view container_;
  const Indices& indices_;
};

// The bytes a template's substitution makes with the indices bound now.
std::expected<std::uint64_t, std::string> SubstitutedBytes(std::string_view pattern,
                                                           const Indices& indices) {
  std::uint64_t bytes = 0;
  for (std::size_t at = 0; at < pattern.size();) {
    if (pattern[at] != '{') {
      ++bytes;
      ++at;
      continue;
    }
    const auto end = pattern.find('}', at + 1);
    if (end == std::string_view::npos) return Fail("recipe template has an unclosed substitution");
    const auto index = indices.Find(pattern.substr(at + 1, end - at - 1));
    if (!index) return Fail("recipe template names an unbound index");
    bytes += std::to_string(*index).size();
    at = end + 1;
  }
  return bytes;
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
    if (end == std::string_view::npos) return Fail("recipe template has an unclosed substitution");
    const auto index = indices.Find(pattern.substr(at + 1, end - at - 1));
    if (!index) return Fail("recipe template names an unbound index");
    out += std::to_string(*index);
    at = end + 1;
  }
  return out;
}

std::expected<void, std::string> CheckTemplate(std::string_view pattern,
                                               std::span<const std::string_view> indices) {
  for (std::size_t at = 0; at < pattern.size();) {
    if (pattern[at] != '{') {
      if (pattern[at] == '}') return Fail("recipe template has an unmatched brace");
      ++at;
      continue;
    }
    const auto end = pattern.find('}', at + 1);
    if (end == std::string_view::npos) return Fail("recipe template has an unclosed substitution");
    const auto key = pattern.substr(at + 1, end - at - 1);
    if (key.empty() || !std::ranges::contains(indices, key))
      return Fail("recipe template uses an index outside its lexical repeat");
    at = end + 1;
  }
  return {};
}

// --------------------------------------------------------------- the costs

struct EvaluatedWeightStorage {
  std::string role;
  std::string tensor;
  format::PartRole part = format::PartRole::kData;
  std::vector<std::uint64_t> source_dims;
};
struct BoundValue {
  std::uint16_t id = 0;
  std::optional<std::uint16_t> host_port{};
};

// A defined value name: the values map holds it as a key and the scope's
// created list as a string; a growing vector can retain both old and new
// backing storage during a geometric reallocation, so three string slots
// cover that peak.
std::uint64_t AliasCost(std::uint64_t name_bytes) {
  constexpr std::uint64_t kMapNode = sizeof(std::pair<const std::string, BoundValue>) + 64;
  constexpr std::uint64_t kVectorNode = kMapNode + 3 * sizeof(std::string);
  return kVectorNode + 2 * name_bytes;
}
// An index bound for an iteration: its stack entry and name.
constexpr std::uint64_t kIndexCost = sizeof(std::pair<const std::string, std::uint64_t>) + 64;

// What expanding one node holds: the component and its link, its value's
// name (the component, the values map and the scope's list), its
// parameters (each schema entry's copy, its name and payload, a scalar
// binding), its weights (the binding, the inventory entry, the evaluated
// contract with its raw dimensions, each name's copies) and its geometry.
// Count and Expand charge exactly this.
struct NodeCost {
  std::uint64_t name_bytes = 0, kind_bytes = 0;
  std::uint64_t param_bytes = 0;  // parameter names and payloads
  std::uint64_t params = 0, scalars = 0, scalar_bytes = 0;
  std::uint64_t weights = 0, weight_bytes = 0, raw_axes = 0;
  std::uint64_t reference_bytes = 0;  // its links' substituted names

  std::expected<std::uint64_t, std::string> Total() const {
    std::uint64_t total = sizeof(ComponentRef) + sizeof(GraphLink) + 512 +
                          sizeof(components::SpatialOperand) + kind_bytes;
    const auto add = [&](std::uint64_t count, std::uint64_t each) {
      std::uint64_t bytes = 0;
      return !__builtin_mul_overflow(count, each, &bytes) && AddBound(bytes, total);
    };
    if (!add(name_bytes, 3) || !add(params, sizeof(Param) + sizeof(ParamValue) + 128) ||
        !AddBound(param_bytes, total) || !add(scalars, sizeof(StageScalarBinding) + 64) ||
        !AddBound(scalar_bytes, total) ||
        !add(weights, sizeof(WeightBinding) + sizeof(RecipeExpansion::Tensor) + 256 +
                          4 * sizeof(std::uint64_t) + sizeof(EvaluatedWeightStorage)) ||
        !AddBound(weight_bytes, total) || !add(raw_axes, 2 * sizeof(std::uint64_t)) ||
        !AddBound(reference_bytes, total))
      return Fail("recipe node metadata bound overflows");
    return total;
  }
};

// ---------------------------------------------------------------- the walk

// A recipe walk's policy hooks: what a node, a defined value, a scope and a
// reference do. Static policies (Check, Collect) visit every step once, with
// both branches and a repeat's body under its lexical index unbound;
// evaluating ones (Count, Expand) run the actual control flow.
struct Value {
  BoundValue bound{};
};

template <class Policy>
class Walker {
 public:
  Walker(Policy& policy, const Env& env, Indices& indices)
      : policy_(policy), env_(env), indices_(indices) {}

  Status Steps(std::span<const RecipeStep> steps, std::size_t depth = 0) {
    if (depth > kMaxParams) return Fail("recipe control flow exceeds nesting bounds");
    if (steps.empty()) return {};
    if (std::ranges::contains(active_, steps.data()))
      return Fail("stage recipe contains a recursive block reference");
    active_.push_back(steps.data());
    for (const auto& step : steps) {
      if (__builtin_add_overflow(visited_, std::uint64_t{1}, &visited_) ||
          visited_ > kMaxComponents)
        return Fail("stage recipe instruction expansion exceeds spec bounds");
      Status result;
      switch (step.kind) {
        case RecipeStep::Kind::kNode:
          if (!step.node) return Fail("recipe node instruction is empty");
          result = policy_.Node(*step.node, lexical_);
          break;
        case RecipeStep::Kind::kRepeat:
          result = Repeat(step.repeat, depth);
          break;
        case RecipeStep::Kind::kIf:
          result = If(step.branch, depth);
          break;
        default:
          return Fail("recipe instruction has an unknown kind");
      }
      if (!result) return result;
    }
    active_.pop_back();
    return {};
  }

  std::uint64_t visited() const { return visited_; }

 private:
  Status Repeat(const RecipeRepeat* repeat, std::size_t depth) {
    if (!repeat || repeat->index.empty() || repeat->body.empty() || repeat->input_name.empty() ||
        repeat->body_yield.empty() || repeat->yield.empty())
      return Fail("recipe repeat instruction is incomplete");
    if (auto ok = policy_.Expression(repeat->count, true); !ok) return ok;
    if (auto ok = policy_.Template(repeat->yield, lexical_); !ok) return ok;
    auto carried = policy_.Reference(repeat->input, lexical_);
    if (!carried) return std::unexpected(carried.error());
    std::int64_t count = 1;
    if constexpr (Policy::kEvaluates) {
      auto counted =
          EvaluateInt(repeat->count, env_, "recipe repeat count exceeds the graph component bound",
                      0, static_cast<std::int64_t>(kMaxComponents), policy_.options());
      if (!counted) return std::unexpected(counted.error());
      count = *counted;
    }
    lexical_.push_back(repeat->index);
    for (std::int64_t i = 0; i < count; ++i) {
      const auto previous = Bind(repeat->index, static_cast<std::uint64_t>(i));
      const auto scope = policy_.Scope();
      Status body = policy_.Define(repeat->input_name, *carried, lexical_);
      if (body) body = Steps(repeat->body, depth + 1);
      std::expected<Value, std::string> yielded = Value{};
      if (body) yielded = policy_.Lookup(repeat->body_yield, lexical_, "recipe repeat yield");
      policy_.EndScope(scope);
      Restore(repeat->index, previous);
      if (!body) return body;
      if (!yielded) return std::unexpected(yielded.error());
      carried = *yielded;
    }
    lexical_.pop_back();
    return policy_.Define(repeat->yield, *carried, lexical_);
  }

  Status If(const RecipeIf* branch, std::size_t depth) {
    if (!branch || branch->input_name.empty() || branch->yield.empty() ||
        branch->then_yield.empty() || branch->else_yield.empty())
      return Fail("recipe conditional instruction is incomplete");
    if (auto ok = policy_.Expression(branch->condition, true); !ok) return ok;
    if (auto ok = policy_.Template(branch->yield, lexical_); !ok) return ok;
    auto input = policy_.Reference(branch->input, lexical_);
    if (!input) return std::unexpected(input.error());
    const auto arm = [&](std::span<const RecipeStep> body,
                         std::string_view yield) -> std::expected<Value, std::string> {
      const auto scope = policy_.Scope();
      Status ran = policy_.Define(branch->input_name, *input, lexical_);
      if (ran) ran = Steps(body, depth + 1);
      std::expected<Value, std::string> value = Value{};
      if (ran) value = policy_.Lookup(yield, lexical_, "recipe conditional yield");
      policy_.EndScope(scope);
      if (!ran) return std::unexpected(ran.error());
      return value;
    };
    std::expected<Value, std::string> chosen = Value{};
    if constexpr (Policy::kEvaluates) {
      auto condition = EvaluateFlag(branch->condition, env_, policy_.options());
      if (!condition) return std::unexpected(condition.error());
      chosen = *condition ? arm(branch->then_steps, branch->then_yield)
                          : arm(branch->else_steps, branch->else_yield);
    } else {
      chosen = arm(branch->then_steps, branch->then_yield);
      if (chosen) chosen = arm(branch->else_steps, branch->else_yield);
    }
    if (!chosen) return std::unexpected(chosen.error());
    return policy_.Define(branch->yield, *chosen, lexical_);
  }

  std::optional<std::uint64_t> Bind(std::string_view index, std::uint64_t value) {
    if constexpr (Policy::kEvaluates) {
      policy_.BindIndex(index);
      return indices_.Bind(index, value);
    } else {
      return std::nullopt;
    }
  }
  void Restore(std::string_view index, std::optional<std::uint64_t> previous) {
    if constexpr (Policy::kEvaluates) indices_.Restore(index, previous);
  }

  Policy& policy_;
  const Env& env_;
  Indices& indices_;
  std::vector<const RecipeStep*> active_;
  std::vector<std::string_view> lexical_;
  std::uint64_t visited_ = 0;
};

// The recipe's own expressions, outside its steps: its width and geometry.
template <class F>
Status RecipeExpressions(const StageRecipe& recipe, F&& visit) {
  if (auto ok = visit(recipe.stream_width); !ok) return ok;
  const auto shape = [&](const RecipeSpatialShape& s) -> Status {
    for (const auto* d : {&s.width, &s.height, &s.channels, &s.images})
      if (auto ok = visit(*d); !ok) return ok;
    return {};
  };
  if (auto ok = shape(recipe.primary_input.shape); !ok) return ok;
  for (const auto& port : recipe.host_ports)
    if (auto ok = shape(port.shape); !ok) return ok;
  for (const auto& output : recipe.expected_outputs)
    if (auto ok = shape(output); !ok) return ok;
  return {};
}

// A node's expressions: its parameters' values and conditions, its weights'
// raw dimensions and their condition, its layer group's index.
template <class F>
Status NodeExpressions(const RecipeNode& node, F&& visit) {
  for (const auto& param : node.params) {
    if (auto ok = visit(param.value); !ok) return ok;
    if (param.when)
      if (auto ok = visit(*param.when); !ok) return ok;
  }
  for (const auto& weight : node.weights) {
    for (const auto& dim : weight.source_dims)
      if (auto ok = visit(dim); !ok) return ok;
    if (weight.source_dims_when)
      if (auto ok = visit(*weight.source_dims_when); !ok) return ok;
  }
  if (node.group == RecipeGroupKind::kLayer) return visit(node.group_index);
  return {};
}

// ------------------------------------------------------------- Check

// Static: every expression's shape, every template's lexical indices,
// every node's required fields and request scalars; references are checked
// when evaluated (a branch a source does not take may read what it lacks).
struct CheckPolicy {
  static constexpr bool kEvaluates = false;
  Status Expression(const RecipeValue& value, bool) { return CheckExpr(value); }
  Status Template(std::string_view pattern, std::span<const std::string_view> lexical) {
    return CheckTemplate(pattern, lexical);
  }
  std::expected<Value, std::string> Reference(const RecipeRef& ref,
                                              std::span<const std::string_view> lexical) {
    if (ref.kind == RecipeRefKind::kNamedValue) {
      if (auto ok = CheckTemplate(ref.name, lexical); !ok) return std::unexpected(ok.error());
    } else if (ref.kind != RecipeRefKind::kStageInput && ref.kind != RecipeRefKind::kHostPort) {
      return Fail("recipe reference has an unknown kind");
    }
    return Value{};
  }
  Status Define(std::string_view name, Value, std::span<const std::string_view> lexical) {
    return CheckTemplate(name, lexical);
  }
  std::expected<Value, std::string> Lookup(std::string_view name,
                                           std::span<const std::string_view> lexical,
                                           std::string_view) {
    if (auto ok = CheckTemplate(name, lexical); !ok) return std::unexpected(ok.error());
    return Value{};
  }
  std::size_t Scope() { return 0; }
  void EndScope(std::size_t) {}
  Status Node(const RecipeNode& node, std::span<const std::string_view> lexical) {
    for (const auto pattern : {node.name, node.component_kind})
      if (auto ok = CheckTemplate(pattern, lexical); !ok) return ok;
    for (const auto* ref : {&node.input, node.has_auxiliary ? &node.auxiliary : nullptr,
                            node.has_carry ? &node.carry : nullptr})
      if (ref)
        if (auto ok = Reference(*ref, lexical); !ok) return std::unexpected(ok.error());
    for (const auto& param : node.params) {
      if (auto ok = CheckTemplate(param.name, lexical); !ok) return ok;
      if (!param.request_scalar.empty() && !spec::IsStageScalarName(param.request_scalar))
        return Fail(std::format("recipe request scalar {} is not one the runtime supplies",
                                param.request_scalar));
    }
    for (const auto& weight : node.weights) {
      if (weight.role.empty() || weight.tensor.empty())
        return Fail("recipe weight binding is incomplete");
      for (const auto pattern : {weight.role, weight.tensor, weight.stage})
        if (auto ok = CheckTemplate(pattern, lexical); !ok) return ok;
    }
    if (node.params.size() > kMaxParams)
      return Fail("recipe node exceeds the component parameter bound");
    return NodeExpressions(node, [](const RecipeValue& value) { return CheckExpr(value); });
  }
};

// ------------------------------------------------------------- Collect

struct CollectPolicy {
  static constexpr bool kEvaluates = false;
  std::set<std::string_view> names;
  Status Expression(const RecipeValue& value, bool) {
    return VisitExpr(value, [&](const Expr& e) -> Status {
      if (e.kind == ExprKind::kValue) names.insert(e.name);
      return {};
    });
  }
  Status Template(std::string_view, std::span<const std::string_view>) { return {}; }
  std::expected<Value, std::string> Reference(const RecipeRef&, std::span<const std::string_view>) {
    return Value{};
  }
  Status Define(std::string_view, Value, std::span<const std::string_view>) { return {}; }
  std::expected<Value, std::string> Lookup(std::string_view, std::span<const std::string_view>,
                                           std::string_view) {
    return Value{};
  }
  std::size_t Scope() { return 0; }
  void EndScope(std::size_t) {}
  Status Node(const RecipeNode& node, std::span<const std::string_view>) {
    return NodeExpressions(node, [&](const RecipeValue& value) { return Expression(value, true); });
  }
};

// The flag a parameter's or contract's condition gives (none: it holds).
std::expected<bool, std::string> Holds(const RecipeValue* when, const Env& env,
                                       EvalOptions options) {
  if (when == nullptr) return true;
  return EvaluateFlag(*when, env, options);
}

// A parameter value's bytes of payload, which must be a scalar's for a
// request-bound one.
std::expected<std::uint64_t, std::string> ParamPayload(const RecipeValue& value, const Env& env) {
  return PayloadBound(value, env);
}

// The cost of expanding `node` with the indices bound now, and its
// tensor bindings' facts.
struct WeightFacts {
  std::uint64_t tensor_bytes = 0, role_bytes = 0, raw_axes = 0;
  bool contract = false;
};
std::expected<NodeCost, std::string> CostOf(const RecipeNode& node, const Env& env,
                                            const Indices& indices, EvalOptions options,
                                            std::vector<WeightFacts>* weights) {
  NodeCost cost;
  auto name = SubstitutedBytes(node.name, indices);
  if (!name) return std::unexpected(name.error());
  cost.name_bytes = *name;
  cost.kind_bytes = node.component_kind.size();
  for (const auto* ref : {&node.input, node.has_auxiliary ? &node.auxiliary : nullptr,
                          node.has_carry ? &node.carry : nullptr}) {
    if (ref == nullptr || ref->kind != RecipeRefKind::kNamedValue) continue;
    auto bytes = SubstitutedBytes(ref->name, indices);
    if (!bytes) return std::unexpected(bytes.error());
    if (!AddBound(*bytes, cost.reference_bytes))
      return Fail("recipe graph-reference metadata bound overflows");
  }
  for (const auto& param : node.params) {
    auto holds = Holds(param.when, env, options);
    if (!holds) return std::unexpected(holds.error());
    if (!*holds) continue;
    auto key = SubstitutedBytes(param.name, indices);
    auto payload = ParamPayload(param.value, env);
    if (!key) return std::unexpected(key.error());
    if (!payload) return std::unexpected(payload.error());
    ++cost.params;
    if (!AddBound(*key + 1, cost.param_bytes) || !AddBound(*payload, cost.param_bytes))
      return Fail("recipe parameter metadata bound overflows");
    if (!param.request_scalar.empty()) {
      ++cost.scalars;
      if (!AddBound(*key + param.request_scalar.size(), cost.scalar_bytes))
        return Fail("recipe scalar metadata bound overflows");
    }
  }
  for (const auto& weight : node.weights) {
    auto role = SubstitutedBytes(weight.role, indices);
    auto tensor = SubstitutedBytes(weight.tensor, indices);
    auto stage = SubstitutedBytes(weight.stage, indices);
    if (!role) return std::unexpected(role.error());
    if (!tensor) return std::unexpected(tensor.error());
    if (!stage) return std::unexpected(stage.error());
    auto contracted = Holds(weight.source_dims_when, env, options);
    if (!contracted) return std::unexpected(contracted.error());
    const auto axes = *contracted ? weight.source_dims.size() : 0;
    ++cost.weights;
    cost.raw_axes += axes;
    if (!AddBound(2 * *role + 3 * *tensor + *stage, cost.weight_bytes))
      return Fail("recipe tensor metadata bound overflows");
    if (weights)
      weights->push_back(
          {.tensor_bytes = *tensor, .role_bytes = *role, .raw_axes = axes, .contract = axes != 0});
  }
  return cost;
}

// ------------------------------------------------------------- Count

// The expansion over the actual source without building it: what each node
// and defined name would hold, and what its scans would compare.
struct CountPolicy {
  static constexpr bool kEvaluates = true;
  const Env& env;
  const Indices& indices;
  std::uint64_t lookups = 0;
  std::uint64_t metadata = 0;
  StageRecipePreflight out{};

  EvalOptions options() { return {.strict = true, .lookups = &lookups}; }
  Status Expression(const RecipeValue& value, bool scalar) {
    auto payload = PayloadBound(value, env);
    if (!payload) return std::unexpected(payload.error());
    if (scalar && *payload != 0) return Fail("recipe control value must be a scalar expression");
    return {};
  }
  Status Template(std::string_view, std::span<const std::string_view>) { return {}; }
  std::expected<Value, std::string> Reference(const RecipeRef& ref,
                                              std::span<const std::string_view>) {
    if (ref.kind == RecipeRefKind::kNamedValue) {
      auto bytes = SubstitutedBytes(ref.name, indices);
      if (!bytes) return std::unexpected(bytes.error());
      if (!AddBound(*bytes, metadata)) return Fail("recipe reference metadata bound overflows");
    }
    return Value{};
  }
  Status Define(std::string_view name, Value, std::span<const std::string_view>) {
    auto bytes = SubstitutedBytes(name, indices);
    if (!bytes) return std::unexpected(bytes.error());
    if (!AddBound(AliasCost(*bytes), metadata))
      return Fail("recipe alias metadata bound overflows");
    return {};
  }
  std::expected<Value, std::string> Lookup(std::string_view name, std::span<const std::string_view>,
                                           std::string_view) {
    auto bytes = SubstitutedBytes(name, indices);
    if (!bytes) return std::unexpected(bytes.error());
    if (!AddBound(*bytes, metadata)) return Fail("recipe yield temporary bound overflows");
    return Value{};
  }
  std::size_t Scope() { return 0; }
  void EndScope(std::size_t) {}
  void BindIndex(std::string_view index) { (void)AddBound(kIndexCost + index.size(), metadata); }

  Status Node(const RecipeNode& node, std::span<const std::string_view>) {
    if (auto ok = NodeExpressions(node, [&](const RecipeValue& v) { return Expression(v, false); });
        !ok)
      return ok;
    std::vector<WeightFacts> weights;
    auto cost = CostOf(node, env, indices, options(), &weights);
    if (!cost) return std::unexpected(cost.error());
    auto total = cost->Total();
    if (!total || !AddBound(*total, metadata))
      return Fail(total ? "recipe metadata bound overflows" : total.error());
    if (__builtin_add_overflow(out.expanded_nodes, std::uint64_t{1}, &out.expanded_nodes) ||
        out.expanded_nodes > kMaxComponents)
      return Fail("stage recipe exceeds the graph component bound");
    std::uint64_t node_roles = 0;
    for (const auto& w : weights) node_roles += w.role_bytes + w.tensor_bytes;
    for (const auto& w : weights) {
      // Its shared-tensor identity scan of the bindings before it, and its
      // evaluated contract's lookup among the node's own weights.
      std::uint64_t identity = 0, lookup = 0;
      if (__builtin_mul_overflow(out.tensor_bindings, w.tensor_bytes + 1, &identity) ||
          __builtin_mul_overflow(static_cast<std::uint64_t>(weights.size()), node_roles + 6,
                                 &lookup) ||
          !AddBound(identity, out.expansion_work) || !AddBound(lookup, out.expansion_work))
        return Fail("recipe expansion work bound overflows");
      ++out.tensor_bindings;
      if (!AddBound(w.tensor_bytes, out.tensor_name_bytes))
        return Fail("recipe tensor-name byte bound overflows");
      out.max_tensor_name_bytes = std::max(out.max_tensor_name_bytes, w.tensor_bytes);
      out.max_weight_role_bytes = std::max(out.max_weight_role_bytes, w.role_bytes);
      out.raw_dimension_axes += w.raw_axes;
      out.raw_dimension_contracts += w.contract ? 1 : 0;
    }
    return {};
  }
};

// ------------------------------------------------------------- Expand

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

std::expected<components::SpatialShape, std::string> EvalShape(const RecipeSpatialShape& shape,
                                                               const Env& env,
                                                               EvalOptions options) {
  std::array<std::uint64_t, 4> dims{};
  const std::array values{&shape.width, &shape.height, &shape.channels, &shape.images};
  for (std::size_t i = 0; i < values.size(); ++i) {
    auto payload = PayloadBound(*values[i], env);
    if (!payload || *payload != 0)
      return Fail(payload ? "spatial recipe dimensions must be scalar expressions"
                          : payload.error());
    auto value = EvaluateInt(*values[i], env, "spatial recipe dimensions must be positive integers",
                             1, std::numeric_limits<std::int64_t>::max(), options);
    if (!value) return std::unexpected(value.error());
    dims[i] = static_cast<std::uint64_t>(*value);
  }
  components::SpatialShape out{dims[0], dims[1], dims[2], dims[3]};
  if (!components::SpatialElements(out))
    return Fail("spatial recipe shape exceeds representable storage");
  return out;
}

struct ExpandPolicy {
  static constexpr bool kEvaluates = true;
  const Env& env;
  const Indices& indices;
  std::uint64_t paid;
  std::uint64_t estimate = 0;
  std::uint64_t lookups = 0;
  Stage stage;
  std::map<std::string, BoundValue, std::less<>> values;
  std::vector<components::SpatialOperand> geometry;
  std::vector<components::SpatialOperand> host_ports;
  std::vector<std::string> created;
  std::vector<RecipeExpansion::Tensor> tensors;

  EvalOptions options() { return {.strict = true, .lookups = &lookups}; }
  bool Charge(std::uint64_t amount) { return AddBound(amount, estimate) && estimate <= paid; }
  Status Expression(const RecipeValue&, bool) { return {}; }
  Status Template(std::string_view, std::span<const std::string_view>) { return {}; }
  void BindIndex(std::string_view) {}

  std::expected<Value, std::string> Reference(const RecipeRef& ref,
                                              std::span<const std::string_view>) const {
    if (ref.kind == RecipeRefKind::kStageInput) return Value{};
    if (ref.kind == RecipeRefKind::kHostPort) return Value{.bound = {.host_port = ref.port}};
    auto name = Substitute(ref.name, indices);
    if (!name) return std::unexpected(name.error());
    const auto it = values.find(*name);
    if (it == values.end()) return Fail("recipe link refers to an unavailable value");
    return Value{.bound = it->second};
  }
  Status Define(std::string_view pattern, Value value, std::span<const std::string_view>) {
    auto name = Substitute(pattern, indices);
    if (!name) return std::unexpected(name.error());
    if (values.contains(*name)) return Fail("recipe value name is defined more than once");
    created.push_back(*name);
    values.emplace(std::move(*name), value.bound);
    return {};
  }
  std::expected<Value, std::string> Lookup(std::string_view pattern,
                                           std::span<const std::string_view>,
                                           std::string_view what) const {
    auto name = Substitute(pattern, indices);
    if (!name) return std::unexpected(name.error());
    const auto it = values.find(*name);
    if (it == values.end()) return Fail(std::format("{} is unavailable", what));
    return Value{.bound = it->second};
  }
  std::size_t Scope() const { return created.size(); }
  void EndScope(std::size_t mark) {
    for (std::size_t i = mark; i < created.size(); ++i) values.erase(created[i]);
    created.resize(mark);
  }

  std::expected<components::SpatialOperand, std::string> Geometry(const BoundValue& value) const {
    if (value.host_port) {
      if (*value.host_port >= host_ports.size())
        return Fail("recipe spatial host port is unavailable");
      return host_ports[*value.host_port];
    }
    if (value.id >= geometry.size()) return Fail("recipe spatial value geometry is unavailable");
    return geometry[value.id];
  }

  Status Node(const RecipeNode& node, std::span<const std::string_view> lexical) {
    if (stage.layers.empty()) stage.layers.emplace_back();
    auto cost = CostOf(node, env, indices, options(), nullptr);
    if (!cost) return std::unexpected(cost.error());
    auto total = cost->Total();
    if (!total || !Charge(*total)) return Fail("recipe exceeds its paid metadata bound");
    auto name = Substitute(node.name, indices);
    if (!name) return std::unexpected(name.error());
    if (name->empty()) return Fail("recipe node name is empty");
    if (values.contains(*name)) return Fail("recipe defines a value more than once");
    if (stage.layers[0].components.size() >= kMaxComponents ||
        stage.layers[0].components.size() >= std::numeric_limits<std::uint16_t>::max())
      return Fail("recipe exceeds the stage component/value-id bound");
    ComponentRef component{.kind = std::string(node.component_kind), .version = node.version};
    component.params.reserve(node.params.size());
    for (const auto& parameter : node.params) {
      auto holds = Holds(parameter.when, env, options());
      if (!holds) return std::unexpected(holds.error());
      if (!*holds) continue;
      auto key = Substitute(parameter.name, indices);
      if (!key) return std::unexpected(key.error());
      auto value = Evaluate(parameter.value, env, options());
      if (!value) return std::unexpected(value.error());
      if (!*value) return Fail("recipe parameter value is absent");
      component.params.push_back({.name = std::move(*key), .value = std::move(**value)});
      if (!parameter.request_scalar.empty()) {
        if (!std::holds_alternative<std::int64_t>(component.params.back().value))
          return Fail("request-bound recipe parameter must have an integer default");
        // Stage scalar component IDs are one-based output value IDs.
        stage.scalar_bindings.push_back(
            {.component = static_cast<std::uint16_t>(stage.layers[0].components.size() + 1),
             .parameter = component.params.back().name,
             .source = std::string(parameter.request_scalar)});
      }
    }
    std::vector<EvaluatedWeightStorage> evaluated;
    evaluated.reserve(node.weights.size());
    component.weights.reserve(node.weights.size());
    for (const auto& weight : node.weights) {
      auto role = Substitute(weight.role, indices);
      auto tensor = Substitute(weight.tensor, indices);
      auto stage_name = Substitute(weight.stage, indices);
      if (!role) return std::unexpected(role.error());
      if (!tensor) return std::unexpected(tensor.error());
      if (!stage_name) return std::unexpected(stage_name.error());
      auto contracted = Holds(weight.source_dims_when, env, options());
      if (!contracted) return std::unexpected(contracted.error());
      std::vector<std::uint64_t> dims;
      if (*contracted) {
        dims.reserve(weight.source_dims.size());
        for (const auto& expression : weight.source_dims) {
          auto payload = PayloadBound(expression, env);
          if (!payload || *payload != 0)
            return Fail("recipe raw source dimension must be a scalar expression");
          auto dim =
              EvaluateInt(expression, env, "recipe raw source dimensions must be positive integers",
                          1, std::numeric_limits<std::int64_t>::max(), options());
          if (!dim) return std::unexpected(dim.error());
          dims.push_back(static_cast<std::uint64_t>(*dim));
        }
      }
      evaluated.push_back(
          {.role = *role, .tensor = *tensor, .part = weight.part, .source_dims = std::move(dims)});
      component.weights.push_back({.role = std::move(*role),
                                   .tensor = std::move(*tensor),
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
      return Fail("recipe assigns one component weight role more than once");
    const auto component_index = static_cast<std::uint16_t>(stage.layers[0].components.size());
    const components::StageFacts facts{
        .width = stage.stream.width, .context = stage.context, .streams = stage.stream.streams};
    auto instance =
        components::MakeInstance(component, facts, components::Site::kLayer, 0, component_index);
    if (!instance) return std::unexpected(instance.error());
    auto role_shapes = instance->def->shapes(*instance);
    if (!role_shapes) return std::unexpected(role_shapes.error());
    std::uint32_t group_index = 0;
    if (node.group == RecipeGroupKind::kLayer && !component.weights.empty()) {
      auto index = EvaluateInt(node.group_index, env, "recipe component group index is invalid", 0,
                               UINT32_MAX, options());
      if (!index) return std::unexpected(index.error());
      group_index = static_cast<std::uint32_t>(*index);
    }
    for (const auto& weight : component.weights) {
      const auto role = components::RoleIndex(*instance->def, weight.role);
      if (!role) return Fail("recipe binds an unknown component weight role");
      const auto shape = std::ranges::find(*role_shapes, *role, &components::RoleShape::role);
      if (shape == role_shapes->end())
        return Fail("recipe binds a component role with no source shape");
      std::vector<std::uint64_t> dimensions;
      dimensions.reserve(shape->rank);
      for (std::size_t d = shape->rank; d > 0; --d) dimensions.push_back(shape->ne[d - 1]);
      const auto previous = std::ranges::find_if(tensors, [&](const RecipeExpansion::Tensor& t) {
        return t.name == weight.tensor && t.part == weight.part;
      });
      const auto declared = std::ranges::find_if(evaluated, [&](const EvaluatedWeightStorage& w) {
        return w.role == weight.role && w.tensor == weight.tensor && w.part == weight.part;
      });
      if (declared == evaluated.end())
        return Fail("recipe weight lost its evaluated source-dimension contract");
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
        return Fail("shared recipe tensor has inconsistent role-derived binding");
      else if (!declared->source_dims.empty()) {
        if (!previous->source_dims.empty() && previous->source_dims != declared->source_dims)
          return Fail("shared recipe tensor has conflicting raw source dimensions");
        if (previous->source_dims.empty()) previous->source_dims = std::move(declared->source_dims);
      }
    }
    GraphLink link;
    std::optional<components::SpatialOperand> operands[3];
    const RecipeRef* refs[3] = {&node.input, node.has_auxiliary ? &node.auxiliary : nullptr,
                                node.has_carry ? &node.carry : nullptr};
    for (unsigned slot = 0; slot < 3; ++slot) {
      if (refs[slot] == nullptr) continue;
      auto value = Reference(*refs[slot], lexical);
      if (!value) return std::unexpected(value.error());
      Attach(value->bound, link, slot);
      auto resolved = Geometry(value->bound);
      if (!resolved) return std::unexpected(resolved.error());
      operands[slot] = *resolved;
    }
    auto output = components::InferSpatialOutput(*instance, *operands[0], operands[1], operands[2]);
    if (!output) return std::unexpected(output.error());
    stage.layers[0].components.push_back(std::move(component));
    stage.links.push_back(link);
    geometry.push_back(*output);
    created.push_back(*name);
    values.emplace(std::move(*name),
                   BoundValue{.id = static_cast<std::uint16_t>(stage.layers[0].components.size())});
    return {};
  }
};

// ------------------------------------------------------------- the calls

struct Source {
  std::span<const MetaEntry> values;
  std::span<const Tensor> tensors;
  std::string_view container;
};

Status Check(const StageRecipe& recipe) {
  if (recipe.name.empty()) return Fail("stage recipe header is invalid");
  if (auto ok = RecipeExpressions(recipe, [](const RecipeValue& v) { return CheckExpr(v); }); !ok)
    return ok;
  for (const auto& output : recipe.exports)
    if (output.name.empty() && output.kind == RecipeRefKind::kNamedValue)
      return Fail("stage export has an empty recipe name");
  const auto encoding_known = [](format::Encoding encoding) {
    return format::PartsOf({.encoding = encoding,
                            .layout = format::Layout::kDense,
                            .shape = ir::ShapeOf({1, 1, 1, 1})})
        .has_value();
  };
  if (!encoding_known(recipe.primary_input.encoding))
    return Fail("spatial recipe port encoding is unknown");
  for (const auto& port : recipe.host_ports)
    if (!encoding_known(port.encoding)) return Fail("spatial recipe port encoding is unknown");
  const auto output_count = recipe.exports.empty() ? std::size_t{1} : recipe.exports.size();
  if (!recipe.expected_outputs.empty() && recipe.expected_outputs.size() != output_count)
    return Fail("expected spatial output count differs from recipe exports");
  CheckPolicy policy;
  const ModelSpec none{};
  Indices indices;
  const RecipeEnv env(none, {}, {}, {}, indices);
  Walker<CheckPolicy> walker(policy, env, indices);
  return walker.Steps(recipe.steps);
}

// Counting: the recipe's geometry and width, then the expansion's walk.
std::expected<StageRecipePreflight, std::string> Count(const StageRecipe& recipe,
                                                       const ModelSpec& family,
                                                       const Source& source) {
  if (auto ok = Check(recipe); !ok) return std::unexpected(ok.error());
  Indices indices;
  const RecipeEnv env(family, source.values, source.tensors, source.container, indices);
  CountPolicy policy{.env = env, .indices = indices, .lookups = 0, .metadata = 4096, .out = {}};
  auto width_payload = PayloadBound(recipe.stream_width, env);
  if (!width_payload || *width_payload != 0)
    return Fail(width_payload ? "stage stream width must be a scalar expression"
                              : width_payload.error());
  auto width = EvaluateInt(recipe.stream_width, env, "stage recipe stream width is invalid", 1,
                           static_cast<std::int64_t>(kMaxWidth), policy.options());
  if (!width) return std::unexpected(width.error());
  const auto shape = [&](const RecipeSpatialShape& s) -> Status {
    auto evaluated = EvalShape(s, env, policy.options());
    if (!evaluated) return std::unexpected(evaluated.error());
    return {};
  };
  if (auto ok = shape(recipe.primary_input.shape); !ok) return std::unexpected(ok.error());
  for (const auto& port : recipe.host_ports)
    if (auto ok = shape(port.shape); !ok) return std::unexpected(ok.error());
  for (const auto& output : recipe.expected_outputs)
    for (const auto* d : {&output.width, &output.height, &output.channels, &output.images}) {
      auto payload = PayloadBound(*d, env);
      if (!payload || *payload != 0)
        return Fail(payload ? "spatial recipe dimensions must be scalar expressions"
                            : payload.error());
    }
  Walker<CountPolicy> walker(policy, env, indices);
  if (auto ok = walker.Steps(recipe.steps); !ok) return std::unexpected(ok.error());
  auto& out = policy.out;
  std::uint64_t geometry_slots = 0, geometry_bytes = 0;
  auto& metadata = policy.metadata;
  if (__builtin_add_overflow(out.expanded_nodes, std::uint64_t{1}, &geometry_slots) ||
      __builtin_mul_overflow(geometry_slots, std::uint64_t{sizeof(components::SpatialOperand)},
                             &geometry_bytes) ||
      !AddBound(geometry_bytes, metadata) ||
      __builtin_mul_overflow(static_cast<std::uint64_t>(recipe.host_ports.size()),
                             std::uint64_t{sizeof(components::SpatialOperand)}, &geometry_bytes) ||
      !AddBound(geometry_bytes, metadata) ||
      __builtin_mul_overflow(static_cast<std::uint64_t>(recipe.expected_outputs.size()),
                             std::uint64_t{sizeof(components::SpatialShape)}, &geometry_bytes) ||
      !AddBound(geometry_bytes, metadata) || !AddBound(recipe.name.size(), metadata))
    return Fail("spatial recipe geometry metadata bound overflows");
  for (const auto& output : recipe.exports) {
    auto bytes = SubstitutedBytes(output.name, indices);
    if (!bytes || !AddBound(sizeof(std::uint16_t) + *bytes + 128, metadata))
      return Fail("stage export metadata bound overflows");
  }
  // The names an evaluation reads, the longest the source's metadata is
  // compared against.
  CollectPolicy names;
  Walker<CollectPolicy> collect(names, env, indices);
  if (auto ok = RecipeExpressions(recipe,
                                  [&](const RecipeValue& v) { return names.Expression(v, true); });
      !ok)
    return std::unexpected(ok.error());
  if (auto ok = collect.Steps(recipe.steps); !ok) return std::unexpected(ok.error());
  for (const auto name : names.names)
    out.max_source_key_bytes =
        std::max(out.max_source_key_bytes, static_cast<std::uint64_t>(name.size()));
  out.visited_steps = walker.visited();
  // Counting's reads, then expansion's: its payload bounds and evaluations
  // read no more than counting did, twice over.
  if (__builtin_mul_overflow(policy.lookups, std::uint64_t{3}, &out.source_value_lookups))
    return Fail("stage recipe source lookup work overflows");
  out.metadata_bytes = metadata;
  out.stream_width = static_cast<std::uint64_t>(*width);
  return out;
}

}  // namespace

std::expected<StageRecipePreflight, std::string> PreflightStageRecipe(
    const spec::StageRecipe& recipe, const spec::ModelSpec& family, std::uint64_t metadata_bytes,
    std::span<const MetaEntry> source_values, std::span<const Tensor> source_tensors,
    std::string_view source_container) {
  if (metadata_bytes == 0) return Bad("stage recipe header or prepaid metadata bound is invalid");
  auto counted = Count(recipe, family, {source_values, source_tensors, source_container});
  if (!counted) return Bad(counted.error());
  if (counted->metadata_bytes > metadata_bytes)
    return Bad("stage recipe exceeds its prepaid metadata bound");
  return counted;
}

std::expected<spec::RecipeExpansion, std::string> ExpandStageRecipe(
    const spec::StageRecipe& recipe, const spec::ModelSpec& family, std::uint64_t metadata_bytes,
    std::span<const MetaEntry> source_values, std::span<const Tensor> source_tensors,
    std::string_view source_container) {
  auto preflight = PreflightStageRecipe(recipe, family, metadata_bytes, source_values,
                                        source_tensors, source_container);
  if (!preflight) return std::unexpected(preflight.error());
  Indices indices;
  const RecipeEnv env(family, source_values, source_tensors, source_container, indices);
  ExpandPolicy expander{.env = env,
                        .indices = indices,
                        .paid = metadata_bytes,
                        .estimate = 0,
                        .lookups = 0,
                        .stage = {.name = std::string(recipe.name),
                                  .role = recipe.role,
                                  .artifact = recipe.artifact,
                                  .stream = {.width = preflight->stream_width}},
                        .values = {},
                        .geometry = {},
                        .host_ports = {},
                        .created = {},
                        .tensors = {}};
  expander.geometry.reserve(static_cast<std::size_t>(preflight->expanded_nodes + 1));
  expander.host_ports.reserve(recipe.host_ports.size());
  auto primary = EvalShape(recipe.primary_input.shape, env, expander.options());
  if (!primary) return Bad(primary.error());
  expander.geometry.push_back({.shape = *primary, .encoding = recipe.primary_input.encoding});
  for (const auto& host : recipe.host_ports) {
    auto shape = EvalShape(host.shape, env, expander.options());
    if (!shape) return Bad(shape.error());
    expander.host_ports.push_back({.shape = *shape, .encoding = host.encoding});
  }
  Walker<ExpandPolicy> walker(expander, env, indices);
  if (auto expanded = walker.Steps(recipe.steps); !expanded) return Bad(expanded.error());
  if (expander.stage.layers.empty() || expander.stage.layers[0].components.empty())
    return Bad("stage recipe produced no components");
  for (const auto& output : recipe.exports) {
    auto value = expander.Reference(output, {});
    if (!value) return Bad(value.error());
    if (value->bound.host_port || value->bound.id == 0)
      return Bad("a stage export must name a computed graph value");
    expander.stage.exports.push_back(value->bound.id);
  }
  for (std::size_t i = 0; i < recipe.expected_outputs.size(); ++i) {
    // A geometry the source does not state is not checked.
    const auto& shape = recipe.expected_outputs[i];
    bool stated = true;
    for (const auto* d : {&shape.width, &shape.height, &shape.channels, &shape.images}) {
      auto value = Evaluate(*d, env, {.strict = false});
      if (!value) return Bad(value.error());
      stated = stated && value->has_value();
    }
    if (!stated) continue;
    auto expected = EvalShape(shape, env, expander.options());
    if (!expected) return Bad(expected.error());
    const auto value_id =
        recipe.exports.empty()
            ? static_cast<std::uint16_t>(expander.stage.layers[0].components.size())
            : expander.stage.exports[i];
    if (value_id >= expander.geometry.size() || expander.geometry[value_id].shape != *expected)
      return Bad("spatial recipe output geometry differs from its declared contract");
  }
  spec::ModelSpec check{.pipeline = spec::PipelineKind::kImage, .stages = {expander.stage}};
  spec::Canonicalize(check);
  if (auto valid = spec::Validate(check); !valid) return Bad(valid.error());
  return spec::RecipeExpansion{.stage = std::move(check.stages[0]),
                               .tensors = std::move(expander.tensors)};
}

std::expected<std::set<std::string_view>, std::string> RecipeValueNames(
    const spec::StageRecipe& recipe) {
  if (auto ok = Check(recipe); !ok) return Bad(ok.error());
  CollectPolicy policy;
  Indices indices;
  const ModelSpec none{};
  const RecipeEnv env(none, {}, {}, {}, indices);
  if (auto ok = RecipeExpressions(recipe,
                                  [&](const RecipeValue& v) { return policy.Expression(v, true); });
      !ok)
    return Bad(ok.error());
  Walker<CollectPolicy> walker(policy, env, indices);
  if (auto ok = walker.Steps(recipe.steps); !ok) return Bad(ok.error());
  return std::move(policy.names);
}

}  // namespace llmp::import
