// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The Jinja-subset evaluator (jinja.h): statements, expressions, and the
// fixed set of filters, tests, methods and globals. Semantics follow Jinja2
// 3.1 under transformers' ImmutableSandboxedEnvironment and Python's own
// for the methods; docs/tokenizer.md records where the subset is narrower
// (integers are 64-bit, some format and strftime directives are refused).
// Case mapping is Python's in full (chat/pycase.h).

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/json.h"
#include "chat/jinja.h"
#include "chat/jinja_internal.h"
#include "chat/pycase.h"

namespace jitllm::chat::jinja {
namespace {

constexpr auto kFilters = std::to_array<std::string_view>(
    {"abs",    "capitalize", "count",      "d",          "default",   "dictsort", "e",
     "escape", "first",      "float",      "format",     "from_json", "indent",   "int",
     "items",  "join",       "last",       "length",     "list",      "lower",    "map",
     "max",    "min",        "reject",     "rejectattr", "replace",   "reverse",  "round",
     "safe",   "select",     "selectattr", "sort",       "string",    "sum",      "title",
     "tojson", "trim",       "unique",     "upper"});

constexpr auto kTests = std::to_array<std::string_view>(
    {"boolean", "callable",  "defined", "divisibleby", "eq",     "equalto",     "escaped",
     "even",    "false",     "float",   "ge",          "gt",     "greaterthan", "in",
     "integer", "iterable",  "le",      "lessthan",    "lower",  "lt",          "mapping",
     "ne",      "none",      "number",  "odd",         "sameas", "sequence",    "string",
     "true",    "undefined", "upper",   "==",          "!=",     "<",           "<=",
     ">",       ">="});

constexpr auto kDictMethods =
    std::to_array<std::string_view>({"clear", "copy", "fromkeys", "get", "items", "keys", "pop",
                                     "popitem", "setdefault", "update", "values"});
constexpr auto kListMethods =
    std::to_array<std::string_view>({"append", "clear", "copy", "count", "extend", "index",
                                     "insert", "pop", "remove", "reverse", "sort"});
constexpr auto kStrMethods = std::to_array<std::string_view>(
    {"capitalize",   "casefold",    "center",    "count",      "encode",       "endswith",
     "expandtabs",   "find",        "format",    "format_map", "index",        "isalnum",
     "isalpha",      "isascii",     "isdecimal", "isdigit",    "isidentifier", "islower",
     "isnumeric",    "isprintable", "isspace",   "istitle",    "isupper",      "join",
     "ljust",        "lower",       "lstrip",    "maketrans",  "partition",    "removeprefix",
     "removesuffix", "replace",     "rfind",     "rindex",     "rjust",        "rpartition",
     "rsplit",       "rstrip",      "split",     "splitlines", "startswith",   "strip",
     "swapcase",     "title",       "translate", "upper",      "zfill"});

template <std::size_t N>
bool In(const std::array<std::string_view, N>& names, std::string_view name) {
  return std::ranges::find(names, name) != names.end();
}

std::unexpected<Error> Fail(Code code, std::string_view reason, std::uint32_t line = 0) {
  return std::unexpected(Error{code, reason, line});
}

// A variable scope: a template's top level, one loop iteration or one
// macro call.
struct Scope final : Object {
  Scope() = default;
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  Scope(Scope&&) = delete;
  Scope& operator=(Scope&&) = delete;
  ~Scope() override {  // without nesting, as List's
    for (auto& [name, v] : vars) {
      ReleaseDeferred(v);
    }
    ReleaseDeferred(std::static_pointer_cast<const Object>(std::move(parent)));
    DrainDeferred();
  }

  std::shared_ptr<const Scope> parent;
  std::vector<std::pair<std::string, Value>> vars;

  // A variable, through the enclosing scopes; null when none has it. `work`
  // grows by what the searches cost (FindMember): a scope may hold as many
  // variables as the template names.
  const Value* Find(std::string_view name, std::uint64_t& work) const {
    for (const Scope* s = this; s != nullptr; s = s->parent.get()) {
      if (const Value* v = FindMember(s->vars, name, work)) {
        return v;
      }
    }
    return nullptr;
  }
  // Sets a variable of this scope, its search charged; false when that
  // reaches a bound.
  bool Set(std::string_view name, Value v, Arena& arena) {
    std::uint64_t work = 0;
    if (Value* existing = FindMember(vars, name, work)) {
      *existing = std::move(v);
    } else {
      vars.emplace_back(std::string(name), std::move(v));
    }
    return arena.Work(work);
  }
};

enum class Flow : std::uint8_t { kNormal, kBreak, kContinue };

// Work charged for each string value an operation makes a piece at a time
// (a string's characters, a split's parts): making one costs about what
// copying this many bytes does, so such operations meet the work bound in
// about the time that copying does.
constexpr std::uint64_t kValueWork = 128;

// Case mapping and case tests (chat/pycase.h) cost more than copying: ASCII
// maps at about 330 MB/s, other text, each code point looked up in the UCD
// tables, at 40 to 80 MB/s (spark-b's GB10). Their bytes are charged that
// much more, so that the work bound spent on them takes about as long as
// on anything else (at most 3.3 s measured, docs/tokenizer.md).
constexpr std::uint64_t kCaseWorkAscii = 2;
constexpr std::uint64_t kCaseWorkOther = 16;

std::uint64_t CaseWork(std::string_view text) {
  std::uint64_t other = 0;
  for (const char c : text) {
    other += static_cast<unsigned char>(c) >= 0x80 ? 1 : 0;
  }
  return ((text.size() - other) * kCaseWorkAscii) + (other * kCaseWorkOther);
}

char AsciiLower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

// Arguments of a filter, test, method or global call.
struct Args {
  std::vector<Value> positional;
  std::vector<std::pair<std::string, Value>> keywords;

  // The argument at `index`, or the keyword `name`; null when absent.
  const Value* Get(std::size_t index, std::string_view name) const {
    if (index < positional.size()) {
      return &positional[index];
    }
    for (const auto& [k, v] : keywords) {
      if (k == name) {
        return &v;
      }
    }
    return nullptr;
  }
};

}  // namespace

// Renders one template once: the scope chain, the arena and the evaluator.
// Input's friend, so not in the anonymous namespace.
class Renderer {
 public:
  Renderer(const Program& program, const std::optional<CivilTime>& now, Budget budget)
      : program_(program), arena_(program.limits, budget), now_(now), usage_(budget.usage) {}
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  Renderer(Renderer&&) = delete;
  Renderer& operator=(Renderer&&) = delete;
  ~Renderer() {
    if (usage_ != nullptr) {
      usage_->steps += arena_.steps();
      usage_->work_bytes += arena_.work();
    }
    for (const std::weak_ptr<Scope>& w : macro_scopes_) {
      if (const auto s = w.lock()) {
        s->vars.clear();
      }
    }
    for (const std::weak_ptr<Namespace>& w : namespaces_) {
      if (const auto n = w.lock()) {
        n->members.clear();
      }
    }
  }

  std::expected<Rendered, Error> Run(const std::vector<std::pair<std::string, Input>>& variables) {
    auto globals = std::make_shared<Scope>();
    for (const auto& [name, input] : variables) {
      auto v = FromInput(input, 1);
      if (!v) {
        return std::unexpected(v.error());
      }
      if (!globals->Set(name, std::move(*v), arena_)) {
        return Bound();
      }
    }
    StrBuilder out;
    std::shared_ptr<Scope> scope = globals;
    auto flow = Exec(program_.body, scope, out);
    if (!flow) {
      return std::unexpected(flow.error());
    }
    if (!arena_.ok()) {
      return Bound();  // a bound reached where no step followed to report it
    }
    Rendered r;
    for (const Range& range : out.raw().trusted) {
      r.trusted.emplace_back(range.offset, range.length);
    }
    r.text = std::move(out.raw().text);
    return r;
  }

 private:
  // ----------------------------------------------------------------- helpers

  std::unexpected<Error> Bound() const {
    return Fail(arena_.cancelled() ? Code::kCancelled : Code::kLimit, arena_.reason());
  }

  // Lookups by name, their searches charged (FindMember). A bound reached
  // fails the next step; loops without one check arena_.ok().
  const Value* FindIn(const Scope& scope, std::string_view name) {
    std::uint64_t work = 0;
    const Value* v = scope.Find(name, work);
    (void)arena_.Work(work);
    return v;
  }
  const Value* FindIn(const Dict& d, std::string_view key) {
    std::uint64_t work = 0;
    const Value* v = d.Find(key, work);
    (void)arena_.Work(work);
    return v;
  }
  Value* FindIn(Namespace& ns, std::string_view key) {
    std::uint64_t work = 0;
    Value* v = ns.Find(key, work);
    (void)arena_.Work(work);
    return v;
  }
  // What a call's arguments cost each filter, test or method that reads
  // them: Args::Get searches the keywords by name.
  bool ChargeArgs(const Args& args) {
    return arena_.Work(kEntryWork * (args.positional.size() + (4 * args.keywords.size())));
  }

  // Records an object that may close a cycle; drops the records of freed
  // ones as the list grows, so it stays proportional to the live ones.
  template <typename T>
  static void Track(std::vector<std::weak_ptr<T>>& list, const std::shared_ptr<T>& object) {
    if (list.size() == list.capacity() && list.size() >= 64) {
      std::erase_if(list, [](const std::weak_ptr<T>& w) { return w.expired(); });
    }
    list.push_back(object);
  }

  // Python's == and <, their work charged (a bound reached fails the next
  // step).
  // Eq's result is meaningless once a bound is reached: callers check
  // arena_.ok(), and Run refuses a rendering that reached one.
  bool Eq(const Value& a, const Value& b) {
    std::uint64_t work = 0;
    const bool equal = Equal(a, b, work, arena_.remaining_work());
    (void)arena_.Work(work);
    return equal;
  }
  std::expected<bool, Error> Lt(const Value& a, const Value& b) {
    std::uint64_t work = 0;
    auto less = Less(a, b, work, arena_.remaining_work());
    if (!arena_.Work(work)) {
      return Bound();
    }
    return less;
  }
  // Eq that reports a bound reached.
  std::expected<bool, Error> CheckedEq(const Value& a, const Value& b) {
    const bool equal = Eq(a, b);
    if (!arena_.ok()) {
      return Bound();
    }
    return equal;
  }
  // A needle a Searcher may take: its tables bounded and charged.
  std::expected<void, Error> CheckNeedle(std::string_view needle) {
    if (needle.size() > arena_.limits().max_live_bytes / kMaxNeedleShare) {
      return Fail(Code::kLimit, "a search for a string longer than its bound");
    }
    if (!arena_.Work(needle.size() * 8)) {
      return Bound();
    }
    return {};
  }

  std::expected<Value, Error> Str(std::string_view text, bool trusted) {
    auto s = MakeStr(&arena_, text, trusted);
    if (!s) {
      return Bound();
    }
    return Value::String(std::move(s));
  }
  std::expected<Value, Error> Take(StrBuilder& b) {
    auto s = b.Take(&arena_);
    if (!s) {
      return Bound();
    }
    return Value::String(std::move(s));
  }
  // A substring of a string value, with its provenance.
  std::expected<Value, Error> Sub(const jinja::Str& s, std::size_t offset, std::size_t length) {
    StrBuilder b;
    b.Append(s, offset, length);
    return Take(b);
  }

  std::expected<Value, Error> MakeList(std::vector<Value> items, bool tuple = false) {
    auto l = std::make_shared<List>();
    std::size_t depth = 1;
    for (const Value& v : items) {
      if (v.is_list()) {
        depth = std::max(depth, v.list().depth + 1);
      } else if (v.is_dict()) {
        depth = std::max(depth, v.dict().depth + 1);
      }
    }
    if (depth > arena_.limits().max_value_depth) {
      return Fail(Code::kLimit, "lists and mappings nest deeper than their bound");
    }
    l->depth = depth;
    l->tuple = tuple;
    l->items = std::move(items);
    if (!arena_.Work(l->items.size() * sizeof(Value)) ||
        !l->Charge(&arena_, (l->items.size() * sizeof(Value)) + 64)) {
      return Bound();
    }
    return Value::MakeList(std::move(l));
  }

  std::expected<Value, Error> MakeDict(std::vector<std::pair<std::string, Value>> members,
                                       bool keys_trusted) {
    auto d = std::make_shared<Dict>();
    std::size_t depth = 1;
    std::size_t bytes = 64;
    for (const auto& [k, v] : members) {
      bytes += k.size() + sizeof(Value) + 32;
      if (v.is_list()) {
        depth = std::max(depth, v.list().depth + 1);
      } else if (v.is_dict()) {
        depth = std::max(depth, v.dict().depth + 1);
      }
    }
    if (depth > arena_.limits().max_value_depth) {
      return Fail(Code::kLimit, "lists and mappings nest deeper than their bound");
    }
    d->depth = depth;
    d->keys_trusted = keys_trusted;
    d->members = std::move(members);
    d->BuildIndex();
    bytes += d->index.size() * 48;
    if (!arena_.Work(bytes) || !d->Charge(&arena_, bytes)) {
      return Bound();
    }
    return Value::MakeDict(std::move(d));
  }

  std::expected<Value, Error> MakeCallable(Callable::Type type, const Value& receiver = {},
                                           std::string_view name = {}, std::size_t index = 0) {
    auto c = std::make_shared<Callable>();
    c->type = type;
    c->receiver = receiver;
    c->name = name;
    c->index = index;
    if (!c->Charge(&arena_, 128 + name.size())) {
      return Bound();
    }
    return Value::MakeCallable(std::move(c));
  }

  std::expected<Value, Error> FromJson(base::json::Value j, std::size_t depth) {
    if (depth > arena_.limits().max_value_depth) {
      return Fail(Code::kLimit, "lists and mappings nest deeper than their bound");
    }
    switch (j.kind()) {
      case base::json::Kind::kNull:
        return Value::None(false);
      case base::json::Kind::kFalse:
      case base::json::Kind::kTrue:
        return Value::Bool(j.boolean(), false);
      case base::json::Kind::kNumber:
        if (j.is_integer()) {
          if (const auto i = j.int64()) {
            return Value::Int(*i, false);
          }
          std::string_view digits = j.number();
          auto s = MakeStr(&arena_, digits, false);
          if (!s) {
            return Bound();
          }
          return Value::BigInt(std::move(s));
        }
        return Value::Float(j.float64().value_or(HUGE_VAL), false);
      case base::json::Kind::kString:
        return Str(j.string(), false);
      case base::json::Kind::kArray: {
        std::vector<Value> items;
        items.reserve(j.size());
        for (std::size_t i = 0; i < j.size(); ++i) {
          auto v = FromJson(j.at(i), depth + 1);
          if (!v) {
            return v;
          }
          items.push_back(std::move(*v));
        }
        return MakeList(std::move(items));
      }
      case base::json::Kind::kObject: {
        std::vector<std::pair<std::string, Value>> members;
        members.reserve(j.size());
        std::unordered_map<std::string_view, std::size_t> seen;  // keys in the document
        for (std::size_t i = 0; i < j.size(); ++i) {
          auto v = FromJson(j.member(i), depth + 1);
          if (!v) {
            return v;
          }
          const std::string_view key = j.key(i);
          if (const auto [it, fresh] = seen.emplace(key, members.size()); !fresh) {
            members[it->second].second = std::move(*v);  // json.loads keeps the last duplicate
          } else {
            members.emplace_back(std::string(key), std::move(*v));
          }
        }
        return MakeDict(std::move(members), false);
      }
    }
    return Value();
  }

  std::expected<Value, Error> FromInput(const Input& in, std::size_t depth);

  // ------------------------------------------------------------ statements

  std::expected<Flow, Error> Exec(const Body& body, std::shared_ptr<Scope>& scope,
                                  StrBuilder& out) {
    for (const StmtPtr& s : body) {
      auto flow = ExecOne(*s, scope, out);
      if (!flow) {
        Error e = flow.error();
        if (e.line == 0) {
          e.line = s->line;
        }
        return std::unexpected(e);
      }
      if (*flow != Flow::kNormal) {
        return flow;
      }
      if (out.size() > arena_.limits().max_output_bytes) {
        return Fail(Code::kLimit, "the rendering is longer than its bound", s->line);
      }
    }
    return Flow::kNormal;
  }

  class Depth {
   public:
    explicit Depth(Renderer& r) : r_(r) { ++r_.depth_; }
    Depth(const Depth&) = delete;
    Depth& operator=(const Depth&) = delete;
    Depth(Depth&&) = delete;
    Depth& operator=(Depth&&) = delete;
    ~Depth() { --r_.depth_; }
    bool ok() const { return r_.depth_ <= r_.arena_.limits().max_eval_depth; }

   private:
    Renderer& r_;
  };

  std::expected<Flow, Error> ExecOne(const Stmt& s, std::shared_ptr<Scope>& scope,
                                     StrBuilder& out) {
    const Depth depth(*this);
    if (!depth.ok()) {
      return Fail(Code::kLimit, "evaluation nests deeper than its bound", s.line);
    }
    if (!arena_.Step()) {
      return Bound();
    }
    switch (s.kind) {
      case StmtKind::kText:
        if (!arena_.Work(s.text->text.size())) {
          return Bound();
        }
        out.Append(*s.text);
        return Flow::kNormal;
      case StmtKind::kPrint: {
        auto v = Eval(*s.expr, *scope);
        if (!v) {
          return std::unexpected(v.error());
        }
        if (v->is_string() && !arena_.Work(v->str().text.size())) {
          return Bound();
        }
        if (auto r = AppendStr(*v, out, arena_); !r) {
          return std::unexpected(r.error());
        }
        return Flow::kNormal;
      }
      case StmtKind::kIf:
        for (const auto& [cond, branch] : s.branches) {
          auto c = Eval(*cond, *scope);
          if (!c) {
            return std::unexpected(c.error());
          }
          if (Truthy(*c)) {
            return Exec(branch, scope, out);
          }
        }
        return Exec(s.else_body, scope, out);
      case StmtKind::kFor:
        return ExecFor(s, scope, out);
      case StmtKind::kSet: {
        auto v = Eval(*s.expr, *scope);
        if (!v) {
          return std::unexpected(v.error());
        }
        if (auto r = Assign(s, std::move(*v), *scope); !r) {
          return std::unexpected(r.error());
        }
        return Flow::kNormal;
      }
      case StmtKind::kSetBlock: {
        StrBuilder captured;
        auto flow = Exec(s.body, scope, captured);
        if (!flow) {
          return flow;
        }
        auto v = Take(captured);
        if (!v) {
          return std::unexpected(v.error());
        }
        for (const ExprPtr& f : s.filters) {
          v = ApplyFilter(*f, std::move(*v), *scope);
          if (!v) {
            return std::unexpected(v.error());
          }
        }
        if (!scope->Set(s.targets[0], std::move(*v), arena_)) {
          return Bound();
        }
        return Flow::kNormal;
      }
      case StmtKind::kMacro: {
        auto c = std::make_shared<Callable>();
        c->type = Callable::Type::kMacro;
        c->macro = &s;
        c->scope = scope;
        Track(macro_scopes_, scope);
        c->name = s.name;
        if (!c->Charge(&arena_, 128)) {
          return Bound();
        }
        if (!scope->Set(s.name, Value::MakeCallable(std::move(c)), arena_)) {
          return Bound();
        }
        return Flow::kNormal;
      }
      case StmtKind::kBreak:
        return Flow::kBreak;
      case StmtKind::kContinue:
        return Flow::kContinue;
    }
    return Flow::kNormal;
  }

  std::expected<void, Error> Unpack(const Stmt& s, Value v, Scope& scope) {
    if (!s.tuple_target) {
      if (!scope.Set(s.targets[0], std::move(v), arena_)) {
        return Bound();
      }
      return {};
    }
    // The length first: a sequence of the wrong length is refused before
    // it is copied or taken apart.
    std::size_t length = 0;
    if (v.is_list()) {
      length = v.list().items.size();
    } else if (v.is_string()) {
      if (!arena_.Work(v.str().text.size())) {
        return Bound();
      }
      length = CodePoints(v.str().text);
    } else {
      return Fail(Code::kRuntime, "unpacking a value that is not a sequence");
    }
    if (length != s.targets.size()) {
      return Fail(Code::kRuntime, "unpacking a sequence of the wrong length");
    }
    std::vector<Value> items;
    if (v.is_list()) {
      items = v.list().items;
    } else {
      auto chars = Iterate(v);
      if (!chars) {
        return std::unexpected(chars.error());
      }
      items = std::move(*chars);
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (!scope.Set(s.targets[i], std::move(items[i]), arena_)) {
        return Bound();
      }
    }
    return {};
  }

  std::expected<void, Error> Assign(const Stmt& s, Value v, Scope& scope) {
    if (!s.attribute_target) {
      return Unpack(s, std::move(v), scope);
    }
    const Value* target = FindIn(scope, s.targets[0]);
    if (!arena_.ok()) {
      return Bound();
    }
    if (target == nullptr || target->kind() != Kind::kNamespace) {
      return Fail(Code::kRuntime, "an attribute assigned on something other than a namespace");
    }
    Namespace& ns = target->ns();
    Value* existing = FindIn(ns, s.targets[1]);
    if (!arena_.ok()) {
      return Bound();
    }
    if (existing != nullptr) {
      *existing = std::move(v);
    } else {
      if (!ns.Charge(&arena_, s.targets[1].size() + sizeof(Value) + 32)) {
        return Bound();
      }
      ns.members.emplace_back(s.targets[1], std::move(v));
    }
    return {};
  }

  // The items a for loop or a filter iterates: a list's, a mapping's keys,
  // a string's characters; nothing for an undefined value.
  std::expected<std::vector<Value>, Error> Iterate(const Value& v) {
    switch (v.kind()) {
      case Kind::kUndefined:
        return std::vector<Value>();
      case Kind::kList:
        if (!arena_.Work(v.list().items.size() * 16)) {
          return Bound();
        }
        return v.list().items;
      case Kind::kDict: {
        if (!arena_.Work(v.dict().members.size() * 16)) {
          return Bound();
        }
        std::vector<Value> keys;
        keys.reserve(v.dict().members.size());
        for (const auto& [k, unused] : v.dict().members) {
          auto key = Str(k, v.dict().keys_trusted);
          if (!key) {
            return std::unexpected(key.error());
          }
          keys.push_back(std::move(*key));
        }
        return keys;
      }
      case Kind::kString: {
        const jinja::Str& s = v.str();
        if (!arena_.Work(s.text.size() * kValueWork)) {  // a value a code point
          return Bound();
        }
        std::vector<Value> chars;
        for (std::size_t at = 0; at < s.text.size();) {
          std::size_t length = 0;
          DecodeAt(s.text, at, length);
          auto c = Sub(s, at, length);
          if (!c) {
            return std::unexpected(c.error());
          }
          chars.push_back(std::move(*c));
          at += length;
        }
        return chars;
      }
      default:
        return Fail(Code::kRuntime, "iterating a value that is not iterable");
    }
  }

  std::expected<Flow, Error> ExecFor(const Stmt& s, std::shared_ptr<Scope>& scope,
                                     StrBuilder& out) {
    auto iterable = Eval(*s.expr, *scope);
    if (!iterable) {
      return std::unexpected(iterable.error());
    }
    // A list's own items are iterated in place (the value keeps them);
    // anything else is a copy, charged as live while the loop runs, so
    // loops nested over one long sequence hold no more than their bound.
    Held held(arena_);
    std::vector<Value> owned;
    const std::vector<Value>* all = nullptr;
    if (iterable->is_list()) {
      if (!arena_.Work(iterable->list().items.size() * 16)) {
        return Bound();
      }
      all = &iterable->list().items;
    } else {
      auto copy = Iterate(*iterable);
      if (!copy) {
        return std::unexpected(copy.error());
      }
      owned = std::move(*copy);
      if (!held.Add(owned.size() * sizeof(Value))) {
        return Bound();
      }
      all = &owned;
    }
    std::vector<Value> filtered;
    if (s.filter) {
      for (const Value& item : *all) {
        auto probe = std::make_shared<Scope>();
        probe->parent = scope;
        if (auto r = Unpack(s, item, *probe); !r) {
          return std::unexpected(r.error());
        }
        auto keep = Eval(*s.filter, *probe);
        if (!keep) {
          return std::unexpected(keep.error());
        }
        if (Truthy(*keep)) {
          if (!held.Add(sizeof(Value))) {
            return Bound();
          }
          filtered.push_back(item);
        }
      }
    }
    const std::vector<Value>& items = s.filter ? filtered : *all;
    if (items.empty()) {
      return Exec(s.else_body, scope, out);
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (!arena_.Step()) {
        return Bound();
      }
      auto iteration = std::make_shared<Scope>();
      iteration->parent = scope;
      if (!iteration->Charge(&arena_, 256)) {
        return Bound();
      }
      if (auto r = Unpack(s, items[i], *iteration); !r) {
        return std::unexpected(r.error());
      }
      auto loop = std::make_shared<Loop>();
      loop->index = i;
      loop->length = items.size();
      if (i > 0) {
        loop->previous = items[i - 1];
      }
      if (i + 1 < items.size()) {
        loop->next = items[i + 1];
      }
      if (!loop->Charge(&arena_, 192)) {
        return Bound();
      }
      if (!iteration->Set("loop", Value::MakeLoop(std::move(loop)), arena_)) {
        return Bound();
      }
      std::shared_ptr<Scope> inner = iteration;
      auto flow = Exec(s.body, inner, out);
      if (!flow) {
        return flow;
      }
      if (*flow == Flow::kBreak) {
        break;
      }
    }
    return Flow::kNormal;
  }

  // ----------------------------------------------------------- expressions

  std::expected<Value, Error> Eval(const Expr& e, const Scope& scope) {
    const Depth depth(*this);
    if (!depth.ok()) {
      return Fail(Code::kLimit, "evaluation nests deeper than its bound", e.line);
    }
    if (!arena_.Step()) {
      return Bound();
    }
    auto v = EvalInner(e, scope);
    if (!v && v.error().line == 0) {
      Error err = v.error();
      err.line = e.line;
      return std::unexpected(err);
    }
    return v;
  }

  std::expected<std::vector<Value>, Error> EvalAll(const std::vector<ExprPtr>& args,
                                                   std::size_t from, const Scope& scope) {
    std::vector<Value> out;
    for (std::size_t i = from; i < args.size(); ++i) {
      auto v = Eval(*args[i], scope);
      if (!v) {
        return std::unexpected(v.error());
      }
      out.push_back(std::move(*v));
    }
    return out;
  }

  std::expected<Args, Error> EvalArgs(const Expr& e, std::size_t from, const Scope& scope) {
    Args a;
    auto positional = EvalAll(e.args, from, scope);
    if (!positional) {
      return std::unexpected(positional.error());
    }
    a.positional = std::move(*positional);
    for (const auto& [name, expr] : e.kwargs) {
      auto v = Eval(*expr, scope);
      if (!v) {
        return std::unexpected(v.error());
      }
      a.keywords.emplace_back(name, std::move(*v));
    }
    return a;
  }

  std::expected<Value, Error> EvalInner(const Expr& e, const Scope& scope) {
    switch (e.kind) {
      case ExprKind::kLiteral:
        return e.constant;
      case ExprKind::kName:
        return Lookup(e.name, scope);
      case ExprKind::kList:
      case ExprKind::kTuple: {
        auto items = EvalAll(e.args, 0, scope);
        if (!items) {
          return std::unexpected(items.error());
        }
        return MakeList(std::move(*items), e.kind == ExprKind::kTuple);
      }
      case ExprKind::kDict: {
        std::vector<std::pair<std::string, Value>> members;
        Held held(arena_);  // the keys copied, until MakeDict charges them
        bool trusted = true;
        for (std::size_t i = 0; i + 1 < e.args.size(); i += 2) {
          auto k = Eval(*e.args[i], scope);
          if (!k) {
            return k;
          }
          if (!k->is_string()) {
            return Fail(Code::kUnsupported, "a mapping key that is not a string");
          }
          trusted = trusted && k->str().all_trusted();
          auto v = Eval(*e.args[i + 1], scope);
          if (!v) {
            return v;
          }
          // A later duplicate key replaces the earlier: each key searches
          // those before it, charged (FindMember).
          std::uint64_t work = 0;
          Value* existing = FindMember(members, k->str().text, work);
          if (!arena_.Work(work)) {
            return Bound();
          }
          if (existing != nullptr) {
            *existing = std::move(*v);
          } else {
            if (!held.Add(k->str().text.size())) {
              return Bound();
            }
            members.emplace_back(k->str().text, std::move(*v));
          }
        }
        held.Clear();
        return MakeDict(std::move(members), trusted);
      }
      case ExprKind::kAttr: {
        auto obj = Eval(*e.args[0], scope);
        if (!obj) {
          return obj;
        }
        return GetAttr(*obj, e.name);
      }
      case ExprKind::kItem: {
        auto obj = Eval(*e.args[0], scope);
        if (!obj) {
          return obj;
        }
        auto key = Eval(*e.args[1], scope);
        if (!key) {
          return key;
        }
        return GetItem(*obj, *key);
      }
      case ExprKind::kSlice:
        return EvalSlice(e, scope);
      case ExprKind::kCall: {
        auto callee = Eval(*e.args[0], scope);
        if (!callee) {
          return callee;
        }
        auto args = EvalArgs(e, 1, scope);
        if (!args) {
          return std::unexpected(args.error());
        }
        return Call(*callee, *args);
      }
      case ExprKind::kFilter: {
        auto subject = Eval(*e.args[0], scope);
        if (!subject) {
          return subject;
        }
        return ApplyFilter(e, std::move(*subject), scope);
      }
      case ExprKind::kTest: {
        auto subject = Eval(*e.args[0], scope);
        if (!subject) {
          return subject;
        }
        auto args = EvalArgs(e, 1, scope);
        if (!args) {
          return std::unexpected(args.error());
        }
        auto r = ApplyTest(e.name, *subject, *args);
        if (!r) {
          return std::unexpected(r.error());
        }
        return Value::Bool(*r != e.negated);
      }
      case ExprKind::kNot: {
        auto v = Eval(*e.args[0], scope);
        if (!v) {
          return v;
        }
        return Value::Bool(!Truthy(*v));
      }
      case ExprKind::kNeg:
      case ExprKind::kPos: {
        auto v = Eval(*e.args[0], scope);
        if (!v) {
          return v;
        }
        if (!v->is_number() || v->kind() == Kind::kBigInt) {
          return Fail(Code::kRuntime, "a sign applied to something other than a number");
        }
        if (e.kind == ExprKind::kPos) {
          return v->is_integral() ? Value::Int(v->integer(), v->trusted()) : *v;
        }
        if (v->kind() == Kind::kFloat) {
          return Value::Float(-v->number(), v->trusted());
        }
        if (v->integer() == std::numeric_limits<std::int64_t>::min()) {
          return Fail(Code::kUnsupported, "an integer beyond 64 bits");
        }
        return Value::Int(-v->integer(), v->trusted());
      }
      case ExprKind::kAnd:
      case ExprKind::kOr: {
        auto left = Eval(*e.args[0], scope);
        if (!left) {
          return left;
        }
        if (Truthy(*left) == (e.kind == ExprKind::kOr)) {
          return left;
        }
        return Eval(*e.args[1], scope);
      }
      case ExprKind::kBinary: {
        auto left = Eval(*e.args[0], scope);
        if (!left) {
          return left;
        }
        auto right = Eval(*e.args[1], scope);
        if (!right) {
          return right;
        }
        return Binary(e.op, *left, *right);
      }
      case ExprKind::kCompare: {
        auto left = Eval(*e.args[0], scope);
        if (!left) {
          return left;
        }
        for (std::size_t i = 0; i < e.ops.size(); ++i) {
          auto right = Eval(*e.args[i + 1], scope);
          if (!right) {
            return right;
          }
          auto r = Compare(e.ops[i], *left, *right);
          if (!r) {
            return std::unexpected(r.error());
          }
          if (!*r) {
            return Value::Bool(false);
          }
          left = std::move(right);
        }
        return Value::Bool(true);
      }
      case ExprKind::kCondition: {
        auto cond = Eval(*e.args[1], scope);
        if (!cond) {
          return cond;
        }
        if (Truthy(*cond)) {
          return Eval(*e.args[0], scope);
        }
        if (e.args[2]) {
          return Eval(*e.args[2], scope);
        }
        return Value();
      }
    }
    return Value();
  }

  std::expected<Value, Error> Lookup(const std::string& name, const Scope& scope) {
    const Value* v = FindIn(scope, name);
    if (!arena_.ok()) {
      return Bound();
    }
    if (v != nullptr) {
      return *v;
    }
    if (name == "range") {
      return MakeCallable(Callable::Type::kRange);
    }
    if (name == "namespace") {
      return MakeCallable(Callable::Type::kNamespace);
    }
    if (name == "dict") {
      return MakeCallable(Callable::Type::kDict);
    }
    if (name == "raise_exception") {
      return MakeCallable(Callable::Type::kRaise);
    }
    if (name == "strftime_now") {
      return MakeCallable(Callable::Type::kStrftime);
    }
    return Value();
  }

  std::expected<Value, Error> GetAttr(const Value& obj, std::string_view name) {
    switch (obj.kind()) {
      case Kind::kUndefined:
        return Fail(Code::kRuntime, "an attribute of an undefined value");
      case Kind::kDict:
        if (In(kDictMethods, name)) {
          return MakeCallable(Callable::Type::kMethod, obj, name);
        }
        if (const Value* v = FindIn(obj.dict(), name); v != nullptr && arena_.ok()) {
          return *v;
        }
        if (!arena_.ok()) {
          return Bound();
        }
        return Value();
      case Kind::kString:
        if (In(kStrMethods, name)) {
          return MakeCallable(Callable::Type::kMethod, obj, name);
        }
        return Value();
      case Kind::kList:
        if (In(kListMethods, name)) {
          return MakeCallable(Callable::Type::kMethod, obj, name);
        }
        return Value();
      case Kind::kLoop: {
        const Loop& l = obj.loop();
        const auto index = static_cast<std::int64_t>(l.index);
        const auto length = static_cast<std::int64_t>(l.length);
        if (name == "index") {
          return Value::Int(index + 1);
        }
        if (name == "index0") {
          return Value::Int(index);
        }
        if (name == "revindex") {
          return Value::Int(length - index);
        }
        if (name == "revindex0") {
          return Value::Int(length - index - 1);
        }
        if (name == "first") {
          return Value::Bool(l.index == 0);
        }
        if (name == "last") {
          return Value::Bool(l.index + 1 == l.length);
        }
        if (name == "length") {
          return Value::Int(length);
        }
        if (name == "depth") {
          return Value::Int(1);
        }
        if (name == "depth0") {
          return Value::Int(0);
        }
        if (name == "previtem") {
          return l.previous.value_or(Value());
        }
        if (name == "nextitem") {
          return l.next.value_or(Value());
        }
        if (name == "cycle") {
          return MakeCallable(Callable::Type::kCycle, Value(), {}, l.index);
        }
        if (name == "changed") {
          return Fail(Code::kUnsupported, "loop.changed");
        }
        return Value();
      }
      case Kind::kNamespace:
        if (const Value* v = FindIn(obj.ns(), name); v != nullptr && arena_.ok()) {
          return *v;
        }
        if (!arena_.ok()) {
          return Bound();
        }
        return Value();
      default:
        return Value();
    }
  }

  std::expected<Value, Error> GetItem(const Value& obj, const Value& key) {
    switch (obj.kind()) {
      case Kind::kUndefined:
        return Fail(Code::kRuntime, "an item of an undefined value");
      case Kind::kDict:
        if (key.is_string()) {
          if (const Value* v = FindIn(obj.dict(), key.str().text); v != nullptr && arena_.ok()) {
            return *v;
          }
          if (!arena_.ok()) {
            return Bound();
          }
          return GetAttr(obj, key.str().text);
        }
        return Value();
      case Kind::kList:
      case Kind::kString: {
        if (!key.is_integral()) {
          return key.is_string() ? GetAttr(obj, key.str().text) : Value();
        }
        const bool is_list = obj.is_list();
        const std::size_t n = is_list ? obj.list().items.size() : CodePoints(obj.str().text);
        std::int64_t i = key.integer();
        if (i < 0) {
          i += static_cast<std::int64_t>(n);
        }
        if (i < 0 || std::cmp_greater_equal(i, n)) {
          return Value();
        }
        if (is_list) {
          return obj.list().items[static_cast<std::size_t>(i)];
        }
        const std::string& text = obj.str().text;
        if (!arena_.Work(text.size())) {
          return Bound();
        }
        const std::size_t at = ByteOffset(text, static_cast<std::size_t>(i));
        std::size_t length = 0;
        DecodeAt(text, at, length);
        return Sub(obj.str(), at, length);
      }
      case Kind::kNamespace:
        return key.is_string() ? GetAttr(obj, key.str().text) : Value();
      default:
        return Value();
    }
  }

  // Python's slice of a sequence of `n` (PySlice_AdjustIndices and
  // PySlice_GetLength): the first index and how many indices, `step`
  // apart, it takes. Every index lies in [0, n), so stepping from one to
  // the next never overflows, whatever the step.
  struct SliceRun {
    std::int64_t start = 0;
    std::uint64_t count = 0;
  };
  static SliceRun AdjustSlice(std::int64_t n, std::optional<std::int64_t> start,
                              std::optional<std::int64_t> stop, std::int64_t step) {
    const auto clamp = [&](std::optional<std::int64_t> v, std::int64_t fallback) {
      if (!v) {
        return fallback;
      }
      std::int64_t x = *v;
      if (x < 0) {
        x += n;
        if (x < 0) {
          x = step < 0 ? -1 : 0;
        }
      } else if (x >= n) {
        x = step < 0 ? n - 1 : n;
      }
      return x;
    };
    SliceRun run;
    if (step > 0) {
      run.start = clamp(start, 0);
      const std::int64_t stop_at = clamp(stop, n);
      if (run.start < stop_at) {
        run.count = ((static_cast<std::uint64_t>(stop_at - run.start) - 1) /
                     static_cast<std::uint64_t>(step)) +
                    1;
      }
    } else {
      run.start = clamp(start, n - 1);
      const std::int64_t stop_at = clamp(stop, -1);
      if (stop_at < run.start) {
        // 0 - step as unsigned: the magnitude, even of the most negative step.
        run.count = ((static_cast<std::uint64_t>(run.start - stop_at) - 1) /
                     (std::uint64_t{0} - static_cast<std::uint64_t>(step))) +
                    1;
      }
    }
    return run;
  }

  std::expected<Value, Error> EvalSlice(const Expr& e, const Scope& scope) {
    auto obj = Eval(*e.args[0], scope);
    if (!obj) {
      return obj;
    }
    std::array<std::optional<std::int64_t>, 3> bounds;
    for (std::size_t i = 0; i < 3; ++i) {
      if (!e.args[i + 1]) {
        continue;
      }
      auto v = Eval(*e.args[i + 1], scope);
      if (!v) {
        return v;
      }
      if (v->is_none() || v->is_undefined()) {
        continue;
      }
      if (!v->is_integral()) {
        return Fail(Code::kRuntime, "a slice bound that is not an integer");
      }
      bounds[i] = v->integer();
    }
    const std::int64_t step = bounds[2].value_or(1);
    if (step == 0) {
      return Fail(Code::kRuntime, "a slice step of zero");
    }
    if (obj->is_undefined()) {
      return Fail(Code::kRuntime, "a slice of an undefined value");
    }
    if (obj->is_list()) {
      const List& l = obj->list();
      if (!arena_.Work(l.items.size() * 16)) {
        return Bound();
      }
      const SliceRun run =
          AdjustSlice(static_cast<std::int64_t>(l.items.size()), bounds[0], bounds[1], step);
      std::vector<Value> items;
      items.reserve(run.count);
      std::int64_t i = run.start;
      for (std::uint64_t k = 0; k < run.count; ++k) {
        items.push_back(l.items[static_cast<std::size_t>(i)]);
        if (k + 1 < run.count) {
          i += step;
        }
      }
      return MakeList(std::move(items), l.tuple);
    }
    if (obj->is_string()) {
      const jinja::Str& s = obj->str();
      if (!arena_.Work(s.text.size() * 2)) {
        return Bound();
      }
      const std::size_t n = CodePoints(s.text);
      const SliceRun run = AdjustSlice(static_cast<std::int64_t>(n), bounds[0], bounds[1], step);
      StrBuilder b;
      if (run.count == 0) {
        return Take(b);
      }
      const auto first = static_cast<std::size_t>(run.start);
      if (step == 1) {
        const std::size_t from = ByteOffset(s.text, first);
        const std::size_t to = ByteOffset(s.text, first + run.count);
        b.Append(s, from, to - from);
        return Take(b);
      }
      // Any other step needs each code point's offset.
      if (!arena_.Work(s.text.size() * 8)) {  // a code point at a time
        return Bound();
      }
      if (n > arena_.limits().max_live_bytes / sizeof(std::size_t)) {
        return Fail(Code::kLimit, "a string sliced with a step longer than its bound");
      }
      std::vector<std::size_t> starts;
      starts.reserve(n);
      for (std::size_t at = 0; at < s.text.size(); ++at) {
        if ((static_cast<unsigned char>(s.text[at]) & 0xC0U) != 0x80U) {
          starts.push_back(at);
        }
      }
      std::int64_t i = run.start;
      for (std::uint64_t k = 0; k < run.count; ++k) {
        const auto at = static_cast<std::size_t>(i);
        const std::size_t to = at + 1 < starts.size() ? starts[at + 1] : s.text.size();
        b.Append(s, starts[at], to - starts[at]);
        if (k + 1 < run.count) {
          i += step;
        }
      }
      return Take(b);
    }
    return Value();
  }

  // ------------------------------------------------------------- operators

  std::expected<Value, Error> Concat(const Value& a, const Value& b) {
    StrBuilder out;
    if (auto r = AppendStr(a, out, arena_); !r) {
      return std::unexpected(r.error());
    }
    if (auto r = AppendStr(b, out, arena_); !r) {
      return std::unexpected(r.error());
    }
    return Take(out);
  }

  // markupsafe's escape(): & < > " ' as entities, provenance kept.
  static void AppendEscaped(const jinja::Str& s, StrBuilder& out) {
    std::size_t from = 0;
    std::size_t range = 0;  // the first trusted range that does not end before i
    for (std::size_t i = 0; i < s.text.size(); ++i) {
      std::string_view entity;
      switch (s.text[i]) {
        case '&':
          entity = "&amp;";
          break;
        case '<':
          entity = "&lt;";
          break;
        case '>':
          entity = "&gt;";
          break;
        case '"':
          entity = "&#34;";
          break;
        case '\'':
          entity = "&#39;";
          break;
        default:
          continue;
      }
      out.Append(s, from, i - from);
      while (range < s.trusted.size() && s.trusted[range].offset + s.trusted[range].length <= i) {
        ++range;
      }
      const bool trusted = range < s.trusted.size() && s.trusted[range].offset <= i;
      out.Append(entity, trusted);
      from = i + 1;
    }
    out.Append(s, from, s.text.size() - from);
  }

  std::expected<Value, Error> Add(const Value& a, const Value& b) {
    if (a.is_string() && b.is_string()) {
      if (a.str().text.size() + b.str().text.size() > arena_.limits().max_string_bytes) {
        return Fail(Code::kLimit, "a string longer than its bound");
      }
      StrBuilder out;
      const bool markup = a.str().markup || b.str().markup;
      if (markup && !a.str().markup) {
        AppendEscaped(a.str(), out);
      } else {
        out.Append(a.str());
      }
      if (markup && !b.str().markup) {
        AppendEscaped(b.str(), out);
      } else {
        out.Append(b.str());
      }
      out.set_markup(markup);
      return Take(out);
    }
    if (a.is_list() && b.is_list() && a.list().tuple == b.list().tuple) {
      std::vector<Value> items = a.list().items;
      items.insert(items.end(), b.list().items.begin(), b.list().items.end());
      return MakeList(std::move(items), a.list().tuple);
    }
    return Arithmetic(Op::kAdd, a, b);
  }

  std::expected<Value, Error> Repeat(const Value& seq, std::int64_t times) {
    std::size_t n = times > 0 ? static_cast<std::size_t>(times) : 0;
    if ((seq.is_string() && seq.str().text.empty()) ||
        (seq.is_list() && seq.list().items.empty())) {
      n = 0;  // nothing to repeat, however often
    }
    if (seq.is_string()) {
      const jinja::Str& s = seq.str();
      if (n != 0 && s.text.size() > arena_.limits().max_string_bytes / n) {
        return Fail(Code::kLimit, "a string longer than its bound");
      }
      StrBuilder out;
      if (n != 0 && (s.trusted.empty() || s.all_trusted())) {
        // One provenance throughout: copy by doubling, not a piece at a time.
        std::string& text = out.raw().text;
        const std::size_t total = s.text.size() * n;
        text.reserve(total);
        text = s.text;
        while (text.size() * 2 <= total) {
          text.append(text.data(), text.size());
        }
        text.append(text.data(), total - text.size());
        if (s.all_trusted()) {
          out.raw().trusted = {{0, total}};
        }
        return Take(out);
      }
      for (std::size_t i = 0; i < n; ++i) {
        out.Append(s);
      }
      return Take(out);
    }
    const List& l = seq.list();
    if (n != 0 && l.items.size() > (arena_.limits().max_live_bytes / sizeof(Value)) / n) {
      return Fail(Code::kLimit, "a list longer than its bound");
    }
    std::vector<Value> items;
    for (std::size_t i = 0; i < n; ++i) {
      items.insert(items.end(), l.items.begin(), l.items.end());
    }
    return MakeList(std::move(items), l.tuple);
  }

  std::expected<Value, Error> Arithmetic(Op op, const Value& a, const Value& b) {
    if (!a.is_number() || !b.is_number() || a.kind() == Kind::kBigInt ||
        b.kind() == Kind::kBigInt) {
      return Fail(Code::kRuntime, "arithmetic on values that do not support it");
    }
    const bool trusted = a.trusted() && b.trusted();
    const bool floats = a.kind() == Kind::kFloat || b.kind() == Kind::kFloat;
    if (op == Op::kDiv) {
      if (b.number() == 0.0) {
        return Fail(Code::kRuntime, "division by zero");
      }
      return Value::Float(a.number() / b.number(), trusted);
    }
    if (floats) {
      const double x = a.number();
      const double y = b.number();
      switch (op) {
        case Op::kAdd:
          return Value::Float(x + y, trusted);
        case Op::kSub:
          return Value::Float(x - y, trusted);
        case Op::kMul:
          return Value::Float(x * y, trusted);
        case Op::kFloorDiv:
          if (y == 0.0) {
            return Fail(Code::kRuntime, "division by zero");
          }
          return Value::Float(std::floor(x / y), trusted);
        case Op::kMod: {
          if (y == 0.0) {
            return Fail(Code::kRuntime, "division by zero");
          }
          double m = std::fmod(x, y);
          if (m != 0.0 && ((m < 0) != (y < 0))) {
            m += y;
          }
          return Value::Float(m, trusted);
        }
        case Op::kPow:
          if (x == 0.0 && y < 0) {
            return Fail(Code::kRuntime, "division by zero");
          }
          return Value::Float(std::pow(x, y), trusted);
        default:
          break;
      }
      return Fail(Code::kRuntime, "an unknown operator");
    }
    const std::int64_t x = a.integer();
    const std::int64_t y = b.integer();
    std::int64_t r = 0;
    bool overflow = false;
    switch (op) {
      case Op::kAdd:
        overflow = __builtin_add_overflow(x, y, &r);
        break;
      case Op::kSub:
        overflow = __builtin_sub_overflow(x, y, &r);
        break;
      case Op::kMul:
        overflow = __builtin_mul_overflow(x, y, &r);
        break;
      case Op::kFloorDiv:
      case Op::kMod: {
        if (y == 0) {
          return Fail(Code::kRuntime, "division by zero");
        }
        if (x == std::numeric_limits<std::int64_t>::min() && y == -1) {
          overflow = true;
          break;
        }
        std::int64_t q = x / y;
        std::int64_t m = x % y;
        if (m != 0 && ((m < 0) != (y < 0))) {
          --q;
          m += y;
        }
        r = op == Op::kFloorDiv ? q : m;
        break;
      }
      case Op::kPow: {
        if (y < 0) {
          if (x == 0) {
            return Fail(Code::kRuntime, "division by zero");
          }
          return Value::Float(std::pow(static_cast<double>(x), static_cast<double>(y)), trusted);
        }
        r = 1;
        for (std::int64_t i = 0; i < y && !overflow; ++i) {
          overflow = __builtin_mul_overflow(r, x, &r);
          if (r == 0 || r == 1) {
            break;  // 0 or 1 stays itself
          }
          if (r == -1) {
            r = (y % 2 == 0) ? 1 : -1;
            break;
          }
          if (!arena_.Step()) {
            return Bound();
          }
        }
        break;
      }
      default:
        return Fail(Code::kRuntime, "an unknown operator");
    }
    if (overflow) {
      return Fail(Code::kUnsupported, "an integer beyond 64 bits");
    }
    return Value::Int(r, trusted);
  }

  std::expected<Value, Error> Binary(Op op, const Value& a, const Value& b) {
    switch (op) {
      case Op::kConcat:
        return Concat(a, b);
      case Op::kAdd:
        return Add(a, b);
      case Op::kMul:
        if ((a.is_string() || a.is_list()) && b.is_integral()) {
          return Repeat(a, b.integer());
        }
        if ((b.is_string() || b.is_list()) && a.is_integral()) {
          return Repeat(b, a.integer());
        }
        return Arithmetic(op, a, b);
      case Op::kMod:
        if (a.is_string()) {
          Args args;
          if (b.is_list() && b.list().tuple) {
            if (!arena_.Work(b.list().items.size() * 16)) {
              return Bound();
            }
            args.positional = b.list().items;
          } else {
            args.positional.push_back(b);
          }
          // Python's str % takes a mapping (a list counts: it is
          // subscriptable) even where no directive uses it.
          return Format(a, args, b.is_dict() || (b.is_list() && !b.list().tuple));
        }
        return Arithmetic(op, a, b);
      default:
        return Arithmetic(op, a, b);
    }
  }
  std::expected<bool, Error> Contains(const Value& container, const Value& needle) {
    switch (container.kind()) {
      case Kind::kUndefined:
        return false;
      case Kind::kString: {
        if (!needle.is_string()) {
          return Fail(Code::kRuntime, "`in` a string with something other than a string");
        }
        const std::string_view sub = needle.str().text;
        if (auto r = CheckNeedle(sub); !r) {
          return std::unexpected(r.error());
        }
        if (!arena_.Work(container.str().text.size())) {
          return Bound();
        }
        return Searcher(sub).Find(container.str().text) != std::string_view::npos;
      }
      case Kind::kList: {
        if (!arena_.Work(container.list().items.size() * 16)) {
          return Bound();
        }
        const bool found = std::ranges::any_of(container.list().items,
                                               [&](const Value& item) { return Eq(item, needle); });
        if (!arena_.ok()) {
          return Bound();
        }
        return found;
      }
      case Kind::kDict:
        if (!needle.is_string()) {
          return false;
        }
        if (const Value* v = FindIn(container.dict(), needle.str().text); arena_.ok()) {
          return v != nullptr;
        }
        return Bound();
      default:
        return Fail(Code::kRuntime, "`in` a value that is not a container");
    }
  }

  std::expected<bool, Error> Compare(Op op, const Value& a, const Value& b) {
    switch (op) {
      case Op::kEq:
        return CheckedEq(a, b);
      case Op::kNe: {
        auto eq = CheckedEq(a, b);
        if (!eq) {
          return eq;
        }
        return !*eq;
      }
      case Op::kLt:
        return Lt(a, b);
      case Op::kGt:
        return Lt(b, a);
      case Op::kLe: {
        auto lt = Lt(b, a);
        if (!lt) {
          return lt;
        }
        return !*lt;
      }
      case Op::kGe: {
        auto lt = Lt(a, b);
        if (!lt) {
          return lt;
        }
        return !*lt;
      }
      case Op::kIn:
        return Contains(b, a);
      case Op::kNotIn: {
        auto in = Contains(b, a);
        if (!in) {
          return in;
        }
        return !*in;
      }
      default:
        return Fail(Code::kRuntime, "an unknown comparison");
    }
  }

  // -------------------------------------------------------------- callables

  std::expected<Value, Error> Call(const Value& callee, const Args& args) {
    if (!ChargeArgs(args)) {
      return Bound();
    }
    if (callee.is_undefined()) {
      return Fail(Code::kRuntime, "calling an undefined value");
    }
    if (callee.kind() != Kind::kCallable) {
      return Fail(Code::kRuntime, "calling a value that is not callable");
    }
    const Callable& c = callee.callable();
    switch (c.type) {
      case Callable::Type::kMacro:
        return CallMacro(c, args);
      case Callable::Type::kRange:
        return CallRange(args);
      case Callable::Type::kNamespace: {
        auto ns = std::make_shared<Namespace>();
        Track(namespaces_, ns);
        if (args.positional.size() > 1) {
          return Fail(Code::kRuntime, "namespace() takes at most one mapping");
        }
        std::size_t bytes = 64;
        if (!args.positional.empty()) {
          if (!args.positional[0].is_dict()) {
            return Fail(Code::kRuntime, "namespace() takes at most one mapping");
          }
          // The members are copied, names and all: charged as built and held.
          const auto& members = args.positional[0].dict().members;
          std::size_t names = 0;
          for (const auto& [k, v] : members) {
            names += k.size();
          }
          if (!arena_.Work(names + (members.size() * 16)) || !ns->Charge(&arena_, names)) {
            return Bound();
          }
          ns->members = members;
        }
        for (const auto& [k, v] : args.keywords) {
          Value* existing = FindIn(*ns, k);
          if (!arena_.ok()) {
            return Bound();
          }
          if (existing != nullptr) {
            *existing = v;
          } else {
            ns->members.emplace_back(k, v);
            bytes += k.size();
          }
        }
        bytes += ns->members.size() * (sizeof(Value) + 48);
        if (!ns->Charge(&arena_, bytes)) {
          return Bound();
        }
        return Value::MakeNamespace(std::move(ns));
      }
      case Callable::Type::kDict:
        if (!args.positional.empty()) {
          return Fail(Code::kUnsupported, "dict() with positional arguments");
        }
        return MakeDict(args.keywords, true);
      case Callable::Type::kRaise:
        return Fail(Code::kRaised, "the chat template raised an exception");
      case Callable::Type::kStrftime:
        return Strftime(args);
      case Callable::Type::kMethod:
        return CallMethod(c.receiver, c.name, args);
      case Callable::Type::kCycle:
        if (args.positional.empty()) {
          return Fail(Code::kRuntime, "loop.cycle() without values");
        }
        return args.positional[c.index % args.positional.size()];
    }
    return Value();
  }

  std::expected<Value, Error> CallMacro(const Callable& c, const Args& args) {
    if (++macro_depth_ > arena_.limits().max_macro_depth) {
      --macro_depth_;
      return Fail(Code::kLimit, "macro calls nest deeper than their bound");
    }
    auto result = [&]() -> std::expected<Value, Error> {
      const Stmt& m = *c.macro;
      auto frame = std::make_shared<Scope>();
      frame->parent = std::static_pointer_cast<const Scope>(c.scope);
      if (!frame->Charge(&arena_, 256)) {
        return Bound();
      }
      if (args.positional.size() > m.params.size() && !m.uses_varargs) {
        return Fail(Code::kRuntime, "a macro called with too many arguments");
      }
      std::vector<bool> bound(m.params.size(), false);
      for (std::size_t i = 0; i < m.params.size() && i < args.positional.size(); ++i) {
        if (!frame->Set(m.params[i].name, args.positional[i], arena_)) {
          return Bound();
        }
        bound[i] = true;
      }
      std::vector<std::pair<std::string, Value>> extra;
      for (const auto& [k, v] : args.keywords) {
        bool matched = false;
        // Every parameter is compared: charged as a search (FindMember).
        std::uint64_t work = 0;
        for (std::size_t i = 0; i < m.params.size(); ++i) {
          work += kEntryWork + (m.params[i].name.size() == k.size() ? k.size() : 0);
          if (m.params[i].name == k) {
            if (bound[i]) {
              return Fail(Code::kRuntime, "a macro argument given twice");
            }
            if (!frame->Set(k, v, arena_)) {
              return Bound();
            }
            bound[i] = true;
            matched = true;
          }
        }
        if (!arena_.Work(work)) {
          return Bound();
        }
        if (!matched) {
          if (!m.uses_kwargs) {
            return Fail(Code::kRuntime, "a macro called with an unknown keyword argument");
          }
          extra.emplace_back(k, v);
        }
      }
      for (std::size_t i = 0; i < m.params.size(); ++i) {
        if (bound[i]) {
          continue;
        }
        if (m.params[i].default_value) {
          auto v = Eval(*m.params[i].default_value, *frame);
          if (!v) {
            return v;
          }
          if (!frame->Set(m.params[i].name, std::move(*v), arena_)) {
            return Bound();
          }
        } else if (!frame->Set(m.params[i].name, Value(), arena_)) {
          return Bound();
        }
      }
      std::vector<Value> rest;
      for (std::size_t i = m.params.size(); i < args.positional.size(); ++i) {
        rest.push_back(args.positional[i]);
      }
      auto varargs = MakeList(std::move(rest), true);
      if (!varargs) {
        return varargs;
      }
      if (!frame->Set("varargs", std::move(*varargs), arena_)) {
        return Bound();
      }
      auto kwargs = MakeDict(std::move(extra), true);
      if (!kwargs) {
        return kwargs;
      }
      if (!frame->Set("kwargs", std::move(*kwargs), arena_)) {
        return Bound();
      }
      StrBuilder out;
      std::shared_ptr<Scope> scope = frame;
      auto flow = Exec(m.body, scope, out);
      if (!flow) {
        return std::unexpected(flow.error());
      }
      return Take(out);
    }();
    --macro_depth_;
    return result;
  }

  std::expected<Value, Error> CallRange(const Args& args) {
    std::array<std::int64_t, 3> a = {0, 0, 1};
    const std::size_t n = args.positional.size();
    if (n == 0 || n > 3 || !args.keywords.empty()) {
      return Fail(Code::kRuntime, "range() takes one to three integers");
    }
    for (const Value& v : args.positional) {
      if (!v.is_integral()) {
        return Fail(Code::kRuntime, "range() takes one to three integers");
      }
    }
    if (n == 1) {
      a[1] = args.positional[0].integer();
    } else {
      a[0] = args.positional[0].integer();
      a[1] = args.positional[1].integer();
      if (n == 3) {
        a[2] = args.positional[2].integer();
      }
    }
    if (a[2] == 0) {
      return Fail(Code::kRuntime, "range() step of zero");
    }
    std::vector<Value> items;
    for (std::int64_t i = a[0]; a[2] > 0 ? i < a[1] : i > a[1]; i += a[2]) {
      if (items.size() >= arena_.limits().max_range) {
        return Fail(Code::kLimit, "a range longer than its bound");
      }
      items.push_back(Value::Int(i));
      if ((a[2] > 0 && i > std::numeric_limits<std::int64_t>::max() - a[2]) ||
          (a[2] < 0 && i < std::numeric_limits<std::int64_t>::min() - a[2])) {
        break;
      }
    }
    return MakeList(std::move(items));
  }

  std::expected<Value, Error> Strftime(const Args& args) {
    if (args.positional.size() != 1 || !args.positional[0].is_string()) {
      return Fail(Code::kRuntime, "strftime_now() takes a format string");
    }
    if (!now_) {
      return Fail(Code::kUnsupported, "strftime_now() without a clock");
    }
    static constexpr std::array<std::string_view, 12> kMonths = {
        "January", "February", "March",     "April",   "May",      "June",
        "July",    "August",   "September", "October", "November", "December"};
    static constexpr std::array<std::string_view, 7> kDays = {
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    const CivilTime& t = *now_;
    const std::string_view fmt = args.positional[0].str().text;
    // Each directive costs a conversion, whatever it prints: the format is
    // charged as scanned at that rate.
    if (!arena_.Work(fmt.size() * 8)) {
      return Bound();
    }
    std::string out;
    const auto number = [&](int v, int width, bool pad) {
      std::string digits = std::to_string(v);
      while (pad && std::cmp_less(digits.size(), width)) {
        digits.insert(digits.begin(), '0');
      }
      out += digits;
    };
    for (std::size_t i = 0; i < fmt.size(); ++i) {
      if (out.size() > arena_.limits().max_string_bytes) {
        return Fail(Code::kLimit, "a string longer than its bound");
      }
      if (fmt[i] != '%') {
        out.push_back(fmt[i]);
        continue;
      }
      if (++i >= fmt.size()) {
        return Fail(Code::kUnsupported, "a strftime format directive the subset leaves out");
      }
      bool pad = true;
      if (fmt[i] == '-' && i + 1 < fmt.size()) {
        pad = false;
        ++i;
      }
      const int hour12 = t.hour % 12 == 0 ? 12 : t.hour % 12;
      switch (fmt[i]) {
        case 'Y':
          number(t.year, 4, pad);
          break;
        case 'y':
          number(t.year % 100, 2, pad);
          break;
        case 'm':
          number(t.month, 2, pad);
          break;
        case 'd':
          number(t.day, 2, pad);
          break;
        case 'e':
          if (pad && t.day < 10) {
            out.push_back(' ');
          }
          number(t.day, 1, false);
          break;
        case 'H':
          number(t.hour, 2, pad);
          break;
        case 'I':
          number(hour12, 2, pad);
          break;
        case 'M':
          number(t.minute, 2, pad);
          break;
        case 'S':
          number(t.second, 2, pad);
          break;
        case 'j':
          number(t.yearday + 1, 3, pad);
          break;
        case 'p':
          out += t.hour < 12 ? "AM" : "PM";
          break;
        case 'B':
          out += kMonths[static_cast<std::size_t>((t.month + 11) % 12)];
          break;
        case 'b':
          out += kMonths[static_cast<std::size_t>((t.month + 11) % 12)].substr(0, 3);
          break;
        case 'A':
          out += kDays[static_cast<std::size_t>(t.weekday % 7)];
          break;
        case 'a':
          out += kDays[static_cast<std::size_t>(t.weekday % 7)].substr(0, 3);
          break;
        case '%':
          out.push_back('%');
          break;
        default:
          return Fail(Code::kUnsupported, "a strftime format directive the subset leaves out");
      }
    }
    return Str(out, args.positional[0].str().all_trusted());
  }

  // ---------------------------------------------------------------- strings

  // Python's str.strip family: whitespace, or the code points in `chars`.
  std::expected<Value, Error> Strip(const Value& v, const Value* chars, bool left, bool right) {
    const jinja::Str& s = v.str();
    const std::string_view text = s.text;
    std::vector<char32_t> set;    // sorted, distinct: a long `chars` must not cost per code point
    std::size_t scan_factor = 1;  // a text code point costs a binary search of `set`
    if (chars != nullptr && !chars->is_none()) {
      if (!chars->is_string()) {
        return Fail(Code::kRuntime, "strip characters that are not a string");
      }
      const std::size_t chars_bytes = chars->str().text.size();
      // The set holds one code point per `chars` byte at most; bound its
      // memory as the slice and reverse paths do, so a huge `chars` cannot
      // allocate past the live budget uncharged.
      if (chars_bytes > arena_.limits().max_live_bytes / sizeof(char32_t)) {
        return Fail(Code::kLimit, "a strip character set longer than its bound");
      }
      if (!arena_.Work(chars_bytes * 8)) {
        return Bound();
      }
      set.reserve(CodePoints(chars->str().text));
      for (std::size_t at = 0; at < chars_bytes;) {
        std::size_t length = 0;
        set.push_back(DecodeAt(chars->str().text, at, length));
        at += length;
      }
      // Sorting costs k*log2(k) and each later membership test log2(k): charge
      // both, so strip meets the work bound in about the time they take rather
      // than at the linear rate (the set's sort and searches are not free).
      std::size_t log_k = 1;
      while ((std::size_t{1} << log_k) < set.size()) {
        ++log_k;
      }
      if (!arena_.Work(set.size() * log_k)) {
        return Bound();
      }
      std::ranges::sort(set);
      const auto distinct = std::ranges::unique(set);
      set.erase(distinct.begin(), distinct.end());
      scan_factor = log_k;
    }
    const auto strip_at = [&](std::size_t at) -> std::size_t {
      if (chars == nullptr || chars->is_none()) {
        return SpaceAt(text, at);
      }
      std::size_t length = 0;
      const char32_t cp = DecodeAt(text, at, length);
      return std::ranges::binary_search(set, cp) ? length : 0;
    };
    if (!arena_.Work(text.size() * scan_factor)) {
      return Bound();
    }
    std::size_t begin = 0;
    if (left) {
      while (begin < text.size()) {
        const std::size_t n = strip_at(begin);
        if (n == 0) {
          break;
        }
        begin += n;
      }
    }
    std::size_t end = text.size();
    if (right) {
      // From the end, a code point at a time, as Python strips.
      while (end > begin) {
        std::size_t at = end - 1;
        while (at > begin && (static_cast<unsigned char>(text[at]) & 0xC0U) == 0x80U) {
          --at;
        }
        if (strip_at(at) == 0) {
          break;
        }
        end = at;
      }
    }
    StrBuilder b;
    b.Append(s, begin, end - begin);
    b.set_markup(s.markup);
    return Take(b);
  }

  std::expected<Value, Error> Split(const Value& v, const Args& args, bool from_right) {
    const jinja::Str& s = v.str();
    const std::string_view text = s.text;
    const Value* sep = args.Get(0, "sep");
    const Value* max = args.Get(1, "maxsplit");
    std::int64_t maxsplit = -1;
    if (max != nullptr) {
      if (!max->is_integral()) {
        return Fail(Code::kRuntime, "a maxsplit that is not an integer");
      }
      maxsplit = max->integer();
    }
    if (!arena_.Work(text.size())) {
      return Bound();
    }
    // Each part becomes a value: no more than the values could hold.
    const std::size_t max_parts = arena_.limits().max_live_bytes / sizeof(Value);
    const auto too_many = [&](std::size_t parts) { return parts >= max_parts; };
    std::vector<std::pair<std::size_t, std::size_t>> parts;  // (offset, length)
    if (sep == nullptr || sep->is_none()) {
      // Runs of whitespace; none at either end.
      std::vector<std::pair<std::size_t, std::size_t>> words;
      for (std::size_t at = 0; at < text.size();) {
        if (const std::size_t n = SpaceAt(text, at)) {
          at += n;
          continue;
        }
        const std::size_t start = at;
        while (at < text.size() && SpaceAt(text, at) == 0) {
          std::size_t length = 0;
          DecodeAt(text, at, length);
          at += length;
        }
        if (too_many(words.size())) {
          return Fail(Code::kLimit, "a split into more parts than their bound");
        }
        words.emplace_back(start, at - start);
      }
      if (maxsplit >= 0 && words.size() > static_cast<std::size_t>(maxsplit) + 1) {
        // The remainder keeps its whitespace on the far side: through the
        // end for split, from the start for rsplit.
        const auto keep = static_cast<std::size_t>(maxsplit);
        if (!from_right) {
          const std::size_t rest = words[keep].first;
          words.resize(keep);
          words.emplace_back(rest, text.size() - rest);
        } else {
          const std::size_t first_kept = words.size() - keep;
          const std::size_t end = words[first_kept - 1].first + words[first_kept - 1].second;
          std::vector<std::pair<std::size_t, std::size_t>> parts_kept;
          parts_kept.emplace_back(0, end);
          parts_kept.insert(parts_kept.end(),
                            words.begin() + static_cast<std::ptrdiff_t>(first_kept), words.end());
          words = std::move(parts_kept);
        }
      }
      parts = std::move(words);
    } else {
      if (!sep->is_string()) {
        return Fail(Code::kRuntime, "a separator that is not a string");
      }
      const std::string_view needle = sep->str().text;
      if (needle.empty()) {
        return Fail(Code::kRuntime, "an empty separator");
      }
      if (auto r = CheckNeedle(needle); !r) {
        return std::unexpected(r.error());
      }
      const Searcher searcher(needle);
      if (!from_right) {
        std::size_t start = 0;
        std::int64_t splits = 0;
        while (maxsplit < 0 || splits < maxsplit) {
          const std::size_t found = searcher.Find(text, start);
          if (found == std::string_view::npos) {
            break;
          }
          if (too_many(parts.size())) {
            return Fail(Code::kLimit, "a split into more parts than their bound");
          }
          parts.emplace_back(start, found - start);
          start = found + needle.size();
          ++splits;
        }
        parts.emplace_back(start, text.size() - start);
      } else {
        std::size_t end = text.size();
        std::int64_t splits = 0;
        std::vector<std::pair<std::size_t, std::size_t>> reversed;
        while (maxsplit < 0 || splits < maxsplit) {
          const std::size_t found = searcher.RFind(text, end);
          if (found == std::string_view::npos) {
            break;
          }
          if (too_many(reversed.size())) {
            return Fail(Code::kLimit, "a split into more parts than their bound");
          }
          reversed.emplace_back(found + needle.size(), end - found - needle.size());
          end = found;
          ++splits;
        }
        reversed.emplace_back(0, end);
        parts.assign(reversed.rbegin(), reversed.rend());
      }
    }
    if (!arena_.Work(parts.size() * kValueWork)) {
      return Bound();
    }
    std::vector<Value> items;
    items.reserve(parts.size());
    for (const auto& [offset, length] : parts) {
      auto part = Sub(s, offset, length);
      if (!part) {
        return part;
      }
      items.push_back(std::move(*part));
    }
    return MakeList(std::move(items));
  }

  std::expected<Value, Error> Replace(const Value& v, const Value& old_value,
                                      const Value& new_value, std::int64_t count) {
    if (!old_value.is_string() || !new_value.is_string()) {
      return Fail(Code::kRuntime, "replace() with something other than strings");
    }
    const jinja::Str& s = v.str();
    const std::string_view needle = old_value.str().text;
    if (!arena_.Work(s.text.size())) {
      return Bound();
    }
    StrBuilder out;
    std::size_t at = 0;
    std::int64_t done = 0;
    if (needle.empty()) {
      // Python inserts the replacement around every code point: work a
      // code point at a time, charged as such.
      if (!arena_.Work(s.text.size() * 16)) {
        return Bound();
      }
      while (count < 0 || done < count) {
        out.Append(new_value.str());
        ++done;
        if (at >= s.text.size()) {
          break;
        }
        std::size_t length = 0;
        DecodeAt(s.text, at, length);
        out.Append(s, at, length);
        at += length;
        if (out.size() > arena_.limits().max_string_bytes) {
          return Fail(Code::kLimit, "a string longer than its bound");
        }
      }
      out.Append(s, at, s.text.size() - at);
      return Take(out);
    }
    if (auto r = CheckNeedle(needle); !r) {
      return std::unexpected(r.error());
    }
    const Searcher searcher(needle);
    while (count < 0 || done < count) {
      const std::size_t found = searcher.Find(s.text, at);
      if (found == std::string::npos) {
        break;
      }
      out.Append(s, at, found - at);
      out.Append(new_value.str());
      at = found + needle.size();
      ++done;
      if (out.size() > arena_.limits().max_string_bytes) {
        return Fail(Code::kLimit, "a string longer than its bound");
      }
    }
    out.Append(s, at, s.text.size() - at);
    return Take(out);
  }

  // Python's case mapping (chat/pycase.h); each code point's mapping keeps
  // the provenance of the code point it maps.
  std::expected<Value, Error> MapCase(const Value& v, CaseOp op) {
    const jinja::Str& s = v.str();
    if (!arena_.Work(CaseWork(s.text))) {
      return Bound();
    }
    class Sink final : public CaseSink {
     public:
      explicit Sink(const jinja::Str& s) : s_(s) {}
      void Emit(std::size_t source_offset, std::size_t source_length,
                std::string_view mapped) override {
        // One code point, or an ASCII run mapped byte for byte: each output
        // byte takes its source byte's provenance.
        const bool bytewise = mapped.size() == source_length;
        std::size_t at = 0;
        while (at < mapped.size()) {
          const std::size_t source = source_offset + (bytewise ? at : 0);
          while (range_ < s_.trusted.size() &&
                 s_.trusted[range_].offset + s_.trusted[range_].length <= source) {
            ++range_;
          }
          const bool in = range_ < s_.trusted.size() && s_.trusted[range_].offset <= source;
          std::size_t n = mapped.size() - at;  // the rest, unless a range edge comes first
          if (bytewise && range_ < s_.trusted.size()) {
            const Range& r = s_.trusted[range_];
            n = std::min(n, (in ? r.offset + r.length : r.offset) - source);
          }
          b.Append(mapped.substr(at, n), in);
          at += n;
        }
      }
      StrBuilder b;

     private:
      const jinja::Str& s_;
      std::size_t range_ = 0;
    } sink(s);
    MapPythonCase(s.text, op, sink);
    // A mapping grows a code point at most threefold.
    if (!arena_.Work(sink.b.size())) {
      return Bound();
    }
    if (sink.b.size() > arena_.limits().max_string_bytes) {
      return Fail(Code::kLimit, "a string longer than its bound");
    }
    sink.b.set_markup(s.markup);
    return Take(sink.b);
  }

  static CaseOp CaseOpNamed(std::string_view name, bool filter) {
    if (name == "upper") {
      return CaseOp::kUpper;
    }
    if (name == "lower") {
      return CaseOp::kLower;
    }
    if (name == "capitalize") {
      return CaseOp::kCapitalize;
    }
    return filter ? CaseOp::kJinjaTitle : CaseOp::kTitle;  // the filter's rule, or str.title
  }

  std::expected<Value, Error> Join(const std::vector<Value>& items, const Value* separator,
                                   const Value* attribute) {
    StrBuilder out;
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (i != 0 && separator != nullptr) {
        if (auto r = AppendStr(*separator, out, arena_); !r) {
          return std::unexpected(r.error());
        }
      }
      Value item = items[i];
      if (attribute != nullptr) {
        auto a = Attribute(item, *attribute);
        if (!a) {
          return a;
        }
        item = std::move(*a);
      }
      if (auto r = AppendStr(item, out, arena_); !r) {
        return std::unexpected(r.error());
      }
      if (out.size() > arena_.limits().max_string_bytes) {
        return Fail(Code::kLimit, "a string longer than its bound");
      }
    }
    return Take(out);
  }

  std::expected<Value, Error> CallMethod(const Value& receiver, std::string_view name,
                                         const Args& args) {
    if (receiver.is_string()) {
      const jinja::Str& s = receiver.str();
      if (name == "strip" || name == "lstrip" || name == "rstrip") {
        return Strip(receiver, args.Get(0, "chars"), name != "rstrip", name != "lstrip");
      }
      if (name == "split" || name == "rsplit") {
        return Split(receiver, args, name == "rsplit");
      }
      if (name == "upper" || name == "lower" || name == "title" || name == "capitalize") {
        return MapCase(receiver, CaseOpNamed(name, false));
      }
      if (name == "startswith" || name == "endswith") {
        const Value* prefix = args.Get(0, "prefix");
        if (prefix == nullptr || args.positional.size() > 1) {
          return Fail(Code::kUnsupported, "startswith or endswith with a range");
        }
        const bool tuple = prefix->is_list() && prefix->list().tuple;
        const std::span<const Value> options = tuple ? std::span<const Value>(prefix->list().items)
                                                     : std::span<const Value>(prefix, 1);
        for (const Value& o : options) {
          if (!o.is_string()) {
            return Fail(Code::kRuntime,
                        "startswith or endswith with something other than a string");
          }
          // Each option compares up to its length.
          if (!arena_.Work(16 + std::min(o.str().text.size(), s.text.size()))) {
            return Bound();
          }
          if (name == "startswith" ? std::string_view(s.text).starts_with(o.str().text)
                                   : std::string_view(s.text).ends_with(o.str().text)) {
            return Value::Bool(true);
          }
        }
        return Value::Bool(false);
      }
      if (name == "replace") {
        const Value* old_value = args.Get(0, "old");
        const Value* new_value = args.Get(1, "new");
        const Value* count = args.Get(2, "count");
        if (old_value == nullptr || new_value == nullptr) {
          return Fail(Code::kRuntime, "replace() needs two strings");
        }
        if (count != nullptr && !count->is_integral()) {
          return Fail(Code::kRuntime, "a replace count that is not an integer");
        }
        return Replace(receiver, *old_value, *new_value, count != nullptr ? count->integer() : -1);
      }
      if (name == "find" || name == "rfind" || name == "count") {
        const Value* sub = args.Get(0, "sub");
        if (sub == nullptr || !sub->is_string() || args.positional.size() > 1) {
          return Fail(Code::kUnsupported, "find, rfind or count other than with one string");
        }
        if (!arena_.Work(s.text.size())) {
          return Bound();
        }
        const std::string_view needle = sub->str().text;
        if (auto r = CheckNeedle(needle); !r) {
          return std::unexpected(r.error());
        }
        if (name == "count") {
          if (needle.empty()) {
            return Value::Int(static_cast<std::int64_t>(CodePoints(s.text) + 1));
          }
          const Searcher searcher(needle);
          std::int64_t n = 0;
          for (std::size_t at = searcher.Find(s.text); at != std::string::npos;
               at = searcher.Find(s.text, at + needle.size())) {
            ++n;
          }
          return Value::Int(n);
        }
        const Searcher searcher(needle);
        const std::size_t at =
            name == "find" ? searcher.Find(s.text) : searcher.RFind(s.text, s.text.size());
        if (at == std::string::npos) {
          return Value::Int(-1);
        }
        return Value::Int(
            static_cast<std::int64_t>(CodePoints(std::string_view(s.text).substr(0, at))));
      }
      if (name == "join") {
        if (args.positional.size() != 1) {
          return Fail(Code::kRuntime, "join() takes one iterable");
        }
        auto items = Iterate(args.positional[0]);
        if (!items) {
          return std::unexpected(items.error());
        }
        for (const Value& item : *items) {
          if (!item.is_string()) {
            return Fail(Code::kRuntime, "str.join() of something other than strings");
          }
        }
        return Join(*items, &receiver, nullptr);
      }
      if (name == "removeprefix" || name == "removesuffix") {
        const Value* affix = args.Get(0, "prefix");
        if (affix == nullptr || !affix->is_string()) {
          return Fail(Code::kRuntime, "removeprefix or removesuffix without a string");
        }
        const std::string_view a = affix->str().text;
        if (!arena_.Work(std::min(a.size(), s.text.size()))) {  // the bytes compared
          return Bound();
        }
        if (name == "removeprefix" && std::string_view(s.text).starts_with(a)) {
          return Sub(s, a.size(), s.text.size() - a.size());
        }
        if (name == "removesuffix" && !a.empty() && std::string_view(s.text).ends_with(a)) {
          return Sub(s, 0, s.text.size() - a.size());
        }
        return receiver;
      }
      if (name == "splitlines") {
        return SplitLines(receiver, false);
      }
      return Fail(Code::kUnsupported, "a string method the subset does not implement");
    }
    if (receiver.is_dict()) {
      const Dict& d = receiver.dict();
      if (name == "get") {
        const Value* key = args.Get(0, "key");
        if (key == nullptr) {
          return Fail(Code::kRuntime, "get() needs a key");
        }
        if (key->is_string()) {
          const Value* v = FindIn(d, key->str().text);
          if (!arena_.ok()) {
            return Bound();
          }
          if (v != nullptr) {
            return *v;
          }
        }
        const Value* fallback = args.Get(1, "default");
        return fallback != nullptr ? *fallback : Value::None();
      }
      if (name == "keys") {
        auto keys = Iterate(receiver);
        if (!keys) {
          return std::unexpected(keys.error());
        }
        return MakeList(std::move(*keys));
      }
      if (name == "values") {
        std::vector<Value> values;
        values.reserve(d.members.size());
        for (const auto& [k, v] : d.members) {
          values.push_back(v);
        }
        return MakeList(std::move(values));
      }
      if (name == "items") {
        return Items(receiver);
      }
      if (name == "copy") {
        return receiver;
      }
      if (name == "pop" || name == "popitem" || name == "setdefault" || name == "update" ||
          name == "clear") {
        return Fail(Code::kRuntime, "the sandbox forbids changing a mapping");
      }
      return Fail(Code::kUnsupported, "a mapping method the subset does not implement");
    }
    if (receiver.is_list()) {
      const List& l = receiver.list();
      if (name == "count") {
        if (args.positional.size() != 1) {
          return Fail(Code::kRuntime, "count() takes one value");
        }
        const auto n = std::ranges::count_if(
            l.items, [&](const Value& item) { return Eq(item, args.positional[0]); });
        if (!arena_.ok()) {
          return Bound();
        }
        return Value::Int(static_cast<std::int64_t>(n));
      }
      if (name == "index") {
        if (args.positional.size() != 1) {
          return Fail(Code::kUnsupported, "index() with a range");
        }
        for (std::size_t i = 0; i < l.items.size(); ++i) {
          auto eq = CheckedEq(l.items[i], args.positional[0]);
          if (!eq) {
            return std::unexpected(eq.error());
          }
          if (*eq) {
            return Value::Int(static_cast<std::int64_t>(i));
          }
        }
        return Fail(Code::kRuntime, "index() of a value not in the list");
      }
      if (name == "copy") {
        return receiver;
      }
      return Fail(Code::kRuntime, "the sandbox forbids changing a list");
    }
    return Fail(Code::kUnsupported, "a method the subset does not implement");
  }

  // Python's str.splitlines().
  std::expected<Value, Error> SplitLines(const Value& v, bool keepends) {
    const jinja::Str& s = v.str();
    const std::string_view text = s.text;
    if (!arena_.Work(text.size())) {
      return Bound();
    }
    std::vector<Value> lines;
    std::size_t start = 0;
    for (std::size_t at = 0; at < text.size();) {
      std::size_t length = 0;
      const char32_t cp = DecodeAt(text, at, length);
      const bool brk = cp == U'\n' || cp == U'\r' || cp == 0x0B || cp == 0x0C || cp == 0x1C ||
                       cp == 0x1D || cp == 0x1E || cp == 0x85 || cp == 0x2028 || cp == 0x2029;
      if (!brk) {
        at += length;
        continue;
      }
      if (!arena_.Work(kValueWork)) {
        return Bound();
      }
      std::size_t end = at + length;
      if (cp == U'\r' && end < text.size() && text[end] == '\n') {
        ++end;
      }
      auto line = Sub(s, start, (keepends ? end : at) - start);
      if (!line) {
        return line;
      }
      lines.push_back(std::move(*line));
      start = end;
      at = end;
    }
    if (start < text.size()) {
      auto line = Sub(s, start, text.size() - start);
      if (!line) {
        return line;
      }
      lines.push_back(std::move(*line));
    }
    return MakeList(std::move(lines));
  }

  std::expected<Value, Error> Items(const Value& v) {
    std::vector<Value> pairs;
    for (const auto& [k, value] : v.dict().members) {
      auto key = Str(k, v.dict().keys_trusted);
      if (!key) {
        return key;
      }
      auto pair = MakeList({std::move(*key), value}, true);
      if (!pair) {
        return pair;
      }
      pairs.push_back(std::move(*pair));
    }
    return MakeList(std::move(pairs));
  }

  // Jinja2's attribute getter for filters: dotted names, each looked up as
  // an item first (environment.getitem).
  std::expected<Value, Error> Attribute(const Value& item, const Value& attribute) {
    if (attribute.is_integral()) {
      return GetItem(item, attribute);
    }
    if (!attribute.is_string()) {
      return Fail(Code::kRuntime, "an attribute name that is not a string");
    }
    Value v = item;
    const std::string_view path = attribute.str().text;
    std::size_t start = 0;
    while (true) {
      const std::size_t dot = path.find('.', start);
      const std::string_view part =
          path.substr(start, dot == std::string_view::npos ? dot : dot - start);
      Value key;
      std::int64_t index = 0;
      bool numeric = !part.empty();
      for (const char c : part) {
        numeric = numeric && c >= '0' && c <= '9';
      }
      if (numeric && part.size() < 18) {
        for (const char c : part) {
          index = (index * 10) + (c - '0');
        }
        key = Value::Int(index);
      } else {
        auto k = Str(part, true);
        if (!k) {
          return k;
        }
        key = std::move(*k);
      }
      auto next = GetItem(v, key);
      if (!next) {
        return next;
      }
      v = std::move(*next);
      if (dot == std::string_view::npos) {
        return v;
      }
      start = dot + 1;
    }
  }

  // `mapping` is str %'s single mapping argument, which need not be used.
  std::expected<Value, Error> Format(const Value& format, const Args& args, bool mapping = false) {
    if (!args.keywords.empty()) {
      return Fail(Code::kUnsupported, "format with keyword arguments");
    }
    const std::string_view f = format.str().text;
    // The format is scanned whatever it builds (a directive may print
    // nothing): charged as such.
    if (!arena_.Work(f.size() * 4)) {
      return Bound();
    }
    StrBuilder out;
    std::size_t next = 0;
    std::size_t from = 0;
    for (std::size_t i = 0; i < f.size(); ++i) {
      if (f[i] != '%') {
        continue;
      }
      out.Append(format.str(), from, i - from);
      if (i + 1 >= f.size()) {
        return Fail(Code::kRuntime, "an incomplete format");
      }
      const char d = f[++i];
      from = i + 1;
      if (d == '%') {
        out.Append("%", format.str().all_trusted());
        continue;
      }
      if (next >= args.positional.size()) {
        return Fail(Code::kRuntime, "not enough arguments for the format");
      }
      const Value& a = args.positional[next++];
      if (d == 's') {
        if (auto r = AppendStr(a, out, arena_); !r) {
          return std::unexpected(r.error());
        }
      } else if (d == 'd' || d == 'i') {
        if (!a.is_number() || a.kind() == Kind::kBigInt) {
          return Fail(Code::kRuntime, "a %d format of something other than a number");
        }
        const std::int64_t n =
            a.is_integral() ? a.integer() : static_cast<std::int64_t>(std::trunc(a.number()));
        out.Append(std::to_string(n), a.trusted());
      } else {
        return Fail(Code::kUnsupported, "a format directive other than %s, %d, %i or %%");
      }
      if (out.size() > arena_.limits().max_string_bytes) {
        return Fail(Code::kLimit, "a string longer than its bound");
      }
    }
    if (next != args.positional.size() && !mapping) {
      return Fail(Code::kRuntime, "too many arguments for the format");
    }
    out.Append(format.str(), from, f.size() - from);
    return Take(out);
  }

  // ---------------------------------------------------------------- filters

  // The template's own values: every string fully the template's.
  static bool DeepTrusted(const Value& v) {
    switch (v.kind()) {
      case Kind::kString:
        return v.str().all_trusted();
      case Kind::kBigInt:
        return false;
      case Kind::kList:
        return std::ranges::all_of(v.list().items, DeepTrusted);
      case Kind::kDict:
        return v.dict().keys_trusted && std::ranges::all_of(v.dict().members, [](const auto& m) {
                 return DeepTrusted(m.second);
               });
      default:
        return v.trusted();
    }
  }

  std::expected<Value, Error> ToJson(const Value& v, const Args& args) {
    JsonOptions o;
    // The text is the template's only when everything it is made of is:
    // the value, and an indent or separators passed as strings.
    bool trusted = true;  // and the value's, once its text is known to be bounded
    if (const Value* a = args.Get(0, "ensure_ascii")) {
      o.ensure_ascii = Truthy(*a);
    }
    if (const Value* indent = args.Get(1, "indent"); indent != nullptr && !indent->is_none()) {
      if (indent->is_integral()) {
        // Python's ' ' * indent; nothing for a negative indent.
        const std::int64_t n = std::max<std::int64_t>(indent->integer(), 0);
        if (std::cmp_greater(n, arena_.limits().max_string_bytes)) {
          return Fail(Code::kLimit, "a string longer than its bound");
        }
        o.indent = std::string(static_cast<std::size_t>(n), ' ');
      } else if (indent->is_string()) {
        o.indent = indent->str().text;
        trusted = trusted && indent->str().all_trusted();
      } else {
        return Fail(Code::kRuntime, "a tojson indent that is not an integer or string");
      }
      o.item_separator = ",";
    }
    if (const Value* sep = args.Get(2, "separators"); sep != nullptr && !sep->is_none()) {
      if (!sep->is_list() || sep->list().items.size() != 2 || !sep->list().items[0].is_string() ||
          !sep->list().items[1].is_string()) {
        return Fail(Code::kRuntime, "tojson separators that are not two strings");
      }
      o.item_separator = sep->list().items[0].str().text;
      o.key_separator = sep->list().items[1].str().text;
      trusted = trusted && sep->list().items[0].str().all_trusted() &&
                sep->list().items[1].str().all_trusted();
    }
    if (const Value* sort = args.Get(3, "sort_keys")) {
      o.sort_keys = Truthy(*sort);
    }
    std::string text;
    if (auto r = AppendJson(v, o, text, arena_); !r) {
      return std::unexpected(r.error());
    }
    // After the text: each value visited printed a byte, so a value shared
    // many times over (lists of lists of one list) costs no more than its
    // bounded text.
    return Str(text, trusted && DeepTrusted(v));
  }

  std::expected<Value, Error> FilterByTest(const Value& seq, const Args& args, bool select,
                                           bool by_attribute) {
    auto items = Iterate(seq);
    if (!items) {
      return std::unexpected(items.error());
    }
    std::size_t at = 0;
    const Value* attribute = nullptr;
    if (by_attribute) {
      attribute = args.Get(0, "attribute");
      if (attribute == nullptr) {
        return Fail(Code::kRuntime, "selectattr or rejectattr without an attribute");
      }
      at = 1;
    }
    std::optional<std::string> test;
    Args test_args;
    if (at < args.positional.size()) {
      if (!args.positional[at].is_string()) {
        return Fail(Code::kRuntime, "a test name that is not a string");
      }
      test = args.positional[at].str().text;
      if (!KnownTest(*test)) {
        return Fail(Code::kUnsupported, "a test the subset does not implement");
      }
      test_args.positional.assign(args.positional.begin() + static_cast<std::ptrdiff_t>(at + 1),
                                  args.positional.end());
    }
    std::vector<Value> kept;
    for (const Value& item : *items) {
      Value subject = item;
      if (attribute != nullptr) {
        auto a = Attribute(item, *attribute);
        if (!a) {
          return a;
        }
        subject = std::move(*a);
      }
      bool pass = false;
      if (test) {
        auto r = ApplyTest(*test, subject, test_args);
        if (!r) {
          return std::unexpected(r.error());
        }
        pass = *r;
      } else {
        pass = Truthy(subject);
      }
      if (pass == select) {
        kept.push_back(item);
      }
    }
    return MakeList(std::move(kept));
  }

  // A sort key: strings lower-cased unless case_sensitive.
  std::expected<Value, Error> SortKey(const Value& v, bool case_sensitive) {
    if (case_sensitive || !v.is_string()) {
      return v;
    }
    return MapCase(v, CaseOp::kLower);
  }

  std::expected<Value, Error> Sorted(std::vector<Value> items, bool reverse, bool case_sensitive,
                                     const Value* attribute) {
    std::vector<std::pair<Value, Value>> keyed;
    for (Value& item : items) {
      Value key = item;
      if (attribute != nullptr && !attribute->is_none()) {
        auto a = Attribute(item, *attribute);
        if (!a) {
          return a;
        }
        key = std::move(*a);
      }
      auto k = SortKey(key, case_sensitive);
      if (!k) {
        return k;
      }
      keyed.emplace_back(std::move(*k), std::move(item));
    }
    std::optional<Error> failed;
    std::ranges::stable_sort(keyed, [&](const auto& a, const auto& b) {
      auto lt = reverse ? Lt(b.first, a.first) : Lt(a.first, b.first);
      if (!lt) {
        failed = lt.error();
        return false;
      }
      return *lt;
    });
    if (failed) {
      return std::unexpected(*failed);
    }
    std::vector<Value> out;
    out.reserve(keyed.size());
    for (auto& [k, v] : keyed) {
      out.push_back(std::move(v));
    }
    return MakeList(std::move(out));
  }

  std::expected<Value, Error> Indent(const Value& v, const Args& args) {
    // The indent with its provenance: a string width is whatever made it,
    // spaces for a number are the template's.
    jinja::Str spaces;
    spaces.text = "    ";
    const jinja::Str* indent = &spaces;
    if (const Value* width = args.Get(0, "width")) {
      if (width->is_string()) {
        indent = &width->str();
      } else if (width->is_integral()) {
        // Python's ' ' * width; nothing for a negative width.
        const std::int64_t n = std::max<std::int64_t>(width->integer(), 0);
        if (std::cmp_greater(n, arena_.limits().max_string_bytes)) {
          return Fail(Code::kLimit, "a string longer than its bound");
        }
        spaces.text.assign(static_cast<std::size_t>(n), ' ');
      } else {
        return Fail(Code::kRuntime, "an indent width that is not an integer or string");
      }
    }
    if (indent == &spaces && !spaces.text.empty()) {
      spaces.trusted = {{0, spaces.text.size()}};
    }
    const Value* first = args.Get(1, "first");
    const Value* blank = args.Get(2, "blank");
    StrBuilder text;
    if (auto r = AppendStr(v, text, arena_); !r) {
      return std::unexpected(r.error());
    }
    text.Append("\n", true);  // Jinja2's quirk for splitlines
    auto whole = Take(text);
    if (!whole) {
      return whole;
    }
    auto lines = SplitLines(*whole, false);
    if (!lines) {
      return lines;
    }
    StrBuilder out;
    const auto& items = lines->list().items;
    for (std::size_t i = 0; i < items.size(); ++i) {
      const jinja::Str& line = items[i].str();
      if (i != 0) {
        out.Append("\n", true);
      }
      const bool indent_line = i == 0
                                   ? (first != nullptr && Truthy(*first))
                                   : (!line.text.empty() || (blank != nullptr && Truthy(*blank)));
      if (indent_line) {
        out.Append(*indent);
      }
      out.Append(line);
      if (out.size() > arena_.limits().max_string_bytes) {
        return Fail(Code::kLimit, "a string longer than its bound");
      }
    }
    return Take(out);
  }

  std::expected<Value, Error> ApplyFilter(const Expr& e, Value subject, const Scope& scope) {
    auto args = EvalArgs(e, 1, scope);
    if (!args) {
      return std::unexpected(args.error());
    }
    return Filter(e.name, std::move(subject), *args);
  }

  std::expected<Value, Error> Filter(std::string_view name, Value v, const Args& args) {
    if (!arena_.Step() || !ChargeArgs(args)) {
      return Bound();
    }
    const auto need_string = [&]() -> std::expected<Value, Error> {
      if (v.is_string()) {
        return v;
      }
      StrBuilder b;
      if (auto r = AppendStr(v, b, arena_); !r) {
        return std::unexpected(r.error());
      }
      return Take(b);
    };
    if (name == "safe") {
      auto s = need_string();
      if (!s) {
        return s;
      }
      StrBuilder b;
      b.Append(s->str());
      b.set_markup(true);
      return Take(b);
    }
    if (name == "escape" || name == "e") {
      if (v.is_string() && v.str().markup) {
        return v;
      }
      auto s = need_string();
      if (!s) {
        return s;
      }
      StrBuilder b;
      AppendEscaped(s->str(), b);
      b.set_markup(true);
      return Take(b);
    }
    if (name == "string") {
      return need_string();
    }
    if (name == "trim") {
      auto s = need_string();
      if (!s) {
        return s;
      }
      return Strip(*s, args.Get(0, "chars"), true, true);
    }
    if (name == "upper" || name == "lower" || name == "capitalize" || name == "title") {
      auto s = need_string();
      if (!s) {
        return s;
      }
      return MapCase(*s, CaseOpNamed(name, true));
    }
    if (name == "replace") {
      auto s = need_string();
      if (!s) {
        return s;
      }
      const Value* old_value = args.Get(0, "old");
      const Value* new_value = args.Get(1, "new");
      const Value* count = args.Get(2, "count");
      if (old_value == nullptr || new_value == nullptr) {
        return Fail(Code::kRuntime, "replace needs two strings");
      }
      if (count != nullptr && !count->is_none() && !count->is_integral()) {
        return Fail(Code::kRuntime, "a replace count that is not an integer");
      }
      StrBuilder plain;
      plain.Append(s->str());
      auto unmarked = Take(plain);
      if (!unmarked) {
        return unmarked;
      }
      // With autoescape off, Jinja2 replaces in str(s): no Markup escaping.
      auto o = old_value->is_string() ? *old_value : Value();
      auto n = new_value->is_string() ? *new_value : Value();
      if (!o.is_string()) {
        StrBuilder b;
        if (auto r = AppendStr(*old_value, b, arena_); !r) {
          return std::unexpected(r.error());
        }
        auto t = Take(b);
        if (!t) {
          return t;
        }
        o = *t;
      }
      if (!n.is_string()) {
        StrBuilder b;
        if (auto r = AppendStr(*new_value, b, arena_); !r) {
          return std::unexpected(r.error());
        }
        auto t = Take(b);
        if (!t) {
          return t;
        }
        n = *t;
      }
      return Replace(*unmarked, o, n,
                     count != nullptr && !count->is_none() ? count->integer() : -1);
    }
    if (name == "length" || name == "count") {
      switch (v.kind()) {
        case Kind::kUndefined:
          return Value::Int(0);
        case Kind::kString:
          if (!arena_.Work(v.str().text.size())) {
            return Bound();
          }
          return Value::Int(static_cast<std::int64_t>(CodePoints(v.str().text)));
        case Kind::kList:
          return Value::Int(static_cast<std::int64_t>(v.list().items.size()));
        case Kind::kDict:
          return Value::Int(static_cast<std::int64_t>(v.dict().members.size()));
        default:
          return Fail(Code::kRuntime, "the length of a value that has none");
      }
    }
    if (name == "default" || name == "d") {
      const Value* fallback = args.Get(0, "default_value");
      const Value* boolean = args.Get(1, "boolean");
      const bool use = v.is_undefined() || (boolean != nullptr && Truthy(*boolean) && !Truthy(v));
      if (!use) {
        return v;
      }
      if (fallback == nullptr) {
        return Str("", true);
      }
      return *fallback;
    }
    if (name == "first" || name == "last") {
      if (v.is_dict() && name == "last") {
        if (v.dict().members.empty()) {
          return Value();
        }
        return Str(v.dict().members.back().first, v.dict().keys_trusted);
      }
      if (v.is_string()) {  // one code point, without taking the string apart
        const jinja::Str& s = v.str();
        if (s.text.empty()) {
          return Value();
        }
        if (name == "first") {
          std::size_t length = 0;
          DecodeAt(s.text, 0, length);
          return Sub(s, 0, length);
        }
        std::size_t at = s.text.size() - 1;
        while (at > 0 && (static_cast<unsigned char>(s.text[at]) & 0xC0U) == 0x80U) {
          --at;
        }
        return Sub(s, at, s.text.size() - at);
      }
      auto items = Iterate(v);
      if (!items) {
        return std::unexpected(items.error());
      }
      if (items->empty()) {
        return Value();
      }
      return name == "first" ? items->front() : items->back();
    }
    if (name == "list") {
      auto items = Iterate(v);
      if (!items) {
        return std::unexpected(items.error());
      }
      return MakeList(std::move(*items));
    }
    if (name == "items") {
      if (v.is_undefined()) {
        return MakeList({});
      }
      if (!v.is_dict()) {
        return Fail(Code::kRuntime, "items of a value that is not a mapping");
      }
      return Items(v);
    }
    if (name == "dictsort") {
      if (!v.is_dict()) {
        return Fail(Code::kRuntime, "dictsort of a value that is not a mapping");
      }
      const Value* case_sensitive = args.Get(0, "case_sensitive");
      const Value* by = args.Get(1, "by");
      const Value* reverse = args.Get(2, "reverse");
      const bool by_value = by != nullptr && by->is_string() && by->str().text == "value";
      if (by != nullptr && (!by->is_string() || (by->str().text != "key" && !by_value))) {
        return Fail(Code::kRuntime, "dictsort by something other than key or value");
      }
      auto pairs = Items(v);
      if (!pairs) {
        return pairs;
      }
      Value position = Value::Int(by_value ? 1 : 0);
      return Sorted(pairs->list().items, reverse != nullptr && Truthy(*reverse),
                    case_sensitive != nullptr && Truthy(*case_sensitive), &position);
    }
    if (name == "join") {
      auto items = Iterate(v);
      if (!items) {
        return std::unexpected(items.error());
      }
      const Value* separator = args.Get(0, "d");
      auto empty = Str("", true);
      if (!empty) {
        return empty;
      }
      return Join(*items, separator != nullptr ? separator : &*empty, args.Get(1, "attribute"));
    }
    if (name == "map") {
      auto items = Iterate(v);
      if (!items) {
        return std::unexpected(items.error());
      }
      std::vector<Value> out;
      if (args.positional.empty()) {
        const Value* attribute = args.Get(99, "attribute");
        if (attribute == nullptr) {
          return Fail(Code::kRuntime, "map without a filter or attribute");
        }
        const Value* fallback = args.Get(99, "default");
        for (const Value& item : *items) {
          auto a = Attribute(item, *attribute);
          if (!a) {
            return a;
          }
          if (a->is_undefined() && fallback != nullptr) {
            out.push_back(*fallback);
          } else {
            out.push_back(std::move(*a));
          }
        }
      } else {
        if (!args.positional[0].is_string() || !KnownFilter(args.positional[0].str().text)) {
          return Fail(Code::kUnsupported, "map with a filter the subset does not implement");
        }
        Args rest;
        rest.positional.assign(args.positional.begin() + 1, args.positional.end());
        rest.keywords = args.keywords;
        const std::string filter = args.positional[0].str().text;
        for (const Value& item : *items) {
          auto r = Filter(filter, item, rest);
          if (!r) {
            return r;
          }
          out.push_back(std::move(*r));
        }
      }
      return MakeList(std::move(out));
    }
    if (name == "select" || name == "reject") {
      return FilterByTest(v, args, name == "select", false);
    }
    if (name == "selectattr" || name == "rejectattr") {
      return FilterByTest(v, args, name == "selectattr", true);
    }
    if (name == "sort") {
      auto items = Iterate(v);
      if (!items) {
        return std::unexpected(items.error());
      }
      const Value* reverse = args.Get(0, "reverse");
      const Value* case_sensitive = args.Get(1, "case_sensitive");
      return Sorted(std::move(*items), reverse != nullptr && Truthy(*reverse),
                    case_sensitive != nullptr && Truthy(*case_sensitive), args.Get(2, "attribute"));
    }
    if (name == "unique") {
      auto items = Iterate(v);
      if (!items) {
        return std::unexpected(items.error());
      }
      const Value* case_sensitive = args.Get(0, "case_sensitive");
      const Value* attribute = args.Get(1, "attribute");
      std::vector<Value> seen;
      std::vector<Value> out;
      for (const Value& item : *items) {
        Value key = item;
        if (attribute != nullptr) {
          auto a = Attribute(item, *attribute);
          if (!a) {
            return a;
          }
          key = std::move(*a);
        }
        auto k = SortKey(key, case_sensitive != nullptr && Truthy(*case_sensitive));
        if (!k) {
          return k;
        }
        if (std::ranges::none_of(seen, [&](const Value& s) { return Eq(s, *k); })) {
          seen.push_back(std::move(*k));
          out.push_back(item);
        }
        if (!arena_.Work(seen.size() * 8)) {
          return Bound();
        }
      }
      return MakeList(std::move(out));
    }
    if (name == "min" || name == "max") {
      auto items = Iterate(v);
      if (!items) {
        return std::unexpected(items.error());
      }
      if (items->empty()) {
        return Value();
      }
      const Value* case_sensitive = args.Get(0, "case_sensitive");
      auto sorted =
          Sorted(std::move(*items), name == "max",
                 case_sensitive != nullptr && Truthy(*case_sensitive), args.Get(1, "attribute"));
      if (!sorted) {
        return sorted;
      }
      return sorted->list().items.front();
    }
    if (name == "sum") {
      auto items = Iterate(v);
      if (!items) {
        return std::unexpected(items.error());
      }
      const Value* attribute = args.Get(0, "attribute");
      const Value* start = args.Get(1, "start");
      if (start != nullptr && start->is_string()) {
        return Fail(Code::kRuntime, "sum() can't sum strings");  // as Python's sum raises
      }
      Value total = start != nullptr ? *start : Value::Int(0);
      for (const Value& item : *items) {
        Value x = item;
        if (attribute != nullptr) {
          auto a = Attribute(item, *attribute);
          if (!a) {
            return a;
          }
          x = std::move(*a);
        }
        auto t = Add(total, x);
        if (!t) {
          return t;
        }
        total = std::move(*t);
      }
      return total;
    }
    if (name == "reverse") {
      if (v.is_string()) {
        const jinja::Str& s = v.str();
        if (!arena_.Work(s.text.size() * 2)) {
          return Bound();
        }
        if (s.trusted.empty() || s.all_trusted()) {
          // One provenance throughout: the code points copied in reverse.
          StrBuilder b;
          std::string& text = b.raw().text;
          text.reserve(s.text.size());
          for (std::size_t end = s.text.size(); end > 0;) {
            std::size_t at = end - 1;
            while (at > 0 && (static_cast<unsigned char>(s.text[at]) & 0xC0U) == 0x80U) {
              --at;
            }
            text.append(s.text, at, end - at);
            end = at;
          }
          if (!s.text.empty() && s.all_trusted()) {
            b.raw().trusted = {{0, text.size()}};
          }
          return Take(b);
        }
        if (!arena_.Work(s.text.size() * 16)) {  // a code point at a time
          return Bound();
        }
        if (CodePoints(s.text) > arena_.limits().max_live_bytes / 16) {
          return Fail(Code::kLimit, "a string reversed longer than its bound");
        }
        std::vector<std::pair<std::size_t, std::size_t>> chars;
        for (std::size_t at = 0; at < s.text.size();) {
          std::size_t length = 0;
          DecodeAt(s.text, at, length);
          chars.emplace_back(at, length);
          at += length;
        }
        StrBuilder b;
        for (const auto& [at, length] : std::views::reverse(chars)) {
          b.Append(s, at, length);
        }
        return Take(b);
      }
      auto items = Iterate(v);
      if (!items) {
        return std::unexpected(items.error());
      }
      std::ranges::reverse(*items);
      return MakeList(std::move(*items));
    }
    if (name == "abs") {
      if (!v.is_number() || v.kind() == Kind::kBigInt) {
        return Fail(Code::kRuntime, "abs of something other than a number");
      }
      if (v.kind() == Kind::kFloat) {
        return Value::Float(std::fabs(v.number()), v.trusted());
      }
      if (v.integer() == std::numeric_limits<std::int64_t>::min()) {
        return Fail(Code::kUnsupported, "an integer beyond 64 bits");
      }
      return Value::Int(v.integer() < 0 ? -v.integer() : v.integer(), v.trusted());
    }
    if (name == "int") {
      const Value* fallback = args.Get(0, "default");
      Value def = fallback != nullptr ? *fallback : Value::Int(0);
      if (args.Get(1, "base") != nullptr) {
        return Fail(Code::kUnsupported, "int with a base");
      }
      if (v.is_string() && !arena_.Work(v.str().text.size() * 4)) {  // copied and parsed twice
        return Bound();
      }
      if (v.is_integral()) {
        return Value::Int(v.integer(), v.trusted());
      }
      if (v.kind() == Kind::kFloat) {
        if (!std::isfinite(v.number()) || std::fabs(v.number()) >= 9.2e18) {
          return def;
        }
        return Value::Int(static_cast<std::int64_t>(std::trunc(v.number())), v.trusted());
      }
      if (v.is_string()) {
        const std::string_view text = PythonNumberText(v.str().text);
        std::int64_t i = 0;
        if (ParseInt(text, i)) {
          return Value::Int(i, v.str().all_trusted());
        }
        double d = 0;
        if (ParseFloat(text, d) && std::isfinite(d) && std::fabs(d) < 9.2e18) {
          return Value::Int(static_cast<std::int64_t>(std::trunc(d)), v.str().all_trusted());
        }
      }
      return def;
    }
    if (name == "float") {
      const Value* fallback = args.Get(0, "default");
      Value def = fallback != nullptr ? *fallback : Value::Float(0.0);
      if (v.is_string() && !arena_.Work(v.str().text.size() * 4)) {  // copied and parsed
        return Bound();
      }
      if (v.is_number() && v.kind() != Kind::kBigInt) {
        return Value::Float(v.number(), v.trusted());
      }
      if (v.is_string()) {
        double d = 0;
        if (ParseFloat(PythonNumberText(v.str().text), d)) {
          return Value::Float(d, v.str().all_trusted());
        }
      }
      return def;
    }
    if (name == "round") {
      if (!v.is_number() || v.kind() == Kind::kBigInt) {
        return Fail(Code::kRuntime, "round of something other than a number");
      }
      std::int64_t precision = 0;
      if (const Value* p = args.Get(0, "precision")) {
        if (!p->is_integral()) {
          return Fail(Code::kRuntime, "a round precision that is not an integer");
        }
        precision = p->integer();
      }
      const Value* method = args.Get(1, "method");
      std::string_view how = "common";
      if (method != nullptr && method->is_string()) {
        how = method->str().text;
      }
      if (how != "common" && how != "ceil" && how != "floor") {
        return Fail(Code::kRuntime, "a round method other than common, ceil or floor");
      }
      if (how == "common") {
        return Round(v, precision);  // Python's round(value, precision)
      }
      // Jinja2's math.ceil or math.floor of value * 10**precision, divided back.
      const double scale =
          std::pow(10.0, static_cast<double>(std::clamp<std::int64_t>(precision, -400, 400)));
      const double x = v.number() * scale;
      return Value::Float((how == "ceil" ? std::ceil(x) : std::floor(x)) / scale, v.trusted());
    }
    if (name == "tojson") {
      return ToJson(v, args);
    }
    if (name == "from_json") {
      if (!v.is_string()) {
        return Fail(Code::kRuntime, "from_json of something other than a string");
      }
      // Parsed at a cost a byte, into no more values than the values could
      // hold (the document lives while they are built).
      if (!arena_.Work(v.str().text.size() * 2)) {
        return Bound();
      }
      const Limits& l = arena_.limits();
      const base::json::Limits json_limits{.max_bytes = l.max_string_bytes,
                                           .max_depth = l.max_value_depth,
                                           .max_values = l.max_live_bytes / sizeof(Value),
                                           .max_string_bytes = l.max_live_bytes};
      auto doc = base::json::Parse(v.str().text, json_limits);
      if (!doc) {
        return Fail(Code::kRuntime, "from_json of text that is not JSON");
      }
      return FromJson(doc->root(), 1);
    }
    if (name == "indent") {
      return Indent(v, args);
    }
    if (name == "format") {
      auto s = need_string();
      if (!s) {
        return s;
      }
      return Format(*s, args);
    }
    return Fail(Code::kUnsupported, "a filter the subset does not implement");
  }

  // Python's round(value, precision): an integer stays one (rounded half to
  // even at a negative precision); a float is rounded correctly, half to
  // even, from its exact value.
  static std::expected<Value, Error> Round(const Value& v, std::int64_t precision) {
    if (v.is_integral()) {
      const std::int64_t x = v.integer();
      if (precision >= 0) {
        return Value::Int(x, v.trusted());
      }
      if (precision < -18) {
        // 10**19 and up: every int64 is within half of zero, but past 5e18
        // (precision -19) rounds to 1e19, beyond 64 bits.
        if (precision == -19 && (x > 5'000'000'000'000'000'000 || x < -5'000'000'000'000'000'000)) {
          return Fail(Code::kUnsupported, "an integer beyond 64 bits");
        }
        return Value::Int(0, v.trusted());
      }
      std::int64_t m = 1;
      for (std::int64_t i = 0; i < -precision; ++i) {
        m *= 10;
      }
      std::int64_t q = x / m;
      const std::int64_t r = x % m;  // the sign of x
      // Compare |r| with m / 2 without overflow.
      const std::uint64_t ar =
          r < 0 ? std::uint64_t{0} - static_cast<std::uint64_t>(r) : static_cast<std::uint64_t>(r);
      const std::uint64_t half = static_cast<std::uint64_t>(m) / 2;  // m is even
      if (ar > half || (ar == half && q % 2 != 0)) {
        q += x < 0 ? -1 : 1;
      }
      std::int64_t out = 0;
      if (__builtin_mul_overflow(q, m, &out)) {
        return Fail(Code::kUnsupported, "an integer beyond 64 bits");
      }
      return Value::Int(out, v.trusted());
    }
    const double x = v.number();
    if (!std::isfinite(x) || precision > 400) {
      return Value::Float(x, v.trusted());
    }
    if (precision < 0) {
      const double scale =
          std::pow(10.0, static_cast<double>(std::max<std::int64_t>(precision, -400)));
      return Value::Float(std::nearbyint(x * scale) / scale, v.trusted());
    }
    // The exact decimal expansion rounded at `precision` digits, half to
    // even, and read back: what Python's round does.
    std::array<char, 1024> buffer{};  // 309 integer digits, a point, 400 decimals
    const auto [end, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), x,
                                         std::chars_format::fixed, static_cast<int>(precision));
    if (ec != std::errc()) {
      return Fail(Code::kUnsupported, "a float rounded beyond what the subset formats");
    }
    double r = 0;
    const auto parsed = std::from_chars(buffer.data(), end, r);
    if (parsed.ec != std::errc()) {
      return Fail(Code::kUnsupported, "a float rounded beyond what the subset formats");
    }
    return Value::Float(r, v.trusted());
  }

  // Python's int()/float() accept surrounding whitespace and underscores
  // between digits; the text without the whitespace.
  static std::string_view PythonNumberText(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || (s.front() >= '\t' && s.front() <= '\r'))) {
      s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || (s.back() >= '\t' && s.back() <= '\r'))) {
      s.remove_suffix(1);
    }
    return s;
  }
  static bool ParseInt(std::string_view s, std::int64_t& out) {
    std::string digits;
    std::size_t i = 0;
    if (!s.empty() && (s[0] == '+' || s[0] == '-')) {
      digits.push_back(s[0]);
      i = 1;
    }
    bool any = false;
    for (; i < s.size(); ++i) {
      if (s[i] >= '0' && s[i] <= '9') {
        digits.push_back(s[i]);
        any = true;
      } else if (s[i] == '_' && any && i + 1 < s.size() && s[i + 1] >= '0' && s[i + 1] <= '9') {
        continue;
      } else {
        return false;
      }
    }
    if (!any) {
      return false;
    }
    const char* begin = digits.data() + (digits[0] == '+' ? 1 : 0);
    const auto [end, ec] = std::from_chars(begin, digits.data() + digits.size(), out);
    return ec == std::errc() && end == digits.data() + digits.size();
  }
  static bool ParseFloat(std::string_view s, double& out) {
    if (s.empty()) {
      return false;
    }
    // As in Python's float(), an underscore only between two digits.
    const auto digit = [](char c) { return c >= '0' && c <= '9'; };
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (s[i] == '_' && (i == 0 || i + 1 == s.size() || !digit(s[i - 1]) || !digit(s[i + 1]))) {
        return false;
      }
    }
    std::string text(s);
    std::erase(text, '_');
    std::string lower;
    for (const char c : text) {
      lower.push_back(AsciiLower(c));
    }
    std::string_view body = lower;
    const bool negative = body.starts_with('-');
    if (body.starts_with('-') || body.starts_with('+')) {
      body.remove_prefix(1);
    }
    if (body == "inf" || body == "infinity") {
      out = negative ? -HUGE_VAL : HUGE_VAL;
      return true;
    }
    if (body == "nan") {
      out = std::nan("");
      return true;
    }
    const char* begin = text.data() + (text[0] == '+' ? 1 : 0);
    const auto [end, ec] = std::from_chars(begin, text.data() + text.size(), out);
    return (ec == std::errc() || ec == std::errc::result_out_of_range) &&
           end == text.data() + text.size();
  }

  // ------------------------------------------------------------------ tests

  std::expected<bool, Error> ApplyTest(std::string_view name, const Value& v, const Args& args) {
    if (!arena_.Step() || !ChargeArgs(args)) {
      return Bound();
    }
    const Value* arg = args.Get(0, "other");
    const auto need_arg = [&]() -> std::expected<const Value*, Error> {
      if (arg == nullptr) {
        return Fail(Code::kRuntime, "a test without its argument");
      }
      return arg;
    };
    if (name == "defined") {
      return !v.is_undefined();
    }
    if (name == "undefined") {
      return v.is_undefined();
    }
    if (name == "none") {
      return v.is_none();
    }
    if (name == "boolean") {
      return v.kind() == Kind::kBool;
    }
    if (name == "true") {
      return v.kind() == Kind::kBool && v.boolean();
    }
    if (name == "false") {
      return v.kind() == Kind::kBool && !v.boolean();
    }
    if (name == "integer") {
      return v.kind() == Kind::kInt || v.kind() == Kind::kBigInt;
    }
    if (name == "float") {
      return v.kind() == Kind::kFloat;
    }
    if (name == "number") {
      return v.is_number();
    }
    if (name == "string") {
      return v.is_string();
    }
    if (name == "mapping") {
      return v.is_dict();
    }
    if (name == "iterable") {
      return v.is_undefined() || v.is_string() || v.is_list() || v.is_dict();
    }
    if (name == "sequence") {
      return v.is_undefined() || v.is_string() || v.is_list() || v.is_dict();
    }
    if (name == "callable") {
      return v.kind() == Kind::kCallable;
    }
    if (name == "escaped") {
      return v.is_string() && v.str().markup;
    }
    if (name == "lower" || name == "upper") {
      // Jinja2's str(value).islower() (isupper()): any value, as its text.
      StrBuilder b;
      if (!v.is_string()) {
        if (auto r = AppendStr(v, b, arena_); !r) {
          return std::unexpected(r.error());
        }
      }
      const std::string_view text = v.is_string() ? std::string_view(v.str().text) : b.text();
      if (!arena_.Work(CaseWork(text))) {
        return Bound();
      }
      return name == "lower" ? PythonIsLower(text) : PythonIsUpper(text);
    }
    if (name == "divisibleby" || name == "even" || name == "odd") {
      if (!v.is_integral()) {
        return Fail(Code::kRuntime, "a divisibility test of something other than an integer");
      }
      std::int64_t by = 2;
      if (name == "divisibleby") {
        auto a = need_arg();
        if (!a) {
          return std::unexpected(a.error());
        }
        if (!(*a)->is_integral() || (*a)->integer() == 0) {
          return Fail(Code::kRuntime, "divisibleby needs a nonzero integer");
        }
        by = (*a)->integer();
      }
      const bool divisible = by == -1 || v.integer() % by == 0;
      return name == "odd" ? !divisible : divisible;
    }
    if (name == "in") {
      auto a = need_arg();
      if (!a) {
        return std::unexpected(a.error());
      }
      return Contains(**a, v);
    }
    if (name == "sameas") {
      auto a = need_arg();
      if (!a) {
        return std::unexpected(a.error());
      }
      const Value& o = **a;
      if (v.kind() != o.kind()) {
        return false;
      }
      if (v.object() || o.object()) {
        return v.object() == o.object();
      }
      return CheckedEq(v, o);
    }
    Op op = Op::kEq;
    if (name == "eq" || name == "equalto" || name == "==") {
      op = Op::kEq;
    } else if (name == "ne" || name == "!=") {
      op = Op::kNe;
    } else if (name == "lt" || name == "lessthan" || name == "<") {
      op = Op::kLt;
    } else if (name == "le" || name == "<=") {
      op = Op::kLe;
    } else if (name == "gt" || name == "greaterthan" || name == ">") {
      op = Op::kGt;
    } else if (name == "ge" || name == ">=") {
      op = Op::kGe;
    } else {
      return Fail(Code::kUnsupported, "a test the subset does not implement");
    }
    auto a = need_arg();
    if (!a) {
      return std::unexpected(a.error());
    }
    return Compare(op, v, **a);
  }

  const Program& program_;
  Arena arena_;  // the program's limits, this rendering's budget and cancellation
  const std::optional<CivilTime>& now_;
  Usage* usage_;  // where to add what this rendering used, if anywhere
  // Objects that can close a reference cycle (a scope holding a macro that
  // holds the scope; a namespace holding itself): emptied when the
  // rendering ends, so that everything it built is freed.
  std::vector<std::weak_ptr<Scope>> macro_scopes_;
  std::vector<std::weak_ptr<Namespace>> namespaces_;
  std::size_t depth_ = 0;
  std::size_t macro_depth_ = 0;
};

std::expected<Value, Error> Renderer::FromInput(const Input& in, std::size_t depth) {
  // Input's members are private to it; Renderer is its friend.
  if (depth > arena_.limits().max_value_depth) {
    return Fail(Code::kLimit, "lists and mappings nest deeper than their bound");
  }
  switch (in.kind_) {
    case Input::Kind::kUndefined:
      return Value();
    case Input::Kind::kNone:
      return Value::None(false);
    case Input::Kind::kBool:
      return Value::Bool(in.b_, false);
    case Input::Kind::kInt:
      return Value::Int(in.i_, false);
    case Input::Kind::kString:
      return Str(in.text_, in.trusted_);
    case Input::Kind::kJson:
      if (!in.json_) {
        return Value();
      }
      return FromJson(*in.json_, depth);
    case Input::Kind::kList: {
      std::vector<Value> items;
      for (const Input& item : in.items_) {
        auto v = FromInput(item, depth + 1);
        if (!v) {
          return v;
        }
        items.push_back(std::move(*v));
      }
      return MakeList(std::move(items));
    }
    case Input::Kind::kMap: {
      std::vector<std::pair<std::string, Value>> members;
      for (const auto& [k, item] : in.members_) {
        auto v = FromInput(item, depth + 1);
        if (!v) {
          return v;
        }
        members.emplace_back(k, std::move(*v));
      }
      return MakeDict(std::move(members), false);
    }
  }
  return Value();
}

bool KnownFilter(std::string_view name) { return In(kFilters, name); }
bool KnownTest(std::string_view name) { return In(kTests, name); }

Input Input::Undefined() { return {}; }
Input Input::None() {
  Input i;
  i.kind_ = Kind::kNone;
  return i;
}
Input Input::Bool(bool b) {
  Input i;
  i.kind_ = Kind::kBool;
  i.b_ = b;
  return i;
}
Input Input::Int(std::int64_t value) {
  Input i;
  i.kind_ = Kind::kInt;
  i.i_ = value;
  return i;
}
Input Input::String(std::string_view text, bool trusted) {
  Input i;
  i.kind_ = Kind::kString;
  i.text_ = text;
  i.trusted_ = trusted;
  return i;
}
Input Input::Json(base::json::Value value) {
  Input i;
  i.kind_ = Kind::kJson;
  i.json_ = value;
  return i;
}
Input Input::List(std::vector<Input> items) {
  Input i;
  i.kind_ = Kind::kList;
  i.items_ = std::move(items);
  return i;
}
Input Input::Map(std::vector<std::pair<std::string, Input>> members) {
  Input i;
  i.kind_ = Kind::kMap;
  i.members_ = std::move(members);
  return i;
}
Input::Input(const Input&) = default;
Input::Input(Input&&) noexcept = default;
Input& Input::operator=(const Input&) = default;
Input& Input::operator=(Input&&) noexcept = default;
Input::~Input() = default;

std::expected<Template, Error> Template::Parse(std::string_view source, const Limits& limits) {
  auto program = std::make_unique<Program>();
  program->limits = limits;
  auto body = ParseProgram(source, program->limits);
  if (!body) {
    return std::unexpected(body.error());
  }
  program->body = std::move(*body);
  return Template(std::move(program));
}

Template::Template(std::unique_ptr<Program> program) : program_(std::move(program)) {}
Template::Template(Template&&) noexcept = default;
Template& Template::operator=(Template&&) noexcept = default;
Template::~Template() = default;

const Limits& Template::limits() const { return program_->limits; }

std::expected<Rendered, Error> Template::Render(
    const std::vector<std::pair<std::string, Input>>& variables,
    const std::optional<CivilTime>& now, Budget budget) const {
  Renderer renderer(*program_, now, budget);
  return renderer.Run(variables);
}

}  // namespace jitllm::chat::jinja
