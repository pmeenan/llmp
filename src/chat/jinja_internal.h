// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The Jinja-subset interpreter's values, syntax tree and accounting
// (jinja.h). Internal to the chat module.
//
// Values follow Python's semantics as Jinja2 exposes them: strings are
// immutable and index by code point, lists and mappings are immutable (the
// immutable sandbox) and shared, and a namespace is the one mutable object.
// Every string carries its provenance: the byte ranges that came from the
// template. Every heap object is charged to the rendering's Arena while it
// lives.

#ifndef JITLLM_CHAT_JINJA_INTERNAL_H_
#define JITLLM_CHAT_JINJA_INTERNAL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chat/jinja.h"

namespace jitllm::chat::jinja {

// ---------------------------------------------------------------- accounting

// What one rendering consumes: steps and bytes built or scanned, counted,
// capped only by a Budget (probes) and checked against the caller's
// cancellation every kCancelSteps steps and kCancelWorkBytes bytes (D-102:
// no time bound, but a long rendering stops when its request ends); and
// bytes held at once, bounded by Limits. The first bound reached, or the
// cancellation, records an error; every later Charge fails too.
class Arena {
 public:
  Arena(const Limits& limits, const Budget& budget)
      : limits_(Lowered(limits, budget)),
        max_steps_(budget.max_steps != 0 ? budget.max_steps : kNone),
        max_work_(budget.max_work_bytes != 0 ? budget.max_work_bytes : kNone),
        cancelled_(budget.cancelled) {}
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&&) = delete;
  Arena& operator=(Arena&&) = delete;
  ~Arena() = default;

  const Limits& limits() const { return limits_; }
  bool Step() {
    if (++steps_ > max_steps_) {
      return Fail("the template took more steps than its budget");
    }
    if ((steps_ & (kCancelSteps - 1)) == 0 && Cancelled()) {
      return false;
    }
    return ok();
  }
  // Bytes about to be built or scanned.
  bool Work(std::size_t bytes) {
    work_ += bytes;
    if (work_ > max_work_) {
      return Fail("the template built or scanned more bytes than its budget");
    }
    if (work_ >= next_check_) {
      next_check_ = work_ + kCancelWorkBytes;
      if (Cancelled()) {
        return false;
      }
    }
    return ok();
  }
  // Whether the caller said to stop; if so the rendering fails with
  // Code::kCancelled.
  bool Cancelled() {
    if (failed_ || cancelled_ == nullptr || !*cancelled_ || !(*cancelled_)()) {
      return false;
    }
    cancel_ = true;
    (void)Fail("the rendering was cancelled: its request ended");
    return true;
  }
  bool cancelled() const { return cancel_; }
  bool Hold(std::size_t bytes) {
    live_ += bytes;
    if (live_ > limits_.max_live_bytes) {
      return Fail("the template's values held more bytes than their bound");
    }
    return ok();
  }
  void Release(std::size_t bytes) { live_ -= bytes; }
  bool Fail(std::string_view reason) {
    if (!failed_) {
      failed_ = true;
      reason_ = reason;
    }
    return false;
  }
  bool ok() const { return !failed_; }
  std::string_view reason() const { return reason_; }
  // Bytes that may still be built or scanned (a comparison stops there).
  std::uint64_t remaining_work() const { return work_ < max_work_ ? max_work_ - work_ : 0; }
  std::uint64_t steps() const { return steps_; }
  std::uint64_t work() const { return work_; }

 private:
  static constexpr std::uint64_t kNone = std::numeric_limits<std::uint64_t>::max();

  // The program's limits, with this rendering's lower memory bounds.
  static Limits Lowered(Limits limits, const Budget& budget) {
    if (budget.max_live_bytes != 0) {
      limits.max_live_bytes = std::min(limits.max_live_bytes, budget.max_live_bytes);
      limits.max_string_bytes = std::min(limits.max_string_bytes, budget.max_live_bytes);
    }
    if (budget.max_output_bytes != 0) {
      limits.max_output_bytes = std::min(limits.max_output_bytes, budget.max_output_bytes);
    }
    return limits;
  }

  const Limits limits_;
  std::uint64_t max_steps_;
  std::uint64_t max_work_;
  const std::function<bool()>* cancelled_;
  std::uint64_t steps_ = 0;
  std::uint64_t work_ = 0;
  std::uint64_t next_check_ = kCancelWorkBytes;
  std::size_t live_ = 0;
  bool failed_ = false;
  bool cancel_ = false;
  std::string_view reason_;
};

// Bytes a rendering holds outside its values (a loop's own copy of what it
// iterates), charged as live while the holder exists.
class Held {
 public:
  explicit Held(Arena& arena) : arena_(arena) {}
  Held(const Held&) = delete;
  Held& operator=(const Held&) = delete;
  Held(Held&&) = delete;
  Held& operator=(Held&&) = delete;
  ~Held() { arena_.Release(bytes_); }
  // False when the live bound is reached.
  bool Add(std::size_t bytes) {
    bytes_ += bytes;
    return arena_.Hold(bytes);
  }
  // Releases what it holds now (the bytes passing to a value that charges
  // them itself).
  void Clear() {
    arena_.Release(bytes_);
    bytes_ = 0;
  }

 private:
  Arena& arena_;
  std::size_t bytes_ = 0;
};

// Work charged for each entry a linear search of names passes (a scope's
// variables, a namespace's or small mapping's members, a call's keyword
// arguments): about what reading this many bytes costs.
inline constexpr std::uint64_t kEntryWork = 4;

// A linear search of (name, value) pairs for `key`: `work` grows by
// kEntryWork an entry passed and the bytes of every name of the key's
// length compared, so a search meets the work bound in about the time it
// takes however many names, or how long a key, it compares.
template <typename Members>
auto FindMember(Members& members, std::string_view key, std::uint64_t& work)
    -> decltype(&members.front().second) {
  for (auto& [k, v] : members) {
    work += kEntryWork;
    if (k.size() == key.size()) {
      work += key.size();
      if (k == key) {
        return &v;
      }
    }
  }
  return nullptr;
}

// Base of every heap value: what it holds is charged to an arena (none for
// the template's own constants) until it is destroyed.
class Object {
 public:
  Object() = default;
  Object(const Object&) = delete;
  Object& operator=(const Object&) = delete;
  Object(Object&&) = delete;
  Object& operator=(Object&&) = delete;
  virtual ~Object() {
    if (arena_ != nullptr) {
      arena_->Release(charged_);
    }
  }
  // Charges `bytes` more to the arena; false when a bound is reached.
  bool Charge(Arena* arena, std::size_t bytes) {
    if (arena == nullptr) {
      return true;
    }
    arena_ = arena;
    charged_ += bytes;
    return arena->Hold(bytes);
  }

 private:
  Arena* arena_ = nullptr;
  std::size_t charged_ = 0;
};

// ------------------------------------------------------------------- values

class Value;
struct Str;
struct List;
struct Dict;
struct Namespace;
struct Callable;

// A trusted byte range of a string.
struct Range {
  std::size_t offset = 0;
  std::size_t length = 0;
};

// A string and its provenance. `markup` is markupsafe's Markup (the `safe`
// filter's result), which escapes what is added to it with `+`.
struct Str final : Object {
  std::string text;
  std::vector<Range> trusted;  // sorted, disjoint, not touching
  bool markup = false;

  bool all_trusted() const {
    return text.empty() || (trusted.size() == 1 && trusted[0].length == text.size());
  }
};

// The containers below (and the evaluator's scopes) hand what they hold to
// ReleaseDeferred when they die, rather than destroying it in place: a chain
// of them (a namespace holding a namespace, a method bound to a mapping
// that holds such a method, each built by one loop iteration) has no depth
// bound, and destroying it recursively would exhaust the stack.
struct List final : Object {
  List() = default;
  List(const List&) = delete;
  List& operator=(const List&) = delete;
  List(List&&) = delete;
  List& operator=(List&&) = delete;
  ~List() override;

  std::vector<Value> items;
  bool tuple = false;
  std::size_t depth = 1;  // nesting, for Limits::max_value_depth
};

struct Dict final : Object {
  Dict() = default;
  Dict(const Dict&) = delete;
  Dict& operator=(const Dict&) = delete;
  Dict(Dict&&) = delete;
  Dict& operator=(Dict&&) = delete;
  ~Dict() override;

  std::vector<std::pair<std::string, Value>> members;  // insertion order; fixed once built
  bool keys_trusted = false;
  std::size_t depth = 1;
  // Key to member, for mappings too large to search (BuildIndex).
  std::unordered_map<std::string_view, std::size_t> index;

  void BuildIndex();
  // `work` grows by what the lookup costs (FindMember; the key's bytes
  // hashed for an indexed mapping).
  const Value* Find(std::string_view key, std::uint64_t& work) const;
};

struct Namespace final : Object {
  Namespace() = default;
  Namespace(const Namespace&) = delete;
  Namespace& operator=(const Namespace&) = delete;
  Namespace(Namespace&&) = delete;
  Namespace& operator=(Namespace&&) = delete;
  ~Namespace() override;

  std::vector<std::pair<std::string, Value>> members;
  Value* Find(std::string_view key, std::uint64_t& work);
};

struct Loop;

enum class Kind : std::uint8_t {
  kUndefined,
  kNone,
  kBool,
  kInt,
  kBigInt,  // a JSON integer beyond int64: its decimal text, printed as is
  kFloat,
  kString,
  kList,
  kDict,
  kNamespace,
  kCallable,
  kLoop,  // a for loop's `loop`
};

class Value {
 public:
  Value() = default;
  static Value None(bool trusted = true) {
    Value v;
    v.kind_ = Kind::kNone;
    v.trusted_ = trusted;
    return v;
  }
  static Value Bool(bool b, bool trusted = true) {
    Value v;
    v.kind_ = Kind::kBool;
    v.int_ = b ? 1 : 0;
    v.trusted_ = trusted;
    return v;
  }
  static Value Int(std::int64_t i, bool trusted = true) {
    Value v;
    v.kind_ = Kind::kInt;
    v.int_ = i;
    v.trusted_ = trusted;
    return v;
  }
  static Value Float(double f, bool trusted = true) {
    Value v;
    v.kind_ = Kind::kFloat;
    v.float_ = f;
    v.trusted_ = trusted;
    return v;
  }
  static Value BigInt(std::shared_ptr<const Str> digits) {
    Value v;
    v.kind_ = Kind::kBigInt;
    v.obj_ = std::move(digits);
    return v;
  }
  static Value String(std::shared_ptr<const Str> s) {
    Value v;
    v.kind_ = Kind::kString;
    v.obj_ = std::move(s);
    return v;
  }
  static Value MakeList(std::shared_ptr<const List> l) {
    Value v;
    v.kind_ = Kind::kList;
    v.obj_ = std::move(l);
    return v;
  }
  static Value MakeDict(std::shared_ptr<const Dict> d) {
    Value v;
    v.kind_ = Kind::kDict;
    v.obj_ = std::move(d);
    return v;
  }
  static Value MakeNamespace(std::shared_ptr<Namespace> n) {
    Value v;
    v.kind_ = Kind::kNamespace;
    v.obj_ = n;
    v.ns_ = std::move(n);
    return v;
  }
  static Value MakeCallable(std::shared_ptr<const Callable> c);  // after Callable
  static Value MakeLoop(std::shared_ptr<const Loop> l);          // after Loop

  Kind kind() const { return kind_; }
  bool is_undefined() const { return kind_ == Kind::kUndefined; }
  bool is_none() const { return kind_ == Kind::kNone; }
  bool is_string() const { return kind_ == Kind::kString; }
  bool is_list() const { return kind_ == Kind::kList; }
  bool is_dict() const { return kind_ == Kind::kDict; }
  // int, bool and float, as Python's numbers.Number (and big integers).
  bool is_number() const {
    return kind_ == Kind::kInt || kind_ == Kind::kBool || kind_ == Kind::kFloat ||
           kind_ == Kind::kBigInt;
  }
  bool is_integral() const { return kind_ == Kind::kInt || kind_ == Kind::kBool; }
  bool trusted() const { return trusted_; }

  bool boolean() const { return int_ != 0; }
  std::int64_t integer() const { return int_; }
  double number() const { return kind_ == Kind::kFloat ? float_ : static_cast<double>(int_); }
  const Str& str() const { return static_cast<const Str&>(*obj_); }
  const std::shared_ptr<const Object>& object() const { return obj_; }
  const List& list() const { return static_cast<const List&>(*obj_); }
  const Dict& dict() const { return static_cast<const Dict&>(*obj_); }
  Namespace& ns() const { return *ns_; }
  const Callable& callable() const;  // after Callable
  const Loop& loop() const;          // after Loop

  // Gives up this value's heap object, for ReleaseDeferred; the value is
  // left undefined.
  std::shared_ptr<const Object> Abandon() {
    ns_.reset();
    kind_ = Kind::kUndefined;
    return std::move(obj_);
  }

 private:
  Kind kind_ = Kind::kUndefined;
  bool trusted_ = true;
  std::int64_t int_ = 0;
  double float_ = 0;
  std::shared_ptr<const Object> obj_;
  std::shared_ptr<Namespace> ns_;
};

// Releases a dying container's references one at a time: the outermost
// release on a thread destroys everything that becomes unreferenced, each
// object's own references queued rather than released in place, so no
// destruction nests (see List).
void ReleaseDeferred(Value& value);
void ReleaseDeferred(std::shared_ptr<const Object> object);
// Runs the queued releases unless one is already running on this thread.
void DrainDeferred();

// ------------------------------------------------------------------- syntax

enum class ExprKind : std::uint8_t {
  kLiteral,  // constant
  kName,     // name
  kList,     // args
  kTuple,    // args
  kDict,     // args: key, value, key, value...
  kAttr,     // args[0] . name
  kItem,     // args[0] [ args[1] ]
  kSlice,    // args[0] [ args[1] : args[2] : args[3] ] (null where absent)
  kCall,     // args[0] ( args[1..], kwargs )
  kFilter,   // args[0] | name ( args[1..], kwargs )
  kTest,     // args[0] is [not] name ( args[1..], kwargs ); negated
  kNot,
  kNeg,
  kPos,
  kBinary,     // op, args[0], args[1]
  kAnd,        // args[0] and args[1]
  kOr,         // args[0] or args[1]
  kCompare,    // args[0] ops[i] args[i+1]...
  kCondition,  // args[0] if args[1] else args[2] (null: undefined)
};

enum class Op : std::uint8_t {
  kAdd,
  kSub,
  kMul,
  kDiv,
  kFloorDiv,
  kMod,
  kPow,
  kConcat,  // ~
  kEq,
  kNe,
  kLt,
  kLe,
  kGt,
  kGe,
  kIn,
  kNotIn,
};

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

struct Expr {
  ExprKind kind = ExprKind::kLiteral;
  Op op = Op::kAdd;
  bool negated = false;
  std::uint32_t line = 0;
  std::string name;
  Value constant;
  std::vector<ExprPtr> args;
  std::vector<std::pair<std::string, ExprPtr>> kwargs;
  std::vector<Op> ops;  // kCompare
};

enum class StmtKind : std::uint8_t {
  kText,      // text
  kPrint,     // expr
  kIf,        // branches (cond, body); else_body
  kFor,       // targets, expr (iterable), filter, body, else_body
  kSet,       // targets (or ns + attr), expr
  kSetBlock,  // targets[0], body, filters
  kMacro,     // name, params, body
  kBreak,
  kContinue,
};

struct Stmt;
using StmtPtr = std::unique_ptr<Stmt>;
using Body = std::vector<StmtPtr>;

struct Param {
  std::string name;
  ExprPtr default_value;  // null: required
};

struct Stmt {
  StmtKind kind = StmtKind::kText;
  std::uint32_t line = 0;
  std::shared_ptr<const Str> text;                 // kText
  ExprPtr expr;                                    // kPrint, kFor iterable, kSet value
  ExprPtr filter;                                  // kFor's `if`
  std::vector<std::pair<ExprPtr, Body>> branches;  // kIf
  Body body;
  Body else_body;
  std::vector<std::string> targets;  // kFor, kSet, kSetBlock; kSet on a namespace: {ns, attr}
  bool attribute_target = false;     // kSet: targets = {namespace name, attribute}
  bool tuple_target = false;         // unpack into targets
  std::string name;                  // kMacro
  std::vector<Param> params;         // kMacro
  bool uses_varargs = false;         // kMacro: its body reads varargs
  bool uses_kwargs = false;          // kMacro: its body reads kwargs
  std::vector<ExprPtr> filters;      // kSetBlock: kFilter nodes with a null subject
};

struct Program {
  Limits limits;
  Body body;
};

struct MacroDef;

// A callable value: a macro, a global, a bound method or loop.cycle.
struct Callable final : Object {
  Callable() = default;
  Callable(const Callable&) = delete;
  Callable& operator=(const Callable&) = delete;
  Callable(Callable&&) = delete;
  Callable& operator=(Callable&&) = delete;
  ~Callable() override;

  enum class Type : std::uint8_t {
    kMacro,
    kRange,
    kNamespace,
    kDict,
    kRaise,
    kStrftime,
    kMethod,  // `receiver`.`name`(...)
    kCycle,   // loop.cycle, at `index`
  };
  Type type = Type::kMacro;
  const Stmt* macro = nullptr;
  std::shared_ptr<const Object> scope;  // the macro's defining scope (a Scope)
  Value receiver;
  std::string name;
  std::size_t index = 0;
};

inline Value Value::MakeCallable(std::shared_ptr<const Callable> c) {
  Value v;
  v.kind_ = Kind::kCallable;
  v.obj_ = std::move(c);
  return v;
}

inline const Callable& Value::callable() const { return static_cast<const Callable&>(*obj_); }

// A for loop's `loop` object: its attributes derive from these (Jinja2's
// LoopContext, depth 1).
struct Loop final : Object {
  Loop() = default;
  Loop(const Loop&) = delete;
  Loop& operator=(const Loop&) = delete;
  Loop(Loop&&) = delete;
  Loop& operator=(Loop&&) = delete;
  ~Loop() override;

  std::size_t index = 0;  // from 0
  std::size_t length = 0;
  std::optional<Value> previous;
  std::optional<Value> next;
};

inline Value Value::MakeLoop(std::shared_ptr<const Loop> l) {
  Value v;
  v.kind_ = Kind::kLoop;
  v.obj_ = std::move(l);
  return v;
}

inline const Loop& Value::loop() const { return static_cast<const Loop&>(*obj_); }

// ------------------------------------------------------------ value helpers

// Builds a string with provenance.
class StrBuilder {
 public:
  void Append(std::string_view text, bool trusted);
  void Append(const Str& s);
  void Append(const Str& s, std::size_t offset, std::size_t length);
  std::size_t size() const { return s_.text.size(); }
  const std::string& text() const { return s_.text; }
  void set_markup(bool markup) { s_.markup = markup; }
  // The string, charged to the arena; null when a bound is reached.
  std::shared_ptr<const Str> Take(Arena* arena);
  Str& raw() { return s_; }

 private:
  Str s_;
};

std::shared_ptr<const Str> MakeStr(Arena* arena, std::string_view text, bool trusted);

// Python's str() of a value, with provenance: strings as they are, numbers
// and constants trusted when the value is, containers as their repr (never
// trusted). Callables have no text: an error.
std::expected<void, Error> AppendStr(const Value& v, StrBuilder& out, Arena& arena);
// Python's repr(). Like AppendJson, it stops with a bound reached once
// `out` grows past Limits::max_string_bytes: a container may hold one
// string many times over, so its text has no other bound.
std::expected<void, Error> AppendRepr(const Value& v, std::string& out, Arena& arena);

struct JsonOptions {
  bool ensure_ascii = false;
  std::optional<std::string> indent;  // None, or the indent string
  std::string item_separator = ", ";
  std::string key_separator = ": ";
  bool sort_keys = false;
};
// Python's json.dumps.
std::expected<void, Error> AppendJson(const Value& v, const JsonOptions& options, std::string& out,
                                      Arena& arena);

bool Truthy(const Value& v);
// Python's ==; `work` counts the bytes compared (16 a value, and the bytes
// of strings). Past `budget` it stops early with a meaningless result: the
// caller's charge of `work` then fails.
bool Equal(const Value& a, const Value& b, std::uint64_t& work, std::uint64_t budget);
// Python's <, or an error for unorderable kinds; `work` and `budget` as
// for Equal.
std::expected<bool, Error> Less(const Value& a, const Value& b, std::uint64_t& work,
                                std::uint64_t budget);

// Substring search in time linear in the text and the needle
// (Knuth-Morris-Pratt), as Python's two-way search is: no search scans a
// text once per needle byte. Its tables take 4 bytes a needle byte for each
// direction searched (the evaluator bounds needles, kMaxNeedleShare). The
// needle must outlive the searcher.
class Searcher {
 public:
  explicit Searcher(std::string_view needle) : needle_(needle) {}
  // The first occurrence starting at or after `from`, or npos.
  std::size_t Find(std::string_view text, std::size_t from = 0) const;
  // The last occurrence that ends at or before `end`, or npos.
  std::size_t RFind(std::string_view text, std::size_t end) const;

 private:
  std::string_view needle_;
  mutable std::vector<std::uint32_t> forward_;   // the needle's prefix function, when needed
  mutable std::vector<std::uint32_t> backward_;  // the reversed needle's
};
// A needle may be at most this share of Limits::max_live_bytes.
inline constexpr std::size_t kMaxNeedleShare = 16;

// Parses a template's source (jinja_parse.cc).
std::expected<Body, Error> ParseProgram(std::string_view source, const Limits& limits);

// Whether the subset has a filter or test of this name (parsing refuses
// others, as Jinja2 refuses unknown names when it compiles).
bool KnownFilter(std::string_view name);
bool KnownTest(std::string_view name);

// Code point helpers on UTF-8 (the template and every input are valid UTF-8).
std::size_t CodePoints(std::string_view s);
// The byte offset of code point `index` (index <= CodePoints(s)).
std::size_t ByteOffset(std::string_view s, std::size_t index);
// Python's str.isspace() for the code point at s[at...]; its length or 0.
std::size_t SpaceAt(std::string_view s, std::size_t at);
// Python's str.isprintable() for a code point.
bool Printable(char32_t cp);
// Decodes the code point at s[at...] (valid UTF-8); `length` receives its size.
char32_t DecodeAt(std::string_view s, std::size_t at, std::size_t& length);

}  // namespace jitllm::chat::jinja

#endif  // JITLLM_CHAT_JINJA_INTERNAL_H_
