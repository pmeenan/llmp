// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A bounded, sandboxed interpreter for the subset of Jinja2 that chat
// templates use (D-067 as amended 2026-10-02): the fallback for a template
// no native family renderer reproduces. It renders as Hugging Face
// transformers' apply_chat_template does (Jinja2's immutable sandbox,
// trim_blocks and lstrip_blocks, the loop-controls extension, transformers'
// tojson, raise_exception and strftime_now, its {% generation %} tag), and
// llama.cpp's from_json filter. docs/tokenizer.md lists the subset.
//
// Untrusted input: the template comes from a checkpoint and the values from
// a client. Nothing reaches files, the network, the environment or process
// state; the only callables are a fixed list of filters, tests, methods and
// globals; and every resource has a bound (Limits). A bound reached is an
// error, never a truncated rendering. Where Jinja2 would raise, rendering
// fails too; where jitLLM does not implement what Jinja2 would do, parsing
// or rendering is refused rather than approximated.
//
// Provenance: every rendered byte is known to come from the template's own
// text (a literal, a value the runtime supplied as trusted such as the BOS
// token text, or a number the template computed) or from the input values.
// Rendered::trusted lists the template's spans; only those may hold
// control tokens (chat::RenderInterpreted marks them).

#ifndef JITLLM_CHAT_JINJA_H_
#define JITLLM_CHAT_JINJA_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/json.h"

namespace jitllm::chat::jinja {

struct Limits {
  std::size_t max_template_bytes = std::size_t{1} << 20U;
  std::size_t max_nodes = std::size_t{1} << 18U;  // AST nodes
  std::size_t max_parse_depth = 256;              // nested blocks and expression tree depth
  std::size_t max_eval_depth = 512;               // evaluator recursion, macros included
  std::size_t max_macro_depth = 32;               // nested macro calls
  std::size_t max_value_depth = 64;               // nested lists and mappings
  std::uint64_t max_steps = 50'000'000;           // evaluation steps
  std::uint64_t max_work_bytes = std::uint64_t{2} << 30U;  // bytes built or scanned
  std::size_t max_live_bytes = std::size_t{256} << 20U;    // bytes held by values at once
  std::size_t max_string_bytes = std::size_t{64} << 20U;   // one string
  std::size_t max_output_bytes = std::size_t{32} << 20U;
  std::size_t max_range = 100'000;  // Jinja2's sandbox MAX_RANGE
};

enum class Code : std::uint8_t {
  kSyntax,       // the template does not parse
  kUnsupported,  // a construct, filter, test or method the subset leaves out
  kLimit,        // a bound reached
  kRaised,       // the template called raise_exception
  kRuntime,      // what Jinja2 raises at run time: an undefined value used, a type error
};

struct Error {
  Code code = Code::kRuntime;
  std::string_view reason;  // static text
  std::uint32_t line = 0;   // in the template, from 1; 0 when none applies

  std::string ToString() const;
};

struct Program;

// A value passed into a rendering: built from JSON (client data, never
// trusted) or scalars and strings the runtime supplies.
class Input {
 public:
  static Input Undefined();
  static Input None();
  static Input Bool(bool b);
  static Input Int(std::int64_t value);
  // `trusted` marks text the runtime vouches for, such as the BOS token's.
  static Input String(std::string_view text, bool trusted = false);
  static Input Json(base::json::Value value);
  static Input List(std::vector<Input> items);
  static Input Map(std::vector<std::pair<std::string, Input>> members);

  Input(const Input&);
  Input(Input&&) noexcept;
  Input& operator=(const Input&);
  Input& operator=(Input&&) noexcept;
  ~Input();

 private:
  friend class Renderer;
  enum class Kind : std::uint8_t { kUndefined, kNone, kBool, kInt, kString, kJson, kList, kMap };
  Input() = default;
  Kind kind_ = Kind::kUndefined;
  bool b_ = false;
  std::int64_t i_ = 0;
  bool trusted_ = false;
  std::string text_;
  std::optional<base::json::Value> json_;
  std::vector<Input> items_;
  std::vector<std::pair<std::string, Input>> members_;
};

// A civil time for strftime_now (local time, as Python's datetime.now()).
struct CivilTime {
  int year = 1970;
  int month = 1;    // 1..12
  int day = 1;      // 1..31
  int hour = 0;     // 0..23
  int minute = 0;   // 0..59
  int second = 0;   // 0..60
  int weekday = 4;  // 0 Sunday .. 6 Saturday
  int yearday = 0;  // 0..365
};

struct Rendered {
  std::string text;
  // Byte ranges of `text` that came only from the template, in order, not
  // overlapping or touching.
  std::vector<std::pair<std::size_t, std::size_t>> trusted;  // (offset, length)
};

// Steps and work bytes renderings used.
struct Usage {
  std::uint64_t steps = 0;
  std::uint64_t work_bytes = 0;
};

// Tighter bounds for one rendering than the template's Limits (probes use
// them); zero keeps the template's own. A rendering adds what it used to
// `usage` when one is given (probes share one pool).
struct Budget {
  std::uint64_t max_steps = 0;
  std::uint64_t max_work_bytes = 0;
  Usage* usage = nullptr;
};

// A parsed template. Immutable; renderings may run concurrently.
class Template {
 public:
  static std::expected<Template, Error> Parse(std::string_view source, const Limits& limits = {});

  Template(Template&&) noexcept;
  Template& operator=(Template&&) noexcept;
  Template(const Template&) = delete;
  Template& operator=(const Template&) = delete;
  ~Template();

  // Renders with the given top-level variables. Without `now`,
  // strftime_now is refused; `budget` lowers the step and work bounds for
  // this rendering.
  std::expected<Rendered, Error> Render(const std::vector<std::pair<std::string, Input>>& variables,
                                        const std::optional<CivilTime>& now = std::nullopt,
                                        Budget budget = {}) const;

  const Limits& limits() const;

 private:
  explicit Template(std::unique_ptr<Program> program);
  std::unique_ptr<Program> program_;
};

}  // namespace jitllm::chat::jinja

#endif  // JITLLM_CHAT_JINJA_H_
