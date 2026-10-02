// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The Jinja-subset lexer and parser (jinja.h). Written from Jinja2 3.1's
// documented grammar and observed behaviour: newlines normalized and one
// trailing newline dropped (keep_trailing_newline off), trim_blocks and
// lstrip_blocks as transformers sets them, `-` and `+` whitespace control,
// string literals decoded as Python's unicode-escape, and Jinja2's operator
// precedence (filters bind tightest, unary minus binds tighter than `**`,
// `**` associates to the left). Every recursion counts against
// Limits::max_parse_depth and every node against Limits::max_nodes.

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base/json.h"
#include "chat/jinja.h"
#include "chat/jinja_internal.h"
#include "tokenizer/unicode.h"

namespace jitllm::chat::jinja {
namespace {

enum class Tok : std::uint8_t {
  kText,
  kVarBegin,
  kVarEnd,
  kBlockBegin,
  kBlockEnd,
  kName,
  kString,
  kInt,
  kFloat,
  kOp,
};

struct Token {
  Tok type = Tok::kText;
  std::uint32_t line = 1;
  std::string text;  // data, a name, an operator or a decoded string
  std::int64_t integer = 0;
  double number = 0;
};

std::unexpected<Error> Fail(Code code, std::string_view reason, std::uint32_t line) {
  return std::unexpected(Error{code, reason, line});
}

// Python's str.rstrip() and the length of a run of whitespace (\s).
std::size_t SpaceRun(std::string_view s, std::size_t at) {
  std::size_t end = at;
  while (const std::size_t n = SpaceAt(s, end)) {
    end += n;
  }
  return end - at;
}

std::string_view RStrip(std::string_view s) {
  // Walk forward: the last non-space code point ends the result.
  std::size_t keep = 0;
  for (std::size_t at = 0; at < s.size();) {
    if (const std::size_t n = SpaceAt(s, at)) {
      at += n;
    } else {
      std::size_t length = 0;
      DecodeAt(s, at, length);
      at += length;
      keep = at;
    }
  }
  return s.substr(0, keep);
}

bool AllSpace(std::string_view s) { return !s.empty() && SpaceRun(s, 0) == s.size(); }

bool IsNameStart(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
bool IsNameChar(char c) { return IsNameStart(c) || (c >= '0' && c <= '9'); }
bool IsDigit(char c) { return c >= '0' && c <= '9'; }

int HexValue(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  return -1;
}

// A string literal's body decoded as Python's unicode-escape (Jinja2's
// lexer): the escapes below, an unknown one kept with its backslash.
std::expected<std::string, Error> DecodeString(std::string_view body, std::uint32_t line) {
  std::string out;
  for (std::size_t i = 0; i < body.size(); ++i) {
    const char c = body[i];
    if (c != '\\') {
      out.push_back(c);
      continue;
    }
    if (i + 1 >= body.size()) {
      return Fail(Code::kSyntax, "a string literal ends in a backslash", line);
    }
    const char e = body[++i];
    switch (e) {
      case '\n':
        break;  // a line continuation
      case '\\':
        out.push_back('\\');
        break;
      case '\'':
        out.push_back('\'');
        break;
      case '"':
        out.push_back('"');
        break;
      case 'a':
        out.push_back('\a');
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'v':
        out.push_back('\v');
        break;
      case 'x':
      case 'u':
      case 'U': {
        std::size_t digits = 8;
        if (e == 'x') {
          digits = 2;
        } else if (e == 'u') {
          digits = 4;
        }
        if (i + digits >= body.size()) {
          return Fail(Code::kSyntax, "a truncated escape in a string literal", line);
        }
        std::uint32_t cp = 0;
        for (std::size_t k = 1; k <= digits; ++k) {
          const int h = HexValue(body[i + k]);
          if (h < 0) {
            return Fail(Code::kSyntax, "a truncated escape in a string literal", line);
          }
          cp = (cp << 4U) | static_cast<std::uint32_t>(h);
        }
        i += digits;
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
          return Fail(Code::kSyntax, "a string literal escapes a surrogate or invalid code point",
                      line);
        }
        tokenizer::unicode::AppendUtf8(static_cast<char32_t>(cp), out);
        break;
      }
      case 'N':
        return Fail(Code::kUnsupported, "a named escape (\\N{...}) in a string literal", line);
      default:
        if (e >= '0' && e <= '7') {
          auto cp = static_cast<std::uint32_t>(e - '0');
          for (int k = 0; k < 2 && i + 1 < body.size() && body[i + 1] >= '0' && body[i + 1] <= '7';
               ++k) {
            cp = (cp << 3U) | static_cast<std::uint32_t>(body[++i] - '0');
          }
          tokenizer::unicode::AppendUtf8(static_cast<char32_t>(cp), out);
        } else {
          out.push_back('\\');
          out.push_back(e);
        }
    }
  }
  return out;
}

class Lexer {
 public:
  Lexer(std::string_view source, const Limits& limits) : src_(source), limits_(limits) {}

  std::expected<std::vector<Token>, Error> Run() {
    bool line_starting = true;
    while (pos_ < src_.size()) {
      std::size_t tag = pos_;
      while (true) {
        tag = src_.find('{', tag);
        if (tag == std::string_view::npos || tag + 1 >= src_.size()) {
          tag = std::string_view::npos;
          break;
        }
        const char k = src_[tag + 1];
        if (k == '{' || k == '%' || k == '#') {
          break;
        }
        ++tag;
      }
      if (tag == std::string_view::npos) {
        Text(src_.substr(pos_));
        Advance(src_.size());
        break;
      }
      const char kind = src_[tag + 1];
      const char sign = tag + 2 < src_.size() && (src_[tag + 2] == '-' || src_[tag + 2] == '+')
                            ? src_[tag + 2]
                            : 0;
      std::string_view text = src_.substr(pos_, tag - pos_);
      if (sign == '-') {
        text = RStrip(text);
      } else if (sign != '+' && kind != '{') {
        // lstrip_blocks: whitespace from the line's start to a block or
        // comment tag goes.
        const std::size_t newline = text.rfind('\n');
        const std::size_t l_pos = newline == std::string_view::npos ? 0 : newline + 1;
        if ((l_pos > 0 || line_starting) && AllSpace(text.substr(l_pos))) {
          text = text.substr(0, l_pos);
        }
      }
      Text(text);
      Advance(tag + 2 + (sign != 0 ? 1 : 0));
      if (kind == '#') {
        auto r = Comment();
        if (!r) {
          return std::unexpected(r.error());
        }
        line_starting = *r;
        continue;
      }
      if (kind == '%' && IsRaw()) {
        auto r = Raw();
        if (!r) {
          return std::unexpected(r.error());
        }
        line_starting = *r;
        continue;
      }
      auto r = Tag(kind == '{');
      if (!r) {
        return std::unexpected(r.error());
      }
      line_starting = *r;
    }
    return std::move(tokens_);
  }

 private:
  void Advance(std::size_t to) {
    for (std::size_t i = pos_; i < to; ++i) {
      if (src_[i] == '\n') {
        ++line_;
      }
    }
    pos_ = to;
  }

  void Text(std::string_view text) {
    if (text.empty()) {
      return;
    }
    if (!tokens_.empty() && tokens_.back().type == Tok::kText) {
      tokens_.back().text += text;
      return;
    }
    tokens_.push_back({Tok::kText, line_, std::string(text), 0, 0});
  }

  // After `{#`: through the comment's end; whether that ended a line.
  std::expected<bool, Error> Comment() {
    const std::size_t end = src_.find("#}", pos_);
    if (end == std::string_view::npos) {
      return Fail(Code::kSyntax, "a comment that never ends", line_);
    }
    const char sign =
        end > pos_ && (src_[end - 1] == '-' || src_[end - 1] == '+') ? src_[end - 1] : 0;
    return EndTag(end + 2, sign);
  }

  // Consumes a tag end at `after` (just past `%}`, `}}` or `#}`): `-`
  // strips following whitespace, trim_blocks one newline after a block or
  // comment; whether the consumed text ended a line.
  bool EndTag(std::size_t after, char sign, bool variable = false) {
    std::size_t to = after;
    if (sign == '-') {
      to += SpaceRun(src_, after);
    } else if (sign != '+' && !variable && after < src_.size() && src_[after] == '\n') {
      ++to;
    }
    Advance(to);
    return to > 0 && src_[to - 1] == '\n';
  }

  bool IsRaw() const {
    std::size_t at = pos_;
    at += SpaceRun(src_, at);
    if (src_.substr(at, 3) != "raw") {
      return false;
    }
    at += 3;
    at += SpaceRun(src_, at);
    return src_.substr(at, 2) == "%}" || src_.substr(at, 3) == "-%}" || src_.substr(at, 3) == "+%}";
  }

  // After `{%` of a raw block: its body is text through `{% endraw %}`.
  std::expected<bool, Error> Raw() {
    std::size_t at = src_.find("%}", pos_);
    const char open_sign = src_[at - 1] == '-' || src_[at - 1] == '+' ? src_[at - 1] : 0;
    EndTag(at + 2, open_sign);
    std::size_t search = pos_;
    while (true) {
      const std::size_t begin = src_.find("{%", search);
      if (begin == std::string_view::npos) {
        return Fail(Code::kSyntax, "a raw block that never ends", line_);
      }
      std::size_t k = begin + 2;
      const char sign = k < src_.size() && (src_[k] == '-' || src_[k] == '+') ? src_[k] : 0;
      k += sign != 0 ? 1 : 0;
      k += SpaceRun(src_, k);
      if (src_.substr(k, 6) == "endraw") {
        k += 6;
        k += SpaceRun(src_, k);
        const char end_sign = k < src_.size() && (src_[k] == '-' || src_[k] == '+') ? src_[k] : 0;
        k += end_sign != 0 ? 1 : 0;
        if (src_.substr(k, 2) == "%}") {
          std::string_view body = src_.substr(pos_, begin - pos_);
          if (sign == '-') {
            body = RStrip(body);
          }
          Text(body);
          Advance(begin);
          return EndTag(k + 2, end_sign);
        }
      }
      search = begin + 2;
    }
  }

  // After `{{` or `{%`: the expression's tokens and the end; whether the
  // end consumed a line's end.
  std::expected<bool, Error> Tag(bool variable) {
    tokens_.push_back({variable ? Tok::kVarBegin : Tok::kBlockBegin, line_, {}, 0, 0});
    std::vector<char> balance;
    while (true) {
      Advance(pos_ + SpaceRun(src_, pos_));
      if (pos_ >= src_.size()) {
        return Fail(Code::kSyntax, "a tag that never ends", line_);
      }
      if (tokens_.size() > limits_.max_nodes * 2) {
        return Fail(Code::kLimit, "the template has more tokens than its bound", line_);
      }
      const std::string_view rest = src_.substr(pos_);
      if (balance.empty()) {
        const std::string_view end = variable ? "}}" : "%}";
        char sign = 0;
        bool found = false;
        if (rest.starts_with(end)) {
          found = true;
        } else if (rest.size() > 2 && (rest[0] == '-' || (!variable && rest[0] == '+')) &&
                   rest.substr(1).starts_with(end)) {
          sign = rest[0];
          found = true;
        }
        if (found) {
          tokens_.push_back({variable ? Tok::kVarEnd : Tok::kBlockEnd, line_, {}, 0, 0});
          return EndTag(pos_ + (sign != 0 ? 3 : 2), sign, variable);
        }
      }
      const char c = rest[0];
      if (IsDigit(c)) {
        if (auto r = Number(); !r) {
          return std::unexpected(r.error());
        }
        continue;
      }
      if (IsNameStart(c)) {
        std::size_t n = 1;
        while (n < rest.size() && IsNameChar(rest[n])) {
          ++n;
        }
        tokens_.push_back({Tok::kName, line_, std::string(rest.substr(0, n)), 0, 0});
        Advance(pos_ + n);
        continue;
      }
      if (c == '\'' || c == '"') {
        std::size_t n = 1;
        while (n < rest.size() && rest[n] != c) {
          n += rest[n] == '\\' ? 2 : 1;
        }
        if (n >= rest.size()) {
          return Fail(Code::kSyntax, "a string literal that never ends", line_);
        }
        auto decoded = DecodeString(rest.substr(1, n - 1), line_);
        if (!decoded) {
          return std::unexpected(decoded.error());
        }
        tokens_.push_back({Tok::kString, line_, std::move(*decoded), 0, 0});
        Advance(pos_ + n + 1);
        continue;
      }
      std::string_view op;
      for (const std::string_view two : {"//", "**", "==", "!=", ">=", "<="}) {
        if (rest.starts_with(two)) {
          op = two;
          break;
        }
      }
      if (op.empty()) {
        if (!std::string_view("+-/*%~[](){}<>=.:|,").contains(c)) {
          return Fail(Code::kSyntax, "an unexpected character in a tag", line_);
        }
        op = rest.substr(0, 1);
      }
      if (op == "(" || op == "[" || op == "{") {
        balance.push_back(op[0]);
      } else if (op == ")" || op == "]" || op == "}") {
        char open = '{';
        if (op == ")") {
          open = '(';
        } else if (op == "]") {
          open = '[';
        }
        if (balance.empty() || balance.back() != open) {
          return Fail(Code::kSyntax, "unbalanced brackets in a tag", line_);
        }
        balance.pop_back();
      }
      tokens_.push_back({Tok::kOp, line_, std::string(op), 0, 0});
      Advance(pos_ + op.size());
    }
  }

  std::expected<void, Error> Number() {
    const std::string_view rest = src_.substr(pos_);
    std::size_t n = 0;
    std::string digits;
    const auto run = [&]() {
      // (\d+_)*\d+
      const std::size_t start = n;
      while (n < rest.size() && (IsDigit(rest[n]) || (rest[n] == '_' && n + 1 < rest.size() &&
                                                      IsDigit(rest[n + 1]) && n > start))) {
        if (rest[n] != '_') {
          digits.push_back(rest[n]);
        }
        ++n;
      }
      return n > start;
    };
    run();
    bool is_float = false;
    if (n + 1 < rest.size() && rest[n] == '.' && IsDigit(rest[n + 1])) {
      digits.push_back('.');
      ++n;
      run();
      is_float = true;
    }
    if (n < rest.size() && (rest[n] == 'e' || rest[n] == 'E')) {
      std::size_t k = n + 1;
      std::string exp = "e";
      if (k < rest.size() && (rest[k] == '+' || rest[k] == '-')) {
        exp.push_back(rest[k]);
        ++k;
      }
      if (k < rest.size() && IsDigit(rest[k])) {
        const std::size_t keep = digits.size();
        digits += exp;
        n = k;
        if (run()) {
          is_float = true;
        } else {
          digits.resize(keep);
        }
      }
    }
    Token t{is_float ? Tok::kFloat : Tok::kInt, line_, {}, 0, 0};
    if (is_float) {
      const auto [end, ec] =
          std::from_chars(digits.data(), digits.data() + digits.size(), t.number);
      if (ec != std::errc() && ec != std::errc::result_out_of_range) {
        return Fail(Code::kSyntax, "a malformed number", line_);
      }
      if (ec == std::errc::result_out_of_range) {
        t.number = HUGE_VAL;
      }
    } else {
      const auto [end, ec] =
          std::from_chars(digits.data(), digits.data() + digits.size(), t.integer);
      if (ec != std::errc()) {
        return Fail(Code::kUnsupported, "an integer literal beyond 64 bits", line_);
      }
    }
    tokens_.push_back(std::move(t));
    Advance(pos_ + n);
    return {};
  }

  std::string_view src_;
  const Limits& limits_;
  std::size_t pos_ = 0;
  std::uint32_t line_ = 1;
  std::vector<Token> tokens_;
};

class Parser {
 public:
  Parser(std::vector<Token> tokens, const Limits& limits)
      : t_(std::move(tokens)), limits_(limits) {}

  std::expected<Body, Error> Run() {
    auto body = ParseBody({});
    if (!body) {
      return body;
    }
    if (at_ < t_.size()) {
      return Fail(Code::kSyntax, "an end tag without its block", t_[at_].line);
    }
    return body;
  }

 private:
  // Depth accounting for every recursive parse.
  class Nest {
   public:
    explicit Nest(Parser& p) : p_(p) { ++p_.depth_; }
    Nest(const Nest&) = delete;
    Nest& operator=(const Nest&) = delete;
    Nest(Nest&&) = delete;
    Nest& operator=(Nest&&) = delete;
    ~Nest() { --p_.depth_; }
    bool ok() const { return p_.depth_ <= p_.limits_.max_parse_depth; }

   private:
    Parser& p_;
  };

  // A left-associative chain (a + b + c, x|f|g, a.b.c) nests its tree one
  // level per link: each link counts as depth while the chain parses.
  class Chain {
   public:
    explicit Chain(Parser& p) : p_(p) {}
    Chain(const Chain&) = delete;
    Chain& operator=(const Chain&) = delete;
    Chain(Chain&&) = delete;
    Chain& operator=(Chain&&) = delete;
    ~Chain() { p_.depth_ -= links_; }
    bool Grow() {
      ++links_;
      ++p_.depth_;
      return p_.depth_ <= p_.limits_.max_parse_depth;
    }

   private:
    Parser& p_;
    std::size_t links_ = 0;
  };

  std::unexpected<Error> TooDeep() const {
    return Fail(Code::kLimit, "expressions nest deeper than their bound", Line());
  }

  std::uint32_t Line() const {
    if (at_ < t_.size()) {
      return t_[at_].line;
    }
    return t_.empty() ? 1 : t_.back().line;
  }
  const Token* Peek(std::size_t ahead = 0) const {
    return at_ + ahead < t_.size() ? &t_[at_ + ahead] : nullptr;
  }
  bool IsOp(std::string_view op, std::size_t ahead = 0) const {
    const Token* t = Peek(ahead);
    return t != nullptr && t->type == Tok::kOp && t->text == op;
  }
  bool IsName(std::string_view name, std::size_t ahead = 0) const {
    const Token* t = Peek(ahead);
    return t != nullptr && t->type == Tok::kName && t->text == name;
  }
  bool Is(Tok type) const {
    const Token* t = Peek();
    return t != nullptr && t->type == type;
  }
  std::expected<void, Error> Expect(Tok type, std::string_view what) {
    if (!Is(type)) {
      return Fail(Code::kSyntax, what, Line());
    }
    ++at_;
    return {};
  }
  std::expected<void, Error> ExpectOp(std::string_view op, std::string_view what) {
    if (!IsOp(op)) {
      return Fail(Code::kSyntax, what, Line());
    }
    ++at_;
    return {};
  }
  std::expected<std::string, Error> ExpectName(std::string_view what) {
    if (!Is(Tok::kName)) {
      return Fail(Code::kSyntax, what, Line());
    }
    return t_[at_++].text;
  }

  std::expected<ExprPtr, Error> Node(ExprKind kind) {
    if (++nodes_ > limits_.max_nodes) {
      return Fail(Code::kLimit, "the template has more nodes than its bound", Line());
    }
    auto e = std::make_unique<Expr>();
    e->kind = kind;
    e->line = Line();
    return e;
  }
  std::expected<StmtPtr, Error> Statement(StmtKind kind, std::uint32_t line) {
    if (++nodes_ > limits_.max_nodes) {
      return Fail(Code::kLimit, "the template has more nodes than its bound", line);
    }
    auto s = std::make_unique<Stmt>();
    s->kind = kind;
    s->line = line;
    return s;
  }

  // Statements up to a block tag named in `stop` (left unconsumed after
  // its kBlockBegin) or the end.
  std::expected<Body, Error> ParseBody(std::initializer_list<std::string_view> stop) {
    const Nest nest(*this);
    if (!nest.ok()) {
      return Fail(Code::kLimit, "blocks nest deeper than their bound", Line());
    }
    Body body;
    while (at_ < t_.size()) {
      const Token& t = t_[at_];
      if (t.type == Tok::kText) {
        auto s = Statement(StmtKind::kText, t.line);
        if (!s) {
          return std::unexpected(s.error());
        }
        (*s)->text = MakeStr(nullptr, t.text, true);
        body.push_back(std::move(*s));
        ++at_;
        continue;
      }
      if (t.type == Tok::kVarBegin) {
        ++at_;
        auto s = Statement(StmtKind::kPrint, t.line);
        if (!s) {
          return std::unexpected(s.error());
        }
        auto e = ParseTuple(true);
        if (!e) {
          return std::unexpected(e.error());
        }
        (*s)->expr = std::move(*e);
        if (auto r = Expect(Tok::kVarEnd, "an expression that does not end at its tag"); !r) {
          return std::unexpected(r.error());
        }
        body.push_back(std::move(*s));
        continue;
      }
      if (t.type != Tok::kBlockBegin) {
        return Fail(Code::kSyntax, "an unexpected token", t.line);
      }
      const Token* name = Peek(1);
      if (name == nullptr || name->type != Tok::kName) {
        return Fail(Code::kSyntax, "a block tag without a name", t.line);
      }
      for (const std::string_view s : stop) {
        if (name->text == s) {
          ++at_;
          return body;
        }
      }
      at_ += 2;
      auto r = ParseStatement(name->text, t.line, body);
      if (!r) {
        return std::unexpected(r.error());
      }
    }
    if (stop.size() != 0) {
      return Fail(Code::kSyntax, "a block that never ends", Line());
    }
    return body;
  }

  // The rest of a block tag after `name`; appends what it parses.
  std::expected<void, Error> ParseStatement(const std::string& name, std::uint32_t line,
                                            Body& body) {
    if (name == "if") {
      return ParseIf(line, body);
    }
    if (name == "for") {
      return ParseFor(line, body);
    }
    if (name == "set") {
      return ParseSet(line, body);
    }
    if (name == "macro") {
      return ParseMacro(line, body);
    }
    if (name == "break" || name == "continue") {
      if (loops_ == 0) {
        return Fail(Code::kSyntax, "break or continue outside a loop", line);
      }
      auto s = Statement(name == "break" ? StmtKind::kBreak : StmtKind::kContinue, line);
      if (!s) {
        return std::unexpected(s.error());
      }
      body.push_back(std::move(*s));
      return Expect(Tok::kBlockEnd, "break or continue with arguments");
    }
    if (name == "generation") {
      // transformers' assistant-mask tag renders its body unchanged.
      if (auto r = Expect(Tok::kBlockEnd, "generation with arguments"); !r) {
        return r;
      }
      auto inner = ParseBody({"endgeneration"});
      if (!inner) {
        return std::unexpected(inner.error());
      }
      ++at_;
      for (StmtPtr& s : *inner) {
        body.push_back(std::move(s));
      }
      return Expect(Tok::kBlockEnd, "endgeneration with arguments");
    }
    for (const std::string_view unsupported :
         {"call", "filter", "include", "import", "from", "extends", "block", "with", "autoescape",
          "do", "trans", "raw"}) {
      if (name == unsupported) {
        return Fail(Code::kUnsupported,
                    "a tag the subset leaves out (call, filter, include, import, extends, block, "
                    "with, autoescape, do, trans)",
                    line);
      }
    }
    return Fail(Code::kSyntax, "an unknown or misplaced block tag", line);
  }

  std::expected<void, Error> ParseIf(std::uint32_t line, Body& body) {
    auto s = Statement(StmtKind::kIf, line);
    if (!s) {
      return std::unexpected(s.error());
    }
    while (true) {
      auto cond = ParseTuple(true);
      if (!cond) {
        return std::unexpected(cond.error());
      }
      if (auto r = Expect(Tok::kBlockEnd, "an if condition that does not end at its tag"); !r) {
        return r;
      }
      auto branch = ParseBody({"elif", "else", "endif"});
      if (!branch) {
        return std::unexpected(branch.error());
      }
      (*s)->branches.emplace_back(std::move(*cond), std::move(*branch));
      const std::string tag = t_[at_++].text;
      if (tag == "elif") {
        continue;
      }
      if (tag == "else") {
        if (auto r = Expect(Tok::kBlockEnd, "else with arguments"); !r) {
          return r;
        }
        auto other = ParseBody({"endif"});
        if (!other) {
          return std::unexpected(other.error());
        }
        (*s)->else_body = std::move(*other);
        ++at_;
      }
      break;
    }
    body.push_back(std::move(*s));
    return Expect(Tok::kBlockEnd, "endif with arguments");
  }

  // Names to assign: `a`, `a, b`, `(a, b)`.
  std::expected<void, Error> ParseTargets(Stmt& s) {
    const bool paren = IsOp("(");
    if (paren) {
      ++at_;
    }
    while (true) {
      auto n = ExpectName("an assignment to something other than a name");
      if (!n) {
        return std::unexpected(n.error());
      }
      s.targets.push_back(std::move(*n));
      if (!IsOp(",")) {
        break;
      }
      ++at_;
      s.tuple_target = true;
      if (paren && IsOp(")")) {
        break;
      }
      if (!paren && !Is(Tok::kName)) {
        break;
      }
    }
    if (paren) {
      s.tuple_target = true;
      return ExpectOp(")", "an unclosed assignment target");
    }
    return {};
  }

  std::expected<void, Error> ParseFor(std::uint32_t line, Body& body) {
    auto s = Statement(StmtKind::kFor, line);
    if (!s) {
      return std::unexpected(s.error());
    }
    if (auto r = ParseTargets(**s); !r) {
      return r;
    }
    if (!IsName("in")) {
      return Fail(Code::kSyntax, "a for loop without `in`", line);
    }
    ++at_;
    auto iter = ParseTuple(false);
    if (!iter) {
      return std::unexpected(iter.error());
    }
    (*s)->expr = std::move(*iter);
    if (IsName("if")) {
      ++at_;
      auto cond = ParseExpression(true);
      if (!cond) {
        return std::unexpected(cond.error());
      }
      (*s)->filter = std::move(*cond);
    }
    if (IsName("recursive")) {
      return Fail(Code::kUnsupported, "a recursive for loop", line);
    }
    if (auto r = Expect(Tok::kBlockEnd, "a for loop that does not end at its tag"); !r) {
      return r;
    }
    ++loops_;
    auto inner = ParseBody({"else", "endfor"});
    --loops_;
    if (!inner) {
      return std::unexpected(inner.error());
    }
    (*s)->body = std::move(*inner);
    if (t_[at_++].text == "else") {
      if (auto r = Expect(Tok::kBlockEnd, "else with arguments"); !r) {
        return r;
      }
      auto other = ParseBody({"endfor"});
      if (!other) {
        return std::unexpected(other.error());
      }
      (*s)->else_body = std::move(*other);
      ++at_;
    }
    body.push_back(std::move(*s));
    return Expect(Tok::kBlockEnd, "endfor with arguments");
  }

  std::expected<void, Error> ParseSet(std::uint32_t line, Body& body) {
    auto s = Statement(StmtKind::kSet, line);
    if (!s) {
      return std::unexpected(s.error());
    }
    if (Is(Tok::kName) && IsOp(".", 1)) {
      (*s)->targets.push_back(t_[at_].text);
      at_ += 2;
      auto attr = ExpectName("an attribute assignment without a name");
      if (!attr) {
        return std::unexpected(attr.error());
      }
      (*s)->targets.push_back(std::move(*attr));
      (*s)->attribute_target = true;
    } else if (auto r = ParseTargets(**s); !r) {
      return r;
    }
    if (IsOp("=")) {
      ++at_;
      auto value = ParseTuple(true);
      if (!value) {
        return std::unexpected(value.error());
      }
      (*s)->expr = std::move(*value);
      body.push_back(std::move(*s));
      return Expect(Tok::kBlockEnd, "a set that does not end at its tag");
    }
    // A block set: {% set x %}...{% endset %}, optionally filtered.
    if ((*s)->attribute_target || (*s)->tuple_target) {
      return Fail(Code::kUnsupported, "a block set into an attribute or a tuple", line);
    }
    (*s)->kind = StmtKind::kSetBlock;
    while (IsOp("|")) {
      ++at_;
      auto f = ParseFilterCall(nullptr);
      if (!f) {
        return std::unexpected(f.error());
      }
      (*s)->filters.push_back(std::move(*f));
    }
    if (auto r = Expect(Tok::kBlockEnd, "a set without a value"); !r) {
      return r;
    }
    auto inner = ParseBody({"endset"});
    if (!inner) {
      return std::unexpected(inner.error());
    }
    ++at_;
    (*s)->body = std::move(*inner);
    body.push_back(std::move(*s));
    return Expect(Tok::kBlockEnd, "endset with arguments");
  }

  std::expected<void, Error> ParseMacro(std::uint32_t line, Body& body) {
    auto s = Statement(StmtKind::kMacro, line);
    if (!s) {
      return std::unexpected(s.error());
    }
    auto name = ExpectName("a macro without a name");
    if (!name) {
      return std::unexpected(name.error());
    }
    (*s)->name = std::move(*name);
    if (auto r = ExpectOp("(", "a macro without parameters"); !r) {
      return r;
    }
    bool defaults = false;
    while (!IsOp(")")) {
      auto param = ExpectName("a macro parameter that is not a name");
      if (!param) {
        return std::unexpected(param.error());
      }
      Param p{std::move(*param), nullptr};
      if (IsOp("=")) {
        ++at_;
        auto value = ParseExpression(true);
        if (!value) {
          return std::unexpected(value.error());
        }
        p.default_value = std::move(*value);
        defaults = true;
      } else if (defaults) {
        return Fail(Code::kSyntax, "a macro parameter without a default after one with", line);
      }
      (*s)->params.push_back(std::move(p));
      if (!IsOp(",")) {
        break;
      }
      ++at_;
    }
    if (auto r = ExpectOp(")", "an unclosed macro parameter list"); !r) {
      return r;
    }
    if (auto r = Expect(Tok::kBlockEnd, "a macro that does not end at its tag"); !r) {
      return r;
    }
    const bool saw_varargs = saw_varargs_;
    const bool saw_kwargs = saw_kwargs_;
    const std::size_t loops = loops_;
    saw_varargs_ = false;
    saw_kwargs_ = false;
    loops_ = 0;
    auto inner = ParseBody({"endmacro"});
    loops_ = loops;
    (*s)->uses_varargs = saw_varargs_;
    (*s)->uses_kwargs = saw_kwargs_;
    saw_varargs_ = saw_varargs || saw_varargs_;
    saw_kwargs_ = saw_kwargs || saw_kwargs_;
    if (!inner) {
      return std::unexpected(inner.error());
    }
    ++at_;
    (*s)->body = std::move(*inner);
    body.push_back(std::move(*s));
    return Expect(Tok::kBlockEnd, "endmacro with arguments");
  }

  // ------------------------------------------------------------ expressions

  // Comma-separated expressions: a tuple when there is a comma.
  std::expected<ExprPtr, Error> ParseTuple(bool with_condition) {
    const std::uint32_t line = Line();
    auto first = ParseExpression(with_condition);
    if (!first) {
      return first;
    }
    if (!IsOp(",")) {
      return first;
    }
    auto tuple = Node(ExprKind::kTuple);
    if (!tuple) {
      return tuple;
    }
    (*tuple)->line = line;
    (*tuple)->args.push_back(std::move(*first));
    while (IsOp(",")) {
      ++at_;
      if (Is(Tok::kVarEnd) || Is(Tok::kBlockEnd) || IsOp(")") || IsName("in")) {
        break;
      }
      auto next = ParseExpression(with_condition);
      if (!next) {
        return next;
      }
      (*tuple)->args.push_back(std::move(*next));
    }
    return tuple;
  }

  std::expected<ExprPtr, Error> ParseExpression(bool with_condition) {
    return with_condition ? ParseCondition() : ParseOr();
  }

  std::expected<ExprPtr, Error> ParseCondition() {
    const Nest nest(*this);
    if (!nest.ok()) {
      return Fail(Code::kLimit, "expressions nest deeper than their bound", Line());
    }
    auto value = ParseOr();
    if (!value) {
      return value;
    }
    Chain chain(*this);
    while (IsName("if")) {
      ++at_;
      if (!chain.Grow()) {
        return TooDeep();
      }
      auto cond = ParseOr();
      if (!cond) {
        return cond;
      }
      auto node = Node(ExprKind::kCondition);
      if (!node) {
        return node;
      }
      (*node)->args.push_back(std::move(*value));
      (*node)->args.push_back(std::move(*cond));
      if (IsName("else")) {
        ++at_;
        auto other = ParseCondition();
        if (!other) {
          return other;
        }
        (*node)->args.push_back(std::move(*other));
      } else {
        (*node)->args.push_back(nullptr);
      }
      value = std::move(node);
    }
    return value;
  }

  std::expected<ExprPtr, Error> Join(ExprKind kind, ExprPtr left, ExprPtr right, Op op = Op::kAdd) {
    auto node = Node(kind);
    if (!node) {
      return node;
    }
    (*node)->op = op;
    (*node)->args.push_back(std::move(left));
    (*node)->args.push_back(std::move(right));
    return node;
  }

  std::expected<ExprPtr, Error> ParseOr() {
    auto left = ParseAnd();
    Chain chain(*this);
    while (left && IsName("or")) {
      ++at_;
      if (!chain.Grow()) {
        return TooDeep();
      }
      auto right = ParseAnd();
      if (!right) {
        return right;
      }
      left = Join(ExprKind::kOr, std::move(*left), std::move(*right));
    }
    return left;
  }

  std::expected<ExprPtr, Error> ParseAnd() {
    auto left = ParseNot();
    Chain chain(*this);
    while (left && IsName("and")) {
      ++at_;
      if (!chain.Grow()) {
        return TooDeep();
      }
      auto right = ParseNot();
      if (!right) {
        return right;
      }
      left = Join(ExprKind::kAnd, std::move(*left), std::move(*right));
    }
    return left;
  }

  std::expected<ExprPtr, Error> ParseNot() {
    if (IsName("not")) {
      const Nest nest(*this);
      if (!nest.ok()) {
        return Fail(Code::kLimit, "expressions nest deeper than their bound", Line());
      }
      ++at_;
      auto operand = ParseNot();
      if (!operand) {
        return operand;
      }
      auto node = Node(ExprKind::kNot);
      if (!node) {
        return node;
      }
      (*node)->args.push_back(std::move(*operand));
      return node;
    }
    return ParseCompare();
  }

  std::expected<ExprPtr, Error> ParseCompare() {
    auto first = ParseMath1();
    if (!first) {
      return first;
    }
    ExprPtr node;
    while (true) {
      Op op = Op::kEq;
      std::size_t width = 1;
      if (IsOp("==")) {
        op = Op::kEq;
      } else if (IsOp("!=")) {
        op = Op::kNe;
      } else if (IsOp("<")) {
        op = Op::kLt;
      } else if (IsOp("<=")) {
        op = Op::kLe;
      } else if (IsOp(">")) {
        op = Op::kGt;
      } else if (IsOp(">=")) {
        op = Op::kGe;
      } else if (IsName("in")) {
        op = Op::kIn;
      } else if (IsName("not") && IsName("in", 1)) {
        op = Op::kNotIn;
        width = 2;
      } else {
        break;
      }
      at_ += width;
      auto right = ParseMath1();
      if (!right) {
        return right;
      }
      if (!node) {
        auto n = Node(ExprKind::kCompare);
        if (!n) {
          return n;
        }
        node = std::move(*n);
        node->args.push_back(std::move(*first));
      }
      node->ops.push_back(op);
      node->args.push_back(std::move(*right));
    }
    if (node) {
      return node;
    }
    return first;
  }

  std::expected<ExprPtr, Error> ParseMath1() {
    auto left = ParseConcat();
    Chain chain(*this);
    while (left && (IsOp("+") || IsOp("-"))) {
      const Op op = IsOp("+") ? Op::kAdd : Op::kSub;
      ++at_;
      if (!chain.Grow()) {
        return TooDeep();
      }
      auto right = ParseConcat();
      if (!right) {
        return right;
      }
      left = Join(ExprKind::kBinary, std::move(*left), std::move(*right), op);
    }
    return left;
  }

  std::expected<ExprPtr, Error> ParseConcat() {
    auto left = ParseMath2();
    Chain chain(*this);
    while (left && IsOp("~")) {
      ++at_;
      if (!chain.Grow()) {
        return TooDeep();
      }
      auto right = ParseMath2();
      if (!right) {
        return right;
      }
      left = Join(ExprKind::kBinary, std::move(*left), std::move(*right), Op::kConcat);
    }
    return left;
  }

  std::expected<ExprPtr, Error> ParseMath2() {
    auto left = ParsePow();
    Chain chain(*this);
    while (left && (IsOp("*") || IsOp("/") || IsOp("//") || IsOp("%"))) {
      Op op = Op::kMod;
      if (IsOp("*")) {
        op = Op::kMul;
      } else if (IsOp("/")) {
        op = Op::kDiv;
      } else if (IsOp("//")) {
        op = Op::kFloorDiv;
      }
      ++at_;
      if (!chain.Grow()) {
        return TooDeep();
      }
      auto right = ParsePow();
      if (!right) {
        return right;
      }
      left = Join(ExprKind::kBinary, std::move(*left), std::move(*right), op);
    }
    return left;
  }

  std::expected<ExprPtr, Error> ParsePow() {
    auto left = ParseUnary(true);
    Chain chain(*this);
    while (left && IsOp("**")) {
      ++at_;
      if (!chain.Grow()) {
        return TooDeep();
      }
      auto right = ParseUnary(true);
      if (!right) {
        return right;
      }
      left = Join(ExprKind::kBinary, std::move(*left), std::move(*right), Op::kPow);
    }
    return left;
  }

  std::expected<ExprPtr, Error> ParseUnary(bool with_filter) {
    const Nest nest(*this);
    if (!nest.ok()) {
      return Fail(Code::kLimit, "expressions nest deeper than their bound", Line());
    }
    std::expected<ExprPtr, Error> node = nullptr;
    if (IsOp("-") || IsOp("+")) {
      const ExprKind kind = IsOp("-") ? ExprKind::kNeg : ExprKind::kPos;
      ++at_;
      auto operand = ParseUnary(false);
      if (!operand) {
        return operand;
      }
      node = Node(kind);
      if (!node) {
        return node;
      }
      (*node)->args.push_back(std::move(*operand));
    } else {
      node = ParsePrimary();
      if (!node) {
        return node;
      }
    }
    node = ParsePostfix(std::move(*node));
    if (node && with_filter) {
      node = ParseFilterExpr(std::move(*node));
    }
    return node;
  }

  std::expected<ExprPtr, Error> Literal(Value v) {
    auto node = Node(ExprKind::kLiteral);
    if (node) {
      (*node)->constant = std::move(v);
    }
    return node;
  }

  std::expected<ExprPtr, Error> ParsePrimary() {
    const Token* t = Peek();
    if (t == nullptr) {
      return Fail(Code::kSyntax, "an expression ends early", Line());
    }
    switch (t->type) {
      case Tok::kName: {
        ++at_;
        const std::string& n = t->text;
        if (n == "true" || n == "True") {
          return Literal(Value::Bool(true));
        }
        if (n == "false" || n == "False") {
          return Literal(Value::Bool(false));
        }
        if (n == "none" || n == "None") {
          return Literal(Value::None());
        }
        if (n == "varargs") {
          saw_varargs_ = true;
        } else if (n == "kwargs") {
          saw_kwargs_ = true;
        }
        auto node = Node(ExprKind::kName);
        if (node) {
          (*node)->name = n;
        }
        return node;
      }
      case Tok::kString: {
        std::string text = t->text;
        ++at_;
        while (Is(Tok::kString)) {  // adjacent literals concatenate
          text += t_[at_++].text;
        }
        return Literal(Value::String(MakeStr(nullptr, text, true)));
      }
      case Tok::kInt:
        ++at_;
        return Literal(Value::Int(t->integer));
      case Tok::kFloat:
        ++at_;
        return Literal(Value::Float(t->number));
      case Tok::kOp:
        if (t->text == "(") {
          ++at_;
          if (IsOp(")")) {
            ++at_;
            return Node(ExprKind::kTuple);
          }
          // `(a,)` is a tuple; `(a)` is a.
          auto inner = ParseTuple(true);
          if (!inner) {
            return inner;
          }
          if (auto r = ExpectOp(")", "an unclosed parenthesis"); !r) {
            return std::unexpected(r.error());
          }
          return inner;
        }
        if (t->text == "[") {
          ++at_;
          auto list = Node(ExprKind::kList);
          if (!list) {
            return list;
          }
          while (!IsOp("]")) {
            auto item = ParseExpression(true);
            if (!item) {
              return item;
            }
            (*list)->args.push_back(std::move(*item));
            if (!IsOp(",")) {
              break;
            }
            ++at_;
          }
          if (auto r = ExpectOp("]", "an unclosed list"); !r) {
            return std::unexpected(r.error());
          }
          return list;
        }
        if (t->text == "{") {
          ++at_;
          auto dict = Node(ExprKind::kDict);
          if (!dict) {
            return dict;
          }
          while (!IsOp("}")) {
            auto key = ParseExpression(true);
            if (!key) {
              return key;
            }
            if (auto r = ExpectOp(":", "a mapping entry without a colon"); !r) {
              return std::unexpected(r.error());
            }
            auto value = ParseExpression(true);
            if (!value) {
              return value;
            }
            (*dict)->args.push_back(std::move(*key));
            (*dict)->args.push_back(std::move(*value));
            if (!IsOp(",")) {
              break;
            }
            ++at_;
          }
          if (auto r = ExpectOp("}", "an unclosed mapping"); !r) {
            return std::unexpected(r.error());
          }
          return dict;
        }
        break;
      default:
        break;
    }
    return Fail(Code::kSyntax, "an unexpected token in an expression", t->line);
  }

  std::expected<ExprPtr, Error> ParsePostfix(ExprPtr node) {
    Chain chain(*this);
    while (true) {
      if ((IsOp(".") || IsOp("[") || IsOp("(")) && !chain.Grow()) {
        return TooDeep();
      }
      if (IsOp(".")) {
        ++at_;
        const Token* t = Peek();
        if (t != nullptr && t->type == Tok::kName) {
          auto attr = Node(ExprKind::kAttr);
          if (!attr) {
            return attr;
          }
          (*attr)->name = t->text;
          (*attr)->args.push_back(std::move(node));
          node = std::move(*attr);
          ++at_;
        } else if (t != nullptr && t->type == Tok::kInt) {
          auto item = Node(ExprKind::kItem);
          if (!item) {
            return item;
          }
          auto index = Literal(Value::Int(t->integer));
          if (!index) {
            return index;
          }
          (*item)->args.push_back(std::move(node));
          (*item)->args.push_back(std::move(*index));
          node = std::move(*item);
          ++at_;
        } else {
          return Fail(Code::kSyntax, "an attribute that is not a name", Line());
        }
      } else if (IsOp("[")) {
        ++at_;
        auto sub = ParseSubscript(std::move(node));
        if (!sub) {
          return sub;
        }
        node = std::move(*sub);
      } else if (IsOp("(")) {
        auto call = ParseCall(std::move(node));
        if (!call) {
          return call;
        }
        node = std::move(*call);
      } else {
        return node;
      }
    }
  }

  std::expected<ExprPtr, Error> ParseSubscript(ExprPtr subject) {
    std::array<ExprPtr, 3> parts;
    std::size_t colons = 0;
    for (std::size_t i = 0; i < 3; ++i) {
      if (!IsOp(":") && !IsOp("]")) {
        auto e = ParseExpression(true);
        if (!e) {
          return e;
        }
        parts[i] = std::move(*e);
      }
      if (i < 2 && IsOp(":")) {
        ++at_;
        ++colons;
        continue;
      }
      break;
    }
    if (IsOp(",")) {
      return Fail(Code::kUnsupported, "a tuple subscript", Line());
    }
    if (auto r = ExpectOp("]", "an unclosed subscript"); !r) {
      return std::unexpected(r.error());
    }
    if (colons == 0) {
      if (!parts[0]) {
        return Fail(Code::kSyntax, "an empty subscript", Line());
      }
      auto item = Node(ExprKind::kItem);
      if (!item) {
        return item;
      }
      (*item)->args.push_back(std::move(subject));
      (*item)->args.push_back(std::move(parts[0]));
      return item;
    }
    auto slice = Node(ExprKind::kSlice);
    if (!slice) {
      return slice;
    }
    (*slice)->args.push_back(std::move(subject));
    for (ExprPtr& p : parts) {
      (*slice)->args.push_back(std::move(p));
    }
    return slice;
  }

  // `( args )` into node->args (after any already there) and kwargs.
  std::expected<void, Error> ParseArgs(Expr& node) {
    if (auto r = ExpectOp("(", "a call without arguments"); !r) {
      return r;
    }
    bool keywords = false;
    while (!IsOp(")")) {
      if (IsOp("*") || IsOp("**")) {
        return Fail(Code::kUnsupported, "*args or **kwargs in a call", Line());
      }
      if (Is(Tok::kName) && IsOp("=", 1)) {
        std::string key = t_[at_].text;
        at_ += 2;
        auto value = ParseExpression(true);
        if (!value) {
          return std::unexpected(value.error());
        }
        node.kwargs.emplace_back(std::move(key), std::move(*value));
        keywords = true;
      } else {
        if (keywords) {
          return Fail(Code::kSyntax, "a positional argument after a keyword argument", Line());
        }
        auto value = ParseExpression(true);
        if (!value) {
          return std::unexpected(value.error());
        }
        node.args.push_back(std::move(*value));
      }
      if (!IsOp(",")) {
        break;
      }
      ++at_;
    }
    return ExpectOp(")", "an unclosed argument list");
  }

  std::expected<ExprPtr, Error> ParseCall(ExprPtr callee) {
    auto call = Node(ExprKind::kCall);
    if (!call) {
      return call;
    }
    (*call)->args.push_back(std::move(callee));
    if (auto r = ParseArgs(**call); !r) {
      return std::unexpected(r.error());
    }
    // Python's compiler, which Jinja2's output goes through, refuses a
    // call that names a keyword twice.
    std::unordered_set<std::string_view> keywords;
    for (const auto& [key, unused] : (*call)->kwargs) {
      if (!keywords.insert(key).second) {
        return Fail(Code::kSyntax, "a keyword argument repeated", Line());
      }
    }
    return call;
  }

  // After `|`: name(.name)* [args], applied to `subject` (null in a block
  // set's filter list).
  std::expected<ExprPtr, Error> ParseFilterCall(ExprPtr subject) {
    auto node = Node(ExprKind::kFilter);
    if (!node) {
      return node;
    }
    auto name = ExpectName("a filter without a name");
    if (!name) {
      return std::unexpected(name.error());
    }
    while (IsOp(".") && Peek(1) != nullptr && Peek(1)->type == Tok::kName) {
      *name += "." + t_[at_ + 1].text;
      at_ += 2;
    }
    if (!KnownFilter(*name)) {
      return Fail(Code::kUnsupported, "a filter the subset does not implement", Line());
    }
    (*node)->name = std::move(*name);
    (*node)->args.push_back(std::move(subject));
    if (IsOp("(")) {
      if (auto r = ParseArgs(**node); !r) {
        return std::unexpected(r.error());
      }
    }
    return node;
  }

  std::expected<ExprPtr, Error> ParseFilterExpr(ExprPtr node) {
    Chain chain(*this);
    while (true) {
      if ((IsOp("|") || IsName("is") || IsOp("(")) && !chain.Grow()) {
        return TooDeep();
      }
      if (IsOp("|")) {
        ++at_;
        auto f = ParseFilterCall(std::move(node));
        if (!f) {
          return f;
        }
        node = std::move(*f);
      } else if (IsName("is")) {
        ++at_;
        auto test = Node(ExprKind::kTest);
        if (!test) {
          return test;
        }
        if (IsName("not")) {
          ++at_;
          (*test)->negated = true;
        }
        auto name = ExpectName("a test without a name");
        if (!name) {
          return std::unexpected(name.error());
        }
        if (!KnownTest(*name)) {
          return Fail(Code::kUnsupported, "a test the subset does not implement", Line());
        }
        (*test)->name = std::move(*name);
        (*test)->args.push_back(std::move(node));
        const Token* t = Peek();
        if (IsOp("(")) {
          if (auto r = ParseArgs(**test); !r) {
            return std::unexpected(r.error());
          }
        } else if (t != nullptr &&
                   (t->type == Tok::kName || t->type == Tok::kString || t->type == Tok::kInt ||
                    t->type == Tok::kFloat || IsOp("[") || IsOp("{")) &&
                   !IsName("else") && !IsName("or") && !IsName("and")) {
          if (IsName("is")) {
            return Fail(Code::kSyntax, "chained tests", Line());
          }
          auto arg = ParsePrimary();
          if (!arg) {
            return arg;
          }
          auto post = ParsePostfix(std::move(*arg));
          if (!post) {
            return post;
          }
          (*test)->args.push_back(std::move(*post));
        }
        node = std::move(*test);
      } else if (IsOp("(")) {
        auto call = ParseCall(std::move(node));
        if (!call) {
          return call;
        }
        node = std::move(*call);
      } else {
        return node;
      }
    }
  }

  std::vector<Token> t_;
  const Limits& limits_;
  std::size_t at_ = 0;
  std::size_t depth_ = 0;
  std::size_t nodes_ = 0;
  std::size_t loops_ = 0;
  bool saw_varargs_ = false;
  bool saw_kwargs_ = false;
};

}  // namespace

std::expected<Body, Error> ParseProgram(std::string_view source, const Limits& limits) {
  if (source.size() > limits.max_template_bytes) {
    return Fail(Code::kLimit, "the template is longer than its bound", 0);
  }
  if (const auto bad = base::json::FirstInvalidUtf8(source)) {
    return Fail(Code::kSyntax, "the template is not valid UTF-8", 0);
  }
  // Newlines normalized to \n, and one trailing newline dropped.
  std::string normalized;
  normalized.reserve(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    if (source[i] == '\r') {
      normalized.push_back('\n');
      if (i + 1 < source.size() && source[i + 1] == '\n') {
        ++i;
      }
    } else {
      normalized.push_back(source[i]);
    }
  }
  if (normalized.ends_with('\n')) {
    normalized.pop_back();
  }
  Lexer lexer(normalized, limits);
  auto tokens = lexer.Run();
  if (!tokens) {
    return std::unexpected(tokens.error());
  }
  Parser parser(std::move(*tokens), limits);
  return parser.Run();
}

}  // namespace jitllm::chat::jinja
