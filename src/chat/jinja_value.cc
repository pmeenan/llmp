// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The interpreter's values as Python prints and compares them (jinja.h).

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chat/jinja_internal.h"
#include "chat/pyjson.h"
#include "tokenizer/unicode.h"

namespace jitllm::chat::jinja {
namespace {

std::unexpected<Error> Fail(Code code, std::string_view reason) {
  return std::unexpected(Error{code, reason, 0});
}

void AppendHex(std::uint32_t value, int digits, std::string& out) {
  constexpr std::string_view kHex = "0123456789abcdef";
  for (int i = digits - 1; i >= 0; --i) {
    out.push_back(kHex[(value >> (static_cast<unsigned>(i) * 4U)) & 0xFU]);
  }
}

void AppendInt(std::int64_t i, std::string& out) {
  std::array<char, 24> buffer{};
  const auto [end, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), i);
  out.append(buffer.data(), static_cast<std::size_t>(end - buffer.data()));
}

// Python's repr() of a float (str() is the same).
void AppendFloatRepr(double f, std::string& out) {
  if (std::isnan(f)) {
    out += "nan";
  } else if (std::isinf(f)) {
    out += f < 0 ? "-inf" : "inf";
  } else {
    AppendPythonFloat(f, out);
  }
}

std::unexpected<Error> TooLong() { return Fail(Code::kLimit, "a string longer than its bound"); }

// Python's repr() of a string; false once `out` is longer than `limit`.
bool AppendStrRepr(std::string_view s, std::size_t limit, std::string& out) {
  const bool single = !s.contains('\'') || s.contains('"');
  const char quote = single ? '\'' : '"';
  out.push_back(quote);
  for (std::size_t at = 0; at < s.size();) {
    if (out.size() > limit) {
      return false;
    }
    std::size_t length = 0;
    const char32_t cp = DecodeAt(s, at, length);
    if (cp == static_cast<char32_t>(quote) || cp == U'\\') {
      out.push_back('\\');
      out.push_back(static_cast<char>(cp));
    } else if (cp == U'\t') {
      out += "\\t";
    } else if (cp == U'\n') {
      out += "\\n";
    } else if (cp == U'\r') {
      out += "\\r";
    } else if (cp < 0x20 || cp == 0x7F) {
      out += "\\x";
      AppendHex(cp, 2, out);
    } else if (cp >= 0x80 && !Printable(cp)) {
      if (cp < 0x100) {
        out += "\\x";
        AppendHex(cp, 2, out);
      } else if (cp < 0x10000) {
        out += "\\u";
        AppendHex(cp, 4, out);
      } else {
        out += "\\U";
        AppendHex(cp, 8, out);
      }
    } else {
      out.append(s.substr(at, length));
    }
    at += length;
  }
  out.push_back(quote);
  return out.size() <= limit;
}

// json.dumps of a string; false once `out` is longer than `limit`.
bool AppendJsonString(std::string_view s, bool ensure_ascii, std::size_t limit, std::string& out) {
  out.push_back('"');
  for (std::size_t at = 0; at < s.size();) {
    if (out.size() > limit) {
      return false;
    }
    std::size_t length = 0;
    const char32_t cp = DecodeAt(s, at, length);
    switch (cp) {
      case U'"':
        out += "\\\"";
        break;
      case U'\\':
        out += "\\\\";
        break;
      case U'\n':
        out += "\\n";
        break;
      case U'\r':
        out += "\\r";
        break;
      case U'\t':
        out += "\\t";
        break;
      case U'\b':
        out += "\\b";
        break;
      case U'\f':
        out += "\\f";
        break;
      default:
        if (cp < 0x20 || (ensure_ascii && cp >= 0x7F)) {
          if (cp >= 0x10000) {
            const char32_t v = cp - 0x10000;
            out += "\\u";
            AppendHex(0xD800 + (v >> 10U), 4, out);
            out += "\\u";
            AppendHex(0xDC00 + (v & 0x3FFU), 4, out);
          } else {
            out += "\\u";
            AppendHex(cp, 4, out);
          }
        } else {
          out.append(s.substr(at, length));
        }
    }
    at += length;
  }
  out.push_back('"');
  return out.size() <= limit;
}

// Python's repr(), stopping once `out` is longer than `limit`.
std::expected<void, Error> Repr(const Value& v, std::size_t limit, std::string& out) {
  switch (v.kind()) {
    case Kind::kUndefined:
      out += "Undefined";  // only inside a container: str() of an undefined is empty
      return {};
    case Kind::kNone:
      out += "None";
      return {};
    case Kind::kBool:
      out += v.boolean() ? "True" : "False";
      return {};
    case Kind::kInt:
      AppendInt(v.integer(), out);
      return {};
    case Kind::kBigInt:
      out += v.str().text;
      return {};
    case Kind::kFloat:
      AppendFloatRepr(v.number(), out);
      return {};
    case Kind::kString:
      if (!AppendStrRepr(v.str().text, limit, out)) {
        return TooLong();
      }
      return {};
    case Kind::kList: {
      const List& l = v.list();
      out.push_back(l.tuple ? '(' : '[');
      for (std::size_t i = 0; i < l.items.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        if (auto r = Repr(l.items[i], limit, out); !r) {
          return r;
        }
        if (out.size() > limit) {
          return TooLong();
        }
      }
      if (l.tuple && l.items.size() == 1) {
        out.push_back(',');
      }
      out.push_back(l.tuple ? ')' : ']');
      return {};
    }
    case Kind::kDict: {
      const Dict& d = v.dict();
      out.push_back('{');
      for (std::size_t i = 0; i < d.members.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        if (!AppendStrRepr(d.members[i].first, limit, out)) {
          return TooLong();
        }
        out += ": ";
        if (auto r = Repr(d.members[i].second, limit, out); !r) {
          return r;
        }
        if (out.size() > limit) {
          return TooLong();
        }
      }
      out.push_back('}');
      return {};
    }
    case Kind::kNamespace:
    case Kind::kCallable:
    case Kind::kLoop:
      return Fail(Code::kUnsupported, "a namespace, loop, macro or method printed as text");
  }
  return {};
}

// json.dumps, stopping once `out` is longer than `limit`. `work` counts what
// sorting keys costs (16 a comparison and the bytes it compares), up to
// `budget`.
std::expected<void, Error> Json(const Value& v, const JsonOptions& o, std::size_t level,
                                std::size_t limit, std::string& out, std::uint64_t& work,
                                std::uint64_t budget) {
  const auto newline = [&](std::size_t at_level) {
    out.push_back('\n');
    for (std::size_t i = 0; i < at_level; ++i) {
      if (out.size() + o.indent->size() > limit) {
        return false;
      }
      out += *o.indent;
    }
    return true;
  };
  switch (v.kind()) {
    case Kind::kNone:
      out += "null";
      return {};
    case Kind::kBool:
      out += v.boolean() ? "true" : "false";
      return {};
    case Kind::kInt:
      AppendInt(v.integer(), out);
      return {};
    case Kind::kBigInt:
      out += v.str().text;
      return {};
    case Kind::kFloat:
      AppendPythonFloat(v.number(), out);
      return {};
    case Kind::kString:
      if (!AppendJsonString(v.str().text, o.ensure_ascii, limit, out)) {
        return TooLong();
      }
      return {};
    case Kind::kList: {
      const List& l = v.list();
      if (l.items.empty()) {
        out += "[]";
        return {};
      }
      out.push_back('[');
      for (std::size_t i = 0; i < l.items.size(); ++i) {
        if (i != 0) {
          out += o.item_separator;
        }
        if (o.indent && !newline(level + 1)) {
          return TooLong();
        }
        if (auto r = Json(l.items[i], o, level + 1, limit, out, work, budget); !r) {
          return r;
        }
        if (out.size() > limit) {
          return TooLong();
        }
      }
      if (o.indent && !newline(level)) {
        return TooLong();
      }
      out.push_back(']');
      return {};
    }
    case Kind::kDict: {
      const Dict& d = v.dict();
      if (d.members.empty()) {
        out += "{}";
        return {};
      }
      std::vector<std::size_t> order(d.members.size());
      for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
      }
      if (o.sort_keys) {
        // Each visit sorts again: charged, so a mapping printed many times
        // over costs what its sorts do, not only its bounded text. Past the
        // budget the order is meaningless and the rendering is refused.
        std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) {
          const std::string& x = d.members[a].first;
          const std::string& y = d.members[b].first;
          work += 16 + std::min(x.size(), y.size());
          return work <= budget && x < y;
        });
        if (work > budget) {
          return Fail(Code::kLimit, "the template built or scanned more bytes than its bound");
        }
      }
      out.push_back('{');
      for (std::size_t i = 0; i < order.size(); ++i) {
        if (i != 0) {
          out += o.item_separator;
        }
        if (o.indent && !newline(level + 1)) {
          return TooLong();
        }
        if (!AppendJsonString(d.members[order[i]].first, o.ensure_ascii, limit, out)) {
          return TooLong();
        }
        out += o.key_separator;
        if (auto r = Json(d.members[order[i]].second, o, level + 1, limit, out, work, budget); !r) {
          return r;
        }
        if (out.size() > limit) {
          return TooLong();
        }
      }
      if (o.indent && !newline(level)) {
        return TooLong();
      }
      out.push_back('}');
      return {};
    }
    case Kind::kUndefined:
    case Kind::kNamespace:
    case Kind::kCallable:
    case Kind::kLoop:
      return Fail(Code::kRuntime, "a value that is not JSON serializable passed to tojson");
  }
  return {};
}

int Compare3(std::string_view a, std::string_view b) {
  const int c = a.compare(b);
  if (c == 0) {
    return 0;
  }
  return c < 0 ? -1 : 1;
}

}  // namespace

void Dict::BuildIndex() {
  if (members.size() <= 16) {
    return;
  }
  index.reserve(members.size());
  for (std::size_t i = 0; i < members.size(); ++i) {
    index.emplace(members[i].first, i);
  }
}

const Value* Dict::Find(std::string_view key, std::uint64_t& work) const {
  if (!index.empty()) {
    work += kEntryWork + (2 * key.size());  // hashed, and compared where it matches
    const auto it = index.find(key);
    return it == index.end() ? nullptr : &members[it->second].second;
  }
  return FindMember(members, key, work);
}

Value* Namespace::Find(std::string_view key, std::uint64_t& work) {
  return FindMember(members, key, work);
}

// ------------------------------------------------------- deferred release

namespace {

struct Graveyard {
  std::vector<std::shared_ptr<const Object>> pending;
  bool draining = false;
};

Graveyard& ThreadGraveyard() {
  thread_local Graveyard graveyard;
  return graveyard;
}

}  // namespace

void ReleaseDeferred(std::shared_ptr<const Object> object) {
  if (object != nullptr) {
    ThreadGraveyard().pending.push_back(std::move(object));
  }
}

void ReleaseDeferred(Value& value) {
  switch (value.kind()) {
    case Kind::kList:
    case Kind::kDict:
    case Kind::kNamespace:
    case Kind::kCallable:
    case Kind::kLoop:
      ReleaseDeferred(value.Abandon());
      break;
    default:
      break;  // a string or a scalar holds nothing further
  }
}

void DrainDeferred() {
  Graveyard& g = ThreadGraveyard();
  if (g.draining) {
    return;  // the outer drain releases what this queued
  }
  g.draining = true;
  while (!g.pending.empty()) {
    std::shared_ptr<const Object> last = std::move(g.pending.back());
    g.pending.pop_back();
    last.reset();  // its destructor may queue more
  }
  g.draining = false;
}

List::~List() {
  for (Value& v : items) {
    ReleaseDeferred(v);
  }
  DrainDeferred();
}

Dict::~Dict() {
  index.clear();
  for (auto& [k, v] : members) {
    ReleaseDeferred(v);
  }
  DrainDeferred();
}

Namespace::~Namespace() {
  for (auto& [k, v] : members) {
    ReleaseDeferred(v);
  }
  DrainDeferred();
}

Callable::~Callable() {
  ReleaseDeferred(receiver);
  ReleaseDeferred(std::move(scope));
  DrainDeferred();
}

Loop::~Loop() {
  if (previous) {
    ReleaseDeferred(*previous);
  }
  if (next) {
    ReleaseDeferred(*next);
  }
  DrainDeferred();
}

// ------------------------------------------------------------- searching

namespace {

// The prefix function of `p` read forwards or backwards: for each length
// k, the longest proper border of p's first (or last) k bytes. Needles are
// strings, below 4 GiB.
std::vector<std::uint32_t> Borders(std::string_view p, bool reversed) {
  const std::size_t m = p.size();
  const auto at = [&](std::size_t i) { return reversed ? p[m - 1 - i] : p[i]; };
  std::vector<std::uint32_t> border(m + 1, 0);
  std::uint32_t k = 0;
  for (std::size_t i = 1; i < m; ++i) {
    while (k > 0 && at(i) != at(k)) {
      k = border[k];
    }
    if (at(i) == at(k)) {
      ++k;
    }
    border[i + 1] = k;
  }
  return border;
}

}  // namespace

std::size_t Searcher::Find(std::string_view text, std::size_t from) const {
  const std::size_t m = needle_.size();
  if (m <= 1 || from > text.size() || m > text.size() - from) {
    return text.find(needle_, from);  // empty, one byte or no room: linear already
  }
  if (forward_.empty()) {
    forward_ = Borders(needle_, false);
  }
  std::size_t k = 0;
  for (std::size_t i = from; i < text.size(); ++i) {
    while (k > 0 && text[i] != needle_[k]) {
      k = forward_[k];
    }
    if (text[i] == needle_[k]) {
      ++k;
    }
    if (k == m) {
      return i + 1 - m;
    }
  }
  return std::string_view::npos;
}

std::size_t Searcher::RFind(std::string_view text, std::size_t end) const {
  const std::size_t m = needle_.size();
  end = std::min(end, text.size());
  if (m > end) {
    return std::string_view::npos;
  }
  if (m <= 1) {
    return text.rfind(needle_, end - m);
  }
  if (backward_.empty()) {
    backward_ = Borders(needle_, true);
  }
  // The reversed needle against the text read backwards from `end`.
  std::size_t k = 0;
  for (std::size_t i = end; i-- > 0;) {
    while (k > 0 && text[i] != needle_[m - 1 - k]) {
      k = backward_[k];
    }
    if (text[i] == needle_[m - 1 - k]) {
      ++k;
    }
    if (k == m) {
      return i;
    }
  }
  return std::string_view::npos;
}

std::string Error::ToString() const {
  std::string s;
  switch (code) {
    case Code::kSyntax:
      s = "syntax";
      break;
    case Code::kUnsupported:
      s = "unsupported";
      break;
    case Code::kLimit:
      s = "limit";
      break;
    case Code::kRaised:
      s = "raised";
      break;
    case Code::kRuntime:
      s = "runtime";
      break;
  }
  s += ": ";
  s += reason;
  if (line != 0) {
    s += " (template line ";
    s += std::to_string(line);
    s += ")";
  }
  return s;
}

void StrBuilder::Append(std::string_view text, bool trusted) {
  if (text.empty()) {
    return;
  }
  if (trusted) {
    if (!s_.trusted.empty() &&
        s_.trusted.back().offset + s_.trusted.back().length == s_.text.size()) {
      s_.trusted.back().length += text.size();
    } else {
      s_.trusted.push_back({s_.text.size(), text.size()});
    }
  }
  s_.text += text;
}

void StrBuilder::Append(const Str& s) { Append(s, 0, s.text.size()); }

void StrBuilder::Append(const Str& s, std::size_t offset, std::size_t length) {
  const std::size_t end = offset + length;
  std::size_t at = offset;
  // The first range that ends after `offset` (ranges are sorted): a string
  // may hold very many, and callers take it apart a code point at a time.
  const auto first = std::ranges::partition_point(
      s.trusted, [&](const Range& r) { return r.offset + r.length <= offset; });
  for (const Range& r : std::ranges::subrange(first, s.trusted.end())) {
    const std::size_t r_end = r.offset + r.length;
    if (r.offset >= end) {
      break;
    }
    const std::size_t from = std::max(r.offset, at);
    const std::size_t to = std::min(r_end, end);
    Append(std::string_view(s.text).substr(at, from - at), false);
    Append(std::string_view(s.text).substr(from, to - from), true);
    at = to;
  }
  Append(std::string_view(s.text).substr(at, end - at), false);
}

std::shared_ptr<const Str> StrBuilder::Take(Arena* arena) {
  auto s = std::make_shared<Str>();
  s->text = std::move(s_.text);
  s->trusted = std::move(s_.trusted);
  s->markup = s_.markup;
  s_.text.clear();
  s_.trusted.clear();
  s_.markup = false;
  if (arena != nullptr) {
    if (s->text.size() > arena->limits().max_string_bytes) {
      arena->Fail("a string longer than its bound");
      return nullptr;
    }
    if (!arena->Work(s->text.size()) ||
        !s->Charge(arena, s->text.size() + (s->trusted.size() * sizeof(Range)) + 64)) {
      return nullptr;
    }
  }
  return s;
}

std::shared_ptr<const Str> MakeStr(Arena* arena, std::string_view text, bool trusted) {
  StrBuilder b;
  b.Append(text, trusted);
  return b.Take(arena);
}

std::expected<void, Error> AppendStr(const Value& v, StrBuilder& out, Arena& arena) {
  switch (v.kind()) {
    case Kind::kUndefined:
      return {};
    case Kind::kString:
      out.Append(v.str());
      return {};
    case Kind::kNone:
    case Kind::kBool:
    case Kind::kInt:
    case Kind::kBigInt:
    case Kind::kFloat: {
      std::string text;
      if (auto r = Repr(v, arena.limits().max_string_bytes, text); !r) {
        return r;
      }
      out.Append(text, v.kind() != Kind::kBigInt && v.trusted());
      return {};
    }
    case Kind::kList:
    case Kind::kDict:
    case Kind::kNamespace:
    case Kind::kCallable:
    case Kind::kLoop: {
      std::string text;
      if (auto r = AppendRepr(v, text, arena); !r) {
        return r;
      }
      out.Append(text, false);
      return {};
    }
  }
  return {};
}

std::expected<void, Error> AppendRepr(const Value& v, std::string& out, Arena& arena) {
  const std::size_t before = out.size();
  if (auto r = Repr(v, before + arena.limits().max_string_bytes, out); !r) {
    return r;
  }
  if (!arena.Work(out.size() - before)) {
    return Fail(Code::kLimit, arena.reason());
  }
  return {};
}

std::expected<void, Error> AppendJson(const Value& v, const JsonOptions& options, std::string& out,
                                      Arena& arena) {
  const std::size_t before = out.size();
  std::uint64_t sort_work = 0;
  if (auto r = Json(v, options, 0, before + arena.limits().max_string_bytes, out, sort_work,
                    arena.remaining_work());
      !r) {
    return r;
  }
  if (!arena.Work(out.size() - before + sort_work)) {
    return Fail(Code::kLimit, arena.reason());
  }
  return {};
}

bool Truthy(const Value& v) {
  switch (v.kind()) {
    case Kind::kUndefined:
    case Kind::kNone:
      return false;
    case Kind::kBool:
    case Kind::kInt:
      return v.integer() != 0;
    case Kind::kBigInt:
      return true;  // beyond int64: never zero
    case Kind::kFloat:
      return v.number() != 0.0;
    case Kind::kString:
      return !v.str().text.empty();
    case Kind::kList:
      return !v.list().items.empty();
    case Kind::kDict:
      return !v.dict().members.empty();
    case Kind::kNamespace:
    case Kind::kCallable:
    case Kind::kLoop:
      return true;
  }
  return false;
}

bool Equal(const Value& a, const Value& b, std::uint64_t& work, std::uint64_t budget) {
  work += 16;
  if (work > budget) {
    return false;
  }
  if (a.is_number() && b.is_number()) {
    if (a.kind() == Kind::kBigInt || b.kind() == Kind::kBigInt) {
      return a.kind() == b.kind() && a.str().text == b.str().text;
    }
    if (a.kind() == Kind::kFloat || b.kind() == Kind::kFloat) {
      return a.number() == b.number();
    }
    return a.integer() == b.integer();
  }
  if (a.kind() != b.kind()) {
    return false;
  }
  switch (a.kind()) {
    case Kind::kUndefined:
    case Kind::kNone:
      return true;
    case Kind::kString: {
      const std::string& x = a.str().text;
      const std::string& y = b.str().text;
      if (x.size() != y.size()) {
        return false;
      }
      work += x.size();
      return work <= budget && x == y;
    }
    case Kind::kList: {
      const List& x = a.list();
      const List& y = b.list();
      if (x.tuple != y.tuple || x.items.size() != y.items.size()) {
        return false;
      }
      for (std::size_t i = 0; i < x.items.size(); ++i) {
        if (!Equal(x.items[i], y.items[i], work, budget)) {
          return false;
        }
      }
      return true;
    }
    case Kind::kDict: {
      const Dict& x = a.dict();
      const Dict& y = b.dict();
      if (x.members.size() != y.members.size()) {
        return false;
      }
      for (const auto& [k, v] : x.members) {
        const Value* other = y.Find(k, work);
        if (work > budget) {
          return false;
        }
        if (other == nullptr || !Equal(v, *other, work, budget)) {
          return false;
        }
      }
      return true;
    }
    case Kind::kNamespace:
    case Kind::kCallable:
    case Kind::kLoop:
      return a.object() == b.object();
    default:
      return false;
  }
}

std::expected<bool, Error> Less(const Value& a, const Value& b, std::uint64_t& work,
                                std::uint64_t budget) {
  work += 16;
  if (work > budget) {
    return false;
  }
  if (a.is_number() && b.is_number() && a.kind() != Kind::kBigInt && b.kind() != Kind::kBigInt) {
    if (a.kind() == Kind::kFloat || b.kind() == Kind::kFloat) {
      return a.number() < b.number();
    }
    return a.integer() < b.integer();
  }
  if (a.is_string() && b.is_string()) {
    work += std::min(a.str().text.size(), b.str().text.size());
    if (work > budget) {
      return false;
    }
    return Compare3(a.str().text, b.str().text) < 0;
  }
  if (a.is_list() && b.is_list() && a.list().tuple == b.list().tuple) {
    const List& x = a.list();
    const List& y = b.list();
    for (std::size_t i = 0; i < x.items.size() && i < y.items.size(); ++i) {
      if (!Equal(x.items[i], y.items[i], work, budget)) {
        return Less(x.items[i], y.items[i], work, budget);
      }
    }
    return x.items.size() < y.items.size();
  }
  return Fail(Code::kRuntime, "values that cannot be ordered compared");
}

char32_t DecodeAt(std::string_view s, std::size_t at, std::size_t& length) {
  const auto b0 = static_cast<unsigned char>(s[at]);
  if (b0 < 0x80) {
    length = 1;
    return b0;
  }
  std::size_t n = 0;
  char32_t cp = 0;
  if ((b0 & 0xE0U) == 0xC0U) {
    n = 2;
    cp = b0 & 0x1FU;
  } else if ((b0 & 0xF0U) == 0xE0U) {
    n = 3;
    cp = b0 & 0x0FU;
  } else {
    n = 4;
    cp = b0 & 0x07U;
  }
  n = std::min(n, s.size() - at);
  for (std::size_t i = 1; i < n; ++i) {
    cp = (cp << 6U) | (static_cast<unsigned char>(s[at + i]) & 0x3FU);
  }
  length = n;
  return cp;
}

std::size_t CodePoints(std::string_view s) {
  std::size_t n = 0;
  for (const char c : s) {
    if ((static_cast<unsigned char>(c) & 0xC0U) != 0x80U) {
      ++n;
    }
  }
  return n;
}

std::size_t ByteOffset(std::string_view s, std::size_t index) {
  std::size_t n = 0;
  for (std::size_t at = 0; at < s.size(); ++at) {
    if ((static_cast<unsigned char>(s[at]) & 0xC0U) != 0x80U) {
      if (n == index) {
        return at;
      }
      ++n;
    }
  }
  return s.size();
}

std::size_t SpaceAt(std::string_view s, std::size_t at) {
  if (at >= s.size()) {
    return 0;
  }
  std::size_t length = 0;
  const char32_t cp = DecodeAt(s, at, length);
  // str.isspace(): bidirectional class WS, B or S, or category Zs.
  const bool space = (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 ||
                     cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
                     cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
  return space ? length : 0;
}

bool Printable(char32_t cp) {
  using tokenizer::unicode::Category;
  if (cp == U' ') {
    return true;
  }
  switch (tokenizer::unicode::GetCategory(cp)) {
    case Category::kCc:
    case Category::kCf:
    case Category::kCs:
    case Category::kCo:
    case Category::kCn:
    case Category::kZl:
    case Category::kZp:
    case Category::kZs:
      return false;
    default:
      return true;
  }
}

}  // namespace jitllm::chat::jinja
