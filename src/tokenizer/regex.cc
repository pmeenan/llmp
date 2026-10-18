// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/regex.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>

#include "base/work_pulse.h"
#include "tokenizer/unicode.h"

namespace llmp::tokenizer {
namespace {
constexpr auto kNoMatch = std::numeric_limits<std::size_t>::max();
constexpr auto kUnlimited = std::numeric_limits<std::uint32_t>::max();
std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t at = kNoItem) {
  return std::unexpected(Error{rule, reason, at});
}
char32_t LowerAscii(char32_t cp) { return cp >= U'A' && cp <= U'Z' ? cp + 32 : cp; }
std::optional<std::uint32_t> CategoryMask(std::u32string_view name) {
  static constexpr std::array<std::u32string_view, 30> names = {
      U"Cn", U"Lu", U"Ll", U"Lt", U"Lm", U"Lo", U"Mn", U"Mc", U"Me", U"Nd",
      U"Nl", U"No", U"Pc", U"Pd", U"Ps", U"Pe", U"Pi", U"Pf", U"Po", U"Sm",
      U"Sc", U"Sk", U"So", U"Zs", U"Zl", U"Zp", U"Cc", U"Cf", U"Cs", U"Co"};
  std::uint32_t mask = 0;
  for (std::size_t i = 0; i < names.size(); ++i)
    if (names[i] == name || (name.size() == 1 && names[i].front() == name.front()))
      mask |= std::uint32_t{1} << i;
  if (mask == 0) return std::nullopt;
  return mask;
}
}  // namespace

std::expected<void, Error> RegexBudget::Pay(std::uint64_t amount) {
  if (amount > remaining) return Fail(Rule::kBounds, "regex transition allowance exhausted", spent);
  remaining -= amount;
  const auto previous = spent;
  spent += amount;
  if ((previous >> 12U) != (spent >> 12U) && !base::Pulse())
    return Fail(Rule::kCancelled, "regex was asked to stop", spent);
  return {};
}

std::uint64_t Regex::ConstructionBytes(std::size_t n) {
  // UTF-8 decode, nodes, explicit parse/compile frames, classes and terms,
  // program and growth overlap. Each syntax character creates at most two
  // nodes, each emitting at most eight instructions. Compact counts never
  // expand the program. Class algebra adds at most three constant-size class
  // records per source character and source-sized parse frames; 2048n also
  // covers these vectors' simultaneous old/new capacities.
  // All code, node and literal offsets are uint32. This worst-case expansion
  // must fit that representation as well as the allocation byte count.
  const auto limit = std::numeric_limits<std::uint64_t>::max();
  return n > (UINT32_MAX - 9U) / 16U || n > (limit - 2048) / 2048
             ? limit
             : 2048 + static_cast<std::uint64_t>(n) * 2048;
}

std::expected<Regex, Error> Regex::Compile(std::string_view expression, RegexDialect dialect) {
  if (ConstructionBytes(expression.size()) == UINT64_MAX)
    return Fail(Rule::kBounds, "regex construction representation overflows", expression.size());
  if (!base::Pulse()) return Fail(Rule::kCancelled, "regex compilation was asked to stop");
  std::vector<char32_t> input;
  if (auto decoded = unicode::DecodeUtf8(expression, input); !decoded)
    return std::unexpected(decoded.error());
  Regex out;
  enum class Kind : std::uint8_t {
    kEmpty,
    kClass,
    kLiteral,
    kGroup,
    kAheadClass,
    kConcat,
    kAlt,
    kRepeat,
    kBegin,
    kEnd,
    kSpace
  };
  struct Node {
    Kind kind = Kind::kEmpty;
    std::uint32_t a = 0, b = 0, min = 0, max = 0;
    bool lazy = false, possessive = false, caseless = false;
  };
  struct Frame {
    std::uint32_t sequence = 0, alternatives = 0;
    bool has_sequence = false, has_alternatives = false, caseless = false, lookahead = false;
  };
  std::vector<Node> nodes;
  nodes.reserve(2 * input.size() + 1);
  const auto node = [&](Node value) {
    nodes.push_back(value);
    return static_cast<std::uint32_t>(nodes.size() - 1);
  };
  std::vector<Frame> frames(1);
  frames.reserve(input.size() + 1);
  std::size_t at = 0;
  const auto error = [&] { return Fail(Rule::kFormat, "malformed regex", at); };
  // Escape parsing is shared by literals and class terms; no recursive calls.
  const auto escape = [&](bool caseless) -> std::expected<Term, Error> {
    if (at >= input.size()) return error();
    const char32_t c = input[at++];
    if (c == U's' || c == U'S') return Term{0, 0, 0, true, c == U'S'};
    if (c == U'p' || c == U'P') {
      if (caseless) return Fail(Rule::kUnsupported, "caseless Unicode category", at);
      if (at == input.size() || input[at++] != U'{') return error();
      const auto begin = at;
      std::size_t property_work = 0;
      while (at < input.size() && input[at] != U'}') {
        if ((++property_work & 255U) == 0 && !base::Pulse())
          return Fail(Rule::kCancelled, "regex compilation was asked to stop", at);
        ++at;
      }
      if (at == input.size()) return error();
      auto mask = CategoryMask(std::u32string_view(input.data() + begin, at++ - begin));
      if (!mask) return Fail(Rule::kUnsupported, "regex Unicode property", begin);
      return Term{0, 0, *mask, false, c == U'P'};
    }
    char32_t literal = c;
    if (c == U'n')
      literal = U'\n';
    else if (c == U'r')
      literal = U'\r';
    else if (c == U't')
      literal = U'\t';
    else if (c == U'f')
      literal = U'\f';
    else if (c == U'v')
      literal = U'\v';
    else if (c >= U'0' && c <= U'9')
      return Fail(Rule::kUnsupported, "regex backreference", at);
    else if ((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z'))
      return Fail(Rule::kUnsupported, "regex escape", at);
    return Term{literal, literal};
  };
  const auto finish_sequence = [&](Frame& frame) {
    auto result = frame.has_sequence ? frame.sequence : node({Kind::kEmpty});
    if (frame.has_alternatives) result = node({Kind::kAlt, frame.alternatives, result});
    return result;
  };
  const auto append = [&](std::uint32_t value) {
    auto& f = frames.back();
    if (f.has_sequence && nodes[value].kind == Kind::kLiteral) {
      const auto last = nodes[f.sequence].kind == Kind::kConcat ? nodes[f.sequence].b : f.sequence;
      auto& previous = nodes[last];
      const auto& current = nodes[value];
      if (previous.kind == Kind::kLiteral && previous.caseless == current.caseless &&
          previous.a + previous.b == current.a) {
        previous.b += current.b;
        return;
      }
    }
    f.sequence = f.has_sequence ? node({Kind::kConcat, f.sequence, value}) : value;
    f.has_sequence = true;
  };
  const auto number = [&]() -> std::expected<std::uint32_t, Error> {
    const auto begin = at;
    std::uint32_t value = 0;
    while (at < input.size() && input[at] >= U'0' && input[at] <= U'9') {
      const auto digit = static_cast<std::uint32_t>(input[at++] - U'0');
      if (value > (kUnlimited - digit) / 10)
        return Fail(Rule::kBounds, "regex repeat count overflow", begin);
      value = value * 10 + digit;
    }
    if (at == begin) return error();
    return value;
  };
  const auto quantified = [&](std::uint32_t value) -> std::expected<std::uint32_t, Error> {
    if (at == input.size()) return value;
    const char32_t c = input[at];
    if (c != U'?' && c != U'*' && c != U'+' && c != U'{') return value;
    ++at;
    Node repeat{Kind::kRepeat, value};
    repeat.min = c == U'+' ? 1 : 0;
    repeat.max = c == U'?' ? 1 : kUnlimited;
    if (c == U'{') {
      auto min = number();
      if (!min) return std::unexpected(min.error());
      repeat.min = repeat.max = *min;
      if (at < input.size() && input[at] == U',') {
        ++at;
        repeat.max = kUnlimited;
        if (at < input.size() && input[at] != U'}') {
          auto max = number();
          if (!max) return std::unexpected(max.error());
          repeat.max = *max;
        }
      }
      if (at == input.size() || input[at++] != U'}' || repeat.min > repeat.max) return error();
    }
    if (at < input.size() && (input[at] == U'?' || input[at] == U'+')) {
      repeat.lazy = input[at] == U'?';
      repeat.possessive = input[at++] == U'+';
    }
    return node(repeat);
  };
  std::size_t parse_work = 0;
  while (at < input.size()) {
    if ((++parse_work & 255U) == 0 && !base::Pulse())
      return Fail(Rule::kCancelled, "regex compilation was asked to stop", at);
    const char32_t c = input[at++];
    if (c == U'|') {
      auto& f = frames.back();
      f.alternatives = finish_sequence(f);
      f.has_alternatives = true;
      f.has_sequence = false;
      continue;
    }
    std::uint32_t value = 0;
    if (c == U')') {
      if (frames.size() == 1) return error();
      const auto result = finish_sequence(frames.back());
      if (frames.back().lookahead) {
        if (nodes[result].kind != Kind::kClass)
          return Fail(Rule::kUnsupported, "regex lookahead other than one class", at);
        value = node({Kind::kAheadClass, nodes[result].a});
      } else
        value = node({Kind::kGroup, result});
      frames.pop_back();
    } else if (c == U'(') {
      // This assertion is a position predicate; it never recursively runs a VM.
      constexpr std::u32string_view assertion = U"?!\\S)";
      if (std::u32string_view(input.data() + at, input.size() - at).starts_with(assertion)) {
        at += assertion.size();
        value = node({Kind::kSpace});
      } else {
        bool caseless = frames.back().caseless;
        bool lookahead = false;
        if (at < input.size() && input[at] == U'?') {
          ++at;
          if (at < input.size() && input[at] == U'=') {
            lookahead = true;
            ++at;
          } else {
            if (at < input.size() && input[at] == U'i') {
              caseless = true;
              ++at;
            }
            if (at == input.size() || input[at++] != U':')
              return Fail(Rule::kUnsupported, "regex group or assertion", at);
          }
        }
        frames.push_back(Frame{.caseless = caseless, .lookahead = lookahead});
        continue;
      }
    } else if (c == U'^' || c == U'$') {
      value = node({c == U'^' ? Kind::kBegin : Kind::kEnd,
                    static_cast<std::uint32_t>(dialect == RegexDialect::kHuggingFace)});
    } else if (c == U'\\' && at < input.size() && (input[at] == U'A' || input[at] == U'z') &&
               dialect == RegexDialect::kHuggingFace) {
      value = node({input[at++] == U'A' ? Kind::kBegin : Kind::kEnd});
    } else {
      if (c == U'*' || c == U'+' || c == U'?' || c == U'{' || c == U'}' || c == U']')
        return error();
      const bool caseless = frames.back().caseless;
      const auto class_node = [&](Class cls) {
        out.classes_.push_back(cls);
        return static_cast<std::uint32_t>(out.classes_.size() - 1);
      };
      std::uint32_t cls = kUnlimited;
      if (c == U'[') {
        struct ClassParseFrame {
          std::uint32_t begin = 0, joined = kUnlimited, intersection = kUnlimited;
          bool inverse = false;
        };
        std::vector<ClassParseFrame> pending;
        const auto open = [&] {
          const bool inverse = at < input.size() && input[at] == U'^';
          if (inverse) ++at;
          pending.push_back(
              {static_cast<std::uint32_t>(out.terms_.size()), kUnlimited, kUnlimited, inverse});
        };
        const auto join = [&](ClassParseFrame& f, std::uint32_t child) {
          f.joined =
              f.joined == kUnlimited ? child : class_node({f.joined, child, ClassOp::kUnion});
        };
        const auto flush = [&](ClassParseFrame& f) {
          const auto count = static_cast<std::uint32_t>(out.terms_.size()) - f.begin;
          if (count) join(f, class_node({f.begin, count, ClassOp::kTerms, false, caseless}));
          f.begin = static_cast<std::uint32_t>(out.terms_.size());
        };
        open();
        while (!pending.empty()) {
          if ((++parse_work & 255U) == 0 && !base::Pulse())
            return Fail(Rule::kCancelled, "regex compilation was asked to stop", at);
          if (at == input.size()) return error();
          auto& f = pending.back();
          if (input[at] == U']') {
            ++at;
            flush(f);
            if (f.joined == kUnlimited && f.intersection == kUnlimited)
              return Fail(Rule::kUnsupported, "empty regex class operand", at);
            if (f.joined == kUnlimited)
              f.joined = class_node({static_cast<std::uint32_t>(out.terms_.size()), 0});
            auto result = f.intersection == kUnlimited
                              ? f.joined
                              : class_node({f.intersection, f.joined, ClassOp::kIntersection});
            if (f.inverse) {
              auto inverted = out.classes_[result];
              inverted.inverse = !inverted.inverse;
              result = class_node(inverted);
            }
            pending.pop_back();
            if (pending.empty())
              cls = result;
            else {
              auto& parent = pending.back();
              parent.begin = static_cast<std::uint32_t>(out.terms_.size());
              join(parent, result);
            }
          } else if (dialect == RegexDialect::kHuggingFace && input[at] == U'[') {
            if (at + 1 < input.size() && input[at + 1] == U':')
              return Fail(Rule::kUnsupported, "regex POSIX character class", at);
            flush(f);
            ++at;
            open();
          } else if (dialect == RegexDialect::kHuggingFace && at + 1 < input.size() &&
                     input[at] == U'&' && input[at + 1] == U'&') {
            flush(f);
            if (f.joined == kUnlimited)
              f.joined = class_node({static_cast<std::uint32_t>(out.terms_.size()), 0});
            f.intersection = f.intersection == kUnlimited
                                 ? f.joined
                                 : class_node({f.intersection, f.joined, ClassOp::kIntersection});
            f.joined = kUnlimited;
            at += 2;
          } else {
            const auto cp = input[at++];
            auto term = cp == U'\\' ? escape(caseless) : std::expected<Term, Error>(Term{cp, cp});
            if (!term) return std::unexpected(term.error());
            if (at + 1 < input.size() && input[at] == U'-' && input[at + 1] != U']') {
              ++at;
              const auto end = input[at++];
              auto last =
                  end == U'\\' ? escape(caseless) : std::expected<Term, Error>(Term{end, end});
              if (!last) return std::unexpected(last.error());
              if (term->categories || term->whitespace || last->categories || last->whitespace ||
                  term->first > last->first)
                return error();
              term->last = last->first;
            }
            if (caseless && term->last > 127)
              return Fail(Rule::kUnsupported, "non-ASCII caseless regex class", at);
            out.terms_.push_back(*term);
          }
        }
      } else if (c == U'.') {
        const auto begin = static_cast<std::uint32_t>(out.terms_.size());
        out.terms_.push_back(Term{U'\n', U'\n'});
        if (dialect == RegexDialect::kGguf)
          for (const auto cp : {U'\r', U'\u2028', U'\u2029'}) out.terms_.push_back(Term{cp, cp});
        cls = class_node({begin, static_cast<std::uint32_t>(out.terms_.size()) - begin,
                          ClassOp::kTerms, true, caseless});
      } else {
        auto term = c == U'\\' ? escape(caseless) : std::expected<Term, Error>(Term{c, c});
        if (!term) return std::unexpected(term.error());
        if (!term->categories && !term->whitespace) {
          const auto offset = static_cast<std::uint32_t>(out.literals_.size());
          out.literals_.push_back(term->first);
          value = node({.kind = Kind::kLiteral, .a = offset, .b = 1, .caseless = caseless});
        } else {
          const auto begin = static_cast<std::uint32_t>(out.terms_.size());
          out.terms_.push_back(*term);
          cls = class_node({begin, 1, ClassOp::kTerms, false, caseless});
        }
      }
      if (cls != kUnlimited) value = node({Kind::kClass, cls});
    }
    auto result = quantified(value);
    if (!result) return std::unexpected(result.error());
    append(*result);
  }
  if (frames.size() != 1) return error();
  const auto root = finish_sequence(frames.back());
  // Children precede their groups. Resolve group boundaries once rather
  // than repeatedly walking an arbitrarily deep chain while compiling.
  std::vector<std::uint32_t> ungrouped;
  ungrouped.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if ((i & 255U) == 0 && !base::Pulse())
      return Fail(Rule::kCancelled, "regex compilation was asked to stop");
    ungrouped.push_back(nodes[i].kind == Kind::kGroup ? ungrouped[nodes[i].a]
                                                      : static_cast<std::uint32_t>(i));
  }
  const auto ungroup = [&](std::uint32_t index) { return ungrouped[index]; };
  for (auto& n : nodes) {
    if (n.kind != Kind::kConcat) continue;
    const auto& left = nodes[ungroup(n.a)];
    const auto& right = nodes[ungroup(n.b)];
    if (left.kind == Kind::kAheadClass && right.kind == Kind::kClass) {
      out.classes_.push_back({left.a, right.a, ClassOp::kIntersection});
      n.kind = Kind::kClass;
      n.a = static_cast<std::uint32_t>(out.classes_.size() - 1);
    }
  }
  // Iterative compiler. Patch tasks run after their children are emitted.
  struct Task {
    std::uint32_t node = 0, phase = 0, pc = 0, slot = 0;
  };
  std::vector<Task> tasks{{root}};
  out.code_.reserve(8 * nodes.size() + 1);
  const auto emit = [&](Instruction instruction) {
    out.code_.push_back(instruction);
    return static_cast<std::uint32_t>(out.code_.size() - 1);
  };
  std::size_t compile_work = 0;
  while (!tasks.empty()) {
    if ((++compile_work & 255U) == 0 && !base::Pulse())
      return Fail(Rule::kCancelled, "regex compilation was asked to stop");
    const auto task = tasks.back();
    tasks.pop_back();
    const auto& n = nodes[task.node];
    if (task.phase == 1) {
      const auto jump = emit({Op::kJump});
      out.code_[task.pc].b = static_cast<std::uint32_t>(out.code_.size());
      tasks.push_back({task.node, 2, jump});
      tasks.push_back({n.b});
    } else if (task.phase == 2) {
      out.code_[task.pc].a = static_cast<std::uint32_t>(out.code_.size());
    } else if (task.phase == 3) {
      emit({Op::kNext, task.slot, task.pc});
      out.code_[task.pc].c = static_cast<std::uint32_t>(out.code_.size());
      emit({Op::kExit, task.slot, static_cast<std::uint32_t>(n.possessive)});
    } else
      switch (n.kind) {
        case Kind::kEmpty:
          break;
        case Kind::kClass:
          emit({Op::kClass, n.a});
          break;
        case Kind::kGroup:
          tasks.push_back({n.a});
          break;
        case Kind::kAheadClass:
          emit({Op::kAheadClass, n.a});
          break;
        case Kind::kLiteral: {
          auto begin = n.a, length = n.b;
          if (n.caseless) {
            begin = static_cast<std::uint32_t>(out.literals_.size());
            for (std::size_t i = n.a; i < n.a + n.b; ++i) {
              if ((i & 255U) == 0 && !base::Pulse())
                return Fail(Rule::kCancelled, "regex compilation was asked to stop");
              const auto folded = unicode::CaseFold(out.literals_[i]);
              out.literals_.insert(out.literals_.end(), folded.view().begin(), folded.view().end());
            }
            length = static_cast<std::uint32_t>(out.literals_.size()) - begin;
          }
          emit({Op::kLiteral, begin, length, static_cast<std::uint32_t>(n.caseless)});
          break;
        }
        case Kind::kBegin:
          emit({Op::kBegin, n.a});
          break;
        case Kind::kEnd:
          emit({Op::kEnd, n.a});
          break;
        case Kind::kSpace:
          emit({Op::kNotNonSpace});
          break;
        case Kind::kConcat:
          tasks.push_back({n.b});
          tasks.push_back({n.a});
          break;
        case Kind::kAlt: {
          const auto pc = emit({Op::kSplit, static_cast<std::uint32_t>(out.code_.size() + 1)});
          tasks.push_back({task.node, 1, pc});
          tasks.push_back({n.a});
          break;
        }
        case Kind::kRepeat: {
          const auto& child = nodes[ungroup(n.a)];
          auto scan =
              child.kind == Kind::kClass ? std::optional<std::uint32_t>(child.a) : std::nullopt;
          if (child.kind == Kind::kLiteral && child.b == 1) {
            const auto cp = out.literals_[child.a];
            const auto folded = unicode::CaseFold(cp);
            // One literal atom must consume complete original code points:
            // s{2} cannot join the two halves of a sharp-s full fold.
            if (!child.caseless || folded.length == 1) {
              const auto begin = static_cast<std::uint32_t>(out.terms_.size());
              out.terms_.push_back(Term{child.caseless ? folded.code_points[0] : cp,
                                        child.caseless ? folded.code_points[0] : cp});
              scan = static_cast<std::uint32_t>(out.classes_.size());
              out.classes_.push_back(Class{begin, 1, ClassOp::kTerms, false, child.caseless});
            }
          }
          if (scan) {
            emit({Op::kScan, *scan, n.min, n.max,
                  static_cast<std::uint32_t>(n.lazy) |
                      (static_cast<std::uint32_t>(n.possessive) << 1U)});
            break;
          }
          const auto slot = out.repeats_++;
          emit({Op::kEnter, slot, n.min, n.max});
          const auto pc = emit({Op::kRepeat, slot, static_cast<std::uint32_t>(n.lazy)});
          tasks.push_back({task.node, 3, pc, slot});
          tasks.push_back({n.a});
          break;
        }
      }
  }
  emit({Op::kAccept});
  return out;
}

std::uint64_t Regex::WorkingBytes(std::size_t n) const {
  // Current registers, choices, flattened register snapshots and one class
  // evaluation stack (8 bytes/record within the 64-byte allowance). Reserve
  // explicitly so vector growth cannot transiently double this allowance.
  const auto limit = std::numeric_limits<std::uint64_t>::max();
  const auto fixed = std::uint64_t{64} * (code_.size() + repeats_ + classes_.size() + 2);
  return n > (limit - fixed) / 64 ? limit : 64 * n + fixed;
}

std::expected<bool, Error> Regex::TestClass(std::uint32_t index, char32_t cp, RegexBudget& budget,
                                            std::vector<ClassFrame>& stack) const {
  stack.clear();
  stack.push_back({index});
  bool result = false;
  while (!stack.empty()) {
    if (auto paid = budget.Pay(); !paid) return std::unexpected(paid.error());
    auto& frame = stack.back();
    const auto& cls = classes_[frame.index];
    if (cls.op != ClassOp::kTerms) {
      if (frame.phase == 0) {
        frame.phase = 1;
        stack.push_back({cls.a});
      } else if (frame.phase == 1) {
        frame.left = result;
        if ((cls.op == ClassOp::kIntersection && !result) ||
            (cls.op == ClassOp::kUnion && result)) {
          result = result != cls.inverse;
          stack.pop_back();
        } else {
          frame.phase = 2;
          stack.push_back({cls.b});
        }
      } else {
        result = (cls.op == ClassOp::kIntersection ? frame.left && result : frame.left || result) !=
                 cls.inverse;
        stack.pop_back();
      }
      continue;
    }
    auto folded = unicode::CaseMapping{{cp, 0, 0}, 1};
    if (cls.caseless) {
      if (auto paid = budget.Pay(16); !paid) return std::unexpected(paid.error());
      folded = unicode::CaseFold(cp);
    }
    bool matched = false;
    for (std::size_t i = cls.a; i < cls.a + cls.b; ++i) {
      if (auto paid = budget.Pay(); !paid) return std::unexpected(paid.error());
      const auto& term = terms_[i];
      bool hit = false;
      if (term.categories)
        hit = (term.categories &
               (std::uint32_t{1} << static_cast<unsigned>(unicode::GetCategory(cp)))) != 0;
      else if (term.whitespace)
        hit = unicode::IsWhiteSpace(cp);
      else if (!cls.caseless)
        hit = cp >= term.first && cp <= term.last;
      else if (folded.length == 1) {
        const auto value = folded.code_points[0];
        hit = value >= term.first && value <= term.last;
        if (!hit && value <= 127) {
          const auto alternate = value >= U'a' && value <= U'z' ? value - 32 : LowerAscii(value);
          hit = alternate >= term.first && alternate <= term.last;
        }
      }
      matched |= hit != term.inverse;
    }
    result = matched != cls.inverse;
    stack.pop_back();
  }
  return result;
}

std::expected<std::size_t, Error> Regex::Match(std::span<const char32_t> text, std::size_t start,
                                               RegexBudget& budget) const {
  struct Counter {
    std::uint32_t count = 0, previous = 0, mark = 0, stalled = 0;
  };
  struct Choice {
    std::uint32_t pc = 0, position = 0, limit = 0;
    std::int32_t step = 0;
  };
  const auto choice_bytes = sizeof(Choice) + repeats_ * sizeof(Counter);
  const auto capacity =
      static_cast<std::size_t>((32 * (text.size() + code_.size() + 1)) / choice_bytes);
  if (auto paid = budget.Pay(repeats_ + classes_.size() + 1); !paid)
    return std::unexpected(paid.error());
  std::vector<ClassFrame> class_stack;
  class_stack.reserve(classes_.size());
  std::vector<Counter> counters(repeats_);
  std::vector<Choice> choices;
  choices.reserve(std::min(capacity, std::size_t{64}));
  std::vector<Counter> snapshots;
  snapshots.reserve(std::min(capacity, std::size_t{64}) * repeats_);
  const auto push = [&](std::uint32_t target, std::size_t position, std::uint32_t limit = 0,
                        std::int32_t step = 0) -> std::expected<void, Error> {
    if (choices.size() == capacity)
      return Fail(Rule::kBounds, "regex choice storage exhausted", choices.size());
    if (auto paid = budget.Pay(repeats_ + 1); !paid) return paid;
    if (choices.size() == choices.capacity()) {
      if (auto paid = budget.Pay(choices.size() * (repeats_ + 1)); !paid) return paid;
      const auto grown = std::min(capacity, std::max(std::size_t{1}, choices.size() * 2));
      choices.reserve(grown);
      snapshots.reserve(grown * repeats_);
    }
    choices.push_back({target, static_cast<std::uint32_t>(position), limit, step});
    snapshots.insert(snapshots.end(), counters.begin(), counters.end());
    return {};
  };
  std::uint32_t pc = 0;
  std::size_t pos = start;
  while (true) {
    if (auto paid = budget.Pay(); !paid) return std::unexpected(paid.error());
    const auto ins = code_[pc];
    bool matched = true;
    switch (ins.op) {
      case Op::kLiteral: {
        std::size_t part = 0;
        while (part < ins.b && pos < text.size()) {
          if (auto paid = budget.Pay(ins.c ? 16 : 1); !paid) return std::unexpected(paid.error());
          const auto folded =
              ins.c ? unicode::CaseFold(text[pos]) : unicode::CaseMapping{{text[pos], 0, 0}, 1};
          if (folded.length > ins.b - part) {
            matched = false;
            break;
          }
          for (const auto cp : folded.view()) {
            if (auto paid = budget.Pay(); !paid) return std::unexpected(paid.error());
            if (cp != literals_[ins.a + part++]) {
              matched = false;
              break;
            }
          }
          if (!matched) break;
          ++pos;
        }
        matched = matched && part == ins.b;
        if (matched) ++pc;
        break;
      }
      case Op::kAheadClass: {
        matched = false;
        if (pos < text.size()) {
          auto hit = TestClass(ins.a, text[pos], budget, class_stack);
          if (!hit) return std::unexpected(hit.error());
          matched = *hit;
        }
        if (matched) ++pc;
        break;
      }
      case Op::kClass: {
        matched = false;
        if (pos < text.size()) {
          auto hit = TestClass(ins.a, text[pos], budget, class_stack);
          if (!hit) return std::unexpected(hit.error());
          matched = *hit;
        }
        if (matched) {
          ++pos;
          ++pc;
        }
        break;
      }
      case Op::kScan: {
        const auto begin = pos;
        while (pos < text.size() && pos - begin < ins.c) {
          if (auto paid = budget.Pay(); !paid) return std::unexpected(paid.error());
          auto hit = TestClass(ins.a, text[pos], budget, class_stack);
          if (!hit) return std::unexpected(hit.error());
          if (!*hit) break;
          ++pos;
        }
        matched = pos - begin >= ins.b;
        if (!matched) break;
        const auto minimum = begin + ins.b;
        const auto maximum = pos;
        const bool lazy = (ins.d & 1U) != 0;
        const bool possessive = (ins.d & 2U) != 0;
        pos = lazy ? minimum : maximum;
        ++pc;
        if (!possessive && maximum > minimum) {
          const auto next = lazy ? minimum + 1 : maximum - 1;
          if (auto saved = push(pc, next, static_cast<std::uint32_t>(lazy ? maximum : minimum),
                                lazy ? 1 : -1);
              !saved)
            return std::unexpected(saved.error());
        }
        break;
      }
      case Op::kSplit:
        if (auto saved = push(ins.b, pos); !saved) return std::unexpected(saved.error());
        pc = ins.a;
        break;
      case Op::kJump:
        pc = ins.a;
        break;
      case Op::kEnter:
        counters[ins.a] = {0, static_cast<std::uint32_t>(pos),
                           static_cast<std::uint32_t>(choices.size()), 0};
        ++pc;
        break;
      case Op::kRepeat: {
        auto& c = counters[ins.a];
        const auto enter = code_[pc - 1];
        if (c.count < enter.b) {
          ++pc;
          break;
        }
        if (c.count >= enter.c || c.stalled != 0) {
          pc = ins.c;
          break;
        }
        const auto body = pc + 1;
        if (auto saved = push(ins.b ? body : ins.c, pos); !saved)
          return std::unexpected(saved.error());
        pc = ins.b ? ins.c : body;
        break;
      }
      case Op::kNext: {
        auto& c = counters[ins.a];
        if (c.count == kUnlimited) return Fail(Rule::kBounds, "regex repeat counter overflow");
        ++c.count;
        c.stalled = c.previous == pos;
        c.previous = static_cast<std::uint32_t>(pos);
        pc = ins.b;
        break;
      }
      case Op::kExit:
        if (ins.b) {
          const auto mark = counters[ins.a].mark;
          if (mark > choices.size()) return Fail(Rule::kBounds, "regex atomic checkpoint");
          if (auto paid = budget.Pay(choices.size() - mark + 1); !paid)
            return std::unexpected(paid.error());
          choices.resize(mark);
          snapshots.resize(mark * repeats_);
        }
        ++pc;
        break;
      case Op::kBegin:
        matched =
            pos == 0 || (ins.a != 0 && pos > 0 && pos < text.size() && text[pos - 1] == U'\n');
        if (matched) ++pc;
        break;
      case Op::kEnd:
        matched = pos == text.size() || (ins.a != 0 && pos < text.size() && text[pos] == U'\n');
        if (matched) ++pc;
        break;
      case Op::kNotNonSpace:
        matched = pos == text.size() || unicode::IsWhiteSpace(text[pos]);
        if (matched) ++pc;
        break;
      case Op::kAccept:
        return pos;
    }
    if (!matched) {
      if (choices.empty()) return kNoMatch;
      if (auto paid = budget.Pay(repeats_ + 1); !paid) return std::unexpected(paid.error());
      const auto choice = choices.back();
      choices.pop_back();
      const auto offset = choices.size() * repeats_;
      std::copy(snapshots.begin() + static_cast<std::ptrdiff_t>(offset), snapshots.end(),
                counters.begin());
      snapshots.resize(offset);
      pc = choice.pc;
      pos = choice.position;
      if (choice.step != 0 && choice.position != choice.limit) {
        const auto next =
            static_cast<std::uint32_t>(static_cast<std::int64_t>(choice.position) + choice.step);
        if (auto saved = push(choice.pc, next, choice.limit, choice.step); !saved)
          return std::unexpected(saved.error());
      }
    }
  }
}

std::expected<void, Error> Regex::Split(std::span<const char32_t> text,
                                        std::vector<std::uint32_t>& lengths,
                                        RegexBudget& budget) const {
  if (!base::Pulse()) return Fail(Rule::kCancelled, "regex was asked to stop");
  if (text.size() >= kUnlimited) return Fail(Rule::kBounds, "regex input code point limit");
  std::size_t gap = 0, pos = 0;
  while (pos <= text.size()) {
    if (auto paid = budget.Pay(); !paid) return paid;
    auto end = Match(text, pos, budget);
    if (!end) return std::unexpected(end.error());
    if (*end == kNoMatch) {
      ++pos;
      continue;
    }
    if (*end == pos) {
      if (pos > gap) lengths.push_back(static_cast<std::uint32_t>(pos - gap));
      gap = pos;
      ++pos;
      continue;
    }
    if (pos > gap) lengths.push_back(static_cast<std::uint32_t>(pos - gap));
    lengths.push_back(static_cast<std::uint32_t>(*end - pos));
    gap = pos = *end;
  }
  if (gap < text.size()) lengths.push_back(static_cast<std::uint32_t>(text.size() - gap));
  return {};
}
}  // namespace llmp::tokenizer
