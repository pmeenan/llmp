// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Builds a Rendered: text, the special tokens a template places (marked so
// the tokenizer encodes exactly those as control tokens) and boundaries.
// Internal to the renderers.
//
// The text is bounded as the interpreter's output is
// (Conversation::max_render_bytes, which serving sets to the template's
// jinja::Limits::max_output_bytes, from the model's context and memory,
// D-102): past the bound the writer drops what it is given, so a renderer
// that repeats client text (a name per tool result) does no more work than
// the bound, and Finish refuses the rendering.

#ifndef LLMP_CHAT_WRITER_H_
#define LLMP_CHAT_WRITER_H_

#include <cstddef>
#include <expected>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include "base/json.h"
#include "chat/chat.h"
#include "chat/jinja.h"
#include "chat/pyjson.h"

namespace llmp::chat {

class Writer {
 public:
  explicit Writer(std::size_t max_bytes = jinja::Limits{}.max_output_bytes)
      : max_bytes_(max_bytes) {}

  void Text(std::string_view text) {
    if (Fits(text.size())) {
      r_.text += text;
    }
  }
  void Special(std::string_view token) {
    if (Fits(token.size())) {
      r_.specials.push_back({r_.text.size(), token.size()});
      r_.text += token;
    }
  }
  // Text in which each occurrence of the given tokens is marked special.
  void TextWithTokens(std::string_view text, std::initializer_list<std::string_view> tokens) {
    while (!text.empty()) {
      std::size_t best = text.size();
      std::string_view found;
      for (const std::string_view t : tokens) {
        const std::size_t at = text.find(t);
        if (at < best || (at == best && at != text.size() && t.size() > found.size())) {
          best = at;
          found = t;
        }
      }
      Text(text.substr(0, best));
      if (found.empty()) {
        return;
      }
      Special(found);
      text.remove_prefix(best + found.size());
    }
  }
  void Json(base::json::Value value) {
    if (!over_ && !AppendPythonJson(value, r_.text, max_bytes_)) {
      over_ = true;  // refused before the text grew past the bound
      r_ = Rendered{};
    }
  }
  void Mark(BoundaryKind kind) { r_.boundaries.push_back({kind, r_.text.size()}); }
  // The rendering, or a refusal once it grew past its bound.
  std::expected<Rendered, Error> Finish() {
    if (over_) {
      return std::unexpected(Error{.rule = Rule::kUnsupported,
                                   .reason = "the rendering is longer than its bound",
                                   .bound = true});
    }
    return std::move(r_);
  }
  // The rendering, unbounded callers only (a fixed prompt around one text).
  Rendered Take() { return std::move(r_); }

 private:
  // Whether `more` bytes fit; once they do not, the text is dropped and
  // every later write ignored.
  bool Fits(std::size_t more) {
    if (!over_ && r_.text.size() + more > max_bytes_) {
      over_ = true;
      r_ = Rendered{};
    }
    return !over_;
  }

  Rendered r_;
  std::size_t max_bytes_;
  bool over_ = false;
};

}  // namespace llmp::chat

#endif  // LLMP_CHAT_WRITER_H_
