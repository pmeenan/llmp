// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// JSON written as Python's json.dumps(value, ensure_ascii=False) writes a
// value json.loads read: what chat templates' `tojson` filter (Hugging Face
// transformers) and DeepSeek's encoder print into prompts, so a renderer's
// bytes match the reference's.
//
// Separators ", " and ": "; object members in document order; strings with
// `"`, `\`, \n, \r, \t, \b, \f escaped and other control characters as
// \u00XX, everything else as is; integers as Python ints print them (-0 is
// 0); other numbers as Python's float repr (shortest round trip, 1e+16 and
// 1e-05 style exponents, Infinity for overflow).

#ifndef JITLLM_CHAT_PYJSON_H_
#define JITLLM_CHAT_PYJSON_H_

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>

#include "base/json.h"

namespace jitllm::chat {

void AppendPythonJson(base::json::Value value, std::string& out);

// The same, refused before `out` would grow past `limit` bytes: false, with
// `out` holding a prefix of the text no longer than `limit`.
bool AppendPythonJson(base::json::Value value, std::string& out, std::size_t limit);

// Python's repr of a finite or infinite double, as json.dumps writes it.
void AppendPythonFloat(double value, std::string& out);

// Python's str() of a value json.loads read, which a template prints with
// {{ }}: a string as itself; None, True and False; integers; floats as
// repr (inf for overflow); lists and dicts as their repr, the strings in
// them quoted and escaped as repr does (members in document order).
void AppendPythonStr(base::json::Value value, std::string& out);

// Python's repr() of a string (quotes chosen as Python does; `\t`, `\n`,
// `\r`, `\xNN`, `\uNNNN`, `\UNNNNNNNN` for what str.isprintable() refuses);
// false once `out` is longer than `limit`. The interpreter's repr too.
bool AppendPythonStringRepr(std::string_view s, std::string& out,
                            std::size_t limit = std::numeric_limits<std::size_t>::max());

// str.isprintable() of one code point: not in categories Cc, Cf, Cs, Co,
// Cn, Zl, Zp or Zs, except the space (UCD 15.1.0's categories).
bool PythonPrintable(char32_t cp);

// The code point at `at` of UTF-8 text and its length in bytes; a sequence
// cut short by the text's end is decoded as far as it goes. For text known
// to be well-formed (JSON strings and the interpreter's are).
char32_t DecodeUtf8At(std::string_view s, std::size_t at, std::size_t& length);

}  // namespace jitllm::chat

#endif  // JITLLM_CHAT_PYJSON_H_
