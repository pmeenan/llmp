// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Preset expression/alias data transcribed from llama.cpp v0.6.0,
// d81235049384534c167caea52b85a694f6103d14, src/llama-vocab.cpp
// (SHA256 18f624c7f5a64f6bd4458248d2cde0ba989cbd46c47b8007be1765e31486c938).
// These are the actual runtime expression strings, including ASCII
// contraction classes and GPT4O/Tekken's category/class lookahead workaround.
// HF inputs supply their original expressions directly and retain their full
// Unicode category and literal-run case-fold semantics. No family matcher code
// is ported here.

#include <algorithm>
#include <array>
#include <span>

#include "tokenizer/pretokenize.h"

namespace llmp::tokenizer {
namespace {
constexpr std::string_view kDEFAULT[] = {
    R"rx([\p{P}\$\+<=>\^~\|]+)rx",
    R"rx('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S))rx",
    R"rx(\p{N}+)rx",
    R"rx([0-9][0-9][0-9])rx",
};
constexpr std::string_view kLLAMA3[] = {
    R"rx((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)rx",
};
constexpr std::string_view kDEEPSEEK_LLM[] = {
    R"rx([\r\n])rx",
    R"rx(\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+)rx",
    R"rx(\s?[!-/:-~！-／：-～‘-‟　-。]+)rx",
    R"rx(\s+$)rx",
    R"rx([一-龥ࠀ-一가-퟿]+)rx",
    R"rx(\p{N}+)rx",
};
constexpr std::string_view kDEEPSEEK3_LLM[] = {
    R"rx(\p{N}{1,3})rx",
    R"rx([一-龥぀-ゟ゠-ヿ]+)rx",
    R"rx([!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)rx",
};
constexpr std::string_view kDEEPSEEK_CODER[] = {
    R"rx([\r\n])rx", R"rx(\s?\p{L}+)rx", R"rx(\s?\p{P}+)rx", R"rx([一-龥ࠀ-一가-퟿]+)rx",
    R"rx(\p{N})rx",
};
constexpr std::string_view kFALCON[] = {
    R"rx([\p{P}\$\+<=>\^~\|`]+)rx",
    R"rx('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S))rx",
    R"rx([0-9][0-9][0-9])rx",
};
constexpr std::string_view kSTARCODER[] = {
    R"rx(\p{N})rx",
    R"rx('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S))rx",
};
constexpr std::string_view kGPT2[] = {
    R"rx('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S))rx",
};
constexpr std::string_view kQWEN2[] = {
    R"rx((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)rx",
};
constexpr std::string_view kQWEN35[] = {
    R"rx((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)rx",
};
constexpr std::string_view kTEKKEN[] = {
    R"rx([^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)rx",
};
constexpr std::string_view kGPT4O[] = {
    R"rx([^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)rx",
};
struct Preset {
  std::string_view name;
  std::span<const std::string_view> expressions;
  bool direct = false, bos = false;
};
const Preset kPresets[] = {
    {"default", kDEFAULT},
    {"llama3", kLLAMA3, true, true},
    {"llama-v3", kLLAMA3, true, true},
    {"llama-bpe", kLLAMA3, true, true},
    {"falcon3", kLLAMA3, true, true},
    {"falcon-h1", kLLAMA3, true, true},
    {"pixtral", kLLAMA3, true, true},
    {"midm-2.0", kLLAMA3, true, true},
    {"lfm2", kLLAMA3, true, true},
    {"jina-v5-nano", kLLAMA3, true, true},
    {"dbrx", kLLAMA3},
    {"smaug-bpe", kLLAMA3},
    {"deepseek-llm", kDEEPSEEK_LLM},
    {"deepseek-coder", kDEEPSEEK_CODER},
    {"deepseek-v3", kDEEPSEEK3_LLM},
    {"joyai-llm", kDEEPSEEK3_LLM},
    {"hunyuan-dense", kDEEPSEEK3_LLM},
    {"hy_v4", kDEEPSEEK3_LLM},
    {"falcon", kFALCON},
    {"starcoder", kSTARCODER},
    {"refact", kSTARCODER},
    {"command-r", kSTARCODER},
    {"smollm", kSTARCODER},
    {"codeshell", kSTARCODER},
    {"exaone", kSTARCODER},
    {"minerva-7b", kSTARCODER},
    {"mellum2", kSTARCODER},
    {"gpt-2", kGPT2},
    {"phi-2", kGPT2},
    {"jina-es", kGPT2},
    {"jina-de", kGPT2},
    {"gigachat", kGPT2},
    {"jina-v2-es", kGPT2},
    {"jina-v2-de", kGPT2},
    {"a.x-4.0", kGPT2},
    {"mellum", kGPT2},
    {"modern-bert", kGPT2},
    {"mpt", kGPT2},
    {"olmo", kGPT2},
    {"jais", kGPT2},
    {"trillion", kGPT2},
    {"granite-docling", kGPT2},
    {"exaone4", kGPT2},
    {"qwen2", kQWEN2},
    {"deepseek-r1-qwen", kQWEN2},
    {"kormo", kQWEN2},
    {"f2llmv2", kQWEN2},
    {"stablelm2", kQWEN2},
    {"megrez", kQWEN2},
    {"hunyuan", kQWEN2},
    {"solar-open", kQWEN2},
    {"qwen35", kQWEN35},
    {"tekken", kTEKKEN, true, true},
    {"gpt-4o", kGPT4O},
    {"llama4", kGPT4O},
    {"kanana2", kGPT4O},
    {"talkie", kGPT4O},
    {"minimax-m2", kGPT4O},
};
}  // namespace
std::expected<PreTokenizerPreset, Error> ReadPreTokenizerPreset(std::string_view name) {
  const auto found = std::ranges::find(kPresets, name, &Preset::name);
  if (found == std::end(kPresets))
    return std::unexpected(Error{Rule::kUnsupported, "GGUF pre-tokenizer preset"});
  PreTokenizerPreset result;
  for (const auto expression : found->expressions) result.expressions.emplace_back(expression);
  result.ignore_merges = found->direct;
  result.add_bos = found->bos;
  return result;
}
}  // namespace llmp::tokenizer
