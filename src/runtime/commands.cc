// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/commands.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <limits>
#include <string>
#include <system_error>

namespace llmp::runtime {
namespace {

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

template <typename T>
bool Number(std::string_view text, T min, T max, T& into) {
  T value{};
  // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage): bounded by its end
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc() || end != text.data() + text.size() || value < min || value > max) {
    return false;
  }
  into = value;
  return true;
}

bool OnOff(std::string_view text, bool& into) {
  if (text != "on" && text != "off") {
    return false;
  }
  into = text == "on";
  return true;
}

// A configured model's name as a request names it (config/node_config.h).
bool PlausibleName(std::string_view name) { return !name.empty() && name.size() <= 64; }

}  // namespace

std::expected<void, std::string> ApplyServiceDiagnostics(
    ServingOptions& options, std::optional<std::string_view> component,
    std::optional<std::string_view> startup_bytes, std::optional<std::string_view> node_cap,
    std::optional<std::string_view> capture_first_verify) {
  if (!component && !startup_bytes && !node_cap && !capture_first_verify) return {};
  std::uint64_t bytes = 0, cap = 0;
  constexpr std::uint64_t extent = std::uint64_t{2} << 20U;
  const bool automatic = startup_bytes && *startup_bytes == "auto";
  if ((capture_first_verify && *capture_first_verify != "1") || !component ||
      (*component != "1" && *component != "all") || !startup_bytes || !node_cap ||
      (!automatic && !Number(*startup_bytes, std::uint64_t{1}, UINT64_MAX, bytes)) ||
      !Number(*node_cap, std::uint64_t{1}, UINT64_MAX, cap) || bytes % extent || cap % extent ||
      bytes > cap)
    return Error(
        "shared service diagnostic needs selector=1 or all, startup bytes (or auto), and aligned "
        "node cap");
  options.diagnostic_component_solo = true;
  options.diagnostic_component_all = *component == "all";
  options.diagnostic_capture_first_verify = capture_first_verify.has_value();
  options.diagnostic_startup_host_bytes = bytes;
  options.diagnostic_auto_startup = automatic;
  options.diagnostic_budget_cap_bytes = cap;
  return {};
}

std::expected<void, std::string> ApplyServiceDiagnosticEnvironment(ServingOptions& options) {
  const auto value = [](const char* name) -> std::optional<std::string_view> {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): before serving threads start
    const char* text = std::getenv(name);
    return text ? std::optional<std::string_view>(text) : std::nullopt;
  };
  // The funded tuple selects every component pipeline; the historical Qwen-only
  // selector keeps its narrower alias contract.
  const auto qwen = value("LLMP_DIAGNOSTIC_COMPONENT_QWEN");
  if (qwen && *qwen != "1") return Error("the Qwen-only component selector requires 1");
  const auto startup = value("LLMP_DIAGNOSTIC_STARTUP_HOST_BYTES");
  const auto cap = value("LLMP_DIAGNOSTIC_NODE_CAP_BYTES");
  const auto capture = value("LLMP_DIAGNOSTIC_CAPTURE_FIRST_VERIFY");
  const bool funded = startup || cap || capture;
  return ApplyServiceDiagnostics(
      options, qwen ? qwen : (funded ? std::optional<std::string_view>("all") : std::nullopt),
      startup, cap, capture);
}

const std::string_view kCommandUsage =
    "Serving commands (after the runtime's own options; local, in this process):\n"
    "  chat [--max-tokens N] [--ignore-stop] [--fresh] [SERVING] --turn MODEL TEXT...\n"
    "      each turn to its model, in order, swapping models as needed; a turn\n"
    "      continues its model's conversation unless --fresh\n"
    "  swap-table [--pairs A:B,...] [--context-text FILE] [--context-tokens N]\n"
    "      [--continue N] [--cycles N] [--zero-context on|off] [--handoff on|off]\n"
    "      [--short-prompt TEXT] [--image-expect SHA256] [SERVING]\n"
    "      M3's swap table: every ordered pair (or those named) A->B->A\n"
    "  SERVING: [--plain] [--image-prompt TEXT] [--image-noise FILE] [--report FILE]\n"
    "  settings [--json]\n"
    "      every configured model's settings, each with its source (derived,\n"
    "      calibrated, override, fallback); reads only, beside a running service\n";

std::expected<CommandOptions, std::string> ParseCommand(std::string_view name,
                                                        std::span<const std::string_view> args) {
  CommandOptions o;
  if (name == "chat") {
    o.command = Command::kChat;
  } else if (name == "swap-table") {
    o.command = Command::kSwapTable;
  } else if (name == "settings") {
    o.command = Command::kSettings;
    for (const std::string_view a : args) {
      if (a != "--json") {
        return Error(std::format("{} is not an option of settings", a));
      }
      o.json = true;
    }
    return o;
  } else {
    return Error(std::format("unknown command '{}'", name));
  }
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    // Flags without a value.
    if (a == "--plain") {
      o.serving.plain = true;
      continue;
    }
    if (o.command == Command::kChat && a == "--ignore-stop") {
      o.chat.ignore_stop = true;
      continue;
    }
    if (o.command == Command::kChat && a == "--fresh") {
      o.chat.fresh = true;
      continue;
    }
    if (o.command == Command::kChat && a == "--turn") {
      if (i + 2 >= args.size()) {
        return Error("--turn needs a model and a text");
      }
      const std::string_view model = args[++i];
      const std::string_view text = args[++i];
      if (!PlausibleName(model)) {
        return Error("--turn names no model");
      }
      if (text.empty()) {
        return Error("a turn's text must not be empty");
      }
      o.chat.turns.push_back({.model = std::string(model), .text = std::string(text)});
      continue;
    }
    if (i + 1 >= args.size()) {
      return Error(std::format("{} needs a value", a));
    }
    const std::string_view v = args[++i];
    bool ok = true;
    if (a == "--image-prompt") {
      ok = !v.empty() && v.size() <= kMaxImagePromptBytes;
      o.serving.image_prompt = v;
    } else if (a == "--image-noise") {
      ok = !v.empty();
      o.serving.image_noise = v;
    } else if (a == "--report") {
      ok = !v.empty();
      o.serving.report = v;
    } else if (o.command == Command::kChat && a == "--max-tokens") {
      // At most what the model's context has left, checked per turn.
      ok =
          Number<std::uint32_t>(v, 1, std::numeric_limits<std::uint32_t>::max(), o.chat.max_tokens);
    } else if (o.command == Command::kSwapTable && a == "--pairs") {
      std::string_view rest = v;
      while (ok && !rest.empty()) {
        const std::size_t comma = rest.find(',');
        const std::string_view pair = rest.substr(0, comma);
        rest = comma == std::string_view::npos ? std::string_view() : rest.substr(comma + 1);
        const std::size_t colon = pair.find(':');
        ok = colon != std::string_view::npos && PlausibleName(pair.substr(0, colon)) &&
             PlausibleName(pair.substr(colon + 1)) &&
             pair.substr(0, colon) != pair.substr(colon + 1);
        if (ok) {
          o.table.pairs.emplace_back(pair.substr(0, colon), pair.substr(colon + 1));
        }
      }
      ok = ok && !o.table.pairs.empty();
    } else if (o.command == Command::kSwapTable && a == "--context-text") {
      ok = !v.empty();
      o.table.context_text = v;
    } else if (o.command == Command::kSwapTable && a == "--context-tokens") {
      // The selected model's usable context decides (Table::Context).
      ok = Number<std::uint32_t>(v, 32, std::numeric_limits<std::uint32_t>::max(),
                                 o.table.context_tokens);
    } else if (o.command == Command::kSwapTable && a == "--continue") {
      ok = Number<std::uint32_t>(v, 1, 1024, o.table.continue_tokens);
    } else if (o.command == Command::kSwapTable && a == "--cycles") {
      ok = Number<std::uint32_t>(v, 1, 8, o.table.cycles);  // 0 would measure nothing
    } else if (o.command == Command::kSwapTable && a == "--zero-context") {
      ok = OnOff(v, o.table.zero_context);
    } else if (o.command == Command::kSwapTable && a == "--handoff") {
      ok = OnOff(v, o.table.handoff);
    } else if (o.command == Command::kSwapTable && a == "--short-prompt") {
      ok = !v.empty();
      o.table.short_prompt = v;
    } else if (o.command == Command::kSwapTable && a == "--image-expect") {
      // A SHA-256 as the table prints it: 64 lowercase hex digits.
      ok = v.size() == 64 && std::ranges::all_of(v, [](char c) {
             return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
      o.table.image_expect = v;
    } else {
      return Error(std::format("{} is not an option of {}", a, name));
    }
    if (!ok) {
      return Error(std::format("{} does not take '{}'", a, v));
    }
  }
  if (o.command == Command::kChat && o.chat.turns.empty()) {
    return Error("chat needs at least one --turn");
  }
  return o;
}

}  // namespace llmp::runtime
