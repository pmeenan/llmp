// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "retained_backing/trace.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "retained_backing/json.h"

namespace llmp::rb {
namespace {

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

std::expected<std::string, std::string> ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Error(std::format("cannot open {}", path));
  }
  std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (in.bad()) {
    return Error(std::format("cannot read {}", path));
  }
  return bytes;
}

std::string Sha256Hex(std::string_view bytes) {
  base::Sha256 hash;
  hash.Update(bytes);
  return base::ToHex(hash.Finish());
}

// A non-negative integer member, or an error.
std::expected<std::uint64_t, std::string> Unsigned(const Json& record, std::string_view key) {
  const Json* value = record.Find(key);
  if (value == nullptr || !value->is_int() || value->integer() < 0) {
    return Error(std::format("`{}` is not a non-negative integer", key));
  }
  return static_cast<std::uint64_t>(value->integer());
}

std::expected<std::string, std::string> Text(const Json& record, std::string_view key) {
  const Json* value = record.Find(key);
  if (value == nullptr || !value->is_string()) {
    return Error(std::format("`{}` is not a string", key));
  }
  return value->string();
}

Ev EventKind(std::string_view name, bool& known) {
  static const std::map<std::string_view, Ev> kNames = {
      {"request", Ev::kRequest}, {"lease", Ev::kLease},    {"release", Ev::kRelease},
      {"use", Ev::kUse},         {"shrink", Ev::kShrink},  {"grow", Ev::kGrow},
      {"evict", Ev::kEvict},     {"restore", Ev::kRestore}};
  const auto found = kNames.find(name);
  known = found != kNames.end();
  return known ? found->second : Ev::kRequest;
}

}  // namespace

std::expected<Trace, std::string> ParseTrace(std::string_view bytes) {
  Trace trace;
  std::size_t at = 0;
  std::size_t line_number = 0;
  auto next_line = [&]() -> std::expected<Json, std::string> {
    const std::size_t end = bytes.find('\n', at);
    if (end == std::string_view::npos) {
      return Error(std::format("line {}: missing newline", line_number + 1));
    }
    const std::string_view text = bytes.substr(at, end - at);
    at = end + 1;
    ++line_number;
    auto parsed = Json::Parse(text);
    if (!parsed) {
      return Error(std::format("line {}: {}", line_number, parsed.error()));
    }
    return parsed;
  };

  auto header = next_line();
  if (!header) {
    return std::unexpected(header.error());
  }
  const auto format = Text(*header, "format");
  const auto version = Unsigned(*header, "version");
  if (!format || *format != "llmp-swap-trace" || !version || *version != 1) {
    return Error("not a version 1 swap trace");
  }
  const auto role = Text(*header, "role");
  const auto budget = Unsigned(*header, "budget_bytes");
  const auto unique = Unsigned(*header, "unique_bytes");
  const auto models = Unsigned(*header, "models");
  const auto groups = Unsigned(*header, "groups");
  const auto chunk = Unsigned(*header, "chunk_bytes");
  const auto align = Unsigned(*header, "file_align");
  if (!role || !budget || !unique || !models || !groups || !chunk || !align) {
    return Error("incomplete header");
  }
  if (*chunk != kChunk || *align != kFileAlign || *groups == 0 || *groups > (1U << 30U) ||
      *models == 0 || *models > 1024) {
    return Error("unexpected header geometry");
  }
  trace.role = *role;
  trace.budget = *budget;
  trace.unique = *unique;

  std::map<std::string, std::uint32_t> model_index;
  for (std::uint64_t m = 0; m < *models; ++m) {
    auto record = next_line();
    if (!record) {
      return std::unexpected(record.error());
    }
    const auto kind = Text(*record, "record");
    const auto name = Text(*record, "name");
    const Json* range = record->Find("groups");
    if (!kind || *kind != "model" || !name || range == nullptr || !range->is_array() ||
        range->items().size() != 2 || !range->items()[0].is_int() || !range->items()[1].is_int()) {
      return Error(std::format("line {}: bad model record", line_number));
    }
    const std::int64_t first = range->items()[0].integer();
    const std::int64_t end = range->items()[1].integer();
    const GroupId expected = trace.models.empty() ? 0 : trace.models.back().end;
    if (std::cmp_not_equal(first, expected) || end <= first || std::cmp_greater(end, *groups) ||
        !model_index.emplace(*name, m).second) {
      return Error(std::format("line {}: model ranges do not tile the groups", line_number));
    }
    trace.models.push_back(Model{
        .name = *name, .first = static_cast<GroupId>(first), .end = static_cast<GroupId>(end)});
  }
  if (trace.models.back().end != *groups) {
    return Error("model ranges do not cover the groups");
  }

  trace.groups.reserve(*groups);
  for (std::uint64_t i = 0; i < *groups; ++i) {
    auto record = next_line();
    if (!record) {
      return std::unexpected(record.error());
    }
    const auto kind = Text(*record, "record");
    const auto id = Unsigned(*record, "id");
    const auto model = Text(*record, "model");
    const auto group_kind = Text(*record, "kind");
    const auto used = Unsigned(*record, "used");
    const auto stored = Unsigned(*record, "stored");
    if (!kind || *kind != "group" || !id || *id != i || !model || !group_kind || !used || !stored) {
      return Error(std::format("line {}: bad group record", line_number));
    }
    const auto owner = model_index.find(*model);
    if (owner == model_index.end() || i < trace.models[owner->second].first ||
        i >= trace.models[owner->second].end) {
      return Error(std::format("line {}: group outside its model's range", line_number));
    }
    if (*used == 0 || *used > *stored || *stored % kFileAlign != 0 ||
        *stored - *used >= kFileAlign || *stored > (std::uint64_t{1} << 40U)) {
      return Error(std::format("line {}: stored bytes are not used bytes aligned", line_number));
    }
    trace.groups.push_back(Group{.model = owner->second,
                                 .expert = *group_kind == "expert",
                                 .used = *used,
                                 .stored = *stored});
  }

  std::vector<std::uint64_t> seen(trace.groups.size(), 0);
  std::uint64_t epoch = 0;
  while (at < bytes.size()) {
    auto record = next_line();
    if (!record) {
      return std::unexpected(record.error());
    }
    const auto name = Text(*record, "ev");
    bool known = false;
    Event event;
    if (name) {
      event.ev = EventKind(*name, known);
    }
    if (!known) {
      return Error(std::format("line {}: unknown event", line_number));
    }
    if (event.ev == Ev::kLease || event.ev == Ev::kRelease) {
      const auto lease = Unsigned(*record, "lease");
      if (!lease || *lease > UINT32_MAX) {
        return Error(std::format("line {}: bad lease", line_number));
      }
      event.lease = static_cast<std::uint32_t>(*lease);
    }
    if (event.ev == Ev::kShrink || event.ev == Ev::kGrow) {
      const auto amount = Unsigned(*record, "bytes");
      if (!amount) {
        return Error(std::format("line {}: bad bytes", line_number));
      }
      event.bytes = *amount;
    }
    if (event.ev == Ev::kLease || event.ev == Ev::kUse || event.ev == Ev::kEvict ||
        event.ev == Ev::kRestore) {
      const Json* list = record->Find("groups");
      if (list == nullptr || !list->is_array()) {
        return Error(std::format("line {}: missing groups", line_number));
      }
      ++epoch;
      event.first = static_cast<std::uint32_t>(trace.ids.size());
      for (const Json& item : list->items()) {
        if (!item.is_int() || item.integer() < 0 ||
            static_cast<std::uint64_t>(item.integer()) >= trace.groups.size()) {
          return Error(std::format("line {}: bad group id", line_number));
        }
        const auto g = static_cast<GroupId>(item.integer());
        if (seen[g] == epoch) {
          return Error(std::format("line {}: repeated group id", line_number));
        }
        seen[g] = epoch;
        trace.ids.push_back(g);
      }
      event.count = static_cast<std::uint32_t>(trace.ids.size() - event.first);
    }
    trace.events.push_back(event);
  }
  return trace;
}

std::expected<Trace, std::string> LoadTrace(const std::string& dir, std::string_view name,
                                            std::string_view role) {
  std::string_view pinned;
  if (role == "primary") {
    pinned = kPrimaryManifest;
  } else if (role == "confirmation") {
    pinned = kConfirmationManifest;
  } else {
    return Error("the role is primary or confirmation");
  }
  auto manifest_bytes = ReadFile(dir + "/manifest.json");
  if (!manifest_bytes) {
    return std::unexpected(manifest_bytes.error());
  }
  if (Sha256Hex(*manifest_bytes) != pinned) {
    return Error(std::format("{}/manifest.json is not the recorded {} manifest", dir, role));
  }
  auto manifest = Json::Parse(*manifest_bytes);
  if (!manifest) {
    return std::unexpected(manifest.error());
  }
  const Json* files = manifest->Find("files");
  const Json* entry = files != nullptr ? files->Find(name) : nullptr;
  if (entry == nullptr) {
    return Error(std::format("the manifest has no {}", name));
  }
  const auto want_sha = Text(*entry, "sha256");
  const auto want_budget = Unsigned(*entry, "budget_bytes");
  if (!want_sha || !want_budget) {
    return Error("bad manifest entry");
  }
  auto bytes = ReadFile(std::format("{}/{}", dir, name));
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  std::string digest = Sha256Hex(*bytes);
  if (digest != *want_sha) {
    return Error(std::format("{}: SHA-256 {} is not the manifest's", name, digest));
  }
  auto trace = ParseTrace(*bytes);
  if (!trace) {
    return std::unexpected(std::format("{}: {}", name, trace.error()));
  }
  if (trace->role != role || trace->budget != *want_budget) {
    return Error(std::format("{}: header disagrees with the manifest", name));
  }
  trace->file = name;
  trace->sha256 = std::move(digest);
  return trace;
}

Reference::Reference(std::span<const Group> groups, std::uint64_t budget)
    : budget_(budget),
      stamp_(groups.size(), 0),
      resident_(groups.size(), false),
      protected_(groups.size(), 0),
      wanted_(groups.size(), 0) {
  size_.reserve(groups.size());
  for (const Group& g : groups) {
    size_.push_back(g.stored);
  }
}

void Reference::Touch(std::span<const GroupId> ids) {
  for (const GroupId i : ids) {
    if (resident_[i]) {
      order_.erase({stamp_[i], i});
      stamp_[i] = ++clock_;
      order_.emplace(stamp_[i], i);
    }
  }
}

std::expected<Reference::Step, std::string> Reference::Need(std::span<const GroupId> ids) {
  const std::uint64_t limit = budget_ - shrunk_;
  std::uint64_t pinned = protected_bytes_;
  std::uint64_t incoming = 0;
  Step step;
  ++epoch_;
  for (const GroupId i : ids) {
    wanted_[i] = epoch_;
    if (protected_[i] == 0) {
      pinned += size_[i];
    }
    if (!resident_[i]) {
      step.restored.push_back(i);
      incoming += size_[i];
    }
  }
  if (pinned > limit) {
    return Error("an access does not fit the budget");
  }
  std::int64_t excess =
      static_cast<std::int64_t>(used_ + incoming) - static_cast<std::int64_t>(limit);
  for (const auto& [stamp, i] : order_) {
    if (excess <= 0) {
      break;
    }
    if (wanted_[i] != epoch_ && protected_[i] == 0) {
      step.evicted.push_back(i);
      excess -= static_cast<std::int64_t>(size_[i]);
    }
  }
  for (const GroupId i : step.evicted) {
    order_.erase({stamp_[i], i});
    resident_[i] = false;
    used_ -= size_[i];
  }
  for (const GroupId i : step.restored) {
    resident_[i] = true;
    used_ += size_[i];
    stamp_[i] = ++clock_;
    order_.emplace(stamp_[i], i);
  }
  Touch(ids);
  return step;
}

std::expected<Reference::Step, std::string> Reference::Apply(const Event& event,
                                                             std::span<const GroupId> ids) {
  switch (event.ev) {
    case Ev::kLease: {
      auto step = Need(ids);
      if (!step) {
        return step;
      }
      if (!leases_.emplace(event.lease, std::vector<GroupId>(ids.begin(), ids.end())).second) {
        return Error("lease reused");
      }
      for (const GroupId i : ids) {
        if (protected_[i]++ == 0) {
          protected_bytes_ += size_[i];
        }
      }
      return step;
    }
    case Ev::kRelease: {
      const auto found = leases_.find(event.lease);
      if (found == leases_.end()) {
        return Error("release of an unknown lease");
      }
      const std::vector<GroupId> held = std::move(found->second);
      leases_.erase(found);
      for (const GroupId i : held) {
        if (--protected_[i] == 0) {
          protected_bytes_ -= size_[i];
        }
      }
      Touch(held);
      return Step{};
    }
    case Ev::kUse:
      return Need(ids);
    case Ev::kShrink:
      shrunk_ += event.bytes;
      if (shrunk_ > budget_) {
        return Error("shrinks below zero");
      }
      return Need({});
    case Ev::kGrow:
      if (event.bytes > shrunk_) {
        return Error("grows more than it shrank");
      }
      shrunk_ -= event.bytes;
      return Step{};
    case Ev::kRequest:
      return Step{};
    case Ev::kEvict:
    case Ev::kRestore:
      break;
  }
  return Error("the reference's own records are not applied");
}

}  // namespace llmp::rb
