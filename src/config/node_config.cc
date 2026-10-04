// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "config/node_config.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/stat.h>
#include <unistd.h>

// toml++ in its TOML 1.0.0 mode, without exceptions (D-066) or formatters;
// only this translation unit includes it (target compile definitions).
#include <dirent.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <toml++/toml.hpp>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "base/report.h"
#include "platform/path_trust.h"

namespace jitllm::config {
namespace {

namespace fs = std::filesystem;

using KeyPath = std::vector<std::string>;

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

// A key path as TOML would write it: bare parts as they are, others quoted.
std::string KeyText(const KeyPath& path) {
  std::string text;
  for (const std::string& part : path) {
    if (!text.empty()) {
      text += '.';
    }
    const bool bare = !part.empty() && std::ranges::all_of(part, [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
             c == '_' || c == '-';
    });
    if (bare) {
      text += part;
    } else {
      text += '"';
      for (const char c : part) {
        const auto byte = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') {
          text += '\\';
          text += c;
        } else if (byte < 0x20 || byte == 0x7F) {
          text += std::format("\\u{:04X}", byte);
        } else {
          text += c;
        }
      }
      text += '"';
    }
  }
  return text;
}

bool IsPrefix(const KeyPath& prefix, const KeyPath& path) {
  return prefix.size() <= path.size() && std::equal(prefix.begin(), prefix.end(), path.begin());
}

// A value or a table as one file wrote it.
struct Entry {
  KeyPath path;
  const toml::node* node = nullptr;
  std::size_t file = 0;
};

class Collector {
 public:
  explicit Collector(std::span<const SourceText> files) : files_(files) {}

  void At(std::size_t file, const toml::node& node, std::string message) {
    const toml::source_position begin = node.source().begin;
    diagnostics_.push_back({.file = files_[file].name,
                            .line = begin.line,
                            .column = begin.column,
                            .message = std::move(message)});
  }
  void At(const Entry& entry, std::string message) {
    At(entry.file, *entry.node, std::move(message));
  }
  void Position(std::size_t file, const toml::source_position& at, std::string message) {
    diagnostics_.push_back({.file = files_[file].name,
                            .line = at.line,
                            .column = at.column,
                            .message = std::move(message)});
  }
  void Document(std::string message) {
    diagnostics_.push_back({.file = {}, .message = std::move(message)});
  }
  void File(std::size_t file, std::string message) {
    diagnostics_.push_back({.file = files_[file].name, .message = std::move(message)});
  }

  bool empty() const { return diagnostics_.empty(); }
  std::vector<Diagnostic> Take() { return std::move(diagnostics_); }
  const std::string& Name(std::size_t file) const { return files_[file].name; }

 private:
  std::span<const SourceText> files_;
  std::vector<Diagnostic> diagnostics_;
};

// What the schema knows (cluster-design.md, D-063, D-073).
enum class Kind : std::uint8_t {
  kAbsolutePath,  // an absolute path in normal form
  kRolePath,      // a path in normal form, absolute or relative
  kUuid,
  kPort,
  kInterfaces,
  kPeerScopes,
  kProfile,
  kClientBind,         // a bind entry or an array of them (D-097)
  kClientPort,         // [client] port
  kClientInteger,      // [client]'s counts, seconds and bytes (kClientIntegers)
  kClientStallAction,  // [client] stall_action
  kMemoryHours,        // [memory] retention_hours
  kMemoryGib,          // [memory] spill_budget_gib
  kMemoryKeep,         // [memory] keep_across_restart
};

// [client]'s integer limits (D-102) and their ranges: counts and seconds
// fit 32 bits; bytes start at a least that leaves the route usable.
struct ClientInteger {
  std::string_view key;
  std::int64_t least;
  std::int64_t most;
  std::string_view unit;  // for messages: "", " seconds", " bytes"
};
constexpr std::int64_t kU32 = 0xFFFF'FFFFLL;
constexpr std::int64_t kSeconds = kMaxClientSeconds;
constexpr std::int64_t kBytes = std::numeric_limits<std::int64_t>::max();
constexpr std::array<ClientInteger, 12> kClientIntegers = {{
    {"max_connections", 1, kU32, ""},
    {"max_queued", 1, kU32, ""},
    {"queue_wait_seconds", 1, kSeconds, " seconds"},
    {"stall_seconds", 1, kSeconds, " seconds"},
    {"deadline_cap_seconds", 1, kSeconds, " seconds"},
    {"idle_seconds", 1, kSeconds, " seconds"},
    {"request_inactivity_seconds", 1, kSeconds, " seconds"},
    {"write_inactivity_seconds", 1, kSeconds, " seconds"},
    {"hang_seconds", kMinHangSeconds, kSeconds, " seconds"},
    {"request_memory_bytes", std::int64_t{1} << 24, kBytes, " bytes"},
    {"max_body_bytes", 1024, static_cast<std::int64_t>(kMaxBodyCeiling), " bytes"},
    {"stream_buffer_bytes", 4096, kBytes, " bytes"},
}};

const ClientInteger* FindClientInteger(std::string_view key) {
  for (const ClientInteger& c : kClientIntegers) {
    if (c.key == key) {
      return &c;
    }
  }
  return nullptr;
}

struct KeySpec {
  KeyPath path;
  Kind kind;
  bool member;  // one of the keys only a cluster member has
};

const std::vector<KeySpec>& Schema() {
  static const std::vector<KeySpec> schema = {
      {.path = {"cluster_file"}, .kind = Kind::kAbsolutePath, .member = true},
      {.path = {"node_id"}, .kind = Kind::kUuid, .member = true},
      {.path = {"credentials", "ca_file"}, .kind = Kind::kAbsolutePath, .member = true},
      {.path = {"credentials", "certificate_file"}, .kind = Kind::kAbsolutePath, .member = true},
      {.path = {"credentials", "private_key_file"}, .kind = Kind::kAbsolutePath, .member = true},
      {.path = {"control", "port"}, .kind = Kind::kPort, .member = true},
      {.path = {"control", "interfaces"}, .kind = Kind::kInterfaces, .member = true},
      {.path = {"control", "peer_scopes"}, .kind = Kind::kPeerScopes, .member = true},
      {.path = {"limits", "profile"}, .kind = Kind::kProfile, .member = false},
      {.path = {"client", "bind"}, .kind = Kind::kClientBind, .member = false},
      {.path = {"client", "port"}, .kind = Kind::kClientPort, .member = false},
      {.path = {"client", "max_connections"}, .kind = Kind::kClientInteger, .member = false},
      {.path = {"client", "max_queued"}, .kind = Kind::kClientInteger, .member = false},
      {.path = {"client", "queue_wait_seconds"}, .kind = Kind::kClientInteger, .member = false},
      {.path = {"client", "stall_seconds"}, .kind = Kind::kClientInteger, .member = false},
      {.path = {"client", "stall_action"}, .kind = Kind::kClientStallAction, .member = false},
      {.path = {"client", "deadline_cap_seconds"}, .kind = Kind::kClientInteger, .member = false},
      {.path = {"client", "idle_seconds"}, .kind = Kind::kClientInteger, .member = false},
      {.path = {"client", "request_inactivity_seconds"},
       .kind = Kind::kClientInteger,
       .member = false},
      {.path = {"client", "write_inactivity_seconds"},
       .kind = Kind::kClientInteger,
       .member = false},
      {.path = {"client", "hang_seconds"}, .kind = Kind::kClientInteger, .member = false},
      {.path = {"client", "request_memory_bytes"}, .kind = Kind::kClientInteger, .member = false},
      {.path = {"client", "max_body_bytes"}, .kind = Kind::kClientInteger, .member = false},
      {.path = {"client", "stream_buffer_bytes"}, .kind = Kind::kClientInteger, .member = false},
      {.path = {"memory", "retention_hours"}, .kind = Kind::kMemoryHours, .member = false},
      {.path = {"memory", "spill_budget_gib"}, .kind = Kind::kMemoryGib, .member = false},
      {.path = {"memory", "keep_across_restart"}, .kind = Kind::kMemoryKeep, .member = false},
      {.path = {"storage", "data_dir"}, .kind = Kind::kAbsolutePath, .member = false},
      {.path = {"storage", "installed"}, .kind = Kind::kRolePath, .member = false},
      {.path = {"storage", "spill"}, .kind = Kind::kRolePath, .member = false},
      {.path = {"storage", "state"}, .kind = Kind::kRolePath, .member = false},
      {.path = {"storage", "checkpoints"}, .kind = Kind::kRolePath, .member = false},
      {.path = {"storage", "long_term"}, .kind = Kind::kAbsolutePath, .member = false},
      {.path = {"storage", "archive"}, .kind = Kind::kRolePath, .member = false},
  };
  return schema;
}

const KeySpec* FindSpec(const KeyPath& path) {
  for (const KeySpec& spec : Schema()) {
    if (spec.path == path) {
      return &spec;
    }
  }
  return nullptr;
}

// [models] and each [models.<name>]: tables whose keys are names.
bool IsModelsTable(const KeyPath& path) {
  return (path.size() == 1 || path.size() == 2) && path.front() == "models";
}

// Whether path names a table the schema has keys in.
bool IsSchemaTable(const KeyPath& path) {
  return IsModelsTable(path) || std::ranges::any_of(Schema(), [&](const KeySpec& spec) {
           return spec.path.size() > path.size() && IsPrefix(path, spec.path);
         });
}

// A model's name: 1 to kMaxModelName of [a-z0-9._-], starting with a letter
// or digit.
bool IsModelName(std::string_view name) {
  const auto allowed = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
  };
  return !name.empty() && name.size() <= kMaxModelName && std::ranges::all_of(name, allowed) &&
         name.front() != '.' && name.front() != '_' && name.front() != '-';
}

// An artifact's or a composition's ID: 64 lowercase hex digits.
bool IsContentId(std::string_view text) {
  return text.size() == 64 && std::ranges::all_of(text, [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

// Adds a file's values (leaves: everything but a table defined by a header
// or dotted keys, so an inline table or an array is one value) and its
// tables to the lists. It descends only into tables the schema has keys in,
// so a table it does not know is one entry however deep it goes.
void Flatten(const toml::table& table, KeyPath& path, std::size_t file, std::vector<Entry>& leaves,
             std::vector<Entry>& tables) {
  for (const auto& [key, node] : table) {
    path.emplace_back(key.str());
    const toml::table* child = node.as_table();
    if (child != nullptr && !child->is_inline()) {
      tables.push_back({.path = path, .node = &node, .file = file});
      if (IsSchemaTable(path) || path == KeyPath{"control", "peer_scopes"}) {
        Flatten(*child, path, file, leaves, tables);
      }
    } else {
      leaves.push_back({.path = path, .node = &node, .file = file});
    }
    path.pop_back();
  }
}

const char* TypeName(const toml::node& node) {
  switch (node.type()) {
    case toml::node_type::table:
      return "a table";
    case toml::node_type::array:
      return "an array";
    case toml::node_type::string:
      return "a string";
    case toml::node_type::integer:
      return "an integer";
    case toml::node_type::floating_point:
      return "a float";
    case toml::node_type::boolean:
      return "a boolean";
    case toml::node_type::date:
    case toml::node_type::time:
    case toml::node_type::date_time:
      return "a date or time";
    case toml::node_type::none:
      break;
  }
  return "nothing";
}

// A path value's problem, if any: the text must be a path in normal form
// (no empty, "." or ".." component, no trailing "/") without control
// characters, and absolute where required.
std::optional<std::string> PathProblem(std::string_view text, bool absolute) {
  if (text.empty()) {
    return "must not be empty";
  }
  if (text.size() >= 4096) {
    return "is longer than 4095 bytes";
  }
  if (std::ranges::any_of(text, [](char c) {
        const auto byte = static_cast<unsigned char>(c);
        return byte < 0x20 || byte == 0x7F;
      })) {
    return "must not contain control characters";
  }
  if (base::Printable(text) != text) {
    return "must not contain control or invisible formatting characters";
  }
  if (absolute && text.front() != '/') {
    return "must be an absolute path";
  }
  if (text != "/" && text.back() == '/') {
    return "must not end in '/'";
  }
  std::string_view rest = text.front() == '/' ? text.substr(1) : text;
  while (!rest.empty() || text == "/") {
    if (text == "/") {
      break;
    }
    const std::size_t slash = rest.find('/');
    const std::string_view part = rest.substr(0, slash);
    if (part.empty() || part == "." || part == "..") {
      return "must be in normal form, without empty, '.' or '..' components";
    }
    if (slash == std::string_view::npos) {
      break;
    }
    rest = rest.substr(slash + 1);
  }
  return std::nullopt;
}

bool IsUuid(std::string_view text) {
  if (text.size() != 36) {
    return false;
  }
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (c != '-') {
        return false;
      }
    } else if ((c < '0' || c > '9') && (c < 'a' || c > 'f')) {
      return false;
    }
  }
  return true;
}

// cluster-design.md's local selectors: "ifname:<exact-name>", with the
// kernel's own rules for interface names, or
// "port:<phys_switch_id>/<phys_port_name>".
std::optional<std::string> SelectorProblem(std::string_view text) {
  const auto printable = [](std::string_view s, bool slash_ok) {
    return std::ranges::all_of(s, [slash_ok](char c) {
      return c > ' ' && c < 0x7F && c != ':' && (slash_ok || c != '/');
    });
  };
  if (text.starts_with("ifname:")) {
    const std::string_view name = text.substr(7);
    if (name.empty() || name.size() > 15 || name == "." || name == ".." ||
        !printable(name, false)) {
      return "names an invalid interface: an interface name is 1-15 printable characters without "
             "'/', ':' or "
             "spaces, and not '.' or '..'";
    }
    return std::nullopt;
  }
  if (text.starts_with("port:")) {
    const std::string_view rest = text.substr(5);
    const std::size_t slash = rest.find('/');
    const std::string_view id = rest.substr(0, slash);
    const std::string_view port =
        slash == std::string_view::npos ? std::string_view() : rest.substr(slash + 1);
    const bool hex_id = !id.empty() && id.size() <= 64 && std::ranges::all_of(id, [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
    if (!hex_id || port.empty() || port.size() > 64 || !printable(port, false)) {
      return "names an invalid port: write port:<phys_switch_id>/<phys_port_name>, with the switch "
             "ID in "
             "lowercase hex";
    }
    return std::nullopt;
  }
  return R"(must be "ifname:<name>" or "port:<phys_switch_id>/<phys_port_name>")";
}

// Whether b is a or lies inside it (both absolute, in normal form).
bool Within(const fs::path& a, const fs::path& b) {
  return std::ranges::mismatch(a, b).in1 == a.end();
}

// Where the values found by the schema pass came from, for later checks.
struct Found {
  std::map<KeyPath, const Entry*> entries;

  const Entry* Get(const KeyPath& path) const {
    const auto it = entries.find(path);
    return it == entries.end() ? nullptr : it->second;
  }
};

class Validator {
 public:
  Validator(Collector& out, fs::path anchor) : out_(out), anchor_(std::move(anchor)) {}

  NodeConfig Run(const std::vector<Entry>& leaves, const std::vector<Entry>& tables) {
    for (const Entry& table : tables) {
      if (table.path.front() == "control" || table.path.front() == "credentials") {
        member_ = true;  // even an empty [control] or [credentials] (D-073)
      }
      if (IsModelsTable(table.path) && table.path.size() == 2) {
        NoteModel(table.path[1], table);  // even an empty [models.<name>]
      }
      if (const KeySpec* spec = FindSpec(table.path)) {
        if (spec->kind == Kind::kPeerScopes) {
          continue;  // written as a table: its entries are its keys
        }
        out_.At(table, std::format("{} must be {}, not a table", KeyText(table.path),
                                   Expected(spec->kind)));
      } else if (IsPeerScopeEntry(table.path)) {
        out_.At(table,
                std::format("{} must be a string selector, not a table", KeyText(table.path)));
      } else if (!IsSchemaTable(table.path)) {
        out_.At(table, std::format("unknown table {}", KeyText(table.path)));
      }
    }
    for (const Entry& leaf : leaves) {
      Leaf(leaf);
    }
    return Build();
  }

 private:
  static bool IsPeerScopeEntry(const KeyPath& path) {
    return path.size() == 3 && path[0] == "control" && path[1] == "peer_scopes";
  }

  static const char* Expected(Kind kind) {
    switch (kind) {
      case Kind::kAbsolutePath:
        return "an absolute path";
      case Kind::kRolePath:
        return "a path";
      case Kind::kUuid:
        return "a UUID string";
      case Kind::kPort:
        return "an integer port";
      case Kind::kInterfaces:
        return "\"auto\" or an array of selectors";
      case Kind::kPeerScopes:
        return "a table of member UUIDs to selectors";
      case Kind::kProfile:
        return "\"initial-v2\"";
      case Kind::kClientBind:
        return "a bind entry or an array of them";
      case Kind::kClientPort:
        return "an integer port";
      case Kind::kClientInteger:
        return "an integer";
      case Kind::kClientStallAction:
        return R"("report" or "fail")";
      case Kind::kMemoryHours:
        return "an integer of hours";
      case Kind::kMemoryGib:
        return "an integer of GiB";
      case Kind::kMemoryKeep:
        return "a boolean";
    }
    return "";
  }

  void Leaf(const Entry& leaf) {
    const std::string key = KeyText(leaf.path);
    if (IsPeerScopeEntry(leaf.path)) {
      PeerScope(leaf, leaf.path[2], *leaf.node);
      return;
    }
    if (leaf.path.size() == 3 && leaf.path.front() == "models") {
      ModelLeaf(leaf);
      return;
    }
    const KeySpec* spec = FindSpec(leaf.path);
    if (spec == nullptr && IsSchemaTable(leaf.path) && leaf.node->is_table()) {
      // An inline table, owned whole by its file: its keys are checked as
      // if written under a header.
      if (leaf.path.front() == "control" || leaf.path.front() == "credentials") {
        member_ = true;
      }
      if (IsModelsTable(leaf.path) && leaf.path.size() == 2) {
        NoteModel(leaf.path[1], leaf);  // even an empty inline table
      }
      for (const auto& [name, value] : *leaf.node->as_table()) {
        KeyPath path = leaf.path;
        path.emplace_back(name.str());
        inline_entries_.push_back(std::make_unique<Entry>(
            Entry{.path = std::move(path), .node = &value, .file = leaf.file}));
        Leaf(*inline_entries_.back());
      }
      return;
    }
    if (spec == nullptr) {
      if (IsSchemaTable(leaf.path)) {
        out_.At(leaf, std::format("{} must be a table, not {}", key, TypeName(*leaf.node)));
      } else {
        out_.At(leaf, std::format("unknown key {}", key));
      }
      return;
    }
    if (spec->member) {
      member_ = true;
    }
    found_.entries[leaf.path] = &leaf;
    const toml::node& node = *leaf.node;
    switch (spec->kind) {
      case Kind::kAbsolutePath:
      case Kind::kRolePath: {
        const auto* text = node.as_string();
        if (text == nullptr) {
          out_.At(leaf, std::format("{} must be a string path, not {}", key, TypeName(node)));
          break;
        }
        if (auto problem = PathProblem(text->get(), spec->kind == Kind::kAbsolutePath)) {
          out_.At(leaf, std::format("{} {}", key, *problem));
          break;
        }
        paths_[leaf.path] = fs::path(text->get());
        break;
      }
      case Kind::kUuid: {
        const auto* text = node.as_string();
        if (text == nullptr || !IsUuid(text->get())) {
          out_.At(leaf, std::format("{} must be a canonical lowercase UUID string", key));
          break;
        }
        node_id_ = text->get();
        break;
      }
      case Kind::kPort: {
        const auto* value = node.as_integer();
        if (value == nullptr) {
          out_.At(leaf, std::format("{} must be an integer, not {}", key, TypeName(node)));
        } else if (value->get() < 1 || value->get() > 65535) {
          out_.At(leaf, std::format("{} must be from 1 to 65535, not {}", key, value->get()));
        } else {
          port_ = static_cast<std::uint16_t>(value->get());
        }
        break;
      }
      case Kind::kInterfaces:
        Interfaces(leaf);
        break;
      case Kind::kPeerScopes: {
        const auto* table = node.as_table();
        if (table == nullptr) {
          out_.At(leaf, std::format("{} must be a table of member UUIDs to selectors, not {}", key,
                                    TypeName(node)));
          break;
        }
        for (const auto& [id, value] : *table) {
          PeerScope(leaf, std::string(id.str()), value);
        }
        break;
      }
      case Kind::kProfile: {
        const auto* text = node.as_string();
        if (text == nullptr || text->get() != kLimitsProfile) {
          out_.At(leaf,
                  std::format("{} must be \"{}\", the only limits profile of schema version {}",
                              key, kLimitsProfile, kSchemaVersion));
        }
        break;
      }
      case Kind::kClientBind:
        ClientBind(leaf);
        break;
      case Kind::kClientPort: {
        const auto* value = node.as_integer();
        if (value == nullptr) {
          out_.At(leaf, std::format("{} must be an integer, not {}", key, TypeName(node)));
        } else if (value->get() < 1 || value->get() > 65535) {
          out_.At(leaf, std::format("{} must be from 1 to 65535, not {}", key, value->get()));
        } else {
          client_port_ = static_cast<std::uint16_t>(value->get());
        }
        break;
      }
      case Kind::kClientInteger: {
        const ClientInteger* range = FindClientInteger(leaf.path.back());
        const auto* value = node.as_integer();
        if (range == nullptr) {
          out_.At(leaf, std::format("unknown key {}", key));  // not reached: the schema's
        } else if (value == nullptr) {
          out_.At(leaf, std::format("{} must be an integer, not {}", key, TypeName(node)));
        } else if (value->get() < range->least || value->get() > range->most) {
          out_.At(leaf, std::format("{} must be from {} to {}{}, not {}", key, range->least,
                                    range->most, range->unit, value->get()));
        } else {
          client_integers_[std::string(range->key)] = static_cast<std::uint64_t>(value->get());
        }
        break;
      }
      case Kind::kClientStallAction: {
        const auto* text = node.as_string();
        if (text == nullptr || (text->get() != "report" && text->get() != "fail")) {
          out_.At(leaf, std::format(R"({} must be "report" or "fail")", key));
        } else {
          stall_action_ = text->get() == "fail" ? StallAction::kFail : StallAction::kReport;
        }
        break;
      }
      case Kind::kMemoryHours:
      case Kind::kMemoryGib: {
        const bool hours = spec->kind == Kind::kMemoryHours;
        const std::int64_t least = hours ? 1 : 0;
        const std::int64_t most = hours ? kMaxRetentionHours : kMaxSpillBudgetGib;
        const auto* value = node.as_integer();
        if (value == nullptr) {
          out_.At(leaf, std::format("{} must be an integer, not {}", key, TypeName(node)));
        } else if (value->get() < least || value->get() > most) {
          out_.At(leaf,
                  std::format("{} must be from {} to {}, not {}", key, least, most, value->get()));
        } else if (hours) {
          retention_hours_ = static_cast<std::uint32_t>(value->get());
        } else {
          spill_budget_gib_ = static_cast<std::uint32_t>(value->get());
        }
        break;
      }
      case Kind::kMemoryKeep: {
        const auto* value = node.as_boolean();
        if (value == nullptr) {
          out_.At(leaf, std::format("{} must be a boolean, not {}", key, TypeName(node)));
        } else {
          keep_across_restart_ = value->get();
        }
        break;
      }
    }
  }

  // [client] bind: one entry, or an array of 1 to kMaxBindEntries.
  void ClientBind(const Entry& leaf) {
    const std::string key = KeyText(leaf.path);
    const toml::node& node = *leaf.node;
    std::vector<BindEntry> entries;
    bool ok = true;
    const auto one = [&](const toml::node& item, const std::string& where) {
      const auto* text = item.as_string();
      if (text == nullptr) {
        out_.At(leaf, std::format("{} must be a string, not {}", where, TypeName(item)));
        ok = false;
        return;
      }
      auto entry = ParseBindEntry(text->get());
      if (!entry) {
        out_.At(leaf, std::format("{} {}", where, entry.error()));
        ok = false;
        return;
      }
      entries.push_back(*entry);
    };
    if (const auto* array = node.as_array()) {
      if (array->empty() || array->size() > kMaxBindEntries) {
        out_.At(leaf, std::format("{} must hold 1 to {} entries, not {}", key, kMaxBindEntries,
                                  array->size()));
        return;
      }
      for (std::size_t i = 0; i < array->size(); ++i) {
        one((*array)[i], std::format("{}[{}]", key, i));
      }
    } else if (node.is_string()) {
      one(node, key);
    } else {
      out_.At(leaf, std::format("{} must be a string or an array of strings, not {}", key,
                                TypeName(node)));
      return;
    }
    if (ok) {
      bind_ = std::move(entries);
    }
  }

  void Interfaces(const Entry& leaf) {
    const std::string key = KeyText(leaf.path);
    const toml::node& node = *leaf.node;
    if (const auto* text = node.as_string()) {
      if (text->get() != "auto") {
        out_.At(leaf, std::format(R"({} must be "auto" or an array of selectors, not "{}")", key,
                                  text->get()));
      }
      return;
    }
    const auto* array = node.as_array();
    if (array == nullptr) {
      out_.At(leaf, std::format("{} must be \"auto\" or an array of selectors, not {}", key,
                                TypeName(node)));
      return;
    }
    if (array->empty()) {
      out_.At(leaf,
              std::format(
                  "{} must not be an empty array; write \"auto\" for the validated interconnects",
                  key));
      return;
    }
    std::set<std::string> seen;
    for (const toml::node& item : *array) {
      const auto* text = item.as_string();
      if (text == nullptr) {
        out_.At(leaf.file, item,
                std::format("{} holds {}, not a selector string", key, TypeName(item)));
        continue;
      }
      if (auto problem = SelectorProblem(text->get())) {
        out_.At(leaf.file, item, std::format("{} entry \"{}\" {}", key, text->get(), *problem));
        continue;
      }
      if (!seen.insert(text->get()).second) {
        out_.At(leaf.file, item, std::format("{} lists \"{}\" twice", key, text->get()));
        continue;
      }
      interfaces_.push_back(text->get());
    }
  }

  void PeerScope(const Entry& leaf, const std::string& id, const toml::node& value) {
    member_ = true;
    const std::string key = KeyText({"control", "peer_scopes", id});
    if (!IsUuid(id)) {
      out_.At(leaf.file, value,
              std::format("{}: the key must be a member's canonical lowercase UUID", key));
      return;
    }
    const auto* text = value.as_string();
    if (text == nullptr) {
      out_.At(leaf.file, value,
              std::format("{} must be a selector string, not {}", key, TypeName(value)));
      return;
    }
    if (auto problem = SelectorProblem(text->get())) {
      out_.At(leaf.file, value, std::format("{} \"{}\" {}", key, text->get(), *problem));
      return;
    }
    peer_scopes_[id] = text->get();
  }

  // A model's table or key was seen: its entry (for problems about the
  // model as a whole, the first place it appears), and its name checked
  // once.
  void NoteModel(const std::string& name, const Entry& at) {
    auto [it, inserted] = models_.try_emplace(name);
    if (!inserted) {
      return;
    }
    it->second.at = &at;
    it->second.entry.name = name;
    if (!IsModelName(name)) {
      it->second.bad_name = true;
      out_.At(at, std::format("{}: a model's name is 1-{} characters of a-z, 0-9, '.', '_' and "
                              "'-', starting with a letter or digit",
                              KeyText({"models", name}), kMaxModelName));
    }
  }

  // A model's key, checked against its schema (ModelKeys): its value kept
  // for the model, by key.
  void ModelLeaf(const Entry& leaf) {
    const std::string key = KeyText(leaf.path);
    NoteModel(leaf.path[1], leaf);
    const ModelKeySpec* spec = FindModelKey(leaf.path[2]);
    if (spec == nullptr) {
      out_.At(leaf, std::format("unknown key {}", key));
      return;
    }
    found_.entries[leaf.path] = &leaf;
    WorkingModel& model = models_.at(leaf.path[1]);
    auto value = ModelValue(*spec, key, *leaf.node);
    if (!value) {
      out_.At(leaf, value.error());
      return;
    }
    model.set.emplace_back(spec, &leaf);
    if (spec->type == ModelKeyType::kId || spec->type == ModelKeyType::kPath) {
      const std::string& text = std::get<std::string>(*value);
      if (spec->key == "artifact") {
        model.entry.artifact = text;
      } else if (spec->key == "composition") {
        model.entry.composition = text;
      } else if (spec->key == "drafter") {
        model.entry.drafter = text;
      } else if (spec->key == "tokenizer") {
        model.entry.tokenizer = fs::path(text);
      } else if (spec->key == "chat_template") {
        model.entry.chat_template = fs::path(text);
      }
      return;
    }
    model.entry.overrides[std::string(spec->key)] = std::move(*value);
  }

  // A value of a model's key, or the problem with it.
  static std::expected<SettingValue, std::string> ModelValue(const ModelKeySpec& spec,
                                                             const std::string& key,
                                                             const toml::node& node) {
    const auto bad = [](std::string message) { return std::unexpected(std::move(message)); };
    const auto real_range = [&](double value) -> std::optional<std::string> {
      const bool low = spec.real_open ? value <= spec.real_least : value < spec.real_least;
      if (!std::isfinite(value) || low || value > spec.real_most) {
        return spec.real_open
                   ? std::format("greater than {} and at most {}{}", spec.real_least,
                                 spec.real_most, spec.unit)
                   : std::format("from {} to {}{}", spec.real_least, spec.real_most, spec.unit);
      }
      return std::nullopt;
    };
    switch (spec.type) {
      case ModelKeyType::kId: {
        const auto* text = node.as_string();
        if (text == nullptr || !IsContentId(text->get())) {
          return bad(std::format("{} must be an ID string: 64 lowercase hexadecimal digits", key));
        }
        return SettingValue(text->get());
      }
      case ModelKeyType::kPath: {
        const auto* text = node.as_string();
        if (text == nullptr) {
          return bad(std::format("{} must be a string path, not {}", key, TypeName(node)));
        }
        if (auto problem = PathProblem(text->get(), true)) {
          return bad(std::format("{} {}", key, *problem));
        }
        return SettingValue(text->get());
      }
      case ModelKeyType::kBool: {
        const auto* value = node.as_boolean();
        if (value == nullptr) {
          return bad(std::format("{} must be a boolean, not {}", key, TypeName(node)));
        }
        return SettingValue(value->get());
      }
      case ModelKeyType::kInteger: {
        const auto* value = node.as_integer();
        if (value == nullptr) {
          return bad(std::format("{} must be an integer, not {}", key, TypeName(node)));
        }
        const std::int64_t v = value->get();
        if (v < spec.least || v > spec.most) {
          return bad(std::format("{} must be from {} to {}{}, not {}", key, spec.least, spec.most,
                                 spec.unit, v));
        }
        if (spec.multiple > 1 && v % spec.multiple != 0) {
          return bad(std::format("{} must be a multiple of {}, not {}", key, spec.multiple, v));
        }
        if (spec.power_of_two && (v & (v - 1)) != 0) {
          return bad(std::format("{} must be a power of two, not {}", key, v));
        }
        return SettingValue(v);
      }
      case ModelKeyType::kReal: {
        std::optional<double> v;
        if (const auto* real = node.as_floating_point()) {
          v = real->get();
        } else if (const auto* integer = node.as_integer()) {
          v = static_cast<double>(integer->get());
        }
        if (!v) {
          return bad(std::format("{} must be a number, not {}", key, TypeName(node)));
        }
        if (auto range = real_range(*v)) {
          return bad(std::format("{} must be {}, not {}", key, *range, *v));
        }
        return SettingValue(*v);
      }
      case ModelKeyType::kChoice: {
        const auto* text = node.as_string();
        if (text != nullptr && std::ranges::find(spec.choices, text->get()) != spec.choices.end()) {
          return SettingValue(text->get());
        }
        std::string list;
        for (std::size_t i = 0; i < spec.choices.size(); ++i) {
          if (i != 0) {
            list += i + 1 == spec.choices.size() ? " or " : ", ";
          }
          list += std::format("\"{}\"", spec.choices[i]);
        }
        return bad(std::format("{} must be {}", key, list));
      }
      case ModelKeyType::kReals: {
        const auto* array = node.as_array();
        if (array == nullptr) {
          return bad(std::format("{} must be an array of numbers, not {}", key, TypeName(node)));
        }
        if (array->empty() || array->size() > spec.most_items) {
          return bad(std::format("{} must hold 1 to {} numbers, not {}", key, spec.most_items,
                                 array->size()));
        }
        std::vector<double> values;
        for (std::size_t i = 0; i < array->size(); ++i) {
          const toml::node& item = (*array)[i];
          std::optional<double> v;
          if (const auto* real = item.as_floating_point()) {
            v = real->get();
          } else if (const auto* integer = item.as_integer()) {
            v = static_cast<double>(integer->get());
          }
          if (!v) {
            return bad(std::format("{}[{}] must be a number, not {}", key, i, TypeName(item)));
          }
          if (auto range = real_range(*v)) {
            return bad(std::format("{}[{}] must be {}, not {}", key, i, *range, *v));
          }
          values.push_back(*v);
        }
        return SettingValue(std::move(values));
      }
      case ModelKeyType::kText: {
        const auto* text = node.as_string();
        if (text == nullptr) {
          return bad(std::format("{} must be a string, not {}", key, TypeName(node)));
        }
        const std::string& s = text->get();
        if (std::cmp_greater(s.size(), spec.most)) {
          return bad(std::format("{} must be at most {} bytes, not {}", key, spec.most, s.size()));
        }
        if (base::Printable(s) != s) {
          return bad(
              std::format("{} must not contain control or invisible formatting characters", key));
        }
        return SettingValue(s);
      }
    }
    return bad(std::format("unknown key {}", key));  // not reached
  }

  // The models, each checked as a whole: exactly one of artifact and
  // composition, each key one its kind takes (ModelKeys), and no artifact
  // serving two models or its own drafter. Their number is not bounded:
  // the library may exceed memory (D-102).
  std::vector<ModelEntry> Models() {
    std::vector<ModelEntry> models;
    std::map<std::string, std::string> used;  // an ID to the model naming it
    for (auto& [name, model] : models_) {
      if (model.bad_name) {
        continue;
      }
      ModelEntry& m = model.entry;
      const std::string key = KeyText({"models", name});
      bool ok = true;
      const auto problem = [&](const std::string& message) {
        out_.At(*model.at, std::format("{}: {}", key, message));
        ok = false;
      };
      if (m.artifact.has_value() == m.composition.has_value()) {
        problem("a model names exactly one of artifact (a model) and composition (a pipeline)");
      } else {
        const std::uint8_t kind = m.artifact ? kArtifactModels : kCompositionModels;
        for (const auto& [spec, at] : model.set) {
          if ((spec->kinds & kind) == 0) {
            out_.At(*at, std::format("{} is a {}'s key, not a {}'s", KeyText(at->path),
                                     kind == kArtifactModels ? "composition" : "model artifact",
                                     kind == kArtifactModels ? "model artifact" : "composition"));
            ok = false;
          }
        }
      }
      if (m.artifact && m.drafter && *m.artifact == *m.drafter) {
        problem("an artifact cannot be its own drafter");
      }
      for (const std::optional<std::string>* id : {&m.artifact, &m.composition, &m.drafter}) {
        if (!id->has_value()) {
          continue;
        }
        const auto [other, inserted] = used.try_emplace(**id, name);
        if (!inserted && other->second != name) {
          problem(std::format("{} is also {}'s; an installed artifact serves one model", **id,
                              KeyText({"models", other->second})));
        }
      }
      if (ok) {
        models.push_back(m);
      }
    }
    return models;
  }

  // Reports a problem with a key, where it was set if it was.
  void KeyProblem(const KeyPath& path, const std::string& message) {
    if (const Entry* entry = found_.Get(path)) {
      out_.At(*entry, message);
    } else {
      out_.Document(message);
    }
  }

  std::optional<fs::path> PathValue(const KeyPath& path) const {
    const auto it = paths_.find(path);
    return it == paths_.end() ? std::nullopt : std::optional<fs::path>(it->second);
  }

  NodeConfig Build() {
    NodeConfig config;
    if (member_) {
      Membership membership;
      const auto require = [&](const KeyPath& path) {
        if (found_.Get(path) == nullptr) {
          out_.Document(
              std::format("{} is required: a cluster member (a node with cluster_file, node_id, "
                          "credentials or control set) needs all of cluster_file, node_id, "
                          "credentials.ca_file, credentials.certificate_file, "
                          "credentials.private_key_file and control.port",
                          KeyText(path)));
        }
      };
      for (const KeyPath& path :
           {KeyPath{"cluster_file"}, KeyPath{"node_id"}, KeyPath{"credentials", "ca_file"},
            KeyPath{"credentials", "certificate_file"}, KeyPath{"credentials", "private_key_file"},
            KeyPath{"control", "port"}}) {
        require(path);
      }
      membership.cluster_file = PathValue({"cluster_file"}).value_or(fs::path());
      membership.node_id = node_id_;
      membership.credentials.ca_file = PathValue({"credentials", "ca_file"}).value_or(fs::path());
      membership.credentials.certificate_file =
          PathValue({"credentials", "certificate_file"}).value_or(fs::path());
      membership.credentials.private_key_file =
          PathValue({"credentials", "private_key_file"}).value_or(fs::path());
      membership.control.port = port_;
      membership.control.interfaces = interfaces_;
      membership.control.peer_scopes = peer_scopes_;
      config.membership = std::move(membership);
    }
    config.storage = Roles();
    config.models = Models();
    if (bind_) {
      config.client.bind = *bind_;
    }
    config.client.port = client_port_.value_or(kDefaultClientPort);
    Client(config.client);
    config.memory.retention_hours = retention_hours_.value_or(kDefaultRetentionHours);
    config.memory.spill_budget_gib = spill_budget_gib_.value_or(kDefaultSpillBudgetGib);
    config.memory.keep_across_restart = keep_across_restart_.value_or(true);
    if (config.membership && config.storage.long_term) {
      for (const KeyPath& path : {KeyPath{"cluster_file"}, KeyPath{"credentials", "ca_file"},
                                  KeyPath{"credentials", "certificate_file"},
                                  KeyPath{"credentials", "private_key_file"}}) {
        const auto value = PathValue(path);
        if (value && Within(*config.storage.long_term, *value)) {
          KeyProblem(path, std::format("{} must not lie inside storage.long_term ({})",
                                       KeyText(path), config.storage.long_term->string()));
        }
      }
    }
    return config;
  }

  // [client]'s limits: what was set, the rest at their defaults (D-102).
  void Client(ClientConfig& client) {
    const auto get = [&](std::string_view key) -> std::optional<std::uint64_t> {
      const auto it = client_integers_.find(std::string(key));
      return it == client_integers_.end() ? std::nullopt : std::optional(it->second);
    };
    const auto u32 = [&](std::string_view key) -> std::optional<std::uint32_t> {
      const auto v = get(key);
      return v ? std::optional(static_cast<std::uint32_t>(*v)) : std::nullopt;
    };
    client.max_connections = u32("max_connections");
    client.max_queued = u32("max_queued");
    client.queue_wait_seconds = u32("queue_wait_seconds");
    client.stall_seconds = u32("stall_seconds").value_or(kDefaultStallSeconds);
    client.stall_action = stall_action_;
    client.deadline_cap_seconds = u32("deadline_cap_seconds");
    client.idle_seconds = u32("idle_seconds").value_or(kDefaultIdleSeconds);
    client.request_inactivity_seconds =
        u32("request_inactivity_seconds").value_or(kDefaultRequestInactivitySeconds);
    client.write_inactivity_seconds = u32("write_inactivity_seconds");
    client.hang_seconds = u32("hang_seconds");
    client.request_memory_bytes = get("request_memory_bytes");
    client.max_body_bytes = get("max_body_bytes");
    client.stream_buffer_bytes = get("stream_buffer_bytes");
    if (client.max_body_bytes && client.request_memory_bytes &&
        *client.request_memory_bytes < *client.max_body_bytes) {
      KeyProblem({"client", "max_body_bytes"},
                 std::format("client.max_body_bytes ({}) must be at most client."
                             "request_memory_bytes ({}): a body is charged to that pool",
                             *client.max_body_bytes, *client.request_memory_bytes));
    }
  }

  Storage Roles() {
    Storage storage;
    storage.data_dir = PathValue({"storage", "data_dir"}).value_or(fs::path(kDefaultDataDir));
    const auto resolve = [&](std::string_view name, std::string_view fallback) {
      const fs::path value = PathValue({"storage", std::string(name)}).value_or(fs::path(fallback));
      return value.is_absolute() ? value : storage.data_dir / value;
    };
    storage.installed = resolve("installed", "models");
    storage.spill = resolve("spill", "spill");
    storage.state = resolve("state", "state");
    storage.checkpoints = resolve("checkpoints", "checkpoints");
    storage.long_term = PathValue({"storage", "long_term"});
    const auto archive = PathValue({"storage", "archive"});
    if (storage.long_term) {
      const fs::path value = archive.value_or(fs::path("archive"));
      storage.archive = value.is_absolute() ? value : *storage.long_term / value;
      if (*storage.archive == *storage.long_term || !Within(*storage.long_term, *storage.archive)) {
        KeyProblem({"storage", "archive"},
                   std::format("storage.archive ({}) must lie inside storage.long_term ({})",
                               storage.archive->string(), storage.long_term->string()));
      }
    } else if (archive) {
      KeyProblem({"storage", "archive"},
                 "storage.archive requires storage.long_term, which it lies inside");
    }

    struct Role {
      std::string_view name;
      const fs::path* path;
    };
    std::vector<Role> roles = {{.name = "installed", .path = &storage.installed},
                               {.name = "spill", .path = &storage.spill},
                               {.name = "state", .path = &storage.state},
                               {.name = "checkpoints", .path = &storage.checkpoints}};
    if (storage.archive) {
      roles.push_back({.name = "archive", .path = &*storage.archive});
    }
    for (std::size_t i = 0; i < roles.size(); ++i) {
      for (std::size_t j = i + 1; j < roles.size(); ++j) {
        const Role& a = roles[i];
        const Role& b = roles[j];
        if (*a.path == *b.path) {
          KeyProblem({"storage", std::string(b.name)},
                     std::format("storage.{} and storage.{} are the same directory, {}", a.name,
                                 b.name, a.path->string()));
        } else if (Within(*a.path, *b.path) || Within(*b.path, *a.path)) {
          const bool b_inside = Within(*a.path, *b.path);
          const Role& inner = b_inside ? b : a;
          const Role& outer = b_inside ? a : b;
          KeyProblem({"storage", std::string(inner.name)},
                     std::format(
                         "storage.{} ({}) lies inside storage.{} ({}); no role may contain another",
                         inner.name, inner.path->string(), outer.name, outer.path->string()));
        }
      }
    }
    if (storage.long_term) {
      for (const Role& role : roles) {
        if (role.name == "checkpoints" || role.name == "archive") {
          continue;  // job-only roles, which may live in the long-term store
        }
        if (Within(*storage.long_term, *role.path) || Within(*role.path, *storage.long_term)) {
          KeyProblem(
              {"storage", "long_term"},
              std::format(
                  "storage.long_term ({}) must not equal, contain or lie inside storage.{} ({})",
                  storage.long_term->string(), role.name, role.path->string()));
        }
      }
    }
    for (const Role& role : roles) {
      if (Within(anchor_, *role.path) || Within(*role.path, anchor_)) {
        KeyProblem(
            {"storage", std::string(role.name)},
            std::format(
                "storage.{} ({}) must not equal, contain or lie inside the enrollment anchor {}",
                role.name, role.path->string(), anchor_.string()));
      }
    }
    return storage;
  }

  // A model as its table and keys were seen.
  struct WorkingModel {
    ModelEntry entry;
    const Entry* at = nullptr;  // where it first appears
    bool bad_name = false;
    // Every valid key it sets, and where.
    std::vector<std::pair<const ModelKeySpec*, const Entry*>> set;
  };

  Collector& out_;
  fs::path anchor_;
  // The entries of inline tables, which found_ points into.
  std::vector<std::unique_ptr<Entry>> inline_entries_;
  std::map<std::string, WorkingModel> models_;
  Found found_;
  bool member_ = false;
  std::map<KeyPath, fs::path> paths_;
  std::string node_id_;
  std::uint16_t port_ = 0;
  std::vector<std::string> interfaces_;
  std::map<std::string, std::string> peer_scopes_;
  std::optional<std::vector<BindEntry>> bind_;
  std::optional<std::uint16_t> client_port_;
  std::map<std::string, std::uint64_t> client_integers_;  // kClientIntegers' keys
  StallAction stall_action_ = StallAction::kReport;
  std::optional<std::uint32_t> retention_hours_;
  std::optional<std::uint32_t> spill_budget_gib_;
  std::optional<bool> keep_across_restart_;
};

constexpr std::array<std::string_view, 3> kWaveForms = {"auto", "speculative", "plain"};
constexpr std::int64_t kU32Max = 0xFFFF'FFFF;

// The schema of a [models.<name>] table (D-096, D-103): what names the
// model's files, then every setting, each with its type and range. The
// settings' sources and defaults are runtime/model_settings.h's.
constexpr std::array kModelKeys = std::to_array<ModelKeySpec>({
    {.key = "artifact",
     .type = ModelKeyType::kId,
     .kinds = kArtifactModels,
     .setting = false,
     .summary = "the installed artifact (a model)"},
    {.key = "composition",
     .type = ModelKeyType::kId,
     .kinds = kCompositionModels,
     .setting = false,
     .summary = "the installed composition (a pipeline)"},
    {.key = "drafter",
     .type = ModelKeyType::kId,
     .setting = false,
     .summary = "the speculative drafter's artifact"},
    {.key = "tokenizer",
     .type = ModelKeyType::kPath,
     .setting = false,
     .summary = "tokenizer.json, for an artifact that keeps none"},
    {.key = "chat_template",
     .type = ModelKeyType::kPath,
     .setting = false,
     .summary = "the chat template, for an artifact that keeps none"},
    {.key = "context",
     .least = kMinContext,
     .most = kU32Max,
     .unit = " tokens",
     .summary = "tokens of conversation state"},
    {.key = "speculation", .type = ModelKeyType::kBool, .summary = "decode with the drafter"},
    {.key = "prefill_chunk",
     .least = 1,
     .most = kMaxPrefillChunk,
     .unit = " rows",
     .summary = "rows of a prefill chunk (capped by the model at its context)"},
    {.key = "max_slots",
     .least = 1,
     .most = kMaxModelSlots,
     .unit = " request slots",
     .summary = "requests at once at most"},
    {.key = "prefill_floor_tok_s",
     .least = 1,
     .most = kMaxPrefillFloor,
     .unit = " tokens a second",
     .summary = "the prefill speed the chat route's allowances assume"},
    {.key = "decode_floor_tok_s",
     .least = 1,
     .most = kMaxDecodeFloor,
     .unit = " tokens a second",
     .summary = "the decode speed the chat route's allowances assume"},
    {.key = "recompute_ms_per_token",
     .type = ModelKeyType::kReal,
     .real_least = 0,
     .real_most = 1e6,
     .real_open = true,
     .unit = " milliseconds a token",
     .summary = "the reclaim order's cost of recomputing dropped state, a token"},
    {.key = "temperature",
     .type = ModelKeyType::kReal,
     .real_least = 0,
     .real_most = 2,
     .summary = "a request's temperature when it sends none (0: greedy)"},
    {.key = "top_p",
     .type = ModelKeyType::kReal,
     .real_least = 0,
     .real_most = 1,
     .real_open = true,
     .summary = "a request's top_p when it sends none"},
    {.key = "top_k",
     .least = 0,
     .most = (std::int64_t{1} << 31) - 1,
     .summary = "a request's top_k when it sends none (0: off)"},
    {.key = "min_p",
     .type = ModelKeyType::kReal,
     .real_least = 0,
     .real_most = 1,
     .summary = "a request's min_p when it sends none (0: off)"},
    {.key = "reasoning_start",
     .type = ModelKeyType::kText,
     .most = 256,
     .summary = "the token opening reasoning (empty: none)"},
    {.key = "reasoning_end",
     .type = ModelKeyType::kText,
     .most = 256,
     .summary = "the token closing reasoning (empty: none)"},
    {.key = "draft_rows",
     .architectures = "deepseek4 qwen4exp",
     .least = 1,
     .most = 16,
     .unit = " rows",
     .summary = "drafts a speculative step proposes at most"},
    {.key = "wave_form",
     .type = ModelKeyType::kChoice,
     .architectures = "deepseek4",
     .choices = kWaveForms,
     .summary = "a speculative wave's form: chosen, or forced"},
    {.key = "wave_costs",
     .type = ModelKeyType::kReals,
     .architectures = "deepseek4",
     .real_least = 0,
     .real_most = 1000,
     .most_items = kMaxWaveCostWidths,
     .summary = "draft-verify wave time over plain, widths 2 up (0: always speculate)"},
    {.key = "prefill_outa_hca",
     .type = ModelKeyType::kBool,
     .architectures = "deepseek4",
     .summary = "output-A/HCA prefill on full 4,096-row chunks (quality-qualified)"},
    {.key = "prefill_outa_hca_partial",
     .type = ModelKeyType::kBool,
     .architectures = "deepseek4",
     .summary = "and on every chunk of 64 rows or more (quality-qualified)"},
    {.key = "draft_vocab",
     .architectures = "qwen4exp",
     .least = 0,
     .most = std::int64_t{1} << 22,
     .unit = " rows",
     .summary = "the draft head's vocabulary rows (0: all)"},
    {.key = "depth_cost_ratio",
     .type = ModelKeyType::kReal,
     .architectures = "qwen4exp",
     .real_least = 0,
     .real_most = 100,
     .real_open = true,
     .summary = "a three-draft step's time over a two-draft step's"},
    {.key = "shared_wave_depth",
     .architectures = "qwen4exp",
     .least = 1,
     .most = 16,
     .unit = " drafts",
     .summary = "drafts a request in a wave of several"},
    {.key = "draft_wave_max",
     .architectures = "qwen4exp",
     .least = 1,
     .most = 16,
     .unit = " requests",
     .summary = "requests drafting in one joined wave; past it each drafts alone"},
    {.key = "wave_lanes",
     .type = ModelKeyType::kBool,
     .architectures = "qwen4exp",
     .summary = "run a wave's independent request operations on concurrent streams"},
    {.key = "wave_read_align",
     .architectures = "qwen4exp",
     .least = 256,
     .most = 65536,
     .power_of_two = true,
     .unit = " positions",
     .summary = "the state a wave reads, rounded up to this"},
    {.key = "image_size",
     .kinds = kCompositionModels,
     .least = 64,
     .most = 2048,
     .multiple = 32,
     .unit = " pixels",
     .summary = "the image's width and height"},
    {.key = "image_steps",
     .kinds = kCompositionModels,
     .least = 2,
     .most = 100,
     .unit = " steps",
     .summary = "denoising steps"},
});

template <typename T>
std::optional<T> OverrideOf(const ModelEntry& entry, std::string_view key) {
  const auto it = entry.overrides.find(key);
  if (it == entry.overrides.end()) {
    return std::nullopt;
  }
  const T* value = std::get_if<T>(&it->second);
  return value == nullptr ? std::nullopt : std::optional<T>(*value);
}

}  // namespace

std::span<const ModelKeySpec> ModelKeys() { return kModelKeys; }

const ModelKeySpec* FindModelKey(std::string_view key) {
  const auto* const it = std::ranges::find(kModelKeys, key, &ModelKeySpec::key);
  return it == kModelKeys.end() ? nullptr : it;
}

bool KeyAppliesTo(const ModelKeySpec& spec, std::string_view architecture) {
  if (spec.architectures.empty()) {
    return true;
  }
  std::string_view rest = spec.architectures;
  while (!rest.empty()) {
    const std::size_t space = rest.find(' ');
    if (rest.substr(0, space) == architecture) {
      return true;
    }
    rest = space == std::string_view::npos ? std::string_view() : rest.substr(space + 1);
  }
  return false;
}

std::optional<bool> ModelEntry::Bool(std::string_view key) const {
  return OverrideOf<bool>(*this, key);
}
std::optional<std::int64_t> ModelEntry::Integer(std::string_view key) const {
  return OverrideOf<std::int64_t>(*this, key);
}
std::optional<double> ModelEntry::Real(std::string_view key) const {
  return OverrideOf<double>(*this, key);
}
std::optional<std::string> ModelEntry::Text(std::string_view key) const {
  return OverrideOf<std::string>(*this, key);
}
std::optional<std::vector<double>> ModelEntry::Reals(std::string_view key) const {
  return OverrideOf<std::vector<double>>(*this, key);
}

std::string FormatDiagnostic(const Diagnostic& diagnostic) {
  // Values from the files reach logs and terminals: nothing in them may
  // forge a line or change how it reads.
  const std::string file = base::Printable(diagnostic.file);
  std::string message = base::Printable(diagnostic.message);
  if (file.empty()) {
    return message;
  }
  if (diagnostic.line == 0) {
    return std::format("{}: {}", file, message);
  }
  return std::format("{}:{}:{}: {}", file, diagnostic.line, diagnostic.column, message);
}

std::expected<BindEntry, std::string> ParseBindEntry(std::string_view text) {
  constexpr std::string_view kForm =
      R"(must be "loopback", "tailscale" or an address with an optional port, such as )"
      R"("127.0.0.1:8114", "0.0.0.0" or "[::1]:8114")";
  if (text == "loopback") {
    return BindEntry{.kind = BindEntry::Kind::kLoopback, .endpoint = {}};
  }
  if (text == "tailscale") {
    return BindEntry{.kind = BindEntry::Kind::kTailscale, .endpoint = {}};
  }
  BindEntry entry{.kind = BindEntry::Kind::kAddress, .endpoint = {}};
  ClientEndpoint& endpoint = entry.endpoint;
  std::string_view address;
  std::optional<std::string_view> port;
  if (text.starts_with('[')) {
    const std::size_t close = text.find(']');
    if (close == std::string_view::npos) {
      return std::unexpected(std::string(kForm));
    }
    address = text.substr(1, close - 1);
    const std::string_view rest = text.substr(close + 1);
    if (!rest.empty()) {
      if (!rest.starts_with(':')) {
        return std::unexpected(std::string(kForm));
      }
      port = rest.substr(1);
    }
    endpoint.ipv6 = true;
  } else {
    const std::size_t colon = text.find(':');
    if (colon != std::string_view::npos && text.find(':', colon + 1) != std::string_view::npos) {
      return std::unexpected(std::format("{}; an IPv6 address is written in brackets", kForm));
    }
    address = text.substr(0, colon);
    if (colon != std::string_view::npos) {
      port = text.substr(colon + 1);
    }
  }
  if (port) {
    if (port->empty() || port->size() > 5 || port->front() == '0' ||
        !std::ranges::all_of(*port, [](char c) { return c >= '0' && c <= '9'; })) {
      return std::unexpected(std::format("{}; the port is 1-65535, in decimal", kForm));
    }
    std::uint32_t number = 0;
    for (const char c : *port) {
      number = (number * 10) + static_cast<std::uint32_t>(c - '0');
    }
    if (number > 65535) {
      return std::unexpected(std::format("{}; the port is 1-65535, in decimal", kForm));
    }
    endpoint.port = static_cast<std::uint16_t>(number);
  }
  const std::string owned(address);
  std::array<char, INET6_ADDRSTRLEN> canonical{};
  if (endpoint.ipv6) {
    in6_addr six{};
    if (::inet_pton(AF_INET6, owned.c_str(), &six) != 1) {
      return std::unexpected(std::string(kForm));
    }
    if (IN6_IS_ADDR_V4MAPPED(&six)) {
      return std::unexpected("must not be an IPv4-mapped IPv6 address; write the IPv4 address");
    }
    (void)::inet_ntop(AF_INET6, &six, canonical.data(), canonical.size());
  } else {
    in_addr four{};
    if (::inet_pton(AF_INET, owned.c_str(), &four) != 1) {
      return std::unexpected(std::string(kForm));
    }
    (void)::inet_ntop(AF_INET, &four, canonical.data(), canonical.size());
  }
  endpoint.address = canonical.data();
  return entry;
}

std::expected<NodeConfig, std::vector<Diagnostic>> ParseNodeConfig(
    std::span<const SourceText> files, const fs::path& anchor) {
  Collector out(files);
  std::size_t total = 0;
  for (const SourceText& file : files) {
    total += file.text.size();
  }
  if (total > kDocumentLimit) {
    out.Document(std::format("the configuration is {} bytes, more than the {}-byte limit", total,
                             kDocumentLimit));
    return std::unexpected(out.Take());
  }

  // Each file parses on its own; its tables live as long as the entries
  // that point into them.
  std::vector<toml::table> documents;
  documents.reserve(files.size());
  std::vector<std::size_t> parsed;  // file index of each document
  for (std::size_t i = 0; i < files.size(); ++i) {
    toml::parse_result result =
        toml::parse(std::string_view(files[i].text), std::string_view(files[i].name));
    if (!result) {
      const toml::parse_error& error = result.error();
      out.Position(i, error.source().begin, std::string(error.description()));
      continue;
    }
    documents.push_back(std::move(result).table());
    parsed.push_back(i);
  }

  std::vector<Entry> leaves;
  std::vector<Entry> tables;
  for (std::size_t d = 0; d < documents.size(); ++d) {
    const std::size_t file = parsed[d];
    const toml::table& document = documents[d];
    // Every file states the schema version, and they agree (D-063).
    const toml::node* version = document.get("schema_version");
    if (version == nullptr) {
      out.File(file,
               std::format("schema_version is missing; every file of the configuration starts with "
                           "schema_version = {}",
                           kSchemaVersion));
    } else if (const auto* number = version->as_integer(); number == nullptr) {
      out.At(file, *version,
             std::format("schema_version must be an integer, not {}", TypeName(*version)));
    } else if (number->get() != kSchemaVersion) {
      out.At(file, *version,
             std::format("schema_version {} is not supported; this build reads version {}",
                         number->get(), kSchemaVersion));
    }
    KeyPath path;
    std::vector<Entry> file_leaves;
    Flatten(document, path, file, file_leaves, tables);
    for (Entry& leaf : file_leaves) {
      if (leaf.path != KeyPath{"schema_version"}) {
        leaves.push_back(std::move(leaf));
      }
    }
  }

  // Every key has one owning file: a value set in two files, or a table
  // one file writes whole (inline) while another adds to it, is fatal.
  std::vector<Entry> merged;
  std::map<KeyPath, const Entry*> owners;
  // The owned key that clashes with path: path itself, a key above it, or
  // one below it (which sorts right after it).
  const auto clash = [&owners](const KeyPath& path) -> const Entry* {
    for (std::size_t n = 1; n <= path.size(); ++n) {
      const auto it =
          owners.find(KeyPath(path.begin(), path.begin() + static_cast<std::ptrdiff_t>(n)));
      if (it != owners.end()) {
        return it->second;
      }
    }
    const auto below = owners.lower_bound(path);
    return below != owners.end() && IsPrefix(path, below->first) ? below->second : nullptr;
  };
  for (const Entry& leaf : leaves) {
    if (const Entry* other = clash(leaf.path)) {
      out.At(leaf, std::format("{} is also set in {} (as {}); each key belongs to exactly one file",
                               KeyText(leaf.path), out.Name(other->file), KeyText(other->path)));
      continue;
    }
    owners[leaf.path] = &leaf;
    merged.push_back(leaf);
  }
  for (const Entry& table : tables) {
    for (std::size_t n = 1; n <= table.path.size(); ++n) {
      const auto it = owners.find(
          KeyPath(table.path.begin(), table.path.begin() + static_cast<std::ptrdiff_t>(n)));
      if (it != owners.end() && it->second->file != table.file) {
        out.At(table,
               std::format("{} is a table here, but {} sets {} as one value", KeyText(table.path),
                           out.Name(it->second->file), KeyText(it->first)));
        break;
      }
    }
  }

  Validator validator(out, anchor);
  NodeConfig config = validator.Run(merged, tables);
  if (!out.empty()) {
    // In reading order: by file, then position; the document's own last.
    std::vector<Diagnostic> diagnostics = out.Take();
    std::map<std::string_view, std::size_t> order;
    for (std::size_t i = 0; i < files.size(); ++i) {
      order.emplace(files[i].name, i);
    }
    const auto rank = [&order, &files](const Diagnostic& d) {
      const auto it = order.find(d.file);
      return std::tuple(it == order.end() ? files.size() : it->second, d.line, d.column);
    };
    std::ranges::stable_sort(diagnostics, [&rank](const Diagnostic& a, const Diagnostic& b) {
      return rank(a) < rank(b);
    });
    if (diagnostics.size() > kDiagnosticLimit) {
      const std::size_t more = diagnostics.size() - kDiagnosticLimit;
      diagnostics.resize(kDiagnosticLimit);
      diagnostics.push_back({.file = {}, .message = std::format("and {} more problems", more)});
    }
    return std::unexpected(std::move(diagnostics));
  }
  for (const SourceText& file : files) {
    config.files.emplace_back(file.name);
  }
  return config;
}

namespace {

// A configuration file after the walk approved its path: opened without
// following a link, it must be the same regular file, owned by root or
// the trusted user and writable by nobody else.
std::expected<std::string, std::string> ReadTrusted(int dir, const std::string& name,
                                                    const struct stat* expected, uid_t trusted,
                                                    std::size_t limit) {
  const int fd =
      ::openat(dir, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NOCTTY | O_NONBLOCK);
  if (fd < 0) {
    const int error = errno;
    return std::unexpected(error == ELOOP ? "is a symbolic link"
                                          : "cannot be opened: " + Errno(error));
  }
  struct stat status{};
  std::string text;
  std::string problem;
  if (::fstat(fd, &status) != 0) {
    problem = "cannot be examined: " + Errno(errno);
  } else if (!S_ISREG(status.st_mode)) {
    problem = "is not a regular file";
  } else if (expected != nullptr &&
             (status.st_dev != expected->st_dev || status.st_ino != expected->st_ino)) {
    problem = "changed while it was being checked";
  } else if (status.st_nlink != 1) {
    // Another name for it could sit in a directory someone else can write.
    problem = std::format("has {} hard links; a configuration file must have one", status.st_nlink);
  } else if (status.st_uid != 0 && status.st_uid != trusted) {
    problem = std::format("is owned by uid {}, not root{}", status.st_uid,
                          trusted == 0 ? std::string() : std::format(" or uid {}", trusted));
  } else if (platform::OthersCanWrite(status, trusted, fd)) {
    problem = std::format("can be written by users other than root and its owner (mode {})",
                          platform::OctalMode(status.st_mode));
  } else {
    // Room for what fstat reports (never more than the limit allows) and
    // one byte to see past the limit; it grows only if the file does.
    text.resize(std::min(static_cast<std::size_t>(std::max<off_t>(status.st_size, 0)), limit) + 1);
    std::size_t size = 0;
    while (size < text.size()) {
      const ssize_t got = ::read(fd, text.data() + size, text.size() - size);
      if (got < 0) {
        if (errno == EINTR) {
          continue;
        }
        problem = "cannot be read: " + Errno(errno);
        break;
      }
      if (got == 0) {
        break;
      }
      size += static_cast<std::size_t>(got);
      if (size == text.size() && size <= limit) {
        text.resize(std::min(text.size() * 2, limit + 1));
      }
    }
    if (problem.empty() && size > limit) {
      problem =
          std::format("would take the configuration past its {}-byte limit ({} bytes were left)",
                      kDocumentLimit, limit);
    }
    text.resize(size);
    text.shrink_to_fit();
  }
  (void)::close(fd);  // read-only
  if (!problem.empty()) {
    return std::unexpected(problem);
  }
  return text;
}

}  // namespace

std::expected<NodeConfig, std::vector<Diagnostic>> LoadNodeConfig(const LoadOptions& options) {
  std::vector<Diagnostic> problems;
  const auto fail = [&](const fs::path& file, std::string message) {
    problems.push_back({.file = file.string(), .message = std::move(message)});
  };
  std::error_code error;
  const fs::path main = fs::absolute(options.main_file, error).lexically_normal();
  if (error || main.extension() != ".toml" || main.filename() == ".toml") {
    fail(options.main_file, "the configuration file's name must end in .toml");
    return std::unexpected(problems);
  }
  fs::path dropins = main;
  dropins.replace_extension(".d");

  std::vector<SourceText> files;
  std::size_t total = 0;
  const auto read = [&](int dir, const std::string& name, const fs::path& shown,
                        const struct stat* status) {
    // Each file may use what the document's limit has left.
    const std::size_t budget = total < kDocumentLimit ? kDocumentLimit - total : 0;
    auto text = ReadTrusted(dir, name, status, options.trusted_uid, budget);
    if (!text) {
      fail(shown, text.error());
      return;
    }
    total += text->size();
    files.push_back({.name = shown.string(), .text = std::move(*text)});
  };

  auto walked = platform::WalkTrusted(main, options.trusted_uid, false);
  if (!walked) {
    fail(main, walked.error());
  } else if (!walked->exists) {
    if (!options.main_file_optional) {
      fail(main, "does not exist");
    }
  } else {
    read(AT_FDCWD, walked->resolved.string(), main, &walked->status);
  }

  auto dir_walk = platform::WalkTrusted(dropins, options.trusted_uid, false);
  if (!dir_walk) {
    fail(dropins, dir_walk.error());
  } else if (dir_walk->exists) {
    const int dir =
        ::open(dir_walk->resolved.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    struct stat status{};
    if (dir < 0 || ::fstat(dir, &status) != 0) {
      fail(dropins, dir < 0 && errno == ENOTDIR ? "is not a directory"
                                                : "cannot be opened: " + Errno(errno));
    } else if (status.st_dev != dir_walk->status.st_dev ||
               status.st_ino != dir_walk->status.st_ino) {
      fail(dropins, "changed while it was being checked");
    } else if (auto private_dir =
                   platform::CheckPrivateDirectory(dropins, status, options.trusted_uid);
               !private_dir) {
      fail(dropins, private_dir.error());
    } else {
      // Regular "*.toml" files, not dotfiles, in bytewise order; other
      // names (such as ".dpkg-old" leftovers) are skipped.
      std::vector<std::string> names;
      const int listing = ::dup(dir);
      DIR* stream = listing < 0 ? nullptr : ::fdopendir(listing);
      if (stream == nullptr) {
        if (listing >= 0) {
          (void)::close(listing);
        }
        fail(dropins, "cannot be listed: " + Errno(errno));
      } else {
        errno = 0;
        while (const dirent* entry =
                   ::readdir(stream)) {  // NOLINT(concurrency-mt-unsafe): one reader
          const std::string name = entry->d_name;
          if (!name.starts_with('.') && name.size() > 5 && name.ends_with(".toml")) {
            names.push_back(name);
          }
        }
        if (errno != 0) {
          fail(dropins, "cannot be listed: " + Errno(errno));
        }
        (void)::closedir(stream);
      }
      std::ranges::sort(names);
      if (names.size() > kDropInLimit) {
        fail(dropins, std::format("holds {} fragments; the configuration takes at most {}",
                                  names.size(), kDropInLimit));
        names.clear();
      }
      for (const std::string& name : names) {
        if (total > kDocumentLimit) {
          break;
        }
        read(dir, name, dropins / name, nullptr);
      }
    }
    if (dir >= 0) {
      (void)::close(dir);
    }
  }
  if (!problems.empty()) {
    return std::unexpected(problems);
  }
  return ParseNodeConfig(files, options.anchor);
}

}  // namespace jitllm::config
