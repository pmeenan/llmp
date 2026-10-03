// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The node's configuration (D-063, D-073): one logical document of strict
// TOML 1.0, a main file plus the fragments in its drop-in directory, in
// cluster-design.md's node-local schema version 2 extended with [storage]
// and the models the node serves ([models.<name>], D-096).
//
// Loading reads the files, refusing any that users other than root and the
// runtime's user could replace (platform/path_trust.h), parses each on its
// own, merges them so that every key has exactly one owning file, and
// validates the result against the schema. Every problem found is
// reported, not just the first: a file that does not parse contributes its
// syntax error, and the rest are still checked.

#ifndef JITLLM_CONFIG_NODE_CONFIG_H_
#define JITLLM_CONFIG_NODE_CONFIG_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/surface_versions.h"

namespace jitllm::config {

inline constexpr std::int64_t kSchemaVersion = surface::kConfigSchemaVersion;
// The merged document's size limit (cluster-design.md).
inline constexpr std::size_t kDocumentLimit = std::size_t{1} << 20;
inline constexpr std::string_view kDefaultConfigFile = "/etc/jitllm/jitllm.toml";
// The enrollment anchor's fixed path when packaged (D-063).
inline constexpr std::string_view kDefaultAnchor = "/var/lib/jitllm/enrollment";
inline constexpr std::string_view kDefaultDataDir = "/var/lib/jitllm";
inline constexpr std::string_view kLimitsProfile = "initial-v2";
// At most this many fragments form the document.
inline constexpr std::size_t kDropInLimit = 256;
// At most this many problems are reported, then a count of the rest.
inline constexpr std::size_t kDiagnosticLimit = 100;

// The storage roles (D-054, D-055, D-063), as absolute paths in normal
// form: relative values resolved against data_dir, or long_term for
// archive. Nothing here has touched the filesystem.
struct Storage {
  std::filesystem::path data_dir;
  std::filesystem::path installed;
  std::filesystem::path spill;
  std::filesystem::path state;
  std::filesystem::path checkpoints;
  std::optional<std::filesystem::path> long_term;
  std::optional<std::filesystem::path> archive;
};

// A local interface selector: "ifname:<name>" or
// "port:<phys_switch_id>/<phys_port_name>" (cluster-design.md).
struct Control {
  std::uint16_t port = 0;
  // Empty means "auto": the validated interconnects.
  std::vector<std::string> interfaces;
  // Member UUID to a local interface selector.
  std::map<std::string, std::string> peer_scopes;
};

struct Credentials {
  std::filesystem::path ca_file;
  std::filesystem::path certificate_file;
  std::filesystem::path private_key_file;
};

// The keys a cluster member adds; a standalone node has none of them.
struct Membership {
  std::filesystem::path cluster_file;
  std::string node_id;
  Credentials credentials;
  Control control;
};

// The models a node serves (D-096): `[models.<name>]`, one table a model.
inline constexpr std::size_t kMaxModels = 16;
inline constexpr std::size_t kMaxModelName = 64;
// A model's conversation state, in tokens: the default, and its bounds.
inline constexpr std::uint32_t kDefaultContext = 262144;
inline constexpr std::uint32_t kMinContext = 512;
inline constexpr std::uint32_t kMaxContext = 1048576;
// A prefill chunk's rows, when a model's are configured.
inline constexpr std::uint32_t kMaxPrefillChunk = 262144;
// The throughput floors, in tokens a second, that the chat route figures a
// model's work at (docs/runtime-serving.md#progress-and-deadlines): the
// defaults, and the bounds of a configured one.
inline constexpr std::uint32_t kDefaultPrefillFloor = 100;
inline constexpr std::uint32_t kMaxPrefillFloor = 1'000'000;
inline constexpr std::uint32_t kDefaultDecodeFloor = 5;
inline constexpr std::uint32_t kMaxDecodeFloor = 100'000;

// How a speculative model steps a wave of concurrent requests
// (docs/runtime-serving.md): chosen per wave from counted acceptance, or
// always draft-verify, or always plain decode (exactness controls).
enum class WaveForm : std::uint8_t { kAuto, kSpeculative, kPlain };

// One model: the name the CLI (and later the API) asks for, and what the
// installed store holds for it. Nothing here has touched the store.
struct ModelEntry {
  // 1-64 characters of [a-z0-9._-], starting with a letter or digit.
  std::string name;
  // Exactly one of: an installed artifact's ID (a model of one component),
  // or an installed composition's ID (a pipeline, D-089). Each is 64
  // lowercase hex digits, the directory's name under storage.installed.
  std::optional<std::string> artifact;
  std::optional<std::string> composition;
  // With an artifact: its speculative drafter's artifact (DSpark, MTP;
  // D-089's drafter binding). Speculation is then the default decode.
  std::optional<std::string> drafter;
  bool speculation = true;
  // With an artifact: a speculative model's form for waves of two or more
  // requests, "auto" (the default), "speculative" or "plain". DeepSeek's
  // waves take it; a model without waves of both forms ignores it.
  WaveForm wave_form = WaveForm::kAuto;
  // With an artifact: the tokens of conversation state its runner holds.
  std::uint32_t context = kDefaultContext;
  // With an artifact: the rows of a prefill chunk (1 to kMaxPrefillChunk);
  // absent, the runtime's default for the model. Either way at most what
  // the model allows at its context (docs/runtime-serving.md#prefill-chunks).
  std::optional<std::uint32_t> prefill_chunk;
  // With an artifact: conservative prefill and decode throughputs (tokens
  // a second, 1 to kMaxPrefillFloor and kMaxDecodeFloor) from which the
  // chat route figures how long a prefill chunk, and a non-streaming
  // request, may take (docs/runtime-serving.md#progress-and-deadlines).
  std::uint32_t prefill_floor_tok_s = kDefaultPrefillFloor;
  std::uint32_t decode_floor_tok_s = kDefaultDecodeFloor;
  // With an artifact whose kept metadata has no tokenizer or chat template
  // (M3's Qwen3.8 import kept config.json only): the checkpoint's
  // tokenizer.json and chat template, absolute paths the runtime reads
  // under the same trust rules as the configuration.
  std::optional<std::filesystem::path> tokenizer;
  std::optional<std::filesystem::path> chat_template;
};

// The chat route's listener (M3's minimal /v1/chat/completions; D-097 as
// the owner amended it on 2026-09-28): `[client]`.
//   bind             a string or an array of 1 to 16 strings, each
//                    "loopback" (127.0.0.1 and [::1]), "tailscale" (the
//                    node's tailnet addresses, found at startup) or an
//                    address with an optional port: "192.168.1.5",
//                    "0.0.0.0:8114", "[::]", "[fd00::5]:9000". Default
//                    ["loopback", "tailscale"]. Any address is accepted;
//                    none needs authentication (D-014's owner note), and
//                    one that is neither loopback nor the tailnet is named
//                    at startup as served without it.
//   port             the port of "loopback", "tailscale" and an address
//                    written without one: 1-65535, default 8114
//   max_connections  open connections, idle ones included: 1-65536,
//                    default 1024
//   max_queued       chat requests waiting behind the running one: 1-1024,
//                    default 64
//   stall_seconds    how long the backend may make no progress (no prefill
//                    chunk, decode step or swap ending) before the request
//                    fails and the backend is marked unhealthy: 30-3600,
//                    default 120 (D-097's owner note of 2026-09-29)
//   deadline_cap_seconds  the most a non-streaming request's scaled
//                    deadline may be: 60-86400, default 14400 (4 hours)
inline constexpr std::uint16_t kDefaultClientPort = 8114;
inline constexpr std::uint32_t kDefaultMaxConnections = 1024;
inline constexpr std::uint32_t kMaxConnectionsCeiling = 65536;
inline constexpr std::uint32_t kDefaultMaxQueued = 64;
inline constexpr std::uint32_t kMaxQueuedCeiling = 1024;
inline constexpr std::uint32_t kDefaultStallSeconds = 120;
inline constexpr std::uint32_t kMinStallSeconds = 30;
inline constexpr std::uint32_t kMaxStallSeconds = 3600;
inline constexpr std::uint32_t kDefaultDeadlineCapSeconds = 14400;
inline constexpr std::uint32_t kMinDeadlineCapSeconds = 60;
inline constexpr std::uint32_t kMaxDeadlineCapSeconds = 86400;
inline constexpr std::size_t kMaxBindEntries = 16;

struct ClientEndpoint {
  std::string address;  // canonical text, as inet_ntop writes it ("127.0.0.1", "::1")
  bool ipv6 = false;
  std::uint16_t port = 0;
};

struct BindEntry {
  enum class Kind : std::uint8_t { kLoopback, kTailscale, kAddress };
  Kind kind = Kind::kLoopback;
  // kAddress: the address, and its port (0: [client] port).
  ClientEndpoint endpoint;
};

struct ClientConfig {
  std::vector<BindEntry> bind = {{.kind = BindEntry::Kind::kLoopback, .endpoint = {}},
                                 {.kind = BindEntry::Kind::kTailscale, .endpoint = {}}};
  std::uint16_t port = kDefaultClientPort;
  std::uint32_t max_connections = kDefaultMaxConnections;
  std::uint32_t max_queued = kDefaultMaxQueued;
  std::uint32_t stall_seconds = kDefaultStallSeconds;
  std::uint32_t deadline_cap_seconds = kDefaultDeadlineCapSeconds;
};

// Conversation state under memory pressure ([memory]; D-055 as amended
// 2026-10-02): how long an idle conversation's state, resident or spilled,
// and its turn checkpoints stay reusable, and how much spilled state the
// spill directory may hold before the least recently used is deleted.
inline constexpr std::uint32_t kDefaultRetentionHours = 24;
inline constexpr std::uint32_t kMaxRetentionHours = 8760;
inline constexpr std::uint32_t kDefaultSpillBudgetGib = 128;
inline constexpr std::uint32_t kMaxSpillBudgetGib = 1048576;

struct MemoryConfig {
  std::uint32_t retention_hours = kDefaultRetentionHours;
  // 0: idle conversations are never kept spilled (dropped instead).
  std::uint32_t spill_budget_gib = kDefaultSpillBudgetGib;
};

// Parses one bind entry ("loopback", "tailscale", "127.0.0.1:8114",
// "[::]", ...); the error says what is wrong. A port is 1-65535, in
// decimal; an IPv6 address is bracketed; IPv4-mapped IPv6 is refused.
std::expected<BindEntry, std::string> ParseBindEntry(std::string_view text);

struct NodeConfig {
  std::optional<Membership> membership;
  // Where and how the service listens for the chat route.
  ClientConfig client;
  MemoryConfig memory;
  std::string limits_profile{kLimitsProfile};
  Storage storage;
  // Sorted by name.
  std::vector<ModelEntry> models;
  // The files the document was formed from, in the order read. Empty when
  // the built-in defaults apply.
  std::vector<std::filesystem::path> files;
};

// One problem, located where the document allows: a file (empty for the
// document as a whole) and a 1-based line and column (0 when unknown).
struct Diagnostic {
  std::string file;
  std::uint32_t line = 0;
  std::uint32_t column = 0;
  std::string message;
};

// "file:line:column: message", with what is unknown left out.
std::string FormatDiagnostic(const Diagnostic& diagnostic);

// One file's text and the name its diagnostics use.
struct SourceText {
  std::string name;
  std::string text;
};

// Parses, merges and validates files already read, in order; the anchor
// is the enrollment anchor's path, which no storage role may equal,
// contain or lie inside. No files at all yields the built-in defaults.
std::expected<NodeConfig, std::vector<Diagnostic>> ParseNodeConfig(
    std::span<const SourceText> files, const std::filesystem::path& anchor);

struct LoadOptions {
  // The main file; its drop-in directory is the same path with ".toml"
  // replaced by ".d". It must end in ".toml".
  std::filesystem::path main_file{kDefaultConfigFile};
  // Only the default main file may be absent (its fragments alone, or
  // nothing at all, then form the document; D-063): its reader says so.
  bool main_file_optional = false;
  std::filesystem::path anchor{kDefaultAnchor};
  // Besides root, the user whose files are trusted: the runtime's.
  uid_t trusted_uid = 0;
};

// Reads the main file and every regular "*.toml" file in the drop-in
// directory, in bytewise order, skipping other names and not recursing,
// then parses them as ParseNodeConfig does.
std::expected<NodeConfig, std::vector<Diagnostic>> LoadNodeConfig(const LoadOptions& options);

}  // namespace jitllm::config

#endif  // JITLLM_CONFIG_NODE_CONFIG_H_
