// SPDX-FileCopyrightText: 2026 llmpalooza contributors
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

#ifndef LLMP_CONFIG_NODE_CONFIG_H_
#define LLMP_CONFIG_NODE_CONFIG_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "base/surface_versions.h"

namespace llmp::config {

inline constexpr std::int64_t kSchemaVersion = surface::kConfigSchemaVersion;
// The merged document's size limit (cluster-design.md).
inline constexpr std::size_t kDocumentLimit = std::size_t{1} << 20;
inline constexpr std::string_view kDefaultConfigFile = "/etc/llmp/llmp.toml";
// The enrollment anchor's fixed path when packaged (D-063).
inline constexpr std::string_view kDefaultAnchor = "/var/lib/llmp/enrollment";
inline constexpr std::string_view kDefaultDataDir = "/var/lib/llmp";
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

// The models a node serves (D-096): `[models.<name>]`, one table a model,
// as many as the configuration names (the library may exceed memory,
// D-102).
inline constexpr std::size_t kMaxModelName = 64;
// A model's conversation state, in tokens: the default, and its least.
// There is no generic ceiling (D-102): registration checks each
// checkpoint's own (runtime/model_settings.h); a 32-bit count is the type's.
inline constexpr std::uint32_t kDefaultContext = 262144;
inline constexpr std::uint32_t kMinContext = 512;
// A prefill chunk's rows, when a model's are configured.
inline constexpr std::uint32_t kMaxPrefillChunk = 262144;
// The throughput floors, in tokens a second, that the chat route figures a
// model's work at (docs/runtime-serving.md#progress-and-deadlines): the
// defaults, and the bounds of a configured one.
inline constexpr std::uint32_t kDefaultPrefillFloor = 100;
inline constexpr std::uint32_t kMaxPrefillFloor = 1'000'000;
inline constexpr std::uint32_t kDefaultDecodeFloor = 5;
inline constexpr std::uint32_t kMaxDecodeFloor = 100'000;
// A model's concurrent request slots, when configured: at most the joined
// products' sixteen rows (engine/request_cohort.h kMaxRequestSlots;
// docs/runtime-serving.md#request-slots).
inline constexpr std::uint32_t kMaxModelSlots = 16;
// wave_costs' widths: 2 to 8, a speculative model's widest costed wave
// (execution/adaptive_wave_mode.h kMaxWidth).
inline constexpr std::size_t kMaxWaveCostWidths = 7;

// How a speculative model steps a wave of concurrent requests
// (docs/runtime-serving.md): chosen per wave from counted acceptance, or
// always draft-verify, or always plain decode (exactness controls).
enum class WaveForm : std::uint8_t { kAuto, kSpeculative, kPlain };

// A model's settings (D-103): every one resolves at registration as a
// default derived from the artifact, a calibration on this machine, or the
// owner's override in [models.<name>] (runtime/model_settings.h). The
// override's value as its table wrote it, checked against its key's schema
// (ModelKeys); a real key's integer is read as a real.
using SettingValue = std::variant<bool, std::int64_t, double, std::string, std::vector<double>>;

// What a [models.<name>] key is: its value's type, and for a setting its
// range. The table below (ModelKeys) is the whole schema of a model's
// table: a key not in it is refused, and so is one the model's kind does
// not take.
enum class ModelKeyType : std::uint8_t {
  kId,       // 64 lowercase hex digits: an installed artifact or composition
  kPath,     // an absolute path in normal form
  kBool,     // true or false
  kInteger,  // least to most, a multiple of `multiple`, a power of two if `power_of_two`
  kReal,     // real_least (exclusive if real_open) to real_most; an integer is a real
  kChoice,   // one of `choices`
  kReals,    // 1 to most_items reals, each real_least to real_most
  kText,     // a string of at most `most` bytes, printable, without control characters
};

// Which models take a key: an artifact's (an LLM), a composition's (a
// pipeline), or both.
inline constexpr std::uint8_t kArtifactModels = 1;
inline constexpr std::uint8_t kCompositionModels = 2;

// Designated initializers in the table may omit any member (NOLINT).
// NOLINTBEGIN(readability-redundant-member-init)
struct ModelKeySpec {
  std::string_view key{};
  ModelKeyType type = ModelKeyType::kInteger;
  std::uint8_t kinds = kArtifactModels;
  // False for what names a model's files (artifact, composition, drafter,
  // tokenizer, chat_template); true for a setting the runtime resolves.
  bool setting = true;
  // The architectures that use the setting, space-separated ("deepseek4",
  // "qwen4exp"); empty, every one of its kinds. Registration logs an
  // override that the model's architecture does not use, and ignores it.
  std::string_view architectures{};
  std::int64_t least = 0;
  std::int64_t most = 0;
  std::int64_t multiple = 1;
  bool power_of_two = false;
  double real_least = 0;
  double real_most = 0;
  bool real_open = false;  // real_least itself excluded
  std::size_t most_items = 0;
  std::span<const std::string_view> choices{};
  std::string_view unit{};  // for messages: " tokens", " rows", ...
  // What it sets, in a few words (the settings command's listing).
  std::string_view summary{};
};
// NOLINTEND(readability-redundant-member-init)

// Every [models.<name>] key, in the order the settings command lists them.
std::span<const ModelKeySpec> ModelKeys();
const ModelKeySpec* FindModelKey(std::string_view key);
// Whether `spec` applies to a model of `architecture` (its kinds aside).
bool KeyAppliesTo(const ModelKeySpec& spec, std::string_view architecture);

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
  // With an artifact whose kept metadata has no tokenizer or chat template
  // (M3's Qwen3.8 import kept config.json only): the checkpoint's
  // tokenizer.json and chat template, absolute paths the runtime reads
  // under the same trust rules as the configuration.
  std::optional<std::filesystem::path> tokenizer;
  std::optional<std::filesystem::path> chat_template;
  // Every setting its table sets, by key (ModelKeys' settings): the
  // owner's overrides (D-103), each of its key's type and in its range.
  std::map<std::string, SettingValue, std::less<>> overrides;

  // An override's value, if set (of the key's type, as ModelKeys says).
  std::optional<bool> Bool(std::string_view key) const;
  std::optional<std::int64_t> Integer(std::string_view key) const;
  std::optional<double> Real(std::string_view key) const;
  std::optional<std::string> Text(std::string_view key) const;
  std::optional<std::vector<double>> Reals(std::string_view key) const;
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
// Every other key is a limit, permissive by default (D-102): absent, a
// count or time limit does not apply and a byte limit follows the host's
// memory (runtime/intake_limits.h). An owner may set a stricter one.
//   max_connections  open connections, idle ones included: absent, the
//                    open-file hard limit less the runtime's own files
//   max_queued       requests waiting behind the active ones: absent, no
//                    count (memory and descriptors bound them)
//   queue_wait_seconds  how long a non-streaming request may wait in the
//                    queue before a 429: absent, no limit
//   stall_seconds    how long the backend may make no progress before the
//                    watchdog reports a stall: default 120
//   stall_action     "report" (default: log, health, service status) or
//                    "fail" (the stalled requests end, what is queued and
//                    what arrives gets 503 until the backend moves)
//   deadline_cap_seconds  set, a non-streaming request's deadline: its
//                    scaled work (watchdog.h), at most this; absent, none
//   idle_seconds     a kept-alive connection idle between requests: 60
//   request_inactivity_seconds  a request's head or body with no byte
//                    arriving for this long gets a 408: default 60
//   write_inactivity_seconds  set, a client that takes no output for this
//                    long is dropped; absent, it gets backpressure only
//   hang_seconds     no progress of any kind (a unit's end, a lane's
//                    completion: a fence, a read or write, a VMM
//                    operation) while work is under way for this long,
//                    past the unit's own allowance, is a confirmed hang,
//                    which is recovered (runtime/hang_ladder.h: the work
//                    cancelled, the model reset, the process restarted
//                    only when nothing less frees it); absent, the larger
//                    of 600 s and five stall times; at least 60
//   model_turn_seconds  resident work before a pending other model pauses
//                    a cooperative cohort at its next completed unit: 30.
//                    One substitute cohort ends, then this cohort resumes
//                    ahead of later arrivals. No response length limit.
//   request_memory_bytes  the most the request memory may hold (bodies,
//                    parses, queued requests, renderings, stop matchers,
//                    unread stream output, responses); absent, the floor
//                    set apart at the start (256 MiB) and the state room
//                    the budget leaves beside the largest model, charged
//                    to the budget as requests grow (runtime/intake_limits.h)
//   max_body_bytes, stream_buffer_bytes  a request body's most and a
//                    stream's unread output before its request pauses:
//                    absent, from the request memory (and the largest
//                    context)
// Seconds are 1 to kMaxClientSeconds (hang_seconds from kMinHangSeconds),
// counts 1 to 2^32-1, bytes from their least to the type's.
inline constexpr std::uint16_t kDefaultClientPort = 8114;
inline constexpr std::uint32_t kDefaultStallSeconds = 120;
inline constexpr std::uint32_t kDefaultIdleSeconds = 60;
inline constexpr std::uint32_t kDefaultRequestInactivitySeconds = 60;
// A confirmed hang's least time: under it, a slow but healthy unit (a swap
// from a slow disk) could pass for one.
inline constexpr std::uint32_t kMinHangSeconds = 60;
// Thirty days: the watchdog's arithmetic bound on an allowance
// (runtime/watchdog.cc), and far past any wait a person means.
inline constexpr std::uint32_t kMaxClientSeconds = 2'592'000;
// The JSON parser's 32-bit offsets bound one request body (base/json.h).
inline constexpr std::uint64_t kMaxBodyCeiling = 0xFFFF'FFFFULL;
inline constexpr std::size_t kMaxBindEntries = 16;

enum class StallAction : std::uint8_t { kReport, kFail };

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
  std::optional<std::uint32_t> max_connections;
  std::optional<std::uint32_t> max_queued;
  std::optional<std::uint32_t> queue_wait_seconds;
  std::uint32_t stall_seconds = kDefaultStallSeconds;
  StallAction stall_action = StallAction::kReport;
  std::optional<std::uint32_t> deadline_cap_seconds;
  std::uint32_t idle_seconds = kDefaultIdleSeconds;
  std::uint32_t request_inactivity_seconds = kDefaultRequestInactivitySeconds;
  std::optional<std::uint32_t> write_inactivity_seconds;
  std::optional<std::uint32_t> hang_seconds;
  std::uint32_t model_turn_seconds = 30;
  std::optional<std::uint64_t> request_memory_bytes;
  std::optional<std::uint64_t> max_body_bytes;
  std::optional<std::uint64_t> stream_buffer_bytes;
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
  // Conversations wholly on disk survive a restart of the service (D-105):
  // their tokens and state stay in owner-only files beneath the spill role
  // until retention or the budget removes them. false: nothing outlives the
  // process (the spill files stay unnamed, as before D-105).
  bool keep_across_restart = true;
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

}  // namespace llmp::config

#endif  // LLMP_CONFIG_NODE_CONFIG_H_
