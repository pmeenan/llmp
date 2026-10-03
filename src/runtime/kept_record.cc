// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/kept_record.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"

namespace jitllm::runtime::kept {
namespace {

namespace json = base::json;

std::unexpected<std::string> Bad(std::string what) {
  return std::unexpected("a kept conversation's record: " + std::move(what));
}

// A time a record gives (milliseconds since the epoch, any 64-bit value an
// edited record may hold) against now: past `retention_ms` before now, or
// more than a minute after it (a clock stepped back a little is allowed),
// with no arithmetic that can overflow.
enum class When : std::uint8_t { kWithin, kExpired, kFuture };
When TimeOf(std::int64_t at, std::int64_t now_ms, std::int64_t retention_ms) {
  constexpr std::int64_t kSkewMs = 60'000;
  if (now_ms < 0 || retention_ms < 0 || at <= now_ms - retention_ms) {
    return When::kExpired;  // now_ms and retention_ms are the runtime's: small
  }
  return at > now_ms + kSkewMs ? When::kFuture : When::kWithin;
}

// ---------------------------------------------------------------- writing

void Number(std::uint64_t value, std::string& out) {
  std::array<char, 24> digits{};
  const auto [end, error] = std::to_chars(digits.data(), digits.data() + digits.size(), value);
  (void)error;
  out.append(digits.data(), end);
}

void Signed(std::int64_t value, std::string& out) {
  std::array<char, 24> digits{};
  const auto [end, error] = std::to_chars(digits.data(), digits.data() + digits.size(), value);
  (void)error;
  out.append(digits.data(), end);
}

void Quoted(std::string_view text, std::string& out) { json::AppendQuoted(text, out); }

void Key(std::string_view key, std::string& out) {
  out += '"';
  out += key;
  out += "\":";
}

void FileFields(const std::string& file, const FileId& id, std::uint64_t bytes, std::string& out) {
  Key("file", out);
  Quoted(file, out);
  out += ',';
  Key("device", out);
  Number(id.device, out);
  out += ',';
  Key("inode", out);
  Number(id.inode, out);
  out += ',';
  Key("generation", out);
  if (id.generation) {
    Number(*id.generation, out);
  } else {
    out += "null";
  }
  out += ',';
  Key("file_bytes", out);
  Number(bytes, out);
}

void Ranges(std::span<const Range> ranges, std::string& out) {
  out += '[';
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    out += i == 0 ? "[" : ",[";
    Number(ranges[i].region, out);
    out += ',';
    Number(ranges[i].offset, out);
    out += ',';
    Number(ranges[i].bytes, out);
    out += ']';
  }
  out += ']';
}

void Words(std::span<const std::uint64_t> words, std::string& out) {
  out += '[';
  for (std::size_t i = 0; i < words.size(); ++i) {
    out += std::format("{}\"{:016x}\"", i == 0 ? "" : ",", words[i]);
  }
  out += ']';
}

std::string Hex(const Digest& digest) { return base::ToHex(digest); }

// ---------------------------------------------------------------- reading

// The member `key` of an object Exactly checked, so present; were it not,
// the object itself, which none of the readers below accepts.
json::Value At(json::Value object, std::string_view key) {
  for (std::size_t i = 0; i < object.size(); ++i) {
    if (object.key(i) == key) {
      return object.member(i);
    }
  }
  return object;
}

// An object whose keys are exactly `keys`, in order.
std::expected<void, std::string> Exactly(json::Value value,
                                         std::initializer_list<std::string_view> keys,
                                         std::string_view what) {
  if (!value.is_object() || value.size() != keys.size()) {
    return Bad(std::format("{} is not an object of its {} fields", what, keys.size()));
  }
  std::size_t i = 0;
  for (const std::string_view key : keys) {
    if (value.key(i) != key) {
      return Bad(std::format("{} has {} where {} belongs", what, value.key(i), key));
    }
    ++i;
  }
  return {};
}

std::optional<std::uint64_t> Unsigned(json::Value value) {
  const std::string_view text = value.number();
  if (!value.is_number() || !value.is_integer() || text.empty() || text.front() == '-') {
    return std::nullopt;
  }
  std::uint64_t out = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
  if (error != std::errc{} || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return out;
}

template <typename T>
std::expected<T, std::string> Get(json::Value value, std::string_view what) {
  const auto got = Unsigned(value);
  if (!got || *got > std::numeric_limits<T>::max()) {
    return Bad(std::format("{} is not a number in its range", what));
  }
  return static_cast<T>(*got);
}

std::expected<std::int64_t, std::string> GetSigned(json::Value value, std::string_view what) {
  const auto got = value.int64();
  if (!value.is_number() || !value.is_integer() || !got) {
    return Bad(std::format("{} is not a whole number", what));
  }
  return *got;
}

std::expected<std::string, std::string> GetString(json::Value value, std::string_view what) {
  if (!value.is_string()) {
    return Bad(std::format("{} is not a string", what));
  }
  return std::string(value.string());
}

std::optional<Digest> ParseDigest(std::string_view hex) {
  if (hex.size() != 64) {
    return std::nullopt;
  }
  Digest digest{};
  for (std::size_t i = 0; i < digest.size(); ++i) {
    const auto pair = hex.substr(2 * i, 2);
    if (std::ranges::any_of(pair,
                            [](char c) { return (c < '0' || c > '9') && (c < 'a' || c > 'f'); })) {
      return std::nullopt;
    }
    const auto [end, error] =
        std::from_chars(pair.data(), pair.data() + pair.size(), digest[i], 16);
    if (error != std::errc{} || end != pair.data() + pair.size()) {
      return std::nullopt;
    }
  }
  return digest;
}

std::expected<Digest, std::string> GetDigest(json::Value value, std::string_view what) {
  const auto digest = value.is_string() ? ParseDigest(value.string()) : std::nullopt;
  if (!digest) {
    return Bad(std::format("{} is not a SHA-256 in lower-case hexadecimal", what));
  }
  return *digest;
}

std::expected<std::vector<std::uint64_t>, std::string> GetWords(json::Value value,
                                                                std::string_view what) {
  if (!value.is_array() || value.size() > 64) {
    return Bad(std::format("{} is not a short array", what));
  }
  std::vector<std::uint64_t> words;
  for (std::size_t i = 0; i < value.size(); ++i) {
    const json::Value word = value.at(i);
    const std::string_view hex = word.string();
    std::uint64_t out = 0;
    if (!word.is_string() || hex.size() != 16 || std::ranges::any_of(hex, [](char c) {
          return (c < '0' || c > '9') && (c < 'a' || c > 'f');
        })) {
      return Bad(std::format("{} holds a word not of 16 hexadecimal digits", what));
    }
    (void)std::from_chars(hex.data(), hex.data() + hex.size(), out, 16);
    words.push_back(out);
  }
  return words;
}

std::expected<std::vector<Range>, std::string> GetRanges(json::Value value, std::string_view what) {
  if (!value.is_array()) {
    return Bad(std::format("{} is not an array", what));
  }
  std::vector<Range> ranges;
  ranges.reserve(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    const json::Value r = value.at(i);
    if (!r.is_array() || r.size() != 3) {
      return Bad(std::format("{} holds a range that is not [region, offset, bytes]", what));
    }
    auto region = Get<std::uint32_t>(r.at(0), what);
    auto offset = Get<std::uint64_t>(r.at(1), what);
    auto bytes = Get<std::uint64_t>(r.at(2), what);
    if (!region || !offset || !bytes) {
      return Bad(std::format("{} holds a range out of its types", what));
    }
    ranges.push_back({.region = *region, .offset = *offset, .bytes = *bytes});
  }
  return ranges;
}

std::expected<FileId, std::string> GetFileId(json::Value object, std::string_view what) {
  FileId id;
  auto device = Get<std::uint64_t>(At(object, "device"), what);
  auto inode = Get<std::uint64_t>(At(object, "inode"), what);
  if (!device || !inode) {
    return Bad(std::format("{}'s file identity is not numbers", what));
  }
  id.device = *device;
  id.inode = *inode;
  const json::Value generation = At(object, "generation");
  if (!generation.is_null()) {
    auto g = Get<std::uint64_t>(generation, what);
    if (!g) {
      return std::unexpected(g.error());
    }
    id.generation = *g;
  }
  return id;
}

std::expected<Checkpoint, std::string> GetCheckpoint(json::Value value) {
  if (auto exact =
          Exactly(value,
                  {"file", "device", "inode", "generation", "file_bytes", "position",
                   "created_unix_ms", "ranges", "footprint", "cursor", "decoding", "digests"},
                  "a checkpoint");
      !exact) {
    return std::unexpected(exact.error());
  }
  Checkpoint c;
  auto file = GetString(At(value, "file"), "a checkpoint's file");
  auto id = GetFileId(value, "a checkpoint");
  auto bytes = Get<std::uint64_t>(At(value, "file_bytes"), "a checkpoint's file_bytes");
  auto position = Get<std::uint32_t>(At(value, "position"), "a checkpoint's position");
  auto created = GetSigned(At(value, "created_unix_ms"), "a checkpoint's created_unix_ms");
  auto ranges = GetRanges(At(value, "ranges"), "a checkpoint's ranges");
  auto footprint = GetRanges(At(value, "footprint"), "a checkpoint's footprint");
  auto cursor = Get<std::uint32_t>(At(value, "cursor"), "a checkpoint's cursor");
  auto decoding = GetWords(At(value, "decoding"), "a checkpoint's decoding");
  for (const std::string* error :
       {file ? nullptr : &file.error(), id ? nullptr : &id.error(),
        bytes ? nullptr : &bytes.error(), position ? nullptr : &position.error(),
        created ? nullptr : &created.error(), ranges ? nullptr : &ranges.error(),
        footprint ? nullptr : &footprint.error(), cursor ? nullptr : &cursor.error(),
        decoding ? nullptr : &decoding.error()}) {
    if (error != nullptr) {
      return std::unexpected(*error);
    }
  }
  const json::Value digests = At(value, "digests");
  if (!digests.is_array()) {
    return Bad("a checkpoint's digests are not an array");
  }
  for (std::size_t i = 0; i < digests.size(); ++i) {
    auto digest = GetDigest(digests.at(i), "a checkpoint's digest");
    if (!digest) {
      return std::unexpected(digest.error());
    }
    c.digests.push_back(*digest);
  }
  c.file = std::move(*file);
  c.id = *id;
  c.file_bytes = *bytes;
  c.position = *position;
  c.created_unix_ms = *created;
  c.ranges = std::move(*ranges);
  c.footprint = std::move(*footprint);
  c.cursor = *cursor;
  c.decoding = std::move(*decoding);
  return c;
}

constexpr std::string_view kDigestKey = R"(,"digest":")";
// The suffix: the digest's key, 64 digits, its quote and the brace.
constexpr std::size_t kSuffixBytes = kDigestKey.size() + 64 + 2;

bool Overflows(std::uint64_t a, std::uint64_t b) {
  return a > std::numeric_limits<std::uint64_t>::max() - b;
}

}  // namespace

std::string StateFileName(std::uint32_t slot) { return std::format("slot-{}.state", slot); }
std::string RecordFileName(std::uint32_t slot) { return std::format("slot-{}.record", slot); }
std::string CheckpointFileName(std::uint32_t slot, std::uint64_t serial) {
  return std::format("slot-{}.turn-{}.state", slot, serial);
}

std::optional<NameOf> ParseFileName(std::string_view name) {
  constexpr std::string_view kSlot = "slot-";
  if (!name.starts_with(kSlot)) {
    return std::nullopt;
  }
  name.remove_prefix(kSlot.size());
  NameOf out;
  const auto [end, error] = std::from_chars(name.data(), name.data() + name.size(), out.slot);
  if (error != std::errc{} || end == name.data() ||
      (name.front() == '0' && end - name.data() > 1)) {
    return std::nullopt;
  }
  name.remove_prefix(static_cast<std::size_t>(end - name.data()));
  if (name == ".state") {
    out.kind = NameOf::Kind::kState;
    return out;
  }
  if (name == ".record") {
    out.kind = NameOf::Kind::kRecord;
    return out;
  }
  constexpr std::string_view kTurn = ".turn-";
  constexpr std::string_view kState = ".state";
  if (!name.starts_with(kTurn) || !name.ends_with(kState) ||
      name.size() <= kTurn.size() + kState.size()) {
    return std::nullopt;
  }
  const std::string_view serial =
      name.substr(kTurn.size(), name.size() - kTurn.size() - kState.size());
  const auto [send, serror] =
      std::from_chars(serial.data(), serial.data() + serial.size(), out.serial);
  if (serror != std::errc{} || send != serial.data() + serial.size() ||
      (serial.front() == '0' && serial.size() > 1)) {
    return std::nullopt;
  }
  out.kind = NameOf::Kind::kCheckpoint;
  return out;
}

bool ValidDirectoryName(std::string_view id) {
  return !id.empty() && id.size() <= 128 && std::ranges::all_of(id, [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

std::string Encode(const Record& r) {
  std::string out;
  out.reserve(4096 + (r.tokens.size() * 8) + (r.extents.size() * 80));
  out += '{';
  Key("format", out);
  Quoted(kFormat, out);
  out += ',';
  Key("version", out);
  Number(kVersion, out);
  out += ',';
  Key("build", out);
  Quoted(r.identity.build, out);
  out += ',';
  Key("artifact", out);
  Quoted(r.identity.artifact, out);
  out += ',';
  Key("drafter", out);
  Quoted(r.identity.drafter, out);
  out += ',';
  Key("layout", out);
  Quoted(r.identity.layout, out);
  out += ',';
  Key("slot", out);
  Number(r.slot, out);
  out += ',';
  FileFields(r.file, r.id, r.file_bytes, out);
  out += ',';
  Key("regions", out);
  out += '[';
  for (std::size_t i = 0; i < r.regions.size(); ++i) {
    if (i != 0) {
      out += ',';
    }
    Number(r.regions[i], out);
  }
  out += "],";
  Key("extents", out);
  out += '[';
  for (std::size_t i = 0; i < r.extents.size(); ++i) {
    out += i == 0 ? "[" : ",[";
    Number(r.extents[i].region, out);
    out += ',';
    Number(r.extents[i].index, out);
    out += ",\"";
    out += Hex(r.extents[i].digest);
    out += "\"]";
  }
  out += "],";
  Key("tokens", out);
  out += '[';
  for (std::size_t i = 0; i < r.tokens.size(); ++i) {
    if (i != 0) {
      out += ',';
    }
    Signed(r.tokens[i], out);
  }
  out += "],";
  Key("cursor", out);
  Number(r.cursor, out);
  out += ',';
  Key("decoding", out);
  Words(r.decoding, out);
  out += ',';
  Key("used_unix_ms", out);
  Signed(r.used_unix_ms, out);
  out += ',';
  Key("checkpoints", out);
  out += '[';
  for (std::size_t i = 0; i < r.checkpoints.size(); ++i) {
    const Checkpoint& c = r.checkpoints[i];
    out += i == 0 ? "{" : ",{";
    FileFields(c.file, c.id, c.file_bytes, out);
    out += ',';
    Key("position", out);
    Number(c.position, out);
    out += ',';
    Key("created_unix_ms", out);
    Signed(c.created_unix_ms, out);
    out += ',';
    Key("ranges", out);
    Ranges(c.ranges, out);
    out += ',';
    Key("footprint", out);
    Ranges(c.footprint, out);
    out += ',';
    Key("cursor", out);
    Number(c.cursor, out);
    out += ',';
    Key("decoding", out);
    Words(c.decoding, out);
    out += ',';
    Key("digests", out);
    out += '[';
    for (std::size_t d = 0; d < c.digests.size(); ++d) {
      out += d == 0 ? "\"" : ",\"";
      out += Hex(c.digests[d]);
      out += '"';
    }
    out += "]}";
  }
  out += "]}";
  // The digest of the record without it, then the record with it.
  const std::string digest = Hex(base::Sha256().Update(out).Finish());
  out.pop_back();
  out += kDigestKey;
  out += digest;
  out += "\"}";
  return out;
}

std::expected<Record, std::string> Decode(std::string_view text) {
  if (text.size() > kMostRecordBytes) {
    return Bad("it is larger than any record");
  }
  if (text.size() < kSuffixBytes + 2 ||
      text.substr(text.size() - kSuffixBytes, kDigestKey.size()) != kDigestKey ||
      !text.ends_with("\"}")) {
    return Bad("it does not end with its digest");
  }
  const auto digest = ParseDigest(text.substr(text.size() - 66, 64));
  if (!digest) {
    return Bad("its digest is not a SHA-256");
  }
  std::string body(text.substr(0, text.size() - kSuffixBytes));
  body += '}';
  if (base::Sha256().Update(body).Finish() != *digest) {
    return Bad("its digest does not match its contents (cut short or changed)");
  }
  json::Limits limits;
  limits.max_bytes = kMostRecordBytes;
  limits.max_depth = 8;
  auto parsed = json::Parse(text, limits);
  if (!parsed) {
    return Bad(parsed.error().ToString());
  }
  const json::Value root = parsed->root();
  if (auto exact = Exactly(
          root,
          {"format", "version", "build",    "artifact",     "drafter",     "layout",  "slot",
           "file",   "device",  "inode",    "generation",   "file_bytes",  "regions", "extents",
           "tokens", "cursor",  "decoding", "used_unix_ms", "checkpoints", "digest"},
          "the record");
      !exact) {
    return std::unexpected(exact.error());
  }
  if (!At(root, "format").is_string() || At(root, "format").string() != kFormat) {
    return Bad("it is not a kept conversation's record");
  }
  if (auto version = Get<std::uint32_t>(At(root, "version"), "its version");
      !version || *version != kVersion) {
    return Bad(std::format("its version is not {}, the only one this build reads", kVersion));
  }
  Record r;
  auto build = GetString(At(root, "build"), "its build");
  auto artifact = GetString(At(root, "artifact"), "its artifact");
  auto drafter = GetString(At(root, "drafter"), "its drafter");
  auto layout = GetString(At(root, "layout"), "its layout");
  auto slot = Get<std::uint32_t>(At(root, "slot"), "its slot");
  auto file = GetString(At(root, "file"), "its file");
  auto id = GetFileId(root, "its file");
  auto bytes = Get<std::uint64_t>(At(root, "file_bytes"), "its file_bytes");
  auto cursor = Get<std::uint32_t>(At(root, "cursor"), "its cursor");
  auto decoding = GetWords(At(root, "decoding"), "its decoding");
  auto used = GetSigned(At(root, "used_unix_ms"), "its used_unix_ms");
  for (const std::string* error :
       {build ? nullptr : &build.error(), artifact ? nullptr : &artifact.error(),
        drafter ? nullptr : &drafter.error(), layout ? nullptr : &layout.error(),
        slot ? nullptr : &slot.error(), file ? nullptr : &file.error(), id ? nullptr : &id.error(),
        bytes ? nullptr : &bytes.error(), cursor ? nullptr : &cursor.error(),
        decoding ? nullptr : &decoding.error(), used ? nullptr : &used.error()}) {
    if (error != nullptr) {
      return std::unexpected(*error);
    }
  }
  r.identity = {.build = std::move(*build),
                .artifact = std::move(*artifact),
                .drafter = std::move(*drafter),
                .layout = std::move(*layout)};
  r.slot = *slot;
  r.file = std::move(*file);
  r.id = *id;
  r.file_bytes = *bytes;
  r.cursor = *cursor;
  r.decoding = std::move(*decoding);
  r.used_unix_ms = *used;
  const json::Value regions = At(root, "regions");
  if (!regions.is_array() || regions.size() == 0 || regions.size() > 64) {
    return Bad("its regions are not an array of 1 to 64");
  }
  for (std::size_t i = 0; i < regions.size(); ++i) {
    auto region = Get<std::uint64_t>(regions.at(i), "a region's bytes");
    if (!region) {
      return std::unexpected(region.error());
    }
    r.regions.push_back(*region);
  }
  const json::Value extents = At(root, "extents");
  if (!extents.is_array()) {
    return Bad("its extents are not an array");
  }
  r.extents.reserve(extents.size());
  for (std::size_t i = 0; i < extents.size(); ++i) {
    const json::Value e = extents.at(i);
    if (!e.is_array() || e.size() != 3) {
      return Bad("an extent is not [region, index, digest]");
    }
    auto region = Get<std::uint32_t>(e.at(0), "an extent's region");
    auto index = Get<std::uint32_t>(e.at(1), "an extent's index");
    auto d = GetDigest(e.at(2), "an extent's digest");
    if (!region || !index || !d) {
      return Bad("an extent is out of its types");
    }
    r.extents.push_back({.region = *region, .index = *index, .digest = *d});
  }
  const json::Value tokens = At(root, "tokens");
  if (!tokens.is_array()) {
    return Bad("its tokens are not an array");
  }
  r.tokens.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    auto token = Get<std::uint32_t>(tokens.at(i), "a token");
    if (!token || *token > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
      return Bad("a token is not a token ID");
    }
    r.tokens.push_back(static_cast<std::int32_t>(*token));
  }
  const json::Value checkpoints = At(root, "checkpoints");
  if (!checkpoints.is_array() || checkpoints.size() > 64) {
    return Bad("its checkpoints are not an array");
  }
  for (std::size_t i = 0; i < checkpoints.size(); ++i) {
    auto c = GetCheckpoint(checkpoints.at(i));
    if (!c) {
      return std::unexpected(c.error());
    }
    r.checkpoints.push_back(std::move(*c));
  }
  return r;
}

std::uint64_t ExtentOffset(std::span<const std::uint64_t> regions, const Extent& extent) {
  std::uint64_t base = 0;
  for (std::uint32_t i = 0; i < extent.region && i < regions.size(); ++i) {
    base += regions[i];
  }
  return base + (std::uint64_t{extent.index} * kExtentBytes);
}

std::vector<Place> CheckpointPlaces(const Checkpoint& checkpoint) {
  std::vector<Place> places;
  places.reserve(checkpoint.ranges.size());
  std::uint64_t at = 0;
  for (const Range& range : checkpoint.ranges) {
    const std::uint64_t transfer =
        (range.bytes + kFileAlignment - 1) / kFileAlignment * kFileAlignment;
    places.push_back({.offset = at, .bytes = transfer});
    at += transfer;
  }
  return places;
}

namespace {

// A range inside its region's layout, within one 2 MiB extent.
std::expected<void, std::string> InLayout(const Range& range,
                                          std::span<const std::uint64_t> layouts,
                                          std::string_view what) {
  if (range.region >= layouts.size() || range.bytes == 0 || range.bytes > kExtentBytes ||
      Overflows(range.offset, range.bytes) || range.offset + range.bytes > layouts[range.region]) {
    return Bad(std::format("{} lies outside the state's layout", what));
  }
  return {};
}

std::expected<void, std::string> CheckCheckpoint(const Checkpoint& c, const Record& r,
                                                 const Expected& e) {
  if (const auto name = ParseFileName(c.file);
      !name || name->kind != NameOf::Kind::kCheckpoint || name->slot != r.slot) {
    return Bad(std::format("checkpoint file {} is not one of slot {}'s", c.file, r.slot));
  }
  if (c.position == 0 || c.position > r.tokens.size()) {
    return Bad("a checkpoint lies past the conversation it belongs to");
  }
  switch (TimeOf(c.created_unix_ms, e.now_unix_ms, e.retention_ms)) {
    case When::kExpired:
      return Bad("a checkpoint is past its retention");
    case When::kFuture:
      return Bad("a checkpoint was made in the future");
    case When::kWithin:
      break;
  }
  if (c.digests.size() != c.ranges.size() || c.ranges.empty()) {
    return Bad("a checkpoint's digests do not match its pages");
  }
  std::uint64_t file_bytes = 0;
  for (const Range& range : c.ranges) {
    if (auto in = InLayout(range, e.layouts, "a checkpoint's page"); !in) {
      return in;
    }
    file_bytes += (range.bytes + kFileAlignment - 1) / kFileAlignment * kFileAlignment;
  }
  if (file_bytes != c.file_bytes) {
    return Bad("a checkpoint's file is not the size of its pages");
  }
  for (const Range& range : c.footprint) {
    if (auto in = InLayout(range, e.layouts, "a checkpoint's footprint"); !in) {
      return in;
    }
  }
  return {};
}

}  // namespace

std::expected<void, std::string> Check(Record& r, const Expected& e,
                                       std::vector<std::string>* dropped) {
  if (r.identity.build != e.identity.build) {
    return Bad("another build wrote it");
  }
  if (r.identity.artifact != e.identity.artifact || r.identity.drafter != e.identity.drafter) {
    return Bad("it holds another model's conversation");
  }
  if (r.identity.layout != e.identity.layout) {
    return Bad("its state layout is not this runner's");
  }
  if (r.slot != e.slot || r.file != StateFileName(e.slot)) {
    return Bad("it names another request slot");
  }
  if (r.regions != e.regions || e.layouts.size() != e.regions.size()) {
    return Bad("its state regions are not this runner's");
  }
  std::uint64_t total = 0;
  for (const std::uint64_t bytes : r.regions) {
    if (bytes % kExtentBytes != 0 || Overflows(total, bytes)) {
      return Bad("a region is not whole extents");
    }
    total += bytes;
  }
  if (r.file_bytes != total) {
    return Bad("its file is not the size of the regions");
  }
  if (r.extents.empty()) {
    return Bad("it keeps no state");
  }
  for (std::size_t i = 0; i < r.extents.size(); ++i) {
    const Extent& x = r.extents[i];
    if (x.region >= r.regions.size() ||
        std::uint64_t{x.index} * kExtentBytes >= r.regions[x.region] ||
        std::uint64_t{x.index} * kExtentBytes >= e.layouts[x.region]) {
      return Bad("an extent lies outside its region");
    }
    if (i != 0 && std::pair(r.extents[i - 1].region, r.extents[i - 1].index) >=
                      std::pair(x.region, x.index)) {
      return Bad("its extents are not in order, once each");
    }
  }
  if (r.tokens.empty() || r.tokens.size() > e.context) {
    return Bad("its conversation does not fit the model's context");
  }
  if (std::ranges::any_of(r.tokens, [&](std::int32_t t) {
        return t < 0 || std::cmp_greater_equal(t, e.vocabulary);
      })) {
    return Bad("a token is not in the model's vocabulary");
  }
  switch (TimeOf(r.used_unix_ms, e.now_unix_ms, e.retention_ms)) {
    case When::kExpired:
      return Bad("it is past its retention ([memory] retention_hours)");
    case When::kFuture:
      return Bad("its last use is in the future");
    case When::kWithin:
      break;
  }
  std::vector<Checkpoint> kept;
  for (Checkpoint& c : r.checkpoints) {
    if (auto valid = CheckCheckpoint(c, r, e); !valid) {
      if (dropped != nullptr) {
        dropped->push_back(std::format("{}: {}", c.file, valid.error()));
      }
      continue;
    }
    if (std::ranges::any_of(kept, [&](const Checkpoint& k) {
          return k.position == c.position || k.file == c.file;
        })) {
      if (dropped != nullptr) {
        dropped->push_back(std::format("{}: a second checkpoint at its position", c.file));
      }
      continue;
    }
    kept.push_back(std::move(c));
  }
  r.checkpoints = std::move(kept);
  return {};
}

}  // namespace jitllm::runtime::kept
