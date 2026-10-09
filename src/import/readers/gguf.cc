// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/readers/gguf.h"

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "base/gguf.h"
#include "base/json.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/files.h"

namespace llmp::import {
namespace {

namespace gg = base::gguf;
namespace json = base::json;

constexpr std::string_view kAlignmentKey = "general.alignment";
constexpr std::string_view kArchitectureKey = "general.architecture";
constexpr std::string_view kSplitNo = "split.no";
constexpr std::string_view kSplitCount = "split.count";
constexpr std::string_view kSplitTensors = "split.tensors.count";

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t file,
                            ItemKind kind = ItemKind::kNone, std::uint64_t item = kNoItem) {
  return std::unexpected(
      Error{.rule = rule, .reason = reason, .file = file, .kind = kind, .item = item});
}

std::optional<std::uint64_t> Add(std::uint64_t a, std::uint64_t b) {
  std::uint64_t out = 0;
  if (__builtin_add_overflow(a, b, &out)) {
    return std::nullopt;
  }
  return out;
}

// x rounded up to a power of two `a`.
std::optional<std::uint64_t> AlignUp(std::uint64_t x, std::uint64_t a) {
  const auto sum = Add(x, a - 1);
  if (!sum) {
    return std::nullopt;
  }
  return *sum & ~(a - 1);
}

// The encoding of each GGML type the format layer names.
std::optional<format::Encoding> EncodingOfGgmlType(std::uint32_t type) {
  for (const format::EncodingInfo& info : format::Encodings()) {
    if (info.ggml && info.ggml->type == type) {
      return info.id;
    }
  }
  return std::nullopt;
}

gg::Limits HeaderLimits() {
  return {.max_keys = kMaxMetadata,
          .max_tensors = kMaxTensors,
          .max_key_bytes = kMaxKeyBytes,
          .max_name_bytes = kMaxNameBytes,
          .max_string_bytes = kMaxGgufString,
          .max_array_elements = kMaxGgufArray};
}

Error FromHeader(const gg::Error& e, std::uint64_t file) {
  Rule rule = Rule::kContainer;
  if (e.fault == gg::Fault::kBounds) {
    rule = Rule::kBounds;
  }
  return {
      .rule = rule, .reason = e.reason, .file = file, .kind = ItemKind::kByte, .item = e.offset};
}

// What one read of a checkpoint may still take, across its split files:
// header bytes read, metadata kept (Cost), tensors.
struct Budget {
  std::uint64_t header = kMaxGgufHeaderBytes;
  std::uint64_t metadata = kMaxGgufMetadataBytes;
  std::uint64_t tensors = kMaxTensors;
};

// What a key's value costs to keep, at least: its key and value bytes, a
// string's or list element's own storage, eight bytes a number.
std::uint64_t Cost(const gg::KeyValue& kv) {
  constexpr std::uint64_t kEntry = 64;
  constexpr std::uint64_t kString = 32;
  std::uint64_t cost = kEntry + kv.key.size();
  if (kv.type == gg::Type::kString) {
    cost += kv.payload.size();
  } else if (kv.type == gg::Type::kArray) {
    // The count is below the array cap and the payload below the header's.
    cost +=
        kv.element == gg::Type::kString ? kv.payload.size() + (kv.count * kString) : kv.count * 8;
  }
  return cost;
}

// A key's value as metadata.
MetaValue ValueOf(const gg::KeyValue& kv) {
  if (kv.type == gg::Type::kString) {
    return std::string(gg::String(kv));
  }
  if (kv.type == gg::Type::kBool) {
    return gg::ScalarBits(kv) != 0;
  }
  if (gg::IsFloat(kv.type)) {
    return gg::FloatValue(kv.type, gg::ScalarBits(kv));
  }
  if (kv.type != gg::Type::kArray) {
    const auto v = gg::SignedValue(kv.type, gg::ScalarBits(kv));
    return v ? MetaValue(*v) : MetaValue();
  }
  if (kv.element == gg::Type::kString) {
    std::vector<std::string> strings;
    strings.reserve(static_cast<std::size_t>(kv.count));
    gg::ForEachString(kv, [&](std::string_view s) { strings.emplace_back(s); });
    return strings;
  }
  if (gg::IsFloat(kv.element)) {
    std::vector<double> reals;
    reals.reserve(static_cast<std::size_t>(kv.count));
    for (std::uint64_t i = 0; i < kv.count; ++i) {
      reals.push_back(gg::FloatValue(kv.element, gg::ElementBits(kv, i)));
    }
    return reals;
  }
  std::vector<std::int64_t> integers;
  integers.reserve(static_cast<std::size_t>(kv.count));
  for (std::uint64_t i = 0; i < kv.count; ++i) {
    const auto v = gg::SignedValue(kv.element, gg::ElementBits(kv, i));
    if (!v) {
      return {};
    }
    integers.push_back(*v);
  }
  return integers;
}

// Equal values, reals compared by their bits (so a NaN equals itself).
bool SameValue(const MetaValue& a, const MetaValue& b) {
  if (a.index() != b.index()) {
    return false;
  }
  if (const auto* x = std::get_if<double>(&a)) {
    return std::bit_cast<std::uint64_t>(*x) == std::bit_cast<std::uint64_t>(std::get<double>(b));
  }
  if (const auto* xs = std::get_if<std::vector<double>>(&a)) {
    const auto& ys = std::get<std::vector<double>>(b);
    return std::ranges::equal(*xs, ys, [](double x, double y) {
      return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y);
    });
  }
  return a == b;
}

// Whether a name has a component ending in "_exps" (llama.cpp's routed
// expert tensors).
bool ExpertName(std::string_view name) {
  std::size_t start = 0;
  while (start <= name.size()) {
    const std::size_t dot = name.find('.', start);
    const std::string_view part =
        name.substr(start, dot == std::string_view::npos ? std::string_view::npos : dot - start);
    if (part.ends_with("_exps")) {
      return true;
    }
    if (dot == std::string_view::npos) {
      break;
    }
    start = dot + 1;
  }
  return false;
}

struct Split {
  std::uint64_t no = 0;
  std::uint64_t count = 1;
  std::optional<std::uint64_t> tensors{};
};

// One file, interpreted.
struct File {
  std::vector<Tensor> tensors{};
  std::map<std::string, MetaValue, std::less<>> metadata;
  std::optional<Split> split{};
};

// A split key's count (gguf-split writes split.no and split.count u16 and
// split.tensors.count i32; any integer type is the same count).
std::expected<std::uint64_t, Error> SplitValue(const gg::KeyValue& kv, std::uint64_t file,
                                               std::uint64_t index) {
  if (!gg::IsInteger(kv.type)) {
    return Fail(Rule::kShards, "a split key that is not an integer", file, ItemKind::kKey, index);
  }
  const auto v = gg::SignedValue(kv.type, gg::ScalarBits(kv));
  if (!v || *v < 0) {
    return Fail(Rule::kShards, "a negative split count", file, ItemKind::kKey, index);
  }
  return static_cast<std::uint64_t>(*v);
}

// The header's file, of `size` bytes, as tensors and metadata.
std::expected<File, Error> Interpret(const gg::Header& header, std::uint64_t size,
                                     std::uint32_t file, Budget& budget) {
  File out;
  if (header.tensors.size() > budget.tensors) {
    return Fail(Rule::kBounds, "more tensors than the reader's cap", file);
  }
  budget.tensors -= header.tensors.size();
  std::uint64_t alignment = gg::kDefaultAlignment;
  std::optional<std::uint64_t> split_no;
  std::optional<std::uint64_t> split_count;
  std::optional<std::uint64_t> split_tensors;
  for (std::size_t k = 0; k < header.metadata.size(); ++k) {
    const gg::KeyValue& kv = header.metadata[k];
    if (kv.key == kAlignmentKey) {
      // gguf.cpp writes a u32; any integer type names the same alignment.
      const auto a = gg::IsInteger(kv.type) ? gg::SignedValue(kv.type, gg::ScalarBits(kv))
                                            : std::optional<std::int64_t>();
      if (!a || *a <= 0 || *a > kMaxGgufAlignment ||
          !std::has_single_bit(static_cast<std::uint64_t>(*a))) {
        return Fail(Rule::kContainer, "general.alignment is not a power of two up to 2^31", file,
                    ItemKind::kKey, k);
      }
      alignment = static_cast<std::uint64_t>(*a);
    } else if (kv.key == kArchitectureKey) {
      if (kv.type != gg::Type::kString || kv.payload.empty() ||
          kv.payload.size() > kMaxArchitecture || json::FirstInvalidUtf8(gg::String(kv)) ||
          !NoControlCharacters(gg::String(kv))) {
        return Fail(Rule::kContainer, "general.architecture is not UTF-8 text of 1 to 200 bytes",
                    file, ItemKind::kKey, k);
      }
    } else if (kv.key == kSplitNo || kv.key == kSplitCount || kv.key == kSplitTensors) {
      auto v = SplitValue(kv, file, k);
      if (!v) {
        return std::unexpected(v.error());
      }
      if (kv.key == kSplitNo) {
        split_no = *v;
      } else if (kv.key == kSplitCount) {
        split_count = *v;
      } else {
        split_tensors = *v;
      }
    }
    if (Cost(kv) > budget.metadata) {
      return Fail(Rule::kBounds, "more metadata than the reader's cap", file, ItemKind::kKey, k);
    }
    budget.metadata -= Cost(kv);
    out.metadata.emplace(std::string(kv.key), ValueOf(kv));
  }
  if (split_count || split_no || split_tensors) {
    out.split = Split{
        .no = split_no.value_or(0), .count = split_count.value_or(1), .tensors = split_tensors};
    if (out.split->count == 0 || out.split->no >= out.split->count) {
      return Fail(Rule::kShards, "a split number outside its count", file);
    }
  }

  const auto data = AlignUp(header.end, alignment);
  if (!data) {
    return Fail(Rule::kBounds, "a data section beyond 64-bit offsets", file);
  }
  // Where each tensor's bytes lie: inside the file, and no two overlapping.
  // gguf.cpp also packs them in info order at the alignment, and ignores
  // bytes after the last; any placement that meets the first two rules
  // describes its tensors as well, so only those are required.
  struct Span {
    std::uint64_t start = 0;
    std::uint64_t end = 0;
    std::size_t tensor = 0;
  };
  std::vector<Span> spans;
  spans.reserve(header.tensors.size());
  out.tensors.reserve(header.tensors.size());
  for (std::size_t i = 0; i < header.tensors.size(); ++i) {
    const gg::TensorInfo& info = header.tensors[i];
    const auto encoding = EncodingOfGgmlType(info.type);
    if (!encoding) {
      return Fail(Rule::kUnsupported, "a GGML type the format layer does not name", file,
                  ItemKind::kTensor, i);
    }
    if (!NoControlCharacters(info.name)) {
      return Fail(Rule::kContainer, "a tensor name with control characters", file,
                  ItemKind::kTensor, i);
    }
    Tensor t{.name = std::string(info.name)};
    t.dims.assign(info.ne.begin(), info.ne.begin() + info.dims);
    if (std::ranges::contains(t.dims, std::uint64_t{0})) {
      return Fail(Rule::kUnsupported, "a tensor with no elements", file, ItemKind::kTensor, i);
    }
    const format::EncodingInfo& e = format::Info(*encoding);
    t.dtype = e.name;
    t.format.encoding = *encoding;
    if (!e.quantized) {
      t.format.layout = format::Layout::kDense;
    } else if (*encoding == format::Encoding::kNvfp4) {
      t.format.layout = format::Layout::kGgmlNvfp4Blocks;
    } else {
      t.format.layout = format::Layout::kGgmlBlocks;
    }
    const auto shape = format::ShapeOf(t.dims);  // at most four dimensions (base/gguf.h)
    if (!shape) {
      return Fail(Rule::kContainer, "a tensor of more than four dimensions", file,
                  ItemKind::kTensor, i);
    }
    t.format.shape = *shape;
    const auto parts = format::PartsOf(t.format);
    if (!parts) {
      if (parts.error() == format::FormatError::kOverflow) {
        return Fail(Rule::kBounds, "a tensor beyond 64-bit sizes", file, ItemKind::kTensor, i);
      }
      return Fail(Rule::kContainer, "a tensor's rows are not whole blocks of its type", file,
                  ItemKind::kTensor, i);
    }
    const std::uint64_t bytes = (*parts)[0].bytes;
    const auto start = Add(*data, info.offset);
    const auto end = start ? Add(*start, bytes) : start;
    if (!end) {
      return Fail(Rule::kBounds, "a tensor beyond 64-bit offsets", file, ItemKind::kTensor, i);
    }
    if (*end > size) {
      return Fail(Rule::kContainer, "a tensor's data past the end of its file", file,
                  ItemKind::kTensor, i);
    }
    spans.push_back({.start = *start, .end = *end, .tensor = i});
    t.extent = {.file = file, .offset = *start, .bytes = bytes};
    t.experts = info.dims == 3 && ExpertName(t.name);
    out.tensors.push_back(std::move(t));
  }
  std::ranges::stable_sort(spans, {}, &Span::start);
  for (std::size_t i = 1; i < spans.size(); ++i) {
    if (spans[i].start < spans[i - 1].end) {
      return Fail(Rule::kContainer, "two tensors' data overlap", file, ItemKind::kTensor,
                  spans[i].tensor);
    }
  }
  return out;
}

// Reads the file's header in growing prefixes, within what the budget
// leaves of the header bytes, and interprets it.
std::expected<File, Error> ReadFile(const OpenFile& f, std::uint32_t file, Budget& budget) {
  const std::uint64_t cap = std::min(f.bytes, budget.header);
  std::vector<std::byte> buffer;
  std::uint64_t have = 0;
  std::uint64_t want = std::min(cap, kGgufFirstRead);
  while (true) {
    buffer.resize(static_cast<std::size_t>(want));
    if (auto ok = ReadExactly(f.fd.get(), have, std::span(buffer).subspan(have), file); !ok) {
      return std::unexpected(ok.error());
    }
    have = want;
    auto header = gg::Parse(buffer, HeaderLimits());
    if (header) {
      budget.header -= header->end;  // within `cap`
      return Interpret(*header, f.bytes, file, budget);
    }
    if (header.error().fault != gg::Fault::kTruncated || have == f.bytes) {
      return std::unexpected(FromHeader(header.error(), file));
    }
    if (have == cap) {
      return Fail(Rule::kBounds, "GGUF headers beyond the reader's cap", file);
    }
    want = std::min(cap, have * 4);
  }
}

// The checkpoint of interpreted files: their tensors in order, every name
// once, and their metadata merged into the first's.
std::expected<Checkpoint, Error> Assemble(std::vector<File> files) {
  Checkpoint c{.container = "gguf"};
  auto& meta = files.front().metadata;
  const auto arch = meta.find(kArchitectureKey);
  if (arch == meta.end()) {
    return Fail(Rule::kContainer, "no general.architecture", 0);
  }
  c.architecture = std::get<std::string>(arch->second);
  std::size_t total = 0;
  for (const File& f : files) {
    total += f.tensors.size();
  }
  if (total > kMaxTensors) {
    return Fail(Rule::kBounds, "more tensors than the reader's cap", kNoItem);
  }
  std::set<std::string_view> names;
  for (std::size_t i = 1; i < files.size(); ++i) {
    for (auto& [key, value] : files[i].metadata) {
      if (key.starts_with("split.")) {
        continue;
      }
      const auto found = meta.find(key);
      if (found == meta.end()) {
        if (meta.size() >= kMaxMetadata) {
          return Fail(Rule::kBounds, "more metadata keys than the reader's cap", i);
        }
        meta.emplace(key, std::move(value));
      } else if (!SameValue(found->second, value)) {
        return Fail(Rule::kShards, "split files disagree on a key's value", i);
      }
    }
  }
  c.tensors.reserve(total);
  for (std::size_t i = 0; i < files.size(); ++i) {
    for (std::size_t t = 0; t < files[i].tensors.size(); ++t) {
      if (!names.insert(files[i].tensors[t].name).second) {
        return Fail(Rule::kShards, "a tensor name in two split files", i, ItemKind::kTensor, t);
      }
    }
  }
  for (File& f : files) {
    std::ranges::move(f.tensors, std::back_inserter(c.tensors));
  }
  c.metadata.reserve(meta.size());
  for (auto& [key, value] : meta) {
    c.metadata.push_back({.key = key, .value = std::move(value)});
  }
  return c;
}

std::string SplitName(std::string_view prefix, std::uint64_t no, std::uint64_t count) {
  return std::format("{}-{:05}-of-{:05}.gguf", prefix, no, count);
}

}  // namespace

std::expected<Checkpoint, Error> ReadGguf(const std::filesystem::path& path, uid_t trusted) {
  auto first = OpenTrustedFile(path, trusted, 0);
  if (!first) {
    return std::unexpected(first.error());
  }
  std::vector<OpenFile> open;
  std::vector<SourceFile> sources;
  open.push_back(std::move(*first));
  sources.push_back({.name = path.filename().string(),
                     .bytes = open[0].bytes,
                     .device = open[0].device,
                     .inode = open[0].inode});
  std::vector<File> files;
  Budget budget;
  auto head = ReadFile(open[0], 0, budget);
  if (!head) {
    return std::unexpected(head.error());
  }
  files.push_back(std::move(*head));
  const Split split = files[0].split.value_or(Split{.tensors = files[0].tensors.size()});
  if (split.no != 0) {
    return Fail(Rule::kShards, "a split GGUF read from a file other than its first", 0);
  }
  if (split.count > kMaxGgufSplits) {
    return Fail(Rule::kBounds, "more split files than the reader's cap", 0);
  }
  if (split.count > 1) {
    const std::string name = sources[0].name;
    const std::string suffix = SplitName("", 1, split.count);
    if (!name.ends_with(suffix) || name.size() == suffix.size()) {
      return Fail(Rule::kShards, "a split GGUF's first file is not named as its splits are", 0);
    }
    const std::string prefix = name.substr(0, name.size() - suffix.size());
    for (std::uint64_t i = 1; i < split.count; ++i) {
      const auto index = static_cast<std::uint32_t>(i);
      sources.push_back({.name = SplitName(prefix, i + 1, split.count)});
      auto f = OpenTrustedFile(path.parent_path() / sources.back().name, trusted, i);
      if (!f) {
        return std::unexpected(f.error());
      }
      open.push_back(std::move(*f));
      sources.back().bytes = open.back().bytes;
      sources.back().device = open.back().device;
      sources.back().inode = open.back().inode;
      auto file = ReadFile(open.back(), index, budget);
      if (!file) {
        return std::unexpected(file.error());
      }
      if (!file->split || file->split->no != i || file->split->count != split.count) {
        return Fail(Rule::kShards, "a split file's split.no or split.count is not its place's", i);
      }
      files.push_back(std::move(*file));
    }
  }
  std::uint64_t total = 0;
  for (const File& f : files) {
    total += f.tensors.size();
  }
  // llama.cpp's loader does not need split.tensors.count; when given, it is
  // the tensors' count.
  if (split.tensors && *split.tensors != total) {
    return Fail(Rule::kShards, "split.tensors.count is not the split files' tensors", 0);
  }
  auto c = Assemble(std::move(files));
  if (!c) {
    return std::unexpected(c.error());
  }
  c->files = std::move(sources);
  c->open = std::move(open);
  return c;
}

std::expected<Checkpoint, Error> ParseGguf(std::span<const std::byte> file) {
  auto header = gg::Parse(file, HeaderLimits());
  if (!header) {
    return std::unexpected(FromHeader(header.error(), 0));
  }
  Budget budget;
  if (header->end > budget.header) {
    return Fail(Rule::kBounds, "GGUF headers beyond the reader's cap", 0);
  }
  budget.header -= header->end;
  auto f = Interpret(*header, file.size(), 0, budget);
  if (!f) {
    return std::unexpected(f.error());
  }
  const Split split = f->split.value_or(Split{.tensors = f->tensors.size()});
  if (split.count != 1) {
    return Fail(Rule::kShards, "a split GGUF given as one file's bytes", 0);
  }
  if (split.tensors && *split.tensors != f->tensors.size()) {
    return Fail(Rule::kShards, "split.tensors.count is not the split files' tensors", 0);
  }
  std::vector<File> files;
  files.push_back(std::move(*f));
  auto c = Assemble(std::move(files));
  if (!c) {
    return std::unexpected(c.error());
  }
  c->files.push_back({.name = "memory", .bytes = file.size()});
  return c;
}

}  // namespace llmp::import
