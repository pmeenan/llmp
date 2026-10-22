// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/checkpoint.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "import/error.h"
#include "import/files.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::import {

std::optional<std::uint64_t> SourcePartBytes(const Tensor& tensor) {
  const auto parts = format::PartsOf(tensor.format);
  const auto part = parts ? parts->Find(tensor.part) : std::nullopt;
  if (!part) return std::nullopt;
  if (!tensor.scalar_storage) return part->bytes;
  if (tensor.part != format::PartRole::kTensorScale ||
      tensor.format.encoding != format::Encoding::kNvfp4 || !tensor.format.params.tensor_scale ||
      tensor.format.stacked != 1 || part->bytes != 4 ||
      (*tensor.scalar_storage != format::Encoding::kBF16 &&
       *tensor.scalar_storage != format::Encoding::kF16))
    return std::nullopt;
  return 2;
}

const Tensor* Checkpoint::FindTensor(std::string_view name) const {
  const auto found = std::ranges::find(tensors, name, &Tensor::name);
  return found == tensors.end() ? nullptr : &*found;
}

const MetaValue* Checkpoint::FindValue(std::string_view key) const {
  const auto found = std::ranges::lower_bound(metadata, key, {}, &MetaEntry::key);
  return found == metadata.end() || found->key != key ? nullptr : &found->value;
}

std::expected<void, Error> Checkpoint::Read(const Extent& extent, std::uint64_t offset,
                                            std::span<std::byte> out) const {
  if (extent.file >= files.size() || offset > extent.bytes || out.size() > extent.bytes - offset) {
    return std::unexpected(Error{.rule = Rule::kBounds,
                                 .reason = "a read outside its tensor's extent",
                                 .file = extent.file});
  }
  const SourceFile& source = files[extent.file];
  if (source.path.empty()) {
    return std::unexpected(Error{.rule = Rule::kIo,
                                 .reason = "a checkpoint parsed from memory has no file to read",
                                 .file = extent.file});
  }
  auto f = OpenCheckpointFile(source.path, extent.file);
  if (!f) {
    return std::unexpected(f.error());
  }
  if (f->device != source.device || f->inode != source.inode) {
    return std::unexpected(Error{.rule = Rule::kIo,
                                 .reason = "a checkpoint file was replaced after it was read",
                                 .file = extent.file});
  }
  return ReadExactly(f->fd.get(), extent.offset + offset, out, extent.file);
}

bool NoControlCharacters(std::string_view text) {
  for (std::size_t i = 0; i < text.size(); ++i) {
    const auto u = static_cast<unsigned char>(text[i]);
    if (u < 0x20 || u == 0x7F) {
      return false;
    }
    // C1's controls, U+0080 to U+009F (a terminal may take U+009B as CSI).
    if (u == 0xC2 && i + 1 < text.size()) {
      const auto next = static_cast<unsigned char>(text[i + 1]);
      if (next >= 0x80 && next <= 0x9F) {
        return false;
      }
    }
  }
  return true;
}

spec::Source SourceOf(const Checkpoint& checkpoint) {
  spec::Source s{.architecture = checkpoint.architecture, .container = checkpoint.container};
  s.tensors.reserve(checkpoint.tensors.size());
  for (const Tensor& t : checkpoint.tensors) {
    s.tensors.push_back({.name = t.name, .format = t.format, .part = t.part, .experts = t.experts});
  }
  return s;
}

std::vector<spec::SourceValue> ValuesOf(const Checkpoint& checkpoint) {
  std::vector<spec::SourceValue> out;
  for (const MetaEntry& e : checkpoint.metadata) {
    if (!spec::IsSourceKey(e.key)) {
      continue;
    }
    const auto add = [&](spec::ParamValue v) {
      out.push_back({.key = e.key, .value = std::move(v)});
    };
    if (const auto* i = std::get_if<std::int64_t>(&e.value)) {
      add(*i);
    } else if (const auto* d = std::get_if<double>(&e.value)) {
      add(*d);
    } else if (const auto* b = std::get_if<bool>(&e.value)) {
      add(*b);
    } else if (const auto* s = std::get_if<std::string>(&e.value)) {
      add(*s);
    } else if (const auto* is = std::get_if<std::vector<std::int64_t>>(&e.value); is != nullptr) {
      add(*is);
    } else if (const auto* ds = std::get_if<std::vector<double>>(&e.value); ds != nullptr) {
      add(*ds);
    }
  }
  return out;
}

}  // namespace llmp::import
