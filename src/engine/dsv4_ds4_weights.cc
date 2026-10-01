// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/dsv4_ds4_weights.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernels/ggml/launch.h"
#include "platform/direct_io.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"

namespace jitllm::engine {
namespace {
namespace kg = kernels::ggml;
constexpr std::uint64_t kAlignment = artifact::kFileAlignment;
constexpr std::uint64_t kMaxPayload = std::uint64_t{256} << 20U;
constexpr std::string_view kCommunityArtifact =
    "cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac";

bool Add(std::uint64_t a, std::uint64_t b, std::uint64_t& result) {
  return !__builtin_add_overflow(a, b, &result);
}

std::expected<Ds4WeightSource, std::string> SourceOf(const artifact::Artifact& a,
                                                     const artifact::Placement& placement) {
  const auto& group = a.groups()[placement.group];
  const auto& shard = a.shards()[group.shard];
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
  std::uint64_t offset = 0;
  if (!Add(shard.data_offset.value(), group.offset.value(), begin) ||
      !Add(begin, group.stored.value(), end) || !Add(begin, placement.offset.value(), offset) ||
      placement.bytes.value() == 0 || offset > end || placement.bytes.value() > end - offset) {
    return std::unexpected("invalid validated preparation source");
  }
  return Ds4WeightSource{.shard = group.shard,
                         .file_offset = offset,
                         .bytes = placement.bytes.value(),
                         .file_end = end};
}

void* Pointer(std::uint64_t address) {
  return std::bit_cast<void*>(static_cast<std::uintptr_t>(address));
}

bool Overlap(std::uint64_t a, std::uint64_t an, std::uint64_t b, std::uint64_t bn) {
  return a < b + bn && b < a + an;
}
}  // namespace

std::expected<Ds4PreparedWeightSet, std::string> PlanDs4PreparedWeights(
    const artifact::Artifact& a) {
  if (a.id() != kCommunityArtifact) {
    return std::unexpected("complete ds4 baseline requires the authenticated community artifact");
  }
  const auto& profile = model::Dsv4Flash();
  auto binding = model::BindDsv4(profile, a);
  if (!binding) return std::unexpected(binding.error());
  Ds4PreparedWeightSet result;
  result.artifact_id = a.id();
  std::set<std::pair<bool, std::uint32_t>> identities;
  auto append = [&](std::string name, std::uint32_t index, bool expert,
                    kg::Ds4AlignedShape shape) -> Status {
    if (!identities.emplace(expert, index).second)
      return std::unexpected("duplicate original prepared tensor identity");
    auto layout = kg::Ds4AlignedLayoutOf(shape);
    if (!layout) return std::unexpected(layout.error().detail);
    Ds4PreparedWeight tensor{.name = std::move(name),
                             .index = index,
                             .expert_array = expert,
                             .shape = shape,
                             .layout = *layout,
                             .sources = {}};
    const std::uint32_t count = expert ? profile.experts : 1;
    for (std::uint32_t e = 0; e < count; ++e) {
      auto placement = expert ? a.SlicePlacement(index, e) : a.ResourcePlacement(index);
      if (!placement) return std::unexpected("invalid preparation placement");
      auto source = SourceOf(a, *placement);
      if (!source) return std::unexpected(source.error());
      if (source->bytes != layout->raw_bytes.value() / count) {
        return std::unexpected("preparation source size differs from aligned tensor geometry");
      }
      tensor.sources.push_back(*source);
      if (expert) result.replaced_groups.push_back(placement->group);
    }
    auto& total = expert ? result.aligned_expert_bytes : result.additive_dense_bytes;
    if (!Add(total, layout->packed_bytes.value(), total)) {
      return std::unexpected("prepared weight total overflows");
    }
    result.largest_tensor_bytes =
        std::max(result.largest_tensor_bytes, layout->packed_bytes.value());
    result.tensors.push_back(std::move(tensor));
    return {};
  };
  for (const auto& layer : binding->layers) {
    for (const auto* tensor : {&layer.gate_exps, &layer.up_exps, &layer.down_exps}) {
      const bool down = tensor == &layer.down_exps;
      const std::string_view type = down ? "Q2_K" : "IQ2_XXS";
      if (tensor->type != type || tensor->ne.size() != 2) {
        return std::unexpected(
            "complete ds4 expert precision differs from original community GGUF");
      }
      const auto& array = a.expert_arrays()[tensor->index];
      if (array.count != profile.experts || array.layer >= profile.layers) {
        return std::unexpected("complete ds4 expert count/layer differs");
      }
      const auto input = static_cast<std::uint32_t>(tensor->ne[0]);
      const auto output = static_cast<std::uint32_t>(tensor->ne[1]);
      if (auto added =
              append(array.name, tensor->index, true,
                     {.kind = down ? kg::Ds4AlignedKind::kQ2K : kg::Ds4AlignedKind::kIq2Xxs,
                      .input = input,
                      .output = output,
                      .groups = profile.experts});
          !added)
        return std::unexpected(added.error());
    }
    // The canonical importer already flattens the original group/rank tensor
    // into [heads*head/groups, o_lora*groups], in the same physical row order
    // that cuda_attention_outa_aligned_ptr repacks into low_dim rows.
    // Prepare that exact additive plane up front, without an F16 weight mirror.
    const auto& out_a = layer.out_a;
    if (out_a.type != "Q8_0" || out_a.ne != std::vector<std::uint64_t>{4096, 8192}) {
      return std::unexpected("original own out-a requires the community Q8 group-major tensor");
    }
    const auto& resource = a.resources()[out_a.index];
    if (auto added = append(
            resource.name, out_a.index, false,
            {.kind = kg::Ds4AlignedKind::kQ8Dense, .input = 4096, .output = 8192, .groups = 1});
        !added)
      return std::unexpected(added.error());
  }
  // Original repack_q8 candidate:2-D Q8_0, K divisible1024, at least2MiB,
  // token embedding excluded. Canonically flattened out_a is already prepared.
  for (std::uint32_t i = 0; i < a.resources().size(); ++i) {
    const auto& tensor = a.resources()[i];
    const auto& repr = tensor.repr;
    if (identities.contains({false, i})) continue;
    if (repr.family != artifact::Family::kGgml || repr.type != "Q8_0" || repr.dims.size() != 2 ||
        repr.dims[0] % 1024 != 0 || tensor.bytes.value() < (std::uint64_t{2} << 20U) ||
        tensor.name.contains("token_embd"))
      continue;
    if (repr.dims[0] > UINT32_MAX || repr.dims[1] > UINT32_MAX) {
      return std::unexpected("dense preparation dimension exceeds its native launch contract");
    }
    if (auto added = append(tensor.name, i, false,
                            {.kind = kg::Ds4AlignedKind::kQ8Dense,
                             .input = static_cast<std::uint32_t>(repr.dims[0]),
                             .output = static_cast<std::uint32_t>(repr.dims[1]),
                             .groups = 1});
        !added)
      return std::unexpected(added.error());
  }
  std::ranges::sort(result.replaced_groups);
  const auto duplicates = std::ranges::unique(result.replaced_groups);
  result.replaced_groups.erase(duplicates.begin(), duplicates.end());
  if (result.replaced_groups.size() != static_cast<std::size_t>(profile.layers) * profile.experts) {
    return std::unexpected("aligned expert replacement does not cover all43 layers");
  }
  return result;
}

std::expected<std::vector<Ds4WeightChunk>, std::string> PlanDs4WeightChunks(
    const Ds4PreparedWeight& tensor, std::uint64_t payload_limit) {
  auto layout = kg::Ds4AlignedLayoutOf(tensor.shape);
  if (!layout || payload_limit == 0 || payload_limit > kMaxPayload ||
      tensor.sources.size() != tensor.shape.groups) {
    return std::unexpected("invalid bounded preparation geometry");
  }
  const auto bytes_per_source = layout->raw_bytes.value() / tensor.shape.groups;
  std::uint64_t block_granularity = 1;
  if (tensor.shape.kind == kg::Ds4AlignedKind::kQ2K) {
    block_granularity = std::uint64_t{2} * layout->blocks_per_row;
  }
  const auto unit_bytes = block_granularity * layout->raw_block_bytes;
  const auto chunk_bytes = payload_limit / unit_bytes * unit_bytes;
  if (chunk_bytes == 0 || bytes_per_source % unit_bytes != 0) {
    return std::unexpected("preparation limit cannot hold a whole original row pair");
  }
  const auto per_source = (bytes_per_source / chunk_bytes) + (bytes_per_source % chunk_bytes != 0);
  if (per_source > 65536 / tensor.shape.groups) {
    return std::unexpected("bounded preparation cannot require more than65536 direct read jobs");
  }
  std::vector<Ds4WeightChunk> result;
  std::uint64_t first_block = 0;
  for (const auto& source : tensor.sources) {
    if (source.bytes != bytes_per_source || source.file_offset > source.file_end ||
        source.bytes > source.file_end - source.file_offset) {
      return std::unexpected("bounded preparation source does not cover one whole slice");
    }
    for (std::uint64_t done = 0; done < source.bytes;) {
      const auto bytes = std::min(chunk_bytes, source.bytes - done);
      const auto offset = source.file_offset + done;
      const auto read_offset = offset / kAlignment * kAlignment;
      const auto payload_offset = offset - read_offset;
      const auto span = payload_offset + bytes;
      const auto read_bytes = ((span + kAlignment - 1) / kAlignment) * kAlignment;
      if (read_offset > source.file_end || read_bytes > source.file_end - read_offset ||
          read_offset > INT64_MAX ||
          read_bytes > static_cast<std::uint64_t>(INT64_MAX) - read_offset) {
        return std::unexpected("aligned preparation read exceeds validated source bounds");
      }
      const auto blocks = bytes / layout->raw_block_bytes;
      result.push_back({.shard = source.shard,
                        .read_offset = read_offset,
                        .read_bytes = read_bytes,
                        .payload_offset = payload_offset,
                        .payload_bytes = bytes,
                        .first_block = first_block,
                        .blocks = blocks});
      first_block += blocks;
      done += bytes;
    }
  }
  if (first_block != layout->blocks)
    return std::unexpected("preparation does not tile the original tensor");
  return result;
}

Status PrepareDs4Weight(PagedNode& node, kg::LaunchContext& launch, const catalog::Closure& closure,
                        std::uint32_t stream, const Ds4PreparedWeight& tensor,
                        std::uint64_t payload_limit, const Ds4WeightPreparationBuffers& b,
                        const Ds4WeightPreparationIo& io) {
  auto layout = kg::Ds4AlignedLayoutOf(tensor.shape);
  auto chunks = PlanDs4WeightChunks(tensor, payload_limit);
  const auto host_address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(b.host));
  if (!layout || !chunks || !io.read || !io.write || b.host == nullptr || launch.capturing() ||
      !launch.UsesStream(node.execution(), node.stream(stream)) || host_address % kAlignment != 0 ||
      b.host_bytes < payload_limit + (2 * kAlignment) ||
      b.host_bytes > kMaxPayload + (2 * kAlignment) || b.raw.bytes < payload_limit ||
      b.raw.address % 64 != 0 || b.packed.bytes < layout->packed_bytes.value() ||
      host_address > UINT64_MAX - b.host_bytes || b.raw.address > UINT64_MAX - b.raw.bytes ||
      b.packed.address > UINT64_MAX - b.packed.bytes ||
      Overlap(host_address, b.host_bytes, b.raw.address, b.raw.bytes) ||
      Overlap(host_address, b.host_bytes, b.packed.address, b.packed.bytes)) {
    return std::unexpected("invalid caller-owned bounded preparation buffers/IO");
  }
  if (auto checked = kg::CheckDs4RepackInitialize(tensor.shape, b.packed); !checked) {
    return std::unexpected(checked.error().detail);
  }
  // Validate every prospective kernel before any work or writer callback.
  for (const auto& chunk : *chunks) {
    if (chunk.read_bytes > b.host_bytes || chunk.payload_bytes > b.raw.bytes) {
      return std::unexpected("bounded preparation staging is too small");
    }
    const kg::Ds4RepackChunk desc{.shape = tensor.shape,
                                  .raw = b.raw,
                                  .packed = b.packed,
                                  .first_block = chunk.first_block,
                                  .blocks = chunk.blocks};
    if (auto checked = kg::CheckDs4RepackChunk(desc); !checked) {
      return std::unexpected(checked.error().detail);
    }
  }
  using scheduler::JobResult;
  auto initialized = node.Job(
      closure,
      [&](providers::NativeStream) {
        const auto submitted = kg::RunDs4RepackInitialize(launch, tensor.shape, b.packed);
        if (submitted) return JobResult::kQueued;
        return submitted.error().error == kg::KernelError::kRejected ? JobResult::kNotStarted
                                                                     : JobResult::kUnknown;
      },
      "initialize original ds4 prepared tensor", stream);
  if (!initialized) return initialized;
  auto* staging = static_cast<std::byte*>(b.host);
  for (const auto& chunk : *chunks) {
    if (auto read = io.read(chunk.shard, chunk.read_offset,
                            {staging, static_cast<std::size_t>(chunk.read_bytes)});
        !read)
      return read;
    const kg::Ds4RepackChunk desc{.shape = tensor.shape,
                                  .raw = b.raw,
                                  .packed = b.packed,
                                  .first_block = chunk.first_block,
                                  .blocks = chunk.blocks};
    auto repacked = node.Job(
        closure,
        [&](providers::NativeStream native) {
          if (!providers::CopyAsync(native, Pointer(b.raw.address), staging + chunk.payload_offset,
                                    static_cast<std::size_t>(chunk.payload_bytes),
                                    providers::CopyKind::kHostToDevice)
                   .ok())
            return JobResult::kUnknown;
          const auto submitted = kg::RunDs4RepackChunk(launch, desc);
          if (submitted) return JobResult::kQueued;
          return submitted.error().error == kg::KernelError::kRejected ? JobResult::kFailed
                                                                       : JobResult::kUnknown;
        },
        "repack original ds4 weight chunk", stream);
    if (!repacked) return repacked;
  }
  for (std::uint64_t done = 0; done < layout->packed_bytes.value();) {
    const auto bytes = std::min(payload_limit, layout->packed_bytes.value() - done);
    auto downloaded = node.Job(
        closure,
        [&](providers::NativeStream native) {
          const auto copied = providers::CopyAsync(
              native, staging, Pointer(b.packed.address + done), static_cast<std::size_t>(bytes),
              providers::CopyKind::kDeviceToHost);
          return copied.ok() ? JobResult::kQueued : JobResult::kUnknown;
        },
        "download original ds4 prepared tensor", stream);
    if (!downloaded) return downloaded;
    if (auto written = io.write(done, {staging, static_cast<std::size_t>(bytes)}); !written)
      return written;
    done += bytes;
  }
  return {};
}

std::expected<Ds4PreparedWeightFile, std::string> PrepareDs4WeightFile(
    PagedNode& node, kg::LaunchContext& launch, const catalog::Closure& closure,
    std::uint32_t stream, const artifact::Artifact& a,
    std::span<const artifact::FileDescriptor> shards, const std::filesystem::path& scratch,
    const Ds4PreparedWeight& tensor, std::uint64_t payload_limit,
    const Ds4WeightPreparationBuffers& b) {
  if (a.id() != kCommunityArtifact || shards.size() != a.shards().size() || payload_limit == 0 ||
      payload_limit > kMaxPayload || payload_limit % kAlignment != 0) {
    return std::unexpected("direct ds4 preparation requires the same model and aligned bounded IO");
  }
  auto layout = kg::Ds4AlignedLayoutOf(tensor.shape);
  if (!layout) return std::unexpected(layout.error().detail);
  auto opened = platform::OpenUnnamedDirectFile(scratch);
  if (!opened) return std::unexpected("could not create private direct-I/O prepared weight file");
  artifact::FileDescriptor file(*opened);
  base::Sha256 hasher;
  std::uint64_t written = 0;
  Ds4WeightPreparationIo io;
  io.read = [&](std::uint32_t shard, std::uint64_t offset, std::span<std::byte> target) -> Status {
    if (shard >= shards.size() || !shards[shard].valid()) {
      return std::unexpected("prepared read names an invalid validated shard");
    }
    const auto& geometry = a.shards()[shard];
    std::uint64_t end = 0;
    if (!Add(geometry.data_offset.value(), geometry.data_bytes.value(), end) ||
        offset < geometry.data_offset.value() || offset > end || target.size() > end - offset) {
      return std::unexpected("prepared read exceeds validated shard data");
    }
    if (auto read = platform::TransferDirectFile(shards[shard].get(), offset, target, false);
        !read) {
      return std::unexpected("prepared tensor direct read failed");
    }
    return {};
  };
  io.write = [&](std::uint64_t offset, std::span<const std::byte> source) -> Status {
    if (offset != written || offset % kAlignment != 0 || source.empty() ||
        offset > layout->packed_bytes.value() ||
        source.size() > layout->packed_bytes.value() - offset) {
      return std::unexpected("prepared file writer received an incomplete or reordered output");
    }
    const auto bytes = ((source.size() + kAlignment - 1) / kAlignment) * kAlignment;
    if (bytes > b.host_bytes || source.data() != b.host) {
      return std::unexpected("prepared direct writer exceeds its owned staging slot");
    }
    auto* staging = static_cast<std::byte*>(b.host);
    std::memset(staging + source.size(), 0, static_cast<std::size_t>(bytes - source.size()));
    const std::span<std::byte> stored(staging, static_cast<std::size_t>(bytes));
    if (auto saved = platform::TransferDirectFile(file.get(), offset, stored, true); !saved) {
      return std::unexpected("prepared tensor direct write failed");
    }
    hasher.Update(std::as_const(stored));
    written += source.size();
    return {};
  };
  if (auto prepared = PrepareDs4Weight(node, launch, closure, stream, tensor, payload_limit, b, io);
      !prepared) {
    return std::unexpected(prepared.error());
  }
  if (written != layout->packed_bytes.value()) {
    return std::unexpected("prepared file writer did not cover every output byte");
  }
  const auto stored = ((written + kAlignment - 1) / kAlignment) * kAlignment;
  return Ds4PreparedWeightFile{.file = std::move(file),
                               .payload_bytes = base::Bytes(written),
                               .stored_bytes = base::Bytes(stored),
                               .sha256 = hasher.Finish()};
}
}  // namespace jitllm::engine
