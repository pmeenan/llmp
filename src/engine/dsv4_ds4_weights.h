// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Private complete-ds4 benchmark preparation, not an artifact format or a
// serving default. Produces the original aligned physical tensors from a
// validated community artifact. The benchmark retains its original content
// identity and hashes the completed output files separately. Aligned expert
// tensors replace raw expert residency; dense Q8 mirrors are additive.
#ifndef JITLLM_ENGINE_DSV4_DS4_WEIGHTS_H_
#define JITLLM_ENGINE_DSV4_DS4_WEIGHTS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "artifact/artifact.h"
#include "base/sha256.h"
#include "engine/paged_node.h"
#include "kernels/ggml/dsv4_ds4_repack.h"
#include "model/dsv4.h"

namespace jitllm::engine {

struct Ds4WeightSource {
  std::uint32_t shard = 0;
  std::uint64_t file_offset = 0;
  std::uint64_t bytes = 0;
  // The validated containing group's end; aligned direct reads cannot
  // consume unvalidated bytes beyond it.
  std::uint64_t file_end = 0;
};

struct Ds4PreparedWeight {
  std::string name;
  std::uint32_t index = 0;
  bool expert_array = false;
  kernels::ggml::Ds4AlignedShape shape{};
  kernels::ggml::Ds4AlignedLayout layout{};
  // Expert-major, exactly one whole slice per expert. Dense: one source.
  std::vector<Ds4WeightSource> sources;
};

struct Ds4PreparedWeightSet {
  std::string artifact_id;
  std::vector<Ds4PreparedWeight> tensors;
  std::vector<std::uint32_t> replaced_groups;
  std::uint64_t aligned_expert_bytes = 0;
  std::uint64_t additive_dense_bytes = 0;
  std::uint64_t largest_tensor_bytes = 0;
};

// Binds all43 Flash layers, refuses another artifact/precision and requires
// IQ2_XXS gate/up, Q2_K down and the original dense Q8 candidate predicates.
// Uses only validated native resource/slice placements, never raw GGUF offsets.
std::expected<Ds4PreparedWeightSet, std::string> PlanDs4PreparedWeights(
    const artifact::Artifact& artifact);

struct Ds4WeightChunk {
  std::uint32_t shard = 0;
  std::uint64_t read_offset = 0;
  std::uint64_t read_bytes = 0;
  std::uint64_t payload_offset = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t first_block = 0;
  std::uint64_t blocks = 0;
};

// Bounded direct-I/O jobs. Q2 chunks stop on complete paired rows, and all
// chunks stop at expert boundaries even where neighboring slices share a
// file. Read slack is bounded by8KiB and is never fed into the repack kernel.
std::expected<std::vector<Ds4WeightChunk>, std::string> PlanDs4WeightChunks(
    const Ds4PreparedWeight& tensor, std::uint64_t payload_limit);

struct Ds4WeightPreparationBuffers {
  // All addresses are caller-owned, cataloged and leased by closure.
  // Host staging is pinned/aligned4096; GPU raw/packed alignment is64.
  void* host = nullptr;
  std::uint64_t host_bytes = 0;
  kernels::ggml::Ds4CacheBuffer raw{}, packed{};
};

struct Ds4WeightPreparationIo {
  // Native artifact shard FDs remain open; the reader must complete the
  // exact requested aligned read before returning. The writer creates a
  // private output and completes each exact write before staging is reused.
  // No callback may submit work onto this node or retain borrowed staging.
  std::function<Status(std::uint32_t, std::uint64_t, std::span<std::byte>)> read;
  std::function<Status(std::uint64_t, std::span<const std::byte>)> write;
};

// Runs actual original GPU repacking through the native device lane. One
// complete tensor at a time bounds memory; each upload/repack/download is
// fenced by node.Job before a slot is reused or a file writer observes it.
// On failure the caller discards incomplete output, performs native node
// teardown and retains all buffers through completion. Initialization and
// transfers belong to preparation, outside resident inference timing.
Status PrepareDs4Weight(PagedNode& node, kernels::ggml::LaunchContext& launch,
                        const catalog::Closure& closure, std::uint32_t stream,
                        const Ds4PreparedWeight& tensor, std::uint64_t payload_limit,
                        const Ds4WeightPreparationBuffers& buffers,
                        const Ds4WeightPreparationIo& io);

struct Ds4PreparedWeightFile {
  artifact::FileDescriptor file;
  base::Bytes payload_bytes;
  base::Bytes stored_bytes;     // payload rounded to4096 with zero file padding
  base::Sha256Digest sha256{};  // complete stored bytes, not a process address
};

// An unnamed owner-only direct-I/O file, retained only by this benchmark.
// No permanent derived artifact format is introduced. All input descriptors
// come from artifact.OpenShardForDirectRead; files remain owned through the
// preparation, native paging and final node teardown. Final stored-byte hash
// and canonical model/tensor provenance belong in the benchmark receipt.
std::expected<Ds4PreparedWeightFile, std::string> PrepareDs4WeightFile(
    PagedNode& node, kernels::ggml::LaunchContext& launch, const catalog::Closure& closure,
    std::uint32_t stream, const artifact::Artifact& artifact,
    std::span<const artifact::FileDescriptor> shards, const std::filesystem::path& scratch,
    const Ds4PreparedWeight& tensor, std::uint64_t payload_limit,
    const Ds4WeightPreparationBuffers& buffers);

}  // namespace jitllm::engine
#endif  // JITLLM_ENGINE_DSV4_DS4_WEIGHTS_H_
