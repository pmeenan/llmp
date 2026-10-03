// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen-Image-2.1 pipeline as a model on a paged node (paged_node.h;
// M3's swap path, docs/experiments/fast-swap/swap.md): its three component
// artifacts,
// joined by their composition (D-089), paged into device VMM through the
// node's landing zone, and each phase run as device jobs over the closure
// of its own component (D-086), with the kernels and the call order of the
// resident harness (benchmarks/qwen_image_exec.cc, whose phases this
// copies, as dsv4_plan.h copies DeepSeek's: the resident harness and its comparison
// with diffusers need no rerun; the pixels are checked equal to its image's
// instead).
//
// Of the engine's skeleton (docs/engine.md) it uses the weights' paging, the
// runner's resources (its own memory, cuBLAS, staging) and the pinned
// places; it holds no conversation state and no GGML plans, and records one
// step through the device runtime rather than the launch context.
//
// - Weights: each component a set of extents (paged_weights.h), only the
//   groups its phase reads (the text encoder's table and language layers,
//   the denoiser, the VAE's decoder), a dense region per group, landed from
//   its shards (D-081). The VAE's weights are F32 in the artifact and paged
//   as they are; the decode job rounds each to BF16 (as
//   from_pretrained(torch_dtype=bfloat16) casts them, and the resident
//   harness does on the host) into its working memory before the decoder.
//   Their places are pinned in the scheduler when registered (D-090).
// - Phases, each a device job over only its component's closure, the
//   image's own memory, the shared workspace (every per-job buffer), the
//   cuBLAS workspace and the staging: encode (one job), denoise (a job per
//   step; the first also projects the text rows and fills the prefix K/V
//   cache), decode (one job). A generation is one request: the driver
//   opens it on the image's stream over everything() (all three
//   components; PagedNode::BeginRequest, M3's lease per request), leased
//   once, and each phase's job runs under that lease, its closure within
//   it. Outside a request each job leases its own component, and a
//   component a phase does not lease can be evicted and paged back
//   meanwhile.
// - The image's own memory (device VMM, mapped at setup, pinned): what
//   lives from one job to the next within a generation — the prompt
//   embeddings, the text rows, the prefix K/V cache, the rotary tables, the
//   latents and the noise prediction. Nothing survives a generation, so a
//   swap has no image state to spill.
// - The endpoint (M3's swap table): FirstOutput encodes the prompt and runs
//   the first denoising step; Finish runs the rest and the decoder, and
//   hashes the image's RGBA pixels.
// - Kernels: the pipeline's (kernels/image/pipeline.h), dispatched through
//   a plan bound against the implementation registry (D-053), the resident
//   harness's plans (QwenImageOptions::plan, fast by default). Steps from
//   the third on replay a CUDA graph of one step, captured in the third's
//   job the first time and kept until the node's reclaim order takes it
//   (charged to the node as recorded, D-090 as amended; recorded again by
//   the next step from the third): every address it holds (the
//   weights' pinned places, D-090, the image's own memory, the workspace)
//   stays put.

#ifndef JITLLM_ENGINE_QWEN_IMAGE_RUNNER_H_
#define JITLLM_ENGINE_QWEN_IMAGE_RUNNER_H_

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "artifact/artifact.h"
#include "catalog/catalog.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "kernels/image/pipeline.h"
#include "memory/reclaim.h"
#include "model/qwen_image.h"

namespace jitllm::engine {

struct QwenImageOptions {
  std::filesystem::path store;  // the installed artifacts
  std::string composition;      // its ID
  std::filesystem::path noise;  // the initial latents (BF16, diffusers' for the seed)
  std::filesystem::path out;
  std::string prompt = "A red ceramic teapot on a plain wooden table, soft daylight, no text.";
  std::uint32_t size = 1024;
  std::uint32_t steps = 40;
  kernels::image::PlanKind plan = kernels::image::PlanKind::kFast;
  // Denoising steps from the third on replayed from one captured step
  // (their inputs uploaded to fixed places first; D-090 keeps the weights'
  // places for the model's life).
  bool graphs = true;
};

class QwenImageRunner final : public PagedModel {
 public:
  using Status = engine::Status;

  QwenImageRunner(PagedNode& node, const QwenImageOptions& options, int owner,
                  std::uint32_t stream);
  ~QwenImageRunner() override;
  QwenImageRunner(const QwenImageRunner&) = delete;
  QwenImageRunner& operator=(const QwenImageRunner&) = delete;
  QwenImageRunner(QwenImageRunner&&) = delete;
  QwenImageRunner& operator=(QwenImageRunner&&) = delete;

  // Before the scheduler exists: the composition and components, bound;
  // the prompt's tokens; the phases' working memory measured; the image's
  // own memory, the cuBLAS workspace and staging mapped; the weights'
  // places reserved and cataloged.
  Status Setup();
  std::uint64_t activations_needed() const { return work_bytes_; }
  // The image runs no GGML plan, so needs no GGML pool: one extent.
  static std::uint64_t pool_needed() { return kPagedExtent; }
  Status Register();
  Status Bind();

  // Encode, then the first denoising step; `sha` the step's noise
  // prediction's SHA-256.
  Status FirstOutput(std::string& sha);
  // The remaining steps and the decoder; `sha` the RGBA pixels' SHA-256.
  Status Finish(std::string& sha);
  // Timings and sizes, JSON.
  std::string Report() const;
  // Its recorded step (driver memory outside the catalog's extents,
  // charged to the node once recorded) as a candidate for the node's
  // reclaim order (memory/reclaim.h), and its reclaim: dropped between
  // jobs, recorded again by the next step from the third. The image
  // keeps no GGML plans; its pipeline's bound plan is a few KiB, not
  // counted.
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out);
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id);
  std::string plan_report() const;
  // What its kept graph took of the device's free memory at its recording
  // (before the floor it is charged at), for the teardown's log.
  std::uint64_t graph_measured_bytes() const;
  // Each component artifact's data directory (after Setup).
  std::vector<std::filesystem::path> data() const;

  std::vector<catalog::ExtentId> weights() const;
  const catalog::Closure& everything() const { return everything_; }
  std::uint64_t weight_read_bytes() const;

  std::uint32_t stream() const override { return stream_; }
  const catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<catalog::ExtentId> managed_extents() const override { return weights(); }
  Status Release() override;

 private:
  struct State;
  Status Encode();
  Status Step(std::uint32_t index, bool hash, std::string* sha);
  Status Decode(std::string& sha);

  PagedNode& node_;
  const QwenImageOptions& o_;
  int owner_;
  std::uint32_t stream_;
  std::unique_ptr<State> s_;
  std::uint64_t work_bytes_ = 0;
  catalog::Closure everything_;
  catalog::Closure fence_;
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_QWEN_IMAGE_RUNNER_H_
