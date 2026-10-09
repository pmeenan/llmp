// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The serving adapter of the image pipeline kind (D-107; docs/runtime-
// serving.md, "Engine-backed models"): one Image over the shared runner,
// for the image pipeline once it moves to the component engine.
//
// A shell for now: it sets up, registers, binds and pages its runner as
// any served model does (its weights, its plans, the reclaim order), but
// the pipeline's work, a text encoder's pass, the denoiser's steps and the
// VAE's decode as one composition of stages (D-089), has no orchestration
// on the shared runner yet, so FirstOutput and Finish refuse. Qwen-Image
// keeps its own serving class (serving.cc) until its migration (M3.6's
// order, after the LLM families) brings the image phases, their
// components and this adapter's units of work.
//
// Vendor-free, every build profile.

#ifndef LLMP_RUNTIME_ENGINE_IMAGE_H_
#define LLMP_RUNTIME_ENGINE_IMAGE_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "config/node_config.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "runtime/serving.h"

namespace llmp::runtime {

// What an engine-backed image pipeline runs: its runner (every stage's
// weights and state, its setup shapes) and step source.
struct EngineImageParts {
  engine::RunnerConfig config;
  std::unique_ptr<engine::StepSource> source;
  std::string architecture;
  // What the step source and the launch borrow, kept until the runner is
  // gone.
  std::shared_ptr<void> keep;
};

using EngineImageBuild =
    std::function<std::expected<EngineImageParts, std::string>(const ModelSettings& settings)>;

class EngineImage final : public Image {
 public:
  EngineImage(engine::PagedNode& node, const config::ModelEntry& entry,
              const ModelSettings& settings, int index, EngineImageBuild build);
  EngineImage(const EngineImage&) = delete;
  EngineImage& operator=(const EngineImage&) = delete;
  EngineImage(EngineImage&&) = delete;
  EngineImage& operator=(EngineImage&&) = delete;
  ~EngineImage() override;

  engine::PagedModel& paged() override;
  Status Setup() override;
  Status Register() override;
  Status Bind() override;
  std::uint64_t activations_needed() const override;
  std::uint64_t pool_needed() const override;
  std::uint64_t host_input_bytes() const override;
  std::uint64_t plan_floor_bytes() const override;
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override;
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override;
  std::vector<catalog::ExtentId> weights() const override;
  std::vector<catalog::ExtentId> state() const override;
  const catalog::Closure& everything() const override;
  std::uint64_t weight_read_bytes() const override;
  Status CheckPlaces() override;
  void DropPlans() override;
  std::string extra() const override;

  // Refused: the pipeline's phases are not on the shared runner yet.
  Status FirstOutput(std::string& sha) override;
  Status Finish(std::string& sha) override;

 private:
  engine::PagedNode& node_;
  int index_;
  EngineImageBuild build_;
  std::string architecture_;
  std::shared_ptr<void> keep_;  // before the runner: it outlives it
  std::unique_ptr<engine::ModelRunner> runner_;
};

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_ENGINE_IMAGE_H_
