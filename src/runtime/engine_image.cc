// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/engine_image.h"

#include <format>
#include <utility>

#include "base/check.h"
#include "runtime/serving_support.h"

namespace llmp::runtime {
namespace {
using serving_support::Error;
}  // namespace

EngineImage::EngineImage(engine::PagedNode& node, const config::ModelEntry& entry,
                         const ModelSettings& settings, int index, EngineImageBuild build)
    : node_(node), index_(index), build_(std::move(build)) {
  name_ = entry.name;
  settings_ = settings;
}

EngineImage::~EngineImage() = default;

engine::PagedModel& EngineImage::paged() {
  base::Check(runner_ != nullptr, "an engine-backed pipeline's runner before its Setup");
  return *runner_;
}

Status EngineImage::Setup() {
  if (runner_ != nullptr) return Error(std::format("model {}: set up already", name_));
  if (!build_) return Error(std::format("model {}: no component pipeline builds it", name_));
  auto parts = build_(settings_);
  if (!parts) return Error(std::format("model {}: {}", name_, parts.error()));
  architecture_ = parts->architecture;
  keep_ = std::move(parts->keep);
  parts->config.slots.model = name_;
  auto runner =
      engine::ModelRunner::Create(node_, std::move(parts->config), std::move(parts->source), index_,
                                  static_cast<std::uint32_t>(index_));
  if (!runner) return Error(std::format("model {}: {}", name_, runner.error()));
  runner_ = std::move(*runner);
  return runner_->Setup();
}

Status EngineImage::Register() {
  if (runner_ == nullptr) return Error(std::format("model {} is not set up", name_));
  return runner_->Register();
}

Status EngineImage::Bind() {
  if (runner_ == nullptr) return Error(std::format("model {} is not set up", name_));
  return runner_->Bind();
}

std::uint64_t EngineImage::activations_needed() const {
  return runner_ ? runner_->activations_needed() : 0;
}
std::uint64_t EngineImage::pool_needed() const { return runner_ ? runner_->pool_needed() : 0; }
std::uint64_t EngineImage::host_input_bytes() const {
  return runner_ ? runner_->host_input_bytes() : 0;
}
std::uint64_t EngineImage::plan_floor_bytes() const {
  return runner_ ? runner_->plan_floor_bytes() : 0;
}
void EngineImage::ReclaimCandidates(std::uint32_t owner, bool running,
                                    std::vector<memory::ReclaimCandidate>& out) {
  if (runner_) runner_->ReclaimCandidates(owner, running, out);
}
std::uint64_t EngineImage::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  return runner_ ? runner_->Reclaim(kind, id) : 0;
}
std::vector<catalog::ExtentId> EngineImage::weights() const {
  return runner_ ? runner_->weights() : std::vector<catalog::ExtentId>{};
}
std::vector<catalog::ExtentId> EngineImage::state() const {
  return runner_ ? runner_->slots().state() : std::vector<catalog::ExtentId>{};
}
const catalog::Closure& EngineImage::everything() const {
  base::Check(runner_ != nullptr, "an engine-backed pipeline's closure before its Setup");
  return runner_->slots().everything();
}
std::uint64_t EngineImage::weight_read_bytes() const {
  return runner_ ? runner_->weight_read_bytes() : 0;
}
Status EngineImage::CheckPlaces() {
  if (runner_ == nullptr) return Error(std::format("model {} is not set up", name_));
  return runner_->slots().CheckPlaces();
}
void EngineImage::DropPlans() {
  if (runner_) runner_->DropPlans();
}
std::string EngineImage::extra() const {
  return std::format(R"({{"engine":"component","architecture":"{}","runner":{}}})", architecture_,
                     runner_ ? runner_->report_json() : std::string("{}"));
}

Status EngineImage::FirstOutput(std::string& /*sha*/) {
  return Error(std::format(
      "model {}: the image pipeline's phases are not on the shared runner yet (they move with "
      "Qwen-Image's migration)",
      name_));
}

Status EngineImage::Finish(std::string& sha) { return FirstOutput(sha); }

}  // namespace llmp::runtime
