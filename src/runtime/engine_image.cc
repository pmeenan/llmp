// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/engine_image.h"

#include <cstring>
#include <utility>

#include "base/check.h"

namespace llmp::runtime {
std::expected<std::span<const std::byte>, std::string> CheckedImageNoise(const ImageNoise& noise,
                                                                         std::uint64_t elements) {
  if (!elements || elements > SIZE_MAX / sizeof(std::uint16_t) ||
      noise.encoding != format::Encoding::kBF16 ||
      noise.layout != ImageNoiseLayout::kRowsChannels ||
      noise.bytes.size() != elements * sizeof(std::uint16_t) || !noise.bytes.paid_bytes())
    return std::unexpected(
        "image noise buffer needs paid BF16 row/channel bytes of the exact extent");
  for (std::size_t i = 0; i < elements; ++i) {
    std::uint16_t value = 0;
    std::memcpy(&value, noise.bytes.data() + i * sizeof(value), sizeof(value));
    if ((value & 0x7F80U) == 0x7F80U) return std::unexpected("image source noise is not finite");
  }
  return std::span<const std::byte>(noise.bytes.data(), noise.bytes.size());
}

std::uint64_t EngineImage::ControlMetadataBytes() {
  return sizeof(EngineImage) + engine::StagedExecution::ControlMetadataBytes();
}
std::string EngineImage::plan_report() const {
  return execution_ ? execution_->plan_report() : std::string{};
}
void EngineImage::set_plan_report_sink(kernels::PlanRecordSink sink) {
  plan_report_sink_ = std::move(sink);
  if (execution_) execution_->SetPlanReportSink(plan_report_sink_);
}
void EngineImage::ReportPlans(const kernels::PlanRecordSink& sink) const {
  if (execution_) execution_->ReportPlans(sink);
}
EngineImage::EngineImage(engine::PagedNode& node, const config::ModelEntry& entry,
                         const ModelSettings& settings, int index, EngineImageBuild build)
    : build_(std::move(build)), node_(node), index_(index) {
  name_ = entry.name;
  settings_ = settings;
}
EngineImage::~EngineImage() = default;
engine::PagedModel& EngineImage::paged() {
  base::Check(static_cast<bool>(execution_), "image before setup");
  return *execution_;
}
void EngineImage::set_log(std::function<void(std::string_view)> log) {
  Image::set_log(log);
  if (execution_) execution_->SetLog(std::move(log));
}
Status EngineImage::Setup() {
  if (execution_ || !build_) return std::unexpected("image setup order or source differs");
  auto parts = build_(settings_);
  if (!parts) return std::unexpected(parts.error());
  // A malformed source is rejected before a node-owned cleanup borrower exists.
  for (const auto& source : parts->weights)
    if (!source.weights) return std::unexpected("image source weights are missing");
  auto owner = engine::StagedExecution::Create(node_, index_);
  if (!owner) return std::unexpected(owner.error());
  execution_ = std::move(*owner);
  execution_->SetPlanReportSink(plan_report_sink_);
  prepare_request_ = std::move(parts->prepare_request);
  layout_ = parts->request.layout;
  // The node's executor outlives this frontend. Copy the observer, never this.
  parts->log = log_observer();
  return execution_->Setup(std::move(*parts));
}
Status EngineImage::PrepareRequest(const ImageInput& input) {
  if (!execution_) return std::unexpected("image preparation before setup");
  if (auto idle = execution_->CheckRequestBoundary(); !idle) return idle;
  if (!prepare_request_) return std::unexpected("image source has no serving request preparation");
  // The installed geometry's: a recovery may have installed a replacement
  // whose ReplaceGeometry failed part way (StagedExecution::AcknowledgeFence).
  if (const auto installed = execution_->request_layout()) layout_ = *installed;
  auto candidate = prepare_request_(input, layout_);
  if (!candidate) return std::unexpected(candidate.error());
  if (auto* request = std::get_if<ImageRequest>(&*candidate))
    return ReplaceRequest(std::move(*request));
  return ReplaceGeometry(std::move(std::get<ImageGeometry>(*candidate)));
}
Status EngineImage::ReplaceRequest(ImageRequest request) {
  if (!execution_) return std::unexpected("image replacement before setup");
  const auto layout = request.layout;
  auto result = execution_->ReplaceRequest(std::move(request));
  if (result) layout_ = layout;
  return result;
}
Status EngineImage::ReplaceGeometry(ImageGeometry geometry) {
  if (!execution_) return std::unexpected("image geometry before setup");
  const auto layout = geometry.request.layout;
  auto result = execution_->ReplaceGeometry(std::move(geometry));
  if (result) layout_ = layout;
  return result;
}
Status EngineImage::Register() {
  return execution_ ? execution_->Register() : std::unexpected("image before setup");
}
Status EngineImage::Bind() {
  return execution_ ? execution_->Bind() : std::unexpected("image before setup");
}
std::uint64_t EngineImage::activations_needed() const {
  return execution_ ? execution_->activations_needed() : 0;
}
std::uint64_t EngineImage::pool_needed() const {
  return execution_ ? execution_->pool_needed() : 0;
}
std::uint64_t EngineImage::host_input_bytes() const {
  return execution_ ? execution_->host_input_bytes() : 0;
}
std::uint64_t EngineImage::startup_host_bytes() const {
  return execution_ ? execution_->startup_host_bytes() : 0;
}
std::uint64_t EngineImage::plan_floor_bytes() const {
  return execution_ ? execution_->plan_floor_bytes() : 0;
}
std::vector<catalog::ExtentId> EngineImage::weights() const {
  return execution_ ? execution_->weights() : std::vector<catalog::ExtentId>{};
}
std::vector<catalog::ExtentId> EngineImage::state() const {
  return execution_ ? execution_->state() : std::vector<catalog::ExtentId>{};
}
const catalog::Closure& EngineImage::everything() const {
  base::Check(static_cast<bool>(execution_), "image before setup");
  return execution_->everything();
}
std::uint64_t EngineImage::weight_read_bytes() const {
  return execution_ ? execution_->weight_read_bytes() : 0;
}
Status EngineImage::AfterLoad() { return execution_ ? execution_->AfterLoad() : Status{}; }
Status EngineImage::CheckPlaces() {
  return execution_ ? execution_->CheckPlaces() : std::unexpected("image before setup");
}
void EngineImage::DropPlans() {
  if (execution_) execution_->DropPlans();
}
std::string EngineImage::extra() const { return execution_ ? execution_->extra() : "{}"; }
const catalog::Closure& EngineImage::fence_closure() const {
  base::Check(static_cast<bool>(execution_), "image before setup");
  return execution_->fence_closure();
}
std::vector<catalog::ExtentId> EngineImage::managed_extents() const {
  return execution_ ? execution_->managed_extents() : std::vector<catalog::ExtentId>{};
}
Status EngineImage::Release() { return execution_ ? execution_->Release() : Status{}; }
Status EngineImage::Abandon() {
  return execution_ ? execution_->Abandon() : std::unexpected("image before setup");
}
Status EngineImage::FirstOutput(std::string& sha) {
  return execution_ ? execution_->FirstOutput(sha) : std::unexpected("image before setup");
}
Status EngineImage::Finish(std::string& sha) {
  return execution_ ? execution_->Finish(sha) : std::unexpected("image before setup");
}
void EngineImage::ReclaimCandidates(std::uint32_t owner, bool running,
                                    std::vector<memory::ReclaimCandidate>& out) {
  if (execution_) execution_->ReclaimCandidates(owner, running, out);
}
std::uint64_t EngineImage::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  return execution_ ? execution_->Reclaim(kind, id) : 0;
}
}  // namespace llmp::runtime
