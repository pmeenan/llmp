// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/step_program.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "base/bytes.h"
#include "base/check.h"
#include "base/json_stream.h"
#include "components/input_values.h"
#include "components/step.h"
#include "execution/registry.h"
#include "kernels/common/comparison_protocol.h"
#include "kernels/decode/selected_source.h"
#include "kernels/exl3/selected_source.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/comparison.h"
#include "kernels/ggml/comparison_region.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/measured_ranks.h"
#include "kernels/ggml/mul_mat_cublas_internal.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/plan_step.h"
#include "kernels/ggml/set_rows_group.h"
#include "providers/device_runtime.h"

namespace llmp::kernels::ggml {

class GeneratedStoreInputs final : public kernels::ResourceDependency {
 public:
  struct Owner {
    GeneratedStoreInputs* value = nullptr;
    kernels::ResourceKeep keep;
  };

  static std::expected<Owner, std::string> Create(
      components::StepInput input, LoweredStep::GeneratedStoreRecipe recipe,
      std::shared_ptr<const ConstructionFunder> funder) {
    // The payload is the pinned allocation itself. The owner object includes
    // its ResourceKeep, ExecutionResources lease and intrusive registry link;
    // charge one allocator envelope and one provider pinned-allocation control
    // envelope in addition to those named storage bytes.
    constexpr std::uint64_t kOwnerAllocationEnvelope = 64;
    constexpr std::uint64_t kPinnedAllocationControlEnvelope = 64;
    const std::uint64_t payload = recipe.slot.bytes;
    std::uint64_t bytes = 0;
    if (!funder || !funder->charge || !funder->release || payload == 0 ||
        __builtin_add_overflow(
            std::uint64_t{sizeof(GeneratedStoreInputs) + kOwnerAllocationEnvelope +
                          kPinnedAllocationControlEnvelope},
            payload, &bytes) ||
        !funder->charge(bytes))
      return std::unexpected("generated store input ownership was not funded");
    const auto retained_funder = funder;
    auto* owner =
        new (std::nothrow) GeneratedStoreInputs(std::move(input), recipe, std::move(funder), bytes);
    if (!owner) {
      retained_funder->release(bytes);
      return std::unexpected("generated store input owner allocation failed");
    }
    return Owner{owner, owner->Adopt()};
  }

  GeneratedStoreInputs(const GeneratedStoreInputs&) = delete;
  GeneratedStoreInputs& operator=(const GeneratedStoreInputs&) = delete;
  bool ready() const override { return initialized_; }
  std::uint64_t host_bytes() const {
    constexpr std::uint64_t kOwnerAllocationEnvelope = 64;
    constexpr std::uint64_t kPinnedAllocationControlEnvelope = 64;
    return sizeof(*this) + kOwnerAllocationEnvelope + kPinnedAllocationControlEnvelope +
           recipe_.slot.bytes;
  }
  std::uint64_t device_address() const { return recipe_.slot.device_address; }
  std::uint64_t bytes() const { return recipe_.slot.bytes; }
  bool dirty() const { return dirty_; }
  bool has_fence() const { return fence_.has_value(); }
  std::uint64_t epoch() const { return epoch_; }
  const std::shared_ptr<const ConstructionFunder>& funder() const { return funder_; }
  void RetainResources(const kernels::ExecutionResources& resources) {
    if (resources && !resources_) resources_ = resources;
  }
  const kernels::ExecutionResources& resources() const { return resources_; }
  bool has_resources() const { return static_cast<bool>(resources_); }
  void AdvanceEpoch() {
    if (epoch_ != UINT64_MAX) ++epoch_;
  }
  std::expected<void, StepFailure> Initialize() {
    if (initialized_) return {};
    auto pinned = providers::AllocatePinned(static_cast<std::size_t>(recipe_.slot.bytes));
    if (!pinned) return std::unexpected(StepFailure{.detail = pinned.error().text()});
    pinned_ = *pinned;
    initialized_ = true;
    return {};
  }
  std::expected<void, StepFailure> Update(const planner::StepShape& shape) {
    if (auto valid = ValidateUpdate(shape); !valid) return valid;
    const auto& segment = shape.segments.front();
    if (populated_ && segment.past == past_) return {};
    if (dirty_ || gate_closed_ || fence_)
      return std::unexpected(
          StepFailure{.detail = "generated store-cell bytes are still used by queued device work"});

    components::InputValueSink output(
        std::span(static_cast<std::byte*>(pinned_), static_cast<std::size_t>(recipe_.slot.bytes)),
        sizeof(std::int64_t));
    if (auto emitted = components::EmitInputValues(input_, shape, {}, output); !emitted)
      return std::unexpected(StepFailure{.detail = emitted.error()});
    populated_ = true;
    past_ = segment.past;
    AdvanceEpoch();
    return {};
  }
  std::expected<void, StepFailure> ValidateUpdate(const planner::StepShape& shape) const {
    if (!initialized_)
      return std::unexpected(StepFailure{.detail = "generated store input is not bound"});
    if (shape.stage != recipe_.stage || shape.phase != recipe_.phase ||
        shape.row_invariant != recipe_.row_invariant || shape.read_align != recipe_.read_align ||
        shape.segments.size() != 1 || !(shape.segments.front() == recipe_.segment))
      return std::unexpected(
          StepFailure{.detail = "generated store input changed a sealed shape field"});
    const auto& segment = shape.segments.front();
    const std::uint64_t capacity = recipe_.slot.cells.capacity;
    if (segment.rows == 0 || segment.past > capacity || segment.rows > capacity - segment.past ||
        capacity > INT64_MAX)
      return std::unexpected(
          StepFailure{.detail = "generated store-cell positions exceed their checked capacity"});
    struct Check {
      std::size_t count = 0;
      std::uint64_t capacity = 0;
      bool valid = true;
    } check{.capacity = capacity};
    const auto emit_check = [](void* context, std::size_t index, std::int64_t value) -> bool {
      auto& checked = *static_cast<Check*>(context);
      if (index != checked.count || value < 0 ||
          static_cast<std::uint64_t>(value) >= checked.capacity)
        checked.valid = false;
      ++checked.count;
      return checked.valid;
    };
    components::InputValueSink preflight(&check, emit_check);
    if (auto emitted = components::EmitInputValues(input_, shape, {}, preflight);
        !emitted || !check.valid || check.count != recipe_.slot.bytes / sizeof(std::int64_t))
      return std::unexpected(StepFailure{
          .detail = "the canonical StoreCells generator did not produce bounded exact values"});

    return {};
  }
  std::expected<void, StepFailure> Copy(providers::NativeStream stream, bool capturing) {
    if (!populated_ || !pinned_ || recipe_.slot.device_address == 0)
      return std::unexpected(StepFailure{.detail = "generated StoreCells values are not prepared"});
    if (!capturing) dirty_ = true;
    AdvanceEpoch();
    const auto copied = providers::CopyAsync(
        stream, reinterpret_cast<void*>(recipe_.slot.device_address), pinned_,
        static_cast<std::size_t>(recipe_.slot.bytes), providers::CopyKind::kHostToDevice);
    if (!copied.ok()) {
      dirty_ = true;
      gate_closed_ = true;
      captured_unknown_ = capturing;
      return std::unexpected(
          StepFailure{.error = StepError::kUnknown,
                      .detail = std::format("generated StoreCells upload: {}", copied.text())});
    }
    return {};
  }
  void MarkSubmitted() {
    dirty_ = true;
    captured_unknown_ = false;
    AdvanceEpoch();
  }
  std::expected<void, StepFailure> Poll(providers::DeviceExecution& execution) {
    if (!fence_)
      return gate_closed_
                 ? std::unexpected(StepFailure{
                       .error = StepError::kUnknown,
                       .detail = "generated StoreCells completion gate is closed without a fence"})
                 : std::expected<void, StepFailure>{};
    auto queried = execution.Query(*fence_);
    if (!queried)
      return std::unexpected(
          StepFailure{.error = StepError::kUnknown, .detail = queried.error().detail});
    if (*queried == providers::FenceState::kPending)
      return std::unexpected(StepFailure{.detail = "generated StoreCells use is still pending"});
    if (auto released = execution.Release(*fence_); !released)
      return std::unexpected(
          StepFailure{.error = StepError::kUnknown, .detail = released.error().detail});
    fence_.reset();
    dirty_ = false;
    captured_unknown_ = false;
    gate_closed_ = false;
    return {};
  }
  std::expected<void, StepFailure> Drain(providers::DeviceExecution& execution,
                                         providers::StreamId stream) {
    if (dirty_ && !fence_) {
      gate_closed_ = true;
      auto recorded = execution.Record(stream);
      if (!recorded)
        return std::unexpected(
            StepFailure{.error = StepError::kUnknown, .detail = recorded.error().detail});
      fence_ = *recorded;
    }
    return Poll(execution);
  }
  void Quarantine() {
    dirty_ = true;
    gate_closed_ = true;
  }

  void RetainRegistry() {
    if (!registered_) {
      registry_keep_ = Keep();
      registered_ = true;
    }
  }
  void ReleaseRegistry() {
    // Keep this entry alive while dropping the private execution lease. That
    // may synchronously destroy its private launch and the recording list.
    auto self = Keep();
    registered_ = false;
    registry_next_ = nullptr;
    resources_ = {};
    registry_keep_ = {};
  }
  void MarkUnknown() {
    dirty_ = true;
    gate_closed_ = true;
    captured_unknown_ = true;
  }
  bool populated() const { return populated_; }

  GeneratedStoreInputs* registry_next_ = nullptr;
  kernels::ResourceKeep registry_keep_{};  // owned and reset only by the launch registry
  bool registered_ = false;
  bool captured_unknown_ = false;
  bool gate_closed_ = false;

 protected:
  ~GeneratedStoreInputs() override {
    if (pinned_) providers::FreePinned(pinned_);
  }
  void Destroy() override {
    auto funder = funder_;
    const auto bytes = paid_bytes_;
    delete this;
    funder->release(bytes);
  }

 private:
  GeneratedStoreInputs(components::StepInput input, LoweredStep::GeneratedStoreRecipe recipe,
                       std::shared_ptr<const ConstructionFunder> funder, std::uint64_t paid_bytes)
      : input_(std::move(input)),
        recipe_(recipe),
        funder_(std::move(funder)),
        paid_bytes_(paid_bytes) {}
  components::StepInput input_{};
  LoweredStep::GeneratedStoreRecipe recipe_{};
  std::shared_ptr<const ConstructionFunder> funder_;
  kernels::ExecutionResources resources_{};
  std::uint64_t paid_bytes_ = 0;
  void* pinned_ = nullptr;
  std::optional<providers::FenceId> fence_{};
  std::uint64_t epoch_ = 0;
  std::uint32_t past_ = 0;
  bool initialized_ = false;
  bool populated_ = false;
  bool dirty_ = false;
};

class GeneratedCaptureOwner final {
 public:
  static std::expected<GeneratedCaptureOwner*, StepFailure> Create(
      GeneratedStoreInputs& generated) {
    const auto funder = generated.funder();
    constexpr std::uint64_t kCaptureNodeAllocationEnvelope = 64;
    std::uint64_t paid = 0;
    if (!funder ||
        __builtin_add_overflow(std::uint64_t{sizeof(GeneratedCaptureOwner)},
                               kCaptureNodeAllocationEnvelope, &paid) ||
        !funder->charge(paid))
      return std::unexpected(
          StepFailure{.detail = "generated capture-owner metadata was not funded"});
    auto* entry = new (std::nothrow) GeneratedCaptureOwner(generated, funder, paid);
    if (!entry) {
      funder->release(paid);
      return std::unexpected(StepFailure{.detail = "generated capture-owner allocation failed"});
    }
    return entry;
  }
  GeneratedCaptureOwner(const GeneratedCaptureOwner&) = delete;
  GeneratedCaptureOwner& operator=(const GeneratedCaptureOwner&) = delete;
  GeneratedStoreInputs* owner() const { return owner_; }
  const kernels::ExecutionResources& resources() const { return resources_; }
  GeneratedCaptureOwner* next = nullptr;
  static void DestroyList(GeneratedCaptureOwner* list) {
    while (list) {
      auto* next = list->next;
      auto funder = std::move(list->funder_);
      const auto paid = std::exchange(list->paid_, 0);
      delete list;
      if (funder) funder->release(paid);
      list = next;
    }
  }

 private:
  GeneratedCaptureOwner(GeneratedStoreInputs& owner,
                        std::shared_ptr<const ConstructionFunder> funder, std::uint64_t paid)
      : owner_(&owner),
        keep_(owner.Keep()),
        resources_(owner.resources()),
        funder_(std::move(funder)),
        paid_(paid) {}
  ~GeneratedCaptureOwner() = default;
  GeneratedStoreInputs* owner_;
  kernels::ResourceKeep keep_;
  kernels::ExecutionResources resources_{};
  std::shared_ptr<const ConstructionFunder> funder_;
  std::uint64_t paid_;
};

class GeneratedCaptureOwners final {
 public:
  GeneratedCaptureOwners() = default;
  explicit GeneratedCaptureOwners(GeneratedCaptureOwner* head) : head_(head) {}
  GeneratedCaptureOwners(const GeneratedCaptureOwners&) = delete;
  GeneratedCaptureOwners& operator=(const GeneratedCaptureOwners&) = delete;
  GeneratedCaptureOwners(GeneratedCaptureOwners&& other) noexcept
      : head_(std::exchange(other.head_, nullptr)) {}
  GeneratedCaptureOwners& operator=(GeneratedCaptureOwners&& other) noexcept {
    if (this == &other) return *this;
    GeneratedCaptureOwner::DestroyList(head_);
    head_ = std::exchange(other.head_, nullptr);
    return *this;
  }
  ~GeneratedCaptureOwners() { GeneratedCaptureOwner::DestroyList(head_); }
  GeneratedCaptureOwner* head() const { return head_; }
  void reset() {
    GeneratedCaptureOwner::DestroyList(head_);
    head_ = nullptr;
  }

 private:
  GeneratedCaptureOwner* head_ = nullptr;
};

namespace {

std::optional<base::Sha256Digest> ComparisonDeviceIdentity(const planner::DeviceFacts& facts) {
  if (facts.device_class.empty() || facts.arch == 0 || facts.multiprocessors == 0) return {};
  return kernels::ComparisonDeviceIdentity({.device_class = facts.device_class,
                                            .device_model = facts.device_model,
                                            .arch = facts.arch,
                                            .features = facts.features,
                                            .multiprocessors = facts.multiprocessors});
}

StepFailure Failure(const KernelFailure& f) {
  return {.error = f.error == KernelError::kUnknown ? StepError::kUnknown : StepError::kRejected,
          .detail = f.detail};
}

class ComparisonFunderAdapter final : public ir::StorageFunder {
 public:
  ComparisonFunderAdapter(std::shared_ptr<const ConstructionFunder> funder,
                          std::uint64_t control_bytes)
      : funder_(std::move(funder)), control_bytes_(control_bytes) {}
  ~ComparisonFunderAdapter() override {
    if (control_bytes_ != 0) funder_->release(control_bytes_);
  }
  bool Charge(std::uint64_t bytes) const override { return funder_->charge(bytes); }
  void Release(std::uint64_t bytes) const override { funder_->release(bytes); }

  static std::expected<std::shared_ptr<const ir::StorageFunder>, std::string> Create(
      const std::shared_ptr<const ConstructionFunder>& funder) {
    constexpr std::uint64_t kControlAllowance = sizeof(ComparisonFunderAdapter) + 2 * 64;
    if (!funder || !funder->charge(kControlAllowance))
      return std::unexpected("comparison funder adapter admission refused");
    return std::shared_ptr<const ir::StorageFunder>(
        std::make_shared<ComparisonFunderAdapter>(funder, kControlAllowance));
  }

 private:
  std::shared_ptr<const ConstructionFunder> funder_;
  std::uint64_t control_bytes_ = 0;
};

// The launch context queues on its own stream, never the one a caller
// names: a runner's job on another stream would not fence its work.
std::expected<void, StepFailure> SameStream(const LaunchContext& context,
                                            providers::NativeStream stream) {
  if (stream.handle != context.native_stream().handle) {
    return std::unexpected(
        StepFailure{.detail = "the step's stream is not its backend's (the runner's job stream)"});
  }
  return {};
}

// A visible staged input does not by itself own auxiliary storage attached to
// its native root. That extension requires the retained staged-input fact.
bool UncoveredComparisonInputRoot(const LoweredStep& lowered, const LoweredLeaf& leaf,
                                  const ComparisonBindingRange& range) {
  const auto* descriptor = PhysicalTensorOf(lowered, leaf.tensor);
  if (!descriptor) return false;
  const auto root =
      std::ranges::find(lowered.physical_roots, descriptor->root, &PhysicalRoot::root);
  return root != lowered.physical_roots.end() && root->operation == ir::Primitive(ir::Op::kInput) &&
         root->bytes > range.bytes;
}

bool ComparisonRangesCurrent(const LoweredStep& lowered, bool has_lanes) {
  const auto& ranges = lowered.comparison_bindings;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const auto& range = ranges[i];
    std::size_t matched = 0;
    std::uint64_t expected = 0, allocation_end = 0, range_end = 0;
    if (range.base == 0 || range.allocation_bytes == 0 || range.bytes == 0 ||
        range.alignment == 0 || !std::has_single_bit(range.alignment) ||
        __builtin_add_overflow(range.base, range.offset, &expected) ||
        __builtin_add_overflow(range.base, range.allocation_bytes, &allocation_end) ||
        __builtin_add_overflow(expected, range.bytes, &range_end) || range_end > allocation_end ||
        expected % range.alignment != 0)
      return false;
    for (const LoweredLeaf& leaf : lowered.leaves) {
      if (leaf.comparison_slab) {
        if (!std::ranges::contains(leaf.comparison_slab->bindings, i)) continue;
        ++matched;
        if (!CheckComparisonExpertSlab(leaf, ranges)) return false;
        continue;
      }
      if (leaf.primitive != range.original_root || leaf.part != range.part) continue;
      ++matched;
      if (leaf.comparison_staged) {
        if (!CheckComparisonStagedInput(lowered, leaf, range)) return false;
      } else if (UncoveredComparisonInputRoot(lowered, leaf, range)) {
        return false;
      }
      if (!leaf.tensor || !leaf.tensor->data ||
          reinterpret_cast<std::uintptr_t>(leaf.tensor->data) != expected ||
          range.bytes < ggml_nbytes(leaf.tensor))
        return false;
    }
    if (matched == 0 && (!lowered.comparison ||
                         lowered.comparison->scope != ComparisonProofScope::kWholeProgramNative ||
                         range.part || !CheckComparisonOmittedInput(lowered, range.original_root)))
      return false;
    // An authenticated omitted source Input has no physical allocation to
    // alias. Its logical range remains checked above and in the source seal.
    if (matched == 0) continue;
    for (std::size_t j = 0; j < i; ++j) {
      const auto& other = ranges[j];
      if (lowered.comparison &&
          lowered.comparison->scope == ComparisonProofScope::kWholeProgramNative && !other.part &&
          CheckComparisonOmittedInput(lowered, other.original_root))
        continue;
      if (other.allocation_class == range.allocation_class &&
          (other.base != range.base || other.allocation_bytes != range.allocation_bytes))
        return false;
      std::uint64_t other_allocation_end = 0;
      if (other.base == 0 || other.allocation_bytes == 0 ||
          __builtin_add_overflow(other.base, other.allocation_bytes, &other_allocation_end))
        return false;
      if (other.allocation_class != range.allocation_class && range.base < other_allocation_end &&
          other.base < allocation_end)
        return false;
      std::uint64_t other_begin = 0, other_end = 0;
      if (__builtin_add_overflow(other.base, other.offset, &other_begin) ||
          __builtin_add_overflow(other_begin, other.bytes, &other_end))
        return false;
      const bool overlaps = expected < other_end && other_begin < range_end;
      if (overlaps && other.allocation_class != range.allocation_class) return false;
      // LaneOrder only relates tensor identities. It cannot order distinct
      // leaves whose actual backing overlaps, so such recipes stay unproved.
      if (has_lanes && overlaps && range.original_root != other.original_root) return false;
    }
  }
  return true;
}

bool ComparisonLeafMatches(const LowerInput& input, const ComparisonBindingRange& range,
                           const LoweredLeaf& leaf) {
  if (leaf.primitive != range.original_root) return false;
  if (leaf.part == range.part) return true;
  if (leaf.part || !range.part) return false;
  const auto weight = std::ranges::find_if(input.io.weights, [&](const auto& candidate) {
    return input.io.graph.root(candidate.tensor) == range.original_root;
  });
  if (weight == input.io.weights.end() || weight->parts.size() != 1) return false;
  const auto parts = format::PartsOf(input.io.graph.tensor(range.original_root).format);
  return parts && parts->size() == 1 && weight->parts.front().second == *range.part &&
         parts->parts().front().role == *range.part;
}

std::optional<LoweredStep::GeneratedStoreRecipe> DescribeGeneratedRopeStore(
    const LowerInput& input, const LoweredStep& lowered) {
  const auto& step = input.token->step;
  const auto& graph = step.graph;
  if (step.saves.size() != 0 || step.state.size() != 1 || input.token->shape.segments.size() != 1 ||
      lowered.plan.steps.empty())
    return {};

  const components::StepState& state = step.state.front();
  if (!state.written || state.borrowed || state.tap || state.ref.segment != 0 ||
      graph.root(state.tensor) != state.tensor)
    return {};
  const components::StepInput* store_input = nullptr;
  const components::StepInput* position_input = nullptr;
  const components::StepInput* feature_input = nullptr;
  for (const auto& candidate : step.inputs) {
    switch (candidate.kind) {
      case components::InputKind::kStoreCells:
        if (store_input) return {};
        store_input = &candidate;
        break;
      case components::InputKind::kPositions:
        if (position_input) return {};
        position_input = &candidate;
        break;
      case components::InputKind::kFeatures:
        if (feature_input) return {};
        feature_input = &candidate;
        break;
      default:
        return {};
    }
  }
  if (!store_input || !position_input || !feature_input ||
      store_input->tensor == position_input->tensor ||
      store_input->tensor == feature_input->tensor ||
      position_input->tensor == feature_input->tensor || store_input->segment != 0 ||
      store_input->cells.capacity == 0 || store_input->cells.ring || store_input->cells.frozen ||
      store_input->cells.fixed || store_input->cells.block != 0 || store_input->cells.per_row ||
      store_input->cells.positions || store_input->cells.recent ||
      store_input->cells.first_row != 0 || store_input->cells.cell_pad != 256 ||
      store_input->read != 0 || store_input->sections != 1 || store_input->component != 0 ||
      store_input->layer != 0 || !store_input->what.empty() || store_input->host.def != nullptr ||
      store_input->host.ratio != 0 || store_input->host.overlap || store_input->ratio != 0 ||
      store_input->overlap || store_input->ring != 0 || store_input->visible_capacity != 0)
    return {};

  const planner::Segment& segment = input.token->shape.segments.front();
  if (segment.rows == 0 || segment.outputs != 0 || segment.past > store_input->cells.capacity ||
      segment.rows > store_input->cells.capacity - segment.past ||
      store_input->cells.capacity > static_cast<std::uint64_t>(INT64_MAX))
    return {};
  const auto& indices_desc = graph.tensor(store_input->tensor);
  if (indices_desc.format.layout != format::Layout::kDense ||
      indices_desc.format.encoding != format::Encoding::kI64 ||
      indices_desc.ne(0) != segment.rows || indices_desc.ne(1) != 1 || indices_desc.ne(2) != 1 ||
      indices_desc.ne(3) != 1)
    return {};
  const auto& positions_desc = graph.tensor(position_input->tensor);
  if (position_input->segment != ir::kJoinedSegment || position_input->sections != 1 ||
      positions_desc.format.layout != format::Layout::kDense ||
      positions_desc.format.encoding != format::Encoding::kI32 ||
      positions_desc.ne(0) != segment.rows || positions_desc.ne(1) != 1 ||
      positions_desc.ne(2) != 1 || positions_desc.ne(3) != 1)
    return {};

  const ir::Node* rope_node = nullptr;
  const ir::Node* cache_store = nullptr;
  std::size_t rope_count = 0, cache_store_count = 0;
  for (const ir::Node& node : graph.nodes()) {
    if (node.key == ir::Primitive(ir::Op::kRope)) {
      rope_node = &node;
      ++rope_count;
    } else if (node.key == ir::Primitive(ir::Op::kCacheStore)) {
      cache_store = &node;
      ++cache_store_count;
    } else if (node.key.op != ir::Op::kInput && node.key.op != ir::Op::kWeight &&
               node.key.op != ir::Op::kState && node.key.op != ir::Op::kView &&
               node.key.op != ir::Op::kReshape) {
      return {};
    }
  }
  if (rope_count != 1 || cache_store_count != 1 || !rope_node || !cache_store ||
      cache_store->nsrc < 3 || cache_store->src[2] != store_input->tensor ||
      graph.root(cache_store->src[0]) != state.tensor || rope_node->nsrc < 2 ||
      rope_node->src[1] != position_input->tensor ||
      rope_node->Params<ir::RopeParams>().mode != ir::RopeMode::kNeox ||
      graph.root(rope_node->src[0]) != graph.root(feature_input->tensor) ||
      graph.root(feature_input->tensor) != feature_input->tensor)
    return {};

  // The original row operand is only the rope result through at most the
  // exact reshape/view chain accepted by RopeStoreMatch. Keep the target
  // state view as an external subrange of the declared state root.
  ir::TensorId rows = cache_store->src[1];
  std::size_t row_views = 0;
  while (graph.producer(rows).valid()) {
    const ir::Node& producer = graph.node(graph.producer(rows));
    if (producer.key.op != ir::Op::kView && producer.key.op != ir::Op::kReshape) break;
    if (++row_views > 2 || producer.nsrc == 0) return {};
    const auto& desc = graph.tensor(rows);
    if (desc.view_offset != 0 || !ir::IsContiguous(desc)) return {};
    rows = producer.src[0];
  }
  if (graph.producer(rows) != graph.producer(rope_node->out[0])) return {};
  const auto& target = graph.tensor(cache_store->src[0]);
  const auto& state_desc = graph.tensor(state.tensor);
  const auto target_bytes = ir::StorageBytes(target);
  const auto state_bytes = ir::StorageBytes(state_desc);
  std::uint64_t target_end = 0;
  const auto encoding_type = [](format::Encoding encoding) -> std::optional<ggml_type> {
    switch (encoding) {
      case format::Encoding::kF16:
        return GGML_TYPE_F16;
      case format::Encoding::kQ8_0:
        return GGML_TYPE_Q8_0;
      case format::Encoding::kQ4_0:
        return GGML_TYPE_Q4_0;
      default:
        return {};
    }
  };
  const auto target_type = encoding_type(target.format.encoding);
  const auto state_type = encoding_type(state_desc.format.encoding);
  const auto valid_storage_format = [](const format::TensorFormat& format) {
    return (format.encoding == format::Encoding::kF16 && format.layout == format::Layout::kDense) ||
           ((format.encoding == format::Encoding::kQ8_0 ||
             format.encoding == format::Encoding::kQ4_0) &&
            format.layout == format::Layout::kGgmlBlocks);
  };
  if (!target_bytes || !state_bytes || graph.root(cache_store->src[0]) != state.tensor ||
      !target_type || !state_type || *target_type != *state_type ||
      target.ne(0) != state_desc.ne(0) || !valid_storage_format(target.format) ||
      !valid_storage_format(state_desc.format) || target.ne(1) != store_input->cells.capacity ||
      target.ne(2) != 1 || target.ne(3) != 1 || state_desc.ne(1) < store_input->cells.capacity ||
      state_desc.ne(2) != 1 || state_desc.ne(3) != 1 ||
      __builtin_add_overflow(target.view_offset, *target_bytes, &target_end) ||
      target_end > *state_bytes)
    return {};

  const InputSlot* input_slot = nullptr;
  for (const InputSlot& candidate : lowered.inputs) {
    if (candidate.tensor != store_input->tensor) continue;
    if (input_slot) return {};
    input_slot = &candidate;
  }
  const std::uint64_t index_bytes = std::uint64_t{segment.rows} * sizeof(std::int64_t);
  if (!input_slot || input_slot->kind != InputKind::kStoreCells || input_slot->segment != 0 ||
      input_slot->device_address == 0 || input_slot->bytes != index_bytes ||
      input_slot->outputs != 0 || input_slot->commit_save ||
      input_slot->cells.capacity != store_input->cells.capacity || input_slot->cells.ring ||
      input_slot->cells.fixed || input_slot->cells.block != 0 || input_slot->cells.per_row ||
      input_slot->cells.first_row != 0 || input_slot->cells.positions || input_slot->cells.recent ||
      input_slot->read != 0 || input_slot->sections != 1 || input_slot->ratio != 0 ||
      input_slot->overlap || input_slot->ring != 0)
    return {};

  const ggml_tensor* rope = nullptr;
  const ggml_tensor* set_rows = nullptr;
  std::size_t physical_ropes = 0, physical_stores = 0;
  for (const auto& physical : lowered.physical_tensors) {
    if (!physical.descriptor) return {};
    if (physical.descriptor->op == GGML_OP_ROPE) {
      rope = physical.descriptor;
      ++physical_ropes;
    } else if (physical.descriptor->op == GGML_OP_SET_ROWS) {
      set_rows = physical.descriptor;
      ++physical_stores;
    } else if (physical.descriptor->op != GGML_OP_NONE && physical.descriptor->op != GGML_OP_VIEW &&
               physical.descriptor->op != GGML_OP_RESHAPE) {
      return {};
    }
  }
  const bool rope_store_checker =
      rope && set_rows &&
      (set_rows->type == GGML_TYPE_F16 ? CheckRopeSetRows(rope, set_rows)
                                       : CheckRopeSetRowsQuant(rope, set_rows));
  const bool primitive_rope_checker =
      rope && set_rows && CheckRope(rope) &&
      (set_rows->type == GGML_TYPE_F16 ? CheckSetRows(set_rows) : CheckSetRowsQuant(set_rows));
  const bool rope_values_match = [&] {
    if (!rope || !set_rows || !set_rows->src[0]) return false;
    const ggml_tensor* values = set_rows->src[0];
    const ggml_tensor* current = values;
    bool reached_rope = false;
    for (std::size_t depth = 0; current && depth <= 2; ++depth) {
      if (current == rope) {
        reached_rope = true;
        break;
      }
      if ((current->op != GGML_OP_VIEW && current->op != GGML_OP_RESHAPE) ||
          current->src[0] == nullptr || current->view_offs != 0 || !ggml_is_contiguous(current) ||
          (current->view_src != nullptr && current->view_src != rope) ||
          current->data != rope->data)
        return false;
      current = current->src[0];
    }
    std::int64_t flattened_width = 0;
    if (!reached_rope || __builtin_mul_overflow(rope->ne[0], rope->ne[1], &flattened_width) ||
        rope->ne[3] != 1 || values->type != GGML_TYPE_F32 || values->ne[0] != flattened_width ||
        values->ne[1] != rope->ne[2] || values->ne[2] != 1 || values->ne[3] != 1 ||
        values->nb[1] != rope->nb[2] || ggml_nbytes(values) != ggml_nbytes(rope))
      return false;
    const auto& original_rows = graph.tensor(cache_store->src[1]);
    for (std::size_t dimension = 0; dimension < 4; ++dimension) {
      if (values->ne[dimension] < 0 ||
          original_rows.ne(dimension) != static_cast<std::uint64_t>(values->ne[dimension]))
        return false;
    }
    return true;
  }();
  const bool selected_fused_rope_store =
      std::ranges::any_of(lowered.plan.steps, [](const PlanStep& planned) {
        return planned.operation == ir::Fused(ir::FusedKind::kRopeStore);
      });
  if (physical_ropes != 1 || physical_stores != 1 || !rope || !set_rows ||
      set_rows->src[1] == nullptr || set_rows->src[1]->data == nullptr ||
      reinterpret_cast<std::uintptr_t>(set_rows->src[1]->data) != input_slot->device_address ||
      !rope_values_match ||
      (selected_fused_rope_store ? !rope_store_checker : !primitive_rope_checker))
    return {};

  const auto state_leaf = std::ranges::find(lowered.leaves, state.tensor, &LoweredLeaf::primitive);
  if (state_leaf == lowered.leaves.end() || !state_leaf->tensor || state_leaf->tensor->view_src)
    return {};
  const auto state_physical =
      std::ranges::find(lowered.physical_tensors, state_leaf->tensor, &PhysicalTensor::descriptor);
  if (state_physical == lowered.physical_tensors.end() || !state_bytes ||
      ggml_nbytes(state_leaf->tensor) < *state_bytes)
    return {};
  const std::uint64_t row_bytes =
      static_cast<std::uint64_t>(ggml_row_size(set_rows->type, set_rows->ne[0]));
  std::uint64_t target_write_end = 0;
  if (!target_bytes || row_bytes == 0 || set_rows->nb[1] < row_bytes ||
      __builtin_mul_overflow(store_input->cells.capacity - 1, set_rows->nb[1], &target_write_end) ||
      __builtin_add_overflow(target_write_end, row_bytes, &target_write_end) ||
      target_write_end > *target_bytes)
    return {};
  const auto feature_leaf =
      std::ranges::find(lowered.leaves, feature_input->tensor, &LoweredLeaf::primitive);
  const auto position_leaf =
      std::ranges::find(lowered.leaves, position_input->tensor, &LoweredLeaf::primitive);
  if (feature_leaf == lowered.leaves.end() || !feature_leaf->tensor ||
      position_leaf == lowered.leaves.end() || !position_leaf->tensor)
    return {};
  const auto root_and_offset =
      [](const ggml_tensor* tensor) -> std::optional<std::pair<const ggml_tensor*, std::uint64_t>> {
    std::uint64_t offset = 0;
    for (std::size_t depth = 0; tensor && tensor->view_src != nullptr; ++depth) {
      if (depth >= 4 || __builtin_add_overflow(offset, tensor->view_offs, &offset)) return {};
      tensor = tensor->view_src;
    }
    if (!tensor) return {};
    return std::pair{tensor, offset};
  };
  const auto actual_x = root_and_offset(rope->src[0]);
  const auto actual_input_root = root_and_offset(feature_leaf->tensor);
  const std::uint64_t expected_x_offset = graph.tensor(rope_node->src[0]).view_offset;
  std::uint64_t expected_total_offset = 0;
  if (!actual_x || !actual_input_root || actual_x->first != actual_input_root->first ||
      __builtin_add_overflow(actual_input_root->second, expected_x_offset,
                             &expected_total_offset) ||
      actual_x->second != expected_total_offset || rope->src[1] != position_leaf->tensor)
    return {};
  const ggml_tensor* cache_root = set_rows->src[2];
  std::uint64_t view_offset = 0;
  for (std::size_t depth = 0; cache_root && cache_root->view_src != nullptr; ++depth) {
    if (depth >= 4 || __builtin_add_overflow(view_offset, cache_root->view_offs, &view_offset))
      return {};
    cache_root = cache_root->view_src;
  }
  if (cache_root != state_leaf->tensor || view_offset != target.view_offset ||
      set_rows->view_src != state_leaf->tensor || set_rows->view_offs != target.view_offset)
    return {};

  bool fused_rope_store = false, primitive_rope = false, primitive_store = false;
  for (const PlanStep& planned : lowered.plan.steps) {
    if (planned.operation == ir::Fused(ir::FusedKind::kRopeStore)) {
      if (fused_rope_store ||
          (planned.implementation != kRopeSetRowsFused &&
           planned.implementation != kRopeSetRowsQuantName) ||
          (set_rows->type == GGML_TYPE_F16 && planned.implementation != kRopeSetRowsFused) ||
          (set_rows->type != GGML_TYPE_F16 && planned.implementation != kRopeSetRowsQuantName) ||
          planned.nodes.size() != 2 || planned.nodes[0] != rope || planned.nodes[1] != set_rows)
        return {};
      fused_rope_store = true;
    } else if (planned.operation == ir::Primitive(ir::Op::kRope)) {
      if (primitive_rope || planned.implementation != kRopeName || planned.nodes.size() != 1 ||
          planned.nodes.front() != rope)
        return {};
      primitive_rope = true;
    } else if (planned.operation == ir::Primitive(ir::Op::kCacheStore)) {
      if (primitive_store ||
          (set_rows->type == GGML_TYPE_F16 && planned.implementation != kSetRowsName) ||
          (set_rows->type != GGML_TYPE_F16 && planned.implementation != kSetRowsQuantName) ||
          planned.nodes.size() != 1 || planned.nodes.front() != set_rows)
        return {};
      primitive_store = true;
    } else {
      return {};
    }
  }
  if (fused_rope_store == (primitive_rope || primitive_store) ||
      (!fused_rope_store && (!primitive_rope || !primitive_store)))
    return {};
  if (fused_rope_store) {
    if (lowered.plan.steps.size() != 1) return {};
  } else if (lowered.plan.steps.size() != 2 ||
             lowered.plan.steps[0].operation != ir::Primitive(ir::Op::kRope) ||
             lowered.plan.steps[1].operation != ir::Primitive(ir::Op::kCacheStore)) {
    return {};
  }

  return LoweredStep::GeneratedStoreRecipe{.slot = *input_slot,
                                           .stage = input.token->shape.stage,
                                           .phase = input.token->shape.phase,
                                           .segment = segment,
                                           .row_invariant = false,
                                           .read_align = input.token->shape.read_align};
}

std::optional<LoweredStep::GeneratedStoreRecipe> DescribeGeneratedStore(
    const LowerInput& input, const LoweredStep& lowered) {
  if (!input.token || !input.comparison || input.comparison->mode != ComparisonRunMode::kFull ||
      input.token->step.saves.size() != 0 || input.token->step.state.empty() ||
      input.token->step.state.size() > kSetRowsGroupMax ||
      input.token->shape.segments.size() != 1 || lowered.plan.steps.empty())
    return {};

  if (auto rope = DescribeGeneratedRopeStore(input, lowered)) return rope;

  for (const PlanStep& step : lowered.plan.steps) {
    const bool ordinary_grouped = step.implementation == kSetRowsGroupedName &&
                                  step.nodes.size() >= 2 && step.nodes.size() <= kSetRowsGroupMax &&
                                  step.operation == ir::Fused(ir::FusedKind::kGroupedStores);
    const bool quant_grouped = step.implementation == kSetRowsQuantGroupName &&
                               step.nodes.size() >= 2 && step.nodes.size() <= kSetRowsGroupMax &&
                               step.operation == ir::Fused(ir::FusedKind::kGroupedStores);
    const bool ordinary_single = step.implementation == kSetRowsName && step.nodes.size() == 1 &&
                                 step.operation == ir::Primitive(ir::Op::kCacheStore);
    const bool quant_single = step.implementation == kSetRowsQuantName && step.nodes.size() == 1 &&
                              step.operation == ir::Primitive(ir::Op::kCacheStore);
    const bool grouped = ordinary_grouped || quant_grouped;
    const bool single = ordinary_single || quant_single;
    if (!grouped && !single) return {};
  }

  const components::StepInput* source = nullptr;
  for (const auto& candidate : input.token->step.inputs) {
    if (candidate.kind != components::InputKind::kStoreCells) continue;
    if (source != nullptr) return {};
    source = &candidate;
  }
  if (!source || source->segment != 0 || source->cells.capacity == 0 || source->cells.ring ||
      source->cells.frozen || source->cells.fixed || source->cells.block != 0 ||
      source->cells.per_row || source->cells.positions || source->cells.recent ||
      source->cells.first_row != 0 || source->cells.cell_pad != 256 || source->read != 0 ||
      source->sections != 1 || source->component != 0 || source->layer != 0 ||
      !source->what.empty() || source->host.def != nullptr || source->host.ratio != 0 ||
      source->host.overlap || source->ratio != 0 || source->overlap || source->ring != 0 ||
      source->visible_capacity != 0)
    return {};

  const planner::Segment& segment = input.token->shape.segments.front();
  const auto& input_desc = input.token->step.graph.tensor(source->tensor);
  if (segment.rows == 0 || segment.outputs != 0 || segment.past > source->cells.capacity ||
      segment.rows > source->cells.capacity - segment.past ||
      source->cells.capacity > static_cast<std::uint64_t>(INT64_MAX) ||
      input_desc.format.layout != format::Layout::kDense ||
      input_desc.format.encoding != format::Encoding::kI64 || input_desc.ne(0) != segment.rows ||
      input_desc.ne(1) != 1 || input_desc.ne(2) != 1 || input_desc.ne(3) != 1)
    return {};

  const InputSlot* input_slot = nullptr;
  for (const auto& slot : lowered.inputs) {
    if (slot.tensor != source->tensor) continue;
    if (input_slot != nullptr) return {};
    input_slot = &slot;
  }
  const std::uint64_t index_bytes = std::uint64_t{segment.rows} * sizeof(std::int64_t);
  if (!input_slot || input_slot->kind != InputKind::kStoreCells || input_slot->segment != 0 ||
      input_slot->device_address == 0 || input_slot->bytes != index_bytes ||
      input_slot->outputs != 0 || input_slot->commit_save ||
      input_slot->cells.capacity != source->cells.capacity || input_slot->cells.ring ||
      input_slot->cells.fixed || input_slot->cells.block != 0 || input_slot->cells.per_row ||
      input_slot->cells.first_row != 0 || input_slot->cells.positions || input_slot->cells.recent ||
      input_slot->read != 0 || input_slot->sections != 1 || input_slot->ratio != 0 ||
      input_slot->overlap || input_slot->ring != 0)
    return {};

  std::array<const ggml_tensor*, kSetRowsGroupMax> stores{};
  std::size_t store_count = 0;
  for (const PlanStep& step : lowered.plan.steps) {
    if (step.nodes.size() > kSetRowsGroupMax - store_count) return {};
    for (const ggml_tensor* node : step.nodes) stores[store_count++] = node;
  }
  if (store_count != input.token->step.state.size()) return {};
  const ggml_tensor* rows = nullptr;
  const ggml_tensor* indices = nullptr;
  std::size_t physical_stores = 0;
  for (const auto& physical : lowered.physical_tensors) {
    if (!physical.descriptor) return {};
    if (physical.descriptor->op == GGML_OP_SET_ROWS)
      ++physical_stores;
    else if (physical.descriptor->op != GGML_OP_NONE && physical.descriptor->op != GGML_OP_VIEW)
      return {};
  }
  if (physical_stores != store_count) return {};
  for (std::size_t i = 0; i < store_count; ++i) {
    const ggml_tensor* node = stores[i];
    if (!node || node->op != GGML_OP_SET_ROWS || !node->src[0] || !node->src[1] || !node->src[2] ||
        node->src[0]->view_src || node->src[1]->view_src || node->src[2]->view_src ||
        node->view_src != node->src[2] || node->view_offs != 0)
      return {};
    if (i == 0) {
      rows = node->src[0];
      indices = node->src[1];
    } else if (node->src[0] != rows || node->src[1] != indices) {
      return {};
    }
    stores[i] = node;
  }
  if (!rows || !indices || indices->type != GGML_TYPE_I64 || indices->ne[0] != segment.rows ||
      indices->ne[1] != 1 || indices->ne[2] != 1 || indices->ne[3] != 1 ||
      reinterpret_cast<std::uintptr_t>(indices->data) != input_slot->device_address ||
      ggml_nbytes(indices) != index_bytes || rows->ne[1] != segment.rows || rows->ne[2] != 1 ||
      rows->ne[3] != 1)
    return {};
  std::size_t store_at = 0;
  for (const PlanStep& step : lowered.plan.steps) {
    if (step.implementation == kSetRowsGroupedName ||
        step.implementation == kSetRowsQuantGroupName) {
      const auto group =
          std::span<const ggml_tensor* const>(stores).subspan(store_at, step.nodes.size());
      const bool checked = step.implementation == kSetRowsGroupedName
                               ? CheckSetRowsGroup(group).has_value()
                               : CheckSetRowsQuantGroup(group).has_value();
      if (!checked) return {};
    } else if (step.implementation == kSetRowsName ? !CheckSetRows(stores[store_at])
                                                   : !CheckSetRowsQuant(stores[store_at])) {
      return {};
    }
    store_at += step.nodes.size();
  }

  std::array<bool, kSetRowsGroupMax> matched_states{};
  for (const auto& state : input.token->step.state) {
    if (!state.written || state.borrowed || state.tap || state.ref.segment != 0) return {};
    const auto& state_desc = input.token->step.graph.tensor(state.tensor);
    const bool valid_state_format = (state_desc.format.encoding == format::Encoding::kF16 &&
                                     state_desc.format.layout == format::Layout::kDense) ||
                                    ((state_desc.format.encoding == format::Encoding::kQ8_0 ||
                                      state_desc.format.encoding == format::Encoding::kQ4_0) &&
                                     state_desc.format.layout == format::Layout::kGgmlBlocks);
    if (!valid_state_format || state_desc.ne(0) != static_cast<std::uint64_t>(rows->ne[0]) ||
        state_desc.ne(1) != source->cells.capacity || state_desc.ne(2) != 1 ||
        state_desc.ne(3) != 1)
      return {};
    const auto leaf = std::ranges::find(lowered.leaves, state.tensor, &LoweredLeaf::primitive);
    if (leaf == lowered.leaves.end() || !leaf->tensor || leaf->tensor->view_src) return {};
    const auto storage_bytes = ir::StorageBytes(state_desc);
    const auto state_physical =
        std::ranges::find(lowered.physical_tensors, leaf->tensor, &PhysicalTensor::descriptor);
    if (!storage_bytes || state_physical == lowered.physical_tensors.end() ||
        ggml_nbytes(leaf->tensor) < *storage_bytes)
      return {};
    std::size_t matches = 0;
    for (std::size_t i = 0; i < store_count; ++i) {
      if (stores[i]->src[2] != leaf->tensor) continue;
      ++matches;
      matched_states[i] = true;
      const bool encoding_matches = (state_desc.format.encoding == format::Encoding::kF16 &&
                                     stores[i]->type == GGML_TYPE_F16) ||
                                    (state_desc.format.encoding == format::Encoding::kQ8_0 &&
                                     stores[i]->type == GGML_TYPE_Q8_0) ||
                                    (state_desc.format.encoding == format::Encoding::kQ4_0 &&
                                     stores[i]->type == GGML_TYPE_Q4_0);
      if (!encoding_matches) return {};
      const std::uint64_t row_bytes =
          static_cast<std::uint64_t>(ggml_row_size(stores[i]->type, stores[i]->ne[0]));
      std::uint64_t write_end = 0;
      if (row_bytes == 0 || stores[i]->nb[1] < row_bytes ||
          __builtin_mul_overflow(source->cells.capacity - 1, stores[i]->nb[1], &write_end) ||
          __builtin_add_overflow(write_end, row_bytes, &write_end) || write_end > *storage_bytes ||
          write_end > ggml_nbytes(stores[i]->src[2]))
        return {};
      if (stores[i]->src[2]->ne[1] != static_cast<std::int64_t>(source->cells.capacity) ||
          stores[i]->src[2]->ne[2] != 1 || stores[i]->src[2]->ne[3] != 1 ||
          stores[i]->ne[1] != stores[i]->src[2]->ne[1])
        return {};
    }
    if (matches != 1) return {};
  }
  if (input.token->step.state.size() != store_count ||
      !std::ranges::all_of(std::span(matched_states).first(store_count),
                           [](bool matched) { return matched; }))
    return {};

  // The private generated upload must be the only writer/reader preparation
  // associated with this physical index leaf in the admitted one-step recipe.
  for (const InputSlot& other : lowered.inputs) {
    if (other.tensor == source->tensor) continue;
    std::uint64_t end = 0, index_end = 0;
    if (other.device_address == 0 ||
        __builtin_add_overflow(other.device_address, other.bytes, &end) ||
        __builtin_add_overflow(input_slot->device_address, input_slot->bytes, &index_end))
      return {};
    if (other.device_address < index_end && input_slot->device_address < end) return {};
  }

  return LoweredStep::GeneratedStoreRecipe{.slot = *input_slot,
                                           .stage = input.token->shape.stage,
                                           .phase = input.token->shape.phase,
                                           .segment = segment,
                                           .row_invariant = input.token->shape.row_invariant,
                                           .read_align = input.token->shape.read_align};
}

// What a planned step holds on the host besides its arena (the plan's
// steps, the activations' places and the bound launches, an upper bound a
// node measured on GB10).
constexpr std::uint64_t kPlanNodeHostBytes = 512;

class GgmlRecorded final : public Recorded {
 public:
  GgmlRecorded(const StepLaunch* owner, std::uint64_t generation, CapturedGraph graph,
               GeneratedCaptureOwners generated)
      : Recorded(owner),
        generated_(std::move(generated)),
        graph_(std::move(graph)),
        generation_(generation) {}
  std::size_t nodes() const override { return graph_.nodes(); }
  double capture_seconds() const override { return graph_.capture_seconds(); }
  double instantiate_seconds() const override { return graph_.instantiate_seconds(); }
  std::expected<double, StepFailure> gpu_duration_ns() const override {
    auto valid = static_cast<const GgmlStepLaunch*>(owner())->ValidateRecording(*this);
    if (!valid) return std::unexpected(valid.error());
    auto duration = graph_.gpu_duration_ns();
    if (!duration) return std::unexpected(Failure(duration.error()));
    return *duration;
  }
  const CapturedGraph& graph() const { return graph_; }
  std::uint64_t generation() const { return generation_; }
  GeneratedCaptureOwner* generated() const { return generated_.head(); }

 private:
  // Declared before graph_: graph destruction precedes release of its copy sources.
  GeneratedCaptureOwners generated_;
  CapturedGraph graph_;
  std::uint64_t generation_;
};

bool DescribeComparisonRemainderCall(const LaunchContext& launch, const PlanStep& step,
                                     ComparisonRemainderCallFacts& facts) {
  if (step.nodes.empty()) return false;
  const auto scratch = PlanCallScratchOn(launch, step);
  if (!scratch) return false;
  facts.scratch_bytes = *scratch;
  if (*scratch != 0) {
    const auto pool = DescribeQ8MainPool(launch);
    if (!pool) return false;
    facts.pool_base = pool->base;
    facts.pool_bytes = pool->bytes;
    facts.pool_lanes = pool->lanes;
    facts.pool_faulted = pool->faulted;
  }
  if ((step.implementation == kMulMatVector || step.implementation == "ggml.mul_mat.mmf") &&
      step.operation == ir::Primitive(ir::Op::kMatMul) && step.nodes.size() == 1) {
    const auto selected = SelectMulMat(launch, step.nodes.front());
    if (!selected) return false;
    facts.has_mul_mat_selection = true;
    facts.mul_mat_vector = *selected == MulMatPath::kVector;
    facts.float_selection = *selected;
  }
  const bool quantized_vector =
      step.implementation == kMulMatVecQ || step.implementation == "ggml.mul_mat_id.mmvq";
  const bool quantized_fused = step.implementation == "ggml.mul_mat_glu.mmvq_fused" ||
                               step.implementation == "ggml.mul_mat_geglu.mmvq_fused";
  const bool quantized_tile = step.implementation == "ggml.mul_mat.mmq";
  if (quantized_vector || quantized_fused || quantized_tile ||
      step.implementation == kMmvqPreparedName) {
    if (step.nodes.size() != (quantized_fused ? 3U : 1U)) return false;
    std::optional<ggml_tensor> prepared;
    const ggml_tensor* product = step.nodes[quantized_fused ? 1 : 0];
    if (step.implementation == kMmvqPreparedName) {
      auto original = MmvqPreparedOriginal(product);
      if (!original) return false;
      prepared = *original;
      product = &*prepared;
    }
    const auto selected = SelectMulMatQ(launch, product);
    if (!selected ||
        *selected != (quantized_tile ? QuantMulMatPath::kTile : QuantMulMatPath::kVector))
      return false;
    facts.quantized_selection = *selected;
  }
  if (step.implementation == kMulMatCublas && step.operation == ir::Primitive(ir::Op::kMatMul) &&
      step.nodes.size() == 1) {
    const auto selected = SelectMulMat(launch, step.nodes[0]);
    const auto plan = PlanMulMatCublas(launch, step.nodes[0]);
    const auto* library = launch.cublas();
    if (!selected || *selected != MulMatPath::kCublas || !plan || !library) return false;
    const auto identity = library->ConfigurationIdentity();
    if (!identity) return false;
    facts.has_mul_mat_selection = true;
    facts.float_selection = *selected;
    facts.cublas_plan = *plan;
    facts.cublas_configuration = *identity;
    facts.library_workspace_base = library->workspace().base;
    facts.library_workspace_bytes = library->workspace().size.value();
    const auto pool = DescribeQ8MainPool(launch);
    if (!pool) return false;
    facts.pool_base = pool->base;
    facts.pool_bytes = pool->bytes;
    facts.pool_lanes = pool->lanes;
    facts.pool_faulted = pool->faulted;
  }
  if (ComparisonBf16Call(step)) {
    const auto* node = step.nodes.front();
    if (UsesGemvBf16Vector(node)) {
      facts.bf16_product = ComparisonBf16Product::kVector;
    } else {
      // The pinned launcher can fall back on library/cache/API state. Only a
      // shape absent from its real table proves this call is always GemmEx.
      if (GemmBf16Pinned(node) &&
          internal::HasPinnedBf16GemmShape(static_cast<int>(node->src[0]->ne[1]),
                                           static_cast<int>(node->src[1]->ne[1]),
                                           static_cast<int>(node->src[0]->ne[0])))
        return false;
      const auto* library = launch.cublas();
      if (!library) return false;
      const auto identity = library->ConfigurationIdentity();
      const auto pool = DescribeQ8MainPool(launch);
      if (!identity || !pool) return false;
      facts.bf16_product = ComparisonBf16Product::kGemmEx;
      facts.cublas_configuration = *identity;
      facts.library_workspace_base = library->workspace().base;
      facts.library_workspace_bytes = library->workspace().size.value();
      facts.pool_base = pool->base;
      facts.pool_bytes = pool->bytes;
      facts.pool_lanes = pool->lanes;
      facts.pool_faulted = pool->faulted;
    }
  }
  if (step.implementation == kMulMatGluFused &&
      step.operation == ir::Fused(ir::FusedKind::kMatMulGlu) && step.nodes.size() == 3) {
    const auto selected = SelectMulMat(launch, step.nodes[1]);
    if (!selected) return false;
    facts.has_mul_mat_selection = true;
    facts.mul_mat_vector = *selected == MulMatPath::kVector;
  }
  if (step.implementation == kMulMatAddFused &&
      step.operation == ir::Fused(ir::FusedKind::kMatMulAdd) && step.nodes.size() == 2) {
    const auto selected = SelectMulMat(launch, step.nodes[0]);
    if (!selected) return false;
    facts.has_mul_mat_selection = true;
    facts.mul_mat_vector = *selected == MulMatPath::kVector;
  }
  if (step.implementation == "ggml.flash_attn_ext.vec" && step.nodes.size() == 1) {
    const auto planned = PlanFlashAttnVec(launch, step.nodes.front());
    if (!planned) return false;
    facts.has_attention_plan = true;
    facts.attention_columns_per_block = planned->columns_per_block;
    facts.attention_parallel_blocks = planned->parallel_blocks;
    facts.attention_scratch_bytes = planned->scratch;
    facts.attention_mask_prepass = planned->mask_prepass;
  }
  return true;
}

// The same concrete wrapper checks used by BoundGraph::BindImpl, without
// constructing a second executable or importing CUDA symbols into tensors.
bool CheckWholeNativeCall(const execution::Registry& registry, const PlanStep& step) {
  const auto at = registry.Find(step.implementation);
  if (!at || registry.at(*at).operation != step.operation || step.nodes.empty()) return false;
  const auto& implementation = registry.at(*at);
  if (UsesCublas(step.implementation) &&
      ((step.implementation != kMulMatCublas && !ComparisonBf16Call(step)) || step.lane != 0))
    return false;
  if (step.operation == ir::Fused(ir::FusedKind::kRmsNormMul)) {
    auto kernel = RmsNormMulKernel::Bind(implementation);
    return kernel && step.nodes.size() == 2 && kernel->Check(step.nodes[0], step.nodes[1]);
  }
  auto kernel = Kernel::Bind(implementation);
  return kernel &&
         kernel->Check(std::span<const ggml_tensor* const>(step.nodes.data(), step.nodes.size()));
}

}  // namespace

class GgmlStepProgram final : public StepProgram, public RoutedProgram {
 public:
  GgmlStepProgram(const GgmlStepLaunch* const* launch, const std::uint64_t* generation,
                  const execution::Registry& registry, LoweredStep lowered, std::uint64_t pool,
                  std::uint64_t private_pool, kernels::LaneConfiguration lanes,
                  std::shared_ptr<const kernels::BackendOwner> owner,
                  GeneratedStoreInputs* generated, kernels::ResourceKeep generated_keep,
                  const LaunchContext* measuring, ComparisonPlanIdentity comparison_identity,
                  const SelectedSource* selected_source)
      : launch_(launch),
        generation_(generation),
        registry_(registry),
        measuring_(measuring),
        comparison_identity_(comparison_identity),
        selected_source_(selected_source),
        lowered_(std::move(lowered)),
        pool_(pool),
        private_pool_(private_pool),
        lanes_(lanes),
        owner_(std::move(owner)),
        generated_(generated),
        generated_keep_(std::move(generated_keep)) {
    for (const PlanStep& step : lowered_.plan.steps) {
      if (const auto at = registry_.Find(step.implementation); at && registry_.at(*at).capability)
        capturable_ = capturable_ && registry_.at(*at).capability->capturable;
      nodes_ += step.nodes.size();
      const auto selected =
          step.nodes.size() == 1
              ? std::ranges::find(lowered_.selected, step.nodes[0], &SelectedNode::tensor)
              : lowered_.selected.end();
      if (step.implementation == kAttentionF32) {
        launches_ += AttentionF32Plan::kLaunchBound;
      } else if (step.implementation == kConv2dTiledF32) {
        // Lowering and physical admission have checked this descriptor already.
        // Each stripe is four actual calls, not one compound launch.
        const auto tile = CheckConv2dTiledF32(step.nodes.front());
        base::Check(tile.has_value(),
                    "an admitted tiled convolution retains its checked descriptor");
        launches_ += tile->launches;
      } else {
        launches_ +=
            selected == lowered_.selected.end() ? step.nodes.size() : selected->call->launches();
      }
    }
    if (measuring_ && lowered_.comparison_bindable &&
        std::ranges::any_of(lowered_.plan.steps, [](const PlanStep& step) {
          return step.implementation == kMulMatCublas || step.implementation == "llmp.gemm.bf16" ||
                 step.implementation == kGemmBf16LtName ||
                 step.implementation == "ggml.mul_mat.mmf";
        }))
      comparison_product_identity_ = ComparisonProductScheduleIdentity(
          lowered_, *measuring_, &DescribeComparisonRemainderCall);
    selected_host_ += lowered_.leaves.capacity() * (sizeof(std::optional<ComparisonExpertSlab>) +
                                                    sizeof(std::optional<ComparisonStagedInput>));
    selected_host_ += lowered_.physical_roots.capacity() * sizeof(PhysicalRoot);
    selected_host_ += lowered_.physical_tensors.capacity() * sizeof(PhysicalTensor);
    selected_host_ += lowered_.saves.capacity() * sizeof(SavedLeaf);
    for (const auto& save : lowered_.saves) selected_host_ += save.what.capacity() + 1;
    if (lowered_.comparison) selected_host_ += sizeof(*lowered_.comparison);
    selected_host_ += lowered_.comparison_bindings.capacity() * sizeof(ComparisonBindingRange);
    selected_host_ +=
        lowered_.comparison_omitted_inputs.capacity() * sizeof(ComparisonOmittedInput);
    if (lowered_.comparison_origins) {
      const auto& origins = *lowered_.comparison_origins;
      selected_host_ +=
          sizeof(origins) + origins.offsets.capacity() * sizeof(std::uint64_t) +
          origins.source_nodes.capacity() * sizeof(ir::NodeId) +
          origins.membership.capacity() * sizeof(std::uint8_t) +
          origins.expansion.capacity() * sizeof(ir::NodeId) +
          origins.planned_nodes.capacity() * sizeof(ir::Node) +
          origins.planned_tensors.capacity() * sizeof(ir::TensorDesc) +
          origins.original_to_planned.capacity() * sizeof(ir::TensorId) +
          origins.input_facts.capacity() * sizeof(ComparisonInputFact) +
          origins.original_bindings.capacity() * sizeof(ComparisonBindingRange) +
          origins.node_local.capacity() * sizeof(std::uint32_t) +
          origins.tensor_local.capacity() * sizeof(std::uint32_t) +
          origins.physical_local.capacity() * sizeof(std::uint32_t) +
          origins.value_ports.capacity() * sizeof(kernels::ComparisonRegionValuePort) +
          origins.accesses.capacity() * sizeof(ComparisonRegionAccessFact) +
          origins.lifetimes.capacity() * sizeof(ComparisonLifetimeFact);
    }
    if (lowered_.comparison_selected_graph) {
      const auto& graph = *lowered_.comparison_selected_graph;
      selected_host_ += sizeof(graph) + graph.planned_nodes.capacity() * sizeof(ir::Node) +
                        graph.planned_tensors.capacity() * sizeof(ir::TensorDesc);
    }
    for (const auto& selected : lowered_.selected) selected_host_ += selected.call->host_bytes();
    if (lowered_.comparison_bindable && !lowered_.selected.empty() && selected_source_)
      comparison_selected_identity_ =
          ComparisonSelectedScheduleIdentity(lowered_, registry_, *selected_source_, pool_);
    if (lowered_.weight_reads) {
      selected_host_ += lowered_.weight_reads->capacity() * sizeof(RoutedWeight);
      for (const auto& w : *lowered_.weight_reads) {
        selected_host_ += w.parts.capacity() * sizeof(decltype(w.parts)::value_type);
        for (const auto& part : w.parts) selected_host_ += part.first.capacity() + 1;
      }
    }
    if (!lowered_.routing.empty()) {
      selected_host_ +=
          lowered_.routing.capacity() * sizeof(LoweredRouting) +
          2 * (lowered_.routing.size() + 1) * sizeof(std::unique_ptr<RoutedRecording>);
      for (const auto& r : lowered_.routing) {
        selected_host_ += r.members.capacity() * sizeof(std::size_t) +
                          r.layout.readback.capacity() * sizeof(RoutedCopy) +
                          r.uploads.capacity() * sizeof(RoutedCopy) +
                          r.descriptors.capacity() * sizeof(decltype(r.descriptors)::value_type);
        // Preparation, complete route readback, all worklist/descriptor uploads,
        // and two opaque segment owners are accounted in the graph plan charge.
        launches_ += 3 + r.uploads.size();
        for (const auto& copy : r.layout.readback) launches_ += copy.rows;
      }
    }
  }

  bool capturable() const override {
    if (!capturable_) return false;
    if (lowered_.selected.empty()) return true;
    if (!bound_ || *launch_ == nullptr || bound_generation_ != *generation_ ||
        !(*launch_)->selected())
      return false;
    return std::ranges::all_of(lowered_.selected, [&](const auto& selected) {
      return selected.call->capturable(*(*launch_)->selected());
    });
  }
  std::uint64_t scratch_bytes() const override { return lowered_.extent; }
  std::uint64_t pool_bytes() const override { return pool_; }
  std::uint64_t private_pool_bytes() const override { return private_pool_; }
  std::uint64_t host_bytes() const override {
    return lowered_.arena->bytes() + (nodes_ * kPlanNodeHostBytes) + selected_host_ +
           (generated_ ? generated_->host_bytes() : 0);
  }
  std::uint64_t launches() const override { return launches_; }
  kernels::PlanCoverage plan_coverage() const override {
    return {.logical_activation_bytes = lowered_.logical_extent,
            .physical_activation_bytes = lowered_.extent,
            .unmaterialized_logical_bytes = lowered_.unmaterialized_logical_bytes,
            .unmaterialized_roots = lowered_.unmaterialized_roots};
  }
  void StreamPlanReport(const kernels::PlanReportSink& sink) const override {
    StreamPhysicalPlanReport(lowered_, registry_, sink,
                             {.pool_bytes = pool_,
                              .lanes = lanes_.count,
                              .lane_scratch_bytes = lanes_.scratch_bytes_per_lane,
                              .launches = launches_});
  }
  std::span<const InputSlot> inputs() const override { return lowered_.inputs; }
  std::span<const OutputSlot> outputs() const override { return lowered_.outputs; }
  std::span<const TensorUse> tensors() const override { return lowered_.tensors; }
  const execution::BoundPlan* bound() const override { return bound_ ? &bound_->bound() : nullptr; }
  std::optional<kernels::ComparisonRecipeSeal> comparison_recipe() const override {
    if (!lowered_.comparison) return {};
    auto seal = *lowered_.comparison;
    const bool current = bound_ && launch_ && *launch_ != nullptr &&
                         bound_generation_ == *generation_ && !(*launch_)->faulted() &&
                         (!owner_ || owner_->Ready().has_value());
    if (seal.state == kernels::ComparisonProofState::kBoundVerified && !current)
      seal.state = kernels::ComparisonProofState::kDescriptorVerified;
    return seal;
  }
  std::optional<ComparisonPlanIdentity> comparison_plan_identity() const override {
    if (comparison_identity_.source_graph_identity != 0 &&
        comparison_identity_.planned_graph_identity != 0)
      return comparison_identity_;
    if (!lowered_.comparison_origins) return {};
    const auto& origins = *lowered_.comparison_origins;
    if (origins.source_identity == 0 || origins.planned_identity == 0) return {};
    return ComparisonPlanIdentity{.source_graph_identity = origins.source_identity,
                                  .planned_graph_identity = origins.planned_identity};
  }
  const StepProgram& comparison_program() const override { return *this; }
  ComparisonRemainderResult CompareComparisonRemainder(
      const StepProgram& candidate,
      const components::OwnedComparisonRegionBoundary& common_boundary,
      ComparisonCertificatePhase phase) const override {
    const auto* peer = dynamic_cast<const GgmlStepProgram*>(&candidate.comparison_program());
    if (!peer) {
      return {.status = ComparisonRemainderStatus::kUnproved,
              .reason = "remainder comparison peer is not a retained GGML program"};
    }
    const auto current_recipe = comparison_recipe();
    const auto peer_recipe = peer->comparison_recipe();
    if (!current_recipe || !peer_recipe)
      return {.status = ComparisonRemainderStatus::kUnproved,
              .reason = "remainder comparison lacks authenticated backend declarations"};
    const auto current_context = ComparisonContextFor(phase);
    const auto peer_context = peer->ComparisonContextFor(phase);
    if (!current_context || !peer_context)
      return {.status = ComparisonRemainderStatus::kUnproved,
              .reason = "remainder comparison lacks the actual device planning context"};
    return llmp::kernels::ggml::CompareComparisonRemainder(
        lowered_, registry_, *current_recipe, current_context, peer->lowered_, peer->registry_,
        *peer_recipe, peer_context, common_boundary, phase, &DescribeComparisonRemainderCall);
  }
  ComparisonRegionDescriptorResult ProjectComparisonRegionDescriptor(
      const components::OwnedComparisonRegionBoundary& boundary) const override {
    ComparisonRegionDescriptorResult result;
    if (lowered_.whole_checkpoint_only) {
      result.reason = "checkpoint-only whole comparison cannot authorize a local projection";
      return result;
    }
    if (lowered_.measuring || bound_ || !generation_ ||
        (launch_ && *launch_ && (*launch_)->faulted()) || (owner_ && !owner_->Ready())) {
      result.reason = "descriptor projection requires a healthy unbound program";
      return result;
    }
    if (!lowered_.comparison_origins || !lowered_.comparison ||
        !lowered_.comparison_origins->descriptor_bindings_verified ||
        !ComparisonRangesCurrent(lowered_, false)) {
      result.reason = "program lacks authenticated descriptor bindings and source provenance";
      return result;
    }
    struct ClearPublishedAddresses {
      std::vector<ComparisonRegionValuePort>& ports;
      ~ClearPublishedAddresses() {
        for (auto& port : ports) port.address = 0;
      }
    } clear_addresses{lowered_.comparison_origins->value_ports};
    const ComparisonQ8Dispatch q8_dispatch{.describe_main_pool = &DescribeQ8MainPool,
                                           .describe_quantize_q8 = &DescribeQuantizeQ8,
                                           .describe_mmvq = &DescribeMmvqDispatch,
                                           .plan_mmvq = &PlanMulMatVecQ,
                                           .plan_prepared_mmvq = &PlanMmvqPrepared};
    const LaunchContext* context = ComparisonContextFor(ComparisonCertificatePhase::kDescriptor);
    const auto analysis = AnalyzeComparisonRegion(
        lowered_, boundary, registry_, *lowered_.comparison, ComparisonRegionPhase::kDescriptor,
        context, &q8_dispatch, ComparisonRegionPurpose::kExecutableComparison,
        &DescribeComparisonRemainderCall);
    result.status = analysis.status;
    result.seal = analysis.seal;
    result.required_members = analysis.required_members;
    result.reason = analysis.reason;
    return result;
  }
  ComparisonRegionDescriptorResult ProjectComparisonRegionCertificate(
      const components::OwnedComparisonRegionBoundary& boundary,
      ComparisonCertificatePhase phase) const override {
    ComparisonRegionDescriptorResult result;
    if (lowered_.whole_checkpoint_only) {
      result.reason = "checkpoint-only whole comparison cannot authorize a local projection";
      return result;
    }
    const bool descriptor = phase == ComparisonCertificatePhase::kDescriptor;
    if (lowered_.measuring || !generation_ || (owner_ && !owner_->Ready()) ||
        (descriptor && (bound_ || (launch_ && *launch_ && (*launch_)->faulted()))) ||
        (!descriptor && (!bound_ || !launch_ || *launch_ == nullptr ||
                         bound_generation_ != *generation_ || (*launch_)->faulted()))) {
      result.reason = descriptor
                          ? "production descriptor certificate requires a healthy ready program"
                          : "production bound certificate requires the current healthy binding";
      return result;
    }
    if (!lowered_.comparison_origins || !lowered_.comparison ||
        !lowered_.comparison_origins->descriptor_bindings_verified ||
        (!descriptor && !lowered_.comparison_origins->bindings_verified) ||
        !ComparisonRangesCurrent(lowered_, false)) {
      result.reason = "production certificate lacks authenticated source and binding facts";
      return result;
    }
    struct ClearPublishedAddresses {
      std::vector<ComparisonRegionValuePort>& ports;
      ~ClearPublishedAddresses() {
        for (auto& port : ports) port.address = 0;
      }
    } clear_addresses{lowered_.comparison_origins->value_ports};
    const ComparisonQ8Dispatch q8_dispatch{.describe_main_pool = &DescribeQ8MainPool,
                                           .describe_quantize_q8 = &DescribeQuantizeQ8,
                                           .describe_mmvq = &DescribeMmvqDispatch,
                                           .plan_mmvq = &PlanMulMatVecQ,
                                           .plan_prepared_mmvq = &PlanMmvqPrepared};
    const LaunchContext* context = ComparisonContextFor(phase);
    const auto analysis = AnalyzeComparisonRegion(
        lowered_, boundary, registry_, *lowered_.comparison,
        descriptor ? ComparisonRegionPhase::kDescriptor : ComparisonRegionPhase::kBound, context,
        &q8_dispatch, ComparisonRegionPurpose::kProductionCertificate,
        &DescribeComparisonRemainderCall);
    result.status = analysis.status;
    result.seal = analysis.seal;
    result.required_members = analysis.required_members;
    result.reason = analysis.reason;
    return result;
  }
  ComparisonRegionResult ProjectComparisonRegion(
      const components::OwnedComparisonRegionBoundary& boundary) const override {
    ComparisonRegionResult result;
    if (lowered_.whole_checkpoint_only) {
      result.reason = "checkpoint-only whole comparison cannot authorize a local projection";
      return result;
    }
    if (!bound_ || !launch_ || *launch_ == nullptr || bound_generation_ != *generation_ ||
        (*launch_)->faulted() || (owner_ && !owner_->Ready())) {
      result.reason = "comparison projection requires the current healthy bound program";
      return result;
    }
    if (!lowered_.comparison_origins || !lowered_.comparison ||
        !lowered_.comparison_origins->bindings_verified ||
        !ComparisonRangesCurrent(lowered_, false)) {
      result.reason = "program was lowered without owned comparison-region provenance";
      return result;
    }
    const auto whole = comparison_recipe();
    if (!whole) {
      result.reason = "whole-program comparison identities are unavailable";
      return result;
    }
    const ComparisonQ8Dispatch q8_dispatch{.describe_main_pool = &DescribeQ8MainPool,
                                           .describe_quantize_q8 = &DescribeQuantizeQ8,
                                           .describe_mmvq = &DescribeMmvqDispatch,
                                           .plan_mmvq = &PlanMulMatVecQ,
                                           .plan_prepared_mmvq = &PlanMmvqPrepared};
    const auto analysis = AnalyzeComparisonRegion(
        lowered_, boundary, registry_, *whole, ComparisonRegionPhase::kBound,
        &(*launch_)->context(), &q8_dispatch, ComparisonRegionPurpose::kExecutableComparison,
        &DescribeComparisonRemainderCall);
    result.status = analysis.status;
    result.required_members = analysis.required_members;
    result.value_ports = analysis.value_ports;
    result.program_nonce = comparison_program_nonce();
    result.generation = bound_generation_;
    result.serial = lowered_.comparison_origins->projection_serial;
    result.reason = analysis.reason;
    if (analysis.status == ComparisonRegionStatus::kClosed) {
      result.projection.emplace(
          MintComparisonRegionProjection(bound_generation_, result.serial, analysis.first,
                                         analysis.end, analysis.seal, analysis.inputs_preserved));
    }
    return result;
  }
  bool ValidateComparisonProjection(const ComparisonRegionProjection& projection) const override {
    if (lowered_.whole_checkpoint_only) return false;
    if (!bound_ || !launch_ || *launch_ == nullptr || bound_generation_ != *generation_ ||
        (*launch_)->faulted() || (owner_ && !owner_->Ready()) || !lowered_.comparison_origins ||
        !lowered_.comparison_origins->bindings_verified ||
        !ComparisonRangesCurrent(lowered_, false))
      return false;
    const auto span = AuthenticatedComparisonRegionSpan(
        projection, bound_generation_, lowered_.comparison_origins->projection_serial);
    return span.has_value() && span->second <= lowered_.plan.steps.size();
  }

  std::expected<void, StepFailure> RunComparisonPrefix(
      const ComparisonRegionProjection& projection, StepLaunch& launch,
      providers::NativeStream stream) const override {
    return RunComparisonSpan(projection, launch, stream, true);
  }
  std::expected<void, StepFailure> RunComparisonRegion(
      const ComparisonRegionProjection& projection, StepLaunch& launch,
      providers::NativeStream stream) const override {
    return RunComparisonSpan(projection, launch, stream, false);
  }

  std::expected<void, StepFailure> TryUpdateGeneratedInputs(
      const planner::StepShape& shape) override {
    if (!generated_) return {};
    if (!launch_ || *launch_ == nullptr)
      return std::unexpected(StepFailure{.detail = "generated input has no current launch"});
    const auto& ggml_launch = **launch_;
    if (ggml_launch.capturing_ || ggml_launch.registry_root_->capturing_)
      return std::unexpected(
          StepFailure{.detail = "generated StoreCells values cannot change during capture"});
    if (auto valid = generated_->ValidateUpdate(shape); !valid) return valid;
    if (auto drained = const_cast<GgmlStepLaunch*>(*launch_)->DrainGeneratedInputs(); !drained)
      return drained;
    return generated_->Update(shape);
  }

  // The pool scratch against the launch's pools, then every implementation
  // bound.
  std::expected<void, StepFailure> Bind(StepLaunch& launch) override {
    bound_.reset();
    bound_generation_ = 0;
    if (lowered_.comparison && lowered_.comparison_bindable) {
      lowered_.comparison->state = kernels::ComparisonProofState::kDescriptorVerified;
      if (lowered_.comparison->missing == kernels::ComparisonMissing::kBindings)
        lowered_.comparison->missing = kernels::ComparisonMissing::kNone;
    } else if (lowered_.comparison) {
      lowered_.comparison->state = kernels::ComparisonProofState::kUnproved;
    }
    if (owner_) {
      if (auto ready = owner_->Ready(); !ready)
        return std::unexpected(StepFailure{.detail = ready.error()});
    }
    if (lowered_.measuring) {
      return std::unexpected(StepFailure{.detail = "a measurement does not bind"});
    }
    if (*launch_ == nullptr || &launch != *launch_) {
      if (lowered_.comparison && lowered_.comparison_bindable && !lowered_.measuring) {
        lowered_.comparison->state = kernels::ComparisonProofState::kUnproved;
        lowered_.comparison->missing = kernels::ComparisonMissing::kBindings;
      }
      return std::unexpected(StepFailure{.detail = "a GGML step binds to its backend's launch"});
    }
    LaunchContext& context = (*launch_)->context();
    GraphPlan& plan = lowered_.plan;
    if (context.lanes() != lanes_.count ||
        (lanes_.count != 0 && context.scratch_size(1).value() != lanes_.scratch_bytes_per_lane))
      return std::unexpected(
          StepFailure{.detail = "the launch's lane resources differ from its plan"});
    auto scratch = PlanScratch(context, plan);
    if (!scratch) return std::unexpected(Failure(scratch.error()));
    if (*scratch > context.scratch_size(0).value()) {
      return std::unexpected(StepFailure{
          .detail = std::format("the step's scratch ({} bytes) exceeds the pool ({} bytes)",
                                *scratch, context.scratch_size(0).value())});
    }
    auto lane = PlanLaneScratch(context, plan);
    if (!lane) return std::unexpected(Failure(lane.error()));
    if (*lane > context.scratch_size(1).value()) {
      return std::unexpected(
          StepFailure{.detail = std::format(
                          "the step's lane scratch ({} bytes) exceeds a lane's pool", *lane)});
    }
    if (lowered_.comparison && !lowered_.measuring &&
        (lowered_.comparison_bindable || lowered_.comparison_origins)) {
      const bool current = ComparisonRangesCurrent(lowered_, !lowered_.plan.regions.empty());
      if (lowered_.comparison_origins) lowered_.comparison_origins->bindings_verified = current;
      if (lowered_.comparison_bindable && !current) {
        lowered_.comparison->state = kernels::ComparisonProofState::kUnproved;
        lowered_.comparison->missing = kernels::ComparisonMissing::kBindings;
        return std::unexpected(
            StepFailure{.detail = "comparison ranges no longer match their owned bound leaves"});
      }
    }
    if (lowered_.comparison && lowered_.comparison_bindable &&
        std::ranges::any_of(plan.steps, [](const PlanStep& step) {
          return step.implementation == kMulMatCublas || step.implementation == "llmp.gemm.bf16" ||
                 step.implementation == kGemmBf16LtName ||
                 step.implementation == "ggml.mul_mat.mmf";
        })) {
      const auto current =
          ComparisonProductScheduleIdentity(lowered_, context, &DescribeComparisonRemainderCall);
      if (!comparison_product_identity_ || !current || *current != *comparison_product_identity_) {
        lowered_.comparison->state = kernels::ComparisonProofState::kUnproved;
        lowered_.comparison->missing = kernels::ComparisonMissing::kBindings;
        return std::unexpected(
            StepFailure{.detail = "comparison product method/configuration/workspace differs from "
                                  "its owned descriptor"});
      }
    }
    const bool shared_q8_recipe =
        std::ranges::any_of(lowered_.plan.steps, [](const PlanStep& step) {
          return step.implementation == kQuantizeQ8Name ||
                 step.implementation == kMmvqPreparedName || step.implementation == kMulMatVecQ;
        });
    if (lowered_.comparison && !lowered_.measuring &&
        (shared_q8_recipe ||
         lowered_.comparison->scope == ComparisonProofScope::kWholeProgramNative)) {
      const auto reject_q8_binding = [&]() -> std::expected<void, StepFailure> {
        lowered_.comparison->state = kernels::ComparisonProofState::kUnproved;
        lowered_.comparison->missing = kernels::ComparisonMissing::kBindings;
        return std::unexpected(
            StepFailure{.detail = "comparison scratch aliases a readable or writable range"});
      };
      auto pool = DescribeQ8MainPool(context);
      if (lanes_.count != 0 || !plan.regions.empty() || !pool ||
          !ComparisonPoolDisjoint(lowered_, *pool, std::max(*scratch, private_pool_)))
        return reject_q8_binding();
    }
    if (lowered_.comparison_bindable && !lowered_.selected.empty()) {
      const auto reject_selected = [&]() -> std::expected<void, StepFailure> {
        lowered_.comparison->state = kernels::ComparisonProofState::kUnproved;
        lowered_.comparison->missing = kernels::ComparisonMissing::kBindings;
        return std::unexpected(StepFailure{
            .detail = "selected comparison source, operands or persistent resources changed"});
      };
      const auto current = selected_source_ ? ComparisonSelectedScheduleIdentity(
                                                  lowered_, registry_, *selected_source_, pool_)
                                            : std::nullopt;
      const auto persistent = (*launch_)->selected_workspace();
      const auto pool = DescribeQ8MainPool(context);
      std::uint64_t persistent_end = 0, pool_end = 0;
      if (!comparison_selected_identity_ || !current ||
          *current != *comparison_selected_identity_ || !(*launch_)->selected() ||
          (*launch_)->selected()->owner() != selected_source_ || !pool ||
          persistent.bytes != selected_source_->persistent_bytes() ||
          __builtin_add_overflow(persistent.base, persistent.bytes, &persistent_end) ||
          __builtin_add_overflow(pool->base, pool->bytes, &pool_end) ||
          (persistent.bytes && persistent.base < pool_end && pool->base < persistent_end) ||
          !ComparisonPoolDisjoint(lowered_,
                                  {.base = persistent.base,
                                   .bytes = persistent.bytes,
                                   .faulted = (*launch_)->faulted()},
                                  persistent.bytes))
        return reject_selected();
    }
    if (generated_) {
      if (auto initialized = generated_->Initialize(); !initialized) return initialized;
      const auto& recipe = *lowered_.generated_store;
      planner::StepShape initial{.stage = recipe.stage,
                                 .phase = recipe.phase,
                                 .segments = {recipe.segment},
                                 .row_invariant = recipe.row_invariant,
                                 .read_align = recipe.read_align};
      if (auto updated = generated_->Update(initial); !updated) return updated;
    }
    auto bound = BoundGraph::BindImpl(registry_, plan, lowered_.selected, (*launch_)->selected(),
                                      {context.workspace().base, context.scratch_size(0).value()},
                                      !lowered_.routing.empty());
    if (!bound) {
      if (lowered_.comparison && lowered_.comparison_bindable && !lowered_.measuring) {
        lowered_.comparison->state = kernels::ComparisonProofState::kUnproved;
        lowered_.comparison->missing = kernels::ComparisonMissing::kBindings;
      }
      return std::unexpected(Failure(bound.error()));
    }
    if (lowered_.comparison_bindable && !lowered_.selected.empty() &&
        lowered_.comparison->execution_method == ComparisonExecutionMethod::kCaptured &&
        std::ranges::any_of(lowered_.selected, [&](const SelectedNode& selected) {
          return !selected.call->capturable(*(*launch_)->selected());
        })) {
      lowered_.comparison->state = kernels::ComparisonProofState::kUnproved;
      lowered_.comparison->missing = kernels::ComparisonMissing::kBindings;
      return std::unexpected(StepFailure{
          .detail = "selected capture eligibility differs from the whole-program method"});
    }
    bound_.emplace(std::move(*bound));
    bound_generation_ = *generation_;
    if (lowered_.comparison && lowered_.comparison_bindable && !lowered_.measuring &&
        !(*launch_)->faulted() && (!owner_ || owner_->Ready().has_value()))
      lowered_.comparison->state = kernels::ComparisonProofState::kBoundVerified;
    // A seal that was already unproved stays unproved for its original reason.
    // Only a descriptor-verified recipe can be promoted, and actual binding
    // failures above are the only failures that replace its reason with Bindings.
    return {};
  }

  std::expected<void, StepFailure> Run(StepLaunch& launch,
                                       providers::NativeStream stream) const override {
    if (!lowered_.routing.empty())
      return std::unexpected(StepFailure{.detail = "routed program refuses whole-step Run"});
    if (owner_) {
      if (auto ready = owner_->Ready(); !ready)
        return std::unexpected(StepFailure{.detail = ready.error()});
    }
    if (!bound_ || &launch != *launch_ || bound_generation_ != *generation_) {
      return std::unexpected(StepFailure{.detail = "the step is not bound to this launch"});
    }
    if (auto same = SameStream((*launch_)->context(), stream); !same) return same;
    if (generated_) {
      auto& ggml_launch = static_cast<GgmlStepLaunch&>(launch);
      const bool capturing = ggml_launch.capturing_ || ggml_launch.registry_root_->capturing_;
      if (auto polled = ggml_launch.PollGeneratedInputs(); !polled) return polled;
      if (auto registered =
              ggml_launch.RegisterGenerated(generated_, capturing, RetainedResources());
          !registered)
        return registered;
      if (auto copied = generated_->Copy(stream, capturing); !copied) return copied;
    }
    if (auto ran = bound_->Run((*launch_)->context()); !ran) {
      return std::unexpected(Failure(ran.error()));
    }
    return {};
  }

  bool warm() const override {
    return !lowered_.prologue.empty() ||
           std::ranges::any_of(lowered_.inputs, [](const InputSlot& s) { return s.invariant; });
  }
  std::expected<void, StepFailure> RunWarm(StepLaunch& launch,
                                           providers::NativeStream stream) const override {
    if (lowered_.prologue.empty()) return Run(launch, stream);
    if (!lowered_.routing.empty())
      return std::unexpected(StepFailure{.detail = "routed program refuses whole-step Run"});
    if (owner_) {
      if (auto ready = owner_->Ready(); !ready)
        return std::unexpected(StepFailure{.detail = ready.error()});
    }
    if (!bound_ || &launch != *launch_ || bound_generation_ != *generation_) {
      return std::unexpected(StepFailure{.detail = "the step is not bound to this launch"});
    }
    if (auto same = SameStream((*launch_)->context(), stream); !same) return same;
    if (generated_) {
      auto& ggml_launch = static_cast<GgmlStepLaunch&>(launch);
      const bool capturing = ggml_launch.capturing_ || ggml_launch.registry_root_->capturing_;
      if (auto polled = ggml_launch.PollGeneratedInputs(); !polled) return polled;
      if (auto registered =
              ggml_launch.RegisterGenerated(generated_, capturing, RetainedResources());
          !registered)
        return registered;
      if (auto copied = generated_->Copy(stream, capturing); !copied) return copied;
    }
    if (auto ran = bound_->RunSkipping((*launch_)->context(), lowered_.prologue); !ran) {
      return std::unexpected(Failure(ran.error()));
    }
    return {};
  }

  std::optional<std::span<const RoutedWeight>> weight_reads() const override {
    if (!lowered_.weight_reads) return {};
    return std::span<const RoutedWeight>(*lowered_.weight_reads);
  }
  RoutedProgram* routed() override { return !lowered_.routing.empty() ? this : nullptr; }
  const RoutedProgram* routed() const override {
    return !lowered_.routing.empty() ? this : nullptr;
  }
  std::span<const RoutedWeight> weights() const override { return *lowered_.weight_reads; }
  std::size_t groups() const override { return lowered_.routing.size(); }
  const RoutedLayout& layout(std::size_t group = 0) const override {
    return lowered_.routing.at(group).layout;
  }
  std::span<const std::size_t> members(std::size_t group) const override {
    return lowered_.routing.at(group).members;
  }
  std::uint64_t pinned_bytes() const override { return lowered_.routed_pinned_bytes; }
  const base::Sha256Digest& routing_identity() const override { return lowered_.routing_identity; }
  const base::Sha256Digest& graph_identity() const override {
    return lowered_.routing.front().graph_identity;
  }
  std::expected<std::unique_ptr<Recorded>, StepFailure> Record(
      StepLaunch& launch, providers::NativeStream stream,
      const StepLaunch::Body& body) const override {
    if (!capturable())
      return std::unexpected(StepFailure{.detail = "selected program is not capturable"});
    if (!lowered_.routing.empty())
      return std::unexpected(StepFailure{.detail = "routed program refuses whole-step Record"});
    return launch.Capture(stream, body);
  }
  std::expected<void, StepFailure> Current(StepLaunch& launch,
                                           providers::NativeStream stream) const {
    if (lowered_.routing.empty() || !bound_ || *launch_ == nullptr || &launch != *launch_ ||
        bound_generation_ != *generation_ || launch.faulted())
      return std::unexpected(
          StepFailure{.detail = "routed program has no current healthy binding"});
    if (owner_)
      if (auto ready = owner_->Ready(); !ready)
        return std::unexpected(StepFailure{.detail = ready.error()});
    return SameStream((*launch_)->context(), stream);
  }
  std::expected<void, StepFailure> Access(const RoutedAccess& access) {
    if (!access.Names(this, access.epoch()) || access.segment() > groups())
      return std::unexpected(StepFailure{.detail = "foreign routed segment access"});
    if (access.segment() == 0 && access.epoch() > progress_.epoch()) {
      if (!progress_.Start(access.epoch(), groups()))
        return std::unexpected(StepFailure{.detail = "previous routed segment remains unretired"});
      pinned_ = {};
    }
    if (!progress_.Ready(access.epoch(), access.segment()))
      return std::unexpected(StepFailure{.detail = "routed segment is out of order or unprepared"});
    return {};
  }
  std::expected<void, StepFailure> RunSegment(const RoutedAccess& access, StepLaunch& launch,
                                              providers::NativeStream stream) override {
    if (auto current = Current(launch, stream); !current) return current;
    if (auto valid = Access(access); !valid) return valid;
    const auto segment = access.segment();
    if (!recording_ && !progress_.Submit(access.epoch(), segment))
      return std::unexpected(StepFailure{.detail = "routed segment was already submitted"});
    if (segment != 0) {
      const auto& previous = lowered_.routing[segment - 1];
      if (pinned_.size() != previous.layout.host_bytes)
        return std::unexpected(StepFailure{.detail = "continuation lost its prepared staging"});
      for (const auto& copy : previous.uploads) {
        auto copied = providers::CopyAsync(stream, reinterpret_cast<void*>(copy.device),
                                           pinned_.data() + copy.host_offset, copy.bytes,
                                           providers::CopyKind::kHostToDevice);
        if (!copied.ok())
          return std::unexpected(
              StepFailure{.error = StepError::kUnknown, .detail = copied.text()});
      }
    }
    const auto begin = segment == 0 ? 0 : lowered_.routing[segment - 1].cut;
    const auto end =
        segment == groups() ? lowered_.plan.steps.size() : lowered_.routing[segment].cut;
    auto& context = (*launch_)->context();
    if (auto ran = bound_->RunRange(context, begin, end); !ran)
      return std::unexpected(Failure(ran.error()));
    if (segment < groups())
      if (auto route = RunMoeRoute(context, lowered_.routing[segment].preparation); !route)
        return std::unexpected(Failure(route.error()));
    return {};
  }
  std::expected<void, StepFailure> CompleteSegment(const RoutedAccess& access) override {
    if (!access.Names(this, progress_.epoch()) || recording_ ||
        !progress_.Complete(access.epoch(), access.segment()))
      return std::unexpected(StepFailure{.detail = "routed completion is stale or out of order"});
    pinned_ = {};
    return {};
  }
  std::expected<void, StepFailure> Prepare(const RoutedAccess& access,
                                           std::span<std::byte> pinned) override {
    pinned_ = {};
    const auto group = access.segment();
    if (group >= groups() || !access.Names(this, progress_.epoch()) ||
        !progress_.CanPrepare(access.epoch(), group) || pinned.size() != layout(group).host_bytes ||
        reinterpret_cast<std::uintptr_t>(pinned.data()) % 256)
      return std::unexpected(
          StepFailure{.detail = "routed preparation lost its completed epoch or staging"});
    const auto& r = layout(group);
    const auto E = r.shape.experts, P = r.shape.used * r.shape.tokens;
    const auto ints = [&](std::uint64_t offset, std::uint64_t count) {
      return std::span(reinterpret_cast<std::int32_t*>(pinned.data() + offset), count);
    };
    const auto valid =
        ValidateGroupedRouteInto(r.shape,
                                 {ints(r.ids, P), ints(r.offsets, E + 1), ints(r.position, P),
                                  ints(r.token, P), ints(r.expert, P)},
                                 {ints(r.work_offsets, E + 1),
                                  ints(r.work_tokens, P),
                                  ints(r.work_slots, P),
                                  {reinterpret_cast<std::uint8_t*>(pinned.data() + r.seen), P}});
    if (!valid)
      return std::unexpected(
          StepFailure{.detail = "actual router IDs or route bijection are invalid"});
    for (const auto& [index, offset] : lowered_.routing[group].descriptors) {
      const auto& call = *lowered_.selected[index].call;
      if (auto staged =
              call.StageDescriptors(pinned.subspan(offset, call.route_descriptor_bytes()));
          !staged)
        return std::unexpected(Failure(staged.error()));
    }
    pinned_ = pinned;
    if (!progress_.Prepared(access.epoch(), group))
      return std::unexpected(StepFailure{.detail = "routed preparation order changed"});
    return {};
  }
  class SegmentRecording final : public RoutedRecording {
   public:
    SegmentRecording(GgmlStepProgram& owner, std::uint64_t generation, std::size_t segment,
                     CapturedGraph graph)
        : owner_(owner), generation_(generation), segment_(segment), graph_(std::move(graph)) {}
    const RoutedProgram* owner() const override { return &owner_; }
    std::size_t segment() const override { return segment_; }
    std::size_t nodes() const override { return graph_.nodes(); }
    double capture_seconds() const override { return graph_.capture_seconds(); }
    double instantiate_seconds() const override { return graph_.instantiate_seconds(); }
    std::expected<void, StepFailure> Replay(const RoutedAccess& access, StepLaunch& launch,
                                            providers::NativeStream stream) const override {
      if (auto current = owner_.Current(launch, stream); !current) return current;
      if (owner_.bound_generation_ != generation_ || access.segment() != segment_)
        return std::unexpected(
            StepFailure{.detail = "routed recording binding or ordinal changed"});
      if (auto valid = owner_.Access(access); !valid) return valid;
      if (!owner_.progress_.Submit(access.epoch(), segment_))
        return std::unexpected(StepFailure{.detail = "routed recording already submitted"});
      if (auto ran = (*owner_.launch_)->context().Launch(graph_); !ran)
        return std::unexpected(Failure(ran.error()));
      return {};
    }

   private:
    GgmlStepProgram& owner_;
    std::uint64_t generation_;
    std::size_t segment_;
    CapturedGraph graph_;
  };
  std::expected<std::unique_ptr<RoutedRecording>, StepFailure> RecordSegment(
      const RoutedAccess& access, StepLaunch& launch, providers::NativeStream stream,
      const StepLaunch::Body& body) override {
    if (auto current = Current(launch, stream); !current) return std::unexpected(current.error());
    if (!capturable())
      return std::unexpected(StepFailure{.detail = "routed program is not capturable"});
    if (auto valid = Access(access); !valid) return std::unexpected(valid.error());
    if (recording_) return std::unexpected(StepFailure{.detail = "nested routed recording"});
    recording_ = true;
    auto captured =
        (*launch_)->context().Capture([&](LaunchContext&) -> std::expected<void, KernelFailure> {
          const auto result = body();
          if (!result)
            return std::unexpected(KernelFailure{
                .error = result.error().error == StepError::kUnknown ? KernelError::kUnknown
                                                                     : KernelError::kRejected,
                .detail = result.error().detail});
          return {};
        });
    recording_ = false;
    if (!captured) return std::unexpected(Failure(captured.error()));
    return std::make_unique<SegmentRecording>(*this, bound_generation_, access.segment(),
                                              std::move(*captured));
  }
  std::expected<void, StepFailure> ReplaySegment(const RoutedAccess& access, StepLaunch& launch,
                                                 providers::NativeStream stream,
                                                 const RoutedRecording& recorded) override {
    if (recorded.owner() != this || recorded.segment() != access.segment())
      return std::unexpected(StepFailure{.detail = "foreign routed recording or segment"});
    return recorded.Replay(access, launch, stream);
  }

 private:
  const LaunchContext* ComparisonContextFor(ComparisonCertificatePhase phase) const {
    if (lowered_.measuring || !generation_ || (owner_ && !owner_->Ready())) return nullptr;
    if (phase == ComparisonCertificatePhase::kDescriptor) {
      if (bound_ || (launch_ && *launch_ && (*launch_)->faulted())) return nullptr;
      if (launch_ && *launch_) return &(*launch_)->context();
      return measuring_;
    }
    if (!bound_ || !launch_ || *launch_ == nullptr || bound_generation_ != *generation_ ||
        (*launch_)->faulted())
      return nullptr;
    return &(*launch_)->context();
  }

  std::expected<void, StepFailure> RunComparisonSpan(const ComparisonRegionProjection& projection,
                                                     StepLaunch& launch,
                                                     providers::NativeStream stream,
                                                     bool prefix) const {
    if (!ValidateComparisonProjection(projection) || &launch != *launch_ || !bound_)
      return std::unexpected(
          StepFailure{.detail = "comparison projection or bound launch is stale"});
    if (owner_)
      if (auto ready = owner_->Ready(); !ready)
        return std::unexpected(StepFailure{.detail = ready.error()});
    if (generated_)
      return std::unexpected(
          StepFailure{.detail = "generated-input region execution is outside this domain"});
    if (auto same = SameStream((*launch_)->context(), stream); !same) return same;
    const auto span = AuthenticatedComparisonRegionSpan(
        projection, bound_generation_, lowered_.comparison_origins->projection_serial);
    if (!span)
      return std::unexpected(StepFailure{.detail = "comparison projection lost its call span"});
    const auto first = prefix ? std::size_t{0} : span->first;
    const auto end = prefix ? span->first : span->second;
    if (auto ran = bound_->RunComparisonRange((*launch_)->context(), first, end); !ran)
      return std::unexpected(Failure(ran.error()));
    return {};
  }

  RoutedProgress progress_;
  bool recording_ = false;
  std::span<std::byte> pinned_{};
  const GgmlStepLaunch* const* launch_;
  const std::uint64_t* generation_;
  std::uint64_t bound_generation_ = 0;
  const execution::Registry& registry_;
  const LaunchContext* measuring_ = nullptr;
  ComparisonPlanIdentity comparison_identity_{};
  const SelectedSource* selected_source_ = nullptr;  // retained by the backend owner
  std::optional<base::Sha256Digest> comparison_product_identity_;
  std::optional<base::Sha256Digest> comparison_selected_identity_;
  LoweredStep lowered_;
  std::uint64_t pool_ = 0, private_pool_ = 0;
  kernels::LaneConfiguration lanes_{};
  std::shared_ptr<const kernels::BackendOwner> owner_;
  GeneratedStoreInputs* generated_ = nullptr;
  kernels::ResourceKeep generated_keep_{};
  std::uint64_t nodes_ = 0;
  std::uint64_t launches_ = 0;
  std::uint64_t selected_host_ = 0;
  bool capturable_ = true;
  std::optional<BoundGraph> bound_;
};

std::expected<std::unique_ptr<Recorded>, StepFailure> GgmlStepLaunch::Capture(
    providers::NativeStream stream, const Body& body) {
  if (private_eager_ || retired())
    return std::unexpected(StepFailure{.detail = "private or retired launch refuses capture"});
  if ((active_ && *active_ != this) || (current_generation_ && *current_generation_ != generation_))
    return std::unexpected(StepFailure{.detail = "the declared launch was retired"});
  if (owner_) {
    if (auto ready = owner_->Ready(); !ready)
      return std::unexpected(StepFailure{.detail = ready.error()});
  }
  if (auto same = SameStream(*launch_, stream); !same) return std::unexpected(same.error());
  if (selected_ && selected_->faulted())
    return std::unexpected(StepFailure{.detail = "selected native source has faulted"});
  if (auto polled = PollGeneratedInputs(); !polled) return std::unexpected(polled.error());
  if (capturing_)
    return std::unexpected(StepFailure{.detail = "nested generated-input capture is unsupported"});
  capturing_ = true;
  captured_generated_ = nullptr;
  auto captured = launch_->Capture(
      [&](LaunchContext&) -> std::expected<void, KernelFailure> {
        auto recorded = body();
        if (!recorded) {
          return std::unexpected(KernelFailure{
              .error = recorded.error().error == StepError::kUnknown ? KernelError::kUnknown
                                                                     : KernelError::kRejected,
              .detail = recorded.error().detail});
        }
        return {};
      },
      capture_measurement_ == CaptureMeasurement::kGpuDuration);
  capturing_ = false;
  if (!captured) {
    for (auto* at = captured_generated_; at; at = at->next) at->owner()->MarkUnknown();
    GeneratedCaptureOwner::DestroyList(std::exchange(captured_generated_, nullptr));
    return std::unexpected(Failure(captured.error()));
  }
  GeneratedCaptureOwners owners(std::exchange(captured_generated_, nullptr));
  return std::make_unique<GgmlRecorded>(this, generation_, std::move(*captured), std::move(owners));
}

std::expected<void, StepFailure> GgmlStepLaunch::ValidateRecording(const Recorded& recorded) const {
  if (private_eager_ || retired())
    return std::unexpected(StepFailure{.detail = "private or retired launch refuses duration"});
  if (faulted())
    return std::unexpected(
        StepFailure{.error = StepError::kUnknown, .detail = "recording owner has faulted"});
  if ((active_ && *active_ != this) || (current_generation_ && *current_generation_ != generation_))
    return std::unexpected(StepFailure{.detail = "the declared launch was retired"});
  if (owner_) {
    if (auto ready = owner_->Ready(); !ready)
      return std::unexpected(StepFailure{.detail = ready.error()});
  }
  if (recorded.owner() != this ||
      static_cast<const GgmlRecorded&>(recorded).generation() != generation_)
    return std::unexpected(StepFailure{.detail = "another launch's recording"});
  return {};
}

std::expected<void, StepFailure> GgmlStepLaunch::Launch(providers::NativeStream stream,
                                                        const Recorded& recorded) {
  if (private_eager_ || retired())
    return std::unexpected(StepFailure{.detail = "private or retired launch refuses replay"});
  if ((active_ && *active_ != this) || (current_generation_ && *current_generation_ != generation_))
    return std::unexpected(StepFailure{.detail = "the declared launch was retired"});
  if (owner_) {
    if (auto ready = owner_->Ready(); !ready)
      return std::unexpected(StepFailure{.detail = ready.error()});
  }
  if (auto same = SameStream(*launch_, stream); !same) return same;
  if (selected_ && selected_->faulted())
    return std::unexpected(StepFailure{.detail = "selected native source has faulted"});
  if (recorded.owner() != this ||
      static_cast<const GgmlRecorded&>(recorded).generation() != generation_) {
    return std::unexpected(StepFailure{.detail = "another launch's recording"});
  }
  if (auto polled = PollGeneratedInputs(); !polled) return polled;
  auto* generated = static_cast<const GgmlRecorded&>(recorded).generated();
  for (auto* at = generated; at; at = at->next) {
    auto* owner = at->owner();
    if (!owner->populated())
      return std::unexpected(
          StepFailure{.detail = "recorded generated-input payload was never initialized"});
    if (auto registered = RegisterGenerated(owner, false, at->resources()); !registered)
      return registered;
    owner->MarkSubmitted();
  }
  if (auto launched = launch_->Launch(static_cast<const GgmlRecorded&>(recorded).graph());
      !launched) {
    for (auto* at = generated; at; at = at->next) at->owner()->Quarantine();
    return std::unexpected(Failure(launched.error()));
  }
  return {};
}

std::expected<void, StepFailure> GgmlStepLaunch::RegisterGenerated(
    GeneratedStoreInputs* owner, bool capturing, const kernels::ExecutionResources& resources) {
  if (owner == nullptr || execution_ == nullptr || !launch_->UsesStream(*execution_, stream_id_))
    return std::unexpected(StepFailure{
        .detail = "generated StoreCells preparation lacks its authenticated launch stream"});
  const auto submission = execution_->Submission(stream_id_);
  if (!submission || submission->handle != launch_->native_stream().handle)
    return std::unexpected(StepFailure{
        .error = StepError::kUnknown,
        .detail = "generated StoreCells preparation cannot authenticate the real stream"});
  if (registry_root_ != this && !resources)
    return std::unexpected(StepFailure{
        .detail = "private generated input lacks its actual retained execution resources"});
  owner->RetainResources(resources);
  if (registry_root_ != this) {
    if (registry_root_->execution_ != execution_ || registry_root_->stream_id_ != stream_id_ ||
        !registry_root_->launch_->UsesStream(*execution_, stream_id_))
      return std::unexpected(StepFailure{
          .error = StepError::kUnknown,
          .detail = "private generated input lost its authenticated parent registry stream"});
  }
  if (auto registered = registry_root_->RegisterOnThis(owner); !registered) return registered;
  auto* const capture_root = registry_root_;
  if (capturing || capture_root->capturing_) {
    for (auto* at = capture_root->captured_generated_; at; at = at->next)
      if (at->owner() == owner) return {};
    auto entry = GeneratedCaptureOwner::Create(*owner);
    if (!entry) return std::unexpected(entry.error());
    (*entry)->next = capture_root->captured_generated_;
    capture_root->captured_generated_ = *entry;
  }
  return {};
}

std::expected<void, StepFailure> GgmlStepLaunch::RegisterOnThis(GeneratedStoreInputs* owner) {
  if (registry_root_ != this) return registry_root_->RegisterOnThis(owner);
  if (owner == nullptr || execution_ == nullptr || !launch_->UsesStream(*execution_, stream_id_))
    return std::unexpected(StepFailure{
        .detail = "generated StoreCells preparation lacks its authenticated launch stream"});
  const auto submission = execution_->Submission(stream_id_);
  if (!submission || submission->handle != launch_->native_stream().handle)
    return std::unexpected(StepFailure{
        .error = StepError::kUnknown,
        .detail = "generated StoreCells preparation cannot authenticate the real stream"});
  if (!owner->registered_) {
    owner->RetainRegistry();
    owner->registry_next_ = generated_inputs_;
    generated_inputs_ = owner;
  }
  return {};
}

std::expected<void, StepFailure> GgmlStepLaunch::PollGeneratedInputs() {
  if (registry_root_ != this) return registry_root_->PollGeneratedInputs();
  if (capturing_ || generated_inputs_ == nullptr) return {};
  if (execution_ == nullptr || !launch_->UsesStream(*execution_, stream_id_))
    return std::unexpected(StepFailure{
        .error = StepError::kUnknown,
        .detail = "generated StoreCells registry lost its authenticated device execution"});
  const auto submission = execution_->Submission(stream_id_);
  if (!submission || submission->handle != launch_->native_stream().handle)
    return std::unexpected(StepFailure{
        .error = StepError::kUnknown,
        .detail = "generated StoreCells registry cannot authenticate its native stream"});
  StepFailure pending{};
  bool has_pending = false;
  auto** link = &generated_inputs_;
  while (*link != nullptr) {
    GeneratedStoreInputs* current = *link;
    auto polled = current->Poll(*execution_);
    if (!polled) {
      if (!has_pending) pending = polled.error();
      has_pending = true;
      link = &current->registry_next_;
      continue;
    }
    if (current->dirty() || current->has_fence()) {
      link = &current->registry_next_;
      continue;
    }
    *link = current->registry_next_;
    current->ReleaseRegistry();
  }
  if (has_pending) return std::unexpected(std::move(pending));
  return {};
}

std::expected<void, StepFailure> GgmlStepLaunch::DrainGeneratedInputs() {
  if (registry_root_ != this) return registry_root_->DrainGeneratedInputs();
  if (capturing_ || generated_inputs_ == nullptr) return {};
  if (execution_ == nullptr || !launch_->UsesStream(*execution_, stream_id_))
    return std::unexpected(StepFailure{
        .error = StepError::kUnknown,
        .detail = "generated StoreCells registry lost its authenticated device execution"});
  const auto submission = execution_->Submission(stream_id_);
  if (!submission || submission->handle != launch_->native_stream().handle)
    return std::unexpected(StepFailure{
        .error = StepError::kUnknown,
        .detail = "generated StoreCells registry cannot authenticate its native stream"});
  StepFailure pending{};
  bool has_pending = false;
  auto** link = &generated_inputs_;
  while (*link != nullptr) {
    GeneratedStoreInputs* current = *link;
    auto drained = current->Drain(*execution_, stream_id_);
    if (!drained) {
      if (!has_pending) pending = drained.error();
      has_pending = true;
      link = &current->registry_next_;
      continue;
    }
    if (current->dirty() || current->has_fence()) {
      link = &current->registry_next_;
      continue;
    }
    *link = current->registry_next_;
    current->ReleaseRegistry();
  }
  if (has_pending) return std::unexpected(std::move(pending));
  return {};
}

// Configuration has stable storage and no callback-address identity. The
// backend/resources still obey the existing rule that they outlive borrowers.
// Binding slots are owned by this private instance, never the global backend.
// The origin authenticates/retains the actual runner before borrowing anything.
class GgmlResources final : public kernels::ResourceInstance {
 public:
  GgmlResources(kernels::ResourceOrigin origin, const GgmlBackend& source,
                std::unique_ptr<LaunchContext> context, SelectedLaunch* selected)
      : ResourceInstance(std::move(origin)), source_(source) {
    launch_ = std::make_unique<GgmlStepLaunch>(
        std::move(context), nullptr, parent_owner(), &active_, &generation_, generation_, selected,
        true, &source_.execution_, source_.stream_, const_cast<GgmlStepLaunch*>(source_.launch_),
        source_.launch_->selected_workspace());
    active_ = launch_.get();
  }
  kernels::StepLaunch& launch() const override { return *launch_; }
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> Lower(
      const kernels::LowerInput& input) const override {
    return source_.LowerFor(input, &active_, &generation_, true);
  }

 private:
  const GgmlBackend& source_;
  const std::uint64_t generation_ = 1;
  const GgmlStepLaunch* active_ = nullptr;
  std::unique_ptr<GgmlStepLaunch> launch_;
};

struct GgmlBackend::Declaration final : kernels::BackendOwner {
  Declaration(GgmlBackend& backend_source, planner::OwnedDeviceFacts owned_facts,
              std::uint64_t workspace_bytes, std::function<bool(std::uint64_t)> reserve,
              std::function<void(std::uint64_t)> refund)
      : source(backend_source),
        facts(std::move(owned_facts)),
        workspace(workspace_bytes),
        charge(std::move(reserve)),
        release(std::move(refund)) {}
  ~Declaration() override = default;
  const kernels::StepBackend& backend() const override { return table; }
  bool Charge(std::uint64_t n) const override { return charge(n); }
  void Release(std::uint64_t n) const override { release(n); }
  std::expected<execution::BackendDeclaration, std::string> Ready() const override {
    if (!sealed || source.cublas_ == nullptr || source.measuring_->faulted() ||
        (source.launch_ && source.launch_->faulted()) ||
        (source.options_.selected &&
         (source.options_.selected->ExecutionIdentity() != selected ||
          source.options_.selected->persistent_bytes() != selected_persistent)))
      return std::unexpected("the declared backend is not prepared or has faulted");
    return *sealed;
  }
  std::expected<std::uint64_t, kernels::ResourceFailure> ResourceBytes(
      kernels::ResourceWorkspace region) const override {
    auto host = LaunchContext::PrivateHostBytes({region.base, base::Bytes(region.bytes)});
    if (!host)
      return std::unexpected(
          kernels::ResourceFailure{kernels::ResourceError::kBounds, host.error().detail});
    constexpr std::uint64_t kAllocationMetadataBytes = 64;
    std::uint64_t bytes = 0;
    if (__builtin_add_overflow(*host,
                               std::uint64_t{sizeof(GgmlResources) + sizeof(GgmlStepLaunch) +
                                             2 * kAllocationMetadataBytes},
                               &bytes))
      return std::unexpected(kernels::ResourceFailure{kernels::ResourceError::kBounds,
                                                      "private backend metadata size overflows"});
    return bytes;
  }
  std::expected<kernels::ResourceInstance*, kernels::ResourceFailure> MakeResources(
      kernels::ResourceOrigin origin, kernels::ResourceWorkspace region) const override {
    if (!source.launch_ || origin.parent_launch() != source.launch_)
      return std::unexpected(kernels::ResourceFailure{
          kernels::ResourceError::kOwnership,
          "private resources do not retain this producer's actual parent launch"});
    auto context = LaunchContext::CreatePrivate(source.launch_->context(),
                                                {region.base, base::Bytes(region.bytes)});
    if (!context)
      return std::unexpected(kernels::ResourceFailure{context.error().error == KernelError::kUnknown
                                                          ? kernels::ResourceError::kFault
                                                          : kernels::ResourceError::kProvider,
                                                      context.error().detail});
    return new GgmlResources(std::move(origin), source, std::move(*context),
                             source.launch_->selected());
  }
  GgmlBackend& source;
  planner::OwnedDeviceFacts facts;
  kernels::StepBackend table{};
  std::uint64_t workspace;
  std::function<bool(std::uint64_t)> charge;
  std::function<void(std::uint64_t)> release;
  std::optional<execution::BackendDeclaration> sealed{};
  std::optional<base::Sha256Digest> selected{};
  std::uint64_t selected_persistent = 0;
};

std::expected<void, std::string> GgmlBackend::FundConstruction(
    std::function<bool(std::uint64_t)> charge, std::function<void(std::uint64_t)> release) {
  if (construction_funder_ || construction_started_ || launch_ || !charge || !release)
    return std::unexpected("the backend needs one complete construction funder before use");
  constexpr std::uint64_t bytes = sizeof(ConstructionFunder) + 2 * 64;
  if (!charge(bytes)) return std::unexpected("backend construction funder metadata was not funded");
  construction_funder_ = std::shared_ptr<const ConstructionFunder>(
      new ConstructionFunder{std::move(charge), std::move(release)}, [](ConstructionFunder* owner) {
        auto refund = std::move(owner->release);
        delete owner;
        refund(bytes);
      });
  // Declarations may precede construction funding (the image path). Export
  // the same paid token without changing any numerical/device declaration.
  if (declaration_) declaration_->table.construction_funder = construction_funder_;
  return {};
}

std::expected<void, std::string> GgmlBackend::Declare(std::span<const execution::TunedShape> tuning,
                                                      std::uint64_t workspace,
                                                      std::function<bool(std::uint64_t)> charge,
                                                      std::function<void(std::uint64_t)> release) {
  if (declaration_ || launch_ || cublas_ ||
      (options_.selected && !options_.selected->ExecutionIdentity()) || !charge || !release ||
      workspace == 0 || workspace % 256 != 0)
    return std::unexpected("the backend cannot declare this configuration");
  // Count/copy admission includes both the owned view and the adapter vector.
  // The input vector itself belongs to the caller and is already funded.
  // Refuse excessive input metadata before the temporary vector allocation.
  std::uint64_t temporary = 0;
  if (__builtin_mul_overflow(static_cast<std::uint64_t>(tuning.size() + facts_.tuned.size()),
                             sizeof(execution::TunedShape), &temporary) ||
      __builtin_add_overflow(temporary, std::uint64_t{512}, &temporary))
    return std::unexpected("the tuning input metadata was not funded");
  for (const auto& name : facts_.preferred) {
    std::uint64_t bytes = sizeof(std::string) + 64;
    if (__builtin_add_overflow(bytes, static_cast<std::uint64_t>(name.size() + 1), &bytes) ||
        __builtin_add_overflow(temporary, bytes, &temporary))
      return std::unexpected("preferred source metadata overflows");
  }
  if (!charge(temporary)) return std::unexpected("the preferred/tuning metadata was not funded");
  // Guards precede the allocations they fund, so every refusal destroys the
  // copied facts before releasing capacity. The retained charge transfers to
  // the shared owner's deleter, which refunds only after complete destruction.
  struct Charge {
    const std::function<void(std::uint64_t)>& refund;
    std::uint64_t bytes;
    ~Charge() {
      if (bytes != 0) refund(bytes);
    }
  } temporary_charge{release, temporary};
  planner::DeviceFacts requested{.device_class = facts_.device_class,
                                 .arch = facts_.arch,
                                 .features = facts_.features,
                                 .device_model = facts_.device_model,
                                 .multiprocessors = facts_.multiprocessors,
                                 .preferred = facts_.preferred};
  requested.tuned = facts_.tuned;
  for (const auto& fact : tuning)
    if (!std::ranges::contains(requested.tuned, fact)) requested.tuned.push_back(fact);
  auto bound = planner::OwnedDeviceFacts::Bound(requested);
  if (!bound) return std::unexpected(bound.error());
  std::uint64_t bytes = 0;
  if (__builtin_add_overflow(*bound, sizeof(Declaration) + 512, &bytes) ||
      __builtin_add_overflow(bytes, temporary, &bytes) || !charge(bytes))
    return std::unexpected("the owned backend metadata was not funded");
  Charge retained_charge{release, bytes};
  auto owned = planner::OwnedDeviceFacts::Create(requested, *bound);
  if (!owned) return std::unexpected(owned.error());
  for (const auto& tune : owned->facts().tuned) {
    if (!registry_->Find(tune.implementation))
      return std::unexpected("the tuning names an implementation outside this backend");
  }
  auto declared = std::shared_ptr<Declaration>(
      new Declaration(*this, std::move(*owned), workspace, charge, release),
      [release, bytes](Declaration* value) {
        delete value;
        release(bytes);
      });
  retained_charge.bytes = 0;
  declared->selected = options_.selected ? options_.selected->ExecutionIdentity() : std::nullopt;
  declared->selected_persistent = options_.selected ? options_.selected->persistent_bytes() : 0;
  declared->table = PlainBackend();
  declared->table.device = declared->facts.facts();
  declaration_ = std::move(declared);
  return {};
}

std::expected<void, std::string> GgmlBackend::PrepareDeclaration() {
  if (!declaration_) return {};
  if (!cublas_ || cublas_->device() != device_ || cublas_->stream() != stream_ ||
      cublas_->workspace().size.value() != declaration_->workspace)
    return std::unexpected("the declared cuBLAS workspace/device/stream differs");
  auto actual = cublas_->ConfigurationIdentity();
  if (!actual) return std::unexpected(actual.error().detail);
  base::Sha256 recipe, build, selection, comparison_selection;
  recipe.Update("llmp-ggml-declared-recipe-v1");
  build.Update("llmp-ggml-qualified-build-v1");
  selection.Update("llmp-ggml-declared-selection-v2");
  comparison_selection.Update("llmp-ggml-comparison-selection-v1");
  selection.Update(
      std::format("lanes:{}:{};", options_.lanes.count, options_.lanes.scratch_bytes_per_lane));
  selection.Update(std::as_bytes(std::span(registry_->RankDigest())));
  comparison_selection.Update(
      std::format("lanes:{}:{};", options_.lanes.count, options_.lanes.scratch_bytes_per_lane));
  comparison_selection.Update(std::as_bytes(std::span(registry_->SelectionDigest())));
  for (std::size_t i = 0; i < registry_->size(); ++i) {
    recipe.Update(std::as_bytes(std::span(registry_->identity(i))));
  }
  const auto append_selection_facts = [&](base::Sha256& hash) {
    if (options_.selected) {
      const auto identity = options_.selected->ExecutionIdentity();
      if (!identity || identity != declaration_->selected ||
          options_.selected->persistent_bytes() != declaration_->selected_persistent)
        return false;
      hash.Update(std::as_bytes(std::span(*identity)));
      hash.Update(std::format("selected-persistent:{}", options_.selected->persistent_bytes()));
    }
    for (const auto& name : facts_.preferred) {
      hash.Update(std::format("preferred:{}:", name.size()));
      hash.Update(name);
    }
    const auto& effective = declaration_->facts.facts();
    for (const auto text :
         {std::string_view(effective.device_class), std::string_view(effective.device_model)}) {
      hash.Update(std::format("fact:{}:", text.size()));
      hash.Update(text);
    }
    hash.Update(std::format("SMs:{};", effective.multiprocessors));
    for (const auto& tune : effective.tuned) {
      hash.Update(std::format("tune:{}:", tune.implementation.size()));
      hash.Update(tune.implementation);
      hash.Update(std::format(":{}:{}:{}:{}:{}:{};", tune.rows, tune.k, tune.n,
                              tune.output_encoding ? static_cast<int>(*tune.output_encoding) : -1,
                              tune.weight_bits, tune.all_rows));
    }
    hash.Update(std::as_bytes(std::span(*actual)));
    hash.Update(std::format("{}:{}:{}:{}:{}:{}", facts_.arch, facts_.features,
                            options_.lower.device_masks, options_.lower.frozen_host_masks,
                            options_.lower.index_scratch_limit, declaration_->workspace));
    return true;
  };
  // The qualified registry declarations carry effective SDK/target/images,
  // source/module revisions and arithmetic build recipes. Server kept records
  // also retain the executable's existing build/stamp identity.
  for (std::size_t i = 0; i < registry_->size(); ++i) {
    const auto& implementation = registry_->at(i);
    for (const auto text :
         std::array<std::string_view, 2>{implementation.build, implementation.revision}) {
      build.Update(std::format("{}:", text.size()));
      build.Update(text);
    }
  }
  if (options_.selected) {
    const auto identity = options_.selected->ExecutionIdentity();
    if (!identity || identity != declaration_->selected ||
        options_.selected->persistent_bytes() != declaration_->selected_persistent)
      return std::unexpected("selected adapter identity changed after declaration");
    recipe.Update(std::as_bytes(std::span(*identity)));
  }
  if (!append_selection_facts(selection) || !append_selection_facts(comparison_selection))
    return std::unexpected("selected adapter identity changed after declaration");
  execution::BackendDeclaration value{.recipe = recipe.Finish(),
                                      .build = build.Finish(),
                                      .selection = selection.Finish(),
                                      .comparison_selection = comparison_selection.Finish()};
  if (declaration_->sealed && *declaration_->sealed != value)
    return std::unexpected("the backend configuration changed after it was sealed");
  declaration_->sealed = value;
  return {};
}

GgmlBackend::GgmlBackend(int device, providers::DeviceExecution& execution,
                         providers::StreamId stream, CublasHandle* cublas,
                         GgmlBackendOptions options)
    : device_(device), execution_(execution), stream_(stream), cublas_(cublas), options_(options) {}

GgmlBackend::~GgmlBackend() = default;

std::expected<std::unique_ptr<GgmlBackend>, std::string> GgmlBackend::Create(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    CublasHandle* cublas, GgmlBackendOptions options) {
  const auto lanes = options.lanes;
  if ((lanes.count == 0 && lanes.scratch_bytes_per_lane != 0) ||
      (lanes.count != 0 && (lanes.count != kMaxLanes || lanes.scratch_bytes_per_lane == 0 ||
                            lanes.scratch_bytes_per_lane % 256 != 0 ||
                            lanes.scratch_bytes_per_lane > UINT64_MAX / lanes.count)))
    return std::unexpected("the backend needs zero lanes or four funded aligned lane pools");
  std::unique_ptr<GgmlBackend> backend(new GgmlBackend(device, execution, stream, cublas, options));
  if (options.lower.selected_source && options.lower.selected_source != options.selected.get())
    return std::unexpected("the selected lowering and owned source differ");
  std::vector<std::string> preferred;
  if (options.selected)
    for (const auto& implementation : options.selected->Implementations())
      preferred.push_back(implementation.name);
  auto decoders = decode::DecodeResidentSelectedSource();
  if (!decoders) return std::unexpected(decoders.error());
  auto exl3 = exl3::DeviceSelectedSource(device);
  if (!exl3) return std::unexpected(exl3.error());
  std::vector<std::shared_ptr<const SelectedSource>> sources;
  if (options.selected) sources.push_back(options.selected);
  sources.push_back(*decoders);
  sources.push_back(*exl3);
  options.selected = decode::CompositeSelectedSource(std::move(sources));
  options.lower.selected_source = options.selected.get();
  backend->options_ = options;
  auto implementations = Implementations();
  if (options.selected) {
    if (options.lower.selected_source && options.lower.selected_source != options.selected.get())
      return std::unexpected("the selected lowering and owned source differ");
    backend->options_.lower.selected_source = options.selected.get();
    auto extra = options.selected->Implementations();
    for (auto& implementation : extra) implementations.push_back(std::move(implementation));
  } else if (options.lower.selected_source) {
    return std::unexpected("a backend must own its selected source");
  }
  auto registry = execution::Registry::Create(std::move(implementations), MeasuredRanks());
  if (!registry) return std::unexpected(registry.error().detail);
  backend->registry_ = std::make_unique<execution::Registry>(std::move(*registry));
  auto measuring =
      LaunchContext::Create(device, execution, stream, {.base = 0, .size = base::Bytes(0)}, cublas);
  if (!measuring) return std::unexpected(measuring.error().detail);
  backend->measuring_ = std::move(*measuring);
  const auto facts = providers::QueryDeviceFacts(device);
  if (!facts) return std::unexpected(std::format("the device's facts: {}", facts.error().text()));
  // The provider's architecture is GGML's compute capability (1210 on the
  // GB10); the planner's is CUDA's major · 10 + minor.
  const std::uint32_t arch = facts->architecture / 10;
  std::uint32_t features = 0;
  if (facts->bf16_mma) features |= execution::kBf16Mma;
  if (facts->int8_mma) features |= execution::kInt8Mma;
  if (facts->fp8_mma) features |= execution::kFp8Mma;
  if (facts->fp4_mma) features |= execution::kFp4Mma;
  if (facts->block_scaled_mma) features |= execution::kBlockScaledMma;
  backend->facts_ = {.device_class = std::format("sm_{}", arch),
                     .arch = arch,
                     .features = features,
                     .device_model = facts->name.data(),
                     .multiprocessors = static_cast<std::uint32_t>(facts->multiprocessors)};
  if (options.selected) backend->facts_.tuned = options.selected->Tuning(arch);
  backend->facts_.preferred = std::move(preferred);
  return backend;
}

std::expected<std::unique_ptr<GgmlBackend>, std::string> GgmlBackend::CreateFromLibrary(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    kernels::StreamLibrary* library, GgmlBackendOptions options) {
  return Create(device, execution, stream, static_cast<CublasHandle*>(library), std::move(options));
}

kernels::StepBackend GgmlBackend::Backend() {
  construction_started_ = true;
  if (!declaration_) return PlainBackend();
  // The owner supplies the table by const reference. In particular, returning
  // its token does not allocate another uncharged tuning/name vector.
  return {.owner = declaration_};
}

kernels::StepBackend GgmlBackend::PlainBackend() {
  // A plan names a group's member but the lowering may resolve it to
  // another (PlanGraph's), so a member of a group with a cuBLAS member
  // stays on the stream too.
  return {.registry = registry_.get(),
          .device = facts_,
          .lanes = options_.lanes,
          .on_stream =
              [this](std::string_view implementation) {
                // Selected native calls share the backend's stream and serial
                // scratch. Independent GGML calls retain declared lane support.
                if (options_.selected && options_.selected->Owns(implementation)) return true;
                const execution::Capability* c = CapabilityOf(implementation);
                return UsesCublas(implementation) ||
                       (c != nullptr && c->group == CapabilityOf("ggml.mul_mat.cublas")->group);
              },
          .written = &WrittenBytes,
          .within = &WithinRoot,
          .lower = [this](const kernels::LowerInput& input) { return Lower(input); },
          .construction_funder = construction_funder_};
}

std::expected<void, std::string> GgmlBackend::LendCublas(CublasHandle* cublas) {
  construction_started_ = true;
  if (launch_ != nullptr) return std::unexpected("the backend's launch was made already");
  CublasHandle* previous = cublas_;
  if (auto lent = measuring_->LendCublas(cublas); !lent)
    return std::unexpected(lent.error().detail);
  cublas_ = cublas;
  if (auto ready = PrepareDeclaration(); !ready) {
    cublas_ = previous;
    const auto restored = measuring_->LendCublas(previous);
    base::Check(restored.has_value(), "an idle descriptor context restores its library loan");
    return ready;
  }
  return {};
}

void GgmlBackend::LaunchReleased() {
  launch_ = nullptr;
  // The caller has fenced and drops its handle next. Do not leave a required
  // declaration apparently ready through a borrowed pointer after teardown.
  const auto released = measuring_->LendCublas(nullptr);
  base::Check(released.has_value(), "an idle descriptor context releases its library loan");
  cublas_ = nullptr;
}

std::expected<std::unique_ptr<kernels::StepLaunch>, std::string> GgmlBackend::Launch(
    LaunchContext::Workspace pool, SelectedWorkspace persistent) {
  construction_started_ = true;
  if (launch_ != nullptr) return std::unexpected("the backend's launch was made already");
  if (launch_generation_ == std::numeric_limits<std::uint64_t>::max())
    return std::unexpected("the backend launch generation is exhausted");
  std::unique_ptr<SelectedLaunch> selected;
  if (options_.selected) {
    const auto bytes = persistent_bytes();
    if (bytes % 256 != 0 || (bytes && (!persistent.base || persistent.base % 256 != 0)) ||
        bytes > persistent.bytes || persistent.base + persistent.bytes < persistent.base ||
        pool.base + pool.size.value() < pool.base ||
        (bytes && persistent.base < pool.base + pool.size.value() &&
         pool.base < persistent.base + bytes))
      return std::unexpected("selected-source state needs distinct owned persistent backing");
    auto made = options_.selected->Launch(device_, execution_, stream_, {persistent.base, bytes});
    if (!made) return std::unexpected(made.error().detail);
    selected = std::move(*made);
    if (!selected || selected->owner() != options_.selected.get())
      return std::unexpected("selected launch belongs to another source");
  }
  auto context = LaunchContext::Create(device_, execution_, stream_, pool, cublas_);
  if (!context) return std::unexpected(context.error().detail);
  if (options_.lanes.count != 0) {
    auto configured = (*context)->ConfigureLanes(
        options_.lanes.count, base::Bytes(options_.lanes.scratch_bytes_per_lane));
    if (!configured) return std::unexpected(configured.error().detail);
  }
  std::optional<execution::BackendDeclaration> backend_declaration;
  if (declaration_) {
    auto ready = declaration_->Ready();
    if (!ready) return std::unexpected(ready.error());
    backend_declaration = *ready;
  }
  auto launch = std::make_unique<GgmlStepLaunch>(
      std::move(*context), std::move(selected), declaration_, &launch_, &launch_generation_,
      launch_generation_ + 1, nullptr, false, &execution_, stream_, nullptr,
      SelectedWorkspace{persistent.base, options_.selected ? persistent_bytes() : 0});
  ++launch_generation_;
  launch_ = launch.get();
  return launch;
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> GgmlBackend::Lower(
    const kernels::LowerInput& input) const {
  return LowerFor(input, &launch_, &launch_generation_);
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> GgmlBackend::LowerFor(
    const kernels::LowerInput& input, const GgmlStepLaunch* const* launch,
    const std::uint64_t* generation, bool /*private_resources*/) const {
  if (!construction_funder_)
    return std::unexpected("native GGML construction requires a paid backend funder");
  if (input.comparison) {
    if (auto checked = ValidateComparisonBindings(input, input.comparison->bindings); !checked)
      return std::unexpected(checked.error());
  }
  auto inventory = LowerConstructionOf(input);
  if (!inventory) return std::unexpected(inventory.error().detail);
  std::optional<execution::BackendDeclaration> backend_declaration;
  if (declaration_) {
    auto ready = declaration_->Ready();
    if (!ready) return std::unexpected(ready.error());
    backend_declaration = *ready;
  }
  // This guard precedes every native allocation (including its own shared
  // control). Declared before lowered/program so all refusal and abandoned
  // result paths destroy native storage before refunding the loan.
  if (!construction_funder_->charge(inventory->bytes))
    return std::unexpected("native GGML construction inventory was not funded");
  auto payment = std::make_shared<ConstructionPayment>();
  payment->funder = construction_funder_;
  payment->bytes = inventory->bytes;
  std::optional<components::ComparisonBoundaryDigest> observed_boundary;
  std::shared_ptr<const ir::StorageFunder> comparison_funder;
  if (input.comparison && input.comparison->expected_boundary) {
    auto adapted = ComparisonFunderAdapter::Create(construction_funder_);
    if (!adapted) return std::unexpected(adapted.error());
    comparison_funder = std::move(*adapted);
    auto observed = ObserveComparisonBoundary(input, input.comparison->bindings, comparison_funder);
    if (!observed) return std::unexpected(observed.error());
    observed_boundary = *observed;
  }
  auto options = options_.lower;
  options.routed_funder = declaration_;
  auto lowered = LowerStep(input, *registry_, DeviceChoicesOf(*measuring_), options);
  if (!lowered) return std::unexpected(lowered.error().detail);
  if (lowered->comparison_origins) {
    lowered->comparison_origins->source_boundary_verified = observed_boundary.has_value();
    lowered->comparison_origins->active_stage =
        observed_boundary && input.comparison ? input.comparison->active_stage : 0;
    if (declaration_ && backend_declaration)
      if (const auto device = ComparisonDeviceIdentity(declaration_->facts.facts()))
        lowered->comparison_origins->backend_device = *device;
  }
  lowered->generated_store = DescribeGeneratedStore(input, *lowered);
  lowered->generated_store_access_closed = lowered->generated_store.has_value();
  if (std::ranges::any_of(lowered->plan.steps,
                          [&](const PlanStep& step) { return step.lane > options_.lanes.count; }))
    return std::unexpected("the step requires lanes outside its backend's execution domain");
  if (options_.lanes.count == 0) lowered->plan.regions.clear();
  auto scratch = PlanScratch(*measuring_, lowered->plan);
  if (!scratch) return std::unexpected(scratch.error().detail);
  std::uint64_t pool = *scratch, private_pool = *scratch;
  std::uint64_t lane_scratch_bytes = 0;
  if (!lowered->plan.regions.empty()) {
    auto lane = PlanLaneScratch(*measuring_, lowered->plan);
    if (!lane) return std::unexpected(lane.error().detail);
    lane_scratch_bytes = *lane;
    if (*lane > options_.lanes.scratch_bytes_per_lane)
      return std::unexpected("the step exceeds its declared per-lane scratch capacity");
    // This work uses the immutable lane slices carved from the launch pool's
    // tail. The runner adds lane_pool_reserve() beside the measured ordinary
    // scratch; charging it here too would fund the same physical slices twice.
  }
  for (const auto& selected : lowered->selected) {
    pool = std::max(pool, selected.call->scratch_bytes());
    private_pool = std::max(private_pool, selected.call->scratch_bytes());
  }
  if (input.comparison) {
    for (auto& leaf : lowered->leaves)
      leaf.comparison_slab = DescribeComparisonExpertSlab(input, *lowered, leaf);
    for (std::size_t index = 0; index < input.comparison->bindings.size(); ++index) {
      const auto& range = input.comparison->bindings[index];
      std::size_t matched = 0;
      for (auto& leaf : lowered->leaves) {
        if (leaf.comparison_slab) {
          if (!std::ranges::contains(leaf.comparison_slab->bindings, index)) continue;
          ++matched;
          if (!CheckComparisonExpertSlab(leaf, input.comparison->bindings))
            return std::unexpected("comparison expert slab lost its complete part declaration");
          continue;
        }
        if (!ComparisonLeafMatches(input, range, leaf)) continue;
        ++matched;
        if (!leaf.tensor || range.bytes < ggml_nbytes(leaf.tensor))
          return std::unexpected(std::format(
              "comparison binding range does not cover every actual lowered leaf extent "
              "(root {}, part {}, declared {}, actual {})",
              range.original_root.v, range.part ? static_cast<int>(*range.part) : -1, range.bytes,
              leaf.tensor ? ggml_nbytes(leaf.tensor) : 0));
        if (!input.addresses.measuring()) {
          std::uint64_t expected = 0;
          leaf.comparison_staged = DescribeComparisonStagedInput(input, *lowered, leaf, index);
          if (leaf.comparison_staged) continue;
          if (UncoveredComparisonInputRoot(*lowered, leaf, range))
            return std::unexpected(
                "comparison staged input lacks its full physical-root ownership");
          if (!leaf.tensor->data || __builtin_add_overflow(range.base, range.offset, &expected) ||
              reinterpret_cast<std::uintptr_t>(leaf.tensor->data) != expected ||
              expected % range.alignment != 0)
            return std::unexpected(std::format(
                "comparison binding range does not own every actual lowered leaf "
                "(root {}, part {}, leaf_part {}, op {}, type {}, bytes {}, expected_offset {}, "
                "actual_in_parent {}, actual_offset {}, aligned {})",
                range.original_root.v, range.part ? static_cast<int>(*range.part) : -1,
                leaf.part ? static_cast<int>(*leaf.part) : -1, ggml_op_name(leaf.tensor->op),
                ggml_type_name(leaf.tensor->type), ggml_nbytes(leaf.tensor), range.offset,
                reinterpret_cast<std::uintptr_t>(leaf.tensor->data) >= range.base &&
                    reinterpret_cast<std::uintptr_t>(leaf.tensor->data) - range.base <
                        range.allocation_bytes,
                reinterpret_cast<std::uintptr_t>(leaf.tensor->data) >= range.base
                    ? reinterpret_cast<std::uintptr_t>(leaf.tensor->data) - range.base
                    : UINT64_MAX,
                expected % range.alignment == 0));
        }
      }
      if (matched == 0 &&
          !(input.comparison->scope == ComparisonProofScope::kWholeProgramNative && !range.part &&
            CheckComparisonOmittedInput(*lowered, range.original_root))) {
        const auto mapped = range.original_root.v < input.plan.tensors.size()
                                ? input.plan.tensors[range.original_root.v]
                                : ir::TensorId{};
        return std::unexpected(
            std::format("comparison binding does not resolve to an original lowered leaf "
                        "(root {}, part {}, planned {}, declared {}, leaves {})",
                        range.original_root.v, range.part ? static_cast<int>(*range.part) : -1,
                        mapped.v, range.bytes, lowered->leaves.size()));
      }
    }
    lowered->comparison_bindings.reserve(input.comparison->bindings.size());
    for (const auto& binding : input.comparison->bindings) {
      auto owned = binding;
      for (const auto& matched : lowered->leaves) {
        if (!matched.comparison_slab && ComparisonLeafMatches(input, binding, matched)) {
          owned.part = matched.part;
          if (matched.comparison_staged) {
            owned.base = reinterpret_cast<std::uintptr_t>(matched.tensor->data);
            owned.bytes = matched.comparison_staged->bytes;
            owned.allocation_bytes = matched.comparison_staged->bytes;
          }
          break;
        }
      }
      lowered->comparison_bindings.push_back(std::move(owned));
    }
    if (lowered->comparison_origins)
      lowered->comparison_origins->descriptor_bindings_verified =
          observed_boundary.has_value() && !input.addresses.measuring() && !lowered->measuring &&
          declaration_ && backend_declaration.has_value();
    const ComparisonQ8Dispatch q8_dispatch{.describe_main_pool = &DescribeQ8MainPool,
                                           .describe_quantize_q8 = &DescribeQuantizeQ8,
                                           .describe_mmvq = &DescribeMmvqDispatch,
                                           .plan_mmvq = &PlanMulMatVecQ,
                                           .plan_prepared_mmvq = &PlanMmvqPrepared};
    lowered->comparison = DescribeComparison(
        input, *lowered, *registry_, backend_declaration ? &*backend_declaration : nullptr,
        {.pool_bytes = pool,
         .private_pool_bytes = private_pool,
         .lanes = options_.lanes.count,
         .lane_scratch_bytes = lane_scratch_bytes},
        observed_boundary, measuring_.get(), &SelectMulMat, &q8_dispatch,
        &DescribeComparisonRemainderCall, &CheckWholeNativeCall, options_.selected.get());
    if (input.comparison->scope == ComparisonProofScope::kWholeProgramNative &&
        !input.addresses.measuring() &&
        !ComparisonRangesCurrent(*lowered, !lowered->plan.regions.empty()))
      return std::unexpected(
          "comparison physical input placement changed its owned alias contract");
    // Only LowerFor's immutable backend owner can authenticate the source
    // declaration. The descriptor helper itself never grants proof authority
    // from a caller-provided digest bundle.
    if (declaration_ && backend_declaration &&
        lowered->comparison->missing == kernels::ComparisonMissing::kNone) {
      if (const auto device = ComparisonDeviceIdentity(declaration_->facts.facts())) {
        lowered->comparison->backend_device = *device;
        lowered->comparison->state = kernels::ComparisonProofState::kDescriptorVerified;
      }
    }
    lowered->comparison_bindable =
        lowered->comparison->state == kernels::ComparisonProofState::kDescriptorVerified;
  }
  GeneratedStoreInputs* generated = nullptr;
  kernels::ResourceKeep generated_keep;
  if (lowered->generated_store) {
    const auto found = std::ranges::find_if(input.token->step.inputs, [&](const auto& candidate) {
      return candidate.tensor == lowered->generated_store->slot.tensor &&
             candidate.kind == components::InputKind::kStoreCells;
    });
    if (found == input.token->step.inputs.end())
      return std::unexpected("the generated StoreCells recipe lost its original input owner");
    auto owned =
        GeneratedStoreInputs::Create(*found, *lowered->generated_store, construction_funder_);
    if (!owned) return std::unexpected(owned.error());
    generated = owned->value;
    generated_keep = std::move(owned->keep);
  }
  auto program = std::make_unique<GgmlStepProgram>(
      launch, generation, *registry_, std::move(*lowered), pool, private_pool, options_.lanes,
      declaration_, generated, std::move(generated_keep), measuring_.get(),
      input.comparison
          ? ComparisonPlanIdentity{.source_graph_identity = input.io.graph.identity(),
                                   .planned_graph_identity = input.plan.graph.identity()}
          : ComparisonPlanIdentity{},
      options_.selected.get());
  if (!program->HoldConstruction(payment))
    return std::unexpected("native construction payment could not follow its program");
  return program;
}

std::expected<void, std::string> GgmlBackend::LendLibrary(kernels::StreamLibrary* library) {
  return LendCublas(static_cast<CublasHandle*>(library));
}

}  // namespace llmp::kernels::ggml

namespace llmp::kernels {

std::expected<std::uint64_t, std::string> StreamLibraryWorkspaceBytes(int device) {
  const auto facts = providers::QueryDeviceFacts(device);
  if (!facts) return std::unexpected(std::format("the device's facts: {}", facts.error().text()));
  return ggml::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture)).value();
}

std::expected<std::unique_ptr<StreamLibrary>, std::string> CreateStreamLibrary(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    providers::DeviceSlice workspace) {
  auto handle = ggml::CublasHandle::Create(device, execution, stream,
                                           {.base = workspace.base, .size = workspace.size});
  if (!handle) return std::unexpected(std::format("cuBLAS: {}", handle.error().detail));
  return std::unique_ptr<StreamLibrary>(std::move(*handle));
}

BackendCapabilities DeviceBackend::Capabilities() { return ggml::GgmlBackend::Capabilities(); }

namespace {

class GgmlDeviceBackend final : public DeviceBackend {
 public:
  GgmlDeviceBackend(int device, providers::DeviceExecution& execution, providers::StreamId stream,
                    std::uint64_t library_workspace_bytes,
                    std::unique_ptr<ggml::GgmlBackend> backend)
      : device_(device),
        execution_(execution),
        stream_(stream),
        library_workspace_bytes_(library_workspace_bytes),
        backend_(std::move(backend)) {}

  std::uint64_t library_workspace_bytes() const override { return library_workspace_bytes_; }
  std::uint64_t persistent_bytes() const override { return backend_->persistent_bytes(); }
  std::uint64_t lane_pool_reserve() const override { return backend_->lane_pool_reserve(); }
  const execution::Registry& registry() const override { return backend_->registry(); }
  const planner::DeviceFacts& device() const override { return backend_->device(); }
  StepBackend Backend() override { return backend_->Backend(); }

  std::vector<execution::TunedShape> ImageAttentionTuning(std::uint64_t head_dim) const override {
    return {{.implementation = ggml::kAttentionF32,
             .rows = 1,
             .k = head_dim,
             .n = head_dim,
             .output_encoding = format::Encoding::kF32,
             .all_rows = true}};
  }

  std::expected<void, std::string> FundConstruction(
      std::function<bool(std::uint64_t)> charge,
      std::function<void(std::uint64_t)> release) override {
    return backend_->FundConstruction(std::move(charge), std::move(release));
  }

  std::expected<void, std::string> Declare(std::span<const execution::TunedShape> tuning,
                                           std::function<bool(std::uint64_t)> charge,
                                           std::function<void(std::uint64_t)> release) override {
    return backend_->Declare(tuning, library_workspace_bytes_, std::move(charge),
                             std::move(release));
  }

  std::expected<void, std::string> BindWorkspaces(std::span<const std::uint64_t> bases) override {
    if (bases.size() != 1U + static_cast<std::size_t>(persistent_bytes() != 0)) {
      return std::unexpected("the kernel workspaces are not mapped");
    }
    selected_state_ = {persistent_bytes() ? bases[1] : 0, persistent_bytes()};
    auto cublas = ggml::CublasHandle::Create(
        device_, execution_, stream_,
        {.base = bases[0], .size = base::Bytes(library_workspace_bytes_)});
    if (!cublas) return std::unexpected(std::format("cuBLAS: {}", cublas.error().detail));
    cublas_ = std::move(*cublas);
    return backend_->LendCublas(cublas_.get());
  }

  void ReleaseWorkspaces() override {
    backend_->LaunchReleased();
    selected_state_ = {};
    cublas_.reset();
  }

  std::expected<std::unique_ptr<StepLaunch>, std::string> Launch(providers::DeviceSlice pool,
                                                                 bool allow_lanes = true) override {
    auto made = backend_->Launch({.base = pool.base, .size = pool.size}, selected_state_);
    if (!made) return std::unexpected(made.error());
    if (!allow_lanes && static_cast<ggml::GgmlStepLaunch*>(made->get())->context().lanes() != 0) {
      return std::unexpected("an image launch has concurrent lanes");
    }
    return made;
  }

 private:
  int device_;
  providers::DeviceExecution& execution_;
  providers::StreamId stream_;
  std::uint64_t library_workspace_bytes_;
  std::unique_ptr<ggml::GgmlBackend> backend_;
  std::unique_ptr<ggml::CublasHandle> cublas_;
  ggml::SelectedWorkspace selected_state_{};
};

}  // namespace

std::expected<std::unique_ptr<DeviceBackend>, std::string> DeviceBackend::Create(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    DeviceBackendOptions options) {
  const auto facts = providers::QueryDeviceFacts(device);
  if (!facts) return std::unexpected(std::format("the device's facts: {}", facts.error().text()));
  const std::uint64_t library_workspace_bytes =
      ggml::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture)).value();
  ggml::GgmlBackendOptions ggml_options{
      .lower = std::move(options.lower),
      .selected = std::move(options.selected),
      .lanes = options.lanes,
  };
  auto backend =
      ggml::GgmlBackend::Create(device, execution, stream,
                                static_cast<ggml::CublasHandle*>(nullptr), std::move(ggml_options));
  if (!backend) return std::unexpected(backend.error());
  return std::make_unique<GgmlDeviceBackend>(device, execution, stream, library_workspace_bytes,
                                             std::move(*backend));
}

}  // namespace llmp::kernels
