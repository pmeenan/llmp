// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/tensors.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "base/check.h"
#include "ggml.h"

namespace jitllm::kernels::ggml {

void TensorArena::Free::operator()(ggml_context* context) const { ggml_free(context); }

std::expected<TensorArena, KernelFailure> TensorArena::Create(std::size_t tensors) {
  const std::size_t overhead = ggml_tensor_overhead();
  if (tensors == 0 || tensors > std::numeric_limits<std::size_t>::max() / overhead) {
    return std::unexpected(
        KernelFailure{.error = KernelError::kRejected,
                      .detail = std::format("no arena holds {} tensors", tensors)});
  }
  return CreateBytes(tensors * overhead);
}

std::expected<TensorArena, KernelFailure> TensorArena::CreateBytes(std::size_t bytes,
                                                                   std::size_t stands_for) {
  const std::size_t size = std::max(bytes, ggml_tensor_overhead());
  const std::size_t graph_capacity = std::max(size, stands_for) / ggml_tensor_overhead();
  // At most half full, so insertion stays bounded even with collisions.
  std::size_t slots = 2;
  while (slots / 2 < graph_capacity) {
    if (slots > std::numeric_limits<std::size_t>::max() / 2)
      return std::unexpected(KernelFailure{.detail = "graph traversal capacity overflow"});
    slots *= 2;
  }
  if (slots > (std::numeric_limits<std::size_t>::max() - size) / sizeof(ggml_tensor*))
    return std::unexpected(KernelFailure{.detail = "arena and graph traversal bytes overflow"});
  std::vector<std::byte> buffer(size);
  std::vector<ggml_tensor*> visited(slots, nullptr);
  // GGML requires its memory pool aligned to GGML_MEM_ALIGN (16).
  base::Check(reinterpret_cast<std::uintptr_t>(buffer.data()) % 16 == 0,
              "operator new returns 16-byte aligned memory");
  ggml_context* context =
      ggml_init(ggml_init_params{.mem_size = size, .mem_buffer = buffer.data(), .no_alloc = true});
  TensorArena arena(std::move(buffer), context, size, std::move(visited), graph_capacity);
  arena.stands_for_ = stands_for;
  return arena;
}

TensorArena::TensorArena(std::vector<std::byte> buffer, ggml_context* context, std::size_t capacity,
                         std::vector<ggml_tensor*> visited, std::size_t graph_capacity)
    : buffer_(std::move(buffer)),
      context_(context),
      capacity_(capacity),
      visited_(std::move(visited)),
      graph_capacity_(graph_capacity) {}

TensorArena::TensorArena(TensorArena&& other) noexcept
    : buffer_(std::move(other.buffer_)),
      context_(std::move(other.context_)),
      capacity_(std::exchange(other.capacity_, 0)),
      stands_for_(std::exchange(other.stands_for_, 0)),
      visited_(std::move(other.visited_)),
      graph_capacity_(std::exchange(other.graph_capacity_, 0)) {
  other.visited_.clear();
}
TensorArena& TensorArena::operator=(TensorArena&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  // The context goes before the buffer it uses.
  context_ = std::move(other.context_);
  buffer_ = std::move(other.buffer_);
  capacity_ = std::exchange(other.capacity_, 0);
  stands_for_ = std::exchange(other.stands_for_, 0);
  visited_ = std::move(other.visited_);
  graph_capacity_ = std::exchange(other.graph_capacity_, 0);
  other.visited_.clear();
  return *this;
}
TensorArena::~TensorArena() = default;

void TensorArena::Reset() {
  std::ranges::fill(visited_, nullptr);
  if (context_ != nullptr) {
    ggml_reset(context_.get());
  }
}

std::size_t TensorArena::used() const {
  return context_ == nullptr ? 0 : ggml_used_mem(context_.get());
}

std::expected<void, KernelFailure> TensorArena::Reserve(std::size_t tensors) const {
  const std::size_t used = context_ == nullptr ? capacity_ : ggml_used_mem(context_.get());
  // An arena sized from an identical build stands for the estimate that
  // build reserved against: the same checks pass and fail as they did
  // there, and what it builds fits, as it did there.
  const std::size_t capacity = stands_for_ != 0 ? stands_for_ : capacity_;
  const std::size_t room = used > capacity ? 0 : (capacity - used) / ggml_tensor_overhead();
  if (tensors > room) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail = std::format("the arena has room for {} more tensors, not {}", room, tensors)});
  }
  return {};
}

void TensorArena::Bind(ggml_tensor* tensor, std::uint64_t address) {
  tensor->data = reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

}  // namespace jitllm::kernels::ggml
