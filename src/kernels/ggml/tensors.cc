// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/tensors.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "base/check.h"
#include "ggml.h"

namespace llmp::kernels::ggml {

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
  auto buffer = std::make_unique_for_overwrite<std::byte[]>(size);
  std::vector<ggml_tensor*> visited(slots, nullptr);
  // GGML requires its memory pool aligned to GGML_MEM_ALIGN (16).
  base::Check(reinterpret_cast<std::uintptr_t>(buffer.get()) % 16 == 0,
              "operator new returns 16-byte aligned memory");
  ggml_context* context =
      ggml_init(ggml_init_params{.mem_size = size, .mem_buffer = buffer.get(), .no_alloc = true});
  TensorArena arena(std::move(buffer), context, size, std::move(visited), graph_capacity);
  arena.stands_for_ = stands_for;
  return arena;
}

TensorArena::TensorArena(std::unique_ptr<std::byte[]> buffer, ggml_context* context,
                         std::size_t capacity, std::vector<ggml_tensor*> visited,
                         std::size_t graph_capacity)
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

std::expected<ggml_type, KernelFailure> GgmlTypeOf(std::string_view name) {
  // GGML names its types in lower case ("q8_0"); the artifact's index in
  // upper case ("Q8_0"), as GGUF tools print them.
  const auto same = [](std::string_view a, std::string_view b) {
    return a.size() == b.size() && std::ranges::equal(a, b, [](char x, char y) {
             const auto lower = [](char c) {
               return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
             };
             return lower(x) == lower(y);
           });
  };
  for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
    const auto type = static_cast<ggml_type>(t);
    const char* type_name = ggml_type_name(type);
    if (ggml_blck_size(type) > 0 && ggml_type_size(type) > 0 && type_name != nullptr &&
        same(name, type_name)) {
      return type;
    }
  }
  return std::unexpected(KernelFailure{.error = KernelError::kRejected,
                                       .detail = std::format("{} is not a GGML type", name)});
}

}  // namespace llmp::kernels::ggml
