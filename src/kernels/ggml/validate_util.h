// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The building blocks of the GGML implementations' operand checks
// (validate.h, validate_ext.h), on the host and in every build profile.
// Internal to the module: callers use the checks, not these.

#ifndef LLMP_KERNELS_GGML_VALIDATE_UTIL_H_
#define LLMP_KERNELS_GGML_VALIDATE_UTIL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml::detail {

inline std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

inline constexpr std::uint64_t kInt32Max = std::numeric_limits<std::int32_t>::max();

inline bool IsF32(const ggml_tensor* tensor) {
  return tensor != nullptr && tensor->type == GGML_TYPE_F32;
}

inline bool Bound(const ggml_tensor* tensor) {
  return tensor != nullptr && tensor->data != nullptr;
}

// The kernels load whole elements, or pairs of them, from the data
// pointer; a misaligned load is a sticky fault that ends every CUDA call in
// the process.
inline bool Aligned(const ggml_tensor* tensor, std::uint64_t alignment) {
  return reinterpret_cast<std::uintptr_t>(tensor->data) % alignment == 0;
}

// The base and every stride are multiples of `alignment`: the kernels
// offset the base by any combination of strides before a paired load.
inline bool AlignedEverywhere(const ggml_tensor* tensor, std::uint64_t alignment) {
  if (!Aligned(tensor, alignment)) {
    return false;
  }
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    if (tensor->nb[i] % alignment != 0) {
      return false;
    }
  }
  return true;
}

// The bytes from a bound tensor's first element to one past its last, by
// checked arithmetic: nothing for an empty tensor, a stride GGML would
// treat as negative, or an extent or end address that overflows
// (ggml_nbytes wraps). A blocked (quantized) type counts its rows in whole
// blocks, as ggml_nbytes does.
inline std::optional<std::uint64_t> Extent(const ggml_tensor* tensor) {
  const auto block = static_cast<std::int64_t>(ggml_blck_size(tensor->type));
  if (block <= 0 || tensor->ne[0] % block != 0) {
    return std::nullopt;
  }
  std::uint64_t extent = 0;
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (tensor->ne[i] <= 0 || tensor->nb[i] > std::numeric_limits<std::int64_t>::max()) {
      return std::nullopt;
    }
  }
  if (block == 1) {
    extent = ggml_type_size(tensor->type);
    std::uint64_t step = 0;
    if (__builtin_mul_overflow(static_cast<std::uint64_t>(tensor->ne[0] - 1), tensor->nb[0],
                               &step) ||
        __builtin_add_overflow(extent, step, &extent)) {
      return std::nullopt;
    }
  } else if (__builtin_mul_overflow(static_cast<std::uint64_t>(tensor->ne[0] / block),
                                    ggml_type_size(tensor->type), &extent)) {
    return std::nullopt;
  }
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    std::uint64_t step = 0;
    if (__builtin_mul_overflow(static_cast<std::uint64_t>(tensor->ne[i] - 1), tensor->nb[i],
                               &step) ||
        __builtin_add_overflow(extent, step, &extent)) {
      return std::nullopt;
    }
  }
  std::uint64_t end = 0;
  if (__builtin_add_overflow(
          static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(tensor->data)), extent,
          &end)) {
    return std::nullopt;
  }
  return extent;
}

inline bool AllSane(std::initializer_list<const ggml_tensor*> tensors) {
  return std::ranges::all_of(tensors,
                             [](const ggml_tensor* tensor) { return Extent(tensor).has_value(); });
}

// Strides exactly those of a dense tensor of its shape: what the dense
// operations accept, since GGML's contiguity test skips extents of one and
// its broadcast launcher merges dimensions assuming packed strides. A
// blocked type's rows are whole blocks.
inline bool Packed(const ggml_tensor* tensor) {
  if (tensor->nb[0] != ggml_type_size(tensor->type)) {
    return false;
  }
  const auto block = static_cast<std::int64_t>(ggml_blck_size(tensor->type));
  if (block <= 0 || tensor->ne[0] % block != 0 ||
      tensor->nb[1] != ggml_row_size(tensor->type, tensor->ne[0])) {
    return false;
  }
  for (int i = 2; i < GGML_MAX_DIMS; ++i) {
    if (tensor->nb[i] != tensor->nb[i - 1] * static_cast<std::uint64_t>(tensor->ne[i - 1])) {
      return false;
    }
  }
  return true;
}

// Packed through its last dimension of more than one element: every stride
// an index can multiply is a dense tensor's, so code that merges
// dimensions assuming packed strides (the broadcast launcher,
// binbcast.cu:196-214) addresses it correctly. A view of one row of a wider
// matrix, [n, 1] at a longer row stride, is one; GGML deems it contiguous.
inline bool PackedThroughLastDim(const ggml_tensor* tensor) {
  if (tensor->nb[0] != ggml_type_size(tensor->type) || ggml_blck_size(tensor->type) != 1) {
    return false;
  }
  int last = 0;
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    if (tensor->ne[i] > 1) {
      last = i;
    }
  }
  for (int i = 1; i <= last; ++i) {
    if (tensor->nb[i] != tensor->nb[i - 1] * static_cast<std::uint64_t>(tensor->ne[i - 1])) {
      return false;
    }
  }
  return true;
}

// Whether a's bytes and b's overlap (both bound); an unmeasurable tensor
// counts as overlapping.
inline bool Overlap(const ggml_tensor* a, const ggml_tensor* b) {
  const auto a_extent = Extent(a);
  const auto b_extent = Extent(b);
  if (!a_extent || !b_extent) {
    return true;
  }
  const auto a_begin = reinterpret_cast<std::uintptr_t>(a->data);
  const auto b_begin = reinterpret_cast<std::uintptr_t>(b->data);
  return a_begin < b_begin + *b_extent && b_begin < a_begin + *a_extent;
}

// An output may share no byte with an input, except, where the operation
// works element by element or row by row, by being exactly that input:
// same address, shape and strides (in place). Upstream's allocator never
// places an output partly over an input.
inline bool Disjoint(const ggml_tensor* output, const ggml_tensor* input, bool in_place) {
  if (!Overlap(output, input)) {
    return true;
  }
  return in_place && output->data == input->data && ggml_are_same_shape(output, input) &&
         ggml_are_same_stride(output, input);
}

// Every view in the chain still points where its source does: a view made
// before its source was bound, or kept after the source was bound again,
// keeps the old address (tensors.h).
inline bool Current(const ggml_tensor* tensor) {
  for (; tensor != nullptr && tensor->view_src != nullptr; tensor = tensor->view_src) {
    const auto* expected = static_cast<const char*>(tensor->view_src->data);
    if (expected == nullptr ||
        static_cast<const char*>(tensor->data) != expected + tensor->view_offs) {
      return false;
    }
  }
  return true;
}

inline bool AllCurrent(std::initializer_list<const ggml_tensor*> tensors) {
  return std::ranges::all_of(tensors, Current);
}

// The tensor that owns a view's storage.
inline const ggml_tensor* Root(const ggml_tensor* tensor) {
  while (tensor->view_src != nullptr) {
    tensor = tensor->view_src;
  }
  return tensor;
}

// Every stride is a whole number of elements, as the launchers divide
// strides by the element size (the broadcast launcher asserts it).
inline bool ElementStrides(const ggml_tensor* tensor) {
  const std::uint64_t size = ggml_type_size(tensor->type);
  return std::ranges::all_of(tensor->nb, [size](std::size_t stride) { return stride % size == 0; });
}

// The elements from the first to the last one the tensor addresses: the
// kernels index within it with 32-bit arithmetic. Only for sane tensors.
inline std::uint64_t Span(const ggml_tensor* tensor) {
  return Extent(tensor).value_or(std::numeric_limits<std::uint64_t>::max()) /
         ggml_type_size(tensor->type);
}

// Upstream skips empty nodes; its launchers divide by extents and build
// fast divisors from them, so an empty operand would abort the process.
inline bool AnyEmpty(std::initializer_list<const ggml_tensor*> tensors) {
  return std::ranges::any_of(tensors, ggml_is_empty);
}

// Every extent and element stride fits 32 bits, as the broadcast launcher
// asserts.
inline bool Fits32(const ggml_tensor* tensor) {
  const std::uint64_t size = ggml_type_size(tensor->type);
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (std::cmp_greater(tensor->ne[i], std::numeric_limits<std::uint32_t>::max()) ||
        tensor->nb[i] / size > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

// The row kernels launch one block per row, channel and sample.
inline bool FitsRowGrid(const ggml_tensor* tensor) {
  return tensor->ne[1] <= std::numeric_limits<std::int32_t>::max() && tensor->ne[2] <= 65535 &&
         tensor->ne[3] <= 65535;
}

inline float ParamF32(const ggml_tensor* tensor, int index) {
  float value = 0.0f;
  std::memcpy(&value, &tensor->op_params[index], sizeof(value));
  return value;
}

// The product of positive extents, or nothing if it overflows.
inline std::optional<std::uint64_t> Product(std::initializer_list<std::int64_t> extents) {
  std::uint64_t product = 1;
  for (const std::int64_t extent : extents) {
    if (extent <= 0 ||
        __builtin_mul_overflow(product, static_cast<std::uint64_t>(extent), &product)) {
      return std::nullopt;
    }
  }
  return product;
}

}  // namespace llmp::kernels::ggml::detail

#endif  // LLMP_KERNELS_GGML_VALIDATE_UTIL_H_
