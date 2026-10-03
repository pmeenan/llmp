// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// GGML tensor descriptors over jitLLM memory (D-053;
// docs/backend-proof.md#ggml-llamacpp-b29c606e2). An arena holds the
// metadata of a bounded number of tensors in a buffer jitLLM owns, in a
// GGML context that never allocates tensor data (no_alloc). Operation
// nodes are built with GGML's own graph functions on context(), so their
// shapes, strides and operation parameters are upstream's, and Bind points
// each tensor at memory the caller owns and keeps valid. Tensors carry no
// GGML backend buffer: jitLLM's launchers never read one.
//
// GGML aborts when a context runs out of room, so callers check Reserve
// before building nodes. Every build profile has this; only CUDA builds
// launch anything on it.

#ifndef JITLLM_KERNELS_GGML_TENSORS_H_
#define JITLLM_KERNELS_GGML_TENSORS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"

namespace jitllm::kernels::ggml {

// Why a kernel-module call failed. kRejected: a precondition did not hold
// and nothing was queued. kUnknown: a launch reported a device error, so
// what was queued is undetermined (a fault, D-048): its operands stay
// protected until recovery proves the stream quiescent.
enum class KernelError : std::uint8_t { kRejected, kUnknown };

struct KernelFailure {
  KernelError error = KernelError::kRejected;
  std::string detail;
};

class TensorArena {
 public:
  // Room for `tensors` tensors, views and operation nodes.
  static std::expected<TensorArena, KernelFailure> Create(std::size_t tensors);
  // Exactly `bytes` of metadata (at least one tensor's): what an earlier
  // build of the same graph used (engine/planned.h SizedArena), standing
  // for the `stands_for` bytes that build was checked against (Reserve
  // answers as it did there; 0: for itself).
  static std::expected<TensorArena, KernelFailure> CreateBytes(std::size_t bytes,
                                                               std::size_t stands_for = 0);
  // Every tensor forgotten: the arena is empty again (a scratch arena's
  // reuse). Nothing built on it may be used afterwards.
  void Reset();
  // Ends a sized arena's standing for its estimate (CreateBytes), once the
  // identical build is done: later checks see its own room.
  void Seal() { stands_for_ = 0; }

  TensorArena(TensorArena&&) noexcept;
  TensorArena& operator=(TensorArena&& other) noexcept;
  TensorArena(const TensorArena&) = delete;
  TensorArena& operator=(const TensorArena&) = delete;
  ~TensorArena();

  // The context GGML's graph functions build on.
  ggml_context* context() const { return context_.get(); }
  // The host bytes it holds (its tensors' metadata), whatever it uses.
  std::size_t bytes() const { return capacity_; }
  // The bytes its tensors use of them (GGML's used memory).
  std::size_t used() const;
  // Refused unless `tensors` more fit.
  std::expected<void, KernelFailure> Reserve(std::size_t tensors) const;

  // Points tensor at `address`, which must hold ggml_nbytes(tensor) bytes
  // that the caller owns until every launch reading or writing them has
  // completed. Views made afterwards follow it; make them after binding.
  static void Bind(ggml_tensor* tensor, std::uint64_t address);

 private:
  struct Free {
    void operator()(ggml_context* context) const;
  };
  TensorArena(std::vector<std::byte> buffer, ggml_context* context, std::size_t capacity);

  std::vector<std::byte> buffer_;  // the metadata, before the context that uses it
  std::unique_ptr<ggml_context, Free> context_;
  std::size_t capacity_ = 0;
  std::size_t stands_for_ = 0;  // CreateBytes
};

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_TENSORS_H_
