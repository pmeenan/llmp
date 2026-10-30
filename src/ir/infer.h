// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What a node makes of its sources (op.h's shape and format rules), shared
// by the builder, which makes outputs from it, and the verifier, which
// checks recorded outputs against it. A fused kind's outputs are its
// decomposition's (decompose.h).

#ifndef LLMP_IR_INFER_H_
#define LLMP_IR_INFER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>

#include "format/encoding.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/verify.h"

namespace llmp::ir {

namespace detail {
class InferenceCache;
}

struct InferenceMemoStats {
  std::uint64_t byte_limit = 0, bytes = 0, peak_bytes = 0;
  std::uint64_t entries = 0, hits = 0, misses = 0, evictions = 0;
};

// A synchronous compiler's memoization over all its builders, rewrites and
// verifications. The caller prepays byte_limit separately from its other
// construction inventories. Storage is lazy, byte bounded and LRU; zero
// disables memoization. No cache survives the scope. A nested compiler owns
// its own budget and restores the outer scope, while fused decompositions
// share this scope. Concurrent scopes share no storage.
class InferenceScope {
 public:
  explicit InferenceScope(std::uint64_t byte_limit);
  ~InferenceScope();
  InferenceScope(const InferenceScope&) = delete;
  InferenceScope& operator=(const InferenceScope&) = delete;
  InferenceScope(InferenceScope&&) = delete;
  InferenceScope& operator=(InferenceScope&&) = delete;
  const InferenceMemoStats& stats() const { return stats_; }
  // Destroy every memo allocation, leaving inference uncached until this scope
  // ends. An optional compiler loan may be returned only after this call.
  void Release();
  // Includes the control allocation, one hash bucket and one result node.
  static std::uint64_t MinimumBytes();
  // A working-set allowance sized from the source's real component inventory,
  // not the IR's maximum graph size. It is a byte budget, not an entry cap;
  // pressure evicts individually even when a component has several queries.
  static std::optional<std::uint64_t> WorkingSetBytes(std::uint64_t components);

 private:
  InferenceMemoStats stats_;
  std::unique_ptr<detail::InferenceCache> cache_;
  detail::InferenceCache* previous_ = nullptr;
};

struct Inferred {
  std::array<TensorDesc, kMaxOutputs> out{};
  std::uint8_t nout = 0;
  // Whether output i is a computed float that may be recorded in another
  // dense float encoding (F32, F16 or BF16; the value rounded once).
  std::array<bool, kMaxOutputs> float_result{};
};

// The outputs of a node of `key` with `params` over `sources` (ids into
// `tensors`, an absent optional source an invalid id). `requested` gives
// the outputs an op cannot derive (a leaf's, a view's, a cast's target, a
// fill's shape); for any other op it may be empty. Each output's role,
// format, strides and view fields are set; its segment, binding and name
// are left to the caller (a leaf's binding comes from its request). The
// error's node is 0: the caller knows which node it is.
std::expected<Inferred, VerifyError> Infer(OpKey key,
                                           std::span<const std::byte, kMaxParamBytes> params,
                                           std::span<const TensorId> sources,
                                           std::span<const TensorDesc> tensors,
                                           std::span<const TensorDesc> requested);

// The dense float encodings a computed result may be recorded in.
bool ResultEncoding(format::Encoding encoding);

}  // namespace llmp::ir

#endif  // LLMP_IR_INFER_H_
