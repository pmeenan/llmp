// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's post-down route weighting followed
// by its six selected contributions' ascending sum. No graph selection,
// shared expert addition, router, allocation or stream ownership.
#ifndef LLMP_KERNELS_GGML_DSV4_WEIGHTED_REDUCE_H_
#define LLMP_KERNELS_GGML_DSV4_WEIGHTED_REDUCE_H_

#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {

class LaunchContext;

inline constexpr std::int64_t kDsv4WeightedReduceWidth = 4096;
inline constexpr std::int64_t kDsv4WeightedReduceSlots = 6;
inline constexpr std::int64_t kDsv4WeightedReduceMaxRows = 4096;
inline constexpr const char* kDsv4WeightedReduceName = "llmp.dsv4.weighted_reduce";

// Metadata-only selection: exact canonical F32 inputs, without requiring
// bound addresses. Unsupported consumers keep their ordinary MUL/ADD graph.
bool Dsv4WeightedReduceFits(const ggml_tensor* down, const ggml_tensor* weights);

// A normal graph node whose two direct dependencies keep down and weights
// live with its separately placed F32 output. The graph owns all metadata.
ggml_tensor* Dsv4OrderedReduce(ggml_context* context, ggml_tensor* down, ggml_tensor* weights);
std::expected<void, KernelFailure> CheckDsv4OrderedReduce(const ggml_tensor* node);
std::expected<void, KernelFailure> RunDsv4OrderedReduce(LaunchContext& launch, ggml_tensor* node);

// A bound F32 tensor and the caller-owned, mapped bytes available starting
// at tensor->data. Tensor metadata is borrowed only during Check/Run.
// Supplied bounds do not prove residency: the caller accounts and protects
// every device operand through completion, including graph replay.
struct Dsv4WeightedReduceOperand {
  const ggml_tensor* tensor = nullptr;
  base::Bytes bytes;
};

// down [4096,6,rows,1], weights [1,6,rows,1], values [4096,rows,1,1],
// all F32 with exact canonical contiguous byte strides. rows is 1..4096.
// Every slot contributes, in original slot order. Read-only operands may
// overlap; the output's accessed range must not overlap either input.
// No intermediate output, scale, shared contribution or scratch is used.
struct Dsv4WeightedReduce {
  Dsv4WeightedReduceOperand down;
  Dsv4WeightedReduceOperand weights;
  Dsv4WeightedReduceOperand values;
};

std::expected<void, KernelFailure> CheckDsv4WeightedReduce(const Dsv4WeightedReduce& desc);

// CUDA only. Refusal queues nothing. A successful call queues on llmpalooza's
// existing stream; it is not completion proof. kUnknown faults the context:
// all operand owners remain protected until recovery proves quiescence.
// Graph capture stores addresses and fixed rows, reads current operand
// contents on replay, and owns no metadata or device storage itself.
// Builds with GGML's device math flags, uses an initial mul_rn and five
// ascending add_rn(mul_rn) steps, with no FMA contraction. Exactness versus
// this build's ordinary GGML MUL/ADD still requires operand qualification.
std::expected<void, KernelFailure> RunDsv4WeightedReduce(LaunchContext& launch,
                                                         const Dsv4WeightedReduce& desc);

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_DSV4_WEIGHTED_REDUCE_H_
