// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Borrowed operands for the pinned ds4 prefill products. Every byte range is
// device-resident, mapped and charged by the caller until proved completion.
// These descriptors own no memory, runtime, stream, handle or allocation.
// CPU builds validate descriptors; CUDA builds alone provide the launchers.
// No ordinary model graph selects these experimental operations implicitly.

#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_PRODUCT_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_PRODUCT_H_

#include <cstdint>
#include <expected>

#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

class LaunchContext;

struct Ds4ProductRead {
  const void* data = nullptr;
  std::uint64_t bytes = 0;
};
struct Ds4ProductWrite {
  void* data = nullptr;
  std::uint64_t bytes = 0;
};
struct Ds4ProductMatrix {
  Ds4ProductRead storage;
  std::uint32_t rows = 0;
  std::uint32_t columns = 0;
  std::uint64_t row_stride = 0;  // bytes, including any physical padding
};
struct Ds4ProductOutput {
  Ds4ProductWrite storage;
  std::uint32_t rows = 0;
  std::uint32_t columns = 0;
  std::uint64_t row_stride = 0;
};

// Complete original input stages. Tokens are caller-validated I32 device
// rows; the original defensive token0 fallback is retained. Embedding rows
// are packed F16 and HC output is packed F32 [T,4*width].
struct Ds4Embedding {
  Ds4ProductRead tokens;
  Ds4ProductMatrix weights;
  Ds4ProductOutput output;
  std::uint32_t hyper_connections = 4;
};
std::expected<void, KernelFailure> CheckDs4Embedding(const Ds4Embedding& d);
std::expected<void, KernelFailure> RunDs4Embedding(LaunchContext& launch, const Ds4Embedding& d);

struct Ds4F16Conversion {
  Ds4ProductMatrix input;
  Ds4ProductOutput output;
};
std::expected<void, KernelFailure> CheckDs4F16Conversion(const Ds4F16Conversion& d);
std::expected<void, KernelFailure> RunDs4F16Conversion(LaunchContext& launch,
                                                       const Ds4F16Conversion& d);

// Original fused wide Q-rank/KV weighted RMS. Q8_1 emission belongs to the
// separate small captured decode tier and is explicitly absent here.
// Exact input/output alias is permitted independently for Q and KV.
struct Ds4QkvNorm {
  Ds4ProductMatrix query;
  Ds4ProductRead query_weight;
  Ds4ProductOutput query_output;
  Ds4ProductMatrix kv;
  Ds4ProductRead kv_weight;
  Ds4ProductOutput kv_output;
  float epsilon = 1.0e-6F;
};
std::expected<void, KernelFailure> CheckDs4QkvNorm(const Ds4QkvNorm& d);
std::expected<void, KernelFailure> RunDs4QkvNorm(LaunchContext& launch, const Ds4QkvNorm& d);

// F16 weights [M,K], preconverted F16 input [T,K], F32 output [T,M].
// Original wide dispatch is T>8. This calls the original GemmEx geometry,
// F32 computation/output and default algorithm using jitLLM's lent handle.
// The handle must have its original 32 MiB charged workspace and stream.
struct Ds4F16Product {
  Ds4ProductMatrix weights;
  Ds4ProductMatrix input;
  Ds4ProductOutput output;
};
std::expected<void, KernelFailure> CheckDs4F16Product(const Ds4F16Product& d);
std::expected<void, KernelFailure> RunDs4F16Product(LaunchContext& launch, const Ds4F16Product& d);

// Original small T1..8 path consumes F32 activation without F16 rounding.
// Exact split-K tier: M<2048,K>=1024 selects min(ceil(2048/M),K/512)
// segments, reduced per row then combined in ascending segment order.
// The native pool holds one row's partials, reused in stream order; no
// global ds4 scratch and no serial/ordered experimental override.
struct Ds4F16Vector {
  Ds4ProductMatrix weights;  // packed F16 [M,K]
  Ds4ProductMatrix input;    // packed F32 [T,K]
  Ds4ProductOutput output;   // packed F32 [T,M]
};
std::expected<void, KernelFailure> CheckDs4F16Vector(const Ds4F16Vector& d);
std::expected<std::uint64_t, KernelFailure> PlanDs4F16Vector(const Ds4F16Vector& d);
std::expected<void, KernelFailure> RunDs4F16Vector(LaunchContext& launch, const Ds4F16Vector& d);

// A producer writes exact block_q8_1_mmq D4 bytes: [K/128][T], each
// block 144 bytes (four F32 scales, 128 signed codes). K/512 is at most
// 65535 for CUDA grid.y. Capacity includes 256 whole zeroable tail blocks.
// Identity is explicit rather than looked
// up in ds4's process-global sidecar registry. The caller vouches for the
// exact source pointer/generation and producer arithmetic; checks refuse a
// stale shape/identity. A producer cannot be carried across a rebind.
struct Ds4D4Sidecar {
  Ds4ProductWrite storage;
  const void* source = nullptr;
  std::uint64_t generation = 0;
  std::uint32_t rows = 0;
  std::uint32_t columns = 0;
};
std::expected<std::uint64_t, KernelFailure> Ds4D4Bytes(std::uint32_t rows, std::uint32_t columns);
std::expected<void, KernelFailure> CheckDs4D4(const Ds4ProductMatrix& input,
                                              std::uint64_t generation, const Ds4D4Sidecar& d);
std::expected<void, KernelFailure> RunDs4D4(LaunchContext& launch, const Ds4ProductMatrix& input,
                                            std::uint64_t generation, const Ds4D4Sidecar& d);

// Raw GGUF Q8_0 rows retain their 34-byte blocks. The optional aligned
// kind-5 representation has separately borrowed F16 scale and signed-code
// planes; both use their original packed physical row strides. Root's
// checked sidepack binding owns conversion and raw/aligned weight identity.
struct Ds4Q8Weights {
  Ds4ProductRead raw{};
  std::uint64_t raw_row_stride = 0;
  Ds4ProductRead scales{};
  std::uint64_t scale_row_stride = 0;
  Ds4ProductRead codes{};
  std::uint64_t code_row_stride = 0;
  std::uint32_t rows = 0;     // M
  std::uint32_t columns = 0;  // K
};
enum class Ds4Q8Path : std::uint8_t { kMmq, kDenseD2r };
struct Ds4Q8Product {
  Ds4Q8Weights weights;
  Ds4ProductMatrix input;   // packed F32 [T,K]
  Ds4ProductOutput output;  // packed F32 [T,M]
  Ds4D4Sidecar quantized;
  std::uint64_t generation = 0;
  Ds4Q8Path path = Ds4Q8Path::kMmq;
  // False: produce D4 from input in this call. True: authenticate and use
  // already-produced bytes; either path zeroes the guarded tail in order.
  bool prepared = false;
};
std::expected<void, KernelFailure> CheckDs4Q8Product(const Ds4Q8Product& d);
std::expected<std::uint64_t, KernelFailure> PlanDs4Q8Product(const LaunchContext& launch,
                                                             const Ds4Q8Product& d);
std::expected<void, KernelFailure> RunDs4Q8Product(LaunchContext& launch, const Ds4Q8Product& d);

// The original small Q8 path has a distinct producer: canonical Q8_1
// [T,padded_K/32], 36-byte blocks with F16 d/sum and 32 signed codes.
// K is padded to 512; T is 1..8. This is never interchangeable with D4.
struct Ds4Q81Sidecar {
  Ds4ProductWrite storage;
  const void* source = nullptr;
  std::uint64_t generation = 0;
  std::uint32_t rows = 0;
  std::uint32_t columns = 0;
};
std::expected<std::uint64_t, KernelFailure> Ds4Q81Bytes(std::uint32_t rows, std::uint32_t columns);
std::expected<void, KernelFailure> CheckDs4Q81(const Ds4ProductMatrix& input,
                                               std::uint64_t generation, const Ds4Q81Sidecar& d);
std::expected<void, KernelFailure> RunDs4Q81(LaunchContext& launch, const Ds4ProductMatrix& input,
                                             std::uint64_t generation, const Ds4Q81Sidecar& d);

// Original default dispatch prefers aligned kind-5 planes for K%1024==0;
// the raw path is its explicit fallback. Both compute every requested row
// of the full head. There is no top2/argmax shortcut or head-cadence policy.
// Raw MMVQ reads complete row tiles before guarded stores: raw capacity
// includes ceil(M/rpb)*rpb rows, rpb=4 for T1,K<1024, 1 for other T1,
// and 2 for T2..8. The extra readable rows are caller-owned physical padding.
enum class Ds4Q8VectorPath : std::uint8_t { kAligned, kRaw };
struct Ds4Q8Vector {
  Ds4Q8Weights weights;
  Ds4ProductMatrix input;   // packed F32 [T,K]
  Ds4ProductOutput output;  // packed F32 [T,M]
  Ds4Q81Sidecar quantized;
  std::uint64_t generation = 0;
  Ds4Q8VectorPath path = Ds4Q8VectorPath::kAligned;
  bool prepared = false;  // authenticate the original Q8_1 producer
};
std::expected<void, KernelFailure> CheckDs4Q8Vector(const Ds4Q8Vector& d);
std::expected<void, KernelFailure> RunDs4Q8Vector(LaunchContext& launch, const Ds4Q8Vector& d);

// Original tail coordinates and YaRN expression order. Positions are
// optional packed I32, one per logical token. Their contents are the
// trusted producer's contract (negative compressor positions are legal).
// Scalar positions must not wrap. This value participates in the caller's
// capture identity; dynamic positions instead use the borrowed array.
struct Ds4ProductRope {
  Ds4ProductRead positions{};
  std::uint32_t first = 0;
  std::uint32_t step = 1;
  std::uint32_t original_context = 0;
  std::uint32_t rotary = 64;
  float base = 10000.0F;
  float scale = 1.0F;
  float extension = 0.0F;
  float attention = 1.0F;
  float beta_fast = 32.0F;
  float beta_slow = 1.0F;
  bool inverse = false;
};
struct Ds4HeadRope {
  Ds4ProductOutput input;  // in-place packed F32 [T,heads*head_width]
  std::uint32_t heads = 0;
  std::uint32_t head_width = 0;
  Ds4ProductRope rope;
  bool normalize = false;
  float epsilon = 1.0e-6F;
};
std::expected<void, KernelFailure> CheckDs4HeadRope(const Ds4HeadRope& d);
std::expected<void, KernelFailure> RunDs4HeadRope(LaunchContext& launch, const Ds4HeadRope& d);

// Own out-a is the original G8,K4096,rank1024,head512,rotary64 HMMA core.
// It keeps the unrotated F32 heads intact and emits interleaved F32 low.
// Low's 32-byte aligned physical capacity includes whole 16-row WMMA stores; the rope
// table has whole 128-row CTA padding for dummy-row reads. Logical rows
// are unchanged. All padding/preparation is explicit and charged.
struct Ds4OutA {
  Ds4Q8Weights weights;        // aligned [8192,4096], raw is not consumed
  Ds4ProductMatrix heads;      // packed F32 [T,32768]
  Ds4ProductOutput low;        // packed F32 [T,8192], padded physical capacity
  Ds4ProductWrite rope_table;  // float2[padded_T128,32]
  Ds4ProductRope rope;         // inverse=true, step=1
  Ds4D4Sidecar quantized{};    // optional dual emit; absent is all-zero descriptor
  std::uint64_t generation = 0;
};
std::expected<void, KernelFailure> CheckDs4OutA(const Ds4OutA& d);
std::expected<void, KernelFailure> RunDs4OutA(LaunchContext& launch, const Ds4OutA& d);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_PRODUCT_H_
