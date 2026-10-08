// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What llmpalooza's EXL3 launchers check on the host before anything is queued
// (D-053; docs/backend-proof.md, "ExLlamaV3"), in every profile: the
// operands of each kernel the native linear launches, the forced launch
// plan, and the rules of launch_contract.h that a host can check. A CUDA
// build's launchers (launch.h) run exactly these checks, then launch; the
// checks read no device memory.
//
// Upstream's host wrappers (exl3_gemm.cu, exl3_gemv.cu, reconstruct.cu,
// hadamard.cu, add.cu) check some of this with TORCH_CHECK and assume the
// rest. These checks refuse, with nothing queued, everything a kernel
// would read or write out of bounds, every plan whose grid is not all
// co-resident (a cooperative kernel would trap or hang), operands aligned
// below what the kernels' vector accesses need, and overlaps the kernels do
// not tolerate. Sizes are in checked arithmetic, and bounded so that the
// kernels' own int arithmetic cannot overflow.
//
// Only the variants the M2 fixtures use exist: the mcg codebook, integer
// rates K = 4, 5, 6 and 8, full-length FP16 side vectors (suh, svh), and the
// GEMV at K = 4 only (docs/exl3-bringup.md). Every dimension of an EXL3
// linear is a multiple of 128: both sides carry a 128-point Hadamard.

#ifndef LLMP_KERNELS_EXL3_VALIDATE_H_
#define LLMP_KERNELS_EXL3_VALIDATE_H_

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace llmp::kernels::exl3 {

// Why a call failed. kRejected: a precondition did not hold and nothing was
// queued. kUnknown: a launch reported a device error, so what was queued is
// undetermined (a fault, D-048): the operands stay protected until
// recovery proves the stream quiescent.
enum class KernelError : std::uint8_t { kRejected, kUnknown };

struct KernelFailure {
  KernelError error = KernelError::kRejected;
  std::string detail;
};

// Upstream's constants at the pin (quant/exl3_kernel_map.cuh,
// exl3_gemm_inner.cuh, exl3_gemv_kernel.cuh, exl3_devctx.cuh), for the
// checks in every profile; a CUDA-profile test holds them to the headers
// (unit.Exl3KernelTablesTest).
inline constexpr int kShapes = 4;  // EXL3_GEMM_NUM_SHAPES
// Indexed by tile shape; index 0 is upstream's unused slot.
inline constexpr std::array<int, 5> kTileK{0, 16, 32, 32, 16};         // EXL3_GEMM_TILESIZE_K
inline constexpr std::array<int, 5> kTileN{0, 128, 128, 256, 512};     // EXL3_GEMM_TILESIZE_N
inline constexpr std::array<int, 5> kBlockDim{0, 256, 512, 512, 256};  // EXL3_GEMM_BLOCKDIM
inline constexpr int kGemmSharedMemory = 90 * 1024;                    // SMEM_MAX
inline constexpr int kGemvMaxRows = 8;                                 // EXL3_GEMV_MAX_M
// The device context's lock slots: MAX_TILES_C split-K locks, then
// 2 * MAX_BARRIERS barrier counters and sense words, then MOE_SCHED_INTS.
inline constexpr std::uint64_t kLockBytes = 4'202'760;
// The rows each 128-point Hadamard block covers, and upstream's r_scale
// (hadamard.cu, had_r_128 with scale 1.0: 1/sqrt(128) as a float literal).
inline constexpr int kHadamardBlock = 128;
inline constexpr float kHadamardScale = 0.088388347648F;
// The bias add's threads per block (add.cu, NUM_THREADS).
inline constexpr int kAddThreads = 1024;
// Upstream's reconstruction slices (exllamav3/modules/quant/exl3.py,
// MAX_RECONSTRUCT_SLICE_N): at most this many output columns at a time.
inline constexpr int kReconstructSliceColumns = 32768;

// Bounds that keep the kernels' int arithmetic in range: m * k and m * n
// (twice, for the two matrices of a multi-GEMM) and the trellis's element
// count stay below 2^31, and a reconstruction grid's k / 16 blocks below
// CUDA's 65,535.
inline constexpr int kMaxDimension = 1 << 19;
inline constexpr int kMaxRows = 1 << 16;

// Whether the rate has kernels in this build: K = 4, 5, 6 or 8.
bool RateCompiled(int bits);

enum class Output : std::uint8_t { kF16, kF32 };
int OutputBytes(Output output);

// One quantized linear's resident tensors, as device addresses: the
// trellis ([k/16][n/16][16 K] uint16), suh (k F16) and svh (n F16).
struct Weights {
  std::uint64_t trellis = 0;
  std::uint64_t suh = 0;
  std::uint64_t svh = 0;
  int k = 0;
  int n = 0;
  int bits = 0;
};
std::uint64_t TrellisBytes(const Weights& weights);
// The shape, rate and alignment of the tensors (trellis 16 bytes, side
// vectors 8), and the bounds above.
std::expected<void, KernelFailure> CheckWeights(const Weights& weights);

// A launch plan the caller fixes, as a decoded tuning record does: the
// tile shape and the grid (docs/backend-proof.md, "Autotuning").
//   exl3_gemm_kernel:  grid (blocks, 1, 1), kBlockDim[shape] threads.
struct GemmPlan {
  int shape = 0;
  int blocks = 0;
};
//   exl3_mgemm_kernel: grid (blocks, 1, concurrency).
struct MultiGemmPlan {
  int shape = 0;
  int blocks = 0;
  int concurrency = 0;
};
//   exl3_gemv_kernel:  grid (blocks, 1, 1), 512 threads for config 0
//   (narrow), 256 for config 1 (wide).
struct GemvPlan {
  int config = 0;
  int blocks = 0;
};
int GemvThreads(int config);
int GemvColumns(int config);  // output columns per block group: 32 or 64

// x (m × k F16) times one linear into y (m × n, F16 or F32), through the
// packed kernels. a_had is scratch for the transformed input, m × k F16,
// which the kernel writes first.
struct LinearOperands {
  Weights weights;
  std::uint64_t x = 0;
  std::uint64_t a_had = 0;
  std::uint64_t y = 0;
  Output output = Output::kF16;
  int m = 0;
};
std::uint64_t ScratchBytes(const LinearOperands& operands);  // a_had's

// `coresident` is how many blocks of the plan's kernel fit on the device at
// once (occupancy per SM times SMs); `locks` the lock area the launch uses.
std::expected<void, KernelFailure> CheckGemm(const LinearOperands& operands, const GemmPlan& plan,
                                             int coresident, std::uint64_t locks);
std::expected<void, KernelFailure> CheckGemv(const LinearOperands& operands, const GemvPlan& plan,
                                             int coresident, std::uint64_t locks);
// A bias (n F16) the packed path adds after the product lies apart from
// what the product writes besides y, which CheckBias covers: a_had and the
// lock area.
std::expected<void, KernelFailure> CheckBiasApart(const LinearOperands& operands,
                                                  std::uint64_t bias, std::uint64_t locks);

// x (m × k F16) times two linears of one input, the fused gate and up, into
// y (2 × m × n). The kernel takes the weights through device tables of two
// addresses each (trellis, suh, svh), which the caller builds from `first`
// and `second`, as upstream's MultiLinear does; a_had is 2 × m × k F16
// scratch. The tables are device memory, so no host check reads them: the
// kernel follows whatever addresses they hold, and `first` and `second`
// are checked in their place. A caller rebuilds them whenever either
// linear's tensors move, and never launches with stale ones
// (docs/backend-proof.md, BP-P5). `written` is the caller's record of what
// it last wrote into the three tables (MultiGemmTables of the linears it
// wrote them for); a launch whose record is not MultiGemmTables(first,
// second) is refused, so a caller that moves a linear and updates its
// operands but not its tables' record is caught before launch. Only the
// record is checked: a caller keeps it with the upload it describes.
struct MultiLinearOperands {
  Weights first;
  Weights second;
  std::uint64_t trellis_table = 0;
  std::uint64_t suh_table = 0;
  std::uint64_t svh_table = 0;
  std::array<std::uint64_t, 6> written{};
  std::uint64_t x = 0;
  std::uint64_t a_had = 0;
  std::uint64_t y = 0;
  Output output = Output::kF16;
  int m = 0;
};
std::uint64_t ScratchBytes(const MultiLinearOperands& operands);
// What the three tables hold for `first` and `second`, in the order the
// kernel reads them: the trellis table's two addresses (16 bytes), then
// the suh table's, then the svh table's.
std::array<std::uint64_t, 6> MultiGemmTables(const Weights& first, const Weights& second);
std::expected<void, KernelFailure> CheckMultiGemm(const MultiLinearOperands& operands,
                                                  const MultiGemmPlan& plan, int coresident,
                                                  std::uint64_t locks);

// Columns [column, column + columns) of the linear's weights, reconstructed
// into w (k × columns F16, row-major): in the rotated basis
// (reconstruct_kernel) or, fused, with both Hadamard transforms and the
// side vectors applied (reconstruct_had_kernel).
struct ReconstructOperands {
  Weights weights;
  std::uint64_t w = 0;
  int column = 0;
  int columns = 0;
  bool fused = false;
};
std::expected<void, KernelFailure> CheckReconstruct(const ReconstructOperands& operands);

// 128-point Hadamard transforms of the rows of x (rows × columns) into y,
// which is x itself or disjoint from it: F16 with the input scale
// (had_hf_r_128_kernel<true, false>), or F16 or F32 with the output scale
// (had_{hf,ff}_r_128_kernel<false, true>).
struct HadamardOperands {
  std::uint64_t x = 0;
  std::uint64_t y = 0;
  std::uint64_t scale = 0;  // columns F16
  int rows = 0;
  int columns = 0;
  Output type = Output::kF16;
  bool input_scale = true;  // else the output scale
};
std::expected<void, KernelFailure> CheckHadamard(const HadamardOperands& operands);

// y = x + bias in F16, bias broadcast over x's rows (add_kernel_hhh);
// y is x itself or disjoint from it.
struct BiasOperands {
  std::uint64_t x = 0;
  std::uint64_t bias = 0;
  std::uint64_t y = 0;
  int rows = 0;
  int columns = 0;
};
std::expected<void, KernelFailure> CheckBias(const BiasOperands& operands);

// A pinned algorithm (exl3-recon-pin.json): the nine CUBLASLT_ALGO_CONFIG
// attributes, in the table's order (algo_id, tile, splitk, reduction,
// swizzle, custom, stages, inner_shape, cluster_shape), and the GEMM they
// were pinned for, as the table records it: m, k, n (the slice's columns,
// the table's n and lda), ldc and the output (HSH F16, HSS F32). The
// approved rule pins each GEMM on its own; ReconGemm runs a pin for exactly
// its GEMM and refuses any other, an unpinned size included
// (docs/backend-proof.md, "EXL3 reconstruction-path linears").
struct LtAlgorithm {
  std::array<std::uint64_t, 9> config{};
  int m = 0;
  int k = 0;
  int n = 0;
  int ldc = 0;
  Output output = Output::kF16;
};

// The reconstruction GEMM's operands (recon_gemm.h).
struct ReconGemmOperands {
  std::uint64_t w = 0;  // k × n F16, row-major
  std::uint64_t x = 0;  // m × k F16, row-major
  std::uint64_t y = 0;  // m rows of n, ldc apart, F16 or F32
  int m = 0;
  int k = 0;
  int n = 0;
  int ldc = 0;
  Output output = Output::kF16;
};
std::expected<void, KernelFailure> CheckReconGemm(const ReconGemmOperands& operands);

// The paths of one linear (linear.h).
enum class LinearPath : std::uint8_t { kPacked, kReconstruct, kReconstructFused };

// Upstream's choice for `rows` rows (exl3.py): packed through 144 rows
// (AUTO_RECONSTRUCT_THRESHOLD), reconstructed above, fused from 1,024 rows.
LinearPath UpstreamPath(int rows);
// Upstream's slices of an n-column reconstruction (exl3.py,
// reconstruct_hgemm): one slice up to 32,768 columns, else slices of
// 32,768 and a remainder, in order.
std::vector<int> ReconstructSlices(int n);
// Its weight scratch: k × min(n, 32,768) F16.
std::uint64_t ReconstructScratchBytes(const Weights& weights);

// One linear through the reconstruction path (linear.h): x (m × k F16)
// into y (m × n), with xh (m × k F16, the transformed input; unused when
// fused) and w (ReconstructScratchBytes) as scratch, and an optional F16
// bias (0 for none).
struct ReconstructedOperands {
  Weights weights;
  std::uint64_t x = 0;
  std::uint64_t xh = 0;
  std::uint64_t w = 0;
  std::uint64_t y = 0;
  Output output = Output::kF16;
  int m = 0;
  std::uint64_t bias = 0;
};
// What the per-launch checks cannot see across launches: the scratch the
// path writes (xh, then w for each slice) overlaps nothing it later reads
// or writes (x, xh, y), and nothing it writes (xh, w, y) holds the weights
// or the bias.
std::expected<void, KernelFailure> CheckReconstructedScratch(const ReconstructedOperands& operands,
                                                             bool fused);

}  // namespace llmp::kernels::exl3

#endif  // LLMP_KERNELS_EXL3_VALIDATE_H_
