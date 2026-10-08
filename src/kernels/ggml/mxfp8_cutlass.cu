// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The dense MXFP8 products (mxfp8_cutlass.h): CUTLASS 4.7.1's SM120
// block-scaled tensor-op GEMM (the kernel CUTLASS's example 79c builds, both
// operands MXFP8 E4M3, no source operand), for sm_121a, with an F32 and a
// BF16 output. The tile (128 x 128 x 128, ping-pong schedule) and the tile
// swizzle were chosen by a quick A/B at Qwen3.8's shapes
// (docs/experiments/qwen38-native/README.md).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "kernels/ggml/mxfp8_cutlass.h"

namespace llmp::kernels::ggml::mxfp8 {

#if defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace {

using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using Tile = cute::Shape<cute::_128, cute::_128, cute::_128>;
using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
constexpr int kAlignAB = 16;  // E4M3 elements: 16 bytes

template <typename ElementD>
struct Config {
  static constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD, kAlignD,
      ElementD, LayoutD, kAlignD, cutlass::epilogue::collective::EpilogueScheduleAuto,
      cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA, kAlignAB,
      ElementB, LayoutB, kAlignAB, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Mainloop,
                                                      Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// The rows past which the tiles are swizzled in groups along N, so that a
// wave's tiles share A and B rows in L2 (the A/B's wide products at 8,192
// rows).
constexpr int kSwizzleRows = 4096;
constexpr int kSwizzle = 8;

template <typename ElementD>
typename Config<ElementD>::Gemm::Arguments ArgumentsOf(const Gemm& gemm, int sms) {
  using G = typename Config<ElementD>::Gemm;
  using Kernel = typename G::GemmKernel;
  using ScaleConfig = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementSF = typename Kernel::CollectiveMainloop::ElementSF;
  const int m = gemm.m;
  const int n = gemm.n;
  const int k = gemm.k;
  typename G::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<const typename G::ElementA*>(gemm.a),
       cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1}),
       static_cast<const typename G::ElementB*>(gemm.b),
       cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1}),
       static_cast<const ElementSF*>(gemm.a_scales),
       ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1)),
       static_cast<const ElementSF*>(gemm.b_scales),
       ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1))},
      {{1.0f, 0.0f},
       nullptr,
       cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1}),
       static_cast<ElementD*>(gemm.d),
       cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1})}};
  arguments.hw_info.device_id = 0;
  arguments.hw_info.sm_count = sms;
  if (m > kSwizzleRows) {
    arguments.scheduler.max_swizzle_size = kSwizzle;
    arguments.scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;
  }
  return arguments;
}

template <typename ElementD>
std::size_t ScratchOf(const Gemm& gemm, int sms) {
  using G = typename Config<ElementD>::Gemm;
  return G::get_workspace_size(ArgumentsOf<ElementD>(gemm, sms));
}

template <typename ElementD>
int RunOf(const Gemm& gemm, int sms, cudaStream_t stream) {
  using G = typename Config<ElementD>::Gemm;
  const auto arguments = ArgumentsOf<ElementD>(gemm, sms);
  G op;
  cutlass::Status status = op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status) + 1;
  }
  status = op.initialize(arguments, gemm.scratch, stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status) + 1;
  }
  status = op.run(stream);
  return status == cutlass::Status::kSuccess ? 0 : static_cast<int>(status) + 1;
}

}  // namespace

bool Available() { return true; }

std::size_t Scratch(const Gemm& gemm, int sms) {
  const std::size_t bytes =
      gemm.bf16 ? ScratchOf<cutlass::bfloat16_t>(gemm, sms) : ScratchOf<float>(gemm, sms);
  return (bytes + 255) / 256 * 256;
}

int Run(const Gemm& gemm, int sms, void* stream) {
  auto* cuda_stream = static_cast<cudaStream_t>(stream);
  return gemm.bf16 ? RunOf<cutlass::bfloat16_t>(gemm, sms, cuda_stream)
                   : RunOf<float>(gemm, sms, cuda_stream);
}

#else

bool Available() { return false; }
std::size_t Scratch(const Gemm& /*gemm*/, int /*sms*/) { return 0; }
int Run(const Gemm& /*gemm*/, int /*sms*/, void* /*stream*/) { return -1; }

#endif  // defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

}  // namespace llmp::kernels::ggml::mxfp8
