// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The routed experts' grouped GEMM (moe_cutlass.h): CUTLASS 4.7.1's SM120
// block-scaled tensor-op grouped GEMM (the kernel CUTLASS's example 79d
// builds, with a BF16 output and no source operand), for sm_121a. The
// problem shapes and operand addresses live on the device: a setup kernel
// writes them from the routing's offsets before each GEMM, so nothing waits
// on the host. The tile (128 x 128 x 256, ping-pong schedule) was chosen by
// a quick A/B at Qwen3.8's shapes (docs/experiments/qwen38-native/README.md).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "kernels/ggml/moe_cutlass.h"

namespace llmp::kernels::ggml::moe {

#if defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace {

using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using Tile = cute::Shape<cute::_128, cute::_128, cute::_256>;
using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
constexpr int kAlignAB = 32;  // FP4 elements: 16 bytes
constexpr int kAlignD = 8;    // BF16 elements: 16 bytes

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD*, kAlignD,
    ElementD, LayoutD*, kAlignD, cutlass::epilogue::collective::EpilogueScheduleAuto,
    cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA*, kAlignAB,
    ElementB, LayoutB*, kAlignAB, float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

using StrideA = typename Kernel::InternalStrideA;
using StrideB = typename Kernel::InternalStrideB;
using StrideD = typename Kernel::InternalStrideD;
using LayoutSFA = typename Kernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Kernel::CollectiveMainloop::InternalLayoutSFB;
using ScaleConfig = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = typename Kernel::CollectiveMainloop::ElementSF;
using Shape3 = typename ProblemShape::UnderlyingProblemShape;

// The per-group argument arrays, each 256-byte aligned, in scratch.
struct Arrays {
  Shape3* shapes;
  const typename Gemm::ElementA** a;
  const typename Gemm::ElementB** b;
  const ElementSF** sfa;
  const ElementSF** sfb;
  ElementD** d;
  StrideA* stride_a;
  StrideB* stride_b;
  StrideD* stride_d;
  LayoutSFA* layout_sfa;
  LayoutSFB* layout_sfb;
  std::size_t bytes;
};

constexpr std::size_t Round(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// The arrays at `base` (0 to size them).
Arrays Carve(std::uintptr_t base, int groups) {
  Arrays arrays{};
  std::size_t used = 0;
  const auto take = [&](auto*& into, std::size_t each) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): device scratch addresses
    into = reinterpret_cast<std::remove_reference_t<decltype(into)>>(base + used);
    used += Round(each * static_cast<std::size_t>(groups));
  };
  take(arrays.shapes, sizeof(Shape3));
  take(arrays.a, sizeof(void*));
  take(arrays.b, sizeof(void*));
  take(arrays.sfa, sizeof(void*));
  take(arrays.sfb, sizeof(void*));
  take(arrays.d, sizeof(void*));
  take(arrays.stride_a, sizeof(StrideA));
  take(arrays.stride_b, sizeof(StrideB));
  take(arrays.stride_d, sizeof(StrideD));
  take(arrays.layout_sfa, sizeof(LayoutSFA));
  take(arrays.layout_sfb, sizeof(LayoutSFB));
  arrays.bytes = used;
  return arrays;
}

// One thread a group: its shape, operand addresses, strides and scale
// layouts.
__global__ void SetupGroups(Arrays arrays, GroupedGemm gemm) {
  const int e = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (e >= gemm.groups) {
    return;
  }
  const int m = gemm.offsets[e + 1] - gemm.offsets[e];
  const int n = gemm.n;
  const int k = gemm.k;
  arrays.shapes[e] = Shape3{m, n, k};
  const auto* a = static_cast<const std::uint8_t*>(gemm.a);
  const auto* sfa = static_cast<const std::uint8_t*>(gemm.a_scales);
  const auto* w = static_cast<const std::uint8_t*>(gemm.weights) +
                  (static_cast<std::uint64_t>(e) * gemm.expert_stride);
  arrays.a[e] = reinterpret_cast<const typename Gemm::ElementA*>(
      a + (static_cast<std::uint64_t>(gemm.offsets[e]) * static_cast<std::uint64_t>(k / 2)));
  arrays.sfa[e] = reinterpret_cast<const ElementSF*>(
      sfa + (static_cast<std::uint64_t>(gemm.sf_rows[e]) * static_cast<std::uint64_t>(k / 16)));
  arrays.b[e] = reinterpret_cast<const typename Gemm::ElementB*>(w + gemm.codes_offset);
  arrays.sfb[e] = reinterpret_cast<const ElementSF*>(w + gemm.scales_offset);
  arrays.d[e] = static_cast<ElementD*>(gemm.d) +
                (static_cast<std::uint64_t>(gemm.offsets[e]) * static_cast<std::uint64_t>(n));
  arrays.stride_a[e] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  arrays.stride_b[e] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  arrays.stride_d[e] = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  arrays.layout_sfa[e] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  arrays.layout_sfb[e] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
}

typename Gemm::Arguments ArgumentsOf(const Arrays& arrays, int groups, int sms) {
  cutlass::KernelHardwareInfo hw;
  hw.device_id = 0;
  hw.sm_count = sms;
  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion;
  fusion.alpha = 1.0f;
  fusion.beta = 0.0f;
  arguments =
      typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                               {groups, arrays.shapes, nullptr},
                               {arrays.a, arrays.stride_a, arrays.b, arrays.stride_b, arrays.sfa,
                                arrays.layout_sfa, arrays.sfb, arrays.layout_sfb},
                               {fusion, nullptr, arrays.stride_d, arrays.d, arrays.stride_d},
                               hw};
  return arguments;
}

}  // namespace

bool GroupedGemmAvailable() { return true; }

std::size_t GroupedGemmScratch(int groups, int sms) {
  const Arrays arrays = Carve(0, groups);
  const typename Gemm::Arguments arguments = ArgumentsOf(arrays, groups, sms);
  return arrays.bytes + Round(Gemm::get_workspace_size(arguments));
}

int RunGroupedGemm(const GroupedGemm& gemm, int sms, void* stream) {
  auto* cuda_stream = static_cast<cudaStream_t>(stream);
  const Arrays arrays = Carve(reinterpret_cast<std::uintptr_t>(gemm.scratch), gemm.groups);
  constexpr int kThreads = 128;
  SetupGroups<<<static_cast<unsigned>((gemm.groups + kThreads - 1) / kThreads), kThreads, 0,
                cuda_stream>>>(arrays, gemm);
  const typename Gemm::Arguments arguments = ArgumentsOf(arrays, gemm.groups, sms);
  Gemm op;
  cutlass::Status status = op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status) + 1;
  }
  void* workspace = static_cast<char*>(gemm.scratch) + arrays.bytes;
  status = op.initialize(arguments, workspace, cuda_stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status) + 1;
  }
  status = op.run(cuda_stream);
  return status == cutlass::Status::kSuccess ? 0 : static_cast<int>(status) + 1;
}

#else

bool GroupedGemmAvailable() { return false; }
std::size_t GroupedGemmScratch(int /*groups*/, int /*sms*/) { return 0; }
int RunGroupedGemm(const GroupedGemm& /*gemm*/, int /*sms*/, void* /*stream*/) { return -1; }

#endif  // defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

}  // namespace llmp::kernels::ggml::moe
