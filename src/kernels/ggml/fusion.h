// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Upstream's fusion gates for the fused implementations in ops.h (D-053;
// docs/backend-proof.md#tier-e-exact): where GGML's graph-compute loop,
// with fusion on (FP16-F), would fuse, so that a native plan fuses exactly
// there. With fusion off (FP16-U, GGML_CUDA_DISABLE_FUSION) upstream
// consults none of them.
//
// Each gate takes a graph's nodes in GGML's execution order (GraphOrder
// gives the order ggml_build_forward_expand records) over tensors bound to
// jitLLM memory, and a node index, and reproduces ggml_cuda_try_fuse's
// conditions for its pattern at llama.cpp b29c606e2: the operations and
// edges, uses confined to the fused nodes (counted as GGML's graph counts
// them, over every node listed), no output flags on elided nodes, and for
// some patterns that the fused output overlaps no input other than an
// elided intermediate. Where upstream measures a tensor's bytes through
// its CUDA buffer type, the gates measure them as that buffer type does,
// from the tensor alone: tensors over jitLLM memory carry no buffer, so
// none counts as a constant weight either.
//
// The gates take a node list rather than GGML's graph object because
// ggml_new_graph_custom sizes that object with pointer arithmetic on a null
// pointer (ggml.c:7424), which the sanitizer builds report as undefined
// behaviour (RE-021).
//
// Upstream tries its patterns in a fixed order at each node
// (ggml-cuda.cu:3432-4180); a planner asks these gates in that order, and
// must show that no pattern jitLLM does not implement applies. For a
// matrix product the gate+GLU pattern comes before the bias pattern. The
// two MMVF gates leave out the one condition that needs the device, its
// MMVF selection (ops.h MulMatVecFusible). A gate that matches says what
// upstream would do; the fused implementation's own checks (validate.h)
// may still refuse the nodes, and then no plan reproduces upstream's.
// Every profile builds these.

#ifndef JITLLM_KERNELS_GGML_FUSION_H_
#define JITLLM_KERNELS_GGML_FUSION_H_

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "ggml.h"

namespace jitllm::kernels::ggml {

// A graph's nodes in execution order: every tensor computed, each after its
// inputs. Leaves (GGML_OP_NONE) are not nodes.
using GraphNodes = std::span<ggml_tensor* const>;

// The nodes ggml_build_forward_expand records for `outputs`, expanded in
// turn, in its order (ggml_visit_parents_graph, ggml.c:7213-7279, with the
// default left-to-right order of inputs).
std::vector<ggml_tensor*> GraphOrder(std::span<ggml_tensor* const> outputs);

// A gate and an up product of one input and the GLU over them, which
// MMVF fuses into one kernel writing the GLU (ops.h MulMatVecGlu).
struct MulMatGluNodes {
  ggml_tensor* gate = nullptr;
  ggml_tensor* up = nullptr;
  ggml_tensor* glu = nullptr;
};

// A product and the add of a bias (or residual) of its shape, which MMVF
// fuses into one kernel writing the add (ops.h MulMatVecBias).
struct MulMatAddNodes {
  ggml_tensor* mul_mat = nullptr;
  ggml_tensor* add = nullptr;
};

// A RoPE, the view that flattens its heads and the KV write that stores
// it, which the fused RoPE writes straight into the destination
// (ops.h RopeSetRows).
struct RopeSetRowsNodes {
  ggml_tensor* rope = nullptr;
  ggml_tensor* view = nullptr;
  ggml_tensor* set_rows = nullptr;
};

// {MUL_MAT, MUL_MAT, GLU} at `index`: ggml_cuda_can_fuse's gate
// (ggml-cuda.cu:3213-3224: ggml_can_fuse_subgraph, ggml.c:7737-7794;
// ggml_cuda_should_fuse_mul_mat, ggml-cuda.cu:1673-1765; and the memory
// ranges, 2971-3033) and ggml_cuda_try_fuse's use of it
// (3937-3955), which requires the gate product first. Any GLU operation
// upstream fuses matches; the implementation takes SwiGLU.
std::optional<MulMatGluNodes> MulMatGluFusionAt(GraphNodes graph, std::size_t index);

// {MUL_MAT, ADD} at `index` (ggml_cuda_try_fuse, ggml-cuda.cu:4074-4128):
// ggml_can_fuse's rules (ggml_can_fuse_ext, ggml-impl.h:681-709) and a
// bias of the product's shape. No memory-range check applies here upstream.
std::optional<MulMatAddNodes> MulMatAddFusionAt(GraphNodes graph, std::size_t index);

// {ROPE, VIEW, SET_ROWS} at `index` (ggml-cuda.cu:3543-3550, through
// ggml_cuda_can_fuse at 3256-3267 and ggml_cuda_should_fuse_rope_set_rows
// at 2666-2698, with the memory ranges). Complete: it needs no device.
std::optional<RopeSetRowsNodes> RopeSetRowsFusionAt(GraphNodes graph, std::size_t index);

// An RMSNorm and the mul that scales it, which GGML's fused launcher writes
// in one kernel (ops.h RmsNormMul).
struct RmsNormMulNodes {
  ggml_tensor* norm = nullptr;
  ggml_tensor* mul = nullptr;
};

// {RMS_NORM, MUL} at `index` (ggml-cuda.cu:4150-4153, through
// ggml_cuda_can_fuse at 3269-3314): ggml_can_fuse's rules, F32 operands,
// no broadcast when the norm is the mul's second operand, and rows that
// are contiguous. Complete: it needs no device. Upstream tries the
// five- and three-node RMSNorm patterns first; UnimplementedFusionAt
// covers them.
std::optional<RmsNormMulNodes> RmsNormMulFusionAt(GraphNodes graph, std::size_t index);

struct RmsNormChainNodes {
  ggml_tensor* norm = nullptr;
  ggml_tensor* mul = nullptr;
  ggml_tensor* out = nullptr;
};
// Three consecutive nodes with exactly one reader of each intermediate,
// no output/view intermediates and checked launcher operands. Kept storage
// readers are additionally excluded by the planner.
std::optional<RmsNormChainNodes> GemmaNormRopeFusionAt(GraphNodes graph, std::size_t index);
std::optional<RmsNormChainNodes> GemmaNormAddFusionAt(GraphNodes graph, std::size_t index);

// Exact RMS_NORM,MUL,GET_ROWS,ADD frontier pattern. The planner defers only
// norm/mul, executes GET_ROWS normally, then launches the checked ADD chain.
std::optional<RmsNormChainNodes> GemmaNormAddGatherFusionAt(GraphNodes graph, std::size_t index);

// Whether a fusion pattern upstream tries at `index`, other than the four
// above, might apply there. It checks the op sequences each such pattern
// requires (ggml_cuda_try_fuse, ggml-cuda.cu:3432-4180, and its matchers),
// and operations no implementation here supports, so a true answer is
// conservative: a planner that must reproduce upstream refuses the graph
// rather than guess. Names the pattern, or nothing.
std::optional<std::string_view> UnimplementedFusionAt(GraphNodes graph, std::size_t index);

// ggml_cuda_check_fusion_memory_ranges (ggml-cuda.cu:2971-3033): whether
// the output node `output` overlaps no input of the `count` nodes from
// `index`, other than a node among them before the one reading it (an
// elided intermediate). Inputs that are leaves (GGML_OP_NONE) are not
// checked, as upstream does not.
bool FusionMemoryClear(GraphNodes graph, std::size_t index, std::size_t count, std::size_t output);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_FUSION_H_
