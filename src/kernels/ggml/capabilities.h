// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What the GGML module's implementations compute, as the planner selects
// them (D-053, D-108; execution/registry.h Capability): each one's IR
// operation, its operands' encodings and layouts, the devices and shapes
// it takes, and, where its launcher or upstream's routing decides more, a
// predicate (`accepts`) for its supported typed operand domain. implementations.cc attaches them to
// the module's declarations; the table itself builds in every profile, so a CPU build plans for the
// GGML module (the lowering's structural tests) with the capabilities a CUDA build declares.
//
// Groups. GGML's matrix product, its expert product and its flash attention
// are families that upstream's routing picks among by device and shape
// (ggml_cuda_mul_mat, ggml_cuda_mul_mat_id, the attention case selection;
// DeviceChoices). Their members declare the group ("ggml.mul_mat",
// "ggml.mul_mat_id", "ggml.flash_attn_ext"): the planner records the
// group, and the lowering (lower.h) resolves its concrete recipe before final placement and
// binding. A member's own predicate only says the family takes the node; where the planner's choice
// depends on the routing itself (a shared Q8_1 preparation, the fused quantized GLU, which upstream
// takes only where it routes the product to MMVQ), the capability carries upstream's routing
// (MmvqRouted).
//
// Architectures are the provider's numbers (CUDA's major · 10 + minor). An
// implementation is selectable only for an architecture this build has
// SASS for (CompiledFor); a CPU build describes the native profile's GB10
// (sm_121a), the device its planning tests model.

#ifndef LLMP_KERNELS_GGML_CAPABILITIES_H_
#define LLMP_KERNELS_GGML_CAPABILITIES_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "execution/registry.h"
#include "format/encoding.h"

namespace llmp::kernels::ggml {

// The capability `implementation` declares, or null where it declares
// none (a family runner's own operation the IR does not name).
const execution::Capability* CapabilityOf(std::string_view implementation);

// What its predicate checks, which its declaration's variant names (an
// identity covers a capability but not its predicate); empty without one.
std::string_view AcceptsTextOf(std::string_view implementation);

// Every implementation that declares a capability, with its operation: a
// registry for planning in a build without the kernels (tests) is made
// from these.
struct CapabilityDeclaration {
  std::string_view name;
  const execution::Capability* capability = nullptr;
};
std::span<const CapabilityDeclaration> CapabilityDeclarations();

// Concrete recipes share this inventory with their validators and launchers.
// Arity zero means validator-owned bounded arity; other values are exact.
struct RecipeDeclaration {
  std::string_view name;
  ir::OpKey operation;
  std::string_view numerical_contract;
  std::size_t arity;
};
std::span<const RecipeDeclaration> RecipeDeclarations();

// Structural registries for CPU planning/lowering checks. These declarations
// include concrete helper recipes, carry their actual keys/numerical contracts,
// and explicitly cannot bind executable kernels of any build.
std::vector<execution::Implementation> PlanningImplementations();

// Whether this build has SASS that runs on a device of `arch`: an `sm_XY`
// cubin on X.Y up to the next major, an arch-specific `sm_XYa` on X.Y only.
bool CompiledFor(std::uint32_t arch);

// Upstream's choice of its vector kernel (MMVQ) for a product of weights
// in `weights` over `columns` activation columns on an NVIDIA device of
// `arch` (ggml_cuda_should_use_mmvq at the pin, mmvq.cu, with GGML's
// compute capability arch · 10): a quantized encoding, and the columns
// within the device's and type's tuned bound.
bool MmvqRouted(format::Encoding weights, std::uint32_t arch, std::uint64_t columns);

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_CAPABILITIES_H_
