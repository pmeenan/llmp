// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Running one phase of the native EXL3 plan (model/qwen2_exl3.h) under
// llmpalooza's dispatch (D-053; docs/backend-proof.md, "Native EXL3 operation
// plan"), CUDA builds only. Bind resolves the phase plan against the
// implementation registry: every operation that launches a kernel is bound
// to the implementation it names, identity and all (GGML's for the norms,
// casts, RoPE, attention, adds, SwiGLU and embedding; ExLlamaV3's for the
// linears and bias adds), or the whole plan is refused (BP-S2, BP-S4); no
// other implementation is ever bound in its place. It builds each GGML
// operation's nodes over the memory it is given and runs every host check
// the implementations make, and for EXL3-O checks that llmpalooza's copy of
// upstream's GEMV choice (upstream_gemv.h) picks exactly the table's plan
// wherever upstream may take the GEMV. Run then queues the operations in
// order on one stream: the GGML and EXL3 launch contexts and the provider
// share it. The KV writes and the host inputs' upload are the provider's
// copies; nothing else is queued.
//
// The plan's identity is a SHA-256 of the registry plan's identity (every
// implementation's identity, in order) and the phase plan's launch digest
// (every operation, tensor and forced launch plan, and the tuning data
// they came from).

#ifndef LLMP_KERNELS_EXL3_QWEN2_H_
#define LLMP_KERNELS_EXL3_QWEN2_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"
#include "model/qwen2.h"
#include "model/qwen2_exl3.h"
#include "providers/device_execution.h"

namespace llmp::kernels::ggml {
class LaunchContext;
}

namespace llmp::kernels::exl3 {

// Where the model's resident tensors are, as device addresses.
struct Qwen2Linear {
  Weights weights;
  std::uint64_t bias = 0;  // F16 [n], or 0
};

struct Qwen2Layer {
  std::uint64_t attn_norm = 0;  // F32 [width], the artifact's BF16 widened exactly
  std::uint64_t mlp_norm = 0;
  Qwen2Linear q, k, v, o, gate, up, down;
  // The multi-GEMM's device tables for gate and up (validate.h
  // MultiLinearOperands) and the caller's record of what they hold.
  std::uint64_t trellis_table = 0;
  std::uint64_t suh_table = 0;
  std::uint64_t svh_table = 0;
  std::array<std::uint64_t, 6> tables_written{};
  // The layer's F16 cache, [cells, KV heads, head size] each.
  std::uint64_t k_cache = 0;
  std::uint64_t v_cache = 0;
};

struct Qwen2Memory {
  std::uint64_t embed = 0;       // BF16 [vocab, width]
  std::uint64_t final_norm = 0;  // F32 [width]
  Qwen2Linear lm_head;
  std::vector<Qwen2Layer> layers;
  int cells = 0;             // the cache's cells per layer
  std::uint64_t region = 0;  // the phase's activations: the plan's region, 256-byte aligned
  std::uint64_t region_bytes = 0;
};

// Called around each operation of a run, with its index in the plan, on
// the submitting thread: a recording harness notes what launched, or
// waits and reads what the operation wrote.
struct Qwen2Hooks {
  std::function<std::expected<void, KernelFailure>(std::size_t)> before;
  std::function<std::expected<void, KernelFailure>(std::size_t)> after;
};

class Qwen2Program {
 public:
  // Refused, with nothing queued, if an operation names an implementation
  // the registry lacks or holds with another identity, an operation's
  // operands fail its implementation's host checks, a tensor's slot lies
  // outside the region, or (EXL3-O) the table's packed plan is not the
  // one upstream's GEMV choice gives on this device.
  static std::expected<std::unique_ptr<Qwen2Program>, KernelFailure> Bind(
      const execution::Registry& registry, const model::Qwen2Profile& profile,
      const model::Exl3PhasePlan& plan, const Qwen2Memory& memory, ggml::LaunchContext& ggml_launch,
      LaunchContext& launch);

  Qwen2Program(const Qwen2Program&) = delete;
  Qwen2Program& operator=(const Qwen2Program&) = delete;
  Qwen2Program(Qwen2Program&&) = delete;
  Qwen2Program& operator=(Qwen2Program&&) = delete;
  ~Qwen2Program();

  // Queues the phase on `stream`, the launch contexts' stream: the inputs'
  // upload from `host_inputs` (pinned, laid out as HostInputs), then every
  // operation. Stops at the first refusal or fault, which it returns.
  std::expected<void, KernelFailure> Run(ggml::LaunchContext& ggml_launch, LaunchContext& launch,
                                         ReconGemm& gemm, providers::DeviceExecution& execution,
                                         providers::StreamId stream, std::uint64_t host_inputs,
                                         const Qwen2Hooks& hooks) const;

  const base::Sha256Digest& identity() const { return identity_; }
  const execution::BoundPlan& bound() const;
  // The most GGML pool scratch one operation draws (the attention's).
  std::uint64_t ggml_scratch() const { return ggml_scratch_; }
  // Where a tensor an operation reads or writes lives: the cache cells an
  // operation reads or writes for k_cache and v_cache, and for a weight
  // the address bound for it (BoundWeights' names).
  std::uint64_t Address(std::size_t op, std::string_view tensor) const;
  // The weights an operation reads, by name, at the addresses bound for
  // them: the embedding's, norms' and biases' by the record's input names;
  // a linear's "trellis", "suh" and "svh"; the multi-GEMM's
  // "gate_proj.trellis" ... "up_proj.svh" and its three device tables,
  // "table.trellis", "table.suh" and "table.svh" (16 bytes each, whose
  // contents the kernel follows). What a recording hashes, so that an
  // operation bound to another layer's or tensor's bytes shows (Tier E).
  std::span<const std::pair<std::string, std::uint64_t>> BoundWeights(std::size_t op) const;

  struct Step;

 private:
  Qwen2Program() = default;

  base::Sha256Digest identity_{};
  std::uint64_t ggml_scratch_ = 0;
  struct State;
  std::unique_ptr<State> state_;
};

// The host inputs of a phase, as Run uploads them: the token IDs (int32
// [rows]), the positions (int32 [rows], past onwards) and the mask (F16
// [rows, padded]: 0 where a column is at or before the row's position, -inf
// elsewhere, the padded columns included), each from a 256-byte boundary.
struct HostInputs {
  std::uint64_t ids = 0;
  std::uint64_t positions = 0;
  std::uint64_t mask = 0;
  std::uint64_t bytes = 0;
};
HostInputs HostInputsLayout(const model::Exl3PhasePlan& plan);
// Writes them into `out` (HostInputsLayout's bytes); refused if the token
// count is not the phase's rows or a token is outside the vocabulary.
std::expected<void, KernelFailure> WriteHostInputs(const model::Qwen2Profile& profile,
                                                   const model::Exl3PhasePlan& plan,
                                                   std::span<const std::int32_t> tokens,
                                                   std::span<std::byte> out);

// Rewrites every layer's multi-GEMM tables for its current gate and up
// weights (validate.h MultiGemmTables), as a caller must whenever either
// moves (BP-P5): 48 bytes per layer into `staging` (host memory the
// provider copies from, which the caller leaves untouched until the copies
// complete), then each table's 16 bytes copied on `stream`. Every layer's
// record (tables_written) is cleared before anything is queued and set
// only once its three copies are queued, ahead in stream order of any
// launch queued after this returns. So after a failure partway, a layer
// whose tables may be stale or half written has no record, and Bind
// refuses its multi-GEMM until the tables are written again.
std::expected<void, KernelFailure> WriteMultiGemmTables(providers::DeviceExecution& execution,
                                                        providers::StreamId stream,
                                                        std::span<std::byte> staging,
                                                        Qwen2Memory& memory);

}  // namespace llmp::kernels::exl3

#endif  // LLMP_KERNELS_EXL3_QWEN2_H_
