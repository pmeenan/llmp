// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The implementation registry and plans (D-053;
// docs/backend-proof.md#dispatch-and-implementations-d-053), as small as
// backend-proof P1 needs them: the operation contract, and with it this
// interface, is settled at P6.
//
// A kernel module declares each implementation it compiles, and a program
// builds the registry from those declarations; the execution layer never
// includes a kernel module (docs/architecture.md#layers-and-dependency-rules).
// An implementation absent from a build, because its module or the device
// code is not compiled in, is never declared, so the registry never sees it.
//
// A plan names one implementation for each of its operations, with the
// identity that implementation had when the plan was made, and the plan's
// identity is a digest of those identities in order: changing any
// operation's implementation, or an implementation's identity, changes the
// plan's. Resolving a plan against a registry binds every operation to the
// implementation it names, or rejects the whole plan with the first
// operation that does not bind: an implementation this build lacks is
// unsupported (BP-S4), one whose identity has changed is stale (BP-S2),
// and neither is ever replaced by another implementation of the operation.

#ifndef JITLLM_EXECUTION_REGISTRY_H_
#define JITLLM_EXECUTION_REGISTRY_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"

namespace jitllm::execution {

// The operations the implementations cover (kernels/ggml/ops.h,
// kernels/exl3/linear.h and launch.h).
enum class Operation : std::uint8_t {
  kRmsNorm,     // rows scaled to unit root mean square
  kRmsNormMul,  // the same, then scaled by a weight
  kAdd,
  kMul,
  kMatMul,
  kGetRows,           // rows gathered by index
  kSetRows,           // rows stored at indices: the KV write
  kRope,              // rotary position embedding
  kRopeSetRows,       // RoPE, then its rows stored at indices
  kSoftMax,           // masked, scaled soft_max over rows
  kCont,              // a copy into packed layout
  kSwiGlu,            // silu(gate) * up
  kMulMatAdd,         // a matrix product plus a bias (or residual) of its shape
  kMulMatGlu,         // gate and up products of one input, then SwiGLU
  kQuantLinear,       // a product with quantized weights and their transforms
  kQuantMultiLinear,  // several such products of one input
  kConvert,           // an element-type conversion of a dense tensor (a cast)
  kFlashAttn,         // masked, scaled attention over a K/V cache, never materializing scores
  kBiasAdd,           // a bias broadcast over rows, added in the operands' own type
  // DeepSeek V4 Flash's and Qwen3.8 Flash's operations (M3).
  kMulMatId,  // each token's rows times its selected experts' weights
  kSub,
  kDiv,
  kScale,  // x * s + b
  kUnary,  // an elementwise function of one tensor, named by the node
  kClamp,
  kFill,    // every element set to one value
  kRepeat,  // a tensor tiled to a larger shape
  kConcat,
  kSumRows,
  kArgsort,
  kTopK,              // the indices of each row's k largest values
  kSwiGluClamp,       // silu(min(gate, limit)) * clamp(up, -limit, limit)
  kSsmConv,           // a depthwise causal convolution over tokens
  kGatedDeltaNet,     // the gated delta rule's recurrence over tokens
  kLightningIndexer,  // DeepSeek V4's sparse-attention cell scores
  kHcComb,            // DeepSeek V4's hyper-connection mixing matrices
  kHcPre,             // hyper-connection streams folded into one
  kHcPost,            // one output spread back over the streams
  // Qwen3.8 Flash's fused hyper-connection and MoE-output work (M3).
  kHcCombine,   // a block's output added to each stream, gated per stream
  kHcNorm,      // each stream's rows scaled to unit RMS, times a weight
  kHcMix,       // gated streams folded into one
  kMoeGlu,      // the experts' scales, then SwiGLU
  kMoeCombine,  // the experts' weighted sum plus the gated shared expert
  kMoeRoute,    // routed rows sorted by expert
  kQuantize,    // activations into a quantized format
  kNormGate,    // rows scaled to unit RMS, times a weight and a sigmoid gate
  // Qwen-Image-2.1's operations (M3; kernels/image/implementations.h).
  kHeadNormRope,       // each head's rows scaled to unit RMS, times a weight, then rotated
  kLayerNormModulate,  // rows layer-normed, then scaled by a modulation row
  kGatedResidual,      // x + tanh(gate) * y, the gate a modulation row
  kGatedResidualNorm,  // the gated residual, then the next layer_norm_modulate of its x
  kEulerStep,          // the flow-matching scheduler's step
  kConv2d,             // a 2-D convolution plus a bias
  kChannelNorm,        // each pixel's channels scaled to unit norm, times a weight
  kUpsample,           // nearest 2x upsampling
  kDupUpAdd,           // a shortcut's channels spread over 2x2 pixels, added
  // Gemma 4 primitives (M3.5), preserving existing operation identities.
  kGeGlu,           // GELU-tanh(gate) * up
  kMulMatGeGlu,     // gate and up products of one input, then GELU-tanh GLU
  kRmsNormMulRope,  // learned RMS normalization, then rotation
  kRmsNormMulAdd,   // learned RMS normalization, then residual addition
};

std::string_view OperationName(Operation operation);

// One compiled implementation of an operation, as its module declares it.
// Its identity is every field (D-053: source, revision, build flags,
// variant, launch configuration and tuning data).
struct Implementation {
  // Unique in a build and the same across builds, such as
  // "ggml.rms_norm_mul.fused": lower-case letters, digits, '.' and '_'.
  std::string name;
  Operation operation = Operation::kRmsNorm;
  // The kernel source, such as "ggml".
  std::string source;
  // The source's exact bytes, such as its prepared tree's digest (D-057).
  std::string revision;
  // How it was compiled: toolchain, target and device architectures.
  std::string build;
  // What it launches: kernels, fusion, launch configuration and tuning.
  std::string variant;
};

// SHA-256 over every field, each length-prefixed.
base::Sha256Digest IdentityOf(const Implementation& implementation);

// Why a registry or a plan was refused.
enum class PlanError : std::uint8_t {
  kInvalid,         // malformed: a bad or duplicate name, empty, too large
  kUnsupported,     // this build has no implementation of that name
  kStale,           // it has one of that name, with another identity
  kWrongOperation,  // the named implementation is of another operation
};

struct PlanRejection {
  PlanError error = PlanError::kInvalid;
  // The plan position of the operation that did not bind, where one did.
  std::size_t operation = 0;
  std::string detail;
};

// The implementations a build compiled. Immutable once created.
class Registry {
 public:
  static constexpr std::size_t kMaxImplementations = 256;
  static constexpr std::size_t kMaxName = 128;

  static std::expected<Registry, PlanRejection> Create(std::vector<Implementation> implementations);

  // Moving keeps every declaration where it is, so a bound plan follows
  // the registry; nothing replaces a registry's declarations in place.
  Registry(Registry&&) = default;
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;
  Registry& operator=(Registry&&) = delete;
  ~Registry() = default;

  // The index of the implementation of that name.
  std::optional<std::size_t> Find(std::string_view name) const;
  std::size_t size() const { return implementations_.size(); }
  const Implementation& at(std::size_t index) const;
  // IdentityOf(at(index)), computed once.
  const base::Sha256Digest& identity(std::size_t index) const;

 private:
  Registry(std::vector<Implementation> implementations, std::vector<base::Sha256Digest> identities,
           std::vector<std::size_t> by_name);

  std::vector<Implementation> implementations_;
  std::vector<base::Sha256Digest> identities_;  // parallel to implementations_
  std::vector<std::size_t> by_name_;            // indices, sorted by name
};

// A planner's choice for one operation.
struct Choice {
  Operation operation = Operation::kRmsNorm;
  std::string implementation;
};

// Operations in launch order, each naming the implementation chosen for it
// and recording that implementation's identity at the time.
class Plan {
 public:
  static constexpr std::size_t kMaxOperations = std::size_t{1} << 16;

  struct Step {
    Operation operation = Operation::kRmsNorm;
    std::string implementation;
    base::Sha256Digest identity{};
  };

  // Each choice must name an implementation of its operation in `registry`.
  static std::expected<Plan, PlanRejection> Build(const Registry& registry,
                                                  std::span<const Choice> choices);

  const base::Sha256Digest& identity() const { return identity_; }
  std::span<const Step> steps() const { return steps_; }

 private:
  Plan(std::vector<Step> steps, base::Sha256Digest identity)
      : steps_(std::move(steps)), identity_(identity) {}

  std::vector<Step> steps_;
  base::Sha256Digest identity_;
};

// A plan with every operation bound to an implementation in one registry,
// which must outlive it (moving the registry keeps it bound).
class BoundPlan {
 public:
  const base::Sha256Digest& identity() const { return identity_; }
  std::size_t size() const { return bound_.size(); }
  const Implementation& at(std::size_t operation) const;

 private:
  friend std::expected<BoundPlan, PlanRejection> Resolve(const Registry& registry,
                                                         const Plan& plan);
  BoundPlan(std::vector<const Implementation*> bound, base::Sha256Digest identity)
      : bound_(std::move(bound)), identity_(identity) {}

  std::vector<const Implementation*> bound_;
  base::Sha256Digest identity_;
};

// Binds every step of `plan` to the implementation it names in `registry`,
// with the identity it recorded, or rejects the plan.
std::expected<BoundPlan, PlanRejection> Resolve(const Registry& registry, const Plan& plan);

}  // namespace jitllm::execution

#endif  // JITLLM_EXECUTION_REGISTRY_H_
