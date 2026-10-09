// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The graph IR's op set, version 1 (D-107; docs/engine-components.md#the-graph-ir):
// the operations a model's step is built from, the fused kinds patterns
// rewrite subgraphs into, and the parameters a node carries. Changing the
// op set is a reviewed change, like an artifact schema: it bumps
// kOpSetVersion.
//
// An operation's semantics are its op, its parameters and its operands'
// formats; an implementation (execution/registry.h) declares which of
// those it computes. Every fused kind has a primitive decomposition, so a
// graph without its pattern is still correct (portability.md's
// primitive-fallback rule). Parameter structs are trivially copyable and
// fit a node's parameter bytes.
//
// So far this header is the vocabulary only; the IR's tensors, graph,
// builder and verifier come with the IR itself.

#ifndef LLMP_IR_OP_H_
#define LLMP_IR_OP_H_

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace llmp::ir {

inline constexpr std::uint32_t kOpSetVersion = 1;

enum class Op : std::uint16_t {
  // Structure.
  kInput,
  kWeight,
  kState,
  kView,
  kReshape,
  kPermute,
  kCont,
  kConcat,
  kRepeat,
  kConvert,  // a dense tensor into another encoding (a cast)
  kGetRows,  // rows gathered by index
  // Products.
  kMatMul,
  kMatMulId,    // grouped: each row times its selected matrices
  kPrepare,     // activations into a product's input form (a quantization)
  kExpertSort,  // routed rows sorted by matrix: a grouped product's worklist
  // Attention and masks.
  kAttention,   // AttnParams
  kMaskBuild,   // MaskParams
  kCacheStore,  // rows stored at indices (set_rows)
  kIndexScore,  // sparse attention's cell scores
  kPool,        // compressed KV pooling
  // Norms and rotations.
  kRmsNorm,
  kL2Norm,
  kLayerNorm,
  kChannelNorm,
  kRope,  // RopeParams
  // Elementwise.
  kAdd,
  kSub,
  kMul,
  kDiv,
  kScale,  // x · scale + bias
  kUnary,  // UnaryParams
  kClamp,
  kFill,
  kGlu,  // GluParams
  kSoftcap,
  // Reductions and selection.
  kSumRows,
  kSoftMax,
  kArgmax,
  kArgsort,
  kTopK,
  // Routing.
  kRoute,         // RouteParams: scores into selected experts and weights
  kExpertReduce,  // the selected experts' outputs, weighted and summed
  // Recurrences and convolutions.
  kSsmConv,
  kGatedDeltaNet,
  kConv2d,
  kUpsample,
  // Hyper-connections: composite ops with a mandatory primitive
  // decomposition.
  kHcPre,   // streams folded into one
  kHcPost,  // one output spread back over the streams
  kHcMix,   // HcParams: the mixing weights
  // Sampling and collectives.
  kSample,
  kAllReduce,
  kAllGather,
  kAllToAll,
  kSend,
  kRecv,
  // A pattern's result; FusedKind names it.
  kFused,
};

enum class FusedKind : std::uint16_t {
  kNone,
  kRmsNormMul,         // RMS norm, then a weight
  kRmsNormMulRope,     // ... then a rotation
  kRmsNormMulAdd,      // ... then a residual add
  kRopeStore,          // a rotation, then its rows stored
  kMatMulGlu,          // gate and up products of one input, then a GLU
  kMatMulAdd,          // a product plus a bias or residual of its shape
  kOwnerAttention,     // attention with its masks and stores, per owner
  kGroupedStores,      // several cache stores as one
  kRouteReduce,        // routing with its experts' reduction
  kMoeGlu,             // the experts' scales, then a GLU
  kMoeCombine,         // the experts' weighted sum plus a gated shared expert
  kHcNorm,             // each stream's rows RMS-normed, times a weight
  kHcNormPost,         // a stream norm with the spread that follows it
  kQkPrep,             // per-head norm and rotation of queries and keys
  kPreparedProducts,   // several products sharing one prepared input
  kNormGate,           // RMS norm, a weight and a sigmoid gate
  kLayerNormModulate,  // a layer norm scaled by a modulation row
  kGatedResidual,      // x + tanh(gate) · y
  kGatedResidualNorm,  // the gated residual, then the next modulated norm
  kEulerStep,          // the flow-matching scheduler's step
  kDupUpAdd,           // a shortcut's channels spread over 2x2 pixels, added
};

struct OpKey {
  Op op = Op::kInput;
  FusedKind fused = FusedKind::kNone;
  auto operator<=>(const OpKey&) const = default;
};

constexpr OpKey Primitive(Op op) { return {.op = op, .fused = FusedKind::kNone}; }
constexpr OpKey Fused(FusedKind kind) { return {.op = Op::kFused, .fused = kind}; }

std::string_view OpName(Op op);
std::string_view FusedKindName(FusedKind kind);
// "mul_mat", or "fused.rms_norm_mul".
std::string OpKeyName(OpKey key);

// ---------------------------------------------------------------- parameters

inline constexpr std::size_t kMaxParamBytes = 64;

struct NormParams {
  float eps = 0.0f;
};

enum class RopeMode : std::uint8_t { kNormal, kNeox, kMulti, kVision };
// GGML's rope_ext's parameters.
struct RopeParams {
  RopeMode mode = RopeMode::kNormal;
  std::uint32_t dims = 0;  // rotated values a head
  std::uint32_t original_context = 0;
  float base = 10000.0f;
  float scale = 1.0f;
  float ext_factor = 0.0f;
  float attn_factor = 1.0f;
  float beta_fast = 32.0f;
  float beta_slow = 1.0f;
  std::array<std::uint32_t, 4> sections{};  // kMulti's
};

enum class MaskKind : std::uint8_t { kCausal, kWindow, kBlock };
struct MaskParams {
  MaskKind kind = MaskKind::kCausal;
  std::uint32_t window = 0;  // kWindow: positions seen
  std::uint32_t block = 0;   // kBlock: positions a block
};

struct AttnParams {
  float scale = 1.0f;
  float softcap = 0.0f;      // 0: none
  float max_bias = 0.0f;     // ALiBi's; 0: none
  std::uint32_t window = 0;  // 0: the whole cache
  bool sinks = false;        // a learned sink per head
};

enum class UnaryFn : std::uint8_t {
  kSilu,
  kGelu,
  kGeluTanh,
  kSigmoid,
  kTanh,
  kRelu,
  kExp,
  kSoftplus,
  kSqrt,
  kSqr,
  kNeg,
  kAbs,
  kSgn,
};
struct UnaryParams {
  UnaryFn fn = UnaryFn::kSilu;
};

enum class GluKind : std::uint8_t { kSwiGlu, kGeGlu, kGeGluTanh, kSwiGluClamp };
struct GluParams {
  GluKind kind = GluKind::kSwiGlu;
  bool swapped = false;  // up, then gate
  float limit = 0.0f;    // kSwiGluClamp's
};

struct ScaleParams {
  float scale = 1.0f;
  float bias = 0.0f;
};

struct ClampParams {
  float min = 0.0f;
  float max = 0.0f;
};

struct FillParams {
  float value = 0.0f;
};

struct SoftcapParams {
  float cap = 0.0f;
};

struct SoftMaxParams {
  float scale = 1.0f;
  float max_bias = 0.0f;
};

struct ArgsortParams {
  bool descending = false;
};

struct TopKParams {
  std::uint32_t k = 0;
};

enum class RouteKind : std::uint8_t { kSoftmaxTopK, kSigmoidTopK, kSqrtSoftplusBias, kHash };
struct RouteParams {
  RouteKind kind = RouteKind::kSoftmaxTopK;
  std::uint32_t experts = 0;
  std::uint32_t used = 0;
  bool normalize = false;  // the selected weights scaled to sum to 1
  float scale = 1.0f;      // then multiplied by this
};

enum class HcMixKind : std::uint8_t { kSigmoid, kSinkhorn };
struct HcParams {
  HcMixKind mix = HcMixKind::kSigmoid;
  std::uint32_t streams = 0;
  std::uint32_t iterations = 0;  // kSinkhorn's
  float eps = 0.0f;
};

struct ConcatParams {
  std::uint8_t dim = 0;
};

struct PermuteParams {
  std::array<std::uint8_t, 4> axes{0, 1, 2, 3};
};

struct Conv2dParams {
  std::uint32_t stride = 1;
  std::uint32_t padding = 0;
  std::uint32_t dilation = 1;
};

struct UpsampleParams {
  std::uint32_t factor = 2;
};

struct PoolParams {
  std::uint32_t ratio = 1;  // cache rows pooled into one
};

enum class SampleKind : std::uint8_t { kGreedy, kSeeded };
struct SampleParams {
  SampleKind kind = SampleKind::kGreedy;
  float temperature = 1.0f;
  std::uint32_t top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
};

struct CollectiveParams {
  std::uint32_t group = 0;  // the partition's device group
  std::uint32_t peer = 0;   // kSend's and kRecv's
};

template <typename P>
inline constexpr bool kIsParams = std::is_trivially_copyable_v<P> && sizeof(P) <= kMaxParamBytes;

static_assert(kIsParams<NormParams> && kIsParams<RopeParams> && kIsParams<MaskParams> &&
              kIsParams<AttnParams> && kIsParams<UnaryParams> && kIsParams<GluParams> &&
              kIsParams<ScaleParams> && kIsParams<ClampParams> && kIsParams<FillParams> &&
              kIsParams<SoftcapParams> && kIsParams<SoftMaxParams> && kIsParams<ArgsortParams> &&
              kIsParams<TopKParams> && kIsParams<RouteParams> && kIsParams<HcParams> &&
              kIsParams<ConcatParams> && kIsParams<PermuteParams> && kIsParams<Conv2dParams> &&
              kIsParams<UpsampleParams> && kIsParams<PoolParams> && kIsParams<SampleParams> &&
              kIsParams<CollectiveParams>);

}  // namespace llmp::ir

#endif  // LLMP_IR_OP_H_
