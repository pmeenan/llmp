// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Planning a GGML graph's execution as upstream's CUDA backend would run
// it (D-053; docs/backend-proof.md, Tier E): which of the module's
// implementations (implementations.h) runs each node, fused or not, and
// where each computed tensor lives.
//
// PlanGraph walks the nodes in order as ggml_cuda_graph_evaluate_and_capture
// does (ggml-cuda.cu:4185-4360). Views and no-ops launch nothing. With
// fusion on (FP16-F), each node is first offered to upstream's fusion
// patterns in ggml_cuda_try_fuse's order, through fusion.h's gates; a graph
// where a pattern jitLLM does not implement might apply is refused, never
// run differently. With fusion off (FP16-U, GGML_CUDA_DISABLE_FUSION),
// upstream consults no pattern: an RMSNorm whose mul follows it runs as the
// unfused RMSNorm-mul implementation (rms_norm's launcher, then mul's), and
// everything else node by node. A matrix product's kernel family is
// upstream's selection on the device (ggml_cuda_mul_mat), which the caller
// supplies (ops.h on a CUDA build; a model of it in tests).
//
// Upstream's fusion gates compare the fused output's memory with its
// inputs', so the plan depends on addresses, and the addresses on the plan
// (what is live together). PlanActivations breaks the circle: plan with
// every computed tensor at distinct addresses, place the tensors by that
// plan's steps (a step's outputs never share memory with anything live
// during it), bind them, plan again with the real addresses and require the
// same plan (SamePlan). Every profile builds this; nothing here launches.

#ifndef JITLLM_KERNELS_GGML_GRAPH_PLAN_H_
#define JITLLM_KERNELS_GGML_GRAPH_PLAN_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"

namespace jitllm::kernels::ggml {

// The device's part of upstream's choices.
struct DeviceChoices {
  // The family ggml_cuda_mul_mat selects for a product, or a refusal where
  // it would take a path no implementation here has.
  std::function<std::expected<MulMatPath, KernelFailure>(const ggml_tensor*)> mul_mat;
  // Pinned local D256/GQA2 solo choice. Without a device selector use
  // the checked group2 MMA primitive; existing attention paths stay intact.
  std::function<bool(const ggml_tensor*)> flash_attn_vec256 = nullptr;
  // ops.h MulMatVecFusible: the MMVF fusion gates' device condition.
  std::function<bool(const ggml_tensor*)> vector_fusible;
  // ops_ext.h SelectMulMatQ: the family upstream routes a quantized
  // mul_mat or mul_mat_id to. Without it, quantized products are refused.
  std::function<std::expected<QuantMulMatPath, KernelFailure>(const ggml_tensor*)> quant;
  // Measured device/shape eligibility for the default-off raw Q2_K D2R
  // product. A CPU planner may supply a model of this predicate.
  std::function<bool(const ggml_tensor*)> q2_d2r_fits = nullptr;
  std::function<bool(const ggml_tensor*)> ds4_hca_fits = nullptr;
  // The device condition of pair_glu (below) for an (up, gate) pair: ops_ext.h
  // MulMatIdQPairGluSupported. Without it, the write-back pair is never planned.
  std::function<bool(const ggml_tensor*, const ggml_tensor*)> pair_glu_fits = nullptr;
  // A speculative verify's plan (D-092): every matrix product of up to
  // kRowsMaxColumns columns (tokens) runs a row-invariant implementation
  // (the quantized ones jitllm.mul_mat*.mmvq_rows, the float ones MMVF),
  // whatever upstream would route it to, so that each row equals the
  // one-row plan's; a wider product is refused. Everything else as
  // upstream.
  bool row_invariant = false;
  // Planned without upstream's fusion, an RMSNorm and the mul that scales
  // it (RmsNormMulFusionAt's structural conditions) still run as the fused
  // RMSNorm-mul: jitLLM's fast plans (DeepSeek V4's, the owner's policy of
  // 2026-09-28), which do not reproduce upstream's unfused arithmetic.
  bool fuse_norms = false;
  // Checked three-node norm fusions, separately requested and default off.
  // Keep/view readers force the existing primitive steps.
  bool fuse_norm_rope = false;
  bool fuse_norm_add = false;
  // Exact Gemma 128/top-eight routing and scaled ordered reduction.
  // Explicit policies retain all descriptors/backing and primitive keep fallback.
  bool fuse_gemma_route = false;
  bool fuse_gemma_reduce = false;
  // Explicit RoPE/direct-view/cache-store fusion without enabling upstream's
  // other fusion gates. Full operand checks and diagnostic keep apply.
  bool fuse_rope_store = false;
  // And float products of at most kRowInvariantColumns columns run GGML's
  // vector kernel (MMVF) whatever upstream would route them to (MMF past
  // one BF16 column): a verify's router and head mixes.
  bool vector_floats = false;
  // The same vector kernel selected for individual float products, rather
  // than all of them. Other products keep the ordinary device selector.
  std::function<bool(const ggml_tensor*)> vector_float_node = nullptr;
  // Separate measured policy for GeGLU MMVF fusion, default off. Ordinary
  // MMVF eligibility alone does not qualify GeGLU: the shared N2112 screen
  // loses. The predicate receives the up product; full structural, precision
  // and memory checks still apply. No current model enables this callback.
  std::function<bool(const ggml_tensor*)> geglu_fusible = nullptr;
  // Fast plans may share preparation across adjacent ordinary MMQ expert
  // products. The default keeps one primitive step per graph node.
  bool pair_experts = false;
  // Use a device-built expert-major MMQ tile list for measured fast
  // quantized shapes. Reference/default plans retain upstream's grid.
  bool compact_experts = false;
  // Keep native Q8 preparation and raw weights, then use ds4's D2R
  // arithmetic for the measured GB10 Q2_K down product (q2_d2r_fits). Not
  // bit-identical; qualified by 32K perplexity
  // (docs/experiments/ds4-prefill-stages). The fast DeepSeek plan's default.
  bool d2r_experts = false;
  // Benchmark-only literal ds4 HCA comparison. Tagged causal metadata and
  // measured device/layout eligibility are both required; fallback is stable.
  bool ds4_hca = false;
  // Share a sparse query tile's KV union only for measured fast shapes.
  // Count-based DeepSeek HCA masks retain the ordinary implementation;
  // selected-list CSA and window masks may use sharing when opted in.
  // Unknown/reference graphs retain the original column selection.
  bool wide_sparse_attention = false;
  // The ds4 prefill stage mechanisms (docs/experiments/ds4-prefill-stages),
  // each byte-identical to the plan without it and each under its own
  // shape guard; the fast DeepSeek plan's defaults (SetDsv4PrefillStages).
  // With compact experts, the measured GB10 IQ2 occupancy-two pair followed
  // by its swiglu_clamp writes the activation from the up product's
  // write-back instead of the up output (where nothing else reads it).
  bool pair_glu = false;
  // And with it, the activation written as the following Q2_K down product's
  // D2S6 input, which that product then reads without quantizing (where
  // nothing else reads the activation). Where d2r_experts takes that down
  // product, the pair writes the F32 activation.
  bool pair_glu_q8 = false;
  // Where an HC post with F16 rows reads the routed experts' ordered
  // reduction plus the shared expert, form that sum in the post's kernel
  // (dsv4_hc_norm.h Dsv4HcPostExpertsAt) instead of writing it.
  bool hc_post_experts = false;
  // Output-A with a coalesced weight repack (identical bytes).
  bool outa_fast_pack = false;
  // Two dense MMQ products of one block-quantized type (not FP4) and one
  // activation share one quantization; the later product runs at the
  // earlier one's place.
  bool dense_pair = false;
};

// The fast DeepSeek plan's device-side prefill stage mechanisms (above) and
// the D2R down product, all on or all off. Reference plans keep them off.
inline void SetDsv4PrefillStages(DeviceChoices& device, bool on) {
  device.d2r_experts = on;
  device.pair_glu = on;
  device.pair_glu_q8 = on;
  device.hc_post_experts = on;
  device.outa_fast_pack = on;
  device.dense_pair = on;
}

// One implementation's run over its nodes, in the order implementations.h
// lists for it.
struct PlanStep {
  execution::Operation operation = execution::Operation::kAdd;
  std::string_view implementation;
  std::vector<ggml_tensor*> nodes;
  // The lane it runs on (AssignLanes): 0, the context's stream, or 1 to
  // kMaxLanes, a concurrent lane inside a region.
  std::uint8_t lane = 0;
};

// Concurrent lanes. A graph builder may tag nodes that run independently of
// one another (a wave's slots' attention and state operations) with a lane
// and a region (LaneTag). AssignLanes gives each step its nodes' lane and
// records each region's span of steps. Over a region's span, BoundGraph::Run
// queues each lane's steps on a stream of its own (LaunchContext lanes): the
// lanes wait for everything queued before the span, a step waits for the
// steps on other lanes that last wrote what it reads or writes, or read
// what it writes since (LaneOrder), and the context's stream waits for
// every lane at the span's end. PlaceActivations keeps every
// tensor a step in a span computes or reads live for the whole span, so
// tensors of concurrent steps never share bytes. Each lane draws its scratch
// from a pool of its own. The same kernels run with the same operands, so
// each step computes what it computes on one stream, bit for bit.
inline constexpr std::uint32_t kMaxLanes = 4;
struct LaneTag {
  std::uint8_t lane = 0;     // 1 to kMaxLanes; 0 keeps the step on the stream
  std::uint32_t region = 0;  // 1 on; 0 is no region
};
using LaneTags = std::vector<std::pair<const ggml_tensor*, LaneTag>>;

struct GraphPlan {
  std::vector<PlanStep> steps;
  // Each region's first and last steps, in order and disjoint.
  struct Region {
    std::uint32_t first = 0;
    std::uint32_t last = 0;
  };
  std::vector<Region> regions;

  // The plan as the registry takes it (execution/registry.h).
  std::vector<execution::Choice> Choices() const;
};

// Gives `plan`'s steps their nodes' lanes and records the regions' spans;
// regions whose spans overlap are joined. A step whose implementation
// `on_stream` names keeps lane 0: one that borrows cuBLAS, whose handle and
// workspace are the context stream's (implementations.h UsesCublas; without
// `on_stream`, GGML's cuBLAS product and jitllm.gemm.bf16).
void AssignLanes(GraphPlan& plan, const LaneTags& tags,
                 const std::function<bool(std::string_view)>& on_stream = {});

// The order concurrent lanes need inside a region (BoundGraph::Run): before
// a step, the other lanes it must wait for. A step waits for the last
// writer of every tensor it reads or writes (read after write, write after
// write) and for every lane that read a tensor it writes since that write
// (write after read). Tensors are storage (a view counts as its source);
// two separate leaf tensors bound to overlapping memory are not linked, so
// lanes must not share memory through distinct leaves.
// Lane 0 is the context's stream; a wait covers everything the awaited lane
// has queued so far.
class LaneOrder {
 public:
  static constexpr std::size_t kLanes = kMaxLanes + 1;

  // The lanes `lane` must wait for before a step reading `reads` and
  // writing `writes`, each once, in lane order.
  std::vector<std::uint8_t> Before(std::uint8_t lane, std::span<const ggml_tensor* const> reads,
                                   std::span<const ggml_tensor* const> writes) const;
  // `waiter` waited for everything `on` has queued.
  void Waited(std::uint8_t waiter, std::uint8_t on);
  // A step queued on `lane`.
  void Queued(std::uint8_t lane, std::span<const ggml_tensor* const> reads,
              std::span<const ggml_tensor* const> writes);
  // A region ended: every lane joined to the stream, nothing outstanding.
  void Reset();

 private:
  struct Access {
    std::uint8_t writer = 0;
    std::uint64_t written = 0;                 // the writer's count at the write; 0: none
    std::array<std::uint64_t, kLanes> read{};  // each lane's count at its last read since
  };
  std::array<std::uint64_t, kLanes> queued_{};
  std::array<std::array<std::uint64_t, kLanes>, kLanes> seen_{};
  std::unordered_map<const ggml_tensor*, Access> access_;
};

// The module's implementation names (implementations.h).
inline constexpr std::string_view kRmsNormMulFused = "ggml.rms_norm_mul.fused";
inline constexpr std::string_view kRmsNormMulUnfused = "ggml.rms_norm_mul.unfused";
inline constexpr std::string_view kRmsNormName = "ggml.rms_norm";
inline constexpr std::string_view kAddName = "ggml.add";
inline constexpr std::string_view kMulName = "ggml.mul";
inline constexpr std::string_view kMulMatVector = "ggml.mul_mat.mmvf";
inline constexpr std::string_view kMulMatTensorCore = "ggml.mul_mat.mmf";
inline constexpr std::string_view kMulMatCublas = "ggml.mul_mat.cublas";
inline constexpr std::string_view kGetRowsName = "ggml.get_rows";
inline constexpr std::string_view kSetRowsName = "ggml.set_rows";
inline constexpr std::string_view kRopeName = "ggml.rope.neox";
inline constexpr std::string_view kRopeSetRowsFused = "ggml.rope_set_rows.fused";
inline constexpr std::string_view kSoftMaxName = "ggml.soft_max";
inline constexpr std::string_view kContName = "ggml.cont";
inline constexpr std::string_view kSwiGluName = "ggml.swiglu";
inline constexpr std::string_view kGeGluName = "ggml.geglu";
inline constexpr std::string_view kMulMatGeGluFused = "ggml.mul_mat_geglu.mmvf_fused";
inline constexpr std::string_view kMulMatAddFused = "ggml.mul_mat_add.mmvf_fused";
inline constexpr std::string_view kMulMatGluFused = "ggml.mul_mat_glu.mmvf_fused";
// M3's (ops_ext.h), planned with fusion off only.
inline constexpr std::string_view kMulMatVecQ = "ggml.mul_mat.mmvq";
inline constexpr std::string_view kMulMatQ = "ggml.mul_mat.mmq";
inline constexpr std::string_view kMulMatQPairDense = "jitllm.mul_mat.mmq_pair_dense";
inline constexpr std::string_view kMulMatHadamard = "ggml.mul_mat.fwht";
inline constexpr std::string_view kMulMatIdVecQ = "ggml.mul_mat_id.mmvq";
inline constexpr std::string_view kMulMatIdQ = "ggml.mul_mat_id.mmq";
inline constexpr std::string_view kMulMatIdQPair = "jitllm.mul_mat_id.mmq_pair";
inline constexpr std::string_view kMulMatIdQCompact = "jitllm.mul_mat_id.mmq_compact";
inline constexpr std::string_view kMulMatIdQPairCompact = "jitllm.mul_mat_id.mmq_pair_compact";
inline constexpr std::string_view kMulMatIdQPairGlu = "jitllm.mul_mat_id.mmq_pair_glu";
inline constexpr std::string_view kMulMatIdQPairGluQ8 = "jitllm.mul_mat_id.mmq_pair_glu_q8";
inline constexpr std::string_view kMulMatIdQCompactPrequant =
    "jitllm.mul_mat_id.mmq_compact_prequant";
inline constexpr std::string_view kMulMatIdQ2D2r = "jitllm.mul_mat_id.q2_d2r";
inline constexpr std::string_view kSubName = "ggml.sub";
inline constexpr std::string_view kDivName = "ggml.div";
inline constexpr std::string_view kScaleName = "ggml.scale";
inline constexpr std::string_view kUnaryName = "ggml.unary";
inline constexpr std::string_view kClampName = "ggml.clamp";
inline constexpr std::string_view kFillName = "ggml.fill";
inline constexpr std::string_view kRepeatName = "ggml.repeat";
inline constexpr std::string_view kConcatName = "ggml.concat";
inline constexpr std::string_view kSumRowsName = "ggml.sum_rows";
inline constexpr std::string_view kArgsortName = "ggml.argsort.bitonic";
inline constexpr std::string_view kTopKName = "ggml.top_k.radix";
inline constexpr std::string_view kSwiGluClampName = "ggml.swiglu_clamp";
inline constexpr std::string_view kRopeExtName = "ggml.rope.ext";
inline constexpr std::string_view kGetRowsExtName = "ggml.get_rows.ext";
inline constexpr std::string_view kSetRowsExtName = "ggml.set_rows.ext";
inline constexpr std::string_view kLightningIndexerName = "ggml.lightning_indexer.wmma";
inline constexpr std::string_view kHcCombName = "ggml.dsv4_hc_comb";
inline constexpr std::string_view kHcPreName = "ggml.dsv4_hc_pre";
inline constexpr std::string_view kHcPostName = "ggml.dsv4_hc_post";
inline constexpr std::string_view kFlashAttnVec256Name = "ggml.flash_attn_ext.vec_d256";
inline constexpr std::string_view kFlashAttnMmaGqa2Name = "ggml.flash_attn_ext.mma_gqa2";
inline constexpr std::string_view kFlashAttnMmaName = "ggml.flash_attn_ext.mma";
inline constexpr std::string_view kFlashAttnMmaWideName = "jitllm.flash_attn_ext.mma_wide";
inline constexpr std::string_view kDsv4HcaTokentileName = "jitllm.dsv4.hca_tokentile";
inline constexpr std::string_view kSsmConvName = "ggml.ssm_conv";
inline constexpr std::string_view kGatedDeltaNetName = "ggml.gated_delta_net";
// jitLLM's own operations on GGML tensors (jitllm_ops.h).
inline constexpr std::string_view kMxfp8MulMatVecName = "jitllm.mxfp8.mul_mat_vec";
inline constexpr std::string_view kMxfp8DequantName = "jitllm.mxfp8.dequant";
inline constexpr std::string_view kNvfp4RowsName = "jitllm.nvfp4.get_rows";
inline constexpr std::string_view kQRowsName = "jitllm.qrows.get_rows";
inline constexpr std::string_view kHcCombineName = "jitllm.hc.combine";
inline constexpr std::string_view kHcNormName = "jitllm.hc.norm";
inline constexpr std::string_view kHcMixName = "jitllm.hc.mix";
inline constexpr std::string_view kMoeGluName = "jitllm.moe.glu";
inline constexpr std::string_view kMoeCombineName = "jitllm.moe.combine";
inline constexpr std::string_view kBf16Name = "jitllm.bf16";
inline constexpr std::string_view kGemmBf16Name = "jitllm.gemm.bf16";
inline constexpr std::string_view kGatedDeltaNetColumnsName = "jitllm.gated_delta_net.columns";
inline constexpr std::string_view kGatedDeltaNetLanesName = "jitllm.gated_delta_net.lanes";
inline constexpr std::string_view kMoeRouteName = "jitllm.moe.route";
inline constexpr std::string_view kMoeQuantizeName = "jitllm.moe.quantize";
inline constexpr std::string_view kMoeGemmName = "jitllm.moe.gemm.cutlass";
inline constexpr std::string_view kMoeGluQuantizeName = "jitllm.moe.glu_quantize";
inline constexpr std::string_view kMoeCombineSortedName = "jitllm.moe.combine_sorted";
inline constexpr std::string_view kMoeGemvName = "jitllm.moe.gemv";
inline constexpr std::string_view kGdnConvName = "jitllm.gdn.conv";
inline constexpr std::string_view kGdnNormGateName = "jitllm.gdn.norm_gate";
inline constexpr std::string_view kArgmaxName = "jitllm.argmax";
inline constexpr std::string_view kMxfp8QuantizeName = "jitllm.mxfp8.quantize";
inline constexpr std::string_view kMxfp8SwizzleName = "jitllm.mxfp8.swizzle";
inline constexpr std::string_view kMxfp8GemmName = "jitllm.mxfp8.gemm.cutlass";
inline constexpr std::string_view kHcPrepName = "jitllm.hc.prep";
inline constexpr std::string_view kHcLoName = "jitllm.hc.lo";
inline constexpr std::string_view kHcMixBf16Name = "jitllm.hc.mix_bf16";
inline constexpr std::string_view kMoeRouterName = "jitllm.moe.router";
inline constexpr std::string_view kGdnHistoryName = "jitllm.gdn.history";
inline constexpr std::string_view kGdnStepName = "jitllm.gdn.step";
inline constexpr std::string_view kGdnGatesName = "jitllm.gdn.gates";
inline constexpr std::string_view kQsaPrepName = "jitllm.qsa.prep";
inline constexpr std::string_view kQsaGateQuantizeName = "jitllm.qsa.gate_quantize";
inline constexpr std::string_view kQsaPoolName = "jitllm.qsa.pool";
inline constexpr std::string_view kQsaTopKName = "jitllm.qsa.topk";
inline constexpr std::string_view kQsaAttnName = "jitllm.qsa.attn";
// DeepSeek V4's fast plan (jitllm_ops.h).
inline constexpr std::string_view kQuantizeQ8Name = "jitllm.q8_1";
inline constexpr std::string_view kVecQName = "jitllm.vecq";
inline constexpr std::string_view kDsv4RouteName = "jitllm.dsv4.route";
inline constexpr std::string_view kDsv4CombineName = "jitllm.dsv4.combine";
inline constexpr std::string_view kDsv4HcMixName = "jitllm.dsv4.hc_mix";
inline constexpr std::string_view kDsv4HcPreName = "jitllm.dsv4.hc_pre";
inline constexpr std::string_view kDsv4CompressName = "jitllm.dsv4.compress";
inline constexpr std::string_view kDsv4LidTopKName = "jitllm.dsv4.lid_topk";
inline constexpr std::string_view kDsv4SparseMaskName = "jitllm.dsv4.sparse_mask";
// A speculative verify's row-invariant products (D-092; ops_ext.h).
inline constexpr std::string_view kMulMatVecQRows = "jitllm.mul_mat.mmvq_rows";
inline constexpr std::string_view kMulMatIdVecQRows = "jitllm.mul_mat_id.mmvq_rows";
inline constexpr std::string_view kMulMatVecFRows = "jitllm.mul_mat.mmvf_rows";
// The most columns (tokens) a row-invariant product takes (ops_ext.h
// kRowsMaxColumns).
inline constexpr std::int64_t kRowInvariantColumns = 8;

// Upstream's no-op nodes (ggml_cuda_is_view_or_noop).
bool LaunchesNothing(const ggml_tensor* node);

// The plan of `graph` (fusion.h's node order) with fusion on or off. The
// tensors `keep` names (PlaceActivations's) are read after the run: an
// implementation that leaves a tensor unwritten or overwrites it with
// another form (pair_glu, pair_glu_q8) is planned only where nothing but its
// consumer reads that tensor, in the graph or after it.
std::expected<GraphPlan, KernelFailure> PlanGraph(GraphNodes graph, bool fusion,
                                                  const DeviceChoices& device,
                                                  std::span<ggml_tensor* const> keep = {});

// Whether two plans run the same implementations over the same nodes.
bool SamePlan(const GraphPlan& a, const GraphPlan& b);

// Where each computed tensor of a plan lives in one region.
struct Placement {
  std::vector<std::pair<ggml_tensor*, std::uint64_t>> offsets;  // tensor, offset
  std::uint64_t extent = 0;  // the region's bytes the placement uses
};

// Places every computed node of `graph` that is no view, and each of
// `inputs` (leaves the region also holds), in one region at offsets
// aligned to `alignment`, each rounded up to it (ggml-alloc's rule). A
// tensor lives from the step that computes it (inputs from the start) to
// the last step reading it or a view of it; the graph's last node, or the
// tensor it views, lives to the end, as do the tensors `keep` names (or
// the tensors they view), which a caller reads after the run. Tensors whose
// lifetimes meet never share bytes, so a step's outputs share none with its
// inputs. Placed largest first, each at the lowest offset free for its
// whole lifetime.
std::expected<Placement, KernelFailure> PlaceActivations(GraphNodes graph, const GraphPlan& plan,
                                                         std::span<ggml_tensor* const> inputs,
                                                         std::uint64_t alignment,
                                                         std::span<ggml_tensor* const> keep = {});

// Points every view among `nodes` into its source's memory (the source's
// address plus the view's offset), once the sources are bound.
void BindViews(std::span<ggml_tensor* const> nodes);

// Gives every computed node that is no view a distinct address from
// `base` on (PlaceActivations's first pass), then binds the views.
void BindDistinct(GraphNodes graph, std::uint64_t base);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_GRAPH_PLAN_H_
