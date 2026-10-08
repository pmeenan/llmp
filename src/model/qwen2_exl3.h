// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen2 adapter for an EXL3 checkpoint and its native operation plan,
// as backend-proof P3 needs them (docs/backend-proof.md, "Native EXL3
// operation plan"; D-068's adapters). Model layer: no vendor or
// kernel-module types. kernels/exl3/qwen2.h runs what this plans.
//
// - BindQwen2Exl3 binds a v0 artifact's resources (the EXL3 importer's
//   names: model.layers.L.self_attn.q_proj.trellis, ...) to the tensors the
//   plan reads, checking every shape, rate, codebook and dtype.
// - PlanPhase gives one phase (a chunk of rows at a starting position) its
//   operations, in the order of the approved record
//   (docs/experiments/backend-proof-p0/exl3-op-plan.json and its P3
//   addendum), each with its owner, the implementation it names in the
//   registry, the tensors it reads and writes (the record's names and
//   dtypes), and for each linear the forced launch plan a Exl3LaunchTable
//   holds: the tile shape and grid decoded from the frozen tuning cache,
//   the GEMV configuration where EXL3-O takes it, or each reconstruction
//   slice's pinned cuBLASLt algorithm with the GEMM it was pinned for.
//   It refuses, before anything runs, a phase kind the record does not
//   hold and a linear case the table does not hold: nothing is chosen at
//   run time and nothing is substituted.
// - The plan places every tensor a phase computes, and the linears'
//   scratch, in one region by lifetime (the same slots in every layer),
//   so a phase's device bytes are known from the plan alone: the itemized
//   buffer plan the EXL3 phase memory limits are judged against.
//
// Upstream's choices this reproduces (ExLlamaV3 6b84a21b): the packed
// linears through 144 rows (exl3.py AUTO_RECONSTRUCT_THRESHOLD), the
// reconstruction from 145 rows and its fused form from 1,024
// (reconstruct_hgemm), gate and up through one multi-linear up to 32 rows
// (mlp.py), and reconstruction slices of at most 32,768 columns.

#ifndef LLMP_MODEL_QWEN2_EXL3_H_
#define LLMP_MODEL_QWEN2_EXL3_H_

#include <array>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "model/qwen2.h"

namespace llmp::artifact {
class Artifact;
}

namespace llmp::model {

// Qwen2.5-0.5B-Instruct as the EXL3 fixtures' config.json gives it (the
// same model as Qwen25Instruct05B, with RoPE's original context 32,768 as
// the record's rope parameters have it).
const Qwen2Profile& Qwen25Instruct05BExl3();

// One EXL3 linear's resources, as artifact resource indices, and its shape.
struct Exl3LinearBinding {
  std::string name;  // "model.layers.0.self_attn.q_proj", ..., "lm_head"
  std::uint32_t trellis = 0;
  std::uint32_t suh = 0;
  std::uint32_t svh = 0;
  std::optional<std::uint32_t> bias;  // F16 [n]
  int k = 0;
  int n = 0;
  int bits = 0;
};

struct Exl3LayerBinding {
  std::uint32_t attn_norm = 0;  // BF16 [width]
  std::uint32_t mlp_norm = 0;
  Exl3LinearBinding q, k, v, o, gate, up, down;
};

struct Exl3Binding {
  std::uint32_t embed = 0;       // BF16 [vocab, width]
  std::uint32_t final_norm = 0;  // BF16 [width]
  std::vector<Exl3LayerBinding> layers;
  Exl3LinearBinding lm_head;
};

// A resource as the EXL3 adapter sees it: its name and representation
// (artifact/representation.h): the family ("plain" or "exl3"), the dtype
// and shape, and for EXL3 its role, rate, codebook and features.
struct Exl3Resource {
  std::string name;
  std::string family;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::string role;
  std::uint32_t k_bits = 0;
  std::uint64_t in_features = 0;
  std::uint64_t out_features = 0;
  std::string codebook;
};

// Refused, naming the resource, unless the resources are a qwen2 EXL3
// checkpoint's with every tensor the plan reads, of the profile's shapes:
// mcg trellises at K = 4, 5, 6 or 8 with FP16 side vectors and an mcg flag,
// F16 biases on q, k and v only, BF16 norms and embedding; and nothing the
// plan does not read. Resource indices are positions in `resources`.
std::expected<Exl3Binding, std::string> BindQwen2Exl3(const Qwen2Profile& profile,
                                                      std::string_view architecture,
                                                      std::span<const Exl3Resource> resources);
// The same over a validated v0 artifact (artifact/artifact.h).
std::expected<Exl3Binding, std::string> BindQwen2Exl3(const Qwen2Profile& profile,
                                                      const artifact::Artifact& artifact);

// EXL3-G (GEMV off) or EXL3-O (upstream's default, GEMV on).
enum class Exl3Arm : std::uint8_t { kG, kO };

// How one linear runs at one row count, as the launch table forces it.
enum class Exl3Path : std::uint8_t { kGemm, kGemv, kMulti, kReconstruct, kReconstructFused };
std::string_view Exl3PathName(Exl3Path path);

// A reconstruction slice's pinned algorithm (exl3-recon-pin.json): the
// nine cuBLASLt attributes and the GEMM they were pinned for.
struct Exl3Pin {
  std::array<std::uint64_t, 9> config{};
  int m = 0;
  int k = 0;
  int n = 0;
  int ldc = 0;
  bool f32 = false;  // HSS, else HSH
};

struct Exl3LinearPlan {
  Exl3Path path = Exl3Path::kGemm;
  int shape = 0;              // GEMM or multi-GEMM tile shape
  int blocks = 0;             // the grid (GEMM, multi-GEMM, GEMV)
  int concurrency = 0;        // multi-GEMM
  int config = 0;             // GEMV configuration
  std::vector<Exl3Pin> pins;  // reconstruction slices, in order
};

// The forced plans of one fixture and arm, keyed by the linear (the gate
// linear for a multi-linear) and the rows; and the identity of the tuning
// data they came from, part of every plan's identity.
class Exl3LaunchTable {
 public:
  explicit Exl3LaunchTable(std::string source) : source_(std::move(source)) {}
  // Refused if the key is already present.
  std::expected<void, std::string> Add(std::string linear, int rows, Exl3LinearPlan plan,
                                       std::string second = {});
  const Exl3LinearPlan* Find(std::string_view linear, int rows) const;
  // The multi-linear's second (up) linear, for its gate linear.
  std::string_view Second(std::string_view linear, int rows) const;
  const std::string& source() const { return source_; }

 private:
  std::string source_;
  std::map<std::pair<std::string, int>, std::pair<Exl3LinearPlan, std::string>, std::less<>> plans_;
};

// A phase: `rows` rows at positions [past, past + rows) of one sequence.
struct Exl3Phase {
  int rows = 0;
  int past = 0;
};
// The attended cells padded to 256: every phase attends K and V over them.
int PaddedCells(const Exl3Phase& phase);
// Whether the record holds the phase's kind: a prefill from position 0 of
// 32, 144, 145, 1,023 or 1,024 rows, or a single-token step whose padded K
// length is 256, 1,024 or 1,280 (the P3 addendum added 1,024: the first
// step after the 1,023-row prefix).
bool RecordedPhase(const Exl3Phase& phase);

enum class Exl3Dtype : std::uint8_t { kF32, kF16, kBF16, kI32 };
std::string_view Exl3DtypeName(Exl3Dtype dtype);
std::uint64_t Exl3DtypeBytes(Exl3Dtype dtype);

// A tensor between operations: the record's name (layer-local inside a
// layer), dtype and row-major shape.
struct Exl3Tensor {
  std::string name;
  Exl3Dtype dtype = Exl3Dtype::kF32;
  std::vector<std::int64_t> shape;
  std::uint64_t bytes() const;
};

enum class Exl3Owner : std::uint8_t { kHost, kGgml, kExl3, kCopy };

// One operation of a phase. `implementation` names its registry entry
// (empty for host uploads and device copies, which launch no kernel).
// The attention operation stands for the record's attention.kv_max (from
// 1,024 rows), attention and attention.combine, which one launcher runs.
struct Exl3Op {
  std::string name;
  int layer = -1;  // -1 outside the layers
  Exl3Owner owner = Exl3Owner::kGgml;
  std::string implementation;
  std::vector<std::string> inputs;  // tensor names; weights by their record names
  std::vector<std::string> outputs;
  // Linears: the linear (the gate linear for gate_up), its plan, and the
  // scratch it writes (a_had, xh, w), in the region.
  std::string linear;
  std::string second;  // gate_up's up linear
  Exl3LinearPlan launch;
};

// Where a tensor or a linear's scratch lives in the phase's region.
struct Exl3Slot {
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct Exl3PhasePlan {
  Exl3Phase phase;
  int padded = 0;  // PaddedCells(phase)
  Exl3Arm arm = Exl3Arm::kG;
  std::vector<Exl3Op> ops;  // embedding, every layer's, then the output
  // Every activation tensor (and each linear's scratch, as
  // "<op>.a_had", "<op>.xh", "<op>.w") by name, in one region; layer
  // tensors take the same slots in every layer.
  std::map<std::string, Exl3Tensor, std::less<>> tensors;
  std::map<std::string, Exl3Slot, std::less<>> slots;
  std::uint64_t region = 0;  // bytes the placement spans
  // SHA-256 over the plan's launch data: the phase, arm, every operation
  // with its implementation name, tensors and forced launch plan, and the
  // launch table's source. The executor combines it with the registry
  // plan's identity (implementation identities) into the plan identity.
  base::Sha256Digest launch_digest{};
};

// The linear scratch the plan places, as the EXL3 launchers size it
// (kernels/exl3/validate.h ScratchBytes and ReconstructScratchBytes): the
// packed GEMM's and GEMV's transformed input (rows × k F16), the
// multi-GEMM's (2 × rows × k), the reconstruction's transformed input
// (rows × k, not for the fused path) and its weight slice (k × min(n,
// 32,768) F16).
inline constexpr int kExl3SliceColumns = 32768;
std::vector<int> Exl3Slices(int n);

std::expected<Exl3PhasePlan, std::string> PlanPhase(const Qwen2Profile& profile,
                                                    const Exl3Binding& binding,
                                                    const Exl3LaunchTable& table, Exl3Arm arm,
                                                    const Exl3Phase& phase);

}  // namespace llmp::model

#endif  // LLMP_MODEL_QWEN2_EXL3_H_
