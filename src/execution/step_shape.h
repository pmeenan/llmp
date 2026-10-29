// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What one step of a model computes, as the planner and the runner name it
// (docs/engine-components.md, "Execution"; D-107): its stage and phase, and
// per owner segment (a request slot's rows in the step) the rows it runs,
// the positions before them, the state cells it reads and the outputs it
// returns. The wave composer (engine/wave_composer.h) makes shapes from
// the slots' pending work; a step source turns a shape into a step program
// (kernels/common/step_program.h); a plan cache keys programs by PlanKey.

#ifndef LLMP_EXECUTION_STEP_SHAPE_H_
#define LLMP_EXECUTION_STEP_SHAPE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "base/sha256.h"

namespace llmp::execution {

// What a step is for: the capture rule, the row-invariance requirement and
// the decoding modes' bookkeeping follow it.
enum class Phase : std::uint8_t {
  kPrefill,  // prompt rows
  kDecode,   // one generated row per owner
  kVerify,   // a speculative verify: 1 + drafted rows per owner (D-092)
  kDraft,    // a drafter's pass
  kScore,    // literal completion likelihoods: every row's outputs
  kCommit,   // commit the accepted rows of a verify's saved intermediates
};

// The rows of one owner (a request slot) in a step.
struct Segment {
  std::uint8_t slot = 0;
  std::uint32_t rows = 0;
  std::uint32_t past = 0;        // positions before the rows (data; not in equality)
  std::uint32_t read_cells = 0;  // state cells the step reads: ReadCellsFor(past + rows)
  std::uint32_t outputs = 0;     // output rows (0: a state-only cut; rows: every row's)
  std::uint8_t group = 0;        // the joined group it belongs to (WaveComposer)
  bool want_head = true;         // false: the head is skipped (state-only)
  bool greedy = false;           // tokens chosen on the device instead of logits returned

  // Present only in a mixed causal target step normalized to kPrefill. It
  // retains each owner's real invocation role for keys and admission.
  std::optional<Phase> origin_phase{};

  bool operator==(const Segment& other) const {
    return slot == other.slot && rows == other.rows && read_cells == other.read_cells &&
           outputs == other.outputs && group == other.group && want_head == other.want_head &&
           greedy == other.greedy && origin_phase == other.origin_phase;
  }
};

struct StepShape {
  std::uint16_t stage = 0;  // the pipeline stage (0: the target; then each drafter)
  Phase phase = Phase::kPrefill;
  std::vector<Segment> segments;  // by slot, ascending
  // Every row computes what its one-row step computes (D-092): a verify's
  // plan, or a wave that must match its owners' lone steps bit for bit.
  bool row_invariant = false;
  // The alignment `read_cells` follow (ReadCellsFor: a multiple of it) and
  // the state is backed through (a positive multiple of 256; 2,048 for a
  // multi-slot wave whose key must repeat across steps).
  std::uint32_t read_align = 256;

  bool operator==(const StepShape&) const = default;
  std::uint32_t rows() const {
    std::uint32_t n = 0;
    for (const Segment& s : segments) n += s.rows;
    return n;
  }
  std::uint32_t outputs() const {
    std::uint32_t n = 0;
    for (const Segment& s : segments) n += s.outputs;
    return n;
  }
};

// The cells a step reads for state through `end` positions: its read class,
// which a plan key carries (Segment::read_cells), and the one rule every
// reader of a step's cells follows (the wave composer, an owner's lone read,
// the runner's row-invariant check). `end` rounded up to `align` (a positive
// multiple of 256), at most `context`. With `buckets` (0: none) the step it
// rounds to doubles each time `end` passes twice `buckets` steps, so a read
// is less than the larger of `align` and `end / buckets` cells past its end
// (masked, but read in full by a kernel that cannot skip masked cells, and
// backed: ModelRunner::Admit) and a context of C cells has about
// 2 * buckets + buckets * log2(C / (2 * buckets * align)) classes instead of
// C / align (36 at 262,144 cells for 4 buckets of 256). Every align * 2^k
// stays a class boundary, so with a power-of-two `align` a read never
// crosses a kernel's power-of-two cell limit before its end does.
constexpr std::uint32_t ReadCellsFor(std::uint64_t end, std::uint32_t align, std::uint32_t buckets,
                                     std::uint32_t context) {
  if (align == 0) return static_cast<std::uint32_t>(std::min<std::uint64_t>(end, context));
  std::uint64_t step = align;
  // end > 2 * buckets * step, without the product.
  if (buckets != 0)
    while (end > step && (end - 1) / step >= 2 * std::uint64_t{buckets}) step *= 2;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>((end + step - 1) / step * step, context));
}

// A step program's identity in a plan cache: the model spec, the shape, the
// pattern set and policy that planned it, and the partition it runs.
struct PlanKey {
  base::Sha256Digest spec{};
  StepShape shape;
  base::Sha256Digest patterns{};
  base::Sha256Digest policy{};
  base::Sha256Digest registry{};
  base::Sha256Digest backend{};
  bool capturable = false;
  std::uint32_t partition = 0;
  // Position-dependent graph dimensions supplied by the step source.
  std::vector<std::uint64_t> parameters{};
  bool operator==(const PlanKey&) const = default;
};

// A plan key's hash, over exactly what its equality compares (a segment's
// `past` is not): the plan cache's index.
inline std::size_t HashOf(const PlanKey& key) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  const auto mix = [&h](std::uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
  };
  const auto digest = [&mix](const base::Sha256Digest& d) {
    std::uint64_t word = 0;
    for (std::size_t i = 0; i < 8; ++i) word |= std::uint64_t{d[i]} << (8U * i);
    mix(word);
  };
  digest(key.spec);
  digest(key.patterns);
  digest(key.policy);
  digest(key.registry);
  digest(key.backend);
  mix(key.capturable);
  mix(key.partition);
  mix((std::uint64_t{key.shape.stage} << 32U) | (std::uint64_t{key.shape.row_invariant} << 16U) |
      static_cast<std::uint64_t>(key.shape.phase));
  mix(key.shape.read_align);
  for (const Segment& s : key.shape.segments) {
    mix((std::uint64_t{s.slot} << 56U) | (std::uint64_t{s.group} << 48U) |
        (std::uint64_t{s.want_head} << 40U) | (std::uint64_t{s.greedy} << 32U) | s.rows);
    mix((std::uint64_t{s.read_cells} << 32U) | s.outputs);
    if (s.origin_phase) mix(0x100U + static_cast<std::uint64_t>(*s.origin_phase));
  }
  for (const std::uint64_t p : key.parameters) mix(p);
  return static_cast<std::size_t>(h);
}

}  // namespace llmp::execution

template <>
struct std::hash<llmp::execution::PlanKey> {
  std::size_t operator()(const llmp::execution::PlanKey& key) const noexcept {
    return llmp::execution::HashOf(key);
  }
};

namespace llmp::planner {

using execution::Phase;
using execution::PlanKey;
using execution::Segment;
using execution::StepShape;

}  // namespace llmp::planner

#endif  // LLMP_EXECUTION_STEP_SHAPE_H_
