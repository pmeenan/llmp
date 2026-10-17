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

#include <cstdint>
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
  std::uint32_t read_cells = 0;  // state cells the step reads: past + rows at the read alignment
  std::uint32_t outputs = 0;     // output rows (0: a state-only cut; rows: every row's)
  std::uint8_t group = 0;        // the joined group it belongs to (WaveComposer)
  bool want_head = true;         // false: the head is skipped (state-only)
  bool greedy = false;           // tokens chosen on the device instead of logits returned

  bool operator==(const Segment& other) const {
    return slot == other.slot && rows == other.rows && read_cells == other.read_cells &&
           outputs == other.outputs && group == other.group && want_head == other.want_head &&
           greedy == other.greedy;
  }
};

struct StepShape {
  std::uint16_t stage = 0;  // the pipeline stage (0: the target; then each drafter)
  Phase phase = Phase::kPrefill;
  std::vector<Segment> segments;  // by slot, ascending
  // Every row computes what its one-row step computes (D-092): a verify's
  // plan, or a wave that must match its owners' lone steps bit for bit.
  bool row_invariant = false;
  // The alignment `read_cells` follow and the state is backed through (a
  // positive multiple of 256; 2,048 for a multi-slot wave whose key must
  // repeat across steps).
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

// A step program's identity in a plan cache: the model spec, the shape, the
// pattern set and policy that planned it, and the partition it runs.
struct PlanKey {
  base::Sha256Digest spec{};
  StepShape shape;
  base::Sha256Digest patterns{};
  base::Sha256Digest policy{};
  std::uint32_t partition = 0;
  // Position-dependent graph dimensions supplied by the step source.
  std::vector<std::uint64_t> parameters{};
  bool operator==(const PlanKey&) const = default;
};

}  // namespace llmp::execution

namespace llmp::planner {

using execution::Phase;
using execution::PlanKey;
using execution::Segment;
using execution::StepShape;

}  // namespace llmp::planner

#endif  // LLMP_EXECUTION_STEP_SHAPE_H_
