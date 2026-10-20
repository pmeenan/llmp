// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A plan key holds each segment's past only through its components'
// shape_parameters words (execution/step_shape.h PlanKey: `past` is data).
// EmitAtPasts emits one component instance at each of `pasts` its shape's
// read cells allow and checks that two pasts with the same words emit the
// same graph and step inputs: a component whose emission depends on the
// position must declare it in its words.

#ifndef LLMP_TESTS_UNIT_PAST_DEPENDENCE_H_
#define LLMP_TESTS_UNIT_PAST_DEPENDENCE_H_

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "components/component.h"
#include "components/step.h"
#include "format/tensor_format.h"
#include "model/shape_parameters.h"
#include "planner/step_shape.h"

namespace llmp::components::test_past {

// The instance's words for `shape`, each segment's in turn, as the
// component step source appends them to the plan key.
inline std::vector<std::uint64_t> Words(const Instance& in, const planner::StepShape& shape) {
  std::vector<std::uint64_t> out;
  if (in.def->shape_parameters == nullptr) return out;
  for (const planner::Segment& s : shape.segments) {
    model::ShapeParameterSink count;
    in.def->shape_parameters(in, s.rows, s.past, shape.read_align, count);
    const auto n = count.Finish();
    EXPECT_TRUE(n.has_value());
    if (!n) return out;
    std::vector<std::uint64_t> segment(*n);
    model::ShapeParameterSink fill(segment);
    in.def->shape_parameters(in, s.rows, s.past, shape.read_align, fill);
    EXPECT_TRUE(fill.Finish().has_value());
    out.insert(out.end(), segment.begin(), segment.end());
  }
  return out;
}

// The graph's digest and what the lowering stages for each input.
inline base::Sha256Digest Emitted(const ComponentGraph& g) {
  base::Sha256 h;
  const base::Sha256Digest graph = g.step.graph.Digest();
  h.Update(std::as_bytes(std::span(graph)));
  for (const StepInput& input : g.step.inputs) {
    h.Update(std::format("{} {} {} {} {} {} {} {} {} {}\n", static_cast<int>(input.kind),
                         input.segment, input.read, input.sections, input.component, input.layer,
                         input.ratio, input.overlap, input.ring, input.visible_capacity));
  }
  return h.Finish();
}

struct PastResult {
  std::size_t built = 0;     // pasts that emitted
  std::size_t compared = 0;  // emissions compared with an earlier one of the same words
  std::size_t distinct = 0;  // distinct emissions
};

// Every segment of `base` at each of `pasts` (those its read cells allow;
// an emission the instance refuses is skipped).
inline PastResult EmitAtPasts(const Instance& in, const planner::StepShape& base,
                              std::span<const format::TensorFormat> formats,
                              std::span<const std::uint64_t> cells, const ComponentInputs& inputs,
                              const std::set<std::uint32_t>& pasts) {
  PastResult r;
  std::map<std::vector<std::uint64_t>, std::pair<std::uint32_t, base::Sha256Digest>> seen;
  std::set<base::Sha256Digest> distinct;
  for (const std::uint32_t past : pasts) {
    planner::StepShape shape = base;
    bool fits = true;
    for (planner::Segment& s : shape.segments) {
      s.past = past;
      fits = fits && std::uint64_t{past} + s.rows <= s.read_cells;
    }
    if (!fits) continue;
    auto graph = BuildComponent(in, shape, formats, cells, inputs);
    if (!graph) continue;
    ++r.built;
    const auto digest = Emitted(*graph);
    distinct.insert(digest);
    const auto [at, first] = seen.try_emplace(Words(in, shape), past, digest);
    if (!first) {
      ++r.compared;
      EXPECT_EQ(at->second.second, digest)
          << "emission at past " << past << " differs from past " << at->second.first
          << " under one plan key: declare the dependence in shape_parameters";
    }
  }
  r.distinct = distinct.size();
  return r;
}

}  // namespace llmp::components::test_past

#endif  // LLMP_TESTS_UNIT_PAST_DEPENDENCE_H_
