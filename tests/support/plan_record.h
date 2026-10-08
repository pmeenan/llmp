// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What a test or benchmark records of an executed plan, and the JSON lines
// it writes them as: the input of
// docs/experiments/backend-proof-p2/plan_compare.py, which compares a
// native plan with the FP16 bridge's recorded one (backend-proof.md, Tier
// E). Test and benchmark support only; no production target links it
// (tests/support/CMakeLists.txt checks).

#ifndef LLMP_TESTS_SUPPORT_PLAN_RECORD_H_
#define LLMP_TESTS_SUPPORT_PLAN_RECORD_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llmp::test_support {

enum class EventKind : std::uint8_t { kKernel, kCopy, kMemset, kCublas };

// One kernel launch, copy, memset or cuBLAS call made while recording.
struct Event {
  EventKind kind = EventKind::kKernel;
  // A kernel's mangled name, or the cuBLAS entry point called.
  std::string name;
  // How a kernel was launched (cudaLaunchKernel, cudaLaunchKernelExC with
  // at most the PDL attribute, cudaLaunchKernelExC+attributes), or
  // a copy's kind (HtoD, DtoH, DtoD, HtoH, default, or driver when the
  // driver API was not told).
  std::string api;
  std::array<unsigned, 3> grid{};
  std::array<unsigned, 3> block{};
  std::size_t shared = 0;  // a kernel's dynamic shared memory
  // The kernel's own attributes, as the runtime reports them.
  int registers = 0;
  std::size_t static_shared = 0;
  std::size_t local = 0;
  std::size_t bytes = 0;  // a copy's or memset's
  int value = 0;          // a memset's
  // A cuBLAS call's m, n, k and batch count (0 when not batched).
  std::array<std::int64_t, 4> shape{};
  const void* stream = nullptr;
};

// Where a chunk of a trajectory begins: its evaluation (1 fresh, 2 repeat,
// 3 restore), index, rows and starting position.
struct Chunk {
  int evaluation = 1;
  int chunk = 0;
  std::int64_t rows = 0;
  std::int64_t n_past = 0;
};

// NVCC names a file's internal-linkage symbols with a per-file token,
// _INTERNAL_<hash>_<length>_<file>_<hash>, whose hashes differ between
// builds; the FP16 gate compares names with both hashes masked.
std::string NormalizedKernelName(std::string_view name);

// The recording's lines, each a JSON object and a newline. A recording is
// a header, then events, with each chunk's between ChunkLine and
// EndChunkLine. `loaded` names each loaded cuBLAS library with its path.
std::string HeaderLine(std::string_view source,
                       const std::vector<std::pair<std::string, std::string>>& loaded);
std::string ChunkLine(const Chunk& chunk);
std::string EndChunkLine();
// Where one operation of a plan begins, inside a chunk: the operation's
// name in the plan's record and its layer (-1 outside the layers). The
// EXL3 plan comparator (docs/experiments/backend-proof-p3/op_plan_compare.py)
// holds each operation's events to the record's.
std::string OpLine(std::string_view name, int layer);
std::string EventLine(const Event& event);

// plan_record_sample.txt: the lines these write for the first seven
// launches of the bridge's decode step (llmp_plan_record_sample).
std::string_view PlanRecordSample();

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_PLAN_RECORD_H_
