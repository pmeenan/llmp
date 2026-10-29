// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The active swap/spec benchmarks' FP16 comparison through the shared component
// engine. Historical BP-P executors and their raw evidence remain separate.
#ifndef LLMP_BENCHMARKS_FP16_RUNNER_H_
#define LLMP_BENCHMARKS_FP16_RUNNER_H_

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "fp16_common.h"
#include "paged_node.h"

namespace llmp::test_support {
class Recording;
}
namespace llmp::benchmarks {

struct Fp16Options {
  std::filesystem::path artifact;
  std::string trajectory;
  std::filesystem::path tokens;
  bool fusion = true;
  std::filesystem::path out;
  // Fixed paid adapter inventory, including retained caller results.
  std::uint64_t host_allowance = 12ULL << 30U;
};

class Fp16Runner final : public test_support::PagedModel {
 public:
  using Status = test_support::Status;
  Fp16Runner(test_support::PagedNode& node, const Fp16Options& options, int owner,
             std::uint32_t stream, test_support::Recording* recording, std::string& record);
  ~Fp16Runner() override;
  Status Setup();
  Status Register();
  Status Bind();
  Status Evaluate(int evaluation, std::vector<float>& result);
  Status RunAlone();
  Status Write(const std::vector<std::vector<float>>& results);
  std::uint64_t activations_needed() const;
  std::uint64_t pool_needed() const;
  std::chrono::steady_clock::time_point first_chunk_done() const;
  const catalog::Closure& everything() const;
  std::vector<catalog::ExtentId> weights() const;
  std::uint64_t weight_read_bytes() const;
  const Trajectory& trajectory() const;
  std::uint32_t stream() const override;
  const catalog::Closure& fence_closure() const override;
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace llmp::benchmarks
#endif  // LLMP_BENCHMARKS_FP16_RUNNER_H_
