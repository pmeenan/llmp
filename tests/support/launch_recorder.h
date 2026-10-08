// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Records the kernel launches, copies, memsets and cuBLAS calls a thread
// makes, for tests and benchmarks that compare an executed plan with a
// recorded one (plan_record.h). Linking llmp_launch_recorder wraps the
// CUDA runtime's launch entry points (<<<>>>'s __cudaLaunchKernel,
// cudaLaunchKernelExC, and the C cudaLaunchKernel and
// cudaLaunchCooperativeKernel that llmpalooza's EXL3 launchers call), its
// copy and memset entry points, the driver's copy, and cuBLAS's and
// cuBLASLt's matrix products at link time (--wrap); each wrapper notes the
// call while a Recording is open on its thread, then makes it. A wrapped
// entry point called from inside another is recorded once, by the outer
// one. What cuBLAS launches inside itself is not seen here (an nsys trace
// of the run supplies it; plan_compare.py --nsys). Never linked into a
// production binary (tests/support/CMakeLists.txt checks).

#ifndef LLMP_TESTS_SUPPORT_LAUNCH_RECORDER_H_
#define LLMP_TESTS_SUPPORT_LAUNCH_RECORDER_H_

#include <string>
#include <utility>
#include <vector>

#include "plan_record.h"

namespace llmp::test_support {

// Records this thread's calls while it lives. One at a time per thread.
class Recording {
 public:
  Recording();
  ~Recording();
  Recording(const Recording&) = delete;
  Recording& operator=(const Recording&) = delete;
  Recording(Recording&&) = delete;
  Recording& operator=(Recording&&) = delete;

  // What was recorded since the last Take, in call order.
  std::vector<Event> Take();

 private:
  std::vector<Event> events_;
};

// The cuBLAS and cuBLASLt libraries this process loaded, each as its file
// name and path, for the recording's header.
std::vector<std::pair<std::string, std::string>> LoadedCublas();

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_LAUNCH_RECORDER_H_
