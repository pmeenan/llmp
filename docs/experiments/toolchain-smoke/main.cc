// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <bit>
#include <cstdio>
#include <span>
#include <utility>
static_assert(__cplusplus >= 202302L);
static_assert(std::byteswap(0x01020304u) == 0x04030201u);
enum class Answer { value = 42 };
static_assert(std::to_underlying(Answer::value) == 42);
#ifdef WITH_CUDA
extern "C" int cuda_smoke();
#endif
int main() {
  std::vector values{1, 2, 3, 4};
  int sum = 0;
  for (int value : std::span{values}) sum += value;
  if (sum != 10) return 1;
  std::printf("CPU PASS: C++23, pointer_bits=%zu, libstdc++=%d\n", sizeof(void*) * 8, __GLIBCXX__);
#ifdef WITH_CUDA
  return cuda_smoke();
#else
  return 0;
#endif
}
