// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#include <iostream>

#include "gemma26_prefill_coarse.h"
int main() { return jitllm::benchmark::coarse::ClockProbe(std::cout) ? 0 : 1; }
