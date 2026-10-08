// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <iostream>

#include "gemma26_prefill_coarse.h"
int main() { return llmp::benchmark::coarse::ClockProbe(std::cout) ? 0 : 1; }
