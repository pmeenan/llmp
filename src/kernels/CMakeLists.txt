# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Build-time implementations of operations, one directory per kernel source
# (D-053; docs/architecture.md#layers-and-dependency-rules), the engine's
# own paging kernels (paging/), and the CPU references implementations are
# checked against (reference/, D-108).
add_subdirectory(reference)
add_subdirectory(ggml)
add_subdirectory(exl3)
add_subdirectory(image)
add_subdirectory(paging)
