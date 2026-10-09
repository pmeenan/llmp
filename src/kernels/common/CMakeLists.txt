# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The vendor-free step interface every kernel source lowers to (D-107;
# docs/engine-components.md, "Execution"): step programs, their launch and
# recording (step_program.h), and the launch over the device runtime's
# recorded work (recording_launch.h). Every profile. It calls the device
# runtime but links no device backend: a program links one (the CUDA
# runtime, or the fake), as for the engine core.
add_library(llmp_step_common STATIC recording_launch.cc)
target_include_directories(llmp_step_common PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_step_common PUBLIC llmp_ir llmp_format llmp_device_memory
                                       PRIVATE llmp_warnings)
