# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The vendor-free step interface every kernel source lowers to (D-107;
# docs/engine-components.md, "Execution"): step programs, their launch and
# recording (step_program.h), what a kernel source gives the engine to plan
# and lower steps (lower.h), the launch over the device runtime's recorded
# work (recording_launch.h), the kernel sources' operations and their
# vendor-free view of a backend's tensors (selected_operation.h,
# tensor_ref.h), the composite of several sources (composite_source.h) and
# the registered kernel set a program's backends come from
# (device_backend.h). Every profile. It names no kernel source. It calls the device
# runtime but links no device backend: a program links one (the CUDA
# runtime, or the fake), as for the engine core.
add_library(llmp_step_common STATIC recording_launch.cc grouped_worklist.cc execution_resources.cc
                                     composite_source.cc device_backend.cc)
target_include_directories(llmp_step_common PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_step_common PUBLIC llmp_ir llmp_format llmp_device_memory llmp_planner
                                              llmp_components
                                       PRIVATE llmp_warnings)

# The fake kernel set (fake_device_backend.h) for host programs and tests,
# in every profile: a program registers it instead of the CUDA build's
# (kernels/compose), never both.
add_library(llmp_fake_kernel_set STATIC fake_device_backend.cc)
target_link_libraries(llmp_fake_kernel_set PUBLIC llmp_step_common PRIVATE llmp_warnings)
