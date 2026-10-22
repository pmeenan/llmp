# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The scheduler (D-048, D-050, D-069): admission and the execution slot with
# the switching policy; the task table and ready queue; the completion
# board; the files page sources read by identity (files.h); lanes (lane.h,
# a template); the storage, device and CPU lanes over the provider
# interfaces (services.h); and the scheduler thread's turn loop
# (scheduler.h). Resource core: no vendor types.
add_library(llmp_scheduler STATIC admission.cc cohort_policy.cc cohort_schedule.cc completions.cc files.cc pagein.cc scheduler.cc
                                    services.cc tasks.cc)
target_include_directories(llmp_scheduler PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_scheduler PUBLIC llmp_base llmp_catalog llmp_memory llmp_storage
                                              llmp_device_memory PRIVATE llmp_warnings)
