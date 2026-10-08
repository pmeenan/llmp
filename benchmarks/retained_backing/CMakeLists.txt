# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The retained-backing comparison's deterministic replay (D-035, D-079;
# docs/experiments/retained-backing/): the candidate backing designs and the
# replay engine, on the fake device-memory provider. The designs are
# candidates for a measurement, not the memory manager; tests/unit's
# rb_replay_test checks their invariants.
add_library(llmp_retained_backing STATIC json.cc trace.cc designs.cc replay.cc)
target_include_directories(llmp_retained_backing PUBLIC "${PROJECT_SOURCE_DIR}/benchmarks")
target_link_libraries(llmp_retained_backing PUBLIC llmp_base llmp_fake_providers PRIVATE llmp_warnings)

add_executable(llmp_rb_replay main.cc)
target_link_libraries(llmp_rb_replay PRIVATE llmp_retained_backing llmp_warnings)
