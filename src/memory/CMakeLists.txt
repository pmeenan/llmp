# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The memory manager's policy (D-050, D-055): the commitment ledger, the
# victim-selection baseline, materialization planning and the reclaim order.
# Resource core: no vendor types.
add_library(jitllm_memory STATIC commitment.cc materialize.cc reclaim.cc victims.cc)
target_include_directories(jitllm_memory PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_memory PUBLIC jitllm_base jitllm_catalog PRIVATE jitllm_warnings)
