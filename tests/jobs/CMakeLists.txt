# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The confined-job proof (D-074). Not a CTest test: it needs a delegated
# cgroup, which tools/job-proof sets up with a transient systemd unit, and
# its seccomp filter would stop qemu-user.
add_executable(job_proof_test job_proof_test.cc)
target_link_libraries(job_proof_test PRIVATE llmp_platform GTest::gmock_main llmp_warnings)
