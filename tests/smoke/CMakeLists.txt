# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The `llmp` binary as shipped, run where each build's tests run: natively,
# under qemu-user, or on a Spark over SSH (tools/run-target). Every test
# carries the `smoke` label.
#
# smoke.doctor runs `llmp doctor` on whatever host runs the tests and
# checks that it reports and that its exit status matches its summary; a
# host without the driver or a GPU this build targets reports problems,
# which is correct. In the CPU-only profile it must also pass, with no
# device provider. smoke.doctor.gpu (label `gpu`, so only on a Spark)
# requires no problems and a GB10 this build targets, with VMM (D-006). In
# a build that also targets discrete GPUs, smoke.doctor.discrete (labels
# `gpu` and `gpu-discrete`) requires the same of a discrete GPU 0 (D-082).
set(emulator "$<JOIN:$<TARGET_PROPERTY:cli_test,CROSSCOMPILING_EMULATOR>,|>")
if(LLMP_CUDA)
  set(mode report)
else()
  set(mode cpu)
endif()
add_test(NAME smoke.doctor
         COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:llmp> "-DEMULATOR=${emulator}" -DMODE=${mode}
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_doctor.cmake)
set_tests_properties(smoke.doctor PROPERTIES LABELS smoke)
# A closed standard output is a failed write (exit 1), not a silent
# success. Natively only: a remote or emulated run would not pass the closed
# descriptor through. `doctor` exits 0 otherwise only in the CPU-only
# profile, which is where its case has teeth.
if(NOT CMAKE_CROSSCOMPILING)
  foreach(command IN ITEMS --version doctor)
    string(REGEX REPLACE "^-+" "" name "${command}")
    add_test(NAME smoke.closed_stdout.${name}
             COMMAND sh -c "\"$0\" $1 >&-; test $? -eq 1" $<TARGET_FILE:llmp> ${command})
    set_tests_properties(smoke.closed_stdout.${name} PROPERTIES LABELS smoke)
  endforeach()
endif()
if(LLMP_CUDA)
  add_test(NAME smoke.doctor.gpu
           COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:llmp> "-DEMULATOR=${emulator}" -DMODE=gb10
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/check_doctor.cmake)
  set_tests_properties(smoke.doctor.gpu PROPERTIES LABELS "smoke;gpu")
endif()
if(LLMP_CUDA_DISCRETE_SASS)
  add_test(NAME smoke.doctor.discrete
           COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:llmp> "-DEMULATOR=${emulator}"
                   -DMODE=discrete -P ${CMAKE_CURRENT_SOURCE_DIR}/check_doctor.cmake)
  set_tests_properties(smoke.doctor.discrete PROPERTIES LABELS "smoke;gpu;gpu-discrete")
endif()
