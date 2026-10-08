# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Writes OUTPUT, a C++ source whose PlanRecordSample() returns INPUT's
# bytes (plan_record_sample.txt), at build time (CMakeLists.txt).
#   cmake -DINPUT=plan_record_sample.txt -DOUTPUT=plan_record_sample.cc -P plan_record_sample.cmake
file(READ "${INPUT}" sample)
if(sample MATCHES "\\)sample\"")
  message(FATAL_ERROR "${INPUT} holds the raw string's delimiter")
endif()
file(CONFIGURE OUTPUT "${OUTPUT}" CONTENT [=[
// Generated from tests/support/plan_record_sample.txt (plan_record_sample.cmake).
#include <string_view>

#include "plan_record.h"

namespace llmp::test_support {
std::string_view PlanRecordSample() {
  return R"sample(@sample@)sample";
}
}  // namespace llmp::test_support
]=] @ONLY)
