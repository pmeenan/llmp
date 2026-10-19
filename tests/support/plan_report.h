// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_TEST_SUPPORT_PLAN_REPORT_H_
#define LLMP_TEST_SUPPORT_PLAN_REPORT_H_

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernels/common/step_program.h"

namespace llmp::test_support {
// Test consumers own their assembly storage. Production sinks consume framed
// fragments directly and never need to allocate the complete JSON record.
inline kernels::PlanRecordSink CollectPlanRecords(std::vector<std::string>& records) {
  return [&records, open = false, record = std::string{}](kernels::PlanReportEvent event,
                                                          std::string_view fragment) mutable {
    if (event == kernels::PlanReportEvent::kBegin) {
      EXPECT_FALSE(open);
      EXPECT_TRUE(fragment.empty());
      open = true;
      record.clear();
    } else if (event == kernels::PlanReportEvent::kFragment) {
      EXPECT_TRUE(open);
      EXPECT_EQ(fragment.find('\n'), std::string_view::npos);
      record.append(fragment);
    } else {
      EXPECT_TRUE(open);
      EXPECT_TRUE(fragment.empty());
      open = false;
      records.push_back(std::move(record));
      record.clear();
    }
  };
}
}  // namespace llmp::test_support

#endif  // LLMP_TEST_SUPPORT_PLAN_REPORT_H_
