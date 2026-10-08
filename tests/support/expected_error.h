// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Reading the error of a std::expected in a test, safely (D-083).
//
// `result.error()` has a precondition: the result holds an error. On one
// that holds a value it reads the value's bytes as an error, so an
// expectation such as EXPECT_EQ(Plan(...).error(), PlanError::kTooBig) can
// pass although Plan succeeded, whenever those bytes happen to read as the
// error expected (an admitted plan's first bytes as an enum's zero value).
// With libstdc++'s assertions, which the test presets build with, it ends
// the whole test binary instead. These helpers check first, so a wrongly
// successful result fails the one expectation that names it:
//
//   EXPECT_EQ(Failed(catalog.Describe(id)), CatalogError::kUnknownId);
//   EXPECT_EQ(FailedCode(memory.Reserve(size)), ProviderError::kInvalid);
//   EXPECT_EQ(Failed(Resolve(registry, plan), &Unresolved::operation), 0U);
//
// A result a test goes on to inspect further can instead be guarded with
// ASSERT_FALSE(result.has_value()) before its first .error().

#ifndef LLMP_TESTS_SUPPORT_EXPECTED_ERROR_H_
#define LLMP_TESTS_SUPPORT_EXPECTED_ERROR_H_

#include <expected>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

namespace llmp::test_support {

// The error a result holds, or nothing when it holds a value.
template <typename T, typename E>
std::optional<E> Failed(const std::expected<T, E>& result) {
  if (result.has_value()) {
    return std::nullopt;
  }
  return result.error();
}

// A part of the error a result holds (`part` is a member pointer, or
// anything else std::invoke takes the error with), or nothing when it holds
// a value.
template <typename T, typename E, typename Part>
auto Failed(const std::expected<T, E>& result, Part part)
    -> std::optional<std::remove_cvref_t<std::invoke_result_t<Part&, const E&>>> {
  if (result.has_value()) {
    return std::nullopt;
  }
  return std::invoke(part, result.error());
}

// The code of the failure a result holds, for failures that carry it as
// their `error` member (ProviderFailure, KernelFailure, ProgramRejection
// and the like), or nothing when it holds a value.
template <typename T, typename E>
  requires requires(const E& failure) { failure.error; }
auto FailedCode(const std::expected<T, E>& result)
    -> std::optional<std::remove_cvref_t<decltype(std::declval<const E&>().error)>> {
  if (result.has_value()) {
    return std::nullopt;
  }
  return result.error().error;
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_EXPECTED_ERROR_H_
