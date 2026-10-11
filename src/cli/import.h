// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_CLI_IMPORT_H_
#define LLMP_CLI_IMPORT_H_
#include <cstdio>
#include <span>
#include <string_view>
namespace llmp::cli {
int Import(std::span<const std::string_view> args, std::FILE* out, std::FILE* err);
}  // namespace llmp::cli
#endif  // LLMP_CLI_IMPORT_H_
