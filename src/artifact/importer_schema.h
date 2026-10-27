// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// v1's `importer` member (D-113), as an artifact's manifest and a
// composition's document both record it: what made the document, so that
// the same sources, the same importer and the same options give the same
// bytes and ID. Internal to src/artifact.

#ifndef LLMP_ARTIFACT_IMPORTER_SCHEMA_H_
#define LLMP_ARTIFACT_IMPORTER_SCHEMA_H_

#include <cstdint>
#include <expected>
#include <string>

#include "artifact/artifact.h"
#include "artifact/error.h"
#include "artifact/json.h"
#include "artifact/schema.h"

namespace llmp::artifact::schema {

// Its name, version, build and options digest; with `mappings` (an
// artifact's, not a composition's), also the optional pair naming the
// architecture mapping the importer ran, both or neither.
inline std::expected<Converter, Error> ReadImporter(json::Value importer, bool mappings) {
  if (auto ok = mappings ? Object(importer, {"name", "version", "build", "options_sha256"},
                                  {"mapping_architecture", "mapping_version"}, "importer keys")
                         : Object(importer, {"name", "version", "build", "options_sha256"}, {},
                                  "importer keys");
      !ok) {
    return std::unexpected(ok.error());
  }
  auto name = Str(Get(importer, "name"), IsText, "importer.name");
  if (!name) {
    return std::unexpected(name.error());
  }
  auto version = Str(Get(importer, "version"), IsText, "importer.version");
  if (!version) {
    return std::unexpected(version.error());
  }
  auto build = Str(Get(importer, "build"), IsText, "importer.build");
  if (!build) {
    return std::unexpected(build.error());
  }
  auto options = Str(Get(importer, "options_sha256"), IsHex, "importer.options_sha256");
  if (!options) {
    return std::unexpected(options.error());
  }
  Converter out{.name = std::string(*name),
                .version = std::string(*version),
                .build = std::string(*build),
                .options_sha256 = DigestFromHex(*options).value_or(Digest{})};
  if (!mappings) {
    return out;
  }
  const auto map_arch = importer.find("mapping_architecture");
  const auto map_ver = importer.find("mapping_version");
  if (map_arch.has_value() != map_ver.has_value()) {
    return Fail(Rule::kSchema, "importer keys");
  }
  if (map_arch) {
    auto arch = Str(*map_arch, IsName, "importer.mapping_architecture");
    if (!arch) {
      return std::unexpected(arch.error());
    }
    auto ver = Int(*map_ver, 1, 65535, "importer.mapping_version");
    if (!ver) {
      return std::unexpected(ver.error());
    }
    out.mapping_architecture = std::string(*arch);
    out.mapping_version = static_cast<std::uint32_t>(*ver);
  }
  return out;
}

}  // namespace llmp::artifact::schema

#endif  // LLMP_ARTIFACT_IMPORTER_SCHEMA_H_
