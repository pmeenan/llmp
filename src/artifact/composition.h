// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The composition reader (D-089; docs/artifact-format.md,
// "Compositions"): a model of several components, such as an image
// pipeline's text encoder, denoiser and VAE, is one ordinary artifact per
// component plus a composition, a small content-addressed document in the
// same store that names each component's artifact by ID and keeps the
// pipeline's own metadata (model_index.json, the scheduler's and the
// processor's files) verbatim. Nothing is copied between artifacts, so a
// component shared by two compositions is stored, verified and counted once.
//
// OpenComposition treats the directory as untrusted input and checks what
// the prototype's verify_composition checks
// (docs/experiments/artifact-layout/import_m3.py), in its order:
//
//   - the directory (itself possibly reached through a link): holding
//     manifest.json, v1's spec.json and meta/ only, regular files, no links
//     inside it;
//   - manifest.json (at most 1 MiB): strict JSON, format
//     "jitllm-composition" and format_version 0, or "llmp-composition" and
//     format_version 1 or 2 (one scoped role level under meta/),
//     canonical bytes, the directory named by D-120's identity digest, every key and
//     value typed and patterned, lists in canonical order; 1-16 components,
//     each a role ([a-z][a-z0-9_]{0,63}), an artifact ID and that
//     artifact's architecture;
//   - no data/ directory; exactly the listed files present; every kept file
//     a recorded source and every source kept, model_index.json among them
//     for v0 (v1 keeps whatever the pipeline has, possibly nothing: a
//     target and its drafter are a composition with no files of its own);
//   - each kept file read whole (64 MiB each, 128 MiB in all) and its
//     digest checked: the files are small, so the open is always deep;
//   - v1's spec.json, when listed (D-113): read whole (kMaxSpecBytes), its
//     digest checked, strict canonical JSON whose root is an object. It is
//     the spec of the model the composition makes, whose stages' artifact
//     indexes count its components in their (role) order; spec/stored.h
//     parses and checks it.
//
// It does not open the components: each is an ordinary artifact, of
// either version, the caller opens by ID (artifact.h, with
// OpenOptions::expected_id) and whose model architecture it compares with
// the one recorded here.

#ifndef LLMP_ARTIFACT_COMPOSITION_H_
#define LLMP_ARTIFACT_COMPOSITION_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/error.h"

namespace llmp::artifact {

inline constexpr std::size_t kMaxComponents = 16;

struct CompositionComponent {
  std::string role;          // model_index.json's entry, such as "transformer"
  std::string artifact;      // the component artifact's ID
  std::string architecture;  // that artifact's model.architecture
};

struct KeptFile {
  std::string name;  // meta/<name>
  std::string bytes;
};

class Composition {
 public:
  const std::string& id() const { return id_; }
  // 0 or 1.
  std::uint32_t format_version() const { return format_version_; }
  const std::string& architecture() const { return architecture_; }
  // v0's converter, or v1's importer (artifact.h's Converter).
  const Converter& converter() const { return converter_; }
  const std::vector<CompositionComponent>& components() const { return components_; }
  const std::vector<Source>& sources() const { return sources_; }

  const CompositionComponent* Find(std::string_view role) const;
  // A kept file's verified bytes.
  std::optional<std::string_view> Metadata(std::string_view name) const;
  // v1: the spec.json bytes, checked as an artifact's, when it has one.
  std::optional<std::string_view> spec_json() const {
    return spec_json_ ? std::optional<std::string_view>(*spec_json_) : std::nullopt;
  }

 private:
  friend class CompositionOpener;
  std::string id_;
  std::uint32_t format_version_ = 0;
  std::optional<std::string> spec_json_;
  std::string architecture_;
  Converter converter_;
  std::vector<CompositionComponent> components_;
  std::vector<Source> sources_;
  std::vector<KeptFile> kept_;
};

// `expected_id` as in OpenOptions: by default the directory's name.
std::expected<Composition, Error> OpenComposition(
    const std::filesystem::path& root,
    const std::optional<std::string>& expected_id = std::nullopt);

}  // namespace llmp::artifact

#endif  // LLMP_ARTIFACT_COMPOSITION_H_
