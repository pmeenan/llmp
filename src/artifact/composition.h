// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The v0 composition reader (D-089; docs/artifact-format.md,
// "Compositions"): a model of several components, such as an image
// pipeline's text encoder, denoiser and VAE, is one ordinary v0 artifact
// per component plus a composition, a small content-addressed document in
// the same store that names each component's artifact by ID and keeps the
// pipeline's own metadata (model_index.json, the scheduler's and the
// processor's files) verbatim. Nothing is copied between artifacts, so a
// component shared by two compositions is stored, verified and counted once.
//
// OpenComposition treats the directory as untrusted input and checks what
// the prototype's verify_composition checks
// (docs/experiments/artifact-layout/import_m3.py), in its order:
//
//   - the directory: a real directory holding manifest.json and meta/ only,
//     regular, singly linked files, no links;
//   - manifest.json (at most 1 MiB): strict JSON, format
//     "jitllm-composition" and format_version 0, canonical bytes, the
//     directory named by their SHA-256, every key and value typed and
//     patterned, lists in canonical order; 1-16 components, each a role
//     ([a-z][a-z0-9_]{0,63}), an artifact ID and that artifact's
//     architecture;
//   - no data/ directory; exactly the listed files present; every kept file
//     a recorded source and every source kept, model_index.json among them;
//   - each kept file read whole (64 MiB each, 128 MiB in all) and its
//     digest checked: the files are small, so the open is always deep.
//
// It does not open the components: each is an ordinary artifact the caller
// opens by ID (artifact.h, with OpenOptions::expected_id) and whose model
// architecture it compares with the one recorded here.

#ifndef LLMP_ARTIFACT_COMPOSITION_H_
#define LLMP_ARTIFACT_COMPOSITION_H_

#include <cstddef>
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
  const std::string& architecture() const { return architecture_; }
  const Converter& converter() const { return converter_; }
  const std::vector<CompositionComponent>& components() const { return components_; }
  const std::vector<Source>& sources() const { return sources_; }

  const CompositionComponent* Find(std::string_view role) const;
  // A kept file's verified bytes.
  std::optional<std::string_view> Metadata(std::string_view name) const;

 private:
  friend class CompositionOpener;
  std::string id_;
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
