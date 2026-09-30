// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The versions of jitLLM's public surfaces (D-062). Each is an integer,
// independent of the product version and of every other surface's. A
// breaking change bumps it, with a CHANGELOG entry and a decisions.md entry.
// A reader accepts exactly the versions it implements and never guesses at
// another. The other surfaces D-062 lists join this file as they are
// implemented.

#ifndef JITLLM_BASE_SURFACE_VERSIONS_H_
#define JITLLM_BASE_SURFACE_VERSIONS_H_

#include <cstdint>

namespace jitllm::surface {

// The bounded inference HTTP subset (D-097, D-100), discovered through
// jitllm-inference-version. Additive OpenAI/vLLM fields stay in version 1.
inline constexpr std::uint32_t kInferenceVersion = 1;

// The representation inside the opaque signature of a reasoning block that
// jitLLM signs (D-047). The signature carries this version with jitLLM's
// identity; M5 fixes what version 1 holds. A runtime cannot verify a
// signature of a version it does not implement, and treats it as any
// signature it cannot verify (docs/client-api-baseline.md).
inline constexpr std::uint32_t kReasoningSignatureVersion = 1;

// The configuration documents' schema_version: the node's jitllm.toml and
// its fragments, and the shared cluster.toml (D-063, D-073,
// docs/cluster-design.md). Every file states it, and a reader accepts only
// this one.
inline constexpr std::int64_t kConfigSchemaVersion = 2;

}  // namespace jitllm::surface

#endif  // JITLLM_BASE_SURFACE_VERSIONS_H_
