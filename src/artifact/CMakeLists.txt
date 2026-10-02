# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The v0 prepared-artifact reader (docs/artifact-format.md, D-056): a strict
# parser of the manifest and index, their validation as untrusted input, and
# the groups, chunks, placements and direct-read plans of a validated
# artifact. The model layer (docs/architecture.md#layers-and-dependency-rules):
# it needs only base and platform, and no vendor SDK.
add_library(jitllm_artifact STATIC artifact.cc composition.cc error.cc gguf_metadata.cc json.cc
                                   layout.cc representation.cc)
target_include_directories(jitllm_artifact PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_artifact PUBLIC jitllm_base PRIVATE jitllm_platform jitllm_warnings)
