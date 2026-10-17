# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The model layer's adapters (docs/architecture.md#layers-and-dependency-rules):
# state representations and their capabilities, a request's live state
# (state.h), state contracts and what derives from them (state_contract.h),
# draft progress (draft_progress.h) and model contexts composed of components
# (context.h), as D-068 needs them expressed; and Qwen-Image-2.1's profiles,
# bindings and host arithmetic (image_pipeline.h), which the image slice moves to
# stored spec data. LLM families have no code here: their specs are the
# artifact's (D-107, D-118).
# Model layer: no vendor or kernel-module types, and
# the resource core never includes it.
add_library(llmp_model STATIC context.cc draft_progress.cc image_pipeline.cc state.cc state_contract.cc)
target_include_directories(llmp_model PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_model PUBLIC llmp_base llmp_catalog llmp_format PRIVATE llmp_artifact llmp_warnings)
