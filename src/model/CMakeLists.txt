# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The model layer's adapters (docs/architecture.md#layers-and-dependency-rules):
# state representations and their capabilities, a request's live state
# (state.h) and model contexts composed of components (context.h), as D-068
# needs them expressed; and architecture adapters, so far Qwen2 dense
# (qwen2.h) for backend-proof P2 and its EXL3 checkpoints and operation plan
# (qwen2_exl3.h) for P3, and DeepSeek V4 Flash (dsv4.h) with its DSpark
# drafter (dspark.h), Qwen3.8 Flash Next (qwen38.h) and Qwen-Image-2.1
# (qwen_image.h) for M3, and Gemma 4 profiles, bindings and bounded inputs
# (gemma4.h) for M3.5, plus the Gemma 3 and Gemma 2 legacy tensor foundations (gemma3.h, gemma2.h).
# Model layer: no vendor or kernel-module types, and
# the resource core never includes it.
add_library(jitllm_model STATIC context.cc dspark.cc dsv4.cc gemma2.cc gemma3.cc gemma4.cc gemma4_assistant.cc qwen2.cc qwen2_exl3.cc qwen38.cc
                                qwen_image.cc state.cc)
target_include_directories(jitllm_model PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_model PUBLIC jitllm_base jitllm_catalog PRIVATE jitllm_artifact jitllm_warnings)
