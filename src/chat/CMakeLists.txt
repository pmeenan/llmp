# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Chat templates (M3; D-067 as amended 2026-10-02): native family renderers
# for DeepSeek V4, Qwen3.8, Gemma 3 and 4, and Qwen-Image's prompt, chosen by
# template hash or by probe equivalence, with each model's stop tokens; the
# bounded Jinja-subset interpreter that renders any other template; and JSON
# written as Python's json.dumps writes it, as templates print tools and
# arguments.
# The model layer (docs/architecture.md#layers-and-dependency-rules), after
# the tokenizer.
add_library(jitllm_chat STATIC chat.cc deepseek_v4.cc gemma.cc jinja_eval.cc jinja_parse.cc
                               jinja_value.cc pycase.cc pyjson.cc qwen.cc template.cc)
target_include_directories(jitllm_chat PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_chat PUBLIC jitllm_json jitllm_tokenizer PRIVATE jitllm_base
                      jitllm_warnings)
