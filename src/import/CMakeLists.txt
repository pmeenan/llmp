# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The native importer (D-109; docs/engine-components.md#the-importer): so far
# its container readers, GGUF and safetensors with config.json, which read a
# checkpoint as untrusted input (D-009) into tensor formats, extents and
# source metadata, the spec::Source an architecture mapping reads. The
# services layer (docs/architecture.md#layers-and-dependency-rules): format,
# spec, artifact, platform and base; never the engine or a kernel module.
add_library(llmp_import STATIC
            checkpoint.cc error.cc files.cc reader.cc
            readers/gguf.cc readers/hf_quant.cc readers/safetensors.cc)
target_include_directories(llmp_import PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_import PUBLIC llmp_format llmp_spec llmp_artifact llmp_json llmp_base
                                  PRIVATE llmp_gguf llmp_platform llmp_warnings)
