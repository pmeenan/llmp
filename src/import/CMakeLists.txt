# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The native importer (D-109; docs/engine-components.md#the-importer): so far
# its container readers (GGUF; safetensors with config.json; MLX; EXL3;
# diffusers pipelines as compositions of component checkpoints), which read a
# checkpoint as untrusted input (D-009) into tensor formats, extents and
# source metadata, the spec::Source an architecture mapping reads; owned
# normalization plans validate the selected mapping against the component catalog. The
# services layer (docs/architecture.md#layers-and-dependency-rules): format,
# spec, components, artifact, platform and base; never the engine or a kernel module.
add_library(llmp_import STATIC
            checkpoint.cc error.cc files.cc reader.cc transforms.cc route.cc indexed_parts.cc expert_relayout.cc parameter_folds.cc image_pipeline.cc stream.cc normalize.cc source_mapping.cc sampling_defaults.cc writer.cc archive_input.cc
            readers/diffusers.cc readers/exl3.cc readers/gguf.cc readers/hf_quant.cc
            readers/mlx.cc readers/safetensors.cc)
target_include_directories(llmp_import PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_import PUBLIC llmp_format llmp_spec llmp_artifact llmp_json llmp_base
                                  PRIVATE llmp_chat llmp_gguf llmp_components llmp_model llmp_platform llmp_warnings)
