#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# Builds the probe on the Spark itself with its CUDA 13.0 toolkit and the
# system GCC as host compiler (a standalone probe, not the SDK build).
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: $0 OUTPUT_BINARY" >&2
  exit 2
fi
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
nvcc=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
"$nvcc" -std=c++20 -O3 -arch=sm_121 -Xcompiler -Wall,-Wextra \
  "$source_dir/dmabuf_probe.cu" -o "$1" -lcuda
