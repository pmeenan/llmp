#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -eu
printf '%s\n' STOCK_REFERENCE_LINKAGE_BEGIN
ldd /helper
ldd /app/libggml-cuda.so
sha256sum /app/libggml-cuda.so /app/libllama.so /app/libggml.so /app/libggml-base.so
ldd /app/libggml-cuda.so | awk '$1 ~ /^lib(cublas|cublasLt|cudart)\.so/ {print $3}' | while IFS= read -r library; do
    test -f "$library"
    sha256sum "$library"
done
printf '%s\n' STOCK_REFERENCE_LINKAGE_END
/helper "$@"
printf '%s\n' STOCK_REFERENCE_WRAPPER_COMPLETE
