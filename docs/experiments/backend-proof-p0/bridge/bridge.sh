#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds the FP16 toolchain bridge on a Spark:
#   bridge.sh SDK LLAMA_SOURCE CUDA_ROOT BUILD_DIR
# SDK is the aarch64 llmpalooza SDK; CUDA_ROOT is the SDK's CUDA 13.4 tree with
# the pinned cuBLAS packages unpacked into it. The compilers and flags are
# the ones cmake/toolchains/sdk.cmake sets for the spark-native profile:
# the SDK's Clang with its GCC 16.2 runtime (static), the host's GNU linker,
# and NVCC with that Clang as its host compiler, for sm_121 SASS only.
# Upstream's image options are kept (GGML_NATIVE=OFF, GGML_CUDA=ON, tests
# off, Release) except: static libraries rather than dynamically loaded
# backends (the SDK's C++ runtime is static only) and no OpenMP (the SDK has
# no OpenMP runtime). Both affect only the CPU backend.
set -eu
SDK=$1 SOURCE=$2 CUDA=$3 BUILD=$4
GCC=$SDK/gcc/aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/16
mkdir -p "$BUILD"
HOST=$BUILD/cuda-host-clang++
cat > "$HOST" <<WRAP
#!/bin/sh
link=yes
for arg in "\$@"; do case "\$arg" in -c|-E|-S|-M|-MM|-fsyntax-only) link= ;; esac; done
if [ -n "\$link" ]; then set -- -fuse-ld=bfd -static-libstdc++ -static-libgcc "\$@"; fi
exec "$SDK/bin/clang++" --target=aarch64-linux-gnu -march=armv8-a --gcc-install-dir=$GCC "\$@"
WRAP
chmod 755 "$HOST"
FLAGS="-march=armv8-a --gcc-install-dir=$GCC"
"$SDK/bin/cmake" -S "$(dirname "$0")" -B "$BUILD" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$SDK/bin/ninja" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$SDK/bin/clang" -DCMAKE_CXX_COMPILER="$SDK/bin/clang++" \
  -DCMAKE_C_COMPILER_TARGET=aarch64-linux-gnu -DCMAKE_CXX_COMPILER_TARGET=aarch64-linux-gnu \
  -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=bfd -static-libstdc++ -static-libgcc" \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$HOST" \
  -DCMAKE_CUDA_ARCHITECTURES=121-real -DCUDAToolkit_ROOT="$CUDA" \
  -DLLAMA_SOURCE="$SOURCE" -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
"$SDK/bin/cmake" --build "$BUILD" --target fp16_reference
