# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# External GGML image reference runner; not a llmpalooza build or SDK pin.
# arm64 manifest of nvidia/cuda:13.0.3-devel-ubuntu24.04 (driver stack is CUDA 13.0).
FROM nvidia/cuda@sha256:2396393dc8a031a6faf810894ec8e40c8d80e2a1177cfda0c2bf81acac9aa240
ARG SDCPP_COMMIT=c92d73c408515c94beef32161bb5960764fde7a0
ARG GGML_COMMIT=4bf5f6000653b7881d00963cd6ddb665ccd62a8d
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake git ninja-build python3 \
    && rm -rf /var/lib/apt/lists/* \
    && dpkg-query -W cmake git ninja-build python3 > /opt/build-packages.txt
RUN git clone --filter=blob:none https://github.com/leejet/stable-diffusion.cpp /src \
    && git -C /src checkout --detach "$SDCPP_COMMIT" \
    && git -C /src submodule update --init ggml \
    && test "$(git -C /src rev-parse HEAD)" = "$SDCPP_COMMIT" \
    && test "$(git -C /src/ggml rev-parse HEAD)" = "$GGML_COMMIT"
# PNG output only: WebP/WebM and the server frontend are not built.
RUN cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real \
      -DSD_WEBP=OFF -DSD_WEBM=OFF -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a \
    && cmake --build /build --target sd-cli -j 16 \
    && install -m 0755 /build/bin/sd-cli /usr/local/bin/sd-cli
ENTRYPOINT ["/usr/local/bin/sd-cli"]
