# syntax=docker/dockerfile:1.26.0@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32
# check=skip=FromPlatformFlagConstDisallowed
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Llmpalooza reference container (D-012, D-049, D-061, D-070): Ubuntu 24.04 by
# digest, the declared host prerequisites from a dated Ubuntu snapshot, mise
# by hash and mise's pinned Python. The SDK is not baked in: `mise run setup`
# inside the container provisions it with the same tools/setup-toolchain a
# workstation uses, into the volumes mounted at the SDK home and cache.
FROM --platform=linux/amd64 docker.io/library/ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3

ARG UBUNTU_SNAPSHOT=20260923T000000Z
ARG MISE_VERSION=2026.9.12
ARG MISE_SHA256=b4058dece685259910d3aba5782445996eea79dbdb3cf952a6eb81aadf0373ff

# The base image has no CA bundle and the snapshot service is HTTPS-only, so
# ca-certificates and its dependencies come from the live archive.
# Every other package comes from the snapshot, which stays apt's default in the
# image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && echo "APT::Snapshot \"${UBUNTU_SNAPSHOT}\";" > /etc/apt/apt.conf.d/50llmp-snapshot \
 && rm -rf /var/lib/apt/lists/*

# The declared prerequisites (version constraints stripped; the snapshot
# decides the versions and tools/check-toolchain checks them), plus git.
COPY toolchains/prerequisites/ubuntu-24.04-x86_64.txt /tmp/prerequisites.txt
RUN apt-get update \
 && sed -e 's/#.*//' -e 's/[[:space:]]*[>=].*//' /tmp/prerequisites.txt \
    | xargs apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* /tmp/prerequisites.txt

ADD --checksum=sha256:${MISE_SHA256} \
    https://github.com/jdx/mise/releases/download/v${MISE_VERSION}/mise-v${MISE_VERSION}-linux-x64.tar.gz \
    /tmp/mise.tar.gz
RUN tar -xzf /tmp/mise.tar.gz -C /tmp \
 && install -m 0755 /tmp/mise/bin/mise /usr/local/bin/mise \
 && rm -rf /tmp/mise /tmp/mise.tar.gz

# The check gate mounts a fresh volume here for its copy of the checkout
# (tools/check); a new named volume inherits this ownership.
RUN install -d -o ubuntu -g ubuntu /workspaces

# The image's existing uid 1000 user. The SDK home and cache exist here so
# fresh named volumes mounted on them inherit this ownership.
USER ubuntu
WORKDIR /home/ubuntu
RUN mkdir -p .local/share/llmp .cache/llmp
ENV MISE_TRUSTED_CONFIG_PATHS=/workspaces

# mise's pinned Python, installed from mise.lock so offline runs need no download.
COPY --chown=ubuntu:ubuntu mise.toml mise.lock /home/ubuntu/.llmp-mise/
RUN cd /home/ubuntu/.llmp-mise && MISE_TRUSTED_CONFIG_PATHS=$PWD mise install --locked \
 && rm -rf /home/ubuntu/.llmp-mise
ENV PATH=/home/ubuntu/.local/share/mise/shims:$PATH
