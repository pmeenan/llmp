# syntax=docker/dockerfile:1.26.0@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32
# check=skip=FromPlatformFlagConstDisallowed
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The arm64 install test's image (D-074): the same Ubuntu 24.04 digest as the
# reference container, as arm64 (always named: RE-015), with systemd, which
# the package's maintainer scripts need. systemd is installed but does not
# run: the test installs, inspects and removes the package without a
# service manager. It runs under qemu-user. Ubuntu's snapshot service has no
# ports archive (RE-016), so systemd comes from the live archive; the test
# prints the version it got.
FROM --platform=linux/arm64 docker.io/library/ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3

RUN apt-get update \
 && apt-get install -y --no-install-recommends systemd \
 && rm -rf /var/lib/apt/lists/*
