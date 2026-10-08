# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
FROM ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
COPY requirements.txt /tmp/image-reference-requirements.txt
RUN python3 -m pip install --break-system-packages --no-deps -r /tmp/image-reference-requirements.txt \
    && python3 -m pip check
ENTRYPOINT ["python3"]
