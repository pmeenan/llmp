# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# External reference runner: the pinned SGLang nightly plus torchcodec. The
# engine's MiMo-V2 input processor imports torchcodec; without it the processor
# silently fails to register and startup aborts even for text-only requests.
FROM lmsysorg/sglang@sha256:9e1fb4c395b9c406136e10aa445b8784d06bca3839623b52cbe4a3b231a157a8
COPY requirements-extra.txt /tmp/requirements-extra.txt
RUN pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements-extra.txt \
    && python3 -c "from torchcodec.decoders import AudioDecoder; import sglang.srt.multimodal.processors.mimo_v2"
