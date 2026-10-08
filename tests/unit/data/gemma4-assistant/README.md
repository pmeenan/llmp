<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Approved Gemma assistant tensor contracts

`assistant26.json` and `assistant31.json` contain the architecture scalars and
complete 49-tensor directories extracted from the pinned approved GGUF headers.
Each file records its exact URL, measured 16 MiB prefix hash, and the header
length/hash through the tensor directory. Vocabulary payloads and model weights
are omitted. These facts drive synthetic shape and zero-tensor kept-metadata
controls; actual paired vocabularies and complete prepared artifacts are tested
separately under the `models` label.

See [assistant binding](../../../../docs/gemma4-assistant.md) for complete source,
carrier, semantic contracts and qualification limits. The code and curated
contract fixtures are llmpalooza-authored Apache-2.0; no upstream kernel/graph code
is ported in this foundation. Model files remain external under their recorded
upstream terms.
