<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 short-screen input preparation

**CPU input freeze passed; no model or likelihood collection ran.** The
[proposal](PROPOSAL.md) remains prospective. Independent scoring-source review
passed; collector review and a separate committed GPU registration remain
required before collection.
The completed Paris control and its older negative evidence are unchanged.

The official pinned-image tokenizer produced 213,814 corpus-body IDs. Its
separate encode and direct calls agreed on every integer. The frozen
[ids.json](ids.json) is BOS 2 exactly once followed by the first 1,023 body IDs:
1,024 inputs, with positions 1–1,023 becoming the eventual teacher targets.
SHA-256: `389216b176c694cd5f43e3068b20d6fad97e09cbd956bcd6c2837d6b400858b2`.
The recipe, corpus provenance and metadata identities are unchanged from the
reviewed preparation proposal. No chat template applies to this corpus.

The first job, `m35-u3-gemma26-short-prep-r1`, hit the registered 120-second
container deadline after preserving both complete tokenizer outputs and source
snapshots. It remains **FAIL**, with no final input-freeze receipt. The helper
called `len(tokenizer)` once per body ID during range validation. The installed
method delegates to `_tokenizer.get_vocab_size(with_added_tokens=True)`; the
captured Python source does not establish the backend operation's complexity.

The reviewed follow-up cached that size once and retained the same tokenization
and checks. A counting fake over 213,814 IDs verified exactly one size query;
out-of-range IDs, duplicate BOS and disagreeing tokenizer outputs still refuse.
The separately authorized `m35-u3-gemma26-short-prep-r2` then completed within
the original limits. Its complete official tokenizer outputs equal r1's; no
tokens, bounds or source model changed. This is a preparation harness correction,
not a model-quality fix.

Both jobs used the installed shared supervisor with 180-second timeout,
30-second termination grace and the immutable reference image; each container
used `runc`, no network or GPU visibility/devices, two CPUs and 3 GiB. Only six
authenticated metadata files, corpus and helpers were mounted. Neither weights
nor a checkpoint directory was mounted; no model was constructed. Both exact
named containers were removed and their absence confirmed. Copied metadata,
corpus and helper pre/post hashes matched. The final supervisor returned zero;
Spark A's post-job check reported no running/waiting jobs or GPU processes.

The [freeze receipt](freeze-receipt.json) records both attempts, exact raw
evidence hashes and lifecycle. [source-hashes.json](source-hashes.json) records
37 actual installed source snapshots; the 15 previously pinned source hashes
also matched. Independent review confirmed teacher-token inclusion even outside
top-1, full-vocabulary normalization, chunk assembly, and the head's 262,144-token
vocabulary after softcap 30. It also traced runner selection. This source review
is not an execution or quality pass. Actual Paris GPU environment selected
`VLLM_USE_V2_MODEL_RUNNER=1`; the CPU preparation does not select a GPU runner.

Raw evidence is outside Git at `/tmp/llmp-gemma26-short-prep-evidence-r{1,2}`,
with supervisors at `/tmp/llmp-gemma26-short-prep-supervisor-r{1,2}` and Spark A
copies under `~/.local/share/llmp/m35-u3-quality/gemma26-short-prep-r{1,2}`.
