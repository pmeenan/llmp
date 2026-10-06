<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 Clear and state-growth attribution

One native C1 warm-and-paid process measured the existing runner phases and
new optional Clear seconds/calls. Its complete paid initialized state, final
head, tokens, history and layout match the fixed C1 own proof byte-for-byte;
warm tokens and history also match. The 262,144-value final head is finite.

| Wall measurement | Warm | Paid |
| --- | ---: | ---: |
| Whole cycle | 23.690795 s | 23.872444 s |
| Clear, one call | 0.749549 ms | 57.452285 ms |
| State growth phase | 162.931112 ms | 162.181846 ms |
| Required planning phase | 18.569570 ms | 0.285598 ms |
| Plan hits / misses | 158 / 2 | 160 / 0 |
| Execution phase | 23.336829 s | 23.543661 s |

The paid cycle has no plan misses. These measurements establish the magnitude
of Clear, state growth and required planning on this input; they do not locate
the cause of the reference gap. Execution and device-stream elapsed wall times
include host gaps. Bind, coverage and cache counters are nested, so counters
are not added or subtracted to infer a residual kernel cost. No provider,
allocation, storage or copy instrumentation was needed for this diagnostic.

The separate unchanged serving baseline measured native mean 23.927816492 s
and reference mean 23.19765 s (+3.147588%). That reference was not rerun here,
and this diagnostic is not a new matched speed comparison or parity gate.

The recipe is the private ordinary 31B serving bridge: C1, context 8192,
prefill 256, assistant off, invariant off, checked norm chains, 129 outputs and
128 decode waves. Seven runtime recipe files are borrowed unchanged from its
frozen source4; they are excluded from this unit's adoption. Only two native
runner files change: optional Clear seconds/calls are separate from the seven
existing phases, and refused Clear calls are timed too. The diagnostic enables
accounting before warm-up, resets it immediately before each unchanged cycle
start and reports after its completed end. Residency and timer boundaries are
unchanged. The benchmark and checker remain external with their identities
recorded in [results.json](results.json), avoiding a duplicate private recipe.

The isolated build on `spark` passed six steps, including the pre-context C4
refusal. On `spark-b`, acquisition and the separate complete-output checker
passed three steps each. All source, binary, build-receipt, library and fixed
input/proof guards passed; both nodes were idle after retirement. GGML remains
locked to official v0.6.0/d812; the installation SDK and native build receipt
are recorded separately. The source snapshot has no Git checkout metadata;
the explicit source manifest binds its main 997 base and private overlay.
Raw logs and payloads remain external. No reference run, full suite, provider
change, model-default adoption or broader quality qualification was performed.
