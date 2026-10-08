<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 C8 attention geometry factor

The geometry factor makes **all 256 paid completed head vectors byte-exact** to the retained original reference. Both candidate repetitions have identical complete heads and initialized frontier/final states. All 264 rows are finite and have identical greedy choices to the reference. The eight prefill frontier rows remain nonexact, with maximum raw logit difference 0.78764248; none lies outside the unchanged pre-oracle C8 bound 2.291207938.

| Screen | Paid exact heads | Strict choice differences | Outside frozen bound |
| --- | ---: | ---: | ---: |
| Original C8 candidate | 0/256 | 12 across 264 rows | 3 |
| Whole-eight partition factor | 256/256 | 0 across 264 rows | 0 |

The original failed comparison stays failed. The original arithmetic factor used no scalar arms and made no reference reacquisition, calibration or allowance changes. The independent 256-target conditional score also passes: full-vocabulary FP64 mean NLL 11.3002489372 native versus 11.3020619410 reference, a relative conditional loss change −0.181136%. These targets cover the eight frontier rows and 248 paid rows. The last eight heads predict position 1024 and have no supplied target. This score is separate from the 1023-transition corpus gate.

The source change preserves physical four-root ownership while deriving stream-K partitions from the real eight-query cohort before the original 5% grid rounding. Positive even whole-eight grids split equally between the two quads. Odd grids retain the existing four-owner geometry. Only a complete one-query eight-owner wave with both quads preflighted and equal cache widths requests this mode; C4, tails and unequal widths keep their prior path. Shared eight-column products, shader arithmetic, both fixups and queue bodies are unchanged. Requested geometry is immutable plan metadata and is funded and validated in both planning passes.

This is bounded causal evidence that the partition geometry produced the paid-row disparity in this screen. It does not qualify other shapes, Gemma26, C12, long contexts or production joined defaults. The actual stock whole-model Q descriptor was not observed. The synthetic proof supplies an authenticated eight-stream operand/plan witness, rather than a claim about that stock descriptor.

Thirteen focused controls pass with zero skips: nine graph/metadata cases, three planning cases and one operator test. The operator test compares existing whole-eight MMA with two independent four-root calls for D256 and D512 at 256/1024 cache cells, in eager execution and graph capture/replay. All outputs are finite and byte-exact. Actual D256 whole/quad grids are 48/24 with 199,936 B scratch per quad; D512 grids 96/48 with 792,832 B. Invalid descriptors, odd-grid fallback, writer preflight, width/tail fallback and startup/placement funding are covered.

Reproduction uses the first eight checkpoint-qualified 1024-ID histories from the twelve-history carrier: prefix 992 at max_rows=256, followed by 32 forced positions 992..1023; max_head_rows=8, context=4096, F16 state, plain norm plus normRoPE/normADD, owner attention requested on the actual GB10. Graphs/fusion stay enabled. Each paid native wave shares eight-column products and executes two attention quads; the retained original uses a normal physical eight-sequence batch. Run the existing joined helper as `ARTIFACT NEW_DIR 31 8 joined norm CARRIER production`. The unchanged publication allowance is 293,715,968 B; neither allocation grants nor head caps were enlarged. Each arm shows 120 selected owner steps/requested cohort-eight steps, 121 plain norms, 120 normRoPE and 120 normADD steps, and 32 graph replays. These are selected-plan counts, not per-replay launch telemetry.

The measured frame is c3300ab plus the candidate 15-v2 and thirteen factor paths; [results.json](results.json) records all 22 source hashes, helper 4514, identical SDK receipt eb7, artifact/input/template pins, unchanged analyzer/judge, retained reference/freeze identities and official records. TensorFold primary was refreshed at this task entry: 609ca419, version 0.6.5; its Gemma26 entry remains MLX, with no comparable GB10 CUDA recipe. Raw logs and full vectors remain external.

Build3, native acquisition and assessment officially complete DONE0; all pre/post source checks and owned container retirement checks pass. The two earlier compile failures were test fixture issues and remain recorded. Full regression is deferred by the owner. Candidate paid timings 4.63433/4.40857 s are recorded without a performance claim. Serving owner/joined defaults remain off, and the other production candidate gates remain separate.

A fresh matched C8 timing bookend reuses the same measured helper 4514 and source22: stock → native → native → stock, with the same eight histories, caps and paid 32-wave recipe above. The mean native elapsed time is **3.57136% higher** (155.385 ms); native spread is 1.930 ms and stock spread 18.180 ms. This short screen establishes the current recipe's elapsed gap, without a parity or sustained-performance claim.

| Arm | Paid elapsed seconds |
| --- | ---: |
| Stock opening | 4.34177 |
| Native first | 4.50721 |
| Native repeat | 4.50528 |
| Stock closing | 4.35995 |

Both fresh native outputs match the retained complete 264 heads and all 16 initialized frontier/final state vectors plus layouts and inputs. Both fresh stock complete heads match the retained reference, authenticating its earlier finite admission. Actual selected-plan counts remain 120 owner steps and 120 requested cohort-eight steps; stock uses physical 8, F16 local 1280/global 4096 caches and normal fusion/graphs. Acquisition 6/6 and the separate retained-output identity check 3/3 officially finish DONE0, including owned container absence and pre/post source authentication. The check performs no numerical rescoring, noise recalibration or allowance change.

This timing measurement still uses immutable source22/private4514/SDKeb7, rather than claiming a model run of the later opt-in core landed at bc8fca6, which excludes six runtime/default-adoption paths and adds parent-storage validation. Native and public client17424 bytes remain unchanged; only output paths and the external metadata/identity checker differ. The timing and source records are added to results.json; raw logs and full outputs stay external. Serving owner/joined defaults and the other qualification gates remain unchanged.
