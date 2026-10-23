<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 scalar schedule transfer result

**PASS for the registered single scalar transfer:** all 1,070 teacher scores
are exactly equal between incumbent MMVQ (A) and repaired VecQ (D). Both
128-token greedy replies are also byte-identical. This establishes the tested
native arithmetic transfer, not general checkpoint quality, frozen-state parity,
assistant or ordinary multirow qualification, a speed gate, or U3/U4 closure.
No frozen expectation or numerical bound changed.

[PROTOCOL.md](PROTOCOL.md) was committed before collection as
`a777309880253b7bc4997956b461276d6e0b1658`. [results.json](results.json) records
exact metrics, bounds, texts, hashes and lifecycle receipts; [pins.json](pins.json)
retains complete inputs and source/runtime/dependency identities. Both arms ran
once on Spark A (`spark-c4e2`, driver580.178.04), with the same retained runtime
`5965e83cf2217ffcfe79fff98e7850699a0afd74b37776f853b6d6aa074d0767`,
module `1c5ae1c7650f88a9b9bd4867aa6a78a1e5821a357ddcdfcba4e9e65c96c59840`,
2903-file source manifest `d3b328506b77b6f00ada0f7c8557b78a577e4cb732672eae077650422fc1ba3e`,
and current target `090809981d2909085798273e665a0db58351a8e6029b13bc04ced857c9130943`.
Context8192, two slots, scalar prefill chunk1, F16 KV and no drafter were common.
A disabled the three registered VecQ selections; D used normal policy. Neither
arm added numerical authority, changed source or substituted weights.

| Registered criterion | Observed | Bound |
| --- | ---: | ---: |
| Aligned teacher targets | 1,070 | Exactly1,070 |
| Absolute PPL change | 0% | ≤0.5% |
| Mean absolute NLL difference | 0 nats | ≤0.05 |
| Maximum absolute NLL difference | 0 nats | ≤1.0 |
| Agreement within0.1 nats | 100% | ≥95% |

Both mean NLLs are9.574996392807655 and both perplexities are14400.185865506815.
These absolute values describe this fixed passage; equality does not establish
absolute model quality. All1,071 echoed token IDs matched the frozen sequence,
with a leading null score and1,070 finite nonpositive actual-teacher scores.
All1,071 scalar input rows completed, including the last row whose head has no
following teacher. Pairing used the unchanged historical bound file; its old
v0/v1 display label does not describe this current same-artifact comparison.

Independent full-plan reviews and root-issued, byte-bound A/D gates established
the actual contrast before D and before pairing. Five completed score plans per
arm account for1,071 uses. A selected incumbent MMVQ; D selected dense VecQ for
Q4_K/Q5_K/Q6_K and fused GLU for Q4_K/Q5_K, including K5376. The target has no
Q8_0; Q6_K is not a GLU gate/up encoding. The comparison established a3,420
operand-descriptor bijection and610 unaffected recipes, accounting for allowed
input preparation and GLU decomposition changes. Norm, Rope and attention
arithmetic/descriptors agree. Aggregate recipe counts alone were not the proof.

Both replies used26 input tokens and128 output tokens, finishing with `length`.
The hash-table response correctly introduces collisions and starts separate
chaining; the story begins a lighthouse keeper's discovery. Both are coherent
but truncated before completing the requested explanation/story. They are not
a complete semantic qualification. Their complete text is in results.json and
raw SSE remains external. Reply SHA256 values, in registered prompt order:

- `5318071dc3e76684c535cebabadfe431c70373aff025cb416cca17748a58b252`
- `fd56b9a14e2e890f4d313102419f96afd1f16b381a39fe58687e00e45e47b357`

## Preserved audit failures and read-only recovery

The original exclusive jobs `m35-u4-gemma31-quality-a-r1` and
`m35-u4-gemma31-quality-d-r1` remain **FAILED, rc1** (150s and145s). In each,
the collector completed with rc0 and the service retired by TERM with rc0;
the subsequent registered auditor failed at its physical-type spelling check.
GGML reports physical `q4_K/q5_K/q6_K`, whereas the auditor expected lowercase
names from semantic encoding metadata. This was a diagnostic schema assumption,
not a numerical failure. The original executors, tracebacks, supervisors,
raw scores and full plans remain unchanged; neither arm was recollected.

Root and an independent reviewer approved a separate read-only adapter
`2a83140d39968a70a7d97db9f71a1fd2194801b8e46b22f3caeaa4035cc42afd`.
It lowercases only physical type matching, checks exact physical/semantic
consistency and re-parses the retained log against the unchanged saved plans.
Separate shared audit jobs passed in1s each; before/after hashes bind the original
bytes. The independent full-plan reviews then supplied the required manual gates:
A `7ac0167a0df1e4a95e21bd59e930247a21d3a434828b5830c711d7d2ca019613`,
D `ca025c4e43e6ca7c5d3ca7304aeea67a09eefb1e0fa23920fd3bde9d8af5e740`.
The original gated pair job `m35-u4-gemma31-pair-a-r1` passed in1s. The prior
CPU metadata wrapper failure also remains recorded. Raw bundles, complete plans,
manual reviews and supervisors are under `/tmp/llmp-u4-gemma31-quality`; adapter
execution receipts are under `/tmp/llmp-u4-gemma31-audit-adapter-r1`.

## Observational clocks and implementation scope

| Prompt | A TTFT s | D TTFT s | A decode tok/s | D decode tok/s | A end-to-end s | D end-to-end s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hash table | 5.749753 | 5.810570 | 10.956425 | 11.250308 | 17.343154 | 17.100920 |
| Lighthouse | 4.360391 | 4.362308 | 10.989723 | 11.274868 | 15.918385 | 15.628496 |

A/D decode-rate ratios are0.973878/0.974710; A/D end-to-end ratios are
1.014165/1.018549; A/D TTFT ratios are0.989533/0.999560. TTFT is
`t_first−t_send`, not isolated prefill time. Decode uses127 divided by the
first-to-last-token interval. These single debug-logged quality samples are
observational; no D-085 performance conclusion follows.

The two-file implementation generalizes the existing Q8 scalar schedule repair
to all18 supported VecQ encodings using actual qk/qi/VDR and incumbent small-K,
iteration-halving and IQ3 exclusion rules. It changes only GB10 dense one-token
calls without IDs, including existing joined one-token callers; ordinary
multirow and routed-ID schedules remain unchanged. No family/shape whitelist.
The actual incumbent operand screen has182 positive inner checks:174 matrix
cells and8 fixed controls, with5 unchanged earlier positives retained. Nine
formats support prepared MMVQ; the other nine require its exact unsupported
refusal and compare against ordinary MMVQ. Original failed fixture/discovery
attempts and interrupted runs remain preserved separately. This result does not
resolve the earlier Gemma26 joined-row or Gemma31 assistant/frozen-state failures.
The registered [C1 adoption speed check](SPEED-RESULTS.md) passes all three
median metrics within its fixed 10% bound. Final native builds passed on both
Sparks. The shared-check union has 3,839 positive GoogleTest cases and 21 other
checks; 77 focused checks close six initial synthetic-fixture failures.

Five affected model checks passed. Gemma26/Gemma31 assistants and DSpark completed
execution and state restoration but retained 15/6/3 frozen differences. A bounded
legacy-drafter control using the same scalar CUDA source reproduces all 24
differences exactly. This establishes observed migration equivalence, not numerical
qualification of those domains; the [drafter report](../../m35-stored-drafter/RESULTS.md)
retains the failures and provenance. No golden, grant or quality bound changed.
