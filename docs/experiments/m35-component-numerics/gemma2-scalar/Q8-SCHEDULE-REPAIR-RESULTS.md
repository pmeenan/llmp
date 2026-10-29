<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Q8 scalar schedule repair: representative screen PASS

The registered R screen **passes**: all 1,070 teacher scores exactly equal the
retained A reference. Mean and maximum absolute NLL differences and PPL delta
are zero; agreement within 0.1 nat is 100%. Both mean NLL/PPL values are
3.5635004927423646 / 35.28650134833719. The unchanged bounds remain 1,070 targets,
absolute PPL delta≤0.5%, mean absolute difference≤0.05 nats, maximum≤1.0 and
at least95% within 0.1 nat. The [original A/D FAIL](RESULTS.md) remains unchanged.
This single Gemma2 scalar screen does not qualify other consumers, refresh frozen
expectations, pass a semantic pack or establish a speed gate.

The [prospective supplement](Q8-SCHEDULE-REPAIR.md), committed initially as
`3735aad4c963bf544fc43986815fbc5b2bbe1452`, was corrected before service launch
by `9ab2fe542e13ccecf8acdddd768c292034d4e373` and
`913b7949c26478ceea4b9958225b0e204d629ad4`. The final registration preserves all
runtime, source, input and numerical bounds. [Aggregate results and raw-file
hashes](q8-schedule-repair-results.json) bind the collection, comparison and
retained failures; candidate pins bind the
2-file production/fixture delta and its four passed operand controls.

R uses retained runtime SHA-256
`2188a8bb2fd0e126a0da02f6d5af1c809ff0e4e1682021d37869e22dc01d0d05`,
compiled module identity
`e777b69317ac8e6f49ded8217d703d73ccc195e881e4b8a8482f6e30d52d309b`,
and actual 2,883-file source manifest
`be392d99e2128f9e7979b0656f4c49f74f659539109bdf26b1178a537799a5da`.
A's existing raw collection is reused without inference. Its previously audited
ASLR-only outer-wrapper failure and recovery remain recorded in the original
report. No threshold or golden changed.

## Input, execution and retirement

The complete 1,071 echoed IDs match A, with one leading BOS 2, leading null and
1,070 finite nonpositive teacher scores at the exact positions and IDs. Their
registered hash is
`ee85f4ffc9be5825936ff379d873322a78c048018375805ffbb4a5cb5b1946c4`.
The same 4,000-byte passage, embedded tokenizer metadata, artifact, F16 KV,
context8192, prefill chunk1, two slots and sequential C1 are retained; scoring
produces zero decode tokens.

Five successfully completed stage0/phase4 score plans have one row and segment,
with 256/256/256/256/47 uses. Each retains 130 VecQ dense, 26 VecQ GLU and one
prepared MMVQ selection. All 506 launch recipes per plan, including attention,
RoPE, quantization and complete physical operand descriptors, equal original D
when excluding only implementation identities, which change with the compiled
module. The generic Q8 schedule repair is exercised without withdrawing VecQ.

Spark B job `m35-u3-q8-scalar-repair-r3`, installed exclusive600/grace30, exits
zero; the service retires with TERM/returncode0. Pre/post runtime, dependencies,
source, registered files, corpus and artifact pins match. Both raw `ldd`
transcripts are retained; only ASLR load addresses are normalized. The final
host receipt reports no active jobs or GPU processes. The fixed pairing ran
once, producing `pair.json` SHA-256
`a738dcb0787c4dc8a1aebdbb8f309a978d0492439b936b4e1021b4f246699ceb`.

## Greedy controls and observational clocks

Both max128 greedy replies finish by length and exactly match A's text hashes.
The hash-table opening still incorrectly calls the index unique and misleadingly
implies similar keys cause collisions. The lighthouse story is coherent but
truncated. Exact agreement preserves those semantic limitations.

| Registered prompt | A decode tok/s | R decode tok/s | A/R rate | A elapsed s | R elapsed s | A/R elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hash-table explanation | 76.067136 | 78.446920 | 0.969664 | 2.401391 | 3.382227 | 0.710003 |
| Lighthouse story | 77.207997 | 79.596968 | 0.969987 | 2.156497 | 3.102392 | 0.695108 |

These are single debug-logged observations against an earlier A sample, not a
matched timing comparison. R has higher observed time to first token despite its higher
decode rate. No speed-gate conclusion follows.

## Preserved pre-launch failures

`m35-u3-q8-scalar-repair-r1` fails preflight because A's pinned `audit.json` and
`plans.json` were absent on B. The exact existing files were copied and
authenticated under the first prospective administrative correction.
`m35-u3-q8-scalar-repair-r2` then exits2 before service launch because the external
wrapper supplied `--label q8-repair`, while the unchanged CLI accepts `old|new`.
The second correction fixes the label/output name to `new`; full argv validation
uses the real parser with collection replaced by a stub. Neither failure ran
inference. R3 is the first and only actual R service collection.

Raw responses, SSE, logs, plans and supervisor receipts remain outside Git at
`/tmp/llmp-u3-q8-quality-r3/`, with both pre-launch failures in sibling r1/r2
directories. The original unexplained operand-test abort, later fixture/harness
failures and subsequent operand passes remain retained in their separate records.
