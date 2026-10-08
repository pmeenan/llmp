<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma4 fresh-zero state preparation

Both approved Gemma4 profiles reuse the shared LiveState-owned, no-victim
preparation ticket and scoped page-in drain. Ordinary 8K/C1 same-native screens
reduce prefill by 1.162% for Gemma26 and 1.050% for Gemma31, and combined paid
prefill/decode by 0.902% and 0.847%. Each mode has two samples; prefill and
combined ranges do not overlap, while decode movement is within off-bookend
movement. These are bounded own-engine results, with no decode or reference
parity claim. Public context, chunk and cohort defaults stay unchanged.

The policy selects eligible fresh zero backing for a known next chunk. It does
not publish future initialized ranges or cursors, evict useful cache, replace
kept restart bytes, or enable CPU lookahead/capture. Kept restart and fresh
sparse-file provenance, broader context/output/cohort performance and remaining
DeepSeek/Qwen adapters stay composite T67 OPEN. The shared protocol and its
cancellation/unknown-completion proof are recorded in
[shared state preparation](../gemma-state-prepare-ahead/README.md).

## Ordinary 8K comparisons

Each profile runs off/on/on/off in four fresh processes, with the same actual
ELF and only the preparation bit changed. Values are seconds.

| Profile / endpoint | O1 | A1 | A2 | O2 | Mean change | Off movement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma26 prefill | 2.332830 | 2.302910 | 2.298490 | 2.322660 | −1.162% | −0.436% |
| Gemma26 decode | 0.666247 | 0.667511 | 0.667620 | 0.668848 | +0.003% | +0.390% |
| Gemma26 combined paid | 2.999077 | 2.970421 | 2.966110 | 2.991508 | −0.902% | −0.252% |
| Gemma31 prefill | 10.390400 | 10.281100 | 10.307400 | 10.416600 | −1.050% | +0.252% |
| Gemma31 decode | 3.230680 | 3.229240 | 3.232110 | 3.243370 | −0.196% | +0.393% |
| Gemma31 combined paid | 13.621080 | 13.510340 | 13.539510 | 13.659970 | −0.847% | +0.286% |

Gemma26 prefill ranges are off 2.322660–2.332830 / on 2.298490–2.302910;
Gemma31 ranges are off 10.390400–10.416600 / on 10.281100–10.307400.
[results.json](results.json) keeps all samples, combined ranges, means, movement,
exact identities and concise provenance. `python3 ../gemma-state-prepare-ahead/compare.py results.json` recomputes all
nine endpoint aggregates, including the separate diagnostic below.

The ordinary recipes use configured context 8192 and 7680 prompt tokens,
followed by three common untimed anchor inputs and 32 paid forced decode units.
Final cursor is 7715. The full standing 8227-ID file is authenticated before
and after each arm. Gemma26 uses 1024-row chunks (seven full chunks and a
512-row tail), local ring 2048, and an equal derived execution budget of
22454332640 bytes. Gemma31 uses thirty 256-row chunks, local ring 1280 and
27879195840 bytes. The budget includes fixed host publication, state capacity,
weights, retained plans/graphs and optional temporary plan allowances. The
serving policy matches the current C1 adapter: checked full final-hidden
arithmetic, max head rows 1, norms/rope/add, G26 route/reduce or G31 quantized
GLU, ordinary lookahead on, capture/features off and the current attention
policy. This C1 prefill measurement is not a joined-prefill speed claim.

Both modes load weights with six rows, then Clear retains precisely the same
charged zero remnant: cursor 0, used bytes 0, healthy state, matching kept
extents and no pending ticket. It is not an empty allocation or a preparation
warmup. Paid prefill includes fresh growth, submission, zero-fill, scoped drain,
CPU planning, graph execution and current publication. Startup/model loading,
six-row loading traversal, three anchors and final payload checks are outside
the timer in both modes. Combined paid is prefill plus the 32-unit decode time.

All non-preparation/non-timing summary fields and budget dictionaries match
within each profile. G26 has seven built/cached future plans; G31 has 29.
Both retain one ordinary capture and 33 replays across actual graph runs.
Bound-plan selection counters describe selected implementations, not executed
kernel counts. Each enabled arm completes/adopts 170 extents for G26 and 680
for G31; disabled arms prepare none. There are no preparation errors/refusals.
Every arm has two full finite 262144-element heads, identical bytes, 32 incoming
choices/forced IDs and all three anchors. Complete initialized state also
matches: G26 581959680 bytes/60 ranges, G31 1698693120 bytes/120 ranges.
All eight processes and the installed supervisor return zero, boot is stable,
observed kernel messages are empty and the GPU is empty after retirement.

## Shared adapter and lifetime qualification

The common runner reserves and accounts for at most 120 initialized-range
descriptors. It derives the known near chunk independently of CPU lookahead.
All verify-save prechecks finish before preparation submission; every slot's
ticket finishes after the current fence and before rollback, future binding,
feature publication or cursor publication. A post-write preparation error
quarantines every current owner and invalidates retained feature authority.
No shared ticket/source/retirement contract changes were needed.

Two actual wrapped, multiowner profile controls run with CPU lookahead disabled,
compare complete heads/features/state, and preserve checkpoint advance/rewind,
protected peers, spill, fresh kept restart and the next continuation. A held
future registration after successful wrapped current GPU work proves all-owner
quarantine, unchanged cursors and refusal of stale feature borrows/copies.
The original combined job passed that failure control and the then-false default
assertion, but both lifetime cases hit the execution budget: their fixture
charged three CPU state copies although only one existed. The correction funds
one maximum-1793-position CPU witness plus eight heads/six features; pinned
copies/checkpoints already have catalog charges. The original budget is
unchanged. Only the two failed cases reran and passed; the original aggregate
stays FAILED and is not pooled into measurements. Unchanged shared proof and
historical suites were not repeated.

Both runner and private ServingOptions now default preparation on. A separate
actual approved 8K/C4 PromptSession fixture compares explicit off with unset
preparation on fresh Servers, pinning ordinary chunks 1024/256 independently
of calibration. It asserts resolved settings and four configured owners while
truthfully dispatching scalar prefill (capacity 1). Distinct owner lengths
4352/4480/4608/4736 give 20 G26 or 72 G31 units / 18176 rows. Owners leave the
execution cohort as they finish; complete state digests captured at departure
match after peer progress. Full finite heads, whole histories/current cursors,
initialized state and actual work match off/default. Off counters are zero;
default completes/adopts 560 G26 or 2260 G31 extents with no errors/refusals,
healthy owners, drained registry and explicit successful teardown.

The first adoption attempt passed the updated default assertion but both new
actual cases refused the final state-copy oracle because departed owners were
inactive. The fixture now reselects all owners only after ALL prompt work ends,
providing valid copy leases without weakening actual departure. Only the two
failed actual cases reran; the positive default assertion is carried. Together
with recipient lifetime/failure controls, positive executions total seven / six
unique cases. Actual adoption checked receipt is 6a2d5af6… / source 216db02d… /
serving fixture 73545127… . Every child and installed supervisor retires with
exit zero, exact pre/post executable/receipt/library identities, stable boot,
empty kernel errors and empty GPU. No timing or historical control was rerun.

## Separate 16K diagnostic

The first G26 screen used configured context 16384 with 8192 prompt tokens,
1024-row chunks and the checked all/normmul-on recipe. It reduced prefill
1.071% and combined paid 0.829% at n=2; off prefill movement was −0.447%.
Decode change +0.058% is inside +0.241% off movement. Full heads/choices and
592445440 initialized state bytes match. This diagnostic motivated the ordinary
8K batch above; it establishes neither ordinary serving adoption nor public
16K admission/reference parity. Its own source/ELF and all samples remain
separate in results.json.

## Replay and identity

Use the existing `jitllm_gemma_prefill` helper, pinned official Spark toolchain
and installed supervised GPU protocol. Approved artifact manifest IDs are
`4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3`
(G26) and `32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08`
(G31); their index SHAs are retained in results.json. Use the standing
`gemma-prefill/input1/ids.i32`: 32908 bytes / 8227 LE I32 IDs, SHA
`6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b`.
[Canonical corpus/tokenizer preparation](../gemma26-capture-ahead/README.md#reproduction-without-disposable-controllers)
recreates that input with `llama_prefill RAW_GGUF CORPUS NEW_INPUT prepare 1024`.
No disposable controller/raw bundle is required.

```text
jitllm_gemma_prefill ARTIFACT IDS FRESH_OUT 26 serving 1024 normmul-on
  state-only lookahead-on phases-off state-chunked capture-ahead-off
  features-off prepare-off 8192 7680
```

Run O1/A1/A2/O2, replacing only prepare-off with prepare-on in A1/A2.
For G31 replace 26/1024 with 31/256. The legacy diagnostic uses 26/all/1024
and omits the trailing context/prefix pair (defaults 16384/8192). The helper
explicitly assigns both preparation modes despite the later true runner default.
Enforce the exact geometry, fresh remnant, work/budget equality, positive on/zero
off counters and full finite payload/state/ID equality above. Authenticate
source, receipt, actual ELF, first-resolved pinned cuBLAS/Lt, input and artifact
metadata before/after. Use 145-second child bounds, owned cleanup, empty GPU and
MemAvailable >= 48 GiB, stable boot/kernel guards and positive installed wait.

The diagnostic uses source 2830d374 / probe ed882c8c; ordinary 8K uses source 78fa8e68 /
probe fab2b4fa, official receipt 0296e41b. Full identities are in results.json.
The later default/wiring delta has its own actual serving proof and does not
relabel either measured binary. Fresh task-entry TensorFold pins were native
main f8fe17d24629aedabf90bbf78279dd776e6d62e7 and retained python-0.6
ed78d6fc204d89d90b045bf033d6551e7714f3a1; applicability was checked, with no
new competitive-reference claim. Full suites, broader contexts/output contracts
and new reference measurements remain deferred.

Final repository light checks pass: REUSE and embedded headers cover 1947 files,
portability boundaries cover 422, all eight changed C++ files are formatted,
and git diff checks plus all six retained state-preparation factors / eighteen
endpoint aggregates replay exactly. All 78 qualified source bindings remain
byte-identical after the actual adoption check. The final documentation adds
this provenance only; full-suite and shipment tiers remain deferred.
