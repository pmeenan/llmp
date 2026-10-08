<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plain greedy decode tokens on the GPU — 2026-10-08

DeepSeek V4 and native/GGUF Qwen3.8 now select an eligible plain greedy
decode token on the GPU and copy its I32 value to the host. The complete
head still computes with the same arithmetic. Sampling, requested scores and
draft/verify paths keep their existing output contracts. There is no new public
setting. The internal false policy remains an explicit comparison control.

Five short own-engine comparisons preserve complete initialized states,
endogenous histories and actual full-vocabulary continuation heads exactly.
Decode means improve 0.37–1.24%; these are bounded n=2 results, not reference
parity or a claim about every context, batch size or quantization. Total paid
changes are smaller and often within bookend movement. The fifth screen
extends DeepSeek adaptive joined plain waves while preserving feature
injection and the drafter ring. Shared prompt-completion first-token
publication remains an open M3.5 transfer.

## Mechanism and limits

`Llm::RunPlainGenerationUnit` serves both a prepared scalar unit and scalar
fallback. It tries the optional token hook only when `DeviceGreedy` permits it;
nullopt retains ordinary rows. It publishes a chosen token only after success,
preserving per-unit failure and processed-prefix handling. Homogeneous plain
waves select token outputs; a wave needing scores or sampling retains rows.

The complete shape key includes output policy. Scalar and joined plans keep
the argmax tensor alive; setup funds the maximum of both variants before any
execution. Qwen reuses funded scalar-logit or wave-ID staging; DeepSeek uses
its existing output staging. `GreedyOutputBytes` authenticates I32 type,
packed count, alignment, producer flavor and containment in current activation
storage. After proven completion every returned ID is checked before any
owner receives a token. An invalid batch quarantines processed state.

The explicit host-greedy argmax flavor matches `ranges::max_element`: the
lowest index wins ties, NaNs after the first element are ignored, and a NaN
at element zero selects zero. Plain Gemma2/3/4 use this flavor too, closing
the old NaN discrepancy. Native draft/verify argmax behavior is unchanged.

Current scope is plain decode, including DeepSeek adaptive joined
`PlainWave`. Common row-free eligibility preserves ordinary `DeviceGreedy`
speculative refusal; adaptive plain waves use the common predicate and return
a single `kept` ID per owner. They preserve all features and injected ring
writes. Mixed sampling/scoring retains rows, and scalar speculation plus
draft/verify keep their prior contracts. Native Qwen speculative scalar/joined
generation always runs lean Draft/Verify ID paths; it has no separate plain
decode fallback. Native and GGUF target T27 decode consumers are adopted.
Every family's prompt-completion head still feeds host first-token choice;
shared device publication there is a separate compatible open plan extension,
including injected prefill, retained features and catch-up/export/carry/
pending-row contracts. This slice changes no public admission or numerical
tolerance and makes no first-token publication claim.

## Paid work and results

Each fixed comparison uses one binary in order **off/on/on/off**, fresh output
directories, identical setup and inputs, and one identical warm traversal.
Clear retains plans, then the paid interval includes independent prefills and
32 endogenous decode units. C1 returns 32 decode tokens; C2 returns 64.
The initial prefill choice is also retained in each history. Model loading
and the warm traversal are excluded and disclosed separately in
[results.json](results.json). All growth, planning, capture/replay and host
output processing occurring after the paid start are included.

Both policies observe complete initialized-state ranges/hashes and histories
at the endpoint, then execute one common full-row continuation per owner
outside the paid interval. Neither DS arm retains or scans diagnostic full
heads during paid decode. Off copies the complete head and chooses on the
host; on runs device argmax and copies four bytes per owner.

| Workload | Host decode seconds, O1 / O2 | Device decode seconds, A1 / A2 | Mean change | Host bookend movement |
| --- | --- | --- | --- | --- |
| Qwen native C2, 1536/1280 prompt | 1.753088708 / 1.753922012 | 1.737580724 / 1.740369786 | −0.829% | +0.048% |
| Qwen native C1, 1536 prompt | 1.169299534 / 1.169786338 | 1.162793658 / 1.161198721 | −0.645% | +0.042% |
| Qwen GGUF C2, 1536/1280 prompt | 1.403611019 / 1.407200842 | 1.386049606 / 1.389818800 | −1.243% | +0.256% |
| DeepSeek C2, 4352/4608 prompt | 1.830888575 / 1.838062516 | 1.826006933 / 1.826831024 | −0.439% | +0.392% |
| DeepSeek injected C2, same prompts | 1.867578391 / 1.878700714 | 1.865759472 / 1.866764120 | −0.367% | +0.596% |

Decode ranges do not overlap in these five screens. This is descriptive
evidence from two samples per policy, not an estimate of long-run variance.

| Workload | Prefill mean change / host movement | Total paid mean change / host movement | Paid host range, seconds | Paid device range, seconds |
| --- | --- | --- | --- | --- |
| Qwen native C2 | −0.088% / +0.258% | −0.413% / +0.166% | 3.994142908–4.000761002 | 3.977980836–3.983903010 |
| Qwen native C1 | +0.068% / +0.118% | −0.288% / +0.080% | 2.345781401–2.347658748 | 2.338724368–2.341216713 |
| Qwen GGUF C2 | +0.043% / +0.353% | −0.291% / +0.328% | 5.403462060–5.421164428 | 5.394502480–5.398657065 |
| DeepSeek C2 | +0.002% / +0.564% | −0.060% / +0.540% | 13.037846177–13.108253057 | 13.055786647–13.074544442 |
| DeepSeek injected C2 | +0.126% / +0.667% | +0.056% / +0.657% | 13.103840962–13.189884220 | 13.130984110–13.177394360 |

The retained aggregates contain every individual endpoint and its range;
`python3 compare.py` recomputes means, deltas and movement. These are
same-engine publication factors, not new llama.cpp, Mia or TensorFold runs.

## Adaptive DeepSeek plain-wave follow-up

The fifth factor uses the approved DSpark drafter with the same DeepSeek
4352/4608 prompts, context 8704/chunk 4096/C2 and output-only O/A/A/O control.
Both modes prefill with feature injection, then execute 32 joined plain decode
units. The complete target head, feature activation and independent drafter
ring writes remain in the graph. The candidate adds host-compatible argmax;
it does not turn off injection or substitute ordinary target-only decoding.
Setup funds the injected token variant with the same drafter model and ring
states. Whole-batch IDs are validated after completion before any `kept`
result is published. Existing `device_tokens=true` selects this compatible
extension; there is no added flag or public setting.

Decode samples do not overlap, but the −0.367% mean is smaller than the
+0.596% host bookend movement. Total paid +0.056% is neutral within +0.657%
movement. These n=2 observations support removal of redundant host row
transfer/scan with exact state; they do not establish a general adaptive
speculation speedup or new reference parity. No timing ladder follows.

The initialized state is 271,794,176 bytes per owner, including the complete
786,432-byte DSpark ring. Both paid and settled range lists authenticate
region 1 at offset 0 with this exact ring length. After the paid endpoint each
owner executes an actual injected full-row continuation, three endogenous
draft proposals and three full target verify rows, then Accept1/Rollback.
All 32 complete vocabulary rows across the four arms are finite; corresponding
payloads match exactly across arms, as do histories, proposal IDs, paid states
and settled states.
Each paid arm has 36 overall units/32 joined replays and 0 off/64 on device IDs.
All four processes and the installed supervisor return 0; kernel messages are
empty and the GPU is empty afterwards.

Six focused controls pass: a graph test preserves every target/feature/ring
operation and independent injection writes; common eligibility and three
existing fake dispatch/refusal controls; one actual DSpark serving fixture.
The serving fixture uses existing `wave_form=plain` with ordinary token
policy unset, exercises unequal C2→C1 departure, mixed sampling/scoring,
protected peer and spill/restore, and then compares wrapped 257/258-row
injection followed by actual joined draft/verify and settled state. C1 still
uses its existing scalar speculative route. The unchanged HostGreedy kernel
and prior target-family controls are carried, rather than rerun broadly.
The five-target Spark build and all six tests pass. No policy/default delta
or extra GPU check is needed after this factor.

Qualified source SHA 76b3fdb1d0a9572f635a81a8dc32ab02aef4e3282b889b7868aba39cd0e57f25,
probe SHA 2acceec3f83596658eefcc4853c46127418979b67233dc8b545bff166e3b4ece,
runtime SHA e8e2ffd87d71f59fc4c8aef4b5ae1dcc1fedcc52c603813a7dbbd40134ac5f1a,
test SHA 4e8ed14c0c3bdb3cf78cc677abaddfb64a38e22a7ab36c0ead61c23f10713c84,
official receipt SHA 0296e41b77f3db5f50dfc06b68ebd87906a683d45350ea2d5586f929c8333ecd.
Prerequisite checked SHA 31829b4508697c1dc65baee42ba1262a8879a7a94c81807cc7414024f779cb95;
factor checked SHA d8ae39b5dcca6491fb2b5efbe407411ac3ea1cd393e353516d5fb88a7a92a10c.
The final runner-header clarification is comment-only; the other 55 bound
source files retain their measured bytes. A first launcher incorrectly named
an absent SDK Python path and retired 127 before the controller ran; its
corrected `/usr/bin/python3` launch ran these exact frozen bytes. That failed
launch contributes no test or timing sample.

Replay the existing DeepSeek input recipe below, with the same target and
DSpark artifact dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5
(index 437561ad6dd1e8d5fd3758be2a8eac52b8a1c81f6a1fb54b27e0b3badfeaa11c).
Run `llmp_dsv4_mask_probe ARTIFACT IDS0 IDS1 NEW_OUT
injected-host|injected-device DRAFTER` four times in host/device/device/host
order under installed supervision and the same pinned library environment.
Require the existing factor retirement/setup/work guards plus exactly one
`PLAIN_TOKEN_INJECTION` record, complete paid/settled target+ring ranges,
valid three-ID `draft{0,1}.i32` and finite full `verify{0,1}.f32` payloads.
Compare every proof across arms; model loading/warm traversal and all
continuation/draft/verify observations stay outside paid time. The aggregate
and reusable comparison script retain all endpoints after raw cleanup.

Task-entry TensorFold pins remain native f8fe17d24629aedabf90bbf78279dd776e6d62e7
and retained Python 0.6 ed78d6fc204d89d90b045bf033d6551e7714f3a1. Native
DeepSeek is under qualification; retained Python CUDA remains applicable.
No cross-engine reference is rerun for this publication-only extension.

## Exactness, funding and selected paths

Every factor returns identical complete finite continuation heads, histories,
initialized state hashes and state-range/layout descriptions across all four
arms. The states are 182,525,952 bytes per Qwen owner and 271,007,744 per
target-only DeepSeek owner (injected state above). Full continuation heads contain 248,320 Qwen or 129,280
DeepSeek floats. Scalar Qwen histories contain prefix+33 IDs; both C2 histories
do too. Independent validation rehashed every full head and checked finiteness.

All paid Qwen C2 arms execute 38 logical units; C1 executes 35. Native C2
records 384 MXFP8 pairs, GGUF C2 545 VecQ pairs. DeepSeek records 36 overall
target units, including its 32 joined replay units, with device raw-mask plans
selected in both arms. Actual device-token counters are zero off and 32/64
on. Every arm has positive actual graph replay; paid capture/eager counts are
retained rather than assumed. Setup is exactly equal between policies:

| Workload | Node budget, bytes | Fixed bytes | Activation capacity, bytes |
| --- | --- | --- | --- |
| Qwen native C2 | 76,865,400,860 | 329,838,620 | 178,257,920 |
| Qwen native C1 | 76,119,256,604 | 321,891,868 | 178,257,920 |
| Qwen GGUF C2 | 55,223,185,772 | 357,495,148 | 195,035,136 |
| DeepSeek C2 | 101,313,155,072 | 2,176,585,728 | 1,807,745,024 |
| DeepSeek injected C2 | 112,582,552,064 | 2,530,306,560 | 2,143,289,344 |

These are benchmark funding envelopes, not public simultaneous-context
admission claims. Results preserve the complete setup receipt.

Initial primitive/graph/fake checks cover odd vocabulary, signed-zero and
infinite ties, all-negative-infinity and NaNs at multiple positions; malformed
output type/count/span/flavor; successful/nullopt/failed token dispatch; and
per-unit prefix/failure handling. Actual native/GGUF/DeepSeek serving checks
compare scalar and joined row/device outputs, unequal owner departure,
protected peer state, initialized-state spill/restore, mixed scoring and
sampling. Actual Gemma2/3 generation and existing Gemma26 scalar/joined
controls cover their plain host-greedy flavor.

The preceding target-only slice's final combined-base check passes exactly
11 controls, including native
Qwen and DeepSeek serving with the diagnostic override unset. This proves
ordinary constructor defaults. The GGUF model case previously passed with an
explicit true override and the same Qwen constructor; its unchanged format
branch is carried, not described as a fresh unset-override GGUF run. Initial
18 prerequisites, six A serving cases and one transported-ELF B GGUF case,
plus the final 11, total 36 accepted positive executions (29 unique cases).
The final seven-target build and CUDA shared-operation merge compile succeed;
no performance factors are repeated after the default change. Final default
checked SHA d31ffbd9bca32be43e5436a1311d9400a4dd109254bd752ee9a9ffbfef6dbda5,
qualified source ffd77be0353a6e09809478672c4992dffb40a38d5454364346d2c96a21444cea.
At that target-only landing, two runner-header comment edits clarified its
speculation exclusion; those executable bytes were unchanged. The injected
follow-up below now extends joined adaptive plain publication.
Full suites, workstation tiers and the full swap table remain deferred under
the owner's focused optimization override. No reference quality gate is
weakened or reinterpreted by these own-engine checks.

## Provenance and replay

Native Qwen and DeepSeek ran on Spark A; GGUF ran on Spark B using the exact
A-built probe/test ELFs, authenticated isolated private cuBLAS libraries and
the standing B artifact. No GGUF checkpoint or artifact was copied, and no B
source/build tree was changed. Driver is 580.178.04, GB10, official pinned
`aarch64-c09daba6ac31edee` SDK. Receipt SHA
0296e41b77f3db5f50dfc06b68ebd87906a683d45350ea2d5586f929c8333ecd;
cuBLAS SHA ee7c1657a03695c0de790aa79e34cef9c9649756b1846b11dd44caca20ba656b;
cuBLASLt SHA ba3b942f4ea43433b65e8c492a7b73de887534dc20146506ddaa4a78c79c5d30.
Actual first-resolved library paths and owned process retirement were checked
before/after, with stable boot, empty compute list, physical-memory admission
and per-arm kernel-log checks. Every accepted arm returns zero with no kernel
errors. Raw logs/telemetry are external and deleted when M3.5 closes.

Source, method, actual ELF and checked-result identities for each measured
factor are retained in results.json. Native C2 predates the additive scalar/
GGUF probe modes; native C1/GGUF share the later identical probe ELF. The DS
counter-only correction changes neither production nor paid work. Afterward
the source is combined with main `002ed48` and defaults selected; explicit
off/on controls keep the measured execution policies available.

Task-entry TensorFold pins were native
f8fe17d24629aedabf90bbf78279dd776e6d62e7 and retained Python0.6
ed78d6fc204d89d90b045bf033d6551e7714f3a1. Native DS/Qwen were under
qualification; retained Python CUDA remains applicable. This transfer reuses
existing reference qualification and makes no new comparative engine claim.

Use [plain_token_probe.cc](../../../benchmarks/plain_token_probe.cc) for Qwen
and the explicit `tokens-host`/`tokens-device` routes of
[dsv4_mask_probe.cc](../../../benchmarks/dsv4_mask_probe.cc) for DeepSeek.
Prepared artifact IDs are Qwen native
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
GGUF `5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78`,
DeepSeek `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`.
Native Qwen derives from checkpoint925d7be6, GGUF UD-IQ3_XXS from38bb39ee;
the existing model support entries retain complete import provenance.

Obtain the authenticated standing Qwen `wave-prompts.json` from
`~/.local/share/llmp/references/qwen-device-masks/inputs` on Spark B or
supply its byte-identical capture, SHA
52dd48eb2b96be3ebc6de1a9c8e3f5155016599a20c85c1dc1dd3778d2123952.
Serialize its named 1536/1280-ID decode prompts as little-endian I32 without
retokenizing. Their hashes are
35ece15522faff9e2895774eb9146b83a7ad623de08b484ac7e6a8feeb29c35d
and e3288c43e2a1fc9c6377aa46f194b289e0085304e43af09f2a05a94c5f88c7d7.
DeepSeek uses the retained `ds4-mask4/ids{0,1}.i32` replay inputs on Spark A,
4352/4608 IDs, hashes
d74255dd4e8ae5d806824e2378007d1599f8115e8ac80d886b60a865350ecadd
and b59367a57eeba410e0ce051780164d1ef8b414a64f2d195e1b855759f04e145d.
Those captured replay inputs are retained independently of disposable logs;
the [DS mask report](../deepseek-device-masks/README.md) also documents their
frozen-source text retrieval/preparation recipe.

Inside installed `spark-job --gpu` supervision, with pinned first-resolved
cuBLAS, run four fresh processes with a new OUT for each, in off/on/on/off
order. Qwen invocation is `llmp_plain_token_probe ARTIFACT IDS0 IDS1 OUT
off|on`; append `c1` for actual scalar Slot hooks, or `gguf` for GGUF C2.
DeepSeek invocation is `llmp_dsv4_mask_probe ARTIFACT IDS0 IDS1 OUT
tokens-host|tokens-device`. The probes fix the registered context/chunk/
owners, warm traversal, 32-unit paid work and output proofs described above.
Require all retirement markers, zero exits, matching setup, actual output
counters, full finite heads and identical histories/state proofs. Preserve
each arm's timings and bookend movement. Never pool a refused arm.

Three harness mistakes are preserved as unpooled failures: the first build
omitted an explicit Qwen spill-place initializer required by warnings-as-errors;
the first DS factor incorrectly treated overall graph counters as scalar-only
and double-counted joined work. It stopped after O1, which retired safely.
The corrected DS guard requires overall36 and joined32 separately. None of the
failures supplies performance evidence or changes the model's computation.
The first combined qualification build succeeded but its direct controller
omitted the generated artifact-corpus environment needed by a DSpark source
fixture. It stopped before model tests; the method-only correction supplies
the existing CMake-generated corpus path. That failed run remains separate.
