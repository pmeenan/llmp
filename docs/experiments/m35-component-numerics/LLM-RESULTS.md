<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# U4 LLM numerical transfer results

The registered native and GGUF speed screens pass, and target equivalence is
verified. Both strict Mia comparisons fail. The frozen bounds and inputs in the
capture, speed and Mia protocols remain unchanged; this is no general model
quality or M3.5 completion claim.

## Arms and actual execution

A is `5ec811a4413982630c05e7b6c2448f51f1191883` plus the registered
physical-plan diagnostic. D adds U4's component/evidence join, the reviewed
owned-device fix and the corrected GDN/PLE/selected-expert domains. Frozen
runtime SHA-256 values are:

- A: `405c5930accdc009ad8e824d2fcdeff00f063877f0d936c2b1a5dda6b28e519e`
- D: `16a8da3639e3a26e4fdd2dec6927510688ee769fe4fbe10acae17bfc21078338`

[The registered arm inventory](llm-capture-pins.json) records the exact source
patches, artifact/configuration identities and helper hashes. All ten resolved
dependencies match between arms. Spark A (`spark-c4e2`) ran the ordinary
captures, supplementary text controls and matched speed screen; Spark B
(`spark-56f5`) ran the independent Mia target control. All model jobs used
exclusive installed supervisors with 600-second limits and successful retirement.
The compiled SDK used Clang 22.1.8, NVCC 13.4.92 and cuBLAS 13.8.0.4; the host's
unrelated system CUDA compiler is not the build identity. Raw logs and samples
remain outside Git.

The independent comparison retains operation arithmetic, all physical operand
formats, shapes, strides, views and backing, logical routes, preparation geometry
and roots. It normalizes only implementation identity hashes. Successful-use
counts and unused installations are recorded separately; this is a completed-plan
inventory, not a per-kernel trace or routing-payload capture.

| Workload | Concrete A/D comparison | Actual teacher targets |
| --- | --- | ---: |
| Qwen native target | All captured target plans equal, including prefill 5/61, decode 1, verify 2–4 and score 997 | 996, every loss identical |
| Qwen native MTP | HC and NVFP4 expert fusions differ at stage 1, rows 1–5 | Target likelihood does not measure the drafter |
| Qwen GGUF | Decode 1, prefill 61 and score 997 equal; prefill 5 transfers existing Q6_K/Q8_0 vector/GLU arithmetic | 996, every loss identical |
| DeepSeek target/drafter | All 11 plans and 177 completed uses equal | 966, every loss identical |

Native target mean NLL/PPL is 2.803101989424089 / 16.49573709131301;
GGUF is 2.8432114725773774 / 17.170820705573195; DeepSeek is
2.7937082602496544 / 16.34150614057285. The native 996-target result remains
informational. GGUF and DeepSeek meet the unchanged registered transfer bounds.
Native has 23 successfully used plan identities per arm, but 391 successful uses
in A and 384 in D; these counts do not support a timing comparison.

The negative B and C captures remain retained. B omitted owned-backend device
facts; C fixed that but changed GDN BF16 product outputs and views into F32.
D restores the intended GDN site: stored MXFP8 QKV/gate weights, within the
existing row/owner domain. It also retains PLE key/value Q8_0 sites and the
accepted GGUF checkpoint's one IQ3_S expert layer. IQ3_S is qualified only for
selected products, with no unrelated dense or wider-row extension. The actual
GGUF five-row transfer and native MTP changes receive separate controls below.

## Independent Mia target control

Both arms completed all four retained 255-ID documents: 254 losses and 255
paid score rows per document, 1,016 targets overall. Both strict Mia comparisons
**FAIL** the original bounds:

| Metric | A and D | Original bound |
| --- | ---: | ---: |
| Absolute aggregate PPL difference | 4.054433875% (signed −4.054433875%) | ≤1% |
| Worst absolute document PPL difference | 9.625914741% | ≤4% |
| Mean absolute target NLL difference | 0.569943046 | ≤0.25 |
| Maximum absolute target NLL difference | 9.214178196 | ≤5 |

A/D `losses.jsonl` are byte-identical, SHA-256
`0438b3cdf68fda76f8eea5ae1793a7d7e0032500c2fad1a5e4ae2b2a0e5c0f18`.
Independent review authenticated the raw responses and all 1,016 target IDs and
losses against the retained reference. Each document completed one target
stage-0, score-phase-4 plan with 255 input rows. Reparsed raw logs agree with the
saved inventories. A and D have identical logical selections and product
arithmetic, all 5,817 physical descriptors, 2,350 graph steps, 2,362 static launch
recipes, 1,623 physical roots, and resource summaries for each document.
Descriptor formats, dimensions, strides, views and source slots are retained;
only implementation-identity hashes are normalized. Logical policy and registry
digests differ, while primitive and graph digests match. These inventories are
successful-use evidence, not measured per-kernel launch counts.

The two exclusive 600-second jobs, `llmp-u4-mia-a-r3` and
`llmp-u4-mia-d-r1`, completed with return code zero and TERM retirement with
return code zero. Frozen runtime hashes match before and after capture; all ten
resolved dependencies match the qualification. The independent recomputation
confirms the strict failures above. Signed per-document PPL differences are
−6.758508646%, −9.625914741%, +4.833830169% and −4.072072905%; the maximum
target-loss difference occurs at document 1, position 147, target ID 407.

A strict failure is not a failed collection, and neither is labeled a strict
quality pass. The existing M3.6 interimplementation target limitation against
Mia’s FP8-KV/BF16-SSM reference can be reused only for unchanged target execution under the prospectively registered
[Mia domain protocol](MIA-DOMAIN-PROTOCOL.md). It supplies no MTP exception.

## Affected controls and limits

The new MTP and GGUF transfer domains retain independent source/formula
oracles and real served-model controls. Native component tests compare HC prep
against FP64 at width 2,560 for rows 1, 5, 8, 9 and 37, including refusal and
no-combine paths; HC low/mix has BF16/MXFP8 plus FP64 controls at rows 21.
NVFP4 expert controls cover 16 experts/top-10, width 2,560/hidden 640, rows
1, 4, 9, 37 and 300 against decoded-quant FP64 and GGML MMQ or whole-operation
oracles. Existing error bounds are unchanged.

The integrated I build passes six real Qwen controls without skips: native MTP
ladder, depth-two full logits, default token schedule, GGUF plain and frozen
heads/state/spill, and unequal owned pending restores with changed-input capture.
The final J build passes all 3,471 unique shared checks with no failures or
skips, plus 31 affected tokenizer/template model checks. The six Qwen controls
remain applicable to its unchanged affected paths. Gemma 2/3's two functional controls
pass in both A and C, with unchanged paths in D; their runtimes are not matched
performance results.

Fixed 128-token greedy controls completed with identical usage and successful
retirement. Native Qwen and DeepSeek text match A exactly; GGUF differs but has
coherent, relevant openings. Every response ends by length; DeepSeek contains
reasoning with empty final content. These observations do not establish completed
answers or a full behavior-pack pass. No threshold, source pin or prompt was
changed in response to results.

## Matched speed and integration

Both families pass the registered nine-metric screen. All three pairs use
A/D, D/A, A/D order on Spark A, with a fresh process and private service state
per arm. DeepSeek's identical target/drafter plans retain their existing evidence;
the registered protocol requires no new speed screen for that unchanged domain.
Every process pays one C1/C2/C4 warm-up; native then pays two measured
passes and GGUF one. All paired prompt indices, widths and full token-usage
objects match, including 128 generated tokens per request and zero cached
prompt tokens. Every request finishes by length with no helper problem.
All twelve exclusive 600-second jobs finish successfully, with TERM retirement,
frozen runtime checks before and after, and all ten dependency hashes matching.
Default runtime environments are empty; no extra priming or debug-plan export
is enabled.

Metrics below are independently recomputed from retained request clocks with
`tools/speed_gate.py`, SHA-256
`0f1135a2283fe63797e15c8dffe48c9df39b0baae91a57f925b5ba21c3e3252e`.
The configuration strings match the registered arms; the corpus SHA-256 remains
`55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`.
The runtime/source pins above are the measured A/D arms, not the combined
integration build. Host `spark-c4e2` uses GB10 and driver 580.178.04.
The installed supervisor job names are `u4-FAMILY-pN-ARM-r1` for native
`qwen38` and `qwen38-gguf`, pairs 1–3 and arms A/D.

Absolute values are the median and [minimum, maximum] of the three run values;
a native run value is the mean of its two measured passes. Ratios retain each
matched pair: D/A for prefill seconds, A/D for throughput. A value above one
means slower D. Acceptance uses the median of the three paired ratios, not a
ratio of the displayed absolute medians. Every median is at most 1.10, and no
individual pair metric exceeds 1.10. These data support the coarse regression
screen, not a finer speedup claim.

### Native Qwen

| Metric | A median [min, max] | D median [min, max] | Pair 1 | Pair 2 | Pair 3 | Median ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 prefill s | 3.0810 [3.0594, 3.0854] | 3.0793 [3.0572, 3.0837] | 0.999281 | 0.998016 | 1.000871 | 0.999281 |
| C1 decode tok/s | 36.5376 [36.5278, 36.7916] | 36.8488 [36.7003, 36.8773] | 0.998449 | 0.990789 | 0.995299 | 0.995299 |
| C1 wall tok/s | 19.4836 [19.4700, 19.6206] | 19.5831 [19.5264, 19.6438] | 0.998816 | 0.994227 | 0.997808 | 0.997808 |
| C2 prefill s | 6.9891 [6.9204, 7.1714] | 6.9685 [6.9190, 7.0195] | 0.999804 | 0.978815 | 0.997056 | 0.997056 |
| C2 decode tok/s | 54.5808 [54.3674, 54.7539] | 54.6899 [54.6259, 54.6978] | 1.002344 | 0.993959 | 0.998005 | 0.998005 |
| C2 wall tok/s | 21.7933 [21.4082, 21.9539] | 21.8907 [21.8195, 21.9976] | 0.998017 | 0.981146 | 0.995551 | 0.995551 |
| C4 prefill s | 17.3286 [16.5140, 17.4203] | 17.1085 [17.0944, 17.4212] | 1.005344 | 1.036003 | 0.981289 | 1.005344 |
| C4 decode tok/s | 63.5238 [62.9077, 63.6753] | 64.5763 [64.4719, 64.7184] | 0.981543 | 0.974160 | 0.987645 | 0.981543 |
| C4 wall tok/s | 20.6773 [20.5517, 21.3075] | 20.9369 [20.6915, 21.0066] | 0.999310 | 1.014323 | 0.981601 | 0.999310 |

### Qwen GGUF

| Metric | A median [min, max] | D median [min, max] | Pair 1 | Pair 2 | Pair 3 | Median ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 prefill s | 6.7024 [6.6815, 6.7033] | 6.6815 [6.6264, 6.6823] | 0.996747 | 0.991757 | 0.996997 | 0.996747 |
| C1 decode tok/s | 31.0592 [30.9973, 31.0818] | 31.0180 [30.8088, 31.1614] | 1.008862 | 0.994732 | 1.001326 | 1.001326 |
| C1 wall tok/s | 11.8609 [11.8585, 11.8735] | 11.8751 [11.8451, 11.9581] | 1.001336 | 0.992920 | 0.998607 | 0.998607 |
| C2 prefill s | 13.5830 [13.5689, 13.6013] | 13.5233 [13.4895, 13.5407] | 0.996640 | 0.993122 | 0.995542 | 0.995542 |
| C2 decode tok/s | 48.3290 [48.2726, 49.0830] | 48.4258 [48.0121, 48.8979] | 1.003785 | 0.996838 | 1.006599 | 1.003785 |
| C2 wall tok/s | 13.5831 [13.5706, 13.6527] | 13.6584 [13.5890, 13.6725] | 0.998555 | 0.994493 | 0.998651 | 0.998555 |
| C4 prefill s | 28.3600 [28.2945, 28.3787] | 28.3304 [28.1974, 28.3622] | 0.998956 | 0.996568 | 0.999418 | 0.998956 |
| C4 decode tok/s | 66.4792 [66.3840, 66.6304] | 66.6426 [66.6238, 66.8189] | 0.997547 | 1.000099 | 0.993491 | 0.997547 |
| C4 wall tok/s | 14.2264 [14.2128, 14.2555] | 14.2413 [14.2405, 14.2941] | 0.998953 | 0.997303 | 0.998050 | 0.998050 |

### Readiness and memory

These ranges keep process readiness separate from steady measured work. Ready
seconds include the production service's startup/readiness path; no pure weight
load time is inferred. Peak occupancy is each run's whole-node `MemAvailable`
drop sampled every 20 ms, not exact allocation or isolated device accounting.
Actual memory admission remains active. Wall time below covers the complete
helper run, including startup, warm-up, measured passes and retirement.

| Family / arm | Ready seconds [min, max] | Whole-run seconds [min, max] | Peak occupancy estimate GiB [min, max] |
| --- | ---: | ---: | ---: |
| Native / A | [6.015, 6.023] | [147.568, 149.865] | [89.744, 90.055] |
| Native / D | [5.620, 6.017] | [147.672, 148.338] | [89.910, 89.998] |
| GGUF / A | [4.216, 4.415] | [143.752, 144.394] | [64.878, 64.897] |
| GGUF / D | [4.215, 4.620] | [143.055, 144.314] | [64.900, 64.922] |

### Integrated validation and scope

Final J's source patch is
`2cfe12da558abb5c84fe2b3b4d1a0641c5edcebe739e3bd847545c093054e775`
on base `dfba7a78`. Its native build and all 3,471 unique shared checks pass;
there are no missing, duplicate, failed or skipped cases. The initial discovery
attempt failed before tests due to concurrent CMake discovery output and is
retained; the completed run supplies the result. All 31 affected tokenizer and
template model cases pass too. This combined build includes the separately
reviewed Gate and tokenizer changes. The standalone U4 source patch
`cdf5609c56eacb83cda798090ba86f6b32393cdd5713f6ed595f8975301533db`
retains the reviewed U4 fixes, adds the formatter-required namespace comment,
and excludes Gate; its pre-Gate norm parameter
contract remains four bytes. These source identities are distinct from the
frozen A/D measurement binaries.

The existing [image results](IMAGE-RESULTS.md) remain valid for unchanged image
paths. The changed MTP domain has its own formula, state, greedy and matched
speed evidence; it inherits no exception from target likelihood. The identical
target paths retain only the prospectively scoped existing interimplementation
limitation, with both strict Mia failures reported above. This report does not
qualify sm86, another encoding, a new device, full chat-answer completion or
general model quality.
