<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.5 record — Foundation work so far

**In progress (written 2026-10-28; M3.5 continues).** This record holds what
the foundation work of M3.5 proved and where the evidence is: the physical
memory modes, the generality follow-ups U1–U10 that the 2026-10-20 M3.6 review
asked for, and the F-series fixes from the foundation reviews of 2026-10-25 and
2026-10-30. It follows [the M3.6 record](m36-record.md). It is not frozen: model
families, the remaining U7 and U9 items, formats and the closeout checks are
still owned by the living [plan](plan.md), which keeps each item's scope,
status and exit and links here for the results. The record is frozen when M3.5
closes.

[Decisions](decisions.md) own the contracts (D-121 to D-127), [model
support](model-support.md) claims only qualified profiles, and the experiment
reports under `docs/experiments/` retain their measured binaries, source pins
and limits; a later documentation commit does not relabel a run as an
execution of the final revision. Raw samples, logs and job telemetry stay
outside the repository (AGENTS.md, rule 8); on 2026-10-28 the slice F10a
deleted the raw dumps, run-once drivers and per-run pin bundles that the
U-series experiments had checked in (each experiment's README says what was
removed and that the files are in Git at `1564dcd6`).

## Range and status

| Range | Meaning |
| --- | --- |
| `ca32156b` (2026-10-20) | The plan for U1–U10, written after the M3.6 review |
| `ca32156b..265c821a` | U1–U10 and the memory modes; the first, reviewed fix pass `b7e6143b`, `f8f106a8`, `7c96e27b`, `265c821a` |
| `265c821a..1564dcd6` (2026-10-28) | The foundation review's F1–F13 fixes, one reviewed commit each |
| `1564dcd6..c36589b2` (2026-10-28) | F14 and F10a; F10b's final checks ran on `c36589b2` (2026-10-29) |
| `c36589b2..` (2026-10-29/30) | F10b's record, then F15's fixes of what it found, F10c and F16 |

Open at this record: U7 (fast paths by capability; only U7a's first GDN-gate
spike is done) and U9 (M4 entry); the
families, formats and media items that M3.5 has always owned ([plan](plan.md)).
The regressions F10b's final checks found are fixed or explained by F15
([F15](#f15-regressions-found-by-the-final-checks-2026-10-29)).

## Memory modes and U1–U10

| Item | Commits | Outcome |
| --- | --- | --- |
| Physical memory modes (D-124) | `87e8fb3b`, accepted by the owner 2026-10-25; F2 `f70ebdef` | `[memory] mode = "dedicated" \| "shared"`. Dedicated fills a pool of empty 2 MiB handles within the budget's headroom at start, recycles them on eviction and keeps swaps from fragmenting host memory; shared allocates on demand and releases on eviction, idle (`idle_release_seconds`) and pressure, and `POST /llmp/v1/memory/release` surrenders the working set. The default follows the device (dedicated on the GB10, shared on a discrete GPU). The accepted M3.6 exception for DeepSeek DSpark's C4 wall (RE-059) has this as its fix. |
| **U1** Engine seams | `61f8426f` | One budget owner held live by the scheduler; one engine-side backend factory (device index, GGML backend, cuBLAS handle, workspace); `runner_resources.h` names no GGML launch types; one unproven-work object cleared only by a positive stream fence (it wraps `HostCopyUnproven` and `KeepPinned`, which survive as thin wrappers; F4 made the contract fence-proof); `PagedNode` split along its driver-thread and catalog-call contracts; `tools/llmp_boundaries.py` checks kernel-library names. Exit met: a recovery test whose retirement is genuinely unprovable serves again. |
| **U2** Test-only paths and v0 removed | `dae227dd` | `PlanGraph`, `Qwen38Commit`, `V0ArtifactSource`, the v0 reader (`artifact/representation`, `format_map`), `artifact_corpus.py`, `LLMP_TEST_HOLD_READS` in the binary and the repeat-pending drafter ledger are gone; the `"spark-v0"` layout profile is `"spark"` version 1 (D-121); the shared suite passes without `m3-artifacts`. `benchmarks/import_fuzz.cc` now reaches `spec::Synthesize`, `import::Normalize`, `planner::Plan` and the v1 readers, with a checked-in seed corpus (`tests/unit/data/import_fuzz/`) and one recorded bounded run (`--iterations=10000`, seed `20261021`, 9 seeds, 2.36 s on `spark-b`, 0 failures, 2026-10-21). The retired v0 stores were deleted on both Sparks on 2026-10-23 ([storage cleanup](environment.md#retired-spark-data-2026-10-23)). |
| **U3** One generic import route (D-125, completed 2026-10-23) | `e9de1e83`, `3f0394ef`, `2f0b141d`, `8c692b12`, `a63a97ff`, `e1832086` and others | ModelOpt and MLX checkpoints use one mapping-driven normalization and publication pipeline; the checkpoint-specific Qwen3.8 normalizers and the derived `SourceMapping` registry are removed. **U3a**: one canonical mapping per architecture with source dialects (metadata, names, parameter chains, folds, selectors, layer schedules, drafter values); source architecture, execution architecture and mapping provenance are distinct; serving consumes stored specs and typed drafting contracts ([closeout](experiments/m35-generic-import/canonical-mappings.md), [dense dialects](experiments/m35-generic-import/dense-dialects-mxfp4.md), [source folds](experiments/m35-source-fold/RESULTS.md), [stored drafters](experiments/m35-stored-drafter/RESULTS.md)). **U3b**: generic part descriptors carry block, global and affine scales, biases and zero points, `{expert}` gathers per-expert tensors, and role/encoding rules plan folds, permutations and scale relayouts at import; Qwen3.8 ModelOpt and MLX [re-import parity](experiments/m35-generic-import/original-parity.md) covers all 323,607 named parts; ModelOpt MXFP4 keeps packed E2M1/E8M0, HF block FP8 keeps E4M3FN/F32 scales with checked geometry (D-127, [checks](experiments/m35-generic-import/block-fp8.md)). **U3c**: Llama3 RoPE frequency scaling through CPU and GPU, with the owner-approved Llama 3.3 NVFP4 [reference likelihood and behavior controls](experiments/m35-u3-quality/RESULTS.md); the original 256-token Fibonacci reply failed and stays unresolved, the registered [512-token supplement](experiments/m35-u3-quality/FIBONACCI512-RESULTS.md) passes. The distinct qwen3_5 MLX [quality screen](experiments/m35-u3-quality/qwen27/RESULTS.md) passes both repeat pairs and six raw-ID cases (its old empty-final chat smoke is retained). Granted: nothing about general Gemma NVFP4 quality, assistant numerics, new devices or encodings, batching, maximum context or competitive speed. |
| **U4** Numerics as component data, per device (D-126, completed 2026-10-23) | `3cf3118c`, `2b3452de` and the registration commits before it | Admission joins component roles, encodings and device evidence the same way for pinned and unpinned artifacts; a device without evidence runs the exact path and says so in its plan report; no `sm_86` numerical grant. Results ([component-numerics](experiments/m35-component-numerics/README.md)): the generic GB10 scalar schedule follows upstream reduction geometry for all 18 VecQ encodings (182 operand controls); the repaired Gemma2 and Gemma31 screens match all 1,070 original-A teacher scores and both bounded replies, and their three-pair adoption speed screens pass; the two-through-eight-column repair passes 178 upstream operand controls; the original Gemma31 A/D screen stays failed under all four unchanged bounds, the repaired R screen matches the 1,070 scores and its C2 speed screen passes (TTFT 0.996109, decode 0.970889, wall 0.987638); the 14 remaining transfer arms (Gemma26, Gemma31, Qwen GGUF, DeepSeek/DSpark) pass; the first C1 speed screen retained Qwen's failed 1.121499 TTFT median, and the Qwen MMVQ routing repair passed a five-row head/state control and the same three-pair bound. Final: native targets, 174 related checks and the 3,696-test shared suite pass without skips. Recorded failures stay recorded: the frozen Gemma26/Gemma31 assistant, DSpark, DeepSeek target and Qwen GGUF differences, both strict Mia comparisons and the unchanged-target limitation ([LLM results](experiments/m35-component-numerics/LLM-RESULTS.md)); [image transfer](experiments/m35-component-numerics/IMAGE-RESULTS.md) keeps both registered geometries and speed pairs. No new device, untested encoding, maximum context, general batching or full-answer quality claim. |
| **U5** Front-door and scheduling seams (completed 2026-10-22) | `5cde838f` (a), `d9fb7f6a` (b), `5f51b17c` (c), `0798f935` (d), `d53347d9` (e) | **a**: route-owned start/delta/finish/error/keepalive events; the Messages encoder has indexed blocks and incremental JSON inputs (the route is not enabled). **b**: owned structured message parts and checked native/interpreted projection keep text boundaries, tool IDs and JSON. **c**: scheduler-owned cohort, turn, drain and substitute policy with socket-free tests (3,483 shared checks, four model cases). **d**: activation and reclamation as named per-attempt phases (13 model swap, recovery and observer cases). **e**: one concrete `Llm` replaces the 70-hook adapter and tests drive the real engine on fake providers (3,525 shared, 461 AArch64 no-CUDA, 31 model cases across both Sparks). Every serving-loop change took the heavy review; reviews were clean. |
| **U6** IR and planner structure (completed 2026-10-25) | `3ad79f4a`, `1aca44e7`, `59a1f310`, `24849831` | One checked trait table (`src/ir/op_traits.inc`) supplies identity, arity, parameter validation, inference, write sets, layout, partition and decomposition dispatch; GGML validates its own extent limits; all 49 registered rewrite and batch patterns have enforced synthetic match and refusal controls; the boundary check finds no vendor or kernel names under `ir/` or `execution/`. **Revised by F6**: the exact-site recipe comparison U6 added to production ran on every plan miss, no row could match a real model, and dense decoders could not finish a prefill plan; it was removed, and the U6 rank rows (including `7c96e27b`'s re-stamp without re-measurement) are retired. The U6 reports under [fusion-ranks](experiments/fusion-ranks/README.md) are history. |
| **U7a** First shared GDN gate spike | `e0338017` | The 48-head and 16-row whitelist is gone, with checked storage bounds, non-reference-shape correctness and a passing registered 48x1 speed screen ([gdn-gates](experiments/gdn-gates/RESULTS.md)). The rest of U7 is open. |
| **U8** Image pipeline as data (completed 2026-10-23) | `c1953ab9`, `48bbb5ed`, `de3c1a50`; F8 `b5b1143d` | Qwen-Image's VAE decoder and DiT block wiring are mapping-owned stage-graph recipes (D-122); import-time recipes account for every source tensor as consumed or explicitly omitted. Exit 1 passed: identical images from the data-driven import (the pinned BF16 full image comparison and frozen pixel digest; the Q4 repeat screen proves completion and determinism only). Exit 2 passed as a bounded native component-import and stored-topology proof for the pinned Ming transformer and first-frame RGBA VAE, with no Ming-specific importer, compiler or runtime wiring; no composed Ming pipeline, serving or image quality is claimed (F8: the second configuration now imports through the one Qwen-Image VAE mapping). Final validation: the 72-path source manifest `0eca5914...`, both Spark native builds, 3,751 shared tests from two complete Spark splits plus three repaired fixtures and 111 focused final tests (not one full-suite rerun on one snapshot), 26 model checks across both Sparks ([evidence](experiments/m35-image-pipeline-data/RESULTS.md)). |
| **U10** Review leftovers (re-verified 2026-10-22) | `dfb22fd9`, `5ec811a4`, `0aa2ffde`, `3bc9a041`, `63cd170d` | Checkpoint, artifact and direct-I/O readers pin the selected regular object before opening it, and a buffered retry uses the same object; ordinary spill and restore keep drafter continuation features (logical replacement still clears them); each activation owns a paid receipt retained by its asynchronous eviction and handoff, and observer names and metadata are paid before allocation (3,458 shared checks and seven swap controls, five on real artifacts); existing regular files are pinned before write/create I/O (nine deterministic race controls). Count/catalog, pending-counter, `CheckOwners`, HTTP deadline/half-close and heap-trim findings are fixed or retired. |

Also landed in this range, outside the U numbering: the native tool-call
foundation (`13b73a91`, `87470b22`, `c7b1b142`, `252794d0`, `0b0fe276`;
[tool-calls.md](tool-calls.md)), with one bounded real-model and client
qualification on 2026-10-22 (six native Qwen API transactions and one pinned
OpenCode 1.19.3 task, [results](experiments/m35-tools/RESULTS.md); broader
served-model and client qualification is open), prefix reuse
([m35-prefix-reuse](experiments/m35-prefix-reuse/README.md)), native SHA-256 on
ARM (`ee300f6e`), seven-day idle conversation retention (`85408951`) and the
Ubuntu 26 harness restoration (`4a86b43f`).

## Foundation review fixes

A cross-verified review of `ca32156b..265c821a` found that broken image
serving, the memory modes, the agent (tool-call) path and the exact-site
comparison machinery in production kept the U-series from being a clean
foundation. The owner (2026-10-25) asked for every finding and suggestion to be
fixed; each slice landed as one reviewed commit, with verification in batches
(owner, 2026-10-26; `66ba9fc2`).

| Slice | Commit | What changed |
| --- | --- | --- |
| F1 | `0d01dda5` | A closed request-scalar vocabulary (every Qwen-Image request was refused at `265c821a`; its text encoder could not import); image generation in the default models set; dense RoPE scaling types mapped or refused instead of run as YaRN; the IMROPE capability aligned with its check; the 256-block scale-atom cap removed. |
| F2 | `f70ebdef` | Live pool target (the dedicated pool charged within the budget; pressure and release free backings first); the shared-mode shrink formula and floor; `SetBudget` as a scheduler command; a device-aware default mode and discrete sizing; D-124 accepted. |
| F6 | `0d86ee9b` | Exact-site comparison moved out of production to `benchmarks/` (about -27k `src/` lines); coarse, versioned `FusionRank` rows in `src/` (none shipped at that commit); a re-check at Qwen-Image VAE shapes kept the fused spatial channel norm (4.5-5.7x faster there). |
| F3a | `971a2c24` | Tool calls: bounded JSON depth, growth-funded output, one request-local refusal channel, one grammar shared by parser, sampler and judge, Llama leading-JSON calls, self-opened reasoning, marked forced spans, linear-time result association. |
| F4 | `e0aa811c` | One fence-proof unproven-work contract with recover-then-serve tests per owner; MTP drafting after plain steps; image fence acknowledgement and request layout; row-read settling; checkpoint staging re-reserved after a fence; history copies. |
| F5 | `2e38631c` | Output delivery as host-only state; one cohort decision function; kept-record commit by descriptor; clock-skew and lock-file edges; a startup allowance that no longer sums sequential phases; the shared-mode floor of the state requests hold; a memory release that waits at most one turn. |
| F3b | `999c2c48` | Device greedy and verify kept when the tool mask is the identity; lazy and class-based masks; template-derived openers; every declared tool rendered ([tool-sampling-speed](experiments/tool-sampling-speed/README.md)). |
| F9 | `5061c346` | Import publication 2-4x faster (one parallel post-write hash pass); reuse provenance; unclassified config keys refused; fuzzing of the newer routes; component-declared import transforms. |
| F12 | `49b2f0af` | Every model test that already failed at `265c821a` fixed, re-recorded with evidence or corrected (stale frozen digests, the Gemma Q8 cache admission, Gemma 4 tool rendering, the gemma2 store artifact, the Qwen3.8 quantized-head tests); drafter tap-row restore; env-gated skips run by default or are justified; the DeepSeek-to-Qwen3.8 8K continuation difference bisected and fixed ([frozen rows](experiments/m35-f12-frozen/README.md)). |
| F13 | `4348ba08` | Known open defects: RE-050 (head wider than the tokenizer), RE-052 (start-up guard against page cache), RE-031 (reference top-k ties), the metadata-funding and tap-row items and the M3.6 record's carried lifecycle, planner and test-infrastructure defects. |
| F8 | `b5b1143d` | A vendor-free selected-operation interface, no kernel-source include cycles and a stricter boundary check; one mapping value language, one Qwen-Image VAE mapping and a shared tensor-name dictionary. |
| process | `66ba9fc2` | Verification in batches; one-pass heavy-path review and challenge; a 900 s shared-set job. |
| harness | `4a86b43f` | Harnesses restored after the Ubuntu 26.04.1 upgrade (glibc 2.43, NVIDIA 595.99.02 on the workstation). |
| F7 | `3be174f0` | Measured rank rows instead of capability vetoes (GGML's MMVQ at 2-8 columns where faster, fused GLUs split where the decomposition is >5% faster; one-pass 8-column VecQ measured slower and dropped), upstream prefetch, a row-strided `llmp.argmax`, stable op identity in digests, cheaper inference and planning; geometric decode buckets built but off until attention skips masked tiles ([vecq-columns](experiments/vecq-columns/README.md)). |
| F13 follow-up | `1564dcd6` | Padding head rows bounded on the device in the heads component: every greedy argmax reads each row's first tokenizer-size logits in place, so padded heads (Qwen3.8, Qwen2.5) keep device greedy in joined waves (RE-050). |
| F10a | `c36589b2` | Hygiene and docs: see below. |
| F14 | `faa99893` | The test-suite audit: dead helpers out, duplicated bodies shared. |
| F10b | `79c6586a` | The closeout checks: [campaign final verification](#campaign-final-verification-f10b-2026-10-29); not passed. |
| F15b | `3c9a938f` | Prompt scores explained (RE-063); wide-output GLU rank rows (an `n` band): the Gemma 4 31B assistant screen −2.01% against M3.6. |
| F15a | `e4e5d8e1` | MTP prompt catch-up, prefill lookahead bounds, a chunk-weighted gate decode rate: every F10b speed failure passes against M3.6. |
| F10c | `ea1bbf5e` | The workstation `native` and `cpu-asan` tiers: [F10c](#f10c-workstation-tiers-2026-10-29); RE-051 fixed and the nine sanitizer findings it hid. |
| F16 | `098fcf89` | Plans build about a third faster with every layout unchanged: [F16](#f16-lowering-placement-cost-2026-10-30). |
| F17 | `ccd645a8` | Parser ownership and output funding: [F17](#f17-parser-ownership-and-output-funding-2026-10-30). |
| F18 | this commit | Image import funding and tensor indexes: [F18](#f18-image-import-funding-and-indexes-2026-10-30). |

## F10a hygiene (2026-10-28)

- **Raw experiment data (rule 8).** Deleted from `docs/experiments`: 28 raw
  dumps (the 343 KB implementation-identity registry dump, the per-run
  `pins.json` bundles with `/tmp` paths, identity maps and crosswalks, the
  relational-plan self-test, the source-change ledger, the identity-helper and
  compiler-delta receipts, the artifact-publication proof, job-telemetry
  sampling refreshes, metadata and source-hash inventories) and 44 Python files
  (12 test files no gate ran, and the run-once drivers, freeze and launch
  scripts of the per-model protocols). Kept: every README, RESULTS, PROTOCOL
  and registration file and every pre-registration (owner, 2026-10-25), the
  aggregates, and the harnesses a README rerun recipe needs
  (`m35-component-numerics/{capture,greedy_capture,plan_reports,reference_inputs,compare_image}.py`
  with `llm-capture-pins.json`, `m35-u3-quality/{reference,native,fibonacci512,collect_*,prepare_*,validate_reference_cli}.py`,
  `tool-sampling-speed/agent_e2e.py`).
- **One test-store resolver.** `LLMP_TEST_V1_STORE`, else the models
  directory's `artifacts`, for `tests/support/v1_frozen.h` and for
  `speed_gate.py`, `k1_quality.py` and `swap_table.py` (`speed_gate.v1_store`).
  The milestone-named `u2-v1-artifacts` and `m36-v1-artifacts` names are no
  longer read; both Sparks already link `artifacts` to the existing store.
- **Leftovers handed over from F14.** Doc links to the deleted
  `benchmarks/paging_cases.h` and the `OpLine` marker fixed; stale comments in
  `composition_test.cc`, `artifact.h` (it no longer reads v0), `exl3_linear_sweep.cc`
  and `gemma_engine_greedy.cc`; three unused constants removed from
  `v1_frozen.h`; the FP16 adapter's refuse-only options deleted; the ModelOpt
  publication test uses the fixture's 38 pins (`qwen38_modelopt_pins.h`).
- **Native-tier fix.** `Prefetches` in `kernels/ggml/vecq.cu` is `[[maybe_unused]]`:
  the `native` preset's sm_86 pass never references it (only the sm_121 pass
  does), and `-Werror` made nvcc's warning #177 an error; the GB10 SASS is
  unchanged.
- **Docs.** Tool-qualification wording, `EngineLlm` references, the release
  route in runtime-serving.md, stale `PlanGraph` comments, GGML patch 0010's
  current use ([upstream/ggml.md](upstream/ggml.md)), AGENTS.md "Current
  status" and the doc map.

## Campaign final verification (F10b, 2026-10-29)

**Not passed.** The final tree is faster than M3.6 for most families and every
swap is exact, but two families regressed in speed, the Gemma 4 31B assistant
screen fails, and literal-route prompt scores changed for Qwen3.8 native and
Gemma 4 while greedy decoding stayed byte-identical. Each item below needs an
owner decision or a fix slice; none is an accepted exception yet.

**Arms.** M3.6 gated against each host's frozen `1facc957-fixed` install, and
its final table ran `final-d373a1bf` (`53c2610b`). Neither can run any more:
the v0 stores were retired on 2026-10-23 and both installs' executables were
removed ([storage cleanup](environment.md#retired-spark-data-2026-10-23)). The
old arm is therefore `53c2610b` rebuilt from Git on each host on 2026-10-28
(`spark-native --locked`, the M3.6 SDK `aarch64-2c4eb5b2f68f3fa5`; runtime
`94e88837…` on `spark`, `e9e79c11…` on `spark-b`), serving M3.6's own v1 store
`m36-v1-artifacts` with M3.6's IDs. It reproduces every M3.6 quality value
exactly (NLL and score digest for all eight LLM configurations, and the image
pin `93356574…`), so it is the M3.6 final state. The new arm is `c36589b2`
(this record's tree; SDK `aarch64-8486b6bf58a61039`; runtime `44620d94…` on
`spark`, `c851db0b…` on `spark-b`), serving `artifacts` with the IDs in
`tests/unit/data/v1_frozen/v1-ids.json`. Both arms got the same per-family
settings (M3.6's gate templates, only IDs and store differ) and no diagnostic
switch: neither arm has the baseline defect that `LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS`
matched. The new arm runs in the default memory mode (`dedicated` on the GB10,
D-124).

**Protocol.** M3.6's tiered pass ([m36-record](m36-record.md#final-speed-verification),
[m36-final](experiments/m36-final/README.md)): a 2-block ABBA per family, passing
when every gated metric's mean slowdown is at most +0.5%; any metric above it
runs the registered gate (`speed_gate.py` minimum blocks, extended while
CONTINUE). Positive is slower. Each family ran wholly on one host, the clean
page cache of build trees, SDKs and both stores was dropped before every session
(RE-057) and the GPU temperature logged after every run.

**Hosts.** `spark` (`spark-c4e2`) and `spark-b` (`spark-56f5`), both Ubuntu
24.04.5, kernel 7.0.0-1019-nvidia, driver 580.178.04, unchanged by the
2026-10-28 update ([environment](environment.md#os-update-2026-10-28)). GPU
39–50 °C at each queue's start, 59–70 °C during the gates, SM clock 2398–2411 MHz.

| Family (host) | Tier, blocks | Verdict | Speed against `53c2610b` (slowdown; mean, upper bound where gated) | Peak, new minus old | Quality against M3.6 (new arm; quality control passage) |
| --- | --- | --- | --- | --- | --- |
| Gemma 2 2B (spark) | 1, 2 | PASS | every metric faster: decode −4.3%/−5.5% (C2/C1), prefill −1.9%/−4.0%, wall −4.1%/−4.2% | +4.13 GiB | NLL 3.5649091 against 3.5635005 (PPL +0.14%); replies equal; per-target agreement 93.1% (tripwire 95%) |
| Gemma 3 4B QAT (spark) | 1, 2 | PASS | every metric faster: decode −3.1%/−4.6%, prefill −1.0%/−3.5% | +3.09 GiB | NLL 3.6733320 against 3.6681064 (PPL +0.52%; M3.6's accepted exception, now +0.62% from its baseline); replies differ |
| Gemma 4 26B-A4B (spark) | 1, 2 | PASS | every metric faster: C1 decode −5.7%, C4 decode −2.2%, C4 prefill −0.7% | +18.15 GiB | **NLL 9.9205757 against 9.7916674 (PPL +13.8%, mean abs. per-target 0.86)**; replies equal |
| Gemma 4 31B (spark) | 2, 9 | **PASS** | C4 decode +0.55%, upper +0.94% (CONTINUE at 7 and 8 blocks: +0.60%/+1.12%, +0.58%/+1.02%); C1 decode −3.7%, prefill −0.3%, C4 prefill −1.1%, wall −1.1%/−2.3% | +22.79 GiB | **NLL 9.7004038 against 9.5749964 (PPL +13.4%, mean abs. 0.72)**; replies equal |
| Gemma 4 31B assistant (spark, `assistant_gate.py`, 8 blocks) | — | **FAIL** | **+3.58%, bound [+3.12%, +4.05%]** (1.3507 s against 1.3045 s for 32 committed tokens; every block +3.0% to +4.5%; the same verified and drafted counts) | — | not scored |
| Qwen3.8 native + MTP (spark-b) | 1, 2 | **FAIL** (not extended) | **decode +31.2%/+26.5%/+26.8% (C1/C2/C4; 49.36 → 37.62 tok/s at C1), wall +12.5%/+6.5%/+11.5%, C4 prefill +5.8%**; C1/C2 prefill −2.8%/−1.6% | +23.89 GiB | **NLL 2.8031018 against 2.7564455 (PPL +4.78%, mean abs. 0.50)**; replies equal |
| Qwen3.8 GGUF (spark-b) | 2, 4 | **FAIL** | **C1 prefill +1.80%, upper +2.35%** (6.508 → 6.625 s; +1.3% to +2.0% in every run); decode −4.9%/−6.4%/−11.8%, C2/C4 prefill −1.2%/−0.2%, wall −0.8% to −2.8% | +47.74 GiB | NLL 2.8432115 against 2.8596571 (PPL −1.63%, mean abs. 0.32): M3.6's accepted exception, now on the other side of llama.cpp b11254 by about the size of llama.cpp's own fused/unfused spread (1.75%) |
| DeepSeek V4 Flash, plain waves (spark-b) | 1, 2 | **FAIL** (not extended) | **C2 decode +10.1% (38.18 → 34.68 tok/s, every run)**; C1 decode −9.0%, C4 decode +0.04%, C2 prefill +0.50%, others at most +0.4% | +4.69 GiB | NLL 2.7937083 against 2.7939955 (PPL −0.03%, mean abs. 0.10); replies differ (U4's recorded scalar-schedule transfer) |
| DeepSeek V4 Flash, DSpark waves (spark-b) | 2, 5 | **PASS** | C1/C2/C4 decode −10.0%/−10.9%/−17.6%; **C4 wall −3.95% (M3.6's accepted +0.94% exception is resolved, RE-059)**; C2 prefill +0.21%, upper +0.77% (CONTINUE at 3 and 4 blocks) | +4.83 GiB | as the plain waves |
| Qwen-Image-2.1 (spark) | 2, 8 | **PASS** | `generation_s` +0.61%, upper +0.91% (34.13 → 34.34 s; CONTINUE at 6 blocks, +0.58%, upper +1.02%) | +37.11 GiB | pixels `93356574…` in every run of both arms (the M3.6 pin; its PSNR/SSIM carry over) |
| Llama 3.2 1B Q8_0 (spark-b, new baseline) | — | baseline | C1 decode 149.4, C4 decode 383.5 tok/s; prefill 0.487/2.040 s; wall 95.7/153.3 tok/s | peak 6.15 GiB | not scored |
| Llama 3.2 1B Q4_K_M (spark-b, new baseline) | — | baseline | C1 decode 208.5, C4 decode 453.0 tok/s; prefill 0.484/1.985 s; wall 117.0/166.5 tok/s | peak 5.23 GiB | not scored |
| Qwen2.5 0.5B FP16 (spark-b, new baseline) | — | baseline | C1 decode 203.6, **C4 decode 203.8** tok/s; prefill 0.797/3.148 s; wall 90.1/93.4 tok/s | peak 4.35 GiB | not scored |
| Qwen2.5 0.5B EXL3 4.0 (spark-b, new baseline) | — | baseline | C1 decode 251.4, **C4 decode 253.1** tok/s; prefill 2.242/8.662 s; wall 46.7/48.6 tok/s | peak 3.12 GiB | not scored |

The dense-decoder rows are one block of four runs on the new arm (run-to-run
sd at most 0.6% except C1 prefill 1.5%), through two gate families this slice
added (`llama32`, `qwen25`: `[models]` context 8,192, chunk 4,096, four slots,
26,000-character prompts, C1 and C4); they are the baseline a later gate
compares with. Qwen2.5's approval admits one owner: four slots run "served,
unqualified" and its C4 cells decode no faster than C1 (a C4 prefill is four
C1 prefills), so Qwen2.5 does not batch.

Peak memory is the whole-node `MemAvailable` drop. The new arm's dedicated
pool fills the budget's headroom at start, so its peaks are larger by design;
it leaves 2.3–3.2 GiB available beside Qwen3.8 and DeepSeek (old arm 7–51 GiB).
That margin is what RE-057 warns about: block 3 of Qwen3.8 GGUF and of DSpark
first ran right after a bisect build whose tree the sessions' cache drop did not
cover; one GGUF new run prefilled 2.2× slower and one DSpark priming run lost its
C4 decode window. Both blocks were set aside and rerun after the drop covered
that tree.

Qwen3.8 native and DeepSeek plain waves were not extended to the registered
gate: their 2-block lower bounds already exceed +1% (C1 decode +28.4%; C2 decode
+5.8%), so more blocks could only confirm the failure.

**IQ2_XXS.** `speed_gate quality` on DeepSeek V4 community IQ2_XXS (v1
`a4fb10cc…`, plain decoding, context 262,144): NLL **2.8566161** (PPL 17.40,
966 targets, scores `b6472f2e…`). The same checkpoint on `53c2610b` (M3.6's v1
`e7c0d0e4…`) scores 2.8490313 (PPL 17.27): +0.76% PPL, mean absolute per-target
difference 0.091 (DeepSeek's M3.6 tripwires, 0.5% and 0.05, fail; the UD-Q2_K_XL
target's same comparison is −0.03% and 0.104). Against UD-Q2_K_XL on the new arm
(2.7937083) it is +6.49% PPL, the expected cost of the lower-bit file. ds4, the
pinned reference for this checkpoint, has only its recorded 32K War-and-Peace
PPL (2.980670302, [ds4-study](experiments/ds4-study/README.md)); no ds4 build or
retained losses exist for the gate passage, so no same-passage reference
comparison was made. F12's re-recorded IQ2_XXS tokens therefore rest on a
+0.76% likelihood shift that matches the target's own U4 transfer in kind but
is larger in aggregate.

**Swap table** (`spark-b`, `swap_table.py`, the current v1 IDs, image pin
`93356574…`, six jobs of one ordered pair): **PASS**, 32 of 32 rows exact and
state-exact, every restored 8K hash and continuation equal, prepared returns kept
and replayed their graphs, every image regeneration equal to the pin. Worst
prepared LLM↔LLM row **8.749 s** (deepseek to qwen3.8, B→A, 0 context) against
the 20 s bound (M3.6 8.505 s). Worst LLM↔LLM row 16.109 s, qwen3.8 to deepseek
first use with 8K (M3.6 8.665 s): its eviction took 6.368 s; deepseek to image
first use 8.913 s (evict 4.612 s; M3.6 4.124 s). Image to deepseek first use
improved to 11.319 s (M3.6 27.354 s). Largest tracked peak 115.32 GiB (M3.6
111.73), lowest MemAvailable 2.27 GiB (M3.6 4.69). The first-use rows stay
within their 40 s target, but the slow first evictions beside the dedicated
pool are new. The Gemma near-8K cases were not rerun.

**Check set.**

| Check | Host | Result |
| --- | --- | --- |
| `spark-native --locked` build of `c36589b2` | both | no-op (already built by F14+F10a's check) |
| Shared set, `-LE 'models\|timing'` | spark | cited, not rerun: 3,965 of 3,965 (job `merge-shared`, the F14+F10a combined check on this tree; F10b changed only a benchmark harness, a gate tool and docs) |
| CPU-only guardrail (`spark-native` with `LLMP_CUDA=OFF`, fresh `build/spark-cpu`, current SDK) | spark-b | 3,426 of 3,426 (a first run in the old cache's directory failed `sources.closure` on objects whose dependency files named the previous SDK; a fresh directory passed) |
| `models\|timing`, Qwen3.8 | spark-b | 32 of 32 |
| `models\|timing`, DeepSeek and DSpark | spark-b | 21 of 21 |
| `models\|timing`, tokenizer, importer and dense decoders | spark-b | 128 of 128, 3 skipped (read below) |
| `models\|timing`, image | spark | 15 of 15 |
| `models\|timing`, everything outside the named areas | spark | 17 of 17 |
| `models\|timing`, Gemma 2/3/4 and assistants | spark | 41 of 41 (one 600 s job reached 17, the other 24 ran in three jobs) |
| All of `models\|timing` | both | 239 distinct tests (`ctest -N`), every one run: 236 passed, 3 skipped, 0 failed |

Skipped, each read in its log: `ImportModelOptModelsTest.ClosedNativeTargetOwnsEverySourceAndPreservesFormats`,
`ImportModelOptModelsTest.PublishesPinnedNativeTargetWithCompleteSourceAuthentication`
(confined proofs: a delegated cgroup and the ModelOpt source and stores) and the
publication half of `ImportMlxNormalizeModelsTest.RealStagedCheckpointNormalizesPublishesAdmitsAndPlansAffineSplitWeights`
(`LLMP_TEST_MLX_PUBLICATION_STORE`): F12's three opt-in proofs
([m35-f12-frozen](experiments/m35-f12-frozen/README.md#opt-in-tests)).

**Where the regressions came from** (bisected on `spark-b` with `53c2610b`'s
history built in a scratch tree; Qwen3.8 native quality control, the store each
commit reads; the two Qwen3.8 v1 artifacts share their weight, index and spec
files, and only the manifest's layout-profile name differs):

- *Prompt scores.* Good at `d3374c67` (NLL 2.7564455, the M3.6 value), changed
  at `e6c3617a` ("Serving: one LLM route, prefill policy in the engine": 2.8031020
  with `b003c9a5`'s one-line scoring fix applied; unpatched, `e6c3617a` and
  `0d28a553` answer every literal score request with HTTP 500 "literal prompt
  rows arrived out of order", which `b003c9a5` fixed), unchanged since. The
  greedy replies are byte-identical at every point, the scored token IDs and
  counts are equal, and the differences start at the first scored row (position
  1: −15.19 against −15.58 nats) and persist through the passage (mean absolute
  difference 0.27 to 1.0 nats in each run of 100 positions). Qwen3.8 native
  and Gemma 4 26B/31B move the most; Gemma 2 hardly. No served-output quality
  claim was re-established for multi-row scoring after `e6c3617a`.
- *Qwen3.8 MTP acceptance* (the decode regression). Still M3.6's at
  `aa92bd1f` and `639583b9` (quality-control replies: 152 of 244 drafts
  accepted; `aa92bd1f`'s speed run 1,655 of 2,088 and C1 decode 49.5 tok/s);
  already changed at `381eec18` (151 of 274) and on `c36589b2` (speed runs 1,371
  of 2,678, 51%). The 101 commits between include U2's re-import, U3 and U4;
  U4's own record notes that "HC and NVFP4 expert fusions differ at stage 1,
  rows 1–5" for the native MTP drafter and that target likelihood does not
  measure the drafter ([U4 LLM results](experiments/m35-component-numerics/LLM-RESULTS.md)),
  and no acceptance or decode-speed comparison against M3.6 followed. Commits
  before `381eec18` cannot read the current drafter artifact (its stored spec
  gained a `drafting` key), so the bisection stops there.
- *DeepSeek plain C2 decode.* Matched `53c2610b` at `aa92bd1f` (R13's table);
  at `381eec18` one speed run gives C2 decode 34.50 tok/s (the regressed value;
  C1 33.69 and C4 50.24, before the later C1 gain). Window `aa92bd1f..381eec18`,
  not narrowed further.
- *Qwen3.8 GGUF C1 prefill* and the *31B assistant screen*: not bisected. At
  `aa92bd1f` (R13) the GGUF C1 prefill was already +3.44% and the assistant
  screen −3.15% against `53c2610b`.

**Changes made by F10b** (in its commit):

- `benchmarks/gemma_engine_greedy.cc`: the assistant screen's harness builds
  its `ModelSettings` by hand and never set `draft_rows`, which the component
  build reads since `381eec18`, so every speculative run refused ("draft depth
  must leave room for its anchor"). It sets 3, the screen's depth. No test
  covers the harness (it is a benchmark); the 8-block screen above is its check.
- `tools/speed_gate.py`: `plan --old-image-pixels-sha256` (and `run
  --image-pixels-sha256` for either arm), so an old image arm that is not the
  frozen baseline names its own pin; the `llama32` and `qwen25` families. Tests
  in `tools/tests/test_speed_gate.py`; protocol text in
  [m36-baselines](experiments/m36-baselines/README.md).
- `docs/engine-components.md`: the retired `PlanGraph` narrated as history.

Raw records, logs, the frozen arms, the `53c2610b` build tree and the bisect
tree were on the Sparks under `~/jobs/f10b` and `~/src` and are deleted
(AGENTS.md, rule 8); this section is the aggregate.

## F15: regressions found by the final checks (2026-10-29)

Two slices, F15a on `spark-b` and F15b on `spark`, each with one review pass,
checked together on the merged tree. Gates are tier-1 (2-block ABBA, +0.5% a
metric) against `53c2610b` rebuilt from Git and serving `m36-v1-artifacts`,
as in F10b; positive is slower.

| Item (F10b) | Root cause | Fix | Gate after |
| --- | --- | --- | --- |
| Qwen3.8 native + MTP decode, about 24% lower | `dae227dd` (U2) deleted the repeat-pending prompt companions as test-only; serving needed them, so the MTP drafter's KV for prompt rows was never written (acceptance 79% → 51%) | `DecodingOrchestrator::CatchUpPrompt` after each prompt chunk writes the drafter's rows from the target's taps (each chunk's first row excepted, on every path); score-only prompts skip it | acceptance equal to M3.6; 4 blocks, every metric faster (decode −0.26/−0.21/−0.33%, prefill −0.91/−0.90/−1.49%, wall −0.61/−0.28/−0.25%) |
| Qwen3.8 native C4 prefill (+5.8%, +8.2% with the catch-up) | `87e8fb3b` made a speculative cohort's last prompt unit build plans for every later exit and joined verify shapes of one width, which adaptive MTP depths never match | `Predict` builds only the ready cohort's continuation, and joins decoding peers only when the verify width is fixed (DSpark's block waves, which `87e8fb3b` was for) | as above |
| Qwen3.8 GGUF C1 prefill +1.80% | the lookahead recorded the prompt's one-row tail graph ahead (135 ms of GPU idle against 25.6 ms) | a chunk records the next chunk's graph ahead only when that chunk is full (the full chunk as `PrefillRows` cuts it) | +0.40% (C2/C4 −0.96/−0.23%, decode −4.9% to −13.0%) |
| DeepSeek plain C2 decode +10% | not the engine: `53c2610b` coalesced tokens into SSE chunks before the common window, and `decode_rate` gave every chunk the reply's mean tokens per chunk | `speed_gate.py` records each chunk's text length and credits in-window chunks by it; mixed records are INVALID ([protocol](experiments/m36-baselines/README.md)) | +0.77% by length (noisier where the arms' replies differ, U4's transfer); window step pace +0.10% (C2), −2.48% (C4), the verdict used |
| Gemma 4 31B assistant screen +3.58% | `3cf3118c` bound Q8 vector-reduction evidence to every GB10 site, moving the 31B's wide FFN GLU (5376×21504) to `llmp.vecq`'s fused GLU; F7's rows, decided on narrower shapes, kept it at 2-4 rows | fusion rank rows gain an output (`n`) band; GLU rows for 8,192+ outputs split from 3 rows (Q4_K, Q6_K) or 4 (Q5_K) ([wide outputs](experiments/vecq-columns/README.md#wide-outputs-2026-10-29)) | 8 blocks −2.01% [−2.28, −1.74]; Gemma 4 31B every metric faster (C4 decode −6.00%) |
| Prompt scores (Gemma 4 PPL +13–14%, Qwen3.8 native +4.78%) | not a defect: since `e6c3617a` literal prompts score in `prefill_chunk` chunks instead of one row a step, and each arm is bit-exact to llama.cpp at its own geometry | none; RE-063 (compare scores at one `prefill_chunk`), [runtime-serving](runtime-serving.md) corrected | `prefill_chunk = 1` reproduces M3.6's digests |

Also gated on the round-1 binary: DSpark (decode −10.1% to −19.3%, wall
−2.8% to −4.2%), Gemma 2 2B (every metric faster) and Llama 3.2 1B Q4_K_M
against `79c6586a` (level; F15b's intermittent C4 prefill 60 ms did not recur
in 2 blocks). No frozen digest changed. The `draft_rows = 2` slow decode seen
once was a probe started after a build without the cache drop (RE-057).

Left as a follow-up: a target plan's lowering costs about 7 ms more than at
`53c2610b` (46 → 59 ms a Qwen3.8 prefill plan at HEAD, 73 ms at `d3374c67`):
`d3374c67`'s physical placement runs twice (`planner::PlaceActivations`'
greedy fit), and its first result lays out DeepSeek's HC carries, so it cannot
simply be skipped. F16 took it up ([F16](#f16-lowering-placement-cost-2026-10-30)).

Checked on the merged tree (2026-10-29): `spark-b`, the shared set 3,974 of
3,974, `-L models -R Qwen38` 32 of 32 and `Dsv4` 21 of 21; `spark`, the
targeted rank, lowering, prefill, decoding and scoring tests 343 of 343,
`Gemma4` 20 of 20 and `GemmaAssistantV1|GemmaNativeV1|DenseDecoder|Gemma31`
4 of 4; no skips. The swap table was not rerun: its exact continuations are
the kept-conversation model tests above.

## F10c: workstation tiers (2026-10-29)

F10's last item: the cuBLAS RUNPATH and `-fno-sanitize=function` fixes (F13)
and the importer under ASan, confirmed on the workstation (Ubuntu 26.04.1,
glibc 2.43) with `tools/build test --locked --fresh <preset>` under
`hostlock shared`.

- **`native`** on `e4e5d8e1`: 3,506 of 3,506 passed, GPU tests skipped by the
  preset. The CUDA test binaries, the cuBLAS ones included, load and run.
- **`cpu-asan`**, first run: 591 of 3,443 failed. Every server, lane or
  engine test whose threads end died in ASan's thread teardown (RE-051:
  `InstallThreadSignalStack` replaced the stack ASan had installed, which ASan
  then failed to unmap). The importer mutation test
  (`ImportAdversarialTest.ConcurrentMutationDuringStreamingDetected`) passed.
- **After the RE-051 fix**, nine failures remained, none seen before because no
  sanitizer run had got that far:

  | Finding | Where | Fix |
  | --- | --- | --- |
  | Signed overflow (UB) | `api_test.cc`, a test's ID generator | 64-bit product |
  | Invalid `ggml_type` load (UB) | `gemma_moe_test.cc` wrote `-1` | `GGML_TYPE_COUNT`, a valid enumerator that is no type |
  | Use-after-free | `plan_cache_test.cc` read an entry after reclaiming it | keep the serial |
  | Timing-mark leaks (3 tests) | fixtures whose refused `Setup` never tore the opened node down | tear down the unstarted node |
  | Leaks of deliberate retention (3 tests) | a dropped supplemental run's state, a retired ring and its landings after an unknown completion, never freed by design but unreachable | `base::RetainForProcess` keeps them reachable (docs/engine.md, "Unproven work") |

- **`cpu-asan`, final:** 3,445 of 3,445 passed. In both presets, the two skips
  (`QwenImage.InstalledComponentsBind`, `ImageFamily.ThePublishedPipelineIsAdmitted`)
  need the published Qwen-Image store, which only the Sparks hold.

Checked on the tree merged with F16 (2026-10-30): `spark`, the shared set
3,977 of 3,977 and `-L models -R Qwen38` 32 of 32; `spark-b`, `Dsv4` 21 of
21; no skips.

## F16: lowering placement cost (2026-10-30)

F15's follow-up, one builder and one review-and-challenge pass. Time per
step of a Qwen3.8 native prefill plan at `e4e5d8e1`, from temporary
instrumentation on `spark-b` (since removed), 61 ms in all:

- **Graph digests, 13 ms.** The hasher fed SHA-256 a byte at a time: 5.1 ms
  for the source graph in `Plan()`, 5.5 ms more when `LowerComponentPlan`
  hashed the source again (the clone never carried its digest back), and
  2.7 ms for the planned graph.
- **The planner's pattern step, 14 ms.** `fusions::GroupedProductOf` scanned
  the graph for every product before its cheap checks (about 28% of `Plan`
  in `perf`), and the merge of the pattern rounds asked every rewrite of
  every node.
- **Placement, 4.6 ms** for all three fits (the planner's logical one and
  the lowering's two physical ones, before and after its schedule), plus root
  lookups and a storage contract built twice.

The fix, with every output identical:

- `src/ir/graph.cc`: the hasher gathers fields into 512-byte blocks;
  `PlanComponentGraph` digests the source before cloning it, so it is hashed
  once a plan.
- `src/kernels/ggml/lower.cc`: the physical storage contract is built once
  (the second placement drops the reads of steps the schedule does not
  launch), `WithinRoot` and `WrittenBytes` are memoized per root, a plan
  without a prologue skips the preserved-root scan, and `WithinRoot` checks
  the gated-norm chain's shape before counting readers.
- `src/planner/placement.{h,cc}`: `FitIntervals` indexes short lifetimes by
  their first step and keeps the offset-ordered scan for plans whose mean
  lifetime is over 16 steps (lanes). A segment tree was 2× faster on prefill
  plans but 2.7× slower on verify plans, measured on 400 real fit inputs,
  and was rejected. The first placement stays: it lays out DeepSeek's HC
  carries and the provisional bind.
- `src/planner/fusions.cc`, `src/planner/pattern.cc`: cheap operand checks
  before the graph scans; the pattern merge indexes each node's makers.

Plan time, Qwen3.8 native, median per plan (ms, same instrumentation,
`spark-b`; all-cell runs vary by about ±5 ms):

| Plan | `e4e5d8e1` | F16 |
| --- | --- | --- |
| Prefill target (≥512 rows) | 58.4 (all cells), 61.2 (C1) | 38.8–40.6 (all cells) |
| Decode target, 1 row | 42.2 | 27.7 |
| Verify, 2–8 rows | 48.6 | 34.6 |
| Verify, 9–511 rows | 73.4 | 52.0 |
| Draft | 4.4 | 2.8 |

Planning per gate run, uninstrumented: Qwen3.8 native 6.19 → 4.47 s (24.8 →
17.7 ms a plan), DeepSeek 8.91 → 7.04 s (127.5 → 100.5 ms), Qwen3.8 GGUF
2.00 → 1.36 s, Gemma 4 26B 2.07 → 1.61 s, Gemma 3 1.31 → 1.01 s, Gemma 2
1.57 → 1.30 s. The target, `53c2610b`'s 46 ms a prefill plan, is met.

Gates, tier 1 against `e4e5d8e1` built on the same host after a cache drop
(every family, since the digest change moves every family's plan time by
more than 1 ms):

| Family (host) | Blocks | Result |
| --- | --- | --- |
| Qwen3.8 native (`spark-b`) | 4 | every mean ≤ +0.11% (C4 decode +0.65% at 2 blocks) |
| DeepSeek plain (`spark-b`) | 4 | pass; C4 decode +0.05% (+0.55% at 2 blocks) |
| Gemma 2 (`spark`) | 6 | pass; C1 prefill +0.39% (+0.71% at 2 blocks, +0.51% at 4) |
| Gemma 3, Gemma 4 26B, Gemma 4 31B, Qwen3.8 GGUF, Qwen-Image (`spark`) | 2 | every mean ≤ +0.17%; the image pixels match the frozen pin in both arms |

Tests: placement pins (`tests/support/placement_digest.h`, recorded with the
old placement code) for the three Qwen3.8 forms and DeepSeek's target and
draft shapes, including the `sinkhorn_hc_pre` carry path; and
`PlannerPlacementTest.TheIndexedFitPlacesEveryIntervalWhereTheScanDid`
against the scan on 400 random interval sets. No frozen digest changed. The
review found no defect; it made the fit's step-length sum unsigned (an int64
overflow no real plan reaches, confirmed under UBSan) and checked the fit
against a copy of the old scan on 200,000 fuzzed sets.

Checked on the tree merged with F10c (2026-10-30): `spark`, the shared set
3,977 of 3,977 and `-L models -R Qwen38` 32 of 32; `spark-b`, `Dsv4` 21 of
21; no skips. The builder's gate trees and records on both Sparks
(`~/.local/share/llmp/f16-gate`, `~/src/llmp-f16base`) are deleted.

## F17: parser ownership and output funding (2026-10-30)

The parser worker returns its parsed result without changing channel state.
The I/O owner publishes streaming intent only after model and context
validation, preserving ordinary HTTP errors for large half-closed requests.

Chat output prepays fixed decoder state and funds unread token capacity and
conservative tool-parser growth for the next causal unit at a completed
boundary. Publication uses the backed grant. Replacement pays both old and
new vector storage, and paused continuations retain their payment. Refusal
cancels only that request before its next unit; a reused generation result is
cleared before an initial refusal. The token allowance no longer prepays
`32 * max_tokens` bytes of output that may never be produced.

On `spark`, native runtime/CLI builds and 155 focused regressions pass without
skips. Three parser half-close/cancellation regressions, repeated ten times
under cross-built ThreadSanitizer, pass without race reports. The merged
F17–F20 native shared set passes 4,003/4,003, without skips. Targeted workstation
ASan/UBSan checks of F17–F19 pass 606 tests; one published-image admission test
skips because the workstation lacks its composition, and passes on `spark`.

## F18: image import funding and indexes (2026-10-30)

Image preflight prices all copied metadata values, lists, paths, ignored tensor
headers and skipped inventories before family synthesis. JSON text and parse
workspace are funded before reading; flattened key/payload bytes are counted
before materialization. Planning consumes the same funded family and parsed
file receipts. The text encoder's profile-only bound precedes descriptor
synthesis, including oversized or sparse-source refusal paths.

Borrowed source-name indexes and an expanded `(name, part)` index replace
repeated tensor scans, preserving ambiguity and raw-dimension checks. Recipe
work accounting visits source-reference leaves in scalar parameters, conditions,
raw axes, groups, controls and geometry. Conservative multiplicity includes
repeated passes and lazy branches without forcing dormant template evaluation.

Native checks pass 44 focused tests without skips, and the real Qwen-Image
pipeline import proof passes on the final merged tree. Work-bound counts for
256/512/1,024 source tensors are 706,550 / 1,558,674 / 3,614,370 comparison
units. A raw-contract plan grows from 76,150,667 to 165,701,569 units when
bindings double from 1,024 to 2,048 (2.18×). These are deterministic charged
work counts, not measured timing improvements. Existing artifact identities
and published specs are unchanged.

## Owner decisions and standing rules so far

- D-124 accepted 2026-10-25 as amended; D-121 (choose the better architecture,
  no migration before the first shipment) governs every item above.
- Verification runs in batches and heavy-path changes get one review-and-challenge
  pass (owner, 2026-10-26; workflow.md "Check cadence").
- Pre-registration and protocol files are kept (owner, 2026-10-25), though
  D-085 makes pre-registered protocols a non-default.
- Retention: idle conversations 168 h (`85408951`).

## Handed on

Carried by the plan: U7 (b-e) and U9; the families, formats, media and
tool-call qualification work.
Candidates noted while writing
this record: `docs/experiments/fusion-ranks/READY-MATRIX-RESULTS.md` (86 KB) and
the other U6 reports repeat registration hashes and review wording and could be
reduced to their tables; patch 0010 of GGML can be narrowed to the one dispatch
split `benchmarks/vecq_upstream.cu` uses.
