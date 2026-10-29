<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.5 record — Foundation work so far

**In progress (written 2026-10-28; M3.5 continues).** This record holds what
the foundation work of M3.5 proved and where the evidence is: the physical
memory modes, the generality follow-ups U1–U10 that the 2026-10-20 M3.6 review
asked for, and the F1–F14 fixes that the 2026-10-25 foundation review asked
for. It follows [the M3.6 record](m36-record.md). It is not frozen: model
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

Open at this record: U7 (fast paths by capability; only U7a's first GDN-gate
spike is done) and U9 (M4 entry); F10b's closeout (the DeepSeek IQ2_XXS quality
score, the full Spark check set and a speed confirmation of every family) and
F14 (the test-suite audit); the families, formats and media items that M3.5 has
always owned ([plan](plan.md)).

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

## Foundation review fixes (F1–F14)

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
| F10a | this commit | Hygiene and docs: see below. |
| F14, F10b | open | The test-suite audit; the closeout (DeepSeek IQ2_XXS quality, full Spark check set, speed confirmation of every family). |

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
tool-call qualification work; F10b's closeout. Candidates noted while writing
this record: `docs/experiments/fusion-ranks/READY-MATRIX-RESULTS.md` (86 KB) and
the other U6 reports repeat registration hashes and review wording and could be
reduced to their tables; patch 0010 of GGML can be narrowed to the one dispatch
split `benchmarks/vecq_upstream.cu` uses; `docs/engine-components.md` still
narrates the retired `PlanGraph`.
