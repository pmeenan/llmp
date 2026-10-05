<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Larger native Gemma prefill screen

Larger chunks reduce native prefill time, but this bounded screen does not
qualify a new policy. All three matched bookends retain strict greedy
mismatches. Production row caps, optimization defaults and quality allowances
are unchanged. The [protocol](PROTOCOL.md) describes the paid work;
[results](results.json) retain exact counts, hashes and aggregate comparisons.

The reference used the C-API default `swa_full=true`, with a full-length
local cache. These historical measurements retain that recipe. The later
[dense31 ring-cache screen](../gemma-swa-ring-h1/README.md) matches the normal
CLI/server setting `swa_full=false`: its final head is byte-exact and all 32
decode choices agree, but native prefill/decode take 16.20%/2.05% more time.
The independent [Gemma26 ring-cache comparison](../gemma26-swa-ring-transfer/README.md)
also matches all 32 decode choices, but both retained full heads differ and
native prefill/decode take 57.64%/0.49% more time. The speed and coarse memory
ratios below do not establish parity for a production ring-cache recipe.

One physical Spark (`spark-c4e2`, SSH `spark`) ran both engines on native base
`c2d2147`. Context is 16,384, KV is F16, masks are device-built, reference
fusion and CUDA graphs are enabled/allowed, and no drafts are used. The same
8,227 explicit IDs from War and Peace are supplied to both engines, SHA-256
`6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b`.
The [earlier protocol](../gemma-prefill/PROTOCOL.md) records corpus provenance.
Setup/loading and six discarded warm rows precede the paid fresh request.
Prefill consumes 8,192 rows; three common untimed inputs
`[236761,108,236913]` bring the prefix to 8,195. The 32 paid fixed-history
units end at 8,227. Staging, plan construction, full-vocabulary publication,
CPU argmax and completion are paid. Head/state file writes are outside timers.

## Paid execution

All times are seconds. The first screen is separate from the adjacent
reference/native/reference bookend. Native128 computes and publishes 63
intermediate heads, native1024 seven, native256 31; reference publishes only
the final prefill head. No intermediate native work is suppressed.

| Profile / policy | Native first / repeat prefill | Native first / repeat 32 units |
| --- | ---: | ---: |
| 26B ordinary128 | 6.86265 | 0.692408 |
| 26B ordinary1024 | 3.43966 / 3.43826 | 0.705680 / 0.705870 |
| 26B all1024 | 4.21685 / 3.81456 | 0.746661 / 0.679228 |
| 31B ordinary128 | 14.0096 | 3.17100 |
| 31B ordinary256 | 12.6528 / 12.5922 | 3.17042 / 3.17052 |
| 31B both256 | 12.5624 / 12.5869 | 3.15931 / 3.16460 |

| Matched bookend | Reference A / native / reference B prefill | Reference A / native / reference B 32 units | Native vs mean prefill / decode |
| --- | ---: | ---: | ---: |
| 26B ordinary1024 | 2.82745 / 3.44366 / 2.83467 | 0.863343 / 0.702542 / 0.866329 | +21.639% / −18.766% |
| 26B all1024 | 2.77688 / 3.82054 / 2.63661 | 0.839639 / 0.675438 / 0.840689 | +41.149% / −19.606% |
| 31B both256 | 12.6234 / 12.5804 / 12.4765 | 3.88918 / 3.18025 / 3.89147 | +0.243% / −18.252% |

26B's all-policy reference prefill bookends move by about 5.18%; 31B's move
by 1.18%. The 31B prefill difference lies inside that movement. These short
rates do not establish parity across depths or production batching.
Reference ubatch1024 (26B) and 256 (31B) were fastest among the previous
profile-specific screens' tested values; this task does not assert global
optimality. Reference `ggml_graph_reused_delta=32` measures GGML reuse,
not a CUDA replay counter. Native cap1024/256 reports one capture and 33
replays; cap128 reports 33 captures and 33 replays.

The ordinary policy requests no optional fusions. 26B `all` selects 60
norm/RoPE, 90 norm/residual, 30 routing and 30 scaled-reduction steps per
reported full chunk. 31B `both` selects 120 norm/RoPE and 120 norm/residual
steps. Prefill and scalar decode counts agree. Shared-Q8, generic
row-invariant, legacy norm and RoPE-store selections remain zero. These are
diagnostic policy axes, not adopted dispatch defaults.

## Numerical and initialized-state limits

Each larger policy's first/repeat has identical finite complete prefill and
final heads, identical initialized-state bytes and the same layout. The
bookend heads/state authenticate against that frozen repeat. Each reference
pair also has identical complete heads. The independent own-repeat receipts
were frozen before the corresponding new reference acquisition/analysis;
none inherits a 128-row allowance. Two retained heads cannot calibrate a
32-row distribution allowance, so the token counts below are **strict argmax
mismatches**, not outside-noise counts. All 32 forced inputs agree across
engines regardless of their predictions.

| Bookend | Strict mismatches / 32 | Prefill max raw / TV / reference-winner NLL delta | Final max raw / TV / reference-winner NLL delta |
| --- | ---: | ---: | ---: |
| 26B ordinary1024 | 4 | 4.37491 / 0.218667 / 0.252304 | 5.85531 / 0.233520 / 0.248832 |
| 26B all1024 | 5 | 4.78952 / 0.293729 / 0.358889 | 6.96425 / 0.168296 / 0.147870 |
| 31B both256 | 3 | 1.09750 / 0.056419 / 0.088266 | 6.73013 / 0.000109 / 0.000108 |

TV uses the full vocabulary and FP64 normalization, as does the absolute
reference-argmax NLL difference. These are two head measurements, not corpus
NLL/PPL or a full-model quality gate. The compound26 policy does not resolve
its numerical failure. Dense31's prior 128-row stock identity cannot be
transferred to these new 256-row inputs. No new speed exception is inferred.

26B cap1024 retains 592,445,440 initialized bytes across 60 ranges, with local
capacity 2,048; cap128 retains 435,159,040 bytes with local capacity 1,280.
31B caps128/256 both retain 1,740,636,160 bytes across 120 ranges with local
capacity 1,280. Layout IDs include max_rows, so equal physical sizes do not
authorize cross-layout restore or retagging. This checks same-shape completed
checkpoint bytes and cursor, not restored continuation/ring/depth qualification.

## Capacity and observed coarse memory

The runner funds its entire max_rows×262,144×4 pinned output envelope:
128 MiB at cap128, 1 GiB at cap1024, 256 MiB at cap256. A separate catalog
charge precedes the caller's 1 MiB frontier allocation. Plan/graph capacity
uses the actual bounded call count (warm + prefill + three warm decode + 32
paid decode) times plan floor ×33; no percentage margin substitutes for state
funding. Every initialized-state snapshot is copied through completion-proven,
separately catalog-charged pinned storage directly to its file.

| Profile / shape | Fixed catalog bytes | Weight extent bytes | State capacity bytes | Plan/graph capacity bytes | Execution capacity bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 26B ordinary128 | 536,871,936 | 17,047,748,608 | 650,117,120 | 7,219,713,600 | 26,105,616,960 |
| 26B ordinary1024 | 3,865,052,160 | 17,047,748,608 | 754,974,720 | 3,176,673,984 | 25,600,472,768 |
| 26B all1024 | 3,865,052,160 | 17,047,748,608 | 754,974,720 | 3,444,306,624 | 25,868,105,408 |
| 31B ordinary/both256 | 1,017,119,744 | 18,895,339,520 | 2,390,753,280 | 4,925,562,048 | 29,620,576,448 |

Capacity reservations and catalog occupancy are not physical peak. One
adjacent reference/native/reference observation used the unchanged
[D-085 coarse observer](../backend-proof-p2/peak_memory.sh): median idle
MemAvailable baseline, approximately 20 ms samples, otherwise quiet Spark,
startup through paid work and retirement, normal mmap/page-cache policy.
Observed deltas were **22,736 / 22,428 / 22,608 MiB** for reference A / ordinary26
cap1024 / reference B (407 / 406 / 390 samples). Native includes its extra
592 MB diagnostic checkpoint; reference has none. This coarse observation
is within 1.1× its reference bookends, but is neither an allocation census
nor full profile memory qualification. RSS/cgroup counters miss device VMM.
Failed token screens do not select further peak/depth acquisitions here.

## Pins and reproduction

Both approved GGUFs execute natively from their current prepared artifacts:
26B artifact `4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3`,
31B artifact `32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08`.
26B raw has 16,947,541,728 bytes, SHA-256
`f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`.
31B raw has 18,822,970,304 bytes, SHA-256
`9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575`,
revision `c1ac76e99d5513b141e8adde7288b85c3f9c32ec`. The newly staged 31B
destination passed fresh raw size/SHA verification and deep prepared-artifact
verification (9,010 chunks); the quiet cross-node copy retired before timing.

TensorFold was freshly observed at 2026-10-05 06:05:56 UTC:
[`609ca419abecebdc5a059498a613680bd3aa847f`](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f),
version 0.6.5. Its primary README/Gemma recipe expose Gemma26 through MLX,
so it supplies no CUDA comparator for these profiles. Same-format comparison
uses llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` (b10964), image
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
Only the thin C API client is compiled; original image math libraries stay
unchanged (CUDA 13.3 versus native pinned SDK 13.4).

| Measured source | Source SHA-256 | Executable SHA-256 |
| --- | --- | --- |
| Initial26 ordinary benchmark | `c2664f9166998851227996e0baccf52b4534d2f318443d120edec464e4a77c09` | `6835e3343ff14f5a81e9868b83f43c9044bc6cdbf15886c69bd54cb3783ae94f` |
| Final26 all /31 benchmark | `8fc6e4b377f5fa120d1dcc82f44531c2b201eb0462905fbe9ff1c68e45b772a2` | `c3d089a1f6db28fc8db75d113764a0f0c716e1291196d3aef3219aca02fb90f8` |
| [Reference client](../gemma-prefill/llama_prefill.cc) | `0e7092d7163d98f109cd7a5fb99b922cedfefe28f3da9632bb95f8cef13403ba` | `bd2dd3aca6e1eeac227f18a3af6099c7a3a8631be7aae27206eb0bac50b28e56` |

The final benchmark only adds the compound policy and explicit prefill MoE
counts to the initial measured source. Both native builds retain receipt SHA
`af910cb38aeeddf819af640e8ea6dcc1b97f594ec0100bd98a073c291ad135ea`.
Production math is base c2d2147 throughout. Final generic profile/parser and
tiny-control additions are analysis-only; historical receipts are unchanged.
Own freeze SHA-256 values are:

- Ordinary26: `d97e0a9401361b6436c8097c06a2c9aae274929c5f061bf7f1bf80a620b905bb`.
- Compound26: `c5999e5e4119fbf87929d1d139d68dd4bdd3bef08c3d38c216e8749b17ae7383`.
- Ordinary31: `7187e1a52057c778182b5e6395572e9449f32557d20cc7e7786ba092261b31ac`.
- Norm-both31: `bf7f95063dfad774871404bd97d6f3fce65c9ea868a077c8fe9817f006a5cf7e`.

Use the existing `jitllm_gemma_prefill` target with:
`ARTIFACT IDS_I32 NEW_OUTPUT_DIR VARIANT POLICY MAX_ROWS`.
The cap allowlist is 128/256/512/1024/2048; measured pairs are 26 ordinary/all
1024 and 31 ordinary/both256. Each first/repeat uses a fresh process/output
directory. Run `freeze.py ROOT IDS_I32 OFFICIAL_NATIVE_LOG POLICY VARIANT`
before reference acquisition. The [reproduction wrapper](reference.sh) preserves the measured original-image
compile and run commands: `26 build`, `26 screen NEW_OUTPUT 1024`, or
`31 build`, `31 screen NEW_OUTPUT 256`. It is a final bounded wrapper around
the unchanged client; acquisition used the external26 wrapper and the existing
[31 wrapper](../gemma31-runner/reference.sh) in `prefill` mode. Copy only the two `.f32` heads into the case directories
under `ROOT/heads` (26) or `ROOT/heads31` (31), retain authenticated official
logs, then run `compare.py ROOT FREEZE_SHA POLICY VARIANT`. Output uses
exclusive creation; `python3 test_controls.py` exercises finite/size refusal,
signed-zero byte identity, first-index ties, the real freeze/no-overwrite path
and parser warm/count/policy refusal.

All acquisitions used installed `spark-job --gpu`, timeout 600, with official
retirement. Raw logs/heads/state remain external under Spark
`~/.local/share/jitllm/{gemma-prefill-large,gemma-prefill-large31,gemma31-prefill}`;
job names begin `m35-gemma-prefill-large-`. Aggregate identities authenticate
`native1`, `freeze3`, `bookend1`, `memory1`, `all1`, `allfreeze1`, `allbookend1`,
`stage31`, `native31`, `freeze31`, `bookend31` logs in `results.json`.

Focused verification: both required native benchmark builds and all supervised
acquisitions retired successfully; four real analyzer/freeze controls passed,
final analysis reproduced all retained metrics, and local REUSE/header (1,367),
boundary (383), format, shell syntax and diff checks passed.
