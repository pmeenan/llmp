<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama Fibonacci completion supplement: results

All four observations pass the separate [512-token protocol](FIBONACCI512-PROTOCOL.md),
registered in `e035571bd0bde0b1a2424b460f060b7aa2c8566b` before collection.
Both fixed native/reference pairs (1,1) and (2,2) finish coherently; each
engine's own repeat verdicts agree. Collections ran on `spark` (`spark-c4e2`,
NVIDIA GB10) on 2026-10-22, using four fresh processes and one request per process.
Independent inspection checked every complete reply and execution receipt.

The allowance was chosen after the original 256-token truncations. This is a
new completion result. The [original cell](RESULTS.md#behavior-and-native-chat)
remains **native FAIL, reference FAIL, comparative UNRESOLVED**; it is not
relabelled. No PPL, model-wide quality, speed, batching, maximum-context or
whole-U3/M3.5 qualification follows from this supplement.

## Four fixed observations

Every request used the original 49 literal IDs and greedy settings, with only
`max_tokens` changed to 512. No special-token insertion, custom stop condition,
input rendering or prompt change was used.

| Observation | Prompt tokens | Completion tokens | Total tokens | Finish | Complete coherent answer | Function checks |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Native 1 | 49 | 445 | 494 | `stop` | PASS | PASS |
| Native 2 | 49 | 445 | 494 | `stop` | PASS | PASS |
| Reference 1 | 49 | 356 | 405 | `stop` | PASS | PASS |
| Reference 2 | 49 | 409 | 458 | `stop` | PASS | PASS |

All replies contain a complete iterative function, finished explanations and
consistent examples, without broken code, repetition or truncated prose.
Every function returns `[0, 1, 13, 55]` for `[0, 1, 7, 10]`. An independent
reviewer inspected the full reply and exact isolated function before each
execution. Only that function ran, under `hostlock shared` and a five-second
process bound; example programs and other response content were excluded.
No generated code was repaired. All four extracted functions are byte-identical,
SHA-256 `899554a88c9fc344f0da5af64d5fa1d6ea34bf86a5e0f9bedcb561165a05a144`.

Native's two answer texts are byte-identical. The reference's greedy repeats
use different headings and explanations, with 356 and 409 completion tokens;
both pass the same complete-answer and function criteria. Cross-engine texts
also differ. Text equality was informational in the registration, not an
acceptance condition. Raw response hashes additionally include run-specific
response fields and therefore do not establish text equality.

## Execution and identity checks

The frozen native executable remains
`32e247723ebe2c00bcdae1cb1aff12884ce7fb7fb6b421e05e47eea9e1c124af`,
source base `3965d69d439d1545b13bb77022a81ac45f8255df` plus patch
`da6310b3138153ac8f22049b0c0bfcb4b6a376363d71e91717a60d016e071405`.
Both native launches authenticate it before and after use, verify shared-library
resolution, and use artifact
`9cca62f1e6582324de2ebce34f38b89a8aa8890cdced957a50782e407ff4d7ee`.
The actual configured tokenizer, source metadata, template and artifact
manifest/index/spec match the original pins. Each native configuration has a
separate fresh state directory; both responses report zero cached prompt tokens.

Both reference launches authenticate the original image
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`
and its loader payload. Actual startup records agree on `LlamaForCausalLM`,
F16 model and F16 KV, ModelOpt FP4, `FlashInferCutlassNvFp4LinearKernel` and
FlashAttention 2, without a conflicting selection or fallback. The source
archive is the original pinned NVIDIA revision. Prefix caching and speculation
are disabled. Reference cached-token counts are null: freshness is established
by fresh processes and disabled prefix caching, not a measured zero cache count.
Both arms retain context 12,288, one slot, 512-row prefill and no drafter or
optional lossy mode. See the registration and [original provenance](RESULTS.md#provenance-and-executed-configuration)
for the complete source/artifact/precision pins.

The registered helper checkout is `e035571b` and the actual helper hashes are:

- `native.py`: `e1a4f6fb9e75e689ef3e4b5b1bf2e2c038365a8efff4aeb15a0407623e754c98`.
- `reference.py`: `d528abd0a390c0f8293be279da0edb2710fb191791ba739e0e42e1b71b6ec6fe`.
- `fibonacci512.py`: `ea381a489a0b101ec02a5932f9a5588445741d5618be51910c7f610cdf45fa42`.
- Native Service helper: `0f1135a2283fe63797e15c8dffe48c9df39b0baae91a57f925b5ba21c3e3252e`.

All four supervised jobs finish with return code zero. Native TERM retirement
returns zero; reference logs/removal/exact-owned-name absence commands all
return zero and positively confirm container absence. The first native and
first reference post-job snapshots are free. The saved second-native snapshot
was taken after reference repeat 2 had started, so it is not a host-free receipt;
native retirement itself is separately confirmed. The final reference snapshot
confirms no active/waiting jobs or GPU compute processes.

## Evidence receipts

The completed bounded-function launch argv, process exit codes and runner
hashes are retained in `execution-launches.json`, SHA-256
`d71ca0dbf9ec394a150c4bb047401560c0d12014b1ac2e68ce3a6d8a29bc381f`,
alongside the independent inspection and execution receipts.

All four request files have SHA-256
`63f52bb49fb00e542f87813e1e4d1e31b18b3c1aa8e41c1eb2b9e0977040a38e`;
the original complete input file is
`ef8b44d0f97e7813edbc506d3105ec0dfd112de05efc4e4bbbe3de9158e9c960`.

| Observation | Raw response SHA-256 | Answer-text SHA-256 |
| --- | --- | --- |
| native-r1 | `d1ec37c0c18eda2348d8bcfcd2efbed9eed6af7e55695ca79b4835c37e287370` | `206bfee1c7a804a82090988352768f99572adffd6eb8c5ed6e1595e945799cb5` |
| native-r2 | `b0a54b3ef15439a8c0a1d806d6b7b51e59ae5602781ce3a24ba76abf3851e943` | `206bfee1c7a804a82090988352768f99572adffd6eb8c5ed6e1595e945799cb5` |
| reference-r1 | `8a46f7bf0ab66cf4b08d6150055b6bcd6a1e6229a7dd84250b584a8fd9289faf` | `d92b97165589bdba7156da296961fb40f439cdac88f319c9c5b2cabbd62c9101` |
| reference-r2 | `1f203b69d5e9b973ca83e33397a147b5256b6d90fd8e00350aedef1138a353d6` | `b189e806a1673015d2dd9190cdea0f9a399e5e7aa868edebd28259d7ffd4b4e7` |

- `native-r1`: launch `73c2e3c98241981ba6457d7853d5f9bf39ec6ebee05fa269216c43896cdece1d`; retirement `102c4a103c269581f1550e5dfe7f616c145397c20f49132725e5a0d85ab71a6b`; independent audit `77c2e662c6f11470d3e4a7f5e13d2bda0f8c75f74e1f2966a75607c5dbc8a09c`; function execution `1a40fe78c567fe06bd02f7da72c9cb847c77c7471b7759b274c0229e7e4e5c63`.
- `native-r2`: launch `4d99cb55a4b97613ae89999b890aacf6c16272ce2ec2a6c05d1c05684f707f46`; retirement `102c4a103c269581f1550e5dfe7f616c145397c20f49132725e5a0d85ab71a6b`; independent audit `7a61b0baa105f9da885c13264049f875c4b377078aa0b801dd9832bc515358e5`; function execution `65b57840a857afdd51faba471bd43d7106af1ccb23a45ed0f2d593bcf04d002c`.
- `reference-r1`: launch `35587918ca1fdd088cbc01b0f2cbf45ce9a92b1ddbbf9d411a7951124ff66f0c`; retirement `07a5256e53b118eb494060a26919c05f20cfc93d8fa7b9d0897f65355233bc07`; independent audit `71393b64d4d10039bd66378b74a743aa33aa55de88a3259a59f43caa65c2b35c`; function execution `9432aaf6458769c1eab1307a346d9bb1aada87c7496638408a85dea761023ae1`.
- `reference-r2`: launch `e412615b4debfcb41bd8cca3eac2b04ab07c2df67da567d47a48a4a20a303fc3`; retirement `8a0514bb4b5f4655e2186623b346fb2b8c82187d1f7b355b84ada3fd2e7ed2d4`; independent audit `7ef4d44a146f5ce6c03c952fce8277fb4ccf3e2c3b88733a0b2efa49c500c3c6`; function execution `d48c588ed043c3a93e54cc3c0ca17b0c14196e3781fa80d28c1f53f0e8bb6843`.

Raw private bundles remain outside Git under
`~/.local/share/llmp/m35-u3-quality/fibonacci512-{native,reference}-r{1,2}`
on `spark`; supervisor jobs are named
`m35-u3-llama-fibonacci512-{native,reference}-r{1,2}`. Local inspection copies are
under `/tmp/llmp-m35-orchestration/llama-fibonacci512-*`; independent inspections,
exact extracted functions and bounded execution receipts are under
`/tmp/llmp-llama-fibonacci512-review/`. These locations and hashes retain the
provenance of this bounded result, subject to milestone raw-data cleanup.
