<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# U3 original checkpoint tensor-content parity — PASS

Both original Qwen checkpoints were re-imported on spark through the ordinary CLI publication path, including deep validation and stored-spec validation. Every original named tensor part, including every expert slice, has identical bytes, format and part role. There are no missing, added or changed parts. This closes the original-checkpoint payload parity control, not U3 serving, quality or speed qualification.

| Control | Parts | Payload bytes | Import elapsed | Old/new digest elapsed |
| --- | ---: | ---: | --- | --- |
| ModelOpt NVFP4 | 99,901 | 103,799,140,484 | 31m46s | 147.296s / 168.565s |
| MLX 4-bit | 223,706 | 110,682,030,208 | 30m15s | 161.720s / 153.432s |

## Sources and publication

- ModelOpt: Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6c14c6c9442ef83e8f05b5a3c39304f69.
- MLX: TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa8066e3be900a0d148d0f5a2f4eb1cf6534. The archive's TensorFold identity is authoritative; historical prose sometimes says Vontra.
- Every old source-file identity remains identical. New manifests additionally retain tokenizer.json, tokenizer_config.json, chat_template.jinja and generation_config.json. Source architecture aliases are qwen3_8_flash_next (ModelOpt) and qwen4_exp (MLX), distinct from declarative mapping qwen4exp/version1. Manifest/spec identity differences do not imply payload differences.

### ModelOpt

Old artifact: `02354af162f30b6ec6a117d5311da7d642c5be3128867750be55eeb7b6cd38b9`.

New artifact: `7175c03c1748aa96af84667f2e51e53d56e954d7fb3b6ccc1083f882b41501a2`.

The new target spec stores temperature 1, top-k 20 and top-p 0.95, with no
min-p, from the retained `generation_config.json` (2,097 bytes, SHA-256
`735ec63e625c2086fbec18327233a03a7b060fddabfd7915af4e6fdcec3187a2`).
The old artifact has neither this retained file nor stored sampling defaults.
These small manifest/spec fields were checked again on `spark` on 2026-10-22;
no additional payload import or inference was needed. This supplies the pinned
checkpoint's structured defaults; model-card fallback for other sources remains
open.

Shared payload aggregate SHA-256: `5d365648ea2ab7824bba41269ec91ed3ad47c3cc5c7d034d246f10fb1bc1e1d8`.

Shared semantic aggregate SHA-256: `024d9180a752e85322bd84bc54d2837adc10214c830d6d045bfcb5af82b74892`.

### MLX

Old artifact: `b842fe59c98887b942a2332a8dbaf83e5e002c96cc86fcd07e872cd1bbe3283b`.

New artifact: `44ef8f918edb8231428be82237e77cd9286a001f3fe96ef97bbf3091e02bbafb`.

Shared payload aggregate SHA-256: `9024ba08f6fb18899b384599e71fe40ad52b0da6307e6a75cf4871ca781f31fd`.

Shared semantic aggregate SHA-256: `ef91111624d6505a41305bb284947e7836bc114c78969026afc6e2d2b284c7b5`.

## Drafter source-alias admission repair

A subsequent MTP qualification attempt on 2026-10-22 paired the new ModelOpt
target `7175c03c…` with existing drafter
`2cb4cdb8d4c8bba3bf7a258d56edbae1b456d9268c238948f00210f7b1e9d0ef`.
Startup refused before readiness: the drafter's recipe targets canonical
`qwen4exp`, while the target manifest retains source name
`qwen3_8_flash_next`. This is a failed startup; the intended inference/tool
control is **NOTRUN**, not a quality or tool-generation result. Its original
failure receipts remain in external scratch under
`/tmp/llmp-m35-orchestration/native-tool-fix-r4/model-qualification/mtp-required/`.

Import, admission, stored-drafter construction and calibration now compare the recipe
against the resolved target mapping's canonical architecture. The existing
container, role, source and structural compatibility checks still apply.
Both retained-metadata and stored-spec drafter paths use that same identity.
No re-import of this existing target/drafter pair is needed: the target mapping
already resolves the source alias, and the existing and newly imported MTP
recipes both name `qwen4exp`. Fresh drafter component publications now identify
the selected canonical drafter architecture rather than the parent checkpoint
architecture, retaining the raw identity in authenticated source/config receipts
and existing mapping metadata. Only component artifacts previously published
under the parent target architecture need re-import; target output and the
artifact schema are unchanged.

Fresh separate required-tool and named-tool cells on Spark B passed on
2026-10-22 with the same `7175c03c…` target and `2cb4cdb8…` drafter. Each
performed an actual `read_marker` call, read `LLMP_TOOL_NATIVE_RESULT_7319`
from the tool result, and completed its streamed follow-up. Their runtime
counters confirm speculative execution rather than just configuration:

| Tool choice | Speculative units | Drafted | Accepted | Discarded | Coverage violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| required | 53 | 58 | 32 | 0 | 0 |
| named | 51 | 64 | 33 | 0 | 0 |

Both cells used temperature 1, top-k 20, top-p 0.95, min-p 0, seed 7319,
three draft rows, requested draft vocabulary 65,536 and depth cost ratio 1.16.
The effective draft vocabulary was 47,172, capped to the companion's selected
head rows; shared wave depth and draft wave maximum were both 2, with 4 verify
rows. Both used 16,384 context, 512 prefill chunk, one slot and F16 KV. The request output
budget was 1,024 tokens each. Source, runtime, artifact and asset checks
matched before and after both cells; process groups retired cleanly and
the listener closed. These are bounded behavior controls, not speed,
reference-quality or broad U3/tool qualification.

The unchanged qualification helper has SHA-256
`324d8441d60f7893c2817f00644a70fe828148c6a5234dbd6d09338dcb3312d9`.
The 2,828-file combined source manifest has SHA-256
`0783bdf0a3b1ab2931515bc25ad1b0e9abb129f81a1351788b9f70f961a94509`;
Spark B's actual runtime has SHA-256
`ea3276a767b59a0ecc7ef907aee52233d2dae157618062cc9599415331ebb74a`;
the compiler/kernel identity is
`8148b25f8c488058047940556525d80736d9d5a4fd4b4401c7bd0fa323146830`.
Raw transaction, authentication, final-counter and retirement receipts are
under external scratch
`/tmp/llmp-m35-w6-checks/model-qualification/{required,named}/`.
The earlier admission failure and its inference **NOTRUN** remain historical
results; these fresh cells do not replace those receipts.

## Reproduction and provenance

Compiled primary base e9de1e8316d68a764e6a718ce44ff4a3291051ff plus the reviewed staged union frozen outside Git in source.patch, SHA-256 a428dc12a5914ac626b803a42472783eb55065c344a4982e3cbf23a3faa2ff80. Importer executable SHA-256 c77b2e5c9f5aa742531d831f5b840374c0505a1901051e811ba8e9ba70343517. Only llmp target built on spark, 77 successful steps. Remote .git deliberately remained unsynchronized: the manifest's importer.build (61f8426f7e2f45931d91645b016069e13f6f2236.dirty) is not the compiled primary source identity; use base + frozen patch + executable hash.

Both long imports were explicitly approved as needed-now supervised jobs with 3600-second bounds based on measured historical duration and input size. Separate digest/compare jobs were shared and bounded to 600 seconds. All jobs passed: u3-original-build; u3-original-{modelopt,mlx}; and each checkpoint's -old-digest, -new-digest and -compare jobs. No inference ran as part of this proof.

The digest harness streams at most 4 MiB at once. Ordinary resource offset is shard.data_offset + group.offset + resource.offset, length resource.bytes. Expert slice offset is shard.data_offset + groups[first_group + expert].offset + group_offset, length slice_bytes; its key is name[expert=N]. Padding and packaging offsets are excluded. Each part receives SHA-256. The payload aggregate hashes canonical sorted JSON name -> per-part hex digest; the semantic aggregate hashes name -> {bytes, format, part, sha256}, both UTF-8 Python json.dumps(sort_keys=True,separators=(',',':')). Comparison rejects any missing/added part or changed field. A synthetic offset/expert fixture validated the harness.

Digest script SHA-256 d50326f27765a54b93e469971191c255cd840af17f88afa11527141fd3da6517; comparator SHA-256 d887eb23ce36ed914799803470fc07755d60576b632851a9a6ce1ae7756e778c.

Raw receipts/logs and per-part digests remain outside Git on spark under ~/.local/share/llmp/jobs/u3-original-* and ~/.local/share/llmp/u3-original-*.json. New artifacts are in ~/.local/share/llmp/models/m35-u3-proof. Compact comparison JSON, scripts, source patch and this report are in /tmp/llmp-u3-original-proof on the workstation. No commits or repository modifications were made by this proof agent.

New spec SHA-256: ModelOpt `02d130bce7462794251cc29e5554565a0ad83a398ce2776fd8078ef4b683618a`; MLX `f184c790c9dd92cdee3b849805a5221b53e00ac12c746444768c34a3950ced87`. Complete old/new index, spec and metadata file identities are retained in `metadata-identities.json` beside this report and the four original manifest receipts.
