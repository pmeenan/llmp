<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 repaired ordinary C2 adoption speed

Status: the registered three-pair C2 comparison completed on Spark B. All six arms passed their request, lifecycle, and environment audits; all nine pair metrics and all three median ratios are within the predeclared 1.10 limit. This is a bounded C2 performance result for this model, runtime, artifact, and configuration. It is not a general batching, quality, full-task-completion, or model-wide performance qualification.

## Frozen comparison

Registration commit `65788392c3f003afe1da821e6c38e739093157f4` bound the exact eight-file package. The enabled external registration envelope on B had SHA-256 `e33c8333918ca1d489860a3fcdec263a176f378f70ec5f5d947aa631a897be91`. The registration was enabled only after the original-A versus R quality pair and independent manual reviews passed. The quality-pair report SHA-256 is `507546bceb36a44d27e5712761a6c01349ff7fe1cddb718cd9c387f06eccaf68`; original-A review SHA-256 `6da9dd76e9769fe4de0dddab0b42ae6671b89e78585e860d450d7e8b5f3e039b`; R review SHA-256 `d70e9717fa5c3404d263fb883236e5eda7d45be8acd6b8ebd7f05a4917e2c765`.

Both arms used the same retained R runtime (SHA-256 `0ba727678e726b0c1bd9800d175358db8ac59207c01f10ce5d34f5ec2fc8d28b`), source manifest (`0a53dea6a1df1b79002b4e3f1af689ed724ee376c9d8ded94c9d8e4f7b0eb99a`), Gemma31 artifact ID `090809981d2909085798273e665a0db58351a8e6029b13bc04ced857c9130943`, and config SHA-256 `f43895787b8b0587da5a4f93399f45f3dd38b6b11184c443f2fbf3e55f6b72d7`. The corpus was `/home/pmeenan/.local/share/llmp/m36-baselines/inputs/corpus-1facc957.md` (SHA-256 `55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`). The registered executor was `run_c2.py` SHA-256 `0e03f9f6588fa0d261951aa841da7127a524101b924142c16f06762d8207651f`; the collector helper was `speed_gate.py` SHA-256 `1b71fda41afb81e316952567d733017bf87fcbb2a6e4205d3ea4c7b143b8e186`.

The workload used context 8192, F16 KV, two slots, prefill chunks of 256, C2, 26,000-character prompts, temperature 0, and 128 output tokens. Every fresh arm did one warmup C2 pass on prompts 0/1 and one measured C2 pass on prompts 2/3. Both measured requests and both warmup requests per arm generated exactly 128 tokens, had zero cached tokens, and finished by the length bound. All six services retired normally (TERM, return code 0); audits recorded no competing GPU compute apps. This is a capped-length timing workload and makes no semantic-completion claim.

A disabled-pattern ordering correction was committed at `5d9a0defae10c9e8100bf862a4cec454c21edf6e`; the subsequent launch correction that passes the pinned runtime library path into the collector child was included in registration commit `65788392c3f003afe1da821e6c38e739093157f4`. The child library check loaded the retained libraries explicitly, returned cuBLASLt version 130800 and matched both dependency hashes; successful measured runtimes also enforced version 130800 at readiness; receipt SHA-256 `b4ec626e30d6b58752ad59ab4a6cbe5ca712f030e0007b9946dd0704a062a2d4`.

## Absolute arm measurements

C2 prefill ends at the last request’s first token. Decode sums the per-request chunk rates over the overlap window; wall divides total output tokens by batch-release-to-last-finish time. These are concurrent workload metrics, not whole-response per-sequence decode rates.

`peak GiB` is the collector’s largest drop in system `MemAvailable` during that arm, not a GPU-memory measurement. Each arm ran in its own supervised exclusive job on Spark B (`spark-56f5`, driver 580.178.04).

| Pair | Order | Arm | Prefill (s) | Decode (tok/s) | Wall (tok/s) | Peak GiB | Record SHA-256 | Runtime-log SHA-256 |
|---:|:---:|:---:|---:|---:|---:|---:|---|---|
| 1 | A/R | A | 25.215268 | 18.089519 | 6.581458 | 22.513054 | `d99a21f94ad35ef43c376cdb8421fe5946f2c2047c45010a778f07d76296cee6` | `7bfef10d4ec45432c1ea79402aff0dc883b1c49ab065f5706209a6b8135825ef` |
| 1 | A/R | R | 25.045082 | 18.566918 | 6.663837 | 22.645454 | `216921812c3f4e19eb89887271f8f7edbd999f69a8eac1dd3bcc65498b0d2df8` | `8bdbc1cdc5b1b8d9e1cf9e9fd9f35918a426c6676a7c11ad45a3d982520acc59` |
| 2 | R/A | R | 25.181023 | 18.640353 | 6.648209 | 22.609241 | `7d49d8888a5c0efee67637dd2d12ae3dcc791eae108758be2042448da3616379` | `bbba845252682064c9f8ebe8166de2e820f6f1773a38e86c9442097eb246f0fd` |
| 2 | R/A | A | 25.251254 | 17.994207 | 6.563426 | 22.554157 | `ef0d3c406e7cf1e2395523529c6196b89635480b5758253c9d04cdde316fa6e4` | `fb01baf33daf74d29bd132d0410a23f588a3c0775189d06ae3562e3ec5c236e7` |
| 3 | A/R | A | 26.874998 | 18.020193 | 6.222972 | 22.613941 | `8df867651b8a2322bf7eae965a1dcd8dd5182965d8b16a2c7d0ee870b2c40e4d` | `0c3e684d3f00659746757f061b39cf36742769f6f134044cd08173ff95d01978` |
| 3 | A/R | R | 26.770422 | 18.560509 | 6.289581 | 22.664238 | `95e4eedde2b019d8e0c8c245933e3780ceebf1c874cc8a37eac0d01e357ee362` | `2cdb80da822d8b4d74bef3d1fce9e60fd1b3aba6216600551ca5a4c552f45db8` |

## Pair and aggregate checks

The predeclared ratios are R/A for C2 prefill time and A/R for C2 decode and wall rates; values at or below 1.10 pass.

| Pair | Prefill R/A | Decode A/R | Wall A/R | Pair verdict |
|---:|---:|---:|---:|:---:|
| 1 | 0.993251 | 0.974288 | 0.987638 | PASS |
| 2 | 0.997219 | 0.965336 | 0.987247 | PASS |
| 3 | 0.996109 | 0.970889 | 0.989410 | PASS |
| Median | 0.996109 | 0.970889 | 0.987638 | PASS |

The [durable aggregate](speed-results.json) retains the six absolute observations and registered verdict. The original aggregate JSON is `/tmp/llmp-u4-gemma31-multirow-c2-speed-r1/output-r3/result.json` (SHA-256 `205096bdf2a5febc03fa9ecfc42ab7f148dee73809b97dd750395f7ede05a0cf`). An independent local recomputation from all six audited run records reproduced every pair ratio and median and confirmed the aggregate pass: `recompute-result.json` SHA-256 `aa74abe04830ec9e0ca109e4c1703b13d7f9920e60f0b9a61b864de3537d6bfd`, script `recompute_aggregate.py` SHA-256 `16f92d5be54673d4b80cbb32173bb00fc3389739d661fdb58c3818d60cd9afa0`.

## Preserved pre-readiness failures

Three earlier attempts are retained and excluded from the six measured samples. None produced a request sample; all measured arms used the fresh final namespace:

1. `setup-failure-r1` stopped in the executor guard before invoking the collector because the external registration was still disabled. Its receipt says there was no collector/inference. Supervisor-log SHA-256 `b8abc536bab90a12a63cb7892d147de591e024a8f6a43692545fb897442496ff`; receipt SHA-256 `c9fee20a0d6acb3a2f5eedacc0fd43180ce2b1b876de0c5914eb8cacdce37dfb`.
2. `output/` ran the collector but the runtime refused its unsorted disabled-pattern list before readiness: “disabled patterns are not ascending and unique.” It had zero requests and no timing metrics. Runtime-log SHA-256 `89af7d3e80ba8654758c72f9720a4373f1d2cfaf1cf5f4b0545e1f35e70970da`; record SHA-256 `ce2ec00a79e8a15b209ff873f02af611286cf22b5bcc9357cb0daccded15e348`; supervisor-log SHA-256 `cfa0d2cca905e192d5824899f4b54bf428a730b245edb0018504f5c2664d362b`.
3. `output-r2/` used the corrected policy but the runtime refused to serve because system cuBLAS/cuBLASLt 130101 were loaded while this build requires 130800. It had zero requests and no timing metrics. Runtime-log SHA-256 `8e3cee6bd60ca19840a54f3c59d61d6d730ee0b419eac752bfa5c96365708b8d`; record SHA-256 `f8643887331031b054d333cc336e06ca8f12175d0f07dd5ce30bc6dcd1febd24`; supervisor-log SHA-256 `db1fae30836ccc308476a44ceceeeac204eb78ba6fb6afccb814137f89dfd1cb`.

The first two collection attempts and setup guard refusal remain in their original output/evidence namespaces; `output-r3` is fresh. The final executor explicitly passes the retained runtime’s pinned `LD_LIBRARY_PATH` only to its collector child, leaving the parent preflight environment clean.

## Retained evidence

The copied, hash-indexed evidence bundle is `/tmp/llmp-m35-orchestration/gemma31-c2-speed-r3/`. It contains all six `output-r3` arm directories (launch, pre/postflight, return status, raw run record, runtime log, and audit), pair JSONs, aggregate result, all three earlier failed output/evidence namespaces, the six supervisor logs, and the registered package snapshot. Manifest SHA-256: `3f14b72047a2712f3e29e082057f2a2d7a583c103c21c26df7d824b9eb759fc8` (`retained-files-manifest.json`). Raw records and logs remain outside Git.
