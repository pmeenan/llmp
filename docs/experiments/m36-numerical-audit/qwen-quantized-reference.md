<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 same-quantized-checkpoint reference

The pinned Mia implementation successfully scored the existing four-document
Qwen corpus: 1,016 next-token targets, mean NLL 3.559477762612 and perplexity
35.144838403. This supplies the quantized reference required by the revised
quality policy; it does not require an original BF16 checkpoint or matching
head bits. It is a short-corpus observation, not a family acceptance gate.

## Observed comparison

The reference and historical native A/C observations use identical four
255-token histories, without BOS/EOS or a chat template. All positions 1–254
are scored independently per document using the actual target token. The
reference tokenizer independently agrees with all four native token prefixes.

| Implementation | Targets | Mean NLL | Perplexity | PPL relative to reference |
| --- | ---: | ---: | ---: | ---: |
| Pinned Mia, target only | 1016 | 3.559477762612 | 35.144838403 | baseline |
| Historical native A, ordinary joined products | 1016 | 3.563363450261 | 35.281665930 | +0.389325% |
| Historical native C, three explicit product tuning keys | 1016 | 3.556988278363 | 35.057454697 | −0.248639% |

These native measurements predate the current main source. Their experimental
policies disable `owner_row_invariant`; C additionally selects existing
physical-eight vector products for 48 routers and two PLE products. Neither
policy is adopted by this comparison. The [broader native screen](qwen-joined-score.md)
records mixed document-level results and a gain dominated by its earlier
prefixes. Current-source comparison, long-context behavior, MTP and serving
migration remain separate work. No claim about closeness to the unquantized
model follows from either direction of perplexity change.

Mia uses FP8 KV and BF16 SSM; these native observations use F16 KV. The
arithmetic and cache choices differ across implementations. We compare their
complete losses without forcing a common rounding sequence.

## Checkpoint, policy and result provenance

Spark B (GB10), 2026-10-12; `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` revision
`925d7be6c14c6c9442ef83e8f05b5a3c39304f69`, using the standing source checkpoint.
The [pinned fast-start launcher](../fast-swap/baselines.md#default-mia-launcher-for-new-runs-2026-10-03)
selects recipe `b8439110eec0230facbe4ddf0dffe01b8f769be0`, image
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
instanttensor 0.2.0 and the existing CPU PLE loader override. The effective
loader, source metadata and indexed payload inventory were checked before
and after execution; this run did not rehash every large weight payload.

Target only, MTP 0, DET 1, `QSA_DET_TOPK=1`, `MOE_DET_FINALIZE=1`, maximum
context 33,792, four slots, batch 2,048, GPU memory utilization 0.786,
26 GiB host reserve, graph mode 0 with full-decode-only captures at sizes
1/2/3/4. A fresh credential-free private recipe uses read-only checkpoint
mounts and the existing packed PLE. Full-context capability is not tested
by the four short documents.

| Identity | SHA-256 |
| --- | --- |
| Safetensors index | `435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9` |
| Target tokenizer | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| 1,020 little-endian int32 input IDs | `3cd946c3876501eccbe079b9d2700e542e66242be241a8549546df8d9b23e468` |
| Collector source | `2808e63be9c08fe1378bf88e5dcf2f09835681d9f50894ea198bc6f61514621f` |
| Reference actual receipt | `1dd33b3265f61cc993973c5651cde03db6b90660b3fa3c9f1cb73c476070f8e9` |
| Historical native actual receipt | `92ad695820d1eb909d5d4336a57a7525d34b2fba5ca622c459b7743f904c535b` |

Each document has 254 targets; reference summed NLL is respectively
747.6428986233683, 1006.8320033252003, 985.9432543228786 and
876.011250541982. All 1,016 unrounded losses have complete, unique target
coordinates and finite nonnegative values. Independent review recomputed
document sums and the token-weighted aggregate.

The supervised reference run completed in 213.255 seconds, including
206.469 seconds to readiness. These are startup observations, not inference
speed measurements. The owned launch group retired, its container was
removed, memory-limit/OOM events were zero and no GPU process remained.
A preceding attempt failed before scoring because a cached absolute model
link was unavailable inside the container; mounting its unchanged resolved
checkpoint read-only fixed the environment. Its failure remains recorded.

## Reusing the collector

Copy [collect-qwen-reference.py](collect-qwen-reference.py) to a fresh scratch
directory as `collect.py`; outputs are created exclusively beside it. Supply
the frozen `inputs.json` and `doc0.txt` through `doc3.txt` from
`/home/pmeenan/scratch/m36-qwen-quant-reference03` on Spark B, or the local
coordination copy `qwen-same-quant-reference03`. These are retained captured
replay inputs, not checked-in raw results. The metadata SHA is
`50110476b05de2b2a87e012b46cff967d179c48917406797368309eb3f27e25d`.
Document hashes are authenticated by that metadata and the collector; do
not substitute new text under the same measurement identity.

Admit and supervise the pinned reference server separately through installed
`spark-job --gpu`; use the declared policy above, model name
`qwen3.8-flash-next` and owned loopback port 18253. Run
`python collect.py http://127.0.0.1:18253 qwen3.8-flash-next` within that job.
The collector bounds responses, rejects missing/nonfinite targets and emits
`result.json` only after complete scoring. It does not start, admit or retire
the model; the owning wrapper must verify source/runtime identity and retire
all work even on failure. Six offline positive/negative controls and the
actual run checked this unchanged collector. Raw losses, logs and actual
receipts remain outside Git.
