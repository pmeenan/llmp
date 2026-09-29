<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 MTP draft-head selection (2026-09-29)

This study isolates the draft-head difference raised by the owner while
closing M3's long-context MTP gap. Draft depth is fixed at 3. It follows
the [MTP sweep](../qwen38-mtp/README.md) and
[long-context diagnosis](../long-context/README.md#phase-2-qwen38-flash-next-flat-with-depth).
The measurement question is whether Mia's selected vocabulary helps beyond
choosing a deeper draft window.

The selected list is worth a native trial: at fixed depth 3 it was 12.6%
faster than our prefix in Mia's engine at 128K, with higher acceptance;
8K was tied. The continuations differ, and jitLLM's selected head has not
been implemented or measured, so this is a lead rather than a native gain.
Our native width control also rejects a full head: it was 12.7% slower.

## What the implementations actually do

jitLLM's default draft head is a view of the first 65,536 BF16 rows of the
target's head, with no extra weight allocation. It returns the winning token
and its probability (the latter serves the experimental adaptive window).
The IDs are a prefix, not a ranking of model-output frequencies.

Mia's pinned
[patch](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/b8439110eec0230facbe4ddf0dffe01b8f769be0/files/patch_mtp_draft_vocab.py)
packs selected BF16 weight rows once with `index_select`, casts the hidden
vector to BF16, computes a standard `torch.nn.functional.linear`, takes
argmax, and maps that row back to its
original token ID. Its greedy path skips the full-vocabulary logits processor
and monotonic scaling/softcap. This is a smaller matrix product with a token
map; it is not a separate trained draft head or a special quantized kernel.
The MTP block and final mixer still come from the checkpoint in both engines.

One further difference is unmeasured: jitLLM's graph computes and copies the
draft probability even when the runtime asks only for draft IDs. Mia's greedy
path takes argmax alone. Making confidence optional is a separate small lead;
this study does not attribute a speed gap to it.

Mia's
[list generator](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/b8439110eec0230facbe4ddf0dffe01b8f769be0/files/build_draft_vocab.py)
counts corpus token frequencies, recommends model-generated output, and
keeps special/added and byte tokens. The shipped English/code list has
47,172 unique IDs, spanning 0–248,076. Its overlap with our prefix is 36,925:
10,247 selected IDs are outside the prefix, while 28,611 prefix IDs are not
in the selected list. Token-ID order is therefore a materially different
selection policy. A particular small output need not favor the selected list.

The actual baseline launches used that 47,172-row list, including both the
short-context MTP comparator and the long-context phase-1 comparator.
Their startup logs report 47,172 of 248,320 rows. The previous MTP report's
description of Mia's head as full vocabulary was wrong.

| Head | BF16 rows | Weight bytes (width 2,560) |
| --- | ---: | ---: |
| Full target vocabulary | 248,320 | 1,271,398,400 |
| jitLLM prefix | 65,536 | 335,544,320 |
| Mia English/code selection | 47,172 | 241,520,640 |

The selected weight matrix is 28% smaller than our prefix. That is a computed
size difference, not measured traffic or a measured jitLLM speedup.
Packing it would also require an extra allocation and token-ID map, unlike
the current view.

## Native width control at 128K

On `spark-b`, GB10, driver 580.178.04, at `2a7e670` (model/kernel code
unchanged by that sampler commit), target `c4fb47a9…`, drafter `056a750e…`.
One run per head, prefix first, 128 greedy output tokens from the existing
128K retrieval prompt (128,799 rendered tokens). Same depth 3, context
131,072, fast plan, artifacts, prompt, and harness. The harness uses
512-row prefills and compares speculation with its own plain decoding;
these rates are not the runtime's 512-output-token ladder rates.

| Draft head | Accepted / drafted | Acceptance | Verifies | Draft ms/step | Verify ms/step | Whole ms/step | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First 65,536 rows | 78 / 146 | 0.5342 | 49 | 9.801 | 56.088 | 66.293 | 39.097 |
| All 248,320 rows | 79 / 143 | 0.5524 | 48 | 20.501 | 56.631 | 77.538 | 34.123 |

The full head is 12.7% slower. Its draft job costs 10.700 ms more per step,
while acceptance improves only 1.8 percentage points. Verify cost stays
within 1%. Width matters, and using the full head does not close this gap.
The earlier 8K sweep also found a full head slower despite better acceptance.

Both head widths produced the same 128-token speculative sequence and the
same 128-token plain sequence. The teacher-forced check matched the plain
argmax at 126 positions, with two near-ties (largest margin 0.2921, existing
bound 1.0), and no violations; the free-running plain/speculative sequences
shared their first 91 tokens. The harness also recorded its own verify noise
in that check: p99 0.7448, maximum 1.1438. Peak memory drop was
79.98 / 79.97 GiB. Neither case had a refused graph or a reported problem.

The speculative output contains three IDs outside the prefix and four
outside Mia's list. These output-coverage counts do not measure draft
acceptance: hidden states and candidate competition also matter. This one
prompt does not establish a general preference between the two reduced heads.

## Selected list versus prefix in Mia's engine

On `spark`, same pinned, unmodified recipe and image, changing only
`MTP_DRAFT_VOCAB`: its shipped 47,172 IDs versus an external file of IDs
0–65,535. Both use depth 3, maximum context 262,144, block drop disabled,
FP8 KV, BF16 recurrent state, and the recipe's normal MTP path. Each head
gets a fresh process; curated first, prefix second. One run per prompt,
128 greedy output tokens, no reused prefix cache. The prompts are phase
1's 8K and 128K retrieval prompts (7,586 and 128,799 rendered tokens).
The 128K prompt IDs also equal the native width control's exactly.

| Depth | Head | Accepted / drafted | Acceptance | Draft iterations | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 8K | Curated 47,172 | 83 / 132 | 0.6288 | 44 | 45.193 |
| 8K | Prefix 65,536 | 84 / 129 | 0.6512 | 43 | 45.081 |
| 128K | Curated 47,172 | 84 / 138 | 0.6087 | 46 | 43.475 |
| 128K | Prefix 65,536 | 73 / 162 | 0.4506 | 54 | 38.615 |

At 8K the rates differ by 0.25%; at 128K the curated head is 12.6%
faster, beyond D-085's coarse 10% threshold, and acceptance rises 15.8
percentage points. Fewer draft iterations are consistent with an acceptance
benefit, not just reduced weight traffic. Counts come from vLLM's metric
deltas, including drafts at the generation limit, rather than only returned
tokens. Start-to-ready was 703.4 / 699.3 s; peak memory drop 101.85 /
101.05 GiB. The memory readings also include page-cache effects and are not
an isolated head-allocation measurement.

The target is unchanged, but the free-running continuations share only
9 tokens (8K) and 15 (128K) before diverging. The prefix's 128-token 128K
sample remains in
reasoning and does not reach the three retrieval answers; the curated
sample contains all three. Both 8K samples contain all three. This is not
a matched-continuation timing or a quality gate: neither repeat variance
nor a full-answer quality comparison was measured. Acceptance reflects
both vocabulary choice and the resulting different continuation. These
two retrieval prompts also do not establish English/code corpus coverage.

## Recommendation

Add a native selected-vocabulary trial alongside adaptive depth and cheaper
verify rows. Pack BF16 rows once, retain the original token-ID map and
verify every draft with the target. Use an independently produced list
under the existing license disposition, and test acceptance and quality on
representative English/code continuations before changing the default.
The current prefix remains the default; no selected list is shipped here.

Do not widen to the full head. Do not assume a 28% traffic reduction or
the reference's 12.6% rate improvement transfers to jitLLM: the native
draft/verify balance, numeric path, and continuations differ. The measured
native verify already accounts for 56.1 of a 66.3 ms step, so its row cost
remains a larger direct timing target.

## Provenance and reproduction

- Native target artifact:
  `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`.
- Native drafter artifact:
  `056a750e3a90be3ae6a4b12bb963ce45290aaa6f52b5ba9799e777d491f80aea`.
- Checkpoint: `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` at
  `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`.
- Mia recipe: `b8439110eec0230facbe4ddf0dffe01b8f769be0`.
  Image `vllm/vllm-openai:qwen38-flash-next`, vLLM
  `0.1.dev20073+g8e685d198`, PyTorch 2.13.0+cu130, CUDA 13.0.1.
  Local image ID:
  `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`.
- Selected list: `files/draft_vocab_en_code_47k.txt`, SHA-256
  `ee819d2560b52ba1351acdd1c4a0244b77bb60694a7660c6363097a482b0afb5`.
  Patch: `files/patch_mtp_draft_vocab.py`, SHA-256
  `2c7d19b8021f2c439920ae7f7df6f7b008eb635256a8423ef03e168d3984911f`.
  Prefix file (IDs 0–65,535, one decimal ID per line with final newline):
  SHA-256 `bac6f4d80bf2772947c877447636c2cda523ec1ed9987ac455fa68a6b94306c5`.
- Native replay input: `spark-b:~/.local/share/jitllm/m3lc2/rv-128k.json`,
  SHA-256 `12fa66def6f5922b2155b48aecddd39ec3d162197a3096af61d31d1ef4d67df7`.
  Its chat entry was moved to the harness's decode group to request 128
  outputs; messages were unchanged. Scratch replay JSON SHA-256
  `7713ffcb26fbf674ebc67dd91a7230a603101990c00a0d82acd18bd5f63504ae`.
- Native raw results: `spark-b:~/scratch/qwen-draft-head/{prefix65536,full248320}/spec.json`;
  supervised job `qwen-draft-head`, 19:26:55–19:39:15 UTC. Before each load,
  free memory was gated at 105 GiB and existing compute processes excluded.
- Reference inputs:
  `spark:~/.local/share/jitllm/m3lc/prompts/qwen3.8/{8k,128k}.json`, SHA-256
  `3f889d5535fe1c31d5f1200f673c490c92e1671ea6ad1efbee734db85901649a` /
  `edc8ba23f091e1be9b0ec9fc6ba0d93b921e2a725a78c24a0ea49b9a4e8bde2a`.
- Reference raw results:
  `spark:~/scratch/qwen-draft-head/{curated47172-v3,prefix65536-v3}/`,
  supervised job `qwen-mia-draft-head3`, 19:44:43–20:11:00 UTC. The
  harness gates at 110 GiB free and excludes any existing compute process
  or running container before each load. Both loads started with 117.2 GiB
  free. The recipe uses its existing checkpoint cache through `HF_HOME`.

Native command: `jitllm_qwen38_spec --check greedy --tokens 128 --repeats 1
--only deep128k --context 131072 --draft 3 --draft-vocab N`, with the artifact,
tokenizer, replay prompt, and output paths above; N is 65,536 or 0 (full).

Reference command: `longctx.py vllm OUT PROMPT8K PROMPT128K --port 18150
--tokens 128 --top 0 --ready-timeout 1800 --start START --stop STOP`.
START runs the pinned `start.sh` with the baseline Docker wrapper in PATH,
`HF_HOME=~/.local/share/jitllm/baselines-20260928/hf`, `PORT=18150`,
`BIND=127.0.0.1`, `MTP_NUM_SPECULATIVE_TOKENS=3`, and one of the two
`MTP_DRAFT_VOCAB` paths. STOP is the pinned `stop.sh`. The complete
commands are in the external steps file and each run's `run.json`.

The recipe and list remain external baseline inputs. Nothing from Mia's
AGPL files is incorporated into jitLLM by this study. The existing
[license disposition](../../licensing.md#miaai-single-spark-recipe) remains
in force; any implementation must follow it and D-080.
