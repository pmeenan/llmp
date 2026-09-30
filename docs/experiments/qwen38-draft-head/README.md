<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 MTP draft-head selection (2026-09-29)

This study examines the draft-head difference raised by the owner while
closing M3's long-context MTP gap. The original width controls fix depth
at 3; the native selected-head trial also tests adaptive depth. It follows
the [MTP sweep](../qwen38-mtp/README.md) and
[long-context diagnosis](../long-context/README.md#phase-2-qwen38-flash-next-flat-with-depth).
The measurement question is whether Mia's selected vocabulary helps beyond
choosing a deeper draft window.

The separate [draft-input experiment](../qwen38-draft-input/README.md)
isolates BF16 rounding of the mixed draft-head input, with vocabulary and
depth fixed. It has no measured result or default change yet.

The selected list is now implemented and tested in jitLLM. Its 12.6% gain
was in Mia's engine; the native same-depth trial at 128K is tied at depth 2
and 4.9% slower at depth 3. The prefix remains the default. Increasing
native depth from 2 to 3 helps this prompt more than changing vocabulary;
the runtime now chooses between these depths from observed acceptance.
Our native width control also rejects a full head: it was 12.7% slower.

## What the implementations actually do

jitLLM's default draft head is a view of the first 65,536 BF16 rows of the
target's head, with no extra weight allocation. It returns the winning token;
its probability is computed only when a caller requests confidence.
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

The native trial also makes confidence optional: runtime greedy drafting
skips the probability reduction and copy. These measurements combine that
change with the new depth controls; they do not isolate its speed gain.

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
The native optional head adds this weight allocation and a 188,688-byte
I32 token map. The default prefix continues to share the target weights.

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

## Native selected-head and depth trial

On `spark-b`, the same checkpoint and target as above, with the independently
implemented importer, token-map gather and optional-confidence graph. The
importer packs selected BF16 rows and original I32 IDs into a separate
drafter artifact. It validates and pins the supplied list; the runtime
checks the map before GPU indexing. Every draft is verified by the target.
Mia's external list is used as experimental data and is not shipped.

The fixed-depth calibration uses the same 128,799-token prompt, context
131,072, 4,096-row prefill chunks and 128 greedy outputs, one run per case.
These short-generation harness rates differ from the runtime's earlier
512-output ladder; their absolute rates should not be mixed.

| Head | Depth | Draft ms/step | Verify ms/step | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Prefix | 2 | 6.740 | 50.345 | 39.370 |
| Selected | 2 | 6.101 | 50.106 | 39.295 |
| Prefix | 3 | 9.602 | 56.538 | 44.223 |
| Selected | 3 | 8.669 | 56.428 | 42.060 |

The selected head reduces the draft job's cost by about 9.5%, while verify
cost stays essentially unchanged. Its end-to-end rate is 0.2% lower at
depth 2 and 4.9% lower at depth 3; neither reaches D-085's 10% threshold.
The prefix's depth-3 rate is 12.3% higher than depth 2 on this prompt.
This single run per setting does not establish a corpus-wide preference.
Whole prefix-step costs are 57.603 / 66.786 ms at depths 2 / 3, the
measured ratio behind the policy's 1.16 calibration.

The runtime's adaptive greedy policy compares kept tokens per calibrated
step cost at depths 2 and 3. It starts with four depth-3 observations, then four depth-2 observations,
uses an acceptance EWMA, requires a 3% predicted advantage to switch, and
periodically probes the other depth. The depth-3/depth-2 cost ratio is 1.16
from the prefix calibration above. The policy uses no clock readings, is
saved with a conversation snapshot, and survives swaps. Seeded sampling
continues to use fixed depth 2; a three-row chunk bound also keeps depth 2.

The initial adaptive trial (depth 2 explored first) generates 128 outputs with two timed passes;
the second replays the first pass's depths and draft proposals as a controlled
timing pass. At 8K (7,586 prompt tokens) the prefix is 40.236 / 42.635 tok/s and
the selected head 39.992 / 42.376. At 256K (258,633 prompt tokens), the
prefix is 51.096 / 51.931 and the selected head 53.068 / 53.927. The 256K
policy uses depth 3 for 32 of 36 prefix steps and 31 of 35 selected steps.
Both heads match all 128 own plain-greedy tokens and their controlled passes
agree, without graph refusals or reported problems. Peak memory
drop is 87.66 / 87.80 GiB at 256K. These rates do not establish that
selected vocabulary is generally faster; continuation acceptance is prompt
dependent, and the measured differences remain below 10%.

At 64K (64,110 coding-prompt tokens), forced rejections cover keeping no
draft, one draft and two drafts. Both heads have 54 rejected steps and
zero state differences from their controls. Each has one own plain/speculative
near-tie among 160 teacher-forced outputs (margins 0.0284 / 0.0172, existing
bound 1.0), with no violations. The swap controls leave a rejected step's
commit pending, swap to the FP16 fixture and back, and continue with the
saved adaptive policy. Both produce identical state, logits and tokens to
their uninterrupted controls; 10 / 12 graphs survive.

Exploration order matters on the 128K prompt: starting at depth 2 settled
there and reached 40.402 / 40.908 tok/s (prefix) and 39.620 / 39.900
(selected). The final policy explores depth 3 first. Its prefix rate is
44.285 / 44.994 tok/s, with 35 depth-3 and eight depth-2 steps; selected
is 41.413 / 41.879, with 39 depth-3 and eight depth-2 steps. These fresh
passes replay depths but recompute all draft proposals; the repeats agree
exactly. This prefix
trial exceeds the recorded Mia rate of 43.475 on that 128-output trial.
The final runtime's original 512-output ladder below still has a gap at
128K. It matches 126 of 128 own
teacher-forced greedy tokens, with two near-ties (largest margin 0.0359,
existing bound 1.0), no violations, and exact replay. Selected matches
all 128. Neither reports a
problem. The changed schedule changes near-tie continuations, so these
are separate native trials, not matched-output acceptance comparisons.

## Final runtime ladder, 512 outputs

On `spark-b`, 2026-09-29 20:47:44–21:02:25 EDT, the final adaptive and
turn-reuse implementation (`4256331` code), context 262,144 and 4,096-row
prefill chunks. Each mode gets a fresh runtime process, a short warm-up,
then phase 1's original 32K, 64K, 128K and 256K prompts with 512 greedy
outputs. Every request completes with zero cached tokens. The selected
head uses the final converter's `8600a998…` artifact. Target caches remain
F16; the pinned Mia comparator uses its recipe's FP8 KV and MTP depth 3.

| Prompt tokens | Plain prefill tok/s | Plain decode tok/s | Prefix adaptive tok/s | Selected adaptive tok/s | Mia MTP-3 tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 31,743 | 2,322.46 | 26.53 | 39.12 | 37.00 | 37.25 |
| 64,110 | 2,315.47 | 25.97 | 37.55 | 41.32 | 40.45 |
| 128,799 | 2,266.02 | 25.43 | 45.03 | 42.88 | 48.65 |
| 258,702 | 2,176.36 | 24.67 | 35.63 | 43.84 | 37.71 |

Speculative prefill stays 2,139–2,270 tok/s for both heads at these depths.
Peak host `MemAvailable` drop is 84.86 GiB plain, 87.39 GiB prefix and
87.47 GiB selected, including weights, used state, workspace and page-cache
effects. The selected head is 10.0% faster than the prefix at 64K and
23.0% faster at 256K, but 5.4% slower at 32K and 4.8% slower at 128K.
These free-running trajectories differ; rates combine draft cost and
continuation-dependent acceptance. They do not isolate vocabulary cost.

The M3 long-context speed gate remains open: neither head reaches Mia at
128K, and the default prefix is also short at 64K and 256K. Plain decode
and all prefill rates exceed their recorded comparators. The shorter native
trial's different generation length and prompt rendering cannot substitute
for this ladder. These timing prompts are not the neutral retrieval gate:
some 512-token continuations remain in reasoning, so missing passphrases
in those continuations are recorded separately from retrieval correctness.

Raw results are external on `spark-b`,
`~/scratch/m3-extrapolation/qw-{plain,prefix,selected}/run.json`, supervised
job `m3-qwen-runtime-final`. Configurations and verified prompt-content
hashes are alongside them in `qw-*.toml` and `prompt-pins.json`; the same
original prompts and pinned Mia run are recorded in the
[long-context report](../long-context/README.md). No job or request errors
occurred; all twelve long requests returned 512 outputs and cached zero
tokens.

## Recommendation

Keep the prefix default and use acceptance to choose draft depth. The
optional selected-head import path enables further trials with externally
supplied, pinned lists. Broader English/code continuations are needed before
changing the default; no selected list is shipped here. Verify rows remain
the larger direct timing target.

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
- Native selected drafter used in the new trial:
  `e0876d69b8677bbecd83523a7fdf4d8cf1ef86c249447760b7de94efcae3f1e2`;
  source list SHA-256 as below, packed BF16 rows and original I32 IDs.
  Re-importing with the final converter changes the artifact identity;
  the pin-validation fixes do not change these tested tensor bytes.
- Final selected drafter:
  `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`,
  imported on `spark-b` at 19:02:35–19:02:47 EDT. Its indexed tensor
  payloads match the earlier trial artifact. An 8K fresh-proposal control
  repeats exactly; the runtime ladder above executes this final artifact.
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
- New native raw results: `spark-b:~/scratch/m3-curated/128k-{prefix2,curated2,prefix3,curated3}/spec.json`
  (fixed calibration, job `m3-qwen-curated-128`, 16:48:48–17:08:10 EDT),
  and `final-{prefix,curated}-{8k,256k,forced,swap}/spec.json`
  (job `m3-qwen-curated-final2`, 17:15:25–17:51:44 EDT). Its mislabeled
  128K rows actually used the 8K input and are excluded. All loads gate
  at 105 GiB available with no existing GPU model process. SDK
  `aarch64-e0a0c85c42806fb1`, CUDA 13.4.92, GB10 sm_121, driver 580.178.04.
- Correct 128K adaptive trials: `spark-b:~/scratch/m3-curated/final-{prefix,curated}-128k/spec.json`
  (depth 2 first, job `m3-qwen-curated-final128`, 17:52:08–18:00:37 EDT),
  and `fresh-{prefix,curated}-128k/spec.json` (final depth-3-first policy,
  fresh draft proposals in both passes, job `m3-qwen-fresh-final2`,
  18:31:33–18:40:19 EDT).
- New trial replay inputs on `spark-b`, one `decode` entry, messages unchanged:
  `~/scratch/m3-curated/prompts-8k.json`, SHA-256
  `26df3204068fbf83f6610d54ae2268cb6bffae322bbe6e764a4072fade3ad0b9`;
  `~/scratch/qwen-draft-head/prompts-128k.json`, SHA-256
  `7713ffcb26fbf674ebc67dd91a7230a603101990c00a0d82acd18bd5f63504ae`;
  `~/scratch/m3-curated/prompts-256k.json`, SHA-256
  `0089ad268538cb8783db3a71774a22ad3b9d6f73446e37190172f83d88fe0e43`;
  `~/.local/share/jitllm/m3lc2/deep.json`, SHA-256
  `5cee70e802ebbe5fcd2bb21ac0aca48dec1dedaecc4ac0c05779b3c582cd4686`.
  The 8K source is the reference input above; 256K is the phase-1 retrieval
  input, SHA-256 `c0e707dc67deab215bf53d4573406da4851c28bab943a937ad2eb39976aff538`.
- Swap fixture: artifact `b93cdc326ba4f4c1c71da613503ecd848cd2a0caf122214f26e04588a12a9073`,
  `~/.local/share/jitllm/p2-fp16exec-20260927/control-tokens.txt`, expected
  logits SHA-256 `bb8ae5e7e3ac6da734173edb1111160a0c80a55c4279b94e67a8f90b142e7571`.

Native command: `jitllm_qwen38_spec --check greedy --tokens 128 --repeats 1
--only deep128k --context 131072 --draft 3 --draft-vocab N`, with the artifact,
tokenizer, replay prompt, and output paths above; N is 65,536 or 0 (full).

New native command: `jitllm_qwen38_spec --qwen38-artifact STORE/TARGET
--drafter STORE/HEAD --tokenizer CHECKPOINT/tokenizer.json --prompts REPLAY
--only deepRUNG --context CONTEXT --prefill-chunk 4096 --check greedy
--tokens 128 --repeats 2 --draft 3 --adaptive-depth on --out OUT`.
The calibration uses `--repeats 1 --adaptive-depth off --draft 2|3`.
The coding controls use `--only deep64k --context 65536 --tokens 160
--repeats 1 --check forced|swap`; swap also supplies the pinned FP16
fixture, control tokens and expected digest recorded in the external steps.

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
