<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native draft-vocabulary study

The owner requested a causal comparison of the prefix and externally
curated Qwen3.8 draft vocabularies, followed by a jitLLM-curated candidate.
The existing context ladder changes prompt content as well as depth, and
its free-running continuations differ. It cannot establish that context
length itself determines which vocabulary wins. No new measurement or
curation result is claimed by this protocol.

The target is the unchanged `c4fb47a9…` checkpoint. Controls use the
65,536-row prefix head `056a750e…` and externally supplied 47,172-row
head `8600a998…`. Preserve the native BF16 head weights, F32 inputs,
original-ID tie order, F16 KV, F32 recurrent state and target verifier.
Selecting vocabulary rows changes the proposal, not the target model.
Use the existing generic selected-row importer; a new list is produced
from independent calibration data, without copying the external list.

## Separate coverage, cost and adaptive depth

First compare both lists on identical captured native MTP input vectors.
Project these vectors through the full original vocabulary once, with
the unchanged native product, then rank each subset using original token
IDs. Require the subset product to reproduce its original captured
logits before interpreting rankings. Report which list contains the
full-head winner, each subset's winner and full-head rank, differences
between those choices, head projection cost and complete charged draft
cost. Full-head projections are an analysis reference, not a proposed
serving default; earlier full-head model trials were slower.

Then compare complete proposals at common target-history anchors. Each
arm drafts and verifies its own unmodified proposals and records natural
acceptance by draft position before restoring the common pre-proposal
checkpoint. Restore the complete target and MTP state: recurrent state,
cache/frontier coverage, pending rows, catch-up position and draft substrate,
as well as the committed history. Advance the frozen known continuation
through one control path, then capture the next common checkpoint. Verify
rollback and MTP catch-up exactly against that control, including identical
MTP input vectors at each anchor. Do not clamp acceptance or treat committing
one common token as proof that the MTP state matches. Report this forced
anchor diagnostic's cost separately from free-running decode throughput.
An ordinary free-running comparison does not substitute for this control. Keep
depth three fixed first. Compare adaptive depth separately after the
vocabulary effect is understood.

Use the same coding question and answer continuation with increasing
irrelevant source context to isolate depth, plus distinct code, prose,
instruction-following and arithmetic prompts to isolate content. Freeze
rendered prompts, exact IDs, continuation IDs, context capacities and
anchor positions before collecting candidate results. Measure 32K,
64K, 128K and 256K only where the bounded first comparisons justify it;
retain actual lengths and all failures. Report acceptance, useful tokens
per verify, draft/head/verify costs and complete decode throughput.

## Independently curated candidate

Freeze calibration and held-out sample identities before collecting
outputs. Calibration must include code, prose and reasoning rather than
just the existing M3 coding ladder. The already inspected M3 prompts and
their outputs are diagnostic cases, not an unseen evaluation set.
Standard harness tasks may supply distinct calibration and evaluation
samples, but evaluation answers and task scores must not enter selection.
Record task, dataset revision, license, sample IDs, tokenizer and template
hashes with the input manifest. Synthetic long-context calibration uses
different seeds and facts from acceptance cases.

Use four equally weighted calibration domains: code, prose, instruction
following and arithmetic/reasoning. Each domain has equal weight at 8K
and 32K, and each example within a domain/depth cell has equal weight.
Average anchors within an example so a longer answer cannot dominate.
Freeze the exact examples, anchor counts, rendered IDs and source licenses
in a manifest before collecting full-head or target outputs. Missing or
failed examples make construction incomplete; do not renormalize them away.

For each vocabulary ID, calculate its weighted frequency as the full-head
greedy proposal and its weighted frequency as the known target token at
the corresponding anchor. Rank by `0.75 * proposal_frequency + 0.25 *
target_frequency`, with ties resolved by the lower original ID. Frequencies
use the same normalization described above; neither ranks nor held-out
task scores enter selection. Reserve every stop ID from the frozen target
model metadata, then fill the remaining slots from this ranking, excluding
already reserved IDs. Zero-frequency rows consequently fall back to the
lower original IDs. Require exactly 47,172 distinct in-range IDs and write
them in ascending original-ID order. This objective and fallback order stay
fixed before any output collection.

Equal width with the external head isolates membership. A smaller or larger
list is a later cost/coverage experiment. Store the list hash, source manifest
and deterministic construction command. Never change weights, use held-out
answers in selection or alter the list to repair a particular test.

Compare the frozen candidate with both controls on held-out common
histories and then fresh free-running requests. Every result carries
source/binary/artifact identities, actual selected rows, proposal coverage,
acceptance by position, complete timing and memory. Speculation must
still meet the existing target agreement, own-repeat, forced-rejection,
saved-policy and restored-continuation checks. Where seeded speculation
is enabled, retain its distribution gate. A draft vocabulary is not a
quality/speed tier that permits changing the target distribution.

No context-dependent head selector is justified by the current ladder.
Consider a deterministic runtime policy only after held-out evidence
shows that measurable acceptance/cost predicts a useful choice. Keep the
prefix default until a complete candidate earns the change. Raw vectors,
full logits, answers and traces remain external; the completed report
will retain aggregate results and reproducible provenance.

## Installation-specific adaptation follow-up

The owner deferred this idea to later optimization passes: learn a draft
shortlist from the installation's actual work. Evaluate this separately from
the frozen calibration candidate above;
online feedback must never enter that candidate or its held-out evaluation.
Start with a broad model/tokenizer-specific list and reserve its special/stop
tokens. Maintain bounded local aggregate counts of committed target output IDs
and verifier-selected IDs missing from the draft list. Prompt IDs may provide
a separately measured signal for identifiers, but do not replace output
coverage. Compare a fixed list with a rolling-frequency list on identical
chronological requests, freezing the update rule and schedule before replay.
Use recent-weighted counts so old tasks do not permanently determine the list.

Rebuild selected weight rows only between requests, with explicit head/graph
identity and invalidation; retain the full target vocabulary and verifier.
Charge row copying, preparation, memory and graph rebuilding to total request
time. Record acceptance by position, out-of-list verifier choices, useful
tokens per verify and complete throughput through workload changes. Preserve
a broad fallback and require existing agreement/distribution gates. No weight
training is needed for this shortlist experiment. Persistent aggregate counts
would be an explicit runtime option with a reset; no prompt or answer text is
retained. No runtime adaptation is implemented or justified by this proposal.

For long conversations, also test a bounded conversation-specific portion
beside the broad core and installation-level portion. Rank it from recent
committed assistant outputs, with user-introduced identifiers as a separate
signal; decay old counts so a topic change can replace stale membership.
Compare this mixture with a fixed list on the same chronological turns and
charge every between-turn rebuild. Conversation counts have explicit scope
and expiry when conversation identity is available; prefix matching alone
does not identify a conversation or its lifetime (D-031). Otherwise use the
bounded rolling installation signal. This is a deferred experiment, not a
chosen partition size or a measured benefit.

For the broad starting list, evaluate a portable curated text corpus across
models: prose, programming languages, identifiers and structured output such
as JSON/tool calls. Tokenize it with each target tokenizer, including space,
newline and punctuation boundary variants; exact matching against decoded
vocabulary labels misses fragments and boundary-dependent tokenizations.
Add the model's actual output-format, reasoning and stop delimiters from its
metadata/template, distinguishing prompt-only role markers from output
tokens. Combine the resulting ID frequencies with representative generated
answers. Freeze this seed corpus and construction rule before replay; it is
a proposed later baseline, not a change to the fixed candidate above.

Frequency-ranked compression has a published precedent in
[FR-Spec](https://aclanthology.org/2025.acl-long.198/); per-step vocabulary
selection is studied in [SpecVocab](https://arxiv.org/abs/2602.13836).
Neither paper's speedup is a measurement of jitLLM or proof that this proposed
between-request policy is faster on the GB10.

## Execution budget

Collect a small calibration/capture sample and evaluate its charged
operator cost first. Estimate each subsequent model batch from the
actual token counts and measured rates, and state its expected duration
before launch. Any batch over ten minutes must decide the current head
choice; a broad recurring suite is not part of the default check set.
All model loads use the existing free-memory and process gates, one
Spark per timed batch, and supervised application completion. This
protocol defines the study; it does not claim any runs have happened.
