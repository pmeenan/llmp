<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 assistant: frozen C1 component control

The approved Gemma31 Q8_0 assistant reproduces every complete original-image
head and recurrent feature byte for byte on one frozen C1/P64 input. One-step
and three-step endogenous chains, each repeated twice, compare all 262,144 F32
head values and all 5,376 F32 postprojection values. Greedy mismatches and maximum
raw differences are zero. Both engines' own repeats are finite and exact.
This closes a bounded component prerequisite; assistant serving, speculation,
full target-chain quality, C2 and performance remain unqualified.

The target's POST-finalnorm feature is row 63. Every assistant query remains at
position 64 and borrows local layer 58 (D256, 16 KV heads) and global layer 59
(D512, 4 KV heads), including the separately authenticated raw K-as-V source.
The completed prefix is 64, the padded read is 256 and physical capacities are
1280/4096. Original complete caches, cell metadata and serialized target state
remain unchanged after every draft step/repeated chain. The actual opaque
sequence-state length is 57,674,696 bytes, below the declared 64 MiB bound.

Native protection covers all four initialized borrowed tensors, 52 MiB in total,
including zero tails beyond the supplied read. Thirteen complete witnesses
match before/after individual steps and chains. Each copy retires before reuse
of the separately catalog-funded 16 MiB pinned buffer. The existing 128 MiB caller
host envelope precedes parsing/vector construction. The reference uses separate
64 MiB opaque-state and 384 MiB known-vector bounds; its model/backend memory is
additional. These are bounds, not measured process peaks or a serialized native
target checkpoint. Native execution includes one eager run, one capture and
six replays; complete heads/features publish after successful retirement.

The original own-freeze preceded release of only eleven stage-zero input files.
Native endogenous own-freeze preceded reading original later recurrence or full
head/feature payloads. No posthoc replay, noise allowance or tolerance was used.

Reproduce with the approved prepared target/assistant pair and the source-bound
[original protocol](../gemma-assistant-reference/PROTOCOL.md#closed-31b-c1-extension)
and [native protocol](../gemma-assistant-execution/PROTOCOL.md).
The existing wrapper's `build31`/`acquire31` modes select the pinned original
image and 19-header closure; use `own_freeze.py --profile 31`. Native invocation is
`jitllm_gemma_assistant_fixture --profile31 STAGE0_DIR STAGE0_MANIFEST NEW_OUTPUT`,
with the qualified SDK/cuBLAS library path. Use exclusive outputs and preserve
the original-own → stage-zero → native-own → comparison order. Legacy 26
invocations retain their existing bounds and behavior. The build receipt reports
`0.1.0-dev+unknown`; the measured source6 and full checkout hashes explicitly
identify the 57f5463-based build.

Eight supervised jobs retired successfully. The first build failed after six
of nine steps because the caller omitted copying the existing retirement helper;
its public-build step refused before any container/model ran. The authenticated
public-only recovery preserved source6 and the compiled native fixture. The
failure and all official records remain external; aggregate identities are in
[results.json](results.json). No full regression suite or timing benchmark ran.

At task entry TensorFold primary HEAD was
`cb2ebf0540f42604e2759b2ddef497861e928248`, version 0.6.6. Its MLX recipe is no
CUDA Q8_0 assistant comparator. Raw inputs, vectors, state, IDs and logs stay
outside Git. No production default or target arithmetic policy changes.
