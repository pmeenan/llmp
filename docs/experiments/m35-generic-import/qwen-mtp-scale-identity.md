<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen ModelOpt MTP scale declaration identity

The historical approval digest named the shared scale declaration
`projection_scales=true`. The native mapping introduced in `e9de1e83`
uses `projection_scales=false` and the three explicit declarations
`gate_scale=true`, `up_scale=true`, and `down_scale=true`. The selected head
has 47,172 rows in both descriptions. The approval's recorded spec digest is
updated to the current declaration; source receipts, transforms, tensor
inventory, envelope and numerical evidence are unchanged.

The current composed spec has SHA-256
`b686a1f3178a95009d7f94ca954604df2898d9dd757686891827b5302b1af860`.
Replacing only those four scale parameters with the historical declaration
reproduces the old approval digest exactly:
`f85a3348251a8ffe2b8d0412a9e6ca12ec7b548f53d5ba17c61d88a83ad99a37`.
This is an offline diagnostic reconstruction, not an admission conversion.
The current JSON came from the real published target/drafter pair on Spark A;
its extracted vocabulary, selected head and output rows all equal 47,172.
Changing the output rows to 65,536 instead does not reproduce the old digest.

The shared MoE component derives each effective scale flag by OR-ing its
individual declaration with `projection_scales`. Required weight roles, IR
emission and the CPU reference all consume these effective flags. The focused
regression compares the required roles, graph digests and exact reference
outputs for stacked and separate expert tables, single and multiple owners,
and distinct non-unit scale values. Disabling one individual scale is a
negative control: its required role, graph and output must differ.

The existing structural eligibility regression compares the numerical
qualification entries and their domains for both descriptions, including a
changed-encoding refusal. It retains each actual semantic identity. LLM
numerical qualification binds the actual source receipts and admitted spec;
its historical approval pointer supplies no permission. The strict `Pinned`
helper applies to image components, which this change does not touch.

Diagnostic evidence used the exact W7 product inputs plus assertion-message
changes only. Its source manifest SHA-256 is
`f93dd6d7347be91de56fbaa2cd55fe3d3a70bb479fb5fa4495ed5349c0fc579c`;
the seven-case diagnostic XML SHA-256 is
`86b7c3bb6ab4fdd9d31bbf5539e1ccfffac18f2a372c7aaa258ef42a8350bb81`.
The original failed approval assertion remains recorded. Native validation on Spark A passed: the new equivalence/negative-control
regression, the existing MoE graph/reference comparison and the structural
numerical eligibility regression (three cases, no skips), followed by the
real published-artifact/pair approval test (one case, no skips). The latter
passes with the current canonical 47,172-row composition. These are
functional controls; no new inference quality or speed claim is made.
