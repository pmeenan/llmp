<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 selected-HC native recipe evidence

G3c2b1b1 corrects the native recipe for the final selected-row HC low/high
products. The shared policy stays default-off and requires proof of the actual
selected output rows and the complete final HC pair before selecting the
existing per-column vector recipe. G3c2b1b remains open: G3c2b1b2 requires
joined heads/state, changed-input replay, old/new persistence and bounded
admission.

The earlier K3 report found unequal outputs with equal complete weights and
equal immediately consumed inputs for all four owner/product comparisons. Its
`output.bytes` is the compared buffer length, not a count of differing bytes;
`first_byte` is the first differing byte index:

| Product | Full output length | Owner 0 first difference | Owner 1 first difference | Reported 1-column → 2-column branch |
| --- | ---: | ---: | ---: | --- |
| Low (`hc_down`) | 1,280 bytes | index 0 | index 0 | `GemvBf16Split<10>` → `cublasGemmEx` |
| High (`hc_up`) | 20,480 bytes | index 7,848 | index 5,442 | `GemvBf16Rows<2>` → `cublasGemmEx` |

The 09c exact bounded screen at alignment 256/context 512 reports 118 checked
boundaries and 112 equal records. Its four K3 product comparisons are equal:

| Product per owner | Weight | Consumed input | Output buffer | Reported 1-/2-column branch |
| --- | ---: | ---: | ---: | --- |
| Low (`hc_down`) | 6,553,600 bytes, equal | 20,480 bytes, equal | 1,280 bytes, equal | `GemvBf16Split<10>` / same |
| High (`hc_up`) | 6,553,600 bytes, equal | 640 bytes, equal | 20,480 bytes, equal | `GemvBf16Rows<2>` / same |

For K1 and K3, each owner's 10,240-byte head input and 188,688-byte logits
match across the complete 47,172 vocabulary logits per owner. The reported operation
branch labels are source-resolved from bound fields and immutable compiled
source; no new dispatch instrumentation was added. The comparison does not
show that the earlier low-output difference propagates through exact `lo`.

**Provenance:** source package manifest SHA-256
`9f6edee71f8c069e4072ee23d05ad8b0bb45b651977821e36f70f1d358d72553` (2,404
files; source JSON `2d9e16b476dd4e41f2d7f68a4f91d9ad5b6ce5811e687f5df739dc94e7030d18`);
executed source manifest `b516164135003db2b27a1009c80174ef16e429db8c4e8e9d4e9321666382b829`;
recipe manifest `0b8c3ac17fea0a3dd4a8a68affd2f574f2d582b0a9bf7a047f871262dc8e8fa1`.
Reports: 08 `66e62259f78edf6c3d669a79ca0e1d97958a997464645a381ceb09651c53c9b9`;
09c `fd6a82c68891a9c4986b94e19d612098af6726684d79b2e30cfbb9da74f14556`.
Execution host: `spark-56f5`; SDK:
`/home/pmeenan/.local/share/llmp/sdk/aarch64-2c4eb5b2f68f3fa5`; artifact
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`, version
`0`.

The cleaned correction passed a separate model-free regression using
the actual `hc.mix` component emitter, canonical `StepContext::OutputRows`
and `InputValues`, and ordinary provider/`StepProgram` execution. It compares
all 2,560 F32 mixed-output scalars per owner against independent singleton
programs for row counts 1+1, 1+3, 3+1, 3+3 and 4+4, with unsorted owner slots.
The same recorded joined program is reused after selected-coordinate changes;
a further replay changes only nonselected rows. Measuring and bound input,
output and use ranges are checked, and uncertain completion retains the whole
owning environment. On Spark B, the focused clean03 build, 169 host checks and
six non-model GPU checks passed without failed or skipped cases, including
this integrated regression. This synthetic component result is distinct from the diagnostic's complete 47,172
vocabulary logits per owner and does not exercise a target/drafter state
transition or native joined request admission.

The clean03 source package is
`17d07456d9bf2c64271cce17acee74a35936e7e13f47cafb1c0f9f1bc08d50bb`;
its complete 2,403-file source identity is
`9d6c2d7ea844634af38ce04601e5a783a09dd493e8eb0093da1e49dc4b136013`.
All six production correction paths are byte-identical to the executed 09c
screen. The cleaned tree removes the scratch observer and model-test overlay;
its permanent integrated test uses normal provider inputs and completion.

This screen makes no claim of joined state correctness, state persistence,
continuation, admission, end-to-end propagation, production qualification, or
the matched 1% migration gate. The owner-model-specific Qwen3.8 HC-sharing M3
experiment remains separate and unchanged.
