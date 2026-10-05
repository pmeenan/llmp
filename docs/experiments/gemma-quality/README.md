<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 26 representative likelihood screen

The first production-reference screen fails quality: native perplexity is
10.03% higher and 476 of 1,024 teacher-forced argmax choices differ outside
the frozen zero-noise allowance. No execution policy is selected from this
result. The ordinary native and independently norm-fused native heads are
byte exact on every row; the production reference repeats are also byte
exact. Full-model qualification remains open.

[Protocol](PROTOCOL.md) fixes the context, teacher-forced arithmetic and
score alignment. This is the first 1,024 token IDs of the existing public
domain War and Peace corpus, with BOS 2 and 1,023 scored targets. It is a
short prose-prefix screen, with no chat template. Both arms have high
absolute perplexity on this input; that does not waive their discrepancy.
The shared LE I32 prefix is 4,096 bytes, SHA-256
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
The input is reproducible with the pinned reference harness's `prepare`
mode on the complete corpus named in the protocol.

| Policy | Mean target NLL | Perplexity | Argmax differences from native |
| --- | ---: | ---: | ---: |
| Native ordinary, device masks | 7.0751330644224675 | 1182.200799205535 | — |
| Pinned reference, fusion enabled | 6.979519368234117 | 1074.4018512197504 | 476 / 1,024 |
| Pinned reference, fusion disabled diagnostic | 7.0751330644224675 | 1182.200799205535 | 0 / 1,024 |

The follow-up fusion-disabled reference and its repeat are byte exact to
native across **all 1,024 full-vocabulary heads**, with identical NLL and
PPL. Direct follow-up SHA-256 verification of all three complete 1 GiB
files (native ordinary, unfused first, unfused repeat) gives
`1c66029f1f088e04a7a4001fbdd0e24afbb97ad3b9e85f20fb73c58292e4b340`.
This control finds no additional native 128-row discrepancy. It does
not qualify native against the fusion-enabled reference, establish
that the reference fusion is defective, or explain every changed operator
in this representative input. The earlier six-row routed-chain diagnosis
is narrower and is not silently generalized to these 1,024 rows.

All later input IDs are explicitly teacher-forced, so an argmax difference
does not change the prefixes used to compare later rows. The final input
head is retained but excluded from NLL; BOS has no target score. The
native-own allowance was frozen before cross-engine analysis, using the
ordinary and norm-fused arms' top-two margin movement. Its p99 is exactly
zero, with all 1,024 complete vocabulary rows byte exact, and all 476
production-reference differences fall outside it. The calibration JSON
SHA-256 is
`2c97b2a4d65277f00b242180d7f54128a395284490d84db0db24e634adcfc901`.
It remains separate from the earlier literal scalar calibration.

The source is reviewed native base `d5539c8`, on physical Spark
`spark-c4e2`. Native shared-Q8, row-invariant and RoPE/store policies remain
off. The same-format pinned llama.cpp reference, container, approved model
and latest TensorFold observation are fixed in the protocol. This
128-row numerical shape control does not constrain the reference's later
competitive prefill chunk size. It does not establish scalar, multi-request
batching, memory or performance qualification.

| Measured artifact | SHA-256 |
| --- | --- |
| Native quality source | `b744cc557b59ac6a95585ed867e56ee801bd8859abb69769538fc3d8a675500f` |
| Native quality executable | `f1115eccc7fae851ce1afe9778bd5411931ef0cbc993bc76877fd59f61b32080` |
| First production-reference source | `0c78afb13a8d93b0dda47462da7123f143884c495ee2cb2a08351347806786b6` |
| First production-reference executable | `c095a7dbcd8ed01b703009656ea063da58af2f071e5aab5bd4d46f55152fecfb` |
| Explicit diagnostic reference source | `eac301a6130cf0e7543e1fa2b90c70b7005f46a9fc45121c9add6dbba7695c2a` |
| Explicit diagnostic reference executable | `e291d8864695c1e4c80a1324fb0758ce127700584384aa3069239ee28cf593f0` |

Native invocation is `jitllm_gemma_quality ARTIFACT IDS_I32 NEW_OUTPUT_DIR
128 ordinary`, and the independent calibration arm substitutes `norm`.
Reference invocation is `llama_quality RAW_GGUF IDS_I32 NEW_OUTPUT_DIR
score 128`, with CUDA fusion and graph disable variables absent. The
reference harness is compiled against the pinned image's C API and
libraries; it is external to jitLLM. `analyze.py ROOT calibrate` creates
an exclusive native-only freeze; `analyze.py ROOT oracle` verifies that
freeze before reading the reference rows. Neither operation replaces an
existing calibration or comparison. Full-logit analysis uses finite
F32 rows, FP64 shifted exponentials and `math.fsum` for normalizers, NLL
and total variation. The diagnostic invocation substitutes `score-unfused` and explicitly sets
`GGML_CUDA_DISABLE_FUSION=1`; graphs remain enabled. Analysis substitutes
`oracle-unfused` and writes a separate comparison, retaining the original
production result and native freeze. The first production source predates
that explicit diagnostic mode and orderly context/model teardown; its
measured source identity remains separate. Raw vectors and logs remain
external.

The initial comparison's byte-exact counter used numerical equality,
which cannot distinguish F32 +0 from -0. The retained whole-file hash
verification establishes byte identity independently of that counter.
The corrected analyzer counts actual row SHA equality, with a signed-zero
rejection control; original measured analyses, source identities and
calibration remain retained. No model runs or NLL scans were repeated for
this evidence correction.

Installed supervised Spark jobs completed the locked native build, two
native calibration arms, two production-reference repeats and two unfused
diagnostic repeats; both full-row analyses completed successfully as
diagnostics, with the production-reference quality failure retained.
