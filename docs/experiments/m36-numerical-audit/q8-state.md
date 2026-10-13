<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded dense Q8_0 cache opt-in

The shared dense-attention component now consumes existing quantized cache stores
and vector attention at D64/D256. F16 remains the default. The explicit per-alias
setting changes cache state, not weights, and does not widen serving admission.
This first screen measures incremental loss against the same checkpoint with
F16 cache. It establishes functional/state support, not an acceptable-loss
threshold, family quality approval, long-context behavior or a speed gain.

## Paired short teacher-forced quality

Spark A; fixed natural text/IDs selected before losses, plain teacher histories,
full finite vocabulary rows, all next-token targets scored. Each pair uses the
same artifact, histories and source policy; only the cache-encoding selection
changes. Its Q8 attention path also quantizes
queries to Q8_1 inside the existing kernel. The measured delta therefore covers
the selected Q8 path, not an isolated storage-only error.

| Model / cache head | Targets | F16 mean NLL / PPL | Q8_0 mean NLL / PPL | PPL change |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B FP16 artifact / D64 | 255 | 4.306976343547 / 74.21574642 | 4.351930424557 / 77.62817371 | +4.597983% |
| Gemma2 2B Q8_0 artifact / D256 | 91 | 5.974900297411 / 393.42887326 | 5.989320294526 / 399.14321787 | +1.452447% |

Both samples show higher loss with Q8 cache. They are short single-passage
observations, not a general quality estimate or a predeclared acceptance gate.
The public option remains explicit; automatic calibration does not choose it.
Original unquantized evidence is not required for this same-weight cache
comparison and no closer-to-original claim is made. The Qwen FP16 artifact's
conversion lineage remains separate from this paired comparison.

## State and memory scope

Logical Q8 cache bytes are 34/64 of F16. Qwen reports layout sizes
6,291,456→3,342,336 bytes. These are state-layout extents, not measured total
process or device occupancy. Gemma's logical state is
54,525,952→28,966,912 bytes, but both actual reserved layouts are 109,051,904
bytes because 52 banks round to 2 MiB each. No physical memory reduction is
claimed for that short Gemma context 512 setup.

The Qwen 76-position same-Q8 control compares 252 complete replay rows across
four evaluations, with checkpoint/spill copies of 3,342,336 bytes, two full-weight
evictions and 980 reloaded extents. The actual selected corpus has 24 quantized
attention operations, and the two replay shapes have 24 each. No cross-encoding
head-bit equality is required.

Gemma uses the normal Server alias and actual native tokenizer. The Q8 control
compares complete initialized 36-position state after checkpoint reconstruction,
spill and fresh-server kept restoration, plus the following 37th-position full
head after spill/kept restoration. There is no independently saved 37-position
state oracle. Its context 512 test does not wrap the 4096-window ring; odd-cell,
blocked geometry, ring/truncate and unsupported snapshot alignment have host
controls. D512, specialized QSA/MQA, sinks, borrowed state and drafters remain
outside the opt-in domain.

## Provenance and bounded execution

Main native build plus 519 host cases across 27 targets pass. The earlier
140-host proof remains applicable; the fixture-only compile and main build
used identical model binaries.
The four actual cases complete with no failures/skips, complete process-group
retirement, zero remaining GPU processes and zero memory-limit/OOM events.
The process/cache peak 480,501,760 bytes is not GPU or whole-model occupancy.
No timing comparison was run. Source/metadata/binary guards pass before and
after execution. Source quality checks are recorded by the final main carry.

Raw receipts stay in coordination scratch: `q8-state-main01/result.json`
SHA `93c46d6b1d3c589260b7e39e948d70428dee2f4e7928a72dd7ff76a20e3947df`
and `q8-state-models02/actual/result.json`
SHA `dd1ed6f9837c68e413a9ee38cb740523f46616ab02df83bfddc93dafaa146d93`.
The Qwen binary SHA is `acc3da660d89c60d70dc48a1fdbc3fef54b3262eafe5ef8f1abeeea2691b80dc`;
Gemma is `4d2d8d2a3fe84f66b54f35ec4ab7ae5ce5284f95242532cd940692cd1e7e3d61`.
Source base 6e939ff1 plus the reviewed 19-path slice; no other arithmetic changes.

Qwen artifact `b93cdc326ba4f4c1c71da613503ecd848cd2a0caf122214f26e04588a12a9073`;
256-ID input SHA `1ca6f2d78042d95978dd4ab34bc95b85dae14ab2c8ee454c8facdf51cac7d423`
reuses the authenticated natural corpus document 0 from the attention qualification.
Gemma artifact `eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870`;
literal UTF8 text SHA `010a99529ef187e6c90769840869608e57c33ccc9737e324faf7d87e299d2f29`,
actual 92 native IDs SHA `dc0714ab2ad1ef7af6d6ceba7a648744337ce0ac85a4b30ede6507fbe68ddfaf`.
The fixed literal remains in the model test; neither losses nor observed outputs
selected the text or cutoff. Full runtime/payment proof is retained in the
reviewed source, D64 applicability, D256 fixture and models02 packages.
