<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek IQ2 worklist sharing: rejected

Building the native J128 worklist once for both gate and up saved 0.039696 ms,
about 0.2% of the resident pair. Retain the existing implementation. This
single operator experiment does not explain the roughly 12% resident-consumer
rate gap against the original ds4 path and makes no whole-model performance claim.

The captured layer-21 input has 4096 tokens, K4096/N2048, 256 experts and six
selected experts per token. Gate and up each use the same 553,648,128-byte
raw IQ2 bank. Both arms use the already qualified native D4 activation and
inverse-source/destination maps. Only the worklist construction changes:
two builders become one; the two original J128 numerical products, launch
geometry and 358 live work items remain unchanged. The compact scratch is
3584 bytes, with capacity 448; products use grid16×448×1, 256 threads and
57,856 dynamic shared bytes. Building the worklist remains inside the timer.

| Chronological arm | Median CUDA-event pair, ms |
| --- | ---: |
| Ordinary, first | 19.688192 |
| Shared builder, first | 19.667168 |
| Shared builder, second | 19.696993 |
| Ordinary, second | 19.755360 |

Each arm has nine positive event samples and a wall measurement. Means of the
two medians are 19.721776 and 19.682081 ms; the corresponding rate ratio is
1.002017. Ordinary bookends differ by 0.34%, shared bookends by 0.15%, so
the observed gain is smaller than the ordinary bookend movement.

All eight complete gate/up outputs are finite and byte-identical to their
native J128 calibration, including own repeats and bookends. Each output
contains 50,331,648 F32 values; CPU comparison found zero changed values,
maximum error, RMS error and NMSE. The unchanged 24 FP64 witnesses are
descriptive controls. Raw-output equality here does not establish task quality
or equality to ds4's different numerical path.

The diagnostic adds an O3 bridge to a complete private copy of the locked
GGML source tree; it deliberately disables source-lock enforcement for that
GGML-only diagnostic override. The actual original numerical instance has
one defining translation unit and its unchanged source/flags are retained.
Compiled resource attributes were unavailable for this instance; no
occupancy or register claim follows. No production source or selector is
adopted from the diagnostic.

Measured on Spark A (spark-c4e2), the target build took about 129 s and the
operator job about 49 s. Both completed in their supervised process groups
with retirement proved. Terminal operator retirement reported 116.740 GiB
free and no active model. The actual A SDK receipt is f38891fc…, with SDK
cuBLAS ee7c1657… and cuBLASLt ba3b942f….

Evidence outside Git is under
`~/scratch/m3-ds4-native-iq2-shared-builder-r1/` on Spark A: actual target
3acb13d8…, outer operator4aa825dd…, native655f0bd5…, full-output
analysisbe8f21b7… and preservation75952299…. Preservation retains 1707
copied source/build/controller files and authenticates 37 complete raw
files retained in place. Source map7a86e88e…, binary213bfdc6… and defining
object3efe6f16… bind the measured implementation.

The next useful native IQ2 scope is raw-weight loading/staging or MMA packing.
This result closes duplicate worklist construction as a substantial source
of the consumer gap. The original J128 and rejected J64 results remain in
[the consumer report](../ds4-native-iq2-consumer/README.md) and
[the J64 report](../ds4-native-iq2-j64/README.md).
