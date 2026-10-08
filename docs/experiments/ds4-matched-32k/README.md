<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Complete ds4 reference at 32K

The temporary native reference matches original ds4 at 32K: all 129,280
final logits are byte-identical, and mean throughput is 0.71% lower on the
same Spark. This extends the [8K pipeline control](../ds4-complete-plan/README.md)
to a context that exercises original indexer score and deep selection.
It establishes fixed-input, single-request pipeline parity, rather than a
task-quality gate, decode result or concurrent-request result. Production
selection and memory guards are unchanged.

## Matched conditions and result

Both engines run on Spark B with the authenticated community IQ2_XXS GGUF,
the same 32,768 token IDs from the retained community perplexity corpus,
context capacity 32,768, all 43 layers, eight 4,096-row chunks and one final
full-vocabulary head. Original cache/product precisions, quantization
producers, physical representations, selector dispatch and output scope are
preserved. MTP, DSpark, profile marks and restoration factors are disabled.
Each process performs one explicit warmup and three fresh, initialized
passes. The native resolved dispatch uses ordinary stream 0.

| Engine | Fresh pass seconds, including final result copy | Mean seconds | Tokens/s |
| --- | --- | ---: | ---: |
| Original ds4 | 30.741708638 / 30.887717338 / 31.030244683 | 30.886556886 | 1060.914628 |
| Native reference | 31.109107350 / 31.065924273 / 31.147558492 | 31.107530038 | 1053.378393 |

Native time is 0.7154% higher and throughput 0.7104% lower, within D-085's
10% threshold. Native maximum/minimum pass time is 1.00263; original is
1.00939. Preparation and weight loading are outside these prefill timers.
The native comparison charges `prefill_wall_seconds + result_copy_seconds`
against the original timer, which includes its final GPU-to-host read.

All four native heads, including warmup, are finite and byte-identical to
each other and to all four original heads: every entry in the 4 × 4
comparison agrees. Head SHA-256 is
`b3d5229f2fe1ce330d4afdee14c4d1cd068ea743e078b0b1461163dabd41c3f5`.
This input exhibits no original or native selector-induced repeat difference;
it does not prove every possible input deterministic.

The native pass records 369 completed jobs, eight chunk dispatch records,
21 CSA indexer layers per chunk, and final-head dispatch only in chunk 7.
Compressed score bands grow from 1,024 to 8,192. Actual native selection is
bitonic-1024, bitonic-2048 and bitonic-4096 in the first three chunks, then
original CUB-8192 for all five deeper chunks. CUB temporary storage is
67,584 bytes against 101,376 bytes of device opt-in shared memory; the
selected dynamic shared-memory request is 67,584 bytes for CUB and zero
for bitonic. Thus each measured pass executes 105 deep CUB selections.
Native choices are observed in resolved dispatch; original choices are
qualified against its unchanged host dispatch source, without added original
runtime telemetry or altered selection algorithms.

Preparing 84,512,276,480 stored bytes takes 293.549 seconds; native page-in
takes 6.462 seconds. Declared execution budget is 99,182,063,104 bytes,
including 744,488,960 bytes of rounded state. Sampled native peak RSS is
901,623,808 bytes and peak node MemAvailable drop is 103,049,506,816 bytes.
The corresponding original samples are 8,423,718,912 and 101,714,128,896
bytes: ratios 0.10703 and 1.01313, respectively, both within 1.1×. Node
drop includes model, paging and file cache; RSS is a separate process measure.
Native child and supervisor exit 0, child reaping is confirmed, and the strong
retirement probe clears with 117.034 GiB available.

## Allocation refusal and scoped cache control

The first native attempt completed preparation but refused its unchanged
full-budget-plus-6-GiB guard before weight loading or inference. Allocation-only
controls found that changing context from 8K to 32K adds only 260,046,848
bytes (248 MiB), while CUDA free memory was about 87.11 GB below Linux
MemAvailable. The diagnostic omits aligned-weight virtual reservations and
catalog entries; it is an allocation ledger, not a complete model run.

The inactive, authenticated 86,720,111,488-byte original GGUF remained fully
resident in file cache after the original engine's buffered SHA checks.
Immediately before a file-scoped `POSIX_FADV_DONTNEED`, `mincore` observes
all 21,171,903 pages resident; immediately afterward it observes zero.
The unchanged 32K allocation probe's CUDA free memory after setup rises
from 23,445,520,384 to 110,266,376,192 bytes, while Linux MemAvailable
remains approximately 120.32 GB. Its unchanged 6-GiB guard changes from
refusal to admission, with 9,380,782,080 bytes above the required headroom.
The immediate pre-advice residency sample follows the earlier allocation
probe, separating that probe's possible reclaim from the file advice.

Only this inactive pinned GGUF receives advice; node-wide caches and core
provider accounting are untouched. Before the successful run, all three
diagnostic source files and the executable are restored to their exact
qualified bytes. The native controller reads metadata and uses bounded
direct-I/O preparation, without hashing the inactive original GGUF again.
Failed attempt and diagnostic records remain preserved separately. No
diagnostic allocation or timing is counted as model performance.

## Qualification and provenance

The expanded benchmark accepts only the matched 8K or 32K geometry;
restoration factors and profile studies retain their 8K-only contracts.
The source slice passed 1,219 locked native tests, including 256 GPU tests,
eleven SDK format checks, seven eligible host/CPU clang-tidy arms,
boundaries and 1,120 REUSE/header checks. The CUDA unit is formatted and
qualified by actual NVCC flags and original-function correspondence;
`tools/check` excludes CUDA translation units from clang-tidy. The proof
retains all 115 original FFN functions and ten original prepared headers.

| Identity | SHA-256 |
| --- | --- |
| Community GGUF | `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` |
| Packed 32,768 IDs | `200f6656e5194ff906724978ee1971ef39a9ca0e58bd158e0897e7b6753cfe5a` |
| Original input receipt | `cc36e5586857d2cd6d46013ebec03b369316f2ccb61bd24f7ff108bd4f326109` |
| Original build receipt | `159000fe69f05d74ed4e3018304010a763f47cc924f189832ade1b3f19f572f6` |
| Original model receipt | `78d1e9f6f53557b76d7d59b14ce5a302e93b66ee458a52ebc195d2255c22ae42` |
| Native qualified source map, 1,240 files | `2fc1a5d285716ec14f9f4f9f18bcf97fd80675c3bf3dcd2ef67b3d2ffe30c5bd` |
| Native locked qualification receipt | `b0d0531200d0ba9323617304a3f3d6cc04cb44fe35e5583c4c4913ae5dbe3437` |
| Native executable | `2b698db46c162d010b7b11a08e4461ebff513f9bc44d649f932af6b26e66346f` |
| Native comparison controller | `e894869c5736429effeac1ac229452f3d5eb7d9ff7a8d6907901cc227fa29866` |
| Successful native comparison receipt | `30095e8f4d384b566271bb81a6b01542dc115e12e7e7511514e665aa60aa132f` |
| Successful native pipeline receipt | `8ac304169a74edb6d7a10e25b82017c9665d84915b87d44831e13f44997e6469` |
| Paired file-cache control receipt | `1ec3f9eaa416043dd3c479e0bba7bc0bb93d3a11bebda25de892e28fdb9b00b1` |

Original ds4 is Entrpi/ds4
`76d51ef82a81b70b78e51a3a6ea11946286de976`. Actual source, SDK,
compile database, build receipt, executable and resolved cuBLAS payload
identities are authenticated before and after the native run.
Measured on Spark B, 2026-10-01, native supervisor 06:33:22–06:40:35 EDT.
Raw original records remain at
`spark-b:~/scratch/m3-ds4-matched-32k-original/`; native qualification,
failed first attempt and successful run at
`spark-b:~/scratch/m3-ds4-matched-32k-native-r1/{check-r1,model-r1,model-r2}/`;
allocation/cache diagnostics and authenticated source/executable restoration
at `spark-b:~/scratch/m3-ds4-matched-32k-budget-r1/`.

The next restoration comparisons can use this control while preserving
precision, paid producers and context. This result supplies no evidence for
multiple CUDA streams, concurrent requests, decode or a production quality
exception. The literal reference remains temporary benchmark support.
