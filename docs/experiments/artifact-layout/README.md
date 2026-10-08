<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Artifact layout study (open question 5)

M0 evidence for D-056 and the [v0 format](../../artifact-format.md). The
harness is a stdlib-only Python reference model of the format:

- it plans layouts from real GGUF/safetensors headers;
- it writes artifacts, verifies them as untrusted input, and publishes them
  atomically;
- it pages artifacts back in with direct reads and compares the bytes
  against the source.

It is the accept/reject oracle for the M5 C++ importer and verifier, not
llmpalooza's runtime parser. `results.json` holds the aggregates. The built
artifacts and raw outputs stay outside Git in
`~/.local/share/llmp/artifact-layout-20260922/` on `spark`.

| File | Purpose |
| --- | --- |
| `layout.py` | Planner, writer, verifier, closure/coalescing model, `loadcheck`, `coldload` |
| `import_m3.py` | M3's importer: `layout.py`, unchanged and checked against its pinned SHA-256, run on GGUF sources checked against the M3 pins, with its own converter identity ([dsv4-native](../dsv4-native/README.md)); for safetensors sources, `modelopt_qwen38.py` |
| `modelopt_qwen38.py` | Qwen3.8 Flash Next's ModelOpt checkpoint (NVFP4 experts, MXFP8 attention and shared expert) planned and written with `layout.py`'s container, index and verifier, its bytes repacked on the way: experts into the CUTLASS layout the grouped GEMM reads (GGML NVFP4 blocks in the first import), the n-gram table's shards into 90-byte rows, linear attention's value heads into tiled order ([qwen38-native](../qwen38-native/README.md)); the converter version carries its SHA-256 |
| `test_modelopt_qwen38.py` | 23 unit tests on a synthetic 4-layer checkpoint: the repacks against element-by-element definitions and both dequantizations, the CUTLASS layout against the harness's former load-time conversion of GGML blocks, a verified build, worker processes, refusals of malformed headers, scale shapes, NaN or negative scales, bad global scales, missing, extra or duplicate tensors and keys, and a config the checkpoint contradicts (`python3 -m unittest test_modelopt_qwen38`) |
| `test_layout.py` | 104 unit tests with synthetic GGUF/safetensors inputs (`python3 -m unittest test_layout`) |
| `conform.py` | Upstream-reader conformance, run in the pinned `llmp-exl3-reference:20260922` image |
| `io_align.py` | A/B direct-read throughput for 2 MiB- versus 4 KiB-aligned file offsets |
| `va_probe.cc` | CUDA VMM virtual-address reservation probe |

## Conditions

All runs were on `spark` on 2026-09-22: GB10, driver 580.178.04, kernel
7.0.0-1019-nvidia, Python 3.12.3. Storage was ext4 on the internal NVMe
(`nvme0n1p2`, 512-byte logical blocks, `max_sectors_kb` 128, `IOV_MAX`
1,024). The unit tests also passed on the workstation (btrfs).

Inputs are the existing pinned files. The builds recomputed each built
source's SHA-256, and all match their existing pins:

- Qwen2.5-0.5B-Instruct FP16 GGUF `8e0ae260…` ([first slice](../first-slice/README.md))
- EXL3 4.0 bpw `66b61bd1…` and 4.5 bpw `94c45f98…` ([EXL3 reference](../exl3-reference/README.md))
- Gemma 4 26B-A4B UD-Q4_K_M `f2c28b3d…` ([reference setup](../reference-setup/README.md))

Ornith, Qwen3.8 and DeepSeek V4 were **planned from headers only** (bounded
reads, no payload read). They are the files pinned in the
[paging study](../paging-feasibility/full-pins.json) and were not re-hashed
here. The model licences in those reports apply; no weights are
redistributed.

The prototype was hardened over ten adversarial challenge rounds (the tenth
came back clean) and two reviews. All figures below come from the final code, whose hash is in
`results.json`. Every artifact was rebuilt from scratch and every check
was re-run.

## Commands

```text
python3 layout.py plan [--tie-check] SOURCE...        # layout statistics
python3 layout.py build OUT SOURCE... [--meta=FILE]   # verify, then atomic publish under OUT
python3 layout.py verify OUT/<id>
python3 layout.py loadcheck OUT/<id> SOURCE...        # readable-range closure, vectored O_DIRECT, vs source
python3 layout.py coldload OUT/<id> [MAX_RUN_MIB]
sudo docker run --rm --network none -e PYTHONPATH=/app/gguf-py -v ...:ro \
  --entrypoint python3 llmp-exl3-reference:20260922 conform.py /art/<id> [SOURCE_GGUF]
python3 io_align.py FILE THREADS READS SEED
g++ -std=c++23 -O2 -Wall -Wextra -Werror -march=armv8-a -I/usr/local/cuda/include \
  va_probe.cc -L/usr/local/cuda/lib64/stubs -lcuda -o va_probe
```

The EXL3 builds passed `config.json`, `generation_config.json`,
`quantization_config.json`, `tokenizer.json`, `tokenizer_config.json`,
`vocab.json` and `merges.txt` as `--meta`. The VA probe used Spark's system
g++ 13.3.0 against the host CUDA 13.0 headers and the driver library. It
is a platform probe, not a D-032 toolchain build.

## Results

**Layouts.** Per-model groups, chunks, padding, expert closures, VA and row
tables are in the format document's
[worked examples](../../artifact-format.md#worked-examples-measured-plans-of-real-files).
With 4 KiB group alignment, disk padding is at most 0.083% (Qwen3.8).
Giving every chunk its own 2 MiB handle would cost 3.49–10.87% in memory;
that is also what 2 MiB-aligned groups on disk would have cost. The pinned
GGML over-read adds 1,541,352 reserved zero bytes to Gemma 4 and 5,347,986
to Qwen3.8. The other five models need none.

**Built artifacts.** Each passed a full `verify`, both inside `build`
before publication and afterwards. Each passed a byte-exact `loadcheck`,
which also confirms that the reserved over-read bytes read back as zero.
The upstream safetensors 0.8.0 reader opened every shard and read every
entry, pads included. `conform.py` checked each entry against the byte
range the shard header names; `verify` separately proves that the header
agrees with the index. For the two GGUF-sourced artifacts, upstream
`gguf-py` read the kept metadata as a zero-tensor GGUF with every other
field byte-identical to the source. The EXL3 artifacts carry their source
JSON/tokenizer files verbatim instead.

| Artifact | Bytes on disk | Build incl. verify / verify | Loadcheck resources, chunks, requests | Upstream entries read |
| --- | ---: | --- | --- | ---: |
| Qwen2.5 FP16 `b93cdc32…` | 994,276,024 | 5.93 s / 1.07 s | 290, 755, 294 | 315 (+29 GGUF fields) |
| EXL3 4.0 bpw `6e96e499…` | 601,043,985 | 2.78 s / 0.56 s | 798, 1,064, 804 | 992 |
| EXL3 4.5 bpw `00d77caf…` | 623,351,494 | 2.82 s / 0.58 s | 798, 1,074, 804 | 992 |
| Gemma 4 `d673bf28…` | 16,958,152,889 | 67.79 s / 14.59 s | 8,278, 13,465, 8,289 | 16,194 (+63 GGUF fields) |

Build time includes hashing every source twice (an identity pass that also
digests every range to be copied, and a change check after writing) and a
full verify before publication. The prototype does this hashing in Python. A rebuild
of Qwen2.5 reproduced its ID, and later rounds reproduced the Qwen2.5 and
Gemma IDs. IDs changed during hardening only through the manifest: once when
its list order became canonical, and twice for EXL3 as `config.json` and then
every kept metadata file joined the source identity (the EXL3 manifests now
list all eight input files). The small artifacts' shard and index bytes did not
change (same sizes; their indexes contain no expert arrays to reorder). The EXL3 artifacts
carry the `exl3` and `plain` representation families, per-tensor `k_bits`,
the `mcg` codebook and 4-byte marker resources. Loadcheck reads each
resource's readable-range chunk closure as vectored O_DIRECT runs (at most
64 MiB and `IOV_MAX` iovecs), one iovec per chunk into its own 2 MiB
anonymous buffer. A resource spanning more than one run explains requests
exceeding resources.

**Offset alignment A/B** (`io_align.py`). Random reads from the 49.6 GB
Qwen3.8 shard 2, O_DIRECT, 32 threads × 3,000 reads, 6 repetitions, arm
order alternated per repetition:

| Arm | Median GB/s | Range |
| --- | ---: | --- |
| 2 MiB offset, 2 MiB length | 9.348 | 9.151–9.444 |
| 4 KiB offset, 2 MiB length | 9.261 | 9.225–9.360 |
| 2 MiB offset, 1,974,272 B (Qwen3.8 closure) | 9.330 | 9.277–9.361 |
| 4 KiB offset, 1,974,272 B | 9.085 | 8.972–9.244 |

An 8-thread pilot (7 GB/s) showed no separable difference. At 32 threads
the 2 MiB-length arms overlap: a 0.9% median gap, within noise. At closure
length the medians differ by 2.6% and the ranges do not overlap, consistent
with one extra 128 KiB device command when an offset is not 128 KiB-aligned.
A 2 MiB layout reads its padding, though. Useful bytes per second for the
Qwen3.8 closure come to about 8.79 GB/s with the 2 MiB profile
(9.348 × 1,971,200/2,097,152) and 9.07 GB/s with the 4 KiB profile
(9.085 × 1,971,200/1,974,272).

**Cold whole-artifact load** (`coldload`, Gemma, 16,939,601,920 data
bytes, 9,059 chunks). Every chunk missing, 8 runs in flight, each run
scattered into per-chunk 2 MiB buffers, 3 repetitions:

| Max run | Requests | GB/s |
| --- | ---: | --- |
| 2 MiB | 9,058 | 7.545, 7.676, 7.438 |
| 16 MiB | 1,119 | 9.939, 10.066, 9.896 |
| 64 MiB | 257 | 10.511, 10.509, 10.517 |

In this Python thread pool, fewer and longer requests were faster, which
probably reflects per-call overhead. D-034's native io_uring already reached
about 15 GB/s with four 2 MiB reads in flight
([I/O path](../io-path/README.md)). These figures compare arms only; they
neither measure llmpalooza load rates nor the native benefit of coalescing.

**Hashing cost** (`openssl speed -evp sha256 -bytes 2097152 -seconds 3`,
OpenSSL 3.0.13, pinned with `taskset`). SHA-256 ran at 2.487 GB/s on a
Cortex-X925 core and 2.117 GB/s on a Cortex-A725 core; BLAKE2b-512
(unpinned) ran at 1.420 GB/s. Verifying every page-in at ~15 GB/s would
take about six cores, so the format hashes only at install and verify time.

**VA reservation** (`va_probe`, 3 runs, host-NUMA pinned properties as in
D-034). The minimum granularity is 2 MiB. The largest single
`cuMemAddressReserve` was 128 TiB (2^47 bytes). 254–255 separate 1 TiB
reservations succeeded in one process. A 2 MiB handle mapped at the last
granule of a 4 TiB reservation was written and read back correctly. So
uniform-stride GGML expert views (93.5 GiB to 37.0 TiB per model) are
possible, while a per-expert pointer table needs only the backing's own VA.

**Index parse.** Qwen3.8's index, with placeholder chunk hashes, is
6,330,330 bytes. The prototype's strict Python parser loaded it in
0.048–0.050 s (3 runs).

## Limits

- The Python writer, verifier and page-in model are reference code. There
  is no C++ implementation, no GPU execution, no GGML/EXL3 view
  construction and no VMM mapping of these artifacts; M2/M3 own that.
- The GGML expert-stride and over-read findings come from reading the pinned
  source, not from executing kernels on these layouts. EXL3 over-read is
  not established.
- I/O figures are single-host, single-device Python measurements on a warm
  system. Only the 49.6 GB shard and the 16 GB Gemma artifact were read.
  There was no concurrent compute, no memory pressure and no native AIO.
- Three MoE models were planned from headers only. Builds, verification
  and load checks covered the three small fixtures and Gemma 4.
