<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSpark block GPU-mask screen — 2026-10-08

The compatible transfer is implemented and checked, but **not selected**:
DSpark keeps host block masks by default. A short two-owner off/on/on/off
screen gives neutral joined-call time and slightly slower whole diagnostic
time. Removing 58,368 constructed host-mask bytes does not reduce the measured
activation, scratch or host-input envelopes. No further mask timing ladder or
reference run is needed for this rejection. DeepSeek's target GPU masks remain
selected; optional compressed visible-count masks remain open.

## Mechanism and bounds

The shared mask producer adds an explicit authenticated `MaskPolicy::kBlock`
(op parameter 6=1; causal remains 0 and parameter 7 remains reserved 0).
For each actual draft position segment, `end=first_position+rows`; a ring
cell holds its latest absolute key below `end`. A query sees a key later
within its current block, or a previous key less than the window behind it.
This preserves DSpark's noncausal current-block visibility. Negative,
noncontiguous or overflowing device positions and padded cells/rows yield
negative infinity.

The primitive supports F16/F32 and exact/padded rows. DSpark selects F16,
exact rows, ring 256/window 128, with the approved one-to-five-row block capacity.
`end<=INT32_MAX` is checked independently of target context: a drafter can
propose past the target endpoint. Source adapters authenticate all eight
parameters, actual position parents and segment offsets, unique producer
membership and omitted host-input membership before embedding/joined
allocation. Skipping the matrix retains the same checked tokens, contiguous
positions and modulo ring cells. Producer activations and graph/source
metadata remain funded. Attention, draft head and capture policy are unchanged.

`Dsv4Options::device_draft_masks=false` preserves the host reference. This is
independent of the selected target raw GPU masks. `bound_draft_masks()` counts
bound planned nodes, and `draft_mask_host_bytes()` counts constructed matrices,
including construction before a later failure; neither counts executed kernels
or actual H2D bytes.

## Fixed factor and result

Spark B (`spark-56f5`, GB10), approved SDK `aarch64-c09daba6ac31edee`, pinned
CUDA13.4/cuBLAS13.8. Four fresh processes use the same binary, C2, draft 3,
verify capacity 4, context 1024, max_rows 512, graphs and joined drafts enabled.
Two distinct synthetic 254-ID prompts make the first three-row block cross
ring 256. No excluded model warm-up is added; each arm executes the existing
solo route followed by joined waves with the same capture/replay schedule.

| Arm | Solo loop s | Whole wave diagnostic s | Mean joined C2 call ms |
| --- | ---: | ---: | ---: |
| Host1 | 1.631370265 | 1.314535243 | 140.059 |
| Device1 | 1.634451982 | 1.316856598 | 140.048 |
| Device2 | 1.637114581 | 1.316458892 | 139.541 |
| Host2 | 1.639289726 | 1.312895394 | 139.831 |

At n=2 per policy, solo mean changes +0.0277% (host bookends +0.4854%),
whole-wave mean +0.2240% (host bookends −0.1247%), and mean joined-call
interval −0.1075% (host bookends −0.1628%, overlapping ranges). These do not
support a useful speed gain. There are only four joined calls per arm;
the rounded per-call averages are diagnostic evidence, not a confidence interval.

Solo and joined generation-loop timers exclude final rollback/state
observation after the loop. The whole joined loop includes its existing
first-owner discard/rerun and departing-peer snapshot; the final protected-peer
comparison is off timer. The width-specific call
interval includes completed `DraftVerifyWave` input construction, staging,
execution, wait and full-head publication, and excludes subsequent checks and
payload retention. Setup/load, prefill, finite scans, hashing/serialization and
settled state reads are outside these intervals. These are runner diagnostic
endpoints, not complete settled generation or HTTP latency.

Each arm completes 62 solo and 46 joined generated-step tokens, with four
C2 calls. Actual capture/replay tuples `[eager,captured,replayed]` are
`[3,3,28]` for scalar drafts, `[8,4,25]` for target graphs and `[1,1,2]`
for wave graphs. These tuples and all work agree across policies. The first
verify discard has zero stale bytes, reruns exactly, and a departing peer's
state remains unchanged.

All 16 retained payloads are complete finite F32 vocabulary rows (129,280
values per head). Per arm, scalar owners retain 46/33 heads and joined owners
46/17; the second joined owner departs early. Every policy repeats identical
full head bytes, proposals, kept-row traces, committed histories and settled
target/drafter initialized-state hashes. The three distinct head hashes are:

| Form / owner | Heads | SHA-256 |
| --- | ---: | --- |
| Solo0 / wave0 | 46 | `8cd2f08903551961fa9972f3a3bcd0812d3dc0730e99921611c5f2f207840acf` |
| Solo1 | 33 | `665420d11fbc2c59e11ba99f668739f6bd4b066e186f7cecb82a12e932776017` |
| Wave1 | 17 | `04a39d7ac290efde176eae0fd2d5a45e4f661c72aa57edee786b87233297c283` |

The device policy binds four mask plan nodes and constructs zero host-mask
bytes, versus zero nodes/58,368 bytes on host. Both policies charge
478,150,656 activation bytes, 48,234,496 scratch bytes, 10,485,760 host-input
bytes and 576,716,800 diagnostic-observation bytes. Device plan floor is
5,999,152 versus 5,998,128 bytes (+1,024); there is no overall envelope saving.
All four processes return0 after explicit teardown. Installed supervisor
wait succeeds, no new kernel/driver error messages appear, boot identity
stays unchanged, and every post-arm GPU-process check is empty.

## Focused controls and provenance

Eight unique Spark B controls pass, with zero skips/errors/disabled cases:

- Three DSpark input/graph controls: independent CPU block visibility,
  skipped matrix with unchanged tokens/positions/cells, and actual scalar/
  joined producer membership and position offsets.
- Two source controls: complete successful tiny-F16 joined embedding assembly
  and atomic malformed descriptor/position/producer refusals before allocation.
- Three GPU controls: block F16/F32 × exact/padded rows 1/3/5 with independent
  absolute-key oracle, wrapped/near-INT32 positions, invalid positions,
  poisoned changed-position replay and protected guards; unchanged shared
  causal and DeepSeek target raw masks.

The first model attempt stopped before paid work: its diagnostic compared
an unrounded byte bound with the catalog's whole-2MiB charge. Funding had
succeeded. The benchmark-only correction rounds `WaveObservationBytes` once
for startup, charging and proof. It rebuilds only the benchmark; the three
control ELFs and all eight positive XML records remain byte-identical.
The failed arm is preserved separately and supplies no timing evidence.
Full regression and broader lifetime/reference qualification are deferred:
this first screen rejects ordinary adoption. Normal source review and an
independent adversarial source/method review are clean.

Target artifact `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`
(`unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93`, UD-Q2_K_XL) and Q8_0
DSpark artifact `dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5`
are the existing approved prepared pair. Manifest SHAs equal these IDs;
index SHAs are respectively
`68ddac15cff7cfcac5162beab7559884632244651797aed1b000cc7325ea24fc`
and `437561ad6dd1e8d5fd3758be2a8eac52b8a1c81f6a1fb54b27e0b3badfeaa11c`.
No raw import or checkpoint copy occurs.

The measured source base is `f8edffc526902f0d246d8f80ca682c50c2a12c59`, with
source manifest `d31f6ad8d3415a14946dd0f5fd688cc3dcc49beccea00b828876b3336e696e19`.
Its benchmark ELF is
`11ac77452dd54bb04651ad4cd13da2a461eea6c6ba3694854d1116addcce3b0d`;
SDK receipt `874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89`.
Actual first-resolved private `libcublas.so.13` and `libcublasLt.so.13` hash to
`ee7c1657a03695c0de790aa79e34cef9c9649756b1846b11dd44caca20ba656b`
and `ba3b942f4ea43433b65e8c492a7b73de887534dc20146506ddaa4a78c79c5d30`.
The successful model packet hashes to
`585ed9f54b0cae1bdd3f1df7f2dc7cd96901445c4c8face2b0d254a225554d74`,
and aggregate runs JSON to
`f99d8e0b4262185f3ce65ed8627166ec3715b31eaadc5a7ae28ecc806940b0ad`.
Jobs `dspark-block-masks-build1`, `dspark-block-masks-build2` and
`dspark-block-masks-screen3` complete 0 and are waited on;
`dspark-block-masks-screen2` remains a failed setup attempt. Raw records are outside
Git at Spark B `~/scratch/m35-dspark-device-masks/` and workstation
`/tmp/jitllm-m35-coordination/dspark-device-mask-raw/`; milestone cleanup can
remove them without losing replay inputs or the aggregate result.

At task entry, checked 2026-10-08T10:27:53Z, TensorFold native main is
`f8fe17d24629aedabf90bbf78279dd776e6d62e7` and Python is
`ed78d6fc204d89d90b045bf033d6551e7714f3a1`. Its pinned documented DeepSeek+
DSpark recipe is MLX/256GB, not this CUDA/GGUF/Q8_0 pair. llama.cpp remains
the applicable same-format comparator, but this rejected same-native factor
makes no new competitive or corpus/PPL claim.

## Replay after raw-output cleanup

The [benchmark](../../../benchmarks/spec_runner.cc) and [payload checker](compare.py)
are durable. Build `jitllm_spec_runner` with the approved Spark SDK and its
private cuBLAS closure. Supply the approved prepared artifacts above from the
standing `~/.local/share/jitllm/m3-artifacts/` store (or recreate them with
those pinned imports); authenticate manifests/indexes before measurement.
Recreate both inputs from the checked-in frozen reference, without new
tokenization or external raw samples:

```python
import hashlib, json, struct
from pathlib import Path
source = Path('docs/experiments/fast-swap/reference-deepseek-v4-flash-0731-llamacpp.json')
assert hashlib.sha256(source.read_bytes()).hexdigest() == '79f946a00f2da83b20c09a4fd814e24de15a0f83384f0666622183436553944e'
reference = {p['id']: p['prompt_token_ids'] for p in json.loads(source.read_text())['prompts']}
expected = ['2378983965087b0a937d847f3d88ebda424c826c8dab966629c59286b8f6a22e',
            '3dd26349493b43d3a3e972379c1835b286ce5fb5e6e827fa8539b393bfc481f2']
rows = []
for name, digest in zip(('capital', 'fibonacci'), expected):
    original = reference[name]
    ids = [original[i % len(original)] for i in range(254)]
    assert hashlib.sha256(struct.pack('<254i', *ids)).hexdigest() == digest
    rows.append({'id': name + '-ring254', 'token_ids': ids})
with Path('prompts.json').open('x') as output:
    json.dump({'decode': rows, 'chat': []}, output, indent=2)
```

Under the installed GPU-exclusive supervisor, run four fresh processes in
host/device/device/host order, with fresh output directories, using:

```sh
build/spark-native/benchmarks/jitllm_spec_runner \
  --dsv4-artifact "$target_artifact" --drafter "$dspark_artifact" \
  --prompts "$prompts" --out "$fresh_output" --check wave --slots 2 \
  --wave-mode verify --tokens 32 --context 1024 --max-rows 512 --draft 3 \
  --graphs on --exact off --wave-lanes on --joined-drafts on \
  --device-draft-masks off --wave-payloads on
```

Change only `--device-draft-masks` in device arms. Bound each application to
120s and the fixed batch to 600s, stop first failure, require return0/`DONE`
after teardown, and wait on the installed supervisor. Guard no foreign GPU
process and MemAvailable>=90GiB before/after every arm; retain kernel cursor,
new messages and boot identity and refuse new NVRM/Xid/UVM/OOM errors.
Bind current source, binary, SDK receipt, inputs and actual cuBLAS identities
before/after; a new build has its own identities rather than pretending to
be the historical binary. Off GPU, under a shared host lock, run
`python3 docs/experiments/dspark-device-masks/compare.py H1 D1 D2 H2`.
This verifies complete finite heads and exact histories/state/work, selected
mask policy, capture/replay and repeated static budgets, then reports the
bounded timing factor. Process retirement and host/kernel guards are separate
requirements, not inferred from the payload checker.

## Cross-family disposition

Only DSpark's noncausal block consumer uses the new mode. The source-shared
causal path remains selected by DeepSeek target, native/GGUF Qwen and Gemma2/3/
26/31, with no new per-family speed claim; causal preservation is checked.
Qwen-Image T/D/V has no staged block-mask consumer. This closes the DSpark
block sub-investigation as measured not-adopted. Composite T22/T69 remain
OPEN for optional compressed visible-count producer/source contracts. The
separate DSpark draft-head quantization T95 is unchanged.
