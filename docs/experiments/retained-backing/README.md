<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retained-backing swap trace — 2026-09-26

The cross-model swap trace for the
[retained-backing comparison](../../backend-proof.md#retained-backing-comparison)
(D-035, D-079). It is replay input, not a measurement. The trace stays
outside Git; this directory holds its generator, parameters, size table and
identity, and the [deterministic replay](replay.md) of part (a)'s 13
designs on its primary seed (2026-09-27): no slab or hybrid design meets
every deterministic criterion at every budget.

| File | Purpose |
| --- | --- |
| `swap_trace.py` | The seeded generator (`generate`) and the trace checker (`verify`); stdlib only |
| `params.json` | Library, episode template, route pools, budgets, shrink probe and both seeds |
| `library.json` | D-056's measured group and closure sizes per model, extracted from headers |
| `extract_library.py` | Produces `library.json` with the layout study's pinned planner |
| `replay.md` | The deterministic replay of the 13 designs: designs as replayed, results, eligibility |
| `replay_report.py` | Checks a replay run is whole and agreed, and applies the deterministic criteria |

Tests are in `tools/tests/test_swap_trace.py` and
`tools/tests/test_rb_replay_report.py`; the replay harness,
`benchmarks/retained_backing/`, is tested by `unit.Rb*` (`mise run check`).

## What the trace is

**Library.** Eight synthetic models. Each takes one measured plan's
groups: kind, layer and used and stored bytes. Stored bytes are the used
bytes rounded up to 4 KiB, as on disk.

| Model | Plan | Dense groups | Expert closures | Stored bytes |
| --- | --- | ---: | --- | ---: |
| `gemma-1`, `gemma-2` | Gemma 4 | 32 | 3,840 of 3,719,168 or 4,337,664 B | 16,939,601,920 each |
| `ornith-1`, `ornith-2` | Ornith 1.5 | 43 | 10,496 of 1,769,472 or 2,039,808 B | 21,702,594,560 each |
| `dsv4-1` | DeepSeek V4, layers 2, 3, 26, 42 | 6 | 1,024 of 8,060,928, 10,878,976 or 9,306,112 B | 10,575,859,712 |
| `qwen25-fp16` | Qwen2.5-0.5B FP16 | 25 | — | 988,221,440 |
| `exl3-4.0`, `exl3-4.5` | Qwen2.5-0.5B EXL3 | 26 | — | 589,004,800; 611,311,616 |

`extract_library.py` ran `layout.py` (`a0d1980a…`, the planner behind
`results.json`) over the pinned model files' headers on `spark`. It checks
every model's groups, chunks, used, disk and handle bytes against
`results.json` before writing, so these are D-056's plans.

**Switching.** The trace is 16 episodes. Each has the shape of the frozen
A→B→A trace: requests A, A, A, B, A, B at 0, 30, 60, 120, 180 and 240 s.
`generate` checks this template against the paging study's `sessions.json`
(`58d3253d…`), the session form of the reference trace (`7625929f…`).
- Episodes are 300 s apart.
- In each of two rounds, every model plays A once, in seeded order. B is a
  seeded draw among the others.
- A model with experts plays its route pool's captured requests in order,
  cycling. A dense model's request leases the whole model.

**Route pools.** These are the traced, retained, full-window runs of the
[paging study](../paging-feasibility/full-study.md), verified against the
event hashes in its `full-evidence.json`:

| Pool | Runs (external) | Requests |
| --- | --- | ---: |
| Gemma | `aba-capture-1/A-b512-r1-s1-t1` (the A→B→A trace), then `small-capture-2/A-b512-r1-s1-t1` | 4 + 4 |
| Ornith | `aba-capture-1/B-b512-r1-s1-t1`, then `small-capture-2/B-b512-r1-s1-t1` | 2 + 4 |
| DeepSeek | `large-capture-2/D-b512-r1-s1-t1` | 3 |

**Access model.** It follows the study's routed demand policy:
- A request leases all its model's dense groups for its duration.
- Each captured prefill or decode step becomes, per layer, one `use` of the
  union of the experts its tokens selected. A use is leased, executed and
  released.
- Before every B request the pool must return 1 GiB to the OS (`shrink`),
  and it gets the 1 GiB back (`grow`) before the next A request.

**Reference.** Per budget, the trace carries a fragmentation-free reference:
- LRU over whole unleased groups at their stored bytes;
- recency is a group's latest lease, use or release, and within one event
  the lower id is older;
- an `evict` record (the victims) and a `restore` record (exactly the
  missing groups) precede the access or shrink that needs them.

Every design performs the reference's evictions first, so the reference's
restores are a lower bound for the designs, and anything beyond them is a
design's own cost. LRU is not an optimal policy in general. The replay
recomputes the reference and requires the records to be its choices.

**Budgets.** Each budget is the largest multiple of 1 GiB not above the
trace's unique bytes divided by 5/4, 3/2 or 2. Unique bytes are the stored
bytes of every group the trace touches. The primary trace touches 28,836
of 29,929 groups (86,584,598,528 bytes), so its budgets are 64, 53 and
40 GiB. Its reference, for comparison with the designs later:

| Budget | Restores | Restored GiB | Evictions |
| ---: | ---: | ---: | ---: |
| 64 GiB | 11,594 | 126.68 | 5,713 |
| 53 GiB | 19,642 | 197.87 | 12,354 |
| 40 GiB | 23,593 | 239.96 | 11,741 |

Each file also holds 96 requests, 783,282 uses and 32 shrink probes.

## What substitutes for what

- **Captured routes stand in for the optimized engine's.** They are the
  only routed captures, recorded with CUDA fusion and graphs disabled
  (RE-006). The optimized plan selects different sets in 40–43% of
  token-layer rows ([fused routes](../fused-routes/README.md)). The
  locality is that configuration's.
- **Two instances per Gemma and Ornith plan stand in for distinct library
  models of those shapes.** They share a route pool, but their groups are
  distinct.
- **DeepSeek keeps four layers, which hold all three closure sizes.** That
  way the 5/4 budget fits on one Spark.
- **Qwen3.8 is left out.** Its captured trajectory failed equivalence. Its
  28.8 GB row table is read by rows; as one whole group it would dominate
  every budget. Its 1.97 MB closures fall inside Ornith's range.
- **Tables and heads are whole groups.** Row-granular table paging is not
  modeled.
- **The shrink probe stands in for non-weight demand**, such as state and
  workspace, which stay outside the pool. Ornith's unused MTP layer is in
  its dense lease, as in the paging study.
- **Arrival seconds are the template's.** Replay time is counted in access
  events.

## Identity

- Generator `swap_trace.py`: `e6ea1214762d25885390d9e9204a808626e0622304408ead6cf07a0c4069c1ee`.
- `params.json`: `42c8a044fa31baa68abbeedb1451ea543d373f56462a9352f6ee7501b2fee4b8`.
- `library.json`: `a64b453ae1d9b4f66fc6aa1440b28cf40775be5c02489706523da5d5577429bd`.
- `extract_library.py`: `fed45acfc700bb20de18b91756a228f7ad3dc2db6dd8b711e6c0743481b62bca`.
- Primary seed 20260926:
  - `manifest.json` `44f9f2b40bb9ed8bab7ac337c5736de08dfdf2e1821ae2755c7a78bb191b7fc9`;
  - `trace-r5-4.jsonl` `01611f43c464dd42b954224b00f21f8aeaef0476e75b9021e9d827a4991c692d`;
  - `trace-r3-2.jsonl` `040ea829349324c8dfd7e9b5320fb4bde3ddf2796863edbf38d037bd5f70ff24`;
  - `trace-r2-1.jsonl` `95255d38fcf005ba0e6b53b0de8e14a4f767bca8914d7c512e6b6c2d378c37dc`.
- Confirmation seed 926202601, generated and checked by hash only; its
  contents were not inspected:
  - `manifest.json` `8aee64a56095a1ef0f3806197e173ae405c1173f007ae417b6bcbec6aaec3f6b`;
  - `trace-r5-4.jsonl` `36ce0dfeb71a643f791cd2f2e96a275fd89fcfa66436b16ce28c49bb0855e9aa`;
  - `trace-r3-2.jsonl` `6ea108f714cff887a6c8b81c875fd422a31c3d86f6bbfb9f34027dc5785bd543`;
  - `trace-r2-1.jsonl` `f75a69cec906f5071678dbb70b3f81547855c61ff006a2a0c8c2dc3e5cd4c3d6`.

The seeds were fixed before anything was generated. The same bytes came
out on `spark` (Python 3.12.3) and on the workstation (the mise-pinned
3.14.7).

## Retrieval and regeneration

The traces are in `~/.local/share/llmp/retained-backing-20260926/`
(`trace-primary/`, `trace-confirmation/`), on `spark` and on the
workstation. The inputs they were generated from are in its `captures/`:
- the five capture files and three receipts listed above;
- `aba-input/sessions.json`.

The originals are in `~/.local/share/llmp/paging/` on `spark`, and on
`spark-b` for `large-capture-2`.

To check a copy, run `python3 swap_trace.py verify DIR`. It checks the
files against `manifest.json`'s hashes, the schema, and a full replay of
each file's reference: it restores exactly the missing groups, evicts no
victim beyond the need and stays within its budget. LRU order is checked
by the unit tests, not by `verify`. The copy is the trace only if
`manifest.json`'s own SHA-256 is the one above. To regenerate, from the repository root, into a directory that
does not exist yet:

```sh
R=docs/experiments/retained-backing
python3 $R/swap_trace.py generate $R/params.json $R/library.json \
  docs/experiments/paging-feasibility/full-evidence.json CAPTURES \
  CAPTURES/aba-input/sessions.json primary NEW_DIRECTORY
```

`CAPTURES` holds `aba-capture-1/`, `small-capture-2/` and
`large-capture-2/`, each with its `receipt.json` and the pool's `.jsonl`
files. `library.json` is regenerated with
`python3 extract_library.py OUT PROFILE=SOURCE[,SOURCE…]…`, over the model
files that [artifact-layout](../artifact-layout/README.md) names.

## Format

The trace is JSON Lines, version 1:
- a `header`;
- one `model` record per model, with its group id range;
- one `group` record per group (`id`, `model`, `kind`, `layer`, `expert`,
  `used`, `stored`);
- the events in order:
  - `request` is informational;
  - `lease` and `release` bracket a request's dense groups;
  - `use` is an access that is released at once;
  - `shrink` and `grow` carry `bytes`;
  - `evict` and `restore` are the reference's, placed before the access or
    shrink they serve.

The header's `access_sha256` covers the budget-independent events, so the
three files of one seed share it. Model and group records carry no file
offsets: a replayer places groups in id order at 4 KiB alignment.
