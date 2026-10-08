<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Which OS counters include VMM backing on the Spark — 2026-09-25

The [memory breakdown](../../architecture.md#memory-breakdown) puts OS
counters beside the catalog's occupancy. M2 owed a measurement of which of
those counters include CUDA VMM backing on the Spark's driver, so the two
can reconcile. This experiment creates backing through llmpalooza's CUDA
provider and records how far each counter moves at each step.

The harness is
[`benchmarks/vmm_counters.cc`](../../../benchmarks/vmm_counters.cc).

## Method

For each backing kind, first device-local and then host-NUMA (D-034), the
harness takes 8 GiB of backing in 4096 extents of 2 MiB through
`VmmProvider`, one step at a time:

1. **reserve:** reserve the address range;
2. **create:** create the backing;
3. **map+access:** map every extent and grant read-write access;
4. **touch:** write every byte. Host backing is written from the CPU;
   device backing gets device copies from a written host extent, completed
   by fence;
5. **unmap:** unmap everything, keeping the backing;
6. **release:** release the backing and free the range.

Each step waits 250 ms, then reads:

- every `/proc/meminfo` field that moved by at least 64 MiB at some step;
- the process's `VmRSS`, `RssAnon`, `RssFile`, `RssShmem`, `VmLck` and
  `VmPin`;
- its cgroup's `memory.current` and `memory.stat` (`anon`, `file`, `shmem`,
  `kernel`);
- the driver's free memory (`cuMemGetInfo`).

The tables give changes from before the reservation, in MiB, as the range
across three consecutive processes.

- **Host:** `spark-c4e2`, GB10, 121.7 GiB `MemTotal`, DGX OS 7.6.0,
  kernel 7.0.0-1019-nvidia, driver 580.178.04, `vm.overcommit_memory=0`.
  The process ran in a user session scope (cgroup v2). The load average
  was 0.44–0.87 (the storage benchmark ran just before), and no settings
  were changed.
- **Build:** the `cross` preset with the pinned SDK
  `x86_64-f469d317c88c3044`.
- **Sources (SHA-256):**
  - `benchmarks/vmm_counters.cc`:
    `7682cb4da1bdb710b8eb2d18d23bf65d1a1212f353fa95113e3ca219e47fd2fe`;
  - `src/providers/cuda/cuda_device_memory.cc`:
    `8c09b9526c2f9c7e3411b47b64afc8ee5794fd19eacfb5542e02cbbe802196c9`;
  - `src/providers/device_memory.cc`:
    `78c2a441694a570706190aafc39abd6f1b64f7d126c5d8d7b35e5ac36a121fdd`.

These are the sources measured. Later review fixes to `device_memory.cc`
cover the error reported by a failed undo after a table-full insert and
quarantine accounting after uncertain provider outcomes. The successful
measurement path does not exercise those failures.

## Results

**Device-local backing, 8 GiB (8192 MiB):**

| Counter | reserve | create | map+access | touch | unmap | release |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MemAvailable`, `MemFree` | 0 | −8346 to −8358 | −8441 to −8448 | −8441 to −8449 | −8351 to −8360 | −8 to −17 |
| `cuMemGetInfo` free | 0 | −8346 to −8358 | −8441 to −8448 | −8441 to −8449 | −8351 to −8360 | −8 to −17 |
| `Slab` (all `SUnreclaim`) | 0 | +135 | +206 to +207 | +207 | +135 to +136 | 0 to +1 |
| Process `VmRSS` | 0 | +4 | +22 | +22 | +22 | +22 |
| Process `RssFile` | 0 | 0 | 0 | 0 | 0 | 0 |
| cgroup `memory.current` | 0 | +4 to +5 | +38 | +38 | +22 | +22 |
| cgroup `kernel` | 0 | 0 | +16 | +16 | 0 | 0 |

**Host-NUMA backing, 8 GiB (8192 MiB):**

| Counter | reserve | create | map+access | touch | unmap | release |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MemAvailable`, `MemFree` | 0 to +1 | −8347 to −8350 | −8462 to −8464 | −8462 to −8464 | −8462 to −8464 | −5 to 0 |
| `cuMemGetInfo` free | 0 to +1 | −8347 to −8350 | −8462 to −8464 | −8462 to −8464 | −8462 to −8464 | −5 to 0 |
| `Mapped` | 0 | 0 | +8192 | +8192 | 0 | 0 |
| `Slab` (all `SUnreclaim`) | 0 | +134 | +218 to +219 | +218 to +219 | +135 | 0 |
| Process `VmRSS` | 0 | 0 | +8198 | +8198 | +6 | +6 |
| Process `RssFile` | 0 | 0 | +8192 | +8192 | 0 | 0 |
| cgroup `memory.current` | 0 | 0 | +39 to +40 | +39 to +40 | +6 | +5 to +6 |
| cgroup `kernel` | 0 | 0 | +33 | +33 | 0 | 0 |

`RssAnon`, `RssShmem`, `VmLck`, `VmPin` and the cgroup's `anon`, `file`
and `shmem` moved by less than the process's own small allocations
(≤ 22 MiB) at every step. No other `/proc/meminfo` field moved by 64 MiB.

## Conclusions

1. **Backing comes out of system memory when it is created, whatever its
   kind.**
   - Device-local and host-NUMA backing alike take their full size from
     `MemFree` and `MemAvailable` at `cuMemCreate`, before any mapping or
     touch. Touching adds nothing.
   - The driver's free memory equals `MemAvailable` throughout. On the GB10
     it is the same pool, not a separate device count (D-004).
   - Releasing the backing returns it all.
   - So the node's physical line can use `MemAvailable`; the driver's
     figure adds nothing.
2. **Per-process and cgroup accounting miss VMM backing.**
   - Device-local backing appears in no per-process counter.
   - Host backing appears in `VmRSS` (as `RssFile`) and in meminfo `Mapped`
     only while mapped with access, and drops out on unmap although it is
     still held.
   - Neither kind is charged to the cgroup: `memory.current` moved by at
     most 40 MiB for 8 GiB. A cgroup memory limit does not bound the
     runtime's VMM backing, which the runtime's own budget `B` must.
   - The "measured footprint" line cannot come from RSS or the cgroup.
     Reconcile against the catalog with the system-wide counters instead.
3. **The driver's per-allocation bookkeeping is kernel memory outside any
   ledger.**
   - Each 2 MiB extent cost about 34 KiB of unreclaimable slab while it
     existed: 134–135 MiB per 4096 extents, about 1.6% of the backing.
   - Mapping with access added about another 17–21 KiB per extent.
   - This lands in `MemAvailable` and in no process or cgroup counter. It
     belongs in `F`, or as a per-extent overhead in the envelopes, and the
     breakdown should show it as its own line.
4. **Unmapped host backing keeps its mapping cost until release.** After an
   unmap, host backing still held about 115 MiB more `MemAvailable` than
   right after it was created. Device backing returned to its
   created-but-unmapped level.

## What the breakdown does now

The architecture's
[memory breakdown](../../architecture.md#memory-breakdown) is updated from
this:
- the physical line uses `MemAvailable`;
- VMM backing is reconciled against the catalog's occupancy, not against
  the process footprint;
- the driver's slab bookkeeping is a separate, derived line;
- the cgroup and RSS counters are shown as covering the runtime's ordinary
  allocations only.

## Limitations

- **Scope:** one process, one GiB size (8), 2 MiB extents and one driver
  version. The per-extent overhead is inferred from this one extent size;
  it is not measured at other sizes.
- **Attribution:** the slab growth is attributed to the driver by its
  timing (it follows create, map and release). The slab caches responsible
  were not identified.
- **Process type:** a user-session scope, not the `llmp.service` unit.
  The cgroup conclusion should hold there too, but it was not measured
  there.

## Reproduce

```bash
tools/build deploy --host spark cross
```

```bash
ssh spark '~/.cache/llmp/deploy/cross-<id>/benchmarks/llmp_vmm_counters 8'
```
