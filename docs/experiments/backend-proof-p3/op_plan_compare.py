#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compares native's executed EXL3 plan with the operation plan record: the plan-first gate.

backend-proof.md's Tier E item "GGML-derived operations inside the EXL3 plan" requires native's
recorded executed plan to equal the record (../backend-proof-p0/exl3-op-plan.json, or P3's
exl3-op-plan-g.json / exl3-op-plan-o.json, same schema) per phase kind before any operation or
model comparison. This tool checks that. It is test and measurement tooling, never part of
llmpalooza's runtime.

  op_plan_compare.py --record RECORD.json --fixture 4.0bpw|4.5bpw RECORDING.jsonl
                     [--nsys RUN.sqlite] [--sass SASS.jsonl] [--demangler c++filt] [--steps 16]

RECORDING is benchmarks/exl3_exec.cc's record.jsonl (--record): tests/support's launch
recorder's JSON lines, each phase between chunk lines and each operation after an op line.
RUN.sqlite is `nsys export --type sqlite` of an `nsys profile -t cuda` trace of the same run:
it shows what cuBLAS launched inside each recorded cublasLtMatmul call, and any launch or copy
the recorder missed. SASS.jsonl is `cuobjdump -sass llmp_exl3_exec | fp16_plan.py sass-hash
--label native`. ExLlamaV3's and cuBLAS's kernels are named as the record names them, by
c++filt (nsys spells some demangled names differently).

Each phase's kind is resolved from its rows at position 0 (a prefill) or, for a single-token
step, its padded K length (Npad = n_past + rows rounded up to 256). Then, in order:
- the operations: `inputs`, the embedding, 24 x the kind's layer order, the output order, then
  `outputs`; native's one `attention` op covers the record's attention.kv_max (from 1,024
  rows), attention and attention.combine, whose launches it concatenates;
- each operation's launches, in order: GGML kernels by mangled name (NVCC's whole _INTERNAL_
  token normalized, as plan_compare.normalized does), SASS encoding hash, registers and static
  shared memory; ExLlamaV3's and cuBLAS's by demangled name; every kernel's grid, block and
  shared memory (static + dynamic). A recorded cuBLAS call stands for the trace's kernels up to
  the next recorded event. `inputs` uploads ids, positions and mask (n x 4, n x 4 and
  n x Npad x 2 bytes), each KV write copies n x 256 bytes and `outputs` downloads the logits
  (n x vocab x 2 bytes); only their sizes are compared;
- between phases, nothing but the cache's zero-fill (memsets of 0);
- the trace against the recording: each recorded launch, copy and memset is the trace's next
  (same kernel, grid, block, dynamic and static shared memory; not registers, which nsys reports
  as allocated, rounded up), from where the recording starts to where it ends (only the first
  evaluation is recorded; the trace goes on).
Only stream identity, addresses, the launch API (with at most the PDL attribute) and copy kinds
are not compared. The linear paths and output dtypes are compared through their launches: each
path, and each output dtype, launches its own kernels.

Exit status: 0 when every phase matched and nothing the gate needs is missing; 1 at the first
difference (named: the phase, the layer and operation, the launch, expected and got) or on an
unreadable input; 2 when what was recorded matched but the gate lacks something (named: the nsys
trace, a phase kind the record lacks, a GGML kernel's SASS, a demangler, a truncated recording,
a trajectory with fewer than --steps steps).
"""
import argparse
import collections
import importlib.util
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pc = _load("plan_compare", HERE.parent / "backend-proof-p2" / "plan_compare.py")
normalized = pc.normalized
FORMAT = pc.FORMAT
# The record's operations native runs as one `attention` op, in this order.
ATTENTION = ("attention.kv_max", "attention", "attention.combine")
# How native may launch its own kernels; "cudaLaunchKernelExC+attributes" (an attribute other
# than PDL) is a difference.
LAUNCH_APIS = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel")
KV_STRIDE = 256


PlanError = pc.PlanError  # an input that cannot be read


class Difference(Exception):
    def __init__(self, message, progress=0):
        super().__init__(message)
        self.progress = progress


def padded(n_past, rows):
    return -(-(n_past + rows) // KV_STRIDE) * KV_STRIDE


def op_label(name, layer):
    return f"layer {layer} {name}" if layer >= 0 else name


# --------------------------------------------------------------------------- the record


class Record:
    """The record for one fixture: each phase kind's operations with their launches."""

    def __init__(self, record, fixture):
        self.name = record.get("tool", "the record")
        self.kernels = {k["id"]: k for k in record["kernels"]}
        self.layers = record["model"]["layers"]
        self.vocab = record["model"]["vocab"]
        self.kinds = record["phase_kinds"]
        self.fixture = fixture
        for pk in self.kinds:
            if fixture not in pk["fixtures"]:
                raise PlanError(f"the record's {pk['phase_kind']} has no fixture {fixture!r}")
        for k in self.kernels.values():
            if k["owner"] not in ("ggml", "exllamav3", "cublas"):
                raise PlanError(f"record kernel {k['id']}: unknown owner {k['owner']!r}")
        self._ops = {}

    def kind(self, rows, n_past):
        for pk in self.kinds:
            if n_past == 0 and pk["first_position"] == 0 and pk["rows"] == rows:
                return pk
            if (n_past > 0 and rows == 1 and pk["rows"] == 1 and pk["first_position"] > 0
                    and pk["padded_kv_length"] == padded(n_past, rows)):
                return pk
        return None

    def launch(self, entry, where):
        if entry[0] == "memcpy":
            return ("copy", entry[2])
        if entry[0] not in self.kernels:
            raise PlanError(f"{where}: the record launches kernel {entry[0]}, which it does not list")
        return ("kernel", entry[0], list(entry[1]), list(entry[2]), entry[3])

    def layer_launches(self, pk, name, layer):
        where = f"{pk['phase_kind']}, layer {layer} {name}"
        if name in pk["layer_launches"]:
            return [self.launch(e, where) for e in pk["layer_launches"][name]]
        groups = pk["fixtures"][self.fixture]["linear_launches"].get(name)
        if groups is None:
            raise PlanError(f"{where}: the record has no launches for this operation")
        for group in groups:
            if layer in group["layers"]:
                return [self.launch(e, where) for e in group["launches"]]
        raise PlanError(f"{where}: the record's {self.fixture} launches do not cover this layer")

    def ops(self, pk):
        """[(name, layer, launches)] as native runs them; a launch is
        ("kernel", id, grid, block, shared) or ("copy", bytes)."""
        key = pk["phase_kind"]
        if key in self._ops:
            return self._ops[key]
        n, npad = pk["rows"], pk["padded_kv_length"]
        ops = [("inputs", -1, [("copy", n * 4), ("copy", n * 4), ("copy", n * npad * 2)])]
        for e in pk["embedding"]:
            ops.append((e["op"], -1, [self.launch(x, key) for x in e["launches"]]))
        order = pk["layer_order"]
        group = [i for i, name in enumerate(order) if name in ATTENTION]
        if "attention" not in order or group != list(range(group[0], group[-1] + 1)):
            raise PlanError(f"{key}: the layer order has no contiguous attention")
        for layer in range(self.layers):
            for i, name in enumerate(order):
                if i == group[0]:
                    ops.append(("attention", layer, [launch for part in order[group[0]:group[-1] + 1]
                                                     for launch in self.layer_launches(pk, part, layer)]))
                elif i not in group:
                    ops.append((name, layer, self.layer_launches(pk, name, layer)))
        output = {e["op"]: e["launches"] for e in pk["output"] + pk["fixtures"][self.fixture]["lm_head_launches"]}
        for name in pk["output_order"]:
            if name not in output:
                raise PlanError(f"{key}: the record has no launches for {name}")
            ops.append((name, -1, [self.launch(x, key) for x in output[name]]))
        ops.append(("outputs", -1, [("copy", n * self.vocab * 2)]))
        self._ops[key] = ops
        return ops

    def describe(self, launch):
        if launch[0] == "copy":
            return f"a {launch[1]}-byte copy"
        k = self.kernels[launch[1]]
        return f"kernel {launch[1]} {k['name'][:90]} grid {launch[2]} block {launch[3]} shared {launch[4]}"


# --------------------------------------------------------------------------- the recording


def read_recording(lines):
    """(header, phases, events outside phases, every event in order, why it is truncated or None)."""
    numbered = [(number, line.strip()) for number, line in enumerate(lines, 1) if line.strip()]
    header, phase, op, prefix, truncated = None, None, None, None, None
    phases, outside, events = [], [], []
    for index, (number, line) in enumerate(numbered):
        try:
            e = json.loads(line)
        except json.JSONDecodeError as error:
            if index == len(numbered) - 1 and header is not None:
                truncated = f"its last line ({number}) is cut off"
                break
            raise PlanError(f"line {number}: not JSON ({error})") from None
        kind = e.get("type")
        if header is None:
            if kind != "header" or e.get("format") != FORMAT:
                raise PlanError(f"line {number}: expected a {FORMAT} header")
            header = e
            continue
        if kind == "chunk":
            if phase is not None:
                raise PlanError(f"line {number}: a chunk begins inside another")
            if e["evaluation"] != 1:
                raise PlanError(f"line {number}: evaluation {e['evaluation']}; only the first is recorded")
            if e["chunk"] == 0:
                prefix = e["rows"]
            phase = {k: e[k] for k in ("chunk", "rows", "n_past")}
            phase.update(prefix=prefix, ops=[], stray=[], partial=False,
                         label=f"prefix {prefix}, chunk {e['chunk']} ({e['rows']} rows, n_past {e['n_past']})")
            op = None
        elif kind == "end_chunk":
            if phase is None:
                raise PlanError(f"line {number}: a chunk ends that never began")
            phases.append(phase)
            phase = None
        elif kind == "op":
            if phase is None:
                raise PlanError(f"line {number}: an operation outside a chunk")
            op = {"name": e["name"], "layer": e["layer"], "events": []}
            phase["ops"].append(op)
        elif kind in ("kernel", "memcpy", "memset", "cublas"):
            e["line"] = number
            if phase is None:
                e["where"] = "between phases"
                outside.append(e)
            elif op is None:
                e["where"] = f"{phase['label']}, before its first operation"
                phase["stray"].append(e)
            else:
                e["where"] = f"{phase['label']}, {op_label(op['name'], op['layer'])}"
                op["events"].append(e)
            events.append(e)
        else:
            raise PlanError(f"line {number}: unknown record {kind!r}")
    if header is None:
        raise PlanError("the recording is empty")
    if phase is not None:
        phase["partial"] = True
        phases.append(phase)
        truncated = truncated or f"it ends inside {phase['label']}"
    return header, phases, outside, events, truncated


def describe_event(e):
    if e["type"] == "kernel":
        return f"kernel {e['name'][:90]} grid {e['grid']} block {e['block']}"
    if e["type"] == "memcpy":
        return f"{e['bytes']}-byte copy"
    if e["type"] == "memset":
        return f"{e['bytes']}-byte memset of {e['value']}"
    return f"{e['function']} call"


# --------------------------------------------------------------------------- the trace


def is_cublas(t):
    return t["type"] == "kernel" and pc.fp16.is_cublas_kernel(t["kernel"], t["api"])


def align(trace, kernels, events, truncated):
    """{id(cuBLAS event): its trace kernels and memsets}: the trace walked beside the recording
    from where the recording's first event is, to its last."""
    if not events:
        return {}
    first = events[0]
    starts = [j for j, t in enumerate(trace)
              if (is_cublas(t) if first["type"] == "cublas" else t["type"] == first["type"])]
    best = None
    for start in starts:
        try:
            return _align_from(start, trace, kernels, events, truncated)
        except Difference as failure:
            if best is None or failure.progress > best.progress:
                best = failure
    raise best or Difference(f"the trace never shows the recording's first event, a {describe_event(first)}")


def _align_from(j, trace, kernels, events, truncated):
    groups = {}

    def fail(index, message):
        e = events[index]
        raise Difference(f"{e['where']} (recording line {e['line']}): {message}", index)

    for index, e in enumerate(events):
        if e["type"] == "cublas":
            group = []
            while j < len(trace) and (is_cublas(trace[j]) or (
                    trace[j]["type"] == "memset" and j + 1 < len(trace) and is_cublas(trace[j + 1]))):
                group.append(trace[j])
                j += 1
            if not group:
                fail(index, "the trace shows no launch for this cuBLAS call")
            groups[id(e)] = group
            continue
        if j >= len(trace):
            fail(index, "the trace ends before this event")
        t = trace[j]
        j += 1
        if t["type"] != e["type"] or (e["type"] == "kernel" and t["kernel"] != e["name"]):
            got = f"kernel {t['kernel'][:90]}" if t["type"] == "kernel" else f"{t['bytes']}-byte {t['type']}"
            fail(index, f"the trace has a {got} where the recording has a {describe_event(e)} (a launch or copy "
                        "the recorder did not see?)")
        if e["type"] == "kernel":
            seen = kernels[t["kernel"]]
            if (t["grid"], t["block"], t["dyn"]) != (e["grid"], e["block"], e["shared"]):
                fail(index, f"the trace launched grid {t['grid']} block {t['block']} shared {t['dyn']}")
            # Registers are not cross-checked: nsys reports the allocated count (rounded up to the
            # allocation granularity, 10 -> 16), the recorder and the record the kernel's own.
            if seen["static_shared"] != e["static_shared"]:
                fail(index, f"the trace's kernel has {seen['static_shared']} bytes of static shared memory; "
                            f"the recording {e['static_shared']}")
        elif t["bytes"] != e["bytes"]:
            fail(index, f"the trace's {t['type']} is {t['bytes']} bytes")
    # The first evaluation ends where the next one's cache clear begins, or the run ends.
    if not truncated and j < len(trace) and trace[j]["type"] != "memset":
        t = trace[j]
        fail(len(events) - 1, f"after the recording's last event the trace has a "
                              f"{('kernel ' + t['kernel'][:90]) if t['type'] == 'kernel' else t['type']} the "
                              "recorder did not see")
    return groups


# --------------------------------------------------------------------------- the comparison


def run_demangler(command):
    def demangle(names):
        try:
            out = subprocess.run([command], input="\n".join(names) + "\n", capture_output=True, text=True,
                                 check=True).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            return None
        return out if len(out) == len(names) else None
    return demangle


class Gate:
    def __init__(self, record, fixture, lines, nsys=None, sass=None, demangle=None, steps=16):
        self.record = Record(record, fixture)
        self.header, self.phases, self.outside, self.events, self.truncated = read_recording(lines)
        self.steps = steps
        self.missing = []
        self.stats = collections.defaultdict(collections.Counter)
        self.groups = None
        if nsys is not None:
            trace, kernels, _ = pc.load_trace(nsys)
            self.groups = align(trace, kernels, self.events, self.truncated)
            self.trace_kernels = kernels
        self.sass = None
        if sass is not None:
            self.sass = collections.defaultdict(set)
            for line in sass:
                if line.strip():
                    r = json.loads(line)
                    self.sass[normalized(r["function"])].add(r["encoding_sha256"])
        names = sorted({e["name"] for e in self.events if e["type"] == "kernel"}
                       | {t["kernel"] for g in (self.groups or {}).values() for t in g if t["type"] == "kernel"})
        demangled = demangle(names) if demangle and names else None
        self.demangled = dict(zip(names, demangled)) if demangled else None

    def lack(self, what):
        if what not in self.missing:
            self.missing.append(what)

    @staticmethod
    def fail(where, message):
        raise Difference(f"{where}: {message}")

    # -- launches

    def items(self, events):
        """Native's launches: recorded kernels and copies, and each cuBLAS call's traced kernels."""
        out = []
        for e in events:
            if e["type"] == "kernel":
                out.append({"kind": "kernel", "name": e["name"], "api": e["api"], "grid": e["grid"],
                            "block": e["block"], "shared": e["shared"] + e["static_shared"],
                            "registers": e["registers"], "static_shared": e["static_shared"]})
            elif e["type"] == "memcpy":
                out.append({"kind": "copy", "bytes": e["bytes"]})
            elif e["type"] == "memset":
                out.append({"kind": "memset", "bytes": e["bytes"], "value": e["value"]})
            elif self.groups is None:
                out.append({"kind": "cublas", "function": e["function"]})
            else:
                for t in self.groups[id(e)]:
                    out.append({"kind": "kernel", "name": t["kernel"], "grid": t["grid"], "block": t["block"],
                                "shared": t["dyn"] + self.trace_kernels[t["kernel"]]["static_shared"]}
                               if t["type"] == "kernel" else {"kind": "memset", "bytes": t["bytes"],
                                                              "value": t["value"]})
        return out

    @staticmethod
    def describe(item):
        if item["kind"] == "kernel":
            return f"kernel {item['name'][:90]} grid {item['grid']} block {item['block']} shared {item['shared']}"
        if item["kind"] == "copy":
            return f"a {item['bytes']}-byte copy"
        if item["kind"] == "memset":
            return f"a {item['bytes']}-byte memset of {item['value']}"
        return f"a {item['function']} call"

    def check_kernel(self, want, got, where):
        _, kid, grid, block, shared = want
        k = self.record.kernels[kid]
        if k["owner"] == "ggml":
            if normalized(got["name"]) != normalized(k["mangled"]):
                self.fail(where, f"{got['name']} where the record launches {k['mangled']} ({k['name'][:70]})")
        elif self.demangled is None:
            self.lack("ExLlamaV3's and cuBLAS's kernel names (no demangler; --demangler)")
        elif self.demangled[got["name"]] != k["name"]:
            self.fail(where, f"{self.demangled[got['name']]} where the record launches {k['name']}")
        for field, expected, actual in (("grid", grid, got["grid"]), ("block", block, got["block"]),
                                        ("shared memory (static + dynamic)", shared, got["shared"])):
            if expected != actual:
                self.fail(where, f"{k['name'][:70]}: {field} {actual}, the record {expected}")
        if "api" in got and got["api"] not in LAUNCH_APIS:
            self.fail(where, f"{k['name'][:70]} launched with {got['api']}; only the launch API and the PDL "
                             "attribute may differ")
        if k["owner"] != "ggml":
            return
        for key in ("registers", "static_shared"):
            if got.get(key) != k[key]:
                self.fail(where, f"{k['name'][:70]}: {key} {got.get(key)}, the record {k[key]}")
        if self.sass is None:
            self.lack("SASS hashes of the native binary (--sass)")
        elif normalized(k["mangled"]) not in self.sass:
            self.lack(f"the SASS of {k['mangled']}")
        elif self.sass[normalized(k["mangled"])] != {k["sass_encoding_sha256"]}:
            self.fail(where, f"{k['name'][:70]}: SASS encoding hash {sorted(self.sass[normalized(k['mangled'])])}, "
                             f"the record {k['sass_encoding_sha256']}")

    def compare_launches(self, expected, events, where, partial):
        got = self.items(events)
        i = j = 0
        while i < len(expected) or j < len(got):
            at = f"{where}, launch {i}"
            if j >= len(got):
                if partial:
                    return j
                self.fail(at, f"native's operation ends; the record continues with {self.record.describe(expected[i])}")
            if i >= len(expected):
                self.fail(at, f"native continues with {self.describe(got[j])} after the record's {len(expected)} "
                              "launches")
            want, item = expected[i], got[j]
            if item["kind"] == "cublas":  # no trace: the call stands for the record's cuBLAS kernels here
                run = 0
                while (i + run < len(expected) and expected[i + run][0] == "kernel"
                       and self.record.kernels[expected[i + run][1]]["owner"] == "cublas"):
                    run += 1
                if not run:
                    self.fail(at, f"native calls {item['function']} where the record has "
                                  f"{self.record.describe(want)}")
                self.lack("an nsys trace of the run (--nsys): cuBLAS's kernels, and any launch or copy the "
                          "recorder missed")
                i, j = i + run, j + 1
                continue
            if want[0] == "copy" and item["kind"] == "copy":
                if want[1] != item["bytes"]:
                    self.fail(at, f"a {item['bytes']}-byte copy where the record copies {want[1]} bytes")
            elif want[0] == "kernel" and item["kind"] == "kernel":
                self.check_kernel(want, item, at)
            else:
                self.fail(at, f"native has {self.describe(item)} where the record has {self.record.describe(want)}")
            i, j = i + 1, j + 1
        return j

    # -- phases

    def compare_phase(self, phase):
        label = phase["label"]
        pk = self.record.kind(phase["rows"], phase["n_past"])
        if pk is None:
            self.lack(f"a record of the phase kind of {label} ({phase['rows']} rows at n_past {phase['n_past']}, "
                      f"K padded to {padded(phase['n_past'], phase['rows'])})")
            return
        where = f"{label} [kind '{pk['phase_kind']}']"
        if phase["stray"]:
            self.fail(where, f"{describe_event(phase['stray'][0])} before the first operation")
        expected, got = self.record.ops(pk), phase["ops"]
        stats = self.stats[pk["phase_kind"]]
        for i in range(max(len(expected), len(got))):
            if i >= len(got):
                if phase["partial"]:
                    break
                name, layer, _ = expected[i]
                self.fail(where, f"native's phase ends after {len(got)} operations; the record continues with "
                                 f"{op_label(name, layer)}")
            op = got[i]
            if i >= len(expected):
                self.fail(where, f"native runs {op_label(op['name'], op['layer'])} after the record's "
                                 f"{len(expected)} operations")
            name, layer, launches = expected[i]
            if (op["name"], op["layer"]) != (name, layer):
                hint = ("; native's attention op covers attention.kv_max, attention and attention.combine"
                        if op["name"] in ATTENTION else "")
                self.fail(where, f"operation {i} is {op_label(op['name'], op['layer'])} where the record has "
                                 f"{op_label(name, layer)}{hint}")
            stats["launches"] += self.compare_launches(launches, op["events"], f"{where}, {op_label(name, layer)}",
                                                       phase["partial"] and i == len(got) - 1)
            stats["operations"] += 1
        stats["phases"] += 1

    def run(self):
        if self.groups is None:
            self.lack("an nsys trace of the run (--nsys): cuBLAS's kernels, and any launch or copy the recorder "
                      "missed")
        for e in self.outside:
            if e["type"] != "memset" or e["value"] != 0:
                self.fail(f"between phases (recording line {e['line']})",
                          f"a {describe_event(e)}; only the cache's zero-fill (memsets of 0) may run there")
        trajectory = None

        def close():
            if trajectory and trajectory["steps"] < self.steps and not self.truncated:
                self.lack(f"the steps of prefix {trajectory['prefix']}: {trajectory['steps']} of {self.steps} "
                          "(a fragment of the run)")

        for phase in self.phases:
            if phase["chunk"] == 0:
                if phase["n_past"] != 0:
                    self.fail(phase["label"], "a trajectory's prefill starts at position 0")
                close()
                trajectory = {"prefix": phase["rows"], "steps": 0}
            else:
                want = trajectory and trajectory["steps"] + 1
                if (not trajectory or phase["chunk"] != want or phase["rows"] != 1
                        or phase["n_past"] != trajectory["prefix"] + phase["chunk"] - 1):
                    self.fail(phase["label"], "out of place: a trajectory is its prefill, then one-row steps at "
                                              "consecutive positions")
                trajectory["steps"] += 1
            self.compare_phase(phase)
        close()
        if self.truncated:
            self.lack(f"the whole recording: {self.truncated}")
        if not self.phases:
            self.lack("any phase")


def compare(record, fixture, lines, nsys=None, sass=None, demangle=None, steps=16):
    """(status, message, summary lines): 0 matched and complete, 1 a difference, 2 matched but incomplete."""
    try:
        gate = Gate(record, fixture, lines, nsys, sass, demangle, steps)
        gate.run()
    except Difference as difference:
        return 1, f"DIFFERENT: {difference}", []
    summary = [f"{kind}: {s['phases']} phases, {s['operations']} operations, {s['launches']} launches"
               for kind, s in gate.stats.items()]
    scope = f"{sum(s['phases'] for s in gate.stats.values())} phases of {len(gate.stats)} kinds ({fixture})"
    if gate.missing:
        return 2, f"INCOMPLETE: {scope} matched the record, but the gate lacks " + "; ".join(gate.missing), summary
    return 0, f"MATCH: {scope} equal the record", summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--record", required=True, help="exl3-op-plan.json, or a P3 record of the same schema")
    ap.add_argument("--fixture", required=True, choices=("4.0bpw", "4.5bpw"))
    ap.add_argument("--nsys", help="nsys export --type sqlite of the same run")
    ap.add_argument("--sass", help="fp16_plan.py sass-hash output for the native binary")
    ap.add_argument("--demangler", default="c++filt")
    ap.add_argument("--steps", type=int, default=16, help="single-token steps per trajectory (default 16)")
    a = ap.parse_args(argv)
    try:
        record = json.loads(pathlib.Path(a.record).read_text())
        lines = pathlib.Path(a.recording).read_text().splitlines()
        sass = pathlib.Path(a.sass).read_text().splitlines() if a.sass else None
        status, message, summary = compare(record, a.fixture, lines, a.nsys, sass, run_demangler(a.demangler),
                                           a.steps)
    except (PlanError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for line in summary:
        print(f"  {line}")
    print(message)
    return status


if __name__ == "__main__":
    sys.exit(main())
