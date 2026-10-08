#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compares a native executed plan with the FP16 bridge's recorded plan.

backend-proof.md's FP16 Tier E gate compares logits only after the native
executed plan matches the bridge's (../backend-proof-p0/fp16-plan.json), per
chunk shape. This tool does that comparison. It is test and measurement
tooling, never part of llmpalooza's runtime.

  plan_compare.py convert RECORDING.jsonl [--cublas-log F --cublaslt-log F]
                  [--sass SASS.jsonl] [--nsys RUN.sqlite] [--library NAME=SHA256 ...] --out NATIVE.json
  plan_compare.py compare --reference fp16-plan.json --arm ARM --native NATIVE.json
                  [--fragment SEQUENCE:START]
  plan_compare.py reference --reference fp16-plan.json --arm ARM --out NATIVE.json
                  [--no-cublas-launches]

`convert` turns what tests/support's launch recorder wrote (JSON lines: a
header, chunk markers, and each kernel launch, copy, memset and cuBLAS call
the process made) into a native plan. The recorder sees the calls llmpalooza
makes, not the kernels cuBLAS launches inside itself. Those come from an
Nsight Systems trace of the same run (--nsys, run where the trace is; the
bridge's record came from one too). cuBLAS's parameters, resolved algorithm
and handle setup come from its own logs, captured as the bridge's were
(CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=F CUBLASLT_LOG_LEVEL=5
CUBLASLT_LOG_FILE=F), and SASS hashes from `cuobjdump -sass BINARY |
fp16_plan.py sass-hash --label executable`. `reference` writes one of the
bridge's arms as a native plan; the tooling tests mutate it.

`compare` walks every recorded chunk and stops at the first difference,
naming the chunk, the token and where it sits in the record's folded
sequence. Compared:
- chunks: order, rows and n_past, and every chunk of the first evaluation;
- tokens in order: each kernel's name (NVCC's per-file _INTERNAL_ hashes
  normalized), grid, block and dynamic shared memory, and its registers,
  static shared and local memory and SASS encoding hash; each copy's size;
  each memset's size and value; each cuBLAS call in place;
- each cuBLAS call's function, parameters, math mode and cuBLASLt record
  (heuristic preference, including workspace limit and operand alignments,
  its result, descriptors, compute descriptor with the SM count, resolved
  algorithm, workspace used), and the kernels and memsets it launched;
- the cuBLAS handle's setup (math mode, stream, 32 MiB workspace) and the
  cuBLAS and cuBLASLt library hashes;
- nothing launched outside the chunks: the bridge only copies and clears
  memory between them, so a kernel or cuBLAS call there is a difference.
Only stream identity, addresses, the launch API (the PDL attribute rides on
cudaLaunchKernelExC; a launch with any other attribute is a difference) and
the copies' memory kinds (host VMM is device memory to CUDA, D-034) are not
compared. Exit status: 0 when everything matched and nothing the gate needs
is missing, 1 at a difference, 2 when what was recorded matched but the
recording lacks something the gate needs (named). A fragment, or a run
without its nsys trace (which also shows any launch or copy the recorder
missed), is never more than 2.
"""
import argparse
import collections
import importlib.util
import json
import pathlib
import sqlite3
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
FORMAT = "llmp-plan-record/1"
# The cuBLAS libraries whose identity is part of the plan (the record's
# `libraries`). The bridge's cudart is dynamic and llmpalooza's static; neither
# chooses a kernel.
PLAN_LIBRARIES = ("libcublas.so.13.8.0.4", "libcublasLt.so.13.8.0.4")
# cuBLAS handle calls that set state, in the order upstream makes them.
HANDLE_SETUP = ("cublasCreate_v2", "cublasSetMathMode", "cublasSetStream_v2", "cublasSetWorkspace_v2")
# How llmpalooza's own kernels may be launched: with or without the PDL
# attribute, the only launch attribute the gate lets differ. The launch
# recorder names any other (cudaLaunchKernelExC+attributes).
LAUNCH_APIS = ("cudaLaunchKernel", "cudaLaunchKernelExC")
LT_FIELDS = ("function", "preference", "heuristic", "Adesc", "Bdesc", "Cdesc", "Ddesc", "computeDesc", "algo",
             "workspace_bytes", "beta")


def _load_fp16_plan():
    spec = importlib.util.spec_from_file_location("fp16_plan", HERE.parent / "backend-proof-p0" / "fp16_plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fp16 = _load_fp16_plan()


class PlanError(ValueError):
    """A recording or record that cannot be read as a plan."""


# --------------------------------------------------------------------------- names


def normalized(name):
    """NVCC's internal-linkage token _INTERNAL_<hash>_<n>_<file>_<hash> with both hashes masked."""
    out = list(name)
    token = "_INTERNAL_"
    at = name.find(token)
    while at != -1:
        p = at + len(token)
        if p + 8 > len(name):
            break
        out[p:p + 8] = "x" * 8
        p += 9
        length = 0
        while p < len(name) and name[p].isdigit():
            length = length * 10 + int(name[p])
            p += 1
        p += 1 + length + 1
        if p + 8 > len(name):
            break
        out[p:p + 8] = "x" * 8
        at = name.find(token, p + 8)
    return "".join(out)


# --------------------------------------------------------------------------- the reference


def unfold(plan, path=()):
    """The folded token list as (token, path) pairs; a path step is (block index, iteration)."""
    for i, node in enumerate(plan):
        if isinstance(node, dict):
            for rep in range(node["repeat"]):
                yield from unfold(node["body"], path + ((i, rep, node["repeat"]),))
        else:
            yield node, path + ((i, None, None),)


def describe_path(path):
    parts = []
    for index, rep, count in path:
        parts.append(f"item {index}" if rep is None else f"item {index} (repeat {rep + 1} of {count})")
    return " > ".join(parts)


def reference_chunks(record, arm_name):
    """One arm of fp16-plan.json: its sequences as token lists with cuBLAS groups collapsed."""
    arms = {a["arm"]: a for a in record["arms"]}
    if arm_name not in arms:
        raise PlanError(f"the record has no arm {arm_name!r} (it has {sorted(arms)})")
    arm = arms[arm_name]
    calls = {c["id"]: c for c in record["cublas_calls"]}
    sequences = {}
    for s in arm["sequences"]:
        flat = list(unfold(s["plan"]))
        tokens = []
        i = 0
        while i < len(flat):
            token, path = flat[i]
            tag = token[-1] if isinstance(token[-1], str) else ""
            if tag.startswith("cublas#"):
                cid = int(tag[len("cublas#"):])
                count = len(calls[cid]["launches"])
                group = [t for t, _ in flat[i:i + count]]
                if len(group) != count or any(t[-1] != tag for t in group):
                    raise PlanError(f"{arm_name}: sequence {s['id']} token {i}: cuBLAS call {cid} is not "
                                    f"{count} consecutive launches")
                tokens.append((["cublas", cid], path))
                i += count
                continue
            tokens.append((token, path))
            i += 1
        sequences[s["id"]] = {"rows": s["rows"], "input_copies": s["input_copies"], "tokens": tokens}
    chunks = [{"chunk": c["chunk"], "rows": c["rows"], "n_past": c["n_past"], "sequence": c["sequence"]}
              for c in arm["chunks_first_evaluation"]]
    return arm, sequences, chunks


def reference_as_native(record, arm_name, cublas_launches=True):
    """The bridge's arm written as a native plan would be: the tooling tests' starting point."""
    arm, sequences, chunks = reference_chunks(record, arm_name)
    kernels = {k["id"]: k for k in record["kernels"]}
    used = collections.OrderedDict()

    def kid(ref_id):
        return used.setdefault(ref_id, len(used))

    call_ids = collections.OrderedDict()
    out_chunks = []
    evaluations = 3 if arm.get("evaluations_identical") else 1
    for evaluation in range(1, evaluations + 1):
        for c in chunks:
            tokens = []
            for token, _ in sequences[c["sequence"]]["tokens"]:
                if token[0] == "kernel":
                    tokens.append(["kernel", kid(token[1]), token[2], token[3], token[4], "s0"])
                elif token[0] == "memcpy":
                    tokens.append(["memcpy", token[1], token[4], "s0"])
                elif token[0] == "memset":
                    tokens.append(["memset", token[1], token[2], "s0"])
                else:
                    tokens.append(["cublas", call_ids.setdefault(token[1], len(call_ids))])
            out_chunks.append({"evaluation": evaluation, "chunk": c["chunk"], "rows": c["rows"],
                               "n_past": c["n_past"], "tokens": tokens})
    calls = {c["id"]: c for c in record["cublas_calls"]}
    for call in call_ids:
        for launch in calls[call]["launches"]:
            if launch[0] == "kernel":
                kid(launch[1])
    out_kernels = []
    for ref_id, new_id in used.items():
        k = kernels[ref_id]
        entry = {"id": new_id, "mangled": k["mangled"], "owner": k["owner"], "registers": k["registers"],
                 "static_shared": k["static_shared"], "local_per_thread": k["local_per_thread"],
                 "launch_api": k["launch_api"]}
        if k["owner"] != "cublas":
            entry["sass_encoding_sha256"] = k["sass_encoding_sha256"]
        out_kernels.append(entry)
    out_calls = []
    for ref_id, new_id in call_ids.items():
        c = calls[ref_id]
        entry = {"id": new_id, "function": c["function"], "args": c["args"], "math_mode": c["math_mode"],
                 "cublaslt": c["cublaslt"]}
        if cublas_launches:
            entry["launches"] = [[l[0], kid(l[1]), *l[2:5]] if l[0] == "kernel" else l[:3]
                                 for l in c["launches"]]
        out_calls.append(entry)
    setup = [{"function": s["function"], "args": s["args"]} for s in arm["cublas_handle_setup"]
             if s["function"] in HANDLE_SETUP]
    plan = {"format": FORMAT, "source": f"reference arm {arm_name}",
            "libraries": {n: record["libraries"][n] for n in PLAN_LIBRARIES},
            "cublas_logs": True, "cublas_launches_observed": cublas_launches, "sass": True,
            "outside_chunks": {"kernels": [], "cublas_calls": 0, "copies": 0, "memsets": 0},
            "kernels": out_kernels, "cublas_calls": out_calls, "cublas_handle_setup": setup,
            "chunks": out_chunks}
    return json.loads(json.dumps(plan))  # shares nothing with the record


# --------------------------------------------------------------------------- the recording


def read_recording(lines):
    header, events, chunk, chunks, outside = None, [], None, [], []
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as error:
            raise PlanError(f"line {number}: not JSON ({error})") from None
        kind = e.get("type")
        if number == 1 or header is None:
            if kind != "header" or e.get("format") != FORMAT:
                raise PlanError(f"line {number}: expected a {FORMAT} header")
            header = e
            continue
        if kind == "chunk":
            if chunk is not None:
                raise PlanError(f"line {number}: a chunk begins inside another")
            chunk = {k: e[k] for k in ("evaluation", "chunk", "rows", "n_past")}
            chunk["events"] = []
        elif kind == "end_chunk":
            if chunk is None:
                raise PlanError(f"line {number}: a chunk ends that never began")
            chunks.append(chunk)
            chunk = None
        elif kind in ("kernel", "memcpy", "memset", "cublas"):
            e["line"] = number
            (chunk["events"] if chunk is not None else outside).append(e)
            events.append(e)
        else:
            raise PlanError(f"line {number}: unknown record {kind!r}")
    if header is None:
        raise PlanError("the recording is empty")
    if chunk is not None:
        raise PlanError("the recording ends inside a chunk")
    return header, events, chunks, outside


# The activity tables fp16_plan.load_nsys reads, as it reads them. An nsys
# export leaves out a table with no rows (a run with no memset has no
# MEMSET table), so a copy of the trace gets empty ones.
ACTIVITY_TABLES = {
    "CUPTI_ACTIVITY_KIND_KERNEL": "correlationId, streamId, demangledName, mangledName, gridX, gridY, gridZ, "
                                  "blockX, blockY, blockZ, dynamicSharedMemory, staticSharedMemory, "
                                  "registersPerThread, localMemoryPerThread",
    "CUPTI_ACTIVITY_KIND_MEMCPY": "correlationId, streamId, bytes, copyKind, srcKind, dstKind",
    "CUPTI_ACTIVITY_KIND_MEMSET": "correlationId, streamId, bytes, value",
}


def load_trace(path):
    try:
        return _load_trace(path)
    except SystemExit as error:  # fp16_plan's own refusals
        raise PlanError(f"the trace: {error}") from None


def _load_trace(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    present = {name for (name,) in db.execute("select name from sqlite_master where type = 'table'")}
    missing = [t for t in ACTIVITY_TABLES if t not in present]
    if not missing:
        db.close()
        return fp16.load_nsys(path)
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp) / "trace.sqlite"
        target = sqlite3.connect(copy)
        db.backup(target)
        db.close()
        for table in missing:
            target.execute(f"create table {table} ({ACTIVITY_TABLES[table]})")
        target.commit()
        target.close()
        return fp16.load_nsys(str(copy))


def cublas_launches_from_nsys(path, events):
    """Each recorded cuBLAS call's own launches, from an nsys trace of the same run.

    The trace's kernels, copies and memsets are walked in API order beside
    the recording: each recorded launch or copy must meet its own event, and
    what follows a recorded cuBLAS call up to the next recorded event is
    what cuBLAS launched for it.
    """
    traced, traced_kernels, _ = load_trace(path)
    j = 0

    def next_event(what):
        nonlocal j
        if j >= len(traced):
            raise PlanError(f"the trace ends before the recording's {what}")
        j += 1
        return traced[j - 1]

    launches = {}
    for index, e in enumerate(events):
        if e["type"] == "cublas":
            group = []
            while j < len(traced) and (
                    (traced[j]["type"] == "kernel" and fp16.is_cublas_kernel(traced[j]["kernel"], traced[j]["api"]))
                    or (traced[j]["type"] == "memset" and j + 1 < len(traced)
                        and traced[j + 1]["type"] == "kernel"
                        and fp16.is_cublas_kernel(traced[j + 1]["kernel"], traced[j + 1]["api"]))):
                group.append(traced[j])
                j += 1
            if not group:
                raise PlanError(f"recording line {e['line']}: the trace shows no launch for this cuBLAS call")
            launches[index] = group
            continue
        t = next_event(f"line {e['line']}")
        if t["type"] != e["type"]:
            raise PlanError(f"recording line {e['line']}: the trace has a {t['type']} where the recording has a "
                            f"{e['type']}")
        if e["type"] == "kernel" and (t["kernel"] != e["name"] or t["grid"] != e["grid"]):
            raise PlanError(f"recording line {e['line']}: the trace launched {t['kernel']} {t['grid']}")
        if e["type"] == "kernel":
            seen = traced_kernels[t["kernel"]]
            if (seen["registers"], seen["static_shared"], seen["local_per_thread"]) != (
                    e["registers"], e["static_shared"], e["local"]):
                raise PlanError(f"recording line {e['line']}: the trace's attributes differ: {seen}")
        if e["type"] in ("memcpy", "memset") and t["bytes"] != e["bytes"]:
            raise PlanError(f"recording line {e['line']}: the trace's {t['type']} is {t['bytes']} bytes")
    if j != len(traced):
        raise PlanError(f"the trace has {len(traced) - j} events after the recording's last")
    return launches, traced_kernels


def convert(lines, cublas_log=None, cublaslt_log=None, sass=None, nsys=None, libraries=()):
    header, events, chunks, outside = read_recording(lines)
    kernel_index, kernels = {}, []

    def kernel_id(name, attributes, api, traced=None):
        """Recorded kernels carry their attributes; cuBLAS's come from the trace."""
        if name not in kernel_index:
            kernel_index[name] = len(kernels)
            own = {k: traced[k] for k in ("registers", "static_shared", "local_per_thread")} if traced else {}
            kernels.append({"id": len(kernels), "mangled": name, "owner": "cublas" if attributes is None else "ggml",
                            "launch_api": set(), **(attributes or own)})
        entry = kernels[kernel_index[name]]
        if attributes is not None:
            for key, value in attributes.items():
                if entry.get(key) != value:
                    raise PlanError(f"kernel {name}: {key} varies between launches")
        if api:
            entry["launch_api"].add(api)
        return entry["id"]

    markers = [e for e in events if e["type"] == "cublas"]
    have_logs = cublas_log is not None and cublaslt_log is not None
    setup = []
    if have_logs:
        legacy, setup_calls = fp16.parse_cublas(cublas_log)
        lt = fp16.parse_cublaslt(cublaslt_log)
        if not len(legacy) == len(lt) == len(markers):
            raise PlanError(f"{len(markers)} recorded cuBLAS calls, {len(legacy)} in cuBLAS's log and {len(lt)} "
                            "in cuBLASLt's")
        for marker, call, ltrec in zip(markers, legacy, lt):
            if call["function"] != marker["function"]:
                raise PlanError(f"recording line {marker['line']}: {marker['function']} but cuBLAS logged "
                                f"{call['function']}")
            marker["log"] = (call, ltrec)
        setup = [{"function": c["function"], "args": c["args"]} for c in setup_calls
                 if c["function"] in HANDLE_SETUP]
    nsys_launches, traced_kernels = cublas_launches_from_nsys(nsys, events) if nsys else (None, {})

    call_table = {}

    def call_id(index, marker):
        if have_logs:
            call, ltrec = marker["log"]
            entry = {"function": call["function"], "args": call["args"], "math_mode": call.get("math_mode"),
                     "cublaslt": ltrec}
        else:
            entry = {"function": marker["function"], "args": {k: marker[k] for k in ("m", "n", "k", "batch")
                                                              if k in marker}}
        if nsys_launches is not None:
            entry["launches"] = [
                ["kernel", kernel_id(t["kernel"], None, t["api"], traced_kernels[t["kernel"]]), t["grid"],
                 t["block"], t["dyn"]]
                if t["type"] == "kernel" else ["memset", t["bytes"], t["value"]]
                for t in nsys_launches[index]]
        key = json.dumps(entry, sort_keys=True)
        if key not in call_table:
            call_table[key] = dict(entry, id=len(call_table))
        return call_table[key]["id"]

    position = {id(e): i for i, e in enumerate(events)}
    streams = {}
    out_chunks = []
    for c in chunks:
        tokens = []
        for e in c["events"]:
            stream = streams.setdefault(e.get("stream"), f"s{len(streams)}")
            if e["type"] == "kernel":
                attributes = {"registers": e["registers"], "static_shared": e["static_shared"],
                              "local_per_thread": e["local"]}
                tokens.append(["kernel", kernel_id(e["name"], attributes, e["api"]), e["grid"], e["block"],
                               e["shared"], stream])
            elif e["type"] == "memcpy":
                tokens.append(["memcpy", e["kind"], e["bytes"], stream])
            elif e["type"] == "memset":
                tokens.append(["memset", e["bytes"], e["value"], stream])
            else:
                tokens.append(["cublas", call_id(position[id(e)], e)])
        out_chunks.append({k: c[k] for k in ("evaluation", "chunk", "rows", "n_past")} | {"tokens": tokens})
    inside = {id(x) for c in chunks for x in c["events"]}
    for e in events:  # calls outside chunks enter the table too, so its ids are stable
        if e["type"] == "cublas" and id(e) not in inside:
            call_id(position[id(e)], e)

    hashes = collections.defaultdict(set)
    if sass is not None:
        for line in sass:
            if line.strip():
                r = json.loads(line)
                if r["label"] == "executable":
                    hashes[r["function"]].add(r["encoding_sha256"])
    for k in kernels:
        k["launch_api"] = sorted(k["launch_api"])
        if k["owner"] == "ggml" and sass is not None:
            found = sorted(hashes.get(k["mangled"], ()))
            k["sass_encoding_sha256"] = found[0] if len(found) == 1 else found
    known = {}
    for name, path in header.get("loaded", {}).items():  # by the plan's names: libcublas.so.13 is 13.8.0.4's
        real = pathlib.Path(path).resolve()
        plan_name = next((p for p in PLAN_LIBRARIES if p.split(".so")[0] == name.split(".so")[0]), None)
        if plan_name and real.is_file():
            known[plan_name] = fp16.sha256_file(real)
    known.update(dict(item.split("=", 1) for item in libraries))
    between = collections.Counter(e["type"] for e in outside)
    return {"format": FORMAT, "source": header.get("source", "native recording"),
            "outside_chunks": {"kernels": sorted({e["name"] for e in outside if e["type"] == "kernel"}),
                               "cublas_calls": between["cublas"], "copies": between["memcpy"],
                               "memsets": between["memset"]},
            "libraries": known, "cublas_logs": have_logs, "cublas_launches_observed": nsys is not None,
            "sass": sass is not None, "kernels": kernels,
            "cublas_calls": sorted(call_table.values(), key=lambda c: c["id"]),
            "cublas_handle_setup": setup, "chunks": out_chunks}


# --------------------------------------------------------------------------- the comparison


class Difference(Exception):
    pass


class Comparison:
    def __init__(self, record, arm_name, native):
        self.record = record
        self.arm_name = arm_name
        self.arm, self.sequences, self.chunks = reference_chunks(record, arm_name)
        self.ref_kernels = {k["id"]: k for k in record["kernels"]}
        self.ref_calls = {c["id"]: c for c in record["cublas_calls"]}
        self.native = native
        self.nat_kernels = {k["id"]: k for k in native["kernels"]}
        self.nat_calls = {c["id"]: c for c in native["cublas_calls"]}
        self.missing = []
        self.matched_calls = {}

    def lack(self, what):
        if what not in self.missing:
            self.missing.append(what)

    def fail(self, where, message):
        raise Difference(f"{where}: {message}")

    # -- tokens

    def describe_ref(self, token):
        if token[0] == "kernel":
            k = self.ref_kernels[token[1]]
            return (f"kernel {token[1]} {k['demangled'][:90]} grid {token[2]} block {token[3]} "
                    f"shared {token[4]}")
        if token[0] == "memcpy":
            return f"{token[4]}-byte copy ({token[1]} {token[2]}->{token[3]})"
        if token[0] == "memset":
            return f"{token[1]}-byte memset of {token[2]}"
        c = self.ref_calls[token[1]]
        return f"cuBLAS call {token[1]} {c['function']} {self.short_args(c['args'])}"

    def describe_nat(self, token):
        if token[0] == "kernel":
            k = self.nat_kernels[token[1]]
            return f"kernel {k['mangled'][:90]} grid {token[2]} block {token[3]} shared {token[4]}"
        if token[0] == "memcpy":
            return f"{token[2]}-byte copy ({token[1]})"
        if token[0] == "memset":
            return f"{token[1]}-byte memset of {token[2]}"
        c = self.nat_calls[token[1]]
        return f"cuBLAS call {c['function']} {self.short_args(c['args'])}"

    @staticmethod
    def short_args(args):
        return "(" + ", ".join(f"{k}={args[k]}" for k in ("m", "n", "k", "batchCount", "batch") if k in args) + ")"

    def same_kernel(self, ref_id, nat_id, where):
        ref, nat = self.ref_kernels[ref_id], self.nat_kernels[nat_id]
        if normalized(ref["mangled"]) != normalized(nat["mangled"]):
            self.fail(where, f"kernel {nat['mangled']} where the record has {ref['mangled']}")
        for key in ("registers", "static_shared", "local_per_thread"):
            if key not in nat:
                self.lack(f"kernel attributes ({key})")
            elif nat[key] != ref[key]:
                self.fail(where, f"kernel {ref_id} ({ref['demangled'][:60]}) has {key} {nat[key]}, "
                                 f"the record {ref[key]}")
        if nat.get("owner") != "cublas":
            other = sorted(set(nat.get("launch_api", ())) - set(LAUNCH_APIS))
            if other:
                self.fail(where, f"kernel {ref_id} ({ref['demangled'][:60]}) launched with {other}; only the "
                                 "PDL attribute may differ")
        if ref["owner"] != "cublas":
            if "sass_encoding_sha256" not in nat:
                self.lack("SASS hashes of the native binary (--sass)")
            elif nat["sass_encoding_sha256"] != ref["sass_encoding_sha256"]:
                self.fail(where, f"kernel {ref_id} ({ref['demangled'][:60]}) has SASS encoding hash "
                                 f"{nat['sass_encoding_sha256']}, the record {ref['sass_encoding_sha256']}")

    def same_launch(self, ref, nat, where):
        """A kernel or memset token: [kernel, id, grid, block, shared, ...] or [memset, bytes, value, ...]."""
        if ref[0] != nat[0]:
            return False
        if ref[0] == "memset":
            return ref[1] == nat[1] and ref[2] == nat[2]
        if list(ref[2]) != list(nat[2]) or list(ref[3]) != list(nat[3]) or ref[4] != nat[4]:
            return False
        self.same_kernel(ref[1], nat[1], where)
        return True

    def same_call(self, ref_id, nat_id, where):
        ref, nat = self.ref_calls[ref_id], self.nat_calls[nat_id]
        if ref["function"] != nat["function"]:
            self.fail(where, f"cuBLAS {nat['function']} where the record calls {ref['function']} "
                             f"(call {ref_id})")
        if not self.native.get("cublas_logs"):
            self.lack("cuBLAS and cuBLASLt logs (--cublas-log, --cublaslt-log)")
            for key, value in nat["args"].items():
                ref_key = "batchCount" if key == "batch" else key
                if ref["args"].get(ref_key) != value:
                    self.fail(where, f"cuBLAS call {ref_id}: {ref_key} {value}, the record "
                                     f"{ref['args'].get(ref_key)}")
        else:
            for key in sorted(set(ref["args"]) | set(nat["args"])):
                if ref["args"].get(key) != nat["args"].get(key):
                    self.fail(where, f"cuBLAS call {ref_id} ({ref['function']}): {key} is "
                                     f"{nat['args'].get(key)!r}, the record {ref['args'].get(key)!r}")
            if ref["math_mode"] != nat.get("math_mode"):
                self.fail(where, f"cuBLAS call {ref_id}: math mode {nat.get('math_mode')}, the record "
                                 f"{ref['math_mode']}")
            for key in LT_FIELDS:
                if ref["cublaslt"].get(key) != nat["cublaslt"].get(key):
                    self.fail(where, f"cuBLAS call {ref_id}: cuBLASLt {key} is {nat['cublaslt'].get(key)!r}, "
                                     f"the record {ref['cublaslt'].get(key)!r}")
        if "launches" not in nat:
            self.lack("the kernels cuBLAS launched (an nsys trace of the run, --nsys)")
        else:
            refs = [["kernel", l[1], l[2], l[3], l[4]] if l[0] == "kernel" else l[:3] for l in ref["launches"]]
            nats = nat["launches"]
            for i in range(max(len(refs), len(nats))):
                at = f"{where}, cuBLAS call {ref_id}'s launch {i}"
                if i >= len(refs):
                    self.fail(at, f"cuBLAS launched {len(nats)} where the record has {len(refs)}")
                if i >= len(nats):
                    self.fail(at, f"cuBLAS launched {len(nats)} where the record has {len(refs)}")
                if refs[i][0] == "kernel" and nats[i][0] == "kernel" and normalized(
                        self.ref_kernels[refs[i][1]]["mangled"]) != normalized(self.nat_kernels[nats[i][1]]["mangled"]):
                    self.fail(at, f"{self.nat_kernels[nats[i][1]]['mangled']} where the record launches "
                                  f"{self.ref_kernels[refs[i][1]]['mangled']}")
                if not self.same_launch(refs[i], nats[i], at):
                    self.fail(at, f"{nats[i]} where the record has {refs[i]}")
        previous = self.matched_calls.setdefault(nat_id, ref_id)
        if previous != ref_id:
            self.fail(where, f"one native cuBLAS call matches record calls {previous} and {ref_id}")

    def key(self, token, reference):
        """What identifies a token for a hint: no attribute checks, no side effects."""
        if token[0] == "kernel":
            table = self.ref_kernels if reference else self.nat_kernels
            return ("kernel", normalized(table[token[1]]["mangled"]), tuple(token[2]), tuple(token[3]), token[4])
        if token[0] == "memcpy":
            return ("memcpy", token[4] if reference else token[2])
        if token[0] == "memset":
            return ("memset", token[1], token[2])
        call = (self.ref_calls if reference else self.nat_calls)[token[1]]
        return ("cublas", call["function"], json.dumps(call["args"], sort_keys=True) if self.native.get(
            "cublas_logs") else self.short_args(call["args"]).replace("batchCount", "batch"))

    def hint(self, ref_tokens, nat_tokens, i):
        """Whether the difference at i looks like a dropped, an inserted or a moved token."""
        window = 8
        ref_keys = [self.key(t, True) for t, _ in ref_tokens[i:i + window]]
        nat_keys = [self.key(t, False) for t in nat_tokens[i:i + window]]
        if ref_keys and ref_keys[0] in nat_keys[1:]:
            return (f"; the record's token comes {nat_keys.index(ref_keys[0], 1)} later in native "
                    "(inserted or reordered launches?)")
        if nat_keys and nat_keys[0] in ref_keys[1:]:
            return (f"; native's token comes {ref_keys.index(nat_keys[0], 1)} later in the record "
                    "(a dropped or reordered launch?)")
        return ""

    def compare_tokens(self, ref_tokens, nat_tokens, where, start=0):
        for i in range(max(len(ref_tokens), len(nat_tokens))):
            at = f"{where}, token {start + i}"
            if i >= len(nat_tokens):
                ref, path = ref_tokens[i]
                self.fail(at, f"native ends; the record continues with {self.describe_ref(ref)} "
                              f"[{describe_path(path)}]")
            if i >= len(ref_tokens):
                self.fail(at, f"native continues with {self.describe_nat(nat_tokens[i])} after the record's "
                              f"{len(ref_tokens)} tokens")
            ref, path = ref_tokens[i]
            at = f"{at} [{describe_path(path)}]"
            nat = nat_tokens[i]
            if ref[0] == "cublas" and nat[0] == "cublas":
                self.same_call(ref[1], nat[1], at)
                continue
            if ref[0] == "kernel" and nat[0] == "kernel":
                rk, nk = self.ref_kernels[ref[1]], self.nat_kernels[nat[1]]
                if normalized(rk["mangled"]) == normalized(nk["mangled"]):
                    for field, index in (("grid", 2), ("block", 3), ("dynamic shared memory", 4)):
                        want, got = ref[index], nat[index]
                        if (list(want) if index < 4 else want) != (list(got) if index < 4 else got):
                            self.fail(at, f"kernel {ref[1]} ({rk['demangled'][:70]}): {field} {got}, the "
                                          f"record {want}")
                    self.same_kernel(ref[1], nat[1], at)
                    continue
            if ref[0] == "memcpy" and nat[0] == "memcpy":
                if ref[4] != nat[2]:
                    self.fail(at, f"a {nat[2]}-byte copy where the record copies {ref[4]} bytes")
                continue
            if ref[0] == "memset" and nat[0] == "memset" and ref[1] == nat[1] and ref[2] == nat[2]:
                continue
            self.fail(at, f"native has {self.describe_nat(nat)} where the record has {self.describe_ref(ref)}"
                          + self.hint(ref_tokens, nat_tokens, i))

    def check_identity(self):
        libraries = self.native.get("libraries", {})
        for name in PLAN_LIBRARIES:
            if name not in libraries:
                self.lack(f"the hash of {name} (--library)")
            elif libraries[name] != self.record["libraries"][name]:
                self.fail("libraries", f"{name} has SHA-256 {libraries[name]}, the record "
                                       f"{self.record['libraries'][name]}")
        if not self.native.get("cublas_logs"):
            self.lack("cuBLAS and cuBLASLt logs (--cublas-log, --cublaslt-log)")
            return
        want = [{"function": s["function"], "args": s["args"]} for s in self.arm["cublas_handle_setup"]
                if s["function"] in HANDLE_SETUP]
        got = self.native.get("cublas_handle_setup", [])
        if got[:len(want)] != want:
            self.fail("cuBLAS handle setup", f"{got[:len(want)]}, the record {want}")

    def run(self, fragment=None):
        self.check_identity()
        if not self.native.get("cublas_launches_observed"):
            self.lack("an nsys trace of the run (--nsys)")
        # The bridge launches nothing outside its chunks: between them it only
        # copies (loading, state save and restore) and clears the KV cache.
        outside = self.native.get("outside_chunks", {})
        if outside.get("kernels") or outside.get("cublas_calls"):
            self.fail("outside the chunks", f"native launches {outside.get('kernels')} and makes "
                                            f"{outside.get('cublas_calls')} cuBLAS calls; the record none")
        chunks = self.native["chunks"]
        if fragment is not None:
            sequence, start = fragment
            if len(chunks) != 1:
                self.fail("fragment", f"a fragment recording has one chunk, not {len(chunks)}")
            if sequence not in self.sequences:
                self.fail("fragment", f"the arm has no sequence {sequence}")
            ref_tokens = self.sequences[sequence]["tokens"]
            if not 0 <= start <= len(ref_tokens):
                self.fail("fragment", f"sequence {sequence} has {len(ref_tokens)} tokens")
            nat_tokens = chunks[0]["tokens"]
            if not nat_tokens:
                self.fail("fragment", "the fragment is empty")
            self.compare_tokens(ref_tokens[start:start + len(nat_tokens)], nat_tokens,
                                f"{self.arm_name} sequence {sequence} fragment", start)
            if start + len(nat_tokens) > len(ref_tokens):
                self.fail("fragment", "the fragment runs past the sequence's end")
            self.lack("every chunk (this is a fragment)")
            return
        seen = collections.defaultdict(list)
        for c in chunks:
            where = f"{self.arm_name}, evaluation {c['evaluation']}, chunk {c['chunk']}"
            if not 0 <= c["chunk"] < len(self.chunks):
                self.fail(where, f"the record has {len(self.chunks)} chunks")
            ref = self.chunks[c["chunk"]]
            expected_next = len(seen[c["evaluation"]])
            if c["chunk"] != expected_next:
                self.fail(where, f"chunk {c['chunk']} runs where chunk {expected_next} comes next")
            seen[c["evaluation"]].append(c["chunk"])
            if (c["rows"], c["n_past"]) != (ref["rows"], ref["n_past"]):
                self.fail(where, f"{c['rows']} rows at n_past {c['n_past']}; the record has {ref['rows']} at "
                                 f"{ref['n_past']}")
            where += f" ({ref['rows']} rows, n_past {ref['n_past']}; record sequence {ref['sequence']})"
            self.compare_tokens(self.sequences[ref["sequence"]]["tokens"], c["tokens"], where)
        if len(seen.get(1, [])) != len(self.chunks):
            self.lack(f"chunks {len(seen.get(1, []))} to {len(self.chunks) - 1} of evaluation 1")


def compare(record, arm_name, native, fragment=None):
    """(status, message): 0 matched and complete, 1 a difference, 2 matched but incomplete."""
    if native.get("format") != FORMAT:
        return 1, f"not a {FORMAT} plan"
    comparison = Comparison(record, arm_name, native)
    try:
        comparison.run(fragment)
    except Difference as difference:
        return 1, f"DIFFERENT: {difference}"
    scope = f"fragment of sequence {fragment[0]} from token {fragment[1]}" if fragment else "every chunk"
    if comparison.missing:
        return 2, f"INCOMPLETE: {scope} matched, but the recording lacks " + "; ".join(comparison.missing)
    return 0, f"MATCH: {scope} of {arm_name}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("recording")
    c.add_argument("--cublas-log")
    c.add_argument("--cublaslt-log")
    c.add_argument("--sass")
    c.add_argument("--nsys")
    c.add_argument("--library", action="append", default=[], help="NAME=SHA256 of a loaded library")
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--reference", required=True)
    m.add_argument("--arm", required=True)
    m.add_argument("--native", required=True)
    m.add_argument("--fragment", help="SEQUENCE:START, a one-chunk recording against part of a sequence")
    r = sub.add_parser("reference")
    r.add_argument("--reference", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--no-cublas-launches", action="store_true")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "convert":
            with open(a.recording) as f:
                lines = f.readlines()
            sass = open(a.sass).readlines() if a.sass else None
            plan = convert(lines, a.cublas_log, a.cublaslt_log, sass, a.nsys, a.library)
            pathlib.Path(a.out).write_text(json.dumps(plan, indent=1) + "\n")
            return 0
        record = json.loads(pathlib.Path(a.reference).read_text())
        if a.cmd == "reference":
            plan = reference_as_native(record, a.arm, not a.no_cublas_launches)
            pathlib.Path(a.out).write_text(json.dumps(plan, indent=1) + "\n")
            return 0
        fragment = None
        if a.fragment:
            sequence, start = a.fragment.split(":")
            fragment = (int(sequence), int(start))
        status, message = compare(record, a.arm, json.loads(pathlib.Path(a.native).read_text()), fragment)
        print(message)
        return status
    except PlanError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
