# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The EXL3 plan gate's comparator (docs/experiments/backend-proof-p3/op_plan_compare.py), no GPU needed.

A native run is synthesized from the record itself, as benchmarks/exl3_exec.cc --record, an
nsys trace of the run and cuobjdump's SASS hashes would give it: it matches, for both fixtures
and every record. Each departure the gate forbids is caught and located, each gap it cannot
judge leaves it incomplete, and the allowed differences (stream, launch API, copy kinds, NVCC's
per-file hashes) pass.
"""

import contextlib
import copy
import importlib.util
import io
import json
import pathlib
import shutil
import sqlite3
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
P3 = ROOT / "docs/experiments/backend-proof-p3"
RECORDS = {"p0": ROOT / "docs/experiments/backend-proof-p0/exl3-op-plan.json",
           "g": P3 / "exl3-op-plan-g.json", "o": P3 / "exl3-op-plan-o.json"}

spec = importlib.util.spec_from_file_location("op_plan_compare", P3 / "op_plan_compare.py")
opc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opc)

ATTENTION = ("attention.kv_max", "attention", "attention.combine")
KV_BYTES = 24 * 2 * 4096 * 256
COOPERATIVE = ("exl3_gemm_kernel", "exl3_mgemm_kernel", "exl3_gemv_kernel")


def fake_mangled(kernel):
    """A stand-in mangled name for ExLlamaV3's and cuBLAS's kernels; demangle() maps it back."""
    if kernel["owner"] == "cublas" and kernel["name"].startswith("nvjet"):
        return kernel["name"]  # extern "C": its own mangled name
    return f"_Z{kernel['owner']}_{kernel['id']}"


def demangler(record):
    names = {fake_mangled(k): k["name"] for k in record["kernels"] if k["owner"] != "ggml"}
    return lambda mangled: [names.get(n, n) for n in mangled]


def native_ops(record, pk, fixture):
    """The record's operations as native runs and names them, built apart from the tool's own."""
    fx = pk["fixtures"][fixture]
    n, npad = pk["rows"], pk["padded_kv_length"]

    def layer(name, l):
        if name in pk["layer_launches"]:
            return pk["layer_launches"][name]
        return next(g["launches"] for g in fx["linear_launches"][name] if l in g["layers"])

    ops = [("inputs", -1, [["memcpy", "HtoD", n * 4], ["memcpy", "HtoD", n * 4],
                           ["memcpy", "HtoD", n * npad * 2]])]
    ops += [(e["op"], -1, e["launches"]) for e in pk["embedding"]]
    for l in range(record["model"]["layers"]):
        attention = []
        for name in pk["layer_order"]:
            if name in ATTENTION:
                if not attention:
                    attention = [("attention", l, [])]
                    ops.append(attention[0])
                attention[0][2].extend(layer(name, l))
            else:
                ops.append((name, l, layer(name, l)))
    outputs = {e["op"]: e["launches"] for e in pk["output"] + fx["lm_head_launches"]}
    ops += [(name, -1, outputs[name]) for name in pk["output_order"]]
    ops.append(("outputs", -1, [["memcpy", "DtoH", n * record["model"]["vocab"] * 2]]))
    return ops


def synthesize(record, fixture, prefixes, steps):
    """(recording, trace): the recording's lines as dicts (a traced event carries its trace
    entry under "_t", the same object the trace list holds), and the trace's events."""
    kernels = {k["id"]: k for k in record["kernels"]}
    kinds = record["phase_kinds"]
    rec = [{"type": "header", "format": opc.FORMAT, "source": "synthetic", "loaded": {}}]
    # Load time: the weights' uploads, a widening kernel, and a memset the size of the clear
    # that is not the recording's start.
    trace = [{"type": "memcpy", "bytes": 1 << 20}, {"type": "memset", "bytes": KV_BYTES, "value": 0},
             {"type": "kernel", "kernel": "_Z6widenPKtPf", "grid": [4, 1, 1], "block": [256, 1, 1], "dyn": 0,
              "api": "cudaLaunchKernel", "registers": 16, "static": 0}]

    def event(e, t):
        rec.append(e)
        if t is not None:
            trace.append(t)
            e["_t"] = t

    def kind(rows, n_past):
        npad = -(-(n_past + rows) // 256) * 256
        for pk in kinds:
            if n_past == 0 and (pk["rows"], pk["first_position"]) == (rows, 0):
                return pk
            if n_past and rows == pk["rows"] == 1 and pk["first_position"] and pk["padded_kv_length"] == npad:
                return pk
        return None

    for prefix in prefixes:
        event({"type": "memset", "bytes": KV_BYTES, "value": 0, "stream": "0x10"},
              {"type": "memset", "bytes": KV_BYTES, "value": 0})
        for chunk in range(steps + 1):
            rows, n_past = (prefix, 0) if chunk == 0 else (1, prefix + chunk - 1)
            pk = kind(rows, n_past)
            if pk is None:  # a kind the record lacks: another kind's plan stands in for it
                pk = next(p for p in kinds if p["rows"] == rows)
            rec.append({"type": "chunk", "evaluation": 1, "chunk": chunk, "rows": rows, "n_past": n_past})
            for name, layer, launches in native_ops(record, pk, fixture):
                rec.append({"type": "op", "name": name, "layer": layer})
                for launch in launches:
                    if launch[0] == "memcpy":
                        event({"type": "memcpy", "kind": "driver", "bytes": launch[2], "stream": "0x10"},
                              {"type": "memcpy", "bytes": launch[2]})
                        continue
                    k = kernels[launch[0]]
                    static = k.get("static_shared", 0)
                    if k["owner"] == "cublas":
                        event({"type": "cublas", "function": "cublasLtMatmul", "m": 1, "n": 1, "k": 1}, None)
                        trace.append({"type": "kernel", "kernel": fake_mangled(k), "grid": list(launch[1]),
                                      "block": list(launch[2]), "dyn": launch[3], "api": "cuLaunchKernelEx",
                                      "registers": 168, "static": 0})
                        continue
                    ggml = k["owner"] == "ggml"
                    api = ("cudaLaunchKernelExC" if ggml else "cudaLaunchCooperativeKernel"
                           if k["name"].split("<")[0].split()[-1] in COOPERATIVE else "cudaLaunchKernel")
                    name = k["mangled"] if ggml else fake_mangled(k)
                    registers = k["registers"] if ggml else 96
                    event({"type": "kernel", "name": name, "api": api, "grid": list(launch[1]),
                           "block": list(launch[2]), "shared": launch[3] - static, "registers": registers,
                           "static_shared": static, "local": 0, "stream": "0x10"},
                          # nsys reports registers as allocated: rounded up to a multiple of 8.
                          {"type": "kernel", "kernel": name, "grid": list(launch[1]), "block": list(launch[2]),
                           "dyn": launch[3] - static, "api": api, "registers": -(-registers // 8) * 8,
                           "static": static})
            rec.append({"type": "end_chunk"})
    # The second evaluation begins: the trace goes on.
    trace += [{"type": "memset", "bytes": KV_BYTES, "value": 0}, copy.deepcopy(trace[-1])]
    return rec, trace


def sass_lines(record):
    return [json.dumps({"label": "native", "function": k["mangled"], "text_sha256": "t",
                        "encoding_sha256": k["sass_encoding_sha256"], "instructions": 1})
            for k in record["kernels"] if k["owner"] == "ggml"]


def write_trace(trace, path):
    """The tables plan_compare.load_trace reads, as nsys export writes them."""
    db = sqlite3.connect(path)
    db.executescript("""
        create table StringIds (id integer, value text);
        create table CUPTI_ACTIVITY_KIND_RUNTIME (correlationId integer, nameId integer);
        create table TARGET_INFO_CUDA_STREAM (streamId integer, flag integer);
        create table ENUM_CUPTI_STREAM_TYPE (id integer, name text, label text);
        create table CUPTI_ACTIVITY_KIND_KERNEL (correlationId integer, streamId integer, demangledName integer,
            mangledName integer, gridX integer, gridY integer, gridZ integer, blockX integer, blockY integer,
            blockZ integer, dynamicSharedMemory integer, staticSharedMemory integer, registersPerThread integer,
            localMemoryPerThread integer);
        create table CUPTI_ACTIVITY_KIND_MEMCPY (correlationId integer, streamId integer, bytes integer,
            copyKind integer, srcKind integer, dstKind integer);
        create table CUPTI_ACTIVITY_KIND_MEMSET (correlationId integer, streamId integer, bytes integer,
            value integer);
        insert into TARGET_INFO_CUDA_STREAM values (7, 1);
        insert into ENUM_CUPTI_STREAM_TYPE values (1, 'X', 'Non-blocking stream');
    """)
    strings = {}

    def sid(text):
        if text not in strings:
            strings[text] = len(strings) + 1
            db.execute("insert into StringIds values (?, ?)", (strings[text], text))
        return strings[text]

    for cid, t in enumerate(trace, 1):
        if t["type"] == "kernel":
            db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?)", (cid, sid(t["api"] + "_v7000")))
            db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (cid, 7, sid("demangled " + t["kernel"]), sid(t["kernel"]), *t["grid"], *t["block"],
                        t["dyn"], t["static"], t["registers"], 0))
        elif t["type"] == "memcpy":
            db.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?,?,?,?,?,?)", (cid, 7, t["bytes"], 8, 2, 2))
        else:
            db.execute("insert into CUPTI_ACTIVITY_KIND_MEMSET values (?,?,?,?)", (cid, 7, t["bytes"], t["value"]))
    db.commit()
    db.close()


def index_of(items, obj):
    """The index of this very object (list.index compares by value, and launches repeat)."""
    return next(i for i, x in enumerate(items) if x is obj)


def lines_of(rec):
    return [json.dumps({k: v for k, v in e.items() if not k.startswith("_")}) for e in rec]


class Run:
    """A synthesized run, mutated in place, then compared."""

    def __init__(self, record, fixture="4.0bpw", prefixes=(145, 1024), steps=1):
        self.record, self.fixture, self.steps = record, fixture, steps
        self.rec, self.trace = synthesize(record, fixture, prefixes, steps)
        self.sass = sass_lines(record)
        self.demangle = demangler(record)

    def compare(self, nsys=True, lines=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "trace.sqlite"
            write_trace(self.trace, path)
            status, message, _ = opc.compare(self.record, self.fixture, lines or lines_of(self.rec),
                                             str(path) if nsys else None, self.sass, self.demangle, self.steps)
        return status, message

    # -- finding and changing events

    def find(self, name, layer=None, chunk=0, prefix=None, nth=0):
        """The index of the op line `name` (in `layer`) of chunk `chunk` of the prefix's trajectory."""
        current, found = None, 0
        for i, e in enumerate(self.rec):
            if e["type"] == "chunk":
                if e["chunk"] == 0:
                    current = e["rows"]
                in_phase = e["chunk"] == chunk and (prefix is None or current == prefix)
            elif e["type"] == "op" and in_phase and e["name"] == name and (layer is None or e["layer"] == layer):
                if found == nth:
                    return i
                found += 1
        raise LookupError(name)

    def op_events(self, at):
        end = next(i for i in range(at + 1, len(self.rec)) if self.rec[i]["type"] in ("op", "end_chunk"))
        return list(range(at + 1, end))

    def insert(self, at, event, traced=True):
        """Inserts a recorded event before rec[at], and its trace entry where it would fall."""
        self.rec.insert(at, event)
        if traced:
            t = {"type": event["type"], "bytes": event.get("bytes")}
            if event["type"] == "kernel":
                t = {"type": "kernel", "kernel": event["name"], "grid": event["grid"], "block": event["block"],
                     "dyn": event["shared"], "api": event["api"], "registers": -(-event["registers"] // 8) * 8,
                     "static": event["static_shared"]}
            elif event["type"] == "memset":
                t["value"] = event["value"]
            after = next(e["_t"] for e in reversed(self.rec[:at]) if "_t" in e)
            self.trace.insert(index_of(self.trace, after) + 1, t)
            event["_t"] = t

    def remove(self, indices):
        for i in sorted(indices, reverse=True):
            e = self.rec.pop(i)
            if "_t" in e:
                del self.trace[index_of(self.trace, e["_t"])]


class Faithful(unittest.TestCase):
    def test_every_record_and_fixture_matches(self):
        # Every phase kind of each record: the steps of prefix 145 are the Npad-256 kind, of 1,024
        # the Npad-1,280 kind; prefix 1023's step (Npad 1,024) only where the record has that kind.
        for name, path in RECORDS.items():
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            kinds = {pk["phase_kind"] for pk in record["phase_kinds"]}
            for fixture in ("4.0bpw", "4.5bpw"):
                with self.subTest(record=name, fixture=fixture):
                    prefixes = (32, 144, 145, 1024)
                    run = Run(record, fixture, prefixes, steps=2)
                    status, message = run.compare()
                    self.assertEqual((status, message), (0, f"MATCH: 12 phases of 6 kinds ({fixture}) equal the "
                                                            "record"))
                    run = Run(record, fixture, (1023,), steps=1 if "step 1023" in kinds else 0)
                    self.assertEqual(run.compare()[0], 0)

    def test_the_summary_and_command_line(self):
        record = json.loads(RECORDS["p0"].read_text())
        run = Run(record, prefixes=(32,), steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            write_trace(run.trace, tmp / "trace.sqlite")
            (tmp / "record.jsonl").write_text("\n".join(lines_of(run.rec)) + "\n")
            (tmp / "sass.jsonl").write_text("\n".join(run.sass) + "\n")
            status, message, summary = opc.compare(record, "4.0bpw", lines_of(run.rec), str(tmp / "trace.sqlite"),
                                                   run.sass, run.demangle, 1)
            self.assertEqual(status, 0, message)
            self.assertEqual(summary[0], "prefill 32: 1 phases, 630 operations, 656 launches")
            # c++filt cannot name the synthetic ExLlamaV3 kernels: the command line differs there.
            args = [str(tmp / "record.jsonl"), "--record", str(RECORDS["p0"]), "--fixture", "4.0bpw", "--nsys",
                    str(tmp / "trace.sqlite"), "--sass", str(tmp / "sass.jsonl"), "--steps", "1"]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(opc.main(args + ["--demangler", str(tmp / "no-such-demangler")]), 2)
                if shutil.which("c++filt"):
                    self.assertEqual(opc.main(args), 1)
            self.assertIn("no demangler", out.getvalue())

    def test_the_normalizers_agree_on_the_record(self):
        # The record's mangled_normalized masks the first hash of NVCC's _INTERNAL_ token; P2's
        # normalizer (the approved "whole token") masks both. They mask the same token, and no two
        # of the record's GGML kernels become one.
        for path in RECORDS.values():
            if not path.exists():
                continue
            ggml = [k for k in json.loads(path.read_text())["kernels"] if k["owner"] == "ggml"]
            for k in ggml:
                ours = opc.normalized(k["mangled"])
                theirs = k["mangled_normalized"].replace("_INTERNAL_x_", "_INTERNAL_xxxxxxxx_")
                self.assertEqual(opc.normalized(theirs), ours, k["mangled"])
                self.assertEqual(ours == k["mangled"], "_INTERNAL_" not in k["mangled"])
            self.assertEqual(len({opc.normalized(k["mangled"]) for k in ggml}), len(ggml))


class Departures(unittest.TestCase):
    """Each departure the gate forbids, made once, fails at its place."""
    record = json.loads(RECORDS["p0"].read_text())

    def setUp(self):
        self.run = Run(self.record)

    def check(self, *fragments, status=1, nsys=True, lines=None):
        got, message = self.run.compare(nsys, lines)
        self.assertEqual(got, status, message)
        for fragment in fragments:
            self.assertIn(fragment, message)
        return message

    def kernel_in(self, op, layer, chunk=0, prefix=None, nth=0):
        at = self.run.find(op, layer, chunk, prefix)
        return [i for i in self.run.op_events(at) if self.run.rec[i]["type"] == "kernel"][nth]

    def test_the_synthesis_matches(self):
        self.check("MATCH", status=0)

    def test_a_changed_grid(self):
        e = self.run.rec[self.kernel_in("rope_q", 5)]
        e["grid"][0] += 1
        e["_t"]["grid"][0] += 1  # the trace saw it too
        self.check("prefix 145, chunk 0 (145 rows, n_past 0) [kind 'prefill 145'], layer 5 rope_q, launch 0",
                   "grid [2031, 1, 1], the record [2030, 1, 1]")

    def test_the_trace_must_agree_with_the_recording(self):
        e = self.run.rec[self.kernel_in("rope_q", 5)]
        e["_t"]["grid"] = [1, 1, 1]
        self.check("layer 5 rope_q (recording line", "the trace launched grid [1, 1, 1]")
        self.setUp()
        name = self.run.rec[self.kernel_in("attention", 0)]["name"]
        for t in self.run.trace:
            if t.get("kernel") == name:
                t["static"] += 16
        self.check("layer 0 attention (recording line", "4624 bytes of static shared memory; the recording 4608")

    def test_a_launch_the_recorder_missed(self):
        at = self.kernel_in("swiglu", 2)
        t = copy.deepcopy(self.run.rec[at]["_t"])
        self.run.trace.insert(index_of(self.run.trace, self.run.rec[at]["_t"]), t)
        self.check("layer 2 swiglu", "the recorder did not see")

    def test_an_extra_kernel(self):
        at = self.kernel_in("swiglu", 2, chunk=1, prefix=1024)
        extra = copy.deepcopy({k: v for k, v in self.run.rec[at].items() if k != "_t"})
        self.run.insert(at + 1, extra)
        self.check("prefix 1024, chunk 1 (1 rows, n_past 1024) [kind 'step 1024'], layer 2 swiglu, launch 1",
                   "native continues with kernel _Z21unary_gated_op_kernel")

    def test_a_missing_op(self):
        at = self.run.find("rope_k.cast_out", 7)
        self.run.remove([at] + self.run.op_events(at))
        self.check("operation", "is layer 7 kv_write.k where the record has layer 7 rope_k.cast_out")

    def test_ops_out_of_order(self):
        # rope_q before its cast, in the recording and the trace alike.
        first = self.run.find("rope_q.cast", 3)
        second = self.run.op_events(first)[-1] + 1
        end = self.run.op_events(second)[-1] + 1
        block_a, block_b = self.run.rec[first:second], self.run.rec[second:end]
        self.run.rec[first:end] = block_b + block_a
        start = index_of(self.run.trace, block_a[1]["_t"])
        self.run.trace[start:start + 2] = [block_b[1]["_t"], block_a[1]["_t"]]
        self.check("is layer 3 rope_q where the record has layer 3 rope_q.cast")

    def test_a_wrong_sass_hash(self):
        index = next(i for i, line in enumerate(self.run.sass) if "rms_norm_f32" in line)
        entry = json.loads(self.run.sass[index])
        entry["encoding_sha256"] = "0" * 64
        self.run.sass[index] = json.dumps(entry)
        self.check("[kind 'prefill 145'], layer 0 attn_norm, launch 0", "SASS encoding hash")

    def test_a_wrong_exllamav3_kernel(self):
        # The fused reconstruction's kernel for 5-bit weights where the record has 4-bit.
        e = self.run.rec[self.kernel_in("q_proj", 0, prefix=1024)]
        e["name"] = e["_t"]["kernel"] = "_Zexllamav3_43"
        self.check("prefix 1024, chunk 0", "layer 0 q_proj, launch 0",
                   "void reconstruct_had_kernel<5, 1, false>", "where the record launches void "
                   "reconstruct_had_kernel<4, 1, false>")

    def test_a_wrong_cublas_kernel(self):
        at = self.run.find("o_proj", 4)
        call = next(i for i in self.run.op_events(at) if self.run.rec[i]["type"] == "cublas")
        # The cuBLAS kernel follows the kernel before the call in the trace.
        before = self.run.rec[call - 1]["_t"]
        t = self.run.trace[index_of(self.run.trace, before) + 1]
        self.assertTrue(t["kernel"].startswith("nvjet") or t["kernel"].startswith("_Zcublas"))
        t["kernel"] = "nvjet_sm121_hsh_mma_64x48x64_7_16x48x64_tmaAB_alignCD4_bz_NNNN"
        self.check("layer 4 o_proj, launch 2", "where the record launches")

    def test_the_attention_grouping(self):
        # Native records the combine as an operation of its own.
        at = self.run.find("attention", 6)
        combine = [i for i in self.run.op_events(at) if "combine" in self.run.rec[i].get("name", "")][0]
        self.run.rec.insert(combine, {"type": "op", "name": "attention.combine", "layer": 6})
        self.check("layer 6 attention, launch 1", "native's operation ends")

    def test_the_attention_order(self):
        # From 1,024 rows the mask pre-pass comes first.
        at = self.run.find("attention", 6, prefix=1024)
        first, second = self.run.op_events(at)[:2]
        a, b = self.run.rec[first], self.run.rec[second]
        self.run.rec[first], self.run.rec[second] = b, a
        i, j = index_of(self.run.trace, a["_t"]), index_of(self.run.trace, b["_t"])
        self.run.trace[i], self.run.trace[j] = b["_t"], a["_t"]
        self.check("prefix 1024, chunk 0", "layer 6 attention, launch 0",
                   "_Z18flash_attn_ext_vec", "where the record launches _Z25flash_attn_mask_to_KV_max")

    def test_an_op_named_as_the_record_splits_it(self):
        at = self.run.find("attention", 2, prefix=1024)
        self.run.rec[at]["name"] = "attention.kv_max"
        self.check("layer 2 attention.kv_max where the record has layer 2 attention",
                   "native's attention op covers")

    def test_a_kernel_between_phases(self):
        at = next(i for i, e in enumerate(self.run.rec) if e["type"] == "end_chunk") + 1
        kernel = {k: v for k, v in self.run.rec[self.kernel_in("embed", -1)].items() if k != "_t"}
        self.run.insert(at, copy.deepcopy(kernel))
        self.check("between phases", "only the cache's zero-fill")

    def test_a_nonzero_fill_between_phases(self):
        self.run.rec[1]["value"] = self.run.rec[1]["_t"]["value"] = 1
        self.check("between phases", "memset of 1")

    def test_copies_of_the_wrong_size(self):
        for op, layer, index, fragment in (("kv_write.k", 3, 0, f"copies {145 * 256} bytes"),
                                           ("inputs", -1, 2, f"copies {145 * 256 * 2} bytes"),
                                           ("outputs", -1, 0, f"copies {145 * 151936 * 2} bytes")):
            with self.subTest(op=op):
                self.setUp()
                e = self.run.rec[self.run.op_events(self.run.find(op, layer))[index]]
                e["bytes"] = e["_t"]["bytes"] = e["bytes"] // 2
                self.check(op, fragment)

    def test_a_launch_attribute_other_than_pdl(self):
        e = self.run.rec[self.kernel_in("mlp_norm", 9)]
        e["api"] = "cudaLaunchKernelExC+attributes"
        self.check("layer 9 mlp_norm, launch 0", "only the launch API and the PDL attribute may differ")

    def test_changed_registers(self):
        # A build that allocates two more: every launch of the kernel, recorded and traced.
        name = self.run.rec[self.kernel_in("attn_residual_add", 1)]["name"]
        for e in self.run.rec:
            if e.get("name") == name:
                e["registers"] = e["_t"]["registers"] = 22
        self.check("layer 0 attn_residual_add", "registers 22, the record 20")

    def test_a_phase_out_of_place(self):
        at = next(i for i, e in enumerate(self.run.rec) if e["type"] == "chunk" and e["chunk"] == 1)
        self.run.rec[at]["n_past"] += 1
        self.check("prefix 145, chunk 1 (1 rows, n_past 146)", "out of place")

    # -- incomplete

    def test_without_the_trace(self):
        message = self.check("an nsys trace of the run", status=2, nsys=False)
        self.assertTrue(message.startswith("INCOMPLETE: 4 phases of 4 kinds"), message)
        # The cuBLAS calls still stand only where the record has cuBLAS kernels.
        at = self.kernel_in("q_proj.bias_add", 0)
        self.run.rec[at] = {"type": "cublas", "function": "cublasLtMatmul", "m": 1, "n": 1, "k": 1}
        self.check("layer 0 q_proj.bias_add, launch 0", "native calls cublasLtMatmul where the record has "
                   "kernel 4 add_kernel_hhh", nsys=False)

    def test_a_phase_kind_the_record_lacks(self):
        # Prefix 1,023's first step pads K to 1,024: not a kind of the P0 record.
        self.run = Run(self.record, prefixes=(1023,), steps=1)
        self.check("the phase kind of prefix 1023, chunk 1 (1 rows, n_past 1023)", "K padded to 1024", status=2)

    def test_a_truncated_recording(self):
        lines = lines_of(self.run.rec)
        at = self.run.find("mlp_norm", 11, prefix=1024)
        self.check("it ends inside prefix 1024, chunk 0", status=2, lines=lines[:at + 1])
        self.check("its last line", status=2, lines=lines[:at + 1] + [lines[at + 1][:20]])
        # A cut recording still differs where it differs.
        self.run.rec[self.kernel_in("attn_norm", 0, prefix=1024)]["grid"][0] += 1
        self.check("layer 0 attn_norm", status=1, lines=lines_of(self.run.rec)[:at + 1])

    def test_a_fragment_of_the_trajectories(self):
        self.run.steps = 16
        self.check("the steps of prefix 145: 1 of 16", status=2)

    def test_missing_sass(self):
        self.run.sass = [line for line in self.run.sass if "rope_neox" not in line]
        self.check("the SASS of _Z9rope_neox", status=2)
        self.run.sass = None
        self.check("SASS hashes of the native binary", status=2)

    def test_without_a_demangler(self):
        self.run.demangle = lambda names: None
        self.check("no demangler", status=2)

    # -- allowed

    def test_allowed_differences_pass(self):
        for e in self.run.rec:
            if "stream" in e:
                e["stream"] = "0x7f00"
            if e["type"] == "memcpy":
                e["kind"] = "HtoD" if e["kind"] == "driver" else "driver"
            if e["type"] == "kernel" and e["api"] == "cudaLaunchKernelExC":
                e["api"] = "cudaLaunchKernel"
            if e["type"] == "kernel" and "_INTERNAL_" in e["name"]:
                renamed = e["name"].replace("_INTERNAL_b727e39e_11_binbcast_cu_6840010b",
                                            "_INTERNAL_01234567_11_binbcast_cu_89abcdef")
                renamed = renamed.replace("_INTERNAL_11a4477a_8_unary_cu_eb6d5366",
                                          "_INTERNAL_76543210_8_unary_cu_fedcba98")
                self.assertNotEqual(renamed, e["name"])
                e["name"] = e["_t"]["kernel"] = renamed
        self.run.sass = [line.replace("b727e39e", "01234567").replace("6840010b", "89abcdef") for line in self.run.sass]
        self.check("MATCH", status=0)

    def test_malformed_recordings_are_refused(self):
        header = json.dumps({"type": "header", "format": opc.FORMAT})
        chunk = json.dumps({"type": "chunk", "evaluation": 1, "chunk": 0, "rows": 1, "n_past": 0})
        for lines, error in (([], "empty"), (["{}"], "header"), ([header, chunk, chunk], "inside another"),
                             ([header, '{"type": "end_chunk"}'], "never began"),
                             ([header, '{"type": "op", "name": "embed", "layer": -1}'], "outside a chunk"),
                             ([header, '{"type": "launch"}'], "unknown"), ([header, "{", chunk], "not JSON"),
                             ([header, chunk.replace('"evaluation": 1', '"evaluation": 2')], "first is recorded")):
            with self.subTest(error=error), self.assertRaisesRegex(opc.PlanError, error):
                opc.compare(self.record, "4.0bpw", lines)


if __name__ == "__main__":
    unittest.main()
