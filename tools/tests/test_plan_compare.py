# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The FP16 gate's plan comparator (docs/experiments/backend-proof-p2/plan_compare.py), no GPU needed.

The bridge's own record, written as a native plan, matches itself in every
arm; each kind of departure the gate forbids is caught and located; the
allowed ones (stream, launch API, copy kind, NVCC's per-file hashes) pass;
and a full recording, synthesized from the record in the launch recorder's
format with cuBLAS's logs, SASS hashes and an nsys trace, converts and
matches, so the converter's inputs are exercised end to end.
"""

import copy
import importlib.util
import json
import pathlib
import sqlite3
import sys
import tempfile
import unittest
import warnings

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
P2 = ROOT / "docs/experiments/backend-proof-p2"
RECORD = ROOT / "docs/experiments/backend-proof-p0/fp16-plan.json"
SAMPLE = ROOT / "tests/support/plan_record_sample.txt"

spec = importlib.util.spec_from_file_location("plan_compare", P2 / "plan_compare.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

ARMS = ("control-fused", "control-unfused", "heldout-fused", "heldout-unfused")


def token_index(plan, chunk, kind, start=0, where=lambda t: True):
    tokens = plan["chunks"][chunk]["tokens"]
    return next(i for i in range(start, len(tokens)) if tokens[i][0] == kind and where(tokens[i]))


class ReferenceMatchesItself(unittest.TestCase):
    record = json.loads(RECORD.read_text())

    def test_every_arm_matches_itself(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                status, message = pc.compare(self.record, arm, pc.reference_as_native(self.record, arm))
                self.assertEqual((status, message), (0, f"MATCH: every chunk of {arm}"))

    def test_another_arm_differs(self):
        native = pc.reference_as_native(self.record, "control-unfused")
        status, message = pc.compare(self.record, "control-fused", native)
        self.assertEqual(status, 1)
        self.assertIn("chunk 0 (32 rows, n_past 0; record sequence 0), token 6", message)
        native = pc.reference_as_native(self.record, "control-fused")
        status, message = pc.compare(self.record, "heldout-fused", native)
        self.assertEqual(status, 1)
        self.assertIn("32 rows at n_past 0; the record has 16", message)

    def test_unobserved_cublas_launches_leave_the_gate_incomplete(self):
        native = pc.reference_as_native(self.record, "heldout-fused", cublas_launches=False)
        status, message = pc.compare(self.record, "heldout-fused", native)
        self.assertEqual(status, 2)
        self.assertIn("the kernels cuBLAS launched", message)

    def test_a_run_without_its_trace_is_incomplete(self):
        # The trace is also what shows a launch or copy the recorder missed.
        native = pc.reference_as_native(self.record, "heldout-fused")
        native["cublas_launches_observed"] = False
        status, message = pc.compare(self.record, "heldout-fused", native)
        self.assertEqual(status, 2, message)
        self.assertIn("an nsys trace of the run", message)

    def test_a_fragment_is_never_a_complete_match(self):
        native = pc.reference_as_native(self.record, "heldout-fused")
        native["chunks"] = [native["chunks"][2]]  # a single-token step, sequence 2
        status, message = pc.compare(self.record, "heldout-fused", native, (2, 0))
        self.assertEqual(status, 2, message)
        self.assertEqual(message, "INCOMPLETE: fragment of sequence 2 from token 0 matched, but the recording "
                                  "lacks every chunk (this is a fragment)")


class MutationsAreCaught(unittest.TestCase):
    """Each departure the FP16 gate forbids, made once, fails at its place."""
    record = json.loads(RECORD.read_text())

    def setUp(self):
        self.arm = "heldout-fused"
        self.native = pc.reference_as_native(self.record, self.arm)

    def check(self, *fragments, status=1):
        got, message = pc.compare(self.record, self.arm, self.native)
        self.assertEqual(got, status, message)
        for fragment in fragments:
            self.assertIn(fragment, message)
        return message

    def ggml_kernel(self, chunk=1, start=6):
        kernels = {k["id"]: k for k in self.native["kernels"]}
        return token_index(self.native, chunk, "kernel", start, lambda t: kernels[t[1]]["owner"] == "ggml")

    def test_a_changed_grid(self):
        i = self.ggml_kernel()
        self.native["chunks"][1]["tokens"][i][2][0] += 1
        self.check("evaluation 1, chunk 1 (17 rows, n_past 16; record sequence 1)", f"token {i}", "grid")

    def test_changed_shared_memory(self):
        i = self.ggml_kernel()
        self.native["chunks"][1]["tokens"][i][4] += 16
        self.check(f"token {i}", "dynamic shared memory")

    def test_a_changed_block(self):
        i = self.ggml_kernel(chunk=18)
        self.native["chunks"][18]["tokens"][i][3][0] *= 2
        self.check("chunk 18 (512 rows, n_past 49; record sequence 3)", "block")

    def test_a_dropped_launch(self):
        i = self.ggml_kernel(chunk=2)
        del self.native["chunks"][2]["tokens"][i]
        self.check(f"chunk 2 (1 rows, n_past 33; record sequence 2), token {i}", "dropped")

    def test_an_extra_launch(self):
        tokens = self.native["chunks"][2]["tokens"]
        tokens.insert(len(tokens) - 1, copy.deepcopy(tokens[self.ggml_kernel(chunk=2)]))
        self.check(f"token {len(tokens) - 2}")

    def test_a_swapped_kernel(self):
        # get_rows' vector variant where the record has the scalar one: the
        # alignment choice.
        by_name = {k["mangled"]: k["id"] for k in self.native["kernels"]}
        kernels = {k["id"]: k for k in self.native["kernels"]}
        scalar = next(i for n, i in by_name.items() if n.startswith("_Z16k_get_rows_float"))
        vector = next(i for n, i in by_name.items() if n.startswith("_Z20k_get_rows_float_vec"))
        i = token_index(self.native, 1, "kernel", 0, lambda t: t[1] == scalar)
        self.native["chunks"][1]["tokens"][i][1] = vector
        message = self.check(f"token {i}", "where the record has kernel 6")
        self.assertIn(kernels[vector]["mangled"][:40], message)

    def test_reordered_launches(self):
        tokens = self.native["chunks"][0]["tokens"]
        i = self.ggml_kernel(chunk=0)
        j = next(k for k in range(i + 1, len(tokens)) if tokens[k] != tokens[i])
        tokens[i], tokens[j] = tokens[j], tokens[i]
        self.check(f"token {i}", "reordered")

    def test_a_changed_cublas_algorithm(self):
        call = self.native["cublas_calls"][0]
        call["cublaslt"] = dict(call["cublaslt"], algo=call["cublaslt"]["algo"].replace("customOption=", "customOption=9"))
        self.check("cuBLASLt algo")

    def test_a_changed_cublas_heuristic_preference(self):
        call = self.native["cublas_calls"][1]
        call["cublaslt"] = dict(call["cublaslt"], preference=call["cublaslt"]["preference"].replace(
            "minBytesAlignmentA=16", "minBytesAlignmentA=8"))
        self.check("cuBLASLt preference")

    def test_a_changed_cublas_call(self):
        call = self.native["cublas_calls"][2]
        call["args"] = dict(call["args"], k=call["args"]["k"] + 1)
        self.check("k is")

    def test_another_cublas_entry_point(self):
        call = self.native["cublas_calls"][0]
        call["function"] = "cublasGemmStridedBatchedEx"
        self.check("cuBLAS cublasGemmStridedBatchedEx where the record calls cublasGemmEx")

    def test_a_changed_math_mode(self):
        self.native["cublas_calls"][0]["math_mode"] = "CUBLAS_DEFAULT_MATH"
        self.check("math mode")

    def test_a_changed_cublas_launch(self):
        call = next(c for c in self.native["cublas_calls"] if len(c["launches"]) > 1)
        call["launches"][1][2][0] += 1
        self.check("launch 1")

    def test_changed_registers(self):
        kernel = next(k for k in self.native["kernels"] if k["owner"] == "ggml")
        kernel["registers"] += 1
        self.check("registers")

    def test_a_changed_sass_hash(self):
        kernel = next(k for k in self.native["kernels"] if k["owner"] == "ggml")
        kernel["sass_encoding_sha256"] = "0" * 64
        self.check("SASS encoding hash")

    def test_embedding_lookup_on_the_device(self):
        # Token ids uploaded instead of looked-up rows: the first input copy.
        tokens = self.native["chunks"][0]["tokens"]
        self.assertEqual(tokens[0][2], 16 * 3584)
        tokens[0][2] = 16 * 4
        self.check("token 0", "a 64-byte copy where the record copies 57344 bytes")

    def test_changed_rows_or_position(self):
        self.native["chunks"][2]["n_past"] += 1
        self.check("chunk 2: 1 rows at n_past 34")

    def test_a_skipped_chunk(self):
        del self.native["chunks"][3]
        self.check("chunk 4 runs where chunk 3 comes next")

    def test_a_missing_tail_is_incomplete(self):
        self.native["chunks"] = [c for c in self.native["chunks"] if c["evaluation"] == 1][:-1]
        self.check("chunks 34 to 34 of evaluation 1", status=2)

    def test_the_handle_workspace(self):
        setup = self.native["cublas_handle_setup"]
        workspace = next(s for s in setup if s["function"] == "cublasSetWorkspace_v2")
        workspace["args"] = {"workspaceSizeInBytes": 4 << 20}
        self.check("cuBLAS handle setup")

    def test_another_cublas_library(self):
        self.native["libraries"]["libcublasLt.so.13.8.0.4"] = "f" * 64
        self.check("libcublasLt.so.13.8.0.4 has SHA-256")

    def test_a_launch_attribute_other_than_pdl(self):
        kernel = next(k for k in self.native["kernels"] if k["owner"] == "ggml")
        kernel["launch_api"] = ["cudaLaunchKernelExC+attributes"]
        self.check("only the PDL attribute may differ")

    def test_a_launch_between_chunks(self):
        self.native["outside_chunks"]["kernels"] = ["_Z6widen"]
        self.check("outside the chunks")

    def test_allowed_differences_pass(self):
        for chunk in self.native["chunks"]:
            for token in chunk["tokens"]:
                if token[0] in ("kernel", "memcpy", "memset"):
                    token[-1] = "another stream"
                if token[0] == "memcpy":
                    token[1] = "driver"
        for kernel in self.native["kernels"]:
            kernel["launch_api"] = ["cudaLaunchKernel"]
            kernel["mangled"] = kernel["mangled"].replace("_INTERNAL_d5c41c42_", "_INTERNAL_01234567_")
        self.check("MATCH", status=0)


def synthesize(record, arm, directory):
    """The arm as the launch recorder, cuBLAS's logs, cuobjdump and nsys would give it."""
    native = pc.reference_as_native(record, arm)
    kernels = {k["id"]: k for k in native["kernels"]}
    calls = {c["id"]: c for c in native["cublas_calls"]}
    lines = [json.dumps({"type": "header", "format": pc.FORMAT, "source": "synthetic", "loaded": {}})]
    legacy, lt, traced = [], [], []

    def trace(kind, **fields):
        traced.append(dict(fields, kind=kind, cid=len(traced) + 1))

    for chunk in native["chunks"]:
        lines.append(json.dumps({"type": "chunk", **{k: chunk[k] for k in ("evaluation", "chunk", "rows", "n_past")}}))
        for t in chunk["tokens"]:
            if t[0] == "kernel":
                k = kernels[t[1]]
                lines.append(json.dumps({
                    "type": "kernel", "name": k["mangled"], "api": "cudaLaunchKernelExC", "grid": t[2],
                    "block": t[3], "shared": t[4], "registers": k["registers"],
                    "static_shared": k["static_shared"], "local": k["local_per_thread"], "stream": "0x10"}))
                trace("kernel", name=k["mangled"], grid=t[2], block=t[3], dyn=t[4], api="cudaLaunchKernelExC")
            elif t[0] == "memcpy":
                lines.append(json.dumps({"type": "memcpy", "kind": "driver", "bytes": t[2], "stream": "0x10"}))
                trace("memcpy", bytes=t[2])
            else:
                c = calls[t[1]]
                lines.append(json.dumps({"type": "cublas", "function": c["function"], "m": c["args"]["m"],
                                         "n": c["args"]["n"], "k": c["args"]["k"]}))
                legacy.append(c)
                lt.append(c["cublaslt"])
                for launch in c["launches"]:
                    if launch[0] == "kernel":
                        name = kernels[launch[1]]["mangled"]
                        trace("kernel", name=name, grid=launch[2], block=launch[3], dyn=launch[4],
                              api="cuLaunchKernelEx" if name.startswith("nvjet") else "cuLaunchKernel")
                    else:
                        trace("memset", bytes=launch[1], value=launch[2])
        lines.append(json.dumps({"type": "end_chunk"}))

    setup = ["I! cuBLAS (v13.8) function cublasStatus_t cublasCreate_v2(cublasContext**) called:\n"]
    for s in native["cublas_handle_setup"][1:]:
        setup.append(f"I! cuBLAS (v13.8) function cublasStatus_t {s['function']}(...) called:\n")
        setup += [f"i!  {key}: type=x; val={value}\n" for key, value in s["args"].items()]
    log = list(setup)
    for c in legacy:
        log.append(f"I! cuBLAS (v13.8) function cublasStatus_t {c['function']}(...) called:\n")
        log += [f"i!  {key}: type=x; val={value}\n" for key, value in c["args"].items()]
        log.append(f"i!Process=1; Thread=1; GPU=0; Handle=0x1; StreamId=0x2 (non-blocking); "
                   f"MathMode={c['math_mode']}\n")
    ltlog = []
    for r in lt:
        ltlog.append(f"[t][cublasLt][1][Api][cublasLtMatmulAlgoGetHeuristicForStream] preference=[{r['preference']}]\n")
        ltlog.append(f"[t][cublasLt][1][Info][cublasLtMatmulAlgoGetHeuristicForStream] {r['heuristic']}\n")
        ltlog.append(f"[t][cublasLt][1][Trace][{r['function']}] A=0x1 Adesc=[{r['Adesc']}] Bdesc=[{r['Bdesc']}] "
                     f"Cdesc=[{r['Cdesc']}] Ddesc=[{r['Ddesc']}] computeDesc=[{r['computeDesc']}] "
                     f"algo=[{r['algo']}] workSpaceSizeInBytes={r['workspace_bytes']} beta={r['beta']} "
                     "outOfPlace=0 stream=0x2\n")
    sass = [json.dumps({"label": "executable", "function": k["mangled"], "text_sha256": "t",
                        "encoding_sha256": k["sass_encoding_sha256"], "instructions": 1})
            for k in native["kernels"] if k["owner"] == "ggml"]

    db = sqlite3.connect(directory / "trace.sqlite")
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

    by_name = {k["mangled"]: k for k in native["kernels"]}
    for e in traced:
        if e["kind"] == "kernel":
            db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?)", (e["cid"], sid(e["api"] + "_v7000")))
            k = by_name[e["name"]]
            db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (e["cid"], 7, sid(e["name"]), sid(e["name"]), *e["grid"], *e["block"], e["dyn"],
                        k["static_shared"], k["registers"], k["local_per_thread"]))
        elif e["kind"] == "memcpy":
            db.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?,?,?,?,?,?)", (e["cid"], 7, e["bytes"], 8, 2, 2))
        else:
            db.execute("insert into CUPTI_ACTIVITY_KIND_MEMSET values (?,?,?,?)", (e["cid"], 7, e["bytes"], e["value"]))
    db.commit()
    db.close()
    (directory / "cublas.log").write_text("".join(log))
    (directory / "cublaslt.log").write_text("".join(ltlog))
    return [line + "\n" for line in lines], sass


class RecordingsConvert(unittest.TestCase):
    record = json.loads(RECORD.read_text())

    def setUp(self):
        # fp16_plan.py's log parsers (frozen P0 evidence) leave their files to the collector.
        warnings.filterwarnings("ignore", category=ResourceWarning)

    def historical_sample(self):
        # The writer's sample follows the current GGML pin; P0 stays frozen.
        # Restore only RMSNorm's old signature for the historical fragment tests.
        current = "_Z12rms_norm_f32ILi256ELb1ELb0ELb0EEvPKfPfilllfS1_lll5uint3S3_S3_S3_S1_lllS3_S3_S3_S3_f"
        historical = "_Z12rms_norm_f32ILi256ELb1ELb0EEvPKfPfilllfS1_lll5uint3S3_S3_S3_S1_lllS3_S3_S3_S3_"
        sample = SAMPLE.read_text()
        self.assertEqual(sample.count(current), 1)
        return sample.replace(current, historical).splitlines()

    def test_current_sample_differs_from_the_historical_rmsnorm_signature(self):
        plan = pc.convert(SAMPLE.read_text().splitlines())
        status, message = pc.compare(self.record, "control-fused", plan, (1, 6))
        self.assertEqual(status, 1, message)
        self.assertIn("token 6", message)
        self.assertIn("rms_norm_f32", message)

    def test_the_historical_sample_matches_the_decode_step(self):
        lines = self.historical_sample()
        plan = pc.convert(lines)
        status, message = pc.compare(self.record, "control-fused", plan, (1, 6))
        self.assertEqual(status, 2, message)
        self.assertIn("fragment of sequence 1 from token 6 matched", message)
        for missing in ("SASS hashes", "cuBLAS and cuBLASLt logs", "libcublas.so.13.8.0.4"):
            self.assertIn(missing, message)
        status, message = pc.compare(self.record, "control-fused", plan, (1, 7))
        self.assertEqual(status, 1, message)
        lines = [line.replace('"grid":[14,1,1]', '"grid":[15,1,1]') for line in lines]
        status, message = pc.compare(self.record, "control-fused", pc.convert(lines), (1, 6))
        self.assertEqual(status, 1)
        self.assertIn("token 8", message)

    def test_a_full_recording_converts_and_matches(self):
        for arm in ("heldout-fused",):  # every phase kind, and both cuBLAS row counts
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                directory = pathlib.Path(tmp)
                lines, sass = synthesize(self.record, arm, directory)
                libraries = [f"{n}={self.record['libraries'][n]}" for n in pc.PLAN_LIBRARIES]
                plan = pc.convert(lines, str(directory / "cublas.log"), str(directory / "cublaslt.log"), sass,
                                  str(directory / "trace.sqlite"), libraries)
                self.assertEqual(pc.compare(self.record, arm, plan), (0, f"MATCH: every chunk of {arm}"))

                # Without the trace the gate is incomplete; without the logs, too.
                plan = pc.convert(lines, str(directory / "cublas.log"), str(directory / "cublaslt.log"), sass,
                                  None, libraries)
                self.assertEqual(pc.compare(self.record, arm, plan)[0], 2)
                plan = pc.convert(lines, None, None, sass, str(directory / "trace.sqlite"), libraries)
                status, message = pc.compare(self.record, arm, plan)
                self.assertEqual(status, 2, message)
                self.assertIn("logs", message)

    def test_logs_and_trace_must_agree_with_the_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            lines, sass = synthesize(self.record, "control-fused", directory)
            cublas_line = next(i for i, line in enumerate(lines) if '"type": "cublas"' in line)
            with self.assertRaisesRegex(pc.PlanError, "recorded cuBLAS calls"):
                pc.convert(lines[:cublas_line] + lines[cublas_line + 1:], str(directory / "cublas.log"),
                           str(directory / "cublaslt.log"))
            kernel_line = next(i for i, line in enumerate(lines) if '"type": "kernel"' in line)
            bad = json.loads(lines[kernel_line])
            bad["grid"] = [9, 9, 9]
            with self.assertRaisesRegex(pc.PlanError, "the trace launched"):
                pc.convert(lines[:kernel_line] + [json.dumps(bad) + "\n"] + lines[kernel_line + 1:],
                           nsys=str(directory / "trace.sqlite"))
            with self.assertRaisesRegex(pc.PlanError, "after the recording's last"):
                pc.convert(lines[:-2] + [lines[-1]], nsys=str(directory / "trace.sqlite"))
            # A kernel recorded between chunks is converted, then refused.
            plan = pc.convert(lines[:1] + [lines[kernel_line]] + lines[1:])
            status, message = pc.compare(self.record, "control-fused", plan)
            self.assertEqual(status, 1)
            self.assertIn("outside the chunks", message)

    def test_a_trace_without_copies_or_memsets(self):
        # nsys leaves out an activity table with no rows; the sample launches
        # kernels only.
        lines = self.historical_sample()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "trace.sqlite"
            db = sqlite3.connect(path)
            db.executescript("""
                create table StringIds (id integer, value text);
                create table CUPTI_ACTIVITY_KIND_RUNTIME (correlationId integer, nameId integer);
                create table TARGET_INFO_CUDA_STREAM (streamId integer, flag integer);
                create table ENUM_CUPTI_STREAM_TYPE (id integer, name text, label text);
                create table CUPTI_ACTIVITY_KIND_KERNEL (correlationId integer, streamId integer,
                    demangledName integer, mangledName integer, gridX integer, gridY integer, gridZ integer,
                    blockX integer, blockY integer, blockZ integer, dynamicSharedMemory integer,
                    staticSharedMemory integer, registersPerThread integer, localMemoryPerThread integer);
                insert into StringIds values (1, 'cudaLaunchKernelExC_v11060');
            """)
            launches = [json.loads(line) for line in lines if '"kernel"' in line]
            for cid, e in enumerate(launches, 2):
                db.execute("insert into StringIds values (?, ?)", (cid, e["name"]))
                db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, 1)", (cid,))
                db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (cid, 7, cid, cid, *e["grid"], *e["block"], e["shared"], e["static_shared"],
                            e["registers"], e["local"]))
            db.commit()
            db.close()
            plan = pc.convert(lines, nsys=str(path))
            self.assertTrue(plan["cublas_launches_observed"])
            self.assertEqual(pc.compare(self.record, "control-fused", plan, (1, 6))[0], 2)
            # The trace's attributes must agree with the runtime's.
            db = sqlite3.connect(path)
            db.execute("update CUPTI_ACTIVITY_KIND_KERNEL set registersPerThread = 99 where correlationId = 2")
            db.commit()
            db.close()
            with self.assertRaisesRegex(pc.PlanError, "attributes differ"):
                pc.convert(lines, nsys=str(path))

    def test_malformed_recordings_are_refused(self):
        header = json.dumps({"type": "header", "format": pc.FORMAT}) + "\n"
        chunk = json.dumps({"type": "chunk", "evaluation": 1, "chunk": 0, "rows": 1, "n_past": 0}) + "\n"
        for lines, error in (([], "empty"), (["{}\n"], "header"), ([header, chunk], "inside a chunk"),
                             ([header, chunk, chunk], "inside another"),
                             ([header, '{"type": "end_chunk"}\n'], "never began"),
                             ([header, '{"type": "launch"}\n'], "unknown"), ([header, "{\n"], "not JSON")):
            with self.subTest(error=error), self.assertRaisesRegex(pc.PlanError, error):
                pc.convert(lines)


if __name__ == "__main__":
    unittest.main()
