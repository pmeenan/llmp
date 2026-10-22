# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock
import mia_capture as mia
from test_plan_reports import header, inventory, completed

REPO = mia.HERE.parents[2]
PINS = json.loads((mia.HERE / "llm-capture-pins.json").read_text())
QP = mia.capture.load_quality_pair(REPO, PINS["quality_pair_sha256"])
IDS = list(range(255))


def response():
    return {"choices": [{"prompt_logprobs": [None] + [
        {str(i): {"logprob": -0.125}} for i in IDS[1:]],
        "logprobs": {"tokens": [f"token_id:{i}" for i in IDS]}}],
        "usage": {"prompt_tokens": 255, "completion_tokens": 0}}


def report_line(record):
    return 'llmp-runtime: debug plan model "qwen3.8": ' + json.dumps(record) + '\n'


def score_header():
    return {**header(), "phase": 4, "rows": 255}


def score_completed():
    return {**completed(), "phase": 4, "rows": 255}


def score_inventory(rows=255):
    records = inventory()
    for r in records:
        if r["kind"] == "physical_node" and r["id"] in (1, 2):
            r["shape"][1] = rows
            r["strides"][2:] = [r["strides"][1] * rows] * 2
    return records


class MiaCaptureTest(unittest.TestCase):
    def validate(self, payload, alter=None):
        with tempfile.TemporaryDirectory() as name:
            path = pathlib.Path(name) / "record.json"
            raw = json.dumps(payload).encode()
            (path.parent / "response.json").write_bytes(raw)
            receipt = {"score_response": {"file": "response.json", "sha256": mia.digest(raw)}}
            if alter:
                alter(receipt)
            return mia.validate_response(path, receipt, IDS, QP)

    def test_all_254_exact_targets_pass(self):
        got = self.validate(response())
        self.assertEqual(len(got.rows), 254)
        self.assertEqual([got.rows[0, p][0] for p in range(1, 255)], IDS[1:])

    def test_first_id_exact_and_all_rows_required(self):
        for mode in ("first", "first_missing", "last_missing", "null", "extra", "scored_first"):
            payload = response()
            choice = payload["choices"][0]
            if mode == "first":
                choice["logprobs"]["tokens"][0] = "token_id:999"
            elif mode == "first_missing":
                choice["logprobs"]["tokens"].pop(0)
            elif mode == "last_missing":
                choice["prompt_logprobs"].pop()
                choice["logprobs"]["tokens"].pop()
            elif mode == "null":
                choice["prompt_logprobs"][80] = None
            elif mode == "extra":
                choice["prompt_logprobs"].append({"255": {"logprob": -0.1}})
                choice["logprobs"]["tokens"].append("token_id:255")
            else:
                choice["prompt_logprobs"][0] = {"0": {"logprob": -0.1}}
            with self.subTest(mode=mode), self.assertRaises((ValueError, QP.Refused)):
                self.validate(payload)

    def test_no_substitute_or_noncanonical_target_or_invalid_loss(self):
        variants = [{}, {"01": {"logprob": -1}}, {"999": {"logprob": -1}},
                    {"1": {"logprob": -1}, "999": {"logprob": -0.1}}]
        variants += [{"1": {"logprob": value}} for value in (True, "-1", 0.1, float("inf"), float("nan"))]
        for row in variants:
            payload = response()
            payload["choices"][0]["prompt_logprobs"][1] = row
            with self.subTest(row=row), self.assertRaises((ValueError, QP.Refused)):
                self.validate(payload)
        payload = response()
        payload["choices"][0]["prompt_logprobs"][1] = {"999": {"logprob": -1}}
        payload["choices"][0]["logprobs"]["tokens"][1] = "token_id:999"
        with self.assertRaises(ValueError):
            self.validate(payload)  # Internally matching echo is insufficient: retained ID is 1.

    def test_hash_choices_and_exact_usage(self):
        with self.assertRaises(QP.Refused):
            self.validate(response(), lambda r: r["score_response"].update(sha256="0" * 64))
        variants = []
        p = response()
        p["choices"].append(copy.deepcopy(p["choices"][0]))
        variants.append(p)
        for usage in ({}, {"prompt_tokens": 254, "completion_tokens": 0},
                      {"prompt_tokens": 255, "completion_tokens": 1}):
            p = response()
            p["usage"] = usage
            variants.append(p)
        for p in variants:
            with self.assertRaises(ValueError):
                self.validate(p)

    def test_later_documents_use_the_global_installed_registry(self):
        records = [score_header(), *score_inventory(), score_completed()]
        records += [{**score_completed(), "sequence": i} for i in (2, 3, 4)]
        with tempfile.TemporaryDirectory() as name:
            path = pathlib.Path(name) / "runtime.log"
            path.write_text(''.join(map(report_line, records)))
            parsed = mia.parse_log(path)
            first = len(records) - 3
            bounds = [0, first, first + 1, first + 2, first + 3]
            docs = [{"document": i, "log_start_line": bounds[i], "log_end_line": bounds[i + 1]}
                    for i in range(4)]
            got = mia.associate_documents(parsed, docs, "qwen3.8")
            self.assertEqual([d["uses"][0]["identity"] for d in got], [("installed", 7)] * 4)
            self.assertEqual([d["score_input_rows"] for d in got], [255] * 4)
            for mode in ("missing", "overlap", "unexpected_stage", "outside", "duplicate"):
                bad, plan = copy.deepcopy(docs), copy.deepcopy(parsed)
                if mode == "missing":
                    bad[1]["log_end_line"] = bad[1]["log_start_line"]
                elif mode == "overlap":
                    bad[1]["log_start_line"] = 0
                elif mode == "unexpected_stage":
                    plan["plans"][0]["completions"][1]["stage"] = 1
                elif mode == "duplicate":
                    plan["plans"][0]["completions"][1]["sequence"] = 1
                else:
                    bad.pop()
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    mia.associate_documents(plan, bad, "qwen3.8")

    def test_only_complete_score_phase_work_is_accepted(self):
        # Both installed and completed headers agree: the generic parser cannot
        # substitute for the collector's phase and paid-row contract.
        for phase, rows in ((1, 255), (0, 255), (3, 255), (True, 255),
                            (4, 0), (4, True), (4, -1), (4, 254), (4, 256)):
            records = [{**score_header(), "phase": phase, "rows": rows},
                       *score_inventory(),
                       {**score_completed(), "phase": phase, "rows": rows}]
            with self.subTest(phase=phase, rows=rows), tempfile.TemporaryDirectory() as name:
                path = pathlib.Path(name) / "runtime.log"
                path.write_text(''.join(map(report_line, records)))
                parsed = mia.parse_log(path)
                with self.assertRaises(ValueError):
                    mia.associate_documents(parsed, [{"document": 0, "log_start_line": 0,
                        "log_end_line": len(records)}], "qwen3.8")

    def test_capacity_split_score_steps_must_cover_exactly_one_document(self):
        records = []
        # Capacity-dependent chunking can install two shapes and reuse both in
        # later requests. Completion rows count inputs, including the first ID.
        for ident, rows in ((7, 128), (8, 127)):
            records += [{**score_header(), "id": ident, "rows": rows},
                        *score_inventory(rows),
                        {**score_completed(), "id": ident, "rows": rows, "sequence": ident - 6}]
        first_end = len(records)
        records += [{**score_completed(), "id": ident, "rows": rows, "sequence": ident - 4}
                    for ident, rows in ((7, 128), (8, 127))]
        with tempfile.TemporaryDirectory() as name:
            path = pathlib.Path(name) / "runtime.log"
            path.write_text(''.join(map(report_line, records)))
            parsed = mia.parse_log(path)
            docs = [{"document": i, "log_start_line": start, "log_end_line": end}
                    for i, (start, end) in enumerate(((0, first_end), (first_end, len(records))))]
            got = mia.associate_documents(parsed, docs, "qwen3.8")
            self.assertEqual([d["score_input_rows"] for d in got], [255, 255])
            self.assertEqual([len(d["uses"]) for d in got], [2, 2])
            for rows in (126, 128):
                bad = copy.deepcopy(parsed)
                bad["plans"][1]["completions"][1]["rows"] = rows
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    mia.associate_documents(bad, docs, "qwen3.8")

    def test_reference_all_positions_and_document_hashes(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            docs = [{"document": d, "tokens": IDS, "text_sha256": mia.digest(b"text")}
                    for d in range(4)]
            for d in range(4):
                (root / f"doc{d}.txt").write_bytes(b"text")
            (root / "inputs.json").write_text(json.dumps({"documents": docs}))
            rows = [{"document": d, "position": p, "target": p, "nll": 1.0}
                    for d in range(4) for p in range(1, 255)]
            def write(rows):
                (root / "losses.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in rows))
                return {"reference_files": {p.name: {"bytes": p.stat().st_size,
                    "sha256": mia.capture.sha256_file(p)} for p in root.iterdir()}}
            pins = write(rows)
            self.assertEqual(len(mia.reference_inputs(root, pins, QP)[1].rows), 1016)
            for altered in (rows[:-1], rows + [rows[0]], [dict(r, target=999) if i == 3 else r
                                                       for i, r in enumerate(rows)]):
                pins = write(altered)
                with self.assertRaises((ValueError, QP.Refused)):
                    mia.reference_inputs(root, pins, QP)
            pins = write(rows)
            (root / "doc0.txt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                mia.reference_inputs(root, pins, QP)

    def test_main_real_service_boundary_and_four_document_capture(self):
        sg = mia.capture.load_speed_gate(REPO)
        docs = [{"document": d, "tokens": IDS, "text_sha256": mia.digest(b"text")} for d in range(4)]
        retained = QP.Losses("synthetic", "losses")
        for d in range(4):
            for p in range(1, 255):
                retained.add(d, p, p, 2.0)
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            runtime = root / "runtime"
            runtime.touch()
            for d in range(4):
                (root / f"doc{d}.txt").write_bytes(b"text")
            out = root / "out"
            original_sha = mia.capture.sha256_file
            def sha(path):
                return mia.EXPECTED["A"] if path == runtime else original_sha(path)
            requests = []
            def request(method, route, raw, headers):
                self.assertEqual((method, route), ("POST", "/v1/completions"))
                body = json.loads(raw)
                self.assertEqual(body, {"model": "qwen3.8", "prompt": "text", "max_tokens": 0,
                    "prompt_logprobs": 0, "echo": True, "logprobs": 0, "return_tokens_as_token_ids": True})
                log = out / "work/qwen38-mia-A/runtime.log"
                with log.open('a') as f:
                    if not requests:
                        f.write(''.join(map(report_line, [score_header(), *score_inventory()])))
                    f.write(report_line({**score_completed(), "sequence": len(requests) + 1}))
                requests.append(body)
            argv = ["mia_capture.py", "--repo", str(REPO), "--arm", "A", "--runtime", str(runtime),
                    "--reference", str(root), "--out", str(out)]
            old_umask = os.umask(0o022)
            try:
                with (mock.patch("sys.argv", argv),
                      mock.patch.object(mia.capture, "sha256_file", side_effect=sha),
                      mock.patch.object(mia.capture, "load_speed_gate", return_value=sg),
                      mock.patch.object(mia.capture, "verify_inputs", return_value=(PINS["configs"]["qwen38"], {})),
                      mock.patch.object(mia, "reference_inputs", return_value=(docs, retained)),
                      mock.patch.object(sg, "host_facts", return_value={}),
                      mock.patch.object(sg, "port_open", return_value=False),
                      mock.patch.object(sg.Service, "wait_ready", return_value=0),
                      mock.patch.object(sg.subprocess, "Popen") as popen,
                      mock.patch.object(mia.http.client, "HTTPConnection") as connection):
                    popen.return_value.poll.return_value = 0
                    popen.return_value.returncode = 0
                    connection.return_value.request.side_effect = request
                    connection.return_value.getresponse.return_value.status = 200
                    connection.return_value.getresponse.return_value.read.return_value = json.dumps(response()).encode()
                    self.assertEqual(mia.main(), 0)
                    self.assertEqual(popen.call_args.kwargs["env"], sg.arm_env({}))
                    self.assertIn("--debug-plans", popen.call_args.args[0])
                    self.assertEqual(len(requests), 4)
                    got = json.loads((out / "capture.json").read_text())
                    self.assertTrue(got["capture_complete"])
                    self.assertEqual(got["mia_strict"]["status"], "FAIL")
                    self.assertEqual(got["mia_strict"]["comparison"]["targets"], 1016)
                    self.assertEqual(len(got["document_plan_uses"]), 4)
                    self.assertEqual(out.stat().st_mode & 0o777, 0o700)
                    with self.assertRaises(FileExistsError):
                        mia.main()
            finally:
                os.umask(old_umask)


if __name__ == "__main__":
    unittest.main()
