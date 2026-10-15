# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""tools/quality_pair.py's loaders, pairing and bounds on synthetic losses, no model needed."""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import pathlib
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("quality_pair", ROOT / "tools/quality_pair.py")
qp = importlib.util.module_from_spec(spec)
sys.modules["quality_pair"] = qp
spec.loader.exec_module(qp)

IDS = [11, 22, 33, 44, 55, 66]  # position 0 is the unscored first token
NLLS = [None, 1.0, 2.0, 3.0, 0.5, 1.5]


def write_jsonl(path, ids=IDS, nlls=NLLS, offsets=None, document=0):
    with open(path, "w") as f:
        for position in range(1, len(ids)):
            nll = nlls[position] + (offsets or {}).get(position, 0.0)
            f.write(json.dumps({"document": document, "position": position,
                                "target": ids[position], "nll": nll}) + "\n")


def write_speed_gate(directory, name="qwen38-quality-new", ids=IDS, nlls=NLLS, echo=True):
    rows = [None] + [{str(ids[p]): {"logprob": -nlls[p], "rank": 1}} for p in range(1, len(ids))]
    choice = {"prompt_logprobs": rows}
    if echo:
        choice["logprobs"] = {"tokens": [f"token_id:{t}" for t in ids]}
    body = json.dumps({"choices": [choice]}).encode()
    (directory / f"{name}-score.json").write_bytes(body)
    record = {"family": "qwen38", "error": None,
              "score_response": {"file": f"{name}-score.json",
                                 "sha256": hashlib.sha256(body).hexdigest()}}
    path = directory / f"{name}.json"
    path.write_text(json.dumps(record))
    return path


def write_k1(path, ids=IDS, nlls=NLLS, prefix=1):
    first = max(prefix, 1)
    record = {"encoding": "f16", "error": None, "prefix_tokens": prefix,
              "scores": [-n for n in nlls[first:]],
              "ids_sha256": hashlib.sha256(json.dumps(ids[first:]).encode()).hexdigest()}
    path.write_text(json.dumps(record))


def write_prereg(path, **bounds):
    path.write_text(json.dumps({"format": qp.PREREG_FORMAT, "name": "test", "bounds": bounds}))
    return path


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = qp.main(list(argv))
        return code, out.getvalue(), err.getvalue()


class Loading(Fixture):
    def test_jsonl_speed_gate_and_k1_agree_on_the_same_losses(self):
        write_jsonl(self.dir / "r.jsonl")
        a, _ = qp.load(str(self.dir / "r.jsonl"))
        b, _ = qp.load(str(write_speed_gate(self.dir)))
        self.assertEqual(a.rows, b.rows)
        write_k1(self.dir / "k.json")
        c, _ = qp.load(str(self.dir / "k.json"))
        self.assertEqual({k: v[1] for k, v in a.rows.items()},
                         {k: v[1] for k, v in c.rows.items()})
        self.assertTrue(all(v[0] is None for v in c.rows.values()))

    def test_bad_losses_are_refused(self):
        for nll in (float("nan"), float("inf"), -0.5, True, "1.0"):
            with self.subTest(nll=nll):
                path = self.dir / "bad.jsonl"
                path.write_text(json.dumps({"document": 0, "position": 1, "target": 1,
                                            "nll": nll}).replace("NaN", "NaN") + "\n")
                with self.assertRaises(qp.Refused):
                    qp.load(str(path))

    def test_duplicates_and_empty_files_are_refused(self):
        path = self.dir / "d.jsonl"
        row = json.dumps({"document": 0, "position": 1, "target": 1, "nll": 1.0})
        path.write_text(row + "\n" + row + "\n")
        with self.assertRaises(qp.Refused):
            qp.load(str(path))
        path.write_text("")
        with self.assertRaises(qp.Refused):
            qp.load(str(path))

    def test_a_changed_or_failed_speed_gate_record_is_refused(self):
        path = write_speed_gate(self.dir)
        (self.dir / "qwen38-quality-new-score.json").write_bytes(b"{}")
        with self.assertRaises(qp.Refused):
            qp.load(str(path))
        record = json.loads(path.read_text())
        record["error"] = "RuntimeError: boom"
        path.write_text(json.dumps(record))
        with self.assertRaises(qp.Refused):
            qp.load(str(path))

    def test_echo_and_scored_target_must_agree(self):
        path = write_speed_gate(self.dir)
        body = json.loads((self.dir / "qwen38-quality-new-score.json").read_text())
        body["choices"][0]["logprobs"]["tokens"][2] = "token_id:99"
        raw = json.dumps(body).encode()
        (self.dir / "qwen38-quality-new-score.json").write_bytes(raw)
        record = json.loads(path.read_text())
        record["score_response"]["sha256"] = hashlib.sha256(raw).hexdigest()
        path.write_text(json.dumps(record))
        with self.assertRaises(qp.Refused):
            qp.load(str(path))

    def test_combined_k1_record_is_refused_with_advice(self):
        path = self.dir / "c.json"
        path.write_text(json.dumps({"format": "llmp-k1-quality/2", "delta": {}}))
        with self.assertRaisesRegex(qp.Refused, "arm record"):
            qp.load(str(path))


class Pairing(Fixture):
    def pair(self, candidate, reference):
        c, _ = qp.load(str(candidate))
        r, _ = qp.load(str(reference))
        return qp.pair(c, r)

    def test_identical_losses_have_zero_deltas(self):
        write_jsonl(self.dir / "r.jsonl")
        rows = self.pair(write_speed_gate(self.dir), self.dir / "r.jsonl")
        result = qp.compare(rows, 0.0)
        self.assertEqual(result["targets"], 5)
        self.assertEqual(result["nll_delta"], 0.0)
        self.assertEqual(result["max_abs_diff"], 0.0)
        self.assertEqual(result["agreement"]["fraction"], 1.0)
        self.assertEqual(result["identical"], 5)

    def test_deltas_match_hand_computation(self):
        write_jsonl(self.dir / "r.jsonl")
        write_jsonl(self.dir / "c.jsonl", offsets={1: 0.1, 2: -0.2, 4: 0.3})
        result = qp.compare(self.pair(self.dir / "c.jsonl", self.dir / "r.jsonl"), 0.15)
        self.assertAlmostEqual(result["nll_delta"], 0.2 / 5)
        self.assertAlmostEqual(result["ppl_delta_pct"], (math.exp(0.04) - 1) * 100)
        self.assertAlmostEqual(result["mean_abs_diff"], 0.6 / 5)
        self.assertAlmostEqual(result["max_abs_diff"], 0.3)
        self.assertEqual(result["max_abs_diff_at"]["position"], 4)
        self.assertEqual(result["max_abs_diff_at"]["target"], 55)
        self.assertEqual((result["candidate_lower"], result["candidate_higher"]), (1, 2))
        self.assertAlmostEqual(result["agreement"]["fraction"], 3 / 5)  # 0.1 and two zeros

    def test_per_document_deltas(self):
        for name, offset in (("c.jsonl", 0.1), ("r.jsonl", 0.0)):
            with open(self.dir / name, "w") as f:
                for document in (0, 1):
                    for position in (1, 2):
                        f.write(json.dumps({"document": document, "position": position,
                                            "target": position, "nll": 1.0 +
                                            (offset if document else 0.0)}) + "\n")
        result = qp.compare(self.pair(self.dir / "c.jsonl", self.dir / "r.jsonl"), None)
        self.assertEqual(result["documents"][0]["ppl_delta_pct"], 0.0)
        self.assertAlmostEqual(result["documents"][1]["ppl_delta_pct"],
                               (math.exp(0.1) - 1) * 100)

    def test_a_different_target_id_is_refused(self):
        write_jsonl(self.dir / "r.jsonl")
        other = list(IDS)
        other[3] = 4444  # a tokenizer difference at one position
        write_jsonl(self.dir / "c.jsonl", ids=other)
        with self.assertRaisesRegex(qp.Refused, "position 3"):
            self.pair(self.dir / "c.jsonl", self.dir / "r.jsonl")

    def test_different_coverage_is_refused(self):
        write_jsonl(self.dir / "r.jsonl")
        write_jsonl(self.dir / "c.jsonl", ids=IDS[:-1], nlls=NLLS[:-1])
        with self.assertRaisesRegex(qp.Refused, "different targets"):
            self.pair(self.dir / "c.jsonl", self.dir / "r.jsonl")

    def test_k1_scores_pair_through_the_id_digest(self):
        write_jsonl(self.dir / "r.jsonl")
        write_k1(self.dir / "k.json")
        rows = self.pair(self.dir / "k.json", self.dir / "r.jsonl")
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0][1], 22)  # the target comes from the reference
        # The reference may have more positions than the passage's: a chat-format prefix.
        write_k1(self.dir / "k2.json", prefix=3)
        rows = self.pair(self.dir / "k2.json", self.dir / "r.jsonl")
        self.assertEqual([r[0][1] for r in rows], [3, 4, 5])

    def test_k1_scores_over_other_tokens_are_refused(self):
        other = list(IDS)
        other[2] = 999
        write_jsonl(self.dir / "r.jsonl", ids=other)
        write_k1(self.dir / "k.json")
        with self.assertRaisesRegex(qp.Refused, "different tokens"):
            self.pair(self.dir / "k.json", self.dir / "r.jsonl")

    def test_k1_against_k1_is_refused(self):
        write_k1(self.dir / "a.json")
        write_k1(self.dir / "b.json")
        with self.assertRaises(qp.Refused):
            self.pair(self.dir / "a.json", self.dir / "b.json")


class Bounds(Fixture):
    def setUp(self):
        super().setUp()
        write_jsonl(self.dir / "r.jsonl")
        write_jsonl(self.dir / "c.jsonl", offsets={1: 0.01, 2: -0.02})
        self.argv = ["pair", "--candidate", str(self.dir / "c.jsonl"),
                     "--reference", str(self.dir / "r.jsonl")]

    def run_with(self, **bounds):
        return self.run_main(*self.argv, "--prereg", str(write_prereg(self.dir / "p.json",
                                                                       **bounds)))

    def test_all_bounds_holding_passes(self):
        code, out, _ = self.run_with(expected_targets=5, max_abs_diff=0.05,
                                     max_mean_abs_diff=0.01, max_abs_ppl_delta_pct=1.0,
                                     max_abs_nll_delta=0.01, min_agreement=0.8,
                                     agreement_tolerance=0.015, max_abs_doc_ppl_delta_pct=1.0)
        self.assertEqual(code, 0, out)
        self.assertTrue(out.rstrip().endswith("PASS"))

    def test_each_bound_can_fail(self):
        cases = {"expected_targets": 6, "min_targets": 6, "max_abs_diff": 0.001,
                 "max_mean_abs_diff": 0.001, "max_abs_nll_delta": 0.0001,
                 "max_abs_ppl_delta_pct": 0.0001, "max_abs_doc_ppl_delta_pct": 0.0001,
                 "min_agreement": 1.0}
        for name, limit in cases.items():
            bounds = {name: limit}
            if name == "min_agreement":
                bounds["agreement_tolerance"] = 0.005
            with self.subTest(bound=name):
                code, out, _ = self.run_with(**bounds)
                self.assertEqual(code, 1, out)
                self.assertIn("FAIL", out)

    def test_an_increase_bound_ignores_improvement(self):
        write_jsonl(self.dir / "c.jsonl", offsets={1: -0.5})
        self.assertEqual(self.run_with(max_ppl_increase_pct=0.0)[0], 0)
        write_jsonl_over = lambda: write_jsonl(self.dir / "c2.jsonl", offsets={1: 0.5})
        write_jsonl_over()
        self.argv[2] = str(self.dir / "c2.jsonl")
        self.assertEqual(self.run_with(max_ppl_increase_pct=0.0)[0], 1)

    def test_refused_inputs_exit_two(self):
        write_jsonl(self.dir / "c.jsonl", ids=[1, 2, 3, 4, 5, 6])
        code, _, err = self.run_with(max_abs_diff=1.0)
        self.assertEqual(code, 2)
        self.assertIn("refused", err)

    def test_bad_preregistrations_are_refused(self):
        bad = [{"format": "other", "bounds": {"max_abs_diff": 1}},
               {"format": qp.PREREG_FORMAT, "bounds": {}},
               {"format": qp.PREREG_FORMAT, "bounds": {"max_abs_dif": 1}},
               {"format": qp.PREREG_FORMAT, "bounds": {"max_abs_diff": -1}},
               {"format": qp.PREREG_FORMAT, "bounds": {"max_abs_diff": True}},
               {"format": qp.PREREG_FORMAT, "bounds": {"min_agreement": 0.5}},
               {"format": qp.PREREG_FORMAT, "bounds": {"agreement_tolerance": 0.5}}]
        for prereg in bad:
            with self.subTest(prereg=prereg):
                (self.dir / "p.json").write_text(json.dumps(prereg))
                self.assertEqual(self.run_main(*self.argv, "--prereg",
                                               str(self.dir / "p.json"))[0], 2)

    def test_the_pinned_reference_hash_is_enforced(self):
        raw = (self.dir / "r.jsonl").read_bytes()
        for digest, code in ((hashlib.sha256(raw).hexdigest(), 0), ("0" * 64, 2)):
            (self.dir / "p.json").write_text(json.dumps(
                {"format": qp.PREREG_FORMAT, "reference_sha256": digest,
                 "bounds": {"max_abs_diff": 1.0}}))
            with self.subTest(digest=digest[:8]):
                self.assertEqual(self.run_main(*self.argv, "--prereg",
                                               str(self.dir / "p.json"))[0], code)

    def test_report_file_and_json_output(self):
        report_path = self.dir / "report.json"
        code, out, _ = self.run_main(*self.argv, "--prereg",
                                     str(write_prereg(self.dir / "p.json", max_abs_diff=1.0)),
                                     "--report", str(report_path), "--json")
        self.assertEqual(code, 0)
        report = json.loads(report_path.read_text())
        self.assertEqual(json.loads(out), report)
        self.assertEqual(report["format"], qp.REPORT_FORMAT)
        self.assertTrue(report["pass"])
        self.assertEqual(report["candidate"]["kind"], "losses")
        self.assertEqual(len(report["prereg"]["sha256"]), 64)


class Extract(Fixture):
    def test_extract_writes_a_reference_that_pairs_with_the_record(self):
        record = write_speed_gate(self.dir)
        out = self.dir / "retained.jsonl"
        self.assertEqual(self.run_main("extract", "--candidate", str(record),
                                       "--out", str(out))[0], 0)
        a, _ = qp.load(str(record))
        b, _ = qp.load(str(out))
        self.assertEqual(a.rows, b.rows)
        # A retained file is never silently replaced.
        self.assertEqual(self.run_main("extract", "--candidate", str(record),
                                       "--out", str(out))[0], 2)

    def test_extract_refuses_records_without_ids(self):
        write_k1(self.dir / "k.json")
        self.assertEqual(self.run_main("extract", "--candidate", str(self.dir / "k.json"),
                                       "--out", str(self.dir / "o.jsonl"))[0], 2)


if __name__ == "__main__":
    unittest.main()
