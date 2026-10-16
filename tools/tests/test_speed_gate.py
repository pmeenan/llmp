# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""tools/speed_gate.py's schedule, statistic and stopping rule on synthetic runs, no GPU needed."""

import contextlib
import dataclasses
import importlib.util
import io
import json
import math
import pathlib
import socket
import sys
import tempfile
import threading
import unittest
import unittest.mock

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("speed_gate", ROOT / "tools/speed_gate.py")
gate = importlib.util.module_from_spec(spec)
sys.modules["speed_gate"] = gate
spec.loader.exec_module(gate)

MARGIN = math.log1p(gate.GATE)


def record(family, block, position, metrics, error=None):
    label = gate.ORDER[position]
    spec = gate.FAMILIES.get(family)
    return {"family": family, "block": block, "position": position, "label": label,
            "metrics": metrics, "error": error, "problems": [],
            "session": f"{family}-s1", "started_unix": 100.0 + 4 * block + position,
            "runtime_sha256": f"{label}-binary", "config_sha256": f"{label}-config",
            "family_spec": json.loads(json.dumps(dataclasses.asdict(spec))) if spec else None}


def prime_record(family, metrics=None):
    """The session's priming run: the old arm, before every measured run."""
    return {"family": family, "label": gate.PRIME_ARM, "block": None, "position": None,
            "prime": True, "session": f"{family}-s1", "started_unix": 1.0,
            "runtime_sha256": f"{gate.PRIME_ARM}-binary",
            "config_sha256": f"{gate.PRIME_ARM}-config",
            "metrics": metrics or {}, "error": None, "problems": []}


def write_blocks(out, family, blocks):
    """blocks: a list of (old, new) metric dicts; each block runs old/new/new/old, all in
    one session after its priming run."""
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "primes").mkdir(parents=True, exist_ok=True)
    (out / "primes" / f"{family}-s1.json").write_text(json.dumps(prime_record(family)))
    for b, (old, new) in enumerate(blocks, start=1):
        for p, label in enumerate(gate.ORDER):
            r = record(family, b, p, old if label == "old" else new)
            (out / "runs" / f"{family}-b{b:02d}-p{p}-{label}.json").write_text(json.dumps(r))


class ImagePins(unittest.TestCase):
    CANDIDATE = "a" * 64

    def records(self, candidate=None):
        family = gate.FAMILIES["image"]
        blocks = [{p: record("image", 1, p, {"generation_s": 33.0, "peak_gib": 32.0})
                   for p in range(4)}]
        prime = prime_record("image")
        prime["turn"] = {"pixels_sha256": gate.IMAGE_PIXELS_SHA256}
        for r in blocks[0].values():
            pin = candidate if r["label"] == "new" and candidate else gate.IMAGE_PIXELS_SHA256
            r["turn"] = {"pixels_sha256": pin}
            if candidate:
                r["image_pixels_sha256"] = pin
        return family, blocks, {"image-s1": prime}

    def test_historical_implicit_baseline_pin_remains_valid(self):
        family, blocks, primes = self.records()
        self.assertEqual(gate.validity(family, blocks, primes), [])

    def test_candidate_can_repeat_its_own_predeclared_output(self):
        family, blocks, primes = self.records(self.CANDIDATE)
        self.assertEqual(gate.validity(family, blocks, primes), [])
        self.assertEqual(gate.block_slowdowns(blocks[0], "generation_s"), 0.0)

    def test_image_run_checks_the_candidate_pin(self):
        for observed in (self.CANDIDATE, gate.IMAGE_PIXELS_SHA256):
            with self.subTest(observed=observed), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                turn = {"swap": {"ready": 1.0}, "first_step_seconds": 2.0,
                        "generation_seconds": 34.0, "peak_bytes": 1024,
                        "pixels_sha256": observed}
                (root / "report.json").write_text(json.dumps({"turns": [turn]}))
                r = {"image_pixels_sha256": self.CANDIDATE}
                with unittest.mock.patch.object(gate, "sha256_file",
                                                return_value=gate.IMAGE_NOISE_SHA256), \
                     unittest.mock.patch.object(gate, "MemorySampler") as sampler, \
                     unittest.mock.patch.object(gate.subprocess, "run") as run:
                    sampler.return_value.stop.return_value = {"peak_gib": 32.0}
                    run.return_value.returncode = 0
                    gate.image_run(gate.FAMILIES["image"], pathlib.Path("/runtime"),
                                   pathlib.Path("/share"), root,
                                   gate.config_template(gate.FAMILIES["image"]), {}, r)
                self.assertEqual(bool(r["problems"]), observed != self.CANDIDATE)
                self.assertEqual(r["metrics"]["generation_s"], 33.0)

    def test_changed_pin_cannot_be_mixed_in_one_arm(self):
        family, blocks, primes = self.records(self.CANDIDATE)
        blocks[0][2]["image_pixels_sha256"] = "b" * 64
        blocks[0][2]["turn"]["pixels_sha256"] = "b" * 64
        self.assertTrue(any("image output pins" in reason
                            for reason in gate.validity(family, blocks, primes)))

    def test_mismatched_or_missing_outputs_including_prime_are_invalid(self):
        for position in (None, 0, 1, 2, 3):
            with self.subTest(position=position):
                family, blocks, primes = self.records(self.CANDIDATE)
                r = primes["image-s1"] if position is None else blocks[0][position]
                r.pop("turn")
                self.assertTrue(any("pixels differ" in reason
                                    for reason in gate.validity(family, blocks, primes)))

    def test_bad_pins_refuse_before_runtime_access(self):
        for family, label, pin in (("image", "new", "x" * 64),
                                   ("image", "new", "a" * 63),
                                   ("image", "old", self.CANDIDATE),
                                   ("gemma2", "new", self.CANDIDATE)):
            with self.subTest(family=family, label=label, pin=pin), self.assertRaises(SystemExit):
                gate.main(["run", "--family", family, "--label", label,
                           "--runtime", "/nonexistent", "--out", "/nonexistent",
                           "--image-pixels-sha256", pin])

    def test_plan_forwards_candidate_pin_only_to_new_runs(self):
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            gate.main(["plan", "--family", "image", "--old", "/o", "--new", "/n",
                       "--out", "/out", "--blocks", "1",
                       "--new-image-pixels-sha256", self.CANDIDATE])
        prime, *runs = text.getvalue().splitlines()
        self.assertNotIn("--image-pixels-sha256", prime)
        for line, label in zip(runs, gate.ORDER):
            self.assertEqual("--image-pixels-sha256" in line, label == "new")
        with self.assertRaises(SystemExit):
            gate.main(["plan", "--family", "gemma2", "--old", "/o", "--new", "/n",
                       "--out", "/out", "--new-image-pixels-sha256", self.CANDIDATE])


class WorkloadIdentity(unittest.TestCase):
    """Both arms must have paid the same work: the same prompts tokenize to the same counts."""

    FAMILY = "gemma2"

    def passes(self, tokens):
        family = gate.FAMILIES[self.FAMILY]
        first = 0
        passes = []
        for p in range(1 + family.repeats):
            cells = {}
            for width in family.cells:
                cells[f"C{width}"] = {"width": width, "problems": [], "requests": [],
                                      "prompts": list(range(first, first + width)),
                                      "prompt_tokens": [tokens + first + i for i in range(width)]}
                first += width
            passes.append({"warmup": p == 0, "cells": cells})
        return passes

    def write(self, out, new_tokens, blocks=None):
        family = gate.FAMILIES[self.FAMILY]
        write_blocks(out, self.FAMILY, [({}, {})] * (blocks or family.min_blocks))
        for path in sorted((out / "runs").glob("*.json")):
            r = json.loads(path.read_text())
            r["passes"] = self.passes(new_tokens if r["label"] == "new" else 8000)
            r["memory"] = {"peak_gib": 10.0}
            path.write_text(json.dumps(r))

    def test_same_token_counts_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            self.write(out, 8000)
            report = gate.verdicts(out, self.FAMILY)[self.FAMILY]
            self.assertEqual([x for x in report["invalid"] if "differ" in x], [])

    def test_a_tokenizer_difference_between_arms_makes_the_family_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            self.write(out, 8003)  # the new arm counts three more tokens a prompt
            report = gate.verdicts(out, self.FAMILY)[self.FAMILY]
            self.assertEqual(report["verdict"], "INVALID")
            mismatches = [x for x in report["invalid"] if "prompt_tokens" in x]
            self.assertTrue(mismatches)
            self.assertTrue(all("new block" in x and "tokenizer" in x for x in mismatches))
            self.assertEqual(gate.verdict_status({self.FAMILY: report}), 1)

    def test_one_run_whose_count_differs_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            self.write(out, 8000)
            path = out / "runs" / f"{self.FAMILY}-b02-p2-new.json"
            r = json.loads(path.read_text())
            r["passes"][1]["cells"]["C1"]["prompt_tokens"] = [1]
            path.write_text(json.dumps(r))
            invalid = gate.verdicts(out, self.FAMILY)[self.FAMILY]["invalid"]
            self.assertEqual(len([x for x in invalid if "differ" in x]), 1)
            self.assertIn("block 2 position 2 pass 1 C1", next(x for x in invalid if "differ" in x))

    def test_different_prompts_or_widths_are_a_mismatch(self):
        family = gate.FAMILIES[self.FAMILY]
        old = {"label": "old", "block": 1, "position": 0, "passes": self.passes(8000)}
        for field, value in (("prompts", [99]), ("width", 7)):
            with self.subTest(field=field):
                new = {"label": "new", "block": 1, "position": 1, "passes": self.passes(8000)}
                new["passes"][0]["cells"]["C1"][field] = value
                reasons = gate.workload_problems([old, new])
                self.assertEqual(len(reasons), 1)
                self.assertIn(field, reasons[0])

    def test_records_without_passes_or_token_counts_are_not_compared(self):
        old = {"label": "old", "block": 1, "position": 0, "passes": self.passes(8000)}
        bare = {"label": "new", "block": 1, "position": 1}
        self.assertEqual(gate.workload_problems([old, bare]), [])
        self.assertEqual(gate.workload_problems([bare]), [])
        partial = {"label": "new", "block": 1, "position": 1,
                   "passes": [{"warmup": True, "cells": {"C1": {"requests": [{"error": "x"}]}}}]}
        self.assertEqual(gate.workload_problems([old, partial]), [])

    def test_usage_in_the_requests_stands_in_for_the_cell_list(self):
        def run(label, tokens):
            return {"label": label, "block": 1, "position": 0, "passes": [{"warmup": False, "cells": {
                "C1": {"width": 1, "prompts": [0],
                       "requests": [{"usage": {"prompt_tokens": tokens}}]}}}]}
        self.assertEqual(gate.workload_problems([run("old", 5), run("new", 5)]), [])
        self.assertEqual(len(gate.workload_problems([run("old", 5), run("new", 6)])), 1)


class Judge(unittest.TestCase):
    def test_identical_blocks_pass(self):
        row = gate.judge([0.0, 0.0, 0.0], MARGIN, 3, 6)
        self.assertEqual(row["verdict"], "PASS")
        self.assertEqual(row["half_width"], 0.0)

    def test_too_few_blocks_give_no_verdict(self):
        self.assertEqual(gate.judge([0.0, 0.0], MARGIN, 3, 6)["verdict"], "INCOMPLETE")
        self.assertEqual(gate.judge([0.0], MARGIN, 1, 6)["verdict"], "INCOMPLETE")

    def test_a_clear_slowdown_fails(self):
        values = [math.log(1.03), math.log(1.031), math.log(1.029)]
        self.assertEqual(gate.judge(values, MARGIN, 3, 6)["verdict"], "FAIL")

    def test_an_unresolved_estimate_continues_then_has_not_passed(self):
        values = [0.004, 0.012, 0.008]
        self.assertEqual(gate.judge(values, MARGIN, 3, 6)["verdict"], "CONTINUE")
        self.assertEqual(gate.judge(values * 2, MARGIN, 3, 6)["verdict"], "NOT PASSED")

    def test_bound_is_one_sided_t(self):
        # Each look's bound: one-sided 97.5% t on n - 1 degrees of freedom.
        values = [0.001, 0.003, 0.002]
        row = gate.judge(values, MARGIN, 3, 6)
        self.assertAlmostEqual(row["half_width"], 4.303 * 0.001 / math.sqrt(3))
        self.assertAlmostEqual(row["upper"], 0.002 + row["half_width"])
        self.assertAlmostEqual(row["lower"], 0.002 - row["half_width"])
        self.assertEqual(gate.t_look(25), gate.T975[20])  # conservative past the table

    def test_looking_after_every_block_keeps_a_true_one_percent_near_five_percent(self):
        sd = 0.003
        at_edge = gate.operating(3, 6, MARGIN, sd, trials=1000)
        at_edge95 = gate.operating(3, 6, MARGIN, sd, gate.t95, trials=1000)
        self.assertLess(at_edge.get("PASS", 0), 0.08)
        self.assertGreater(at_edge95.get("PASS", 0), at_edge.get("PASS", 0))
        self.assertGreater(gate.operating(3, 6, 0.0, sd, trials=300)["PASS"], 0.95)


class Spread(unittest.TestCase):
    def test_blocks_needed_follow_the_resolution(self):
        self.assertEqual(gate.blocks_needed(0.0), 3)
        # t(0.95, 2) * 0.2% / sqrt(3) = 0.34% fits three blocks; 0.4% needs four
        # (2.920 * 0.4% / sqrt(3) = 0.67%, 2.353 * 0.4% / 2 = 0.47%).
        self.assertEqual(gate.blocks_needed(0.002), 3)
        self.assertEqual(gate.blocks_needed(0.004), 4)
        self.assertIsNone(gate.blocks_needed(0.05))

    def test_same_binary_runs_report_their_spread(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            (out / "runs").mkdir()
            values = [[1.00, 1.01, 0.99, 1.00], [1.02, 1.01, 1.03, 1.02]]
            for b, xs in enumerate(values, start=1):
                for p, x in enumerate(xs):
                    r = record("f", b, p, {"C1.prefill_s": x, "peak_gib": 10.0 + p})
                    (out / "runs" / f"f-{b}-{p}.json").write_text(json.dumps(r))
            gate.FAMILIES.setdefault("f", gate.Family("f", "llm", "f", "s", ""))
            row = gate.spread(out, "f")["f"]["C1.prefill_s"]
            self.assertEqual(row["runs"], 8)
            self.assertEqual(row["blocks"], 2)
            self.assertGreater(row["run_sd"], 0.005)
            self.assertLess(row["run_sd"], 0.02)
            self.assertAlmostEqual(gate.spread(out, "f")["f"]["peak_gib"]["range"], 3.0)


class Blocks(unittest.TestCase):
    def block(self, old, new):
        return {p: record("f", 1, p, old if gate.ORDER[p] == "old" else new) for p in range(4)}

    def test_times_and_rates_are_slowdowns_when_new_is_slower(self):
        b = self.block({"C1.prefill_s": 1.0, "C1.decode_tok_s": 50.0},
                       {"C1.prefill_s": 1.02, "C1.decode_tok_s": 49.0})
        self.assertAlmostEqual(gate.block_slowdowns(b, "C1.prefill_s"), math.log(1.02))
        self.assertAlmostEqual(gate.block_slowdowns(b, "C1.decode_tok_s"), math.log(50 / 49))

    def test_peak_memory_is_an_absolute_difference(self):
        b = self.block({"peak_gib": 10.0}, {"peak_gib": 10.25})
        self.assertAlmostEqual(gate.block_slowdowns(b, "peak_gib"), 0.25)

    def test_incomplete_failed_or_misordered_blocks_are_skipped(self):
        b = self.block({"x.prefill_s": 1.0}, {"x.prefill_s": 1.0})
        del b[3]
        self.assertIsNone(gate.block_slowdowns(b, "x.prefill_s"))
        b = self.block({"x.prefill_s": 1.0}, {"x.prefill_s": 1.0})
        b[1]["error"] = "RuntimeError: boom"
        self.assertIsNone(gate.block_slowdowns(b, "x.prefill_s"))
        b = self.block({"x.prefill_s": 1.0}, {"x.prefill_s": 1.0})
        b[0]["label"] = "new"
        self.assertIsNone(gate.block_slowdowns(b, "x.prefill_s"))

    def test_bookends_cancel_linear_drift(self):
        # Each run 1% slower than the one before, the two binaries identical.
        b = {p: record("f", 1, p, {"C1.prefill_s": 1.0 + 0.01 * p}) for p in range(4)}
        self.assertAlmostEqual(gate.block_slowdowns(b, "C1.prefill_s"), 0.0)


class Verdicts(unittest.TestCase):
    SAME = {"C1.prefill_s": 2.0, "C1.decode_tok_s": 40.0, "peak_gib": 30.0}

    def test_more_memory_is_reported_without_overriding_speed_or_run_validity(self):
        n = gate.FAMILIES["gemma2"].min_blocks
        for slowdown, problem, expected in ((1.0, None, "PASS"),
                                            (1.02, None, "FAIL"),
                                            (1.0, "capacity admission failed", "INVALID")):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                out = pathlib.Path(tmp)
                candidate = dict(self.SAME, peak_gib=100.0)
                candidate["C1.prefill_s"] *= slowdown
                write_blocks(out, "gemma2", [(self.SAME, candidate)] * n)
                if problem:
                    self.rewrite(out, "gemma2-b01-p1-new.json",
                                 lambda r: r.update(problems=[problem]))
                report = gate.verdicts(out, "gemma2")["gemma2"]
                self.assertEqual(report["verdict"], expected)
                memory = report["metrics"]["peak_gib"]
                self.assertEqual(memory["verdict"], "INFORMATIONAL")
                self.assertEqual(memory["mean"], 70.0)
                self.assertEqual(memory["new_mean"], 100.0)

    def test_memory_observations_alone_cannot_pass_the_speed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            n = gate.FAMILIES["gemma2"].min_blocks
            write_blocks(out, "gemma2", [({"peak_gib": 30.0}, {"peak_gib": 100.0})] * n)
            self.assertEqual(gate.verdicts(out, "gemma2")["gemma2"]["verdict"], "INCOMPLETE")

    def rewrite(self, out, name, change):
        path = out / "runs" / name
        r = json.loads(path.read_text())
        change(r)
        path.write_text(json.dumps(r))

    def test_failed_runs_problems_or_mixed_binaries_make_a_family_invalid(self):
        n = gate.FAMILIES["gemma2"].min_blocks
        cases = {
            "gemma2-b02-p1-new.json": lambda r: r.update(error="RuntimeError: boom"),
            "gemma2-b03-p2-new.json": lambda r: r.update(problems=["C1: 1 cached prompt tokens"]),
            "gemma2-b04-p2-new.json": lambda r: r.update(runtime_sha256="rebuilt"),
            "gemma2-b01-p3-old.json": lambda r: r.update(config_sha256="other"),
            "gemma2-b02-p2-new.json": lambda r: r.update(
                runtime_env={"LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS": "1"}),
            "gemma2-b01-p0-old.json": lambda r: r["family_spec"].update(repeats=2),
        }
        with tempfile.TemporaryDirectory() as tmp:  # block counts are the verdict's own
            out = pathlib.Path(tmp)
            write_blocks(out, "gemma2", [(self.SAME, self.SAME)] * n)
            self.rewrite(out, "gemma2-b01-p0-old.json", lambda r: r["family_spec"].update(
                min_blocks=3, max_blocks=6))
            self.assertEqual(gate.verdicts(out, "gemma2")["gemma2"]["verdict"], "PASS")
        for name, change in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                out = pathlib.Path(tmp)
                write_blocks(out, "gemma2", [(self.SAME, self.SAME)] * (n + 1))
                self.rewrite(out, name, change)
                report = gate.verdicts(out, "gemma2")["gemma2"]
                self.assertEqual(report["verdict"], "INVALID")
                self.assertTrue(report["invalid"])
                self.assertEqual(gate.verdict_status({"gemma2": report}), 1)

    def test_a_missing_run_makes_a_family_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            write_blocks(out, "gemma2", [(self.SAME, self.SAME)] * 6)
            (out / "runs" / "gemma2-b03-p1-new.json").unlink()
            report = gate.verdicts(out, "gemma2")["gemma2"]
            self.assertEqual(report["verdict"], "INVALID")
            self.assertIn("block 3: no record at positions [1]", report["invalid"])

    def test_blocks_past_the_maximum_do_not_count(self):
        family = gate.FAMILIES["deepseek"]
        slow = dict(self.SAME, **{"C1.prefill_s": 2.0 * 1.006})
        noisy = [(self.SAME, dict(slow, **{"C1.prefill_s": 2.0 * x}))
                 for x in (1.000, 1.012, 1.004, 1.010, 1.002, 1.008)]
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            write_blocks(out, "deepseek", noisy + [(self.SAME, self.SAME)] * 6)
            report = gate.verdicts(out, "deepseek")["deepseek"]
            self.assertEqual(report["ignored_blocks"], list(range(family.max_blocks + 1, 13)))
            self.assertEqual(report["metrics"]["C1.prefill_s"]["blocks"], family.max_blocks)
            self.assertEqual(report["metrics"]["C1.prefill_s"]["verdict"], "NOT PASSED")
            self.assertEqual(report["verdict"], "NOT PASSED")

    def test_the_exit_status_fails_anything_but_a_pass(self):
        self.assertEqual(gate.verdict_status({}), 1)
        self.assertEqual(gate.verdict_status({"a": {"verdict": "PASS"}}), 0)
        self.assertEqual(gate.verdict_status({"a": {"verdict": "PASS"},
                                              "b": {"verdict": "CONTINUE"}}), 2)
        for bad in ("FAIL", "NOT PASSED", "INVALID"):
            self.assertEqual(gate.verdict_status({"a": {"verdict": "CONTINUE"},
                                                  "b": {"verdict": bad}}), 1)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main(["verdict", tmp]), 1)

    def test_a_family_passes_only_when_every_metric_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            same = {"C1.prefill_s": 2.0, "C1.decode_tok_s": 40.0, "peak_gib": 30.0}
            slower = {"C1.prefill_s": 2.0, "C1.decode_tok_s": 38.0, "peak_gib": 30.0}
            write_blocks(out, "gemma2", [(same, same)] * gate.FAMILIES["gemma2"].min_blocks)
            write_blocks(out, "gemma3", [(same, slower)] * gate.FAMILIES["gemma3"].min_blocks)
            report = gate.verdicts(out, None)
            self.assertEqual(report["gemma2"]["verdict"], "PASS")
            self.assertEqual(report["gemma3"]["verdict"], "FAIL")
            self.assertEqual(report["gemma3"]["metrics"]["C1.prefill_s"]["verdict"], "PASS")
            self.assertEqual(report["gemma3"]["metrics"]["C1.decode_tok_s"]["verdict"], "FAIL")
            self.assertEqual(report["gemma2"]["metrics"]["peak_gib"]["unit"], "GiB")

    def test_plan_schedules_abba_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                gate.main(["plan", "--family", "gemma2", "--old", "/o", "--new", "/n",
                           "--out", tmp, "--blocks", "2", "--first", "5"])
            prime, *lines = text.getvalue().splitlines()
            self.assertEqual(len(lines), 8)
            labels = [line.split("--label ")[1].split()[0] for line in lines]
            self.assertEqual(labels, list(gate.ORDER) * 2)
            self.assertIn("--block 5 --position 0", lines[0])
            self.assertIn("--block 6 --position 3", lines[7])
            self.assertIn("--runtime /o", lines[0])
            self.assertIn("--runtime /n", lines[1])
            self.assertNotIn("--config-template", lines[0])
            # The session starts with the old arm's priming run; every run is its session's.
            self.assertEqual(gate.PRIME_ARM, "old")
            self.assertIn("--runtime /o --label old", prime)
            self.assertIn("--prime --session ", prime)
            self.assertNotIn("--block", prime)
            session = prime.split("--session ")[1].split()[0]
            self.assertTrue(session.startswith("gemma2-b05p0-"))
            for line in lines:
                self.assertIn(f"--session {session}", line)

    def test_plan_splits_jobs_into_sessions_each_primed(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = pathlib.Path(tmp) / "deepseek"
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                gate.main(["plan", "--family", "deepseek", "--old", "/o", "--new", "/n",
                           "--out", tmp, "--blocks", "2", "--job-runs", "3",
                           "--steps", str(prefix), "--new-config", "/n.toml"])
            paths = text.getvalue().splitlines()
            self.assertEqual(paths, [f"{prefix}-{n:02d}.steps" for n in (1, 2, 3)])
            jobs = [pathlib.Path(p).read_text().splitlines() for p in paths]
            self.assertEqual([len(j) for j in jobs], [4, 4, 3])  # 3 + 3 + 2 runs, each primed
            sessions = set()
            measured = []
            for job in jobs:
                session = job[0].split("--session ")[1].split()[0]
                sessions.add(session)
                self.assertIn("--prime", job[0])
                self.assertIn("--runtime /o --label old", job[0])
                self.assertNotIn("--config-template", job[0])  # the old arm's own
                for line in job[1:]:
                    self.assertNotIn("--prime", line)
                    self.assertIn(f"--session {session}", line)
                    measured.append(line.split("--block ")[1].split()[0] + "/"
                                    + line.split("--position ")[1].split()[0])
            self.assertEqual(len(sessions), 3)
            self.assertEqual(measured, [f"{b}/{p}" for b in (1, 2) for p in range(4)])
            with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
                gate.main(["plan", "--family", "deepseek", "--old", "/o", "--new", "/n",
                           "--out", tmp, "--blocks", "2", "--job-runs", "3"])

    def test_a_priming_run_needs_a_session_and_no_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = ["run", "--family", "gemma2", "--runtime", "/nonexistent", "--label", "old",
                    "--out", tmp]
            for extra in (["--prime"], ["--prime", "--session", "s", "--block", "1"],
                          ["--block", "1"], ["--position", "0"]):
                with self.subTest(extra), self.assertRaises(SystemExit):
                    gate.main(base + extra)
            self.assertFalse((pathlib.Path(tmp) / "runs").exists())
            self.assertFalse((pathlib.Path(tmp) / "primes").exists())

    def test_priming_runs_count_in_no_statistic_but_must_succeed_first(self):
        n = gate.FAMILIES["gemma2"].min_blocks
        # A priming run far slower than every measured run changes nothing.
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            write_blocks(out, "gemma2", [(self.SAME, self.SAME)] * n)
            slow = {k: v * 3 for k, v in self.SAME.items()}
            (out / "primes" / "gemma2-s1.json").write_text(
                json.dumps(prime_record("gemma2", slow)))
            report = gate.verdicts(out, "gemma2")["gemma2"]
            self.assertEqual(report["verdict"], "PASS")
            self.assertEqual(report["metrics"]["C1.prefill_s"]["runs"], {"old": 2 * n,
                                                                          "new": 2 * n})
            self.assertEqual(report["metrics"]["C1.prefill_s"]["old_mean"], 2.0)
            self.assertEqual(gate.spread(out, "gemma2")["gemma2"]["C1.prefill_s"]["runs"], 4 * n)
            self.assertEqual(gate.spread(out, "gemma2")["gemma2"]["C1.prefill_s"]["unprimed"], 0)
            # spread still reads runs from before priming, and counts them.
            (out / "primes" / "gemma2-s1.json").unlink()
            report = gate.spread(out, "gemma2")["gemma2"]["C1.prefill_s"]
            self.assertEqual((report["runs"], report["unprimed"]), (4 * n, 4 * n))

        def prime(change):
            def apply(out):
                path = out / "primes" / "gemma2-s1.json"
                r = json.loads(path.read_text())
                change(r)
                path.write_text(json.dumps(r))
            return apply

        def run(name, change):
            return lambda out: self.rewrite(out, name, change)

        cases = {
            "failed": prime(lambda r: r.update(error="RuntimeError: boom")),
            "problem": prime(lambda r: r.update(problems=["C1: 1 cached prompt tokens"])),
            "missing": lambda out: (out / "primes" / "gemma2-s1.json").unlink(),
            "another family's": prime(lambda r: r.update(family="gemma3")),
            "late": prime(lambda r: r.update(started_unix=200.0)),
            "the new arm's": prime(lambda r: r.update(label="new", runtime_sha256="new-binary",
                                                      config_sha256="new-config")),
            "another binary": prime(lambda r: r.update(runtime_sha256="other-binary")),
            "another configuration": prime(lambda r: r.update(config_sha256="other-config")),
            "no session": run("gemma2-b02-p1-new.json", lambda r: r.pop("session")),
            "other session": run("gemma2-b02-p1-new.json",
                                 lambda r: r.update(session="gemma2-s2")),
        }
        for name, change in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                out = pathlib.Path(tmp)
                write_blocks(out, "gemma2", [(self.SAME, self.SAME)] * n)
                change(out)
                report = gate.verdicts(out, "gemma2")["gemma2"]
                self.assertEqual(report["verdict"], "INVALID")
                self.assertTrue(report["invalid"])
                self.assertEqual(gate.verdict_status({"gemma2": report}), 1)

    def test_plan_gives_each_arm_its_configuration(self):
        text = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(text):
            gate.main(["plan", "--family", "deepseek", "--old", "/o", "--new", "/n",
                       "--out", tmp, "--blocks", "1", "--new-config", "/n.toml"])
        prime, *lines = text.getvalue().splitlines()
        self.assertNotIn("--config-template", prime)
        self.assertNotIn("--config-template", lines[0])
        self.assertIn("--config-template /n.toml", lines[1])
        self.assertIn("--config-template /n.toml", lines[2])
        self.assertNotIn("--config-template", lines[3])


    def test_plan_gives_each_arm_its_diagnostic_switches(self):
        text = io.StringIO()
        switch = "LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1"
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(text):
            gate.main(["plan", "--family", "gemma2", "--old", "/o", "--new", "/n",
                       "--out", tmp, "--blocks", "1", "--new-env", switch])
        prime, *lines = text.getvalue().splitlines()
        self.assertNotIn("--runtime-env", prime)
        self.assertNotIn("--runtime-env", lines[0])
        self.assertIn(f"--runtime-env {switch}", lines[1])
        self.assertIn(f"--runtime-env {switch}", lines[2])
        self.assertNotIn("--runtime-env", lines[3])

    def test_only_diagnostic_switches_reach_the_runtime(self):
        self.assertEqual(gate.parse_runtime_env(["LLMP_DIAGNOSTIC_B=1", "LLMP_DIAGNOSTIC_A=x"]),
                         {"LLMP_DIAGNOSTIC_A": "x", "LLMP_DIAGNOSTIC_B": "1"})
        for bad in (["CUDA_VISIBLE_DEVICES=1"], ["LLMP_DIAGNOSTIC_A"],
                    ["LLMP_DIAGNOSTIC_A=1", "LLMP_DIAGNOSTIC_A=2"]):
            with self.subTest(bad), self.assertRaises(SystemExit):
                gate.parse_runtime_env(bad)
        with unittest.mock.patch.dict(gate.os.environ, {"LLMP_DIAGNOSTIC_STRAY": "1"}):
            env = gate.arm_env({"LLMP_DIAGNOSTIC_A": "1"})
        self.assertNotIn("LLMP_DIAGNOSTIC_STRAY", env)
        self.assertEqual(env["LLMP_DIAGNOSTIC_A"], "1")
        self.assertEqual(env["CUDA_DISABLE_PTX_JIT"], "1")

    def test_a_switch_the_runtime_does_not_name_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = pathlib.Path(tmp) / "llmp-runtime"
            runtime.write_bytes(b"\x7fELF...LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS\x00more"
                                b"XLLMP_DIAGNOSTIC_EMBEDDED\x00")
            known = {"LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS": "1"}
            gate.check_runtime_env(runtime, known)
            gate.check_runtime_env(runtime, {})
            for bad in ({"LLMP_DIAGNOSTIC_COMPONENT_SERVING": "qwen_image_2_1"},
                        {"LLMP_DIAGNOSTIC_SKIP_TURN": "1"},  # a prefix is not the name
                        {"LLMP_DIAGNOSTIC_EMBEDDED": "1"}):  # nor inside a longer one
                with self.subTest(bad), self.assertRaises(ValueError):
                    gate.check_runtime_env(runtime, bad)
            # A run records the refusal as its error, beside no stray run directory.
            out = pathlib.Path(tmp) / "out"
            code = gate.main(["run", "--family", "image", "--runtime", str(runtime),
                              "--label", "new", "--out", str(out), "--block", "1",
                              "--position", "1", "--runtime-env",
                              "LLMP_DIAGNOSTIC_COMPONENT_SERVING=qwen_image_2_1"])
            self.assertEqual(code, 1)
            record = json.loads((out / "runs" / "image-b01-p1-new.json").read_text())
            self.assertIn("reads no diagnostic switch", record["error"])
            # plan refuses it for an existing binary.
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                gate.main(["plan", "--family", "image", "--old", str(runtime),
                           "--new", str(runtime), "--out", tmp, "--blocks", "1",
                           "--new-env", "LLMP_DIAGNOSTIC_COMPONENT_SERVING=qwen_image_2_1"])

    def test_an_arm_with_one_switch_set_in_every_run_is_valid(self):
        n = gate.FAMILIES["gemma2"].min_blocks
        switch = {"LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            write_blocks(out, "gemma2", [(self.SAME, self.SAME)] * n)
            for path in (out / "runs").glob("*-new.json"):
                r = json.loads(path.read_text())
                r["runtime_env"] = switch
                path.write_text(json.dumps(r))
            report = gate.verdicts(out, "gemma2")["gemma2"]
            self.assertEqual(report["verdict"], "PASS", report["invalid"])
            self.assertEqual(report["diagnostics"]["new"], [json.dumps(switch)])
            # The priming run is the old arm's, switches included.
            prime = out / "primes" / "gemma2-s1.json"
            r = json.loads(prime.read_text())
            r["runtime_env"] = switch
            prime.write_text(json.dumps(r))
            self.assertEqual(gate.verdicts(out, "gemma2")["gemma2"]["verdict"], "INVALID")


class Cells(unittest.TestCase):
    def request(self, send, times, end, tokens):
        return {"t_send": send, "t_first": times[0], "t_last": times[-1], "t_end": end,
                "times": times, "usage": {"completion_tokens": tokens}}

    def test_one_request_decodes_from_its_first_to_its_last_token(self):
        r = self.request(0.0, [1.0, 1.5, 2.0, 3.0], 3.1, 7)  # three steps of two tokens
        m = gate.cell_metrics([r])
        self.assertAlmostEqual(m["prefill_s"], 1.0)
        self.assertAlmostEqual(m["decode_tok_s"], 6 / 2.0)
        self.assertAlmostEqual(m["wall_tok_s"], 7 / 3.1)

    def test_a_cohort_decodes_only_while_every_request_does(self):
        # The first request decodes alone while the second prefills; the window is [2, 4].
        a = self.request(0.0, [1.0, 1.5, 2.5, 3.0, 3.5, 4.0], 4.1, 6)
        b = self.request(0.0, [2.0, 2.5, 3.0, 3.5, 4.0, 4.5], 4.6, 6)
        m = gate.cell_metrics([a, b])
        self.assertAlmostEqual(m["prefill_s"], 2.0)
        # Inside [2, 4]: a's chunks 2.5..4.0 (3 steps in 1.5 s), b's 2.0..4.0 (4 in 2 s).
        self.assertAlmostEqual(m["decode_tok_s"], 3 / 1.5 + 4 / 2.0)

    def test_a_chunk_at_the_window_edge_does_not_change_the_rate(self):
        # Two requests in one wave, steps every 0.5 s; b's chunks land 1 ms after a's.
        a = self.request(0.0, [1.0 + 0.5 * i for i in range(9)], 5.1, 9)
        b = self.request(0.0, [1.001 + 0.5 * i for i in range(9)], 5.2, 9)
        self.assertAlmostEqual(gate.cell_metrics([a, b])["decode_tok_s"], 4.0, places=2)

    def test_no_shared_window_has_no_decode_rate(self):
        a = self.request(0.0, [1.0, 1.5], 1.6, 2)
        b = self.request(0.0, [2.0, 2.5], 2.6, 2)
        self.assertNotIn("decode_tok_s", gate.cell_metrics([a, b]))


class Inputs(unittest.TestCase):
    def test_passages_are_deterministic_line_starts(self):
        corpus = "".join(f"line {i}\n" for i in range(5000))
        a = gate.passage(corpus, 3, 100)
        self.assertEqual(a, gate.passage(corpus, 3, 100))
        self.assertEqual(len(a), 100)
        self.assertTrue(a.startswith("line "))
        self.assertNotEqual(a, gate.passage(corpus, 4, 100))
        self.assertTrue(gate.prompt(corpus, 3, 100).startswith("Request 3. "))

    def test_a_template_fills_share_data_and_port(self):
        family = gate.FAMILIES["deepseek"]
        default = gate.arm_template(family, None)
        self.assertEqual(gate.render_config(default, pathlib.Path("/s"), pathlib.Path("/d"), 7),
                         gate.config_text(family, pathlib.Path("/s"), pathlib.Path("/d"), 7))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "new.toml"
            path.write_text('[storage]\ndata_dir = "{data}"\n[client]\nport = {port}\n'
                            '[models.deepseek]\nartifact = "{share}/x"\n')
            text = gate.render_config(gate.arm_template(family, str(path)), pathlib.Path("/s"),
                                      pathlib.Path("/d"), 7)
            self.assertEqual(text, '[storage]\ndata_dir = "/d"\n[client]\nport = 7\n'
                                   '[models.deepseek]\nartifact = "/s/x"\n')
        with self.assertRaises(SystemExit):
            gate.render_config('data_dir = "{data}"\n', pathlib.Path("/s"), pathlib.Path("/d"), 7)

    def test_each_arm_defaults_to_its_labels_artifacts(self):
        for name, family in gate.FAMILIES.items():
            old = gate.arm_template(family, None, "old")
            new = gate.arm_template(family, None, "new")
            self.assertEqual(old, gate.config_template(family))  # the frozen baseline's
            self.assertIn(f'installed = "{{share}}/{family.store}"', old)
            if name == "image":  # the image slice's: both arms as today
                self.assertEqual(old, new)
                continue
            self.assertIn('installed = "{share}/m36-v1-artifacts"', new)
            self.assertNotEqual(old, new)
            # The same settings, only the artifact and drafter IDs differ.
            ids = {"artifact", "drafter"}
            keep = lambda text: [line for line in text.splitlines()
                                 if line.split(" = ")[0] not in ids
                                 and not line.startswith("installed")]
            self.assertEqual(keep(old), keep(new), name)
            self.assertEqual(gate.arm_template(family, None, "new"),
                             gate.config_template(family, "new"))
        v1 = gate.config_template(gate.FAMILIES["qwen38"], "new")
        self.assertIn(f'artifact = "{gate.QWEN_V1}"', v1)
        self.assertIn(f'drafter = "{gate.QWEN_MTP_V1}"', v1)
        self.assertNotIn(gate.QWEN_V0, v1)
        self.assertIn(gate.DSV4_V0, gate.config_template(gate.FAMILIES["deepseek"]))
        self.assertIn(f'drafter = "{gate.DSPARK_V1}"',
                      gate.config_template(gate.FAMILIES["deepseek"], "new"))
        self.assertIn(gate.GEMMA31_V1, gate.config_text(
            gate.FAMILIES["gemma31"], pathlib.Path("/s"), pathlib.Path("/d"), 7, "new"))

    def test_an_explicit_template_overrides_the_label_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "t.toml"
            path.write_text("x = 1\n")
            for label in ("old", "new"):
                self.assertEqual(gate.arm_template(gate.FAMILIES["gemma2"], str(path), label),
                                 "x = 1\n")

    def test_config_binds_loopback_and_names_the_share(self):
        text = gate.config_text(gate.FAMILIES["qwen38"], pathlib.Path("/s"), pathlib.Path("/d"),
                                18114)
        self.assertIn('data_dir = "/d"', text)
        self.assertIn('installed = "/s/m3-artifacts"', text)
        self.assertIn('bind = ["loopback"]', text)
        self.assertIn("port = 18114", text)
        self.assertIn('tokenizer = "/s/models/', text)
        self.assertNotIn("{share}", text)


class Service(unittest.TestCase):
    def test_a_runtime_already_on_the_port_is_refused(self):
        with socket.socket() as listener, tempfile.TemporaryDirectory() as tmp:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "already listens"):
                gate.Service(pathlib.Path("/nonexistent"), pathlib.Path("/c"),
                             pathlib.Path(tmp), port, {})

    def test_a_request_that_cannot_connect_releases_its_cell(self):
        with socket.socket() as listener, socket.socket() as closed:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            closed.bind(("127.0.0.1", 0))  # bound, never listening: connections are refused
            barrier = threading.Barrier(2)
            results = [None, None]

            def one(i, port):
                results[i] = gate.stream_chat(port, {}, barrier)

            threads = [threading.Thread(target=one, args=(0, listener.getsockname()[1])),
                       threading.Thread(target=one, args=(1, closed.getsockname()[1]))]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            self.assertFalse(any(t.is_alive() for t in threads))
            self.assertIn("BrokenBarrierError", results[0]["error"])
            self.assertIn("ConnectionRefusedError", results[1]["error"])


if __name__ == "__main__":
    unittest.main()
