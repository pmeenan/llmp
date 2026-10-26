# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""tools/assistant_gate.py's steps, log parsing and verdict on synthetic logs, no GPU needed."""

import contextlib
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
spec = importlib.util.spec_from_file_location("assistant_gate", ROOT / "tools/assistant_gate.py")
ag = importlib.util.module_from_spec(spec)
sys.modules["assistant_gate"] = ag
spec.loader.exec_module(ag)

RETIRED = ("GREEDY_NATIVE_RETIRED profile=31 mode=unit32 C=1 P=64 prefill_query=64 emitted=32 "
           "endpoint=96 warm=1 quality=2 timed=2")


def pinned_v1(name):
    """The pinned in-tree map's v1 ID for `name` (what the models tests serve)."""
    data = json.loads((ROOT / "tests/unit/data/v1_frozen/v1-ids.json").read_text())
    return next(data["artifacts"][key] for key, named in data["names"].items() if named == name)


def log(first, repeat, retired=RETIRED):
    done = "GREEDY32_COMPLETED mode=unit32 phase={} emitted=32 endpoint=96 verified=46 drafted=34 "
    return "\n".join([
        "GREEDY32_POLICY phase=prefill last_built_rows=64",
        done.format("quality-first") + "seconds=0",
        done.format("timed-first") + f"seconds={first}",
        done.format("timed-repeat") + f"seconds={repeat}",
        retired]) + "\n"


def write_blocks(out, blocks, old=1.4, retired=RETIRED):
    """blocks: a list of the new arm's seconds; the old arm takes `old`."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "prime-b01.log").write_text(log(old, old))
    for b, new in enumerate(blocks, start=1):
        for p, arm in enumerate(ag.ORDER):
            seconds = new if arm == "new" else old
            (out / f"b{b:02d}-p{p}-{arm}.log").write_text(
                log(seconds, seconds, retired if arm == "new" else RETIRED))


class Steps(unittest.TestCase):
    def arms(self):
        return {"old": ("/o/ref", "/s/old/T", "/s/old/A"),
                "new": ("/n/engine", "/s/new/T2", "/s/new/A2")}

    def test_priming_run_then_abba_blocks(self):
        lines = ag.step_lines("/out", 3, 2, self.arms(), "/in.i32")
        self.assertEqual(len(lines), 1 + 8)
        self.assertIn("/out/prime-b03", lines[0])
        self.assertIn("/o/ref /s/old/T /s/old/A /out/prime-b03 31 /in.i32 unit32", lines[0])
        order = [("old" if "/o/ref" in line else "new") for line in lines[1:]]
        self.assertEqual(order, ["old", "new", "new", "old"] * 2)
        self.assertIn("/out/b04-p2-new", lines[7])
        self.assertTrue(all("/s/new/T2 /s/new/A2" in line for line in lines if "/n/engine" in line))
        self.assertTrue(all(line.startswith("rm -rf ") and "CUDA_DISABLE_PTX_JIT=1" in line
                            for line in lines))

    def test_paths_with_spaces_are_quoted(self):
        arms = {"old": ("/o/my ref", "/s/T", "/s/A"), "new": ("/n/e", "/s/T", "/s/A")}
        self.assertIn("'/o/my ref'", ag.step_lines("/out", 1, 1, arms, "/in")[0])

    def test_command_writes_a_plan_and_refuses_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "run"
            argv = ["steps", "--out", str(out), "--old", "/o/ref", "--new", "/n/e",
                    "--old-artifacts", "/s/old", "--new-artifacts", "/s/new",
                    "--target", "T", "--assistant", "A", "--inputs", "/nonexistent.i32"]
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                self.assertEqual(ag.main(argv + ["--blocks", "1"]), 0)
            lines = text.getvalue().splitlines()
            self.assertEqual(len(lines), 5)
            self.assertIn("/s/old/T /s/old/A", lines[1])
            self.assertIn("/s/new/T /s/new/A", lines[2])  # same IDs unless overridden
            plan = json.loads((out / "plan.json").read_text())
            self.assertEqual(plan["arms"]["new"]["target"], "/s/new/T")
            self.assertIsNone(plan["inputs_sha256"])
            with contextlib.redirect_stdout(io.StringIO()):
                ag.main(argv + ["--first", "2"])  # the same plan again: a later job
                with self.assertRaises(SystemExit):
                    ag.main(argv[:-1] + ["/other.i32"])

    def test_the_new_store_may_name_the_artifacts_differently(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                ag.main(["steps", "--out", tmp, "--old", "/o", "--new", "/n",
                         "--old-artifacts", "/s/old", "--new-artifacts", "/s/new",
                         "--target", "T", "--assistant", "A", "--new-target", "T1",
                         "--new-assistant", "A1", "--inputs", "/i", "--blocks", "1"])
            self.assertIn("/s/new/T1 /s/new/A1", text.getvalue().splitlines()[2])

    def test_defaults_to_the_v1_gemma31_target_and_assistant(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                ag.main(["steps", "--out", tmp, "--old", "/o", "--new", "/n",
                         "--old-artifacts", "/s/old", "--new-artifacts", "/s/new",
                         "--inputs", "/i", "--blocks", "1"])
            self.assertIn(f"/s/old/{ag.DEFAULT_TARGET} /s/old/{ag.DEFAULT_ASSISTANT}",
                          text.getvalue().splitlines()[1])
            self.assertEqual(ag.DEFAULT_TARGET, pinned_v1("gemma31"))
            self.assertEqual(ag.DEFAULT_ASSISTANT, pinned_v1("gemma31-assistant"))


class Parse(unittest.TestCase):
    def test_a_complete_log(self):
        r = ag.parse_log(log(1.5, 1.3))
        self.assertAlmostEqual(r["seconds"], 1.4)
        self.assertEqual(r["identity"]["P"], "64")
        self.assertEqual(r["identity"]["timed"], "2")

    def test_the_engine_harness_line_is_read_like_the_reference_harness(self):
        engine = RETIRED.replace("GREEDY_NATIVE_RETIRED", "GREEDY_ENGINE_RETIRED")
        self.assertEqual(ag.parse_log(log(1.5, 1.3, engine)), ag.parse_log(log(1.5, 1.3)))

    def test_an_incomplete_log_is_refused(self):
        for text in (log(1.4, 1.4).replace(RETIRED, ""),
                     log(1.4, 1.4).replace("phase=timed-repeat", "phase=quality-repeat"),
                     log(1.4, 1.4).replace("drafted=34", "drafted=33"),
                     ""):
            with self.subTest(text=text[-40:]), self.assertRaises(ValueError):
                ag.parse_log(text)

    def test_a_retired_line_missing_a_field_is_refused(self):
        with self.assertRaises(ValueError):
            ag.parse_log(log(1, 1, RETIRED.replace(" P=64", "")))


class Verdict(unittest.TestCase):
    def report(self, blocks, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            write_blocks(out, blocks, **kw)
            return ag.verdict_report(out)

    def test_equal_speed_passes_after_the_minimum_blocks(self):
        r = self.report([1.4] * 8)
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(ag.status(r["verdict"]), 0)
        self.assertEqual(len(r["blocks"]), 8)

    def test_too_few_blocks_continue(self):
        r = self.report([1.4] * 5)
        self.assertEqual(ag.status(r["verdict"]), 2)

    def test_a_clear_slowdown_fails(self):
        r = self.report([1.4 * 1.05, 1.4 * 1.051] * 4)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertEqual(ag.status("FAIL"), 1)
        self.assertGreater(r["judged"]["lower"], ag.LIMIT)

    def test_the_slowdown_is_the_log_ratio_of_means(self):
        r = self.report([1.4 * 1.02])
        self.assertAlmostEqual(r["blocks"][0]["slowdown"], math.log(1.02))
        self.assertAlmostEqual(r["blocks"][0]["new_s"], 1.4 * 1.02)

    def test_the_limit_is_speed_gates_one_percent(self):
        self.assertAlmostEqual(ag.LIMIT, math.log(1.01))

    def test_another_workload_in_the_new_arm_is_invalid(self):
        for change in ("P=64", "prefill_query=64", "endpoint=96", "profile=31", "mode=unit32"):
            with self.subTest(change=change):
                other = RETIRED.replace(change, change.split("=")[0] + "=7")
                r = self.report([1.4] * 8, retired=other)
                self.assertEqual(r["verdict"], "INVALID")
                self.assertTrue(any(change.split("=")[0] in x for x in r["invalid"]))
                self.assertEqual(ag.status(r["verdict"]), 1)

    def test_an_incomplete_or_missing_run_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            write_blocks(out, [1.4] * 8)
            (out / "b03-p1-new.log").write_text("crashed\n")
            (out / "b05-p2-new.log").unlink()
            r = ag.verdict_report(out)
            self.assertEqual(r["verdict"], "INVALID")
            self.assertEqual(len(r["blocks"]), 6)
            self.assertTrue(any("b03-p1-new.log" in x for x in r["invalid"]))
            self.assertTrue(any("b05-p2-new.log: missing" in x for x in r["invalid"]))

    def test_an_incomplete_priming_log_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            write_blocks(out, [1.4] * 8)
            (out / "prime-b01.log").write_text("")
            self.assertEqual(ag.verdict_report(out)["verdict"], "INVALID")

    def test_blocks_past_the_maximum_do_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            write_blocks(out, [1.4] * 11 + [3.0])
            r = ag.verdict_report(out)
            self.assertEqual(r["judged"]["blocks"], 11)
            self.assertEqual(r["verdict"], "PASS")

    def test_the_command_prints_and_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            write_blocks(out, [1.4] * 8)
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                self.assertEqual(ag.main(["verdict", str(out)]), 0)
            self.assertIn("8 blocks: slowdown +0.000%", text.getvalue())
            self.assertTrue(text.getvalue().rstrip().endswith("PASS"))
            with contextlib.redirect_stdout(io.StringIO()) as j:
                ag.main(["verdict", str(out), "--json"])
            self.assertEqual(json.loads(j.getvalue())["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()
