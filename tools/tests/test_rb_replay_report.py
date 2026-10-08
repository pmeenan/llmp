# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The retained-backing replay report: a whole, agreeing run, and the deterministic criteria applied mechanically."""

import importlib.util
import json
import pathlib
import sys
import unittest

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parents[2] / "docs/experiments/retained-backing"
SPEC = importlib.util.spec_from_file_location("replay_report", HERE / "replay_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)
GIB = 1 << 30


def record(design, file, **overrides):
    r = dict(design=design, file=file, role="primary", trace_sha256=report.FILES["primary"][file],
             replays_agree=True, budget_bytes=64 * GIB, ticks=100, waste_sum=100 * GIB, waste_peak=2 * GIB,
             content_lost_bytes=5 * GIB, extra_restored_bytes=4 * GIB, shrink_extra_evicted_bytes=GIB,
             restored_bytes=100 * GIB, refusals=0, refused_at=-1, delayed_admissions=0, relocated_bytes=0,
             calls={k: 10 for k in report.CALLS})
    r.update(overrides)
    return r


def run(**per_design):
    """Lines for every design at every budget; per_design overrides one design's fields."""
    lines = []
    for file in report.FILES["primary"]:
        for design in report.DESIGNS:
            lines.append(json.dumps(record(design, file, **per_design.get(design, {}))))
    return lines


class ReplayReportTest(unittest.TestCase):
    def test_ties_pass_and_any_worse_metric_fails(self):
        base = record("d033", "trace-r5-4.jsonl")
        self.assertTrue(report.judge(base, dict(base))["eligible"])
        for field, value in (("waste_peak", 2 * GIB + 1), ("content_lost_bytes", 5 * GIB + 1),
                             ("waste_sum", 100 * GIB + 1)):
            checks = report.judge(base, dict(base, **{field: value}))
            self.assertFalse(checks["eligible"], field)
        better = dict(base, waste_sum=GIB, waste_peak=GIB, content_lost_bytes=0)
        self.assertTrue(report.judge(base, better)["eligible"])

    def test_waste_mean_is_compared_exactly_over_each_replays_ticks(self):
        base = record("d033", "trace-r5-4.jsonl", ticks=3, waste_sum=10)
        self.assertTrue(report.judge(base, dict(base, ticks=6, waste_sum=20))["waste_mean"])
        self.assertFalse(report.judge(base, dict(base, ticks=6, waste_sum=21))["waste_mean"])

    def test_a_replay_that_stopped_at_a_refusal_is_not_eligible(self):
        base = record("d033", "trace-r5-4.jsonl", refusals=1)
        stopped = dict(base, waste_sum=0, waste_peak=0, content_lost_bytes=0)
        self.assertFalse(report.judge(base, stopped)["eligible"])  # the stricter reading

    def test_eligibility_is_per_budget(self):
        lines = run()
        worse = json.loads(lines[len(report.DESIGNS) + 1])  # slab32m-run at the second budget
        worse["waste_peak"] += 1
        lines[len(report.DESIGNS) + 1] = json.dumps(worse)
        role, records = report.load(lines)
        text, eligible = report.report(role, records)
        self.assertEqual(eligible["slab32m-run"], [64, 64])  # every file claims 64 GiB here
        self.assertEqual(eligible["slab1g-run"], [64, 64, 64])
        self.assertIn("no: waste_peak", text)

    def test_a_disagreeing_incomplete_or_foreign_run_is_void(self):
        with self.assertRaisesRegex(report.VoidRun, "disagree"):
            report.load(run(**{"slab1g-run": dict(replays_agree=False)}))
        with self.assertRaisesRegex(report.VoidRun, "no result"):
            report.load(run()[:-1])
        with self.assertRaisesRegex(report.VoidRun, "not the recorded"):
            report.load(run(**{"d033": dict(trace_sha256="0" * 64)}))
        with self.assertRaisesRegex(report.VoidRun, "repeated"):
            report.load(run() + run()[:1])
        with self.assertRaisesRegex(report.VoidRun, "not one of"):
            report.load(run() + [json.dumps(record("slab2g-run", "trace-r5-4.jsonl"))])
        with self.assertRaisesRegex(report.VoidRun, "two parts"):
            report.load(run(**{"slab1g-run": dict(content_lost_bytes=0)}))


if __name__ == "__main__":
    unittest.main()
