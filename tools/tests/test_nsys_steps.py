# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""tools/nsys_steps.py on synthetic nsys sqlite exports, no GPU needed.

Two traces of three repeated steps: the reference fuses a gate/up product the native
trace runs as two launches, and the native step has a device copy inside it leaving an
idle hole. Steps split at idle gaps, the span window selects them, kernels match by
name, first template argument and grid, and copies and holes are reported.
"""

import importlib.util
import io
import pathlib
import sqlite3
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("nsys_steps", ROOT / "tools/nsys_steps.py")
nsys_steps = importlib.util.module_from_spec(spec)
sys.modules["nsys_steps"] = nsys_steps
spec.loader.exec_module(nsys_steps)

US = 1000
MS = 1000 * US


def trace(path, steps):
    """Writes an export with `steps`: lists of (name, demangled, grid, µs) or ('copy', bytes, µs)."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, "
               "shortName INTEGER, demangledName INTEGER, gridX INTEGER, streamId INTEGER)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, "
               "bytes INTEGER, copyKind INTEGER)")
    strings = {}

    def sid(text):
        if text not in strings:
            strings[text] = len(strings) + 1
            db.execute("INSERT INTO StringIds VALUES (?, ?)", (strings[text], text))
        return strings[text]

    at = 0
    for step in steps:
        for item in step:
            if item[0] == "copy":
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, 8)",
                           (at, at + item[2] * US, item[1]))
                at += item[2] * US
                continue
            name, demangled, grid, length = item
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, 13)",
                       (at, at + length * US, sid(name), sid(demangled), grid))
            at += length * US
        at += 2 * MS  # the host's turn between steps
    db.commit()
    db.close()


GATE_UP = ("mul_mat_vec_q", "void mul_mat_vec_q<(ggml_type)12, 1, true>(...)", 21504)
DOWN = ("mul_mat_vec_q", "void mul_mat_vec_q<(ggml_type)14, 1, false>(...)", 5376)


class NsysStepsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.ref = str(pathlib.Path(self.dir.name) / "ref.sqlite")
        self.nat = str(pathlib.Path(self.dir.name) / "nat.sqlite")
        prefill = [("prefill", "void prefill<1>()", 1, 30 * 1000)]
        ref_step = [(*GATE_UP, 60 * 1000), (*DOWN, 30 * 1000)]
        nat_step = [(*GATE_UP, 31 * 1000), (*GATE_UP, 31 * 1000), ("copy", 32768, 12),
                    (*DOWN, 30 * 1000)]
        trace(self.ref, [prefill] + [ref_step] * 3)
        trace(self.nat, [prefill] + [nat_step] * 3)

    def test_key_takes_the_first_template_argument_and_grid(self):
        self.assertEqual(nsys_steps.kernel_key("k", "void k<(ggml_type)12, 1, true>(int)", 4),
                         "k<(ggml_type)12> g4")
        self.assertEqual(nsys_steps.kernel_key("k", "void k<A<B, C>, 2>()", 1), "k<A<B, C>> g1")
        self.assertEqual(nsys_steps.kernel_key("k", "k(int)", 8), "k g8")

    def test_steps_split_at_idle_gaps_and_the_span_selects_them(self):
        steps = nsys_steps.load(self.nat, 250 * US)
        self.assertEqual(len(steps), 4)
        chosen = nsys_steps.select(steps, 80, 100, 0)
        self.assertEqual(len(chosen), 3)
        self.assertEqual([len(s.copies) for s in chosen], [1, 1, 1])

    def test_report_matches_launches_and_shows_copies_and_holes(self):
        out = io.StringIO()
        ref = nsys_steps.load(self.ref, 250 * US)
        nat = nsys_steps.load(self.nat, 250 * US)
        a = nsys_steps.summarize(ref, nsys_steps.select(ref, 80, 100, 0), 5)
        b = nsys_steps.summarize(nat, nsys_steps.select(nat, 80, 100, 0), 5)
        self.assertAlmostEqual(a.kernels["mul_mat_vec_q<(ggml_type)12> g21504"][0], 60.0)
        self.assertEqual(b.kernels["mul_mat_vec_q<(ggml_type)12> g21504"], (62.0, 2.0))
        self.assertEqual(b.copies["copy8"][1], 1.0)
        self.assertAlmostEqual(b.holes[0][0], 12.0)
        self.assertAlmostEqual(b.span_ms - a.span_ms, 2.012)
        nsys_steps.report(a, b, ("reference", "native"), 10, out)
        text = out.getvalue()
        self.assertIn("mul_mat_vec_q<(ggml_type)12> g21504", text)
        self.assertIn("copies and memsets inside native's steps", text)
        self.assertIn("longest idle holes", text)

    def test_main_refuses_an_empty_window(self):
        self.assertEqual(nsys_steps.main([self.ref, self.nat, "--span", "500", "600"]), 1)


if __name__ == "__main__":
    unittest.main()
