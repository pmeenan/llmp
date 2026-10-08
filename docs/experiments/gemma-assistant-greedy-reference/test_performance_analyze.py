#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Small fixed-output accounting and first-common-prefix controls."""
import unittest

import performance_analyze as a


class FixedOutput(unittest.TestCase):
    def metadata(self, spec, keeps):
        count = rows = drafts = 0
        units = []
        for keep in keeps:
            depth = min(3, 31-count) if spec and count < 31 else 0
            units.append(dict(past=64+count, depth=depth, verified_rows=depth+1, keep=keep))
            count += keep
            rows += depth+1
            drafts += depth
        return dict(mode="unit32" if spec else "plain32", emitted=32, endpoint=96,
                    prefill_queries=64, max_rows=128, context=4096, eog_ignored=True,
                    eos_id=106, eos_occurrences=1, verified_rows=rows,
                    drafted_rows=drafts, units=units, seconds=1.0)

    def test_tail_depth_preserves_identical32_work_and_ignores_eos(self):
        tokens = [106] + [1]*31
        self.assertEqual(a.counters(self.metadata(False, [1]*32), tokens), (32, 0))
        self.assertEqual(a.counters(self.metadata(True, [4]*8), tokens), (32, 8))
        self.assertEqual(a.counters(self.metadata(True, [1]*32), tokens), (122, 31))

    def test_extra_committed_or_false_work_is_refused(self):
        tokens = [106] + [1]*31
        m = self.metadata(True, [4]*8)
        m["units"][-1]["keep"] = 5
        with self.assertRaises(ValueError):
            a.counters(m, tokens)
        m = self.metadata(False, [1]*32)
        m["verified_rows"] += 1
        with self.assertRaises(ValueError):
            a.counters(m, tokens)

    def test_first_divergence_never_compares_later_contexts(self):
        self.assertIsNone(a.first_divergence([1, 2, 3], [1, 2, 3]))
        self.assertEqual(a.first_divergence([1, 7, 3], [1, 2, 4]), 1)


if __name__ == "__main__":
    unittest.main()
