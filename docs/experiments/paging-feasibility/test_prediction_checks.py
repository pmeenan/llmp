# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path
import unittest
from prediction_checks import (capture_comparison, validate_prediction_pair,
                               validate_prediction_records, validate_session_predictions)
from run import sha


class PredictionChecksTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.left=Path(self.temp.name)/'left';self.right=Path(self.temp.name)/'right'
        self.left.write_text('0 1\n0 2\n1 3\n');self.right.write_text('0 1\n0 4\n1 5\n')

    def receipts(self):
        return dict(predictions_sha256=sha(self.left)),dict(predictions_sha256=sha(self.right))

    def test_capture_strict_default_rejects(self):
        with self.assertRaises(ValueError):capture_comparison(self.left,self.right)

    def test_explicit_capture_records_failed_equivalence_without_threshold(self):
        comparison=capture_comparison(self.left,self.right,True)
        self.assertEqual(comparison['different_records'],2)
        self.assertEqual(comparison['equivalence'],'failed')

    def test_replay_requires_both_recorded_failure_and_explicit_permission(self):
        control,traced=self.receipts()
        with self.assertRaises(ValueError):
            validate_prediction_pair(control,traced,self.left,self.right,True)
        traced.update(prediction_drift_recorded=True,
                      prediction_comparison=capture_comparison(self.left,self.right,True))
        with self.assertRaises(ValueError):
            validate_prediction_pair(control,traced,self.left,self.right)
        self.assertEqual(validate_prediction_pair(control,traced,self.left,self.right,True)['different_records'],2)

    def test_drift_permission_does_not_allow_truncation(self):
        self.right.write_text('0 1\n')
        with self.assertRaises(ValueError):capture_comparison(self.left,self.right,True)

    def test_drift_permission_does_not_allow_changed_request_ids(self):
        self.right.write_text('0 1\n1 2\n1 3\n')
        with self.assertRaises(ValueError):capture_comparison(self.left,self.right,True)

    def test_falsified_drift_count_rejected(self):
        control,traced=self.receipts()
        comparison=capture_comparison(self.left,self.right,True);comparison['different_records']=1
        traced.update(prediction_drift_recorded=True,prediction_comparison=comparison)
        with self.assertRaises(ValueError):
            validate_prediction_pair(control,traced,self.left,self.right,True)

    def test_equal_predictions_need_no_opt_in(self):
        self.right.write_bytes(self.left.read_bytes());control,traced=self.receipts()
        self.assertEqual(validate_prediction_pair(control,traced,self.left,self.right)['equivalence'],'exact_top1_match')

    def test_identically_truncated_pair_rejected_against_session(self):
        self.left.write_text('0 1\n0 2\n');self.right.write_bytes(self.left.read_bytes())
        control,traced=self.receipts()
        self.assertEqual(validate_prediction_pair(control,traced,self.left,self.right)['equivalence'],'exact_top1_match')
        requests=[dict(request=0,decode=1),dict(request=1,decode=0)]
        for path in (self.left,self.right):
            with self.assertRaisesRegex(ValueError,'Truncated'):
                validate_session_predictions(path,requests)

    def test_complete_session_includes_prefill_prediction(self):
        validate_session_predictions(self.left,[dict(request=0,decode=1),dict(request=1,decode=0)])

    def test_same_total_with_wrong_request_counts_rejected(self):
        self.left.write_text('0 1\n1 2\n1 3\n')
        with self.assertRaisesRegex(ValueError,'identity'):
            validate_session_predictions(self.left,[dict(request=0,decode=1),dict(request=1,decode=0)])

    def test_extra_session_prediction_rejected(self):
        with self.assertRaisesRegex(ValueError,'Extra'):
            validate_session_predictions(self.left,[dict(request=0,decode=1)])

    def test_complete_parallel_predictions(self):
        self.left.write_text(''.join(f'{step} {seq} 10\n' for step in range(2) for seq in range(4)))
        validate_prediction_records(self.left,((step,seq) for step in range(2) for seq in range(4)))

    def test_identically_truncated_parallel_pair_rejected(self):
        self.left.write_text(''.join(f'0 {seq} 10\n' for seq in range(4)))
        self.right.write_bytes(self.left.read_bytes());control,traced=self.receipts()
        validate_prediction_pair(control,traced,self.left,self.right)
        for path in (self.left,self.right):
            with self.assertRaisesRegex(ValueError,'Truncated'):
                validate_prediction_records(path,((step,seq) for step in range(2) for seq in range(4)))

    def test_parallel_sequence_order_rejected(self):
        self.left.write_text('0 0 10\n0 2 10\n0 1 10\n0 3 10\n')
        with self.assertRaisesRegex(ValueError,'identity'):
            validate_prediction_records(self.left,((0,seq) for seq in range(4)))


if __name__=='__main__':unittest.main()
