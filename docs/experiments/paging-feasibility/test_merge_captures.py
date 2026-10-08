# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest
from merge_captures import merge
from run import IMAGE, REVISION, sha


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def capture(self, name, model):
        root = self.root / name
        root.mkdir()
        receipt = dict(image=IMAGE, revision=REVISION, cuda_disable_fusion=True,
                       cuda_disable_graphs=True, host=name, source_sha256=name,
                       spec=dict(models=[dict(id=model)], configurations=[[512, 1, 1]],
                                 trace_sha256='workload'), runs=[])
        for trace in (False, True):
            run = dict(name=f'{model}-{int(trace)}', model=model, batch=512, reuse=1,
                       swa=1, trace=trace)
            for extension, key in [('jsonl', 'events_sha256'), ('predictions', 'predictions_sha256')]:
                path = root / (run['name'] + '.' + extension)
                path.write_text('0 1\n')
                run[key] = sha(path)
            (root / (run['name'] + '.log')).write_text('historical unbound log')
            receipt['runs'].append(run)
        self.write(root, receipt)
        return root, receipt

    def write(self, root, receipt):
        (root / 'receipt.json').write_text(json.dumps(receipt))

    def test_disjoint_models_preserve_distinct_provenance(self):
        left, _ = self.capture('spark', 'D')
        right, _ = self.capture('spark-b', 'Q')
        merge([left, right], self.root / 'merged')
        result = json.loads((self.root / 'merged/receipt.json').read_text())
        self.assertNotIn('host', result)
        self.assertEqual([c['receipt']['host'] for c in result['components']], ['spark', 'spark-b'])
        self.assertEqual([c['receipt_sha256'] for c in result['components']],
                         [sha(left / 'receipt.json'), sha(right / 'receipt.json')])
        self.assertTrue(all('log_sha256' in r for r in result['runs']))
        self.assertEqual(len(result['runs']), 4)

    def test_duplicate_model_across_roots_rejected(self):
        left, _ = self.capture('left', 'D')
        right, _ = self.capture('right', 'D')
        with self.assertRaises(ValueError):
            merge([left, right], self.root / 'merged')
        self.assertFalse((self.root / 'merged').exists())

    def test_duplicate_model_spec_rejected(self):
        root, receipt = self.capture('left', 'D')
        receipt['spec']['models'].append(dict(id='D'))
        self.write(root, receipt)
        with self.assertRaises(ValueError):
            merge([root], self.root / 'merged')

    def test_incomplete_pair_rejected(self):
        root, receipt = self.capture('left', 'D')
        receipt['runs'].pop()
        self.write(root, receipt)
        with self.assertRaises(ValueError):
            merge([root], self.root / 'merged')

    def test_corrupted_bound_file_rejected(self):
        root, _ = self.capture('left', 'D')
        (root / 'D-0.jsonl').write_text('corruption')
        with self.assertRaises(ValueError):
            merge([root], self.root / 'merged')

    def test_mismatched_predictions_rejected_even_when_each_hash_valid(self):
        root, receipt = self.capture('left', 'D')
        path = root / 'D-1.predictions'
        path.write_text('different prediction')
        receipt['runs'][1]['predictions_sha256'] = sha(path)
        self.write(root, receipt)
        with self.assertRaises(ValueError):
            merge([root], self.root / 'merged')

    def test_merge_preserves_explicit_failed_equivalence(self):
        from prediction_checks import capture_comparison
        root,receipt=self.capture('left','Q')
        path=root/'Q-1.predictions';path.write_text('0 2\n')
        traced=receipt['runs'][1]
        traced.update(predictions_sha256=sha(path),prediction_drift_recorded=True,
                      prediction_comparison=capture_comparison(root/'Q-0.predictions',path,True))
        self.write(root,receipt)
        merge([root],self.root/'merged')
        combined=json.loads((self.root/'merged/receipt.json').read_text())
        self.assertTrue(combined['runs'][1]['prediction_drift_recorded'])
        self.assertEqual(combined['runs'][1]['prediction_comparison']['equivalence'],'failed')

    def test_missing_event_hash_is_not_silently_adopted(self):
        root, receipt = self.capture('left', 'D')
        del receipt['runs'][0]['events_sha256']
        self.write(root, receipt)
        with self.assertRaises(ValueError):
            merge([root], self.root / 'merged')


if __name__ == '__main__':
    unittest.main()
