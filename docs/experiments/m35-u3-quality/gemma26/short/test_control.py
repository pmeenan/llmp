#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Short-score contract fixtures; no device, model, container or network."""
import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import control
import source


def native():
    ids = control.inputs()
    rows = [None] + [{str(i): {'logprob': -2.0}, str(i + 1): {'logprob': -0.1}} for i in ids[1:]]
    return dict(choices=[dict(prompt_logprobs=rows, finish_reason='length',
                             logprobs={'tokens': [f'token_id:{i}' for i in ids]})],
                usage=dict(prompt_tokens=1024, completion_tokens=0, total_tokens=1024))


def reference():
    ids = control.inputs()
    rows = [None] + [{i: NS(logprob=-2.0), i + 1: NS(logprob=-0.1)} for i in ids[1:]]
    return [NS(prompt_token_ids=ids, prompt_logprobs=rows, finished=True,
               outputs=[NS(token_ids=[50429], text='Paris', finish_reason='length', stop_reason=None)])]


class Contracts(unittest.TestCase):
    def test_exact_request_and_sampling(self):
        request = control.body()
        self.assertEqual(request['prompt'], control.inputs())
        self.assertEqual((request['max_tokens'], request['echo'], request['prompt_logprobs']), (0, True, 1))
        raw = control.raw_reference()
        before = raw.sampling_kwargs()
        after = source.sampling_kwargs(raw)
        self.assertEqual(after, dict(before, max_tokens=1, prompt_logprobs=1))
        self.assertEqual(raw.sampling_kwargs(), before)
        self.assertEqual(raw.IDS, [818, 5279, 529, 7001, 563])

    def test_prior_registrations_do_not_authorize_collection(self):
        for prefix in ('03655eaf', '3202f46f', '24846f55'):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                control.registration(prefix + '0' * 32)

    def test_pair_actual_teacher_outside_top_one(self):
        a = control.native_output(native())
        b, unused = control.source_output(reference())
        self.assertEqual(a, b)
        self.assertEqual(len(a), 1023)
        self.assertEqual((a[0]['position'], a[-1]['position']), (1, 1023))
        self.assertTrue(all(row['nll'] == 2.0 for row in a))
        self.assertFalse(unused['used_as_teacher'])
        self.assertEqual(unused['token_ids'], [50429])

    def test_native_missing_changed_or_truncated_echo_refused(self):
        for mode in ('missing', 'changed', 'short'):
            obj = native(); labels = obj['choices'][0]['logprobs']
            if mode == 'missing': labels.pop('tokens')
            elif mode == 'changed': labels['tokens'][512] = 'token_id:999'
            else: labels['tokens'].pop()
            with self.subTest(mode=mode), self.assertRaises(ValueError): control.native_output(obj)

    def test_native_teacher_missing_even_with_top_one(self):
        obj = native(); pos = 512
        del obj['choices'][0]['prompt_logprobs'][pos][str(control.inputs()[pos])]
        with self.assertRaises(ValueError): control.native_output(obj)

    def test_source_exact_prompt_required(self):
        for pos in (0, 512, 1023):
            obj = reference(); obj[0].prompt_token_ids[pos] += 1
            with self.subTest(pos=pos), self.assertRaises(ValueError): control.source_output(obj)

    def test_source_missing_teacher_not_replaced_with_top_one(self):
        obj = reference(); pos = 513
        del obj[0].prompt_logprobs[pos][control.inputs()[pos]]
        with self.assertRaises(ValueError): control.source_output(obj)

    def test_row_geometry(self):
        for mode in ('leading', 'missing', 'extra'):
            a, b = native(), reference()
            ar, br = a['choices'][0]['prompt_logprobs'], b[0].prompt_logprobs
            if mode == 'leading': ar[0] = {}; br[0] = {}
            elif mode == 'missing': ar.pop(512); br.pop(512)
            else: ar.append(ar[-1]); br.append(br[-1])
            with self.subTest(mode=mode), self.assertRaises(ValueError): control.native_output(a)
            with self.subTest(mode=mode), self.assertRaises(ValueError): control.source_output(b)

    def test_invalid_scores_both_engines(self):
        for value in (None, float('nan'), float('inf'), 0.1, True):
            a, b = native(), reference(); token = control.inputs()[513]
            a['choices'][0]['prompt_logprobs'][513][str(token)]['logprob'] = value
            b[0].prompt_logprobs[513][token].logprob = value
            with self.subTest(value=value), self.assertRaises(ValueError): control.native_output(a)
            with self.subTest(value=value), self.assertRaises(ValueError): control.source_output(b)

    def test_decode_not_counted_or_allowed_in_native(self):
        obj = native(); obj['usage'].update(completion_tokens=1, total_tokens=1025)
        with self.assertRaises(ValueError): control.native_output(obj)
        obj = reference(); obj[0].outputs[0].token_ids.append(106)
        with self.assertRaises(ValueError): control.source_output(obj)

    def test_jsonl_is_one_actual_target_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = control.native_output(native()); out = Path(tmp)
            receipt = control.write_losses(out, rows)
            self.assertEqual([json.loads(line) for line in (out/'losses.jsonl').read_text().splitlines()], rows)
            self.assertEqual(receipt['targets'], 1023)

    def test_raw_source_return_precedes_failure(self):
        obj = reference(); obj[0].prompt_token_ids[512] += 1
        raw = control.raw_reference()
        fake = NS(validate=lambda: {}, checkpoint_check=lambda _: {},
                  sampling_kwargs=raw.sampling_kwargs, engine_kwargs=raw.engine_kwargs,
                  IMAGE=raw.IMAGE, inspect_model=raw.inspect_model,
                  check_dispatch=lambda _: None, write_private=lambda p, b: p.write_bytes(b))
        class LLM:
            def __init__(self, **kwargs): pass
            def apply_model(self, _): return [{}]
            def generate(self, *args, **kwargs): return obj
        def sampling(**kwargs): return NS(**kwargs, skip_reading_prefix_cache=True)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', VLLM_USE_V2_MODEL_RUNNER='1'), \
             patch.dict(sys.modules, {'vllm': NS(LLM=LLM, SamplingParams=sampling),
                                      'vllm.inputs': NS(TokensPrompt=lambda **kw: kw)}), \
             patch.object(control, 'source_check', return_value={}), \
             patch.object(control, 'raw_reference', return_value=fake), \
             patch.object(sys, 'argv', ['source.py', 'run', '--out', tmp, '--registration', 'f'*40]):
            with self.assertRaises(ValueError): source.main()
            self.assertEqual((Path(tmp)/'returned.txt').read_text(), repr(obj)+'\n')
            self.assertEqual(json.loads((Path(tmp)/'result.json').read_text())['status'], 'failed')
            self.assertFalse((Path(tmp)/'losses.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
