#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Protocol contracts only; no runtime, device, checkpoint or network calls."""
import sys
sys.dont_write_bytecode = True

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import control
import source


def response(text='Paris', finish='stop', tokens=None):
    tokens = [1000, 106] if tokens is None else tokens
    n = len(tokens)
    return {'choices': [{'text': text, 'finish_reason': finish,
                         'logprobs': {'tokens': [f'token_id:{i}' for i in tokens],
                                      'token_logprobs': [-0.1] * n}}],
            'usage': {'prompt_tokens': 35, 'completion_tokens': n, 'total_tokens': 35+n}}


class Contracts(unittest.TestCase):
    def test_literal_request_uses_committed_ids(self):
        request = control.body()
        self.assertEqual(request['prompt'], control.inputs()['prompt_token_ids'])
        self.assertEqual(len(request['prompt']), 35)
        self.assertFalse(request['add_special_tokens'])
        self.assertEqual(request['max_tokens'], 64)

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_bytes(Path(control.__file__).with_name('input.json').read_bytes() + b' ')
            with self.assertRaises(ValueError): control.inputs(path)

    def test_reference_binding_keeps_implementation_and_recipe(self):
        raw = control.raw_reference()
        original_run, original_check = raw.run, raw.check_dispatch
        engine, sampling = raw.engine_kwargs(), raw.sampling_kwargs()
        self.assertIs(source.bind(), raw)
        self.assertEqual(raw.IDS, control.body()['prompt'])
        self.assertIs(raw.run, original_run)
        self.assertIs(raw.check_dispatch, original_check)
        self.assertEqual((raw.engine_kwargs(), raw.sampling_kwargs()), (engine, sampling))
        self.assertEqual(control.sha(raw.__file__), control.RAW_REFERENCE_SHA)

    def test_reference_rejects_old_five_id_response(self):
        raw = source.bind()
        with self.assertRaises(ValueError):
            raw.collect_output([NS(prompt_token_ids=[818,5279,529,7001,563])])

    def test_reference_preserves_valid_negative_repetition(self):
        raw = source.bind()
        token = NS(logprob=-0.1, rank=1, decoded_token=' is')
        out = NS(token_ids=[563]*64, text=' is'*64, logprobs=[{563:token}]*64,
                 finish_reason='length', stop_reason=None, cumulative_logprob=-6.4)
        value = raw.collect_output([NS(prompt_token_ids=list(raw.IDS), finished=True, outputs=[out])])
        self.assertEqual(control.verdict(value['text'], value['finish_reason']), 'fail')
        self.assertEqual(value['completion_tokens'],64)

    def test_native_valid_complete_and_stop_ids(self):
        value = control.native_output(response())
        self.assertEqual(value['semantic_verdict'], 'pass')
        self.assertEqual(value['output_token_ids'], [1000,106])

    def test_finite_selected_scores_required(self):
        for invalid in (None, float('nan'), float('inf'), 0.01, True):
            obj=response();obj['choices'][0]['logprobs']['token_logprobs'][0]=invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): control.native_output(obj)

    def test_native_bad_counts_or_ids_rejected(self):
        bad=response();bad['usage']['prompt_tokens']=5
        bad_id=response();bad_id['choices'][0]['logprobs']['tokens'][0]='Paris'
        missing=response();missing['choices'][0]['logprobs']['token_logprobs'].pop()
        for obj in (bad,bad_id,missing):
            with self.subTest(obj=obj), self.assertRaises(ValueError): control.native_output(obj)

    def test_unexpected_stops_rejected(self):
        for obj in (response(tokens=[1000]), response(tokens=[106,1000,106]),
                    response(finish='length',tokens=[563]*63+[106])):
            with self.subTest(obj=obj), self.assertRaises(ValueError): control.native_output(obj)

    def test_semantic_failure_is_retained(self):
        for obj in (response(text='London'), response(text='Paris '+ ' is'*63,
                     finish='length',tokens=[563]*64)):
            self.assertEqual(control.native_output(obj)['semantic_verdict'],'fail')

    def test_input_freeze_is_not_gpu_registration(self):
        for value in ('03655eaf'+'0'*32, 'main', 'abcdef01'):
            with self.assertRaises(ValueError): control.registration(value)


if __name__ == '__main__':
    unittest.main()
