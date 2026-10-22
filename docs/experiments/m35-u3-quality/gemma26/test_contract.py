#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Adversarial result and retirement checks; no Docker, model or vLLM imports."""
import sys
sys.dont_write_bytecode = True

import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import launch
from reference import IDS, collect_output, check_dispatch


def output():
    return [NS(prompt_token_ids=list(IDS), finished=True,
               outputs=[NS(token_ids=[563] * 64, text=' is' * 64,
                           finish_reason='length', stop_reason=None,
                           logprobs=[{563: NS(logprob=-0.1, rank=1, decoded_token=' is')}]*64,
                           cumulative_logprob=-6.4)])]


class Contracts(unittest.TestCase):
    def test_literal_repetition_is_preserved_as_data(self):
        actual = collect_output(output())
        self.assertEqual(actual['text'], ' is' * 64)
        self.assertEqual(actual['completion_tokens'], 64)
        self.assertNotIn('passed', actual)

    def test_input_mutation_and_invalid_scores_are_rejected(self):
        for ids in ([2] + IDS, IDS[:-1], IDS[::-1]):
            item = output()
            item[0].prompt_token_ids = ids
            with self.assertRaisesRegex(ValueError, 'prompt IDs'):
                collect_output(item)
        for value in (float('nan'), float('inf'), 0.5):
            item = output()
            item[0].outputs[0].logprobs[0] = {563: NS(logprob=value, rank=1, decoded_token=' is')}
            with self.assertRaisesRegex(ValueError, 'log probability'):
                collect_output(item)
        for value in (None, float('nan'), float('inf'), 0.5):
            item = output()
            item[0].outputs[0].cumulative_logprob = value
            with self.assertRaisesRegex(ValueError, 'cumulative log probability'):
                collect_output(item)

    def test_incomplete_missing_target_and_early_length_are_rejected(self):
        for kind in ('missing-target', 'incomplete', 'early-length'):
            item = output()
            if kind == 'missing-target':
                item[0].outputs[0].logprobs[0] = {}
            elif kind == 'incomplete':
                item[0].finished = False
            else:
                item[0].outputs[0].token_ids.pop()
            with self.assertRaises(ValueError):
                collect_output(item)

    def test_dispatch_refuses_arithmetic_and_missing_block_changes(self):
        record = dict(model='vllm.Gemma4ForCausalLM', quant_config='vllm.ModelOptNvFp4Config',
                      quant_method='NVFP4', linear_method='ModelOptNvFp4LinearMethod',
                      modules=[dict(quant_method='vllm.ModelOptNvFp4FusedMoE', use_a16=False,
                                    nvfp4_backend='FLASHINFER', moe_kernel='vllm.Kernel')
                               for _ in range(30)] +
                              [dict(attention_backend='vllm.FlashAttentionBackend',
                                    kv_cache_dtype='bfloat16') for _ in range(30)] +
                              [dict(quant_method='vllm.ModelOptNvFp4LinearMethod',
                                    kernel='vllm.CutlassNvFp4LinearKernel')])
        check_dispatch(record)
        for change in ('a16', 'missing', 'kv', 'marlin', 'linear-marlin',
                       'linear-humming', 'linear-missing', 'linear-a16'):
            candidate = copy.deepcopy(record)
            if change == 'a16':
                candidate['modules'][0]['use_a16'] = True
            elif change == 'missing':
                candidate['modules'].pop(0)
            elif change == 'kv':
                candidate['modules'][-2]['kv_cache_dtype'] = 'fp8'
            elif change == 'marlin':
                candidate['modules'][0]['nvfp4_backend'] = 'MARLIN'
            elif change == 'linear-missing':
                del candidate['modules'][-1]['kernel']
            elif change == 'linear-a16':
                candidate['modules'][-1]['quant_method'] = 'vllm.ModelOptNvFp4W4A16LinearMethod'
            else:
                candidate['modules'][-1]['kernel'] = ('vllm.MarlinNvFp4LinearKernel'
                    if change == 'linear-marlin' else 'vllm.HummingNvFp4LinearKernel')
            with self.assertRaises(ValueError):
                check_dispatch(candidate)

    def test_stop_tokens_and_reasons_match_the_frozen_request(self):
        for token, reason in ((1, None), (106, 106), (50, 50)):
            item = output()
            result = item[0].outputs[0]
            result.token_ids = [token]
            result.logprobs = [{token: NS(logprob=-0.1, rank=1, decoded_token='stop')}]
            result.finish_reason = 'stop'
            result.stop_reason = reason
            collect_output(item)
            result.stop_reason = 'unexpected stop string'
            with self.assertRaisesRegex(ValueError, 'stop token or reason'):
                collect_output(item)
        item = output()
        item[0].outputs[0].token_ids[0] = 106
        with self.assertRaisesRegex(ValueError, 'continued past'):
            collect_output(item)
        item = output()
        item[0].outputs[0].stop_reason = 106
        with self.assertRaisesRegex(ValueError, 'contradicts stop'):
            collect_output(item)

    def test_remove_timeout_still_requires_successful_absence_query(self):
        with patch.object(launch.subprocess, 'run', side_effect=[
                subprocess.TimeoutExpired('docker rm', 4),
                NS(returncode=0, stdout='', stderr='')]):
            self.assertTrue(launch.retire('owned')['retired'])
        with patch.object(launch.subprocess, 'run', side_effect=[
                NS(returncode=0, stderr=''), NS(returncode=1, stdout='', stderr='bad daemon')]):
            self.assertFalse(launch.retire('owned')['retired'])

    def test_log_open_failure_and_process_timeout_still_retire(self):
        for failure in ('log-open', 'timeout'):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                if failure == 'log-open':
                    (out / 'container.log').mkdir()
                runs = [NS(returncode=0), NS(stdout=json.dumps([{'Image': launch.IMAGE}]).encode())]
                if failure == 'timeout':
                    runs.append(subprocess.TimeoutExpired('docker start', 510))
                receipt = {}
                with patch.object(launch.subprocess, 'run', side_effect=runs), \
                     patch.object(launch, 'retire', return_value={'retired': True}) as retired:
                    with self.assertRaises((IsADirectoryError, subprocess.TimeoutExpired)):
                        launch.execute(['create'], 'owned', out, receipt)
                    retired.assert_called_once_with('owned')
                self.assertTrue(json.loads((out/'launch.json').read_text())['retirement']['retired'])

    def test_prestart_daemon_failures_retain_diagnostics(self):
        for phase in ('create', 'inspect'):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                failure = subprocess.CalledProcessError(
                    1, ['docker', phase], output=b'partial output',
                    stderr=b'daemon rejected request')
                runs = ([NS(returncode=0)] if phase == 'inspect' else []) + [failure]
                with patch.object(launch.subprocess, 'run', side_effect=runs), \
                     patch.object(launch, 'retire', return_value={'retired': True}):
                    with self.assertRaises(subprocess.CalledProcessError):
                        launch.execute(['create'], 'owned', out, {})
                receipt = json.loads((out / 'launch.json').read_text())
                self.assertEqual(receipt['failed_command'], ['docker', phase])
                self.assertEqual(receipt['failed_stderr'], 'daemon rejected request')
                self.assertEqual(receipt['failed_stdout'], 'partial output')
                self.assertTrue(receipt['retirement']['retired'])


if __name__ == '__main__':
    unittest.main()
