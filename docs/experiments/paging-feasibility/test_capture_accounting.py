# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent tests for logged allocation and parallel trace validation."""
import json
import copy
from pathlib import Path
import tempfile
import unittest
from run import sha
from switch_replay import log_allocations, read_capture_log, state_allocation, EXTENT, load_events
from parallel_replay import input_shape, load


class AllocationTest(unittest.TestCase):
    def test_additive_state_partitions_not_summary_lines(self):
        log = '''llama_kv_cache: CUDA0 KV buffer size = 640.00 MiB
llama_kv_cache: CUDA0 KV buffer size = 6400.00 MiB
llama_memory_recurrent: CUDA0 RS buffer size = 62.81 MiB
cache: CUDA0 DSV4 csa state buffer size = 16.00 MiB
cache: CUDA0 DSV4 hca state buffer size = 8.00 MiB
cache: CUDA0 DSV4 lid state buffer size = 4.00 MiB
cache: csa ratio = 4, size = 16.00 MiB
compute: CUDA0 compute buffer size = 1500.00 MiB
compute: CUDA0 compute buffer size = 1499.00 MiB
compute: CUDA_Host compute buffer size = 30.00 MiB
compute: CUDA_Host output buffer size = 1.00 MiB
'''
        self.assertEqual(log_allocations(log), (7132 * (1 << 20), 1532 * (1 << 20)))

    def test_mode_allocation_not_model_name(self):
        model = dict(id='D', kv_bytes=100 * EXTENT, normal_kv_bytes=10 * EXTENT)
        self.assertEqual(state_allocation(model), 100 * EXTENT)
        self.assertEqual(state_allocation(model, True), 10 * EXTENT)

    def test_missing_allocation_fails(self):
        with self.assertRaises(ValueError):
            log_allocations('CUDA0 compute buffer size = 500 MiB')

    def test_optional_bound_log_digest_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.log'
            path.write_text('original')
            run = dict(name='run', log_sha256=sha(path))
            self.assertEqual(read_capture_log(directory, run), 'original')
            path.write_text('changed')
            with self.assertRaises(ValueError):
                read_capture_log(directory, run)


class ParallelTraceTest(unittest.TestCase):
    def test_shape_includes_each_prefill_and_shortest_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.tokens'
            path.write_text('4\n' + ''.join(f'1 {d}\n' + ' '.join(['3'] * (1 + d)) + '\n'
                                            for d in (2, 3, 4, 5)))
            shape = input_shape(path, dict(tokens_sha256=sha(path), context=64))
            self.assertEqual(shape, [('prefill', 1)] * 4 + [('decode', 4)] * 2)

    def test_self_consistent_truncation_rejected_against_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            events = [dict(event='batch', step=0, phase='prefill', tokens=1),
                      dict(event='batch', step=1, phase='decode', tokens=4),
                      dict(event='end', decode_batches=1, batch_size=4)]
            path.write_text(''.join(json.dumps(e) + '\n' for e in events))
            run = dict(events_sha256=sha(path), trace=False)
            with self.assertRaises(ValueError):
                load(path, run, dict(layers=1), [('prefill', 1), ('decode', 4), ('decode', 4)])


class OutputOnlyRoutesTest(unittest.TestCase):
    def events(self):
        events=[dict(event='request',request=0,prompt=4,decode=1,reused=0)]
        for step,phase,n in [(0,'prefill',4),(1,'decode',1)]:
            for layer in range(2):
                output_only=phase=='prefill' and layer==1
                events.append(dict(event='routes',request=0,step=step,phase=phase,layer=layer,
                                   output_only=output_only,routes=[[0]]*(1 if output_only else n)))
            events.append(dict(event='step',request=0,step=step,phase=phase,tokens=n,us=1))
        events.append(dict(event='end',request=0,history=5,state_bytes=100))
        return events

    def parse(self,events):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            path.write_text(''.join(json.dumps(e)+'\n' for e in events))
            return load_events(path,dict(events_sha256=sha(path),batch=512,trace=True),
                               dict(context=100,layers=2,experts=2,topk=1))

    def test_last_prefill_layer_can_route_only_output_row(self):
        requests=self.parse(self.events())
        self.assertEqual(requests[0]['step_us'][0]['tokens'],4)
        self.assertEqual(len(requests[0]['groups'][1]['routes']),1)

    def test_pruned_row_without_explicit_flag_rejected(self):
        events=self.events();del events[2]['output_only']
        with self.assertRaises(ValueError):self.parse(events)

    def test_nonfinal_layer_pruning_rejected(self):
        events=self.events();events[1]['routes']=[[0]];events[1]['output_only']=True
        with self.assertRaises(ValueError):self.parse(events)

    def test_decode_pruning_flag_rejected(self):
        events=self.events();events[5]['output_only']=True
        with self.assertRaises(ValueError):self.parse(events)


if __name__ == '__main__':
    unittest.main()
