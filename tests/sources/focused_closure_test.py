#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""No-compile controls for explicit focused Ninja dependency auditing."""
import argparse
import importlib.util
import contextlib
import io
import json
import sys
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

parser = argparse.ArgumentParser()
parser.add_argument('--checker', type=pathlib.Path,
                    default=pathlib.Path(__file__).with_name('check_closure.py'))
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('closure', args.checker)
closure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(closure)


class FocusedClosureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        (self.root / 'CMakeFiles').mkdir()
        (self.root / 'CMakeFiles/rules.ninja').write_text(
            'rule CXX_COMPILER__a\n  deps = gcc\nrule CXX_EXECUTABLE_LINKER__a\n')
        for output in ('a.o', 'app', 'other.o'):
            (self.root / output).write_bytes(b'fixture')
        self.rules = ('a.o: CXX_COMPILER__a\napp: CXX_EXECUTABLE_LINKER__a\n'
                      'alias: phony\nother.o: CXX_COMPILER__a\nall: phony\n')
        self.records = {
            'a.o': 'a.o: #deps 1 (VALID)\n    a.h\n',
            'other.o': 'other.o: #deps 1 (STALE)\n    old.h\n',
        }
        self.calls = []
        self.extra_inputs = ""

    def tearDown(self):
        self.temp.cleanup()

    def ninja(self, command, **kwargs):
        self.calls.append(command)
        tool = command[command.index('-t') + 1]
        if tool == 'targets':
            result = self.rules
        elif tool == 'inputs':
            targets = command[command.index('-E') + 1:]
            self.assertEqual(targets, ['alias'] if targets != ['all'] else ['all'])
            result = 'a.h\0a.o\0app\0' + self.extra_inputs + ('other.o\0' if targets == ['all'] else '')
        else:
            self.assertEqual(tool, 'deps')
            result = ''.join(self.records.get(output, '') for output in command[command.index('deps') + 1:])
        return subprocess.CompletedProcess(command, 0, result, '')

    def scope(self, targets=('alias',)):
        problems = []
        scope = closure.focused_outputs('ninja', self.root, list(targets), problems)
        inputs, rules = closure.recorded_inputs('ninja', self.root, problems, scope)
        return problems, inputs, rules

    def test_selected_closure_requires_current_dependencies_and_skips_unrelated_stale_outputs(self):
        # CMake's always-run verification tasks are graph dependencies, not file outputs.
        self.rules += 'verify: CUSTOM_COMMAND\n'
        self.extra_inputs = 'verify\0'
        with mock.patch.object(closure.subprocess, 'run', self.ninja):
            problems, inputs, rules = self.scope()
        self.assertEqual(problems, [])
        self.assertEqual(inputs, {'a.o': ['a.h']})
        self.assertIn('other.o', rules)  # classification still knows all declared outputs
        deps_calls = [call for call in self.calls if '-t' in call and 'deps' in call]
        self.assertTrue(deps_calls)
        self.assertTrue(all('other.o' not in call for call in deps_calls))

    def test_selected_absent_output_is_refused(self):
        for output in ('app', 'a.o'):
            with self.subTest(output=output):
                (self.root / output).unlink()
                with mock.patch.object(closure.subprocess, 'run', self.ninja):
                    problems, _, _ = self.scope()
                self.assertTrue(any(f'focused output {output} is absent' in problem for problem in problems))
                (self.root / output).write_bytes(b'fixture')

    def test_selected_stale_dependency_record_is_refused(self):
        self.records['a.o'] = 'a.o: #deps 1 (STALE)\n    old.h\n'
        with mock.patch.object(closure.subprocess, 'run', self.ninja):
            problems, _, _ = self.scope()
        self.assertTrue(any('a.o has no current dependency record' in problem for problem in problems))

    def test_unknown_selected_target_is_refused_before_inputs_lookup(self):
        with mock.patch.object(closure.subprocess, 'run', self.ninja):
            problems, _, _ = self.scope(('unknown',))
        self.assertTrue(any('unknown focused target' in problem for problem in problems))
        self.assertFalse(any('inputs' in call for call in self.calls))

    def test_default_audit_keeps_requiring_unrelated_current_outputs(self):
        with mock.patch.object(closure.subprocess, 'run', self.ninja):
            problems = []
            closure.recorded_inputs('ninja', self.root, problems)
        self.assertTrue(any('other.o has no current dependency record' in problem for problem in problems))

        # Exercise main's component classifier and coverage gate in both modes.
        selected, unused = self.root / 'selected', self.root / 'unused'
        selected.mkdir()
        unused.mkdir()
        (self.root / 'llmp-receipt.json').write_text(json.dumps({'components': [
            {'id': 'selected', 'source': str(selected), 'source_tree': 'a' * 64},
            {'id': 'unused', 'source': str(unused), 'source_tree': 'b' * 64},
        ]}))
        (self.root / 'CMakeCache.txt').write_text('')
        (self.root / 'compile_commands.json').write_text('[]')
        (self.root / 'build.ninja').write_text('build app: CXX_EXECUTABLE_LINKER__a linked.o || verify\n')
        (self.root / 'linked.o').write_bytes(b'fixture')
        argv = ['check', '--build-dir', str(self.root), '--source-dir', str(self.root),
                '--sdk', str(self.root / 'sdk'), '--ninja', 'ninja', '--cross']
        inputs = ({'a.o': [str(selected / 'header.h')]},
                  {'a.o': 'CXX_COMPILER__a', 'app': 'CXX_EXECUTABLE_LINKER__a', 'linked.o': 'CUSTOM_COMMAND'})
        for focused in (False, True):
            with self.subTest(focused=focused), contextlib.ExitStack() as stack:
                stdout, stderr = io.StringIO(), io.StringIO()
                stack.enter_context(contextlib.redirect_stdout(stdout))
                stack.enter_context(contextlib.redirect_stderr(stderr))
                stack.enter_context(mock.patch.object(sys, 'argv', argv + (['--target', 'alias'] if focused else [])))
                stack.enter_context(mock.patch.object(closure, 'focused_outputs', return_value={'a.o', 'app', 'linked.o', 'verify'}))
                stack.enter_context(mock.patch.object(closure, 'recorded_inputs', return_value=inputs))
                stack.enter_context(mock.patch.object(closure, 'link_input_kind', return_value='object'))
                stack.enter_context(mock.patch.object(closure, 'exception_machinery', return_value={}))
                result = closure.main()
                self.assertEqual(result, 0 if focused else 1)
                if focused:
                    self.assertIn('Receipt components outside focused inputs: unused', stdout.getvalue())
                    self.assertIn('whole build/package audit deferred', stdout.getvalue())
                    (self.root / 'linked.o').unlink()
                    self.assertEqual(closure.main(), 1)
                    self.assertIn('focused link app input linked.o is absent', stderr.getvalue())
                    (self.root / 'linked.o').write_bytes(b'fixture')
                else:
                    self.assertIn('unused, which nothing compiles or links', stderr.getvalue())


if __name__ == '__main__':
    unittest.main(argv=['focused_closure_test'])
