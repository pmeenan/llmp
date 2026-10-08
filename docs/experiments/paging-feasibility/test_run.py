# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import subprocess
import unittest
from unittest.mock import patch
import run


class CleanupTest(unittest.TestCase):
    def test_success_cleanup(self):
        with patch.object(run.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,'','')) as call:
            run.container(['docker'], ['docker','run'], ['image'])
            self.assertEqual(call.call_count,2)
            name = call.call_args_list[0].args[0][3]
            self.assertEqual(call.call_args_list[1].args[0],['docker','rm','-f',name])

    def test_failed_run_cleanup(self):
        failure = subprocess.CalledProcessError(1,['docker'])
        with patch.object(run.subprocess,'run',side_effect=[failure,subprocess.CompletedProcess([],0,'','')]) as call:
            with self.assertRaises(subprocess.CalledProcessError):
                run.container(['docker'],['docker','run'],['image'])
            self.assertEqual(call.call_count,2)

    def test_interruption_cleanup(self):
        with patch.object(run.subprocess,'run',side_effect=[KeyboardInterrupt(),subprocess.CompletedProcess([],0,'','')]) as call:
            with self.assertRaises(KeyboardInterrupt):
                run.container(['docker'],['docker','run'],['image'])
            self.assertEqual(call.call_count,2)

    def test_cleanup_failure_is_reported(self):
        with patch.object(run.subprocess,'run',side_effect=[subprocess.CompletedProcess([],0),subprocess.CompletedProcess([],1,'','daemon failed')]):
            with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
                run.container(['docker'],['docker','run'],['image'])


if __name__ == '__main__':
    unittest.main()
