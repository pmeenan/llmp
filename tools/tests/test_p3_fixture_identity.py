# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Which EXL3 fixture an artifact or oracle is (docs/experiments/backend-proof-p3/fixture_identity.py).

pack_run.py and tierc_check.py judge a native run against its fixture's bounds and oracle; the
fixture comes from what the artifact was prepared from and what the oracle hashes to, never from
a run's label. Synthetic artifact directories and oracles, no GPU or numpy.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
P3 = ROOT / "docs/experiments/backend-proof-p3"

spec = importlib.util.spec_from_file_location("fixture_identity", P3 / "fixture_identity.py")
fid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fid)

PROBES = json.loads((P3 / "exl3-op-plan-g.json").read_text())["provenance"]["probes"]


class FixtureIdentityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def artifact(self, checkpoint, name=None):
        """An installed artifact directory whose manifest names checkpoint as its source."""
        manifest = json.dumps({"source": [{"name": "config.json", "sha256": "0" * 64},
                                          {"name": "model.safetensors", "sha256": checkpoint}]}).encode()
        directory = self.root / (name or hashlib.sha256(manifest).hexdigest())
        directory.mkdir()
        (directory / "manifest.json").write_bytes(manifest)
        return directory

    def test_an_artifact_is_the_fixture_of_its_checkpoint(self):
        for fixture, probe in PROBES.items():
            directory = self.artifact(probe["weight_sha256"])
            self.assertEqual(fid.artifact_fixture(directory), (fixture, directory.name))

    def test_an_artifact_of_another_checkpoint_or_renamed_is_refused(self):
        with self.assertRaises(fid.Unidentified):
            fid.artifact_fixture(self.artifact("1" * 64))
        renamed = self.artifact(PROBES["4.0bpw"]["weight_sha256"], name="f" * 64)
        with self.assertRaises(fid.Unidentified):
            fid.artifact_fixture(renamed)

    def test_an_oracle_is_the_fixture_whose_output_it_hashes_to(self):
        oracle = self.root / "oracle.npz"
        oracle.write_bytes(b"an oracle")
        digest = hashlib.sha256(b"an oracle").hexdigest()
        envelope = self.root / "envelope.json"
        envelope.write_text(json.dumps({"oracle": {"output_sha256": {"4.0bpw": "2" * 64, "4.5bpw": digest}}}))
        self.assertEqual(fid.oracle_fixture(oracle, envelope), "4.5bpw")
        # The recorded envelope names neither fixture for these bytes.
        with self.assertRaises(fid.Unidentified):
            fid.oracle_fixture(oracle)


if __name__ == "__main__":
    unittest.main()
