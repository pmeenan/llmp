# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# Run inside the pinned image-reference container (torch, safetensors).
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

from safetensors.torch import save_file
import torch

import identity


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        g = torch.Generator().manual_seed(3)
        self.q = torch.randn(4, 3, generator=g).to(torch.bfloat16)
        self.k = torch.randn(4, 3, generator=g).to(torch.bfloat16)
        self.conv = torch.randn(2, 2, 3, 3, generator=g)

    def save(self, name, tensors):
        path = Path(self.temp.name) / name
        save_file(tensors, str(path))
        return str(path)

    def run_tool(self, candidate, reference):
        argv = ["identity.py", "--label", "t", "--candidate", candidate, "--reference", reference]
        out = io.StringIO()
        with unittest.mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            identity.main()
        return json.loads(out.getvalue())

    def test_fused_squeezed_and_rounded_match(self):
        ref = self.save("ref.safetensors", {"q": self.q, "k": self.k, "conv": self.conv})
        cand = self.save("cand.safetensors", {
            "qk": torch.cat([self.q, self.k]),                      # fused rows
            "conv3d": self.conv.to(torch.bfloat16).unsqueeze(2),    # singleton depth, BF16-rounded
        })
        r = self.run_tool(cand, ref)
        self.assertEqual((r["matched_tensors"], r["row_concatenated_tensors"]), (1, 1))
        self.assertEqual(r["unused_reference_tensors"], 0)
        self.assertEqual(r["matched_bytes"], r["candidate_bytes"])
        self.assertEqual(r["unmatched_candidate_shapes"], [])

    def test_changed_value_is_unmatched(self):
        changed = self.k.clone()
        changed[0, 0] += 1
        ref = self.save("ref.safetensors", {"q": self.q, "k": self.k})
        cand = self.save("cand.safetensors", {"qk": torch.cat([self.q, changed])})
        r = self.run_tool(cand, ref)
        self.assertEqual(r["row_concatenated_tensors"], 0)
        self.assertEqual(r["unmatched_candidate_shapes"], ["(8, 3)"])

    def test_reference_consumed_once(self):
        ref = self.save("ref.safetensors", {"q": self.q, "k": self.k})
        cand = self.save("cand.safetensors", {"a": torch.cat([self.q, self.k]), "b": self.q})
        r = self.run_tool(cand, ref)
        self.assertEqual(r["matched_tensors"] + r["row_concatenated_tensors"], 1)


if __name__ == "__main__":
    unittest.main()
