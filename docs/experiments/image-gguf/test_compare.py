# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# Run inside the pinned image-reference container (numpy, Pillow).
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import compare


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        rng = np.random.default_rng(7)
        self.base = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    def save(self, name, array, mode="RGB"):
        path = Path(self.temp.name) / name
        Image.fromarray(array).convert(mode).save(path)
        return str(path)

    def test_identical(self):
        a, b = self.save("a.png", self.base), self.save("b.png", self.base.copy())
        result = compare.compare(a, b)
        self.assertTrue(result["identical_pixels"])
        self.assertIsNone(result["psnr_db"])
        self.assertAlmostEqual(result["ssim_luma"], 1.0, places=6)

    def test_alpha_mode_is_not_identical(self):
        a, b = self.save("a.png", self.base, "RGBA"), self.save("b.png", self.base)
        self.assertFalse(compare.compare(a, b)["identical_pixels"])

    def test_small_perturbation_orders_metrics(self):
        noisy = np.clip(self.base.astype(int) + 2, 0, 255).astype(np.uint8)
        worse = np.clip(self.base.astype(int) + np.random.default_rng(1).integers(-60, 60, self.base.shape),
                        0, 255).astype(np.uint8)
        ref = self.save("ref.png", self.base)
        near = compare.compare(self.save("near.png", noisy), ref)
        far = compare.compare(self.save("far.png", worse), ref)
        self.assertFalse(near["identical_pixels"])
        self.assertGreater(near["psnr_db"], far["psnr_db"])
        self.assertGreater(near["ssim_luma"], far["ssim_luma"])

    def test_size_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            compare.compare(self.save("a.png", self.base), self.save("b.png", self.base[:32]))


if __name__ == "__main__":
    unittest.main()
