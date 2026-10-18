# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""tools/gen-unicode-tables: pinned inputs only, and the checked-in tables regenerate exactly."""

import importlib.machinery
import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parent.parent
REPO = TOOLS.parent

loader = importlib.machinery.SourceFileLoader("gen_unicode_tables", str(TOOLS / "gen-unicode-tables"))
spec = importlib.util.spec_from_loader(loader.name, loader)
gen = importlib.util.module_from_spec(spec)
loader.exec_module(gen)

# Where a Spark keeps the pinned UCD files (docs/tokenizer.md); another
# directory can be named with LLMP_TEST_UCD.
UCD = pathlib.Path(os.environ.get("LLMP_TEST_UCD", pathlib.Path.home() / ".local/share/llmp/ucd/15.1.0"))


class GenUnicodeTables(unittest.TestCase):
    def test_refuses_files_other_than_the_pinned_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in gen.FILES:
                (pathlib.Path(tmp) / name).write_text("0041;LATIN CAPITAL LETTER A;Lu;0;L;;;;;N;;;;0061;\n")
            with self.assertRaisesRegex(gen.Error, "SHA-256"):
                gen.generate(pathlib.Path(tmp))

    def test_default_full_folding_is_distinct_from_simple_and_turkic_mappings(self):
        self.assertEqual(gen.default_case_folds(
            "0041; C; 0061; # A\n00DF; F; 0073 0073; # sharp s\n"
            "0130; F; 0069 0307;\n0130; T; 0069;\n1E9E; S; 00DF;\n"),
            [(0x41, [0x61]), (0xDF, [0x73, 0x73]), (0x130, [0x69, 0x307])])
        for source in ("0041; X; 0061;", "0041; C; ;", "0041; C; 110000;",
                       "0041; C; 0061;\n0041; C; 0061;"):
            with self.assertRaises(gen.Error):
                gen.default_case_folds(source)

    def test_refuses_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(gen.Error, "--download"):
            gen.generate(pathlib.Path(tmp))

    @unittest.skipUnless(all((UCD / name).is_file() for name in gen.FILES), f"no UCD {gen.VERSION} files in {UCD}")
    def test_regenerates_the_checked_in_tables(self):
        generated = gen.generate(UCD)
        self.assertEqual(generated, (REPO / "src/tokenizer/unicode_data.cc").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
