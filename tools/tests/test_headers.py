# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tools/llmp_headers.py, the embedded-header check (D-029).

The REUSE report is synthetic here: by default it says what REUSE reads from
each fixture's own header or sidecar. The check step runs the real tool.
Tag names are spelled in parts, so neither REUSE nor the check reads the
fixtures in this file as tags.
"""

import os
import pathlib
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import llmp_headers as headers  # noqa: E402

COPYRIGHT_TAG, LICENSE_TAG = "SPDX-" + "FileCopyrightText", "SPDX-" + "License-Identifier"
IGNORE_START, IGNORE_END = "REUSE-" + "IgnoreStart", "REUSE-" + "IgnoreEnd"
COPYRIGHT = f"{COPYRIGHT_TAG}: 2026 Example"
LICENSE = f"{LICENSE_TAG}: MIT"
EMPTY_TAGS = f"# {COPYRIGHT_TAG}:\n# {LICENSE_TAG}:   \n"
HASH_HEADER = f"# {COPYRIGHT}\n# {LICENSE}\n"
HTML_HEADER = f"<!-- {COPYRIGHT} -->\n<!-- {LICENSE} -->\n"
SLASH_HEADER = f"// {COPYRIGHT}\n// {LICENSE}\n"


def entry(path: str, source: str | None = None, license: str = "MIT", holder: str = "2026 Example") -> dict:
    """REUSE's report for one file whose information comes from `source` (the file itself by default)."""
    source = source or path
    return {"path": path, "copyrights": [{"value": f"{COPYRIGHT_TAG}: {holder}", "source": source}],
            "spdx_expressions": [{"value": license, "source": source}]}


class Check(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))

    def write(self, files: dict[str, str]) -> None:
        for name, text in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def check(self, files: dict[str, str], report: dict | None = None, names: list[str] | None = None) -> list[str]:
        """Checks the files against `report`, by default what REUSE reads from each file's own metadata."""
        self.write(files)
        if report is None:
            covered = [n for n, text in files.items() if text and not n.endswith(".license")
                       and not n.startswith("LICENSES/") and not headers.overrides(n)]
            report = {"files": [entry(n, n + ".license" if n + ".license" in files else n) for n in covered]}
        return headers.check(self.root, sorted(names or files), report)[1]

    def test_each_comment_style_is_accepted_after_a_shebang_or_directives(self):
        self.assertEqual(self.check({
            "doc.md": HTML_HEADER + "\n# Title\n",
            "src/a.cc": SLASH_HEADER + "int x;\n",
            "src/b.cuh": f"/* {COPYRIGHT}\n * {LICENSE}\n */\n",
            "src/c.h": f"/* {COPYRIGHT} */\n/* {LICENSE} */\n",
            "tools/run": "#!/usr/bin/env python3\n" + HASH_HEADER,
            "tools/lib.py": HASH_HEADER,
            "Dockerfile": "# syntax=docker/dockerfile:1\n# check=skip=X\n" + HASH_HEADER,
            "img/Dockerfile.variant": HASH_HEADER,
            "CMakeLists.txt": HASH_HEADER,
            "exp/requirements-extra.txt": HASH_HEADER + "pkg==1\n",
            "toolchains/prerequisites/ubuntu.txt": HASH_HEADER,
            ".clang-tidy": HASH_HEADER,
        }), [])

    def test_a_missing_tag_is_named(self):
        problems = self.check({"a.py": f"# {COPYRIGHT}\n", "b.md": f"<!-- {LICENSE} -->\n"})
        self.assertEqual(len(problems), 2)
        self.assertIn("a.py: no SPDX-License-Identifier tag", problems[0])
        self.assertIn("b.md: no SPDX-FileCopyrightText tag", problems[1])

    def test_tags_must_be_in_the_header_and_in_the_files_comment_syntax(self):
        late = "\n" * headers.HEADER_LINES + HASH_HEADER
        problems = self.check({
            "late.py": late,
            "wrong.cc": HASH_HEADER,           # a # line is not a C++ comment
            "prose.md": f"{COPYRIGHT}\n{LICENSE}\n",
            "unclosed.md": f"<!-- {COPYRIGHT}\n<!-- {LICENSE}\n",
            "empty.py": EMPTY_TAGS,
        })
        self.assertEqual(sorted({p.split(":", 1)[0] for p in problems}),
                         ["empty.py", "late.py", "prose.md", "unclosed.md", "wrong.cc"])

    def test_a_file_that_cannot_hold_a_comment_needs_a_sidecar(self):
        files = {"data.json": "{}", "fix.patch": "--- a\n", "NOTICE": "text\n", "mise.lock": "[tools]\n",
                 "notes.txt": "text\n"}
        problems = self.check(files)
        self.assertEqual(len(problems), 5)
        self.assertTrue(all("needs a" in p and "sidecar" in p for p in problems))
        sidecars = {f"{n}.license": f"{COPYRIGHT}\n{LICENSE}\n" for n in files}
        self.assertEqual(self.check({**files, **sidecars}), [])

    def test_a_sidecar_may_not_replace_a_possible_header_or_outlive_its_file(self):
        problems = self.check({"a.py": HASH_HEADER, "a.py.license": "x\n", "gone.json.license": "x\n"})
        self.assertEqual(len(problems), 2)
        self.assertIn("a.py: can hold a comment", problems[0])
        self.assertIn("gone.json.license: a sidecar for a file that does not exist", problems[1])

    def test_reuse_toml_and_dep5_are_refused_anywhere_tracked_or_not(self):
        problems = self.check({"REUSE.toml": "version = 1\n", "sub/REUSE.toml": "version = 1\n",
                               ".reuse/dep5": "Format: x\n"})
        self.assertEqual(len(problems), 3)
        self.assertTrue(all("can override embedded headers" in p for p in problems))
        # Git may ignore them, but REUSE still reads them.
        problems = self.check({"a.py": HASH_HEADER, ".reuse/dep5": "Format: x\n", "REUSE.toml": "version = 1\n"},
                              names=["a.py"])
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("even if Git ignores it" in p for p in problems))

    def test_unknown_types_fail_closed(self):
        problems = self.check({"logo.png": "x", "tools/helper": "echo hi\n"})
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("does not classify" in p for p in problems))

    def test_what_reuse_skips_is_skipped(self):
        files = {"LICENSE": "text", "LICENSES/MIT.txt": "text", "third_party/x/COPYING": "text",
                 "sub/LICENSE-MIT": "text", "sub/LICENSE.md": "text", "sub/COPYING.txt": "text", "empty.cc": ""}
        self.write(files)
        os.symlink("elsewhere.png", self.root / "link.png")
        checked, problems = headers.check(self.root, sorted([*files, "link.png"]), {"files": []})
        self.assertEqual((checked, problems), (0, []))

    def test_anything_but_a_license_text_named_like_one_fails(self):
        problems = self.check({"src/LICENSE-helpers.cc": "int x;\n", "tools/COPYING.py": "x = 1\n",
                               "tools/LICENSE-gen": "#!/bin/sh\n", "k/LICENSE.S": "nop\n",
                               "k/COPYING-sm121.ptx": "x\n", "p/COPYING-fix.patch": "x\n", "d/LICENSE.svg": "x\n",
                               "t/COPYING.LIB": "x\n"}, report={"files": []})
        self.assertEqual(len(problems), 8)
        self.assertTrue(all("REUSE skips files named like license texts" in p for p in problems))

    def test_licenses_holds_only_license_texts(self):
        problems = self.check({"LICENSES/LicenseRef-helper.py": "x = 1\n", "LICENSES/MIT": "text"},
                              report={"files": []})
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("LICENSES/ holds only license texts" in p for p in problems))


class ReuseReport(unittest.TestCase):
    """REUSE must read each file's license from the file's own header or sidecar, and nothing else."""

    def setUp(self):
        self.root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "a.py").write_text(HASH_HEADER)
        (self.root / "d.json").write_text("{}")
        (self.root / "d.json.license").write_text(f"{COPYRIGHT}\n{LICENSE}\n")
        self.names = ["a.py", "d.json", "d.json.license"]

    def problems(self, *entries) -> list[str]:
        return headers.check(self.root, self.names, {"files": list(entries)})[1]

    def test_agreement_passes(self):
        self.assertEqual(self.problems(entry("a.py"), entry("d.json", "d.json.license")), [])

    def test_a_hidden_header_is_caught(self):
        # A header hidden by ignore markers, with other tags later: REUSE reads the later ones.
        problems = self.problems(entry("a.py", license="GPL-3.0-only", holder="2019 Someone Else"),
                                 entry("d.json", "d.json.license"))
        self.assertTrue(all(p.startswith("a.py: ") for p in problems))
        self.assertIn("a.py: REUSE reads the license GPL-3.0-only, but the header says MIT", problems)
        self.assertIn("a.py: REUSE does not read the header's copyright 2026 Example", problems)

    def test_information_from_anywhere_else_is_caught(self):
        dep5 = entry("a.py")
        dep5["spdx_expressions"].append({"value": "MIT", "source": ".reuse/dep5"})
        problems = self.problems(dep5, entry("d.json", "d.json"))
        self.assertEqual(len(problems), 2)
        self.assertIn("a.py: REUSE reads license information from .reuse/dep5, not a.py", problems[0])
        self.assertIn("d.json: REUSE reads license information from d.json, not d.json.license", problems[1])

    def test_a_tag_reuse_does_not_read_is_caught_anywhere(self):
        deep = HASH_HEADER + "\n" * 300
        for body in (f"# {IGNORE_START}\n# {LICENSE_TAG}: GPL-3.0-only\n# {IGNORE_END}\n",  # hidden by markers
                     f"# {LICENSE_TAG}: GPL-3.0-only\n",       # a tag REUSE's report leaves out
                     f"x = '{COPYRIGHT_TAG}: 2019 Someone Else'\n"):
            with self.subTest(body=body):
                (self.root / "a.py").write_text(deep + body)
                problems = self.problems(entry("a.py"), entry("d.json", "d.json.license"))
                self.assertTrue(problems)
                self.assertTrue(all(p.startswith("a.py: ") for p in problems))

    def test_a_sidecar_files_own_tags_must_agree_with_the_sidecar(self):
        (self.root / "d.json").write_text(f'{{"x": "{LICENSE_TAG}: GPL-3.0-only"}}')
        problems = self.problems(entry("a.py"), entry("d.json", "d.json.license"))
        self.assertEqual(len(problems), 1)
        self.assertIn("d.json: a License-Identifier tag REUSE does not read for its file: 'GPL-3.0-only\"}'",
                      problems[0])
        # A tag must repeat a whole expression REUSE reads: one side of an OR, or a license without its
        # exception, would show a reader narrower terms than REUSE records.
        for content, recorded in (("MIT", "MIT"), ("GPL-3.0-only", "MIT OR GPL-3.0-only"),
                                  ("GPL-2.0-only", "GPL-2.0-only WITH Classpath-exception-2.0")):
            with self.subTest(content=content):
                (self.root / "d.json").write_text(f"{LICENSE_TAG}: {content}\n")
                problems = self.problems(entry("a.py"), entry("d.json", "d.json.license", recorded))
                self.assertEqual(problems != [], content != recorded)

    def test_a_sidecars_own_text_is_scanned_too(self):
        for text in (f"{COPYRIGHT}\n{LICENSE}\n{IGNORE_START}\n{LICENSE_TAG}: GPL-3.0-only\n{IGNORE_END}\n",
                     f"{COPYRIGHT}\nylno-0.3-LPG DNA {LICENSE_TAG}: MIT AND GPL-3.0-only\n"):
            with self.subTest(text=text):
                (self.root / "d.json.license").write_text(text)
                problems = self.problems(entry("a.py"), entry("d.json", "d.json.license"))
                self.assertTrue(problems)
                self.assertTrue(all(p.startswith("d.json.license: ") for p in problems))

    def test_every_file_is_utf8_without_control_characters(self):
        for data in (f"{LICENSE_TAG}: GPL-3.0-only".encode("utf-16"),              # UTF-16 with a BOM
                     f"{LICENSE_TAG}: GPL-3.0-only".encode("utf-16-le"),           # without one: NULs
                     f"SPDX-License-\x1b[0mIdentifier: GPL-3.0-only".encode()):   # a terminal hides the escape
            with self.subTest(data=data):
                (self.root / "d.json").write_bytes(data)
                problems = self.problems(entry("a.py"), entry("d.json", "d.json.license"))
                self.assertEqual(len(problems), 1)
                self.assertIn("d.json: not UTF-8 text without control characters", problems[0])
        (self.root / "d.json").write_text("{}")
        (self.root / "d.json.license").write_text(f"{COPYRIGHT}\n{LICENSE}\n\x00")
        problems = self.problems(entry("a.py"), entry("d.json", "d.json.license"))
        self.assertEqual(problems, ["d.json.license: not UTF-8 text without control characters"])

    def test_markup_cannot_hide_a_tag_from_the_scan(self):
        (self.root / "b.md").write_text(HTML_HEADER)
        self.names.append("b.md")
        for split in ("*Identifier*", "`Identifier`", "<span></span>Identifier", "<!-- -->Identifier"):
            with self.subTest(split=split):
                (self.root / "b.md").write_text(HTML_HEADER + f"\n{LICENSE_TAG.replace('Identifier', split)}: GPL-3.0-only\n")
                problems = self.problems(entry("a.py"), entry("b.md"), entry("d.json", "d.json.license"))
                self.assertTrue(problems)
                self.assertTrue(all(p.startswith("b.md: ") for p in problems))

    def test_correct_headers_pass_however_reuse_spells_them(self):
        for holder in ("2026 Acme\u2122 Inc.", "2024 \u682a\u5f0f\u4f1a\u793e\uff21\uff22\uff23", "2026 Lab\u00b2"):
            with self.subTest(holder=holder):
                (self.root / "a.py").write_text(f"\ufeff# {COPYRIGHT_TAG}: {holder}\n# {LICENSE}\n")
                self.assertEqual(self.problems(entry("a.py", holder=holder), entry("d.json", "d.json.license")), [])

    def test_tags_are_read_as_a_reader_sees_them(self):
        spellings = [f"{LICENSE_TAG}: GPL-3.0-only".replace("-Identifier", "\u200b-Identifier"),  # zero-width space
                     f"{LICENSE_TAG}: GPL-3.0-only".replace("-", "\u2011", 2),                    # non-breaking hyphens
                     f"{LICENSE_TAG.lower()}: GPL-3.0-only",
                     f"{LICENSE_TAG} : GPL-3.0-only",
                     f"{LICENSE_TAG}\uff1a GPL-3.0-only"]                                          # full-width colon
        for spelling in spellings:
            with self.subTest(spelling=spelling):
                (self.root / "a.py").write_text(HASH_HEADER + f"# {spelling}\n")
                self.assertTrue(self.problems(entry("a.py"), entry("d.json", "d.json.license")))
        markdown = LICENSE_TAG.replace("-", "\\-")
        self.assertEqual(headers.visible(f"{markdown}&#58; GPL", headers.HTML), f"{LICENSE_TAG}: GPL")

    def test_holders_keep_every_script(self):
        (self.root / "a.py").write_text(HASH_HEADER + "\n" * headers.HEADER_LINES
                                        + f"# {COPYRIGHT_TAG}: 2026 Example \u682a\u5f0f\u4f1a\u793e\n")
        problems = self.problems(entry("a.py"), entry("d.json", "d.json.license"))
        self.assertEqual(len(problems), 1)
        self.assertIn("FileCopyrightText tag REUSE does not read", problems[0])

    def test_copyrights_compare_as_reuse_spells_them(self):
        (self.root / "a.py").write_text(f"# {COPYRIGHT_TAG}: (c) 2026  Example\n# {LICENSE}\n")
        report = entry("a.py", holder="(C) 2026 Example")
        self.assertEqual(self.problems(report, entry("d.json", "d.json.license")), [])

    def test_coverage_must_match_both_ways(self):
        problems = self.problems(entry("d.json", "d.json.license"), entry("elsewhere.cc"))
        self.assertEqual(len(problems), 2)
        self.assertIn("a.py: REUSE lint does not cover it", problems[0])
        self.assertIn("elsewhere.cc: REUSE lint covers it, but the header check does not", problems[1])


if __name__ == "__main__":
    unittest.main()
