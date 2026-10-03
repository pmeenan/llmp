#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Writes the reference for Python's str case operations (README.md).

    pycase_reference.py OUT.json

Run with the reference Python (3.12.3, Unicode 15.0.0) and Jinja2 3.1.6
(the tokenizer reference's environment). For every code point it records
where str.upper(), str.lower() and str.title() of the one-character string
differ from it (str.capitalize() of one character is its title()), and the
code points for which str.islower() and str.isupper() hold, as ranges; then
seeded random strings from a pool of the hard cases (sigma in every
context, case-ignorable and cased neighbours, title-case letters,
SpecialCasing's expansions, Jinja2's title separators) with each
operation's result and Jinja2's `title` filter's. `pycase_test` compares
chat/pycase.h with it over all 1,114,112 code points and every string.
"""

import sys

sys.dont_write_bytecode = True

import json
import random
import unicodedata

import jinja2.filters


def ranges(cps):
    out, start, prev = [], None, None
    for cp in cps:
        if start is None:
            start = prev = cp
        elif cp == prev + 1:
            prev = cp
        else:
            out.append([start, prev])
            start = prev = cp
    if start is not None:
        out.append([start, prev])
    return out


def main() -> int:
    singles = {"upper": {}, "lower": {}, "title": {}}
    lower_true, upper_true = [], []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue  # not representable in UTF-8 text
        c = chr(cp)
        for name in singles:
            mapped = getattr(c, name)()
            if mapped != c:
                singles[name][f"{cp:X}"] = [ord(x) for x in mapped]
        if c.islower():
            lower_true.append(cp)
        if c.isupper():
            upper_true.append(cp)

    pool = ["Σ", "σ", "ς", "a", "A", "'", "ͅ", "́", ".", " ", "-", "(", "[", "{", "<", "\t", "　",
            "ß", "İ", "ﬃ", "ŉ", "ǅ", "ǆ", "Ǆ", "ΐ", "ᾳ", "ᾼ", "1", "東", "Ⅰ", "ⅰ", "Ⓐ", "ⓐ", "­", "ʰ",
            "Ä", "ä", "ǰ", "İ", "ﬀ", "x", "_", "‍", ":", "Ω", "ω", "ʼ"]
    rng = random.Random(20261002)
    texts = ["", "Σ", "ΑΣ", "ΑΣ'", "ΑΣ'Α", "Σ'Α", "'Σ", "ΑΣ Α", "aΣb", "aΣ.", "hello world's", "o'neil-o'NEIL (x)",
             "ǆemal ǅ", "straße", "İstanbul", "ﬃ ﬀ", "ΑΣ-Β", "aͅΣ", "ΣΣ", "Σ́"]
    for _ in range(1000):
        texts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 8))))
    strings = []
    for t in texts:
        strings.append({"text": t, "upper": t.upper(), "lower": t.lower(), "capitalize": t.capitalize(),
                        "title": t.title(), "jinja_title": jinja2.filters.do_title(t), "islower": t.islower(),
                        "isupper": t.isupper()})
    out = {"python": sys.version.split()[0], "unicode": unicodedata.unidata_version,
           "jinja2": jinja2.__version__, "singles": singles, "islower": ranges(lower_true),
           "isupper": ranges(upper_true), "strings": strings}
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
        f.write("\n")
    # Case data of the Unicode Character Database, as Python holds it. The
    # tags are spelled in parts, so that REUSE does not read them as this
    # script's own.
    copyright_tag, license_tag = "SPDX-" + "FileCopyrightText", "SPDX-" + "License-Identifier"
    with open(sys.argv[1] + ".license", "w", encoding="utf-8") as f:
        f.write(f"{copyright_tag}: 2026 jitLLM contributors\n"
                f"{copyright_tag}: 1991-2023 Unicode, Inc.\n"
                f"{license_tag}: Apache-2.0 AND Unicode-3.0\n")
    print({k: len(v) for k, v in singles.items()}, len(lower_true), len(upper_true), len(strings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
