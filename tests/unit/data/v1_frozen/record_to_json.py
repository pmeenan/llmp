#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Turns the lines a v1 equivalence test records into the frozen JSON files.

    LLMP_TEST_V1_RECORD=/path/record.txt <test binary> ...   # appends "family key value"
    record_to_json.py record.txt OUTDIR --provenance 'host, commit, v0 and v1 IDs'

Writes OUTDIR/<family>.json ({"family", "provenance", "digests"}), canonical
and sorted. A key recorded twice must repeat its value. The tests read the
files from LLMP_TEST_DATA/v1_frozen (tests/support/v1_frozen.h).
"""
import argparse
import json
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    parser.add_argument("outdir")
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()
    families: dict[str, dict[str, str]] = {}
    for number, line in enumerate(pathlib.Path(args.record).read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(" ", 2)
        if len(parts) != 3:
            print(f"line {number}: expected 'family key value'", file=sys.stderr)
            return 1
        family, key, value = parts
        old = families.setdefault(family, {}).setdefault(key, value)
        if old != value:
            print(f"{family} {key}: recorded twice with different values", file=sys.stderr)
            return 1
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for family, digests in sorted(families.items()):
        document = {"family": family, "provenance": args.provenance,
                    "digests": dict(sorted(digests.items()))}
        (out / f"{family}.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        print(f"{family}: {len(digests)} digests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
