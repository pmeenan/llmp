#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Turns the lines a v1 equivalence test records into the frozen JSON files.

One-command re-record (runs CTest with LLMP_TEST_V1_RECORD set and updates
tests/unit/data/v1_frozen/<family>.json, merging into existing keys by default):

    python3 tests/unit/data/v1_frozen/record_to_json.py \\
        --ctest 'GemmaNativeV1ServingGpu' --provenance 'host, commit, reason'

Or from a recorded text file:

    LLMP_TEST_V1_RECORD=/path/record.txt <test binary> ...   # appends "family key value"
    record_to_json.py record.txt [OUTDIR] --provenance 'host, commit, v0 and v1 IDs'

Writes OUTDIR/<family>.json ({"family", "provenance", "digests"}), canonical
and sorted. A key recorded twice in one run must repeat its value. By default
existing keys in OUTDIR/<family>.json not touched by the run are preserved
(pass --replace to overwrite the whole family file).
"""
import argparse
import datetime
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]


def default_provenance() -> str:
    host = socket.gethostname()
    commit = "unknown"
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip():
            commit = res.stdout.strip()
    except OSError:
        pass
    today = datetime.date.today().isoformat()
    return f"Recorded {today} on {host} (commit {commit})."


def parse_record_lines(text: str) -> tuple[dict[str, dict[str, str]], str | None]:
    families: dict[str, dict[str, str]] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(" ", 2)
        if len(parts) != 3:
            return {}, f"line {number}: expected 'family key value'"
        family, key, value = parts
        old = families.setdefault(family, {}).setdefault(key, value)
        if old != value:
            return {}, f"{family} {key}: recorded twice with different values"
    return families, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", nargs="?", help="recorded text file (omit when using --ctest)")
    parser.add_argument("outdir", nargs="?", default=str(SCRIPT_DIR),
                        help="output directory (default: tests/unit/data/v1_frozen)")
    parser.add_argument("--ctest", metavar="REGEX",
                        help="run ctest -R REGEX with LLMP_TEST_V1_RECORD set and update JSON files")
    parser.add_argument("--preset", default="spark-native",
                        help="build/test preset directory for --ctest (default: spark-native)")
    parser.add_argument("--provenance", help="provenance string to store in updated JSON files")
    parser.add_argument("--replace", action="store_true",
                        help="replace all digests in touched family files instead of merging")
    args = parser.parse_args()

    if bool(args.record) == bool(args.ctest):
        parser.error("specify either a RECORD file or --ctest REGEX")

    if args.ctest:
        with tempfile.NamedTemporaryFile(prefix="llmp-v1-record-", suffix=".txt", delete=False) as tmp:
            record_path = pathlib.Path(tmp.name)
        try:
            env = dict(os.environ, LLMP_TEST_V1_RECORD=str(record_path))
            cmd = ["mise", "run", "test", "--", args.preset, "--locked", "--",
                   "-R", args.ctest, "--output-on-failure"]
            res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
            if res.returncode != 0:
                print(f"ctest failed with exit code {res.returncode}", file=sys.stderr)
                return res.returncode
            text = record_path.read_text()
        finally:
            record_path.unlink(missing_ok=True)
    else:
        text = pathlib.Path(args.record).read_text()

    families, err = parse_record_lines(text)
    if err is not None:
        print(err, file=sys.stderr)
        return 1
    if not families:
        print("no digest lines recorded", file=sys.stderr)
        return 1

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for family, digests in sorted(families.items()):
        path = out / f"{family}.json"
        existing: dict = {}
        if path.is_file() and not args.replace:
            try:
                existing = json.loads(path.read_text())
            except (OSError, ValueError):
                existing = {}
        merged_digests = dict(existing.get("digests", {})) if not args.replace else {}
        merged_digests.update(digests)
        provenance = args.provenance or existing.get("provenance") or default_provenance()
        document = dict(existing)
        document["family"] = family
        document["provenance"] = provenance
        document["digests"] = dict(sorted(merged_digests.items()))
        path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        print(f"{family}: {len(digests)} recorded ({len(merged_digests)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
