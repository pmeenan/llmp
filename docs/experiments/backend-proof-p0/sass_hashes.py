#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel SASS hashes of a CUDA binary, and comparisons between builds.

  sass_hashes.py hash BINARY [--cuobjdump PATH]          -> {demangled name: sha256}
  sass_hashes.py compare A.json B.json                    -> counts and the identical kernels
  sass_hashes.py merge HASHES.json LAUNCH.json... --out OUT.json
                                                          -> launch records with sass_sha256

The SASS of each function (cuobjdump -sass) is hashed with its instruction
addresses stripped; names are demangled with c++filt. `merge` adds
`sass_sha256` to every launch of exl3_launch_record.py outputs whose kernel
name matches, and fails if an ExLlamaV3 launch (not cuBLAS's nvjet_* or
cutlass kernels, not PyTorch's at:: kernels) has no hash.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def hashes(binary, cuobjdump):
    text = subprocess.run([cuobjdump, "-sass", str(binary)], capture_output=True, text=True, check=True).stdout
    funcs, current, lines = {}, None, []
    for line in text.splitlines(keepends=True):
        m = re.match(r"\s+Function : (\S+)", line)
        if m:
            if current:
                funcs[current] = hashlib.sha256("".join(lines).encode()).hexdigest()
            current, lines = m.group(1), []
        elif current is not None:
            lines.append(re.sub(r"/\*[0-9a-f]{4,}\*/", "", line))
    if current:
        funcs[current] = hashlib.sha256("".join(lines).encode()).hexdigest()
    names = list(funcs)
    demangled = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True,
                               check=True).stdout.splitlines()
    return {d: funcs[n] for n, d in zip(names, demangled)}


def library(name):
    return name.startswith(("nvjet", "void cutlass", "void at::", "cutlass"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("hash")
    h.add_argument("binary", type=Path)
    h.add_argument("--cuobjdump", default="cuobjdump")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    m = sub.add_parser("merge")
    m.add_argument("hashes", type=Path)
    m.add_argument("launches", type=Path, nargs="+")
    m.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "hash":
        print(json.dumps(hashes(args.binary, args.cuobjdump), indent=1, sort_keys=True))
    elif args.command == "compare":
        a, b = json.loads(args.a.read_text()), json.loads(args.b.read_text())
        common = sorted(set(a) & set(b))
        same = [k for k in common if a[k] == b[k]]
        print(json.dumps({"a": str(args.a), "b": str(args.b), "kernels_a": len(a), "kernels_b": len(b),
                          "common": len(common), "identical_sass": len(same), "identical": same}, indent=1))
    else:
        table = json.loads(args.hashes.read_text())
        merged, missing = {}, []
        for path in args.launches:
            record = json.loads(path.read_text())
            for case in record["cases"]:
                for launch in case["launches"]:
                    if launch["name"] in table:
                        launch["sass_sha256"] = table[launch["name"]]
                    elif not library(launch["name"]):
                        missing.append(launch["name"])
            merged[path.stem.replace("launch-g-", "")] = record
        if missing:
            raise SystemExit(f"{len(missing)} ExLlamaV3 launches without a SASS hash, e.g. {missing[0]}")
        args.out.write_text(json.dumps(merged, indent=1) + "\n")


if __name__ == "__main__":
    main()
