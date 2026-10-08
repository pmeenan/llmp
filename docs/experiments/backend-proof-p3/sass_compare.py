#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compares the SASS of every ExLlamaV3 kernel llmpalooza's port launches with the reference build's.

  sass_compare.py --reference EXT.so --port BINARY --launches REF.json... [--cuobjdump PATH] [--out OUT.json]

EXT.so is upstream's extension as the reference built it (P0's NVCC 13.4.92 bridge build,
build-cache-nvcc134, SHA-256 aa8b9f16...), BINARY a llmpalooza binary that links the port
(benchmarks/exl3_linear_sweep.cc's), and REF.json the reference runs of the per-linear sweep
(linear_reference.py --profile), whose recorded launches name every kernel the sweep's cases run.
Each function's SASS is hashed two ways, as ../backend-proof-p0/fp16_plan.py sass-hash does: its
instruction text with addresses stripped, and its raw encodings. Unlike P0's sass_hashes.py, a
function ends at the next function or at the end of its cubin (a "Fatbin" or "...." line), so no
hash takes in another member's header. A kernel matches when both hashes are equal.

Checked: every ExLlamaV3 kernel the reference runs launched (cuBLAS's and PyTorch's are not the
port's) exists in both binaries with identical SASS; and every kernel the port's binary holds
that the extension also holds is reported, identical or not. Exits 1 on any launched kernel
missing or different. Reference-only experiment tooling.
"""

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend-proof-p0"))
import fp16_plan  # noqa: E402


def hashes(binary, cuobjdump):
    text = subprocess.run([cuobjdump, "-sass", str(binary)], capture_output=True, text=True, check=True).stdout
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fp16_plan.sass_hash(str(binary), io.StringIO(text))
    table = {}
    for line in out.getvalue().splitlines():
        record = json.loads(line)
        table.setdefault(record["function"], set()).add(
            (record["text_sha256"], record["encoding_sha256"], record["instructions"]))
    names = sorted(table)
    demangled = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True,
                               check=True).stdout.splitlines()
    return table, dict(zip(demangled, names))


def library(name):
    return name.startswith(("nvjet", "void cutlass", "cutlass", "void at::", "void (anonymous namespace)"))


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--port", type=Path, required=True)
    parser.add_argument("--launches", type=Path, nargs="+", required=True)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    version = [line for line in subprocess.run([args.cuobjdump, "--version"], capture_output=True,
                                               text=True).stdout.splitlines() if "release" in line]
    reference, reference_names = hashes(args.reference, args.cuobjdump)
    port, port_names = hashes(args.port, args.cuobjdump)
    launched = set()
    for path in args.launches:
        for case in json.loads(path.read_text())["cases"]:
            for launch in case.get("launches") or []:
                if not library(launch["name"]):
                    launched.add(launch["name"])
    problems, kernels = [], []
    for name in sorted(launched):
        ref_mangled, port_mangled = reference_names.get(name), port_names.get(name)
        if ref_mangled is None or port_mangled is None:
            problems.append(f"{name}: in the reference {ref_mangled is not None}, in the port {port_mangled is not None}")
            continue
        ref_sass, port_sass = reference[ref_mangled], port[port_mangled]
        same = len(ref_sass) == 1 and ref_sass == port_sass
        (text, encoding, instructions), = sorted(port_sass)[:1]
        kernels.append({"kernel": name, "identical": same, "text_sha256": text, "encoding_sha256": encoding,
                        "instructions": instructions})
        if not same:
            problems.append(f"{name}: SASS differs")
    common = sorted(set(port) & set(reference))
    identical = [n for n in common if len(reference[n]) == 1 and reference[n] == port[n]]
    summary = {
        "cuobjdump": version[-1].strip() if version else args.cuobjdump,
        "reference": {"path": str(args.reference), "sha256": digest(args.reference), "functions": len(reference)},
        "port": {"path": str(args.port), "sha256": digest(args.port), "functions": len(port)},
        "launched": len(launched), "launched_identical": sum(k["identical"] for k in kernels),
        "port_functions_in_reference": len(common), "port_functions_identical": len(identical),
        "port_functions_differing": [n for n in common if n not in identical],
        "kernels": kernels,
    }
    text = json.dumps(summary, indent=1) + "\n"
    if args.out:
        args.out.write_text(text)
    print(json.dumps({k: v for k, v in summary.items() if k != "kernels"}, indent=1))
    for problem in problems:
        print("  ", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
