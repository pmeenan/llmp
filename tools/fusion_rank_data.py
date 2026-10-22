#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate declarative fusion comparison data and compile it to C++.

No source text is accepted from data. Kernel identities exclude this data but
include this generator; registry identity includes every generated field.
"""
import argparse
import json
import re
from pathlib import Path


def cpp_string(value):
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError("expected nonempty ASCII text")
    # JSON emits other ASCII controls as universal-character escapes, which
    # are not valid C++ string-literal characters. Keep the common escapes
    # JSON and C++ share; refuse NUL and the remaining controls explicitly.
    if any(ord(c) < 32 and c not in "\b\f\n\r\t" for c in value):
        raise ValueError("unsupported control character in C++ text")
    return json.dumps(value)


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) or set(value) == {"0"}:
        raise ValueError("expected nonzero lowercase SHA-256")
    return "{" + ",".join("0x" + value[i:i + 2] for i in range(0, 64, 2)) + "}"


def positive(value, bits=64):
    if type(value) is not int or not 0 < value < 2**bits:
        raise ValueError(f"expected positive uint{bits}")
    return str(value) + ("ULL" if bits == 64 else "U")


def fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError("unexpected or missing fields")


def implementation(value):
    fields(value, "name identity")
    if not isinstance(value["name"], str) or not re.fullmatch(r"[a-z0-9._]+", value["name"]):
        raise ValueError("invalid implementation name")
    return "{" + cpp_string(value["name"]) + "," + digest(value["identity"]) + "}"


def compile_data(data):
    fields(data, "version comparisons")
    if type(data["version"]) is not int or data["version"] != 1 or not isinstance(data["comparisons"], list):
        raise ValueError("invalid comparison data version or array")
    records, seen = [], set()
    for r in data["comparisons"]:
        fields(r, "shape_class query selection_environment fusion decomposition fused_ns decomposition_ns samples provenance")
        if r["provenance"] == "UNREVIEWED":
            raise ValueError("unreviewed measurement cannot be installed")
        fusion = implementation(r["fusion"])
        if not isinstance(r["decomposition"], list) or not r["decomposition"]:
            raise ValueError("missing own decomposition")
        parts = [implementation(p) for p in r["decomposition"]]
        if any(p["name"] == r["fusion"]["name"] for p in r["decomposition"]):
            raise ValueError("fusion is its own comparator")
        key = r["fusion"]["name"], r["query"]
        if key in seen:
            raise ValueError("duplicate fusion comparison")
        seen.add(key)
        records.append("{" + ",".join([
            cpp_string(r["shape_class"]), digest(r["query"]), digest(r["selection_environment"]), fusion,
            "{" + ",".join(parts) + "}", positive(r["fused_ns"]),
            positive(r["decomposition_ns"]), positive(r["samples"], 32),
            cpp_string(r["provenance"]),
        ]) + "}")
    return """// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Generated from declarative comparison data; do not edit.
#include "execution/rank_data.h"
namespace llmp::kernels::ggml {
std::vector<execution::FusionComparison> MeasuredFusionComparisons() {
  return {""" + ",\n".join(records) + """};
}
}  // namespace llmp::kernels::ggml
"""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.source.read_text(), object_pairs_hook=unique_object)
    result = compile_data(data)
    if not args.output.exists() or args.output.read_text() != result:
        args.output.write_text(result)


if __name__ == "__main__":
    main()
