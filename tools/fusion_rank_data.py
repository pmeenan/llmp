#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate declarative fusion comparison data and compile it to C++.

No source text is accepted from data. Kernel identities exclude this data but
include this generator; registry identity includes every generated field.
"""
import argparse
import json
import pathlib
import re
from pathlib import Path


def cpp_string(value):
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError("expected nonempty ASCII text")
    if any(ord(c) < 32 and c not in "\b\f\n\r\t" for c in value):
        raise ValueError("unsupported control character in C++ text")
    return json.dumps(value)


def recipe_cpp_string(value):
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError("expected nonempty ASCII recipe text")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("control character in recipe text")
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


def operation_tokens():
    traits = pathlib.Path(__file__).resolve().parents[1] / "src/ir/op_traits.inc"
    result = {}
    for line in traits.read_text().splitlines():
        match = re.match(r'\s*LLMP_PRIMITIVE\((k[A-Za-z0-9_]+),\s*"([a-z0-9_]+)"', line)
        if match and match.group(1) != "kFused":
            result[match.group(2)] = f"ir::Primitive(ir::Op::{match.group(1)})"
            continue
        match = re.match(r'\s*LLMP_FUSED\((k[A-Za-z0-9_]+),\s*"([a-z0-9_]+)"', line)
        if match and match.group(1) != "kNone":
            result[f"fused.{match.group(2)}"] = f"ir::Fused(ir::FusedKind::{match.group(1)})"
    return result


def arm(value):
    names = ("planned_region", "physical_recipe", "backend_recipe", "backend_build",
             "backend_selection")
    fields(value, " ".join(names))
    return "{" + ",".join(digest(value[name]) for name in names) + "}"


def recipe_comparison(value, ops):
    outer = ("key incumbent_ns alternative_ns samples protocol receipt provenance shape_class")
    if not isinstance(value, dict):
        raise ValueError("expected recipe comparison object")
    disposition = value.get("disposition", "latency")
    if disposition not in ("latency", "reject_incumbent_quality"):
        raise ValueError("unknown recipe comparison disposition")
    fields({k: v for k, v in value.items() if k != "disposition"}, outer)
    quality = disposition == "reject_incumbent_quality"
    for name in ("incumbent_ns", "alternative_ns", "samples"):
        if quality and (type(value[name]) is not int or value[name] != 0):
            raise ValueError("quality rejection must have zero timing and samples")
    key = value["key"]
    key_names = ("version region_canonicalizer discovery_operation mode subject boundary policy "
                 "device selection_environment preparation_domain incumbent alternative")
    fields(key, key_names)
    if type(key["version"]) is not int or key["version"] != 1 or \
       type(key["region_canonicalizer"]) is not int or key["region_canonicalizer"] != 2:
        raise ValueError("unknown recipe comparison key version")
    op = key["discovery_operation"]
    if not isinstance(op, str) or op not in ops:
        raise ValueError("unknown discovery operation spelling")
    mode = key["mode"]
    modes = {"full": "kFull", "warm": "kWarm"}
    if not isinstance(mode, str) or mode not in modes:
        raise ValueError("unknown recipe comparison run mode")
    provenance = value["provenance"]
    if provenance == "UNREVIEWED":
        raise ValueError("unreviewed recipe comparison cannot be installed")
    if key["incumbent"] == key["alternative"]:
        raise ValueError("recipe comparison arms must have distinct identities")
    incumbent = arm(key["incumbent"])
    alternative = arm(key["alternative"])
    literals = [
        "{.key = {" + ",".join([
            ".version = 1U", ".region_canonicalizer = 2U",
            ".discovery_operation = " + ops[op],
            ".mode = execution::RecipeRunMode::" + modes[mode],
            ".subject = " + digest(key["subject"]),
            ".boundary = " + digest(key["boundary"]),
            ".policy = " + digest(key["policy"]),
            ".device = " + digest(key["device"]),
            ".selection_environment = " + digest(key["selection_environment"]),
            ".preparation_domain = " + digest(key["preparation_domain"]),
            ".incumbent = " + incumbent,
            ".alternative = " + alternative,
        ]) + "}, " + ", ".join([
            ".incumbent_ns = " + ("0ULL" if quality else positive(value["incumbent_ns"])),
            ".alternative_ns = " + ("0ULL" if quality else positive(value["alternative_ns"])),
            ".samples = " + ("0U" if quality else positive(value["samples"], 32)),
            ".protocol = " + digest(value["protocol"]),
            ".receipt = " + digest(value["receipt"]),
            ".provenance = " + recipe_cpp_string(provenance),
            ".shape_class = " + recipe_cpp_string(value["shape_class"]),
        ]) + "}",
    ]
    if quality:
        return literals[0][:-1] + ", .kind = execution::RecipeComparisonKind::kRejectIncumbentQuality}"
    return literals[0]


def compile_data(data):
    if not isinstance(data, dict) or type(data.get("version")) is not int:
        raise ValueError("invalid comparison data version")
    version = data["version"]
    if version == 1:
        fields(data, "version comparisons")
        recipe_rows = []
    elif version == 2:
        fields(data, "version comparisons recipe_comparisons")
        recipe_rows = data["recipe_comparisons"]
        if not isinstance(recipe_rows, list):
            raise ValueError("recipe_comparisons must be an array")
    else:
        raise ValueError("invalid comparison data version")
    if not isinstance(data["comparisons"], list):
        raise ValueError("comparisons must be an array")
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
    ops = operation_tokens()
    recipe_records, recipe_seen = [], set()
    for row in recipe_rows:
        literal = recipe_comparison(row, ops)
        identity = json.dumps(row["key"], sort_keys=True, separators=(",", ":"))
        if identity in recipe_seen:
            raise ValueError("duplicate exact recipe comparison key")
        recipe_seen.add(identity)
        recipe_records.append(literal)
    return """// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Generated from declarative comparison data; do not edit.
#include "execution/rank_data.h"
namespace llmp::kernels::ggml {
std::vector<execution::FusionComparison> MeasuredFusionComparisons() {
  return {""" + ",\n".join(records) + """};
}
std::vector<execution::RecipeComparison> MeasuredRecipeComparisons() {
  return {""" + ",\n".join(recipe_records) + """};
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
