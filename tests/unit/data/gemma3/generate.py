# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

"""Network-free facts extraction from an authenticated prefix and bounded range.

Usage: generate.py PREFIX RANGE OUTPUT
The whole-file SHA is the exact-revision primary HF API LFS/HEAD claim, not
a locally measured payload hash. Neither raw input belongs in the repository.
"""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys

REPOSITORY = "ggml-org/gemma-3-4b-it-qat-GGUF"
REVISION = "bbcac0d065076c47042838c0675c602411b0dd4c"
FULL_BYTES = 2526080992
FULL_SHA = "ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a"
PREFIX_BYTES = 6514895
PREFIX_SHA = "3073f37db9c1977ed7d68cd64707ce32043eecd250000a929a2ceb6723b1ad4c"
RANGE_BYTES = 262144
RANGE_SHA = "62abe6abbea861399b424e362fb77839ccc6ab67a9a3cb09b12bf978419d1c33"
TABLE_BYTES = 26361
TABLE_SHA = "e47e271674ae26f87da138b369f4e8518f99f1be5af0413bb9f2f98b68065b4a"
GGML_COMMIT = "d81235049384534c167caea52b85a694f6103d14"


def generate(prefix_path, range_path):
    prefix = Path(prefix_path).read_bytes()
    part = Path(range_path).read_bytes()
    if (len(prefix), hashlib.sha256(prefix).hexdigest()) != (PREFIX_BYTES, PREFIX_SHA):
        raise ValueError("prefix identity differs")
    if (len(part), hashlib.sha256(part).hexdigest()) != (RANGE_BYTES, RANGE_SHA):
        raise ValueError("range identity differs")
    repo = Path(__file__).resolve().parents[4]
    module_path = repo / "docs/experiments/artifact-layout/layout.py"
    spec = importlib.util.spec_from_file_location("layout", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Use the current native artifact type table, not a guessed quant size.
    traits_path = repo / "src/artifact/representation.cc"
    traits = {
        int(type_id): (name, int(block), int(size))
        for name, type_id, block, size in re.findall(
            r'GgmlType\{"([^"]+)", (\d+), (\d+), (\d+)\}', traits_path.read_text()
        )
    }
    f = io.BytesIO(prefix + part)
    reader = module.Reader(f, len(prefix) + len(part))
    if reader.take(4) != b"GGUF" or reader.unpack("I")[0] != 3:
        raise ValueError("not GGUF v3")
    if reader.unpack("QQ") != (444, 41):
        raise ValueError("unexpected tensor or metadata count")
    metadata = {}
    for _ in range(41):
        key = reader.string()
        if key in metadata:
            raise ValueError("duplicate metadata")
        metadata[key] = module._gguf_value(reader, reader.unpack("I")[0], True)
    if f.tell() != PREFIX_BYTES or "general.alignment" in metadata:
        raise ValueError("metadata boundary or alignment differs")
    tensors = []
    names = set()
    for _ in range(444):
        name = reader.string()
        rank = reader.unpack("I")[0]
        if name in names or not 1 <= rank <= 4:
            raise ValueError("duplicate name or invalid rank")
        names.add(name)
        ne = list(reader.unpack("Q" * rank))
        type_id = reader.unpack("I")[0]
        offset = reader.unpack("Q")[0]
        if type_id not in (0, 2, 8) or not all(0 < dim < 2**63 for dim in ne):
            raise ValueError("unexpected type or dimension")
        dtype, block, size = traits[type_id]
        if ne[0] % block:
            raise ValueError("partial quant block")
        nbytes = ne[0] // block * size
        for dim in ne[1:]:
            nbytes *= dim
        tensors.append(dict(name=name, ggml_type=type_id, type=dtype, ne=ne,
                            relative_offset=offset, bytes=nbytes))
    end = f.tell()
    if end - PREFIX_BYTES != TABLE_BYTES or hashlib.sha256(part[:TABLE_BYTES]).hexdigest() != TABLE_SHA:
        raise ValueError("tensor table identity differs")
    base = module.align(end, 32)
    ordered = sorted(tensors, key=lambda tensor: tensor["relative_offset"])
    if any(t["relative_offset"] % 32 or base + t["relative_offset"] + t["bytes"] > FULL_BYTES
           for t in ordered):
        raise ValueError("unaligned or out-of-file tensor")
    if any(a["relative_offset"] + a["bytes"] > b["relative_offset"]
           for a, b in zip(ordered, ordered[1:])):
        raise ValueError("overlapping tensors")
    if base + ordered[-1]["relative_offset"] + ordered[-1]["bytes"] != FULL_BYTES:
        raise ValueError("final tensor boundary differs")
    kept = {key: value for key, value in metadata.items()
            if key.startswith("gemma3.") or key in (
                "general.architecture", "general.license", "general.quantization_version",
                "general.file_type")}
    kept["vocab_count"] = metadata["tokenizer.ggml.tokens"]["array_len"]
    return dict(source=dict(repository=REPOSITORY, revision=REVISION,
                            file="gemma-3-4b-it-qat-Q4_0.gguf", full_bytes=FULL_BYTES,
                            full_sha256=FULL_SHA,
                            full_sha256_origin="exact-revision primary HF API LFS oid and HEAD X-Linked-ETag; no local payload hash",
                            prefix_bytes=PREFIX_BYTES, prefix_sha256=PREFIX_SHA,
                            range_start=PREFIX_BYTES, range_bytes=RANGE_BYTES,
                            range_sha256=RANGE_SHA,
                            range_response="206 bytes 6514895-6777038/2526080992",
                            ggml_type_source_commit=GGML_COMMIT,
                            ggml_type_helper_origin="retained b29c606e table; used F32/Q4_0/Q8_0 IDs and block geometry agree with d812",
                            ggml_type_helper="src/artifact/representation.cc",
                            ggml_type_helper_sha256=hashlib.sha256(traits_path.read_bytes()).hexdigest()),
                metadata=kept, tensor_table_bytes=TABLE_BYTES, tensor_table_sha256=TABLE_SHA,
                complete_header_bytes=end,
                complete_header_sha256=hashlib.sha256((prefix + part)[:end]).hexdigest(),
                alignment=32, alignment_origin="GGUF v3 default; general.alignment absent",
                data_offset=base, tensors=tensors)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    Path(sys.argv[3]).write_text(json.dumps(generate(sys.argv[1], sys.argv[2]), indent=2) + "\n")
