# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""What a model archive variant holds, read from its files' headers (docs/model-archive.md).

Bounded readers of GGUF and safetensors headers, plus config.json and
ModelOpt's hf_quant_config.json, give each variant its container,
architecture, scheme, bytes per encoding, bits, parameters and quantizer.
Only headers are read. The result describes; support is the importer's
and the admission gate's to decide (D-109).
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import struct

import llmp_archive as arc

GGUF_HEADER_CAP = 256 << 20
SAFETENSORS_HEADER_CAP = 100_000_000
CONFIG_CAP = 16 << 20
GGUF_COUNT_CAP = 1 << 24
GGUF_STRING_CAP = 1 << 24

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
              24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
              35: "TQ2_0", 39: "MXFP4"}
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
              11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
              17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
              23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
              30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}
NAME_HINT = re.compile(r"(?i)(?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z]+|Q\d_K(?:_[A-Z]+)?|Q\d_[01]|TQ\d_0|MXFP4(?:_MOE)?"
                       r"|BF16|F16|F32))(?=$|[-_.])")
DTYPE_BITS = {"F64": 64, "F32": 32, "F16": 16, "BF16": 16, "F8_E4M3": 8, "F8_E5M2": 8, "I64": 64, "I32": 32,
              "I16": 16, "I8": 8, "U8": 8, "U16": 16, "U32": 32, "U64": 64, "BOOL": 8}


class HeaderError(Exception):
    pass


class _Reader:
    def __init__(self, f, cap: int):
        self.f, self.cap, self.pos = f, cap, 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > self.cap:
            raise HeaderError(f"header over {self.cap} bytes")
        data = self.f.read(n)
        if len(data) != n:
            raise HeaderError("file ends inside its header")
        self.pos += n
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > GGUF_STRING_CAP:
            raise HeaderError("string over its cap")
        return self.take(n).decode("utf-8", "replace")


_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _gguf_value(r: _Reader, vtype: int, keep: bool):
    if vtype in _SCALARS:
        fmt = _SCALARS[vtype]
        return struct.unpack(fmt, r.take(struct.calcsize(fmt)))[0]
    if vtype == 8:
        return r.string()
    if vtype == 9:
        itype, count = r.u32(), r.u64()
        if count > GGUF_COUNT_CAP or itype == 9:
            raise HeaderError("array too long, or of arrays")
        items = [_gguf_value(r, itype, keep) for _ in range(count)]
        return items if keep and count <= 64 else None
    raise HeaderError(f"unknown value type {vtype}")


def read_gguf(path: pathlib.Path) -> dict:
    """{metadata (short values only), tensors: [(name, type, elements, offset)], data_start, size}."""
    with open(path, "rb") as f:
        r = _Reader(f, GGUF_HEADER_CAP)
        if r.take(4) != b"GGUF":
            raise HeaderError("not GGUF")
        version = r.u32()
        if version not in (2, 3):
            raise HeaderError(f"GGUF version {version}")
        n_tensors, n_kv = r.u64(), r.u64()
        if n_tensors > GGUF_COUNT_CAP or n_kv > GGUF_COUNT_CAP:
            raise HeaderError("counts over their caps")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            value = _gguf_value(r, vtype, keep=key.startswith("general.") or key.startswith("split."))
            if value is not None and not isinstance(value, list):
                meta[key] = value
            elif isinstance(value, list):
                meta[key] = value
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            if n_dims > 8:
                raise HeaderError("too many dimensions")
            dims = [r.u64() for _ in range(n_dims)]
            ttype, offset = r.u32(), r.u64()
            tensors.append((name, ttype, math.prod(dims), offset))
        alignment = meta.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
            alignment = 32
        data_start = (r.pos + alignment - 1) // alignment * alignment
    return {"metadata": meta, "tensors": tensors, "data_start": data_start, "size": path.stat().st_size}


def read_safetensors(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise HeaderError("too short for safetensors")
        n = struct.unpack("<Q", raw)[0]
        if n > SAFETENSORS_HEADER_CAP:
            raise HeaderError("header over 100,000,000 bytes")
        data = f.read(n)
    if len(data) != n:
        raise HeaderError("file ends inside its header")
    try:
        header = json.loads(data)
    except ValueError:
        raise HeaderError("header is not JSON") from None
    if not isinstance(header, dict):
        raise HeaderError("header is not an object")
    return header


def _config(directory: pathlib.Path, name: str) -> dict:
    try:
        data = arc.read_capped(directory / name, CONFIG_CAP)
        value = json.loads(data)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, RecursionError, arc.ArchiveError):
        return {}


def _name_hint(paths: list[str]) -> str | None:
    match = NAME_HINT.search(paths[0].rsplit("/", 1)[-1][:-5]) if paths else None
    return match.group(1) if match else None


def _detect_gguf(directory: pathlib.Path, paths: list[str]) -> dict:
    encodings, elements, meta, notes = {}, 0, {}, []
    for i, p in enumerate(paths):
        h = read_gguf(directory / p)
        if i == 0:
            meta = h["metadata"]
        ordered = sorted(h["tensors"], key=lambda t: t[3])
        for j, (_, ttype, count, offset) in enumerate(ordered):
            end = ordered[j + 1][3] if j + 1 < len(ordered) else h["size"] - h["data_start"]
            name = GGML_TYPES.get(ttype, f"type{ttype}")
            encodings[name] = encodings.get(name, 0) + max(0, end - offset)
            elements += count
    total = sum(encodings.values())
    ftype = meta.get("general.file_type")
    scheme = FILE_TYPES.get(ftype) if isinstance(ftype, int) else None
    if scheme is None and encodings:
        scheme = max(encodings, key=encodings.get)
    return {"container": "gguf", "architecture": meta.get("general.architecture"),
            "scheme": f"gguf:{scheme}" if scheme else None, "encodings": dict(sorted(encodings.items())),
            "bits": round(total * 8 / elements, 2) if elements else None, "parameters": elements or None,
            "quantizer": meta.get("general.quantized_by"), "notes": notes}


def _detect_safetensors(directory: pathlib.Path, paths: list[str], vid: str) -> dict:
    sub = directory if vid == "*" else directory / vid
    config = _config(sub, "config.json") or _config(directory, "config.json")
    quant_cfg = _config(sub, "hf_quant_config.json") or _config(directory, "hf_quant_config.json")
    index = _config(directory, "model_index.json")
    encodings, elements, names, mlx, notes = {}, 0, set(), False, []
    for p in paths:
        header = read_safetensors(directory / p)
        meta = header.get("__metadata__") if isinstance(header.get("__metadata__"), dict) else {}
        mlx |= meta.get("format") == "mlx"
        for name, t in header.items():
            if name == "__metadata__" or not isinstance(t, dict):
                continue
            dtype, offsets, shape = t.get("dtype"), t.get("data_offsets"), t.get("shape")
            if isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(o, int) for o in offsets):
                encodings[str(dtype)] = encodings.get(str(dtype), 0) + max(0, offsets[1] - offsets[0])
            if isinstance(shape, list) and all(isinstance(d, int) for d in shape):
                elements += math.prod(shape)
            names.add(name.rsplit(".", 1)[-1])
    qc = config.get("quantization_config") if isinstance(config.get("quantization_config"), dict) else {}
    mlx_q = config.get("quantization") if isinstance(config.get("quantization"), dict) else None
    method = str(qc.get("quant_method") or "").lower() or None
    container = "diffusers" if index else ("mlx" if mlx else ("exl3" if method == "exl3" else "safetensors"))
    scheme = bits = quantizer = None
    params = elements or None
    algo = (quant_cfg.get("quantization") or {}).get("quant_algo") if isinstance(
        quant_cfg.get("quantization"), dict) else None
    if method == "modelopt" or algo:
        algo = algo or qc.get("quant_algo")
        scheme = f"modelopt:{algo}" if algo else "modelopt"
        producer = quant_cfg.get("producer") if isinstance(quant_cfg.get("producer"), dict) else {}
        quantizer = " ".join(str(producer[k]) for k in ("name", "version") if producer.get(k)) or "modelopt"
        bits = {"NVFP4": 4, "MXFP8": 8, "FP8": 8}.get(str(algo).upper())
        if str(algo).upper() == "NVFP4" and not ("U8" in encodings and "F8_E4M3" in encodings):
            notes.append("NVFP4 without U8 codes and F8_E4M3 scales")
        params = None
    elif mlx_q is not None or mlx:
        q = mlx_q or {}
        bits = q.get("bits")
        scheme = (f"mlx:{q.get('mode', 'affine')}-{bits}bit-g{q.get('group_size')}" if mlx_q else
                  f"mlx:{max(encodings, key=encodings.get).lower()}" if encodings else "mlx")
        quantizer = "mlx"
        if mlx_q and "scales" not in names:
            notes.append("MLX quantization without scales tensors")
        if mlx_q:
            params = None
    elif method == "exl3":
        bits = qc.get("bits")
        scheme = f"exl3:{bits}bpw"
        quantizer = "exllamav3"
        params = None
        if "trellis" not in names:
            notes.append("EXL3 without trellis tensors")
    elif method:
        bits = qc.get("bits") if isinstance(qc.get("bits"), (int, float)) else None
        scheme = f"{method}:{bits}bit" if bits else method
        quantizer = method
        params = None
    elif encodings:
        dominant = max(encodings, key=encodings.get)
        scheme, bits = dominant.lower(), DTYPE_BITS.get(dominant)
    architecture = index.get("_class_name") if index else config.get("model_type") or (
        (config.get("architectures") or [None])[0] if isinstance(config.get("architectures"), list) else None)
    return {"container": container, "architecture": architecture, "scheme": scheme,
            "encodings": dict(sorted(encodings.items())), "bits": bits, "parameters": params,
            "quantizer": quantizer, "notes": notes}


def detect(directory: pathlib.Path, files: list[dict], previous: list[dict], touched: set) -> list[dict]:
    """The manifest's `variants`: kept from `previous` where none of a variant's files changed."""
    by_path = {f["path"]: f for f in files}
    old = {v["id"]: v for v in previous}
    out = []
    for vid, v in sorted(arc.group_variants(list(by_path)).items()):
        paths = v["files"]
        present = [p for p in paths if by_path[p]["present"]]
        state = True if len(present) == len(paths) else ("partial" if present else False)
        entry = {"id": vid, "files": paths, "role": v["role"], "present": state,
                 "bytes": sum(by_path[p]["size"] for p in paths)}
        prior = old.get(vid)
        if prior and prior.get("present") == state and not touched & set(paths) and prior.get("detected"):
            entry["detected"] = prior["detected"]
        else:
            hint = _name_hint(paths) if v["kind"] == "gguf" else None
            detected = {"from": "names", "container": v["kind"], "architecture": None, "scheme": None,
                        "encodings": {}, "bits": None, "parameters": None, "quantizer": None, "notes": []}
            if present and (v["kind"] == "safetensors" or present[0] == paths[0]):
                try:
                    found = (_detect_gguf(directory, present) if v["kind"] == "gguf"
                             else _detect_safetensors(directory, present, vid))
                    arc.dumps(found)  # a NaN or other value the manifest cannot hold fails here
                    detected.update(found, **{"from": "files" if state is True else "partial"})
                except Exception as error:  # noqa: BLE001 (a header the readers did not expect never blocks a manifest)
                    detected["notes"] = [f"unreadable header: {error}"]
            detected["name_hint"] = hint
            entry["detected"] = detected
        out.append(entry)
    return out
