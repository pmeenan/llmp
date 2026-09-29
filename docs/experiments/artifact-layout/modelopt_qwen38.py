# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 Flash Next's ModelOpt checkpoint as a D-056 artifact (M3).

import_m3.py runs this for safetensors sources. The checkpoint (NVFP4 routed
experts, MXFP8 attention, linear attention and shared expert, BF16 elsewhere,
an NVFP4 n-gram table in 128 shards) is planned and written with layout.py's
container, index and verifier, but its bytes are repacked on the way, so it
has its own writer: layout.build copies source ranges verbatim.

What is repacked, all losslessly (docs/experiments/qwen38-native/README.md):

- Routed experts: ModelOpt NVFP4 (two E2M1 codes per byte, element 2k in the
  low nibble; one E4M3 scale per 16 elements in a separate tensor) is written
  in the layout CUTLASS's SM1xx block-scaled grouped GEMM and jitLLM's vector
  products read (src/kernels/ggml/moe_layout.h; docs/artifact-format.md,
  "Executable views"), so the runtime maps each expert group as it is. An
  expert group holds four byte arrays (GGML I8, packed back to back): gate's
  and up's codes as one block of 2f rows (verbatim: ModelOpt's codes are
  CUTLASS's), their scales swizzled into 128-row by 4-scale atoms of 512
  bytes (expert_scales_sf1xx), then down's codes and scales the same way.
  Each layer's per-expert global scales (weight_scale_2) are gathered into
  F32 [experts] vectors in the layer group. The first Qwen3.8 artifact wrote
  GGML's block_nvfp4 instead (nvfp4_to_ggml, kept as the tests' reference),
  which the Qwen3.8 harness rewrote into this layout at every load.
- MXFP8 matrices keep ModelOpt's layout (E4M3 [out, in] and E8M0 [out, in/32])
  as plain resources, `.weight` and `.weight_scale`.
- The n-gram table's shards are concatenated, each row interleaved as its 80
  code bytes then its 10 scale bytes (90 bytes), so a row lookup reads one
  contiguous row; its global scale is a plain F32 [1].
- Linear attention's value heads are reordered from grouped (by key head) to
  tiled order, as llama.cpp's converter does (conversion/qwen.py
  _LinearAttentionVReorderBase): rows of the QKV (value part), Z, alpha and
  beta projections, the dt bias, A and the convolution's value channels, and
  the output projection's input columns (whole 128-column heads, so their
  32-column MXFP8 scale blocks move with them).
- BF16 matrices keep their bytes as GGML BF16; norm weights become F32 with
  the (1 + w) the model applies folded in (all but linear attention's gated
  norm, as llama.cpp's converter); A becomes -exp(A_log) in F32; dt_bias and
  the convolution kernels become F32.

Names follow llama.cpp's qwen4exp tensors (src/llama-arch.cpp), the graph the
native one ports. The vision tower and the MTP block are not imported.

The MTP block is its own drafter artifact (plan_mtp, build_mtp;
architecture qwen4exp-mtp), as DeepSeek's DSpark drafter is (D-089's note):
one full-attention layer with hyper-connections and the routed experts, the
fc_embedding and fc_hidden projections with their norms, and its own final
mixer, from the checkpoint's last shard alone. It holds no token table or
head: the runtime binds the target artifact's (vLLM's Qwen3_8FlashNextMTP
loads the same two). Its attention, indexer, router and shared expert are
BF16 in the checkpoint and stay so; its experts take the target's CUTLASS
layout; norms fold (1 + w) as the target's. Importing the drafter reads and
hashes only that shard and config.json, so the target artifact is neither
re-imported nor changed.
"""
import array
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import struct
from pathlib import Path

ARCH = "qwen4exp"
PREFIX = "model.language_model."

# ------------------------------------------------------------------ config


class Config:
    """The text model's hyperparameters from config.json, validated as
    untrusted input: every value the plan or a tensor shape depends on is
    bounded and consistent, or the import is refused."""

    def __init__(self, doc):
        t = doc.get("text_config", doc) if type(doc) is dict else None
        if type(t) is not dict:
            raise ValueError("config.json: no text_config object")

        def num(key, lo=1, hi=1 << 20):
            v = t.get(key)
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError(f"config.json: {key} = {v!r} is not an integer in [{lo}, {hi}]")
            return v

        self.hidden = num("hidden_size")
        self.layers = num("num_hidden_layers", 1, 1024)
        types = t.get("layer_types")
        if type(types) is not list or len(types) != self.layers or \
                not all(x in ("linear_attention", "full_attention") for x in types):
            raise ValueError("config.json: layer_types must name each layer linear_attention or full_attention")
        self.linear = [x == "linear_attention" for x in types]
        self.heads = num("num_attention_heads", 1, 1024)
        self.kv_heads = num("num_key_value_heads", 1, 1024)
        self.head_dim = num("head_dim", 1, 4096)
        self.lin_k_heads = num("linear_num_key_heads", 1, 1024)
        self.lin_v_heads = num("linear_num_value_heads", 1, 1024)
        self.lin_k_dim = num("linear_key_head_dim", 1, 4096)
        self.lin_v_dim = num("linear_value_head_dim", 1, 4096)
        self.conv = num("linear_conv_kernel_dim", 1, 64)
        self.experts = num("num_experts", 1, 1 << 16)
        self.experts_used = num("num_experts_per_tok", 1, 1 << 16)
        self.moe_ff = num("moe_intermediate_size")
        self.shared_ff = num("shared_expert_intermediate_size")
        self.hc = num("hc_count", 2, 64)
        self.hc_rank = num("hc_lowrank")
        self.idx_heads = num("indexer_n_heads", 1, 1024)
        self.idx_dim = num("indexer_head_dim", 1, 4096)
        self.idx_kv_heads = num("indexer_kv_heads", 1, 1)
        self.ngram = num("ngram_size", 2, 16)
        self.heads_per_ngram = num("heads_per_ngram", 1, 1024)
        self.ple_parts = num("split_ngram_parts", 1, 1 << 16)
        self.ple_dim = num("ple_embed_dim")
        self.ple_conv = num("ple_conv_kernel_size", 1, 64)
        self.vocab = num("vocab_size", 1, 1 << 24)
        ple = t.get("ple_layer_ids")
        if type(ple) is not list or len(ple) != 1 or type(ple[0]) is not int or not 1 <= ple[0] <= self.layers:
            raise ValueError("config.json: ple_layer_ids must name exactly one layer (1-based)")
        self.ple_layer = ple[0] - 1
        if t.get("tie_word_embeddings") is not False or t.get("output_gate_type") != "sigmoid":
            raise ValueError("config.json: untied embeddings and a sigmoid output gate are required")
        self.ple_heads = (self.ngram - 1) * self.heads_per_ngram
        if self.experts_used > self.experts:
            raise ValueError("config.json: more experts used than exist")
        if self.lin_v_heads % self.lin_k_heads or self.ple_dim % self.ple_heads:
            raise ValueError("config.json: value heads must group evenly over key heads, and the n-gram "
                             "embedding over its heads")
        self.ple_row = self.ple_dim // self.ple_heads  # values per table row
        if not self.linear[self.ple_layer]:
            raise ValueError("config.json: the n-gram layer must be a linear attention layer")
        # Block sizes the formats need: NVFP4 GGML blocks of 64 along every
        # expert product's input, MXFP8 blocks of 32 along every MXFP8 one's,
        # NVFP4 sub-blocks of 16 along a table row.
        for n in (self.hidden, self.moe_ff):
            if n % 64:
                raise ValueError("config.json: NVFP4 expert inputs must be multiples of 64")
        for n in (self.hidden, self.lin_v_heads * self.lin_v_dim, self.heads * self.head_dim,
                  self.shared_ff, self.lin_v_dim):
            if n % 32:
                raise ValueError("config.json: MXFP8 inputs must be multiples of 32")
        if self.ple_row % 16:
            raise ValueError("config.json: an n-gram table row must be a multiple of 16 values")

    @property
    def conv_dim(self):
        return 2 * self.lin_k_heads * self.lin_k_dim + self.lin_v_heads * self.lin_v_dim

    @property
    def hc_dim(self):
        return self.hc * self.hidden


# ------------------------------------------------------------------ byte repacks

def _or_bytes(a, b):
    """Bytewise OR of two equal-length byte strings."""
    n = len(a)
    return (int.from_bytes(a, "little") | int.from_bytes(b, "little")).to_bytes(n, "little")


_LO = bytes(x & 0x0F for x in range(256))
_LO_TO_HI = bytes((x & 0x0F) << 4 for x in range(256))
_HI_TO_LO = bytes(x >> 4 for x in range(256))
_HI = bytes(x & 0xF0 for x in range(256))

# Scale values ModelOpt produces, and the only ones whose meaning every reader
# agrees on. An NVFP4 block scale is a non-negative finite E4M3 (0x00-0x7E):
# GGML reads it as UE4M3, its CPU path ignoring the sign bit, its CUDA
# dequantizers honouring it and turning NaN (0x7F, 0xFF) into 0, and the
# Blackwell MMQ path handing the raw byte to the tensor core's unsigned
# ue4m3 scale; jitLLM's n-gram lookup decodes signed E4M3 with NaN. An E8M0
# scale is finite below 0xFF (NaN). A global scale is a positive finite F32.
_E4M3_SCALE = bytes(range(0x7F))
_E8M0_SCALE = bytes(range(0xFF))


def check_e4m3_scales(data, where):
    bad = bytes(data).translate(None, _E4M3_SCALE)
    if bad:
        raise ValueError(f"{where}: {len(bad)} E4M3 scales negative or NaN (e.g. {bad[0]:#04x})")


def check_e8m0_scales(data, where):
    if bytes(data).translate(None, _E8M0_SCALE):
        raise ValueError(f"{where}: E8M0 scale 0xff (NaN)")


def check_global_scales(data, where):
    for v in _floats(data):
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"{where}: global scale {v!r} is not positive and finite")
    return data


def nvfp4_to_ggml(codes, scales, rows, k):
    """ModelOpt NVFP4 [rows, k] (codes: k/2 bytes a row, element 2i in the low
    nibble; scales: k/16 E4M3 bytes a row) as GGML block_nvfp4 rows: per 64
    elements, the 4 scale bytes then 32 code bytes, byte j of each 16-element
    sub-block holding element j (low nibble) and element j + 8 (high). GGML
    reads a scale byte as half its E4M3 value against doubled E2M1 values
    (kvalues_fp4), so the bytes carry over unchanged."""
    if k % 64 or len(codes) != rows * k // 2 or len(scales) != rows * k // 16:
        raise ValueError("NVFP4 tensor sizes disagree with its shape")
    check_e4m3_scales(scales, "NVFP4 block scales")
    codes = bytes(codes)
    # Within each sub-block (8 code bytes, elements 0-15 in nibble order),
    # output byte 2i takes elements 2i and 2i + 8 ... as nibble pairs:
    # out[2i]   = lo(in[i]) | lo(in[4 + i]) << 4   (elements 2i, 2i + 8)
    # out[2i+1] = hi(in[i]) | hi(in[4 + i])        (elements 2i + 1, 2i + 9)
    shuffled = bytearray(len(codes))
    for i in range(4):
        a, b = codes[i::8], codes[4 + i::8]
        shuffled[2 * i::8] = _or_bytes(a.translate(_LO), b.translate(_LO_TO_HI))
        shuffled[2 * i + 1::8] = _or_bytes(a.translate(_HI_TO_LO), b.translate(_HI))
    out = bytearray(rows * k // 64 * 36)
    for i in range(4):
        out[i::36] = scales[i::4]
    for j in range(32):
        out[4 + j::36] = shuffled[j::32]
    return bytes(out)


SF_ATOM_ROWS = 128  # rows of one scale atom
SF_ATOM_BYTES = 512  # 128 rows x 4 scales


def sf1xx_atoms(rows, k):
    """The 512-byte atoms of a [rows, k] NVFP4 matrix's swizzled scales
    (rows padded to 128)."""
    return -(-rows // SF_ATOM_ROWS) * (k // 64)


def sf1xx_offset(row, block, blocks):
    """The byte of row `row`'s scale `block` (of `blocks`, a multiple of 4) in
    CUTLASS's SM1xx scale layout: moe_cutlass.h SfOffset."""
    return ((((row // 128) * (blocks // 4)) + (block // 4)) * 512 + (row % 32) * 16
            + ((row % 128) // 32) * 4 + block % 4)


def expert_scales_sf1xx(scales, rows, k):
    """E4M3 scales [rows, k/16], row-major as ModelOpt stores them, in CUTLASS's
    swizzled layout (sf1xx_offset); rows past `rows` up to the atom's 128 are
    zero. Every 4 scales of a row stay together, so the permutation moves
    32-bit words: row r's words go to one lane of each of its atoms."""
    blocks = k // 16
    if k % 64 or rows <= 0 or len(scales) != rows * blocks:
        raise ValueError("NVFP4 scale sizes disagree with their shape")
    check_e4m3_scales(scales, "NVFP4 block scales")
    words = blocks // 4
    src = array.array("I")
    if src.itemsize != 4:
        raise ValueError("no 32-bit array type")
    src.frombytes(bytes(scales))
    out = array.array("I", bytes(sf1xx_atoms(rows, k) * SF_ATOM_BYTES))
    lanes = SF_ATOM_BYTES // 4
    for r in range(rows):
        first = (r // SF_ATOM_ROWS) * words * lanes + (r % 32) * 4 + (r % SF_ATOM_ROWS) // 32
        out[first:first + words * lanes:lanes] = src[r * words:(r + 1) * words]
    return out.tobytes()


def expert_to_sf1xx(gate, up, down, ffn, width):
    """One expert's (codes, scales) of gate and up [ffn, width] and down [width,
    ffn] as the four arrays of its group: gate's and up's codes, their scales
    as one 2·ffn-row matrix, down's codes, down's scales."""
    for (codes, scales), (rows, k) in zip((gate, up, down), ((ffn, width), (ffn, width), (width, ffn))):
        if k % 64 or len(codes) != rows * k // 2 or len(scales) != rows * k // 16:
            raise ValueError("NVFP4 tensor sizes disagree with its shape")
    return [bytes(gate[0]) + bytes(up[0]),
            expert_scales_sf1xx(bytes(gate[1]) + bytes(up[1]), 2 * ffn, width),
            bytes(down[0]),
            expert_scales_sf1xx(down[1], width, ffn)]


def ggml_to_sf1xx_reference(gate, up, down, ffn, width):
    """The Qwen3.8 harness's load-time conversion (jitllm_moe.cu ConvertBlocks)
    of an expert's GGML block_nvfp4 projections, byte by byte (tests)."""
    def part(projections, rows, k):
        codes = bytearray(rows * k // 2)
        scales = bytearray(sf1xx_atoms(rows, k) * SF_ATOM_BYTES)
        for data, row0, rows_here in projections:
            for r in range(rows_here):
                for blk in range(k // 64):
                    g = data[(r * (k // 64) + blk) * 36:(r * (k // 64) + blk + 1) * 36]
                    v = [0] * 64
                    for s in range(4):
                        for j in range(8):
                            v[16 * s + j] = g[4 + 8 * s + j] & 15
                            v[16 * s + j + 8] = g[4 + 8 * s + j] >> 4
                        scales[sf1xx_offset(row0 + r, blk * 4 + s, k // 16)] = g[s]
                    at = (row0 + r) * (k // 2) + blk * 32
                    codes[at:at + 32] = bytes(v[2 * b] | v[2 * b + 1] << 4 for b in range(32))
        return bytes(codes), bytes(scales)
    gu_codes, gu_scales = part([(gate, 0, ffn), (up, ffn, ffn)], 2 * ffn, width)
    d_codes, d_scales = part([(down, 0, width)], width, ffn)
    return [gu_codes, gu_scales, d_codes, d_scales]


def nvfp4_to_ggml_reference(codes, scales, rows, k):
    """The same as nvfp4_to_ggml, element by element (tests)."""
    out = bytearray()
    for r in range(rows):
        vals = []
        for i in range(k // 2):
            byte = codes[r * k // 2 + i]
            vals += [byte & 15, byte >> 4]
        for blk in range(k // 64):
            out += bytes(scales[r * k // 16 + blk * 4:r * k // 16 + blk * 4 + 4])
            for sub in range(4):
                e = vals[blk * 64 + sub * 16:blk * 64 + sub * 16 + 16]
                out += bytes(e[j] | e[j + 8] << 4 for j in range(8))
    return bytes(out)


def bf16_to_f32(data):
    out = bytearray(len(data) * 2)
    out[2::4] = data[0::2]
    out[3::4] = data[1::2]
    return bytes(out)


def _unique_keys(pairs):
    """json object_pairs_hook refusing duplicate keys (json.loads keeps the
    last; another reader may keep the first)."""
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def _floats(data_f32):
    a = array.array("f")
    a.frombytes(data_f32)
    return a


def norm_plus_one(data_bf16):
    """F32 (1 + w) of BF16 w: exact in double, one rounding to F32 as torch's
    F32 add gives."""
    return array.array("f", [x + 1.0 for x in _floats(bf16_to_f32(data_bf16))]).tobytes()


def neg_exp(data_bf16):
    return array.array("f", [-math.exp(x) for x in _floats(bf16_to_f32(data_bf16))]).tobytes()


def tiled_order(k_heads, v_per_k):
    """Value heads in tiled order: output head t = j·k_heads + g holds the
    grouped head g·v_per_k + j (llama.cpp's _reorder_v_heads)."""
    return [g * v_per_k + j for j in range(v_per_k) for g in range(k_heads)]


def reorder_rows(data, row_bytes, order, head_rows, first_row=0):
    """Rows [first_row, first_row + len(order)·head_rows) permuted in blocks of
    head_rows rows: output block t is input block order[t]."""
    base = first_row * row_bytes
    block = head_rows * row_bytes
    if sorted(order) != list(range(len(order))) or base + len(order) * block > len(data):
        raise ValueError("row reorder out of range")
    pieces = [data[:base]]
    pieces += [data[base + h * block:base + (h + 1) * block] for h in order]
    pieces.append(data[base + len(order) * block:])
    out = b"".join(pieces)
    if len(out) != len(data):
        raise ValueError("row reorder out of range")
    return out


def reorder_cols(data, rows, row_bytes, order, head_bytes):
    """Within every row, the first len(order)·head_bytes bytes permuted in
    blocks of head_bytes."""
    if (len(order) * head_bytes != row_bytes or len(data) != rows * row_bytes
            or sorted(order) != list(range(len(order)))):
        raise ValueError("column reorder out of range")
    out = bytearray(len(data))
    for t, h in enumerate(order):
        for r in range(rows):
            src = r * row_bytes + h * head_bytes
            dst = r * row_bytes + t * head_bytes
            out[dst:dst + head_bytes] = data[src:src + head_bytes]
    return bytes(out)


def interleave_rows(a, a_row, b, b_row, rows):
    """Rows of a (a_row bytes) each followed by the same row of b (b_row)."""
    width = a_row + b_row
    out = bytearray(rows * width)
    for i in range(a_row):
        out[i::width] = a[i::a_row]
    for i in range(b_row):
        out[a_row + i::width] = b[i::b_row]
    return bytes(out)


# ------------------------------------------------------------------ sources


class Sources:
    """The checkpoint's tensors by name (layout.read_safetensors validated each
    header) and a reader for their bytes."""

    def __init__(self, layout, paths):
        self.parts = [layout.read_safetensors(p) for p in paths]
        self.tensors = {}
        for part in self.parts:
            # read_safetensors keeps a duplicated name's last entry; refuse it.
            with open(part["path"], "rb") as fh:
                head = fh.read(part["header_len"])
            if hashlib.sha256(head).hexdigest() != part["header_sha256"]:
                raise ValueError(f"{part['path']}: header changed while it was read")
            json.loads(head[8:].decode("utf-8"), object_pairs_hook=_unique_keys)
            for t in part["tensors"]:
                if t["name"] in self.tensors:
                    raise ValueError("duplicate tensor across shards: " + t["name"])
                self.tensors[t["name"]] = t
        self.handles = {}

    def get(self, name, dtype, shape):
        t = self.tensors.get(name)
        if t is None:
            raise ValueError(f"the checkpoint has no {name}")
        if t["dtype"] != dtype or list(t["shape"]) != list(shape):
            raise ValueError(f"{name} is {t['dtype']} {t['shape']}, not {dtype} {list(shape)}")
        return t

    def read(self, t):
        fh = self.handles.get(t["path"])
        if fh is None:
            fh = self.handles[t["path"]] = open(t["path"], "rb")
        fh.seek(t["offset"])
        data = fh.read(t["nbytes"])
        if len(data) != t["nbytes"]:
            raise ValueError("source truncated: " + t["name"])
        return data

    def close(self):
        for fh in self.handles.values():
            fh.close()
        self.handles = {}


# ------------------------------------------------------------------ plan


def _ggml(type_name, ne):
    return {"family": "ggml", "type": type_name, "ne": list(ne)}


def _plain(dtype, shape):
    return {"family": "plain", "dtype": dtype, "shape": list(shape)}


def plan(layout, cfg, src, shard_target=None):
    """The artifact's groups, members and expert arrays, with every source
    tensor's shape and type checked. Members carry `produce(src)`, which
    returns their bytes."""
    groups, arrays = [], []
    h, e = cfg.hidden, cfg.experts
    order = tiled_order(cfg.lin_k_heads, cfg.lin_v_heads // cfg.lin_k_heads)
    used = set()

    def get(name, dtype, shape):
        used.add(PREFIX + name if not name.startswith(("lm_head", "mtp.", "model.")) else name)
        return src.get(PREFIX + name if not name.startswith(("lm_head", "mtp.", "model.")) else name,
                       dtype, shape)

    def member(name, rep, produce, access=None):
        m = dict(name=name, repr=rep, produce=produce, roles=[name])
        m["nbytes"] = layout.repr_bytes(rep, name)
        m["readable"] = layout.readable_for(rep, m["nbytes"])
        if access:
            m["access"] = access
        return m

    def verbatim(t):
        return lambda s: s.read(t)

    def bf16_matrix(name, hf, rows, cols, access=None):
        """BF16 [rows, cols] as GGML BF16 ne [cols, rows]."""
        t = get(hf, "BF16", [rows, cols])
        return member(name, _ggml("BF16", [cols, rows]), verbatim(t), access)

    def f32_vector(name, hf, n, fn=bf16_to_f32, shape=None):
        t = get(hf, "BF16", shape or [n])
        return member(name, _ggml("F32", [n]), lambda s: fn(s.read(t)))

    def mxfp8(name, hf, rows, cols, fix=None):
        """MXFP8 [rows, cols]: E4M3 weight and E8M0 scale, with an optional
        (weight, scale) -> (weight, scale) reorder."""
        if cols % 32:
            raise ValueError(f"{hf}: MXFP8 input dimension {cols} not a multiple of 32")
        w = get(hf + ".weight", "F8_E4M3", [rows, cols])
        s = get(hf + ".weight_scale", "U8", [rows, cols // 32])
        cache = {}

        def pair(src_):
            if "v" not in cache:
                wb, sb = src_.read(w), src_.read(s)
                check_e8m0_scales(sb, s["name"])
                cache["v"] = fix(wb, sb) if fix else (wb, sb)
            return cache["v"]

        def weight(src_):
            return pair(src_)[0]

        def scale(src_):
            v = pair(src_)[1]
            cache.clear()
            return v

        return [member(name + ".weight", _plain("F8_E4M3", [rows, cols]), weight),
                member(name + ".weight_scale", _plain("U8", [rows, cols // 32]), scale)]

    def group(kind, layer=None, expert=None):
        g = dict(kind=kind, layer=layer, expert=expert, members=[])
        groups.append(g)
        return g

    # Token table.
    group("table")["members"].append(
        bf16_matrix("token_embd.weight", "embed_tokens.weight", cfg.vocab, h, access="rows"))

    # The n-gram table: its shards' rows in order, each row's codes then scales.
    row_codes, row_scales = cfg.ple_row // 2, cfg.ple_row // 16
    pl = f"layers.{cfg.ple_layer}.ple.ple_embedding."
    shards, rows = [], 0
    for i in range(cfg.ple_parts):
        name = f"{pl}ngram_embedding.shard_{i}.weight"
        t = src.tensors.get(PREFIX + name)
        if t is None or t["dtype"] != "U8" or len(t["shape"]) != 2 or t["shape"][1] != row_codes:
            raise ValueError(f"{PREFIX + name}: missing or not U8 [rows, {row_codes}]")
        n = t["shape"][0]
        w = get(name, "U8", [n, row_codes])
        sc = get(f"{pl}ngram_embedding.shard_{i}.weight_scale", "F8_E4M3", [n, row_scales])
        shards.append((w, sc, n))
        rows += n

    def ple_table(src_):
        for w, sc, n in shards:
            scales = src_.read(sc)
            check_e4m3_scales(scales, sc["name"])
            yield interleave_rows(src_.read(w), row_codes, scales, row_scales, n)

    group("table")["members"].append(
        member("per_layer_token_embd.weight", _plain("U8", [rows, row_codes + row_scales]), ple_table,
               access="rows"))

    for il in range(cfg.layers):
        lp = f"layers.{il}."
        g = group("layer", il)
        mem = g["members"]
        for kind, hf in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
            mem.append(f32_vector(f"blk.{il}.hc_{kind}_norm.weight", f"{lp}{hf}.hc_norm.weight", cfg.hc_dim,
                                  norm_plus_one))
            mem.append(bf16_matrix(f"blk.{il}.hc_{kind}_down.weight", f"{lp}{hf}.input_mix_weight_down.weight",
                                   cfg.hc_rank, cfg.hc_dim))
            mem.append(bf16_matrix(f"blk.{il}.hc_{kind}_up.weight", f"{lp}{hf}.input_mix_weight_up.weight",
                                   cfg.hc_dim, cfg.hc_rank))
            mem.append(bf16_matrix(f"blk.{il}.hc_{kind}_inject.weight",
                                   f"{lp}{hf}.block_inject_weight.weight", cfg.hc, cfg.hc_dim))
        if cfg.linear[il]:
            la = lp + "linear_attn."
            kdim = cfg.lin_k_heads * cfg.lin_k_dim
            vdim = cfg.lin_v_heads * cfg.lin_v_dim
            vh = cfg.lin_v_dim

            def fix_rows(head_rows, first=0):
                def fix(w, s):
                    return (reorder_rows(w, h, order, head_rows, first),
                            reorder_rows(s, h // 32, order, head_rows, first))
                return fix

            def fix_out(w, s):
                return (reorder_cols(w, h, vdim, order, vh),
                        reorder_cols(s, h, vdim // 32, order, vh // 32))

            mem += mxfp8(f"blk.{il}.attn_qkv", la + "in_proj_qkv", cfg.conv_dim, h, fix_rows(vh, 2 * kdim))
            mem += mxfp8(f"blk.{il}.attn_gate", la + "in_proj_z", vdim, h, fix_rows(vh))
            mem += mxfp8(f"blk.{il}.ssm_beta", la + "in_proj_b", cfg.lin_v_heads, h, fix_rows(1))
            mem += mxfp8(f"blk.{il}.ssm_alpha", la + "in_proj_a", cfg.lin_v_heads, h, fix_rows(1))

            def per_head(fn):
                return lambda d: reorder_rows(fn(d), 4, order, 1)

            mem.append(f32_vector(f"blk.{il}.ssm_dt.bias", la + "dt_bias", cfg.lin_v_heads, per_head(bf16_to_f32)))
            mem.append(f32_vector(f"blk.{il}.ssm_a", la + "A_log", cfg.lin_v_heads, per_head(neg_exp)))
            conv = get(la + "conv1d.weight", "BF16", [cfg.conv_dim, 1, cfg.conv])
            mem.append(member(f"blk.{il}.ssm_conv1d.weight", _ggml("F32", [cfg.conv, cfg.conv_dim]),
                              lambda s, t=conv: reorder_rows(bf16_to_f32(s.read(t)), 4 * cfg.conv, order, vh,
                                                             2 * kdim)))
            mem.append(f32_vector(f"blk.{il}.ssm_norm.weight", la + "norm.weight", vh))
            mem += mxfp8(f"blk.{il}.ssm_out", la + "out_proj", h, vdim, fix_out)
        else:
            sa = lp + "self_attn."
            mem += mxfp8(f"blk.{il}.attn_q", sa + "q_proj", 2 * cfg.heads * cfg.head_dim, h)
            mem += mxfp8(f"blk.{il}.attn_k", sa + "k_proj", cfg.kv_heads * cfg.head_dim, h)
            mem += mxfp8(f"blk.{il}.attn_v", sa + "v_proj", cfg.kv_heads * cfg.head_dim, h)
            mem += mxfp8(f"blk.{il}.attn_output", sa + "o_proj", h, cfg.heads * cfg.head_dim)
            mem.append(f32_vector(f"blk.{il}.attn_q_norm.weight", sa + "q_norm.weight", cfg.head_dim,
                                  norm_plus_one))
            mem.append(f32_vector(f"blk.{il}.attn_k_norm.weight", sa + "k_norm.weight", cfg.head_dim,
                                  norm_plus_one))
            mem += mxfp8(f"blk.{il}.indexer.qk_proj", sa + "indexer.index_qk_proj",
                         (cfg.idx_heads + 1) * cfg.idx_dim, h)
            mem.append(f32_vector(f"blk.{il}.indexer.q_norm.weight", sa + "indexer.q_layernorm.weight",
                                  cfg.idx_dim, norm_plus_one))
            mem.append(f32_vector(f"blk.{il}.indexer.k_norm.weight", sa + "indexer.k_layernorm.weight",
                                  cfg.idx_dim, norm_plus_one))
        if il == cfg.ple_layer:
            pp = lp + "ple."
            mem.append(bf16_matrix(f"blk.{il}.ple_key.weight", pp + "key_proj.weight", cfg.hc_dim, cfg.ple_dim))
            mem.append(bf16_matrix(f"blk.{il}.ple_value.weight", pp + "value_proj.weight", h, cfg.ple_dim))
            for part in ("key", "query", "conv"):
                mem.append(f32_vector(f"blk.{il}.ple_norm_{part}.weight", f"{pp}norm_{part}.weight", cfg.hc_dim,
                                      norm_plus_one))
            pconv = get(pp + "conv1d.weight", "BF16", [cfg.hc_dim, 1, cfg.ple_conv])
            mem.append(member(f"blk.{il}.ple_conv1d.weight", _ggml("F32", [cfg.ple_conv, cfg.hc_dim]),
                              lambda s, t=pconv: bf16_to_f32(s.read(t))))
            for name, hf, n in (("ple_multipliers", "layer_multipliers", cfg.ngram),
                                ("ple_head_offsets", "ngram_heads_offsets", cfg.ple_heads),
                                ("ple_head_vocab", "ngram_heads_vocab_sizes", cfg.ple_heads)):
                t = get(pl + hf, "I64", [n])
                mem.append(member(f"blk.{il}.{name}", _plain("I64", [n]), verbatim(t)))
            s2 = get(pl + "ngram_embedding.weight_scale_2", "F32", [1])
            mem.append(member("per_layer_token_embd.weight_scale_2", _plain("F32", [1]),
                              lambda s, t=s2: check_global_scales(s.read(t), t["name"])))
        # MoE: router, shared expert and its gate, the routed experts' scales.
        mp = lp + "mlp."
        mem.append(bf16_matrix(f"blk.{il}.ffn_gate_inp.weight", mp + "gate.weight", e, h))
        gate_shexp = get(mp + "shared_expert_gate.weight", "BF16", [1, h])
        mem.append(member(f"blk.{il}.ffn_gate_inp_shexp.weight", _ggml("BF16", [h]), verbatim(gate_shexp)))
        mem += mxfp8(f"blk.{il}.ffn_gate_shexp", mp + "shared_expert.gate_proj", cfg.shared_ff, h)
        mem += mxfp8(f"blk.{il}.ffn_up_shexp", mp + "shared_expert.up_proj", cfg.shared_ff, h)
        mem += mxfp8(f"blk.{il}.ffn_down_shexp", mp + "shared_expert.down_proj", h, cfg.shared_ff)
        shapes = {"gate": (cfg.moe_ff, h), "up": (cfg.moe_ff, h), "down": (h, cfg.moe_ff)}
        experts = {}
        for proj, (n_out, k_in) in shapes.items():
            ex = []
            for x in range(e):
                base = f"{mp}experts.{x}.{proj}_proj."
                ex.append((get(base + "weight", "U8", [n_out, k_in // 2]),
                           get(base + "weight_scale", "F8_E4M3", [n_out, k_in // 16]),
                           get(base + "weight_scale_2", "F32", [])))
            experts[proj] = ex
            mem.append(member(f"blk.{il}.ffn_{proj}_exps.weight_scale_2", _ggml("F32", [e]),
                              lambda s, ex=ex: b"".join(check_global_scales(s.read(t2), t2["name"])
                                                        for _, _, t2 in ex)))
        # The routed experts in CUTLASS's layout: per expert group, four byte
        # arrays packed back to back (each a multiple of 256 bytes at
        # Qwen3.8's shapes, so at the offsets moe_layout.h ExpertLayout gives).
        first = len(groups)
        egroups = [group("expert", il, x) for x in range(e)]
        ff = cfg.moe_ff
        parts = ((f"blk.{il}.ffn_gate_up_exps.codes", [h // 2, 2 * ff]),
                 (f"blk.{il}.ffn_gate_up_exps.scales", [SF_ATOM_BYTES, sf1xx_atoms(2 * ff, h)]),
                 (f"blk.{il}.ffn_down_exps.codes", [ff // 2, h]),
                 (f"blk.{il}.ffn_down_exps.scales", [SF_ATOM_BYTES, sf1xx_atoms(h, ff)]))
        for x in range(e):
            reads = tuple(((w["path"], w["offset"], w["nbytes"]), (sc["path"], sc["offset"], sc["nbytes"]))
                          for w, sc, _ in (experts[p][x] for p in ("gate", "up", "down")))
            # The same repack as reads a worker process can make.
            egroups[x]["job"] = (reads, ff, h)
        for i, (name, ne) in enumerate(parts):
            rep = _ggml("I8", ne)
            arr = dict(name=name, layer=il, count=e, repr=rep, slice_bytes=layout.repr_bytes(rep, name),
                       members=[])
            arr["readable"] = layout.readable_for(rep, arr["slice_bytes"])
            for x in range(e):
                pairs = [experts[p][x][:2] for p in ("gate", "up", "down")]
                m = member(f"{name}#{x}", rep,
                           lambda s, pairs=pairs, i=i: expert_to_sf1xx(
                               *[(s.read(w), s.read(sc)) for w, sc in pairs], ff, h)[i])
                m["roles"] = []
                m["array"] = len(arrays)
                egroups[x]["members"].append(m)
                arr["members"].append(m)
            arr["first_group"] = first
            arrays.append(arr)

    head = group("head")
    head["members"].append(bf16_matrix("output.weight", "lm_head.weight", cfg.vocab, h))
    head["members"].append(f32_vector("output_hc_norm.weight", "hyper_connection_mixer.hc_norm.weight",
                                      cfg.hc_dim, norm_plus_one))
    head["members"].append(bf16_matrix("output_hc_down.weight", "hyper_connection_mixer.input_mix_weight_down.weight",
                                       cfg.hc_rank, cfg.hc_dim))
    head["members"].append(bf16_matrix("output_hc_up.weight", "hyper_connection_mixer.input_mix_weight_up.weight",
                                       cfg.hc_dim, cfg.hc_rank))

    # Every text-model tensor is either imported or deliberately left out: the
    # vision tower, the MTP block, and the routed experts' static activation
    # scales (input_scale), which GGML's products do not read (they quantize
    # each activation row with its own scale).
    skipped = [n for n in src.tensors if n not in used and not n.startswith(("model.visual.", "mtp."))
               and not (n.startswith(PREFIX + "layers.") and n.endswith(".input_scale"))]
    if skipped:
        raise ValueError(f"{len(skipped)} checkpoint tensors have no place in the plan, e.g. {skipped[:3]}")
    return _place(layout, groups, arrays, shard_target or layout.SHARD_TARGET)


MTP_ARCH = "qwen4exp-mtp"


def check_mtp_config(cfg, doc):
    """The MTP block config.json describes: one hybrid full-attention layer
    that reads the target's hidden streams and shares its embeddings."""
    t = doc.get("text_config", doc)
    mtp = t.get("mtp")
    if (t.get("mtp_num_hidden_layers") != 1 or t.get("mtp_use_dedicated_embeddings") is not False
            or type(mtp) is not dict or mtp.get("num_hidden_layers") != 1
            or mtp.get("layer_types") != ["full_attention"] or mtp.get("hybrid") is not True
            or mtp.get("mtp_use_hidden_state_from_layer") is not None):
        raise ValueError("config.json: not one hybrid full-attention MTP layer over the last hidden "
                         "streams with shared embeddings")
    rope = t.get("rope_parameters", {})
    if type(rope) is not dict or mtp.get("rope_theta") != rope.get("rope_theta"):
        raise ValueError("config.json: the MTP layer's rope_theta differs from the model's")


def draft_vocabulary(data, vocab):
    """Ascending original token IDs: order also preserves argmax tie breaking."""
    if len(data) > 12 * vocab or not data:
        raise ValueError("draft vocabulary is empty or too large")
    lines = data.splitlines()
    if len(lines) > vocab or any(not line or not line.isdigit() for line in lines):
        raise ValueError("draft vocabulary must contain one decimal token ID per line")
    ids = [int(line) for line in lines]
    if any(i >= vocab or i > 0x7fffffff for i in ids):
        raise ValueError("draft vocabulary token ID is outside the vocabulary")
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ValueError("draft vocabulary token IDs must be strictly ascending")
    return ids


def selected_rows(src, tensor, ids, width):
    """Bounded reads of consecutive selected BF16 rows, without a full-head copy."""
    row_bytes = 2 * width
    at = 0
    while at < len(ids):
        end = at + 1
        while (end < len(ids) and ids[end] == ids[end - 1] + 1 and
               (end - at) * row_bytes < 4 << 20):
            end += 1
        yield src.read(dict(tensor, offset=tensor["offset"] + ids[at] * row_bytes,
                            nbytes=(end - at) * row_bytes))
        at = end


def plan_mtp(layout, cfg, src, shard_target=None, draft_ids=None):
    """The MTP drafter artifact's groups, members and expert arrays: its
    layer (group `layer`, layer 0), its 512 expert groups and its head (the
    final mixer), every source tensor's shape and type checked and every
    mtp.* tensor placed."""
    groups, arrays = [], []
    h, e, hd = cfg.hidden, cfg.experts, cfg.head_dim
    used = set()

    def get(name, dtype, shape):
        used.add(name)
        return src.get(name, dtype, shape)

    def member(name, rep, produce):
        m = dict(name=name, repr=rep, produce=produce, roles=[name])
        m["nbytes"] = layout.repr_bytes(rep, name)
        m["readable"] = layout.readable_for(rep, m["nbytes"])
        return m

    def bf16_matrix(name, hf, rows, cols):
        t = get(hf, "BF16", [rows, cols])
        return member(name, _ggml("BF16", [cols, rows]), lambda s, t=t: s.read(t))

    def f32_vector(name, hf, n, fn=bf16_to_f32):
        t = get(hf, "BF16", [n])
        return member(name, _ggml("F32", [n]), lambda s, t=t: fn(s.read(t)))

    def group(kind, layer=None, expert=None):
        g = dict(kind=kind, layer=layer, expert=expert, members=[])
        groups.append(g)
        return g

    mem = group("layer", 0)["members"]
    mem.append(bf16_matrix("fc_embd.weight", "mtp.fc_embedding.weight", h, h))
    mem.append(bf16_matrix("fc_hidden.weight", "mtp.fc_hidden.weight", h, h))
    mem.append(f32_vector("norm_embd.weight", "mtp.pre_fc_norm_embedding.weight", h, norm_plus_one))
    mem.append(f32_vector("norm_hidden.weight", "mtp.pre_fc_norm_hidden.weight", cfg.hc_dim,
                          norm_plus_one))
    lp = "mtp.layers.0."
    for kind, hf in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
        mem.append(f32_vector(f"blk.0.hc_{kind}_norm.weight", f"{lp}{hf}.hc_norm.weight", cfg.hc_dim,
                              norm_plus_one))
        mem.append(bf16_matrix(f"blk.0.hc_{kind}_down.weight", f"{lp}{hf}.input_mix_weight_down.weight",
                               cfg.hc_rank, cfg.hc_dim))
        mem.append(bf16_matrix(f"blk.0.hc_{kind}_up.weight", f"{lp}{hf}.input_mix_weight_up.weight",
                               cfg.hc_dim, cfg.hc_rank))
        mem.append(bf16_matrix(f"blk.0.hc_{kind}_inject.weight", f"{lp}{hf}.block_inject_weight.weight",
                               cfg.hc, cfg.hc_dim))
    sa = lp + "self_attn."
    mem.append(bf16_matrix("blk.0.attn_q.weight", sa + "q_proj.weight", 2 * cfg.heads * hd, h))
    mem.append(bf16_matrix("blk.0.attn_k.weight", sa + "k_proj.weight", cfg.kv_heads * hd, h))
    mem.append(bf16_matrix("blk.0.attn_v.weight", sa + "v_proj.weight", cfg.kv_heads * hd, h))
    mem.append(bf16_matrix("blk.0.attn_output.weight", sa + "o_proj.weight", h, cfg.heads * hd))
    mem.append(f32_vector("blk.0.attn_q_norm.weight", sa + "q_norm.weight", hd, norm_plus_one))
    mem.append(f32_vector("blk.0.attn_k_norm.weight", sa + "k_norm.weight", hd, norm_plus_one))
    mem.append(bf16_matrix("blk.0.indexer.qk_proj.weight", sa + "indexer.index_qk_proj.weight",
                           (cfg.idx_heads + 1) * cfg.idx_dim, h))
    mem.append(f32_vector("blk.0.indexer.q_norm.weight", sa + "indexer.q_layernorm.weight", cfg.idx_dim,
                          norm_plus_one))
    mem.append(f32_vector("blk.0.indexer.k_norm.weight", sa + "indexer.k_layernorm.weight", cfg.idx_dim,
                          norm_plus_one))
    mp = lp + "mlp."
    mem.append(bf16_matrix("blk.0.ffn_gate_inp.weight", mp + "gate.weight", e, h))
    gate_shexp = get(mp + "shared_expert_gate.weight", "BF16", [1, h])
    mem.append(member("blk.0.ffn_gate_inp_shexp.weight", _ggml("BF16", [h]),
                      lambda s, t=gate_shexp: s.read(t)))
    mem.append(bf16_matrix("blk.0.ffn_gate_shexp.weight", mp + "shared_expert.gate_proj.weight",
                           cfg.shared_ff, h))
    mem.append(bf16_matrix("blk.0.ffn_up_shexp.weight", mp + "shared_expert.up_proj.weight",
                           cfg.shared_ff, h))
    mem.append(bf16_matrix("blk.0.ffn_down_shexp.weight", mp + "shared_expert.down_proj.weight", h,
                           cfg.shared_ff))
    shapes = {"gate": (cfg.moe_ff, h), "up": (cfg.moe_ff, h), "down": (h, cfg.moe_ff)}
    experts = {}
    for proj, (n_out, k_in) in shapes.items():
        ex = []
        for x in range(e):
            base = f"{mp}experts.{x}.{proj}_proj."
            ex.append((get(base + "weight", "U8", [n_out, k_in // 2]),
                       get(base + "weight_scale", "F8_E4M3", [n_out, k_in // 16]),
                       get(base + "weight_scale_2", "F32", [])))
        experts[proj] = ex
        mem.append(member(f"blk.0.ffn_{proj}_exps.weight_scale_2", _ggml("F32", [e]),
                          lambda s, ex=ex: b"".join(check_global_scales(s.read(t2), t2["name"])
                                                    for _, _, t2 in ex)))
    # The routed experts in the target's CUTLASS layout (plan's, the same
    # repack in worker processes).
    first = len(groups)
    egroups = [group("expert", 0, x) for x in range(e)]
    ff = cfg.moe_ff
    parts = (("blk.0.ffn_gate_up_exps.codes", [h // 2, 2 * ff]),
             ("blk.0.ffn_gate_up_exps.scales", [SF_ATOM_BYTES, sf1xx_atoms(2 * ff, h)]),
             ("blk.0.ffn_down_exps.codes", [ff // 2, h]),
             ("blk.0.ffn_down_exps.scales", [SF_ATOM_BYTES, sf1xx_atoms(h, ff)]))
    for x in range(e):
        reads = tuple(((w["path"], w["offset"], w["nbytes"]), (sc["path"], sc["offset"], sc["nbytes"]))
                      for w, sc, _ in (experts[p][x] for p in ("gate", "up", "down")))
        egroups[x]["job"] = (reads, ff, h)
    for i, (name, ne) in enumerate(parts):
        rep = _ggml("I8", ne)
        arr = dict(name=name, layer=0, count=e, repr=rep, slice_bytes=layout.repr_bytes(rep, name),
                   members=[])
        arr["readable"] = layout.readable_for(rep, arr["slice_bytes"])
        for x in range(e):
            pairs = [experts[p][x][:2] for p in ("gate", "up", "down")]
            m = member(f"{name}#{x}", rep,
                       lambda s, pairs=pairs, i=i: expert_to_sf1xx(
                           *[(s.read(w), s.read(sc)) for w, sc in pairs], ff, h)[i])
            m["roles"] = []
            m["array"] = len(arrays)
            egroups[x]["members"].append(m)
            arr["members"].append(m)
        arr["first_group"] = first
        arrays.append(arr)

    head = group("head")
    head["members"].append(f32_vector("output_hc_norm.weight", "mtp.hyper_connection_mixer.hc_norm.weight",
                                      cfg.hc_dim, norm_plus_one))
    head["members"].append(bf16_matrix("output_hc_down.weight",
                                       "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
                                       cfg.hc_rank, cfg.hc_dim))
    head["members"].append(bf16_matrix("output_hc_up.weight",
                                       "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
                                       cfg.hc_dim, cfg.hc_rank))
    if draft_ids is not None:
        ids = draft_vocabulary(draft_ids, cfg.vocab)
        tensor = get("lm_head.weight", "BF16", [cfg.vocab, h])
        head["members"].append(member("draft_output.weight", _ggml("BF16", [h, len(ids)]),
                                      lambda s: selected_rows(s, tensor, ids, h)))
        packed_ids = struct.pack(f"<{len(ids)}i", *ids)
        head["members"].append(member("draft_output.ids", _ggml("I32", [1, len(ids)]),
                                      lambda s: packed_ids))
    # Every MTP tensor has a place; the given sources hold nothing else the
    # drafter should have read.
    skipped = [n for n in src.tensors if n.startswith("mtp.") and n not in used]
    if skipped:
        raise ValueError(f"{len(skipped)} MTP tensors have no place in the plan, e.g. {skipped[:3]}")
    p = _place(layout, groups, arrays, shard_target or layout.SHARD_TARGET)
    p["arch"] = MTP_ARCH
    return p


def _place(layout, groups, arrays, shard_target):
    """layout.plan's placement: members at 256-byte offsets (readable bytes
    reserved), groups 4 KiB-stored, shards filled to the target, chunks
    numbered in order."""
    chunk, shards = 0, [dict(stored=0, groups=[])]
    for gid, g in enumerate(groups):
        off = 0
        for m in g["members"]:
            off = layout.align(off, layout.MEMBER_ALIGN)
            m["group"], m["offset"] = gid, off
            off += m["readable"]
        g["used"], g["stored"] = off, max(layout.FILE_ALIGN, layout.align(off, layout.FILE_ALIGN))
        g["chunks"] = -(-g["stored"] // layout.CHUNK)
        shard = shards[-1]
        if shard["stored"] and shard["stored"] + g["stored"] > shard_target:
            shards.append(dict(stored=0, groups=[]))
            shard = shards[-1]
        g["id"], g["shard"], g["file_offset"], g["first_chunk"] = gid, len(shards) - 1, shard["stored"], chunk
        shard["groups"].append(gid)
        shard["stored"] += g["stored"]
        chunk += g["chunks"]
    for arr in arrays:
        offsets = {m["offset"] for m in arr["members"]}
        sizes = {groups[m["group"]]["stored"] for m in arr["members"]}
        if len(offsets) != 1 or len(sizes) != 1:
            raise ValueError("expert array not uniform: " + arr["name"])
        arr["group_offset"] = offsets.pop()
    return dict(arch=ARCH, groups=groups, shards=shards, expert_arrays=arrays, chunks=chunk, ties={})


# ------------------------------------------------------------------ writing


class _GroupSink:
    """A group's bytes into its shard, each 2 MiB chunk hashed as it passes."""

    def __init__(self, layout, f, whole, g, hashes):
        self.layout, self.f, self.whole, self.g, self.hashes = layout, f, whole, g, hashes
        self.pos, self.cur = 0, hashlib.sha256()

    def write(self, data):
        chunk = self.layout.CHUNK
        mv = memoryview(data)
        while len(mv):
            piece = mv[:chunk - self.pos % chunk]
            self.cur.update(piece)
            self.whole.update(piece)
            self.f.write(piece)
            self.pos += len(piece)
            mv = mv[len(piece):]
            if self.pos % chunk == 0:
                self.hashes[self.g["first_chunk"] + self.pos // chunk - 1] = self.cur.hexdigest()
                self.cur = hashlib.sha256()

    def zeros(self, n):
        while n > 0:
            k = min(n, 4 << 20)
            self.write(bytes(k))
            n -= k

    def close(self):
        if self.pos != self.g["stored"]:
            raise ValueError("group written short or long")
        if self.pos % self.layout.CHUNK:
            self.hashes[self.g["first_chunk"] + self.pos // self.layout.CHUNK] = self.cur.hexdigest()


def _produce(m, src):
    out = m["produce"](src)
    if isinstance(out, (bytes, bytearray)):
        yield out
    else:
        yield from out


def _expert_group_bytes(job):
    """Worker: one expert group's members' bytes, from its gate, up and down
    (path, offset, nbytes) reads of codes and scales, repacked (run in a
    process pool; returns a list of bytes)."""
    reads, ffn, width = job

    def read(path, offset, n):
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(n)
        if len(data) != n:
            raise ValueError("source truncated")
        return data

    pairs = [(read(*w), read(*s)) for w, s in reads]
    try:
        return expert_to_sf1xx(*pairs, ffn, width)
    except ValueError as e:
        raise ValueError(f"{reads[0][1][0]} at {reads[0][1][1]}: {e}") from None


def write(layout, p, src, work, converter, sources, metas, workers=8):
    """Writes the shards, index and manifest into `work`, returns the manifest
    bytes. `sources` is {name: (bytes, sha256)} of every identity file."""
    (work / "data").mkdir(parents=True)
    (work / "meta").mkdir()
    hashes, shard_meta, files = [None] * p["chunks"], [], []
    # Forked workers inherit this module (import_m3.py loads it by path).
    pool = (concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                                                   mp_context=multiprocessing.get_context("fork"))
            if workers > 1 else None)
    try:
        for s, shard in enumerate(p["shards"]):
            raw, data_offset = layout.st_header(p, s)
            rel = f"data/{s:05d}.safetensors"
            whole = hashlib.sha256()
            with open(work / rel, "wb", buffering=16 << 20) as f:
                head = struct.pack("<Q", len(raw)) + raw
                f.write(head)
                whole.update(head)
                gids = shard["groups"]
                pending, ahead = {}, 0

                def submit_upto(limit):
                    # Expert groups are repacked in worker processes, a bounded
                    # window ahead of the writer.
                    nonlocal ahead
                    while pool is not None and ahead < len(gids) and len(pending) < limit:
                        g_ = p["groups"][gids[ahead]]
                        if g_["kind"] == "expert":
                            pending[gids[ahead]] = pool.submit(
                                _expert_group_bytes, g_["job"])
                        ahead += 1

                for gid in gids:
                    submit_upto(4 * workers)
                    g = p["groups"][gid]
                    sink = _GroupSink(layout, f, whole, g, hashes)
                    ready = pending.pop(gid).result() if gid in pending else None
                    for i, m in enumerate(g["members"]):
                        sink.zeros(m["offset"] - sink.pos)
                        n = 0
                        for piece in ([ready[i]] if ready is not None else _produce(m, src)):
                            sink.write(piece)
                            n += len(piece)
                        if n != m["nbytes"]:
                            raise ValueError(f"{m['name']}: produced {n} bytes, expected {m['nbytes']}")
                    sink.zeros(g["stored"] - sink.pos)
                    sink.close()
                f.flush()
                os.fsync(f.fileno())
            shard_meta.append({"path": rel, "data_offset": data_offset, "data_bytes": shard["stored"],
                               "header_sha256": hashlib.sha256(head).hexdigest()})
            files.append({"path": rel, "role": "shard", "bytes": data_offset + shard["stored"],
                          "sha256": whole.hexdigest()})
            print(json.dumps({"shard": s, "of": len(p["shards"]), "bytes": data_offset + shard["stored"]}),
                  flush=True)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    for rel, path in metas:
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != sources[Path(path).name][1]:
            raise ValueError(f"{path}: metadata changed during import")
        layout._fsync_write(work / rel, data)
        files.append({"path": rel, "role": "source-metadata", "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    index = layout.dumps(layout.index_doc(p, hashes, shard_meta))
    layout._fsync_write(work / "index.json", index)
    files.append({"path": "index.json", "role": "index", "bytes": len(index),
                  "sha256": hashlib.sha256(index).hexdigest()})
    families = sorted({m["repr"]["family"] for g in p["groups"] for m in g["members"]})
    transformations = [{"kind": "expert-slice", "tensor": a["name"], "count": a["count"]}
                       for a in p["expert_arrays"]]
    manifest = {
        "format": layout.FORMAT, "format_version": layout.FORMAT_VERSION, "experimental": True,
        "layout": layout.LAYOUT,
        "model": {"architecture": p["arch"], "expert_count": len(p["expert_arrays"][0]["members"])
                  if p["expert_arrays"] else 0, "representation": families},
        "source": sorted(({"name": n, "bytes": b, "sha256": d} for n, (b, d) in sources.items()),
                         key=lambda x: x["name"]),
        "transformations": sorted(transformations, key=layout._canon_key),
        "converter": converter,
        "files": sorted(files, key=lambda x: x["path"]),
    }
    mbytes = layout.dumps(manifest)
    layout._fsync_write(work / "manifest.json", mbytes)
    return mbytes


def _hash_files(paths, extra=()):
    """{name: (bytes, sha256)} of whole files, and SHA-256 of (path, offset,
    n) ranges, hashed from the same reads; files in parallel threads (hashlib
    releases the GIL on large updates)."""
    wanted = {}
    for path, off, n in extra:
        wanted.setdefault(os.path.realpath(path), []).append((off, n))

    def one(path):
        ranges = sorted(wanted.get(os.path.realpath(path), []))
        hashers = [hashlib.sha256() for _ in ranges]
        whole, pos = hashlib.sha256(), 0
        with open(path, "rb") as fh:
            while block := fh.read(16 << 20):
                whole.update(block)
                end = pos + len(block)
                for (a, n), hs in zip(ranges, hashers):
                    if a < end and a + n > pos:
                        hs.update(block[max(a, pos) - pos:min(a + n, end) - pos])
                pos = end
        return path, pos, whole.hexdigest(), [((path, a, n), hs.hexdigest()) for (a, n), hs in zip(ranges, hashers)]

    sources, ranges = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for path, size, digest, rs in ex.map(one, paths):
            sources[Path(path).name] = (size, digest)
            for key, d in rs:
                ranges[(os.path.realpath(key[0]), key[1], key[2])] = d
    return sources, ranges


def build(layout, out_root, shard_paths, expected=None, converter=None, shard_target=None, workers=8,
          mtp=False, draft_vocab_ids=None):
    """Plans, writes, verifies and publishes the artifact; returns its path.
    With `mtp`, the MTP drafter's artifact (plan_mtp) from the shards given,
    which must hold every mtp.* tensor.

    `expected` maps each identity file's name (the shards and config.json) to
    its pinned SHA-256; the files are hashed before planning is trusted and
    again after writing, so a source that changes during the import aborts it."""
    out_root = Path(out_root)
    paths = [str(p) for p in shard_paths]
    names = [Path(p).name for p in paths]
    if len(set(names)) != len(names) or not all(layout.NAME.fullmatch(n) for n in names):
        raise ValueError(f"source file names must be unique and safe: {names}")
    config_path = Path(paths[0]).parent / "config.json"
    if any(Path(p).parent != config_path.parent for p in paths):
        raise ValueError("the shards must share one directory, with its config.json")
    config_bytes = config_path.read_bytes()
    doc = json.loads(config_bytes, object_pairs_hook=_unique_keys)
    cfg = Config(doc)
    ids_path, ids_bytes = None, None
    if draft_vocab_ids is not None:
        if not mtp:
            raise ValueError("a draft vocabulary needs an MTP drafter import")
        ids_path = Path(draft_vocab_ids)
        if not layout.NAME.fullmatch(ids_path.name) or ids_path.name in names + [config_path.name]:
            raise ValueError("draft vocabulary file name is unsafe or duplicates a source")
        # A bounded read, then the same bytes are planned, hashed and kept.
        with ids_path.open("rb") as f:
            ids_bytes = f.read(12 * cfg.vocab + 1)
        draft_vocabulary(ids_bytes, cfg.vocab)
    if mtp:
        check_mtp_config(cfg, doc)
    src = Sources(layout, paths)
    try:
        p = (plan_mtp(layout, cfg, src, shard_target, ids_bytes) if mtp else
             plan(layout, cfg, src, shard_target))
        identity = paths + [str(config_path)]
        metas = [("meta/config.json", config_path)]
        if ids_path is not None:
            identity.append(str(ids_path))
            metas.append((f"meta/{ids_path.name}", ids_path))
            if expected is not None:
                expected = dict(expected)
                expected.setdefault(ids_path.name, hashlib.sha256(ids_bytes).hexdigest())
        headers = [(part["path"], 0, part["header_len"]) for part in src.parts]
        sources, ranges = _hash_files(identity, headers)
        for part in src.parts:
            if ranges[(os.path.realpath(part["path"]), 0, part["header_len"])] != part["header_sha256"]:
                raise ValueError(f"{part['path']}: header changed after planning")
        if sources[config_path.name][1] != hashlib.sha256(config_bytes).hexdigest():
            raise ValueError("config.json changed after planning")
        if ids_path is not None and sources[ids_path.name][1] != hashlib.sha256(ids_bytes).hexdigest():
            raise ValueError("draft vocabulary changed after planning")
        if expected is not None and {n: d for n, (_, d) in sources.items()} != expected:
            raise ValueError("sources differ from their recorded identities")
        converter = converter or {"name": "artifact-layout/modelopt_qwen38.py", "version": "test"}
        staging = layout._staging_dir(out_root)
        job = layout.job_name(sorted([n, d] for n, (_, d) in sources.items()), [], converter)
        lock = layout._try_lock(staging / f"{job}.lock")
        if lock is None:
            raise layout.ArtifactError("busy", f"import {job} is already running")
        try:
            work = staging / job
            if work.is_symlink():
                work.unlink()
            elif work.exists():
                shutil.rmtree(work)
            mbytes = write(layout, p, src, work, converter, sources, metas,
                           workers=workers)
            if _hash_files(identity)[0] != sources:
                raise ValueError("a source changed during import")
            artifact_id = hashlib.sha256(mbytes).hexdigest()
            layout.verify(work, deep=True, expected_id=artifact_id)
            for d in (work / "data", work / "meta", work):
                layout._fsync_dir(d)
            final = out_root / artifact_id
            if final.exists() or final.is_symlink():
                layout.verify(final)
                shutil.rmtree(work)
            else:
                os.rename(work, final)
            layout._fsync_dir(staging)
            layout._fsync_dir(out_root)
            return final, p
        finally:
            os.close(lock)
    finally:
        src.close()
