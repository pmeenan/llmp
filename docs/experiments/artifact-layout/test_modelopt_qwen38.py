# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for modelopt_qwen38.py (through import_m3.py). Stdlib only:

  python3 -m unittest test_modelopt_qwen38
"""
import array
import hashlib
import json
import random
import struct
import tempfile
import unittest
from pathlib import Path

import import_m3
import test_layout

LAYOUT = import_m3.load_layout()
MO, _ = import_m3.load_modelopt()

# A tiny model of the same structure: 4 layers (3 linear, 1 full), the
# n-gram layer second, every block size the formats need.
CONFIG = {
    "model_type": "qwen3_8_flash_next",
    "text_config": {
        "hidden_size": 64, "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 32,
        "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
        "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 32, "hc_count": 2, "hc_lowrank": 16,
        "indexer_n_heads": 2, "indexer_head_dim": 32, "indexer_kv_heads": 1,
        "ngram_size": 3, "heads_per_ngram": 2, "split_ngram_parts": 2, "ple_embed_dim": 64,
        "ple_conv_kernel_size": 4, "ple_layer_ids": [2], "vocab_size": 16,
        "tie_word_embeddings": False, "output_gate_type": "sigmoid",
    },
}


def bf16(values):
    return b"".join(struct.pack("<f", v)[2:] for v in values)


def rnd(n, seed):
    r = random.Random(seed)
    return bytes(r.randrange(256) for _ in range(n))


def e4m3_scales(n, seed):
    """Random non-negative finite E4M3 scale bytes (0x00-0x7E)."""
    return bytes(b % 0x7F for b in rnd(n, seed))


def small_bf16(n, seed):
    r = random.Random(seed)
    return bf16([r.choice([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0]) for _ in range(n)])


def checkpoint(cfg):
    """{name: (dtype, shape, bytes)} of a synthetic checkpoint."""
    t = cfg["text_config"]
    h, e, hc = t["hidden_size"], t["num_experts"], t["hc_count"] * t["hidden_size"]
    kdim = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    vdim = t["linear_num_value_heads"] * t["linear_value_head_dim"]
    conv_dim = 2 * kdim + vdim
    P = "model.language_model."
    out, seed = {}, [0]

    def put(name, dtype, shape, data=None):
        seed[0] += 1
        n = 1
        for s in shape:
            n *= s
        size = n * {"BF16": 2, "F32": 4, "I64": 8, "U8": 1, "F8_E4M3": 1}[dtype]
        if data is None:
            data = small_bf16(n, seed[0]) if dtype == "BF16" else (
                struct.pack(f"<{n}f", *[0.5] * n) if dtype == "F32" else rnd(size, seed[0]))
            if name.endswith(".weight_scale"):  # valid scales: E4M3 below 0x7F, E8M0 below 0xFF
                data = e4m3_scales(size, seed[0]) if dtype == "F8_E4M3" else bytes(b % 0xFF for b in data)
        assert len(data) == size, name
        out[name] = (dtype, shape, data)

    def mx(name, rows, cols):
        put(name + ".weight", "F8_E4M3", [rows, cols])
        put(name + ".weight_scale", "U8", [rows, cols // 32])

    put(P + "embed_tokens.weight", "BF16", [t["vocab_size"], h])
    put("lm_head.weight", "BF16", [t["vocab_size"], h])
    for part in ("hc_norm.weight",):
        put(P + "hyper_connection_mixer." + part, "BF16", [hc])
    put(P + "hyper_connection_mixer.input_mix_weight_down.weight", "BF16", [t["hc_lowrank"], hc])
    put(P + "hyper_connection_mixer.input_mix_weight_up.weight", "BF16", [hc, t["hc_lowrank"]])
    ple_heads = (t["ngram_size"] - 1) * t["heads_per_ngram"]
    row = t["ple_embed_dim"] // ple_heads
    pl = P + f"layers.{t['ple_layer_ids'][0] - 1}.ple."
    for i in range(t["split_ngram_parts"]):
        put(f"{pl}ple_embedding.ngram_embedding.shard_{i}.weight", "U8", [5 + i, row // 2])
        put(f"{pl}ple_embedding.ngram_embedding.shard_{i}.weight_scale", "F8_E4M3", [5 + i, row // 16])
    put(f"{pl}ple_embedding.ngram_embedding.weight_scale_2", "F32", [1])
    put(f"{pl}ple_embedding.layer_multipliers", "I64", [t["ngram_size"]])
    put(f"{pl}ple_embedding.ngram_heads_offsets", "I64", [ple_heads])
    put(f"{pl}ple_embedding.ngram_heads_vocab_sizes", "I64", [ple_heads])
    put(pl + "key_proj.weight", "BF16", [hc, t["ple_embed_dim"]])
    put(pl + "value_proj.weight", "BF16", [h, t["ple_embed_dim"]])
    for part in ("key", "query", "conv"):
        put(f"{pl}norm_{part}.weight", "BF16", [hc])
    put(pl + "conv1d.weight", "BF16", [hc, 1, t["ple_conv_kernel_size"]])
    for il, kind in enumerate(t["layer_types"]):
        lp = P + f"layers.{il}."
        for hf in ("attn_hyper_connection", "mlp_hyper_connection"):
            put(f"{lp}{hf}.hc_norm.weight", "BF16", [hc])
            put(f"{lp}{hf}.input_mix_weight_down.weight", "BF16", [t["hc_lowrank"], hc])
            put(f"{lp}{hf}.input_mix_weight_up.weight", "BF16", [hc, t["hc_lowrank"]])
            put(f"{lp}{hf}.block_inject_weight.weight", "BF16", [t["hc_count"], hc])
        if kind == "linear_attention":
            la = lp + "linear_attn."
            mx(la + "in_proj_qkv", conv_dim, h)
            mx(la + "in_proj_z", vdim, h)
            mx(la + "in_proj_a", t["linear_num_value_heads"], h)
            mx(la + "in_proj_b", t["linear_num_value_heads"], h)
            put(la + "A_log", "BF16", [t["linear_num_value_heads"]])
            put(la + "dt_bias", "BF16", [t["linear_num_value_heads"]])
            put(la + "conv1d.weight", "BF16", [conv_dim, 1, t["linear_conv_kernel_dim"]])
            put(la + "norm.weight", "BF16", [t["linear_value_head_dim"]])
            mx(la + "out_proj", h, vdim)
        else:
            sa = lp + "self_attn."
            hd = t["head_dim"]
            mx(sa + "q_proj", 2 * t["num_attention_heads"] * hd, h)
            mx(sa + "k_proj", t["num_key_value_heads"] * hd, h)
            mx(sa + "v_proj", t["num_key_value_heads"] * hd, h)
            mx(sa + "o_proj", h, t["num_attention_heads"] * hd)
            put(sa + "q_norm.weight", "BF16", [hd])
            put(sa + "k_norm.weight", "BF16", [hd])
            mx(sa + "indexer.index_qk_proj", (t["indexer_n_heads"] + 1) * t["indexer_head_dim"], h)
            put(sa + "indexer.q_layernorm.weight", "BF16", [t["indexer_head_dim"]])
            put(sa + "indexer.k_layernorm.weight", "BF16", [t["indexer_head_dim"]])
        mp = lp + "mlp."
        put(mp + "gate.weight", "BF16", [e, h])
        put(mp + "shared_expert_gate.weight", "BF16", [1, h])
        for proj, (n_out, k_in) in {"gate": (32, h), "up": (32, h), "down": (h, 32)}.items():
            mx(f"{mp}shared_expert.{proj}_proj", n_out, k_in)
        ff = t["moe_intermediate_size"]
        for x in range(e):
            for proj, (n_out, k_in) in {"gate": (ff, h), "up": (ff, h), "down": (h, ff)}.items():
                base = f"{mp}experts.{x}.{proj}_proj."
                put(base + "weight", "U8", [n_out, k_in // 2])
                put(base + "weight_scale", "F8_E4M3", [n_out, k_in // 16])
                put(base + "weight_scale_2", "F32", [], struct.pack("<f", 0.25 * (x + 1)))
                put(base + "input_scale", "F32", [], struct.pack("<f", 1.0))
    # Left out on purpose: a vision tensor and an MTP tensor.
    put("model.visual.pos_embed.weight", "BF16", [2, 4])
    put("mtp.fc_hidden.weight", "BF16", [4, 4])
    return out


def write_checkpoint(directory, tensors, config=CONFIG, split=2):
    names = sorted(tensors)
    paths = []
    for s in range(split):
        part = names[s::split]
        data = test_layout.st_bytes([(n, tensors[n][0], tensors[n][1], tensors[n][2]) for n in part])
        path = directory / f"model-{s + 1:05d}-of-{split:05d}.safetensors"
        path.write_bytes(data)
        paths.append(path)
    (directory / "config.json").write_text(json.dumps(config))
    return paths


class RepackTests(unittest.TestCase):
    def test_nvfp4_repack_matches_the_element_by_element_definition(self):
        for rows, k, seed in ((1, 64, 1), (3, 128, 2), (5, 640, 3)):
            codes, scales = rnd(rows * k // 2, seed), e4m3_scales(rows * k // 16, seed + 100)
            self.assertEqual(MO.nvfp4_to_ggml(codes, scales, rows, k),
                             MO.nvfp4_to_ggml_reference(codes, scales, rows, k))

    def test_nvfp4_repack_places_elements_where_ggml_reads_them(self):
        # Element i of a row is code i: GGML's dequantizer reads sub-block s,
        # byte j as elements s*16 + j (low nibble) and s*16 + j + 8 (high).
        k = 64
        codes = bytes((2 * i % 16) | ((2 * i + 1) % 16) << 4 for i in range(k // 2))
        scales = bytes(range(10, 14))
        out = MO.nvfp4_to_ggml(codes, scales, 1, k)
        self.assertEqual(out[:4], scales)
        for s in range(4):
            for j in range(8):
                byte = out[4 + s * 8 + j]
                self.assertEqual(byte & 15, (s * 16 + j) % 16)
                self.assertEqual(byte >> 4, (s * 16 + j + 8) % 16)

    def test_nvfp4_repack_dequantizes_to_the_same_values(self):
        # ModelOpt: E2M1 code times its E4M3 block scale. GGML (ggml-quants.c
        # dequantize_row_nvfp4): kvalues_fp4 (doubled E2M1) times
        # ggml_ue4m3_to_fp32, which is half the E4M3 value. Equal up to the
        # sign of zero (E2M1 code 8 is -0, kvalues_fp4[8] is +0).
        e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        e2m1 += [-x for x in e2m1]
        kvalues_fp4 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]

        def e4m3(b):
            e, m = b >> 3 & 15, b & 7
            return m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)

        def ue4m3_half(b):  # ggml-impl.h ggml_ue4m3_to_fp32
            return 0.0 if b in (0, 0x7F) else e4m3(b & 0x7F) * 0.5

        rows, k = 3, 128
        codes, scales = rnd(rows * k // 2, 7), e4m3_scales(rows * k // 16, 8)
        out = MO.nvfp4_to_ggml(codes, scales, rows, k)
        for r in range(rows):
            want = [e2m1[(codes[r * k // 2 + i // 2] >> 4 * (i % 2)) & 15] * e4m3(scales[r * k // 16 + i // 16])
                    for i in range(k)]
            got = [0.0] * k
            for b in range(k // 64):
                blk = out[(r * k // 64 + b) * 36:(r * k // 64 + b + 1) * 36]
                for s in range(4):
                    d = ue4m3_half(blk[s])
                    for j in range(8):
                        q = blk[4 + s * 8 + j]
                        got[b * 64 + s * 16 + j] = kvalues_fp4[q & 15] * d
                        got[b * 64 + s * 16 + j + 8] = kvalues_fp4[q >> 4] * d
            self.assertEqual(got, want)

    def test_sf1xx_offsets_are_a_permutation(self):
        for rows, k in ((128, 64), (256, 256), (1280, 2560)):
            blocks = k // 16
            seen = {MO.sf1xx_offset(r, b, blocks) for r in range(rows) for b in range(blocks)}
            self.assertEqual(seen, set(range(rows * blocks)), msg=(rows, k))

    def test_sf1xx_layout_is_the_harness_conversion_of_ggml_blocks(self):
        # What the importer writes from ModelOpt's tensors equals what the
        # Qwen3.8 harness wrote at load from GGML's blocks (jitllm_moe.cu
        # ConvertBlocks), for gate/up rows (2f) and down rows (w) that are
        # whole 128-row atoms, and for down rows that are not (padded).
        for ffn, width, seed in ((64, 128, 1), (128, 256, 2), (64, 64, 3)):
            def proj(rows, k, s):
                return rnd(rows * k // 2, s), e4m3_scales(rows * k // 16, s + 50)
            gate, up, down = proj(ffn, width, seed), proj(ffn, width, seed + 1), proj(width, ffn, seed + 2)
            got = MO.expert_to_sf1xx(gate, up, down, ffn, width)
            want = MO.ggml_to_sf1xx_reference(MO.nvfp4_to_ggml(*gate, ffn, width),
                                              MO.nvfp4_to_ggml(*up, ffn, width),
                                              MO.nvfp4_to_ggml(*down, width, ffn), ffn, width)
            self.assertEqual(got, want, msg=(ffn, width))
            self.assertEqual(got[0], gate[0] + up[0])  # codes verbatim
            self.assertEqual(got[2], down[0])
            self.assertEqual([len(x) for x in got],
                             [ffn * width, MO.sf1xx_atoms(2 * ffn, width) * 512, ffn * width // 2,
                              MO.sf1xx_atoms(width, ffn) * 512])
        with self.assertRaisesRegex(ValueError, "E4M3 scales negative or NaN"):
            MO.expert_scales_sf1xx(bytes([0x38, 0x38, 0x80, 0x38]), 1, 64)
        with self.assertRaisesRegex(ValueError, "disagree"):
            MO.expert_scales_sf1xx(bytes(8), 1, 64)

    def test_scales_outside_what_every_reader_agrees_on_are_refused(self):
        codes = rnd(64 // 2, 1)
        for good in (0x00, 0x01, 0x7E):
            MO.nvfp4_to_ggml(codes, bytes([good] * 4), 1, 64)
        for bad in (0x7F, 0x80, 0xC3, 0xFF):
            with self.assertRaisesRegex(ValueError, "E4M3 scales negative or NaN", msg=hex(bad)):
                MO.nvfp4_to_ggml(codes, bytes([0x38, 0x38, bad, 0x38]), 1, 64)
            with self.assertRaisesRegex(ValueError, "negative or NaN"):
                MO.check_e4m3_scales(bytes([bad]), "t")
        MO.check_e8m0_scales(bytes(range(0xFF)), "t")
        with self.assertRaisesRegex(ValueError, "E8M0"):
            MO.check_e8m0_scales(bytes([127, 0xFF]), "t")
        MO.check_global_scales(struct.pack("<2f", 1e-30, 3.0), "t")
        for bad in (0.0, -0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "global scale", msg=repr(bad)):
                MO.check_global_scales(struct.pack("<2f", 1.0, bad), "t")

    def test_tiled_order_matches_llama_cpp(self):
        # Grouped [G0v0, G0v1, G0v2, G1v0, G1v1, G1v2] -> tiled [G0v0, G1v0, G0v1, ...].
        self.assertEqual(MO.tiled_order(2, 3), [0, 3, 1, 4, 2, 5])
        data = bytes(range(12))
        self.assertEqual(MO.reorder_rows(data, 2, [1, 0], 2, first_row=2),
                         bytes([0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7]))
        self.assertEqual(MO.reorder_cols(bytes(range(8)), 2, 4, [1, 0], 2), bytes([2, 3, 0, 1, 6, 7, 4, 5]))
        # Head counts that do not cover the tensor exactly are refused.
        with self.assertRaisesRegex(ValueError, "out of range"):
            MO.reorder_rows(bytes(range(12)), 2, [1, 0, 2], 2, first_row=2)
        with self.assertRaisesRegex(ValueError, "out of range"):
            MO.reorder_cols(bytes(range(8)), 2, 4, [1, 0, 2], 2)
        with self.assertRaisesRegex(ValueError, "out of range"):
            MO.reorder_cols(bytes(range(8)), 2, 4, [1, 0], 1)
        with self.assertRaisesRegex(ValueError, "out of range"):  # not a permutation
            MO.reorder_rows(bytes(range(12)), 2, [1, 1], 2, first_row=2)
        with self.assertRaisesRegex(ValueError, "out of range"):
            MO.reorder_cols(bytes(range(8)), 2, 4, [1, 1], 2)
        # A tail after the reordered heads (none in the plan) stays in place.
        self.assertEqual(MO.reorder_rows(bytes(range(6)), 1, [1, 0], 2), bytes([2, 3, 0, 1, 4, 5]))

    def test_norms_and_a(self):
        self.assertEqual(array.array("f", MO.norm_plus_one(bf16([0.5, -1.0]))).tolist(), [1.5, 0.0])
        a = array.array("f", MO.neg_exp(bf16([0.0, 1.0]))).tolist()
        self.assertEqual(a[0], -1.0)
        self.assertAlmostEqual(a[1], -2.718281828, places=6)

    def test_rows_interleave(self):
        self.assertEqual(MO.interleave_rows(b"abcdef", 3, b"XY", 1, 2), b"abcXdefY")


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.tensors = checkpoint(CONFIG)
        self.paths = write_checkpoint(self.dir, self.tensors)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, workers=1, **kw):
        final, p = MO.build(LAYOUT, self.dir / "store", self.paths, shard_target=1 << 16, workers=workers, **kw)
        return final, p

    def test_the_artifact_verifies_and_holds_the_repacked_bytes(self):
        final, _ = self.build()
        manifest, index = LAYOUT.verify(final)
        self.assertEqual(manifest["model"], {"architecture": "qwen4exp", "expert_count": 4,
                                             "representation": ["ggml", "plain"]})
        self.assertEqual({s["name"] for s in manifest["source"]},
                         {p.name for p in self.paths} | {"config.json"})
        self.assertEqual(len(index["expert_arrays"]), 16)
        res = {r["name"]: r for r in index["resources"]}
        groups = index["groups"]

        def stored(r_or_group, offset, n):
            g = groups[r_or_group]
            shard = index["shards"][g["shard"]]
            with open(final / shard["path"], "rb") as f:
                f.seek(shard["data_offset"] + g["offset"] + offset)
                return f.read(n)

        def resource(name):
            r = res[name]
            return stored(r["group"], r["offset"], r["bytes"])

        P = "model.language_model."
        # Experts: the CUTLASS layout's four arrays packed from each expert
        # group's start.
        arrays = [a for a in index["expert_arrays"] if a["layer"] == 2]
        self.assertEqual([(a["name"], a["repr"], a["group_offset"]) for a in arrays], [
            ("blk.2.ffn_gate_up_exps.codes", {"family": "ggml", "type": "I8", "ne": [32, 128]}, 0),
            ("blk.2.ffn_gate_up_exps.scales", {"family": "ggml", "type": "I8", "ne": [512, 1]}, 4096),
            ("blk.2.ffn_down_exps.codes", {"family": "ggml", "type": "I8", "ne": [32, 64]}, 4608),
            ("blk.2.ffn_down_exps.scales", {"family": "ggml", "type": "I8", "ne": [512, 1]}, 6656)])
        g = next(i for i, g in enumerate(groups) if g["kind"] == "expert" and g["layer"] == 2 and g["expert"] == 3)
        pairs = [(self.tensors[P + f"layers.2.mlp.experts.3.{p}_proj.weight"][2],
                  self.tensors[P + f"layers.2.mlp.experts.3.{p}_proj.weight_scale"][2]) for p in ("gate", "up", "down")]
        self.assertEqual([stored(g, a["group_offset"], a["slice_bytes"]) for a in arrays],
                         MO.expert_to_sf1xx(*pairs, 64, 64))
        self.assertEqual(array.array("f", resource("blk.2.ffn_down_exps.weight_scale_2")).tolist(),
                         [0.25, 0.5, 0.75, 1.0])
        # MXFP8 verbatim where nothing reorders; value heads tiled where it does.
        self.assertEqual(resource("blk.3.attn_q.weight"), self.tensors[P + "layers.3.self_attn.q_proj.weight"][2])
        z = self.tensors[P + "layers.0.linear_attn.in_proj_z.weight"][2]
        order = MO.tiled_order(2, 2)  # [0, 2, 1, 3]
        self.assertEqual(order, [0, 2, 1, 3])
        head = 32 * 64
        self.assertEqual(resource("blk.0.attn_gate.weight"), b"".join(z[h * head:(h + 1) * head] for h in order))
        qkv = self.tensors[P + "layers.0.linear_attn.in_proj_qkv.weight"][2]
        kq = 2 * 64 * 64
        self.assertEqual(resource("blk.0.attn_qkv.weight")[:kq], qkv[:kq])
        self.assertEqual(resource("blk.0.attn_qkv.weight")[kq:], b"".join(qkv[kq + h * head:kq + (h + 1) * head]
                                                                             for h in order))
        # Norms fold (1 + w), linear attention's gated norm does not.
        hn = array.array("f", resource("blk.1.hc_attn_norm.weight"))
        src = array.array("f", MO.bf16_to_f32(self.tensors[P + "layers.1.attn_hyper_connection.hc_norm.weight"][2]))
        self.assertEqual(hn.tolist(), [x + 1 for x in src])
        self.assertEqual(resource("blk.0.ssm_norm.weight"),
                         MO.bf16_to_f32(self.tensors[P + "layers.0.linear_attn.norm.weight"][2]))
        # The n-gram table: shard rows in order, codes then scales per row.
        table = resource("per_layer_token_embd.weight")
        self.assertEqual(res["per_layer_token_embd.weight"]["repr"],
                         {"family": "plain", "dtype": "U8", "shape": [11, 9]})
        pl = P + "layers.1.ple.ple_embedding.ngram_embedding."
        want = b"".join(MO.interleave_rows(self.tensors[f"{pl}shard_{i}.weight"][2], 8,
                                           self.tensors[f"{pl}shard_{i}.weight_scale"][2], 1, 5 + i)
                        for i in range(2))
        self.assertEqual(table, want)
        # BF16 matrices keep their bytes, with GGML's dimension order.
        self.assertEqual(res["token_embd.weight"]["repr"], {"family": "ggml", "type": "BF16", "ne": [64, 16]})
        self.assertEqual(res["token_embd.weight"]["access"], "rows")
        self.assertEqual(resource("output.weight"), self.tensors["lm_head.weight"][2])
        self.assertNotIn("mtp.fc_hidden.weight", res)

    def test_worker_processes_write_the_same_artifact(self):
        a, _ = self.build(workers=1)
        b, _ = MO.build(LAYOUT, self.dir / "store2", self.paths, shard_target=1 << 16, workers=3)
        self.assertEqual(a.name, b.name)

    def test_unplaced_tensors_and_bad_shapes_are_refused(self):
        extra = dict(self.tensors)
        extra["model.language_model.layers.0.linear_attn.surprise"] = ("BF16", [2], bf16([1.0, 2.0]))
        for p in self.paths:
            p.unlink()
        self.paths = write_checkpoint(self.dir, extra)
        with self.assertRaisesRegex(ValueError, "no place in the plan"):
            self.build()
        bad = dict(self.tensors)
        bad["model.language_model.layers.3.self_attn.k_norm.weight"] = ("BF16", [16], bf16([0.0] * 16))
        for p in self.paths:
            p.unlink()
        self.paths = write_checkpoint(self.dir, bad)
        with self.assertRaisesRegex(ValueError, "k_norm"):
            self.build()

    def test_config_is_validated(self):
        for key, value in (("hidden_size", 96), ("ple_layer_ids", [4]), ("linear_num_value_heads", 5),
                           ("tie_word_embeddings", True), ("num_experts", 0),
                           ("hidden_size", 0), ("hidden_size", True), ("hidden_size", 64.0), ("hidden_size", "64"),
                           ("hidden_size", 1 << 30), ("num_hidden_layers", 5), ("num_hidden_layers", 1 << 20),
                           ("layer_types", ["linear_attention"] * 3 + ["sliding_attention"]),
                           ("num_experts_per_tok", 5), ("moe_intermediate_size", 96),
                           ("shared_expert_intermediate_size", 48), ("linear_value_head_dim", 48),
                           ("ple_embed_dim", 72), ("ple_layer_ids", [0]), ("ple_layer_ids", [2, 3]),
                           ("hc_count", 1), ("split_ngram_parts", 0), ("ngram_size", 1),
                           ("output_gate_type", "swish"), ("vocab_size", None)):
            cfg = json.loads(json.dumps(CONFIG))
            cfg["text_config"][key] = value
            with self.assertRaises(ValueError, msg=f"{key}={value!r}"):
                MO.Config(cfg)
        for doc in ([], {"text_config": []}):
            with self.assertRaises(ValueError):
                MO.Config(doc)

    # ------------------------------------------------------------ adversarial checkpoints

    P = "model.language_model."

    def rewrite(self, tensors=None, config=CONFIG):
        for p in self.paths:
            p.unlink()
        self.paths = write_checkpoint(self.dir, self.tensors if tensors is None else tensors, config)

    def refused(self, pattern, tensors=None, config=CONFIG, workers=1):
        self.rewrite(tensors, config)
        with self.assertRaisesRegex(ValueError, pattern):
            self.build(workers=workers)

    def with_bytes(self, name, patch):
        """The checkpoint with `name`'s bytes changed by patch(bytearray)."""
        t = dict(self.tensors)
        dtype, shape, data = t[name]
        data = bytearray(data)
        patch(data)
        t[name] = (dtype, shape, bytes(data))
        return t

    def test_bad_nvfp4_scales_are_refused(self):
        e = self.P + "layers.1.mlp.experts.2.up_proj.weight_scale"
        for bad in (0x7F, 0x80, 0xC3, 0xFF):
            def patch(d, bad=bad):
                d[5] = bad
            for workers in (1, 2):
                with self.subTest(bad=hex(bad), workers=workers):
                    self.refused("E4M3 scales negative or NaN", self.with_bytes(e, patch), workers=workers)

    def test_bad_ngram_scales_are_refused(self):
        s = self.P + "layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight_scale"
        for bad in (0x7F, 0x80, 0xFF):
            def patch(d, bad=bad):
                d[-1] = bad
            self.refused("shard_1.weight_scale.*E4M3", self.with_bytes(s, patch))

    def test_nan_mxfp8_scales_are_refused(self):
        for name in ("layers.0.linear_attn.out_proj.weight_scale", "layers.3.self_attn.q_proj.weight_scale",
                     "layers.2.mlp.shared_expert.down_proj.weight_scale"):
            def patch(d):
                d[len(d) // 2] = 0xFF
            self.refused(name + ".*E8M0", self.with_bytes(self.P + name, patch))

    def test_bad_global_scales_are_refused(self):
        for name in ("layers.0.mlp.experts.3.down_proj.weight_scale_2",
                     "layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2"):
            for bad in (0.0, -0.0, -0.25, float("nan"), float("inf")):
                def patch(d, bad=bad):
                    d[:] = struct.pack("<f", bad)
                with self.subTest(name=name, bad=bad):
                    self.refused("global scale", self.with_bytes(self.P + name, patch))

    def test_wrong_scale_shapes_and_types_are_refused(self):
        base = self.P + "layers.0.mlp.experts.1.gate_proj."
        cases = {
            # NVFP4 scales per 32 instead of per 16, and as U8.
            base + "weight_scale": [("F8_E4M3", [64, 2], bytes(128)), ("U8", [64, 4], bytes(256))],
            # A global scale of shape [1] instead of a scalar, and as BF16.
            base + "weight_scale_2": [("F32", [1], struct.pack("<f", 1.0)), ("BF16", [], bf16([1.0]))],
            # MXFP8 scales per 16 instead of per 32, and as F8_E4M3.
            self.P + "layers.3.self_attn.o_proj.weight_scale": [("U8", [64, 4], bytes(256)),
                                                                ("F8_E4M3", [64, 2], bytes(128))],
            # NVFP4 codes as F8_E4M3, and one row short.
            base + "weight": [("F8_E4M3", [64, 32], bytes(2048)), ("U8", [63, 32], bytes(2016))],
            # n-gram rows without their scale.
            self.P + "layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight_scale":
                [("F8_E4M3", [5, 0], b"")],
        }
        for name, variants in cases.items():
            for v in variants:
                with self.subTest(name=name, variant=v[:2]):
                    self.refused(name.rsplit(".", 2)[-2] + r"\.", {**self.tensors, name: v})

    def test_missing_extra_and_duplicate_tensors_are_refused(self):
        P = self.P
        t = dict(self.tensors)
        del t[P + "layers.2.mlp.experts.3.up_proj.weight"]
        self.refused("has no .*experts.3.up_proj.weight", t)
        t = dict(self.tensors)
        del t[P + "layers.2.mlp.experts.3.up_proj.weight_scale_2"]
        self.refused("has no", t)
        # A fifth expert in one layer, a third n-gram shard, a lost shard.
        t = dict(self.tensors)
        for suffix in ("weight", "weight_scale", "weight_scale_2"):
            t[P + "layers.1.mlp.experts.4.up_proj." + suffix] = t[P + "layers.1.mlp.experts.3.up_proj." + suffix]
        self.refused("no place in the plan", t)
        ng = P + "layers.1.ple.ple_embedding.ngram_embedding."
        t = dict(self.tensors)
        t[ng + "shard_2.weight"], t[ng + "shard_2.weight_scale"] = t[ng + "shard_1.weight"], t[ng + "shard_1.weight_scale"]
        self.refused("no place in the plan", t)
        t = dict(self.tensors)
        del t[ng + "shard_1.weight"]
        self.refused("shard_1.weight: missing", t)
        t = dict(self.tensors)
        del t[ng + "shard_1.weight_scale"]
        self.refused("has no .*shard_1.weight_scale", t)
        # The same name in two shards.
        self.rewrite()
        name = P + "layers.0.linear_attn.norm.weight"
        other = next(p for p in self.paths if name not in self.header(p))
        dtype, shape, data = self.tensors[name]
        self.mutate(other, lambda h, n: h.__setitem__(name, {"dtype": dtype, "shape": shape,
                                                             "data_offsets": [n, n + len(data)]}), extra=data)
        with self.assertRaisesRegex(ValueError, "duplicate tensor across shards"):
            self.build()

    def header(self, path):
        data = path.read_bytes()
        (n,) = struct.unpack("<Q", data[:8])
        return json.loads(data[8:8 + n])

    def mutate(self, path, fn, extra=b"", raw=None):
        """Rewrites a shard's header: fn(header, data_bytes) edits it, `extra`
        is appended to the data; or `raw` replaces the header text."""
        data = path.read_bytes()
        (n,) = struct.unpack("<Q", data[:8])
        h = json.loads(data[8:8 + n])
        body = data[8 + n:]
        if raw is None:
            fn(h, len(body))
            raw = json.dumps(h).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + body + extra)

    def test_malformed_safetensors_headers_are_refused(self):
        name = self.P + "layers.2.mlp.experts.0.down_proj.weight"

        def where():
            return next(p for p in self.paths if name in self.header(p))

        def offsets(fn):
            return lambda h, n: h[name].__setitem__("data_offsets", fn(h[name]["data_offsets"], n))

        cases = [
            ("inconsistent", offsets(lambda o, n: [n + 64 - 2048, n + 64])),     # past the file's end
            ("inconsistent", offsets(lambda o, n: [o[0], o[1] - 1])),            # shorter than dtype x shape
            ("inconsistent", offsets(lambda o, n: [o[1], o[0]])),                # reversed
            ("overlaps", offsets(lambda o, n: [o[0] + 16, o[1] + 16])),          # into its neighbour
            ("inconsistent", lambda h, n: h[name].__setitem__("shape", [64, 16])),  # shape vs bytes
            ("inconsistent", lambda h, n: h[name].__setitem__("dtype", "BF16")),    # dtype vs bytes
            ("malformed", lambda h, n: h[name].__setitem__("shape", [64, -32])),
            ("malformed", lambda h, n: h[name].__setitem__("dtype", "F4")),
            ("malformed", lambda h, n: h[name].__setitem__("data_offsets", [0])),
        ]
        for pattern, fn in cases:
            with self.subTest(pattern=pattern):
                self.rewrite()
                self.mutate(where(), fn)
                with self.assertRaisesRegex(ValueError, pattern):
                    self.build()
        # Neighbours in file order: shift an entry onto the one before it.
        self.rewrite()
        path = where()
        h = self.header(path)
        ordered = sorted(h, key=lambda k: h[k]["data_offsets"][0] if k != "__metadata__" else -1)
        i = ordered.index(name)
        prev = ordered[i - 1] if i else ordered[i + 1]
        self.mutate(path, lambda h, n: h[name].__setitem__(
            "data_offsets", [h[prev]["data_offsets"][0], h[prev]["data_offsets"][0] + 2048]))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.build()
        # A header length past the file, and a duplicated key in one header.
        self.rewrite()
        path = where()
        data = path.read_bytes()
        path.write_bytes(struct.pack("<Q", len(data)) + data[8:])
        with self.assertRaisesRegex(ValueError, "header size"):
            self.build()
        self.rewrite()
        path = where()
        text = json.dumps(self.header(path))
        dup = json.dumps({name: self.header(path)[name]})[1:-1]
        self.mutate(path, None, raw=(text[:-1] + ", " + dup + "}").encode())
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.build()

    def test_config_disagreeing_with_the_checkpoint_is_refused(self):
        for key, value in (("num_experts", 8), ("num_experts", 2), ("split_ngram_parts", 3),
                           ("split_ngram_parts", 1), ("linear_num_value_heads", 2), ("linear_num_key_heads", 1),
                           ("hidden_size", 128), ("moe_intermediate_size", 128), ("num_hidden_layers", 3),
                           ("heads_per_ngram", 1), ("vocab_size", 32), ("hc_count", 3)):
            cfg = json.loads(json.dumps(CONFIG))
            cfg["text_config"][key] = value
            if key == "num_hidden_layers":
                cfg["text_config"]["layer_types"] = cfg["text_config"]["layer_types"][:value]
            with self.subTest(key=key, value=value):
                self.refused("", config=cfg)
        # A config.json with a key twice (json.loads would keep the last).
        self.rewrite()
        text = json.dumps(CONFIG)
        (self.dir / "config.json").write_text(text.replace('"num_experts": 4', '"num_experts": 8, "num_experts": 4'))
        with self.assertRaisesRegex(ValueError, "duplicate JSON key 'num_experts'"):
            self.build()

    def test_expected_identities_are_enforced(self):
        wrong = {p.name: "0" * 64 for p in self.paths}
        wrong["config.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "recorded identities"):
            self.build(expected=wrong)

    def test_import_m3_takes_the_shards_and_their_config_from_the_pins(self):
        files = [{"path": f"Q/{p.name}", "bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in self.paths]
        cfg = self.dir / "config.json"
        files.append({"path": "Q/config.json", "bytes": cfg.stat().st_size,
                      "sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()})
        pins = self.dir / "pins.json"
        pins.write_text(json.dumps({"models": [{"id": "q", "files": files}]}))
        out = self.dir / "store3"
        import_m3.main(["import_m3.py", "build", str(out), str(pins), "q", *map(str, self.paths)])
        (artifact,) = [p for p in out.iterdir() if p.name != ".staging"]
        manifest, _ = LAYOUT.verify(artifact)
        self.assertTrue(manifest["converter"]["version"].startswith(import_m3.converter()["version"] + "+modelopt"))


# The MTP block of the tiny model: one full-attention layer, BF16 but for
# its NVFP4 experts, and its final mixer.
MTP_CONFIG = json.loads(json.dumps(CONFIG))
MTP_CONFIG["text_config"].update({
    "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
    "rope_parameters": {"rope_theta": 10000000},
    "mtp": {"hybrid": True, "layer_types": ["full_attention"], "mtp_use_hidden_state_from_layer": None,
            "num_hidden_layers": 1, "rope_theta": 10000000}})


def mtp_checkpoint(cfg):
    """{name: (dtype, shape, bytes)} of a synthetic MTP block."""
    t = cfg["text_config"]
    h, e, hc, hd = t["hidden_size"], t["num_experts"], t["hc_count"] * t["hidden_size"], t["head_dim"]
    out, seed = {}, [100]

    def put(name, dtype, shape, data=None):
        seed[0] += 1
        n = 1
        for s in shape:
            n *= s
        size = n * {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1}[dtype]
        if data is None:
            data = small_bf16(n, seed[0]) if dtype == "BF16" else (
                e4m3_scales(size, seed[0]) if dtype == "F8_E4M3" else rnd(size, seed[0]))
        out[name] = (dtype, shape, data)

    put("mtp.fc_embedding.weight", "BF16", [h, h])
    put("mtp.fc_hidden.weight", "BF16", [h, h])
    put("mtp.pre_fc_norm_embedding.weight", "BF16", [h])
    put("mtp.pre_fc_norm_hidden.weight", "BF16", [hc])
    for part in ("hc_norm.weight",):
        put("mtp.hyper_connection_mixer." + part, "BF16", [hc])
    put("mtp.hyper_connection_mixer.input_mix_weight_down.weight", "BF16", [t["hc_lowrank"], hc])
    put("mtp.hyper_connection_mixer.input_mix_weight_up.weight", "BF16", [hc, t["hc_lowrank"]])
    lp = "mtp.layers.0."
    for hf in ("attn_hyper_connection", "mlp_hyper_connection"):
        put(f"{lp}{hf}.hc_norm.weight", "BF16", [hc])
        put(f"{lp}{hf}.input_mix_weight_down.weight", "BF16", [t["hc_lowrank"], hc])
        put(f"{lp}{hf}.input_mix_weight_up.weight", "BF16", [hc, t["hc_lowrank"]])
        put(f"{lp}{hf}.block_inject_weight.weight", "BF16", [t["hc_count"], hc])
    sa = lp + "self_attn."
    put(sa + "q_proj.weight", "BF16", [2 * t["num_attention_heads"] * hd, h])
    put(sa + "k_proj.weight", "BF16", [t["num_key_value_heads"] * hd, h])
    put(sa + "v_proj.weight", "BF16", [t["num_key_value_heads"] * hd, h])
    put(sa + "o_proj.weight", "BF16", [h, t["num_attention_heads"] * hd])
    put(sa + "q_norm.weight", "BF16", [hd])
    put(sa + "k_norm.weight", "BF16", [hd])
    put(sa + "indexer.index_qk_proj.weight", "BF16", [(t["indexer_n_heads"] + 1) * t["indexer_head_dim"], h])
    put(sa + "indexer.q_layernorm.weight", "BF16", [t["indexer_head_dim"]])
    put(sa + "indexer.k_layernorm.weight", "BF16", [t["indexer_head_dim"]])
    mp = lp + "mlp."
    put(mp + "gate.weight", "BF16", [e, h])
    put(mp + "shared_expert_gate.weight", "BF16", [1, h])
    sff = t["shared_expert_intermediate_size"]
    for proj, (n_out, k_in) in {"gate": (sff, h), "up": (sff, h), "down": (h, sff)}.items():
        put(f"{mp}shared_expert.{proj}_proj.weight", "BF16", [n_out, k_in])
    ff = t["moe_intermediate_size"]
    for x in range(e):
        for proj, (n_out, k_in) in {"gate": (ff, h), "up": (ff, h), "down": (h, ff)}.items():
            base = f"{mp}experts.{x}.{proj}_proj."
            put(base + "weight", "U8", [n_out, k_in // 2])
            put(base + "weight_scale", "F8_E4M3", [n_out, k_in // 16])
            put(base + "weight_scale_2", "F32", [], struct.pack("<f", 0.5 * (x + 1)))
    return out


class MtpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.tensors = mtp_checkpoint(MTP_CONFIG)
        self.paths = write_checkpoint(self.dir, self.tensors, config=MTP_CONFIG, split=1)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **kw):
        return MO.build(LAYOUT, self.dir / "store", self.paths, shard_target=1 << 16, workers=1, mtp=True,
                        **kw)

    def test_the_drafter_verifies_and_holds_the_mtp_block(self):
        final, _ = self.build()
        manifest, index = LAYOUT.verify(final)
        self.assertEqual(manifest["model"], {"architecture": "qwen4exp-mtp", "expert_count": 4,
                                             "representation": ["ggml"]})
        res = {r["name"]: r for r in index["resources"]}
        # No token table or head of its own: the target's are bound at load.
        self.assertNotIn("token_embd.weight", res)
        self.assertNotIn("output.weight", res)
        groups = index["groups"]

        def resource(name):
            r = res[name]
            g = groups[r["group"]]
            shard = index["shards"][g["shard"]]
            with open(final / shard["path"], "rb") as f:
                f.seek(shard["data_offset"] + g["offset"] + r["offset"])
                return f.read(r["bytes"])

        # BF16 linears verbatim, in GGML's order; norms fold (1 + w).
        self.assertEqual(res["blk.0.attn_q.weight"]["repr"], {"family": "ggml", "type": "BF16", "ne": [64, 128]})
        self.assertEqual(resource("blk.0.attn_q.weight"), self.tensors["mtp.layers.0.self_attn.q_proj.weight"][2])
        self.assertEqual(resource("fc_hidden.weight"), self.tensors["mtp.fc_hidden.weight"][2])
        self.assertEqual(resource("blk.0.ffn_down_shexp.weight"),
                         self.tensors["mtp.layers.0.mlp.shared_expert.down_proj.weight"][2])
        norm = array.array("f", resource("norm_hidden.weight"))
        src = array.array("f", MO.bf16_to_f32(self.tensors["mtp.pre_fc_norm_hidden.weight"][2]))
        self.assertEqual(norm.tolist(), [x + 1 for x in src])
        self.assertEqual(res["output_hc_norm.weight"]["repr"], {"family": "ggml", "type": "F32", "ne": [128]})
        # The experts in the target's CUTLASS layout, their global scales gathered.
        arrays = index["expert_arrays"]
        self.assertEqual([a["name"] for a in arrays], ["blk.0.ffn_gate_up_exps.codes", "blk.0.ffn_gate_up_exps.scales",
                                                       "blk.0.ffn_down_exps.codes", "blk.0.ffn_down_exps.scales"])
        self.assertEqual(array.array("f", resource("blk.0.ffn_up_exps.weight_scale_2")).tolist(),
                         [0.5, 1.0, 1.5, 2.0])
        g = next(i for i, g in enumerate(groups) if g["kind"] == "expert" and g["expert"] == 2)
        shard = index["shards"][groups[g]["shard"]]
        pairs = [(self.tensors[f"mtp.layers.0.mlp.experts.2.{p}_proj.weight"][2],
                  self.tensors[f"mtp.layers.0.mlp.experts.2.{p}_proj.weight_scale"][2]) for p in ("gate", "up", "down")]
        with open(final / shard["path"], "rb") as f:
            got = []
            for a in arrays:
                f.seek(shard["data_offset"] + groups[g]["offset"] + a["group_offset"])
                got.append(f.read(a["slice_bytes"]))
        self.assertEqual(got, MO.expert_to_sf1xx(*pairs, 64, 64))

    def test_missing_and_unplaced_mtp_tensors_are_refused(self):
        for mutate, pattern in ((lambda t: t.pop("mtp.fc_embedding.weight"), "no mtp.fc_embedding"),
                                (lambda t: t.__setitem__("mtp.surprise", ("BF16", [2], bf16([1.0, 2.0]))),
                                 "no place in the plan"),
                                (lambda t: t.__setitem__("mtp.layers.0.self_attn.k_norm.weight",
                                                         ("BF16", [16], bf16([0.0] * 16))), "k_norm")):
            tensors = dict(self.tensors)
            mutate(tensors)
            for p in self.paths:
                p.unlink()
            self.paths = write_checkpoint(self.dir, tensors, config=MTP_CONFIG, split=1)
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                self.build()

    def test_the_mtp_config_is_validated(self):
        for key, value in (("mtp_num_hidden_layers", 2), ("mtp_use_dedicated_embeddings", True),
                           ("mtp", {"hybrid": True, "layer_types": ["linear_attention"],
                                    "mtp_use_hidden_state_from_layer": None, "num_hidden_layers": 1,
                                    "rope_theta": 10000000}),
                           ("mtp", {"hybrid": True, "layer_types": ["full_attention"],
                                    "mtp_use_hidden_state_from_layer": None, "num_hidden_layers": 1,
                                    "rope_theta": 5}),
                           ("mtp", None)):
            cfg = json.loads(json.dumps(MTP_CONFIG))
            cfg["text_config"][key] = value
            (self.dir / "config.json").write_text(json.dumps(cfg))
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "MTP"):
                self.build()

    def test_selected_head_rows_and_original_ids_are_packed_exactly(self):
        vocab = MTP_CONFIG["text_config"]["vocab_size"]
        width = MTP_CONFIG["text_config"]["hidden_size"]
        raw = small_bf16(vocab * width, 876)
        self.tensors["lm_head.weight"] = ("BF16", [vocab, width], raw)
        for p in self.paths:
            p.unlink()
        self.paths = write_checkpoint(self.dir, self.tensors, config=MTP_CONFIG, split=1)
        ids = [0, 2, 3, vocab - 1]
        selection = self.dir / "selection.txt"
        selection.write_text("".join(f"{i}\n" for i in ids))
        final, _ = self.build(draft_vocab_ids=selection)
        manifest, index = LAYOUT.verify(final)
        resources = {r["name"]: r for r in index["resources"]}
        def read(name):
            r = resources[name]
            g = index["groups"][r["group"]]
            shard = index["shards"][g["shard"]]
            with (final / shard["path"]).open("rb") as f:
                f.seek(shard["data_offset"] + g["offset"] + r["offset"])
                return f.read(r["bytes"])
        self.assertEqual(read("draft_output.weight"),
                         b"".join(raw[i * 2 * width:(i + 1) * 2 * width] for i in ids))
        self.assertEqual(read("draft_output.ids"), struct.pack("<4i", *ids))
        self.assertEqual((final / f"meta/{selection.name}").read_bytes(), selection.read_bytes())
        source = next(s for s in manifest["source"] if s["name"] == selection.name)
        self.assertEqual(source["sha256"], hashlib.sha256(selection.read_bytes()).hexdigest())
        self.assertEqual(resources["draft_output.ids"]["repr"]["ne"], [1, 4])

    def test_invalid_draft_vocabulary_is_refused_before_writing(self):
        for data in (b"", b"-1\n", b"2\n2\n", b"3\n2\n", b"9999999\n", b"1 2\n",
                     b"\xff\n", b"1\n\n"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                MO.draft_vocabulary(data, 16)
        self.assertEqual(MO.draft_vocabulary(b"0\n1\n15\n", 16), [0, 1, 15])

    def test_a_supplied_selection_pin_is_not_replaced_by_the_current_file(self):
        width = MTP_CONFIG["text_config"]["hidden_size"]
        vocab = MTP_CONFIG["text_config"]["vocab_size"]
        self.tensors["lm_head.weight"] = ("BF16", [vocab, width], small_bf16(vocab * width, 876))
        for p in self.paths:
            p.unlink()
        self.paths = write_checkpoint(self.dir, self.tensors, config=MTP_CONFIG, split=1)
        selection = self.dir / "selection.txt"
        selection.write_bytes(b"0\n2\n")
        sources = [*self.paths, self.dir / "config.json"]
        expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        expected[selection.name] = hashlib.sha256(b"0\n1\n").hexdigest()
        with self.assertRaisesRegex(ValueError, "sources differ"):
            self.build(draft_vocab_ids=selection, expected=expected)
        self.assertFalse((self.dir / "store").exists())

    def test_import_m3_imports_the_drafter_from_its_pins(self):
        files = [{"path": f"Q/{p.name}", "bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in self.paths]
        cfg = self.dir / "config.json"
        files.append({"path": "Q/config.json", "bytes": cfg.stat().st_size,
                      "sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()})
        pins = self.dir / "pins.json"
        pins.write_text(json.dumps({"models": [{"id": "q", "files": files}]}))
        out = self.dir / "store3"
        import_m3.main(["import_m3.py", "drafter", str(out), str(pins), "q", *map(str, self.paths)])
        (artifact,) = [p for p in out.iterdir() if p.name != ".staging"]
        manifest, _ = LAYOUT.verify(artifact)
        self.assertEqual(manifest["model"]["architecture"], "qwen4exp-mtp")
        self.assertTrue(manifest["converter"]["version"].endswith("+mtp"))

    def test_cli_selected_head_honors_the_list_pin_and_requires_one(self):
        width = MTP_CONFIG["text_config"]["hidden_size"]
        vocab = MTP_CONFIG["text_config"]["vocab_size"]
        self.tensors["lm_head.weight"] = ("BF16", [vocab, width], small_bf16(vocab * width, 876))
        for p in self.paths:
            p.unlink()
        self.paths = write_checkpoint(self.dir, self.tensors, config=MTP_CONFIG, split=1)
        selection = self.dir / "selection.txt"
        selection.write_bytes(b"0\n1\n")
        identities = [*self.paths, self.dir / "config.json", selection]
        files = [{"path": f"Q/{p.name}", "bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in identities]
        pins = self.dir / "pins.json"
        out = self.dir / "cli-selected"
        args = ["import_m3.py", "drafter", str(out), str(pins), "q", *map(str, self.paths),
                "--draft-vocab-ids", str(selection)]
        pins.write_text(json.dumps({"models": [{"id": "q", "files": files}]}))
        selection.write_bytes(b"0\n2\n")  # valid, same length, different content
        with self.assertRaisesRegex(ValueError, "sources differ"):
            import_m3.main(args)
        self.assertFalse(out.exists())
        selection.write_bytes(b"0\n1\n")
        pins.write_text(json.dumps({"models": [{"id": "q", "files": files[:-1]}]}))
        with self.assertRaises(SystemExit):
            import_m3.main(args)
        self.assertFalse(out.exists())
        pins.write_text(json.dumps({"models": [{"id": "q", "files": files}]}))
        import_m3.main(args)
        (artifact,) = [p for p in out.iterdir() if p.name != ".staging"]
        manifest, index = LAYOUT.verify(artifact)
        self.assertTrue(any(r["name"] == "draft_output.ids" for r in index["resources"]))
        self.assertEqual(next(s["sha256"] for s in manifest["source"]
                              if s["name"] == selection.name), files[-1]["sha256"])


if __name__ == "__main__":
    unittest.main()
