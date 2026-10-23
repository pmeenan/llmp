// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The native Qwen3.8 MTP drafter import (`llmp import --component mtp`),
// over a descriptor-only checkpoint of the pinned closed domain: its draft
// head list as untrusted input, the drafter's normalization with and
// without a selected head and its refusals, and the runtime's approval of
// the published drafter (runtime/approvals.h) against the plan's
// transforms; its inventory and composition are pinned from the published
// artifact (ApprovalsModels).
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "components/step.h"
#include "import/normalize.h"
#include "qwen38_fixture.h"
#include "runtime/approvals.h"
#include "runtime/component_admission.h"
#include "spec/mappings/qwen38.h"

namespace llmp::import {
namespace {
namespace fs = std::filesystem;
using format::Encoding;
using format::PartRole;

NormalizationTarget NativeTarget() {
  auto fixture = test_support::qwen38::NativeModelOpt();
  return {&spec::mappings::Qwen38NativeMapping(), std::move(fixture.model),
          std::move(fixture.source)};
}
std::expected<NormalizedPlan, Error> NormalizeDrafterFixture(const Checkpoint& source,
                                                             std::optional<DraftHeadList> head,
                                                             NormalizationLimits limits) {
  auto target = NativeTarget();
  return Normalize(source, limits, &target, {.component = "mtp", .draft_head = std::move(head)});
}

enum class Kind : std::uint8_t { kDense, kCodes, kScales, kScale2 };
struct Role {
  std::string_view pattern, dtype;
  std::vector<std::uint64_t> dims;
};
// The checkpoint's dense mtp.* descriptors (modelopt_qwen38.cc's pinned
// rules); the hyper-connections' and the experts' are MtpDescriptors'.
std::vector<Role> Roles() {
  std::vector<Role> out = {
      {"mtp.fc_embedding.weight", "BF16", {2560, 2560}},
      {"mtp.fc_hidden.weight", "BF16", {2560, 2560}},
      {"mtp.pre_fc_norm_embedding.weight", "BF16", {2560}},
      {"mtp.pre_fc_norm_hidden.weight", "BF16", {10240}},
      {"mtp.hyper_connection_mixer.hc_norm.weight", "BF16", {10240}},
      {"mtp.hyper_connection_mixer.input_mix_weight_down.weight", "BF16", {320, 10240}},
      {"mtp.hyper_connection_mixer.input_mix_weight_up.weight", "BF16", {10240, 320}},
      {"mtp.layers.0.mlp.gate.weight", "BF16", {512, 2560}},
      {"mtp.layers.0.mlp.shared_expert.down_proj.weight", "BF16", {2560, 640}},
      {"mtp.layers.0.mlp.shared_expert.gate_proj.weight", "BF16", {640, 2560}},
      {"mtp.layers.0.mlp.shared_expert.up_proj.weight", "BF16", {640, 2560}},
      {"mtp.layers.0.mlp.shared_expert_gate.weight", "BF16", {1, 2560}},
      {"mtp.layers.0.self_attn.indexer.index_qk_proj.weight", "BF16", {640, 2560}},
      {"mtp.layers.0.self_attn.indexer.k_layernorm.weight", "BF16", {128}},
      {"mtp.layers.0.self_attn.indexer.q_layernorm.weight", "BF16", {128}},
      {"mtp.layers.0.self_attn.k_norm.weight", "BF16", {256}},
      {"mtp.layers.0.self_attn.k_proj.weight", "BF16", {512, 2560}},
      {"mtp.layers.0.self_attn.o_proj.weight", "BF16", {2560, 6144}},
      {"mtp.layers.0.self_attn.q_norm.weight", "BF16", {256}},
      {"mtp.layers.0.self_attn.q_proj.weight", "BF16", {12288, 2560}},
      {"mtp.layers.0.self_attn.v_proj.weight", "BF16", {512, 2560}},
  };
  return out;
}

struct Descriptor {
  std::string name;
  std::string_view dtype;
  std::vector<std::uint64_t> dims;
  Kind kind = Kind::kDense;
  std::uint64_t k = 0, n = 0;  // a quantized expert projection's
};

std::vector<Descriptor> MtpDescriptors() {
  std::vector<Descriptor> out;
  for (const auto& r : Roles()) out.push_back({std::string(r.pattern), r.dtype, r.dims});
  for (const auto hc : {"attn", "mlp"}) {
    const auto at = [&](std::string_view rest) {
      return std::format("mtp.layers.0.{}_hyper_connection.{}", hc, rest);
    };
    out.push_back({at("block_inject_weight.weight"), "BF16", {4, 10240}});
    out.push_back({at("hc_norm.weight"), "BF16", {10240}});
    out.push_back({at("input_mix_weight_down.weight"), "BF16", {320, 10240}});
    out.push_back({at("input_mix_weight_up.weight"), "BF16", {10240, 320}});
  }
  for (std::uint32_t e = 0; e < 512; ++e)
    for (const auto proj : {"gate", "up", "down"}) {
      const bool down = std::string_view(proj) == "down";
      const std::uint64_t n = down ? 2560 : 640, k = down ? 640 : 2560;
      const auto base = std::format("mtp.layers.0.mlp.experts.{}.{}_proj.", e, proj);
      out.push_back({base + "weight", "U8", {n, k / 2}, Kind::kCodes, k, n});
      out.push_back({base + "weight_scale", "F8_E4M3", {n, k / 16}, Kind::kScales, k, n});
      out.push_back({base + "weight_scale_2", "F32", {}, Kind::kScale2, k, n});
    }
  return out;
}

Tensor Make(const Descriptor& d, std::uint32_t file, std::uint64_t& offset) {
  Tensor t{.name = d.name, .dtype = d.dtype, .dims = d.dims};
  if (d.kind == Kind::kDense) {
    t.format = {.encoding = Encoding::kBF16,
                .layout = format::Layout::kDense,
                .shape = {.ne = {1, 1, 1, 1}, .rank = static_cast<std::uint8_t>(d.dims.size())}};
    for (std::size_t i = 0; i < d.dims.size(); ++i)
      t.format.shape.ne[i] = d.dims[d.dims.size() - 1 - i];
  } else {
    t.format = {.encoding = Encoding::kNvfp4,
                .params = {.tensor_scale = true},
                .layout = format::Layout::kSplit,
                .shape = {.ne = {d.k, d.n, 1, 1}, .rank = 2}};
    t.part = d.kind == Kind::kCodes    ? PartRole::kCodes
             : d.kind == Kind::kScales ? PartRole::kBlockScales
                                       : PartRole::kTensorScale;
  }
  const auto bytes = format::PartsOf(t.format)->Find(t.part)->bytes;
  t.extent = {.file = file, .offset = offset, .bytes = bytes};
  offset += bytes;
  return t;
}

// config.json's checked keys (modelopt_qwen38.cc Config), sorted.
std::vector<MetaEntry> Metadata() {
  std::vector<MetaEntry> m;
  const auto i = [&](std::string_view k, std::int64_t v) { m.push_back({std::string(k), v}); };
  const auto r = [&](std::string_view k, double v) { m.push_back({std::string(k), v}); };
  const auto t = [&](std::string_view k, std::string_view v) {
    m.push_back({std::string(k), std::string(v)});
  };
  const auto b = [&](std::string_view k, bool v) { m.push_back({std::string(k), v}); };
  t("model_type", "qwen3_8_flash_next");
  t("dtype", "bfloat16");
  b("is_encoder_decoder", false);
  b("tie_word_embeddings", false);
  r("text_config.attention_dropout", 0.0);
  b("text_config.attention_bias", false);
  i("text_config.bos_token_id", 248044);
  t("text_config.dtype", "bfloat16");
  i("text_config.eos_token_id", 248044);
  i("text_config.hc_count", 4);
  i("text_config.hc_lowrank", 320);
  i("text_config.head_dim", 256);
  i("text_config.heads_per_ngram", 8);
  t("text_config.hidden_act", "silu");
  i("text_config.hidden_size", 2560);
  i("text_config.indexer_budget", 2048);
  i("text_config.indexer_compress_ratio", 4);
  i("text_config.indexer_head_dim", 128);
  i("text_config.indexer_kv_heads", 1);
  i("text_config.indexer_n_heads", 4);
  b("text_config.is_encoder_decoder", false);
  std::vector<std::string> layers;
  for (std::size_t l = 0; l < 48; ++l)
    layers.emplace_back(l % 4 == 3 ? "full_attention" : "linear_attention");
  m.push_back({"text_config.layer_types", layers});
  i("text_config.linear_conv_kernel_dim", 4);
  i("text_config.linear_key_head_dim", 128);
  i("text_config.linear_num_key_heads", 16);
  i("text_config.linear_num_value_heads", 48);
  i("text_config.linear_value_head_dim", 128);
  i("text_config.make_ngram_vocab_size_divisible_by", 128);
  t("text_config.mamba_ssm_dtype", "float32");
  i("text_config.max_position_embeddings", 262144);
  t("text_config.model_type", "qwen3_8_flash_next_text");
  i("text_config.moe_intermediate_size", 640);
  m.push_back({"text_config.mtp.layer_types", std::vector<std::string>{"full_attention"}});
  i("text_config.mtp.num_hidden_layers", 1);
  i("text_config.mtp_num_hidden_layers", 1);
  i("text_config.ngram_size", 3);
  i("text_config.ngram_vocab_size_base", 20000000);
  i("text_config.num_attention_heads", 24);
  i("text_config.num_experts", 512);
  i("text_config.num_experts_per_tok", 10);
  i("text_config.num_hidden_layers", 48);
  i("text_config.num_key_value_heads", 2);
  t("text_config.output_gate_type", "sigmoid");
  i("text_config.ple_conv_kernel_size", 4);
  i("text_config.ple_embed_dim", 2560);
  t("text_config.ple_embedding_dtype", "nvfp4");
  m.push_back({"text_config.ple_layer_ids", std::vector<std::int64_t>{2}});
  r("text_config.rms_norm_eps", 0.000001);
  b("text_config.rope_parameters.mrope_interleaved", true);
  m.push_back({"text_config.rope_parameters.mrope_section", std::vector<std::int64_t>{11, 11, 10}});
  r("text_config.rope_parameters.partial_rotary_factor", 0.25);
  r("text_config.rope_parameters.rope_theta", 10000000.0);
  t("text_config.rope_parameters.rope_type", "default");
  i("text_config.shared_expert_intermediate_size", 640);
  i("text_config.split_ngram_parts", 128);
  b("text_config.tie_word_embeddings", false);
  i("text_config.vocab_size", 248320);
  i("vision_config.depth", 27);
  i("vision_config.hidden_size", 1152);
  i("vision_config.in_channels", 3);
  i("vision_config.intermediate_size", 4304);
  t("vision_config.model_type", "qwen3_5_vision");
  i("vision_config.num_heads", 16);
  i("vision_config.num_position_embeddings", 2304);
  i("vision_config.out_hidden_size", 2560);
  i("vision_config.patch_size", 16);
  i("vision_config.spatial_merge_size", 2);
  i("vision_config.temporal_patch_size", 2);
  std::ranges::sort(m, {}, &MetaEntry::key);
  return m;
}

constexpr std::array<std::string_view, 6> kFiles = {"config.json",
                                                    "hf_quant_config.json",
                                                    "model-00001-of-00034.safetensors",
                                                    "model-00033-of-00034.safetensors",
                                                    "model-00034-of-00034.safetensors",
                                                    "model.safetensors.index.json"};

// A descriptor-only checkpoint: what the safetensors reader gives the
// pinned source, with one target tensor and the head in their shards.
Checkpoint Source() {
  Checkpoint c{.role = {},
               .container = "safetensors",
               .architecture = "qwen3_8_flash_next",
               .quantization = "modelopt MIXED_PRECISION"};
  for (std::size_t f = 0; f < kFiles.size(); ++f)
    c.files.push_back({.name = std::string(kFiles[f]),
                       .path = fs::path("/fixture") / kFiles[f],
                       .bytes = std::uint64_t{8} << 30U,
                       .device = 1,
                       .inode = 100 + f,
                       .parsed = {64, base::Sha256().Update(kFiles[f]).Finish()}});
  c.metadata = Metadata();
  std::uint64_t offset = 4096;
  for (const auto& d : MtpDescriptors()) c.tensors.push_back(Make(d, 4, offset));
  std::uint64_t other = 4096;
  c.tensors.push_back(
      Make({"model.language_model.embed_tokens.weight", "BF16", {248320, 2560}}, 2, other));
  std::uint64_t head = 8192;
  c.tensors.push_back(Make({"lm_head.weight", "BF16", {248320, 2560}}, 3, head));
  return c;
}

fs::path WriteList(std::string_view name, std::string_view text) {
  const auto dir = fs::path(testing::TempDir()) / "mtp-draft-head";
  fs::create_directories(dir);
  const auto path = dir / name;
  std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
  return path;
}

NormalizationLimits Paid(const Checkpoint& c, std::size_t ids) {
  auto bound = NormalizationBound(c, {.component = "mtp", .draft_head_rows = ids});
  EXPECT_TRUE(bound) << (bound ? "" : bound.error().ToString());
  return bound ? NormalizationLimits{bound->metadata_bytes, bound->work_units}
               : NormalizationLimits{};
}

// The pinned files' identities as the published manifest lists them.
std::vector<artifact::Source> Origins(bool head, std::string_view list = "ids.txt") {
  const auto hex = [](std::string_view text) {
    base::Sha256Digest d{};
    for (std::size_t i = 0; i < d.size(); ++i)
      d[i] = static_cast<std::uint8_t>(std::stoi(std::string(text.substr(2 * i, 2)), nullptr, 16));
    return d;
  };
  std::vector<artifact::Source> out = {
      {"config.json", base::Bytes(109897),
       hex("2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e")},
      {"hf_quant_config.json", base::Bytes(99088),
       hex("d60c54caa65317fd8f1338bdeb81b9014512ac8f21ec18da22541cb88c6784b7")},
      {"model-00034-of-00034.safetensors", base::Bytes(1597270800),
       hex("fe0a0a227e39c4d02dc66c726d3e1ad87ef082e90a6b023ac90b2bb301672203")},
      {"model.safetensors.index.json", base::Bytes(33074286),
       hex("435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9")}};
  if (head) {
    out.push_back({"model-00033-of-00034.safetensors", base::Bytes(5021547836),
                   hex("0fbcdf16c13cd830447efdcd3be19b82ec0e76b7f57f0e96952c1f2b62002b11")});
    out.push_back({std::string(list), base::Bytes(12), base::Sha256Digest{}});
  }
  return out;
}

// The approved drafter: Mia's 47,172-token head.
const runtime::Approval& MtpApproval() {
  const auto* a = runtime::FindApprovalNamed("qwen3.8-flash-next-modelopt-mtp-mia47172");
  EXPECT_NE(a, nullptr);
  return *a;
}
// The canonical tensors of the plan's outputs `pick` selects, sorted.
template <class Plan, class Pick>
std::vector<std::string> NamesOf(const Plan& plan, Pick pick) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < plan.bindings().size(); ++i)
    if (pick(plan.bindings()[i].output)) out.push_back(plan.canonical().tensors[i].name);
  std::ranges::sort(out);
  return out;
}
// A transform's record as approvals.h digests it.
bool Recorded(const runtime::ApprovedTransform& want, std::string_view name,
              const std::vector<std::string>& sorted) {
  base::Sha256 hash;
  for (const auto& t : sorted) hash.Update(t).Update("\n");
  return want.name == name && want.tensors == sorted.size() &&
         base::ToHex(hash.Finish()) == want.digest;
}
// The plan's fold and relayout, exactly the approval's records: its nine
// norms folded (by role), its experts' four tables relaid out.
template <class Plan>
void ExpectApprovedTransforms(const Plan& plan) {
  const auto& a = MtpApproval();
  ASSERT_EQ(a.fold_tensors.size(), 1U);
  ASSERT_EQ(a.relayouts.size(), 1U);
  ASSERT_EQ(plan.planned_folds().size(), 1U);
  const auto& folded = plan.planned_folds()[0].outputs;
  EXPECT_TRUE(
      Recorded(a.fold_tensors[0], plan.planned_folds()[0].fold,
               NamesOf(plan, [&](std::uint32_t o) { return std::ranges::contains(folded, o); })));
  EXPECT_TRUE(Recorded(a.relayouts[0], "split", NamesOf(plan, [&](std::uint32_t o) {
                         return plan.parts().outputs()[o].relayout_from.has_value();
                       })));
  EXPECT_EQ(plan.canonical().folds, std::vector<std::string>(a.folds.begin(), a.folds.end()));
}

const spec::SourceValue* Find(std::span<const spec::SourceValue> values, std::string_view key) {
  const auto at = std::ranges::find(values, key, &spec::SourceValue::key);
  return at == values.end() ? nullptr : &*at;
}

TEST(ImportModelOptDrafterTest, ReadsAStrictAscendingDraftHeadList) {
  auto list = ReadDraftHeadList(WriteList("ids.txt", "0\n1\n2\n7\n248319\n"), 248320);
  ASSERT_TRUE(list) << list.error().ToString();
  EXPECT_EQ(list->ids, (std::vector<std::int32_t>{0, 1, 2, 7, 248319}));
  EXPECT_EQ(list->file.name, "ids.txt");
  EXPECT_EQ(list->file.parsed.bytes, list->file.bytes);
  EXPECT_EQ(list->file.parsed.sha256, base::Sha256().Update("0\n1\n2\n7\n248319\n").Finish());
  auto crlf = ReadDraftHeadList(WriteList("crlf.txt", "3\r\n9"), 248320);
  ASSERT_TRUE(crlf) << crlf.error().ToString();
  EXPECT_EQ(crlf->ids, (std::vector<std::int32_t>{3, 9}));
  for (const auto& [name, text] :
       std::vector<std::pair<std::string, std::string>>{{"empty.txt", ""},
                                                        {"blank.txt", "\n"},
                                                        {"gap.txt", "1\n\n2\n"},
                                                        {"descending.txt", "2\n1\n"},
                                                        {"repeated.txt", "1\n1\n"},
                                                        {"outside.txt", "248320\n"},
                                                        {"negative.txt", "-1\n"},
                                                        {"space.txt", "1 \n"},
                                                        {"plus.txt", "+1\n"},
                                                        {"word.txt", "one\n"},
                                                        {"huge.txt", "99999999999\n"}})
    EXPECT_FALSE(ReadDraftHeadList(WriteList(name, text), 248320)) << name;
  EXPECT_FALSE(ReadDraftHeadList(WriteList("bad name.txt", "1\n"), 248320));
  EXPECT_FALSE(ReadDraftHeadList(WriteList("small.txt", "0\n1\n2\n3\n"), 3));
  EXPECT_FALSE(ReadDraftHeadList(fs::path(testing::TempDir()) / "absent.txt", 248320));
}

TEST(ImportModelOptDrafterTest, NormalizesTheDrafterInItsSourceEncodings) {
  const auto source = Source();
  auto plan = NormalizeDrafterFixture(source, std::nullopt, Paid(source, 0));
  ASSERT_TRUE(plan) << plan.error().ToString();
  // Only the configuration documents and the shard holding mtp.*.
  std::vector<std::string> files;
  for (const auto& f : plan->parts().source().files) files.push_back(f.name);
  EXPECT_EQ(files, (std::vector<std::string>{"config.json", "hf_quant_config.json",
                                             "model-00034-of-00034.safetensors",
                                             "model.safetensors.index.json"}));
  EXPECT_EQ(plan->parts().source().architecture, source.architecture);
  EXPECT_EQ(plan->parts().source().tensors.size(), 29U + 9U * 512U);
  EXPECT_TRUE(plan->generated().empty());
  EXPECT_EQ(plan->selected_rows(), 0U);
  EXPECT_TRUE(plan->options().empty());
  // 29 dense tensors, the relaid gate-up and down tables' two parts each,
  // and three projections' tensor-scale vectors.
  ASSERT_EQ(plan->canonical().tensors.size(), 29U + 4U + 3U);
  EXPECT_EQ(plan->bindings().size(), plan->canonical().tensors.size());
  const auto relaid = NamesOf(
      *plan, [&](std::uint32_t o) { return plan->parts().outputs()[o].relayout_from.has_value(); });
  const auto folded = NamesOf(*plan, [&](std::uint32_t o) {
    return !plan->planned_folds().empty() &&
           std::ranges::contains(plan->planned_folds()[0].outputs, o);
  });
  EXPECT_EQ(relaid.size(), 4U);
  EXPECT_EQ(folded.size(), 9U);
  for (const auto& t : plan->canonical().tensors) {
    EXPECT_EQ(t.experts, std::ranges::contains(relaid, t.name)) << t.name;
    if (t.name.ends_with("_exps.weight_scale_2")) {
      EXPECT_EQ(t.format.encoding, Encoding::kF32) << t.name;
      EXPECT_EQ(t.format.shape.rank, 1U);
      EXPECT_EQ(t.format.shape.ne[0], 512U);
    } else if (t.experts) {
      EXPECT_EQ(t.format.encoding, Encoding::kNvfp4) << t.name;
      EXPECT_EQ(t.format.layout, format::Layout::kScaleAtoms) << t.name;
      EXPECT_FALSE(t.format.params.tensor_scale);
      EXPECT_EQ(t.format.stacked, t.name.contains("gate_up") ? 2U : 1U) << t.name;
      EXPECT_EQ(t.format.shape.rank, 3U);
      EXPECT_EQ(t.format.shape.ne[2], 512U);
    } else {
      // BF16 as the checkpoint holds them; the norms' 1 + w in F32.
      EXPECT_EQ(t.format.encoding,
                std::ranges::contains(folded, t.name) ? Encoding::kF32 : Encoding::kBF16)
          << t.name;
    }
  }
  ASSERT_EQ(plan->planned_folds().size(), 1U);
  EXPECT_EQ(plan->planned_folds()[0].fold, "norm.plus_one");
  EXPECT_EQ(plan->planned_folds()[0].outputs.size(), 9U);
  for (const auto o : plan->planned_folds()[0].outputs)
    EXPECT_TRUE(std::ranges::contains(folded, plan->parts().outputs()[o].name));
  const auto gate = std::ranges::find(plan->canonical().tensors, "blk.0.ffn_gate_inp_shexp.weight",
                                      &spec::SourceTensor::name);
  ASSERT_NE(gate, plan->canonical().tensors.end());
  EXPECT_EQ(gate->format.shape.rank, 1U);
  EXPECT_EQ(plan->canonical().folds, std::vector<std::string>{"norm.plus_one"});
  // Composed with the native target: the drafter stage, its norms' fold
  // (no zero-centered flag left), its head its target's.
  ASSERT_EQ(plan->spec().stages.size(), 2U);
  const auto& stage = plan->spec().stages[1];
  EXPECT_EQ(stage.role, spec::StageRole::kDrafter);
  EXPECT_EQ(stage.artifact, 1U);
  EXPECT_EQ(stage.folds, std::vector<std::string>{"norm.plus_one"});
  ASSERT_TRUE(stage.heads.output.has_value());
  EXPECT_TRUE(std::ranges::all_of(stage.heads.output->weights,
                                  [](const auto& w) { return w.stage == "target"; }));
  ASSERT_TRUE(stage.embedding.has_value());
  EXPECT_EQ(stage.embedding->Find("norm_zero_centered"), nullptr);
  const auto* vocab = Find(plan->values(), "qwen4exp-mtp.draft_vocab");
  ASSERT_NE(vocab, nullptr);
  EXPECT_EQ(std::get<std::int64_t>(vocab->value), 65536);
  EXPECT_EQ(Find(plan->values(), "qwen4exp-mtp.draft_head_rows"), nullptr);
  // The transforms the approval records; without a head, two tensors
  // fewer than the approved drafter's.
  ExpectApprovedTransforms(*plan);
  EXPECT_EQ(plan->canonical().tensors.size() + 2, MtpApproval().tensors);
}

TEST(ImportModelOptDrafterTest, GathersASelectedHeadInItsListsOrder) {
  const auto source = Source();
  auto list = ReadDraftHeadList(WriteList("ids.txt", "0\n1\n2\n10\n248319\n"), 248320);
  ASSERT_TRUE(list);
  auto plan = NormalizeDrafterFixture(source, std::move(*list), Paid(source, 5));
  ASSERT_TRUE(plan) << plan.error().ToString();
  const auto& sub = plan->parts().source();
  ASSERT_EQ(sub.files.size(), 6U);
  EXPECT_EQ(sub.files[3].name, "model-00034-of-00034.safetensors");
  EXPECT_EQ(sub.files[2].name, "model-00033-of-00034.safetensors");
  EXPECT_EQ(sub.files.back().name, "ids.txt");
  // Three runs of rows from lm_head.weight, in order.
  const auto lm = *source.FindTensor("lm_head.weight");
  std::vector<Extent> runs;
  for (const auto& t : sub.tensors)
    if (std::ranges::any_of(plan->parts().members(), [&](const auto& member) {
          return member.source == static_cast<std::uint32_t>(&t - sub.tensors.data()) &&
                 plan->parts().outputs()[member.output].name == "draft_output.weight";
        }))
      runs.push_back(t.extent);
  ASSERT_EQ(runs.size(), 3U);
  EXPECT_EQ(runs[0], (Extent{.file = 2, .offset = lm.extent.offset, .bytes = 3 * 5120}));
  EXPECT_EQ(runs[1], (Extent{.file = 2, .offset = lm.extent.offset + 10 * 5120, .bytes = 5120}));
  EXPECT_EQ(runs[2],
            (Extent{.file = 2, .offset = lm.extent.offset + 248319ULL * 5120, .bytes = 5120}));
  const auto head_at =
      std::ranges::find(plan->parts().outputs(), "draft_output.weight", &IndexedOutput::name);
  ASSERT_NE(head_at, plan->parts().outputs().end());
  const auto& head = *head_at;
  EXPECT_EQ(head.name, "draft_output.weight");
  EXPECT_EQ(head.format.shape.ne[1], 5U);
  // The token map: the last canonical tensor, generated from the list.
  ASSERT_EQ(plan->generated().size(), 1U);
  const auto& ids = plan->generated()[0];
  EXPECT_EQ(ids.tensor, plan->canonical().tensors.size() - 1);
  EXPECT_EQ(ids.file, sub.files.size() - 1);
  ASSERT_EQ(ids.bytes.size(), 20U);
  EXPECT_EQ(ids.bytes[12], std::byte{10});
  EXPECT_EQ(ids.bytes[16], std::byte{0xFF});
  EXPECT_EQ(ids.bytes[17], std::byte{0xC9});
  EXPECT_EQ(ids.bytes[18], std::byte{0x03});
  EXPECT_EQ(plan->canonical().tensors.back().name, "draft_output.ids");
  EXPECT_EQ(plan->canonical().tensors.back().format.encoding, Encoding::kI32);
  EXPECT_EQ(plan->bindings().size() + 1, plan->canonical().tensors.size());
  EXPECT_EQ(plan->selected_rows(), 5U);
  EXPECT_EQ(plan->options(), R"({"component":"mtp","draft_head":"ids.txt"})");
  const auto* rows = Find(plan->values(), "qwen4exp-mtp.draft_head_rows");
  ASSERT_NE(rows, nullptr);
  EXPECT_EQ(std::get<std::int64_t>(rows->value), 5);
  EXPECT_EQ(std::get<std::int64_t>(Find(plan->values(), "qwen4exp-mtp.draft_vocab")->value), 5);
  // Its own selected head binds the drafter's tensors, not its target's.
  const auto& output = plan->spec().stages[1].heads.output;
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(std::ranges::all_of(output->weights, [](const auto& w) { return w.stage.empty(); }));
  ExpectApprovedTransforms(*plan);
  EXPECT_EQ(plan->canonical().tensors.size(), MtpApproval().tensors);
}

TEST(ImportModelOptDrafterTest, RefusesWhatItCannotDescribeOrIsNotPaidFor) {
  const auto source = Source();
  const auto refused = [&](Checkpoint c, std::optional<DraftHeadList> head = std::nullopt) {
    const auto paid = Paid(source, head ? head->ids.size() : 0);
    return !NormalizeDrafterFixture(c, std::move(head), paid).has_value();
  };
  {  // an expert part missing
    auto c = source;
    std::erase_if(c.tensors, [](const Tensor& t) {
      return t.name == "mtp.layers.0.mlp.experts.17.up_proj.weight_scale";
    });
    EXPECT_TRUE(refused(c));
  }
  {  // a descriptor differs
    auto c = source;
    for (auto& t : c.tensors)
      if (t.name == "mtp.fc_hidden.weight") t.dims = {2560, 2561};
    EXPECT_TRUE(refused(c));
  }
  {  // an unknown mtp.* tensor
    auto c = source;
    std::uint64_t at = std::uint64_t{7} << 30U;
    c.tensors.push_back(Make({"mtp.extra.weight", "BF16", {4}}, 4, at));
    EXPECT_TRUE(refused(c));
  }
  {  // a configuration outside the contract
    auto c = source;
    for (auto& e : c.metadata)
      if (e.key == "text_config.mtp_num_hidden_layers") e.value = std::int64_t{2};
    EXPECT_TRUE(refused(c));
  }
  {  // a head without lm_head.weight
    auto c = source;
    std::erase_if(c.tensors, [](const Tensor& t) { return t.name == "lm_head.weight"; });
    auto list = ReadDraftHeadList(WriteList("ids.txt", "1\n"), 248320);
    ASSERT_TRUE(list);
    EXPECT_TRUE(refused(c, std::move(*list)));
  }
  {  // a head list named as a checkpoint file, read or not
    for (const std::string name : {"config.json", "model-00001-of-00034.safetensors"}) {
      auto list = ReadDraftHeadList(WriteList(name, "1\n"), 248320);
      ASSERT_TRUE(list);
      EXPECT_TRUE(refused(source, std::move(*list))) << name;
    }
  }
  {  // another checkpoint kind
    auto c = source;
    c.quantization = "mlx affine";
    auto normalized = NormalizeDrafterFixture(c, std::nullopt, Paid(c, 0));
    EXPECT_TRUE(normalized);  // tensor descriptors, rather than provenance text, govern assembly
  }
  // Not paid for.
  auto bound = NormalizationBound(source, {.component = "mtp"});
  ASSERT_TRUE(bound);
  EXPECT_FALSE(NormalizeDrafterFixture(source, std::nullopt,
                                       {bound->metadata_bytes - 1, bound->work_units}));
}

// Historical performance evidence identifies its list, while valid component
// structure admits independently of that evidence.
TEST(ImportModelOptDrafterTest, TheApprovalNamesOneListAndItsInventory) {
  const auto source = Source();
  auto list = ReadDraftHeadList(WriteList("ids.txt", "4\n5\n"), 248320);
  ASSERT_TRUE(list);
  auto plan = NormalizeDrafterFixture(source, std::move(*list), Paid(source, 2));
  ASSERT_TRUE(plan) << plan.error().ToString();
  const auto& a = MtpApproval();
  EXPECT_TRUE(a.drafter);
  EXPECT_EQ(a.target, "qwen3.8-flash-next-modelopt");
  EXPECT_EQ(runtime::FindApproval("qwen4exp-mtp", Origins(true)), nullptr);
  EXPECT_EQ(runtime::FindApproval("qwen4exp-mtp", Origins(false)), nullptr);
  std::vector<artifact::Source> approved;
  for (const auto& r : a.sources) {
    base::Sha256Digest d{};
    for (std::size_t i = 0; i < d.size(); ++i)
      d[i] =
          static_cast<std::uint8_t>(std::stoi(std::string(r.sha256.substr(2 * i, 2)), nullptr, 16));
    approved.push_back({std::string(r.name), base::Bytes(r.bytes), d});
  }
  EXPECT_EQ(runtime::FindApproval("qwen4exp-mtp", approved), &a);
  EXPECT_EQ(runtime::FindApproval("qwen4exp", approved), nullptr);
  // Two rows, not Mia's 47,172: not the approved inventory.
  // A normalization plan has not executed its folds. Admission must refuse
  // that original projection, even when its source identity is approved.
  EXPECT_FALSE(runtime::CheckComponentDrafterAdmission(plan->spec(), plan->canonical(), approved));
  auto published = plan->canonical();  // synthetic post-publication descriptors only
  for (const auto& fold : plan->planned_folds()) {
    std::vector<std::string> tensors;
    for (std::size_t i = 0; i < plan->bindings().size(); ++i)
      if (std::ranges::contains(fold.outputs, plan->bindings()[i].output))
        tensors.push_back(published.tensors[i].name);
    std::ranges::sort(tensors);
    tensors.erase(std::unique(tensors.begin(), tensors.end()), tensors.end());
    ASSERT_FALSE(tensors.empty());
    published.preprocessing.push_back({.name = fold.fold, .tensors = std::move(tensors)});
  }
  ASSERT_TRUE(plan->applied_folds().empty());
  const auto admitted = runtime::CheckComponentDrafterAdmission(plan->spec(), published, approved);
  ASSERT_TRUE(admitted) << admitted.error();
  auto other = published;
  other.folds.push_back("unknown.fold");
  EXPECT_FALSE(runtime::CheckComponentDrafterAdmission(plan->spec(), other, approved));
}

TEST(ImportModelOptDrafterTest, TargetSourceAliasesUseCanonicalMappingIdentity) {
  const auto source = Source();
  const auto limits = Paid(source, 0);
  std::optional<base::Sha256Digest> canonical_digest;
  for (const std::string architecture : {"qwen4exp", "qwen3_8_flash_next", "qwen4_exp"}) {
    SCOPED_TRACE(architecture);
    auto target = NativeTarget();
    target.source.architecture = architecture;
    ASSERT_TRUE(ValidateTarget(target, spec::mappings::Qwen38MtpNativeMapping()));
    const auto plan = Normalize(source, limits, &target, {.component = "mtp"});
    ASSERT_TRUE(plan) << plan.error().ToString();
    EXPECT_EQ(plan->canonical().architecture, "qwen4exp-mtp");
    EXPECT_EQ(plan->source().architecture, "qwen3_8_flash_next");
    if (!canonical_digest) canonical_digest = plan->spec().Digest();
    EXPECT_EQ(plan->spec().Digest(), *canonical_digest);
  }
  for (unsigned mutation = 0; mutation < 5; ++mutation) {
    SCOPED_TRACE(mutation);
    auto target = NativeTarget();
    target.source.architecture = "qwen3_8_flash_next";
    if (mutation == 0) {
      target.source.architecture = "gemma4";
      target.source.container = "gguf";
      target.mapping = spec::FindMapping(target.source.architecture, target.source.container);
      ASSERT_NE(target.mapping, nullptr);
    }
    if (mutation == 1) target.source.container = "gguf";
    if (mutation == 2) target.mapping = &spec::mappings::Qwen38MtpNativeMapping();
    if (mutation == 3) target.model.stages.front().role = spec::StageRole::kDrafter;
    if (mutation == 4) target.model.stages.front().stream.width = 1;
    EXPECT_FALSE(ValidateTarget(target, spec::mappings::Qwen38MtpNativeMapping()));
    EXPECT_FALSE(Normalize(source, limits, &target, {.component = "mtp"}));
  }
}

TEST(ImportModelOptDrafterTest, SelectedIdsMustFitBothActualTargetAndCheckpointHeadRows) {
  auto target = NativeTarget();
  for (auto& tensor : target.source.tensors)
    if (tensor.name == "token_embd.weight" || tensor.name == "output.weight")
      tensor.format.shape.ne[1] = 300000;
  ASSERT_TRUE(ValidateTarget(target, spec::mappings::Qwen38MtpNativeMapping()));
  auto head = ReadDraftHeadList(WriteList("outside-source-ids.txt", "248320\n"), 300000);
  ASSERT_TRUE(head);
  const auto source = Source();
  const auto normalized = Normalize(source, Paid(source, 1), &target,
                                    {.component = "mtp", .draft_head = std::move(*head)});
  ASSERT_FALSE(normalized);
  EXPECT_NE(normalized.error().reason.find("head"), std::string::npos);
}

TEST(ImportModelOptDrafterTest, MissingTargetRefusesAndPublishedMetadataExtractionUsesActualFacts) {
  const auto source = Source();
  const auto missing = Normalize(source, Paid(source, 0), nullptr, {.component = "mtp"});
  ASSERT_FALSE(missing);
  EXPECT_NE(missing.error().reason.find("--target"), std::string::npos);
  auto target = NativeTarget();
  target.model.stages.front().context = 123456;
  auto own = test_support::qwen38::MtpSource(test_support::families::Qwen38Flash(), 321);
  for (auto& tensor : target.source.tensors)
    if (tensor.name == "output.weight") tensor.format.shape.ne[1] = 100003;
  auto values = spec::ExtractValues(spec::mappings::Qwen38MtpNativeMapping(), target.model,
                                    target.source, own);
  ASSERT_TRUE(values) << values.error();
  EXPECT_EQ(std::get<std::int64_t>(Find(*values, "qwen4exp-mtp.context_length")->value), 123456);
  EXPECT_EQ(std::get<std::int64_t>(Find(*values, "qwen4exp-mtp.draft_head_rows")->value), 321);
  EXPECT_EQ(std::get<std::int64_t>(Find(*values, "qwen4exp-mtp.draft_vocab")->value), 321);
  std::erase_if(own.tensors, [](const spec::SourceTensor& tensor) {
    return tensor.name.starts_with("draft_output.");
  });
  values = spec::ExtractValues(spec::mappings::Qwen38MtpNativeMapping(), target.model,
                               target.source, own);
  ASSERT_TRUE(values);
  EXPECT_EQ(std::get<std::int64_t>(Find(*values, "qwen4exp-mtp.draft_vocab")->value), 65536);
}

TEST(ImportModelOptDrafterTest, TargetExtractionPreservesDeclaredDrafterMath) {
  auto source = Source();
  for (auto& entry : source.metadata) {
    if (entry.key == "text_config.rms_norm_eps") entry.value = 0.000003;
    if (entry.key == "text_config.rope_parameters.rope_theta") entry.value = 900000.0;
  }
  const auto target = NativeTarget();
  auto plan = Normalize(source, Paid(source, 0), &target, {.component = "mtp"});
  ASSERT_TRUE(plan) << plan.error().ToString();
  const auto* epsilon = Find(plan->values(), "qwen4exp-mtp.attention.layer_norm_rms_epsilon");
  const auto* rope = Find(plan->values(), "qwen4exp-mtp.rope.freq_base");
  ASSERT_NE(epsilon, nullptr);
  ASSERT_NE(rope, nullptr);
  EXPECT_DOUBLE_EQ(std::get<double>(epsilon->value), 0.000003);
  EXPECT_DOUBLE_EQ(std::get<double>(rope->value), 900000.0);
  auto composed_target = plan->spec().stages.front();
  // Composition adds taps for its declared reads without changing target math.
  composed_target.taps = target.model.stages.front().taps;
  EXPECT_EQ(composed_target, target.model.stages.front());
}

// The drafter's expert layout comes from its own tensors, never its target's:
// a relaid ModelOpt drafter (stacked gate/up, tensor scales) composes with an
// unstacked GGUF or MLX target of the same geometry.
TEST(ImportModelOptDrafterTest, StackedAndScaledFollowTheDrafterTensors) {
  const auto target = NativeTarget();
  const auto flag = [](const std::vector<spec::SourceValue>& values, std::string_view key) {
    const auto* v = Find(values, key);
    return v != nullptr && std::get<bool>(v->value);
  };
  auto own = test_support::qwen38::MtpSource(test_support::families::Qwen38Flash(), 321);
  std::erase_if(own.tensors, [](const spec::SourceTensor& tensor) {
    return tensor.name.contains("ffn_gate_up_exps") ||
           tensor.name.ends_with("_exps.weight_scale_2");
  });
  auto values = spec::ExtractValues(spec::mappings::Qwen38MtpNativeMapping(), target.model,
                                    target.source, own);
  ASSERT_TRUE(values) << values.error();
  EXPECT_FALSE(flag(*values, "llmp.qwen4exp.experts_stacked"));
  EXPECT_FALSE(flag(*values, "llmp.qwen4exp.expert_scales"));
  own.tensors.push_back({.name = "blk.0.ffn_gate_up_exps.weight"});
  own.tensors.push_back({.name = "blk.0.ffn_gate_exps.weight_scale_2"});
  values = spec::ExtractValues(spec::mappings::Qwen38MtpNativeMapping(), target.model,
                               target.source, own);
  ASSERT_TRUE(values) << values.error();
  EXPECT_TRUE(flag(*values, "llmp.qwen4exp.experts_stacked"));
  EXPECT_TRUE(flag(*values, "llmp.qwen4exp.expert_scales"));
}

}  // namespace
}  // namespace llmp::import
