// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen-Image-2.1 adapter (model/qwen_image.h): the flow-matching
// schedule against the values diffusers set for the reference run
// (docs/experiments/qwen-image-native, reference.py's sigmas), the tensor
// lists and their binding, the VAE decoder's plan, the rotary tables and
// the timestep's sinusoid by their definitions, and the pixels' rounding.
// The installed component artifacts are bound where the M3 store holds
// them (a Spark) and skipped elsewhere.

#include "model/qwen_image.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"

namespace {

namespace md = llmp::model;

TEST(QwenImage, TheScheduleIsDiffusers) {
  // FlowMatchEulerDiscreteScheduler.sigmas for 40 steps over 4,096 image
  // tokens (float32 values as Python printed them from the reference run).
  const std::array<double, 41> want = {1.0,
                                       0.9869637489318848,
                                       0.97359299659729,
                                       0.9598749279975891,
                                       0.9457955360412598,
                                       0.9313406348228455,
                                       0.9164948463439941,
                                       0.9012421369552612,
                                       0.8855656385421753,
                                       0.8694472312927246,
                                       0.8528681993484497,
                                       0.8358084559440613,
                                       0.8182465434074402,
                                       0.8001601696014404,
                                       0.7815254926681519,
                                       0.7623172402381897,
                                       0.7425084114074707,
                                       0.7220702171325684,
                                       0.7009725570678711,
                                       0.679182767868042,
                                       0.6566662788391113,
                                       0.6333861351013184,
                                       0.6093027591705322,
                                       0.5843738913536072,
                                       0.5585541129112244,
                                       0.5317949056625366,
                                       0.5040440559387207,
                                       0.4752454161643982,
                                       0.4453383684158325,
                                       0.41425788402557373,
                                       0.3819332718849182,
                                       0.3482884168624878,
                                       0.31324076652526855,
                                       0.2767007350921631,
                                       0.2385709285736084,
                                       0.19874519109725952,
                                       0.1571078896522522,
                                       0.11353254318237305,
                                       0.06788057088851929,
                                       0.019999980926513672,
                                       0.0};
  const auto s = md::QwenImageSchedulerSigmas(md::QwenImage21().scheduler, 40, 4096);
  ASSERT_TRUE(s.has_value()) << s.error();
  EXPECT_DOUBLE_EQ(s->mu, 0.6935483870967742);
  ASSERT_EQ(s->sigmas.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    EXPECT_EQ(s->sigmas[i], static_cast<float>(want[i])) << i;
  }
  ASSERT_EQ(s->timesteps.size(), 40U);
  EXPECT_EQ(s->timesteps[1], 986.9637451171875f);
  EXPECT_FALSE(md::QwenImageSchedulerSigmas(md::QwenImage21().scheduler, 1, 4096).has_value());
}

TEST(QwenImage, TensorListsNameTheCheckpoint) {
  const md::QwenImageProfile& p = md::QwenImage21();
  const auto text = md::TextEncoderTensors(p.text);
  ASSERT_EQ(text.size(), 1 + (36 * static_cast<std::size_t>(md::TextTensor::kCount)));
  EXPECT_EQ(text[0].name, "model.language_model.embed_tokens.weight");
  EXPECT_EQ(text[1 + static_cast<std::size_t>(md::TextTensor::kK)].shape,
            (std::vector<std::uint64_t>{1024, 4096}));
  const auto dit = md::DenoiserTensors(p.denoiser);
  ASSERT_EQ(dit.size(), static_cast<std::size_t>(md::DenoiserGlobal::kCount) +
                            (32 * static_cast<std::size_t>(md::BlockTensor::kCount)));
  EXPECT_EQ(dit[static_cast<std::size_t>(md::DenoiserGlobal::kModulation)].shape,
            (std::vector<std::uint64_t>{16384, 4096}));
  EXPECT_EQ(dit.back().name, "transformer_blocks.31.img_mlp.out.weight");
  const auto vae = md::VaeDecoderTensors(p.vae);
  for (const auto& t : vae) {
    EXPECT_EQ(t.dtype, "F32") << t.name;
    EXPECT_TRUE(t.name.starts_with("decoder.") || t.name.starts_with("post_quant_conv")) << t.name;
  }
}

TEST(QwenImage, BindingRefusesWhatTheProfileDoesNotRead) {
  const std::vector<md::QwenImageTensor> tensors = {{"a.weight", "BF16", {4, 2}},
                                                    {"b.weight", "F32", {3}}};
  std::vector<md::QwenImageResource> resources = {{{"b.weight"}, "plain", "F32", {3}},
                                                  {{"a.weight", "alias"}, "plain", "BF16", {4, 2}}};
  const auto bound = md::BindQwenImageComponent(tensors, "Arch", "Arch", resources);
  ASSERT_TRUE(bound.has_value()) << bound.error();
  EXPECT_EQ(*bound, (std::vector<std::uint32_t>{1, 0}));
  EXPECT_FALSE(md::BindQwenImageComponent(tensors, "Arch", "Other", resources).has_value());
  resources[0].dtype = "BF16";
  EXPECT_FALSE(md::BindQwenImageComponent(tensors, "Arch", "Arch", resources).has_value());
  resources[0].dtype = "F32";
  resources[1].shape = {2, 4};
  EXPECT_FALSE(md::BindQwenImageComponent(tensors, "Arch", "Arch", resources).has_value());
  resources[1].shape = {4, 2};
  resources[1].family = "ggml";
  EXPECT_FALSE(md::BindQwenImageComponent(tensors, "Arch", "Arch", resources).has_value());
  resources.pop_back();
  EXPECT_FALSE(md::BindQwenImageComponent(tensors, "Arch", "Arch", resources).has_value());
}

TEST(QwenImage, TheVaePlanUpsamplesSixteenFold) {
  const md::QwenImageVaeProfile& p = md::QwenImage21().vae;
  const auto tensors = md::VaeDecoderTensors(p);
  const auto steps = md::VaeDecoderPlan(p, 64, 64);
  int upsamples = 0;
  int dupups = 0;
  for (const md::VaeStep& s : steps) {
    upsamples += s.kind == md::VaeStep::Kind::kUpsample ? 1 : 0;
    dupups += s.kind == md::VaeStep::Kind::kAddDupUp ? 1 : 0;
    EXPECT_LT(s.weight, static_cast<std::int32_t>(tensors.size()));
    EXPECT_LT(s.bias, static_cast<std::int32_t>(tensors.size()));
    EXPECT_LT(s.in, md::kVaeBuffers);
    EXPECT_LT(s.out, md::kVaeBuffers);
    if (s.kind == md::VaeStep::Kind::kConv3x3 || s.kind == md::VaeStep::Kind::kConv1x1) {
      const auto& w = tensors[static_cast<std::size_t>(s.weight)].shape;
      EXPECT_EQ(w[0], s.out_channels);
      EXPECT_EQ(w[1], s.in_channels);
    }
  }
  EXPECT_EQ(upsamples, 4);
  EXPECT_EQ(dupups, 4);
  // The first step is post_quant_conv on the 64-channel latent; the last
  // writes the 4 output channels at 1024 x 1024 into x.
  EXPECT_EQ(steps.front().in_channels, 64U);
  EXPECT_EQ(steps.back().out_channels, 4U);
  EXPECT_EQ(steps.back().height, 1024U);
  EXPECT_EQ(steps.back().out, md::kVaeX);
  // The last two up blocks' shortcuts are temporal (factor 2) then spatial only.
  std::vector<std::uint32_t> factors;
  for (const md::VaeStep& s : steps) {
    if (s.kind == md::VaeStep::Kind::kAddDupUp) {
      factors.push_back(s.factor_t);
    }
  }
  EXPECT_EQ(factors, (std::vector<std::uint32_t>{2, 2, 2, 1}));
}

TEST(QwenImage, RotaryTablesFollowTheirDefinitions) {
  const md::QwenImageDenoiserProfile& d = md::QwenImage21().denoiser;
  const std::uint32_t text = 3;
  const std::uint32_t grid = 4;
  const auto f = md::QwenImageDenoiserRotary(d, text, grid);
  ASSERT_EQ(f.size(), (text + (grid * grid)) * 64U * 2U);
  // Row 0 is position 0 on every axis: cos 1, sin 0.
  for (std::size_t k = 0; k < 64; ++k) {
    EXPECT_EQ(f[2 * k], 1.0f);
    EXPECT_EQ(f[(2 * k) + 1], 0.0f);
  }
  // The image's first token: frame 3, height -2, width -2 (a grid of 4
  // centred on 0); the frame axis' first pair turns by 3 radians.
  const std::size_t row = text;
  EXPECT_FLOAT_EQ(f[row * 128], std::cos(3.0f));
  // Height's first pair (pair 8) turns by -2.
  EXPECT_FLOAT_EQ(f[(row * 128) + 16], std::cos(-2.0f));
  EXPECT_FLOAT_EQ(f[(row * 128) + 17], std::sin(-2.0f));
  // Text rotary: position 2, the first frequency is 1 (theta^0).
  const auto t = md::QwenImageTextRotary(md::QwenImage21().text, 3);
  EXPECT_EQ(t.cos[std::size_t{2} * 128], md::ToBf16(std::cos(2.0f)));
  EXPECT_EQ(t.sin[(std::size_t{2} * 128) + 64], md::ToBf16(std::sin(2.0f)));
}

TEST(QwenImage, TimestepSinusoidAndPixelsRoundAsDiffusers) {
  const md::QwenImageDenoiserProfile& d = md::QwenImage21().denoiser;
  const auto s = md::QwenImageTimestepSinusoid(d, 1000.0f);
  ASSERT_EQ(s.size(), 512U);
  EXPECT_EQ(s[0], md::ToBf16(std::cos(1000.0f)));  // freq 1: cos(1000)
  EXPECT_EQ(s[128], md::ToBf16(std::sin(1000.0f)));
  EXPECT_EQ(s[256], md::ToBf16(1.0f));  // the t = 0 row
  EXPECT_EQ(s[256 + 128], md::ToBf16(0.0f));
  // -1 -> 0, 1 -> 255, 0 -> 127.5 -> 128 (half to even), beyond 1 clamps.
  const std::vector<std::uint16_t> decoded = {md::ToBf16(-1.0f), md::ToBf16(1.0f), md::ToBf16(0.0f),
                                              md::ToBf16(3.0f)};
  const auto px = md::QwenImagePixels(decoded, 1, 2, 2);
  EXPECT_EQ(px, (std::vector<std::uint8_t>{0, 255, 128, 255}));
  EXPECT_EQ(md::ToBf16(1.0f), 0x3f80);
  EXPECT_EQ(md::FromBf16(0x3f80), 1.0f);
}

TEST(QwenImage, InstalledComponentsBind) {
  const char* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  const std::filesystem::path store =
      std::filesystem::path(home != nullptr ? home : "/") / ".local/share/llmp/m3-artifacts";
  std::error_code error;
  std::optional<llmp::artifact::Composition> composition;
  for (const auto& entry : std::filesystem::directory_iterator(store, error)) {
    auto c = llmp::artifact::OpenComposition(entry.path());
    if (c && c->architecture() == md::QwenImage21().pipeline_architecture) {
      composition.emplace(std::move(*c));
    }
  }
  if (!composition) {
    GTEST_SKIP() << "no installed Qwen-Image composition in " << store;
  }
  const md::QwenImageProfile& p = md::QwenImage21();
  const std::array<std::pair<const char*, std::vector<md::QwenImageTensor>>, 3> roles = {{
      {"text_encoder", md::TextEncoderTensors(p.text)},
      {"transformer", md::DenoiserTensors(p.denoiser)},
      {"vae", md::VaeDecoderTensors(p.vae)},
  }};
  const std::array<std::string_view, 3> archs = {p.text_architecture, p.denoiser_architecture,
                                                 p.vae_architecture};
  for (std::size_t i = 0; i < roles.size(); ++i) {
    const auto* c = composition->Find(roles[i].first);
    ASSERT_NE(c, nullptr) << roles[i].first;
    auto a = llmp::artifact::Artifact::Open(store / c->artifact);
    ASSERT_TRUE(a.has_value()) << a.error().ToString();
    const auto bound = md::BindQwenImageComponent(roles[i].second, archs[i], *a);
    EXPECT_TRUE(bound.has_value()) << roles[i].first << ": " << bound.error();
  }
}

}  // namespace
