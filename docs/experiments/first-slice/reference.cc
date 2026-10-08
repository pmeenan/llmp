// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// External llama.cpp reference only; this does not implement llmpalooza inference.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;

static constexpr char prompt[] =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nKeep the label cedar-17. Count: 1, 2, 3. "
    "Text: caf\xc3\xa9, \xe4\xbd\xa0\xe5\xa5\xbd.\n"
    "Code: x = 2 + 3\nWhat label did I give you?<|im_end|>\n"
    "<|im_start|>assistant\nThe label is cedar-17. The sum is 5.";

struct Result {
    std::vector<float> logits;
    size_t state_bytes = 0;
};

static Result evaluate(llama_model * model, const llama_context_params & params,
                       const std::vector<llama_token> & tokens, bool restore) {
    Context ctx(llama_init_from_model(model, params), llama_free);
    require(bool(ctx), "Context creation failed");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int split = 32;
    Result result;
    result.logits.reserve(tokens.size() * size_t(vocab));
    for (int pos = 0; pos < int(tokens.size());) {
        const int count = pos == 0 ? split : 1;
        auto batch = llama_batch_init(count, 0, 1);
        require(batch.token && batch.pos && batch.n_seq_id && batch.seq_id && batch.logits,
                "Batch allocation failed");
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = tokens[size_t(pos + i)];
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        const int status = llama_decode(ctx.get(), batch);
        llama_synchronize(ctx.get());
        llama_batch_free(batch);
        require(status == 0, "Decode failed");
        for (int i = 0; i < count; ++i) {
            const float * row = llama_get_logits_ith(ctx.get(), i);
            require(row != nullptr, "Missing logits");
            require(std::all_of(row, row + vocab, [](float x) { return std::isfinite(x); }),
                    "Nonfinite logits");
            result.logits.insert(result.logits.end(), row, row + vocab);
        }
        pos += count;
        if (restore && pos == split) {
            std::vector<uint8_t> state(llama_state_get_size(ctx.get()));
            require(!state.empty(), "Empty state");
            const size_t written = llama_state_get_data(ctx.get(), state.data(), state.size());
            require(written > 0 && written <= state.size(), "Invalid state write size");
            state.resize(written);
            result.state_bytes = written;
            ctx.reset();
            ctx.reset(llama_init_from_model(model, params));
            require(bool(ctx), "Restored context creation failed");
            require(llama_state_set_data(ctx.get(), state.data(), state.size()) == state.size(),
                    "State restore failed");
        }
    }
    return result;
}

static size_t differing(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "Logit shape mismatch");
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        count += std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i]);
    }
    return count;
}

int main(int argc, char ** argv) try {
    require(argc == 4, "Usage: reference MODEL OUTPUT_DIRECTORY cpu|cuda");
    const std::string backend = argv[3];
    require(backend == "cpu" || backend == "cuda", "Unknown backend");
    require(std::endian::native == std::endian::little, "Expected little-endian host");
    const bool cuda = backend == "cuda";
    const std::filesystem::path output(argv[2]);
    require(std::filesystem::create_directory(output), "Output directory must be new");
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = cuda ? 999 : 0;
    Model model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    require(bool(model), "Model load failed");
    const auto * vocab = llama_model_get_vocab(model.get());
    require(llama_vocab_n_tokens(vocab) == 151936, "Unexpected vocabulary size");
    std::vector<llama_token> tokens(512);
    const int n = llama_tokenize(vocab, prompt, sizeof(prompt) - 1,
                                 tokens.data(), int(tokens.size()), false, true);
    require(n > 40 && n <= 512, "Unexpected token count");
    tokens.resize(size_t(n));
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = cp.n_ubatch = 64;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.offload_kqv = cp.op_offload = cuda;
    const auto first = evaluate(model.get(), cp, tokens, false);
    const auto repeat = evaluate(model.get(), cp, tokens, false);
    const auto restored = evaluate(model.get(), cp, tokens, true);
    const size_t repeat_differences = differing(first.logits, repeat.logits);
    const size_t restore_differences = differing(first.logits, restored.logits);
    std::ofstream ids(output / "tokens.txt");
    for (auto token : tokens) ids << token << '\n';
    ids.close();
    require(bool(ids), "Token write failed");
    std::ofstream raw(output / "logits.f32le", std::ios::binary);
    raw.write(reinterpret_cast<const char *>(first.logits.data()),
              std::streamsize(first.logits.size() * sizeof(float)));
    raw.close();
    require(bool(raw), "Logit write failed");
    const std::string summary = "{\"backend\":\"" + backend +
        "\",\"tokens\":" + std::to_string(n) +
        ",\"vocabulary\":151936,\"logit_values\":" + std::to_string(first.logits.size()) +
        ",\"restore_prefix_tokens\":32,\"state_bytes\":" + std::to_string(restored.state_bytes) +
        ",\"repeat_bit_differences\":" + std::to_string(repeat_differences) +
        ",\"restore_bit_differences\":" + std::to_string(restore_differences) + "}\n";
    std::ofstream record(output / "summary.json");
    record << summary;
    record.close();
    require(bool(record), "Summary write failed");
    std::fputs(summary.c_str(), stdout);
    model.reset();
    llama_backend_free();
    require(repeat_differences == 0, "Fresh-reference repeat changed logits");
    require(restore_differences == 0, "Restored reference changed logits");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
