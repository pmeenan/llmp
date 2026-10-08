// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// External llama.cpp reference harness for the backend proof's P0 controls
// (docs/backend-proof.md); it does not implement llmpalooza inference. It builds
// against the pinned reference image's llama.cpp and against the toolchain
// bridge (the same source built with llmpalooza's SDK).
//
//   fp16_reference MODEL OUTPUT_DIRECTORY cpu|cuda control
//   fp16_reference MODEL OUTPUT_DIRECTORY cpu|cuda heldout IDS_FILE
//
// control: M0's 76-token trajectory with M0's settings (first-slice): a
//   512-token context, batches of 64, a 32-token prefill then single tokens,
//   and a restore after 32 tokens.
// heldout: the first 577 of the declared held-out IDs (little-endian int64)
//   in chunks of 16 and 17 (either side of GGML's MMF/cuBLAS boundary), 16
//   single tokens, a 512-token prefill and 16 more single tokens, with a
//   1024-token context and batches of 512, restored after 33 tokens.
//
// Each run evaluates three times (fresh, repeat, restored) and requires the
// repeat and the restored continuation to be bit-identical to the first.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
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

struct Trajectory {
    std::vector<llama_token> tokens;
    std::vector<int> chunks;  // batch sizes, in order; they sum to tokens.size()
    size_t restore_after = 0;  // tokens evaluated before the restore
    uint32_t context = 0;
    uint32_t batch = 0;
};

struct Result {
    std::vector<float> logits;
    size_t state_bytes = 0;
};

static Result evaluate(llama_model * model, const llama_context_params & params, const Trajectory & t,
                       bool restore) {
    Context ctx(llama_init_from_model(model, params), llama_free);
    require(bool(ctx), "Context creation failed");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    Result result;
    result.logits.reserve(t.tokens.size() * size_t(vocab));
    int pos = 0;
    for (const int count : t.chunks) {
        auto batch = llama_batch_init(count, 0, 1);
        require(batch.token && batch.pos && batch.n_seq_id && batch.seq_id && batch.logits,
                "Batch allocation failed");
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = t.tokens[size_t(pos + i)];
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
        if (restore && size_t(pos) == t.restore_after) {
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
    require(size_t(pos) == t.tokens.size(), "Chunks do not cover the trajectory");
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

static Trajectory control(const llama_vocab * vocab) {
    Trajectory t;
    t.tokens.resize(512);
    const int n = llama_tokenize(vocab, prompt, sizeof(prompt) - 1, t.tokens.data(), int(t.tokens.size()),
                                 false, true);
    require(n > 40 && n <= 512, "Unexpected token count");
    t.tokens.resize(size_t(n));
    t.chunks.push_back(32);
    t.chunks.insert(t.chunks.end(), size_t(n - 32), 1);
    t.restore_after = 32;
    t.context = 512;
    t.batch = 64;
    return t;
}

static Trajectory heldout(const char * path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    require(bytes.size() == 1040 * 8, "Expected 1040 little-endian int64 IDs");
    Trajectory t;
    t.chunks = {16, 17};
    t.chunks.insert(t.chunks.end(), 16, 1);
    t.chunks.push_back(512);
    t.chunks.insert(t.chunks.end(), 16, 1);
    size_t total = 0;
    for (const int c : t.chunks) total += size_t(c);
    for (size_t i = 0; i < total; ++i) {
        int64_t id = 0;
        for (int b = 7; b >= 0; --b) id = (id << 8) | uint8_t(bytes[i * 8 + size_t(b)]);
        require(id >= 0 && id < 151643, "Held-out ID outside the regular vocabulary");
        t.tokens.push_back(llama_token(id));
    }
    t.restore_after = 33;
    t.context = 1024;
    t.batch = 512;
    return t;
}

int main(int argc, char ** argv) try {
    require(argc == 5 || argc == 6, "Usage: fp16_reference MODEL OUTPUT_DIRECTORY cpu|cuda control|heldout [IDS]");
    const std::string backend = argv[3];
    const std::string which = argv[4];
    require(backend == "cpu" || backend == "cuda", "Unknown backend");
    require((which == "control" && argc == 5) || (which == "heldout" && argc == 6), "Unknown trajectory");
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
    const Trajectory t = which == "control" ? control(vocab) : heldout(argv[5]);
    auto cp = llama_context_default_params();
    cp.n_ctx = t.context;
    cp.n_batch = cp.n_ubatch = t.batch;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.offload_kqv = cp.op_offload = cuda;
    const auto first = evaluate(model.get(), cp, t, false);
    const auto repeat = evaluate(model.get(), cp, t, false);
    const auto restored = evaluate(model.get(), cp, t, true);
    const size_t repeat_differences = differing(first.logits, repeat.logits);
    const size_t restore_differences = differing(first.logits, restored.logits);
    std::ofstream ids(output / "tokens.txt");
    for (auto token : t.tokens) ids << token << '\n';
    ids.close();
    require(bool(ids), "Token write failed");
    std::ofstream raw(output / "logits.f32le", std::ios::binary);
    raw.write(reinterpret_cast<const char *>(first.logits.data()),
              std::streamsize(first.logits.size() * sizeof(float)));
    raw.close();
    require(bool(raw), "Logit write failed");
    std::string chunks;
    for (const int c : t.chunks) chunks += (chunks.empty() ? "" : ",") + std::to_string(c);
    const std::string summary = "{\"backend\":\"" + backend + "\",\"trajectory\":\"" + which +
        "\",\"tokens\":" + std::to_string(t.tokens.size()) + ",\"chunks\":[" + chunks +
        "],\"context\":" + std::to_string(t.context) + ",\"batch\":" + std::to_string(t.batch) +
        ",\"vocabulary\":151936,\"logit_values\":" + std::to_string(first.logits.size()) +
        ",\"restore_prefix_tokens\":" + std::to_string(t.restore_after) +
        ",\"state_bytes\":" + std::to_string(restored.state_bytes) +
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
