// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// External reference harness: link only to the pinned llama.cpp image.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Capture {
    FILE * out;
    int step = 0;
    int tokens = 0;
    int layer = 0;
    const char * phase = "prefill";
};
static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
// No exception may cross the backend's C callback boundary.
static bool routes(ggml_tensor * t, bool ask, void * opaque) noexcept {
    auto & c = *static_cast<Capture *>(opaque);
    if (std::strncmp(t->name, "ffn_moe_topk-", 13) != 0) return false;
    if (ask) return true;
    try {
        const auto expected = "ffn_moe_topk-" + std::to_string(c.layer);
        require(expected == t->name && t->type == GGML_TYPE_I32 &&
                t->ne[0] == 8 && t->ne[1] == c.tokens &&
                t->ne[2] == 1 && t->ne[3] == 1, "Unexpected routing tensor/order");
        std::vector<unsigned char> bytes(ggml_nbytes(t));
        ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        std::fprintf(c.out, "{\"step\":%d,\"phase\":\"%s\",\"layer\":%d,\"routes\":[",
                     c.step, c.phase, c.layer);
        for (int i = 0; i < c.tokens; ++i) {
            std::fprintf(c.out, "%s[", i ? "," : "");
            bool seen[128] = {};
            for (int k = 0; k < 8; ++k) {
                int32_t id;
                const size_t off = i * t->nb[1] + k * t->nb[0];
                require(off + sizeof(id) <= bytes.size(), "Invalid routing stride");
                std::memcpy(&id, bytes.data() + off, sizeof(id));
                require(id >= 0 && id < 128 && !seen[id], "Invalid/duplicate expert");
                seen[id] = true;
                std::fprintf(c.out, "%s%d", k ? "," : "", id);
            }
            std::fputc(']', c.out);
        }
        std::fprintf(c.out, "]}\n");
        require(!std::ferror(c.out), "Route write failed");
        ++c.layer;
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Route capture failed: %s\n", e.what());
        std::abort(); // fail closed; incomplete capture is rejected by replay
    } catch (...) {
        std::abort();
    }
}
int main(int argc, char ** argv) try {
    require(argc == 7, "Usage: capture MODEL TOKENS BATCH ROUTES|- PREDICTIONS CONTEXT");
    const int batch_size = std::stoi(argv[3]);
    const int context = std::stoi(argv[6]);
    require(batch_size > 0 && batch_size <= 512 && context == 32768, "Invalid configuration");
    std::ifstream input(argv[2]);
    int n_prompt, n_decode;
    require(bool(input >> n_prompt >> n_decode) && n_prompt > 0 && n_decode > 0 &&
            n_prompt <= context && n_decode <= context - n_prompt, "Invalid input counts");
    std::vector<llama_token> tokens(n_prompt + n_decode);
    for (auto & token : tokens) require(bool(input >> token), "Missing input token");
    std::string extra;
    require(!(input >> extra), "Extra input data");
    const bool tracing = std::string(argv[4]) != "-";
    FILE * output = tracing ? std::fopen(argv[4], "wx") : nullptr;
    require(!tracing || output, "Cannot create route output");
    FILE * predictions = std::fopen(argv[5], "wx");
    require(predictions, "Cannot create prediction output");
    Capture c{output};
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    auto * model = llama_model_load_from_file(argv[1], mp);
    require(model, "Model load failed");
    const auto * vocab = llama_model_get_vocab(model);
    for (auto token : tokens) require(token >= 0 && token < llama_vocab_n_tokens(vocab), "Invalid token");
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = cp.n_ubatch = batch_size;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.swa_full = true;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    if (tracing) { cp.cb_eval = routes; cp.cb_eval_user_data = &c; }
    auto * ctx = llama_init_from_model(model, cp);
    require(ctx, "Context creation failed");
    for (int pos = 0; pos < int(tokens.size()); ) {
        c.phase = pos < n_prompt ? "prefill" : "decode";
        c.tokens = pos < n_prompt ? std::min(batch_size, n_prompt - pos) : 1;
        c.layer = 0;
        require(llama_decode(ctx, llama_batch_get_one(tokens.data() + pos, c.tokens)) == 0,
                "Decode failed");
        llama_synchronize(ctx);
        require(!tracing || c.layer == 30, "Missing routing layers");
        if (pos + c.tokens >= n_prompt) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            require(logits, "Missing logits");
            const auto best = std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits;
            std::fprintf(predictions, "%td\n", best);
        }
        pos += c.tokens;
        ++c.step;
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    require(std::fclose(predictions) == 0, "Predictions close failed");
    if (output) require(std::fclose(output) == 0, "Routes close failed");
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
