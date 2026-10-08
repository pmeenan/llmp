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
    int request = 0;
    int layers = 0;
    int experts = 0;
    int topk = 0;
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
        const bool output_only=c.layer==c.layers-1 && std::strcmp(c.phase,"prefill")==0 &&
                               c.tokens>1 && t->ne[1]==1;
        require(expected == t->name && t->type == GGML_TYPE_I32 &&
                t->ne[0] == c.topk && (t->ne[1] == c.tokens || output_only) &&
                t->ne[2] == 1 && t->ne[3] == 1, "Unexpected routing tensor/order");
        std::vector<unsigned char> bytes(ggml_nbytes(t));
        ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        std::fprintf(c.out, "{\"event\":\"routes\",\"request\":%d,\"step\":%d,\"phase\":\"%s\",\"layer\":%d,\"output_only\":%s,\"routes\":[",
                     c.request, c.step, c.phase, c.layer, output_only?"true":"false");
        for (int i = 0; i < t->ne[1]; ++i) {
            std::fprintf(c.out, "%s[", i ? "," : "");
            std::vector<bool> seen(c.experts);
            for (int k = 0; k < c.topk; ++k) {
                int32_t id;
                const size_t off = i * t->nb[1] + k * t->nb[0];
                require(off + sizeof(id) <= bytes.size(), "Invalid routing stride");
                std::memcpy(&id, bytes.data() + off, sizeof(id));
                require(id >= 0 && id < c.experts && !seen[id], "Invalid/duplicate expert");
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
    require(argc == 13, "Usage: session_capture MODEL INPUT EVENTS PRED LAYERS EXPERTS TOPK CTX BATCH TRACE REUSE SWA_FULL");
    const int layers=std::stoi(argv[5]), experts=std::stoi(argv[6]), topk=std::stoi(argv[7]);
    const int context=std::stoi(argv[8]), batch_size=std::stoi(argv[9]);
    const bool tracing=std::stoi(argv[10]) != 0, reuse=std::stoi(argv[11]) != 0;
    require(layers>0 && layers<=128 && experts>0 && experts<=1024 && topk>0 && topk<=experts &&
            context>0 && context<=65536 && batch_size>0 && batch_size<=512, "Invalid bounds");
    std::ifstream input(argv[2]);
    int requests;
    require(bool(input >> requests) && requests>0 && requests<=100, "Invalid request count");
    FILE * output=std::fopen(argv[3],"wx"), * predictions=std::fopen(argv[4],"wx");
    require(output && predictions,"Cannot create exclusive outputs");
    Capture c{output}; c.layers=layers; c.experts=experts; c.topk=topk;
    ggml_backend_load_all(); llama_backend_init();
    auto mp=llama_model_default_params(); mp.n_gpu_layers=999; mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
    auto * model=llama_model_load_from_file(argv[1],mp); require(model,"Model load failed");
    auto cp=llama_context_default_params();
    cp.n_ctx=context; cp.n_batch=cp.n_ubatch=batch_size; cp.n_threads=cp.n_threads_batch=8;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.swa_full=std::stoi(argv[12]) != 0; cp.type_k=cp.type_v=GGML_TYPE_F16;
    if(tracing) {cp.cb_eval=routes; cp.cb_eval_user_data=&c;}
    auto * ctx=llama_init_from_model(model,cp); require(ctx,"Context creation failed");
    const auto * vocab=llama_model_get_vocab(model);
    std::vector<llama_token> history;
    for(int req=0; req<requests; ++req) {
        c.request=req; c.step=0;
        int n_prompt,n_decode;
        require(bool(input>>n_prompt>>n_decode) && n_prompt>0 && n_decode>=0 &&
                n_prompt<=context && n_decode<=context-n_prompt,"Invalid token counts");
        std::vector<llama_token> tokens(n_prompt+n_decode);
        for(auto & t:tokens) require(bool(input>>t) && t>=0 && t<llama_vocab_n_tokens(vocab),"Invalid token");
        int common=0;
        if(reuse) while(common<int(history.size()) && common<n_prompt && history[common]==tokens[common]) ++common;
        // Always evaluate the last prompt token so first-output logits exist.
        common=std::min(common,n_prompt-1);
        const int requested_common=common;
        const int n_swa=llama_model_n_swa(model);
        const int memory_min=llama_memory_seq_pos_min(llama_get_memory(ctx),0);
        // Sequence snapshots may omit old SWA cells even with full-SWA allocation.
        // No older checkpoint exists in this harness: missing coverage means reset.
        const int coverage_threshold=std::max(0,common-n_swa);
        const bool coverage_reset=common>0 && n_swa>0 && memory_min>0 && memory_min>=coverage_threshold;
        if(coverage_reset) {
            common=0; llama_memory_clear(llama_get_memory(ctx),true);
        }
        if(common<int(history.size()) && !llama_memory_seq_rm(llama_get_memory(ctx),0,common,-1)) {
            common=0; llama_memory_clear(llama_get_memory(ctx),true);
        }
        std::fprintf(output,"{\"event\":\"request\",\"request\":%d,\"prompt\":%d,\"decode\":%d,\"reused\":%d,\"requested_reuse\":%d,\"memory_min\":%d,\"swa_window\":%d,\"coverage_reset\":%s}\n",req,n_prompt,n_decode,common,requested_common,memory_min,n_swa,coverage_reset?"true":"false");
        for(int pos=common; pos<int(tokens.size());) {
            c.phase=pos<n_prompt?"prefill":"decode";
            c.tokens=pos<n_prompt?std::min(batch_size,n_prompt-pos):1; c.layer=0;
            const auto start=ggml_time_us();
            require(llama_decode(ctx,llama_batch_get_one(tokens.data()+pos,c.tokens))==0,"Decode failed");
            llama_synchronize(ctx);
            const auto elapsed=ggml_time_us()-start;
            require(!tracing || c.layer==layers,"Missing routing layers");
            std::fprintf(output,"{\"event\":\"step\",\"request\":%d,\"step\":%d,\"phase\":\"%s\",\"tokens\":%d,\"us\":%lld}\n",req,c.step,c.phase,c.tokens,(long long)elapsed);
            if(pos+c.tokens>=n_prompt) {
                const auto * logits=llama_get_logits_ith(ctx,-1); require(logits,"Missing logits");
                const auto best=std::max_element(logits,logits+llama_vocab_n_tokens(vocab))-logits;
                std::fprintf(predictions,"%d %td\n",req,best);
            }
            pos+=c.tokens; ++c.step;
        }
        history=std::move(tokens);
        std::fprintf(output,"{\"event\":\"end\",\"request\":%d,\"state_bytes\":%zu,\"history\":%zu}\n",req,llama_state_seq_get_size(ctx,0),history.size());
        require(std::fflush(output)==0 && !std::ferror(predictions),"Output failed");
    }
    std::string extra; require(!(input>>extra),"Unexpected trailing input");
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    require(std::fclose(predictions)==0 && std::fclose(output)==0,"Close failed");
    return 0;
} catch(const std::exception & e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
