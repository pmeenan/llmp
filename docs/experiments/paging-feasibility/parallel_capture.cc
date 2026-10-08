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
    require(argc==10,"Usage: parallel_capture MODEL INPUT EVENTS PRED LAYERS EXPERTS TOPK CTX TRACE");
    std::ifstream input(argv[2]);int sequences;require(bool(input>>sequences) && sequences==4,"Need four requests");
    const int context=std::stoi(argv[8]); require(context>=4096 && context<=131072,"Context bound");
    std::vector<std::vector<llama_token>> prompts(4),outputs(4);
    for(int i=0;i<4;++i) {
        int n,d;require(bool(input>>n>>d) && n>0 && d>0 && n<=context/4 && d<=context/4-n,"Input bounds");
        prompts[i].resize(n);outputs[i].resize(d);
        for(auto & t:prompts[i]) require(bool(input>>t),"Missing prompt token");
        for(auto & t:outputs[i]) require(bool(input>>t),"Missing output token");
    }
    std::string extra;require(!(input>>extra),"Extra input");
    FILE * out=std::fopen(argv[3],"wx"),*pred=std::fopen(argv[4],"wx");require(out && pred,"Exclusive outputs required");
    Capture c{out};c.layers=std::stoi(argv[5]);c.experts=std::stoi(argv[6]);c.topk=std::stoi(argv[7]);
    require(c.layers>0&&c.layers<=128&&c.experts>0&&c.experts<=1024&&c.topk>0&&c.topk<=c.experts,"Model bounds");
    ggml_backend_load_all();llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=999;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
    auto * model=llama_model_load_from_file(argv[1],mp);require(model,"Model load failed");
    auto cp=llama_context_default_params();cp.n_ctx=context;cp.n_batch=cp.n_ubatch=512;cp.n_seq_max=4;
    cp.n_threads=cp.n_threads_batch=8;cp.swa_full=true;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.type_k=cp.type_v=GGML_TYPE_F16;
    const bool tracing=std::stoi(argv[9])!=0;if(tracing){cp.cb_eval=routes;cp.cb_eval_user_data=&c;}
    auto * ctx=llama_init_from_model(model,cp);require(ctx,"Context creation failed");
    auto batch=llama_batch_init(512,0,1);
    auto evaluate=[&]() {
        c.tokens=batch.n_tokens;c.layer=0;
        require(llama_decode(ctx,batch)==0,"Decode failed");llama_synchronize(ctx);
        require(!tracing || c.layer==c.layers,"Missing layer");
        std::fprintf(out,"{\"event\":\"batch\",\"step\":%d,\"phase\":\"%s\",\"tokens\":%d}\n",c.step,c.phase,c.tokens);
        ++c.step;
    };
    auto add=[&](int token,int position,int seq,bool logits) {
        const int n=batch.n_tokens++;
        require(token>=0&&token<llama_vocab_n_tokens(llama_model_get_vocab(model)),"Invalid token");
        batch.token[n]=token;batch.pos[n]=position;batch.n_seq_id[n]=1;batch.seq_id[n][0]=seq;batch.logits[n]=logits;
    };
    c.phase="prefill";
    for(int seq=0;seq<4;++seq) {
        for(int pos=0;pos<int(prompts[seq].size());) {
            batch.n_tokens=0;
            for(int j=0;j<512&&pos<int(prompts[seq].size());++j,++pos) add(prompts[seq][pos],pos,seq,pos+1==int(prompts[seq].size()));
            evaluate();
        }
    }
    c.phase="decode";
    const int rounds=int(std::min({outputs[0].size(),outputs[1].size(),outputs[2].size(),outputs[3].size()}));
    for(int t=0;t<rounds;++t) {
        batch.n_tokens=0;
        for(int seq=0;seq<4;++seq)add(outputs[seq][t],int(prompts[seq].size())+t,seq,true);
        evaluate();
        for(int seq=0;seq<4;++seq) {
            auto * logits=llama_get_logits_ith(ctx,seq);require(logits,"Missing logits");
            auto best=std::max_element(logits,logits+llama_vocab_n_tokens(llama_model_get_vocab(model)))-logits;
            std::fprintf(pred,"%d %d %td\n",t,seq,best);
        }
    }
    std::fprintf(out,"{\"event\":\"end\",\"decode_batches\":%d,\"batch_size\":4}\n",rounds);
    llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
    require(std::fclose(out)==0&&std::fclose(pred)==0,"Output failure");return 0;
} catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
