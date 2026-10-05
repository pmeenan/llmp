#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2023-2026 The ggml authors
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Bounded external-reference controller diagnostic at the exact b29 pin.

Original CUDA math launchers remain in the digest-pinned image. Trace
selection without requesting graph intermediates; require full-head identity.
"""
import hashlib
import pathlib
import sys

PINNED_SHA = '523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634'

HELPER = r'''
// External jitLLM diagnosis only; no changes to model graph tensors.
// 0 preserves all original selections, 1/2 select one norm family, 3 none, 4 both.
static int jitllm_reference_policy() {
    static const int policy = [] {
        const char * p = getenv("JITLLM_REFERENCE_FUSION_POLICY");
        if (p == nullptr || strcmp(p, "all") == 0) return 0;
        if (strcmp(p, "norm_rope") == 0) return 1;
        if (strcmp(p, "norm_add") == 0) return 2;
        if (strcmp(p, "none") == 0) return 3;
        if (strcmp(p, "both") == 0) return 4;
        GGML_ABORT("unknown external reference fusion policy");
    }();
    return policy;
}
'''

ALLOW = r'''
    static const bool announced = [] {
        fprintf(stderr, "JITLLM_REF_CONTROLLER policy=%d\n", jitllm_reference_policy());
        return true;
    }();
    (void) announced;
    const int diagnostic_policy = jitllm_reference_policy();
    if (diagnostic_policy == 1 || diagnostic_policy == 4) {
        if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
            ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, cgraph->nodes[i], cgraph->nodes[i+1], cgraph->nodes[i+2], cgraph->nodes[i+4]);
            return 4;
        }
        if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
            ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, cgraph->nodes[i], cgraph->nodes[i+1], cgraph->nodes[i+2], nullptr);
            return 2;
        }
        if (diagnostic_policy == 1) return 0;
    }
    if (diagnostic_policy == 2 || diagnostic_policy == 4) {
        if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
            ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, cgraph->nodes[i], cgraph->nodes[i+1], cgraph->nodes[i+2]);
            return 2;
        }
        return 0;
    }
    if (diagnostic_policy == 3) return 0;
'''

TRACE = r'''
                static const bool trace = getenv("JITLLM_REFERENCE_TRACE") != nullptr;
                if (trace && nodes_to_skip > 0) {
                    fprintf(stderr, "JITLLM_REF_FUSION count=%d", nodes_to_skip + 1);
                    for (int f = i; f <= i + nodes_to_skip; ++f) {
                        const auto * t = cgraph->nodes[f];
                        fprintf(stderr, " | %s:%s:%s:[%lld,%lld,%lld,%lld]",
                                t->name, ggml_op_name(t->op), ggml_type_name(t->type),
                                (long long)t->ne[0], (long long)t->ne[1],
                                (long long)t->ne[2], (long long)t->ne[3]);
                    }
                    fprintf(stderr, "\n");
                }
'''


def main():
    assert len(sys.argv) == 3
    source, destination = map(pathlib.Path, sys.argv[1:])
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINNED_SHA, 'wrong upstream source identity'
    assert not destination.exists(), 'diagnostic destination must be new'
    text = raw.decode()
    anchor = '// try and fuse nodes and return the number of nodes to skip\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, HELPER + '\n' + anchor)
    anchor = '    ggml_tensor * node = cgraph->nodes[i];\n'
    start = text.index('static int ggml_cuda_try_fuse(')
    end = text.index('static void ggml_cuda_graph_evaluate_and_capture(', start)
    function = text[start:end]
    assert function.count(anchor) == 1
    text = text[:start] + function.replace(anchor, ALLOW + '\n' + anchor) + text[end:]
    anchor = '                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + TRACE)
    # Allocator dependencies for the original weighted reduction apply only
    # when its original fusion policy remains selectable. Leave everything
    # else (including graph tensors, streams and capture) unchanged.
    start = text.index('static void ggml_backend_cuda_graph_optimize(')
    anchor = '    if (!disable_fusion) {'
    offset = text.index(anchor, start)
    text = text[:offset] + text[offset:].replace(
        anchor, '    if (!disable_fusion && jitllm_reference_policy() == 0) {', 1)
    destination.write_text(text)
    print('original_sha256=' + PINNED_SHA)
    print('diagnostic_sha256=' + hashlib.sha256(destination.read_bytes()).hexdigest())
    return 0


if __name__ == '__main__':
    sys.exit(main())
