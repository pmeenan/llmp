#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2023-2026 The ggml authors
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Removable copy-only instrumentation over the authenticated original observer."""
import hashlib
import pathlib
import re
import runpy
import sys

observer = runpy.run_path(str(pathlib.Path(__file__).resolve().parent.parent /
                               'gemma26-dispatch-observation/observe_controller.py'))
BEGIN = '\n// JITLLM_LATE_BEGIN\n'
END = '// JITLLM_LATE_END\n'


def insert(text, anchor, code, after=False):
    if text.count(anchor) != 1:
        raise ValueError('unexpected capture anchor: ' + anchor[:100])
    marked = BEGIN + code + '\n' + END
    return text.replace(anchor, anchor + marked if after else marked + anchor)


def original(text):
    stripped = re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END), '', text, flags=re.S)
    return observer['original'](stripped)


def transform(raw):
    text = observer['transform'](raw).decode()
    text = insert(text, 'static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,',
                  '#include "capture.h"')
    text = insert(text,
        '    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);',
        '    jitllm_late_graph(cgraph, use_cuda_graph, cuda_graph_update_required);')
    route = '''                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&'''
    text = insert(text, route, '''                jitllm_late_route(*cuda_ctx, node, logits, weights, ids,
                    ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2),
                    ggml_cuda_should_use_topk_moe(node, logits, weights, ids),
                    ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, true));''')
    call = '''                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);'''
    text = insert(text, call, '''                const int jitllm_capture_layer = jitllm_late_reduction_layer(node);
                if (jitllm_capture_layer >= 0)
                    jitllm_late_reduction_before(*cuda_ctx, jitllm_capture_layer,
                        match.experts, match.expert_scale, match.weights, match.dst);''')
    text = insert(text, call, '''                if (jitllm_capture_layer >= 0)
                    jitllm_late_reduction_after(*cuda_ctx, jitllm_capture_layer, match.dst);''', after=True)
    text = insert(text, '                GGML_ASSERT(ok);',
                  '                jitllm_late_after_node(*cuda_ctx, node);', after=True)
    if original(text).encode() != raw:
        raise ValueError('capture changed original source bytes')
    return text.encode()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('capture_controller.py EXACT_ORIGINAL NEW_CONTROLLER')
    raw = pathlib.Path(sys.argv[1]).read_bytes()
    output = transform(raw)
    with pathlib.Path(sys.argv[2]).open('xb') as stream:
        stream.write(output)
    print('original_sha256=' + hashlib.sha256(raw).hexdigest())
    print('capture_sha256=' + hashlib.sha256(output).hexdigest())
    print('reconstructed_sha256=' + hashlib.sha256(original(output.decode()).encode()).hexdigest())
