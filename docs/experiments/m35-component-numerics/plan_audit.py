# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Strict physical/semantic quant dtype reconciliation for retained plans."""

DENSE_QUANT = {
    'q4_K': 'q4_k',
    'q5_K': 'q5_k',
    'q6_K': 'q6_k',
}


def dense_quant_encoding(source):
    """Return a known semantic encoding only when physical and semantic fields agree.

    The physical names are case-sensitive (`q4_K`) while the IR encodings use
    lowercase (`q4_k`). Unknown source types are ignored; a quantized physical
    or semantic claim with a missing or inconsistent counterpart is rejected.
    """
    if source is None:  # A physical node may have an unused optional source.
        return None
    physical = source.get('dtype')
    ir_format = source.get('ir_format')
    semantic = ir_format.get('encoding') if isinstance(ir_format, dict) else None
    is_physical_quant = physical in DENSE_QUANT
    is_semantic_quant = semantic in DENSE_QUANT.values()
    if not is_physical_quant and not is_semantic_quant:
        return None
    if not is_physical_quant or not is_semantic_quant:
        raise ValueError(f'physical/semantic dense quant dtype is incomplete: {physical!r}/{semantic!r}')
    if semantic != DENSE_QUANT[physical]:
        raise ValueError(f'physical/semantic dense quant dtype conflicts: {physical!r}/{semantic!r}')
    if source.get('shape') != ir_format.get('shape'):
        raise ValueError('physical and semantic dense quant shapes differ')
    if ir_format.get('layout') != 'ggml_blocks':
        raise ValueError(f'unsupported dense quant layout: {ir_format.get("layout")!r}')
    return semantic


def ordinary_decode_segments(plan):
    """Return explicit or source-inferred C2 evidence; never fabricate a header.

    Inference requires the frozen target-only, tool-free ordinary decode callers:
    DecodingOrchestrator::Single emits one row per work item, prefill peers
    contribute one row, and CheckShape enforces distinct slots. The outer
    preflight binds that complete source chain and the fixed request/config.
    Pure phase-1 rows=2 therefore means two one-row segments, even for transient
    completions whose debug report intentionally lacks a segment-count field.
    """
    h = plan['header']
    if h.get('stage') != 0 or h.get('phase') != 1 or h.get('rows') != 2:
        return None
    completions = plan.get('completions', [])
    if not completions or any(c.get('stage') != 0 or c.get('phase') != 1 or
                              c.get('rows') != 2 for c in completions):
        raise ValueError('C2 decode lacks matching positive completion evidence')
    if 'segments' in h:
        if h['segments'] != 2:
            raise ValueError('explicit segment count conflicts with ordinary C2 decode')
        return {'segments': 2, 'evidence': 'explicit installed-plan header'}
    if h.get('transient') is not True:
        raise ValueError('missing segment count outside a transient completion')
    return {'segments': 2,
            'evidence': 'inferred from pinned ordinary decode callers; explicit header absent'}


def dense_product_recipe(name):
    return name.startswith(('llmp.vecq', 'ggml.mul_mat.mmvq', 'llmp.mmvq',
                            'llmp.mul_mat.mmvq_rows'))
