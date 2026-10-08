#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact prediction comparisons; explicit drift records never establish equivalence."""
from run import sha


def validate_prediction_records(path, expected_identities):
    """Bind predictions to the complete workload, not just another prediction file."""
    lines=iter(path.read_text().splitlines())
    for expected in expected_identities:
        line=next(lines,None)
        if line is None:raise ValueError('Truncated prediction records')
        fields=line.split()
        if len(fields)!=len(expected)+1 or any(not n.isascii() or not n.isdigit() for n in fields):
            raise ValueError('Invalid prediction record')
        if tuple(map(int,fields[:-1]))!=tuple(expected):
            raise ValueError('Prediction/workload identity mismatch')
    if next(lines,None) is not None:raise ValueError('Extra prediction records')


def validate_session_predictions(path, requests):
    validate_prediction_records(path,((r['request'],) for r in requests for _ in range(r['decode']+1)))


def compare_predictions(left, right):
    a=left.read_text().splitlines();b=right.read_text().splitlines()
    if not a or len(a)!=len(b):raise ValueError('Prediction record count mismatch')
    # Drift may change the final token ID, never request/step/sequence identity.
    for x,y in zip(a,b):
        xp=x.split();yp=y.split()
        if len(xp)<2 or len(xp)!=len(yp) or xp[:-1]!=yp[:-1]:
            raise ValueError('Prediction record identity mismatch')
        if any(not n.isdigit() for n in xp+yp):raise ValueError('Invalid prediction record')
    differences=sum(x!=y for x,y in zip(a,b))
    return dict(left_count=len(a),right_count=len(b),different_records=differences,
                equivalence='failed' if differences else 'exact_top1_match')


def capture_comparison(left, right, record_drift=False):
    comparison=compare_predictions(left,right)
    if comparison['different_records'] and not record_drift:
        raise ValueError('Control/capture prediction equivalence failed')
    return comparison


def validate_prediction_pair(control, traced, left, right, allow_drift=False):
    if sha(left)!=control['predictions_sha256'] or sha(right)!=traced['predictions_sha256']:
        raise ValueError('Prediction hash mismatch')
    comparison=compare_predictions(left,right)
    if 'prediction_comparison' in traced and traced['prediction_comparison']!=comparison:
        raise ValueError('Prediction comparison receipt mismatch')
    if comparison['different_records']:
        if not allow_drift or traced.get('prediction_drift_recorded') is not True or 'prediction_comparison' not in traced:
            raise ValueError('Control/capture prediction equivalence failed; explicit recorded drift permission required')
    elif traced.get('prediction_drift_recorded'):
        raise ValueError('Spurious prediction drift receipt')
    return comparison
