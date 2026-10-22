#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Snapshot the short-score collector; reuse the reviewed Gemma container lifecycle."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import shutil
import uuid

import control


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--staging-evidence', type=Path, required=True)
    parser.add_argument('--registration', required=True)
    args = parser.parse_args()
    registration = control.registration(args.registration)
    control.inputs()
    raw, lifecycle = control.raw_reference(), control.raw_launcher()
    source = raw.checkpoint_check(args.checkpoint)
    control.checked(args.checkpoint / 'chat_template.jinja', control.TEMPLATE_SHA)
    control.checked(args.staging_evidence, control.STAGE_SHA)
    pin = args.repo / 'docs/experiments/fast-swap/mia-launch.json'
    control.checked(pin, control.LOADER_SHA)
    pins = json.loads(pin.read_bytes())
    control.require(pins['image'] == raw.IMAGE and pins['loader_version'] == '0.2.0', 'loader pin changed')
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    out = args.out.resolve()
    snapshot = out / 'helpers/u3quality'
    helpers = snapshot / 'gemma26'
    (helpers / 'short').mkdir(parents=True)
    here = Path(__file__).resolve().parent
    sources = [here / name for name in ('source_launch.py', 'source.py', 'control.py', 'ids.json',
                                       'source-hashes.json', 'pair-prereg.json', 'PROTOCOL.md')]
    sources += [here.parent / name for name in ('reference.py', 'launch.py', 'paris/control.py')]
    sources += [here.parent.parent / 'qwen27/contract.py']
    before = {str(path): control.sha(path) for path in sources}
    copies = {}
    for path in sources:
        dest = snapshot / path.relative_to(here.parent.parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        control.checked(dest, before[str(path)])
        copies[str(dest)] = before[str(path)]
    loader = out / 'loader'
    installed = Path.home() / '.local/share/mia-load-study/pylib'
    for relative, digest in pins['loader_files'].items():
        control.checked(installed / relative, digest)
        dest = loader / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(installed / relative, dest)
        control.checked(dest, digest)
    shutil.copyfile(args.staging_evidence, out / 'staging-evidence')
    shutil.copyfile(args.checkpoint / 'llmp-archive.json', out / 'source-archive.json')
    cache = Path.home() / '.cache/vllm/triton-u3-gemma26'
    cache.mkdir(parents=True, exist_ok=True)
    name = 'llmp-gemma26-short-' + uuid.uuid4().hex
    create = lifecycle.DOCKER + [
        'create', '--name', name, '--device', 'nvidia.com/gpu=all',
        '--network', 'none', '--ipc', 'host', '--entrypoint', 'python3',
        '-v', f'{args.checkpoint.resolve()}:/model:ro', '-v', f'{out / "helpers"}:/helpers:ro',
        '-v', f'{loader}:/opt/instanttensor:ro', '-v', f'{cache}:/triton', '-v', f'{out}:/output',
        '-e', 'PYTHONPATH=/opt/instanttensor:/helpers/u3quality/gemma26',
        '-e', 'TRITON_CACHE_DIR=/triton', '-e', 'VLLM_USE_V2_MODEL_RUNNER=1',
        '-e', f'{raw.SERIALIZATION_ENV}=1', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
        raw.IMAGE, '/helpers/u3quality/gemma26/short/source.py', 'run', '--out', '/output',
        '--registration', registration,
    ]
    receipt = dict(format='llmp-gemma26-short-reference-launch/1', argv=create, image=raw.IMAGE,
                   registration_commit=registration, input_sha256=control.INPUT_SHA, source=source,
                   protocol_sha256=control.sha(here / 'PROTOCOL.md'),
                   staging_evidence_sha256=control.STAGE_SHA, loader_pin_sha256=control.LOADER_SHA,
                   loader_files=pins['loader_files'], helper_hashes_before=before,
                   helper_snapshot_hashes=copies, collection_completed=False)
    (out / 'launch.json').write_bytes(control.encoded(receipt))
    try:
        lifecycle.execute(create, name, out, receipt)
    finally:
        receipt['helper_hashes_after'] = {str(path): control.sha(path) for path in sources}
        receipt['helper_snapshot_hashes_after'] = {path: control.sha(path) for path in copies}
        receipt['loader_hashes_after'] = {rel: control.sha(loader / rel) for rel in pins['loader_files']}
        receipt['source_after'] = raw.checkpoint_check(args.checkpoint)
        receipt['template_sha256_after'] = control.checked(args.checkpoint / 'chat_template.jinja',
                                                         control.TEMPLATE_SHA)
        (out / 'launch.json').write_bytes(control.encoded(receipt))
        control.require(receipt['helper_hashes_after'] == before and
                        receipt['helper_snapshot_hashes_after'] == copies and
                        receipt['loader_hashes_after'] == pins['loader_files'] and
                        receipt['source_after'] == source, 'helper/loader/source changed')


if __name__ == '__main__':
    main()
