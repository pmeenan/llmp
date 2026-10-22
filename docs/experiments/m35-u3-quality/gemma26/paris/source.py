#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bind the new frozen request to unchanged, authenticated Gemma reference checks."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path

import control


def bind():
    raw = control.raw_reference()
    # Only request data changes. The old file on disk, implementations, model
    # arithmetic, backend checks and all five-ID result directories stay intact.
    raw.IDS = list(control.inputs()['prompt_token_ids'])
    return raw


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate', 'run'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--registration', required=True)
    args = parser.parse_args()
    registration = control.registration(args.registration)
    raw = bind()
    if args.mode == 'validate':
        print(control.encoded(raw.validate()).decode(), end='')
        return
    control.require(args.out is not None and args.out.is_dir(), 'private mounted output required')
    request = dict(prompt_token_ids=list(raw.IDS), sampling=raw.sampling_kwargs(),
                   tokenization_kwargs={'add_special_tokens': False})
    raw.write_private(args.out / 'request.json', control.encoded(request))
    # The existing implementation validates the real parser/callback APIs before
    # model construction, checks live dispatch before generation, and saves the
    # complete returned objects before inspecting any output fields.
    try:
        raw.run(args.out)
    finally:
        result_path = args.out / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_bytes())
            result.update(control='gemma26-source-template-paris64',
                          registration_commit=registration, input_sha256=control.INPUT_SHA,
                          raw_reference_sha256=control.RAW_REFERENCE_SHA)
            if 'output' in result:
                output = result['output']
                result['semantic_verdict'] = control.verdict(output['text'], output['finish_reason'])
            raw.write_private(result_path, control.encoded(result))


if __name__ == '__main__':
    main()
