#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""One ordinary source prompt-logprob request; no custom forward or head adapter."""
import sys
sys.dont_write_bytecode = True

import argparse
import os
from pathlib import Path
import control


def sampling_kwargs(raw):
    return dict(raw.sampling_kwargs(), max_tokens=1, prompt_logprobs=1)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--registration', required=True)
    args = parser.parse_args()
    registration = control.registration(args.registration)
    ids, raw = control.inputs(), control.raw_reference()
    control.require(args.out.is_dir(), 'private mounted output required')
    control.require(os.environ.get('VLLM_USE_V2_MODEL_RUNNER') == '1', 'runner selection changed')
    source_hashes = control.source_check()
    # Parent validation authenticates engine/API/callback/dispatch prerequisites;
    # its old request data are not used for this new ordinary scoring request.
    api = raw.validate()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    options = sampling_kwargs(raw)
    sampling = SamplingParams(**options)
    control.require(sampling.max_tokens == 1 and sampling.prompt_logprobs == 1 and
                    sampling.logprobs == 1 and sampling.temperature == 0 and
                    sampling.skip_reading_prefix_cache is True,
                    'actual scoring sampling options differ')
    request = dict(prompt_token_ids=ids, sampling=options,
                   actual_sampling_repr=repr(sampling), tokenization_kwargs={'add_special_tokens': False})
    raw.write_private(args.out / 'request.json', control.encoded(request))
    record = dict(format='llmp-gemma26-short-source/1', registration_commit=registration,
                  input_sha256=control.INPUT_SHA, image=raw.IMAGE, api=api,
                  source_hashes=source_hashes, checkpoint=raw.checkpoint_check(Path('/model')),
                  native_kv_dtype='float16', reference_dtype='bfloat16',
                  reference_kv_dtype='bfloat16', runner_env='1', status='starting')
    save = lambda: raw.write_private(args.out / 'result.json', control.encoded(record))
    save()
    try:
        llm = LLM(**raw.engine_kwargs())
        dispatch = llm.apply_model(raw.inspect_model)
        record['dispatch'] = dispatch
        save()
        control.require(len(dispatch) == 1, 'expected one model worker')
        raw.check_dispatch(dispatch[0])
        outputs = llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling,
                               use_tqdm=False, tokenization_kwargs={'add_special_tokens': False})
        raw.write_private(args.out / 'returned.txt', (repr(outputs) + '\n').encode())
        rows, unused = control.source_output(outputs)
        record.update(control.write_losses(args.out, rows))
        # Return host ownership even though the container writes private output.
        owner = args.out.stat()
        os.chown(args.out / 'losses.jsonl', owner.st_uid, owner.st_gid)
        record.update(unused_generated_output=unused,
                      actual_prompt_token_ids=list(outputs[0].prompt_token_ids),
                      status='collected-not-quality-qualified')
        save()
    except BaseException as error:
        record.update(status='failed', failure=f'{type(error).__name__}: {error}')
        save()
        raise
    finally:
        record['source_hashes_after'] = control.source_check()
        save()
        control.require(record['source_hashes_after'] == source_hashes, 'installed source changed')


if __name__ == '__main__':
    main()
