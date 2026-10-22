#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned-image Gemma raw-ID diagnostic; validate never constructs a model."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

IMAGE = 'sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8'
IDS = [818, 5279, 529, 7001, 563]
STOP_IDS = [1, 106, 50]
SOURCE = 'nvidia/Gemma-4-26B-A4B-NVFP4'
REVISION = 'a19cfe00be84568a6867111c9a68c9c44fdcffe6'
SERIALIZATION_ENV = 'VLLM_ALLOW_INSECURE_SERIALIZATION'
SOURCE_HASHES = {
    'envs.py': 'fb701e932ff263612d520c1447dfdbd882da3286c889c4e01735c8456c3db508',
    'v1/serial_utils.py': 'd75c17a098231b2398cbdb7da6f9983e5b9229c0ba138b2874934d45fafd1f28',
    'v1/engine/llm_engine.py': '415b7d02460984e8678f78420be22dbc336d8bb73734d7a753b2badb62232744',
    'v1/worker/worker_base.py': 'b0684c84f2f2c4bff2d7f0574e13f48c01b4d76dbf555e7c52b32e9a702a9061',
    'model_executor/models/gemma4.py': '6074e4a2f0c40a5bb28d101060fb4fe64f11ac6ee0cb6f524a9bb3deaceab797',
    'model_executor/layers/quantization/modelopt.py': '3f3ca743fd3c66d72be92b7544591a8632f1aa422b73bb50c83e8a3281196e7d',
    'inputs/llm.py': 'd7bf21f71d089709b5566c9232ce20bf3da7029228ca6256b5e1e61057fba5d2',
    'inputs/preprocess.py': '8e2fe8558064319af1a10ebc5db12b7b8e76ddd78e2c10322dff9c2118c583bc',
    'renderers/inputs/preprocess.py': 'd3ec875c43f81208444c111da53bea5854bb9ff0cc0c793eeec6c693a4bdd25c',
    'renderers/base.py': 'a358921359b241df2e5c08b547897b874a0eb5c73bc9051293bab800511aeedb',
    'renderers/params.py': 'b9667d21614cc474e881124b44762db6a1d43efe21b9ec0350fc5c80229bf67d',
    'engine/arg_utils.py': '75636c2a7903738f0e8954394fcedfced318d30a41598a8aca533be2ea2c38c4',
    'entrypoints/llm.py': '52de4ac99489e004ef6c61d0bedc84aa96020dd58b8bd1ae500814b548b2b83e',
}
METADATA_HASHES = {
    'llmp-archive.json': '4f36e82e02c29b3e99530efa38495459667a4c461a4d03bc9f610664db162ca1',
    'config.json': '4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98',
    'hf_quant_config.json': 'fca2ea21cded31e6cff2c56ee83a162dcd5d3ff292cbf2f6083702e9fc324454',
    'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
    'tokenizer_config.json': '90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df',
    'generation_config.json': 'd4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de',
    'model.safetensors.index.json': 'ac5e677ed9f8d9b689170bbae0c88ce163e284d02b09100e14afddaa9ec4a15c',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def write_private(path, data):
    path.write_bytes(data)
    # The image runs as root; return its private receipts to the host-owned mount.
    owner = path.parent.stat()
    os.chown(path, owner.st_uid, owner.st_gid)


def engine_kwargs():
    # No checkpoint code, source rewrites, inherited sampling or precision fallback.
    return dict(model='/model', tokenizer='/model', trust_remote_code=False,
                hf_overrides={'architectures': ['Gemma4ForCausalLM']},
                runner='generate', dtype='bfloat16', kv_cache_dtype='bfloat16',
                quantization='modelopt_fp4', load_format='instanttensor',
                generation_config='vllm', gpu_memory_utilization=0.65,
                max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=512,
                enable_chunked_prefill=True, enable_prefix_caching=False,
                max_logprobs=1, logprobs_mode='raw_logprobs', seed=0,
                tensor_parallel_size=1, pipeline_parallel_size=1,
                speculative_config=None, cpu_offload_gb=0)


def sampling_kwargs():
    return dict(n=1, temperature=0, top_p=1, top_k=-1, max_tokens=64,
                min_tokens=0, repetition_penalty=1, presence_penalty=0,
                frequency_penalty=0, stop_token_ids=list(STOP_IDS),
                ignore_eos=False, logprobs=1, seed=0)


def source_check():
    root = Path(importlib.util.find_spec('vllm').origin).parent
    actual = {name: sha(root / name) for name in SOURCE_HASHES}
    require(actual == SOURCE_HASHES, 'installed reference source changed')
    return actual


def validate_callback_transport(*, enabled=True):
    """Round-trip only our own callback through the pinned official RPC codec."""
    from types import SimpleNamespace
    from vllm.v1.serial_utils import MsgpackEncoder, MsgpackDecoder, UtilityResult
    from vllm.v1.worker.worker_base import WorkerBase

    require(os.environ.get(SERIALIZATION_ENV, '0') == ('1' if enabled else '0'),
            'callback serialization environment differs from the selected validation')
    # These are the collective_rpc arguments produced by LLMEngine.apply_model.
    request = ('apply_model', None, (inspect_model,), None)
    if not enabled:
        try:
            MsgpackEncoder().encode(request)
        except TypeError as error:
            require('function' in str(error) and SERIALIZATION_ENV in str(error),
                    'unexpected default-off codec failure')
            return dict(enabled=False, callback_rejected=True, failure=str(error))
        raise ValueError('default-off codec unexpectedly accepted the callback')

    buffers = MsgpackEncoder().encode(request)
    decoded = MsgpackDecoder().decode(buffers)
    require(len(decoded) == 4 and decoded[0] == 'apply_model' and
            decoded[1] is None and decoded[3] is None and len(decoded[2]) == 1,
            'callback RPC envelope changed')
    restored = decoded[2][0]
    require(callable(restored) and restored.__name__ == inspect_model.__name__ and
            restored.__code__ == inspect_model.__code__, 'callback code changed in transport')

    class FixtureModel:
        quant_config = SimpleNamespace(quant_method='NVFP4',
                                       LinearMethodCls=type('ModelOptNvFp4LinearMethod', (), {}))

        def named_modules(self):
            # No tensors, weights, model constructor or device calls. These objects
            # exercise both quantized-kernel and attention provenance branches.
            yield 'expert', SimpleNamespace(quant_method=SimpleNamespace(
                use_a16=False, nvfp4_backend='FLASHINFER_CUTLASS', moe_kernel=object()))
            yield 'attention', SimpleNamespace(kv_cache_dtype='bfloat16',
                get_attn_backend=lambda: type('TritonAttentionBackend', (), {}))

    fixture = FixtureModel()
    expected = inspect_model(fixture)
    actual = WorkerBase.apply_model(SimpleNamespace(get_model=lambda: fixture), restored)
    require(actual == expected, 'transported callback changed inspected metadata')
    result_buffers = MsgpackEncoder().encode(UtilityResult([actual]))
    result = MsgpackDecoder(UtilityResult).decode(result_buffers).result
    require(result == [expected], 'callback utility result changed in transport')
    return dict(enabled=True, callback='inspect_model', callback_code_equal=True,
                official_worker_apply_passed=True, utility_result_equal=True,
                encoded_bytes=sum(len(buf) for buf in buffers), model_constructed=False)


def validate():
    """Exercise the real official parser and renderer with a tokenizer tripwire."""
    sources = source_check()
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.renderers.base import BaseRenderer
    from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
    from vllm.renderers.params import TokenizeParams

    class Tripwire:
        def __getattr__(self, name):
            raise RuntimeError(f'raw IDs unexpectedly touched tokenizer: {name}')

    class RendererFixture:
        tokenizer = Tripwire()

        def _tokenize_prompt(self, *args):
            raise RuntimeError('raw IDs unexpectedly invoked text tokenization')

        def _detokenize_prompt(self, *args):
            raise RuntimeError('raw IDs unexpectedly invoked detokenization')

    params = TokenizeParams(max_total_tokens=4096, max_output_tokens=64,
                            add_special_tokens=False)
    prompt = parse_dec_only_prompt(TokensPrompt(prompt_token_ids=list(IDS)))
    rendered = BaseRenderer._tokenize_singleton_prompt(RendererFixture(), prompt, params)
    require(rendered['prompt_token_ids'] == IDS, 'official parser/renderer changed IDs')
    require(list(rendered) == ['prompt_token_ids'], 'unexpected prompt fields')
    # Constructing arguments validates the public constructor contract; deliberately
    # do not call create_engine_config(), which would resolve a model/device.
    kwargs = engine_kwargs()
    parsed = EngineArgs(**kwargs)
    for key, value in kwargs.items():
        require(getattr(parsed, key) == value, f'EngineArgs changed {key}')
    sampling = SamplingParams(**sampling_kwargs())
    require(sampling.temperature == 0 and sampling.max_tokens == 64,
            'sampling contract changed')
    transport = validate_callback_transport()
    return dict(format='llmp-gemma26-reference-api/1', image=IMAGE,
                callback_transport=transport,
                vllm_version=vllm.__version__, sources=sources,
                engine_kwargs=kwargs, sampling_kwargs=sampling_kwargs(),
                prompt_token_ids=rendered['prompt_token_ids'], bos_added=False,
                tokenizer_called=False, model_constructed=False,
                limits='No device/model configuration, weights, startup or inference tested')


def checkpoint_check(path):
    actual = {name: sha(path / name) for name in METADATA_HASHES}
    require(actual == METADATA_HASHES, 'checkpoint metadata differs from frozen source')
    archive = json.loads((path / 'llmp-archive.json').read_text())
    require(archive['repo'] == SOURCE and archive['commit'] == REVISION,
            'archive identity mismatch')
    # Full payload validation belongs to official stage-model. Preserve its receipt;
    # here authenticate the manifest and check every archived shard's actual size.
    shards = [f for f in archive['files'] if f['path'].endswith('.safetensors')]
    require(len(shards) == 2, 'unexpected archive shard set')
    for item in shards:
        require((path / item['path']).stat().st_size == item['size'],
                f"shard size mismatch: {item['path']}")
    return dict(metadata=actual, shards=shards, whole_shards_rehashed=False)


def inspect_model(model):
    """Read runtime dispatch only; no monkeypatches or weight transformations."""
    def typename(value):
        cls = type(value)
        return cls.__module__ + '.' + cls.__qualname__

    quant = model.quant_config
    record = dict(model=typename(model), quant_config=typename(quant),
                  quant_method=quant.quant_method,
                  linear_method=quant.LinearMethodCls.__name__, modules=[])
    for name, module in model.named_modules():
        method = getattr(module, 'quant_method', None)
        row = {'name': name}
        if method is not None:
            row['quant_method'] = typename(method)
            for attr in ('use_a16', 'nvfp4_backend', 'backend'):
                if hasattr(method, attr):
                    value = getattr(method, attr)
                    row[attr] = value if isinstance(value, bool) else str(value)
            for attr in ('kernel', 'moe_kernel', 'experts_cls'):
                value = getattr(method, attr, None)
                if value is not None:
                    row[attr] = (value.__module__ + '.' + value.__qualname__
                                 if isinstance(value, type) else typename(value))
        if hasattr(module, 'get_attn_backend'):
            backend = module.get_attn_backend()
            row['attention_backend'] = backend.__module__ + '.' + backend.__qualname__
            row['kv_cache_dtype'] = str(module.kv_cache_dtype)
        if len(row) > 1:
            record['modules'].append(row)
    return record


def check_dispatch(record):
    require(record['model'].endswith('.Gemma4ForCausalLM'), 'wrong model wrapper')
    require(record['quant_config'].endswith('.ModelOptNvFp4Config'), 'wrong quant config')
    require(record['quant_method'] == 'NVFP4', 'reference is not NVFP4 W4A4')
    require(record['linear_method'] == 'ModelOptNvFp4LinearMethod', 'wrong linear arithmetic')
    for row in record['modules']:
        method = row.get('quant_method', '')
        require('W4A16' not in method, 'linear method widened to A16')
        if method.endswith('.ModelOptNvFp4LinearMethod'):
            kernel = row.get('kernel', '')
            require(kernel, 'missing linear kernel dispatch')
            # The pinned selector may return these weight-only kernels even
            # with use_a16=False; the quantization class alone is not evidence.
            require(not any(name in kernel.upper() for name in ('MARLIN', 'HUMMING')),
                    'linear kernel widened to A16')
    experts = [m for m in record['modules']
               if m.get('quant_method', '').endswith('.ModelOptNvFp4FusedMoE')]
    require(len(experts) == 30, 'expected all 30 NVFP4 expert blocks')
    for row in experts:
        require(row.get('use_a16') is False, 'expert activation widened to A16')
        require(row.get('nvfp4_backend') and row.get('moe_kernel'), 'missing expert dispatch')
        require('MARLIN' not in row['nvfp4_backend'].upper(), 'unexpected Marlin fallback')
    attention = [m for m in record['modules'] if 'attention_backend' in m]
    require(len(attention) == 30, 'missing attention backend provenance')
    require(all(m['kv_cache_dtype'] == 'bfloat16' for m in attention), 'KV dtype changed')


def collect_output(outputs):
    require(len(outputs) == 1, 'expected one request')
    request = outputs[0]
    require(list(request.prompt_token_ids) == IDS, 'engine inserted/truncated/replaced prompt IDs')
    require(request.finished and len(request.outputs) == 1, 'request incomplete or forked')
    output = request.outputs[0]
    ids = list(output.token_ids)
    require(0 < len(ids) <= 64, 'invalid output length')
    require(output.finish_reason in ('stop', 'length'), 'unexpected finish reason')
    require(output.finish_reason != 'length' or len(ids) == 64, 'early length finish')
    require(not any(token in STOP_IDS for token in ids[:-1]), 'continued past a stop token')
    if output.finish_reason == 'length':
        require(output.stop_reason is None and ids[-1] not in STOP_IDS,
                'length finish contradicts stop contract')
    else:
        # The pinned scheduler reports primary EOS as None and explicit token
        # stops as the token ID. Its returned token IDs include that final token.
        require((output.stop_reason is None and ids[-1] == 1) or
                (type(output.stop_reason) is int and output.stop_reason in STOP_IDS
                 and ids[-1] == output.stop_reason), 'unexpected stop token or reason')
    require(output.logprobs is not None and len(output.logprobs) == len(ids),
            'missing generated-token log probabilities')
    rows = []
    for token, probabilities in zip(ids, output.logprobs, strict=True):
        require(token in probabilities, 'selected token missing from logprob row')
        row = {str(k): {'logprob': v.logprob, 'rank': v.rank,
                        'decoded_token': v.decoded_token} for k, v in probabilities.items()}
        require(all(math.isfinite(v['logprob']) and v['logprob'] <= 0 for v in row.values()),
                'nonfinite/positive log probability')
        rows.append(row)
    require(output.cumulative_logprob is not None and
            math.isfinite(output.cumulative_logprob) and output.cumulative_logprob <= 0,
            'missing/nonfinite/positive cumulative log probability')
    return dict(prompt_token_ids=list(request.prompt_token_ids), prompt_tokens=len(IDS),
                output_token_ids=ids, completion_tokens=len(ids), text=output.text,
                finish_reason=output.finish_reason, stop_reason=output.stop_reason,
                logprobs=rows, cumulative_logprob=output.cumulative_logprob)


def run(out):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    record = {'format': 'llmp-gemma26-source-diagnostic/1', 'image': IMAGE,
              'api': validate(), 'checkpoint': checkpoint_check(Path('/model')),
              'native_kv_dtype': 'float16', 'reference_dtype': 'bfloat16',
              'reference_kv_dtype': 'bfloat16', 'status': 'starting'}
    save = lambda: write_private(out / 'result.json', encoded(record))
    save()
    try:
        llm = LLM(**engine_kwargs())
        dispatch = llm.apply_model(inspect_model)
        record['dispatch'] = dispatch
        save()
        require(len(dispatch) == 1, 'expected one model worker')
        check_dispatch(dispatch[0])
        record['status'] = 'ready-dispatch-validated'
        save()
        outputs = llm.generate([TokensPrompt(prompt_token_ids=list(IDS))],
                               SamplingParams(**sampling_kwargs()), use_tqdm=False,
                               tokenization_kwargs={'add_special_tokens': False})
        # Save the complete returned objects before validating their shape. A failed
        # exact-ID contract retains the raw negative evidence and cannot become a pass.
        write_private(out / 'returned.txt', (repr(outputs) + '\n').encode())
        record['output'] = collect_output(outputs)
        record['status'] = 'collected-not-quality-qualified'
        save()
    except BaseException as error:
        record.update(status='failed', failure=f'{type(error).__name__}: {error}')
        save()
        raise


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate', 'validate-rpc-disabled', 'run'])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.mode == 'validate-rpc-disabled':
        print(encoded(dict(sources=source_check(),
                           callback_transport=validate_callback_transport(enabled=False))).decode(), end='')
    elif args.mode == 'validate':
        print(encoded(validate()).decode(), end='')
    else:
        require(args.out is not None and args.out.is_dir(), 'private mounted output required')
        run(args.out)


if __name__ == '__main__':
    main()
