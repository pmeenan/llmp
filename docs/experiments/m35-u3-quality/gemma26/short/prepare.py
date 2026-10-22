#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only source inspection and corpus tokenization; never construct a model."""
import sys
sys.dont_write_bytecode = True

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path

IMAGE = 'sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8'
CORPUS = '55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb'
PARENTS = {
    'paris_freeze.py': 'f8c21049c1e783ba865ac7bcc9bb9764e134b6b7bbfeeda3de4504dc52f7d8f2',
    'raw_reference.py': '847191af80ee1cead7979e06d825fc6460e1e73a408e6a415c48c9bc2c28d7ef',
}
KNOWN_SOURCE = {
    'sampling_params.py': '73e5bf250cb13803e4ce390407ae059b497898c8b81f9985d2b5c055d4db8f7d',
    'outputs.py': '54c21778c544e78509420817ab53d7b91a2beec6d7ca11a65e32458dc2cf3058',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(data)
    owner = Path('/output').stat()
    os.chmod(path, 0o600)
    os.chown(path, owner.st_uid, owner.st_gid)
    # Nested source directories must remain readable by the host owner.
    parent = path.parent
    while parent != Path('/output'):
        os.chown(parent, owner.st_uid, owner.st_gid)
        parent = parent.parent


def parent(path, name):
    require(sha(path) == PARENTS[name], 'parent helper identity differs: ' + name)
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_evidence(out, raw):
    # Locate installed package metadata without importing vLLM or torch.
    distribution = importlib.metadata.distribution('vllm')
    spec = importlib.util.find_spec('vllm')
    require(spec is not None and spec.origin is not None, 'installed vLLM has no source origin')
    root = Path(spec.origin).parent
    expected = dict(raw.SOURCE_HASHES, **KNOWN_SOURCE)
    actual = {name: sha(root / name) for name in expected}
    write(out / 'source-authentication.json', encoded(dict(
        image=IMAGE, package_version=distribution.version, package_root=str(root),
        expected=expected, actual=actual)))
    require(actual == expected, 'known pinned vLLM sources differ')
    # Capture the actual prompt-score path for subsequent human review. New file
    # hashes are observations from the immutable image, not invented prior pins.
    selected = set(expected) | {
        'logprobs.py', 'v1/engine/logprobs.py', 'v1/sample/sampler.py',
        'model_executor/layers/logits_processor.py', 'v1/worker/gpu_worker.py',
        'config/vllm.py', 'v1/core/kv_cache_manager.py',
    }
    scanned = {}
    total = 0
    for path in sorted((root / 'v1').rglob('*.py')):
        require(path.stat().st_size <= 4 << 20, 'unexpected source-file size')
        data = path.read_bytes()
        total += len(data)
        require(total <= 64 << 20, 'unexpected installed source inventory size')
        name = path.relative_to(root).as_posix()
        scanned[name] = hashlib.sha256(data).hexdigest()
        if any(term in data for term in (b'prompt_logprobs', b'def compute_logprobs',
                                        b'def gather_logprobs', b'log_softmax')):
            selected.add(name)
    records = {}
    for name in sorted(selected):
        path = root / name
        if not path.is_file():
            records[name] = {'missing': True}
            continue
        data = path.read_bytes()
        write(out / 'source' / name, data)
        lines = data.decode().splitlines()
        hits = set()
        for index, line in enumerate(lines):
            if any(term in line for term in ('prompt_logprobs', 'log_softmax', 'gather_logprobs',
                                            'compute_logprobs', 'max_tokens < 1')):
                hits.update(range(max(0, index - 4), min(len(lines), index + 12)))
        records[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                            excerpt=[{'line': i + 1, 'text': lines[i]} for i in sorted(hits)])
    evidence = dict(files=records, scanned_sha256=scanned, scanned_bytes=total,
                    semantic_review='pending: actual teacher inclusion and full-vocabulary normalization')
    write(out / 'source-evidence.json', encoded(evidence))
    return sha(out / 'source-evidence.json')


def freeze_ids(body, direct, tokenizer, generation_bos):
    # Some tokenizer implementations materialize the vocabulary in __len__.
    # Obtain this once, never once per corpus token.
    vocab_size = len(tokenizer)
    require(type(body) is list and type(direct) is list and body == direct,
            'separate encode and direct tokenizer IDs differ')
    require(len(body) >= 1023 and all(type(x) is int and 0 <= x < vocab_size for x in body),
            'invalid corpus IDs')
    require(tokenizer.bos_token_id == generation_bos == 2, 'BOS identity differs')
    require(2 not in body[:1023], 'body already contains BOS; do not substitute or strip it')
    return [2] + body[:1023], vocab_size


def main():
    os.umask(0o077)
    out, model = Path('/output'), Path('/metadata')
    freeze = parent(Path('/helpers/paris_freeze.py'), 'paris_freeze.py')
    raw = parent(Path('/helpers/raw_reference.py'), 'raw_reference.py')
    require(os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void' and
            not list(Path('/dev').glob('nvidia*')), 'GPU visible')
    require(os.environ.get('HF_HUB_OFFLINE') == '1' and
            os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline flags missing')
    actual = {name: sha(model / name) for name in freeze.METADATA}
    require(actual == freeze.METADATA and sorted(p.name for p in model.iterdir()) == sorted(actual),
            'metadata changed or unexpected files')
    data = Path('/corpus.txt').read_bytes()
    require(len(data) == 817275 and hashlib.sha256(data).hexdigest() == CORPUS, 'corpus differs')
    source_sha = source_evidence(out, raw)
    import transformers
    import tokenizers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    # Preserve the actual implementation before diagnosis, len(), or validation.
    implementations = {}
    for name in ('__len__', 'get_vocab'):
        method = getattr(type(tokenizer), name)
        path = Path(inspect.getsourcefile(method))
        implementations[name] = dict(file=str(path), file_sha256=sha(path),
                                     source=inspect.getsource(method))
    write(out / 'tokenizer-length-source.json', encoded(dict(
        tokenizer_class=type(tokenizer).__name__, implementations=implementations,
        prior_timeout='r1 called len(tokenizer) once per each of 213814 body IDs',
        correction='cache vocabulary size once before the range-validation loop')))
    body = tokenizer.encode(data.decode('utf-8'), add_special_tokens=False)
    direct = tokenizer(data.decode('utf-8'), add_special_tokens=False)['input_ids']
    diagnostic = dict(encode_signature=str(inspect.signature(tokenizer.encode)),
                      encode_source_sha256=sha(inspect.getsourcefile(tokenizer.encode)),
                      tokenizer_class=type(tokenizer).__name__, body_ids=body,
                      direct_ids=direct, bos_token_id=tokenizer.bos_token_id,
                      generation_bos=json.loads((model / 'generation_config.json').read_text())['bos_token_id'],
                      metadata=actual, corpus_sha256=CORPUS)
    write(out / 'tokenization-diagnostic.json', encoded(diagnostic))
    ids, vocab_size = freeze_ids(body, direct, tokenizer, diagnostic['generation_bos'])
    diagnostic['vocab_size'] = vocab_size
    write(out / 'tokenization-diagnostic.json', encoded(diagnostic))
    write(out / 'ids.json', encoded(ids))
    receipt = dict(status='frozen-inputs-only-source-review-pending', image=IMAGE,
                   ids_sha256=sha(out / 'ids.json'), input_ids=1024, first_scored_position=1,
                   targets=1023, corpus_sha256=CORPUS, metadata=actual,
                   recipe='[BOS 2] + encode(corpus, add_special_tokens=False)[:1023]',
                   source_evidence_sha256=source_sha, model_constructed=False, weights_mounted=False,
                   transformers_version=transformers.__version__, tokenizers_version=tokenizers.__version__,
                   helper_sha256=sha(__file__), parents=PARENTS,
                   tokenizer_length_source_sha256=sha(out / 'tokenizer-length-source.json'),
                   tokenization_diagnostic_sha256=sha(out / 'tokenization-diagnostic.json'),
                   nvidia_device_nodes=[], nvidia_visible_devices=os.environ['NVIDIA_VISIBLE_DEVICES'])
    write(out / 'result.json', encoded(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
