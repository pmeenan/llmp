#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only official-tokenizer preparation; no model construction or weights."""
import sys
sys.dont_write_bytecode = True

import hashlib
import inspect
import json
import os
from collections.abc import Mapping
from pathlib import Path

IMAGE = 'sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8'
METADATA = {
    'config.json': '4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98',
    'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
    'tokenizer_config.json': '90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df',
    'chat_template.jinja': '94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413',
    'generation_config.json': 'd4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de',
    'llmp-archive.json': '4f36e82e02c29b3e99530efa38495459667a4c461a4d03bc9f610664db162ca1',
}
MESSAGES = [
    {'role': 'system', 'content': 'Answer only the requested capital city, without explanation.'},
    {'role': 'user', 'content': 'What is the capital of France?'},
]
RENDERED = ('<bos><|turn>system\nAnswer only the requested capital city, without explanation.'
            '<turn|>\n<|turn>user\nWhat is the capital of France?<turn|>\n'
            '<|turn>model\n<|channel>thought\n<channel|>')
RENDERED_SHA256 = 'd01c83b27d5600c694366037eeccd563440a46beac7555e70c186bbbc0b0651e'


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_private(path, value):
    path.write_bytes(encoded(value))
    owner = path.parent.stat()
    os.chmod(path, 0o600)
    os.chown(path, owner.st_uid, owner.st_gid)


def api_value(value):
    """Record the complete ordinary Python tokenization return, including mappings."""
    if isinstance(value, Mapping):
        return {key: api_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [api_value(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    # Preserve unexpected shapes as evidence; these cannot pass the flat-ID gate.
    return {'unsupported_type': type(value).__module__ + '.' + type(value).__qualname__,
            'repr': repr(value)}


def main():
    import transformers
    import tokenizers
    from transformers import AutoTokenizer
    model, out = Path('/metadata'), Path('/output')
    actual = {name: sha(model / name) for name in METADATA}
    require(actual == METADATA, 'metadata identity changed')
    require(sorted(p.name for p in model.iterdir()) == sorted(METADATA), 'unexpected metadata file')
    require(os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void', 'GPU visibility is not void')
    require(not list(Path('/dev').glob('nvidia*')), 'unexpected NVIDIA device node')
    require(os.environ.get('HF_HUB_OFFLINE') == '1' and
            os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline flags missing')
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True,
                                               trust_remote_code=False)
    method = tokenizer.apply_chat_template
    signature = inspect.signature(method)
    method_path = Path(inspect.getsourcefile(method))
    method_source = inspect.getsource(method)
    diagnostic = dict(metadata=actual, transformers_version=transformers.__version__,
                      tokenizers_version=tokenizers.__version__, tokenizer_class=type(tokenizer).__name__,
                      helper_sha256=sha(__file__), method_signature=str(signature),
                      method_source_file=str(method_path), method_source_file_sha256=sha(method_path),
                      method_source=method_source, method_doc=inspect.getdoc(method),
                      return_dict_parameter_supported='return_dict' in signature.parameters)
    save_diagnostic = lambda: write_private(out / 'tokenization-diagnostic.json', diagnostic)
    save_diagnostic()
    options = dict(tools=None, enable_thinking=False, add_generation_prompt=True)
    rendered = method(MESSAGES, tokenize=False, **options)
    diagnostic['rendered'] = rendered
    save_diagnostic()
    require(rendered == RENDERED and len(rendered.encode()) == 178 and
            hashlib.sha256(rendered.encode()).hexdigest() == RENDERED_SHA256,
            'official template rendering differs')
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    diagnostic.update(encode_type=type(ids).__module__ + '.' + type(ids).__qualname__,
                      encode_value=api_value(ids), encode_repr=repr(ids))
    save_diagnostic()
    default_direct = method(MESSAGES, tokenize=True, **options)
    diagnostic.update(default_direct_type=type(default_direct).__module__ + '.' + type(default_direct).__qualname__,
                      default_direct_value=api_value(default_direct), default_direct_repr=repr(default_direct))
    save_diagnostic()
    require('return_dict' in signature.parameters, 'actual API has no explicit return_dict parameter')
    explicit_direct = method(MESSAGES, tokenize=True, return_dict=False, **options)
    diagnostic.update(explicit_direct_options=dict(tokenize=True, return_dict=False, **options),
                      explicit_direct_type=type(explicit_direct).__module__ + '.' + type(explicit_direct).__qualname__,
                      explicit_direct_value=api_value(explicit_direct), explicit_direct_repr=repr(explicit_direct))
    save_diagnostic()
    if isinstance(default_direct, Mapping):
        # The official method's recorded source must itself expose this projection.
        require('input_ids' in default_direct and
                ('["input_ids"]' in method_source or "['input_ids']" in method_source),
                'default mapping has no source-documented input_ids projection')
        default_ids = default_direct['input_ids']
        diagnostic['default_projection'] = 'input_ids'
    else:
        default_ids = default_direct
        diagnostic['default_projection'] = 'identity'
    diagnostic['default_projected_value'] = api_value(default_ids)
    save_diagnostic()
    require(type(ids) is list and type(default_ids) is list and type(explicit_direct) is list,
            'expected three flat integer lists')
    require(ids == default_ids == explicit_direct,
            'separate encode, default projection and explicit flat direct call disagree')
    require(ids and all(type(x) is int and 0 <= x < len(tokenizer) for x in ids), 'invalid IDs')
    require(ids[0] == 2 and ids.count(2) == 1, 'BOS not present exactly once')
    input_value = dict(messages=MESSAGES, options=options, rendered=rendered,
                       rendered_sha256=RENDERED_SHA256, prompt_token_ids=ids,
                       add_special_tokens=False)
    write_private(out / 'input.json', input_value)
    module_path = Path(inspect.getfile(type(tokenizer)))
    receipt = dict(status='frozen-inputs-only', image=IMAGE, metadata=actual,
                   input_sha256=sha(out / 'input.json'), prompt_tokens=len(ids),
                   tokenizer_class=type(tokenizer).__name__,
                   tokenizer_module=str(module_path), tokenizer_module_sha256=sha(module_path),
                   transformers_version=transformers.__version__, tokenizers_version=tokenizers.__version__,
                   helper_sha256=sha(__file__), model_constructed=False, weights_mounted=False,
                   nvidia_device_nodes=[], nvidia_visible_devices=os.environ['NVIDIA_VISIBLE_DEVICES'],
                   method_signature=str(signature), method_source_file_sha256=sha(method_path),
                   default_direct_type=diagnostic['default_direct_type'],
                   default_projection=diagnostic['default_projection'],
                   explicit_direct_options=diagnostic['explicit_direct_options'],
                   tokenization_diagnostic_sha256=sha(out / 'tokenization-diagnostic.json'),
                   separate_encode_equals_default_projection_equals_explicit_flat=True)
    write_private(out / 'result.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
