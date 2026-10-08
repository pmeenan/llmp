#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Input-only preparation. extract PPL NATURAL NEW_OUT;
render TEMPLATE NATURAL NEW_OUT; pack TEXTS TEXT_SHA NATIVE PUBLIC PROFILE NEW_OUT.
extract-validation PPL NEW_OUT;
pack-validation TEXTS TEXT_SHA NATIVE PUBLIC NEW_OUT (closed26 histories13/14).
No model outputs are read. Tokenization is performed by separately reviewed clients.
"""
import hashlib
import json
import pathlib
import struct
import sys

CORPUS_SHA = 'c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d'
NATURAL_SHA = '116f5d709cfb18fc5681fabc6045d5b4d47d774ab9deba45c6a9d50dd5721b1c'
TEMPLATE_SHA = '845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b'
METADATA_SHA = {
    '26': '5d15e0c84e29af2ed9b4525dacdd31ccd7bae061eb28dde2dad984a9b1e45a59',
    '31': '79c54f13f654d98e93ba1496062c017baea53ddaed59de086fe3fa7cf09d8461'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, cap, sha=None):
    assert path.is_file() and path.stat().st_size <= cap
    raw = path.read_bytes()
    assert len(raw) <= cap
    if sha is not None:
        assert digest(raw) == sha, 'input identity changed'
    return raw


def fresh(path):
    path.mkdir(mode=0o700)


def save(path, value):
    with path.open('x') as file:
        json.dump(value, file, indent=2)
        file.write('\n')


def paragraphs(corpus, indices):
    slices = []
    # Paragraph starts strictly after each fixed byte offset. End at the first
    # complete paragraph boundary after 32 KiB, fixed before any tokenization.
    for index in indices:
        offset = 65536 * index
        boundary = corpus.find(b'\n\n', offset)
        assert boundary >= 0
        start = boundary + 2
        while corpus[start:start+1] == b'\n':
            start += 1
        boundary = corpus.find(b'\n\n', start + 32768)
        assert boundary >= 0
        end = boundary + 2
        text = corpus[start:end]
        text.decode('utf-8')
        assert len(text) <= 65536
        slices.append((f'paragraph-{index:02}', text, offset, start, end))
    return slices


def extract(corpus_path, natural_path, out):
    corpus = read(corpus_path, 4 << 20, CORPUS_SHA)
    assert len(corpus) == 3274124
    natural = json.loads(read(natural_path, 65536, NATURAL_SHA))
    assert [item['name'] for item in natural['cases']] == [
        'explanation', 'arithmetic', 'instruction', 'retrieval']
    slices = paragraphs(corpus, range(1, 13))
    fresh(out)
    histories = []
    for name, text, offset, start, end in slices:
        (out / (name + '.txt')).write_bytes(text)
        histories.append(dict(name=name, requested_offset=offset, start=start, end=end,
                              text_bytes=len(text), text_sha256=digest(text)))
    for item in natural['cases']:
        (out / (item['name'] + '.txt')).write_bytes(item['user'].encode())
    save(out / 'text-manifest.json', dict(schema=1, corpus_sha256=CORPUS_SHA,
         corpus_bytes=len(corpus), natural_sha256=NATURAL_SHA, histories=histories,
         natural_cases=natural['cases'], tokenization_done=False))


def extract_validation(corpus_path, out):
    corpus = read(corpus_path, 4 << 20, CORPUS_SHA)
    assert len(corpus) == 3274124
    slices = paragraphs(corpus, (13, 14))
    fresh(out)
    histories = []
    for name, text, offset, start, end in slices:
        (out / (name + '.txt')).write_bytes(text)
        histories.append(dict(name=name, requested_offset=offset, start=start, end=end,
                              text_bytes=len(text), text_sha256=digest(text)))
    save(out / 'text-manifest.json', dict(schema=1, corpus_sha256=CORPUS_SHA,
         corpus_bytes=len(corpus), histories=histories, tokenization_done=False,
         purpose='history13 native schedule calibration; history14 heldout'))


def render(template_path, natural_path, out):
    # Same pinned reference renderer used by the committed serving fixtures.
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    assert jinja2.__version__ == '3.1.6'
    template = read(template_path, 65536, TEMPLATE_SHA).decode()
    natural = json.loads(read(natural_path, 65536, NATURAL_SHA))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=['jinja2.ext.loopcontrols'])
    def refuse(text):
        raise ValueError(text)
    env.globals['raise_exception'] = refuse
    env.filters['from_json'] = json.loads
    env.filters['tojson'] = lambda obj: json.dumps(obj, ensure_ascii=False, sort_keys=True)
    template = env.from_string(template)
    results = []
    for case in natural['cases']:
        text = template.render(messages=[dict(role='user', content=case['user'])], tools=[],
                               bos_token='<bos>', eos_token='<eos>',
                               add_generation_prompt=True, enable_thinking=False).encode()
        assert len(text) <= 65536
        results.append((case['name'], text))
    fresh(out)
    for name, text in results:
        (out / (name + '.txt')).write_bytes(text)
    save(out / 'render-manifest.json', dict(schema=1, template_sha256=TEMPLATE_SHA,
         natural_sha256=NATURAL_SHA, renderer='Jinja2 3.1.6 immutable sandbox',
         cases=[dict(name=name, rendered_sha256=digest(text), rendered_bytes=len(text))
                for name, text in results]))


def ids(raw):
    assert len(raw) % 4 == 0
    values = struct.unpack('<%di' % (len(raw) // 4), raw)
    assert values and values[0] == 2 and values.count(2) == 1
    assert all(0 <= value < 262144 for value in values)
    return values


def pack(texts, expected_text_sha, native, public, profile, out, validation=False):
    assert profile in METADATA_SHA
    text_manifest = json.loads(read(texts / 'text-manifest.json', 65536, expected_text_sha))
    assert text_manifest['schema'] == 1 and text_manifest['corpus_sha256'] == CORPUS_SHA
    indices = (13, 14) if validation else range(1, 13)
    assert text_manifest['tokenization_done'] is False
    assert [x['name'] for x in text_manifest['histories']] == [
        f'paragraph-{i:02}' for i in indices]
    if validation:
        assert profile == '26' and 'natural_cases' not in text_manifest
    else:
        assert text_manifest['natural_sha256'] == NATURAL_SHA
    prepared, receipts = [], []
    cases = [(x['name'], x['text_sha256'], 'literal') for x in text_manifest['histories']]
    if not validation:
        cases += [(x['name'], digest(x['user'].encode()), 'chat')
                  for x in text_manifest['natural_cases']]
    for name, text_sha, mode in cases:
        read(texts / (name + '.txt'), 65536, text_sha)
        receipt = json.loads(read(native / name / 'receipt.json', 65536))
        assert receipt['schema'] == 1 and receipt['mode'] == mode
        assert receipt['metadata_sha256'] == METADATA_SHA[profile]
        assert receipt['text_sha256'] == text_sha and receipt['template_sha256'] == TEMPLATE_SHA
        assert receipt['model_loaded'] is False
        assert receipt['enable_thinking'] is False and receipt['add_generation_prompt'] is True
        raw = read(native / name / 'ids.i32', 4 << 20, receipt['input_sha256'])
        values = ids(raw)
        assert receipt['input_bytes'] == len(raw) and receipt['tokens'] == len(values)
        counterpart = read(public / name / 'ids.i32', 4 << 20)
        if mode == 'literal':
            assert len(values) >= 1024, 'fixed paragraph has fewer than 1024 tokens; stop'
            raw = raw[:4096]
            assert len(counterpart) == 4096 and raw == counterpart
            ids(raw)
        else:
            assert raw == counterpart
            rendered = read(native / name / 'rendered.txt', 65536, receipt['rendered_sha256'])
            assert rendered == read(public / 'rendered' / (name + '.txt'), 65536)
        prepared.append((name, raw, mode))
        receipts.append(receipt)
    fresh(out)
    histories = []
    for (name, raw, mode), receipt in zip(prepared, receipts):
        directory = out / name
        fresh(directory)
        (directory / 'ids.i32').write_bytes(raw)
        manifest = dict(schema=1, rows=len(raw)//4, input_bytes=len(raw),
                        input_sha256=digest(raw), bos=2, bos_count=1, vocab=262144,
                        profile=profile, mode=mode, text_manifest_sha256=expected_text_sha,
                        metadata_sha256=METADATA_SHA[profile], template_sha256=TEMPLATE_SHA,
                        tokenizer_agreement='complete supplied IDs byte exact')
        if mode == 'literal':
            manifest['scored_targets'] = 1023
            if validation:
                manifest['history_index'] = int(name.removeprefix('paragraph-'))
        save(directory / 'input-manifest.json', manifest)
        histories.append(manifest)
    if validation:
        save(out / 'validation-manifest.json', dict(schema=1, profile=profile,
             text_manifest_sha256=expected_text_sha, histories=histories,
             calibration_history=13, heldout_history=14,
             scored_transitions_per_history=1023, final_row_target=None))
        return
    carrier = b''.join(raw for _, raw, mode in prepared if mode == 'literal')
    assert len(carrier) == 12 * 1024 * 4
    (out / 'cohort12.i32').write_bytes(carrier)
    save(out / 'carrier-manifest.json', dict(schema=1, profile=profile,
         input_bytes=len(carrier), input_sha256=digest(carrier), histories=histories[:12], natural_cases=histories[12:],
         history_count=12, tokens_per_history=1024, order='owner-major paragraph-01..12',
         cohort_counts=[4, 8, 12], scored_frontier_positions=list(range(992, 1024)),
         final_frontier_target=None, corpus_scored_transitions_per_history=1023))


def main():
    args = sys.argv[1:]
    if len(args) == 4 and args[0] in ('extract', 'render'):
        (extract if args[0] == 'extract' else render)(*[pathlib.Path(x) for x in args[1:]])
    elif len(args) == 3 and args[0] == 'extract-validation':
        extract_validation(pathlib.Path(args[1]), pathlib.Path(args[2]))
    elif len(args) == 6 and args[0] == 'pack-validation':
        pack(pathlib.Path(args[1]), args[2], pathlib.Path(args[3]), pathlib.Path(args[4]),
             '26', pathlib.Path(args[5]), validation=True)
    elif len(args) == 7 and args[0] == 'pack':
        pack(pathlib.Path(args[1]), args[2], pathlib.Path(args[3]), pathlib.Path(args[4]),
             args[5], pathlib.Path(args[6]))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
