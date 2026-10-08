#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed literal corpus excerpts and complete native/public token identity; no inference."""
import hashlib
import json
import struct
import sys
from pathlib import Path

CORPUS = 'c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d'
METADATA = '79c54f13f654d98e93ba1496062c017baea53ddaed59de086fe3fa7cf09d8461'
TEMPLATE = '845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(p, cap, expected=None):
    assert p.is_file() and not p.is_symlink() and p.stat().st_size <= cap
    b = p.read_bytes()
    assert len(b) <= cap and (expected is None or sha(b) == expected)
    return b


def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')


def extract(corpus, out):
    b = read(corpus, 4 << 20, CORPUS)
    assert len(b) == 3274124
    out.mkdir(mode=0o700)
    histories = []
    for index in range(1, 5):
        requested = index * 262144
        boundary = b.find(b'\n\n', requested)
        assert boundary >= 0
        start = boundary + 2
        while b[start:start+1] == b'\n': start += 1
        boundary = b.find(b'\n\n', start + 131072)
        assert boundary >= 0
        end = boundary + 2
        text = b[start:end]; text.decode('utf-8')
        assert 131072 <= len(text) <= 262144
        name = f'paragraph-{index}'
        with (out / (name + '.txt')).open('xb') as f: f.write(text)
        histories.append(dict(name=name, requested_offset=requested, start=start, end=end,
                              bytes=len(text), sha256=sha(text)))
    save(out / 'manifest.json', dict(schema=1, corpus_sha256=CORPUS, corpus_bytes=len(b),
                                    histories=histories, tokenization_done=False))


def pack(texts, text_sha, native, public, out):
    manifest = json.loads(read(texts / 'manifest.json', 65536, text_sha))
    assert manifest['corpus_sha256'] == CORPUS and not manifest['tokenization_done']
    assert len(manifest['histories']) == 4
    out.mkdir(mode=0o700)
    carrier = bytearray(); records = []
    for item in manifest['histories']:
        text = read(texts / (item['name'] + '.txt'), 262144, item['sha256'])
        assert len(text) == item['bytes']
        case = native / item['name']
        receipt = json.loads(read(case / 'receipt.json', 65536))
        assert receipt['mode'] == 'literal' and receipt['model_loaded'] is False
        assert receipt['metadata_sha256'] == METADATA and receipt['template_sha256'] == TEMPLATE
        assert receipt['text_sha256'] == item['sha256']
        n = read(case / 'ids.i32', 4 << 20)
        assert len(n) >= 8192 * 4 and len(n) % 4 == 0
        assert sha(n) == receipt['input_sha256']
        prefix = n[:8192 * 4]
        p = read(public / item['name'] / 'ids.i32', 8192 * 4)
        assert p == prefix
        ids = struct.unpack('<8192i', prefix)
        assert ids[0] == 2 and ids.count(2) == 1 and all(0 <= t < 262144 for t in ids)
        carrier.extend(prefix)
        records.append(dict(name=item['name'], text_sha256=item['sha256'],
                            native_complete_ids_sha256=sha(n), complete_native_rows=len(n)//4,
                            public_prefix_sha256=sha(p), rows=8192,
                            native_receipt_sha256=sha(read(case / 'receipt.json', 65536))))
    with (out / 'inputs.i32').open('xb') as f: f.write(carrier)
    save(out / 'manifest.json', dict(schema=1, profile=31, rows=8192, owners=4,
         input_sha256=sha(carrier), text_manifest_sha256=text_sha, metadata_sha256=METADATA,
         template_sha256=TEMPLATE, histories=records, model_contexts_created=0))


if __name__ == '__main__':
    if sys.argv[1] == 'extract' and len(sys.argv) == 4:
        extract(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == 'pack' and len(sys.argv) == 7:
        pack(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6]))
    else:
        raise SystemExit('extract CORPUS NEW_TEXTS; pack TEXTS TEXT_SHA NATIVE PUBLIC NEW_OUT')
