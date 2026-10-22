#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import argparse,pathlib,json,hashlib
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=pathlib.Path,required=True);p.add_argument('--corpus',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
raw=a.corpus.read_bytes();assert hashlib.sha256(raw).hexdigest()=='55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb';assert len(raw)==817275
assert hashlib.sha256((a.checkpoint/'tokenizer.json').read_bytes()).hexdigest()=='6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b'
tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True,trust_remote_code=False)
ids=tok.encode(raw.decode(),add_special_tokens=True)[:9216];assert len(ids)==9216 and ids[0]==128000 and all(0<=i<128256 for i in ids)
a.out.mkdir(parents=True,exist_ok=True);b=json.dumps(ids,separators=(',',':')).encode();(a.out/'ids.json').write_bytes(b)
(a.out/'inputs.json').write_text(json.dumps({'corpus_sha256':hashlib.sha256(raw).hexdigest(),'ids_sha256':hashlib.sha256(b).hexdigest(),'count':len(ids),'tokenizer_class':type(tok).__name__,'special_tokens_map':tok.special_tokens_map},indent=2));print('frozen ids',hashlib.sha256(b).hexdigest())
