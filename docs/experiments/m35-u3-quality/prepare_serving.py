#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import argparse,pathlib,json,hashlib
p=argparse.ArgumentParser();p.add_argument('model',choices=['llama33','qwen27']);p.add_argument('checkpoint',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);p.add_argument('--store',default='/home/pmeenan/.local/share/llmp/models/m35-u3-proof');a=p.parse_args()
ids={'llama33':'9cca62f1e6582324de2ebce34f38b89a8aa8890cdced957a50782e407ff4d7ee','qwen27':'ca25eee473b02f4a13fe1d8680dea5f2aa244b556d26ee50eaa0b62b8719d3b5'}
expected={'llama33':{'config.json':'e7cbee89103cc488ef55c142c2a2a376ea00ce67b49c3ae9282d944107d97c33','generation_config.json':'b52d7103195cf7520887804dc703c715515dca31c7a2db20afdb2bf8f78203f0','tokenizer_config.json':'5c756ad6048c884974a331963ae3c3e1624b548a11174dadda02a69041c87352','tokenizer.json':'6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b'},'qwen27':{'config.json':'14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85','generation_config.json':'e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e','tokenizer_config.json':'792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b','tokenizer.json':'06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'}}
a.output.mkdir(parents=True,exist_ok=True);hashes={}
for name,want in expected[a.model].items():
 raw=(a.checkpoint/name).read_bytes();got=hashlib.sha256(raw).hexdigest();assert got==want,(name,got,want);hashes[name]=got
src=a.checkpoint/'chat_template.jinja'
if src.exists():template=src.read_text()
else:template=json.loads((a.checkpoint/'tokenizer_config.json').read_text())['chat_template']
assert isinstance(template,str);dst=a.output/'chat_template.jinja';dst.write_text(template);hashes['effective_chat_template']=hashlib.sha256(template.encode()).hexdigest()
# Keep state separate from the service and each other proof.
conf=f'''schema_version = 2
[storage]
data_dir = {json.dumps(str(a.output.resolve()/'runtime'))}
installed = {json.dumps(a.store)}
[models.{a.model}]
artifact = "{ids[a.model]}"
tokenizer = {json.dumps(str((a.checkpoint/'tokenizer.json').resolve()))}
chat_template = {json.dumps(str(dst.resolve()))}
context = 12288
max_slots = 1
prefill_chunk = 512
kv_cache_encoding = "f16"
temperature = 0.0
top_p = 1.0
top_k = 0
[client]
bind = "127.0.0.1"
port = 18974
'''
(a.output/'native.toml').write_text(conf)
receipt={'artifact':ids[a.model],'checkpoint':str(a.checkpoint.resolve()),'hashes':hashes,'checkpoint_sampling':json.loads((a.checkpoint/'generation_config.json').read_text()),'native_override':{'temperature':0,'top_p':1,'top_k':0},'context':12288,'max_slots':1}
(a.output/'pins.json').write_text(json.dumps(receipt,indent=2,sort_keys=True));print(json.dumps(receipt))
