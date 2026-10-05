#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Retire only this acquisition's labelled CID, including SIGKILL recovery."""
import json
import os
import pathlib
import re
import subprocess
import sys

LABEL = 'gemma26-late-moe'


def docker(*args):
    return subprocess.check_output(['sudo','-n','docker',*args],text=True).strip()


def retire(scratch,name):
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,99}',name):
        raise ValueError('outside owned container name')
    cid_path = scratch/(name+'.cid')
    if not cid_path.exists():
        # No CID means no proof: record nothing; caller retains this failure.
        return
    cid = cid_path.read_text().strip()
    if not re.fullmatch('[0-9a-f]{64}',cid):
        raise ValueError('malformed owned CID')
    remaining = docker('ps','-aq','--no-trunc','--filter','id='+cid)
    if remaining:
        if remaining != cid:
            raise ValueError('ambiguous Docker CID query')
        metadata = json.loads(docker('inspect',cid))
        if len(metadata) != 1 or metadata[0]['Id'] != cid or \
                metadata[0]['Config']['Labels'].get('jitllm.observer') != LABEL or \
                metadata[0]['Name'] != '/jitllm-gemma26-late-moe-'+name:
            raise ValueError('refuse another container owner')
        docker('rm','-f',cid)
    if docker('ps','-aq','--no-trunc','--filter','id='+cid):
        raise ValueError('owned container still exists')
    final = scratch/(name+'-container-retired.json')
    record = {'cid':cid,'name':'jitllm-gemma26-late-moe-'+name,
              'owner_label':LABEL,'container_absent_after_checked_docker_query':True}
    if final.exists():
        if json.loads(final.read_text()) != record:
            raise ValueError('retirement proof changed')
        return
    temporary = final.with_name(final.name+'.partial.'+str(os.getpid()))
    with temporary.open('x') as file:
        json.dump(record,file,sort_keys=True);file.write('\n');file.flush();os.fsync(file.fileno())
    try:
        os.link(temporary,final)
    finally:
        temporary.unlink()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('SCRATCH OWNED_NAME')
    retire(pathlib.Path(sys.argv[1]),sys.argv[2])
