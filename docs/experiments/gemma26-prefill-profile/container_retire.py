#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Retire only this acquisition's labelled CID, including SIGKILL recovery."""
import json
import os
import pathlib
import re
import subprocess
import sys

LABEL = 'gemma26-prefill-profile'


def docker(*args):
    return subprocess.check_output(['sudo','-n','docker',*args],text=True).strip()


def retire(scratch,name):
    assert re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',name)
    cid_path = scratch/(name+'.cid')
    if not cid_path.exists():
        # No CID means no proof: record nothing; caller retains this failure.
        return
    cid = cid_path.read_text().strip()
    assert re.fullmatch('[0-9a-f]{64}',cid)
    remaining = docker('ps','-aq','--no-trunc','--filter','id='+cid)
    if remaining:
        assert remaining == cid
        metadata = json.loads(docker('inspect',cid))
        assert len(metadata) == 1 and metadata[0]['Id'] == cid
        assert metadata[0]['Config']['Labels']['llmp.observer'] == LABEL
        assert metadata[0]['Name'] == '/llmp-gemma26-prefill-profile-'+name
        docker('rm','-f',cid)
    assert not docker('ps','-aq','--no-trunc','--filter','id='+cid)
    final = scratch/(name+'-container-retired.json')
    record = {'cid':cid,'name':'llmp-gemma26-prefill-profile-'+name,
              'owner_label':LABEL,'container_absent_after_checked_docker_query':True}
    if final.exists():
        assert json.loads(final.read_text()) == record
        return
    temporary = final.with_name(final.name+'.partial.'+str(os.getpid()))
    with temporary.open('x') as file:
        json.dump(record,file,sort_keys=True);file.write('\n');file.flush();os.fsync(file.fileno())
    try:
        os.link(temporary,final)
    finally:
        temporary.unlink()


if __name__ == '__main__':
    assert len(sys.argv) == 3
    retire(pathlib.Path(sys.argv[1]),sys.argv[2])
