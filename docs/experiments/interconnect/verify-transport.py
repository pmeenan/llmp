#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify channel logs, mapped host buffers and HCA counters for every NCCL run."""
import argparse
import json
from pathlib import Path
import re

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('raw', type=Path)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
receipts = json.loads((a.raw/'nccl-main/receipts.json').read_text())
expected_names = {f'main-{op}-{arm}-r{r}' for op in ['sendrecv','all_reduce','all_gather']
                  for arm in ['auto','h0','h1'] for r in range(3)}
assert len(receipts) == len(expected_names) and {r['name'] for r in receipts} == expected_names
result = []
devices = ['rocep1s0f1', 'roceP2p1s0f1']
for rec in receipts:
    name = rec['name']
    arm = name.rsplit('-', 2)[1]
    selected = devices if arm == 'auto' else [devices[int(arm[1])]]
    for host in ['spark', 'spark-b']:
        logs = list((a.raw/'nccl-logs'/host).glob(name+'.*.log'))
        assert len(logs) == 1, (host, name, logs)
        content = logs[0].read_text()
        channels = [l for l in content.splitlines() if 'Channel ' in l and ' via NET/' in l]
        assert channels and all('via NET/IB/' in l and '/GDRDMA' not in l for l in channels), (host,name)
        assert any('[send]' in l for l in channels) and any('[receive]' in l for l in channels)
        ids = set(re.findall(r'via NET/IB/(\d+)', '\n'.join(channels)))
        assert ids == ({'0','1'} if arm == 'auto' else {'0'}), (name,ids)
        using = [l for l in content.splitlines() if 'NET/IB : Using ' in l]
        assert len(using) == 1
        for dev in devices:
            assert (dev+':1/RoCE' in using[0]) == (dev in selected), (name, using)
            if dev in selected:
                assert re.search(r"GPU Direct RDMA Disabled for HCA \d+ '"+re.escape(dev)+"'",content)
        buffers = [l for l in content.splitlines()
                   if re.search(r'transport/net.cc:(679|957|1135) .*Cuda Host Alloc Size', l)]
        assert buffers, (host,name)
        snapshots = [json.loads((a.raw/'nccl-main'/f'{name}.{host}.{side}.json').read_text())
                     for side in ['before','after']]
        deltas = {}
        for dev in devices:
            before, after = [x['devices'][dev]['files'] for x in snapshots]
            fields = ['ports/1/counters/port_xmit_data', 'ports/1/counters/port_rcv_data',
                      'ports/1/hw_counters/rx_write_requests']
            deltas[dev] = {field: int(after[field])-int(before[field]) for field in fields}
            if dev in selected:
                assert all(v > 0 for v in deltas[dev].values()), (host,name,deltas)
            else:
                assert all(v == 0 for v in deltas[dev].values()), (host,name,deltas)
            for field, value in before.items():
                if any(t in field for t in ['error', 'discard', 'drop', 'out_of_buffer',
                                             'timeout', 'retry', 'nak', 'err_','_err']):
                    if isinstance(value,str) and value.isdigit():
                        assert int(after[field]) == int(value), (host,name,dev,field)
        result.append({'name':name, 'host':host, 'selected':selected,
                       'channel_lines':len(channels), 'host_buffer_allocations':len(buffers),
                       'counter_deltas':deltas})
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(f'{len(result)} rank logs and before/after counter sets verified')
