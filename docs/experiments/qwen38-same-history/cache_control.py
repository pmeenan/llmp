# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Materialize the exact measured cache-control drivers from pinned originals.

Run the emitted driver separately under installed spark-job --gpu. This only
writes diagnostic sources; it does not launch a service or change engine code.
"""
import argparse
import hashlib
from pathlib import Path

PINS = {
    'controller': '151e257fd1d8ac43dcfe8fbf57a98103bee6ba0836f40208506cf713caf85b87',
    'client': '1d7c801bf1fdf25dede04eb28b59576e38253b287bf46fbd045dc519ae2888be',
    'helper': '53b0cc57c2ae77f2d56c6984d0257c3c9c2ceafc86b3acf455d9f02573e342a1',
}
OUTPUTS = {
    'mia-cold-p3-client.py': 'ffc35c10eb81f7ba13bddb68d5a9f167cbd7e72d43e5c5b3b4434a0a7e44af0c',
    'mia-cold-p3-run.py': 'd5e2abfe11180251684822c0a3f9ae0a2196ec001460a1c66cc5c1a3d1e1a68a',
    'mia-cache-paths-client.py': '2fdc80afd2b02f6c63f6125215f63624e8e10daed0f2d8c59e856f4ec6c3fc81',
    'mia-cache-paths-run.py': '710dcab7a48dbb1a099578df8855545086741be1dc64ee69f9e9f1111686fc40',
    'mia-det1-helper.py': '8f24cf10db459066ff24f55be4348a71e295320eb5d67e66b257f51739a85536',
    'mia-cache-det1-run.py': '7ce595c975a45dcb27dca078967331005e4ed01011d137e3be0db44769b09b9e',
    'mia-cold-all-client.py': '32c03d9ee10e7f55b6bb2aad23b64ca42ec4f197300f6ed72c129fc02fd779bc',
    'mia-cold-all-run.py': '15ef442afd3baef3eeac184ca03f3aba2e56e9020e30b4726e14f00b2860cab4',
}


def change(source, old, new):
    if source.count(old) != 1:
        raise ValueError('exact single diagnostic replacement required: ' + old)
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('controller', 'client', 'helper', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    originals = {}
    for name, pin in PINS.items():
        raw = getattr(args, name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError('pinned original differs: ' + name)
        originals[name] = raw.decode()
    made = {}
    cold_client = change(originals['client'], 'schedule.insert(1, (0, True))',
        'schedule = [(3, False), (3, True), (0, False), (1, False), (2, False)]\n'
        '        require(args.limit == 4 and args.previous is None, "exact cold-p3 cache-control schedule required")')
    made['mia-cold-p3-client.py'] = cold_client
    cold = change(originals['controller'], "'mia-first-verify-fast2'", "'mia-cold-p3'")
    cold = change(cold, "client = FROZEN / 'mia-anchor-client.py'", "client = BASE / 'mia-cold-p3-client.py'")
    cold = change(cold, "sha(BASE / 'mia-launch.py'), launcher_pin",
        "sha(BASE / 'mia-launch.py'), controller_sha256=sha(Path(__file__)), client_sha256=sha(BASE / 'mia-cold-p3-client.py'), launcher_pin")
    made['mia-cold-p3-run.py'] = cold
    paths_client = change(originals['client'], 'schedule.insert(1, (0, True))',
        'schedule = [(3, False), (3, True), (0, False), (3, False), (3, True)]\n'
        '        require(args.limit == 4 and args.previous is None, "exact two-cache-path schedule required")')
    paths_client = change(paths_client, 'prompt = prefix + continuation[:position]',
        'if index == 2:\n                salt = "jitllm-anchor-" + uuid.uuid4().hex\n'
        '            prompt = prefix + continuation[:position]')
    paths_client = change(paths_client, "'position': position, 'repeat': repeat, 'complete': False",
        "'position': position, 'repeat': repeat, 'cache_salt': salt, 'complete': False")
    made['mia-cache-paths-client.py'] = paths_client
    paths = cold.replace('mia-cold-p3', 'mia-cache-paths')
    made['mia-cache-paths-run.py'] = paths
    helper = change(originals['helper'],
        "expected_env = {'VLLM_QSA_DET_TOPK': '0', 'VLLM_MOE_DET_FINALIZE': '0'",
        "expected_env = {'VLLM_QSA_DET_TOPK': '1', 'VLLM_MOE_DET_FINALIZE': '1'")
    helper = change(helper, 'actual DET0/draft/offload environment differs',
                    'actual diagnostic DET1/draft/offload environment differs')
    made['mia-det1-helper.py'] = helper
    raw = helper.encode()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    det = change(paths, "helper_path = FROZEN / 'jitllm-mia-concurrent-run.py'", "helper_path = BASE / 'mia-det1-helper.py'")
    det = change(det, '01d606e9d9c7c362216ef9e0e3c18f01c0477f9b', blob)
    det = change(det, "out = BASE / 'mia-cache-paths'", "out = BASE / 'mia-cache-det1'")
    det = change(det, "requested.update(TP1_CONTAINER_NAME=name, PORT=str(PORT), READY_TIMEOUT_S='400')",
        "requested.update(TP1_CONTAINER_NAME=name, PORT=str(PORT), READY_TIMEOUT_S='400',\n"
        "                     VLLM_QSA_DET_TOPK='1', VLLM_MOE_DET_FINALIZE='1')")
    det = change(det, 'requested_environment=requested, observer_sha256=',
        "requested_environment=requested, diagnostic_settings='both recipe determinism knobs on; no timing qualification', helper_sha256=sha(helper_path), observer_sha256=")
    made['mia-cache-det1-run.py'] = det
    all_client = change(originals['client'], 'schedule.insert(1, (0, True))',
        'schedule = [(0, False), (1, False), (2, False), (3, False), (3, True)]\n'
        '        require(args.limit == 4 and args.previous is None, "four cold anchors and a cold p3 repeat required")')
    all_client = change(all_client, 'prompt = prefix + continuation[:position]',
        "salt = 'jitllm-anchor-' + uuid.uuid4().hex\n            prompt = prefix + continuation[:position]")
    all_client = change(all_client, "'position': position, 'repeat': repeat, 'complete': False",
        "'position': position, 'repeat': repeat, 'cache_salt': salt, 'complete': False")
    made['mia-cold-all-client.py'] = all_client
    all_cold = change(originals['controller'], "BASE = Path('/home/pmeenan/scratch/acceptance')",
        "BASE = Path('/home/pmeenan/scratch/qwen-cold-anchors')")
    all_cold = change(all_cold, "'mia-first-verify-fast2'", "'mia-cold-all'")
    all_cold = change(all_cold, "client = FROZEN / 'mia-anchor-client.py'", "client = BASE / 'mia-cold-all-client.py'")
    all_cold = change(all_cold,
        "                  launcher_sha256=sha(BASE / 'mia-launch.py'),",
        "                  controller_sha256=sha(Path(__file__)), client_sha256=sha(BASE / 'mia-cold-all-client.py'),\n"
        "                  launcher_sha256=sha(BASE / 'mia-launch.py'),")
    all_cold = change(all_cold, "        rows = [json.loads(p.read_text()) for p in traces.glob('*.json')]",
        "        require(all(r['cached_tokens']==0 for r in anchors['requests']), 'every independently salted cold request must consume all prompt IDs')\n"
        "        rows = [json.loads(p.read_text()) for p in traces.glob('*.json')]")
    made['mia-cold-all-run.py'] = all_cold
    for name, body in made.items():
        raw = body.encode()
        if hashlib.sha256(raw).hexdigest() != OUTPUTS[name]:
            raise ValueError('generated source differs from measured pin: ' + name)
        path = args.out / name
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError('existing diagnostic source differs: ' + str(path))
        else:
            with path.open('xb') as file:
                file.write(raw)
        print(name, OUTPUTS[name])


if __name__ == '__main__':
    main()
