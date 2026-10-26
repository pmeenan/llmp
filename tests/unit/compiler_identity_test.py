#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Narrow compiler receipt controls; native compiler fixture is opt-in."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("compiler_identity", ROOT / "tools/compiler_identity.py")
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


def ownership_catalog(build, paths, generated=()):
    path = build / 'ownership.json'
    path.write_text(json.dumps({'schema': 1, 'owned_inputs': sorted(str(p) for p in paths),
                               'declared_generated': sorted(str(p) for p in generated)}))
    return str(path)


class CompilerIdentityTest(unittest.TestCase):
    def test_paths_do_not_rewrite_semantic_flags(self):
        roots = [("/owner/source", "$SOURCE"), ("/owner/build", "$BUILD")]
        args = ["/sdk/clang", '-DNAME="/owner/source/value"',
                "-unknown=/owner/source/value", "-I/owner/source/include",
                "-I/owner/source-other", "-o", "/owner/build/file.o", "/owner/source/file.cc"]
        got = IDENTITY.normalized_arguments(args, roots, args[-1])
        self.assertEqual(got[1:3], args[1:3])
        self.assertEqual(got[3], "-I$SOURCE/include")
        self.assertEqual(got[4], args[4])
        self.assertEqual(got[-2:], ["$BUILD/file.o", "$SOURCE/file.cc"])

    def test_xcompiler_prefix_map_commas_and_target_cache_invalidation(self):
        self.assertEqual(
            IDENTITY.split_xcompiler("-ffile-prefix-map=/owner/path,with,comma=llmp/source"),
            ["-ffile-prefix-map=/owner/path,with,comma=llmp/source"])
        self.assertEqual(
            IDENTITY.split_xcompiler("-ffile-prefix-map=/a=b,-D__DATE__=\"override\""),
            ["-ffile-prefix-map=/a=b", "-D__DATE__=\"override\""])
        IDENTITY.supported_compile_options([
            "/sdk/nvcc", "-Xcompiler=-ffile-prefix-map=/owner/path,with,comma=llmp/source",
            "-c", "file.cu", "-o", "file.o"])
        with self.assertRaises(IDENTITY.Refusal):
            IDENTITY.supported_compile_options([
                "/sdk/nvcc", "-Xcompiler=-ffile-prefix-map=/a=b,-D__DATE__=\"override\"",
                "-c", "file.cu", "-o", "file.o"])
        for bad in ("-Xcompiler=-ffile-prefix-map=/a=b,@rsp.txt",
                    "-Xcompiler=-ffile-prefix-map=/a=b,extra.cu",
                    "-Xcompiler=-ffile-prefix-map=/a"):
            with self.assertRaises(IDENTITY.Refusal):
                IDENTITY.supported_compile_options(["/sdk/nvcc", bad, "-c", "file.cu", "-o", "file.o"])
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "CMakeFiles").mkdir()
            build_ninja = build / "build.ninja"
            rules_ninja = build / "CMakeFiles/rules.ninja"
            build_ninja.write_text("build a.o: CXX\n")
            rules_ninja.write_text("rule CXX\n")
            config = {"build": str(build), "ninja": "/sdk/ninja"}
            calls = []
            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                text = "a.o: CXX\nall: phony\n" if len(calls) == 1 else "a.o: CXX\nb.o: CXX\nall: phony\n"
                return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")
            with mock.patch.object(IDENTITY.subprocess, "run", side_effect=fake_run):
                first, _ = IDENTITY.build_outputs(config)
                second, _ = IDENTITY.build_outputs(config)
                self.assertEqual(first, {str(build / "a.o")})
                self.assertEqual(second, first)
                self.assertEqual(len(calls), 1)
                build_ninja.write_text("build a.o: CXX\nbuild b.o: CXX\n")
                third, _ = IDENTITY.build_outputs(config)
                self.assertEqual(third, {str(build / "a.o"), str(build / "b.o")})
                self.assertEqual(len(calls), 2)

    def test_make_dependency_format(self):
        self.assertEqual(IDENTITY.depfile_paths("file.o: a.cc path\\ with\\ space.h \\\n other\\#name.h dollar$$.h\n"),
                         ["a.cc", "path with space.h", "other#name.h", "dollar$.h"])
        for text in ("file.o: $(variable)", "file.o: x.h\nother.o: x.h", "file.o:"):
            with self.assertRaises(IDENTITY.Refusal):
                IDENTITY.depfile_paths(text)

    def test_launcher_dependency_rule(self):
        # The native/ninja fixture below checks actual consumed names and build
        # behavior. Our compiler-depfile reader is not an oracle for Ninja.
        for path in ('line\nbreak', 'carriage\rreturn', 'nul\0byte', 'tab\tname',
                     'trailing\\', 'back\\$dollar'):
            with self.assertRaises(IDENTITY.Refusal):
                IDENTITY.dependency_rule('object.o', {path})

    def test_missing_and_stale_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = Path(directory) / "file.o"
            obj.write_bytes(b"object")
            with self.assertRaises(IDENTITY.Refusal):
                IDENTITY.validated_receipt(obj)
            primary = {"source": "fixture"}
            receipt = {"schema": 1, "object": str(obj), "object_record": IDENTITY.snapshot(obj),
                       "primary": primary, "identity": IDENTITY.sha(IDENTITY.canonical(primary)), "inputs": {},
                       "compiler_dependency_paths": [str(obj)]}
            archive = Path(directory) / 'ownership.json'
            archive.write_text('{}')
            receipt['ownership'] = {'archive': str(archive), 'record': IDENTITY.snapshot(archive)}
            IDENTITY.receipt_path(obj).write_text(json.dumps(receipt))
            IDENTITY.validated_receipt(obj)
            obj.write_bytes(b"changed")
            with self.assertRaises(IDENTITY.Refusal):
                IDENTITY.validated_receipt(obj)

    def test_actual_dependency_directory_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'header').mkdir()
            ordinary = root / 'header/name.h'
            ordinary.write_text('normal')
            IDENTITY.check_dependency_aliases([str(ordinary)])
            unrelated = root / 'unrelated\\name.h'
            unrelated.write_text('not a candidate')
            IDENTITY.check_dependency_aliases([str(ordinary)])
            for name, raw in [('header\\name.h', str(ordinary)),
                              ('header\\\\name.h', str(root)+'/header//name.h'),
                              ('header\\..\\name.h', str(root)+'/header/../name.h')]:
                alias = root / name
                alias.write_text('ambiguous')
                with self.assertRaises(IDENTITY.Refusal):
                    IDENTITY.check_dependency_aliases([raw])
                alias.unlink()
            alias_dir = root / 'header\\nested'
            alias_dir.mkdir()
            with self.assertRaises(IDENTITY.Refusal):
                IDENTITY.check_dependency_aliases([str(root / 'header/nested/file.h')])
            parent_alias = root / 'a\\b'
            parent_alias.mkdir()
            for suffix in ('a/b/../header.h', 'a//b/../header.h', 'a/./b/../header.h'):
                with self.assertRaises(IDENTITY.Refusal):
                    IDENTITY.check_dependency_aliases([str(root)+'/'+suffix])
            # Ordinary parent components remain supported.
            IDENTITY.check_dependency_aliases([str(root)+'/header/../ordinary.h'])
            # Raw escaping outside the supported compiler format never drops
            # a backslash and accidentally authenticates a different filename.
            with self.assertRaises(IDENTITY.Refusal):
                IDENTITY.depfile_paths('object.o: header\\name.h\n')

    def test_capture_refuses_changed_or_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, build, sdk = (root / name for name in ('source', 'build', 'sdk'))
            for path in (source, build, sdk / 'bin'):
                path.mkdir(parents=True)
            (sdk / 'sdk.json').write_text('{"identity":"fixture","created":"receipt-only"}')
            src, header = source / 'file.cc', source / 'header.h'
            src.write_text('input')
            header.write_text('header')
            compiler = sdk / 'bin/clang++'
            mode = source / 'mode'
            mode.write_text('normal')
            compiler.write_text('#!/usr/bin/python3\nimport pathlib,sys\n'
                'a=sys.argv; obj=pathlib.Path(a[a.index("-o")+1]); dep=pathlib.Path(a[a.index("-MF")+1])\n'
                'src=pathlib.Path(a[a.index("-c")+1]); header=src.parent/"header.h"\n'
                'mode=(src.parent/"mode").read_text()\n'
                'if mode=="source": src.write_text("changed")\n'
                'if mode=="header": header.write_text("changed")\n'
                'obj.write_bytes(b"object")\n'
                'extra=" "+str(obj.parent/"ownership.json") if mode=="catalog_dependency" else ""\n'
                'if mode!="missing": dep.write_text(str(obj)+": "+str(src)+" "+str(header)+extra+"\\n")\n')
            compiler.chmod(0o755)
            config = build / 'config.json'
            config.write_text(json.dumps({'schema': 1, 'source': str(source), 'build': str(build),
                'sdk': str(sdk), 'prepared': str(root / 'prepared'), 'compilers': [str(compiler)],
                'generated': {}, 'excluded_sources': [], 'identity': {},
                'ownership_catalog': ownership_catalog(build, [src, header, source / 'flags.rsp'])}))
            obj, dep = build / 'file.o', build / 'file.o.d'
            flags = ['--no-default-config', '-Werror=date-time', '-D__DATE__="Jan  1 1970"',
                     '-D__TIME__="00:00:00"', '-D__TIMESTAMP__="Thu Jan  1 00:00:00 1970"']
            for name, mapped in ((source, 'source'), (build, 'build'), (sdk, 'sdk'), (root/'prepared', 'prepared')):
                flags.append('-ffile-prefix-map='+str(name)+'=llmp/'+mapped)
            response = source / 'flags.rsp'
            response.write_text('-DSEMANTIC="/owner/source/literal"')
            command = [str(compiler), *flags, '@'+str(response), '-MD', '-MT', str(obj),
                       '-MF', str(dep), '-o', str(obj), '-c', str(src)]
            original = Path.cwd()
            try:
                os.chdir(build)
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(IDENTITY.compile_with_receipt(config, command), 0)
                    IDENTITY.validated_receipt(obj)
                    dependencies = set(IDENTITY.depfile_paths(dep.read_text()))
                    self.assertEqual(dependencies, {str(p) for p in
                        (src, header, response, config, ROOT / 'tools/compiler_identity.py', sdk / 'sdk.json')})
                    original_receipt = IDENTITY.validated_receipt(obj)
                    catalog = build / 'ownership.json'
                    original_catalog = catalog.read_bytes()
                    report = source / 'unrelated-report.md'
                    report.write_text('unrelated authored document')
                    ownership_catalog(build, [src, header, response, report])
                    current = IDENTITY.load_config(config)
                    IDENTITY.validate_receipt_authority(original_receipt, config, current)
                    self.assertEqual(IDENTITY.validated_receipt(obj), original_receipt)
                    self.assertEqual(Path(original_receipt['ownership']['archive']).read_bytes(), original_catalog)
                    self.assertNotEqual(current['_ownership_audit']['record']['sha256'],
                                        original_receipt['ownership']['record']['sha256'])
                    for paths, generated in (([src, response], []), ([src, header, response], [header])):
                        ownership_catalog(build, paths, generated)
                        with self.assertRaises(IDENTITY.Refusal):
                            IDENTITY.validate_receipt_authority(original_receipt, config, IDENTITY.load_config(config))
                    ownership_catalog(build, [src, header, response])
                    current = IDENTITY.load_config(config)
                    current['_build_outputs'] = {str(header)}
                    with self.assertRaisesRegex(IDENTITY.Refusal, 'generated input has no declared recipe'):
                        IDENTITY.validate_receipt_authority(original_receipt, config, current)
                    # A catalog is control-only only if the compiler did not
                    # consume it. A real dependency still needs a recipe.
                    mode.write_text('catalog_dependency')
                    with self.assertRaisesRegex(IDENTITY.Refusal, 'generated input has no declared recipe'):
                        IDENTITY.compile_with_receipt(config, command)
                    self.assertFalse(obj.exists())
                    mode.write_text('normal')
                    # Mutate authority during the actual compiler call. The
                    # object is discarded even though all code bytes match.
                    real_run = IDENTITY.subprocess.run
                    def mutate_catalog(*args, **kwargs):
                        result = real_run(*args, **kwargs)
                        ownership_catalog(build, [src, header, response, report])
                        return result
                    with mock.patch.object(IDENTITY.subprocess, 'run', side_effect=mutate_catalog):
                        with self.assertRaisesRegex(IDENTITY.Refusal, 'catalog/build graph changed'):
                            IDENTITY.compile_with_receipt(config, command)
                    self.assertFalse(obj.exists())
                    self.assertFalse(IDENTITY.receipt_path(obj).exists())
                    catalog.write_bytes(original_catalog)
                    self.assertEqual(IDENTITY.compile_with_receipt(config, command), 0)
                    for override in (['-D', '__DATE__="override"'], ['-U__TIME__'],
                                     ['--define-macro=__TIMESTAMP__="override"'],
                                     ['-Xcompiler=-D__DATE__="override"'],
                                     ['-Xcompiler=-D,__DATE__="override"'],
                                     ['--define-macro=OTHER=1,__TIME__="override"']):
                        with self.assertRaises(IDENTITY.Refusal):
                            IDENTITY.compile_with_receipt(config, command+override)
                    original_config = config.read_text()
                    template, generator = source / 'template.cc', source / 'generator.py'
                    template.write_bytes(src.read_bytes())
                    generator.write_text('import shutil,sys;shutil.copyfile(sys.argv[1],sys.argv[2])')
                    derived = json.loads(original_config)
                    derived['excluded_sources'] = [str(src)]
                    derived['derived'] = {str(src): {'inputs': [str(template), str(generator)],
                        'value_inputs': [str(template)], 'identity': 'test-constant',
                        'command': ['/usr/bin/python3', str(generator), str(template), '@OUTPUT@']}}
                    ownership_catalog(build, [src, header, response, template, generator])
                    config.write_text(json.dumps(derived))
                    self.assertEqual(IDENTITY.compile_with_receipt(config, command), 0)
                    first = IDENTITY.validated_receipt(obj)['identity']
                    header.write_text('forced-header code changed')
                    self.assertEqual(IDENTITY.compile_with_receipt(config, command), 0)
                    second = IDENTITY.validated_receipt(obj)['identity']
                    self.assertNotEqual(first, second)
                    derived_receipt = IDENTITY.validated_receipt(obj)
                    for paths, generated in (([src, header, response, generator], []),
                                             ([src, header, response, template, generator], [template])):
                        ownership_catalog(build, paths, generated)
                        with self.assertRaises(IDENTITY.Refusal):
                            IDENTITY.validate_receipt_authority(derived_receipt, config, IDENTITY.load_config(config))
                    ownership_catalog(build, [src, header, response, template, generator])
                    template.write_text('new value')
                    src.write_text('new value')
                    self.assertEqual(IDENTITY.compile_with_receipt(config, command), 0)
                    self.assertEqual(IDENTITY.validated_receipt(obj)['identity'], second)
                    src.write_text('extra code outside generator')
                    with self.assertRaises(IDENTITY.Refusal):
                        IDENTITY.compile_with_receipt(config, command)
                    config.write_text(original_config)
                    response.write_text('-DSEMANTIC="other"')
                    with self.assertRaises(IDENTITY.Refusal):
                        IDENTITY.validated_receipt(obj)
                    for changed in ('source', 'header', 'missing'):
                        mode.write_text(changed)
                        dep.unlink(missing_ok=True)
                        with self.assertRaises((IDENTITY.Refusal, OSError)):
                            IDENTITY.compile_with_receipt(config, command)
                        self.assertFalse(obj.exists())
                    os.environ['LD_PRELOAD'] = '/unsupported/compiler-hook.so'
                    with self.assertRaises(IDENTITY.Refusal):
                        IDENTITY.compile_with_receipt(config, command)
                    os.environ.clear()
                    os.chdir(source)
                    with self.assertRaises(IDENTITY.Refusal):
                        IDENTITY.compile_with_receipt(config, command)
            finally:
                os.chdir(original)


def compiler_alias_fixture(work, compiler, sdk):
    """Actual pinned Clang: lossy MD spelling must not bless a collider."""
    work.mkdir(parents=True)
    source, build = work / 'source', work / 'build'
    source.mkdir()
    build.mkdir()
    (source / 'header').mkdir()
    ordinary, alias = source / 'header/name.h', source / 'header\\name.h'
    ordinary.write_text('#define MARKER 999\n')
    src = source / 'source.cc'
    src.write_text('#include "header/name.h"\nstatic_assert(MARKER == 999);\nint value = MARKER;\n')
    sdk = Path(sdk)
    arch = json.loads((sdk / 'sdk.json').read_text())['host_arch']
    assert arch in ('x86_64', 'aarch64')
    triple = arch + '-linux-gnu'
    config = build / 'config.json'
    config.write_text(json.dumps({'schema': 1, 'source': str(source), 'build': str(build),
        'sdk': str(sdk), 'prepared': str(work / 'prepared'), 'compilers': [compiler],
        'generated': {}, 'excluded_sources': [], 'identity': {},
        'ownership_catalog': ownership_catalog(build, [src, ordinary])}))
    obj, dep = build / 'object.o', build / 'object.o.d'
    flags = ['--target='+triple, '-march='+('x86-64' if arch == 'x86_64' else 'armv8-a'),
             '--gcc-install-dir='+str(sdk / 'gcc' / triple / 'lib/gcc' / triple / '16'),
             '--no-default-config', '-std=c++23', '-Werror=date-time', '-Wno-builtin-macro-redefined',
             '-D__DATE__="Jan  1 1970"', '-D__TIME__="00:00:00"',
             '-D__TIMESTAMP__="Thu Jan  1 00:00:00 1970"']
    for name, mapped in ((source, 'source'), (build, 'build'), (sdk, 'sdk'), (work/'prepared', 'prepared')):
        flags.append('-ffile-prefix-map='+str(name)+'=llmp/'+mapped)
    command = [compiler, *flags, '-MD', '-MT', str(obj), '-MF', str(dep), '-o', str(obj), '-c', str(src)]
    # Ordinary sibling creation/removal must not invalidate a dependency merely
    # because a shared ancestor's directory timestamp changes during inspection.
    stop_activity = threading.Event()
    activity_errors = []
    def unrelated_activity():
        try:
            while not stop_activity.is_set():
                for directory in (work.parent, source, build):
                    with tempfile.NamedTemporaryFile(prefix='ordinary-activity-', dir=directory):
                        stop_activity.wait(0.0005)
        except BaseException as error:
            activity_errors.append(error)
    activity = threading.Thread(target=unrelated_activity)
    activity.start()
    previous = Path.cwd()
    try:
        os.chdir(build)
        with mock.patch.dict(os.environ, {}, clear=True):
            assert IDENTITY.compile_with_receipt(config, command) == 0
            IDENTITY.validated_receipt(obj)
            launcher = [sys.executable, '-B', str(ROOT / 'tools/compiler_identity.py'),
                        'compile', '--config', str(config), '--', *command]
            (build / 'build.ninja').write_text('rule compile\n  command = '+shlex.join(launcher)+
                '\n  depfile = '+str(dep)+'\n  deps = gcc\nbuild '+str(obj)+': compile '+str(src)+'\n')
            ninja = sdk / 'ninja/ninja'
            subprocess.run([str(ninja)], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            original = IDENTITY.receipt_path(obj).read_bytes()
            unchanged = IDENTITY.snapshot(obj)
            noop = subprocess.run([str(ninja)], check=True, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert 'no work to do' in noop.stdout
            assert IDENTITY.snapshot(obj) == unchanged
            assert IDENTITY.receipt_path(obj).read_bytes() == original
            alias.write_text('#define MARKER 123\n')
            # Even an old otherwise-current receipt must detect the new alias.
            try:
                IDENTITY.validated_receipt(obj)
                raise AssertionError('new alias bypassed receipt validation')
            except IDENTITY.Refusal as error:
                assert 'ambiguous compiler dependency path' in str(error)
            assert IDENTITY.receipt_path(obj).read_bytes() == original
            for marker in (123, 456):
                alias.write_text(f'#define MARKER {marker}\n')
                src.write_text('#include "header\\name.h"\n'+f'static_assert(MARKER == {marker});\nint value = MARKER;\n')
                expanded = subprocess.run([compiler, *flags, '-E', '-P', str(src)],
                    env=IDENTITY.effective_environment(IDENTITY.load_config(config)),
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                assert f'int value = {marker};' in expanded.stdout
                try:
                    IDENTITY.compile_with_receipt(config, command)
                    raise AssertionError('compiler-normalized collider was accepted')
                except IDENTITY.Refusal as error:
                    assert 'ambiguous compiler dependency path' in str(error)
                assert not obj.exists() and not IDENTITY.receipt_path(obj).exists()
            alias.unlink()
            # Clang also loses a literal-backslash directory before '..'.
            # The actual parent differs from lexical slash-path normalization.
            (source / 'a/b').mkdir(parents=True)
            (source / 'a\\b').mkdir()
            (source / 'a/header.h').write_text('#define MARKER 999\n')
            for marker in (123, 456):
                (source / 'header.h').write_text(f'#define MARKER {marker}\n')
                src.write_text('#include "a\\b/../header.h"\n'+
                               f'static_assert(MARKER == {marker});\nint value = MARKER;\n')
                expanded = subprocess.run([compiler, *flags, '-E', '-P', str(src)],
                    env=IDENTITY.effective_environment(IDENTITY.load_config(config)),
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                assert f'int value = {marker};' in expanded.stdout
                try:
                    IDENTITY.compile_with_receipt(config, command)
                    raise AssertionError('compiler-normalized parent collider was accepted')
                except IDENTITY.Refusal as error:
                    assert 'ambiguous compiler dependency path' in str(error)
                assert not obj.exists() and not IDENTITY.receipt_path(obj).exists()
            (source / 'a\\b').rmdir()
            src.write_text('#include "header/name.h"\nstatic_assert(MARKER == 999);\nint value = MARKER;\n')
            assert IDENTITY.compile_with_receipt(config, command) == 0
            IDENTITY.validated_receipt(obj)
    finally:
        os.chdir(previous)
        stop_activity.set()
        activity.join(timeout=5)
        assert not activity.is_alive() and not activity_errors
    print('Actual Clang ordinary/build-noop/concurrent-siblings/collider123/mutated456/parent-collider123/parent-mutated456/restored999 controls PASS', flush=True)


def ninja_dependency_fixture(ninja):
    """The actual build-tool consumer, including no-op and changed-input edges."""
    with tempfile.TemporaryDirectory(prefix='llmp-ninja-dependencies-') as directory:
        work = Path(directory)
        names = ['ordinary.h', 'space name.h', 'hash#name.h', 'dollar$name.h',
                 'colon:name.h', 'back\\slash.h', 'two\\\\slashes.h',
                 'back\\ space.h', 'two\\\\ spaces.h', 'back\\#hash.h',
                 'back\\:colon.h', 'mixed #$ name.h']
        for name in names:
            (work / name).write_text('input')
        (work / 'rule.txt').write_text(IDENTITY.dependency_rule('object.o', set(names)))
        (work / 'build.ninja').write_text(
            'rule emit\n  command = cp rule.txt object.o.d && touch object.o\n'
            '  depfile = object.o.d\n  deps = gcc\nbuild object.o: emit rule.txt\n')
        def run(*args):
            result = subprocess.run([ninja, *args], cwd=work, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            return result.stdout
        run()
        consumed = run('-t', 'deps').splitlines()[1:]
        assert {line[4:] for line in consumed if line.startswith('    ')} == set(names), consumed
        assert 'no work to do' in run('-n', '-d', 'explain')
        for name in names:
            (work / name).write_text('changed dependency')
            assert 'cp rule.txt object.o.d' in run('-n', '-d', 'explain'), name
            run()
            assert 'no work to do' in run('-n', '-d', 'explain'), name
    print('Actual Ninja dependency consumption/no-op/change controls PASS', flush=True)


def native_fixture(root, work, cmake, toolchain, python, cuda):
    """Use the production CMake module and launcher, not a parallel mock."""
    work.mkdir(parents=True, exist_ok=True)
    source, build = work / "source", work / "build"
    source.mkdir()
    for name in ("cmake", "tools", "src/common", "src/engine", "src/kernels", "src/public"):
        (source / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "tools/compiler_identity.py", source / "tools/compiler_identity.py")
    shutil.copyfile(root / "cmake/LlmpSourceClosure.cmake", source / "cmake/LlmpSourceClosure.cmake")
    (source / "CMakePresets.json").write_text('{}\n')
    (source / "src/CMakeLists.txt").write_text('add_subdirectory(common)\nadd_subdirectory(engine)\nadd_subdirectory(public)\n')
    (source / "src/public/CMakeLists.txt").write_text('add_library(complete STATIC)\ntarget_link_libraries(complete PUBLIC code)\n')
    (source / "src/kernels/CMakeLists.txt").write_text('# fixture\n')
    (source / "src/common/CMakeLists.txt").write_text('add_library(code OBJECT code.cc)\n'
        'target_compile_definitions(code PRIVATE OPTION_VALUE=${OPTION_VALUE})\n'
        'target_include_directories(code PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/chosen")\n'
        'add_library(provider INTERFACE)\n'
        'target_sources(provider INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/provided.cc")\n'
        'target_include_directories(provider INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/fallback")\n'
        'target_link_libraries(code PRIVATE provider)\n'
        'if(LLMP_CUDA)\n target_sources(code PRIVATE device.cu)\nendif()\n')
    (source / "src/common/code.cc").write_text('#include"nested.inl"\nint Value() { return VALUE + OPTION_VALUE; }\n')
    (source / "src/common/nested.inl").write_text('#/**/inc\\\nlude "value.h"\n')
    clock_check = ('constexpr bool ClockEqual(const char* a,const char* b) { while(*a && *a==*b) {++a;++b;} return *a==*b; }\n'
        'static_assert(ClockEqual(__DATE__, "Jan  1 1970"));\n'
        'static_assert(ClockEqual(__TIME__, "00:00:00"));\n'
        'static_assert(ClockEqual(__TIMESTAMP__, "Thu Jan  1 00:00:00 1970"));\n')
    (source / "src/common/value.h").write_text('#define VALUE 1\n'+clock_check)
    for name, value in (("chosen", 3), ("fallback", 4)):
        (source / f"src/common/{name}").mkdir()
        (source / f"src/common/{name}/context.h").write_text(f'#define CONTEXT_VALUE {value}\n')
    (source / "src/common/provided.cc").write_text('#include <context.h>\nint Context() { return CONTEXT_VALUE; }\n')
    (source / "src/common/device.cu").write_text('#include "value.h"\n__global__ void Kernel(int* p) { *p = VALUE; }\n')
    (source / "src/engine/CMakeLists.txt").write_text('add_library(engine STATIC graph.cc unused.cc)\n')
    (source / "src/engine/graph.cc").write_text('int Graph() { return 3; }\n')
    (source / "src/engine/unused.cc").write_text('int Unused() { return 4; }\n')
    (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 4.4.3)
project(compiler_identity_fixture LANGUAGES CXX)
if(LLMP_CUDA)
 enable_language(CUDA)
endif()
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(LLMP_SOURCES_DIR "${PROJECT_SOURCE_DIR}/prepared")
include(cmake/LlmpSourceClosure.cmake)
llmp_identity_freeze_sources()
add_subdirectory(src)
set(output "${PROJECT_BINARY_DIR}/identity.cc")
target_sources(complete PRIVATE "${output}")
llmp_compiler_identity(OUTPUT "${output}" TARGETS code CONSTANT_TARGET complete
  SELECT_TARGET engine SELECT_SOURCE "${PROJECT_SOURCE_DIR}/src/engine/graph.cc")
add_dependencies(complete llmp_compiler_identity_ready)
''')
    def run(*args, success=True, env=None, cwd=None):
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        print(result.stdout, end="", flush=True)
        if success and result.returncode:
            raise RuntimeError(f"fixture command failed ({result.returncode}): {args}")
        if not success and result.returncode == 0:
            raise RuntimeError(f"fixture unexpectedly accepted: {args}")
        return result
    def configure(src=source, out=build, option=5):
        run(cmake, '-G', 'Ninja', '-S', str(src), '-B', str(out),
            '-DCMAKE_TOOLCHAIN_FILE='+toolchain, '-DLLMP_CUDA='+('ON' if cuda else 'OFF'),
            '-DLLMP_PYTHON='+python, '-DCMAKE_BUILD_TYPE=RelWithDebInfo', '-DOPTION_VALUE='+str(option))
    def compile(out=build):
        run(cmake, '--build', str(out), '--target', 'complete', '-j4')
        return json.loads((out / 'compiler_identity_receipt.json').read_text())['identity']
    # Bootstrap an actual pre-repair Ninja dependency database: that launcher
    # recorded all receipt inputs but published only compiler-emitted headers.
    # Upgrade through the configured launcher hash, never by touching objects or
    # editing receipts/dependency metadata to force recompilation.
    launcher = source / 'tools/compiler_identity.py'
    repaired_launcher = launcher.read_text()
    publication = '        write_if_changed(depfile, dependency_rule(operand(args, "-MT"), inputs).encode())'
    assert repaired_launcher.count(publication) == 1
    launcher.write_text(repaired_launcher.replace(publication, '        # Legacy compiler-only dependency rule'))
    configure()
    compiler_config = json.loads((build / 'compiler_identity.json').read_text())
    compiler_alias_fixture(work / 'compiler-alias-control', compiler_config['constant_compiler'], compiler_config['sdk'])
    cache = (build / 'CMakeCache.txt').read_text().splitlines()
    ninja = next(line.split('=', 1)[1] for line in cache if line.startswith('CMAKE_MAKE_PROGRAM:'))
    ninja_dependency_fixture(ninja)
    legacy = compile()
    legacy_objects = [Path(p) for p in (build / 'compiler_objects.txt').read_text().splitlines()]
    legacy_snapshots = {p: IDENTITY.snapshot(p) for p in legacy_objects}
    launcher.write_text(repaired_launcher)
    initial = compile()
    assert initial != legacy, 'launcher recipe update did not change implementation identity'
    assert all(IDENTITY.snapshot(p) != old for p, old in legacy_snapshots.items()), 'legacy object did not recompile through the configured command stamp'
    assert not (build / 'src/engine/CMakeFiles/engine.dir/unused.cc.o').exists()
    assert compile() == initial
    # Re-running real CMake without changed semantics must not chmod the host
    # wrapper or force a recompile. Compare inode/mtime/ctime as well as bytes.
    inventory = [Path(p) for p in (build / 'compiler_objects.txt').read_text().splitlines()]
    outputs = inventory + [build / 'src/public/libcomplete.a']
    wrapper = build / 'toolchain/cuda-host-clang++'
    if cuda:
        outputs.append(wrapper)
    before_reconfigure = {p: IDENTITY.snapshot(p) for p in outputs}
    configure()
    assert compile() == initial, 'same-configuration regeneration changed identity'
    assert {p: IDENTITY.snapshot(p) for p in outputs} == before_reconfigure, 'no-op configure rewrote a compiled input/output'
    # Catalog-only changes re-run provenance validation, never compilation.
    # Include the constant TU and archive, receipts, and semantic config so the
    # check cannot pass by merely producing equivalent replacement objects.
    stable = outputs + list(build.rglob('*.llmp-compile.json')) + [build / 'compiler_identity.json']
    stable += [p for p in build.rglob('*.o') if p not in stable]
    catalog = build / 'compiler_ownership.json'
    catalog_before = IDENTITY.snapshot(catalog)
    stable_before = {p: IDENTITY.snapshot(p) for p in stable}
    report = source / 'unrelated-report.md'
    report.write_text('unrelated authored document\n')
    assert compile() == initial, 'unrelated catalog addition changed implementation identity'
    assert {p: IDENTITY.snapshot(p) for p in stable} == stable_before, 'catalog addition rebuilt a compiled object/input'
    catalog_after = IDENTITY.snapshot(catalog)
    assert catalog_after['sha256'] != catalog_before['sha256'], 'new report was not cataloged'
    audit = json.loads((build / 'compiler_identity_receipt.json').read_text())['ownership']
    assert audit['record'] == catalog_after and Path(audit['archive']).read_bytes() == catalog.read_bytes()
    report.unlink()
    assert compile() == initial, 'unrelated catalog removal changed implementation identity'
    assert {p: IDENTITY.snapshot(p) for p in stable} == stable_before, 'catalog removal rebuilt a compiled object/input'
    assert IDENTITY.snapshot(catalog)['sha256'] == catalog_before['sha256']
    checker = [python, str(source / 'tools/compiler_identity.py'), 'check', '--config',
               str(build / 'compiler_identity.json'), '--objects', str(build / 'compiler_objects.txt'),
               '--output', str(build / 'identity.cc'), '--receipt', str(build / 'compiler_identity_receipt.json')]
    catalog_bytes = catalog.read_bytes()
    for operation in ('remove', 'generated', 'declaration'):
        changed_catalog = json.loads(catalog_bytes)
        header_path = str(source / 'src/common/value.h')
        if operation in ('remove', 'declaration'):
            changed_catalog['owned_inputs'].remove(header_path if operation == 'remove' else str(source / 'CMakeLists.txt'))
        else:
            changed_catalog['declared_generated'].append(header_path)
        catalog.write_text(json.dumps(changed_catalog))
        refused = run(*checker, success=False)
        assert ('outside the frozen authored-source inventory' if operation in ('remove', 'declaration') else
                'generated input has no declared recipe') in refused.stdout
        assert {p: IDENTITY.snapshot(p) for p in stable} == stable_before
        catalog.write_bytes(catalog_bytes)
        run(*checker)
    # Ninja also classifies generated inputs; no catalog edit or object change
    # can hide a new generator edge for an already compiled source-root header.
    graph = build / 'build.ninja'
    original_graph = graph.read_bytes()
    graph.write_bytes(original_graph + ('\nbuild '+str(source / 'src/common/value.h')+
                                       ': CUSTOM_COMMAND\n  COMMAND = true\n').encode())
    refused = run(*checker, success=False)
    assert 'generated input has no declared recipe' in refused.stdout
    graph.write_bytes(original_graph)
    run(*checker)
    # Exercise the full digest with an authority mutation after it has loaded
    # the catalog, while object/source bytes and every receipt remain intact.
    validate = IDENTITY.validated_receipt
    def mutate_catalog_during_digest(obj):
        receipt = validate(obj)
        catalog.write_bytes(catalog_bytes + b' ')
        return receipt
    with mock.patch.object(IDENTITY, 'validated_receipt', side_effect=mutate_catalog_during_digest):
        try:
            IDENTITY.digest(build / 'compiler_identity.json', build / 'compiler_objects.txt',
                            build / 'identity.cc', build / 'compiler_identity_receipt.json', True)
        except IDENTITY.Refusal as error:
            assert 'catalog/build graph changed' in str(error), error
        else:
            raise AssertionError('digest accepted a mid-operation catalog mutation')
    catalog.write_bytes(catalog_bytes)
    run(*checker)
    print('Catalog addition/removal: zero object rebuilds, stable identity; actual dependency removal/reclassification refused', flush=True)
    if cuda:
        # A direct wrapper alteration is not a configured recipe. The new
        # object dependency must enter the launcher and refuse it automatically.
        wrapper.write_bytes(wrapper.read_bytes() + b'\n# unconfigured wrapper change\n')
        result = run(cmake, '--build', str(build), '--target', 'complete', '-j4', success=False)
        assert 'generated input differs from its configured recipe' in result.stdout
        configure()  # production toolchain restores its generated bytes
        assert compile() == initial
    # A legitimate recipe edit must configure/recompile, not reuse old receipts.
    closure = source / 'cmake/LlmpSourceClosure.cmake'
    original_closure = closure.read_text()
    closure.write_text(original_closure + '\n# changed fixture recipe\n')
    assert compile() != initial
    closure.write_text(original_closure)
    assert compile() == initial
    copied, copied_build = work / 'other-source', work / 'other-build'
    shutil.copytree(source, copied)
    configure(copied, copied_build)
    assert compile(copied_build) == initial, 'cross-root implementation identity differs'
    run(cmake, '--build', str(build), '--target', 'clean')
    assert compile() == initial, 'clean rebuild changed implementation identity'
    (source / 'src/common/value.h').write_text('#define VALUE 2\n'+clock_check)
    changed = compile()
    assert changed != initial
    (source / 'src/common/chosen/context.h').write_text('#define CONTEXT_VALUE 7\n')
    selected = compile()
    assert selected != changed
    (source / 'src/common/fallback/context.h').write_text('#define CONTEXT_VALUE 8\n')
    assert compile() == selected, 'unselected include entered compiled identity'
    configure(option=6)
    flags = compile()
    assert flags != selected
    # A stale input cannot pass the identity checker without recompilation.
    (source / 'src/common/value.h').write_text('#define VALUE 9\n'+clock_check)
    checker = [python, str(source / 'tools/compiler_identity.py'), 'check', '--config',
               str(build / 'compiler_identity.json'), '--objects', str(build / 'compiler_objects.txt'),
               '--output', str(build / 'identity.cc'), '--receipt', str(build / 'compiler_identity_receipt.json')]
    run(*checker, success=False)
    current = compile()
    run(*checker)
    commands_file = build / 'compile_commands.json'
    original_commands = commands_file.read_text()
    commands = json.loads(original_commands)
    selected_command = next(entry for entry in commands if entry['file'].endswith('/code.cc'))
    selected_command['command'] += ' -DUNBUILT_CHANGE=1'
    commands_file.write_text(json.dumps(commands))
    run(*checker, success=False)
    commands_file.write_text(original_commands)
    run(*checker)
    selected_obj = Path(selected_command['output'])
    if not selected_obj.is_absolute():
        selected_obj = build / selected_obj
    prior_receipt = IDENTITY.receipt_path(selected_obj)
    actual_receipt = prior_receipt.read_bytes()
    original_object = selected_obj.read_bytes()
    source_control = source / 'src/common/code.cc'
    original_control = source_control.read_text()
    source_control.write_text(original_control+'// dependency-only control\n')
    raw_command = shlex.split(selected_command['command'])
    raw_command += ['-MD', '-MT', str(selected_obj), '-MF', str(selected_obj)+'.d', '-M']
    # This command returns success from Clang without producing an object.
    run(python, str(source / 'tools/compiler_identity.py'), 'compile', '--config',
        str(build / 'compiler_identity.json'), '--', *raw_command, success=False, cwd=build)
    assert selected_obj.read_bytes() == original_object and prior_receipt.read_bytes() == actual_receipt
    run(*checker, success=False)
    source_control.write_text(original_control)
    assert compile() == current
    original_object = selected_obj.read_bytes()
    actual_receipt = prior_receipt.read_bytes()
    for option in ('-fprofile-instr-use=/untracked/profile.profdata',
                   '-fplugin=/untracked/plugin.so', '-Xclang', '-Wp,-include,/untracked/header.h'):
        run(python, str(source / 'tools/compiler_identity.py'), 'compile', '--config',
            str(build / 'compiler_identity.json'), '--', *raw_command[:-1], option,
            success=False, cwd=build)
        assert selected_obj.read_bytes() == original_object and prior_receipt.read_bytes() == actual_receipt
    if cuda:
        device_command = next(entry for entry in commands if entry['file'].endswith('/device.cu'))
        device_obj = Path(device_command['output'])
        if not device_obj.is_absolute():
            device_obj = build / device_obj
        device_args = shlex.split(device_command['command']) + ['-MD', '-MT', str(device_obj),
                                                              '-MF', str(device_obj)+'.d']
        prior = IDENTITY.receipt_path(device_obj).read_bytes()
        for option in ('-Xcompiler=-D,__DATE__="override"', '-Xcompiler=-U,__TIME__',
                       '--define-macro=OTHER=1,__TIMESTAMP__="override"',
                       '-Xcompiler=-Xclang,-load,/untracked/plugin.so',
                       '--compiler-options=-Xclang,-load,/untracked/plugin.so',
                       '--compiler-options', '-Xcompiler=-fprofile-instr-use=/untracked/profile.profdata',
                       '-Xcompiler=-Wp,-include,/untracked/header.h'):
            run(python, str(source / 'tools/compiler_identity.py'), 'compile', '--config',
                str(build / 'compiler_identity.json'), '--', *device_args, option,
                success=False, cwd=build)
            assert IDENTITY.receipt_path(device_obj).read_bytes() == prior
        run(python, str(source / 'tools/compiler_identity.py'), 'compile', '--config',
            str(build / 'compiler_identity.json'), '--', *device_args, '-ccbin',
            '/untracked/host-wrapper', success=False, cwd=build)
        assert IDENTITY.receipt_path(device_obj).read_bytes() == prior
    cmake_file = source / 'CMakeLists.txt'
    original_cmake = cmake_file.read_text()
    for directive in ('target_compile_options(complete PRIVATE -include "${PROJECT_SOURCE_DIR}/src/common/value.h")',
                      'target_compile_options(complete PRIVATE "-Wp,-include,${PROJECT_SOURCE_DIR}/src/common/value.h")',
                      'target_compile_definitions(complete PRIVATE ModuleSourcesDigest=Unexpected)',
                      'target_precompile_headers(code PRIVATE "${PROJECT_SOURCE_DIR}/src/common/value.h")',
                      'set_property(TARGET code PROPERTY CXX_COMPILER_LAUNCHER /usr/bin/env)'):
        cmake_file.write_text(original_cmake.replace('set(output',
                                                     directive+'\nset(output'))
        if directive.startswith(('target_compile_options', 'target_compile_definitions')):
            configure(option=6)
            run(cmake, '--build', str(build), '--target', 'complete', '-j4', success=False)
        else:
            run(cmake, '-S', str(source), '-B', str(build), success=False)
        cmake_file.write_text(original_cmake)
        configure(option=6)
    code = source / 'src/common/code.cc'
    original_code = code.read_text()
    cmake_file.write_text(original_cmake.replace('set(output',
        'file(WRITE "${PROJECT_BINARY_DIR}/unknown.h" "#define GENERATED 1\\n")\n'
        'target_include_directories(code PRIVATE "${PROJECT_BINARY_DIR}")\n'
        'set(output'))
    code.write_text('#include "unknown.h"\n'+original_code)
    configure(option=6)
    run(cmake, '--build', str(build), '--target', 'complete', '-j4', success=False)
    code.write_text(original_code)
    cmake_file.write_text(original_cmake)
    configure(option=6)
    assert compile() == current
    # Configure cannot rewrite authored inputs, even outside the build root.
    for generated_path, content in ((source / 'src/common/value.h', '#define VALUE 99'),
                                    (source / 'src/common/new_configured.h', '#define NEW 1')):
        cmake_file.write_text(original_cmake.replace('set(output',
            'file(WRITE "'+str(generated_path)+'" "'+content+'")\nset(output'))
        run(cmake, '-S', str(source), '-B', str(build), success=False)
        if generated_path.name == 'value.h':
            generated_path.write_text('#define VALUE 9\n'+clock_check)
        else:
            generated_path.unlink(missing_ok=True)
        cmake_file.write_text(original_cmake)
        configure(option=6)
    # Declared GENERATED source-root headers need a recipe even when preexisting.
    cmake_file.write_text(original_cmake.replace('set(output',
        'set_source_files_properties("${PROJECT_SOURCE_DIR}/src/common/value.h" PROPERTIES GENERATED TRUE)\nset(output'))
    configure(option=6)
    run(cmake, '--build', str(build), '--target', 'complete', '-j4', success=False)
    cmake_file.write_text(original_cmake)
    configure(option=6)
    assert compile() == current
    # Ambient compiler/loader overrides are refused before compiler execution.
    environment = os.environ.copy()
    environment['CPATH'] = str(source)
    run(cmake, '--build', str(build), '--target', 'clean')
    run(cmake, '--build', str(build), '--target', 'complete', '-j4', success=False, env=environment)
    assert compile() == current
    print('Compiler identity native fixture PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--cmake')
    parser.add_argument('--toolchain')
    parser.add_argument('--python', default=os.sys.executable)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--ninja', help='run only the actual Ninja dependency consumer control')
    parser.add_argument('--compiler-alias', action='store_true')
    parser.add_argument('--compiler')
    parser.add_argument('--sdk')
    args = parser.parse_args()
    if args.native:
        native_fixture(args.root, args.work, args.cmake, args.toolchain, args.python, args.cuda)
    elif args.ninja:
        ninja_dependency_fixture(args.ninja)
    elif args.compiler_alias:
        compiler_alias_fixture(args.work, args.compiler, args.sdk)
    else:
        unittest.main(argv=[os.sys.argv[0]])
