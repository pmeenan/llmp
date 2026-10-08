# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests D-062's version rules on synthetic Git repositories.

    python3 derive_test.py --repo REPO --cmake CMAKE --work DIR

Drives the real cmake/version/derive.cmake and cmake/version/stamp.cmake with
the build's CMake: dev versions counted from the root or the highest release
tag, the tree's modified state, release tags and the tags that are not,
refused versions, shallow clones, unreadable metadata, the environment's Git
variables, linked worktrees, the Debian mapping's ordering, and the stamp
rewriting its outputs only when they change.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import unittest

ARGS = None

# Fixture repositories ignore the user's and the system's Git configuration
# (signing, hooks, default branch) and get fixed identities and dates.
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}

DRIVER = """\
cmake_minimum_required(VERSION 4.4.3)
include("${REPO}/cmake/version/derive.cmake")
llmp_version_derive(v SOURCE_DIR "${SOURCE_DIR}" PROJECT_VERSION "${PROJECT_VERSION}")
llmp_version_json(json v)
file(WRITE "${OUT}" "${json}")
"""


def environment(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_ENV, **extra)
    return env


class Repo:
    def __init__(self, path: pathlib.Path, init: bool = True):
        self.path = path
        self.commits = 0
        if init:
            path.mkdir(parents=True)
            self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.path), *args], env=environment(), check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(self, name: str = "file.txt", text: str | None = None) -> None:
        self.commits += 1
        (self.path / name).write_text(text or f"change {self.commits}\n")
        self.git("add", name)
        self.git("commit", "-q", "-m", f"change {self.commits}")

    def tag(self, name: str, annotated: bool = True) -> None:
        if annotated:
            self.git("tag", "-a", "-m", name, name)
        else:
            self.git("tag", name)

    def head(self) -> str:
        return self.git("rev-parse", "HEAD")

    def short(self) -> str:
        return self.git("rev-parse", "--short=12", "HEAD")


class Case(unittest.TestCase):
    def setUp(self):
        self.work = ARGS.work / self.id().rsplit(".", 1)[-1]
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True)
        self.driver = self.work / "driver.cmake"
        self.driver.write_text(DRIVER)

    def repo(self, name: str = "repo") -> Repo:
        return Repo(self.work / name)

    def run_derive(self, source: pathlib.Path, version: str, env=None) -> subprocess.CompletedProcess:
        out = self.work / "version.json"
        out.unlink(missing_ok=True)
        cmd = [str(ARGS.cmake), f"-DREPO={ARGS.repo}", f"-DSOURCE_DIR={source}", f"-DPROJECT_VERSION={version}",
               f"-DOUT={out}", "-P", str(self.driver)]
        result = subprocess.run(cmd, env=env or environment(), capture_output=True, text=True)
        result.version = json.loads(out.read_text()) if result.returncode == 0 else None
        return result

    def derive(self, source: pathlib.Path, version: str, env=None) -> dict:
        result = self.run_derive(source, version, env)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.version

    def refuse(self, source: pathlib.Path, version: str, message: str) -> None:
        result = self.run_derive(source, version)
        self.assertNotEqual(result.returncode, 0, result.version)
        self.assertIn(message, " ".join(result.stderr.split()))

    def dev(self, repo: Repo, version: str, distance: int, dirty: bool = False) -> dict:
        product = f"{version}-dev.{distance}+g{repo.short()}" + (".dirty" if dirty else "")
        return {"product": product, "debian": product.replace("-", "~") + "-1", "commit": repo.head(),
                "modified": dirty, "origin": "git"}


class Derive(Case):
    def test_no_git_metadata(self):
        tree = self.work / "export"
        tree.mkdir()
        self.assertEqual(self.derive(tree, "0.3.0"), {
            "product": "0.3.0-dev+unknown", "debian": "0.3.0~dev+unknown-1", "commit": None,
            "modified": False, "origin": "none"})

    def test_no_tag_counts_from_the_root(self):
        repo = self.repo()
        for _ in range(3):
            repo.commit()
        self.assertEqual(self.derive(repo.path, "0.1.0"), self.dev(repo, "0.1.0", 3))
        self.assertEqual(len(repo.short()), 12)

    def test_modified_tree(self):
        repo = self.repo()
        repo.commit(".gitignore", "ignored.txt\n")
        (repo.path / "ignored.txt").write_text("ignored\n")
        self.assertFalse(self.derive(repo.path, "0.1.0")["modified"], "an ignored file")
        (repo.path / "untracked.txt").write_text("new\n")
        self.assertEqual(self.derive(repo.path, "0.1.0"), self.dev(repo, "0.1.0", 1, dirty=True))
        (repo.path / "untracked.txt").unlink()
        (repo.path / ".gitignore").write_text("changed\n")
        self.assertTrue(self.derive(repo.path, "0.1.0")["modified"], "a changed tracked file")
        repo.git("add", ".gitignore")
        self.assertTrue(self.derive(repo.path, "0.1.0")["modified"], "a staged change")
        repo.git("reset", "-q", "--hard")
        self.assertFalse(self.derive(repo.path, "0.1.0")["modified"])

    def test_release_tag(self):
        repo = self.repo()
        repo.commit()
        repo.commit()
        repo.tag("v0.1.0")
        self.assertEqual(self.derive(repo.path, "0.1.0"), {
            "product": "0.1.0", "debian": "0.1.0-1", "commit": repo.head(), "modified": False, "origin": "git"})
        (repo.path / "file.txt").write_text("edited\n")
        self.assertEqual(self.derive(repo.path, "0.1.0"), self.dev(repo, "0.1.0", 0, dirty=True))

    def test_release_tag_needs_its_own_version(self):
        repo = self.repo()
        repo.commit()
        repo.tag("v0.1.0")
        self.refuse(repo.path, "0.2.0", "HEAD is the release tag v0.1.0, but project(VERSION) is 0.2.0")

    def test_next_version_can_be_checked_before_commit(self):
        repo = self.repo()
        repo.commit("CMakeLists.txt", "project(llmp VERSION 0.1.0)\n")
        repo.tag("v0.1.0")
        (repo.path / "CMakeLists.txt").write_text("project(llmp VERSION 0.1.1)\n")
        self.assertEqual(self.derive(repo.path, "0.1.1"), self.dev(repo, "0.1.1", 0, dirty=True))
        repo.git("add", "CMakeLists.txt")
        self.assertEqual(self.derive(repo.path, "0.1.1"), self.dev(repo, "0.1.1", 0, dirty=True))
        self.refuse(repo.path, "0.0.9", "below the release tag v0.1.0")

    def test_after_a_release_the_version_rises(self):
        repo = self.repo()
        repo.commit()
        repo.tag("v0.1.0")
        repo.commit()
        repo.commit()
        for version in ("0.1.0", "0.0.9"):
            self.refuse(repo.path, version, f"project(VERSION) is {version}, not above the last release tag v0.1.0")
        self.assertEqual(self.derive(repo.path, "0.1.1"), self.dev(repo, "0.1.1", 2))
        self.assertEqual(self.derive(repo.path, "1.0.0"), self.dev(repo, "1.0.0", 2))

    def test_tags_that_are_not_releases(self):
        repo = self.repo()
        repo.commit()
        repo.git("checkout", "-q", "-b", "other")
        repo.commit("other.txt")
        repo.tag("v9.0.0")  # on a commit HEAD does not contain
        repo.git("checkout", "-q", "main")
        repo.commit()
        repo.tag("v0.5.0", annotated=False)  # lightweight
        for name in ("v0.2.0-rc.1", "v01.0.0", "v1.0", "v1.0.0.0", "0.4.0", "release/v0.3.0", "v1.0.0+build",
                     "v9.1.0;x"):  # `;` is CMake's list separator
            repo.tag(name)
        self.assertEqual(self.derive(repo.path, "0.1.0"), self.dev(repo, "0.1.0", 2))

    def test_highest_release_tag_counts(self):
        repo = self.repo()
        repo.commit()
        repo.tag("v0.3.0")
        repo.commit()
        repo.tag("v0.2.0")  # a lower tag on a later commit
        repo.commit()
        self.refuse(repo.path, "0.3.0", "not above the last release tag v0.3.0")
        self.assertEqual(self.derive(repo.path, "0.3.1"), self.dev(repo, "0.3.1", 2))

    def test_merged_release_branch(self):
        repo = self.repo()
        repo.commit()
        repo.git("checkout", "-q", "-b", "release")
        repo.commit("release.txt")
        repo.tag("v0.1.0")
        repo.git("checkout", "-q", "main")
        repo.commit()
        repo.git("merge", "-q", "--no-edit", "release")
        # Commits HEAD has that v0.1.0 does not: main's commit and the merge.
        self.assertEqual(self.derive(repo.path, "0.2.0"), self.dev(repo, "0.2.0", 2))

    def test_shallow_clone_is_refused(self):
        repo = self.repo()
        repo.commit()
        repo.commit()
        clone = self.work / "clone"
        subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{repo.path}", str(clone)],
                       env=environment(), check=True)
        self.refuse(clone, "0.1.0", "is a shallow clone")

    def test_unreadable_metadata_is_refused(self):
        tree = self.work / "broken"
        tree.mkdir()
        (tree / ".git").write_text("")
        self.refuse(tree, "0.1.0", "The version comes from Git (D-062), but `git rev-parse")

    def test_linked_worktree(self):
        repo = self.repo()
        repo.commit()
        repo.git("worktree", "add", "-q", "-b", "side", str(self.work / "side"))
        side = Repo(self.work / "side", init=False)
        side.commit("side.txt")
        self.assertEqual(self.derive(side.path, "0.1.0"), self.dev(side, "0.1.0", 2))

    def test_environment_does_not_redirect_git(self):
        repo = self.repo()
        repo.commit()
        repo.commit()
        other = self.repo("other")
        other.commit()
        other.commit()
        other.commit()
        env = environment(GIT_DIR=str(other.path / ".git"), GIT_WORK_TREE=str(other.path),
                          GIT_INDEX_FILE=str(other.path / ".git" / "index"))
        self.assertEqual(self.derive(repo.path, "0.1.0", env), self.dev(repo, "0.1.0", 2))
        # A graft file (the environment's or the repository's) that makes HEAD
        # a root, or a shallow file that names none, would change N.
        grafts = self.work / "grafts"
        grafts.write_text(repo.head() + "\n")
        for env in (environment(GIT_GRAFT_FILE=str(grafts)), environment(GIT_SHALLOW_FILE=str(grafts))):
            self.assertEqual(self.derive(repo.path, "0.1.0", env), self.dev(repo, "0.1.0", 2))
        shutil.copy(grafts, repo.path / ".git" / "info" / "grafts")
        self.assertEqual(self.derive(repo.path, "0.1.0"), self.dev(repo, "0.1.0", 2))

    def test_project_version_form(self):
        tree = self.work / "export"
        tree.mkdir()
        for version in ("0.1", "01.0.0", "0.1.0.0", "0.1.0-rc.1", "v0.1.0"):
            self.refuse(tree, version, "the product version is MAJOR.MINOR.PATCH without leading zeros")

    @unittest.skipUnless(shutil.which("dpkg"), "needs dpkg")
    def test_debian_versions_sort_as_releases_do(self):
        ordered = ["0.1.0~dev+unknown-1", "0.1.0~dev.0+g0123456789ab.dirty-1", "0.1.0~dev.9+gffffffffffff-1",
                   "0.1.0~dev.10+g000000000000-1", "0.1.0-1", "0.1.1~dev.1+g0123456789ab-1", "0.1.1-1"]
        for lower, higher in zip(ordered, ordered[1:]):
            result = subprocess.run(["dpkg", "--compare-versions", lower, "lt", higher])
            self.assertEqual(result.returncode, 0, f"{lower} < {higher}")


class Stamp(Case):
    def stamp(self, repo: Repo, sdk: str = "x86_64-0123456789abcdef") -> subprocess.CompletedProcess:
        cmd = [str(ARGS.cmake), f"-DSOURCE_DIR={repo.path}", "-DPROJECT_VERSION=0.1.0",
               f"-DOUTPUT={self.output}", f"-DRECEIPT={self.receipt}", "-DLICENSE_PROFILE=core",
               f"-DSDK={sdk}", "-DTARGET=x86_64-linux-gnu", "-P", str(ARGS.repo / "cmake/version/stamp.cmake")]
        return subprocess.run(cmd, env=environment(), capture_output=True, text=True)

    def setUp(self):
        super().setUp()
        self.output = self.work / "build_info.cc"
        self.receipt = self.work / "llmp-receipt.json"
        self.receipt.write_text(json.dumps({"schema": 1, "version": {"product": "stale"}, "sdk": "kept"}))

    def test_outputs_follow_the_checkout(self):
        repo = self.repo()
        repo.commit()
        result = self.stamp(repo)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt["version"], self.derive(repo.path, "0.1.0"))
        self.assertEqual(receipt["sdk"], "kept")
        source = self.output.read_text()
        self.assertIn(f'.version = "0.1.0-dev.1+g{repo.short()}",', source)
        self.assertIn(f'.commit = "{repo.head()}",', source)
        self.assertIn(".modified = false,", source)
        self.assertIn('.sdk = "x86_64-0123456789abcdef",', source)

        # Unchanged: neither file is rewritten, so nothing rebuilds.
        times = (self.output.stat().st_mtime_ns, self.receipt.stat().st_mtime_ns)
        os.utime(self.output, ns=(1, 1))
        os.utime(self.receipt, ns=(1, 1))
        self.assertEqual(self.stamp(repo).returncode, 0)
        self.assertEqual((self.output.stat().st_mtime_ns, self.receipt.stat().st_mtime_ns), (1, 1), times)

        # A commit with no configure in between reaches both.
        repo.commit()
        self.assertEqual(self.stamp(repo).returncode, 0)
        self.assertIn(f'.version = "0.1.0-dev.2+g{repo.short()}",', self.output.read_text())
        self.assertEqual(json.loads(self.receipt.read_text())["version"]["commit"], repo.head())

    def test_refuses_what_a_build_identity_never_holds(self):
        repo = self.repo()
        repo.commit()
        result = self.stamp(repo, sdk='x"; int evil;')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("internal error: SDK is", result.stderr)
        self.assertFalse(self.output.exists())

    def test_needs_the_receipt(self):
        repo = self.repo()
        repo.commit()
        self.receipt.unlink()
        result = self.stamp(repo)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is missing; configure writes it", " ".join(result.stderr.split()))


def main() -> int:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--cmake", type=pathlib.Path, required=True)
    parser.add_argument("--work", type=pathlib.Path, required=True)
    ARGS, rest = parser.parse_known_args()
    result = unittest.main(argv=[sys.argv[0], *rest], exit=False, verbosity=2).result
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
