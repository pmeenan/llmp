# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""A minimal Hugging Face Hub client for the model archive (docs/model-archive.md).

It speaks exactly the two requests the native `llmp install` will make
(D-114), with the standard library only:

- `GET {endpoint}/api/models/{repo}/revision/{revision}?blobs=true`: the
  revision's commit, card data, gated flag and every file with its size,
  Git blob ID and, for LFS files, SHA-256;
- `GET {endpoint}/{repo}/resolve/{commit}/{path}` with an optional `Range`,
  following redirects to the storage backend.

The token comes from the environment only (HF_TOKEN, then the older
HUGGING_FACE_HUB_TOKEN); mise exports it from the repository's ignored
/.env. It is sent as an unredirected header, so it never reaches the host
a resolve request redirects to, and `scrub` removes it from any text this
module or its callers print.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://huggingface.co"
INFO_CAP = 64 << 20  # a revision's metadata; the largest repositories' lists are a few MB
TIMEOUT = 60  # seconds without progress before a request fails and is retried
USER_AGENT = "llmp-archive/1"


class HubError(Exception):
    """A Hub request failed. `status` is the HTTP status, or None for a network error."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


TOKEN_RE = re.compile(r"[\x21-\x7e]+")


def token_from_env(environ=None) -> str | None:
    environ = os.environ if environ is None else environ
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = environ.get(name, "").strip()
        if value:
            if not TOKEN_RE.fullmatch(value):
                # Never echo the value: an HTTP library's own complaint about a bad header would.
                raise HubError(f"{name} has spaces, line breaks or non-ASCII characters, which a token "
                               "cannot; check the repository's .env")
            return value
    return None


class Hub:
    def __init__(self, endpoint: str = ENDPOINT, token: str | None = None, opener=None):
        self.endpoint = endpoint.rstrip("/")
        self._token = token
        self._opener = opener or urllib.request.build_opener()

    def scrub(self, text: str) -> str:
        """`text` with the token, if any, replaced; for every message that may echo a request."""
        return text.replace(self._token, "<token>") if self._token else text

    def _request(self, url: str, headers: dict | None = None) -> urllib.request.Request:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
        if self._token:
            # Not forwarded on redirects (urllib copies only redirected headers),
            # so the storage backend a resolve redirects to never sees it.
            request.add_unredirected_header("Authorization", f"Bearer {self._token}")
        return request

    def _open(self, request: urllib.request.Request, what: str):
        try:
            return self._opener.open(request, timeout=TIMEOUT)
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            hint = ""
            if status == 404:
                hint = " (no such repository or revision, or it is private and needs HF_TOKEN)"
            if status in (401, 403):
                hint = (" (gated or private: accept the license on the Hub and put HF_TOKEN in the"
                        " repository's .env)" if not self._token else " (the token has no access)")
            raise HubError(self.scrub(f"{what}: HTTP {status}{hint}"), status,
                           retryable=status == 429 or status >= 500) from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
            raise HubError(self.scrub(f"{what}: {error}"), None, retryable=True) from None

    def revision_info(self, repo: str, revision: str) -> dict:
        url = (f"{self.endpoint}/api/models/{urllib.parse.quote(repo, safe='/')}/revision/"
               f"{urllib.parse.quote(revision, safe='')}?blobs=true")
        with self._open(self._request(url, {"Accept": "application/json"}), f"{repo}@{revision}") as response:
            data = response.read(INFO_CAP + 1)
        if len(data) > INFO_CAP:
            raise HubError(f"{repo}@{revision}: metadata over {INFO_CAP} bytes")
        try:
            info = json.loads(data)
        except ValueError:
            raise HubError(f"{repo}@{revision}: metadata is not JSON") from None
        if not isinstance(info, dict):
            raise HubError(f"{repo}@{revision}: metadata is not an object")
        return info

    def open_file(self, repo: str, commit: str, path: str, start: int = 0):
        """A readable response for `path` from byte `start`; its `start` attribute says
        where it really begins (0 when the server ignored the range)."""
        url = (f"{self.endpoint}/{urllib.parse.quote(repo, safe='/')}/resolve/{commit}/"
               f"{urllib.parse.quote(path, safe='/')}")
        headers = {"Range": f"bytes={start}-"} if start else {}
        response = self._open(self._request(url, headers), f"{repo}@{commit[:8]}/{path}")
        begins = 0
        if start and response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            try:
                begins = int(content_range.split(" ", 1)[1].split("-", 1)[0])
            except (IndexError, ValueError):
                response.close()
                raise HubError(f"{path}: unreadable Content-Range", retryable=True) from None
            if begins != start:
                response.close()
                raise HubError(f"{path}: range began at {begins}, not {start}", retryable=True)
        response.start = begins
        return response
