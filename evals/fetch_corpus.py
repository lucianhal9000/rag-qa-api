"""Download and verify the frozen eval corpus.

Usage:
    python evals/fetch_corpus.py

First run: sha256 fields in corpus_manifest.yaml are empty, so files are
downloaded and their hashes printed. Paste those hashes into the manifest.
Every later run verifies against them and fails loudly on any mismatch.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MANIFEST = HERE / "corpus_manifest.yaml"
CORPUS_DIR = HERE / "corpus"
UA = {"User-Agent": "rag-qa-api-eval-harness/1.0"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        while chunk := resp.read(65536):
            out.write(chunk)


def main() -> int:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    unpinned: list[tuple[str, str]] = []
    failures: list[str] = []

    for doc in manifest["documents"]:
        dest = CORPUS_DIR / f"{doc['id']}.{doc['format']}"
        expected = (doc.get("sha256") or "").strip()

        if not dest.exists():
            print(f"downloading {doc['id']} ...")
            try:
                download(doc["url"], dest)
            except Exception as exc:
                dest.unlink(missing_ok=True)
                failures.append(f"{doc['id']}: download failed: {exc}")
                continue

        actual = sha256_of(dest)

        if not expected:
            unpinned.append((doc["id"], actual))
        elif actual != expected:
            failures.append(
                f"{doc['id']}: hash mismatch\n"
                f"    expected {expected}\n"
                f"    actual   {actual}\n"
                f"    The corpus has drifted. Delete the file and re-fetch, or "
                f"pin the source URL to a specific version."
            )
        else:
            print(f"ok  {doc['id']}  ({dest.stat().st_size:,} bytes)")

    if unpinned:
        print("\nPaste these into corpus_manifest.yaml:\n")
        for doc_id, digest in unpinned:
            print(f"  {doc_id}: {digest}")

    if failures:
        print("\nFAILURES:\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
