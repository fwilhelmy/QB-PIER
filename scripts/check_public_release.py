
"""Conservative pre-release scan of the current file tree (not Git history)."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BLOCKED_DIRS = {"experiments", ".git", "logs", "outputs", "data", "wandb", "__pycache__", ".venv"}
BLOCKED_SUFFIXES = {".ipynb", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".parquet", ".pkl", ".pickle", ".sqlite", ".h5", ".hdf5"}
BLOCKED_NAMES = {".env", "HELPER.md", "queue.txt"}
# Heuristics: review any matches. Never print a detected secret's value.
CONTENT_PATTERNS = (
    ("webhook URL", re.compile(r"https?://(?:discord(?:app)?\.com)/api/webhooks/\d+/[A-Za-z0-9_-]+", re.I)),
    ("private key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    ("personal home path", re.compile(r"(?:[A-Za-z]:\\Users\\|/(?:Users|home)/)(?!<|example|user/)", re.I)),
    ("ÉTS cluster login", re.compile(r"(?:rep" + r"lica\.logti\.etsmtl\.ca|AP" + r"09650)", re.I)),
    ("likely GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
)
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".")
    root = Path(parser.parse_args().path).resolve()
    issues = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        parts = rel.parts
        if any(part in BLOCKED_DIRS for part in parts):
            # Ignore .git and local working artifacts when checking an existing clone.
            # Public content should be inspected using a fresh source checkout.
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in BLOCKED_SUFFIXES or path.name in BLOCKED_NAMES:
            issues.append((rel.as_posix(), "forbidden file type/name"))
            continue
        if path.stat().st_size > 2_000_000:
            issues.append((rel.as_posix(), "unexpected >2 MB source file"))
            continue
        try:
            contents = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            issues.append((rel.as_posix(), "unreadable/non-text file; manual review"))
            continue
        for label, pattern in CONTENT_PATTERNS:
            if pattern.search(contents):
                issues.append((rel.as_posix(), label))
    for path, label in issues:
        print(f"REVIEW {path}: {label}")
    if issues:
        print(f"Release guard: {len(issues)} issue(s) found. Inspect before publishing.")
        return 1
    print("Release guard: no flagged working-tree files. History and IP not covered.")
    return 0
if __name__ == "__main__":
    sys.exit(main())
