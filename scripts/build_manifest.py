"""Write a SHA-256 manifest for the public-safe repository contents."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "run_outputs", "__pycache__"}
SKIP_NAMES = {"MANIFEST.sha256"}


def main() -> None:
    rows = []
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    for relative in sorted(set(filter(None, names))):
        path = ROOT / relative
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        relative_parts = set(path.relative_to(ROOT).parts)
        if relative_parts & SKIP_PARTS:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        digest = digest.hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    destination = ROOT / "MANIFEST.sha256"
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"manifest_entries={len(rows)}")


if __name__ == "__main__":
    main()
