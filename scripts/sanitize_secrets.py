"""
One-shot sanitiser — replaces every hardcoded API key and /PlantVQA/ absolute
path in this repository with an environment-variable read.  Run once before
pushing to GitHub.

Usage:    python scripts/sanitize_secrets.py [--dry-run]
"""

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS = list((REPO_ROOT / "eval").glob("*.py"))

# Literal patterns we need to replace.  Order matters — longer patterns first.
REPLACEMENTS = [
    # Assignments: replace RHS with env-var read
    (re.compile(r'(GEMINI_API_KEY\s*=\s*)"AIzaSy[A-Za-z0-9_\-]{20,}"'),
     r'\1os.environ["GEMINI_API_KEY"]'),
    (re.compile(r'(OPENROUTER_API_KEY\s*=\s*)"sk-or-v1-[A-Za-z0-9]{40,}"'),
     r'\1os.environ["OPENROUTER_API_KEY"]'),
    (re.compile(r'(OPENAI_API_KEY\s*=\s*)"sk-proj-[A-Za-z0-9_\-]{40,}"'),
     r'\1os.environ["OPENAI_API_KEY"]'),
    (re.compile(r'(OPENAI_API_KEY\s*=\s*)"sk-[A-Za-z0-9]{40,}"'),
     r'\1os.environ["OPENAI_API_KEY"]'),

    # Absolute image / dataset paths.  We leave the variable name, just point
    # it at the in-repo or env-var location.
    (re.compile(r'"/media/rmedu/NewVolume/sns/PlantVQA/images"'),
     r'os.environ.get("PLANTINQUIRY_IMAGE_DIR", "images")'),
    (re.compile(r'"/media/rmedu-4090/Storage_21/PlantVQA"'),
     r'os.environ.get("PLANTINQUIRY_HOME", ".")'),
    (re.compile(r'"/media/rmedu/NewVolume/sns/PlantVQA"'),
     r'os.environ.get("PLANTINQUIRY_HOME", ".")'),
    (re.compile(r'"/media/rmedu/NewVolume/sns/PlantInquiryVQA"'),
     r'os.environ.get("PLANTINQUIRY_HOME", ".")'),
    (re.compile(r'"/media/rmedu/NewVolume/sns/PlantVQA/eval"'),
     r'os.path.join(os.environ.get("PLANTINQUIRY_HOME", "."), "eval")'),
    (re.compile(r'"/media/rmedu/NewVolume/sns/PlantVQA/dataset_splits/plantvqa_golden_test_500img\.csv"'),
     r'os.path.join(os.environ.get("PLANTINQUIRY_HOME", "."), "dataset/plantinquiryvqa_test_subset.csv")'),
]


def ensure_os_import(src: str) -> str:
    """Make sure the file imports `os` (most already do)."""
    if re.search(r"^\s*import\s+os\b", src, re.M):
        return src
    # Insert `import os` after the top-of-file docstring / first non-blank line
    lines = src.splitlines()
    insert_at = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith(("#", "\"\"\"", "'''")) or not ln.strip():
            continue
        insert_at = i
        break
    lines.insert(insert_at, "import os")
    return "\n".join(lines) + ("\n" if src.endswith("\n") else "")


def process_file(path: Path, dry_run: bool):
    src = path.read_text()
    new = src
    n_changes = 0
    for rx, repl in REPLACEMENTS:
        new, k = rx.subn(repl, new)
        n_changes += k
    if n_changes and "os.environ" in new:
        new = ensure_os_import(new)
    if new != src:
        action = "DRY" if dry_run else "fix"
        print(f"[{action}] {path.relative_to(REPO_ROOT)}   ({n_changes} replacements)")
        if not dry_run:
            path.write_text(new)
    return n_changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    total = 0
    for p in TARGETS:
        total += process_file(p, args.dry_run)
    print(f"\n{'Would change' if args.dry_run else 'Changed'}: {total} occurrences across {len(TARGETS)} files.")


if __name__ == "__main__":
    main()
