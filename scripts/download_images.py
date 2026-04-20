"""
Download the PlantInquiryVQA image corpus (~3.5 GB, 24,950 files) from the
HuggingFace Hub into the local `images/` folder.

Usage:
    python scripts/download_images.py                  # default target: ./images/
    python scripts/download_images.py --to /tmp/pivqa  # custom location
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_REPO = os.environ.get("PLANTINQUIRY_HF_REPO", "<user>/PlantInquiryVQA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default=str(REPO_ROOT / "images"),
                    help="Destination directory (default: ./images)")
    ap.add_argument("--repo", default=HF_REPO,
                    help="HuggingFace dataset repo (override with PLANTINQUIRY_HF_REPO)")
    args = ap.parse_args()

    target = Path(args.to).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install huggingface_hub first:  pip install huggingface_hub")

    print(f"Downloading {args.repo} → {target}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=["images/*", "*.csv", "*.json"],
    )
    print("Done.")


if __name__ == "__main__":
    main()
