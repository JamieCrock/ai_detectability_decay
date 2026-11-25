"""Download the HC3 (Human ChatGPT Comparison Corpus) English split.

HC3 is released by Hello-SimpleAI under CC BY-SA 4.0:
https://huggingface.co/datasets/Hello-SimpleAI/HC3

Usage:  python scripts/download_hc3.py [--subset all|finance|medicine|open_qa|reddit_eli5|wiki_csai]
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{subset}.jsonl"
SUBSETS = ("all", "finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", choices=SUBSETS, default="all")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    out = Path(args.out_dir) / f"hc3_{args.subset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} already exists - skipping download")
        return
    url = BASE.format(subset=args.subset)
    print(f"Downloading {url}")
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.rename(out)
    print(f"Saved to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
