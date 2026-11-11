#!/usr/bin/env python3
"""Download only the official OLMo tokenizer, pinned to a resolved revision."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "allenai/Olmo-3.1-32B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=ROOT / "models/olmo-tokenizer")
    args = parser.parse_args()
    revision = HfApi().model_info(MODEL_ID, revision=args.revision).sha
    snapshot_download(MODEL_ID, revision=revision, local_dir=args.output,
                      cache_dir=ROOT / ".cache/huggingface",
                      allow_patterns=["tokenizer.json", "tokenizer_config.json",
                                      "special_tokens_map.json", "vocab.json",
                                      "merges.txt", "chat_template.jinja", "config.json"])
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in args.output.iterdir() if p.is_file() and p.name != "download.json"}
    (args.output / "download.json").write_text(json.dumps(
        {"model_id": MODEL_ID, "revision": revision, "sha256": files}, indent=2) + "\n")
    print(f"Tokenizer: {args.output}; revision: {revision}")


if __name__ == "__main__":
    main()
