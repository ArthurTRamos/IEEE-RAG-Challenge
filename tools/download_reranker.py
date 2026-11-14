#!/usr/bin/env python3
"""Download a revision-pinned local reranker including its modular scoring head."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='cross-encoder/ettin-reranker-68m-v1')
    p.add_argument('--revision',default='d166fa88ddde3c42bc3ee92f7df476d941c8204a')
    p.add_argument('--output',type=Path,default=ROOT/'models/ettin-reranker-68m-v1')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use a new model directory')
    info = HfApi().model_info(args.model,revision=args.revision,files_metadata=True)
    files = [f for f in info.siblings if not f.rfilename.startswith(('onnx/','openvino/'))
             and (f.rfilename.endswith(('.json','.safetensors')) or f.rfilename=='README.md')]
    size = sum(f.size for f in files)
    if size > 500_000_000:
        p.error('Selected model resources exceed 500 MB')
    snapshot_download(args.model,revision=info.sha,local_dir=args.output,
                      cache_dir=ROOT/'.cache/huggingface',allow_patterns=[f.rfilename for f in files])
    hashes = {f.rfilename:hashlib.sha256((args.output/f.rfilename).read_bytes()).hexdigest() for f in files}
    (args.output/'provenance.json').write_text(json.dumps(
        {'model_id':args.model,'revision':info.sha,'size_bytes':size,'files':hashes},indent=2)+'\n')
    print(args.output,info.sha,size)


if __name__=='__main__':
    main()
