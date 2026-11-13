#!/usr/bin/env python3
"""Derive a reviewed bundle from validated v1 artifacts, without network calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.dbpedia_builder import apply_redirect_review, digest, publish, validate_bundle


def revise(source, output, review):
    manifest = validate_bundle(source)
    rows = lambda filename: [json.loads(l) for l in (source/filename).read_text(encoding='utf-8').splitlines()]
    # Only consume the exact N-Triples subset emitted and validated by this builder.
    triples = []
    for line in (source/'domain_kg.ttl').read_text(encoding='utf-8').splitlines():
        subject, predicate, obj = line[:-2].split(' ', 2)
        triples.append((subject[1:-1], predicate[1:-1], obj))
    data = {'cards': rows('kg_rag_chunks.jsonl'), 'edges': rows('edges.jsonl'), 'triples': triples,
            'seeds': manifest['seeds'], 'missing_metadata_uris': manifest['missing_metadata_uris'],
            'derivation': {'source_manifest_sha256': digest((source/'manifest.json').read_bytes()),
                           'source_path': str(source), 'revision_code_sha256': digest(Path(__file__).read_bytes()),
                           'builder_code_sha256': digest((Path(__file__).parent/'dbpedia_builder.py').read_bytes())}}
    revised = apply_redirect_review(data, review)
    config = {**manifest['config'], 'redirect_review': review}
    return publish(output, revised, config, manifest['cache_namespace'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path, default=Path('notebooks/dbpedia-redirect-review.json'))
    args = parser.parse_args()
    print(revise(args.source, args.output, json.loads(args.review.read_text(encoding='utf-8'))))


if __name__ == '__main__':
    main()
