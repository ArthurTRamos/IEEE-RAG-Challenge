#!/usr/bin/env python3
"""Build the two frozen DBpedia submission candidates without downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.dbpedia_builder import validate_bundle

VARIANTS = {
    'original-legacy': ('data/kg_rag_chunks.jsonl', 'legacy',
                        'experiments/runs/public22-kg-legacy-replay-2026-11-12'),
    'v2-contextual': ('experiments/runs/dbpedia-reviewed-v2/kg_rag_chunks.jsonl', 'contextual',
                      'experiments/runs/public22-v2-contextual-decision-2026-11-12'),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(output_dir):
    baseline = json.loads((ROOT/'experiments/submissions/2026-11-10-hybrid-k1_1p25-b0p75-rrf60-dw1-bw0p75.json').read_text())
    assets = []
    for name, expected in baseline['zip_entry_sha256'].items():
        if name.startswith(('embed_model/', 'docling_models/')):
            path = ROOT/'src'/name
            if sha(path.read_bytes()) != expected:
                raise ValueError('Auxiliary model differs from score-60 baseline: '+name)
            assets.append((name, path))
    validate_bundle(ROOT/'experiments/runs/dbpedia-reviewed-v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = [(name, output_dir/f'2026-11-12-dbpedia-{name}.zip') for name in VARIANTS]
    for _, path in planned:
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError(path)
    for name, path in planned:
        resource, policy, reference = VARIANTS[name]
        kg = ROOT/resource
        config = {'variant': name, 'policy': policy, 'mode': 'aliases', 'kg_sha256': sha(kg.read_bytes()),
                  'source_resource': resource, 'source': 'https://dbpedia.org/',
                  'citation': 'DBpedia; see https://www.dbpedia.org/about/ for source and licensing information.'}
        entries = [('main.py', ROOT/'src/main_kg.py'), ('retrieval.py', ROOT/'src/retrieval.py'),
                   ('kg_expansion.py', ROOT/'src/kg_expansion.py'), ('kg_resources/kg_rag_chunks.jsonl', kg), *assets]
        temp = path.with_suffix('.zip.tmp')
        try:
            with zipfile.ZipFile(temp, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for target, source in sorted(entries):
                    archive.write(source, target)
                archive.writestr('kg_config.json', json.dumps(config, ensure_ascii=False, indent=2)+'\n')
            if temp.stat().st_size > 1_000_000_000:
                raise ValueError('Submission exceeds 1 GB')
            with zipfile.ZipFile(temp) as archive:
                if archive.testzip() is not None:
                    raise ValueError('ZIP CRC failed')
                hashes = {n: sha(archive.read(n)) for n in archive.namelist()}
                if len(hashes) != len(archive.namelist()):
                    raise ValueError('Duplicate archive entries')
            temp.rename(path)
        finally:
            if temp.exists(): temp.unlink()
        manifest = {'archive': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size,
                    'sha256': sha(path.read_bytes()), 'kg_config': config,
                    'zip_entry_sha256': hashes, 'baseline_config': baseline['config'],
                    'context_chunk_text_budget': 10000, 'max_new_tokens': 2048,
                    'reference_run': reference, 'models_match_score60_baseline': True,
                    'validation': {'zip_crc': 'passed', 'full_llm_inference_tested': False,
                                   'h100_total_runtime_tested': False},
                    'official_submission_status': 'not_submitted'}
        path.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        print(f'{path}: {path.stat().st_size} bytes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'submissions')
    build(parser.parse_args().output_dir.resolve())
