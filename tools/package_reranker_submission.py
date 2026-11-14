#!/usr/bin/env python3
"""Build the top-100 cross-encoder submission from local resources only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def constants(source):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT/'models/ms-marco-MiniLM-L6-v2')
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        parser.error('Refusing to overwrite existing archive or manifest')
    source = (ROOT/'src/main.py').read_text()
    if constants(source).get('RERANK_ENABLED') is not False:
        parser.error('Expected baseline source with reranker disabled')
    source = source.replace('RERANK_ENABLED = False', 'RERANK_ENABLED = True')
    config = constants(source)
    expected = {'RETRIEVAL_METHOD':'hybrid','BM25_K1':1.25,'BM25_B':.75,
                'RRF_K':60,'RRF_DENSE_WEIGHT':1.,'RRF_BM25_WEIGHT':.75,
                'RERANK_ENABLED':True,'RERANK_CANDIDATES':100,
                'RERANK_MAX_LENGTH':512,'RERANK_BATCH_SIZE':16}
    if any(config.get(k) != v for k,v in expected.items()):
        parser.error('Source configuration differs from requested submission')
    folders = {'embed_model':ROOT/'src/embed_model', 'docling_models':ROOT/'src/docling_models',
               'reranker_model':args.model}
    required = {'embed_model':['model.safetensors','modules.json'],
                'docling_models':['docling-project--docling-layout-heron/model.safetensors'],
                'reranker_model':['model.safetensors','config.json','tokenizer.json']}
    for folder, files in required.items():
        for file in files:
            if not (folders[folder]/file).is_file(): parser.error(f'Missing model file: {folder}/{file}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in ('main.py','retrieval.py','reranking.py'):
            data = source.encode() if name=='main.py' else (ROOT/'src'/name).read_bytes()
            compile(data, name, 'exec')
            archive.writestr(name,data)
            entries[name] = hashlib.sha256(data).hexdigest()
        for folder, directory in folders.items():
            for path in sorted(directory.rglob('*')):
                relative = path.relative_to(directory)
                if not path.is_file() or any(p.startswith('.') or p=='__pycache__' for p in relative.parts): continue
                if path.suffix in ('.pyc','.Identifier'): continue
                name = f'{folder}/{relative.as_posix()}'
                archive.write(path,name)
                entries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.output.stat().st_size > 1_000_000_000:
        raise ValueError('Archive exceeds 1 GB')
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None: raise ValueError('ZIP integrity check failed')
    record = {'archive':str(args.output), 'size_bytes':args.output.stat().st_size,
              'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(), 'config':expected,
              'context_chunk_text_budget':10000,'max_new_tokens':config['MAX_NEW_TOKENS'],
              'reranker_device':'cpu','reranker_cpu_threads':'environment default; no global thread override',
              'tail_policy':'append unscored candidates in unchanged hybrid order',
              'model_provenance':json.loads((args.model/'provenance.json').read_text()),
              'zip_entry_sha256':entries,'validation':{'zip_crc':'passed','size_under_1GB':True},
              'reference_run':'experiments/runs/public22-ce-minilm-l6-2026-11-13/hybrid-ce-minilm-l6-top100-maxlen512-cpu',
              'official_submission_status':'not_uploaded'}
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(json.dumps(record,indent=2)+'\n')
    print(args.output, record['size_bytes'], record['sha256'])


if __name__=='__main__': main()
