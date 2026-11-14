#!/usr/bin/env python3
"""Offline ZIP replay and official answer smoke; final generator is a fixture."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np

from prepare_evaluation import ROOT, prepare
from compare_retrieval import load_frozen_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    args = parser.parse_args()
    record = json.loads(args.manifest.read_text())
    archive_path = Path(record['archive'])
    if hashlib.sha256(archive_path.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('Archive hash mismatch')
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    questions,judgments,_ = prepare(ROOT/'data/pdfs')
    chunks,dense,_ = load_frozen_run(ROOT/'experiments/runs/baseline-public22-ranking',questions,judgments)
    ettin = record['variant']=='ettin1024-selected'
    source = ROOT/('experiments/runs/public22-ce-ettin68m-top100-maxlen1024-2026-11-14'
                   if ettin else 'experiments/runs/public22-ce-minilm-l6-query-preserved-2026-11-14')
    scores = {str(q['id']):q for q in json.loads((source/'scores.json').read_text())}
    reference = {str(q['id']):q for q in json.loads((ROOT/record['reference_run']/'context.json').read_text())['questions']}
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'models/olmo-tokenizer'),local_files_only=True)
    with tempfile.TemporaryDirectory(prefix='ieee-order-validation-') as temporary:
        folder = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            if set(archive.namelist()) != set(record['zip_entry_sha256']):
                raise ValueError('Entry list differs from manifest')
            for name in archive.namelist():
                if hashlib.sha256(archive.read(name)).hexdigest()!=record['zip_entry_sha256'][name]:
                    raise ValueError(f'Entry hash mismatch: {name}')
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('Invalid archive path')
            archive.extractall(folder)
        sys.path.insert(0,str(folder))
        spec = importlib.util.spec_from_file_location('packaged_main',folder/'main.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        retrieval = sys.modules['retrieval']
        assert Path(retrieval.__file__).parent==folder
        assert list(inspect.signature(module.answer).parameters)==['question','corpus_dir','model_id']
        assert module.RERANK_ENABLED and module.RERANK_MODE=='selected_context'
        index = SimpleNamespace(chunks=chunks,embeddings=None,
                                embedder=SimpleNamespace(encode=lambda *a,**kw:np.array([[1.]],dtype=np.float32)),
                                lexical_index=None)
        class ReplayReranker:
            def rerank(self,question,ranking,candidates):
                assert candidates==100
                qid = next(str(q['id']) for q in questions if q['question']==question)
                saved = scores[qid]['chunks']
                assert [c['chunk_id'] for c in ranking[:100]]==[c['chunk_id'] for c in saved]
                # Use the packaged stable prefix implementation.
                from reranking import reorder_prefix
                return reorder_prefix(ranking,[c['rerank_score'] for c in saved])
        config = dict(method=module.RETRIEVAL_METHOD,k1=module.BM25_K1,b=module.BM25_B,
                      rrf_k=module.RRF_K,dense_weight=module.RRF_DENSE_WEIGHT,
                      bm25_weight=module.RRF_BM25_WEIGHT,rerank_candidates=module.RERANK_CANDIDATES,
                      rerank_mode=module.RERANK_MODE)
        for q in questions:
            qid = str(q['id'])
            index.embeddings = dense[qid].reshape(-1,1)
            result = retrieval.retrieve(q['question'],index,tokenizer,reranker=ReplayReranker(),**config)
            assert result['context']==reference[qid]['context'], f'Context mismatch: {qid}'
            assert [c['chunk_id'] for c in result['selected_chunks']]==[c['chunk_id'] for c in reference[qid]['chunks']]
        # Exercise the actual packaged CrossEncoder, official entry point and chat template.
        q = questions[0]
        index.embeddings = dense[str(q['id'])].reshape(-1,1)
        captured = []
        class GeneratorFixture:
            device = torch.device('cpu')
            def generate(self, **kwargs):
                assert kwargs['max_new_tokens']==2048 and kwargs['do_sample'] is False
                return torch.cat([kwargs['input_ids'],torch.tensor([tokenizer.encode('validation-ok',add_special_tokens=False)])],dim=1)
        original_template = tokenizer.apply_chat_template
        def capture(messages,**kwargs):
            captured.append(messages)
            return original_template(messages,**kwargs)
        module._load = lambda *args:(tokenizer,GeneratorFixture(),index)
        with patch.object(tokenizer,'apply_chat_template',side_effect=capture):
            result = module.answer(q['question'],'frozen-corpus','generator-fixture')
            assert result=='validation-ok'
            assert captured[-1][1]['content']==reference[str(q['id'])]['context']+q['question']
            real = module._load_reranker()
            assert real is module._load_reranker()
            long_query = 'query '*2000
            expected = retrieval.retrieve(long_query,index,tokenizer,**config)['context']
            with patch.object(real.model,'predict',side_effect=AssertionError('Must skip overflow')):
                module.answer(long_query,'frozen-corpus','generator-fixture')
            assert captured[-1][1]['content']==expected+long_query
    record['validation'].update(archive_entry_hashes='passed',official_answer_signature='passed',
                               frozen_context_replay_questions=22,real_ce_answer_smoke='passed',
                               real_ce_scored_candidates=100,overflow_answer_fallback='passed',
                               reranker_cached_once=True,validation_cpu_threads=4,
                               final_llm='fixture; OLMo weights not loaded',
                               dense_scores='frozen; BM25 and cross-encoder real in answer smoke',
                               h100_total_runtime_tested=False)
    args.manifest.write_text(json.dumps(record,indent=2)+'\n')
    print('Validated:',archive_path)


if __name__=='__main__':
    main()
