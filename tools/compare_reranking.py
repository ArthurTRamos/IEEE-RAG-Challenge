#!/usr/bin/env python3
"""Frozen hybrid vs cross-encoder prefix reranking with document-grade proxies."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np
from prepare_evaluation import ROOT, prepare, write_json, fingerprint
from compare_retrieval import load_frozen_run
from evaluate_retrieval import evaluate, load_judgments, dcg
from sweep_retrieval import metrics_for_context
sys.path.insert(0, str(ROOT))
from src.retrieval import BM25Retriever, rank_chunks, select_context
from src.reranking import ChunkReranker, reorder_prefix


def grade_proxy(rows, grades, corpus, k=10):
    values = [grades[c['document_id']] for c in rows[:k]]
    ideal = dcg(sorted([grades[c['document_id']] for c in corpus], reverse=True)[:k])
    return {'ndcg_document_grade_proxy_at10': dcg(values)/ideal if ideal else None,
            'chunks_from_grade2_documents_at10': sum(g==2 for g in values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=ROOT/'experiments/runs/baseline-public22-ranking')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=ROOT/'models/olmo-tokenizer')
    p.add_argument('--candidates', type=int, nargs='+', default=[50,100,200])
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): p.error('Output must be empty')
    if not args.candidates or min(args.candidates)<=0 or len(set(args.candidates))!=len(args.candidates): p.error('Candidate counts must be unique and positive')
    if args.threads<=0: p.error('Thread count must be positive')
    started = time.perf_counter()
    import torch
    torch.set_num_threads(args.threads)
    questions, judgments, manifest = prepare(ROOT/'data/pdfs')
    chunks, similarities, source = load_frozen_run(args.source_run, questions, judgments)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    token_counts = {c['chunk_id']:len(tokenizer.encode(c['text'])) for c in chunks}
    tick = time.perf_counter()
    reranker = ChunkReranker(args.model)
    load_seconds = time.perf_counter()-tick
    bm25 = BM25Retriever(chunks, 1.25, .75)
    qrels = load_judgments(judgments)
    modes = [0, *sorted(args.candidates)]
    outputs = {n:{'ranking':[], 'context':[], 'diagnostics':[]} for n in modes}
    score_records=[]
    for q in questions:
        qid = str(q['id'])
        ranking = rank_chunks(chunks, similarities[qid], q['question'], 'hybrid', bm25, 60, 1, .75)
        prefix = ranking[:max(modes)]
        pairs = reranker.model.tokenizer([q['question']]*len(prefix), [c['text'] for c in prefix], truncation=False, padding=False)
        lengths = [len(ids) for ids in pairs['input_ids']]
        tick = time.perf_counter()
        scores = reranker.score(q['question'], prefix)
        duration = time.perf_counter()-tick
        if len(scores)!=len(prefix): raise ValueError('Incorrect cross-encoder score count')
        score_records.append({'id':q['id'],'chunks': [{'chunk_id':c['chunk_id'], 'rerank_score':float(s), 'pair_tokens':length}
                              for c,s,length in zip(prefix,scores,lengths)], 'score_seconds':duration})
        for n in modes:
            ordered = reorder_prefix(ranking, scores[:n]) if n else ranking
            selected = select_context(ordered, tokenizer, token_counts=token_counts)
            base = {'id':q['id'],'question':q['question'],'domain':q['domain']}
            outputs[n]['ranking'].append({**base,'chunks':[{k:v for k,v in c.items() if k not in ('text','source_path')} for c in ordered]})
            outputs[n]['context'].append({**base,'chunks':selected['selected_chunks'],'context':selected['context']})
            counts=Counter(str(qrels[qid][c['document_id']]) for c in selected['selected_chunks'])
            outputs[n]['diagnostics'].append({**base,**grade_proxy(ordered,qrels[qid],chunks),
                'candidate_document_ids':sorted({c['document_id'] for c in ranking[:n]}),
                'context_chunks_by_document_grade':dict(counts), 'selected_chunks':len(selected['selected_chunks']),
                'truncated_cross_encoder_pairs':sum(v>512 for v in lengths[:n]),
                'scored_pairs':min(n,len(prefix)),
                'selected_unscored_tail_chunks':sum(c.get('hybrid_rank', c['rank'])>n for c in selected['selected_chunks']) if n else 0})
        print(f"Scored question {qid}: {len(prefix)} pairs, {duration:.2f}s",flush=True)
    summaries=[]
    write_json(args.output_dir/'judgments.json',judgments)
    for n in modes:
        name='hybrid-control' if not n else f'hybrid-ce-minilm-l6-top{n}-maxlen512-cpu'
        folder=args.output_dir/name
        data=outputs[n]
        rp={'variant':name,'questions':data['ranking']}; cp={'variant':name,'questions':data['context']}
        rm=evaluate(rp,judgments); cm=metrics_for_context(cp,judgments)
        for filename,payload in [('ranking',rp),('context',cp),('ranking-metrics',rm),('context-metrics',cm),('diagnostics',data['diagnostics'])]: write_json(folder/f'{filename}.json',payload)
        diag=data['diagnostics']; counts=Counter()
        for row in diag: counts.update(row['context_chunks_by_document_grade'])
        summary={'name':name,'candidates':n,'question_count':len(questions),
            'document_ndcg_at10':rm['macro_at_document_k']['10']['ndcg']['value'],
            'document_grade_proxy_ndcg_at10':sum(r['ndcg_document_grade_proxy_at10'] for r in diag)/len(diag),
            'mean_grade2_chunks_at10':sum(r['chunks_from_grade2_documents_at10'] for r in diag)/len(diag),
            'context_recall_ge2':cm['macro_full_context_document_coverage']['recall_ge2']['value'],
            'context_recall_ge1':cm['macro_full_context_document_coverage']['recall_ge1']['value'],
            'context_chunks_by_document_grade':dict(counts),
            'selected_chunks':sum(counts.values()),
            'truncated_cross_encoder_pairs':sum(r['truncated_cross_encoder_pairs'] for r in diag),
            'scored_pairs':sum(r['scored_pairs'] for r in diag)}
        summaries.append(summary); write_json(folder/'summary.json',summary)
        write_json(folder/'manifest.json',{'status':'complete','config':{'candidates':n,'max_length':512,'batch_size':16,'device':'cpu','tail_policy':'append unchanged after reranked prefix'},'model_files':{p.name:fingerprint(p) for p in args.model.iterdir() if p.is_file()}})
    manifest.update({'status':'complete','source_manifest':source,'config':{'k1':1.25,'b':.75,'rrf_k':60,'weights':[1,.75],'token_budget':10000,'candidates':args.candidates,'cpu_threads':args.threads},
        'model_load_seconds':load_seconds,'model_path':str(args.model),'model_files':{p.name:fingerprint(p) for p in args.model.iterdir() if p.is_file()},
        'source_ranking':fingerprint(args.source_run/'ranking.json'),
        'code':{s:fingerprint(ROOT/s) for s in ('src/retrieval.py','src/reranking.py','tools/compare_reranking.py')},
        'score_seconds_total':sum(r['score_seconds'] for r in score_records),
        'elapsed_seconds':time.perf_counter()-started,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'timing_scope':'CPU frozen replay, scores shared across candidate depths; excludes PDF parsing, dense embeddings and final LLM'})
    write_json(args.output_dir/'manifest.json',manifest)
    write_json(args.output_dir/'scores.json',score_records)
    write_json(args.output_dir/'summary.json',summaries)


if __name__=='__main__': main()
