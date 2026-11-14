#!/usr/bin/env python3
"""Evaluate fixed chunk windows and actual contexts from frozen reranker runs.

Grades are inherited from documents, never interpreted as passage annotations.
This analysis requires no model inference and does not change saved rankings.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

if __package__:
    from .evaluate_retrieval import dcg, load_judgments, ranked_chunks
    from .prepare_evaluation import ROOT, fingerprint, write_json
else:
    from evaluate_retrieval import dcg, load_judgments, ranked_chunks
    from prepare_evaluation import ROOT, fingerprint, write_json


def chunk_window_scores(chunks, judgments, corpus_document_ids, k):
    if type(k) is not int or k <= 0:
        raise ValueError('Chunk cutoff must be a positive integer')
    selected = chunks[:k]
    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Duplicate chunk IDs')
    if any(doc not in judgments for doc in corpus_document_ids):
        raise ValueError('Unjudged corpus document')
    if any(c['document_id'] not in judgments for c in chunks):
        raise ValueError('Unjudged retrieved document')
    grades = [judgments[c['document_id']] for c in selected]
    counts = Counter(grades)
    ideal = dcg(sorted((judgments[doc] for doc in corpus_document_ids), reverse=True)[:k])
    distinct = {c['document_id'] for c in selected}
    result = {'returned_chunks': len(selected), 'grade0_chunks': counts[0],
              'grade1_chunks': counts[1], 'grade2_chunks': counts[2],
              'precision_grade_proxy_ge2': counts[2]/k,
              'precision_grade_proxy_ge1': (counts[1]+counts[2])/k,
              'ndcg_grade_proxy': dcg(grades)/ideal if ideal else None,
              'unique_documents': len(distinct),
              'duplicate_document_fraction': 1-len(distinct)/len(selected) if selected else None}
    for threshold in (1, 2):
        relevant = {d for d,g in judgments.items() if g >= threshold}
        result[f'document_recall_ge{threshold}'] = len(distinct & relevant)/len(relevant) if relevant else None
    return result


def macro(rows):
    result = {}
    for key in rows[0]:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = sum(values)/len(values) if values else None
    return result


def index_questions(payload, expected):
    records = payload['questions']
    indexed = {str(q['id']): q for q in records}
    if len(indexed) != len(records) or set(indexed) != set(expected):
        raise ValueError('Runs must contain every judged question exactly once')
    return indexed


def analyze(source, k=50):
    judgments = json.loads((source/'judgments.json').read_text())
    qrels = load_judgments(judgments)
    config_rows = json.loads((source/'summary.json').read_text())
    controls = [r for r in config_rows if r['candidates']==0]
    if len(controls)!=1:
        raise ValueError('Exactly one hybrid control required')
    control = index_questions(json.loads((source/controls[0]['name']/'ranking.json').read_text()), qrels)
    baseline = {qid:list(ranked_chunks(q)) for qid,q in control.items()}
    summaries, details, sources = [], {}, {}
    for config in config_rows:
        name = config['name']
        folder = source/name
        ranking = index_questions(json.loads((folder/'ranking.json').read_text()), qrels)
        context = index_questions(json.loads((folder/'context.json').read_text()), qrels)
        sources[name] = {file:fingerprint(folder/file) for file in ('ranking.json','context.json')}
        rows, context_rows, paired = [], [], []
        for qid in qrels:
            ranked = list(ranked_chunks(ranking[qid]))
            catalog = {c['chunk_id']:c['document_id'] for c in baseline[qid]}
            if {c['chunk_id']:c['document_id'] for c in ranked} != catalog:
                raise ValueError(f'Corpus differs across rankings: {name}/{qid}')
            if ranking[qid]['question'] != control[qid]['question'] or context[qid]['question'] != control[qid]['question']:
                raise ValueError(f'Question text differs: {qid}')
            actual = list(ranked_chunks(context[qid]))
            if any(catalog.get(c['chunk_id']) != c['document_id'] for c in actual):
                raise ValueError('Context chunk absent from source catalog')
            corpus_docs = list(catalog.values())
            window = chunk_window_scores(ranked, qrels[qid], corpus_docs, k)
            actual_metrics = chunk_window_scores(actual, qrels[qid], corpus_docs, max(1,len(actual)))
            original = chunk_window_scores(baseline[qid], qrels[qid], corpus_docs, k)
            previous = {c['chunk_id'] for c in baseline[qid][:k]}
            current = {c['chunk_id'] for c in ranked[:k]}
            current_context = {c['chunk_id'] for c in actual}
            pair = {'id':qid, 'window':window, 'actual_context':actual_metrics,
                    'same_topk_set_as_control':previous==current,
                    'entered_topk':sorted(current-previous), 'left_topk':sorted(previous-current),
                    'topk_chunks_in_actual_context':len(current & current_context),
                    'delta_vs_control':{key:window[key]-original[key] if window[key] is not None and original[key] is not None else None for key in window}}
            rows.append(window); context_rows.append(actual_metrics); paired.append(pair)
        summary = {'name':name,'rerank_candidates':config['candidates'], 'chunk_k':k,
                   'question_count':len(rows), 'window_macro':macro(rows),
                   'actual_context_macro':macro(context_rows),
                   'context_chunks_min':min(r['returned_chunks'] for r in context_rows),
                   'context_chunks_max':max(r['returned_chunks'] for r in context_rows),
                   'contexts_below_k':sum(r['returned_chunks']<k for r in context_rows),
                   'contexts_above_k':sum(r['returned_chunks']>k for r in context_rows),
                   'questions_with_unchanged_topk_set':sum(r['same_topk_set_as_control'] for r in paired)}
        summary['paired_ndcg_vs_control'] = {
            label:sum(test(r['delta_vs_control']['ndcg_grade_proxy']) for r in paired if r['delta_vs_control']['ndcg_grade_proxy'] is not None)
            for label,test in [('improved',lambda v:v>1e-12),('worse',lambda v:v< -1e-12),('tied',lambda v:abs(v)<=1e-12)]}
        summaries.append(summary); details[name] = paired
    return summaries, details, {'judgments':fingerprint(source/'judgments.json'),'runs':sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'experiments/runs/public22-ce-minilm-l6-2026-11-13')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty')
    summaries, details, sources = analyze(args.source,args.k)
    write_json(args.output_dir/'summary.json',summaries)
    write_json(args.output_dir/'per-question.json',details)
    write_json(args.output_dir/'manifest.json',{'status':'complete','chunk_k':args.k,'sources':sources,
        'code':fingerprint(Path(__file__)),
        'definitions':{'cutoff':'First k chunks, not k distinct documents',
            'ndcg':'Gain 2**document_grade-1; ideal over full frozen chunk catalog with inherited document grades',
            'precision':'Chunks from relevant documents divided by k; actual-context fractions use actual context size',
            'recall':'Distinct relevant documents seen in window divided by all judged relevant documents',
            'limitation':'Document-grade proxies do not establish passage usefulness or answer accuracy'}})
    flat = [{'name':r['name'],'rerank_candidates':r['rerank_candidates'],**r['window_macro']} for r in summaries]
    with (args.output_dir/'topk-summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()
