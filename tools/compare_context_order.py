#!/usr/bin/env python3
"""Cross frozen context selection and ordering without repacking or inference."""
import argparse
import json
from pathlib import Path

if __package__:
    from .evaluate_chunk_windows import chunk_window_scores, index_questions, macro
    from .evaluate_retrieval import load_judgments
    from .prepare_evaluation import ROOT, fingerprint, write_json
else:
    from evaluate_chunk_windows import chunk_window_scores, index_questions, macro
    from evaluate_retrieval import load_judgments
    from prepare_evaluation import ROOT, fingerprint, write_json


def order_selected(selected, ranking):
    """Permute exactly the selected chunks, keeping text and provenance intact."""
    positions = {c['chunk_id']: i for i,c in enumerate(ranking)}
    ids = [c['chunk_id'] for c in selected]
    if len(positions) != len(ranking) or len(ids) != len(set(ids)):
        raise ValueError('Duplicate chunk IDs')
    if not set(ids) <= positions.keys():
        raise ValueError('Selected chunk absent from ranking')
    return [dict(c) for c in sorted(selected, key=lambda c:positions[c['chunk_id']])]


def format_context(selected):
    body = '\n\n'.join(f"[Fragment {i}]\n{c['text']}" for i,c in enumerate(selected, 1))
    return f'Context extracted from a blind search over the corpus:\n\n{body}\n\n---\n\n' if selected else ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'experiments/runs/public22-ce-minilm-l6-query-preserved-2026-11-14')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--reranker-key', default='ce100', help='Short label identifying the reranker')
    args = parser.parse_args()
    for folder in (args.output_dir, args.results_dir):
        if folder.exists() and any(folder.iterdir()):
            parser.error('Use new output directories')
    if not args.reranker_key or args.reranker_key == 'hybrid' or any(
            c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.reranker_key):
        parser.error('Use a distinct lowercase reranker key')
    source = args.source
    read = lambda path:json.loads(path.read_text())
    configs = read(source/'summary.json')
    names = {}
    for key, depth in [('hybrid',0), (args.reranker_key,100)]:
        matches = [c['name'] for c in configs if c['candidates']==depth]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one variant at depth {depth}')
        names[key] = matches[0]
    judgments = read(source/'judgments.json')
    qrels = load_judgments(judgments)
    rankings, contexts, sources = {}, {}, {}
    for key,name in names.items():
        rankings[key] = index_questions(read(source/name/'ranking.json'), qrels)
        contexts[key] = index_questions(read(source/name/'context.json'), qrels)
        sources[key] = {f:fingerprint(source/name/f) for f in ('ranking.json','context.json')}
    summaries, details = [], {}
    for selection in names:
        for order in names:
            name = f'select-{selection}-order-{order}-query-preserved'
            records, metrics = [], []
            for qid in qrels:
                original = contexts[selection][qid]
                control = rankings['hybrid'][qid]
                if any(data[qid]['question'] != control['question']
                       for data in [*rankings.values(), *contexts.values()]):
                    raise ValueError('Mismatched query text')
                catalog = {c['chunk_id']:c['document_id'] for c in control['chunks']}
                if catalog != {c['chunk_id']:c['document_id'] for c in rankings[order][qid]['chunks']}:
                    raise ValueError('Mismatched chunk catalog')
                selected = order_selected(original['chunks'], rankings[order][qid]['chunks'])
                if format_context(original['chunks']) != original['context']:
                    raise ValueError('Unexpected source context format')
                # Reordering must preserve every selected byte and its document identity.
                if any(catalog[c['chunk_id']] != c['document_id'] for c in selected):
                    raise ValueError('Mismatched document identity')
                context = format_context(selected)
                if selection == order and context != original['context']:
                    raise ValueError('Control reconstruction changed context')
                score = chunk_window_scores(selected, qrels[qid], list(catalog.values()), max(1,len(selected)))
                top10 = chunk_window_scores(selected, qrels[qid], list(catalog.values()), 10)
                score['ndcg_proxy_at10'] = top10['ndcg_grade_proxy']
                metrics.append(score)
                records.append({'id':original['id'],'question':original['question'], 'domain':original['domain'],
                                'chunks':[{**c, 'source_rank':c['rank'], 'rank':i}
                                          for i,c in enumerate(selected, 1)], 'context':context})
            write_json(args.output_dir/name/'context.json', {'variant':name,'questions':records})
            details[name] = [{'id':r['id'], **m} for r,m in zip(records,metrics)]
            summaries.append({'name':name,'selection':selection,'order':order,
                              'question_count':len(records),'macro':macro(metrics)})
    manifest = {'sources':sources,'judgments':fingerprint(source/'judgments.json'),
                'code':fingerprint(Path(__file__)), 'repacking':False,'new_model_inference':False,
                'definitions':{'rank':'Context position; source_rank retains source retrieval position',
                               'ndcg':'Document-grade proxy; full-context cutoff varies with selection',
                               'token_budget':'Same selected texts as source, no repacking after ordering'},
                'limitations':['No final LLM or measured answer accuracy.',
                               'Source context budget counts chunk texts, excluding headers.']}
    for folder in (args.output_dir, args.results_dir):
        write_json(folder/'summary.json', summaries)
        write_json(folder/'per-question.json', details)
        write_json(folder/'manifest.json', manifest)
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
