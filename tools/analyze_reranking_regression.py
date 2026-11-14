#!/usr/bin/env python3
"""Audit frozen reranking contexts without generation or new retrieval scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from prepare_evaluation import ROOT, fingerprint, write_json


def question_index(payload):
    records = payload['questions']
    result = {str(row['id']): row for row in records}
    if len(result) != len(records):
        raise ValueError('Duplicate question IDs')
    return result


def summarize(rows):
    numeric = [key for key, value in rows[0].items()
               if isinstance(value, (int, float)) and key != 'id']
    return {key: mean(row[key] for row in rows) for key in numeric}


def analyze(source, catalog_path, tokenizer_path, reranker_tokenizer_path):
    from transformers import AutoTokenizer

    source_files = [source/'judgments.json', source/'scores.json',
                    source/'summary.json', catalog_path]
    judgments = json.loads(source_files[0].read_text())
    grades = {str(q['id']): {r['document_id']: r['grade'] for r in q['judgments']}
              for q in judgments['questions']}
    scores = {str(q['id']): {r['chunk_id']: r for r in q['chunks']}
              for q in json.loads(source_files[1].read_text())}
    configs = json.loads(source_files[2].read_text())
    controls = [c for c in configs if c['candidates'] == 0]
    if len(controls) != 1:
        raise ValueError('Exactly one control required')
    catalog = json.loads(catalog_path.read_text())['questions'][0]['chunks']
    chunks = {c['chunk_id']: c for c in catalog}
    if len(chunks) != len(catalog):
        raise ValueError('Duplicate catalog chunks')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    lengths = {cid: len(tokenizer.encode(c['text'])) for cid, c in chunks.items()}
    context_path = source/controls[0]['name']/'context.json'
    baseline = question_index(json.loads(context_path.read_text()))
    source_files.append(context_path)
    ce_tokenizer = AutoTokenizer.from_pretrained(str(reranker_tokenizer_path), local_files_only=True)
    query_truncation, query_lengths = {}, {}
    for qid, question in baseline.items():
        query = question['question']
        query_lengths[qid] = len(ce_tokenizer.encode(query, add_special_tokens=False))
        candidate_ids = list(scores[qid])
        pairs = ce_tokenizer([query]*len(candidate_ids),
                             [chunks[cid]['text'] for cid in candidate_ids],
                             max_length=512, truncation='longest_first', padding=False)
        query_truncation[qid] = {
            cid: pairs.sequence_ids(i).count(0) < query_lengths[qid]
            for i, cid in enumerate(candidate_ids)
        }
    details, summaries = {}, []
    for config in configs:
        name, candidates = config['name'], config['candidates']
        paths = [source/name/f'{kind}.json' for kind in ('ranking', 'context')]
        source_files.extend(paths)
        ranking, contexts = [question_index(json.loads(p.read_text())) for p in paths]
        if not (set(ranking) == set(contexts) == set(baseline) == set(grades) == set(scores)):
            raise ValueError('Mismatched question IDs')
        rows = []
        for qid, question in contexts.items():
            ordered, selected = ranking[qid]['chunks'], question['chunks']
            ids = [c['chunk_id'] for c in selected]
            previous = [c['chunk_id'] for c in baseline[qid]['chunks']]
            if {c['chunk_id'] for c in ordered} != set(chunks) or len(ordered) != len(chunks):
                raise ValueError('Ranking catalog mismatch')
            if ids != [c['chunk_id'] for c in ordered[:len(ids)]]:
                raise ValueError('Context is not a ranking prefix')
            if any(c['text'] != chunks[c['chunk_id']]['text'] for c in selected):
                raise ValueError('Truncated/changed context requires separate token accounting')
            used = sum(lengths[cid] for cid in ids)
            next_tokens = lengths[ordered[len(ids)]['chunk_id']]
            if not used <= 10000 < used + next_tokens:
                raise ValueError('Context does not reproduce first-overflow policy')
            old, new = set(previous), set(ids)
            removed, added = old-new, new-old
            grade = lambda cid: grades[qid][chunks[cid]['document_id']]
            docs = Counter(c['document_id'] for c in selected)
            scored = [c for c in selected if c.get('hybrid_rank', c['rank']) <= candidates]
            long_selected = [c for c in scored if scores[qid][c['chunk_id']]['pair_tokens'] > 512]
            row = {
                'id': qid, 'domain': question['domain'], 'question': question['question'],
                'context_chunks': len(ids), 'used_chunk_tokens': used,
                'unused_chunk_tokens': 10000-used,
                'mean_chunk_tokens': used/len(ids), 'next_chunk_tokens': next_tokens,
                'retained_chunks': len(old & new), 'removed_chunks': len(removed),
                'added_chunks': len(added), 'baseline_retention': len(old & new)/len(old),
                'baseline_top10_retained_in_context': len(set(previous[:10]) & new),
                'top10_overlap': len(set(previous[:10]) & set(ids[:10])),
                'unique_documents': len(docs), 'largest_document_share': max(docs.values())/len(ids),
                'scored_selected_chunks': len(scored), 'truncated_scored_selected_chunks': len(long_selected),
                'tokens_in_truncated_scored_selected_chunks': sum(lengths[c['chunk_id']] for c in long_selected),
                'query_tokens_before_truncation': query_lengths[qid],
                'query_truncated_candidates': sum(query_truncation[qid][cid]
                                                  for cid in list(scores[qid])[:candidates]),
                'query_truncated_selected_chunks': sum(query_truncation[qid][c['chunk_id']] for c in scored),
                'removed_ids': sorted(removed), 'added_ids': sorted(added),
            }
            for g in (0, 1, 2):
                row[f'removed_grade{g}'] = sum(grade(cid) == g for cid in removed)
                row[f'added_grade{g}'] = sum(grade(cid) == g for cid in added)
            rows.append(row)
        details[name] = rows
        summaries.append({'name': name, 'candidates': candidates, 'question_count': len(rows),
                          'macro': summarize(rows),
                          'by_domain': {d: summarize([r for r in rows if r['domain'] == d])
                                        for d in sorted({r['domain'] for r in rows})},
                          'totals': {k: sum(r[k] for r in rows) for k in
                                     ('context_chunks', 'removed_chunks', 'added_chunks',
                                      'removed_grade2', 'added_grade2',
                                      'truncated_scored_selected_chunks',
                                      'query_truncated_candidates', 'query_truncated_selected_chunks',
                                      'tokens_in_truncated_scored_selected_chunks', 'used_chunk_tokens')}})
    manifest = {'sources': {str(p.relative_to(ROOT)): fingerprint(p) for p in source_files},
                'tokenizer_files': {p.name: fingerprint(p) for p in tokenizer_path.iterdir() if p.is_file()},
                'reranker_tokenizer_files': {p.name: fingerprint(p) for p in reranker_tokenizer_path.iterdir()
                                            if p.name in ('tokenizer.json', 'tokenizer_config.json',
                                                          'special_tokens_map.json', 'vocab.txt')},
                'pair_truncation_policy': 'longest_first, max_length=512; reproduces installed Sentence Transformers default',
                'code': fingerprint(Path(__file__)),
                'limitations': ['Public development set; no hidden responses or final LLM inference.',
                                'Document grades are proxies, not passage support labels.',
                                'Tokens in truncated pairs count whole chunks, not omitted tokens.',
                                'Token budget counts chunk text only, matching baseline.']}
    return summaries, details, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'experiments/runs/public22-ce-minilm-l6-2026-11-13')
    parser.add_argument('--catalog', type=Path, default=ROOT/'experiments/runs/baseline-public22-ranking/ranking.json')
    parser.add_argument('--tokenizer', type=Path, default=ROOT/'models/olmo-tokenizer')
    parser.add_argument('--reranker-tokenizer', type=Path, default=ROOT/'models/ms-marco-MiniLM-L6-v2')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty')
    summaries, details, manifest = analyze(args.source.resolve(), args.catalog.resolve(),
                                         args.tokenizer.resolve(), args.reranker_tokenizer.resolve())
    for name, payload in [('summary', summaries), ('per-question', details), ('manifest', manifest)]:
        write_json(args.output_dir/f'{name}.json', payload)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
