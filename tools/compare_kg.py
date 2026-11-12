#!/usr/bin/env python3
"""Controlled KG ablation with section/document-grade context diagnostics."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

from prepare_evaluation import ROOT, prepare, write_json, fingerprint
from compare_retrieval import load_frozen_run
from sweep_retrieval import metrics_for_context
from evaluate_retrieval import evaluate, load_judgments
sys.path.insert(0, str(ROOT))
from src.retrieval import BM25Retriever, rank_chunks, select_context
from src.kg_expansion import KnowledgeExpansion
from src.chunk_diagnostics import annotate_sections, context_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT/'experiments/runs/baseline-public22-ranking')
    parser.add_argument('--kg', type=Path, default=ROOT/'data/kg_rag_chunks.jsonl')
    parser.add_argument('--tokenizer', type=Path, default=ROOT/'models/olmo-tokenizer')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty')
    started = time.perf_counter()
    questions, judgments, manifest = prepare(ROOT/'data/pdfs')
    chunks, dense, source_manifest = load_frozen_run(args.source_run, questions, judgments)
    annotations = annotate_sections(chunks)
    kg = KnowledgeExpansion(args.kg, chunks)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    counts = {c['chunk_id']: len(tokenizer.encode(c['text'])) for c in chunks}
    bm25 = BM25Retriever(chunks, 1.25, .75)
    qrels = load_judgments(judgments)
    write_json(args.output_dir/'sections.json', annotations)
    write_json(args.output_dir/'judgments.json', judgments)
    write_json(args.output_dir/'entity-chunk-links.json', {u: sorted(v) for u,v in sorted(kg.postings.items()) if v})
    summaries = []
    for mode in ('control', 'aliases', 'one_hop'):
        tick = time.perf_counter()
        name = f'hybrid-bw0p75-k1_1p25-b0p75-rrf60-kg-{mode}'
        folder = args.output_dir/name
        rankings, contexts, diagnostics = [], [], []
        for question in questions:
            qid = str(question['id'])
            query, trace = kg.expand(question['question'], mode)
            lexical = SimpleNamespace(scores=lambda _: bm25.scores(query))
            rows = rank_chunks(chunks, dense[qid], question['question'], 'hybrid', lexical, 60, 1, .75)
            result = select_context(rows, tokenizer, token_counts=counts)
            selected = result['selected_chunks']
            base = {'id': question['id'], 'question': question['question'], 'domain': question['domain']}
            rankings.append({**base, 'chunk_ids': [c['chunk_id'] for c in rows], 'scores': [c['score'] for c in rows]})
            contexts.append({**base, 'chunks': [{**c, **annotations[c['chunk_id']]} for c in selected], 'context': result['context']})
            diagnostics.append({**base, 'expanded_bm25_query': query, 'linking': trace,
                                **context_diagnostics(selected, qrels[qid], annotations)})
        rp, cp = {'variant': name, 'questions': rankings}, {'variant': name, 'questions': contexts}
        rm, cm = evaluate(rp, judgments), metrics_for_context(cp, judgments)
        totals = {}
        for key in ('by_document_grade', 'by_primary_section', 'section_by_document_grade'):
            combined = Counter()
            for q in diagnostics: combined.update(q[key])
            totals[key] = dict(combined)
        summary = {'name': name, 'mode': mode, 'question_count': len(questions),
                   'ndcg_at10': rm['macro_at_document_k']['10']['ndcg']['value'],
                   'full_context_coverage': cm['macro_full_context_document_coverage'],
                   'selected_chunks': sum(q['selected_chunks'] for q in diagnostics),
                   'mean_fraction_from_grade2_documents': sum(q['fraction_from_grade2_documents'] for q in diagnostics)/len(diagnostics),
                   'questions_with_seeds': sum(bool(q['linking']['seed_uris']) for q in diagnostics),
                   'questions_with_added_phrases': sum(bool(q['linking']['additions']) for q in diagnostics),
                   'totals': totals, 'incremental_seconds': time.perf_counter()-tick}
        for file, value in [('ranking', rp), ('context', cp), ('ranking-metrics', rm), ('context-metrics', cm),
                            ('diagnostics', diagnostics), ('summary', summary)]:
            write_json(folder/f'{file}.json', value)
        write_json(folder/'manifest.json', {
            'status': 'complete', 'mode': mode, 'question_count': len(questions),
            'config': {'k1': 1.25, 'b': .75, 'rrf_k': 60, 'dense_weight': 1, 'bm25_weight': .75,
                       'token_budget': 10000, 'dense_query': 'unchanged', 'llm_question': 'unchanged'},
            'sources': {'kg_cards': fingerprint(args.kg),
                        'dense_ranking': fingerprint(args.source_run/'ranking.json'),
                        'judgments': fingerprint(args.output_dir/'judgments.json'),
                        'sections': fingerprint(args.output_dir/'sections.json')}})
        summaries.append(summary)
        print(name, summary['ndcg_at10'], 'selected:', summary['selected_chunks'], flush=True)
    manifest.update({'status': 'complete', 'source_manifest': source_manifest,
        'sources': {**manifest['sources'], 'kg_cards': fingerprint(args.kg),
                    'dense_ranking': fingerprint(args.source_run/'ranking.json'),
                    'code': {p: fingerprint(ROOT/p) for p in ('src/kg_expansion.py','src/chunk_diagnostics.py','tools/compare_kg.py','src/retrieval.py')}},
        'config': {'k1':1.25,'b':.75,'rrf_k':60,'dense_weight':1,'bm25_weight':.75,'token_budget':10000,
                   'seed_limit':8,'aliases_per_seed':2,'neighbor_limit':8,'max_phrase_words':8,
                   'dense_query':'unchanged','llm_question':'unchanged','kg_prompt_context':False},
        'kg_statistics': {'nodes':len(kg.nodes),'directed_card_edges':sum(map(len,kg.edges.values())),
                          'linked_entities':sum(bool(v) for v in kg.postings.values()),
                          'ambiguous_label_edges_skipped':kg.ambiguous_edges,
                          'definitions_present':sum(bool(n.get('definition')) for n in kg.nodes.values())},
        'limitations': ['Graph uses related_concepts resolved by unique labels; export-time caps depend on the source bundle. Structured URI edges are not consumed.',
                       'Section classification is a heading heuristic, not a Docling semantic section label.',
                       'Document grade does not establish chunk relevance; overlap may span sections.',
                       'Timing excludes source PDF parsing/dense embeddings and final LLM.'],
        'elapsed_seconds':time.perf_counter()-started,
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    write_json(args.output_dir/'manifest.json', manifest)
    write_json(args.output_dir/'summary.json', summaries)
    with (args.output_dir/'section-grade-counts.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'section', 'document_grade', 'chunk_occurrences'])
        for row in summaries:
            for pair, count in sorted(row['totals']['section_by_document_grade'].items()):
                section, grade = pair.split('|grade_')
                writer.writerow([row['name'], section, grade, count])


if __name__ == '__main__':
    main()
