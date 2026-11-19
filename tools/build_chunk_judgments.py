#!/usr/bin/env python3
"""Materialize Codex chunk reviews and inherited document-zero judgments."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

from prepare_evaluation import (ROOT, DEFAULT_JUDGMENTS, EXCLUDED_IDS,
                                EXPECTED_IDS, write_json, fingerprint)


def chunk_signature(doc):
    payload = [(c['chunk_id'], c['sha256']) for c in doc['chunks']]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def write_report(path, question, rows, catalog):
    positives = sorted((r for r in rows if r['grade'] in (1, 2)),
                       key=lambda r: (-r['grade'], r['document_id'],
                                      catalog[r['chunk_id']]['chunk_index']))
    lines = [f"# Pergunta {question['id']}", '', question['question'], '',
             'Revisão pelo Codex. Este relatório lista os graus 1 e 2; '
             'o JSON correspondente contém todos os chunks, incluindo os zeros.', '',
             '| Grau | Chunk | Grau do documento | Justificativa |',
             '| --- | --- | --- | --- |']
    for row in positives:
        chunk = catalog[row['chunk_id']]
        link = os.path.relpath(ROOT/chunk['path'], path.parent)
        reason = row['rationale'].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {row['grade']} | [{row['chunk_id']}]({link}) | "
                     f"{row['document_grade']} | {reason} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, default=ROOT/'experiments/runs/pdf-markdown-chunks')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/judments/public22-chunks')
    parser.add_argument('--require-complete', action='store_true',
                        help='Fail if any semantic review is missing.')
    args = parser.parse_args()
    args.export = args.export.resolve()
    args.output = args.output.resolve()
    source = json.loads(DEFAULT_JUDGMENTS.read_text())
    manifest = json.loads((args.export/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    docs = {d['document_id']: d for d in manifest['documents']}
    questions = [q for q in source['questions'] if str(q['id']) not in EXCLUDED_IDS]
    assert len(questions) == 22 and {str(q['id']) for q in questions} == EXPECTED_IDS
    assert set(docs) == {d['document_id'] for d in source['corpus_documents']}
    source_hash = fingerprint(DEFAULT_JUDGMENTS)['sha256']
    reviews = {}
    review_sources = []
    for p in sorted((args.output/'reviews').glob('*.json')):
        review = json.loads(p.read_text())
        doc = docs[review['document_id']]
        assert review['markdown_sha256'] == doc['markdown_sha256'], p
        assert review['source_judgments_sha256'] == source_hash, p
        assert review['chunks_sha256'] == chunk_signature(doc), p
        assert review['scope'] == 'all_chunks_semantically_reviewed', p
        review_sources.append(fingerprint(p))
        for q in review['questions']:
            key = (str(q['id']), doc['document_id'])
            assert key not in reviews, key
            assert q['zero_rationale'].strip(), key
            entries = {i: (0, q['zero_rationale']) for i in range(doc['chunk_count'])}
            explicit = set()
            for decision in q['decisions']:
                assert type(decision['grade']) is int and decision['grade'] in (0, 1, 2)
                assert decision['rationale'].strip()
                for i in decision['indices']:
                    assert i in entries and i not in explicit, (p, i)
                    explicit.add(i)
                    entries[i] = decision['grade'], decision['rationale']
            reviews[key] = entries
    catalog = []
    for doc in docs.values():
        assert fingerprint(args.export/doc['markdown_path'])['sha256'] == doc['markdown_sha256']
        assert len(doc['chunks']) == doc['chunk_count']
        for index, chunk in enumerate(doc['chunks']):
            path = args.export/chunk['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == chunk['sha256']
            catalog.append({**chunk, 'chunk_index': index, 'document_id': doc['document_id'],
                            'path': str(path.relative_to(ROOT))})
    write_json(args.output/'chunks.json', catalog)
    by_chunk = {c['chunk_id']: c for c in catalog}
    assert len(by_chunk) == len(catalog)
    totals = Counter()
    summaries = []
    needed = set()
    qrels = []
    for question in questions:
        qid = str(question['id'])
        rows, stats = [], Counter()
        assert {j['document_id'] for j in question['judgments']} == set(docs)
        for judgment in question['judgments']:
            did, dg = judgment['document_id'], judgment['grade']
            assert type(dg) is int and dg in (0, 1, 2)
            if dg:
                needed.add((qid, did))
            for i, chunk in enumerate(docs[did]['chunks']):
                if dg == 0:
                    grade, reason, method = 0, 'Documento com grau 0 para esta pergunta; zero herdado conforme regra solicitada.', 'inherited_document_zero'
                elif (qid, did) in reviews:
                    grade, reason = reviews[qid, did][i]
                    method = 'codex_semantic_review'
                else:
                    grade, reason, method = None, None, 'pending_codex_review'
                rows.append({'chunk_id': chunk['chunk_id'], 'document_id': did,
                             'document_grade': dg, 'grade': grade,
                             'method': method, 'rationale': reason})
                stats[str(grade)] += 1
                stats[method] += 1
                if grade is not None:
                    qrels.append(f'{qid}\t0\t{chunk["chunk_id"]}\t{grade}\n')
        assert len(rows) == len(catalog) == 5156
        write_json(args.output/'questions'/f'{int(qid):02d}.json', {
            'id': question['id'], 'question': question['question'],
            'reference_answer_used': True, 'judgments': rows})
        write_report(args.output/'reports'/f'{int(qid):02d}.md', question, rows, by_chunk)
        summaries.append({'id': question['id'], 'counts': dict(stats)})
        totals.update(stats)
    assert set(reviews) <= needed
    pending = sorted(needed-set(reviews), key=lambda x:(int(x[0]), x[1]))
    if args.require_complete:
        assert not pending, f'Missing semantic reviews: {pending}'
    # Never leave a partially populated qrels file that looks complete.
    qrels_path = args.output/'qrels.tsv'
    if pending:
        qrels_path.unlink(missing_ok=True)
    else:
        qrels_path.write_text(''.join(qrels), encoding='utf-8')
    with (args.output/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['question_id', 'non_relevant', 'partially_relevant',
                         'relevant', 'inherited_document_zero',
                         'codex_semantic_review', 'pending_codex_review'])
        for q in summaries:
            counts = q['counts']
            writer.writerow([q['id'], *(counts.get(k, 0) for k in
                             ['0', '1', '2', 'inherited_document_zero',
                              'codex_semantic_review', 'pending_codex_review'])])
    write_json(args.output/'manifest.json', {
        'schema_version': 1, 'unit': 'chunk',
        'status': 'in_progress' if pending else 'complete',
        'assessor': 'Codex semantic review; not independently human-validated',
        'labels': {'0': 'não relevante', '1': 'parcialmente relevante', '2': 'relevante'},
        'criteria': {'2': 'Suporte direto a pelo menos uma parte material da resposta, inclusive dado para cálculo.',
                     '1': 'Contexto útil mas insuficiente, ambíguo, conflitante ou incompleto devido à extração.',
                     '0': 'Sem suporte útil à pergunta; nos documentos grau 0, rótulo herdado por instrução.'},
        'source_judgments': fingerprint(DEFAULT_JUDGMENTS),
        'source_export': fingerprint(args.export/'manifest.json'),
        'review_sources': review_sources,
        'code': fingerprint(Path(__file__)),
        'reference_answers_used': True,
        'excluded_question_ids': sorted(EXCLUDED_IDS, key=int),
        'chunk_grade_not_capped_by_document_grade': True,
        'counts': dict(totals), 'question_count': len(questions),
        'chunk_count': len(catalog), 'pair_count': len(questions)*len(catalog),
        'reviewed_question_document_pairs': len(reviews),
        'pending_question_document_pairs': pending,
        'questions': summaries,
        'limitations': ['Julgamentos da IA, não ground truth humano independente.',
                       'Zeros documentais são herdados e não foram reavaliados semanticamente.',
                       'A referência orienta a revisão, mas o chunk precisa conter a evidência.',
                       'Figuras ausentes e fórmulas danificadas não têm conteúdo visual inferido.']})
    print(json.dumps({'counts': dict(totals), 'pending_document_pairs': len(pending)}, indent=2))


if __name__ == '__main__':
    main()
