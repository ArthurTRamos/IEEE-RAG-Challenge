#!/usr/bin/env python3
"""Domain alignment diagnostics on frozen rankings; no filtering or inference."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

if __package__:
    from .evaluate_chunk_windows import analyze, index_questions
    from .evaluate_retrieval import load_judgments, ranked_chunks
    from .prepare_evaluation import ROOT, DEFAULT_JUDGMENTS, DEFAULT_QUESTIONS, fingerprint, write_json
else:
    from evaluate_chunk_windows import analyze, index_questions
    from evaluate_retrieval import load_judgments, ranked_chunks
    from prepare_evaluation import ROOT, DEFAULT_JUDGMENTS, DEFAULT_QUESTIONS, fingerprint, write_json


def canonical_domain(value):
    mapping = {'DONES':'IFMIF-DONES','IFMIF-DONES':'IFMIF-DONES',
               'AI Papers':'AI/mathematics','AI/mathematics':'AI/mathematics'}
    if value not in mapping:
        raise ValueError(f'Unknown domain: {value!r}')
    return mapping[value]


def domain_counts(chunks, question_domain, document_domains, grades):
    question_domain = canonical_domain(question_domain)
    counts, cross = Counter(), Counter()
    for chunk in chunks:
        doc = chunk['document_id']
        domain = canonical_domain(document_domains[doc])
        counts[domain] += 1
        cross[f"{'same' if domain==question_domain else 'other'}|grade{grades[doc]}"] += 1
    total = len(chunks)
    same = counts[question_domain]
    return {'total_chunks':total,'same_domain_chunks':same,'other_domain_chunks':total-same,
            'same_domain_fraction':same/total if total else None,
            'same_domain_grade0_chunks':cross['same|grade0'],
            'same_domain_grade1_chunks':cross['same|grade1'],
            'same_domain_grade2_chunks':cross['same|grade2'],
            'other_domain_grade1_or2_chunks':cross['other|grade1']+cross['other|grade2']}


def summarize(rows):
    total = sum(r['total_chunks'] for r in rows)
    valid = [r['same_domain_fraction'] for r in rows if r['same_domain_fraction'] is not None]
    return {'question_count':len(rows),
            'mean_same_domain_chunks':sum(r['same_domain_chunks'] for r in rows)/len(rows),
            'mean_total_chunks':total/len(rows),
            'macro_same_domain_fraction':sum(valid)/len(valid) if valid else None,
            'micro_same_domain_fraction':sum(r['same_domain_chunks'] for r in rows)/total if total else None,
            'mean_same_domain_grade0_chunks':sum(r['same_domain_grade0_chunks'] for r in rows)/len(rows),
            'other_domain_relevant_chunk_occurrences':sum(r['other_domain_grade1_or2_chunks'] for r in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'experiments/runs/public22-ce-minilm-l6-2026-11-13')
    p.add_argument('--metadata', type=Path, default=DEFAULT_JUDGMENTS)
    p.add_argument('--questions', type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument('--k',type=int,default=50)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): p.error('Output directory must be empty')
    # Reuse complete question/catalog/context consistency checks from window evaluator.
    _,_,validated_sources=analyze(args.source,args.k)
    judged=json.loads((args.source/'judgments.json').read_text());qrels=load_judgments(judged)
    metadata=json.loads(args.metadata.read_text()); documents={d['document_id']:d for d in metadata['corpus_documents']}
    for doc in judged['corpus_documents']:
        if doc['pdf_sha256']!=documents[doc['document_id']]['pdf_sha256']:
            raise ValueError('Domain metadata references another PDF version')
    domains={doc:canonical_domain(d['domain']) for doc,d in documents.items()}
    questions={str(q['id']):q for q in json.loads(args.questions.read_text())}
    if not set(qrels)<=set(questions): raise ValueError('Missing question domain metadata')
    configs=json.loads((args.source/'summary.json').read_text())
    rows=[]
    for config in configs:
        folder=args.source/config['name']
        for scope,filename in ((f'top{args.k}','ranking.json'),('actual_context','context.json')):
            records=index_questions(json.loads((folder/filename).read_text()),qrels)
            for qid,q in records.items():
                expected=questions[qid]
                if q['question']!=expected['question'] or canonical_domain(q['domain'])!=canonical_domain(expected['domain']):
                    raise ValueError(f'Question domain/text mismatch: {qid}')
                chunks=list(ranked_chunks(q))
                if scope!='actual_context': chunks=chunks[:args.k]
                rows.append({'run':config['name'],'candidates':config['candidates'],'scope':scope,'id':qid,
                             'question_domain':canonical_domain(expected['domain']),
                             **domain_counts(chunks,expected['domain'],domains,qrels[qid])})
    controls={(r['scope'],r['id']):r for r in rows if r['candidates']==0}
    for row in rows:
        baseline=controls[row['scope'],row['id']]
        row['delta_same_domain_chunks']=row['same_domain_chunks']-baseline['same_domain_chunks']
        row['delta_same_domain_fraction']=(row['same_domain_fraction']-baseline['same_domain_fraction']
            if row['same_domain_fraction'] is not None and baseline['same_domain_fraction'] is not None else None)
    summaries=[]
    for config in configs:
        for scope in (f'top{args.k}','actual_context'):
            selected=[r for r in rows if r['run']==config['name'] and r['scope']==scope]
            by_domain={d:summarize([r for r in selected if r['question_domain']==d]) for d in sorted({r['question_domain'] for r in selected})}
            summaries.append({'run':config['name'],'candidates':config['candidates'],'scope':scope,
                              **summarize(selected),'by_question_domain':by_domain,
                              'paired_fraction':{label:sum(test(r['delta_same_domain_fraction']) for r in selected if r['delta_same_domain_fraction'] is not None)
                               for label,test in [('improved',lambda x:x>1e-12),('worse',lambda x:x< -1e-12),('tied',lambda x:abs(x)<=1e-12)]}})
    write_json(args.output_dir/'summary.json',summaries)
    write_json(args.output_dir/'per-question.json',rows)
    with (args.output_dir/'per-question.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(args.output_dir/'manifest.json',{'status':'complete','sources':{**validated_sources,'domain_metadata':fingerprint(args.metadata),'questions':fingerprint(args.questions)},
        'code':fingerprint(Path(__file__)),'chunk_k':args.k,
        'definitions':{'domain':'Source document domain from imported metadata, not semantic passage classification',
            'macro':'Mean of per-question domain fractions','micro':'Same-domain chunk occurrences / all chunk occurrences',
            'limits':'Domain alignment is not passage relevance or QA accuracy; no domain filtering was applied'}})
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
