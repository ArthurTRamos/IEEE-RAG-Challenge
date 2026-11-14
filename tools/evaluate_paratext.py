#!/usr/bin/env python3
"""Heuristic front matter / bibliography diagnostics; never filters retrieval."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

if __package__:
    from .evaluate_chunk_windows import analyze, index_questions
    from .evaluate_retrieval import load_judgments, ranked_chunks
    from .prepare_evaluation import ROOT, fingerprint, write_json
else:
    from evaluate_chunk_windows import analyze, index_questions
    from evaluate_retrieval import load_judgments, ranked_chunks
    from prepare_evaluation import ROOT, fingerprint, write_json

HEADING = re.compile(r'^(#{1,6})\s+(.+)$', re.M)
ANCHOR = re.compile(r'^(?:#{1,6}\s+)?(?:\*\*)?(?:\d+[.\s]*)?(?:abstract|resumo|introduction|introdução)\b', re.M|re.I)
REFERENCES = re.compile(r'^(?:\d+[.\s]*)?(?:references|bibliography|referências|literature cited)\s*[:.]?$', re.I)
CATEGORIES = ('front_matter_candidate', 'bibliography', 'body_candidate', 'unknown')


def annotate_catalog(chunks, overlap=150):
    documents=defaultdict(list)
    for chunk in chunks: documents[chunk['document_id']].append(chunk)
    annotations={}
    for rows in documents.values():
        rows.sort(key=lambda r:int(r['chunk_id'].rsplit('#',1)[1]))
        text=''; intervals=[]
        for i,row in enumerate(rows):
            tail=min(overlap,len(rows[i-1]['text'])) if i else 0
            if tail and row['text'][:tail]!=text[-tail:]:
                raise ValueError(f"Invalid overlap: {row['chunk_id']}")
            start=len(text)-tail
            text+=row['text'][tail:]
            intervals.append((start,len(text),row))
        anchor=ANCHOR.search(text)
        events={0:'front_matter_candidate' if anchor and anchor.start()>0 else 'unknown'}
        if anchor: events[anchor.start()]='body_candidate'
        reference_level=None
        for heading in HEADING.finditer(text):
            level=len(heading.group(1)); title=heading.group(2).strip().strip('* ')
            if REFERENCES.fullmatch(title):
                reference_level=level; events[heading.start()]='bibliography'
            elif reference_level is not None and level<=reference_level:
                reference_level=None; events[heading.start()]='body_candidate' if anchor and heading.start()>=anchor.start() else 'unknown'
        points=sorted(events)
        regions=[(pos, points[i+1] if i+1<len(points) else len(text),events[pos]) for i,pos in enumerate(points)]
        for start,end,row in intervals:
            spans=[{'start':max(start,a)-start,'end':min(end,b)-start,'category':label}
                   for a,b,label in regions if min(end,b)>max(start,a)]
            annotations[row['chunk_id']]={'document_id':row['document_id'],'original_length':len(row['text']),
                'spans':spans,'body_anchor_detected':anchor is not None}
    return annotations


def classify(annotation, length=None):
    length=annotation['original_length'] if length is None else length
    if length<0 or length>annotation['original_length']:
        raise ValueError('Selected text length outside original chunk')
    counts=Counter()
    for span in annotation['spans']:
        counts[span['category']]+=max(0,min(span['end'],length)-span['start'])
    counts=Counter({k:v for k,v in counts.items() if v>0})
    primary=max(counts,key=counts.get) if counts else 'unknown'
    return {'primary_category':primary,'mixed':len(counts)>1,
            'front_matter_characters':counts['front_matter_candidate'],
            'bibliography_characters':counts['bibliography'],
            'body_characters':counts['body_candidate'],'unknown_characters':counts['unknown'],
            'front_or_bibliography_fraction':(counts['front_matter_candidate']+counts['bibliography'])/length if length else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'experiments/runs/public22-ce-minilm-l6-2026-11-13')
    parser.add_argument('--catalog-source',type=Path,default=ROOT/'experiments/runs/baseline-public22-ranking/ranking.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--k',type=int,default=50)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):parser.error('Output must be empty')
    _,_,sources=analyze(args.source,args.k)
    origin=json.loads((args.source/'manifest.json').read_text())
    if fingerprint(args.catalog_source)['sha256'] != origin['source_ranking']['sha256']:
        raise ValueError('Text catalog is not the dense source used by the reranker run')
    judgments=json.loads((args.source/'judgments.json').read_text());qrels=load_judgments(judgments)
    configs=json.loads((args.source/'summary.json').read_text())
    catalog_payload=json.loads(args.catalog_source.read_text())
    chunks=catalog_payload['questions'][0]['chunks']
    catalog={c['chunk_id']:c for c in chunks}
    if len(catalog)!=len(chunks):raise ValueError('Duplicate catalog IDs')
    annotations=annotate_catalog(chunks)
    write_json(args.output_dir/'annotations.json',annotations)
    per_question=[]; per_chunk=[]; summaries=[]
    for config in configs:
        folder=args.source/config['name']
        for scope,file in ((f'top{args.k}','ranking.json'),('actual_context','context.json')):
            records=index_questions(json.loads((folder/file).read_text()),qrels)
            scope_rows=[]
            for qid,q in records.items():
                ranked=list(ranked_chunks(q))
                if any(c['chunk_id'] not in catalog or c['document_id']!=catalog[c['chunk_id']]['document_id'] for c in ranked):
                    raise ValueError('Ranking/catalog mismatch')
                selected=ranked[:args.k] if scope!='actual_context' else ranked
                counts=Counter(); chars=Counter(); mixed=0
                for c in selected:
                    original=catalog[c['chunk_id']]['text']; actual=c.get('text',original)
                    if not original.startswith(actual):raise ValueError('Selected text is not a prefix of original chunk')
                    label=classify(annotations[c['chunk_id']],len(actual))
                    counts[label['primary_category']]+=1;mixed+=label['mixed']
                    for key in ('front_matter_characters','bibliography_characters','body_characters','unknown_characters'):chars[key]+=label[key]
                    per_chunk.append({'run':config['name'],'scope':scope,'id':qid,'chunk_id':c['chunk_id'],
                        'document_id':c['document_id'],'document_grade':qrels[qid][c['document_id']],
                        **label,'excerpt':actual[:180].replace('\n',' ')})
                row={'run':config['name'],'candidates':config['candidates'],'scope':scope,'id':qid,
                    'total_chunks':len(selected),**{k:counts[k] for k in CATEGORIES},'mixed_chunks':mixed,
                    'any_front_or_bibliography_chunks':sum(r['front_or_bibliography_fraction']>0 for r in per_chunk[-len(selected):]) if selected else 0,
                    **dict(chars)}
                scope_rows.append(row);per_question.append(row)
            totals={key:sum(r[key] for r in scope_rows) for key in ('total_chunks',*CATEGORIES,'mixed_chunks','any_front_or_bibliography_chunks')}
            summaries.append({'run':config['name'],'candidates':config['candidates'],'scope':scope,'question_count':len(scope_rows),
                'totals':totals,'mean_per_question':{k:v/len(scope_rows) for k,v in totals.items()}})
    for file,value in [('summary',summaries),('per-question',per_question)]:write_json(args.output_dir/f'{file}.json',value)
    for file,rows in [('per-question',per_question),('per-chunk',per_chunk)]:
        with (args.output_dir/f'{file}.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(args.output_dir/'manifest.json',{'status':'complete','sources':{**sources,'catalog':fingerprint(args.catalog_source)},
        'code':fingerprint(Path(__file__)),'k':args.k,
        'definitions':{'primary_category':'Largest character span; ties prefer earlier span',
            'mixed':'More than one nonempty category span; overlaps count as present',
            'front_matter_candidate':'Prefix before first recognized abstract/introduction marker; not verified cover page',
            'bibliography':'Recognized references heading until next same/shallower heading',
            'body_candidate':'After body marker, outside bibliography; not a relevance judgment',
            'unknown':'No confident structural anchor; never counted as body by default',
            'limits':'Heuristics on exported Markdown; no PDF page/layout provenance. No filtering applied.'}})
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
