#!/usr/bin/env python3
"""Check paired Colab outputs against frozen inputs; does not judge QA accuracy."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows):
    result = {str(r['id']):r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate question IDs')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports',type=Path,default=ROOT/'experiments/exports/colab-selection-order-2026-11-14')
    p.add_argument('--contexts',type=Path,default=ROOT/'experiments/runs/public22-ce-selection-order-2026-11-14')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hybrid-results',type=Path,help='Explicit hybrid-order result file')
    p.add_argument('--ce-results',type=Path,help='Explicit CE-order result file')
    args = p.parse_args()
    sample_path = ROOT/'IEEE-CIS-DSQA-Starter-Kit/sample/sample_qa.json'
    refs = indexed([q for q in read(sample_path) if int(q['id']) not in (16,18,21)])
    module = ast.parse((ROOT/'src/main.py').read_text())
    system_prompt = next(ast.literal_eval(n.value) for n in module.body
                         if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and
                         t.id=='SYSTEM_PROMPT' for t in n.targets))
    summaries, records, sources, metadata, contexts = [], {}, {}, {}, {}
    for order in ('hybrid','ce100'):
        name = f'select-hybrid-order-{order}-query-preserved'
        inp = args.exports/f'{name}.json'
        explicit = args.hybrid_results if order=='hybrid' else args.ce_results
        out = explicit if explicit is not None else args.exports/'results'/f'{name}_answers.json'
        ctx = args.contexts/name/'context.json'
        payload = read(out)
        metadata[order] = payload['metadata']
        if digest(inp) != payload['metadata']['input_sha256']:
            raise ValueError('Input hash mismatch')
        prompts, answers = indexed(read(inp)), indexed(payload['answers'])
        contexts[order] = indexed(read(ctx)['questions'])
        if not set(refs) == set(prompts) == set(answers) == set(contexts[order]):
            raise ValueError('Expected exactly the 22 active IDs')
        rows = []
        for qid,q in prompts.items():
            a, c, ref = answers[qid], contexts[order][qid], refs[qid]
            if ([m['role'] for m in q['prompt']] != ['system','user'] or
                    q['prompt'][0]['content'] != system_prompt):
                raise ValueError('System prompt mismatch')
            if q['expected_answer'] != ref['answer'] or a['expected_answer'] != ref['answer']:
                raise ValueError('Reference mismatch')
            if c['question'] != ref['question'] or q['prompt'][1]['content'] != c['context']+ref['question']:
                raise ValueError('Prompt mismatch')
            rows.append({'id':q['id'],'status':a['status'],'input_tokens':a.get('input_tokens'),
                         'answer_sha256':hashlib.sha256((a.get('answer') or '').encode()).hexdigest(),
                         'output_tokens':a.get('output_tokens'),
                         'output_at_limit':a.get('output_tokens')==payload['metadata']['max_new_tokens'],
                         'peak_allocated_gib':a['peak_allocated_gib'],
                         'peak_reserved_gib':a['peak_reserved_gib']})
        records[order] = rows
        summaries.append({'order':order,'questions':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
                          'output_limit_ids':[r['id'] for r in rows if r['output_at_limit']],
                          'mean_output_tokens':mean(r['output_tokens'] for r in rows if r['output_tokens'] is not None),
                          'peak_allocated_gib':max(r['peak_allocated_gib'] for r in rows),
                          'peak_reserved_gib':max(r['peak_reserved_gib'] for r in rows)})
        sources[order] = {str(f.resolve().relative_to(ROOT) if f.resolve().is_relative_to(ROOT)
                              else f.resolve()):digest(f) for f in (inp,out,ctx)}
    if {k:v for k,v in metadata['hybrid'].items() if k!='input_sha256'} != {
            k:v for k,v in metadata['ce100'].items() if k!='input_sha256'}:
        raise ValueError('Recorded generation configurations differ')
    for qid in refs:
        a,b = (contexts[o][qid] for o in ('hybrid','ce100'))
        key = lambda c:(c['chunk_id'],c['document_id'],c['text'])
        if Counter(map(key,a['chunks'])) != Counter(map(key,b['chunks'])):
            raise ValueError('Selected evidence differs')
    payload = {'summary':summaries,'per_question':records,'metadata':metadata,'sources':sources,
               'sample_sha256':digest(sample_path),'code_sha256':digest(Path(__file__)),
               'checks':{'input_hashes':True,'references':True,'prompts_match_contexts':True,
                         'system_prompt_matches_main':True,
                         'selected_evidence_equal':True,'recorded_generation_settings_equal':True},
               'limitations':['Structural audit, no automated or official answer judge.',
                              'Reaching token limit is an indicator; EOS stop reason was not logged.',
                              'Device map, GPU identity, model/tokenizer revisions and runtime were not recorded.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
