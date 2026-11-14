#!/usr/bin/env python3
"""Audit four frozen Colab order variants; no automatic answer correctness score."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from audit_colab_results import ROOT, read, digest, indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory')
    ex = ROOT/'experiments/exports'
    snapshots = ROOT/'experiments/results/colab-selection-order-review-2048-2026-11-14'
    old = ex/'colab-selection-order-2026-11-14'
    e512 = ex/'colab-ettin68m-selection-order-2026-11-14'
    e1024 = ex/'colab-ettin68m-maxlen1024-selection-order-2026-11-14'
    run = ROOT/'experiments/runs'
    configs = {
        'hybrid':(old,'select-hybrid-order-hybrid-query-preserved',snapshots/'hybrid-answers-2048.json',run/'public22-ce-selection-order-2026-11-14'),
        'minilm512':(old,'select-hybrid-order-ce100-query-preserved',snapshots/'ce100-answers-2048.json',run/'public22-ce-selection-order-2026-11-14'),
        'ettin512':(e512,'select-hybrid-order-ettin68m-query-preserved',None,run/'public22-ettin68m-selection-order-2026-11-14'),
        'ettin1024':(e1024,'select-hybrid-order-ettin68m-l1024-query-preserved',None,run/'public22-ettin68m-l1024-selection-order-2026-11-14'),
    }
    refs = indexed([r for r in read(ROOT/'IEEE-CIS-DSQA-Starter-Kit/sample/sample_qa.json') if r['id'] not in (16,18,21)])
    tree = ast.parse((ROOT/'src/main.py').read_text())
    system = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='SYSTEM_PROMPT' for t in n.targets))
    all_prompts, all_answers, metadata, summaries, sources = {}, {}, {}, [], {}
    evidence = None
    for key,(folder,name,result,context_run) in configs.items():
        inp = folder/f'{name}.json'
        result = result or folder/'results'/f'{name}_answers.json'
        context_path = context_run/name/'context.json'
        d = read(result)
        prompts, answers, contexts = indexed(read(inp)), indexed(d['answers']), indexed(read(context_path)['questions'])
        if not set(prompts)==set(answers)==set(contexts)==set(refs):
            raise ValueError('Expected the 22 active IDs exactly once')
        if digest(inp)!=d['metadata']['input_sha256']:
            raise ValueError(f'Input hash mismatch: {key}')
        current_evidence = {}
        for qid,q in prompts.items():
            a,c,ref = answers[qid],contexts[qid],refs[qid]
            if a['expected_answer']!=ref['answer'] or q['expected_answer']!=ref['answer']:
                raise ValueError('Reference mismatch')
            expected_prompt = [{'role':'system','content':system},
                               {'role':'user','content':c['context']+ref['question']}]
            if q['prompt']!=expected_prompt or c['question']!=ref['question']:
                raise ValueError('Prompt mismatch')
            if a['status']!='ok' or not isinstance(a['answer'],str) or not a['answer'].strip():
                raise ValueError('Incomplete or empty result')
            current_evidence[qid] = Counter((r['chunk_id'],r['document_id'],r['text']) for r in c['chunks'])
        if evidence is not None and current_evidence!=evidence:
            raise ValueError('Different selected evidence')
        evidence = current_evidence
        all_prompts[key], all_answers[key], metadata[key] = prompts, answers, d['metadata']
        sources[key] = {str(p.relative_to(ROOT)):digest(p) for p in (inp,result,context_path)}
        summaries.append({'variant':key,'questions':len(answers),
                          'output_limit_ids':[r['id'] for r in answers.values() if r['output_tokens']==d['metadata']['max_new_tokens']],
                          'mean_output_tokens':mean(r['output_tokens'] for r in answers.values()),
                          'peak_allocated_gib':max(r['peak_allocated_gib'] for r in answers.values()),
                          'peak_reserved_gib':max(r['peak_reserved_gib'] for r in answers.values())})
    settings = [{k:v for k,v in m.items() if k!='input_sha256'} for m in metadata.values()]
    if any(m!=settings[0] for m in settings):
        raise ValueError('Recorded generation settings differ')
    paired = []
    for qid in refs:
        changed_prompt = all_prompts['ettin512'][qid]['prompt']!=all_prompts['ettin1024'][qid]['prompt']
        changed_answer = all_answers['ettin512'][qid]['answer']!=all_answers['ettin1024'][qid]['answer']
        paired.append({'id':refs[qid]['id'],'ettin_window_prompt_changed':changed_prompt,
                       'ettin_window_answer_changed':changed_answer,
                       'answer_hashes':{k:digest_text(a[qid]['answer']) for k,a in all_answers.items()},
                       'output_tokens':{k:a[qid]['output_tokens'] for k,a in all_answers.items()}})
    payload = {'summary':summaries,'per_question':paired,'metadata':metadata,'sources':sources,
               'same_selected_evidence':True,'same_recorded_generation_settings':True,
               'code_sha256':digest(Path(__file__)),
               'limitations':['No correctness judge or official score; qualitative review is separate.',
                              'EOS stop reason, model revisions, GPU device map and timing are not recorded.']}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    for key,(_,_,path,_) in configs.items():
        # Preserve newly supplied results against later replacement in the upload folder.
        (args.output_dir/f'{key}-answers.json').write_text(json.dumps(
            {'metadata':metadata[key],'answers':list(all_answers[key].values())},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'summary':summaries,'changed_prompts':[p['id'] for p in paired if p['ettin_window_prompt_changed']],
                      'changed_answers':[p['id'] for p in paired if p['ettin_window_answer_changed']]},indent=2))


def digest_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__=='__main__':
    main()
