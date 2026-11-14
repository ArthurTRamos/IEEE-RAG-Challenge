#!/usr/bin/env python3
"""Export frozen contexts as ready chat prompts, with references kept separate."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(context_path, sample_path):
    sample = json.loads(sample_path.read_text())
    references = {str(q['id']):q for q in sample if int(q['id']) not in (16,18,21)}
    questions = json.loads(context_path.read_text())['questions']
    if len({str(q['id']) for q in questions}) != len(questions):
        raise ValueError('Duplicate question IDs')
    module = ast.parse((ROOT/'src/main.py').read_text())
    system = next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='SYSTEM_PROMPT' for t in n.targets))
    rows = []
    for q in questions:
        ref = references[str(q['id'])]
        if q['question'] != ref['question']:
            raise ValueError(f"Question differs from official sample: {q['id']}")
        rows.append({'id':q['id'], 'expected_answer':ref['answer'], 'prompt':[
            {'role':'system','content':system},
            {'role':'user','content':q['context']+q['question']}]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',type=Path,required=True)
    parser.add_argument('--sample',type=Path,default=ROOT/'IEEE-CIS-DSQA-Starter-Kit/sample/sample_qa.json')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rows = export(args.context,args.sample)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(rows,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(f'{len(rows)} prompts: {args.output}')


if __name__ == '__main__':
    main()
