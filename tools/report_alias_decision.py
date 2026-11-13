#!/usr/bin/env python3
"""Compare alias decision ablations against their unchanged hybrid controls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from report_dbpedia_comparison import load_run, question_diff, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-run', type=Path, required=True)
    parser.add_argument('--v2-run', type=Path, required=True)
    parser.add_argument('--contextual-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    paths = {'v1_legacy': args.v1_run, 'v2_legacy': args.v2_run, 'v2_contextual': args.contextual_run}
    runs = {label: load_run(path) for label, path in paths.items()}
    manifests = {label: read(path/'manifest.json') for label, path in paths.items()}
    reference = manifests['v1_legacy']
    output = {'runs': {}, 'differences': {}}
    for label, run in runs.items():
        m = manifests[label]
        assert m['source_manifest'] == reference['source_manifest']
        assert m['sources']['dense_ranking']['sha256'] == reference['sources']['dense_ranking']['sha256']
        assert read(paths[label]/'judgments.json') == read(args.v1_run/'judgments.json')
        for key in ('k1', 'b', 'rrf_k', 'dense_weight', 'bm25_weight', 'token_budget'):
            assert m['config'][key] == reference['config'][key]
        for qid in run['control']['contexts']:
            assert run['control']['contexts'][qid]['context'] == runs['v1_legacy']['control']['contexts'][qid]['context']
            assert run['control']['ranking'][qid] == runs['v1_legacy']['control']['ranking'][qid]
        diagnostics = list(run['aliases']['diagnostics'].values())
        decisions = [c for d in diagnostics for c in d['linking'].get('candidate_decisions', [])]
        output['runs'][label] = {'summary': run['aliases']['summary'], 'manifest': m,
                                 'abstained': sum(d['linking'].get('abstained', False) for d in diagnostics),
                                 'abstained_with_seeds': sum(d['linking'].get('abstained', False) and bool(d['linking']['seed_uris']) for d in diagnostics),
                                 'decision_reasons': dict(Counter(c['reason'] for c in decisions))}
    output['control'] = runs['v1_legacy']['control']['summary']
    for name, left, right in [('resource', runs['v1_legacy']['aliases'], runs['v2_legacy']['aliases']),
                              ('policy', runs['v2_legacy']['aliases'], runs['v2_contextual']['aliases']),
                              ('final_vs_control', runs['v2_contextual']['control'], runs['v2_contextual']['aliases'])]:
        qs = [question_diff(left, right, qid) for qid in left['contexts']]
        output['differences'][name] = {'questions': qs,
                                     'wins_ties_losses': dict(Counter('win' if q['delta_ndcg10'] > 1e-12 else 'loss' if q['delta_ndcg10'] < -1e-12 else 'tie' for q in qs)),
                                     'identical_contexts': sum(q['context_identical'] for q in qs)}
    args.output.mkdir(parents=True)
    for label, path in paths.items():
        directory = args.output/label
        directory.mkdir()
        for filename in ('summary.json', 'manifest.json'):
            shutil.copyfile(path/filename, directory/filename)
        for mode in ('control', 'aliases'):
            folder = directory/mode
            folder.mkdir()
            for filename in ('ranking-metrics.json', 'context-metrics.json', 'diagnostics.json'):
                shutil.copyfile(path/runs[label][mode]['summary']['name']/filename, folder/filename)
    (args.output/'comparison.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
