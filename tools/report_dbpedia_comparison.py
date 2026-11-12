#!/usr/bin/env python3
"""Summarize a DBpedia bundle and two frozen compare_kg runs (no model calls)."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.dbpedia_builder import validate_bundle, digest


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def cards(path):
    return {c['uri']: c for c in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())}


def stats(nodes):
    names = defaultdict(set)
    for u, c in nodes.items():
        for name in [c['label'], *c['aliases']]:
            names[name.casefold()].add(u)
    return {'cards': len(nodes), 'definitions': sum(bool(c['definition']) for c in nodes.values()),
            'fields': {k: {'present': sum(bool(c[k]) for c in nodes.values()),
                           'values': sum(len(c[k]) for c in nodes.values())}
                       for k in ['aliases', 'properties', 'related_concepts']},
            'ambiguous_casefolded_names': sum(len(v) > 1 for v in names.values()),
            'name_checks': {n: sorted(names[n.casefold()]) for n in ['OTR', 'DONES', 'IFMIF-DONES', 'AGI', 'FCM', 'DEMO']}}


def load_run(path):
    variants = {}
    for s in read(path/'summary.json'):
        folder = path/s['name']
        variants[s['mode']] = {'summary': s, 'ranking': {str(q['id']): q for q in read(folder/'ranking-metrics.json')['questions']},
                               'contexts': {str(q['id']): q for q in read(folder/'context.json')['questions']},
                               'diagnostics': {str(q['id']): q for q in read(folder/'diagnostics.json')}}
    return variants


def question_diff(a, b, qid):
    ac, bc = a['contexts'][qid], b['contexts'][qid]
    old, new = {c['chunk_id']: c for c in ac['chunks']}, {c['chunk_id']: c for c in bc['chunks']}
    ar, br = a['ranking'][qid], b['ranking'][qid]
    diag_a, diag_b = a['diagnostics'][qid], b['diagnostics'][qid]
    return {'id': qid, 'domain': bc['domain'], 'question': bc['question'],
            'old_ndcg10': ar['at_document_k']['10']['ndcg'], 'new_ndcg10': br['at_document_k']['10']['ndcg'],
            'delta_ndcg10': br['at_document_k']['10']['ndcg']-ar['at_document_k']['10']['ndcg'],
            'old_top10_documents': ar['document_ranking'][:10], 'new_top10_documents': br['document_ranking'][:10],
            'old_additions': diag_a['linking']['additions'], 'new_additions': diag_b['linking']['additions'],
            'old_seed_uris': diag_a['linking']['seed_uris'], 'new_seed_uris': diag_b['linking']['seed_uris'],
            'context_identical': ac['context'] == bc['context'],
            'context_chunk_jaccard': len(old.keys() & new.keys())/len(old.keys() | new.keys()),
            'added_chunks': [{'chunk_id': k, 'document_id': new[k]['document_id'], 'section_type': new[k]['section_type'],
                              'excerpt': new[k]['text'][:350]} for k in sorted(new.keys()-old.keys())],
            'removed_chunks': [{'chunk_id': k, 'document_id': old[k]['document_id'], 'section_type': old[k]['section_type'],
                                'excerpt': old[k]['text'][:350]} for k in sorted(old.keys()-new.keys())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--legacy-cards', type=Path, default=ROOT/'data/kg_rag_chunks.jsonl')
    parser.add_argument('--old-run', type=Path, required=True)
    parser.add_argument('--new-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file')
    manifest = validate_bundle(args.bundle)
    old_cards, new_cards = cards(args.legacy_cards), cards(args.bundle/'kg_rag_chunks.jsonl')
    old, new = load_run(args.old_run), load_run(args.new_run)
    old_m, new_m = read(args.old_run/'manifest.json'), read(args.new_run/'manifest.json')
    assert old_m['config'] == new_m['config'], 'Configuration changed'
    assert old_m['sources']['dense_ranking']['sha256'] == new_m['sources']['dense_ranking']['sha256']
    assert read(args.old_run/'judgments.json') == read(args.new_run/'judgments.json')
    assert old_m['source_manifest'] == new_m['source_manifest']
    comparisons = {}
    for mode in old:
        qs = [question_diff(old[mode], new[mode], qid) for qid in old[mode]['contexts']]
        summary = {'old': old[mode]['summary'], 'new': new[mode]['summary'], 'questions': qs,
                   'wins_ties_losses': dict(Counter('win' if q['delta_ndcg10'] > 1e-12 else 'loss' if q['delta_ndcg10'] < -1e-12 else 'tie' for q in qs)),
                   'identical_contexts': sum(q['context_identical'] for q in qs),
                   'domains': {domain: {'questions': sum(q['domain'] == domain for q in qs),
                                       'old_ndcg10': sum(q['old_ndcg10'] for q in qs if q['domain'] == domain)/sum(q['domain'] == domain for q in qs),
                                       'new_ndcg10': sum(q['new_ndcg10'] for q in qs if q['domain'] == domain)/sum(q['domain'] == domain for q in qs)}
                               for domain in sorted({q['domain'] for q in qs})}}
        comparisons[mode] = summary
    assert comparisons['control']['identical_contexts'] == 22, 'Control contexts differ'
    for q in new['control']['ranking']:
        assert old['control']['ranking'][q] == new['control']['ranking'][q], 'Control ranking differs'
    delta_control = {mode: [question_diff(new['control'], new[mode], qid) for qid in new[mode]['contexts']]
                     for mode in ('aliases', 'one_hop')}
    output = {'bundle_validation': 'passed', 'bundle_manifest': manifest,
              'bundle_total_bytes': sum(p.stat().st_size for p in args.bundle.iterdir() if p.is_file()),
              'old_cards': stats(old_cards), 'new_cards': stats(new_cards),
              'added_uris': sorted(new_cards.keys()-old_cards.keys()), 'removed_uris': sorted(old_cards.keys()-new_cards.keys()),
              'changed_labels': [{'uri': u, 'old': old_cards[u]['label'], 'new': new_cards[u]['label']}
                                 for u in sorted(old_cards.keys() & new_cards.keys()) if old_cards[u]['label'] != new_cards[u]['label']],
              'runs': {label: {'path': str(path), 'manifest_sha256': digest((path/'manifest.json').read_bytes()),
                              'kg_statistics': m['kg_statistics'], 'seconds': m['elapsed_seconds'],
                              'peak_rss_bytes': m['process_peak_rss_bytes']}
                       for label, path, m in [('old', args.old_run, old_m), ('new', args.new_run, new_m)]},
              'comparisons': comparisons, 'new_vs_control': delta_control,
              'limitations': ['Document-level provisional judgments; chunk excerpts are not evidence annotations.',
                              'No PDF parsing, embedding recomputation or LLM generation in this comparison.',
                              'Bundle construction completeness does not imply seed redirect semantic equivalence.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
