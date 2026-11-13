import json
from pathlib import Path
import tempfile
import unittest

from src.kg_expansion import KnowledgeExpansion
from tools.dbpedia_builder import name_from_uri, apply_redirect_review, uri


class ContextualAliasesTests(unittest.TestCase):
    def kg(self, cards, chunks):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)/'cards.jsonl'
        path.write_text('\n'.join(json.dumps(c) for c in cards))
        return KnowledgeExpansion(path, chunks, policy='contextual')

    def test_slashes_and_percent_encoding_survive_uri_name_extraction(self):
        self.assertEqual(name_from_uri(uri('dbr:T1/2')), 'T1/2')
        self.assertEqual(name_from_uri(uri('dbr:T1%2F2')), 'T1/2')
        self.assertEqual(name_from_uri(uri('dbr:Optical_transition_radiation')), 'Optical transition radiation')

    def test_support_required_and_no_repeated_query_terms(self):
        kg = self.kg([{'uri': 'n', 'label': 'Nitrogen', 'aliases': ['ATC code ABC', 'Nitrogen gas']}],
                     [{'chunk_id': 'a', 'text': 'Nitrogen gas temperature trap operating.'}])
        query, trace = kg.expand('What is the temperature of the Nitrogen trap?', 'aliases')
        self.assertEqual(query.count('Nitrogen'), 1)
        self.assertEqual(trace['additions'][0]['query_terms_added'], ['gas'])
        self.assertTrue(any(c['phrase'] == 'ATC code ABC' and c['reason'] == 'no_corpus_mention'
                            for c in trace['candidate_decisions']))
        self.assertEqual(trace['additions'][0]['support_chunk_ids'], ['a'])

    def test_unrelated_chunk_and_insufficient_context_abstain(self):
        kg = self.kg([{'uri': 'n', 'label': 'Nitrogen', 'aliases': ['Nitrogen gas']}],
                     [{'chunk_id': 'a', 'text': 'Nitrogen gas pharmaceutical coding.'}])
        for question in ['Nitrogen temperature trap?', 'Nitrogen?']:
            query, trace = kg.expand(question, 'aliases')
            self.assertEqual(query, question)
            self.assertTrue(trace['abstained'])

    def test_ambiguous_alias_does_not_link_or_expand(self):
        kg = self.kg([{'uri': 'a', 'label': 'Alpha', 'aliases': ['Shared alias']},
                      {'uri': 'b', 'label': 'Beta', 'aliases': ['Shared alias']}],
                     [{'chunk_id': 'a', 'text': 'Alpha Shared alias temperature trap.'}])
        self.assertEqual(kg.link('Shared alias'), [])
        query, trace = kg.expand('Alpha temperature trap', 'aliases')
        self.assertEqual(query, 'Alpha temperature trap')
        self.assertTrue(trace['abstained'])

    def test_numeric_alias_cannot_expand_and_control_is_exact(self):
        kg = self.kg([{'uri': 'n', 'label': 'Nitrogen', 'aliases': ['2']}],
                     [{'chunk_id': 'a', 'text': 'Nitrogen temperature trap 2.'}])
        question = 'Nitrogen temperature trap?'
        self.assertEqual(kg.expand(question, 'control')[0], question)
        self.assertEqual(kg.expand(question, 'aliases')[0], question)
        with self.assertRaises(ValueError): kg.expand(question, 'one_hop')

    def test_review_removes_equivalence_not_specific_node_and_preserves_original(self):
        source, target, excluded = uri('dbr:Specific'), uri('dbr:General'), uri('dbr:Demo')
        data = {'cards': [dict(uri=u, label=u.rsplit('/', 1)[-1], aliases=['Specific'],
                              names=[dict(text='Specific', kind='redirect', source_uri=source)], related_concepts=[])
                          for u in [source, target, excluded]],
                'triples': [[excluded, uri('rdfs:label'), '"Demo"@en']], 'edges': [],
                'seeds': [{'requested_uri': source, 'canonical_uri': target, 'metadata_status': 'present'}]}
        review = {'rejected_equivalences': [{'source_uri': source, 'target_uri': target, 'reason': 'Distinct concepts'}],
                  'excluded_entities': [{'uri': excluded, 'reason': 'Mixed target'}]}
        result = apply_redirect_review(data, review)
        self.assertEqual(len(data['cards']), 3)
        self.assertEqual(len(result['cards']), 2)
        self.assertEqual(result['seeds'][0]['canonical_uri'], source)
        self.assertEqual(next(c for c in result['cards'] if c['uri'] == target)['aliases'], [])
        self.assertFalse(result['triples'])


if __name__ == '__main__':
    unittest.main()
