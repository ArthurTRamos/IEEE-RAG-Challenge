import json
import re
import tempfile
import unittest
from pathlib import Path

from tools.dbpedia_builder import (Client, IncompleteQuery, alias_index, build, extract, publish, rdf_value,
                                   resolve_seeds, uri, validate_bundle)


def binding(value, kind='uri', **extra):
    return {'value': value, 'type': kind, **extra}


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = json.loads(Path('notebooks/dbpedia-config.json').read_text())
        self.cfg.update(page_size=2, max_pages=5, SEED_ENTITIES=['dbr:Old'],
                        SEED_CATEGORIES=['dbc:Test'], depth=0)

    def client(self, transport, cfg=None):
        c = Client(cfg or self.cfg, self.root / 'cache.sqlite', transport, lambda _: None)
        self.addCleanup(c.close)
        return c

    def response(self, rows, headers=None):
        return json.dumps({'results': {'bindings': rows}}).encode(), headers or {}

    def test_cache_is_scoped_and_partial_json_is_not_reused(self):
        calls = []
        def transport(endpoint, q):
            calls.append(q)
            return self.response([{'x': binding('a')}], {'X-SQL-State': 'S1TAT'} if len(calls) == 1 else {})
        c = self.client(transport)
        with self.assertRaises(IncompleteQuery):
            c.query('q', retries=1)
        self.assertEqual(c.conn.execute('SELECT status FROM responses_v3').fetchone()[0], 'partial')
        self.assertEqual(len(c.query('q', retries=1)), 1)
        c.query('q')
        self.assertEqual(len(calls), 2)
        cfg = dict(self.cfg, snapshot='new')
        self.client(transport, cfg).query('q')
        self.assertEqual(len(calls), 3)

    def test_malformed_response_retried_and_failed_marked(self):
        c = self.client(lambda e, q: (b'{"results":{"bindings": [S1TAT', {}))
        with self.assertRaises(IncompleteQuery): c.query('q', retries=1)
        self.assertEqual(c.conn.execute('SELECT status FROM responses_v3').fetchone()[0], 'partial')
        other = self.client(lambda e, q: (_ for _ in ()).throw(OSError('offline')))
        with self.assertRaises(IncompleteQuery): other.query('other', retries=1)
        self.assertIn('failed', [r[0] for r in other.conn.execute('SELECT status FROM responses_v3')])

    def test_reported_row_cap_is_not_accepted_as_complete(self):
        c = self.client(lambda e, q: self.response([{'x': binding('a')}], {'X-SPARQL-MaxRows': '1'}))
        with self.assertRaisesRegex(IncompleteQuery, 'row limit'): c.query('cap', retries=1)
        c = self.client(lambda e, q: self.response([{'x': binding('a')}], {'X-SPARQL-MaxRows': '10000'}))
        self.assertEqual(len(c.query('not-capped', retries=1)), 1)

    def test_rdf_values_preserve_datatypes_languages_and_escaping(self):
        self.assertEqual(rdf_value(binding('3.14', 'typed-literal', datatype='http://www.w3.org/2001/XMLSchema#decimal')),
                         '"3.14"^^<http://www.w3.org/2001/XMLSchema#decimal>')
        self.assertEqual(rdf_value(binding('a "quote"\n', 'literal', **{'xml:lang': 'en'})),
                         json.dumps('a "quote"\n') + '@en')
        with self.assertRaises(ValueError): rdf_value(binding('x', 'literal', **{'xml:lang': 'en . bad'}))
        with self.assertRaises(ValueError): uri('http://example.org/bad name')

    def test_ordered_pagination_including_exact_page_boundary(self):
        calls = []
        def transport(endpoint, q):
            calls.append(q)
            start = int(re.search(r'OFFSET (\d+)', q)[1])
            return self.response([{'x': binding(str(i))} for i in range(4)][start:start+2])
        c = self.client(transport)
        self.assertEqual(len(c.paged('SELECT DISTINCT ?x WHERE { ?x ?p ?o }')), 4)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all('ORDER BY ?x' in q for q in calls))

    def test_repeated_pages_and_budget_exhaustion_rejected(self):
        c = self.client(lambda e, q: self.response([{'x': binding('a')}, {'x': binding('b')}]))
        with self.assertRaisesRegex(IncompleteQuery, 'repeated'): c.paged('SELECT DISTINCT ?x WHERE {}')
        c = self.client(lambda e, q: self.response([{'x': binding(q)}, {'x': binding('b')}]), dict(self.cfg, max_pages=1))
        with self.assertRaisesRegex(IncompleteQuery, 'budget'): c.paged('SELECT DISTINCT ?x WHERE { ?x ?p ?o }')

    def test_batch_split_and_single_failure(self):
        c = self.client(lambda e, q: self.response([]))
        calls = []
        def paged(q):
            calls.append(q)
            if len(q.split(',')) > 1: raise IncompleteQuery('heavy')
            return [{'x': binding(q)}]
        c.paged = paged
        self.assertEqual(len(c.batch(['b', 'a'], lambda b: ','.join(b))), 2)
        self.assertEqual(calls, ['a,b', 'a', 'b'])
        c.paged = lambda q: (_ for _ in ()).throw(IncompleteQuery('heavy singleton'))
        with self.assertRaises(IncompleteQuery): c.batch(['a'], str)
        self.assertEqual(c.failures[0]['uri'], 'a')

    def endpoint(self, endpoint, q):
        offset = int(re.search(r'OFFSET (\d+)', q)[1])
        if 'SELECT DISTINCT ?target ' in q:
            rows = [{'target': binding(uri('dbr:Canonical'))}] if '<' + uri('dbr:Old') + '>' in q else []
        elif 'SELECT DISTINCT ?entity ?cat ' in q:
            rows = []
        elif 'SELECT DISTINCT ?entity ?label ?comment ' in q:
            rows = [{'entity': binding(uri('dbr:Canonical')), 'label': binding('Canonical', 'literal', **{'xml:lang': 'en'})}]
        elif 'SELECT DISTINCT ?entity ?redirect ' in q:
            rows = [{'entity': binding(uri('dbr:Canonical')), 'redirect': binding(uri('dbr:' + name))}
                    for name in ['Old', 'An_alternative_name_that_is_longer_than_35_characters', *[f'Alias_{i}' for i in range(20)]]]
        else:
            rows = []
        return self.response(rows[offset:offset+self.cfg['page_size']])

    def test_full_build_preserves_seed_chain_and_uncapped_names(self):
        # Lots of aliases require more than the five-page default test budget.
        self.cfg['max_pages'] = 30
        output = self.root / 'run'
        build(self.cfg, self.root / 'cache.sqlite', output, self.endpoint, lambda _: None)
        manifest = validate_bundle(output)
        self.assertEqual(manifest['seeds'][0]['chain'], [uri('dbr:Old'), uri('dbr:Canonical')])
        card = json.loads((output / 'kg_rag_chunks.jsonl').read_text())
        self.assertGreater(len(card['aliases']), 15)
        self.assertTrue(any(len(a) > 35 for a in card['aliases']))
        self.assertTrue(any(n['kind'] == 'seed_redirect' for n in card['names']))
        before = (output / 'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            publish(output, extract(self.client(self.endpoint)), self.cfg, 'test')
        self.assertEqual(before, (output / 'manifest.json').read_bytes())

    def test_cycle_ambiguous_redirect_and_hop_limit(self):
        for targets, expected in [(['Old'], 'redirect_cycle'), (['A', 'B'], 'ambiguous_redirect'), (['Next'], 'hop_limit')]:
            c = self.client(lambda e, q: self.response([]), dict(self.cfg, max_redirect_hops=1))
            c.batch = lambda v, q: [{'target': binding(uri('dbr:' + x))} for x in targets]
            self.assertEqual(resolve_seeds(c, ['dbr:Old'])[0]['status'], expected)

    def test_missing_seed_does_not_publish_and_writes_diagnostics(self):
        def endpoint(e, q):
            if '?label ?comment' in q: return self.response([])
            return self.endpoint(e, q)
        output = self.root / 'bad'
        with self.assertRaisesRegex(IncompleteQuery, 'without labels'):
            build(self.cfg, self.root / 'cache.sqlite', output, endpoint, lambda _: None)
        self.assertFalse(output.exists())
        self.assertEqual(json.loads(Path(str(output)+'.failed.json').read_text())['status'], 'failed')

    def test_explicit_missing_seed_exception_is_reported_without_fake_card(self):
        self.cfg['max_pages'] = 30
        self.cfg['SEED_ENTITIES'].append('dbr:Absent')
        self.cfg['allowed_missing_seed_labels'] = {uri('dbr:Absent'): 'Known absent in this snapshot'}
        result = extract(self.client(self.endpoint))
        excluded = next(s for s in result['seeds'] if s['requested_uri'] == uri('dbr:Absent'))
        self.assertEqual(excluded['metadata_status'], 'excluded_missing_label')
        self.assertIn(uri('dbr:Absent'), result['missing_metadata_uris'])
        self.assertNotIn(uri('dbr:Absent'), {c['uri'] for c in result['cards']})
        output = self.root / 'with-exception'
        publish(output, result, self.cfg, 'ns')
        self.assertIn(excluded, validate_bundle(output)['seeds'])
        self.cfg['allowed_missing_seed_labels'] = {}
        with self.assertRaises(IncompleteQuery) as caught:
            extract(self.client(self.endpoint))
        failed = json.loads(str(caught.exception).split(': ', 1)[1])
        self.assertEqual([s['requested_uri'] for s in failed], [uri('dbr:Absent')])

    def test_seed_exception_policy_preserves_query_cache_namespace(self):
        original = dict(self.cfg)
        original.pop('allowed_missing_seed_labels', None)
        first = self.client(self.endpoint, original)
        changed = dict(original, allowed_missing_seed_labels={uri('dbr:Absent'): 'Explicit exception'})
        self.assertEqual(first.namespace, self.client(self.endpoint, changed).namespace)

    def test_ambiguity_is_preserved(self):
        cards = [{'uri': uri('dbr:' + n), 'names': [{'text': 'Lithium-7'}]} for n in ['Lithium-7', 'Isotopes_of_lithium']]
        self.assertEqual(len(alias_index(cards)['lithium-7']), 2)

    def test_publication_deterministic_budget_and_corruption(self):
        card = {'uri': uri('dbr:A'), 'label': 'A', 'names': [{'text': 'A'}]}
        data = {'cards': [card], 'triples': [(uri('dbr:A'), uri('rdfs:label'), '"A"@en')],
                'edges': [], 'seeds': [], 'missing_metadata_uris': []}
        a, b = self.root / 'a', self.root / 'b'
        publish(a, data, self.cfg, 'ns')
        publish(b, data, self.cfg, 'ns')
        for p in a.iterdir(): self.assertEqual(p.read_bytes(), (b / p.name).read_bytes())
        with self.assertRaisesRegex(ValueError, 'budget'):
            publish(self.root / 'too-big', data, dict(self.cfg, max_bundle_bytes=1), 'ns')
        self.assertFalse((self.root / 'too-big').exists())
        self.assertFalse(list(self.root.glob('.dbpedia-*')))
        (a / 'domain_kg.ttl').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): validate_bundle(a)

    def test_reviewed_alias_requires_source_and_present_target(self):
        self.cfg['max_pages'] = 30
        self.cfg['reviewed_aliases'] = [{'text': 'CAN', 'canonical_uri': 'dbr:Canonical'}]
        with self.assertRaisesRegex(ValueError, 'text and source'): extract(self.client(self.endpoint))
        self.cfg['reviewed_aliases'][0]['source'] = 'local-review:document#chunk'
        result = extract(self.client(self.endpoint))
        self.assertIn('CAN', result['cards'][0]['aliases'])
        self.cfg['reviewed_aliases'][0]['canonical_uri'] = 'dbr:Missing'
        with self.assertRaisesRegex(ValueError, 'absent'): extract(self.client(self.endpoint))


if __name__ == '__main__':
    unittest.main()
