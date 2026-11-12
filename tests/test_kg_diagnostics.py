import json
from pathlib import Path
import tempfile
import unittest
from src.retrieval import _chunk_markdown
from src.chunk_diagnostics import annotate_sections, context_diagnostics
from src.kg_expansion import KnowledgeExpansion


class KGDiagnosticsTests(unittest.TestCase):
    def test_section_reconstruction_preserves_mixed_and_reference_chunks(self):
        text = '# Title\n\n## Introduction\n\n' + 'intro text '*100 + '\n\n## References\n\n' + 'citation '*100
        parts = _chunk_markdown(text)
        chunks = [{'chunk_id':f'A.pdf#{i}', 'document_id':'A', 'text':s} for i,s in enumerate(parts)]
        annotations = annotate_sections(chunks)
        self.assertEqual(annotations[chunks[-1]['chunk_id']]['section_type'], 'references')
        self.assertTrue(any(a['mixed_sections'] for a in annotations.values()))
        result = context_diagnostics(chunks, {'A':2}, annotations)
        self.assertEqual(result['fraction_from_grade2_documents'], 1)
        self.assertEqual(sum(result['by_primary_section'].values()), len(chunks))
        broken = [dict(c) for c in chunks]; broken[1]['text'] = 'wrong overlap'
        with self.assertRaises(ValueError): annotate_sections(broken)

    def test_alias_linking_boundaries_one_hop_and_no_match(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'kg.jsonl'
            cards = [{'uri':'a','label':'Alpha physics','aliases':['Alphatron'],'related_concepts':['Beta physics']},
                     {'uri':'b','label':'Beta physics','aliases':[],'related_concepts':['Gamma physics']},
                     {'uri':'c','label':'Gamma physics','aliases':[],'related_concepts':[]}]
            p.write_text('\n'.join(json.dumps(c) for c in cards))
            chunks = [{'chunk_id':str(i),'text':c['label']} for i,c in enumerate(cards)]
            kg = KnowledgeExpansion(p, chunks)
            self.assertEqual(kg.link('notAlphatron'), [])
            alias, _ = kg.expand('Alphatron', 'aliases')
            self.assertIn('Alpha physics', alias)
            hop, _ = kg.expand('Alphatron', 'one_hop')
            self.assertIn('Beta physics', hop)
            self.assertNotIn('Gamma physics', hop)
            self.assertEqual(kg.expand('no match', 'one_hop')[0], 'no match')
            self.assertEqual(kg.expand('Alphatron', 'control')[0], 'Alphatron')

    def test_ambiguous_alias_is_not_linked(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'kg.jsonl'
            p.write_text('\n'.join(json.dumps({'uri':u,'label':u,'aliases':['ambiguous']}) for u in ('aa','bb')))
            self.assertEqual(KnowledgeExpansion(p, []).link('ambiguous'), [])
