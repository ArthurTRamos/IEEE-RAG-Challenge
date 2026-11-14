import unittest
from tools.evaluate_paratext import annotate_catalog, classify
from src.retrieval import _chunk_markdown


class ParatextTests(unittest.TestCase):
    def catalog(self,text):
        return [{'chunk_id':f'A.pdf#{i}','document_id':'A','text':part} for i,part in enumerate(_chunk_markdown(text))]

    def test_cover_abstract_references_appendix_and_mixed(self):
        text='## Paper Title\n\nAuthor and affiliation\n\n## Abstract\n\nScientific content\n\n## References\n\n[1] Citation\n\n## Appendix\n\nProof'
        rows=self.catalog(text);annotation=annotate_catalog(rows)[rows[0]['chunk_id']]
        result=classify(annotation)
        self.assertTrue(result['mixed']);self.assertGreater(result['front_matter_characters'],0)
        self.assertGreater(result['bibliography_characters'],0)
        cover=classify(annotation,len('## Paper Title'))
        self.assertEqual(cover['primary_category'],'front_matter_candidate')
        self.assertEqual(cover['bibliography_characters'],0)
        self.assertEqual(sum(result[k] for k in ('front_matter_characters','bibliography_characters','body_characters','unknown_characters')),len(text))

    def test_no_body_anchor_is_unknown_not_cover_or_body(self):
        rows=self.catalog('Content without recognizable sections')
        self.assertEqual(classify(annotate_catalog(rows)['A.pdf#0'])['primary_category'],'unknown')

    def test_overlap_validation_and_references_inherited(self):
        rows=self.catalog('## Abstract\n\n'+'science '*150+'\n\n## References\n\n'+'citation '*150)
        labels=annotate_catalog(rows)
        self.assertEqual(classify(labels[rows[-1]['chunk_id']])['primary_category'],'bibliography')
        rows[1]['text']='corrupt overlap'
        with self.assertRaises(ValueError):annotate_catalog(rows)
