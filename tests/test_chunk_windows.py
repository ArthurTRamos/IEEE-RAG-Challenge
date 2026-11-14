import unittest
from tools.evaluate_chunk_windows import chunk_window_scores


class ChunkWindowTests(unittest.TestCase):
    def test_cutoff_counts_chunks_and_duplicate_documents_consume_slots(self):
        rows=[{'chunk_id':str(i),'document_id':doc} for i,doc in enumerate(['A','A','C','B'])]
        scores=chunk_window_scores(rows,{'A':2,'B':0,'C':1},['A','A','B','C'],2)
        self.assertEqual(scores['precision_grade_proxy_ge2'],1)
        self.assertEqual(scores['document_recall_ge1'],.5)
        self.assertEqual(scores['unique_documents'],1)
        self.assertEqual(scores['ndcg_grade_proxy'],1)

    def test_short_and_empty_rankings_and_invalid_input(self):
        rows=[{'chunk_id':'1','document_id':'A'}]
        result=chunk_window_scores(rows,{'A':2},['A'],2)
        self.assertEqual(result['precision_grade_proxy_ge2'],.5)
        self.assertEqual(chunk_window_scores([],{'A':0},['A'],2)['ndcg_grade_proxy'],None)
        with self.assertRaises(ValueError):chunk_window_scores(rows,{'B':2},['B'],2)
        with self.assertRaises(ValueError):chunk_window_scores(rows*2,{'A':2},['A'],2)
        with self.assertRaises(ValueError):chunk_window_scores(rows,{'A':2},['A'],0)

    def test_same_set_changes_ndcg_but_not_counts_or_coverage(self):
        rows=[{'chunk_id':'a','document_id':'A'}, {'chunk_id':'b','document_id':'B'}]
        early=chunk_window_scores(rows,{'A':2,'B':0},['A','B'],2)
        late=chunk_window_scores(rows[::-1],{'A':2,'B':0},['A','B'],2)
        self.assertEqual(early['grade2_chunks'],late['grade2_chunks'])
        self.assertEqual(early['document_recall_ge2'],late['document_recall_ge2'])
        self.assertGreater(early['ndcg_grade_proxy'],late['ndcg_grade_proxy'])
