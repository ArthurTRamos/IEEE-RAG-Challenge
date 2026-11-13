import unittest
import numpy as np
from types import SimpleNamespace
from src.reranking import reorder_prefix, ChunkReranker
from src.retrieval import retrieve


class RerankingTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'chunk_id':str(i),'rank':i+1,'score':.1/(i+1),'text':str(i)} for i in range(5)]

    def test_stable_prefix_order_and_unchanged_tail(self):
        result=reorder_prefix(self.rows,[.2,.9,.9])
        self.assertEqual([r['chunk_id'] for r in result],['1','2','0','3','4'])
        self.assertEqual([r['rank'] for r in result],list(range(1,6)))
        self.assertEqual([r['hybrid_rank'] for r in result],[2,3,1,4,5])
        self.assertNotIn('rerank_score',result[-1])
        self.assertEqual(self.rows[0]['rank'],1)
        self.assertEqual(result[0]['score'],self.rows[1]['score'])

    def test_empty_prefix_and_bad_scores(self):
        self.assertEqual([r['chunk_id'] for r in reorder_prefix(self.rows,[])],[str(i) for i in range(5)])
        for scores in ([float('nan')],[float('inf')],np.zeros(6)):
            with self.assertRaises(ValueError):reorder_prefix(self.rows,scores)

    def test_rerank_limits_scoring_and_checks_model_output(self):
        reranker=object.__new__(ChunkReranker)
        reranker.score=lambda q, rows:np.arange(len(rows))
        result=reranker.rerank('query',self.rows,2)
        self.assertEqual([r['chunk_id'] for r in result],['1','0','2','3','4'])
        with self.assertRaises(ValueError):reranker.rerank('query',self.rows,0)
        reranker.score=lambda *args:np.zeros(1)
        with self.assertRaises(ValueError):reranker.rerank('query',self.rows,2)

    def test_retrieval_pipeline_opt_in_preserves_baseline_default(self):
        index = SimpleNamespace(chunks=self.rows, embeddings=np.array([[5.],[4.],[3.],[2.],[1.]]),
                                embedder=SimpleNamespace(encode=lambda *a, **kw: np.array([[1.]])),
                                lexical_index=None)
        baseline = retrieve('query', index)['chunks']
        reranker = object.__new__(ChunkReranker)
        reranker.score = lambda q, rows: np.arange(len(rows))
        ranked = retrieve('query', index, reranker=reranker, rerank_candidates=2)['chunks']
        self.assertEqual([r['chunk_id'] for r in baseline], ['0','1','2','3','4'])
        self.assertEqual([r['chunk_id'] for r in ranked], ['1','0','2','3','4'])
