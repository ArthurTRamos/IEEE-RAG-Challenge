import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from types import SimpleNamespace
from src.reranking import reorder_prefix, ChunkReranker, QueryTooLongError
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


@unittest.skipUnless((Path(__file__).resolve().parents[1]/'models/ms-marco-MiniLM-L6-v2').is_dir(),
                     'Local CE tokenizer not available')
class QueryPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(Path(__file__).resolve().parents[1]/'models/ms-marco-MiniLM-L6-v2'),
            local_files_only=True)

    def setUp(self):
        self.reranker = object.__new__(ChunkReranker)
        self.reranker.max_length, self.reranker.batch_size = 512, 16
        self.reranker.model = SimpleNamespace(tokenizer=self.tokenizer, predict=Mock())

    def test_long_pair_preserves_query_ids_and_truncates_only_passage(self):
        query = ' '.join(['theorem']*300) + ' Why is Y lush?'
        passage = ' '.join(['proof']*800)
        rows = [{'text':passage}]

        def predict(pairs, **kwargs):
            encoded = self.tokenizer([q for q,c in pairs], [c for q,c in pairs],
                                     **kwargs['processing_kwargs']['text'])
            ids = encoded['input_ids'][0]
            query_ids = [v for v,s in zip(ids, encoded.sequence_ids(0)) if s == 0]
            self.assertEqual(query_ids, self.tokenizer.encode(query, add_special_tokens=False))
            self.assertEqual(len(ids), 512)
            self.assertLess(encoded.sequence_ids(0).count(1), 800)
            return [1.5]

        self.reranker.model.predict.side_effect = predict
        np.testing.assert_equal(self.reranker.score(query, rows), [1.5])
        self.assertEqual(rows[0]['text'], passage)

    def test_no_passage_space_falls_back_without_scoring_or_mutating_ranking(self):
        for count in (509, 510, 600):  # BERT pair markers consume three positions.
            query = ' '.join(['query']*count)
            rows = [{'rank':1,'text':'evidence','score':.25}]
            with self.assertRaises(QueryTooLongError):
                self.reranker.score(query, rows)
            with self.assertLogs('src.reranking', level='WARNING'):
                result = self.reranker.rerank(query, rows)
            self.assertEqual(result, rows)
            self.assertIsNot(result[0], rows[0])
        self.reranker.model.predict.assert_not_called()

    def test_one_passage_token_available_and_empty_candidates(self):
        query = ' '.join(['query']*508)
        self.assertEqual(self.reranker.query_diagnostics(query)['passage_token_budget'], 1)
        self.reranker.model.predict.return_value = [2.]
        self.assertEqual(self.reranker.score(query, [{'text':'proof'}]).tolist(), [2.])
        self.assertEqual(self.reranker.score('query', []).size, 0)
