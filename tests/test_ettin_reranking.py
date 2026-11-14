"""Offline smoke checks for Ettin's modular cross-encoder."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.reranking import ChunkReranker

MODEL = Path(__file__).resolve().parents[1]/'models/ettin-reranker-68m-v1'


@unittest.skipUnless((MODEL/'4_Dense/model.safetensors').exists(), 'Local Ettin resources required')
class EttinIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.reranker = ChunkReranker(MODEL)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def test_modular_head_predicts_relevance(self):
        r = self.reranker
        self.assertEqual([type(m).__name__ for m in r.model.children()],
                         ['Transformer','Pooling','Dense','LayerNorm','Dense'])
        scores = r.score('Which planet is known as the Red Planet?',
                         [{'text':'Mars is known as the Red Planet.'},
                          {'text':"Venus is often called Earth's twin."}])
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[0], scores[1])

    def test_predict_preserves_complete_query_and_skips_overflow(self):
        r = self.reranker
        tokenizer = r.model.tokenizer
        query = 'theorem '*300 + 'Why is Y lush?'
        expected = tokenizer.encode(query, add_special_tokens=False)
        original = r.model.preprocess
        calls = []

        def check(*args, **kwargs):
            features = original(*args, **kwargs)
            for ids in features['input_ids']:
                self.assertEqual(ids[1:1+len(expected)].tolist(), expected)
                self.assertEqual(int(ids[1+len(expected)]), tokenizer.sep_token_id)
            calls.append(True)
            return features

        with patch.object(r.model, 'preprocess', side_effect=check):
            scores = r.score(query, [{'text':'proof '*1000}])
        self.assertTrue(calls)
        self.assertTrue(np.isfinite(scores).all())
        rows = [{'rank':1,'text':'evidence','score':.5}]
        with patch.object(r.model, 'predict', side_effect=AssertionError('Must skip CE')):
            with self.assertLogs('src.reranking', level='WARNING'):
                self.assertEqual(r.rerank('query '*600, rows), rows)

    def test_1024_window_uses_more_passage_without_cutting_query(self):
        r = ChunkReranker(MODEL, max_length=1024)
        tokenizer = r.model.tokenizer
        query = 'theorem '*300 + 'Why is Y lush?'
        expected = tokenizer.encode(query, add_special_tokens=False)
        original = r.model.preprocess
        observed = []

        def check(*args, **kwargs):
            features = original(*args, **kwargs)
            for ids, mask in zip(features['input_ids'], features['attention_mask']):
                self.assertEqual(ids[1:1+len(expected)].tolist(), expected)
                self.assertEqual(int(mask.sum()), 1024)
                observed.append(True)
            return features

        with patch.object(r.model, 'preprocess', side_effect=check):
            scores = r.score(query, [{'text':'proof '*2000}])
        self.assertTrue(observed)
        self.assertTrue(np.isfinite(scores).all())
        rows = [{'rank':1,'text':'evidence','score':.5}]
        with patch.object(r.model, 'predict', side_effect=AssertionError('Must skip CE')):
            with self.assertLogs('src.reranking', level='WARNING'):
                self.assertEqual(r.rerank('query '*1200, rows), rows)
