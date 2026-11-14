"""Offline integration checks with the actual MiniLM model and tokenizers."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.reranking import ChunkReranker
from src.retrieval import retrieve

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'models/ms-marco-MiniLM-L6-v2'


@unittest.skipUnless((MODEL/'model.safetensors').is_file(), 'Local CE weights required')
class RerankingIntegrationTests(unittest.TestCase):
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

    def checked_scores(self, question, passages):
        """Inspect the tensors actually passed through predict(), not a replica."""
        model = self.reranker.model
        original = model.preprocess
        expected = model.tokenizer.encode(question, add_special_tokens=False)
        observed = []

        def check(*args, **kwargs):
            features = original(*args, **kwargs)
            for ids, types, mask in zip(features['input_ids'], features['token_type_ids'],
                                        features['attention_mask']):
                active = mask.bool()
                query_ids = ids[active & (types == 0)][1:-1].tolist()
                self.assertEqual(query_ids, expected)
                self.assertLessEqual(int(mask.sum()), 512)
                observed.append(len(query_ids))
            return features

        with patch.object(model, 'preprocess', side_effect=check):
            scores = self.reranker.score(question, [{'text':p} for p in passages])
        self.assertEqual(len(observed), len(passages))
        self.assertTrue(np.isfinite(scores).all())
        return scores

    def test_real_predict_preserves_long_question_across_batches(self):
        question = ' '.join(['theorem']*300) + ' Why is Y lush?'
        passages = ['proof '*800, 'short proof', '']*6
        self.checked_scores(question, passages)

    def test_one_remaining_passage_token_works_in_real_forward(self):
        self.checked_scores(' '.join(['query']*508), ['proof '*800, 'proof', ''])

    def test_fallback_preserves_hybrid_ranking_selected_text_and_context(self):
        tokenizer = self.reranker.model.tokenizer
        chunks = [{'chunk_id':str(i), 'document_id':f'doc{i}', 'text':text,
                   'section':'body'} for i,text in enumerate(
                       ['query proof evidence', 'theorem consequence', 'query example'])]
        index = SimpleNamespace(chunks=chunks, embeddings=np.array([[.1],[.9],[.4]]),
                                embedder=SimpleNamespace(encode=lambda *a, **kw:np.array([[1.]])),
                                lexical_index=None)
        before = copy.deepcopy(chunks)
        for n in (509, 510, 600):
            question = ' '.join(['query']*n)
            baseline = retrieve(question, index, tokenizer, method='hybrid',
                                bm25_weight=.75, max_tokens=12)
            with patch.object(self.reranker.model, 'predict', side_effect=AssertionError('CE must be skipped')):
                with self.assertLogs('src.reranking', level='WARNING'):
                    result = retrieve(question, index, tokenizer, method='hybrid',
                                      bm25_weight=.75, max_tokens=12, reranker=self.reranker)
            self.assertEqual(result, baseline)
        self.assertEqual(chunks, before)
        self.checked_scores('What is the theorem?', ['a theorem', 'proof'])

    def test_unrelated_failure_is_not_silently_converted_to_fallback(self):
        with patch.object(self.reranker.model, 'predict', side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                self.reranker.rerank('query', [{'rank':1, 'text':'passage'}])

    @unittest.skipUnless((ROOT/'experiments/runs/public22-ce-minilm-l6-query-preserved-2026-11-14/scores.json').exists(),
                         'Frozen public run required')
    def test_question13_real_forward_matches_recorded_corrected_scores(self):
        source = ROOT/'experiments/runs/public22-ce-minilm-l6-query-preserved-2026-11-14'
        scores = next(q for q in json.loads((source/'scores.json').read_text()) if q['id']==13)
        ranking = json.loads((ROOT/'experiments/runs/baseline-public22-ranking/ranking.json').read_text())
        question = next(q for q in ranking['questions'] if q['id']==13)
        catalog = {c['chunk_id']:c['text'] for c in question['chunks']}
        actual = self.checked_scores(question['question'], [catalog[c['chunk_id']] for c in scores['chunks']])
        np.testing.assert_allclose(actual, [c['rerank_score'] for c in scores['chunks']], atol=1e-4, rtol=0)
