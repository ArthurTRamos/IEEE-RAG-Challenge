import copy
from types import SimpleNamespace
import unittest

import numpy as np

from src.reranking import ChunkReranker, QueryTooLongError
from src.retrieval import retrieve


class Tokenizer:
    def encode(self, text):
        return list(text)


class SelectedContextTests(unittest.TestCase):
    def setUp(self):
        self.index = SimpleNamespace(
            chunks=[{'chunk_id':str(i),'document_id':str(i),'text':t} for i,t in enumerate(['a','b','c'])],
            embeddings=np.array([[3.],[2.],[1.]]),
            embedder=SimpleNamespace(encode=lambda *a, **kw:np.array([[1.]])), lexical_index=None)
        self.reranker = object.__new__(ChunkReranker)
        self.reranker.score = lambda q,rows:np.arange(len(rows))

    def test_reorders_selected_set_without_repacking(self):
        before = copy.deepcopy(self.index.chunks)
        actual = retrieve('q',self.index,Tokenizer(),max_tokens=2,reranker=self.reranker,
                          rerank_mode='selected_context')
        self.assertEqual([c['text'] for c in actual['selected_chunks']],['b','a'])
        self.assertNotIn('c',actual['context'].split('[Fragment')[-1])
        self.assertEqual(self.index.chunks,before)
        legacy = retrieve('q',self.index,Tokenizer(),max_tokens=2,reranker=self.reranker)
        self.assertEqual([c['text'] for c in legacy['selected_chunks']],['c','b'])

    def test_preserves_first_oversized_chunk_text(self):
        self.index.chunks[0]['text'] = 'abcdefgh'
        result = retrieve('q',self.index,Tokenizer(),max_tokens=4,reranker=self.reranker,
                          rerank_mode='selected_context')
        self.assertEqual(len(result['selected_chunks']),1)
        self.assertEqual(result['selected_chunks'][0]['text'],'abcd')
        self.assertTrue(result['selected_chunks'][0]['truncated'])

    def test_fallback_is_identical_to_baseline(self):
        baseline = retrieve('q',self.index,Tokenizer(),max_tokens=2)
        def fail(*args):
            raise QueryTooLongError('test overflow')
        self.reranker.score = fail
        with self.assertLogs('src.reranking',level='WARNING'):
            result = retrieve('q',self.index,Tokenizer(),max_tokens=2,reranker=self.reranker,
                              rerank_mode='selected_context')
        self.assertEqual(result,baseline)

    def test_requires_tokenizer_and_valid_mode(self):
        with self.assertRaises(ValueError):
            retrieve('q',self.index,reranker=self.reranker,rerank_mode='selected_context')
        with self.assertRaises(ValueError):
            retrieve('q',self.index,rerank_mode='invalid')
