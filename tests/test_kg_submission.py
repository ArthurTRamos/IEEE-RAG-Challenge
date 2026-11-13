import unittest
from types import SimpleNamespace
import numpy as np
from src.retrieval import RetrievalIndex, retrieve, rank_chunks, BM25Retriever


class SubmissionRetrievalTests(unittest.TestCase):
    def test_expansion_changes_only_lexical_channel(self):
        chunks = [{'chunk_id': f'{i}.pdf#0', 'text': t, 'document_id': str(i)}
                  for i, t in enumerate(['alpha alpha', 'beta beta', 'gamma delta'])]
        calls = []
        def encode(text, **kwargs):
            calls.append(text)
            return np.array([[1.]], dtype=np.float32)
        index = RetrievalIndex(chunks, np.array([[.8], [.7], [.6]], dtype=np.float32),
                               SimpleNamespace(encode=encode), [], [], 0)
        result = retrieve('alpha', index, method='hybrid', bm25_weight=.75, lexical_question='beta')
        expected = rank_chunks(chunks, index.embeddings[:, 0], 'beta', 'hybrid',
                               BM25Retriever(chunks), 60, 1, .75)
        self.assertEqual(result['chunks'], expected)
        self.assertEqual(calls, [['alpha']])
        original = retrieve('alpha', index, method='hybrid', bm25_weight=.75)
        self.assertNotEqual(original['chunks'][0]['chunk_id'], result['chunks'][0]['chunk_id'])


if __name__ == '__main__': unittest.main()
