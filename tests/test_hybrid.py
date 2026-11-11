"""BM25 math, RRF votes and shared context-budget contracts."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import BM25Retriever, lexical_tokens, rank_chunks, select_context


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"chunk_id": f"{doc}.pdf#0", "document_id": doc, "text": text}
                       for doc, text in zip("ABC", ("alpha alpha beta", "gamma", "delta"))]

    def test_bm25_matches_manual_okapi_formula(self):
        bm25 = BM25Retriever(self.chunks, k1=1.25, b=.75)
        idf = math.log((3 - 1 + .5) / (1 + .5))
        expected = idf * (2 * (1.25 + 1)) / (2 + 1.25 * (1 - .75 + .75 * 3 / (5 / 3)))
        scores = bm25.scores("alpha")
        np.testing.assert_allclose(scores, [expected, 0, 0])
        ranking = rank_chunks(self.chunks, None, "ALPHA", "bm25", bm25)
        self.assertEqual(ranking[0]["document_id"], "A")
        self.assertNotIn("dense_score", ranking[0])

    def test_b_one_handles_chunks_without_lexical_tokens(self):
        chunks = [{"text": text} for text in ("---", "alpha", "beta")]
        bm25 = BM25Retriever(chunks, k1=1.25, b=1.0)
        scores = bm25.scores("alpha")
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores[0], 0)
        self.assertGreater(scores[1], 0)
        np.testing.assert_array_equal(bm25.scores("unmatched"), np.zeros(3))

    def test_rrf_formula_uses_one_based_ranks_and_only_positive_lexical_votes(self):
        class Lexical:
            def scores(self, question):
                return np.array([0., 5., 1.])
        rows = rank_chunks(self.chunks, np.array([.9, .8, .7]), "query",
                           "hybrid", Lexical(), rrf_k=60)
        self.assertEqual([r["document_id"] for r in rows], ["B", "C", "A"])
        self.assertAlmostEqual(rows[0]["score"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(rows[2]["score"], 1 / 61)

    def test_no_lexical_matches_preserves_dense_order(self):
        bm25 = BM25Retriever(self.chunks)
        rows = rank_chunks(self.chunks, np.array([.7, .9, .8]), "unknown",
                           "hybrid", bm25)
        self.assertEqual([r["document_id"] for r in rows], ["B", "C", "A"])

    def test_weighted_rrf_formula_and_zero_lexical_weight(self):
        class Lexical:
            def scores(self, question):
                return np.array([0., 5., 1.])
        for weight in (.75, .5):
            rows = rank_chunks(self.chunks, np.array([.9, .8, .7]), "query",
                               "hybrid", Lexical(), rrf_k=30, bm25_weight=weight)
            scores = {r["document_id"]: r["score"] for r in rows}
            self.assertAlmostEqual(scores["B"], 1 / 32 + weight / 31)
            self.assertAlmostEqual(scores["C"], 1 / 33 + weight / 32)
            self.assertAlmostEqual(scores["A"], 1 / 31)
        rows = rank_chunks(self.chunks, np.array([.9, .8, .7]), "query",
                           "hybrid", Lexical(), bm25_weight=0)
        self.assertEqual([r["document_id"] for r in rows], ["A", "B", "C"])

    def test_cached_token_counts_preserve_context_and_truncation(self):
        class Tokenizer:
            def encode(self, text):
                return list(text)
        rows = [{**c, "rank": i + 1} for i, c in enumerate(self.chunks)]
        cache = {c["chunk_id"]: len(c["text"]) for c in rows}
        for budget in (1, 15, 25, 10000):
            self.assertEqual(select_context(rows, Tokenizer(), budget),
                             select_context(rows, Tokenizer(), budget, token_counts=cache))

    def test_unicode_tokenization_is_shared_by_queries_and_documents(self):
        self.assertEqual(lexical_tokens("IFMIF-DONES 1.25 ÉNERGIE"),
                         ["ifmif", "dones", "1", "25", "énergie"])

    def test_context_is_not_cut_at_ten_chunks(self):
        class Tokenizer:
            def encode(self, text):
                return list(text)
        chunks = [{"text": "a", "chunk_id": f"A.pdf#{i}", "rank": i + 1}
                  for i in range(100)]
        result = select_context(chunks, Tokenizer(), max_tokens=25)
        self.assertEqual(len(result["selected_chunks"]), 25)

    def test_invalid_bm25_and_rrf_parameters_fail(self):
        for k1, b in ((0, .75), (1.25, -1), (1.25, 1.1), (float("nan"), .75)):
            with self.subTest(k1=k1, b=b), self.assertRaises(ValueError):
                BM25Retriever(self.chunks, k1, b)
        with self.assertRaises(ValueError):
            rank_chunks(self.chunks, np.ones(3), "alpha", rrf_k=0)
        for dense_weight, bm25_weight in ((-1, .5), (1, float("nan")), (0, 0)):
            with self.assertRaises(ValueError):
                rank_chunks(self.chunks, np.ones(3), "alpha", dense_weight=dense_weight,
                            bm25_weight=bm25_weight)


if __name__ == "__main__":
    unittest.main()
