"""Hand-computed chunk metrics, with no model downloads or inference."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_chunk_context import channel_top_chunks, context_metrics, document_diversity, prompt_token_count, submission_settings


class ChunkContextTests(unittest.TestCase):
    def test_ideal_uses_unretrieved_chunks_and_graded_gain(self):
        grades = {"a": 1, "b": 0, "c": 2, "d": 2}
        result = context_metrics(["a", "b"], grades)
        self.assertAlmostEqual(result["ndcg"], 1 / (3 + 3 / math.log2(3)))
        self.assertAlmostEqual(result["ge1"]["recall"], 1 / 3)
        self.assertEqual(result["ge2"]["recall"], 0)
        self.assertEqual(result["ge2"]["reciprocal_rank"], 0)

    def test_same_document_chunks_occupy_distinct_ranks(self):
        result = context_metrics(["A#0", "A#1", "B#0"], {"A#0": 0, "A#1": 0, "B#0": 2})
        self.assertAlmostEqual(result["ge2"]["reciprocal_rank"], 1 / 3)
        self.assertEqual(result["ge2"]["recall"], 1)

    def test_no_strict_positives_is_undefined(self):
        result = context_metrics(["a"], {"a": 1, "b": 0})
        self.assertIsNone(result["ge2"]["recall"])
        self.assertIsNone(result["ge2"]["reciprocal_rank"])
        self.assertEqual(result["ndcg"], 1)
        empty = context_metrics([], {"a": 2})
        self.assertEqual(empty["ndcg"], 0)
        self.assertEqual(empty["ge2"]["recall"], 0)

    def test_invalid_context_rejected(self):
        for ids in (["a", "a"], ["missing"]):
            with self.assertRaises(ValueError):
                context_metrics(ids, {"a": 2})

    def test_diversity_and_token_concentration(self):
        rows = [{"document_id": "A", "tokens": 10},
                {"document_id": "A", "tokens": 20},
                {"document_id": "B", "tokens": 70}]
        result = document_diversity(rows)
        self.assertEqual(result["unique_documents"], 2)
        self.assertAlmostEqual(result["dominant_document_chunk_share"], 2 / 3)
        self.assertAlmostEqual(result["dominant_document_token_share"], .7)
        self.assertAlmostEqual(result["effective_documents"], math.exp(-2 / 3 * math.log(2 / 3) - 1 / 3 * math.log(1 / 3)))

    def test_literal_settings_match_submission(self):
        result = submission_settings()
        self.assertEqual(result["RETRIEVAL_METHOD"], "hybrid")
        self.assertEqual(result["RRF_BM25_WEIGHT"], .75)

    def test_prompt_tokens_count_input_ids_not_encoding_fields(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = {
            "input_ids": SimpleNamespace(shape=(1, 10500)),
            "attention_mask": SimpleNamespace(shape=(1, 10500)),
        }
        messages = [{"role": "user", "content": "evidence and question"}]
        self.assertEqual(prompt_token_count(tokenizer, messages), 10500)
        tokenizer.apply_chat_template.assert_called_once_with(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

    def test_channel_ranks_use_original_order_instead_of_fused_ties(self):
        import numpy as np

        chunks = [{"chunk_id": str(i), "document_id": "A", "text": f"chunk {i}"}
                  for i in range(150)]
        # Reverse fusion order deliberately; the channel rankings must restore
        # src's native ties from canonical input, and BM25 must keep zero scores.
        ranking = [{**chunks[i], "rank": rank, "score": 1 / rank,
                    "dense_score": .5, "bm25_score": 0.0}
                   for rank, i in enumerate(reversed(range(150)), 1)]
        result = channel_top_chunks(ranking, chunks)
        dense_order = np.argsort(-np.full(150, .5, dtype=np.float32))[:100]
        self.assertEqual([c["chunk_id"] for c in result["semantic"]], [str(i) for i in dense_order])
        self.assertEqual([c["chunk_id"] for c in result["bm25"]], [str(i) for i in range(100)])
        self.assertEqual(result["bm25"][0]["rank"], 1)
        self.assertEqual(result["bm25"][0]["rrf_rank"], 150)
        self.assertEqual(result["bm25"][0]["score"], 0)
        self.assertEqual(result["bm25"][0]["text"], "chunk 0")

    def test_channels_can_retrieve_different_chunks(self):
        chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        ranking = [{"chunk_id": "a", "rank": 1, "score": .03, "dense_score": .9, "bm25_score": 1},
                   {"chunk_id": "b", "rank": 2, "score": .02, "dense_score": .1, "bm25_score": 7}]
        result = channel_top_chunks(ranking, chunks, limit=1)
        self.assertEqual(result["semantic"][0]["chunk_id"], "a")
        self.assertEqual(result["bm25"][0]["chunk_id"], "b")
        self.assertAlmostEqual(result["semantic"][0]["score"], .9, places=6)
        self.assertEqual(result["bm25"][0]["score"], 7)


if __name__ == "__main__":
    unittest.main()
