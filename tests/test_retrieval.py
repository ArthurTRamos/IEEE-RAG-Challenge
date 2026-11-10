"""Contract tests; synthetic rankings are not benchmark results."""

import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.retrieval import RetrievalIndex, retrieve, discover_pdfs, _chunk_markdown
from evaluate_retrieval import evaluate
from prepare_evaluation import prepare, EXPECTED_IDS


class Embedder:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0]])


class Tokenizer:
    def encode(self, text):
        return list(text)


def fixture_index():
    texts = ["first evidence", "second evidence", "third evidence"]
    chunks = [{"chunk_id": f"{doc}.pdf#{i}", "document_id": doc, "text": text}
              for i, (doc, text) in enumerate(zip(("A", "A", "B"), texts))]
    return RetrievalIndex(chunks, np.array([[1., 0.], [.9, .1], [.8, .2]]),
                          Embedder(), ["A", "B"], [], 0.)


class RetrievalTests(unittest.TestCase):
    def test_context_matches_original_starter(self):
        # Execute the unchanged original selection function as an independent oracle.
        source = ast.parse((ROOT / "IEEE-CIS-DSQA-Starter-Kit/02_rag/main.py").read_text())
        function = next(n for n in source.body
                        if isinstance(n, ast.FunctionDef) and n.name == "_build_context")
        namespace = {"np": np, "MAX_CONTEXT_TOKENS": 10000}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "starter", "exec"), namespace)
        index = fixture_index()
        for budget in (1, 14, 28, 10000):
            with self.subTest(budget=budget):
                original = namespace["_build_context"](
                    "query", [c["text"] for c in index.chunks], index.embeddings,
                    index.embedder, Tokenizer(), budget)
                result = retrieve("query", index, Tokenizer(), budget)
                self.assertEqual(result["context"], original)
        self.assertIsNone(retrieve("query", index)["selected_chunks"])

    def test_chunking_matches_starter(self):
        import re
        source = ast.parse((ROOT / "IEEE-CIS-DSQA-Starter-Kit/02_rag/main.py").read_text())
        function = next(n for n in source.body
                        if isinstance(n, ast.FunctionDef) and n.name == "_chunk_markdown")
        namespace = {"re": re, "CHUNK_SIZE": 800, "CHUNK_OVERLAP": 150}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "starter", "exec"), namespace)
        for text in ("", "long " * 300, "\n\n".join(["paragraph " * 20] * 15)):
            self.assertEqual(_chunk_markdown(text), namespace["_chunk_markdown"](text))

    def test_recursive_discovery_rejects_ambiguous_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested/A.pdf").touch()
            self.assertEqual(len(discover_pdfs(root)), 1)
            (root / "A.pdf").touch()
            with self.assertRaisesRegex(ValueError, "unique"):
                discover_pdfs(root)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.judgments = {"unit": "document", "coverage": "all_local_documents",
                          "corpus_documents": [{"document_id": "A"}, {"document_id": "B"}],
                          "questions": [{"id": "1", "judgments": [
                              {"document_id": "A", "grade": 1},
                              {"document_id": "B", "grade": 2}]}]}
        self.payload = {"questions": [{"id": "1", "chunks": fixture_index().chunks}]}
        for i, row in enumerate(self.payload["questions"][0]["chunks"], 1):
            row["rank"] = i

    def test_duplicates_consume_chunk_slots_not_document_slots(self):
        result = evaluate(self.payload, self.judgments, (1, 2, 3))
        row = result["questions"][0]
        self.assertEqual(row["at_document_k"]["2"]["recall_ge2"], 1.)
        self.assertEqual(row["at_chunk_k_document_coverage"]["2"]["recall_ge2"], 0.)
        self.assertEqual(row["at_document_k"]["2"]["mrr_ge2"], .5)
        self.assertAlmostEqual(row["at_document_k"]["2"]["ndcg"], .7967075809905066)

    def test_incomplete_evaluation_fails(self):
        q = copy.deepcopy(self.judgments["questions"][0])
        q["id"] = "2"
        self.judgments["questions"].append(q)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            evaluate(self.payload, self.judgments)

    def test_no_direct_evidence_has_null_recall(self):
        self.judgments["questions"][0]["judgments"][1]["grade"] = 0
        result = evaluate(self.payload, self.judgments)
        self.assertIsNone(result["macro_at_document_k"]["1"]["recall_ge2"]["value"])
        self.assertEqual(result["questions_without_direct_evidence"], ["1"])

    def test_real_dataset_scope_and_hash_validation(self):
        questions, judgments, manifest = prepare(ROOT / "data/pdfs")
        self.assertEqual({str(q["id"]) for q in questions}, EXPECTED_IDS)
        pdf_ids = {p.stem for p in (ROOT / "data/pdfs").rglob("*.pdf")}
        self.assertEqual({d["document_id"] for d in judgments["corpus_documents"]}, pdf_ids)
        self.assertEqual(manifest["document_count"], len(pdf_ids))

    def test_missing_corpus_sources_are_reported_and_not_in_recall_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = next((ROOT / "data/pdfs").rglob("AI-001.pdf"))
            (root / source.name).symlink_to(source)
            _, judgments, manifest = prepare(root)
            self.assertIn(1, manifest["questions_without_available_direct_evidence"])
            self.assertEqual(len(judgments["corpus_documents"]), 1)
            self.assertTrue(all(len(q["judgments"]) == 1 for q in judgments["questions"]))

    def test_changed_pdf_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "AI-001.pdf").write_bytes(b"different version")
            with self.assertRaisesRegex(ValueError, "differs from judged"):
                prepare(root)


if __name__ == "__main__":
    unittest.main()
