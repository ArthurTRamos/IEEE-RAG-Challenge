"""Frozen-run validation prevents silently comparing different evidence."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_retrieval import load_frozen_run


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.questions = [{"id": 1, "question": "alpha?"}, {"id": 2, "question": "beta?"}]
        self.judgments = {"unit": "document", "coverage": "all_local_documents",
            "corpus_documents": [{"document_id": d, "pdf_sha256": d} for d in "AB"],
            "questions": [{"id": q["id"], "judgments": [
                {"document_id": "A", "grade": 2}, {"document_id": "B", "grade": 0}]}
                for q in self.questions]}
        rows = [{"chunk_id": f"{d}.pdf#0", "source_path": f"{d}.pdf",
                 "document_id": d, "text": d, "score": score, "rank": rank}
                for rank, (d, score) in enumerate((("B", .9), ("A", .2)), 1)]
        self.payload = {"questions": [{**q, "chunks": copy.deepcopy(rows)} for q in self.questions]}
        self.write("manifest.json", {"config": {}, "chunk_count": 2})
        self.write("judgments.json", self.judgments)
        self.write("ranking.json", self.payload)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_recovers_original_catalog_and_query_scores(self):
        chunks, scores, _ = load_frozen_run(self.root, self.questions, self.judgments)
        self.assertEqual([c["document_id"] for c in chunks], ["A", "B"])
        self.assertAlmostEqual(float(scores["1"][1]), .9, places=6)

    def test_changed_corpus_or_question_is_rejected(self):
        changed = copy.deepcopy(self.judgments)
        changed["corpus_documents"][0]["pdf_sha256"] = "another-version"
        with self.assertRaisesRegex(ValueError, "corpus differs"):
            load_frozen_run(self.root, self.questions, changed)
        changed_questions = copy.deepcopy(self.questions)
        changed_questions[0]["question"] = "other question"
        with self.assertRaisesRegex(ValueError, "text differs"):
            load_frozen_run(self.root, changed_questions, self.judgments)

    def test_inconsistent_chunks_between_questions_are_rejected(self):
        self.payload["questions"][1]["chunks"][0]["text"] = "changed passage"
        self.write("ranking.json", self.payload)
        with self.assertRaisesRegex(ValueError, "content differs"):
            load_frozen_run(self.root, self.questions, self.judgments)


if __name__ == "__main__":
    unittest.main()
