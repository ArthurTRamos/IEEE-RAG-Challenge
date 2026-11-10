"""Exercise artifact generation with synthetic retrieval, without model weights."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_retrieval
from src.retrieval import RetrievalIndex


class RunnerTests(unittest.TestCase):
    def run_fixture(self, with_context):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            output = root / "run"
            # The actual corpus is validated; only expensive parsing and models
            # are replaced. Scores from this fixture are never benchmark results.
            chunks = [{"chunk_id": "AI-001.pdf#0", "document_id": "AI-001",
                       "source_path": "AI-001.pdf", "text": "Synthetic evidence."}]
            embedder = SimpleNamespace(encode=lambda *a, **kw: np.array([[1., 0.]]))
            index = RetrievalIndex(chunks, np.array([[1., 0.]]), embedder,
                                   ["AI-001"], [], 0.)
            mode = ["--tokenizer", str(models)] if with_context else ["--ranking-only"]
            argv = ["run_retrieval.py", "--embed-model", str(models),
                    "--docling-models", str(models), "--output-dir", str(output), *mode]
            tokenizer = SimpleNamespace(encode=lambda text: list(text))
            loader = unittest.mock.Mock(return_value=tokenizer)
            fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=loader))
            with patch.object(sys, "argv", argv), \
                    patch.object(run_retrieval, "build_index", return_value=index), \
                    patch.dict(sys.modules, {"transformers": fake_transformers}), \
                    patch("builtins.print"):
                run_retrieval.main()
                with self.assertRaises(SystemExit):
                    run_retrieval.main()
            metrics = json.loads((output / "ranking-metrics.json").read_text())
            self.assertEqual(metrics["question_count"], 22)
            self.assertEqual(metrics["judged_questions_not_in_input"], [])
            self.assertEqual((output / "context-metrics.json").exists(), with_context)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIn("process_peak_rss_bytes", manifest)
            if with_context:
                loader.assert_called_once_with(str(models), local_files_only=True)
                context = json.loads((output / "context.json").read_text())
                self.assertEqual(len(context["questions"]), 22)
                self.assertTrue(context["questions"][0]["context"])
            else:
                loader.assert_not_called()

    def test_ranking_only_artifacts(self):
        self.run_fixture(False)

    def test_context_artifacts_without_generative_model(self):
        self.run_fixture(True)


if __name__ == "__main__":
    unittest.main()
