"""Verify descriptive run names and complete sweep artifact generation."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import sweep_retrieval


class SweepTests(unittest.TestCase):
    def test_default_grid_is_unique_and_contains_requested_weights(self):
        configs = list(sweep_retrieval.configurations([.75, 1.25, 1.75], [.5, .75, 1],
                                                    [1, .75, .5], [30, 60, 90]))
        names = [sweep_retrieval.experiment_name(c) for c in configs]
        self.assertEqual(len(configs), 91)
        self.assertEqual(len(set(names)), 91)
        self.assertIn("hybrid-dw1-bw0p75-k1_1p25-b0p75-rrf60", names)
        self.assertIn("hybrid-dw1-bw0p5-k1_1p25-b0p75-rrf60", names)

    def test_small_sweep_writes_each_run_with_config_and_shared_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "ranking.json").write_text("{}")
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            output = root / "sweep"
            questions = [{"id": 1, "question": "alpha?", "domain": "DONES"}]
            judgments = {"unit": "document", "coverage": "all_local_documents",
                         "corpus_documents": [{"document_id": "A"}, {"document_id": "B"}],
                         "questions": [{"id": 1, "judgments": [
                             {"document_id": "A", "grade": 2}, {"document_id": "B", "grade": 0}]}]}
            chunks = [{"chunk_id": f"{doc}.pdf#0", "document_id": doc,
                       "source_path": f"{doc}.pdf", "text": doc} for doc in "AB"]
            tokenizer = SimpleNamespace(encode=lambda text: list(text))
            transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: tokenizer))
            argv = ["sweep_retrieval.py", "--source-run", str(source),
                    "--tokenizer", str(tokenizer_dir), "--output-dir", str(output),
                    "--k1", "1.25", "--b", ".75", "--bm25-weight", ".75", ".5",
                    "--rrf-k", "60"]
            with patch.object(sys, "argv", argv), \
                    patch.object(sweep_retrieval, "prepare", return_value=(questions, judgments, {"sources": {}})), \
                    patch.object(sweep_retrieval, "load_frozen_run", return_value=(chunks, {"1": np.array([.9, .1])}, {})), \
                    patch.dict(sys.modules, {"transformers": transformers}), patch("builtins.print"):
                sweep_retrieval.main()
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary), 4)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "complete")
            for row in summary:
                run = output / row["name"]
                manifest = json.loads((run / "manifest.json").read_text())
                self.assertEqual(manifest["config"]["method"], row["method"])
                metrics = json.loads((run / "context-metrics.json").read_text())
                self.assertEqual(metrics["question_count"], 1)
                ranking = json.loads((run / "ranking.json").read_text())
                self.assertEqual(len(ranking["questions"][0]["chunk_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
