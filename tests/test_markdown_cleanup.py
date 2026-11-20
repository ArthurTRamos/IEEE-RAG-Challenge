import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval import clean_markdown, markdown_cleanup, _chunk_markdown


class CleanupTests(unittest.TestCase):
    def test_front_matter_and_numbered_sections(self):
        for header in ("## Abstract", "## **ABSTRACT**", "## 1. Introduction", "## I. INTRODUCTION"):
            raw = f"## Paper title\n\nAuthors\n\n{header}\n\nEvidence.\n"
            self.assertEqual(clean_markdown(raw), f"{header}\n\nEvidence.\n")

    def test_inline_abstract_and_first_marker_in_reading_order(self):
        raw = "## Title\n\nAbstract—Useful summary.\n\n## I. INTRODUCTION\n\nBody."
        self.assertTrue(clean_markdown(raw).startswith("Abstract—Useful summary."))
        raw = "Title\n\n## 1 Introduction\n\nBody.\n\n## Abstract\n\nLate summary."
        self.assertIn("Body.", clean_markdown(raw))

    def test_missing_markers_leave_prefix_and_prose_mentions_untouched(self):
        raw = "## Title\n\nIntroduction to references is useful.\n\n## 2 Reference electrode system\n\nData."
        self.assertEqual(clean_markdown(raw), raw)

    def test_bibliography_only_preserves_appendix_even_wrong_level(self):
        for heading in ("## **References**", "## 7. References:", "## BIBLIOGRAPHY", "## Reference"):
            raw = f"## Introduction\n\nEvidence.\n\n{heading}\n\n[1] Citation.\n\n### Appendix\n\nProof."
            cleaned = clean_markdown(raw)
            self.assertNotIn("Citation", cleaned)
            self.assertIn("### Appendix\n\nProof.", cleaned)
            self.assertIn("Evidence.", cleaned)

    def test_terminal_bibliography_and_false_heading_matches(self):
        self.assertEqual(clean_markdown("Evidence.\n\n## References\n\n[1] Citation."), "Evidence.\n\n")
        raw = "## Resources and References\n\nTeaching content.\n\n## ACM Reference Format:\n\nCitation."
        self.assertEqual(clean_markdown(raw), raw)

    def test_code_fences_are_not_sections(self):
        raw = "Body\n\n```markdown\n## Abstract\n## References\n```\n\nKeep this."
        self.assertEqual(clean_markdown(raw), raw)

    def test_empty_fallback_and_offset_audit(self):
        raw = "## References\n\nOnly references."
        self.assertEqual(clean_markdown(raw), raw)
        self.assertEqual(markdown_cleanup(raw)["fallback"], "cleanup_would_empty_document")
        raw = "Title\n\n## Introduction\n\nEvidence\n\n## References\n\nCitation."
        result = markdown_cleanup(raw)
        for cut in reversed(result["removed"]):
            raw = raw[:cut["start"]] + raw[cut["end"]:]
        self.assertEqual(raw, result["text"])
        self.assertEqual(clean_markdown(result["text"]), result["text"])
        self.assertFalse(any("Citation" in c for c in _chunk_markdown(result["text"])))


if __name__ == "__main__":
    unittest.main()
