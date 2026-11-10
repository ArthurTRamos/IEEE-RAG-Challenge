#!/usr/bin/env python3
"""Derive the public-22 evaluation set from historical judgments and local PDFs."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_IDS = {"16", "18", "21"}
EXPECTED_IDS = {str(i) for i in range(1, 26)} - EXCLUDED_IDS
DEFAULT_QUESTIONS = ROOT / "IEEE-CIS-DSQA-Starter-Kit/sample/sample_qa.json"
DEFAULT_JUDGMENTS = ROOT / "experiments/judments/public-22-questions.json"


def fingerprint(path):
    path = Path(path)
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def prepare(corpus_dir, questions_path=DEFAULT_QUESTIONS,
            judgments_path=DEFAULT_JUDGMENTS):
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    questions = [q for q in questions if str(q["id"]) not in EXCLUDED_IDS]
    ids = [str(q["id"]) for q in questions]
    if len(ids) != 22 or set(ids) != EXPECTED_IDS:
        raise ValueError("Public evaluation must contain exactly the 22 active IDs")
    original = json.loads(Path(judgments_path).read_text(encoding="utf-8"))
    from evaluate_retrieval import load_judgments
    load_judgments(original)
    catalog = {d["document_id"]: d for d in original["corpus_documents"]}
    pdfs = sorted(Path(corpus_dir).rglob("*.pdf"))
    if not pdfs or len({p.stem for p in pdfs}) != len(pdfs):
        raise ValueError("Corpus must contain PDFs with unique document IDs")
    documents = []
    for pdf in pdfs:
        if pdf.stem not in catalog:
            raise ValueError(f"PDF has no historical judgments: {pdf}")
        source = fingerprint(pdf)
        if source["sha256"] != catalog[pdf.stem]["pdf_sha256"]:
            raise ValueError(f"PDF differs from judged version: {pdf}")
        documents.append({"document_id": pdf.stem,
                          "relative_path": pdf.relative_to(corpus_dir).as_posix(),
                          "pdf_sha256": source["sha256"],
                          "title": catalog[pdf.stem]["title"]})
    available = {d["document_id"] for d in documents}
    old_questions = {str(q["id"]): q for q in original["questions"]}
    derived_questions, coverage = [], []
    for question in questions:
        qid = str(question["id"])
        old = old_questions[qid]
        if question["question"] != old["question"]:
            raise ValueError(f"Question text changed: {qid}")
        derived_questions.append({"id": question["id"],
                                  "judgments": [{"document_id": r["document_id"],
                                                 "grade": r["grade"]}
                                                for r in old["judgments"]
                                                if r["document_id"] in available]})
        direct = {r["document_id"] for r in old["judgments"] if r["grade"] == 2}
        coverage.append({"id": question["id"], "domain": question["domain"],
                         "historical_direct_documents": sorted(direct),
                         "available_direct_documents": sorted(direct & available),
                         "missing_direct_documents": sorted(direct - available),
                         "historical_coverage_status": old["coverage_status"]})
    manifest = {"excluded_question_ids": sorted(EXCLUDED_IDS),
                "question_count": len(questions), "document_count": len(documents),
                "missing_document_ids": sorted(set(catalog) - available),
                "questions_without_available_direct_evidence": [
                    q["id"] for q in coverage if not q["available_direct_documents"]],
                "coverage": coverage,
                "sources": {"questions": fingerprint(questions_path),
                            "historical_judgments": fingerprint(judgments_path)}}
    judgments = {"schema_version": 1, "unit": "document",
                 "status": original["status"], "coverage": "all_local_documents",
                 "limitations": original["limitations"],
                 "derivation": "Active public questions and available hash-verified PDFs only; original grades unchanged.",
                 "corpus_documents": documents, "questions": derived_questions}
    load_judgments(judgments)
    return questions, judgments, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/pdfs")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/evaluation/public-22")
    args = parser.parse_args()
    questions, judgments, manifest = prepare(args.corpus, args.questions, args.judgments)
    for name, payload in (("questions", questions), ("judgments", judgments),
                          ("manifest", manifest)):
        write_json(args.output_dir / f"{name}.json", payload)
    print(f"Prepared {len(questions)} questions / {manifest['document_count']} PDFs")


if __name__ == "__main__":
    main()
