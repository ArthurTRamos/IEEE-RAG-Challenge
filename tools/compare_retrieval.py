#!/usr/bin/env python3
"""Compare dense/BM25/RRF on a validated, frozen baseline ranking and tokenizer.

Reuses PDF extraction and dense similarities, not model inference. Timings are
incremental replay costs and must not be presented as full pipeline runtimes.
"""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from prepare_evaluation import ROOT, prepare, write_json, fingerprint
from evaluate_retrieval import evaluate, load_judgments

sys.path.insert(0, str(ROOT))
from src.retrieval import BM25Retriever, rank_chunks, select_context, MAX_CONTEXT_TOKENS


def load_frozen_run(directory, questions, judgments):
    manifest = json.loads((directory / "manifest.json").read_text())
    old_judgments = json.loads((directory / "judgments.json").read_text())
    if load_judgments(old_judgments) != load_judgments(judgments):
        raise ValueError("Frozen judgments differ from current public evaluation")
    old_hashes = {d["document_id"]: d["pdf_sha256"] for d in old_judgments["corpus_documents"]}
    hashes = {d["document_id"]: d["pdf_sha256"] for d in judgments["corpus_documents"]}
    if hashes != old_hashes:
        raise ValueError("Frozen corpus differs from current PDFs")
    if manifest["config"].get("method", "dense") != "dense":
        raise ValueError("Replay source must be a dense baseline run")
    payload = json.loads((directory / "ranking.json").read_text())
    records = payload["questions"]
    expected = {str(q["id"]): q["question"] for q in questions}
    if len(records) != len(expected) or {str(q["id"]) for q in records} != set(expected):
        raise ValueError("Frozen run must contain exactly the active question IDs")
    chunks = [{key: row[key] for key in ("chunk_id", "document_id", "source_path", "text")}
              for row in records[0]["chunks"]]
    chunks.sort(key=lambda c: (c["source_path"], int(c["chunk_id"].rsplit("#", 1)[1])))
    catalog = {c["chunk_id"]: c for c in chunks}
    if len(catalog) != len(chunks) or len(chunks) != manifest["chunk_count"]:
        raise ValueError("Frozen chunk catalog is incomplete or duplicated")
    scores = {}
    for question in records:
        qid = str(question["id"])
        if question["question"] != expected[qid]:
            raise ValueError(f"Frozen question text differs: {qid}")
        rows = question["chunks"]
        if len(rows) != len(catalog) or {r["chunk_id"] for r in rows} != set(catalog):
            raise ValueError(f"Inconsistent frozen chunk IDs: {qid}")
        for row in rows:
            if any(row[key] != catalog[row["chunk_id"]][key] for key in catalog[row["chunk_id"]]):
                raise ValueError(f"Frozen chunk content differs: {qid}/{row['chunk_id']}")
        by_id = {r["chunk_id"]: r["score"] for r in rows}
        values = np.asarray([by_id[c["chunk_id"]] for c in chunks], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite dense scores")
        replay = rank_chunks(chunks, values, question["question"])
        if [r["chunk_id"] for r in replay] != [r["chunk_id"] for r in rows]:
            raise ValueError(f"Dense replay cannot reproduce source order: {qid}")
        scores[qid] = values
    return chunks, scores, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/pdfs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k1", type=float, default=1.25)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--rrf-k", type=float, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    started = time.perf_counter()
    questions, judgments, manifest = prepare(args.corpus)
    chunks, similarities, source_manifest = load_frozen_run(args.source_run, questions, judgments)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    tick = time.perf_counter()
    lexical = BM25Retriever(chunks, args.k1, args.b)
    bm25_build_seconds = time.perf_counter() - tick
    summary = {}
    for method in ("dense", "bm25", "hybrid"):
        tick = time.perf_counter()
        ranking = {"variant": method, "questions": []}
        context = {"variant": f"{method}_selected_context", "questions": []}
        for question in questions:
            query_started = time.perf_counter()
            qid = str(question["id"])
            rows = rank_chunks(chunks, similarities[qid] if method != "bm25" else None,
                               question["question"], method, lexical, args.rrf_k,
                               args.dense_weight, args.bm25_weight)
            result = select_context(rows, tokenizer)
            base = {"id": question["id"], "domain": question["domain"],
                    "question": question["question"]}
            ranking["questions"].append({**base, "chunks": rows})
            selected = result["selected_chunks"]
            context["questions"].append({**base, "chunks": selected, "context": result["context"],
                "chunk_text_tokens": sum(len(tokenizer.encode(c["text"])) for c in selected),
                "context_tokens_with_headers": len(tokenizer.encode(result["context"])),
                "replay_seconds": time.perf_counter() - query_started})
        folder = args.output_dir / method
        write_json(folder / "judgments.json", judgments)
        summaries = {}
        for name, payload in (("ranking", ranking), ("context", context)):
            write_json(folder / f"{name}.json", payload)
            metrics = evaluate(payload, judgments)
            metrics["sources"] = {"input": fingerprint(folder / f"{name}.json"),
                                  "judgments": fingerprint(folder / "judgments.json")}
            # Recall over every selected context chunk, not only the first ten.
            if name == "context":
                all_k = max(len(q["chunks"]) for q in payload["questions"])
                full = evaluate(payload, judgments, (max(1, all_k),))
                metrics["macro_full_context_document_coverage"] = full[
                    "macro_at_chunk_k_document_coverage"][str(max(1, all_k))]
                summaries["full_context_document_coverage"] = metrics["macro_full_context_document_coverage"]
            write_json(folder / f"{name}-metrics.json", metrics)
            summaries[name] = metrics["macro_at_document_k"]
        summaries["mean_context_chunks"] = sum(len(q["chunks"]) for q in context["questions"]) / len(questions)
        summaries["mean_chunk_text_tokens"] = sum(q["chunk_text_tokens"] for q in context["questions"]) / len(questions)
        summaries["incremental_replay_seconds_with_serialization"] = time.perf_counter() - tick
        summary[method] = summaries
        print(f"[comparison] {method}: {summaries['mean_context_chunks']:.1f} chunks/context", flush=True)
    manifest.update({"mode": "frozen_dense_replay", "chunk_count": len(chunks),
                     "source_ranking": fingerprint(args.source_run / "ranking.json"),
                     "source_manifest": source_manifest,
                     "bm25_build_seconds": bm25_build_seconds,
                     "config": {"k1": args.k1, "b": args.b, "rrf_k": args.rrf_k,
                                "rrf_weights": [args.dense_weight, args.bm25_weight],
                                "lexical_votes": "positive BM25 scores only",
                                "lexical_tokenization": "Unicode word regex, casefold; no stemming/stopwords",
                                "context_chunk_text_budget": MAX_CONTEXT_TOKENS,
                                "tokenizer": str(args.tokenizer.resolve())},
                     "tokenizer_files": {p.name: fingerprint(p) for p in args.tokenizer.iterdir() if p.is_file()},
                     "elapsed_seconds": time.perf_counter() - started,
                     "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
