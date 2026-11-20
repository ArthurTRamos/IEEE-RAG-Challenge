#!/usr/bin/env python3
"""Isolate technical lexical tokenization using the saved public-22 dense scores."""
import argparse
import csv
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_evaluation import fingerprint, write_json
from run_chunk_context import (aggregate, context_metrics, document_diversity,
                               load_inputs, prompt_token_count, submission_settings)
from src.retrieval import BM25Retriever, rank_chunks, select_context, lexical_tokens
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "experiments/runs/public22-rrf-chunk-context-2026-11-19-top100")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    started = time.perf_counter()
    settings = submission_settings()
    questions, chunks, qrels, _, _ = load_inputs(
        ROOT / "experiments/runs/pdf-markdown-chunks",
        ROOT / "experiments/judments/public22-chunks", ROOT / "data/pdfs")
    source_manifest = json.loads((args.baseline / "manifest.json").read_text())
    if source_manifest["settings"] != {k: v for k, v in settings.items() if k != "SYSTEM_PROMPT"}:
        raise ValueError("Baseline retrieval settings differ")
    order = [c["chunk_id"] for c in chunks]
    if order != json.loads((args.baseline / "chunk-order.json").read_text()):
        raise ValueError("Baseline chunk order differs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/olmo-tokenizer"), local_files_only=True)
    tokens = {c["chunk_id"]: len(tokenizer.encode(c["text"])) for c in chunks}
    t = time.perf_counter()
    model = BM25Retriever(chunks, settings["BM25_K1"], settings["BM25_B"], tokenization="technical")
    index_seconds = time.perf_counter() - t
    print(f"Technical BM25 built in {index_seconds:.2f}s", flush=True)
    rows = {"legacy": [], "technical": []}
    for q in questions:
        qid, stem = str(q["id"]), f"{int(q['id']):02}"
        cached = json.loads((args.baseline / "rankings" / f"{stem}.json").read_text())["chunks"]
        byid = {c["chunk_id"]: c for c in cached}
        dense = np.array([byid[cid]["dense_score"] for cid in order], dtype=np.float32)
        old_scores = np.array([byid[cid]["bm25_score"] for cid in order])
        new_scores = model.scores(q["question"])
        grades = {cid: r["grade"] for cid, r in qrels[qid].items()}
        for variant, values in (("legacy", old_scores), ("technical", new_scores)):
            lexical = SimpleNamespace(scores=lambda _, values=values: values)
            ranking = rank_chunks(chunks, dense, q["question"], "hybrid", lexical,
                settings["RRF_K"], settings["RRF_DENSE_WEIGHT"], settings["RRF_BM25_WEIGHT"])
            selected = select_context(ranking, tokenizer, 10000, token_counts=tokens)
            chosen = [{**c, "tokens": tokens[c["chunk_id"]], "grade": grades[c["chunk_id"]]}
                      for c in selected["selected_chunks"]]
            if any(c["truncated"] for c in chosen):
                raise ValueError("Truncated chunks require new judgments")
            ids = [c["chunk_id"] for c in chosen]
            row = {"id": qid, "domain": q["domain"], "question": q["question"],
                   "context": context_metrics(ids, grades),
                   "context_top10": context_metrics(ids[:10], grades),
                   "diversity": document_diversity(chosen),
                   "chunk_tokens": sum(c["tokens"] for c in chosen),
                   "prompt_tokens": prompt_token_count(tokenizer, [
                       {"role": "system", "content": settings["SYSTEM_PROMPT"]},
                       {"role": "user", "content": selected["context"] + q["question"]}])}
            if variant == "legacy":
                original = json.loads((args.baseline / "questions" / f"{stem}.json").read_text())
                if ids != [c["chunk_id"] for c in original["selected_chunks"]] or row["context"] != original["context"]:
                    raise ValueError(f"Baseline replay mismatch: {qid}")
            rows[variant].append(row)
            folder = args.output_dir / variant
            write_json(folder / "questions" / f"{stem}.json", {**row, "selected_chunks": chosen})
            (folder / "questions" / f"{stem}.md").write_text(selected["context"], encoding="utf-8")
            write_json(folder / "rankings" / f"{stem}.json", {
                "id": qid, "chunks": [{k: c[k] for k in ("chunk_id", "document_id", "rank", "score", "dense_score", "bm25_score")}
                                      for c in ranking]})
        print(f"Q{qid}: {rows['legacy'][-1]['context']['ndcg']:.4f} -> {rows['technical'][-1]['context']['ndcg']:.4f}", flush=True)
    summaries = {name: aggregate(values) for name, values in rows.items()}
    write_json(args.output_dir / "metrics.json", {"summary": summaries, "questions": rows})
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "old_ndcg", "new_ndcg", "delta", "old_recall_ge2", "new_recall_ge2"])
        for old, new in zip(rows["legacy"], rows["technical"]):
            writer.writerow([old["id"], old["context"]["ndcg"], new["context"]["ndcg"],
                new["context"]["ndcg"] - old["context"]["ndcg"],
                old["context"]["ge2"]["recall"], new["context"]["ge2"]["recall"]])
    examples = {text: {mode: lexical_tokens(text, mode=mode) for mode in rows}
                for text in ("ZeroDiff", "ZeroDiff++", "10^-3", "10^3", "10⁻³", "1.25", "1/25", "not required")}
    write_json(args.output_dir / "manifest.json", {
        "status": "complete", "baseline_replay_verified_questions": 22,
        "dense_scores_reused": True, "legacy_bm25_scores_reused": True,
        "technical_bm25_recomputed": True, "generative_inference": False,
        "settings": {k: v for k, v in settings.items() if k != "SYSTEM_PROMPT"},
        "index_seconds": index_seconds, "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu_used": False, "token_examples": examples,
        "code": [fingerprint(Path(__file__)), fingerprint(ROOT / "src/retrieval.py")],
        "baseline": fingerprint(args.baseline / "manifest.json"),
        "judgments": fingerprint(ROOT / "experiments/judments/public22-chunks/manifest.json")})
    print(json.dumps({name: val["context"] for name, val in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
