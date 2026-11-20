#!/usr/bin/env python3
"""Compare dense and RRF rankings against matching chunk and document qrels."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

from compare_embeddings import (BM25_B, BM25_K1, EMBED_MODEL_PATH, RRF_BM25_WEIGHT,
                                RRF_DENSE_WEIGHT, RRF_K, BM25Retriever, encode_chunks, rank_chunks)
from evaluate_retrieval import load_judgments, scores
from judge_chunks import load_inputs
from prepare_evaluation import ROOT, fingerprint, prepare, write_json


CUTOFFS = (20, 40)
METRICS = ("ndcg", "recall_ge1", "recall_ge2", "hit_ge1", "hit_ge2")
COVERAGE = ("distinct_docs", "docs_grade_1", "docs_grade_2", "docs_with_evidence",
            "docs_with_direct_evidence")
METHODS = ("dense", "hybrid")
ENCODERS = ("minilm-l6-v2", "arctic-embed-xs")
LIMITS = (256, 512)


def load_chunk_qrels(folder, catalog, chunk_digest, doc_qrels):
    validation = json.loads((folder / "validation.json").read_text(encoding="utf-8"))
    path = folder / "qrels.txt"
    if (validation["chunk_set_sha256"] != chunk_digest
            or validation["qrels_sha256"] != fingerprint(path)["sha256"]):
        raise ValueError("Chunk qrels do not match the exported Markdown and validation hashes")
    qrels = {qid: {} for qid in doc_qrels}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            qid, _, cid, value = line.split()
            grade = int(value)
            if cid not in catalog or grade not in (0, 1, 2) or cid in qrels[qid]:
                raise ValueError(f"Invalid chunk qrel: {line.strip()}")
            if grade and not doc_qrels[qid][catalog[cid][0]]:
                raise ValueError(f"Positive chunk in grade-zero document: {line.strip()}")
            qrels[qid][cid] = grade
    if any(grades.keys() != catalog.keys() for grades in qrels.values()):
        raise ValueError("Chunk qrels must cover the whole corpus for each question")
    return qrels


def cutoff_metrics(ranking, qrels, cutoff):
    row = scores(ranking, qrels, cutoff)
    for threshold in (1, 2):
        recall = row[f"recall_ge{threshold}"]
        row[f"hit_ge{threshold}"] = int(recall > 0) if recall is not None else None
    return {metric: row[metric] for metric in METRICS}


def evaluate_ranking(ranking, catalog, chunks, docs):
    document_ranking = list(dict.fromkeys(catalog[cid][0] for cid in ranking))
    top40 = ranking[:40]
    present = list(dict.fromkeys(catalog[cid][0] for cid in top40))
    evidence = {catalog[cid][0] for cid in top40 if chunks[cid] >= 1}
    direct = {catalog[cid][0] for cid in top40 if chunks[cid] == 2}
    coverage = {
        "distinct_docs": len(present),
        "docs_grade_1": sum(docs[doc] == 1 for doc in present),
        "docs_grade_2": sum(docs[doc] == 2 for doc in present),
        "docs_with_evidence": len(evidence),
        "docs_with_direct_evidence": len(direct),
        "documents": [{"document_id": doc, "document_grade": docs[doc],
                       "chunk_evidence": doc in evidence, "direct_chunk_evidence": doc in direct}
                      for doc in present],
    }
    metrics = {unit: {str(k): cutoff_metrics(order, qrels, k) for k in CUTOFFS}
               for unit, order, qrels in (("chunk", ranking, chunks), ("document", document_ranking, docs))}
    return {**metrics, "coverage_in_top40_chunks": coverage}


def average(values):
    defined = [value for value in values if value is not None]
    return mean(defined) if defined else None


def summarize(encoder, limit, method, records):
    rows = []
    domains = sorted({row["domain"] for row in records})
    for domain in ("ALL", *domains):
        selected = [r for r in records if domain == "ALL" or r["domain"] == domain]
        row = {"encoder": encoder, "limit": limit, "method": method, "domain": domain,
               "questions": len(selected)}
        for unit in ("chunk", "document"):
            for k in CUTOFFS:
                for metric in METRICS:
                    row[f"{unit}_{metric}@{k}"] = average([r[unit][str(k)][metric] for r in selected])
        for metric in COVERAGE:
            row[f"top40_{metric}"] = average([r["coverage_in_top40_chunks"][metric] for r in selected])
        rows.append(row)
    return rows


def document_changes(baseline, candidate):
    changes = []
    for before, after in zip(baseline, candidate, strict=True):
        if before["id"] != after["id"]:
            raise ValueError("Question order changed between rankings")
        sets = []
        for row in (before, after):
            top40 = row["coverage_in_top40_chunks"]["documents"]
            sets.append({
                "grade_1": {r["document_id"] for r in top40 if r["document_grade"] == 1},
                "grade_2": {r["document_id"] for r in top40 if r["document_grade"] == 2},
                "with_chunk_evidence": {r["document_id"] for r in top40 if r["chunk_evidence"]},
            })
        entries = {}
        for label, ids in sets[0].items():
            gained, lost = sets[1][label] - ids, ids - sets[1][label]
            if gained or lost:
                entries[label] = {"added": sorted(gained), "removed": sorted(lost)}
        if entries:
            changes.append({"id": before["id"], "domain": before["domain"], **entries})
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--arctic-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty")

    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    questions, judgments, sources = prepare(ROOT / "data/pdfs")
    doc_qrels = load_judgments(judgments)
    reviewed_questions, catalog, chunk_digest = load_inputs(args.export)
    if any(reviewed_questions[str(q["id"])]["question"] != q["question"] for q in questions):
        raise ValueError("Question text differs between the two qrels")
    chunk_qrels = load_chunk_qrels(args.qrels, catalog, chunk_digest, doc_qrels)
    chunks = [{"chunk_id": cid, "document_id": doc, "text": text} for cid, (doc, text) in catalog.items()]
    print(f"Indexing {len(chunks)} matched chunks and {len(questions)} questions", flush=True)
    lexical = BM25Retriever(chunks, BM25_K1, BM25_B)
    lexical_cache = {q["question"]: lexical.scores(q["question"]) for q in questions}
    cached = SimpleNamespace(scores=lexical_cache.__getitem__)

    summary, results, encoder_runs = [], {}, []
    for name, model_path, prompt in zip(ENCODERS, (EMBED_MODEL_PATH, args.arctic_model), (None, "query"), strict=True):
        encoder = SentenceTransformer(str(model_path), device=None, local_files_only=True)
        weights = fingerprint(model_path / "model.safetensors")
        for limit in LIMITS:
            encoder.max_seq_length = limit
            tick = time.perf_counter()
            embeddings = encode_chunks(chunks, encoder)
            encoder_runs.append({"encoder": name, "limit": limit, "device": str(encoder.device),
                                 "weights": weights, "embedding_seconds": time.perf_counter() - tick,
                                 "query_prompt": encoder.prompts[prompt] if prompt else None})
            records, rankings = defaultdict(list), defaultdict(list)
            for q in questions:
                qid, question = str(q["id"]), q["question"]
                vector = encoder.encode([question], prompt_name=prompt, normalize_embeddings=True,
                                        show_progress_bar=False)[0]
                dense = embeddings @ vector
                base = {"id": q["id"], "domain": q["domain"]}
                for method in METHODS:
                    rows = rank_chunks(chunks, dense, question, method, cached,
                                       RRF_K, RRF_DENSE_WEIGHT, RRF_BM25_WEIGHT)
                    ranking = [row["chunk_id"] for row in rows]
                    rankings[method].append({**base, "chunk_ids": ranking})
                    records[method].append({**base, **evaluate_ranking(
                        ranking, catalog, chunk_qrels[qid], doc_qrels[qid])})
            for method in METHODS:
                path = output / f"{name}-{limit}" / method
                write_json(path / "ranking.json", {"questions": rankings[method]})
                write_json(path / "metrics-by-question.json", records[method])
                summary.extend(summarize(name, limit, method, records[method]))
                results[(name, limit, method)] = records[method]
            print(f"Finished {name}-{limit} on {encoder.device}", flush=True)
        del encoder

    pairs = [((e, k, "dense"), (e, k, "hybrid")) for e in ENCODERS for k in LIMITS]
    pairs += [((e, 256, m), (e, 512, m)) for e in ENCODERS for m in METHODS]
    pairs += [((ENCODERS[0], k, m), (ENCODERS[1], k, m)) for k in LIMITS for m in METHODS]
    write_json(output / "document-changes.json", [
        {"baseline": before, "candidate": after, "questions": document_changes(results[before], results[after])}
        for before, after in pairs])
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(output / "manifest.json", {
        "elapsed_seconds": time.perf_counter() - started, "chunks": len(catalog),
        "questions": len(questions), "documents": sources["document_count"],
        "sources": {**sources["sources"], "export": fingerprint(args.export / "manifest.json"),
                    "chunk_qrels": fingerprint(args.qrels / "qrels.txt"),
                    "retrieval": fingerprint(ROOT / "src/retrieval.py"), "benchmark": fingerprint(__file__)},
        "chunk_set_sha256": chunk_digest, "encoders": encoder_runs,
        "rrf": {"k": RRF_K, "dense_weight": RRF_DENSE_WEIGHT, "bm25_weight": RRF_BM25_WEIGHT,
                "bm25_k1": BM25_K1, "bm25_b": BM25_B},
        "metrics": {"chunk_cutoffs": CUTOFFS, "document_cutoffs": CUTOFFS,
                    "document_order": "First occurrence in the full ranked chunk list",
                    "distinct_document_coverage": "Within the first 40 ranked chunks",
                    "aggregation": "Macro mean over questions with a defined denominator",
                    "hit": "One if at least one relevant item appears, zero if none, null if no relevant items"},
    })


if __name__ == "__main__":
    main()
